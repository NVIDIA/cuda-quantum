/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Grid/Tdgp.h"

#include "Math/Geometry/ConvexSet.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Grid/Odgp.h"
#include "Support/StreamOps.h"
#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <utility>

#define DEBUG_TYPE "cudaq-synth"

namespace cudaq::synth {

namespace {

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// Cached constant used in the per-beta fattening factor. Held as a static
/// to avoid the GMP/MPFR allocation that would otherwise repeat for every
/// beta processed.
const Real &tdgp_ten() {
  static const Real value(10.0);
  return value;
}

} // namespace

//===----------------------------------------------------------------------===//
// TdgpStepper
//===----------------------------------------------------------------------===//

TdgpStepper::TdgpStepper(Integer k, const ConvexSet &setA,
                         const ConvexSet &setB, const GridOp &opG_inv,
                         Rectangle bboxA, Rectangle bboxB,
                         Interval bboxA_y_fattened, Interval bboxB_y_fattened)
    : k_(std::move(k)), setA_(&setA), setB_(&setB), opG_inv_(opG_inv),
      bboxA_(std::move(bboxA)), bboxB_(std::move(bboxB)),
      bboxA_y_fattened_(std::move(bboxA_y_fattened)),
      bboxB_y_fattened_(std::move(bboxB_y_fattened)) {
  CUDAQ_CUDAQ_SYNTH_OPEN_SUB("TdgpStepper");
  LLVM_DEBUG(cudaq::synth::dbgs() << "k=" << static_cast<int64_t>(k_) << '\n');

  // The x-anchor is a single one-shot solve: only the first solution to the
  // x-ODGP is needed as a fixed reference for the per-beta line scan. The
  // local stepper is destroyed at the end of this block.
  {
    OdgpScaledStepper x_gen(bboxA_.I_x(), bboxB_.I_x(), k_ + 1);
    const DSqrt2 *first_x = x_gen.next();
    if (!first_x) {
      exhausted_ = true;
      close_reason_ = "no x-direction anchor";
      return;
    }
    alpha0_ = *first_x;
  }
  LLVM_DEBUG(cudaq::synth::dbgs() << "alpha0=" << alpha0_ << '\n');

  // dx_ is the grid spacing in the t-parameter; v_common_ is the
  // direction-vector image of that spacing under opG_inv (and v_conj_ its
  // sqrt(2)-conjugate). These are loop invariants across all beta values.
  dx_ = DSqrt2::power_of_inv_sqrt2(k_);
  v_common_ = opG_inv_ * DOmega::from_dsqrt2_vector(dx_, DSqrt2{0}, k_);
  v_conj_ = v_common_.conj_sq2();

  // 2^k appears in the per-beta fattening factor. Precompute once.
  two_pow_k_ = Real((Integer(1) << k_));

  beta_gen_.emplace(bboxA_y_fattened_, bboxB_y_fattened_, k_ + 1);
}

TdgpStepper::~TdgpStepper() {
  // Emit the skip count (if any) just before the close line so the diagnostic
  // tree stays well-nested under the "TdgpStepper" scope.
  if (skipped_betas_ > 0)
    CUDAQ_SYNTH_ACTION("Skip") << skipped_betas_ << " betas\n";

  if (yielded_ > 0) {
    CUDAQ_CUDAQ_SYNTH_CLOSE_SUCCESS("yielded " + std::to_string(yielded_) + " candidates");
  } else if (!close_reason_.empty()) {
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE(close_reason_);
  } else {
    CUDAQ_CUDAQ_SYNTH_CLOSE_FAILURE("no candidates");
  }
}

bool TdgpStepper::advance_to_next_beta() {
  // Pull beta values out of beta_gen_ until one yields a non-empty
  // (intA, intB) intersection; for that beta, materialise the inner
  // alpha-stepper and set current_beta_ before returning. beta values whose
  // line misses A or B entirely are counted in skipped_betas_ and skipped.
  while (true) {
    const DSqrt2 *beta = beta_gen_->next();
    if (!beta)
      return false;

    DOmega z0 = opG_inv_ * DOmega::from_dsqrt2_vector(alpha0_, *beta, k_ + 1);
    auto t_A_opt = setA_->intersect(z0, v_common_);
    auto t_B_opt = setB_->intersect(z0.conj_sq2(), v_conj_);
    if (!t_A_opt.has_value() || !t_B_opt.has_value()) {
      ++skipped_betas_;
      continue;
    }

    auto [tA_l, tA_r] = *t_A_opt;
    auto [tB_l, tB_r] = *t_B_opt;
    Interval intA(tA_l, tA_r);
    Interval intB(tB_l, tB_r);

    DSqrt2 parity = absorb_sqrt2_power(*beta - alpha0_, k_);

    // Fatten the t-interval to absorb the MPFR rounding errors accumulated
    // along the line-intersection path. Wider intervals need less padding,
    // hence the inverse-width scaling capped at the constant tdgp_ten().
    Real intA_width = intA.width();
    Real intB_width = intB.width();
    Real dtA = tdgp_ten() / std::max(tdgp_ten(), two_pow_k_ * intB_width);
    Real dtB = tdgp_ten() / std::max(tdgp_ten(), two_pow_k_ * intA_width);
    intA = fatten(intA, dtA);
    intB = fatten(intB, dtB);

    LLVM_DEBUG(cudaq::synth::dbgs()
               << "beta=" << *beta << ", parity=" << parity
               << ", intA_fat=" << intA << ", intB_fat=" << intB << '\n');

    current_beta_ = *beta;
    alpha_gen_.emplace(intA, intB, Integer(1), parity);
    return true;
  }
}

const DOmega *TdgpStepper::next() {
  if (exhausted_)
    return nullptr;

  // Two-level loop: outer over beta (driven by beta_gen_, materialised
  // lazily via advance_to_next_beta), inner over alpha for the current
  // beta. Candidates that survive the line scan but fail the exact in-A,
  // in-B membership check are silently dropped (the line scan operates on
  // a slightly fattened interval, so over-approximation is expected).
  for (;;) {
    if (!alpha_gen_) {
      if (!advance_to_next_beta()) {
        exhausted_ = true;
        return nullptr;
      }
    }

    const DSqrt2 *alpha = alpha_gen_->next();
    if (!alpha) {
      alpha_gen_.reset();
      continue;
    }

    // Map the line-scan parameter back to (alpha, beta) coordinates, build
    // the candidate D[omega] element, and undo the upright transform.
    DSqrt2 new_alpha = *alpha * dx_ + alpha0_;
    DOmega candidate = DOmega::from_dsqrt2_vector(new_alpha, current_beta_, k_);
    DOmega z_tr = opG_inv_ * candidate;
    bool in_A = setA_->contains(z_tr);
    bool in_B = setB_->contains(z_tr.conj_sq2());
    LLVM_DEBUG(cudaq::synth::dbgs()
               << "candidate=" << candidate.to_string() << ", in_A? " << in_A
               << ", in_B? " << in_B << '\n');
    if (in_A && in_B) {
      last_sol_.assign(z_tr.u(), z_tr.k());
      ++yielded_;
      return &last_sol_;
    }
  }
}

} // namespace cudaq::synth
