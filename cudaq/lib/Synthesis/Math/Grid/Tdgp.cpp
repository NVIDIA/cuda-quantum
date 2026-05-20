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

const Real &tdgp_ten() {
  static const Real value(10.0);
  return value;
}

} // namespace

TdgpStepper::TdgpStepper(Integer k, const ConvexSet &setA,
                         const ConvexSet &setB, const GridOp &opG_inv,
                         Rectangle bboxA, Rectangle bboxB,
                         Interval bboxA_y_fattened, Interval bboxB_y_fattened)
    : k_(std::move(k)), setA_(&setA), setB_(&setB), opG_inv_(opG_inv),
      bboxA_(std::move(bboxA)), bboxB_(std::move(bboxB)),
      bboxA_y_fattened_(std::move(bboxA_y_fattened)),
      bboxB_y_fattened_(std::move(bboxB_y_fattened)) {
  SYNTH_OPEN_SUB("solve_tdgp");
  LLVM_DEBUG(cudaq::synth::dbgs() << "k=" << static_cast<i64>(k_) << '\n');

  // x-direction: only the first solution is needed as an anchor for the
  // line-scan step. The local stepper is destroyed at the end of this scope.
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

  dx_ = DSqrt2::power_of_inv_sqrt2(k_);
  v_common_ = opG_inv_ * DOmega::from_dsqrt2_vector(dx_, DSqrt2{0}, k_);
  v_conj_ = v_common_.conj_sq2();

  two_pow_k_ = Real((Integer(1) << k_));

  beta_gen_.emplace(bboxA_y_fattened_, bboxB_y_fattened_, k_ + 1);
}

TdgpStepper::~TdgpStepper() {
  if (skipped_betas_ > 0)
    SYNTH_ACTION("Skip") << skipped_betas_ << " betas\n";

  if (yielded_ > 0) {
    SYNTH_CLOSE_SUCCESS("yielded " + std::to_string(yielded_) + " candidates");
  } else if (!close_reason_.empty()) {
    SYNTH_CLOSE_FAILURE(close_reason_);
  } else {
    SYNTH_CLOSE_FAILURE("no candidates");
  }
}

bool TdgpStepper::advance_to_next_beta() {
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

  for (;;) {
    if (!alpha_gen_) {
      if (!advance_to_next_beta()) {
        exhausted_ = true;
        return nullptr;
      }
    }

    const DSqrt2 *alpha = alpha_gen_->next();
    if (!alpha) {
      // Inner α-stepper exhausted for the current β; move to next β.
      alpha_gen_.reset();
      continue;
    }

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
    // filter rejected -- continue with next α
  }
}

TdgpStepper solve_tdgp(Integer k, const ConvexSet &setA, const ConvexSet &setB,
                       const GridOp &opG_inv, Rectangle bboxA, Rectangle bboxB,
                       Interval bboxA_y_fattened, Interval bboxB_y_fattened) {
  return TdgpStepper(std::move(k), setA, setB, opG_inv, std::move(bboxA),
                     std::move(bboxB), std::move(bboxA_y_fattened),
                     std::move(bboxB_y_fattened));
}

} // namespace cudaq::synth
