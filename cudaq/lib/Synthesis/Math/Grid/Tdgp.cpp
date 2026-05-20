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

#define DEBUG_TYPE "cudaq-synth"

namespace cudaq::synth {

namespace {
/// Scratch space for TDGP hot-loop Real temporaries.
/// Stack-allocated once per solve_tdgp() call to avoid repeated
/// MPFR init/clear overhead.
struct TdgpScratch {
  Real intA_width, intB_width;
  Real dtA, dtB;
  Real two_pow_k;

  static const Real TEN;
};

const Real TdgpScratch::TEN = Real(10.0);

} // namespace

generator<DOmega> solve_tdgp(Integer k, const ConvexSet &setA,
                             const ConvexSet &setB, const GridOp &opG_inv,
                             Rectangle bboxA, Rectangle bboxB,
                             Interval bboxA_y_fattened,
                             Interval bboxB_y_fattened) {
  LLVM_DEBUG(llvm::dbgs() << "[grid] solve_tdgp: k=" << static_cast<i64>(k)
                          << ", bboxA=" << bboxA.I_x().width() << "x"
                          << bboxA.I_y().width()
                          << ", bboxB=" << bboxB.I_x().width() << "x"
                          << bboxB.I_y().width()
                          << ", bboxA_y_fat=" << bboxA_y_fattened.width()
                          << ", bboxB_y_fat=" << bboxB_y_fattened.width()
                          << '\n');

  TdgpScratch scratch;

  // x-direction: only the first solution is needed as an anchor for the
  // line-scan step.
  auto x_gen = solve_odgp_scaled(bboxA.I_x(), bboxB.I_x(), k + 1);
  auto x_it = x_gen.begin();
  if (x_it == x_gen.end())
    co_return;

  DSqrt2 alpha0 = *x_it;
  LLVM_DEBUG(llvm::dbgs() << "[grid] solve_tdgp: alpha0=" << alpha0 << '\n');

  DSqrt2 dx = DSqrt2::power_of_inv_sqrt2(k);
  DOmega v_common = opG_inv * DOmega::from_dsqrt2_vector(dx, DSqrt2{0}, k);
  DOmega v_conj = v_common.conj_sq2();

  for (const DSqrt2 &beta :
       solve_odgp_scaled(bboxA_y_fattened, bboxB_y_fattened, k + 1)) {
    DOmega z0 = opG_inv * DOmega::from_dsqrt2_vector(alpha0, beta, k + 1);
    auto t_A_opt = setA.intersect(z0, v_common);
    auto t_B_opt = setB.intersect(z0.conj_sq2(), v_conj);

    if (!t_A_opt.has_value() || !t_B_opt.has_value()) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[grid] solve_tdgp: beta=" << beta
                 << ", t_A_valid=" << t_A_opt.has_value()
                 << ", t_B_valid=" << t_B_opt.has_value() << " -- skip\n");
      continue;
    }

    auto [tA_l, tA_r] = *t_A_opt;
    auto [tB_l, tB_r] = *t_B_opt;

    Interval intA(tA_l, tA_r);
    Interval intB(tB_l, tB_r);

    DSqrt2 parity = absorb_sqrt2_power(beta - alpha0, k);

    scratch.intA_width = intA.width();
    scratch.intB_width = intB.width();
    scratch.two_pow_k = Real((Integer(1) << k));
    scratch.dtA =
        TdgpScratch::TEN /
        std::max(TdgpScratch::TEN, scratch.two_pow_k * scratch.intB_width);
    scratch.dtB =
        TdgpScratch::TEN /
        std::max(TdgpScratch::TEN, scratch.two_pow_k * scratch.intA_width);
    intA = fatten(intA, scratch.dtA);
    intB = fatten(intB, scratch.dtB);

    LLVM_DEBUG(llvm::dbgs()
               << "[grid] solve_tdgp: beta=" << beta << ", parity=" << parity
               << ", intA_fat=" << intA << ", intB_fat=" << intB << '\n');

    for (const DSqrt2 &alpha :
         solve_odgp_scaled_with_parity(intA, intB, 1, parity)) {
      DSqrt2 new_alpha = alpha * dx + alpha0;
      DOmega candidate = DOmega::from_dsqrt2_vector(new_alpha, beta, k);

      DOmega z_tr = opG_inv * candidate;
      bool in_A = setA.contains(z_tr);
      bool in_B = setB.contains(z_tr.conj_sq2());
      LLVM_DEBUG(llvm::dbgs()
                 << "[grid] solve_tdgp: candidate=" << candidate.to_string()
                 << ", in_A? " << in_A << ", in_B? " << in_B << '\n');
      if (in_A && in_B)
        co_yield z_tr;
    }
  }
}

} // namespace cudaq::synth
