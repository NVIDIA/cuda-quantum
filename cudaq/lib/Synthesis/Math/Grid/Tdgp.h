/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/GridOp.h"
#include "Math/Geometry/Interval.h"
#include "Math/Geometry/Rectangle.h"
#include "Math/Grid/Odgp.h"
#include "Support/Stepper.h"
#include "cudaq/Synthesis/Math/Ring/Domega.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"

#include <optional>
#include <string>

namespace cudaq::synth {
class ConvexSet;
} // namespace cudaq::synth

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// Two-Dimensional Grid Problem (TDGP)
//===----------------------------------------------------------------------===//
//
// Reference: Ross & Selinger, arXiv:1403.2975, sec. 5.1-5.7
// (Theorem 5.18, Propositions 5.21-5.22).
//
// Definition 5.3 / 5.20: given convex sets A, B in R^2 and a denominator
// exponent k >= 0, find all u in (1/sqrt(2)^k) * Z[omega] with u in A and
// u* in B, where (-)* is the sqrt(2)-conjugation.
//
// Algorithm shape (Theorem 5.18):
//   1. Upright preprocessing (delegated to ToUpright): a grid operator G is
//      computed such that G(A) and G*(B) are 1/6-upright; the TDGP solver is
//      handed the inverse G^-1 together with the upright bounding boxes.
//   2. Reduction to 1D problems (Lemma 5.6): any u = alpha + beta*i (even
//      case) or u = alpha + beta*i + omega (odd case) has alpha, beta in
//      Z[sqrt(2)], so the 2D constraint decomposes into two independent 1D
//      ODGPs, one on the real axis and one on the imaginary axis.
//   3. Line scan: take a single x-solution alpha_0 as an anchor, parametrise
//      a line through the candidates as z(t) = z_0 + t * v, intersect that
//      line with A and B to get a t-range, and run a 1D scaled-with-parity
//      ODGP on t.
//   4. Filter: undo the upright transform via G^-1 and re-verify membership
//      against the original A and B. The fattened t-range used in step 3
//      slightly over-approximates, so a small number of false positives are
//      expected here and silently skipped.
//
// Deviation from paper: rather than running two separate 1D grid problems
// for the even and odd (omega-offset) cases (Lemma 5.6), this implementation
// folds both into the scaled-with-parity ODGP variant by fattening the y
// bounding boxes -- see bboxA_y_fattened / bboxB_y_fattened in the
// constructor signature.
//
// The caller iterates k = 0, 1, 2, ... over fresh `TdgpStepper` instances
// until a solution is found, which produces candidates in order of increasing
// T-count (Lemma 7.3, Proposition 5.22).
//
// The whole pipeline (x-ODGP, y-ODGP, scaled-with-parity ODGP, this TDGP)
// is lazy: early termination by the caller propagates through every layer
// via RAII without performing unnecessary downstream work.

//===----------------------------------------------------------------------===//
// TdgpStepper
//===----------------------------------------------------------------------===//

/// Stepper for the scaled TDGP at a fixed denominator exponent k.
///
/// Yields all u in (1/sqrt(2)^k) * Z[omega] satisfying u in `setA` and
/// u* in `setB`. Composes `OdgpScaledStepper` for the beta-iteration and
/// `OdgpScaledWithParityStepper` for the alpha-iteration on each beta line.
/// Candidates that pass the line-scan refinement but fail the exact
/// membership re-check (because the line-scan operates on a slightly fattened
/// interval) are silently dropped without bumping the yield counter.
///
/// Pointer contract matches the rest of the synth steppers: the value
/// returned by `next()` is valid until the next call to `next()` / `++it`.
class TdgpStepper : public StepperBase<TdgpStepper, DOmega> {
public:
  TdgpStepper(Integer k, const ConvexSet &setA, const ConvexSet &setB,
              const GridOp &opG_inv, Rectangle bboxA, Rectangle bboxB,
              Interval bboxA_y_fattened, Interval bboxB_y_fattened);
  ~TdgpStepper();

  TdgpStepper(const TdgpStepper &) = delete;
  TdgpStepper &operator=(const TdgpStepper &) = delete;
  TdgpStepper(TdgpStepper &&) = delete;
  TdgpStepper &operator=(TdgpStepper &&) = delete;

  const DOmega *next();

private:
  // Inputs captured by the constructor. setA_ and setB_ are non-owning
  // pointers because they live in the caller's frame for the stepper's
  // lifetime.
  Integer k_;
  const ConvexSet *setA_;
  const ConvexSet *setB_;
  GridOp opG_inv_;
  Rectangle bboxA_;
  Rectangle bboxB_;
  Interval bboxA_y_fattened_;
  Interval bboxB_y_fattened_;

  // Line-scan parameters computed in the constructor (after the one-shot
  // x-anchor solve). alpha0_ is the anchor; dx_ is the per-step grid
  // spacing in t; v_common_ and v_conj_ are the direction vector and its
  // sqrt(2)-conjugate; two_pow_k_ is precomputed for the per-beta fattening
  // factor.
  DSqrt2 alpha0_;
  DSqrt2 dx_;
  DOmega v_common_;
  DOmega v_conj_;
  Real two_pow_k_;

  // beta-iteration cursor. beta_gen_ is engaged once the constructor finds
  // an x-anchor; current_beta_ / alpha_gen_ are reset and re-emplaced each
  // time `advance_to_next_beta()` moves on.
  std::optional<OdgpScaledStepper> beta_gen_;
  DSqrt2 current_beta_;
  std::optional<OdgpScaledWithParityStepper> alpha_gen_;

  // Buffer aliased by the pointer returned from next().
  DOmega last_sol_;

  // Termination flag plus diagnostic counters that drive the destructor's
  // CUDAQ_SYNTH_CLOSE_* / CUDAQ_SYNTH_ACTION emission.
  bool exhausted_ = false;
  int yielded_ = 0;
  int skipped_betas_ = 0;
  std::string close_reason_;

  /// Walk beta_gen_ forward until a beta with a non-empty (A, B) line
  /// intersection is found. On success: sets `current_beta_`, emplaces
  /// `alpha_gen_`, and returns true. On exhaustion: returns false. Betas
  /// whose line misses A or B are counted in `skipped_betas_`.
  bool advance_to_next_beta();
};

} // namespace cudaq::synth
