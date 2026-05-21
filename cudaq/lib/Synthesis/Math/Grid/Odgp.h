/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/Interval.h"
#include "Support/Stepper.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"

#include <cmath>
#include <mpfr.h>
#include <optional>
#include <string>

namespace cudaq::synth {

//===----------------------------------------------------------------------===//
// One-Dimensional Grid Problem (ODGP)
//===----------------------------------------------------------------------===//
//
// Reference: Ross & Selinger, arXiv:1403.2975, sec. 4 (Proposition 4.5).
//
// Definition 4.3: given intervals I and J, find all alpha in Z[sqrt(2)] with
// alpha in I and alpha* in J, where (-)* is the sqrt(2)-conjugation that
// flips the sign of the sqrt(2) coefficient.
//
// Algorithm outline:
//   1. Shift to center the search.
//   2. Apply lambda-rescalings (lambda = 1 + sqrt(2), the fundamental unit of
//      Z[sqrt(2)]) until J is narrow enough for direct enumeration.
//   3. Enumerate over the resulting two-level (a, b) loop with exact bounds
//      checks against the *original* (un-shifted) interval.
//
// Variants implemented below: parity-constrained (Lemma 5.5, omega-offset
// case), scaled (Proposition 5.21, enumerating in (1/sqrt(2)^k)*Z[sqrt(2)]),
// and the combination.
//
// All entry points are stepper classes -- single-pass lazy ranges-- with the
// iterator inheriting from `llvm::iterator_facade_base`. Solutions stream out
// in lexicographic (a, b) order; destroying the stepper before exhaustion
// safely releases all resources via RAII.

//===----------------------------------------------------------------------===//
// OdgpStepper
//===----------------------------------------------------------------------===//

/// Stepper for the core ODGP (Definition 4.3).
///
/// Construction performs the one-shot shift / lambda-rescale / bounds caching;
/// `next()` resumes the (a, b) enumeration.
///
/// Pointer contract: the pointer returned by `next()` (and the reference
/// returned by `*it`) is valid only until the next call to `next()` / `++it`.
/// Callers must consume or copy before advancing.
class OdgpStepper : public StepperBase<OdgpStepper, ZSqrt2> {
public:
  OdgpStepper(Interval I, Interval J);
  ~OdgpStepper();

  OdgpStepper(const OdgpStepper &) = delete;
  OdgpStepper &operator=(const OdgpStepper &) = delete;
  OdgpStepper(OdgpStepper &&) = delete;
  OdgpStepper &operator=(OdgpStepper &&) = delete;

  /// Advance one solution; returns nullptr when exhausted.
  const ZSqrt2 *next();

private:
  // Owning the mpfr_t scratch directly amortizes mpfr_init2 / mpfr_clear over
  // the entire enumeration rather than paying it once per yielded solution.
  mpfr_t orig_I_lo_, orig_I_hi_;
  mpfr_t orig_J_lo_, orig_J_hi_;
  mpfr_t real_base_, real_slope_;
  mpfr_t conj_base_, conj_slope_;
  mpfr_t real_step_, conj_step_;
  mpfr_t cur_real_, cur_conj_;
  mpfr_t scratch1_, scratch2_;

  // Loop invariants computed once during construction.
  Integer scale_a_, scale_b_, two_scale_b_;
  Integer shift_a_, shift_b_;
  Integer delta_result_a_, delta_result_b_;
  Integer a_min_, a_max_;
  Real cur_J_l_, cur_J_r_;
  bool swapped_ = false;

  // Resumable cursor. After a successful next() these reflect the iteration
  // that produced the yielded value; the post-yield update runs on the
  // *following* call before the next bounds check.
  Integer a_, b_, b_hi_;
  Integer result_a_, result_b_;
  bool started_ = false;
  bool exhausted_ = false;

  // Buffer aliased by the pointer returned from next().
  ZSqrt2 last_sol_;

  // Drives the SYNTH_CLOSE_SUCCESS / SYNTH_CLOSE_FAILURE line emitted by the
  // destructor.
  int yielded_ = 0;
  std::string close_reason_;

  bool setup_current_a();
  void post_yield_update();
  bool b_in_bounds() const;

  void cache_interval_bounds(const Interval &I, const Interval &J);
  void compute_slopes();
  void compute_linear_bases(const Integer &offset_a, const Integer &offset_b);
  void init_current_values(const Integer &b_adj);
  void refine_range_against_bounds(const mpfr_t &bound_lo,
                                   const mpfr_t &bound_hi, const mpfr_t &slope,
                                   const mpfr_t &base, Integer &range_lo,
                                   Integer &range_hi);
};

//===----------------------------------------------------------------------===//
// OdgpWithParityStepper
//===----------------------------------------------------------------------===//

/// Stepper for the omega-offset ODGP variant (Lemma 5.5).
///
/// Yields alpha in Z[sqrt(2)] with alpha in I, alpha* in J, and the constant
/// coefficient of alpha matching the parity of `parity_hint`. Drives an inner
/// OdgpStepper over rescaled intervals; per-yield work is a coefficient
/// transform of the inner value.
class OdgpWithParityStepper
    : public StepperBase<OdgpWithParityStepper, ZSqrt2> {
public:
  OdgpWithParityStepper(Interval I, Interval J, ZSqrt2 parity_hint);
  ~OdgpWithParityStepper();

  OdgpWithParityStepper(const OdgpWithParityStepper &) = delete;
  OdgpWithParityStepper &operator=(const OdgpWithParityStepper &) = delete;
  OdgpWithParityStepper(OdgpWithParityStepper &&) = delete;
  OdgpWithParityStepper &operator=(OdgpWithParityStepper &&) = delete;

  const ZSqrt2 *next();

private:
  OdgpStepper inner_;
  i32 parity_p_;
  ZSqrt2 last_sol_;
  int yielded_ = 0;
};

//===----------------------------------------------------------------------===//
// OdgpScaledStepper
//===----------------------------------------------------------------------===//

/// Stepper for the scaled ODGP (Proposition 5.21).
///
/// Yields alpha in (1/sqrt(2)^denom_exp) * Z[sqrt(2)] with alpha in I and
/// alpha* in J. The intervals are pre-scaled (and conjugate-flipped for odd
/// exponents) before being handed to the inner OdgpStepper.
class OdgpScaledStepper : public StepperBase<OdgpScaledStepper, DSqrt2> {
public:
  OdgpScaledStepper(Interval I, Interval J, Integer denom_exp);
  ~OdgpScaledStepper();

  OdgpScaledStepper(const OdgpScaledStepper &) = delete;
  OdgpScaledStepper &operator=(const OdgpScaledStepper &) = delete;
  OdgpScaledStepper(OdgpScaledStepper &&) = delete;
  OdgpScaledStepper &operator=(OdgpScaledStepper &&) = delete;

  const DSqrt2 *next();

private:
  // Optional defers construction until the rescaled intervals are computed
  // in the body of this stepper's constructor; OdgpStepper itself is
  // non-movable so it cannot live in an initializer list expression.
  std::optional<OdgpStepper> inner_;
  Integer denom_exp_;
  DSqrt2 last_sol_;
  int yielded_ = 0;
};

//===----------------------------------------------------------------------===//
// OdgpScaledWithParityStepper
//===----------------------------------------------------------------------===//

/// Stepper for the combined scaled-with-parity ODGP.
///
/// Branches at construction on `denom_exp`: zero composes
/// OdgpWithParityStepper directly; positive values reduce the exponent by
/// one and delegate to OdgpScaledStepper, applying a constant offset to
/// each yielded value.
class OdgpScaledWithParityStepper
    : public StepperBase<OdgpScaledWithParityStepper, DSqrt2> {
public:
  OdgpScaledWithParityStepper(Interval I, Interval J, Integer denom_exp,
                              DSqrt2 parity_hint);
  ~OdgpScaledWithParityStepper();

  OdgpScaledWithParityStepper(const OdgpScaledWithParityStepper &) = delete;
  OdgpScaledWithParityStepper &
  operator=(const OdgpScaledWithParityStepper &) = delete;
  OdgpScaledWithParityStepper(OdgpScaledWithParityStepper &&) = delete;
  OdgpScaledWithParityStepper &
  operator=(OdgpScaledWithParityStepper &&) = delete;

  const DSqrt2 *next();

private:
  // Exactly one branch is engaged for the lifetime of the stepper.
  std::optional<OdgpWithParityStepper> direct_;  // denom_exp == 0
  std::optional<OdgpScaledStepper> recursive_;   // denom_exp > 0
  DSqrt2 offset_;
  DSqrt2 last_sol_;
  int yielded_ = 0;
};

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//
//
// Each factory returns a stepper by value. The stepper types are non-movable;
// guaranteed copy elision (C++17 prvalue rules) is what makes both the
// `return Foo(...)` form and caller-side `auto x = solve_foo(...)` legal.

/// Core ODGP (Definition 4.3): all alpha in Z[sqrt(2)] with alpha in I and
/// alpha* in J.
OdgpStepper solve_odgp(Interval I, Interval J);

/// ODGP that fixes the parity of alpha's constant coefficient.
OdgpWithParityStepper solve_odgp_with_parity(Interval I, Interval J,
                                             ZSqrt2 parity_hint);

/// Scaled ODGP: enumerate in (1/sqrt(2)^denom_exp) * Z[sqrt(2)].
OdgpScaledStepper solve_odgp_scaled(Interval I, Interval J, Integer denom_exp);

/// Scaled ODGP with parity constraint.
OdgpScaledWithParityStepper
solve_odgp_scaled_with_parity(Interval I, Interval J, Integer denom_exp,
                              DSqrt2 parity_hint);

} // namespace cudaq::synth
