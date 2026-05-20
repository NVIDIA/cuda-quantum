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

/// One-Dimensional Grid Problem (ODGP) solver.
///
/// Reference: Ross & Selinger, arXiv:1403.2975, §4 (Proposition 4.5).
///
/// The one-dimensional grid problem (Definition 4.3):
///   Given intervals I and J, find all α ∈ Z[√2] satisfying α ∈ I and α● ∈ J.
///
/// Algorithm: Initial shift to center the search, iterative `λ-rescaling` until
/// the interval is narrow enough, then direct enumeration with exact bounds
/// checks. Variants support parity constraints (Lemma 5.5) and scaled
/// problems (Proposition 5.21).
///
/// All entry points are exposed as hand-rolled steppers (no coroutines), with
/// the iterator inheriting from `llvm::iterator_facade_base`. Each stepper
/// yields solutions on demand in lexicographic (a, b) order. Early termination
/// (destroying the stepper before exhaustion) is safe and releases all
/// resources via RAII.

/// Lazy stepper for the core ODGP (Definition 4.3): find all α ∈ Z[√2] with
/// α ∈ I and α● ∈ J.
///
/// Usage:
///   for (const ZSqrt2 &alpha : OdgpStepper(I, J)) { ... }
///   auto vec = to_vector(OdgpStepper(I, J));
///
/// Setup (shift, λ-rescale, bounds caching) runs once in the constructor; the
/// enumeration is resumed across calls to `next()`. There is no heap-allocated
/// `coroutine` frame, and the inner `b`-loop is plain C++ -- the compiler can
/// inline `next()` into the calling code and keep loop variables in registers.
///
/// Yielded reference contract: the pointer returned by `next()` and the
/// reference returned by `*it` are valid until the next call to `next()` /
/// `++it`. Callers must consume or copy the value before advancing.
///
/// Non-copyable, non-movable. Owns mpfr_t scratch buffers that are
/// `mpfr_init2`-ed once in the constructor and `mpfr_clear`-ed in the
/// destructor.
class OdgpStepper : public StepperBase<OdgpStepper, ZSqrt2> {
public:
  OdgpStepper(Interval I, Interval J);
  ~OdgpStepper();

  OdgpStepper(const OdgpStepper &) = delete;
  OdgpStepper &operator=(const OdgpStepper &) = delete;
  OdgpStepper(OdgpStepper &&) = delete;
  OdgpStepper &operator=(OdgpStepper &&) = delete;

  /// Advance to the next solution. Returns a pointer valid until the next
  /// call to `next()` or destruction. Returns `nullptr` when exhausted.
  const ZSqrt2 *next();

private:
  // mpfr_t scratch buffers used by the inner loop. Initialized in the
  // constructor, cleared in the destructor.
  mpfr_t orig_I_lo_, orig_I_hi_;
  mpfr_t orig_J_lo_, orig_J_hi_;
  mpfr_t real_base_, real_slope_;
  mpfr_t conj_base_, conj_slope_;
  mpfr_t real_step_, conj_step_;
  mpfr_t cur_real_, cur_conj_;
  mpfr_t scratch1_, scratch2_;

  // Loop-invariant integer / real state computed in the constructor.
  Integer scale_a_, scale_b_, two_scale_b_;
  Integer shift_a_, shift_b_;
  Integer delta_result_a_, delta_result_b_;
  Integer a_min_, a_max_;
  Real cur_J_l_, cur_J_r_;
  bool swapped_ = false;

  // Resumable enumeration state. After a successful `next()`, these reflect
  // the iteration that produced the yielded value (not yet post-updated).
  Integer a_, b_, b_hi_;
  Integer result_a_, result_b_;
  bool started_ = false;
  bool exhausted_ = false;

  // Storage for the value pointed to by the return of `next()`.
  ZSqrt2 last_sol_;

  // Diagnostics: counts yielded solutions; on destruction emits either
  // "yielded N" or a failure reason via the SYNTH_CLOSE_* macros.
  int yielded_ = 0;
  std::string close_reason_;

  // -- Internal helpers (defined in Odgp.cpp) --
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

/// Lazy stepper for ODGP with parity (for ω-offset case, Lemma 5.5).
///
/// Yields α ∈ Z[√2] with `α ∈ I`, `α● ∈ J`, and `α ≡ parity_hint (mod 2)` in
/// the constant coefficient. Composes `OdgpStepper` over rescaled intervals.
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

/// Lazy stepper for the scaled ODGP (Proposition 5.21): find all
/// α ∈ (1/√2^denom_exp)·Z[√2] with α ∈ I and α● ∈ J.
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
  // Pre-rescaled intervals must be passed to `inner_` at construction; the
  // simplest way to do that without a helper constructor is to compute them
  // in the public constructor body and pass them on via `inner_emplace`.
  // Since `OdgpStepper` is non-movable we hold it through `std::optional`
  // and `emplace` it once the rescaled intervals are ready.
  std::optional<OdgpStepper> inner_;
  Integer denom_exp_;
  DSqrt2 last_sol_;
  int yielded_ = 0;
};

/// Lazy stepper for the scaled-with-parity ODGP variant. Internally branches
/// on `denom_exp == 0` (composes `OdgpWithParityStepper` directly) vs the
/// recursive case (composes `OdgpScaledStepper` over shifted intervals).
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
  // Exactly one of `direct_` / `recursive_` is engaged based on `denom_exp`.
  std::optional<OdgpWithParityStepper> direct_;   // denom_exp == 0
  std::optional<OdgpScaledStepper> recursive_;    // denom_exp > 0
  DSqrt2 offset_;
  DSqrt2 last_sol_;
  int yielded_ = 0;
};

// ---------------------------------------------------------------------------
// Factory functions
// ---------------------------------------------------------------------------
//
// All return concrete steppers by value. Each stepper is non-movable but the
// return is a prvalue (guaranteed copy elision since C++17), so the
// caller-side variable is constructed directly without invoking a move.

/// Core ODGP (Definition 4.3): find all α ∈ Z[√2] with α ∈ I and α● ∈ J.
OdgpStepper solve_odgp(Interval I, Interval J);

/// ODGP with parity constraint (Lemma 5.5).
OdgpWithParityStepper solve_odgp_with_parity(Interval I, Interval J,
                                             ZSqrt2 parity_hint);

/// Scaled ODGP (Proposition 5.21).
OdgpScaledStepper solve_odgp_scaled(Interval I, Interval J, Integer denom_exp);

/// Scaled ODGP with parity.
OdgpScaledWithParityStepper
solve_odgp_scaled_with_parity(Interval I, Interval J, Integer denom_exp,
                              DSqrt2 parity_hint);

} // namespace cudaq::synth
