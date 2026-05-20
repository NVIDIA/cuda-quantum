/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Math/Geometry/Interval.h"
#include "Support/Generator.h"
#include "cudaq/Synthesis/Math/Ring/Dsqrt2.h"
#include "cudaq/Synthesis/Math/Ring/Zsqrt2.h"

#include <cmath>
#include <iterator>
#include <mpfr.h>
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
/// The core ODGP is exposed as `OdgpStepper`, a hand-rolled stepper class
/// (no coroutine) so the inner enumeration loop runs as plain C++ without
/// suspend/resume overhead. The parity / scaled / scaled-with-parity wrappers
/// remain coroutines for compositional readability; each is a thin transform
/// over the stepper or another wrapper.
///
/// All entry points yield solutions on demand in lexicographic (a, b) order.
/// Early termination (destroying the stepper / generator before exhaustion)
/// is safe and releases all resources via RAII.

// NOTE: The `coroutine` wrappers below take parameters by value to avoid the
// dangling-reference pitfall: `coroutine` frames store copies of parameters,
// but for reference parameters only the reference (pointer) is copied.
// If the caller passed a temporary, the reference dangles after the
// `coroutine`'s first suspension point. `OdgpStepper` is not a coroutine, so
// this rule does not apply to it -- but for API symmetry it also takes
// parameters by value.

/// Lazy stepper for the core ODGP (Definition 4.3): find all α ∈ Z[√2] with
/// α ∈ I and α● ∈ J.
///
/// Usage:
///   for (const ZSqrt2 &alpha : OdgpStepper(I, J)) { ... }
///   auto vec = to_vector(OdgpStepper(I, J));
///
/// Replaces an earlier C++20 `coroutine` implementation. The setup work
/// (shift, λ-rescale, bounds caching) runs once in the constructor; the
/// enumeration is resumed across calls to `next()`. There is no heap-allocated
/// `coroutine` frame, and the inner `b`-loop is plain C++ -- the compiler can
/// inline `next()` into the calling code and keep loop variables in registers
/// instead of spilling them to a frame.
///
/// Yielded reference contract: the pointer returned by `next()` and the
/// reference returned by `*it` are valid until the next call to `next()` /
/// `++it`. Callers must consume or copy the value before advancing.
///
/// Non-copyable, non-movable. Owns mpfr_t scratch buffers that are
/// `mpfr_init2`-ed once in the constructor and `mpfr_clear`-ed in the
/// destructor.
class OdgpStepper {
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

  // --- Range-for support (input-iterator semantics, matching generator<T>) ---
  struct sentinel {};

  struct iterator {
    using iterator_category = std::input_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = ZSqrt2;
    using reference = const ZSqrt2 &;
    using pointer = const ZSqrt2 *;

    OdgpStepper *s_ = nullptr;
    const ZSqrt2 *v_ = nullptr;

    iterator &operator++() {
      v_ = s_->next();
      return *this;
    }
    void operator++(int) { ++*this; }
    reference operator*() const { return *v_; }
    pointer operator->() const { return v_; }

    friend bool operator==(const iterator &it, sentinel) noexcept {
      return it.v_ == nullptr;
    }
    friend bool operator!=(const iterator &it, sentinel s) noexcept {
      return !(it == s);
    }
    friend bool operator==(sentinel s, const iterator &it) noexcept {
      return it == s;
    }
    friend bool operator!=(sentinel s, const iterator &it) noexcept {
      return !(it == s);
    }
  };

  iterator begin() { return iterator{this, next()}; }
  sentinel end() const noexcept { return {}; }

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

  /// Find the next a in [a_, a_max_] with a non-empty refined b-range,
  /// initializing b_, b_hi_, result_a_, result_b_, and the mpfr running state.
  /// Returns false (and leaves the stepper exhausted) if no such a exists.
  bool setup_current_a();

  /// Advance one step within the current a-line: ++b_, update result_*,
  /// step the cur_real_/cur_conj_ MPFR running state. Mirrors the four
  /// statements after `co_yield sol;` in the pre-stepper implementation.
  void post_yield_update();

  /// Test whether the current (cur_real_, cur_conj_) lies inside the cached
  /// original I × J rectangle.
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

// Core ODGP (Definition 4.3): find all α ∈ Z[√2] with α ∈ I and α● ∈ J.
///
/// Returned by value via guaranteed copy elision; `OdgpStepper` is
/// non-movable but the return is a prvalue construction, so no move is
/// invoked.
OdgpStepper solve_odgp(Interval I, Interval J);

// With parity constraint (for ω-offset case, Lemma 5.5)
generator<ZSqrt2> solve_odgp_with_parity(Interval I, Interval J,
                                         ZSqrt2 parity_hint);

// Scaled ODGP (Proposition 5.21): find all α ∈ (1/√2^denom_exp)Z[√2] with
// α ∈ I and α● ∈ J
generator<DSqrt2> solve_odgp_scaled(Interval I, Interval J, Integer denom_exp);

// Scaled with parity
generator<DSqrt2> solve_odgp_scaled_with_parity(Interval I, Interval J,
                                                Integer denom_exp,
                                                DSqrt2 parity_hint);

} // namespace cudaq::synth
