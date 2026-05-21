/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Grid/Odgp.h"

#include "Support/StreamOps.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>

#define DEBUG_TYPE "cudaq-synth"

namespace cudaq::synth {

namespace {

//===----------------------------------------------------------------------===//
// Internal helpers
//===----------------------------------------------------------------------===//

/// Computes n = floor(log_y(x)) and the remainder r = x / y^n.
///
/// Uses a squaring chain plus binary search on the exponent so the cost
/// scales with log|n|, not |n|. The chain is cached on a fixed-size local
/// array because m grows like log log x and stays well under the 64-slot
/// bound in practice.
std::pair<Integer, Real> floor_log(const Real &x, const Real &y) {
  assert(x > Real(0) && "floor_log: non-positive input (floating point error)");

  // Phase 1: bracket log_y(x) within [-2^m, 2^m), filling pow_y[0..m] with
  // y^(2^i) so phase 2 can binary-search without recomputing the chain.
  std::array<Real, 64> pow_y;
  pow_y[0] = y;
  int m = 0;
  while (x >= pow_y[m] || x * pow_y[m] < Real(1)) {
    pow_y[m + 1] = pow_y[m] * pow_y[m]; // y^(2^(m+1))
    ++m;
  }

  // Phase 2: extract the bits of n from high to low. pow_y[m] is used only
  // for the x < 1 init shift; pow_y[0..m-1] carry the per-bit divisors.
  Integer n;
  Real r;
  if (x >= Real(1)) {
    n = 0;
    r = x;
  } else {
    n = -1;
    r = x * pow_y[m];
  }
  for (int i = m - 1; i >= 0; --i) {
    n <<= 1;
    if (r > pow_y[i]) {
      r /= pow_y[i];
      n += 1;
    }
  }
  return {n, r};
}

/// Cached powers of the Z[sqrt(2)] fundamental unit lambda = 1 + sqrt(2),
/// including the sqrt(2)-conjugate lambda* = 1 - sqrt(2) and the inverse
/// lambda^-1 = -lambda* = -1 + sqrt(2). The real-valued projections are
/// precomputed for fast access in the rescaling loop.
struct LambdaPowers {
  ZSqrt2 lambda_n;
  ZSqrt2 lambda_conj_n;
  ZSqrt2 lambda_inv_n;
  Real lambda_n_real;
  Real lambda_conj_n_real;
};

/// Thread-local memoized table indexed by exponent. The first call eagerly
/// fills the first ~10 entries (covering the common shallow-rescale case)
/// and later calls extend on demand.
const LambdaPowers &get_lambda_powers(const Integer &n) {
  assert(n >= 0 && "get_lambda_powers requires non-negative exponent");
  size_t idx = static_cast<size_t>(n);
  static thread_local std::vector<LambdaPowers> cache;
  if (cache.empty()) {
    LambdaPowers base{ZSqrt2{1}, ZSqrt2{1}, ZSqrt2{1}, Real(1.0), Real(1.0)};
    cache.push_back(base);
    static const ZSqrt2 LAMBDA_CONJ = ZSqrt2::lambda().conj_sq2();
    static const ZSqrt2 LAMBDA_INV = *inv(ZSqrt2::lambda());
    for (int i = 0; i < 8; ++i) {
      const auto &prev = cache.back();
      LambdaPowers next{prev.lambda_n * ZSqrt2::lambda(),
                        prev.lambda_conj_n * LAMBDA_CONJ,
                        prev.lambda_inv_n * LAMBDA_INV, Real(0.0), Real(0.0)};
      next.lambda_n_real = to_real(next.lambda_n);
      next.lambda_conj_n_real = to_real(next.lambda_conj_n);
      cache.emplace_back(next);
    }
  }
  if (idx >= cache.size()) {
    static const ZSqrt2 LAMBDA_CONJ = ZSqrt2::lambda().conj_sq2();
    static const ZSqrt2 LAMBDA_INV = *inv(ZSqrt2::lambda());
    while (cache.size() <= idx) {
      const auto &prev = cache.back();
      LambdaPowers next{prev.lambda_n * ZSqrt2::lambda(),
                        prev.lambda_conj_n * LAMBDA_CONJ,
                        prev.lambda_inv_n * LAMBDA_INV, Real(0.0), Real(0.0)};
      next.lambda_n_real = to_real(next.lambda_n);
      next.lambda_conj_n_real = to_real(next.lambda_conj_n);
      cache.emplace_back(next);
    }
  }
  return cache[idx];
}

/// Below this absolute slope the linear bound-refinement is treated as
/// degenerate and the range-intersection step is skipped (the bounds are
/// determined entirely by the base value).
constexpr double SLOPE_ZERO_TOLERANCE = 1e-40;

} // namespace

//===----------------------------------------------------------------------===//
// OdgpStepper
//===----------------------------------------------------------------------===//

OdgpStepper::OdgpStepper(Interval I, Interval J) {
  SYNTH_OPEN_SUB("solve_odgp");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "I_width=" << I.width() << ", J_width=" << J.width() << '\n');

  // mpfr_t buffers must be initialized before any path that the destructor
  // could subsequently follow -- the destructor always clears them, even on
  // the early-return paths below.
  const mpfr_prec_t prec = Real::get_default_precision();
  mpfr_init2(orig_I_lo_, prec);
  mpfr_init2(orig_I_hi_, prec);
  mpfr_init2(orig_J_lo_, prec);
  mpfr_init2(orig_J_hi_, prec);
  mpfr_init2(real_base_, prec);
  mpfr_init2(real_slope_, prec);
  mpfr_init2(conj_base_, prec);
  mpfr_init2(conj_slope_, prec);
  mpfr_init2(real_step_, prec);
  mpfr_init2(conj_step_, prec);
  mpfr_init2(cur_real_, prec);
  mpfr_init2(cur_conj_, prec);
  mpfr_init2(scratch1_, prec);
  mpfr_init2(scratch2_, prec);

  if (I.width() < 0 || J.width() < 0) {
    exhausted_ = true;
    close_reason_ = "empty interval";
    return;
  }

  // Shift to bring the search window towards the origin so the lambda
  // rescaling below converges quickly.
  Integer shift_a = floor_to_integer((I.l() + J.l()) / 2);
  Integer shift_b = floor_to_integer(Real::sqrt2() * (I.l() - J.l()) / 4);
  ZSqrt2 shift(shift_a, shift_b);

  Interval cur_I = I - to_real(shift);
  Interval cur_J = J - to_real(shift.conj_sq2());

  // The rescaling step assumes |cur_I| <= |cur_J|; if the post-shift widths
  // come out the other way around we swap and remember to undo the swap when
  // computing per-iteration deltas.
  ZSqrt2 scale{1};
  swapped_ = false;
  if (cur_I.width() > 0 && cur_J.width() <= 0) {
    std::swap(cur_I, cur_J);
    swapped_ = true;
  }

  // Repeatedly multiply cur_I by lambda^n and cur_J by (lambda*)^n until
  // cur_J is narrow enough for direct enumeration. Each iteration shrinks
  // cur_J by lambda^(2n) so the loop terminates quickly.
  static const Real LAMBDA_REAL = to_real(ZSqrt2::lambda());
  while (cur_J.width() > 0) {
    auto [n, _] = floor_log(cur_J.width(), LAMBDA_REAL);
    if (n <= 0)
      break;
    const auto &powers = get_lambda_powers(n);
    cur_I = cur_I * powers.lambda_n_real;
    cur_J = cur_J * powers.lambda_conj_n_real;
    scale = scale * powers.lambda_inv_n;
  }

  LLVM_DEBUG(cudaq::synth::dbgs()
             << "after rescale: cur_I_width=" << cur_I.width()
             << ", cur_J_width=" << cur_J.width() << ", swapped=" << swapped_
             << '\n');

  cache_interval_bounds(I, J);

  scale_a_ = scale.a();
  scale_b_ = scale.b();
  shift_a_ = shift.a();
  shift_b_ = shift.b();
  two_scale_b_ = scale_b_ * Integer(2);

  // direction == -1 reflects the swap above; it propagates into the per-step
  // delta and the cur_real_/cur_conj_ running update.
  int direction = swapped_ ? -1 : 1;

  compute_slopes();
  if (direction > 0) {
    mpfr_set(real_step_, real_slope_, MPFR_RNDN);
    mpfr_set(conj_step_, conj_slope_, MPFR_RNDN);
  } else {
    mpfr_neg(real_step_, real_slope_, MPFR_RNDN);
    mpfr_neg(conj_step_, conj_slope_, MPFR_RNDN);
  }

  a_min_ = ceil_to_integer((cur_I.l() + cur_J.l()) / 2);
  a_max_ = floor_to_integer((cur_I.r() + cur_J.r()) / 2);
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "a_min=" << a_min_ << ", a_max=" << a_max_ << '\n');

  delta_result_a_ = (direction > 0) ? two_scale_b_ : -two_scale_b_;
  delta_result_b_ = (direction > 0) ? scale_a_ : -scale_a_;

  cur_J_l_ = cur_J.l();
  cur_J_r_ = cur_J.r();
}

OdgpStepper::~OdgpStepper() {
  mpfr_clear(orig_I_lo_);
  mpfr_clear(orig_I_hi_);
  mpfr_clear(orig_J_lo_);
  mpfr_clear(orig_J_hi_);
  mpfr_clear(real_base_);
  mpfr_clear(real_slope_);
  mpfr_clear(conj_base_);
  mpfr_clear(conj_slope_);
  mpfr_clear(real_step_);
  mpfr_clear(conj_step_);
  mpfr_clear(cur_real_);
  mpfr_clear(cur_conj_);
  mpfr_clear(scratch1_);
  mpfr_clear(scratch2_);

  if (yielded_ > 0) {
    SYNTH_CLOSE_SUCCESS("yielded " + std::to_string(yielded_));
  } else if (!close_reason_.empty()) {
    SYNTH_CLOSE_FAILURE(close_reason_);
  } else {
    SYNTH_CLOSE_FAILURE("no solutions");
  }
}

const ZSqrt2 *OdgpStepper::next() {
  if (exhausted_)
    return nullptr;

  // Contract of the b cursor: at function entry on every call after the first,
  // (a_, b_, result_a_, result_b_, cur_real_, cur_conj_) all reflect the
  // iteration that just yielded. We must therefore step forward *before*
  // running the bounds check on the new position.
  if (!started_) {
    a_ = a_min_;
    if (!setup_current_a()) {
      exhausted_ = true;
      return nullptr;
    }
    started_ = true;
  } else {
    post_yield_update();
    if (b_ > b_hi_) {
      ++a_;
      if (!setup_current_a()) {
        exhausted_ = true;
        return nullptr;
      }
    }
  }

  for (;;) {
    if (b_in_bounds()) {
      last_sol_.assign(result_a_, result_b_);
      ++yielded_;
      return &last_sol_;
    }
    post_yield_update();
    if (b_ > b_hi_) {
      ++a_;
      if (!setup_current_a()) {
        exhausted_ = true;
        return nullptr;
      }
    }
  }
}

bool OdgpStepper::setup_current_a() {
  // Walk a_ forward until we find an outer index whose refined b-range is
  // non-empty, materialising the per-a line state for it. Returns false if
  // a_ ran past a_max_ without finding a usable line.
  while (a_ <= a_max_) {
    Integer b_lo = ceil_to_integer(Real::sqrt2() * (a_ - cur_J_r_) / 2);
    Integer b_hi_local = floor_to_integer(Real::sqrt2() * (a_ - cur_J_l_) / 2);
    if (b_hi_local < b_lo) {
      ++a_;
      continue;
    }

    Integer base_a = a_ * scale_a_;
    Integer base_b = a_ * scale_b_;
    Integer offset_a = base_a + shift_a_;
    Integer offset_b = base_b + shift_b_;

    compute_linear_bases(offset_a, offset_b);

    // The bound-refinement runs in the (un-swapped) b coordinate; if we
    // entered swapped mode above, mirror the range across zero on the way in
    // and back on the way out.
    Integer b_adj_lo = swapped_ ? -b_hi_local : b_lo;
    Integer b_adj_hi = swapped_ ? -b_lo : b_hi_local;

    refine_range_against_bounds(orig_I_lo_, orig_I_hi_, real_slope_, real_base_,
                                b_adj_lo, b_adj_hi);
    if (b_adj_hi < b_adj_lo) {
      ++a_;
      continue;
    }
    refine_range_against_bounds(orig_J_lo_, orig_J_hi_, conj_slope_, conj_base_,
                                b_adj_lo, b_adj_hi);
    if (b_adj_hi < b_adj_lo) {
      ++a_;
      continue;
    }

    if (swapped_) {
      b_lo = std::max(b_lo, Integer(-b_adj_hi));
      b_hi_local = std::min(b_hi_local, Integer(-b_adj_lo));
    } else {
      b_lo = std::max(b_lo, Integer(b_adj_lo));
      b_hi_local = std::min(b_hi_local, Integer(b_adj_hi));
    }
    if (b_hi_local < b_lo) {
      ++a_;
      continue;
    }

    Integer b_adj_start = swapped_ ? -b_lo : b_lo;
    result_a_ = offset_a + two_scale_b_ * b_adj_start;
    result_b_ = offset_b + scale_a_ * b_adj_start;
    init_current_values(b_adj_start);

    b_ = b_lo;
    b_hi_ = b_hi_local;
    return true;
  }
  return false;
}

void OdgpStepper::post_yield_update() {
  ++b_;
  result_a_ += delta_result_a_;
  result_b_ += delta_result_b_;
  mpfr_add(cur_real_, cur_real_, real_step_, MPFR_RNDN);
  mpfr_add(cur_conj_, cur_conj_, conj_step_, MPFR_RNDN);
}

bool OdgpStepper::b_in_bounds() const {
  return mpfr_cmp(cur_real_, orig_I_lo_) >= 0 &&
         mpfr_cmp(cur_real_, orig_I_hi_) <= 0 &&
         mpfr_cmp(cur_conj_, orig_J_lo_) >= 0 &&
         mpfr_cmp(cur_conj_, orig_J_hi_) <= 0;
}

void OdgpStepper::cache_interval_bounds(const Interval &I, const Interval &J) {
  mpfr_set(orig_I_lo_, I.l().get_mpfr(), MPFR_RNDN);
  mpfr_set(orig_I_hi_, I.r().get_mpfr(), MPFR_RNDN);
  mpfr_set(orig_J_lo_, J.l().get_mpfr(), MPFR_RNDN);
  mpfr_set(orig_J_hi_, J.r().get_mpfr(), MPFR_RNDN);

  // The inner-loop check (cur_real in [orig_I_lo, orig_I_hi], etc.) compares
  // a value computed incrementally (cur = base + b*slope, then cur += step,
  // up to ~2 ULPs of drift) against bounds computed through a different MPFR
  // path. When the exact solution sits on the interval boundary (e.g.
  // w* = -1/sqrt(2) in the Hadamard inner TDGP) these two representations
  // can disagree by a few ULPs, rejecting valid candidates.
  //
  // Widening the cached bounds by 4 ULPs covers the worst-case drift. False
  // positives that slip past this widened gate are caught downstream by the
  // exact DSqrt2/DOmega integer arithmetic in the TDGP's contains() check.
  for (int i = 0; i < 4; ++i) {
    mpfr_nextbelow(orig_I_lo_);
    mpfr_nextabove(orig_I_hi_);
    mpfr_nextbelow(orig_J_lo_);
    mpfr_nextabove(orig_J_hi_);
  }
}

void OdgpStepper::compute_slopes() {
  mpfr_set_z(real_slope_, two_scale_b_.get_mpz_t(), MPFR_RNDN);
  mpfr_mul_z(scratch1_, Real::sqrt2().get_mpfr(), scale_a_.get_mpz_t(),
             MPFR_RNDN);
  mpfr_add(real_slope_, real_slope_, scratch1_, MPFR_RNDN);

  mpfr_set_z(conj_slope_, two_scale_b_.get_mpz_t(), MPFR_RNDN);
  mpfr_sub(conj_slope_, conj_slope_, scratch1_, MPFR_RNDN);
}

void OdgpStepper::compute_linear_bases(const Integer &offset_a,
                                       const Integer &offset_b) {
  mpfr_set_z(real_base_, offset_a.get_mpz_t(), MPFR_RNDN);
  mpfr_mul_z(scratch1_, Real::sqrt2().get_mpfr(), offset_b.get_mpz_t(),
             MPFR_RNDN);
  mpfr_add(real_base_, real_base_, scratch1_, MPFR_RNDN);

  mpfr_set_z(conj_base_, offset_a.get_mpz_t(), MPFR_RNDN);
  mpfr_sub(conj_base_, conj_base_, scratch1_, MPFR_RNDN);
}

void OdgpStepper::init_current_values(const Integer &b_adj) {
  mpfr_mul_z(scratch1_, real_slope_, b_adj.get_mpz_t(), MPFR_RNDN);
  mpfr_add(cur_real_, real_base_, scratch1_, MPFR_RNDN);

  mpfr_mul_z(scratch1_, conj_slope_, b_adj.get_mpz_t(), MPFR_RNDN);
  mpfr_add(cur_conj_, conj_base_, scratch1_, MPFR_RNDN);
}

void OdgpStepper::refine_range_against_bounds(
    const mpfr_t &bound_lo, const mpfr_t &bound_hi, const mpfr_t &slope,
    const mpfr_t &base, Integer &range_lo, Integer &range_hi) {
  // Near-zero slope: there is no b that materially changes the value, so
  // either every b is in-bounds (no refinement) or none is (empty range).
  mpfr_abs(scratch1_, slope, MPFR_RNDN);
  if (mpfr_cmp_d(scratch1_, SLOPE_ZERO_TOLERANCE) <= 0) {
    if (mpfr_cmp(base, bound_lo) < 0 || mpfr_cmp(base, bound_hi) > 0)
      range_hi = range_lo - Integer(1);
    return;
  }

  mpfr_sub(scratch2_, bound_lo, base, MPFR_RNDN);
  mpfr_div(scratch2_, scratch2_, slope, MPFR_RNDN);
  Integer lo_new;
  mpfr_get_z(lo_new.get_mpz_t(), scratch2_, MPFR_RNDU);

  mpfr_sub(scratch2_, bound_hi, base, MPFR_RNDN);
  mpfr_div(scratch2_, scratch2_, slope, MPFR_RNDN);
  Integer hi_new;
  mpfr_get_z(hi_new.get_mpz_t(), scratch2_, MPFR_RNDD);

  // Negative slope inverts the [lo, hi] mapping computed above.
  if (mpfr_sgn(slope) < 0)
    std::swap(lo_new, hi_new);

  range_lo = std::max(range_lo, lo_new);
  range_hi = std::min(range_hi, hi_new);
}

//===----------------------------------------------------------------------===//
// OdgpWithParityStepper
//===----------------------------------------------------------------------===//

namespace {

/// Pre-rescale I for the parity transform: the inner OdgpStepper enumerates
/// in a coordinate where the parity constraint is folded into a shift and
/// a 1/sqrt(2) factor.
Interval scaled_parity_I(const Interval &I, int p) {
  return (I + (-static_cast<Real>(p))) * (Real::sqrt2() / 2);
}
Interval scaled_parity_J(const Interval &J, int p) {
  return (J + (-static_cast<Real>(p))) * (-Real::sqrt2() / 2);
}

} // namespace

OdgpWithParityStepper::OdgpWithParityStepper(Interval I, Interval J,
                                             ZSqrt2 parity_hint)
    : inner_(scaled_parity_I(I, parity_hint.parity()),
             scaled_parity_J(J, parity_hint.parity())),
      parity_p_(parity_hint.parity()) {
  SYNTH_OPEN_SUB("solve_odgp_with_parity");
  LLVM_DEBUG(cudaq::synth::dbgs() << "parity=" << parity_p_ << '\n');
}

OdgpWithParityStepper::~OdgpWithParityStepper() {
  if (yielded_ > 0)
    SYNTH_CLOSE_SUCCESS("yielded " + std::to_string(yielded_));
  else
    SYNTH_CLOSE_FAILURE("no solutions");
}

const ZSqrt2 *OdgpWithParityStepper::next() {
  const ZSqrt2 *alpha = inner_.next();
  if (!alpha)
    return nullptr;
  // Undo the per-iteration shift introduced by scaled_parity_I/J. Symbolically
  //   sol = alpha * ZSqrt2{0, 1} + ZSqrt2{p}
  //       = (a + b*sqrt(2))*sqrt(2) + p
  //       = (2b + p) + a*sqrt(2)
  // so the new coefficients are (2b + p, a). Computing them directly avoids
  // constructing the intermediate ZSqrt2 temporaries.
  Integer new_a = (alpha->b() << 1) + Integer(parity_p_);
  Integer new_b = alpha->a();
  last_sol_.assign(std::move(new_a), std::move(new_b));
  ++yielded_;
  return &last_sol_;
}

//===----------------------------------------------------------------------===//
// OdgpScaledStepper
//===----------------------------------------------------------------------===//

OdgpScaledStepper::OdgpScaledStepper(Interval I, Interval J, Integer denom_exp)
    : denom_exp_(std::move(denom_exp)) {
  SYNTH_OPEN_SUB("solve_odgp_scaled");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "denom_exp=" << static_cast<i64>(denom_exp_)
             << ", I_width=" << I.width() << ", J_width=" << J.width() << '\n');
  // Scale I and J by sqrt(2)^denom_exp; the J side picks up a sign flip for
  // odd exponents because the sqrt(2)-conjugation sends sqrt(2) -> -sqrt(2).
  Real scale = pow_sqrt2(denom_exp_);
  Interval scaled_I = I * scale;
  Interval scaled_J = (denom_exp_ & 1) ? J * (-scale) : J * scale;
  inner_.emplace(scaled_I, scaled_J);
}

OdgpScaledStepper::~OdgpScaledStepper() {
  if (yielded_ > 0)
    SYNTH_CLOSE_SUCCESS("yielded " + std::to_string(yielded_));
  else
    SYNTH_CLOSE_FAILURE("no solutions");
}

const DSqrt2 *OdgpScaledStepper::next() {
  const ZSqrt2 *alpha = inner_->next();
  if (!alpha)
    return nullptr;
  last_sol_.assign(*alpha, denom_exp_);
  ++yielded_;
  return &last_sol_;
}

//===----------------------------------------------------------------------===//
// OdgpScaledWithParityStepper
//===----------------------------------------------------------------------===//

OdgpScaledWithParityStepper::OdgpScaledWithParityStepper(
    Interval I, Interval J, Integer denom_exp, DSqrt2 parity_hint) {
  SYNTH_OPEN_SUB("solve_odgp_scaled_with_parity");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "denom_exp=" << static_cast<i64>(denom_exp)
             << ", parity=" << parity_hint << '\n');

  // denom_exp == 0 is the parity-only base case; positive exponents reduce
  // by one and absorb a 1/sqrt(2)^denom_exp offset to satisfy the parity
  // constraint at the next level down.
  if (denom_exp == 0) {
    ZSqrt2 beta_z = with_denom_exp(parity_hint, 0).alpha();
    direct_.emplace(I, J, beta_z);
    return;
  }

  int p = with_denom_exp(parity_hint, denom_exp).parity();
  offset_ = (p == 0) ? DSqrt2{0} : DSqrt2::power_of_inv_sqrt2(denom_exp);
  Interval shifted_I = I + (-to_real(offset_));
  Interval shifted_J = J + (-to_real(offset_.conj_sq2()));
  recursive_.emplace(shifted_I, shifted_J, denom_exp - 1);
}

OdgpScaledWithParityStepper::~OdgpScaledWithParityStepper() {
  if (yielded_ > 0)
    SYNTH_CLOSE_SUCCESS("yielded " + std::to_string(yielded_));
  else
    SYNTH_CLOSE_FAILURE("no solutions");
}

const DSqrt2 *OdgpScaledWithParityStepper::next() {
  if (direct_) {
    const ZSqrt2 *a = direct_->next();
    if (!a)
      return nullptr;
    last_sol_.assign(*a, Integer(0));
    ++yielded_;
    return &last_sol_;
  }
  const DSqrt2 *a = recursive_->next();
  if (!a)
    return nullptr;
  last_sol_ = *a + offset_;
  ++yielded_;
  return &last_sol_;
}

//===----------------------------------------------------------------------===//
// Factory functions
//===----------------------------------------------------------------------===//

OdgpStepper solve_odgp(Interval I, Interval J) {
  return OdgpStepper(std::move(I), std::move(J));
}

OdgpWithParityStepper solve_odgp_with_parity(Interval I, Interval J,
                                             ZSqrt2 parity_hint) {
  return OdgpWithParityStepper(std::move(I), std::move(J),
                               std::move(parity_hint));
}

OdgpScaledStepper solve_odgp_scaled(Interval I, Interval J, Integer denom_exp) {
  return OdgpScaledStepper(std::move(I), std::move(J), std::move(denom_exp));
}

OdgpScaledWithParityStepper
solve_odgp_scaled_with_parity(Interval I, Interval J, Integer denom_exp,
                              DSqrt2 parity_hint) {
  return OdgpScaledWithParityStepper(std::move(I), std::move(J),
                                     std::move(denom_exp),
                                     std::move(parity_hint));
}

} // namespace cudaq::synth
