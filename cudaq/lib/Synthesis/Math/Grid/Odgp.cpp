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

/// Computes n = ⌊log_y(x)⌋ and the remainder r = x / y^n.
std::pair<Integer, Real> floor_log(const Real &x, const Real &y) {
  assert(x > Real(0) && "floor_log: non-positive input (floating point error)");

  // Phase 1: bracket log_y(x) within [-2^m, 2^m), collecting the squaring
  // chain into a fixed array so Phase 2 doesn't recompute it.
  // m <= ~60 in practice (O(log log x) growth), so a fixed array is safe.
  std::array<Real, 64> pow_y;
  pow_y[0] = y;
  int m = 0;
  while (x >= pow_y[m] || x * pow_y[m] < Real(1)) {
    pow_y[m + 1] = pow_y[m] * pow_y[m]; // y^(2^(m+1))
    ++m;
  }

  // Phase 2: binary search — extract bits of n from high to low.
  // pow_y[m] = y^(2^m) used only for the x < 1 init; pow_y[0..m-1] for bits.
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

struct EnumerationScratch {
  mpfr_t orig_I_lo, orig_I_hi;
  mpfr_t orig_J_lo, orig_J_hi;
  mpfr_t real_base, real_slope;
  mpfr_t conj_base, conj_slope;
  mpfr_t real_step, conj_step;
  mpfr_t cur_real, cur_conj;
  mpfr_t scratch1, scratch2, scratch3;

  EnumerationScratch() {
    const mpfr_prec_t prec = Real::get_default_precision();
    mpfr_init2(orig_I_lo, prec);
    mpfr_init2(orig_I_hi, prec);
    mpfr_init2(orig_J_lo, prec);
    mpfr_init2(orig_J_hi, prec);
    mpfr_init2(real_base, prec);
    mpfr_init2(real_slope, prec);
    mpfr_init2(conj_base, prec);
    mpfr_init2(conj_slope, prec);
    mpfr_init2(real_step, prec);
    mpfr_init2(conj_step, prec);
    mpfr_init2(cur_real, prec);
    mpfr_init2(cur_conj, prec);
    mpfr_init2(scratch1, prec);
    mpfr_init2(scratch2, prec);
    mpfr_init2(scratch3, prec);
  }

  ~EnumerationScratch() {
    mpfr_clear(orig_I_lo);
    mpfr_clear(orig_I_hi);
    mpfr_clear(orig_J_lo);
    mpfr_clear(orig_J_hi);
    mpfr_clear(real_base);
    mpfr_clear(real_slope);
    mpfr_clear(conj_base);
    mpfr_clear(conj_slope);
    mpfr_clear(real_step);
    mpfr_clear(conj_step);
    mpfr_clear(cur_real);
    mpfr_clear(cur_conj);
    mpfr_clear(scratch1);
    mpfr_clear(scratch2);
    mpfr_clear(scratch3);
  }

  EnumerationScratch(const EnumerationScratch &) = delete;
  EnumerationScratch &operator=(const EnumerationScratch &) = delete;
};

// -----------------------------------------------------------------------------
// Lambda power cache
// -----------------------------------------------------------------------------
struct LambdaPowers {
  ZSqrt2 lambda_n;
  ZSqrt2 lambda_conj_n;
  ZSqrt2 lambda_inv_n;
  Real lambda_n_real;
  Real lambda_conj_n_real;
};

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

// -----------------------------------------------------------------------------
// Helper functions
// -----------------------------------------------------------------------------
static constexpr double SLOPE_ZERO_TOLERANCE = 1e-40;

void cache_interval_bounds(EnumerationScratch &s, const Interval &I,
                           const Interval &J) {
  mpfr_set(s.orig_I_lo, I.l().get_mpfr(), MPFR_RNDN);
  mpfr_set(s.orig_I_hi, I.r().get_mpfr(), MPFR_RNDN);
  mpfr_set(s.orig_J_lo, J.l().get_mpfr(), MPFR_RNDN);
  mpfr_set(s.orig_J_hi, J.r().get_mpfr(), MPFR_RNDN);

  // The inner-loop comparison (cur_real ∈ [orig_I_lo, orig_I_hi], etc.) uses
  // values computed incrementally: cur = base + b * slope, then cur += step.
  // Each MPFR RNDN operation contributes up to 0.5 ULP of error; with ~4 ops
  // in init_current_values + the loop step, total drift can reach ~2 ULP.
  //
  // Meanwhile, the cached bounds come from bbox/ellipse computations through a
  // different MPFR path. When the exact solution sits on the interval boundary
  // (e.g., w† = -1/√2 for the Hadamard inner TDGP), these two independent
  // representations of the same value can disagree by a few ULPs, causing the
  // strict comparison to reject a valid candidate.
  //
  // Widening by 4 ULPs covers the worst-case drift. The TDGP's exact
  // contains() check (DSqrt2/DOmega integer arithmetic) filters false
  // positives.
  for (int i = 0; i < 4; ++i) {
    mpfr_nextbelow(s.orig_I_lo);
    mpfr_nextabove(s.orig_I_hi);
    mpfr_nextbelow(s.orig_J_lo);
    mpfr_nextabove(s.orig_J_hi);
  }
}

void compute_slopes(EnumerationScratch &s, const Integer &scale_a,
                    const Integer &two_scale_b) {
  mpfr_set_z(s.real_slope, two_scale_b.get_mpz_t(), MPFR_RNDN);
  mpfr_mul_z(s.scratch1, Real::sqrt2().get_mpfr(), scale_a.get_mpz_t(),
             MPFR_RNDN);
  mpfr_add(s.real_slope, s.real_slope, s.scratch1, MPFR_RNDN);

  mpfr_set_z(s.conj_slope, two_scale_b.get_mpz_t(), MPFR_RNDN);
  mpfr_sub(s.conj_slope, s.conj_slope, s.scratch1, MPFR_RNDN);
}

void compute_linear_bases(EnumerationScratch &s, const Integer &offset_a,
                          const Integer &offset_b) {
  mpfr_set_z(s.real_base, offset_a.get_mpz_t(), MPFR_RNDN);
  mpfr_mul_z(s.scratch1, Real::sqrt2().get_mpfr(), offset_b.get_mpz_t(),
             MPFR_RNDN);
  mpfr_add(s.real_base, s.real_base, s.scratch1, MPFR_RNDN);

  mpfr_set_z(s.conj_base, offset_a.get_mpz_t(), MPFR_RNDN);
  mpfr_sub(s.conj_base, s.conj_base, s.scratch1, MPFR_RNDN);
}

void init_current_values(EnumerationScratch &s, const Integer &b_adj) {
  mpfr_mul_z(s.scratch1, s.real_slope, b_adj.get_mpz_t(), MPFR_RNDN);
  mpfr_add(s.cur_real, s.real_base, s.scratch1, MPFR_RNDN);

  mpfr_mul_z(s.scratch1, s.conj_slope, b_adj.get_mpz_t(), MPFR_RNDN);
  mpfr_add(s.cur_conj, s.conj_base, s.scratch1, MPFR_RNDN);
}

void refine_range_against_bounds(EnumerationScratch &s, const mpfr_t &bound_lo,
                                 const mpfr_t &bound_hi, const mpfr_t &slope,
                                 const mpfr_t &base, Integer &range_lo,
                                 Integer &range_hi) {
  mpfr_abs(s.scratch1, slope, MPFR_RNDN);
  if (mpfr_cmp_d(s.scratch1, SLOPE_ZERO_TOLERANCE) <= 0) {
    if (mpfr_cmp(base, bound_lo) < 0 || mpfr_cmp(base, bound_hi) > 0)
      range_hi = range_lo - Integer(1);
    return;
  }

  mpfr_sub(s.scratch2, bound_lo, base, MPFR_RNDN);
  mpfr_div(s.scratch2, s.scratch2, slope, MPFR_RNDN);
  Integer lo_new;
  mpfr_get_z(lo_new.get_mpz_t(), s.scratch2, MPFR_RNDU);

  mpfr_sub(s.scratch2, bound_hi, base, MPFR_RNDN);
  mpfr_div(s.scratch2, s.scratch2, slope, MPFR_RNDN);
  Integer hi_new;
  mpfr_get_z(hi_new.get_mpz_t(), s.scratch2, MPFR_RNDD);

  if (mpfr_sgn(slope) < 0)
    std::swap(lo_new, hi_new);

  range_lo = std::max(range_lo, lo_new);
  range_hi = std::min(range_hi, hi_new);
}

} // namespace

// -----------------------------------------------------------------------------
// Public API — lazy generators
// -----------------------------------------------------------------------------
generator<ZSqrt2> solve_odgp(Interval I, Interval J) {
  SYNTH_OPEN_SUB("solve_odgp");
  cudaq::synth::CloseGuard guard;
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "I_width=" << I.width() << ", J_width=" << J.width() << '\n');

  if (I.width() < 0 || J.width() < 0) {
    guard.fail("empty interval");
    co_return;
  }

  Integer shift_a = floor_to_integer((I.l() + J.l()) / 2);
  Integer shift_b = floor_to_integer(Real::sqrt2() * (I.l() - J.l()) / 4);
  ZSqrt2 shift(shift_a, shift_b);

  Interval cur_I = I - to_real(shift);
  Interval cur_J = J - to_real(shift.conj_sq2());

  ZSqrt2 scale{1};
  bool swapped = false;

  if (cur_I.width() > 0 && cur_J.width() <= 0) {
    std::swap(cur_I, cur_J);
    swapped = true;
  }

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
             << ", cur_J_width=" << cur_J.width() << ", swapped=" << swapped
             << '\n');

  // --- Enumeration (inlined from enumerate_solutions) ---
  EnumerationScratch s;
  cache_interval_bounds(s, I, J);

  const Integer &scale_a = scale.a();
  const Integer &scale_b = scale.b();
  const Integer &shift_a_ref = shift.a();
  const Integer &shift_b_ref = shift.b();

  Integer two_scale_b = scale_b * Integer(2);
  int direction = swapped ? -1 : 1;

  compute_slopes(s, scale_a, two_scale_b);

  if (direction > 0) {
    mpfr_set(s.real_step, s.real_slope, MPFR_RNDN);
    mpfr_set(s.conj_step, s.conj_slope, MPFR_RNDN);
  } else {
    mpfr_neg(s.real_step, s.real_slope, MPFR_RNDN);
    mpfr_neg(s.conj_step, s.conj_slope, MPFR_RNDN);
  }

  Integer a_min = ceil_to_integer((cur_I.l() + cur_J.l()) / 2);
  Integer a_max = floor_to_integer((cur_I.r() + cur_J.r()) / 2);

  LLVM_DEBUG(cudaq::synth::dbgs()
             << "a_min=" << a_min << ", a_max=" << a_max << '\n');

  Integer delta_result_a = (direction > 0) ? two_scale_b : -two_scale_b;
  Integer delta_result_b = (direction > 0) ? scale_a : -scale_a;

  int yielded = 0;
  for (Integer a = a_min; a <= a_max; ++a) {
    Integer b_lo = ceil_to_integer(Real::sqrt2() * (a - cur_J.r()) / 2);
    Integer b_hi = floor_to_integer(Real::sqrt2() * (a - cur_J.l()) / 2);
    if (b_hi < b_lo)
      continue;

    Integer base_a = a * scale_a;
    Integer base_b = a * scale_b;
    Integer offset_a = base_a + shift_a_ref;
    Integer offset_b = base_b + shift_b_ref;

    compute_linear_bases(s, offset_a, offset_b);

    Integer b_adj_lo = swapped ? -b_hi : b_lo;
    Integer b_adj_hi = swapped ? -b_lo : b_hi;

    refine_range_against_bounds(s, s.orig_I_lo, s.orig_I_hi, s.real_slope,
                                s.real_base, b_adj_lo, b_adj_hi);
    if (b_adj_hi < b_adj_lo)
      continue;

    refine_range_against_bounds(s, s.orig_J_lo, s.orig_J_hi, s.conj_slope,
                                s.conj_base, b_adj_lo, b_adj_hi);
    if (b_adj_hi < b_adj_lo)
      continue;

    if (swapped) {
      b_lo = std::max(b_lo, Integer(-b_adj_hi));
      b_hi = std::min(b_hi, Integer(-b_adj_lo));
    } else {
      b_lo = std::max(b_lo, Integer(b_adj_lo));
      b_hi = std::min(b_hi, Integer(b_adj_hi));
    }
    if (b_hi < b_lo)
      continue;

    Integer b_adj_start = swapped ? -b_lo : b_lo;
    Integer result_a = offset_a + two_scale_b * b_adj_start;
    Integer result_b = offset_b + scale_a * b_adj_start;

    init_current_values(s, b_adj_start);

    for (Integer b = b_lo; b <= b_hi; ++b) {
      if (mpfr_cmp(s.cur_real, s.orig_I_lo) >= 0 &&
          mpfr_cmp(s.cur_real, s.orig_I_hi) <= 0 &&
          mpfr_cmp(s.cur_conj, s.orig_J_lo) >= 0 &&
          mpfr_cmp(s.cur_conj, s.orig_J_hi) <= 0) {
        ZSqrt2 sol(result_a, result_b);
        ++yielded;
        co_yield sol;
      }

      result_a += delta_result_a;
      result_b += delta_result_b;
      mpfr_add(s.cur_real, s.cur_real, s.real_step, MPFR_RNDN);
      mpfr_add(s.cur_conj, s.cur_conj, s.conj_step, MPFR_RNDN);
    }
  }
  if (yielded > 0)
    guard.succeed("yielded " + std::to_string(yielded));
  else
    guard.fail("no solutions");
}

generator<ZSqrt2> solve_odgp_with_parity(Interval I, Interval J,
                                         ZSqrt2 parity_hint) {
  SYNTH_OPEN_SUB("solve_odgp_with_parity");
  cudaq::synth::CloseGuard guard;
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "parity=" << parity_hint.parity() << '\n');
  int p = parity_hint.parity();
  Interval scaled_I = (I + (-static_cast<Real>(p))) * (Real::sqrt2() / 2);
  Interval scaled_J = (J + (-static_cast<Real>(p))) * (-Real::sqrt2() / 2);

  int yielded = 0;
  for (const ZSqrt2 &alpha : solve_odgp(scaled_I, scaled_J)) {
    ZSqrt2 sol = alpha * ZSqrt2{0, 1} + ZSqrt2{p};
    ++yielded;
    co_yield sol;
  }
  if (yielded > 0)
    guard.succeed("yielded " + std::to_string(yielded));
  else
    guard.fail("no solutions");
}

generator<DSqrt2> solve_odgp_scaled(Interval I, Interval J, Integer denom_exp) {
  SYNTH_OPEN_SUB("solve_odgp_scaled");
  cudaq::synth::CloseGuard guard;
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "denom_exp=" << static_cast<i64>(denom_exp)
             << ", I_width=" << I.width() << ", J_width=" << J.width() << '\n');
  Real scale = pow_sqrt2(denom_exp);
  Interval scaled_I = I * scale;
  Interval scaled_J = (denom_exp & 1) ? J * (-scale) : J * scale;

  int yielded = 0;
  for (const ZSqrt2 &alpha : solve_odgp(scaled_I, scaled_J)) {
    DSqrt2 sol(alpha, denom_exp);
    ++yielded;
    co_yield sol;
  }
  if (yielded > 0)
    guard.succeed("yielded " + std::to_string(yielded));
  else
    guard.fail("no solutions");
}

generator<DSqrt2> solve_odgp_scaled_with_parity(Interval I, Interval J,
                                                Integer denom_exp,
                                                DSqrt2 parity_hint) {
  SYNTH_OPEN_SUB("solve_odgp_scaled_with_parity");
  cudaq::synth::CloseGuard guard;
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "denom_exp=" << static_cast<i64>(denom_exp)
             << ", parity=" << parity_hint << '\n');

  int yielded = 0;
  if (denom_exp == 0) {
    ZSqrt2 beta_z = with_denom_exp(parity_hint, 0).alpha();
    for (const ZSqrt2 &a : solve_odgp_with_parity(I, J, beta_z)) {
      DSqrt2 sol = DSqrt2::from_zsqrt2(a);
      ++yielded;
      co_yield sol;
    }
    if (yielded > 0)
      guard.succeed("yielded " + std::to_string(yielded));
    else
      guard.fail("no solutions");
    co_return;
  }

  int p = with_denom_exp(parity_hint, denom_exp).parity();
  DSqrt2 offset = (p == 0) ? DSqrt2{0} : DSqrt2::power_of_inv_sqrt2(denom_exp);
  Interval shifted_I = I + (-to_real(offset));
  Interval shifted_J = J + (-to_real(offset.conj_sq2()));

  for (const DSqrt2 &a :
       solve_odgp_scaled(shifted_I, shifted_J, denom_exp - 1)) {
    DSqrt2 sol = a + offset;
    ++yielded;
    co_yield sol;
  }
  if (yielded > 0)
    guard.succeed("yielded " + std::to_string(yielded));
  else
    guard.fail("no solutions");
}

} // namespace cudaq::synth
