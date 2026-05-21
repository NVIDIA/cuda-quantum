/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Math/Diophantine.h"
#include "Support/StreamOps.h"
#include "cudaq/Synthesis/Math/Integer.h"
#include "cudaq/Synthesis/Math/Ring/Zomega.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <random>
#include <variant>
#include <vector>

//===----------------------------------------------------------------------===//
// Diophantine solver for conj(t)*t = xi   (Appendix C of arXiv:1403.2975)
//===----------------------------------------------------------------------===//
//
// Given xi in D[sqrt(2)], find t in D[omega] with conj(t) * t = xi, or
// determine that no such t exists. Two distinct involutions appear together
// throughout the algorithm:
//
//   conj(z)         -- complex conjugation on Z[omega] / D[omega] (the
//                      paper's (-)^dagger). Sends omega to omega^-1.
//   conj_sq2(z)     -- sqrt(2)-conjugation (the paper's (-)^bullet).
//                      Sends sqrt(2) to -sqrt(2) and fixes the imaginary
//                      unit.
//
// Algorithm shape (Theorem 6.2 / Proposition C.24):
//
//   1. Necessary conditions: xi >= 0 and conj_sq2(xi) >= 0       (Lemma 6.1)
//   2. Reduce D[sqrt(2)] -> Z[sqrt(2)] via delta-scaling         (Lemma C.25)
//   3. Split xi into a self-associate part d ~ conj_sq2(d) and
//      a self-coprime part eta with gcd(eta, conj_sq2(eta)) ~ 1  (Lemma C.23)
//   4. Factor through integer primes -> Z[sqrt(2)]-primes
//      -> Z[omega]                                               (Lemmas
//      C.8-C.13)
//   5. adj-decompose each prime factor                           (Lemma C.20)
//   6. Combine partial solutions, adjust the residual unit       (Lemma C.16)
//
// The only super-polynomial step is integer factoring, handled here via a
// Pollard-Brent rho heuristic with a caller-supplied timeout.

#define DEBUG_TYPE "cudaq-synth"

using namespace cudaq::synth;

namespace {

//===----------------------------------------------------------------------===//
// Result type
//===----------------------------------------------------------------------===//

/// Outcome of an internal adj-decomposition step. The "NeedFactoring" arm
/// covers the case where the prime classification could not be decided
/// without further integer factoring -- the caller is expected to factor
/// more aggressively and retry.
struct NoSolution {};
struct NeedFactoring {};
using DiophantineResult = std::variant<ZOmega, NoSolution, NeedFactoring>;

inline bool is_no_solution(const DiophantineResult &r) {
  return std::holds_alternative<NoSolution>(r);
}
inline bool is_need_factoring(const DiophantineResult &r) {
  return std::holds_alternative<NeedFactoring>(r);
}
inline bool is_success(const DiophantineResult &r) {
  return std::holds_alternative<ZOmega>(r);
}

//===----------------------------------------------------------------------===//
// Random number generator
//===----------------------------------------------------------------------===//

/// RAII wrapper around GMP's Mersenne Twister random state.
///
/// gmp_randstate_t is not copyable; each thread gets its own instance via
/// global_rng() seeded once at construction. All subsequent calls reuse the
/// same state, so the per-call cost is just the GMP iteration step.
struct GmpRng {
  gmp_randstate_t state;

  GmpRng() {
    gmp_randinit_mt(state);
    // Pack two 32-bit random_device samples into a single 64-bit seed.
    // unsigned long is 64-bit on LP64, so gmp_randseed_ui takes the seed
    // directly without going through an mpz_t temporary.
    std::random_device rd;
    const unsigned long seed = (static_cast<unsigned long>(rd()) << 32) | rd();
    gmp_randseed_ui(state, seed);
  }

  ~GmpRng() { gmp_randclear(state); }

  GmpRng(const GmpRng &) = delete;
  GmpRng &operator=(const GmpRng &) = delete;
};

/// Returns the thread-local GMP random state. Construction (and therefore
/// seeding) happens exactly once per thread; destruction (gmp_randclear)
/// runs at thread exit.
GmpRng &global_rng() {
  static thread_local GmpRng rng;
  return rng;
}

//===----------------------------------------------------------------------===//
// Utility
//===----------------------------------------------------------------------===//

/// Number of decimal digits in |n|. Wrapper over mpz_sizeinbase used by the
/// debug-logging paths that report factoring progress.
size_t num_decimal_digits(const Integer &n) {
  return static_cast<size_t>(mpz_sizeinbase(n.get_mpz_t(), 10));
}

//===----------------------------------------------------------------------===//
// Modular arithmetic
//===----------------------------------------------------------------------===//

/// Modular exponentiation via GMP. Returns 0 for the trivial moduli m in
/// {0, 1} so callers do not have to guard the corner cases.
[[maybe_unused]] Integer mod_pow(const Integer &base, const Integer &exp,
                                 const Integer &mod) {
  if (mod == 0 || mod == 1)
    return Integer(0);
  Integer b_norm = base;
  mpz_mod(b_norm.get_mpz_t(), b_norm.get_mpz_t(), mod.get_mpz_t());
  Integer result;
  mpz_powm(result.get_mpz_t(), b_norm.get_mpz_t(), exp.get_mpz_t(),
           mod.get_mpz_t());
  return result;
}

/// Uniform random integer in [low, high] using GMP's arbitrary-precision RNG.
///
/// mpz_urandomm produces a uniform value in [0, range); shifting by low maps
/// it into the requested range. `range` is constructed locally so that the
/// temporary lives for the duration of the mpz_urandomm call.
Integer urand_between(const Integer &low, const Integer &high, GmpRng &rng) {
  Integer range = high - low + 1;
  Integer result;
  mpz_urandomm(result.get_mpz_t(), rng.state, range.get_mpz_t());
  return result + low;
}

//===----------------------------------------------------------------------===//
// Extension field F_p^2 = F_p[x] / (x^2 - base)   (for root_mod / Cipolla)
//===----------------------------------------------------------------------===//
//
// Cipolla's algorithm computes square roots mod p by working in the quadratic
// extension F_p^2 = F_p[x] / (x^2 - delta), where delta is a quadratic
// non-residue mod p. An element a + b*x is represented as a pair (a, b) with
// the multiplication rule x^2 = delta.
//
// The ring parameters (the prime p and the non-residue delta, called `base`
// because x^2 equals it in the extension) are captured in Fp2Ctx and passed
// explicitly to every arithmetic operation. This avoids mutable global state
// and makes the code thread-safe with no temporal coupling between calls.
// Fp2 itself is a plain aggregate with no constructor normalization -- the
// arithmetic helpers cooperate to keep coefficients reduced mod p on entry
// and on exit.
//
// Reference: Rabin [12]; Ross & Selinger sec. 8 / Algorithm 7.6 step 2(b).

/// Ring context: holds references to the prime p and the non-residue delta.
/// Both must outlive every Fp2 element and every fp2_mul / fp2_pow call that
/// uses this context.
struct Fp2Ctx {
  const Integer &p;
  const Integer &base;
};

/// Plain aggregate a + b*x in F_p^2 with the invariant that both coefficients
/// are in [0, p). Helpers below preserve the invariant on every input/output.
struct Fp2 {
  Integer a;
  Integer b;
};

/// Multiply two F_p^2 elements:
///   (a_1 + b_1 x) * (a_2 + b_2 x)
///     = (a_1 a_2 + b_1 b_2 delta) + (a_1 b_2 + b_1 a_2) x
/// All intermediate products are reduced mod p exactly once.
Fp2 fp2_mul(const Fp2Ctx &ctx, const Fp2 &lhs, const Fp2 &rhs) {
  Integer bb_mod = lhs.b * rhs.b % ctx.p;
  Integer new_a = (lhs.a * rhs.a + bb_mod * ctx.base) % ctx.p;
  Integer new_b = (lhs.a * rhs.b + lhs.b * rhs.a) % ctx.p;
  return {std::move(new_a), std::move(new_b)};
}

/// Square-and-multiply in F_p^2. The exponent is consumed by value, shifted
/// right per iteration, so the caller's copy is destroyed.
///
/// Complexity: O(log e) Fp2 multiplications, each O(M(log p)) where M is
/// GMP's multiplication cost.
Fp2 fp2_pow(const Fp2Ctx &ctx, Fp2 base_elem, Integer e) {
  assert(!(e < 0) && "fp2_pow: negative exponent");
  Fp2 result{Integer(1), Integer(0)};
  while (e > 0) {
    if (e.is_odd())
      result = fp2_mul(ctx, result, base_elem);
    base_elem = fp2_mul(ctx, base_elem, base_elem);
    e >>= 1;
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Integer factoring  (Pollard-Brent rho)
//===----------------------------------------------------------------------===//

/// Pollard-Brent rho: return a non-trivial factor of n, or failure if either
/// the iteration budget L or the wall-clock timeout is exhausted.
///
/// Used as the factoring sub-oracle in the Diophantine solver -- the paper
/// notes this is the only super-polynomial step (sec. 8, Algorithm 7.6 step
/// 2b). Heuristic; the iteration cap is set from a digit-count power law.
llvm::FailureOr<Integer> find_factor(const Integer &n,
                                     int32_t factoring_timeout_ms,
                                     int32_t batch_size = 128) {
  CUDAQ_SYNTH_OPEN_SUB("find_factor");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "n has " << num_decimal_digits(n)
             << " digits, timeout=" << factoring_timeout_ms << "ms\n");
  // Quick trial-division pass: catches every n with a tiny prime factor
  // without spinning up the rho machine. Cast to unsigned long matches the
  // GMP mpz_divisible_ui_p ABI; the values fit losslessly on LP64.
  static constexpr uint64_t small_primes[] = {
      2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,
      43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101,
      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
      173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
      241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311};
  const auto *n_mpz = n.get_mpz_t();
  for (uint64_t p : small_primes) {
    if (mpz_divisible_ui_p(n_mpz, static_cast<unsigned long>(p))) {
      if (mpz_cmp_ui(n_mpz, static_cast<unsigned long>(p)) > 0) {
        CUDAQ_SYNTH_CLOSE_SUCCESS("small prime " + std::to_string(p));
        return Integer(static_cast<int64_t>(p));
      }
    }
  }
  if (n <= int64_t(3)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("n <= 3");
    return llvm::failure();
  }

  // Iteration budget heuristic: L ~= 10^(digits/4) * 1.1774. The constant
  // comes from sqrt(pi/log(2)) / 2 in the textbook rho complexity estimate.
  size_t digits = num_decimal_digits(n);
  double pow_term = std::pow(10.0, static_cast<double>(digits) / 4.0);
  int64_t L = static_cast<int64_t>(pow_term * 1.1774 + 10.0);

  GmpRng &rng = global_rng();
  Integer a_int = urand_between(1, n - 1, rng);
  const auto *a_mpz = a_int.get_mpz_t();

  // Pre-allocate every GMP temporary used in the hot loop so the inner
  // batches do not pay an mpz_init/mpz_clear pair per step.
  Integer y_val, x_val, q_val, y0_val, tmp, diff, g_val;
  auto *y = y_val.get_mpz_t();
  auto *x = x_val.get_mpz_t();
  auto *q = q_val.get_mpz_t();
  auto *y0 = y0_val.get_mpz_t();
  auto *t = tmp.get_mpz_t();
  auto *d = diff.get_mpz_t();
  auto *g = g_val.get_mpz_t();

  mpz_set(y, a_mpz);

  int64_t r = 1, k = 0;
  auto start = std::chrono::steady_clock::now();

  auto make_result = [&](mpz_t src) -> llvm::FailureOr<Integer> {
    Integer out;
    mpz_set(out.get_mpz_t(), src);
    return out;
  };

  while (true) {
    // Brent phase: save x = y + n so that x - y stays non-negative, then
    // advance the hare y in batched steps with a single GCD per batch.
    mpz_add(x, y, n_mpz);

    while (k < r) {
      mpz_set_ui(q, 1);
      mpz_set(y0, y);

      int64_t batch_end = std::min(k + static_cast<int64_t>(batch_size), r);
      for (; k < batch_end; ++k) {
        // y <- (y^2 + a) mod n
        mpz_mul(t, y, y);
        mpz_add(t, t, a_mpz);
        mpz_mod(y, t, n_mpz);

        // q <- q * (x - y) mod n
        mpz_sub(d, x, y);
        mpz_mul(t, q, d);
        mpz_mod(q, t, n_mpz);
      }

      mpz_gcd(g, q, n_mpz);

      if (mpz_cmp_ui(g, 1) != 0) {
        if (mpz_cmp(g, n_mpz) == 0) {
          // GCD collapsed to n: the batched product picked up too many
          // factors at once. Replay the batch step-by-step from the saved
          // y0 to recover the first non-trivial GCD.
          mpz_set(y, y0);
          for (int64_t j = 0; j < batch_size; ++j) {
            mpz_mul(t, y, y);
            mpz_add(t, t, a_mpz);
            mpz_mod(y, t, n_mpz);
            mpz_sub(d, x, y);
            mpz_gcd(g, d, n_mpz);
            if (mpz_cmp_ui(g, 1) != 0) {
              if (mpz_cmp(g, n_mpz) == 0) {
                CUDAQ_SYNTH_CLOSE_FAILURE("backtrack collapsed");
                return llvm::failure();
              }
              CUDAQ_SYNTH_CLOSE_SUCCESS("Pollard-Brent backtrack");
              return make_result(g);
            }
          }
          CUDAQ_SYNTH_CLOSE_FAILURE("backtrack exhausted");
          return llvm::failure();
        }
        CUDAQ_SYNTH_CLOSE_SUCCESS("Pollard-Brent rho");
        return make_result(g);
      }

      auto now = std::chrono::steady_clock::now();
      if (k >= L ||
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                  .count() >= factoring_timeout_ms) {
        LLVM_DEBUG(cudaq::synth::dbgs()
                   << "exhausted budget for " << digits
                   << "-digit number (L=" << L << ", k=" << k << ")\n");
        CUDAQ_SYNTH_CLOSE_FAILURE("budget exhausted (L=" + std::to_string(L) +
                                  ", k=" + std::to_string(k) + ")");
        return llvm::failure();
      }
    }
    r <<= 1;
  }
}

//===----------------------------------------------------------------------===//
// Square roots in Z_p
//===----------------------------------------------------------------------===//

/// Find x with x^2 == -1 (mod p) for a prime p == 1 (mod 4).
///
/// Samples b uniformly from [1, p-1] and checks whether b^((p-1)/4) is a
/// fourth root of unity whose square is -1. Each trial succeeds with
/// probability >= 1/2 (Lemma C.20 / Remark C.22, citing Rabin [12]). A
/// non-trivial Fermat witness aborts: it proves p is composite.
llvm::FailureOr<Integer> sqrt_negative_one(const Integer &p,
                                           int32_t batch_size = 128) {
  if (p <= 2)
    return llvm::failure();

  const auto *p_mpz = p.get_mpz_t();
  Integer exp = (p - 1) >> 2; // (p-1)/4
  Integer p_minus_1 = p - 1;
  const auto *exp_mpz = exp.get_mpz_t();
  const auto *p_minus_1_mpz = p_minus_1.get_mpz_t();

  Integer b, h, r, tmp;
  GmpRng &rng = global_rng();

  for (int32_t i = 0; i < batch_size; ++i) {
    b = urand_between(1, p_minus_1, rng);
    // h = b^((p-1)/4) mod p; then check r = h^2 == -1 mod p.
    mpz_powm(h.get_mpz_t(), b.get_mpz_t(), exp_mpz, p_mpz);
    mpz_mul(tmp.get_mpz_t(), h.get_mpz_t(), h.get_mpz_t());
    mpz_mod(r.get_mpz_t(), tmp.get_mpz_t(), p_mpz);

    if (mpz_cmp(r.get_mpz_t(), p_minus_1_mpz) == 0)
      return h;
    if (mpz_cmp_ui(r.get_mpz_t(), 1) != 0)
      return llvm::failure(); // r is neither 1 nor -1 -- p is composite.
  }
  return llvm::failure();
}

/// Find y with y^2 == x (mod p) via Cipolla's algorithm in F_p^2.
///
/// First applies the Euler criterion (x^((p-1)/2) == 1 mod p) to confirm x
/// is a quadratic residue. Then searches for a random b such that b^2 - x
/// is a non-residue, and computes the answer in F_p[t]/(t^2 - (b^2 - x))
/// (Rabin [12]).
llvm::FailureOr<Integer> root_mod(const Integer &x, const Integer &p,
                                  int32_t batch_size = 128) {
  Integer x_norm = x % p;
  if (x_norm < 0)
    x_norm += p;

  if (p == 2)
    return x_norm;
  if (x_norm == 0)
    return Integer(0);
  if (!(p.is_odd()) && p > 2)
    return llvm::failure(); // even "prime" > 2 -- bail out

  const auto *p_mpz = p.get_mpz_t();
  const auto *x_norm_mpz = x_norm.get_mpz_t();

  Integer exp_half = (p - 1) / 2;
  Integer p_minus_1 = p - 1;
  Integer power = (p + 1) / 2; // Cipolla exponent
  const auto *exp_half_mpz = exp_half.get_mpz_t();
  const auto *p_minus_1_mpz = p_minus_1.get_mpz_t();

  // Euler criterion: x^((p-1)/2) must equal 1 for x to be a QR mod p.
  Integer t;
  mpz_powm(t.get_mpz_t(), x_norm_mpz, exp_half_mpz, p_mpz);
  if (mpz_cmp_ui(t.get_mpz_t(), 1) != 0)
    return llvm::failure();

  Integer b, r, candidate_base, check, tmp;
  auto &rng = global_rng();

  for (int32_t i = 0; i < batch_size; ++i) {
    b = urand_between(1, p_minus_1, rng);

    // Fermat test: b^(p-1) must be 1 if p is genuinely prime. A failure
    // is a witness that p is composite.
    mpz_powm(r.get_mpz_t(), b.get_mpz_t(), p_minus_1_mpz, p_mpz);
    if (mpz_cmp_ui(r.get_mpz_t(), 1) != 0)
      return llvm::failure();

    // candidate_base = b^2 - x (mod p), the "base" delta for the F_p^2
    // extension. Add p before subtracting to keep the intermediate positive.
    mpz_mul(tmp.get_mpz_t(), b.get_mpz_t(), b.get_mpz_t());
    mpz_add(tmp.get_mpz_t(), tmp.get_mpz_t(), p_mpz);
    mpz_sub(tmp.get_mpz_t(), tmp.get_mpz_t(), x_norm_mpz);
    mpz_mod(candidate_base.get_mpz_t(), tmp.get_mpz_t(), p_mpz);

    // Re-check Euler criterion: candidate_base must be a *non-residue* for
    // the F_p^2 construction to be valid.
    mpz_powm(check.get_mpz_t(), candidate_base.get_mpz_t(), exp_half_mpz,
             p_mpz);
    if (mpz_cmp_ui(check.get_mpz_t(), 1) != 0) {
      // Cipolla: y = (b + t)^((p+1)/2) in F_p[t]/(t^2 - candidate_base).
      // The real component of the result is the desired square root of x
      // mod p.
      Fp2Ctx ctx{p, candidate_base};
      Fp2 rfp = fp2_pow(ctx, Fp2{b % p, Integer(1)}, power);
      return std::move(rfp.a);
    }
  }
  return llvm::failure();
}

//===----------------------------------------------------------------------===//
// Coprime factorization helpers
//===----------------------------------------------------------------------===//

using Factor = std::pair<Integer, Integer>;

/// Rewrite a product prod(b_i ^ k_i) over Z into u * prod(c_j ^ e_j) where
/// u is a unit (+/-1) and the bases c_j are pairwise coprime. This is *not*
/// full prime factorization: Lemma C.19 says adj-decomposability factors
/// over coprime parts, so pairwise coprimality of the bases is sufficient.
std::pair<Integer, std::vector<Factor>>
decompose_into_coprime_factors(const std::vector<Factor> &factors) {
  Integer unit = 1;

  std::vector<Factor> pending(factors.rbegin(), factors.rend());

  std::vector<Factor> coprime_factors;
  coprime_factors.reserve(factors.size());
  while (!pending.empty()) {
    auto [b, k_b] = pending.back();
    pending.pop_back();

    // +/-1 bases collapse into the unit sign and never enter the table.
    if (b == 1)
      continue;
    if (b == -1 && (k_b & 1)) {
      unit = -unit;
      continue;
    }

    size_t i = 0;
    while (true) {
      // Walked off the end without finding overlap -- record b as a fresh
      // coprime factor.
      if (i >= coprime_factors.size()) {
        coprime_factors.emplace_back(b, k_b);
        break;
      }

      auto &&[a, k_a] = coprime_factors[i];

      // Case 1: bases agree up to sign. Merge exponents; if a = -b, an odd
      // power of b contributes an extra -1 to the unit.
      if (a == b || a == -b) {
        if (a == -b && (k_b & 1))
          unit = -unit;
        k_a += k_b;
        break;
      }

      // Case 2: already coprime -- move on to the next existing base.
      Integer g = gcd(a, b);
      if (g == 1 || g == -1) {
        ++i;
        continue;
      }

      // Case 3: a and b share a non-trivial divisor g. Factor a as
      //   a = g * (a/g)
      // and rewrite a^k_a * b^k_b = (a/g)^k_a * g^(k_a + k_b) * (b/g)^k_b.
      // Recursively coprime-decompose the first two terms; push (b/g, k_b)
      // back onto the pending stack to be handled in a subsequent round.
      std::vector<Factor> partial = {{a / g, k_a}, {g, k_a + k_b}};
      auto [partial_unit, partial_factors] =
          decompose_into_coprime_factors(partial);

      unit *= partial_unit;

      coprime_factors[i] = partial_factors[0];
      coprime_factors.insert(coprime_factors.end(), partial_factors.begin() + 1,
                             partial_factors.end());

      pending.emplace_back(b / g, k_b);
      break;
    }
  }
  return {unit, coprime_factors};
}

using ZSqrt2Factor = std::pair<ZSqrt2, Integer>;

/// Pairwise-coprime decomposition for Z[sqrt(2)] factors. Structurally
/// identical to the integer variant above; differences are confined to the
/// associate-equivalence test (are_associates instead of ==) and the use of
/// the Z[sqrt(2)] gcd.
std::pair<ZSqrt2, std::vector<ZSqrt2Factor>>
decompose_into_coprime_factors(const std::vector<ZSqrt2Factor> &factors) {
  ZSqrt2 unit{1};
  std::vector<ZSqrt2Factor> pending(factors.rbegin(), factors.rend());

  std::vector<ZSqrt2Factor> coprime_factors;
  coprime_factors.reserve(factors.size());
  while (!pending.empty()) {
    auto [b, k_b] = pending.back();
    pending.pop_back();

    if (are_associates(b, ZSqrt2{1})) {
      for (Integer j = 0; j < k_b; ++j)
        unit = unit * b;
      continue;
    }

    size_t i = 0;
    while (true) {
      if (i >= coprime_factors.size()) {
        coprime_factors.emplace_back(b, k_b);
        break;
      }
      auto &&[a, k_a] = coprime_factors[i];
      if (are_associates(a, b)) {
        ZSqrt2 quotient = b / a;
        for (Integer j = 0; j < k_b; ++j)
          unit = unit * quotient;
        k_a += k_b;
        break;
      }
      ZSqrt2 g = gcd(a, b);
      if (are_associates(g, ZSqrt2{1})) {
        ++i;
        continue;
      }
      std::vector<ZSqrt2Factor> partial = {{a / g, k_a}, {g, k_a + k_b}};
      auto [partial_unit, partial_factors] =
          decompose_into_coprime_factors(partial);
      unit = unit * partial_unit;
      coprime_factors[i] = partial_factors[0];
      coprime_factors.insert(coprime_factors.end(), partial_factors.begin() + 1,
                             partial_factors.end());
      pending.emplace_back(b / g, k_b);
      break;
    }
  }
  return {unit, coprime_factors};
}

//===----------------------------------------------------------------------===//
// adj-decomposition of integer primes   (Lemma C.20)
//===----------------------------------------------------------------------===//

/// adj-decompose a positive integer prime p: find t in Z[omega] with
/// conj(t) * t ~ p (up to units in Z[sqrt(2)]).
///
/// Case split by p mod 8 (Lemma C.20):
///
///   p == 2:         delta * conj(delta) = lambda * sqrt(2) ~ sqrt(2), and
///                   sqrt(2) * sqrt(2) = 2. The returned t is i*sqrt(2),
///                   which in the Z[omega] basis is omega + omega^3
///                   = ZOmega(-1, 0, 1, 0).
///   p == 1 (mod 4): -1 is a QR mod p. Solve h^2 == -1 (mod p) and take
///                   t = gcd(h + i, p) in Z[omega].
///   p == 3 (mod 8): -2 is a QR mod p. Solve h^2 == -2 (mod p) and take
///                   t = gcd(h + i*sqrt(2), p), where i*sqrt(2) is
///                   ZOmega(1, 0, 1, 0).
///   p == 7 (mod 8): NOT adj-decomposable (Lemma C.20). We confirm by
///                   checking that 2 is a QR mod p -- if so, return
///                   NoSolution. If primality was wrong, ask for further
///                   factoring.
///
/// If the probabilistic primality test is misleading, the function returns
/// NeedFactoring so the caller can attempt deeper factoring.
DiophantineResult adj_decompose_prime(Integer p) {
  if (p < 0)
    p = -p;
  if (p == 0 || p == 1)
    return ZOmega::from_int(p);
  if (p == 2)
    return ZOmega(-1, 0, 1, 0);

  LLVM_DEBUG(cudaq::synth::dbgs()
             << "adj_decompose_prime(int): p mod 8 = "
             << static_cast<int64_t>(p & 7) << ", " << num_decimal_digits(p)
             << " digits, prime=" << is_probably_prime(p) << '\n');

  if (is_probably_prime(p)) {
    if ((p & 0b11) == 1) {
      // p == 1 (mod 4) -- covers both p == 1 and p == 5 (mod 8).
      llvm::FailureOr<Integer> h = sqrt_negative_one(p);
      if (llvm::failed(h))
        return NeedFactoring{};
      ZOmega t =
          gcd(ZOmega(0, 1, 0, 0) + ZOmega(0, 0, 0, *h), ZOmega::from_int(p));
      // Sanity check: conj(t) * t should equal +/-p (up to a unit). If not,
      // the gcd picked up the wrong representative and we need more work.
      if (t.conj() * t == ZOmega::from_int(p) ||
          t.conj() * t == ZOmega::from_int(-p))
        return t;
      return NeedFactoring{};
    }

    if ((p & 0b111) == 3) {
      llvm::FailureOr<Integer> h = root_mod(-2, p);
      if (llvm::failed(h))
        return NeedFactoring{};
      ZOmega t =
          gcd(ZOmega(1, 0, 1, 0) + ZOmega(0, 0, 0, *h), ZOmega::from_int(p));
      if (t.conj() * t == ZOmega::from_int(p) ||
          t.conj() * t == ZOmega::from_int(-p))
        return t;
      return NeedFactoring{};
    }

    if ((p & 0b111) == 7) {
      // Confirm the no-solution case: 2 must be a QR mod p.
      llvm::FailureOr<Integer> h = root_mod(2, p);
      if (llvm::succeeded(h))
        return NoSolution{};
      return NeedFactoring{};
    }

    return NeedFactoring{};
  }

  // p is composite (or the primality test got an inconclusive answer); the
  // p == 7 (mod 8) test above is still meaningful and can short-circuit to
  // NoSolution. Anything else requires further factoring.
  if ((p & 0b111) == 7) {
    llvm::FailureOr<Integer> h = root_mod(2, p);
    if (llvm::succeeded(h))
      return NoSolution{};
    return NeedFactoring{};
  }
  return NeedFactoring{};
}

//===----------------------------------------------------------------------===//
// adj-decomposition of prime powers   (Lemma C.21)
//===----------------------------------------------------------------------===//

/// adj-decompose p^k for an integer prime p.
///
/// Lemma C.21: p^k is adj-decomposable iff k is even, or p != 7 (mod 8).
///   k even:  t = p^(k/2) (trivially, since p in Z embeds in Z[omega] with
///            conj(p) = p).
///   k odd:   t = (t_prime)^k where conj(t_prime) * t_prime ~ p, obtained
///            from adj_decompose_prime above.
DiophantineResult adj_decompose_prime_power(const Integer &p,
                                            const Integer &k) {
  if (!(k & 1)) {
    Integer e = k / 2;
    ZOmega z = ZOmega::from_int(1);
    ZOmega base = ZOmega::from_int(p);
    while (e > 0) {
      if (e.is_odd())
        z = z * base;
      base = base * base;
      e >>= 1;
    }
    return z;
  }

  DiophantineResult t = adj_decompose_prime(p);
  if (!is_success(t))
    return t;
  Integer e = k - 1;
  const ZOmega &t_val = std::get<ZOmega>(t);
  ZOmega acc = t_val;
  ZOmega base = t_val;
  while (e > 0) {
    if (e.is_odd())
      acc = acc * base;
    base = base * base;
    e >>= 1;
  }
  return acc;
}

//===----------------------------------------------------------------------===//
// adj-decomposition: integer n = conj_sq2(xi) * xi   (Proposition C.24)
//===----------------------------------------------------------------------===//

/// adj-decompose an integer n (typically the norm conj_sq2(xi) * xi).
///
/// Iteratively factors n with Pollard-Brent rho, adj-decomposes each prime
/// power individually (Lemma C.21), and combines the partials (Lemma C.19
/// allows the combination over coprime factors). Returns NoSolution if any
/// prime power has no decomposition; returns NoSolution on diophantine
/// timeout since an incomplete factorization cannot guarantee solvability.
DiophantineResult adj_decompose(Integer n, int32_t diophantine_timeout_ms,
                                int32_t factoring_timeout_ms,
                                std::chrono::steady_clock::time_point start) {
  CUDAQ_SYNTH_OPEN_SUB("adj_decompose(int)");
  if (n < 0)
    n = -n;
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "n has " << num_decimal_digits(n) << " digits\n");
  std::vector<Factor> factors = {{n, 1}};
  ZOmega t = ZOmega::from_int(1);
  while (!factors.empty()) {
    auto [p, k] = factors.back();
    factors.pop_back();
    DiophantineResult t_p = adj_decompose_prime_power(p, k);
    if (is_no_solution(t_p)) {
      CUDAQ_SYNTH_CLOSE_FAILURE("prime power has no solution");
      return NoSolution{};
    }

    if (is_need_factoring(t_p)) {
      llvm::FailureOr<Integer> factor = find_factor(p, factoring_timeout_ms);
      if (llvm::failed(factor)) {
        // Push the unfactored term back and keep going only if there is
        // still time on the overall diophantine budget.
        factors.emplace_back(p, k);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                .count();
        if (elapsed >= diophantine_timeout_ms) {
          CUDAQ_SYNTH_CLOSE_FAILURE("diophantine timeout while factoring");
          return NoSolution{};
        }
      } else {
        LLVM_DEBUG(cudaq::synth::dbgs()
                   << "found factor (" << num_decimal_digits(*factor)
                   << " digits) of " << num_decimal_digits(p)
                   << "-digit number\n");
        factors.emplace_back(p / *factor, k);
        factors.emplace_back(*factor, k);
        auto decomposed = decompose_into_coprime_factors(factors);
        factors = decomposed.second;
      }
      continue;
    }

    t = t * std::get<ZOmega>(t_p);
  }
  CUDAQ_SYNTH_CLOSE_SUCCESS("");
  return t;
}

//===----------------------------------------------------------------------===//
// adj-decomposition: self-associate part   (xi ~ conj_sq2(xi), Lemma C.10)
//===----------------------------------------------------------------------===//

/// adj-decompose xi in Z[sqrt(2)] under the assumption that xi is associate
/// to its sqrt(2)-conjugate. By Lemma C.10 such xi has the form xi ~ n or
/// xi ~ n * sqrt(2) for some n in Z. We extract the integer content
/// n = gcd(a, b) (where xi = a + b*sqrt(2)), recurse to adj_decompose on n,
/// and absorb the residual sqrt(2) factor using delta = 1 + omega
/// (delta * conj(delta) = lambda * sqrt(2) ~ sqrt(2), Remark 3.6).
DiophantineResult
adj_decompose_selfassociate(const ZSqrt2 &xi, int32_t diophantine_timeout_ms,
                            int32_t factoring_timeout_ms,
                            std::chrono::steady_clock::time_point start) {
  if (xi == ZSqrt2{0})
    return ZOmega::from_int(0);

  Integer n = gcd(xi.a(), xi.b());
  ZSqrt2 r = xi / ZSqrt2{n};

  DiophantineResult t1 =
      adj_decompose(n, diophantine_timeout_ms, factoring_timeout_ms, start);

  // delta = 1 + omega in the Z[omega] basis is ZOmega(0, 0, 1, 1).
  ZOmega t2 = ((r % ZSqrt2{0, 1}) == ZSqrt2{0}) ? ZOmega(0, 0, 1, 1)
                                                : ZOmega::from_int(1);

  if (!is_success(t1))
    return t1;
  return std::get<ZOmega>(t1) * t2;
}

//===----------------------------------------------------------------------===//
// adj-decomposition of Z[sqrt(2)]-primes   (Lemma C.20, non-integer primes)
//===----------------------------------------------------------------------===//

/// adj-decompose a Z[sqrt(2)]-prime eta with eta not associate to
/// conj_sq2(eta) (i.e., eta is a true split prime above an integer prime
/// p == 1 or 7 (mod 8); Lemma C.11 gives p ~ conj_sq2(eta) * eta).
///
/// The norm p = |conj_sq2(eta) * eta| is the integer prime sitting under
/// eta. The case split mirrors the integer-prime case above (Lemma C.20)
/// but the gcds are computed in Z[omega] against the embedded eta rather
/// than against p.
DiophantineResult adj_decompose_prime(const ZSqrt2 &eta) {
  Integer p = eta.norm();
  if (p < 0)
    p = -p;
  if (p == 0 || p == 1)
    return ZOmega::from_int(p);
  if (p == 2)
    return ZOmega(-1, 0, 1, 0);

  if (is_probably_prime(p)) {
    if ((p & 0b11) == 1) {
      llvm::FailureOr<Integer> h = sqrt_negative_one(p);
      if (llvm::failed(h))
        return NeedFactoring{};

      ZOmega t = gcd(ZOmega(0, 1, 0, 0) + ZOmega(0, 0, 0, *h),
                     ZOmega::from_zsqrt2(eta));
      if (are_associates(ZSqrt2::from_zomega(t.conj() * t), eta))
        return t;

      return NeedFactoring{};
    }

    if ((p & 0b111) == 3) {
      llvm::FailureOr<Integer> h = root_mod(-2, p);
      if (llvm::failed(h))
        return NeedFactoring{};
      ZOmega t = gcd(ZOmega(1, 0, 1, 0) + ZOmega(0, 0, 0, *h),
                     ZOmega::from_zsqrt2(eta));
      if (are_associates(ZSqrt2::from_zomega(t.conj() * t), eta))
        return t;

      return NeedFactoring{};
    }

    if ((p & 0b111) == 7) {
      llvm::FailureOr<Integer> h = root_mod(2, p);
      if (llvm::succeeded(h))
        return NoSolution{};
      return NeedFactoring{};
    }

    return NeedFactoring{};
  }

  if ((p & 0b111) == 7) {
    llvm::FailureOr<Integer> h = root_mod(2, p);
    if (llvm::succeeded(h))
      return NoSolution{};
    return NeedFactoring{};
  }
  return NeedFactoring{};
}

/// adj-decompose eta^k for a Z[sqrt(2)]-prime eta with eta not associate to
/// conj_sq2(eta). Lemma C.21:
///   k even: t = eta^(k/2) embedded into Z[omega].
///   k odd:  t = (t_prime)^k where conj(t_prime) * t_prime ~ eta.
DiophantineResult adj_decompose_prime_power(const ZSqrt2 &eta,
                                            const Integer &k) {
  if (!k.is_odd()) {
    Integer e = k / 2;
    ZSqrt2 eta_pow = ZSqrt2{1};
    ZSqrt2 base = eta;
    while (e > 0) {
      if (e.is_odd())
        eta_pow = eta_pow * base;
      base = base * base;
      e >>= 1;
    }
    return ZOmega::from_zsqrt2(eta_pow);
  }

  DiophantineResult t = adj_decompose_prime(eta);
  if (!is_success(t))
    return t;
  Integer e = k - 1;
  const ZOmega &t_val = std::get<ZOmega>(t);
  ZOmega acc = t_val;
  ZOmega base_z = t_val;
  while (e > 0) {
    if (e.is_odd())
      acc = acc * base_z;
    base_z = base_z * base_z;
    e >>= 1;
  }
  return acc;
}

//===----------------------------------------------------------------------===//
// adj-decomposition: self-coprime part   (gcd(xi, conj_sq2(xi)) ~ 1)
//===----------------------------------------------------------------------===//

/// adj-decompose xi in Z[sqrt(2)] under the assumption that gcd(xi,
/// conj_sq2(xi)) ~ 1. All prime factors eta of xi then satisfy eta not ~
/// conj_sq2(eta) -- they are the split primes from Lemma C.11. We
/// iteratively factor the norm |conj_sq2(eta) * eta| through Pollard-rho,
/// adj-decompose each Z[sqrt(2)]-prime power, and combine via Lemma C.19.
DiophantineResult
adj_decompose_selfcoprime(const ZSqrt2 &xi, int32_t diophantine_timeout_ms,
                          int32_t factoring_timeout_ms,
                          std::chrono::steady_clock::time_point start) {
  CUDAQ_SYNTH_OPEN_SUB("adj_decompose_selfcoprime");
  LLVM_DEBUG(cudaq::synth::dbgs() << "xi=" << xi << '\n');
  std::vector<std::pair<ZSqrt2, Integer>> factors = {{xi, 1}};
  ZOmega t = ZOmega::from_int(1);
  while (!factors.empty()) {
    LLVM_DEBUG({
      std::string flist;
      for (const auto &[f, e] : factors)
        flist += f.to_string() + "^" + e.to_string() + " ";
      cudaq::synth::dbgs() << factors.size() << " pending factors: [" << flist
                           << "]\n";
    });

    auto [eta, k] = factors.back();
    factors.pop_back();
    DiophantineResult t_eta = adj_decompose_prime_power(eta, k);
    if (is_no_solution(t_eta)) {
      CUDAQ_SYNTH_CLOSE_FAILURE("prime power has no solution");
      return NoSolution{};
    }

    if (is_need_factoring(t_eta)) {
      Integer n = eta.norm();
      if (n < 0)
        n = -n;
      llvm::FailureOr<Integer> fac_n = find_factor(n, factoring_timeout_ms);
      if (llvm::failed(fac_n)) {
        factors.emplace_back(eta, k);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                .count();
        if (elapsed >= diophantine_timeout_ms) {
          CUDAQ_SYNTH_CLOSE_FAILURE("diophantine timeout while factoring");
          return NoSolution{};
        }
      } else {
        LLVM_DEBUG(cudaq::synth::dbgs()
                   << "split eta via " << num_decimal_digits(*fac_n)
                   << "-digit factor\n");
        // Lift the integer factor back into Z[sqrt(2)] via gcd with xi to
        // get the actual split prime, then continue the loop.
        ZSqrt2 fac = gcd(xi, ZSqrt2{*fac_n});
        factors.emplace_back(eta / fac, k);
        factors.emplace_back(fac, k);
        auto decomposed = decompose_into_coprime_factors(factors);
        factors = decomposed.second;
      }
      continue;
    }

    t = t * std::get<ZOmega>(t_eta);
  }
  CUDAQ_SYNTH_CLOSE_SUCCESS("");
  return t;
}

//===----------------------------------------------------------------------===//
// Main Z[sqrt(2)] solver: adj_decompose   (Proposition C.24)
//===----------------------------------------------------------------------===//

/// Solve conj(t) * t ~ xi (up to units in Z[sqrt(2)]) by splitting xi as
///   d   = gcd(xi, conj_sq2(xi))      -- the self-associate part
///   eta = xi / d                     -- the self-coprime part
/// and adj-decomposing each piece independently (Lemma C.19 allows the
/// independent treatment). Matches Lemma C.23 / Proposition C.24.
DiophantineResult adj_decompose(const ZSqrt2 &xi,
                                int32_t diophantine_timeout_ms,
                                int32_t factoring_timeout_ms,
                                std::chrono::steady_clock::time_point start) {
  CUDAQ_SYNTH_OPEN_SUB("adj_decompose(ZSqrt2)");
  if (xi == ZSqrt2{0}) {
    CUDAQ_SYNTH_CLOSE_SUCCESS("xi == 0");
    return ZOmega::from_int(0);
  }

  ZSqrt2 xi_bullet = xi.conj_sq2();
  ZSqrt2 d = gcd(xi, xi_bullet);
  ZSqrt2 eta = xi / d;

  LLVM_DEBUG(cudaq::synth::dbgs() << "self-associate d=" << d
                                  << ", self-coprime eta=" << eta << '\n');

  DiophantineResult t1 = adj_decompose_selfassociate(
      d, diophantine_timeout_ms, factoring_timeout_ms, start);
  if (is_no_solution(t1)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("self-associate part has no solution");
    return t1;
  }

  DiophantineResult t2 = adj_decompose_selfcoprime(eta, diophantine_timeout_ms,
                                                   factoring_timeout_ms, start);
  if (is_no_solution(t2)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("self-coprime part has no solution");
    return t2;
  }

  if (!is_success(t1) || !is_success(t2)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("need factoring");
    return NeedFactoring{};
  }

  CUDAQ_SYNTH_CLOSE_SUCCESS("");
  return std::get<ZOmega>(t1) * std::get<ZOmega>(t2);
}

//===----------------------------------------------------------------------===//
// Z[sqrt(2)] exact solver   (Theorem 6.2 / Lemma C.16)
//===----------------------------------------------------------------------===//

/// Solve conj(t) * t = xi (exact equality) for xi in Z[sqrt(2)].
///
/// First runs adj_decompose for the up-to-units form conj(t) * t ~ xi. The
/// residual unit u = xi / (conj(t) * t) is doubly positive (since both xi
/// and conj(t)*t are), so Lemma C.2 gives u = v^2 for some v in Z[sqrt(2)],
/// and the exact solution is t' = v * t (Lemma C.16).
DiophantineResult diophantine(const ZSqrt2 &xi, int32_t diophantine_timeout_ms,
                              int32_t factoring_timeout_ms) {
  CUDAQ_SYNTH_OPEN_SUB("diophantine");
  auto start = std::chrono::steady_clock::now();
  LLVM_DEBUG(cudaq::synth::dbgs() << "xi=" << xi << '\n');

  if (xi == ZSqrt2{0}) {
    CUDAQ_SYNTH_CLOSE_SUCCESS("xi == 0");
    return ZOmega::from_int(0);
  }

  // Necessary conditions (Lemma 6.1): xi >= 0 and conj_sq2(xi) >= 0.
  if (xi < ZSqrt2{0} || xi.conj_sq2() < ZSqrt2{0}) {
    CUDAQ_SYNTH_CLOSE_FAILURE(
        "necessary conditions failed (xi<0 or xi_bullet<0)");
    return NoSolution{};
  }

  DiophantineResult t =
      adj_decompose(xi, diophantine_timeout_ms, factoring_timeout_ms, start);
  if (!is_success(t)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("adj_decompose failed");
    return t;
  }

  // Unit adjustment (Lemma C.16).
  const ZOmega &t_val = std::get<ZOmega>(t);
  ZSqrt2 xi_approx = ZSqrt2::from_zomega(t_val.conj() * t_val);
  ZSqrt2 u = xi / xi_approx;

  // u is doubly positive, so u = v^2 in Z[sqrt(2)] (Lemma C.2).
  llvm::FailureOr<ZSqrt2> v_or = sqrt(u);
  if (llvm::failed(v_or)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("sqrt(u) failed");
    return NoSolution{};
  }

  ZOmega v_zomega = ZOmega::from_zsqrt2(*v_or);
  CUDAQ_SYNTH_CLOSE_SUCCESS("unit adjustment succeeded");
  return v_zomega * t_val;
}

} // namespace

//===----------------------------------------------------------------------===//
// Public API: D[sqrt(2)] solver   (Theorem 6.2 / Lemma C.25)
//===----------------------------------------------------------------------===//

llvm::FailureOr<DOmega>
cudaq::synth::diophantine_dyadic(const DSqrt2 &xi, int32_t diophantine_timeout,
                                 int32_t factoring_timeout) {
  CUDAQ_SYNTH_OPEN_SUB("diophantine_dyadic");
  LLVM_DEBUG(cudaq::synth::dbgs()
             << "denom_exp=" << static_cast<int64_t>(xi.k())
             << ", dioph_timeout=" << diophantine_timeout
             << "ms, fact_timeout=" << factoring_timeout << "ms\n");

  Integer k_div_2 = xi.k() >> 1;
  Integer k_mod_2 = xi.k() & 1;

  // Clear the sqrt(2) denominator so the underlying solver can work in
  // Z[sqrt(2)]. If k is even, alpha = xi * sqrt(2)^k is already in Z[sqrt(2)]
  // and we pass it directly. If k is odd, write k = 2m + 1 and multiply by
  // one extra sqrt(2) so the total exponent is 2m + 2; in the Z[sqrt(2)]
  // representation this is multiplication by ZSqrt2(1, 1) since
  //   (a + b*sqrt(2)) * sqrt(2) = (2b + a*sqrt(2)).
  ZSqrt2 arg = k_mod_2 ? (xi.alpha() * ZSqrt2(1, 1)) : xi.alpha();

  DiophantineResult t =
      diophantine(arg, diophantine_timeout, factoring_timeout);

  if (!is_success(t)) {
    CUDAQ_SYNTH_CLOSE_FAILURE("no solution for denom_exp=" +
                              xi.k().to_string());
    return llvm::failure();
  }

  // Undo the "extra sqrt(2)" introduced when k was odd, using delta = 1 +
  // omega. Lemma C.25: delta * conj(delta) = lambda * sqrt(2), i.e. delta is
  // a unit whose norm is sqrt(2) up to another unit lambda. Dividing by
  // sqrt(2) therefore corresponds to multiplying by delta / (delta *
  // conj(delta)); in the Z[omega] basis this is multiplication by the fixed
  // element ZOmega(0, -1, 1, 0).
  ZOmega z = std::get<ZOmega>(t);
  if (k_mod_2)
    z = z * ZOmega(0, -1, 1, 0);

  CUDAQ_SYNTH_CLOSE_SUCCESS("solution found for denom_exp=" +
                            xi.k().to_string());
  return DOmega(z, k_div_2 + k_mod_2);
}
