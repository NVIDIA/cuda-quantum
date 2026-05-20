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

// =============================================================================
// Diophantine solver for t†t = ξ  (Appendix C of arXiv:1403.2975)
//
// Given ξ ∈ D[√2], finds t ∈ D[ω] such that t†t = ξ, or determines
// that no solution exists.
//
// Algorithm outline (Theorem 6.2 / Proposition C.24):
//   1. Check necessary conditions: ξ ≥ 0 and ξ• ≥ 0       (Lemma 6.1)
//   2. Reduce from D[√2] to Z[√2] via δ-scaling           (Lemma C.25)
//   3. Split ξ into self-associate and self-coprime parts (Lemma C.23)
//      - Self-associate part: gcd(ξ, ξ•), where ξ ~ ξ•    (Lemma C.10)
//      - Self-coprime part:   ξ / gcd(ξ, ξ•)
//   4. Factor via integer primes → Z[√2] primes → Z[ω]    (Lemmas C.8–C.13)
//   5. †-decompose each prime factor                      (Lemma C.20)
//   6. Combine and adjust unit                            (Lemma C.16)
//
// The only super-polynomial step is integer factoring (Pollard-Brent ρ).
// =============================================================================

#define DEBUG_TYPE "cudaq-synth"

using namespace cudaq::synth;

namespace {
// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

// Outcome of an internal †-decomposition step (solving t†t ~ ξ up to units).
//
// - ZOmega:        success — t was found.
// - NoSolution:    the equation provably has no solution.
// - NeedFactoring: the equation could not be decided without further integer
//                  factoring (the prime classification test was inconclusive
//                  because n = ξ•ξ is not yet fully factored).

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

// ---------------------------------------------------------------------------
// Random number generator
// ---------------------------------------------------------------------------

/// RAII wrapper around GMP's Mersenne Twister random state.
///
/// GMP's gmp_randstate_t is not copyable, so this struct is non-copyable.
/// Each thread gets its own instance via global_rng(), seeded once from
/// std::random_device at construction.  All subsequent calls to
/// _rand_between reuse the same state, avoiding the cost of reseeding.
struct GmpRng {
  gmp_randstate_t state;

  GmpRng() {
    gmp_randinit_mt(state);
    // Combine two 32-bit words from the OS entropy source into a 64-bit
    // seed.  gmp_randseed_ui takes unsigned long (64-bit on LP64), so this
    // fills the seed register completely without constructing a temporary
    // mpz_t.
    std::random_device rd;
    // gmp_randseed_ui takes unsigned long (GMP API); cast is lossless on LP64.
    const unsigned long seed = (static_cast<unsigned long>(rd()) << 32) | rd();
    gmp_randseed_ui(state, seed);
  }

  ~GmpRng() { gmp_randclear(state); }

  GmpRng(const GmpRng &) = delete;
  GmpRng &operator=(const GmpRng &) = delete;
};

/// Returns the thread-local GMP random state.
///
/// Thread-local storage guarantees that construction (and therefore seeding)
/// happens exactly once per thread, and that the destructor — which calls
/// gmp_randclear — runs when the thread exits.
GmpRng &global_rng() {
  static thread_local GmpRng rng;
  return rng;
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

/// Number of decimal digits in |n|.
size_t num_decimal_digits(const Integer &n) {
  return static_cast<size_t>(mpz_sizeinbase(n.get_mpz_t(), 10));
}

// ---------------------------------------------------------------------------
// Modular arithmetic
// ---------------------------------------------------------------------------

/// Compute (base^exp) mod m  via GMP mpz_powm.
/// Returns 0 when m ∈ {0, 1}.
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
/// mpz_urandomm produces a value in [0, range) with uniform distribution
/// over the full bit-width of range, with no casting or truncation.
/// Adding low shifts the result into [low, high].
///
/// The range Integer is constructed here rather than at call sites so that
/// the temporary lives for the duration of the mpz_urandomm call.
Integer urand_between(const Integer &low, const Integer &high, GmpRng &rng) {
  Integer range = high - low + 1;
  Integer result;
  mpz_urandomm(result.get_mpz_t(), rng.state, range.get_mpz_t());
  return result + low;
}

// ---------------------------------------------------------------------------
// Extension field F_p[x] / (x² − base)      [used by root_mod / Cipolla]
// ---------------------------------------------------------------------------
//
// Cipolla's algorithm computes square roots mod p by working in the
// quadratic extension  F_p² = F_p[x] / (x² − δ),  where δ is a quadratic
// non-residue mod p.  An element of F_p² is a pair (a, b) representing
// a + b·x, with multiplication rule  x² = δ.
//
// The ring parameters (the prime p and the non-residue δ) are captured in
// Fp2Ctx and passed explicitly to every arithmetic operation.  This avoids
// mutable global state and makes the code thread-safe and free of temporal
// coupling.  The element struct Fp2 is a plain aggregate with no constructor
// normalization — callers are responsible for providing coefficients already
// reduced mod p, and every arithmetic function preserves that invariant.
//
// Reference: Rabin [12]; Ross & Selinger §8 / Algorithm 7.6 step 2(b).
// ---------------------------------------------------------------------------

/// Ring parameters for F_p² = F_p[x] / (x² − base).
///
/// Holds const references to the prime p and the quadratic non-residue δ
/// (called `base` because x² = δ in the extension). Both must outlive every
/// Fp2 element and every call to fp2_mul / fp2_pow that uses this context —
/// which is trivially satisfied when Fp2Ctx is constructed from local
/// variables in root_mod.
struct Fp2Ctx {
  const Integer &p;
  const Integer &base;
};

/// An element a + b·x of F_p² = F_p[x] / (x² − base).
///
/// Plain aggregate — no constructor normalization. Coefficients must be in
/// [0, p) on entry; all arithmetic functions preserve this invariant so that
/// no redundant modular reductions are performed on the hot path.
struct Fp2 {
  Integer a;
  Integer b;
};

/// Multiply two F_p² elements:
///   (a₁ + b₁·x)(a₂ + b₂·x) = (a₁a₂ + b₁b₂·δ) + (a₁b₂ + b₁a₂)·x
///
/// All intermediate products are reduced mod p exactly once; the result
/// coefficients are in [0, p).
///
/// @param ctx Ring context (prime p and non-residue δ).
/// @param lhs Left operand, coefficients in [0, p).
/// @param `rhs` Right operand, coefficients in [0, p).
/// @return Product in F_p², coefficients in [0, p).
Fp2 fp2_mul(const Fp2Ctx &ctx, const Fp2 &lhs, const Fp2 &rhs) {
  // real part: a₁·a₂ + b₁·b₂·δ  (mod p)
  Integer bb_mod = lhs.b * rhs.b % ctx.p;
  Integer new_a = (lhs.a * rhs.a + bb_mod * ctx.base) % ctx.p;
  // imaginary part: a₁·b₂ + b₁·a₂  (mod p)
  Integer new_b = (lhs.a * rhs.b + lhs.b * rhs.a) % ctx.p;
  return {std::move(new_a), std::move(new_b)};
}

/// Exponentiate an F_p² element by repeated squaring.
///
/// Computes base_elem^e in F_p² using the standard square-and-multiply
/// algorithm. The exponent is consumed by value and shifted right on each
/// iteration, so it is destroyed in the process.
///
/// Complexity: O(log e) multiplications in F_p², each costing O(M(log p))
/// where M is the GMP multiplication cost.
///
/// @param ctx Ring context (prime p and non-residue δ).
/// @param base_elem The element to exponentiate, coefficients in [0, p).
/// @param e Non-negative exponent (consumed by value).
/// @return base_elem^e in F_p², coefficients in [0, p).
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

// ---------------------------------------------------------------------------
// Integer factoring  (Pollard–Brent ρ)
// ---------------------------------------------------------------------------

/// Pollard–Brent ρ factoring: return one non-trivial factor of n, or `nullopt`
/// if the allotted time or iteration budget is exhausted.
///
/// Used as the factoring sub-oracle in the Diophantine solver.  The paper
/// (Section 8, Algorithm 7.6 step 2b) notes that factoring is the only
/// super-polynomial step; this implementation is heuristic.
///
/// @param n                    Composite integer > 1 to factor.
/// @param factoring_timeout_ms Wall-clock timeout in milliseconds.
/// @param batch_size           Steps between batched-GCD checks (default 128).
llvm::FailureOr<Integer> find_factor(const Integer &n,
                                     i32 factoring_timeout_ms,
                                     i32 batch_size = 128) {
  LLVM_DEBUG(llvm::dbgs() << "[diophantine] find_factor: n has "
                          << num_decimal_digits(n)
                          << " digits, timeout=" << factoring_timeout_ms
                          << "ms\n");
  // --- Quick trial division by small primes ---
  // u64 values; cast to unsigned long at mpz_divisible_ui_p callsite (GMP ABI).
  static constexpr u64 small_primes[] = {
      2,   3,   5,   7,   11,  13,  17,  19,  23,  29,  31,  37,  41,
      43,  47,  53,  59,  61,  67,  71,  73,  79,  83,  89,  97,  101,
      103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167,
      173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233, 239,
      241, 251, 257, 263, 269, 271, 277, 281, 283, 293, 307, 311};
  const auto *n_mpz = n.get_mpz_t();
  for (u64 p : small_primes) {
    if (mpz_divisible_ui_p(n_mpz, static_cast<unsigned long>(p))) {
      if (mpz_cmp_ui(n_mpz, static_cast<unsigned long>(p)) > 0)
        return Integer(static_cast<i64>(p));
    }
  }
  if (n <= i64(3))
    return llvm::failure();

  // Iteration budget heuristic: L ≈ 10^(digits/4) · 1.1774.
  size_t digits = num_decimal_digits(n);
  double pow_term = std::pow(10.0, static_cast<double>(digits) / 4.0);
  i64 L = static_cast<i64>(pow_term * 1.1774 + 10.0);

  GmpRng &rng = global_rng();
  Integer a_int = urand_between(1, n - 1, rng);
  const auto *a_mpz = a_int.get_mpz_t();

  // Pre-allocate GMP temporaries for the hot loop.
  Integer y_val, x_val, q_val, y0_val, tmp, diff, g_val;
  auto *y = y_val.get_mpz_t();
  auto *x = x_val.get_mpz_t();
  auto *q = q_val.get_mpz_t();
  auto *y0 = y0_val.get_mpz_t();
  auto *t = tmp.get_mpz_t();
  auto *d = diff.get_mpz_t();
  auto *g = g_val.get_mpz_t();

  mpz_set(y, a_mpz);

  i64 r = 1, k = 0;
  auto start = std::chrono::steady_clock::now();

  auto make_result = [&](mpz_t src) -> llvm::FailureOr<Integer> {
    Integer out;
    mpz_set(out.get_mpz_t(), src);
    return out;
  };

  while (true) {
    // Brent phase: save x = y + n (so x − y ≥ 0), advance hare y.
    mpz_add(x, y, n_mpz);

    while (k < r) {
      mpz_set_ui(q, 1);
      mpz_set(y0, y);

      i64 batch_end = std::min(k + static_cast<i64>(batch_size), r);
      for (; k < batch_end; ++k) {
        // y ← (y² + a) mod n
        mpz_mul(t, y, y);
        mpz_add(t, t, a_mpz);
        mpz_mod(y, t, n_mpz);

        // q ← q · (x − y) mod n
        mpz_sub(d, x, y);
        mpz_mul(t, q, d);
        mpz_mod(q, t, n_mpz);
      }

      mpz_gcd(g, q, n_mpz);

      if (mpz_cmp_ui(g, 1) != 0) {
        if (mpz_cmp(g, n_mpz) == 0) {
          // Product collapsed — backtrack step-by-step from y0.
          mpz_set(y, y0);
          for (i64 j = 0; j < batch_size; ++j) {
            mpz_mul(t, y, y);
            mpz_add(t, t, a_mpz);
            mpz_mod(y, t, n_mpz);
            mpz_sub(d, x, y);
            mpz_gcd(g, d, n_mpz);
            if (mpz_cmp_ui(g, 1) != 0) {
              if (mpz_cmp(g, n_mpz) == 0)
                return llvm::failure();
              return make_result(g);
            }
          }
          return llvm::failure();
        }
        return make_result(g);
      }

      auto now = std::chrono::steady_clock::now();
      if (k >= L ||
          std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                  .count() >= factoring_timeout_ms) {
        LLVM_DEBUG(llvm::dbgs() << "[diophantine] find_factor: exhausted "
                                   "budget for "
                                << digits << "-digit number (L=" << L
                                << ", k=" << k << ")\n");
        return llvm::failure();
      }
    }
    r <<= 1;
  }
}

// ---------------------------------------------------------------------------
// Square roots in Z_p
// ---------------------------------------------------------------------------

/// Find x with x² ≡ −1 (mod p) for prime p ≡ 1 (mod 4).
///
/// Randomly samples b ∈ [1, p−1] and checks whether b^((p−1)/4) is a
/// fourth root of unity whose square is −1.  Succeeds with probability
/// ≥ 1/2 per trial (Lemma C.20 / Remark C.22, citing Rabin [12]).
///
/// @param p          Odd prime with p ≡ 1 (mod 4).
/// @param batch_size Number of random candidates to try (default 128).
llvm::FailureOr<Integer> sqrt_negative_one(const Integer &p,
                                           i32 batch_size = 128) {
  if (p <= 2)
    return llvm::failure();

  const auto *p_mpz = p.get_mpz_t();
  Integer exp = (p - 1) >> 2; // (p−1)/4
  Integer p_minus_1 = p - 1;
  const auto *exp_mpz = exp.get_mpz_t();
  const auto *p_minus_1_mpz = p_minus_1.get_mpz_t();

  Integer b, h, r, tmp;
  GmpRng &rng = global_rng();

  for (i32 i = 0; i < batch_size; ++i) {
    b = urand_between(1, p_minus_1, rng);
    // h = b^((p-1)/4) mod p
    mpz_powm(h.get_mpz_t(), b.get_mpz_t(), exp_mpz, p_mpz);
    // r = h² mod p
    mpz_mul(tmp.get_mpz_t(), h.get_mpz_t(), h.get_mpz_t());
    mpz_mod(r.get_mpz_t(), tmp.get_mpz_t(), p_mpz);

    if (mpz_cmp(r.get_mpz_t(), p_minus_1_mpz) == 0)
      return h; // h² ≡ −1 (mod p)
    if (mpz_cmp_ui(r.get_mpz_t(), 1) != 0)
      return llvm::failure(); // p is not prime (witness found)
  }
  return llvm::failure();
}

/// Find y with y² ≡ x (mod p) using Cipolla's algorithm in F_p².
///
/// First checks the Euler criterion (x^((p−1)/2) ≡ 1 mod p) to verify
/// x is a quadratic residue.  Then searches for a random b such that
/// b² − x is a quadratic non-residue mod p, and computes the answer in
/// the extension field F_p[t]/(t² − (b²−x))  (Rabin [12]).
///
/// @param x          Value whose square root is sought.
/// @param p          Odd prime modulus.
/// @param batch_size Number of random candidates to try (default 128).
llvm::FailureOr<Integer> root_mod(const Integer &x, const Integer &p,
                                  i32 batch_size = 128) {
  Integer x_norm = x % p;
  if (x_norm < 0)
    x_norm += p;

  if (p == 2)
    return x_norm;
  if (x_norm == 0)
    return Integer(0);
  if (!(p.is_odd()) && p > 2)
    return llvm::failure(); // even "prime" > 2 — bail out

  const auto *p_mpz = p.get_mpz_t();
  const auto *x_norm_mpz = x_norm.get_mpz_t();

  Integer exp_half = (p - 1) / 2;
  Integer p_minus_1 = p - 1;
  Integer power = (p + 1) / 2; // Exponent for Cipolla: (p+1)/2
  const auto *exp_half_mpz = exp_half.get_mpz_t();
  const auto *p_minus_1_mpz = p_minus_1.get_mpz_t();

  // Euler criterion: x^((p-1)/2) must be 1 for x to be a QR mod p.
  Integer t;
  mpz_powm(t.get_mpz_t(), x_norm_mpz, exp_half_mpz, p_mpz);
  if (mpz_cmp_ui(t.get_mpz_t(), 1) != 0)
    return llvm::failure(); // x is a quadratic non-residue

  Integer b, r, candidate_base, check, tmp;
  auto &rng = global_rng();

  for (i32 i = 0; i < batch_size; ++i) {
    b = urand_between(1, p_minus_1, rng);

    // Verify b is in (Z/pZ)* (Fermat test: b^(p-1) ≡ 1).
    mpz_powm(r.get_mpz_t(), b.get_mpz_t(), p_minus_1_mpz, p_mpz);
    if (mpz_cmp_ui(r.get_mpz_t(), 1) != 0)
      return llvm::failure(); // p is composite (Fermat witness)

    // candidate_base = b² − x  (mod p).  This will be the "base" for F_p².
    mpz_mul(tmp.get_mpz_t(), b.get_mpz_t(), b.get_mpz_t());
    mpz_add(tmp.get_mpz_t(), tmp.get_mpz_t(), p_mpz);
    mpz_sub(tmp.get_mpz_t(), tmp.get_mpz_t(), x_norm_mpz);
    mpz_mod(candidate_base.get_mpz_t(), tmp.get_mpz_t(), p_mpz);

    // Check that candidate_base is a quadratic non-residue (Euler criterion).
    mpz_powm(check.get_mpz_t(), candidate_base.get_mpz_t(), exp_half_mpz,
             p_mpz);
    if (mpz_cmp_ui(check.get_mpz_t(), 1) != 0) {
      // Cipolla step: y = (b + t)^((p+1)/2) in F_p[t]/(t² − candidate_base).
      // The result's real component is the desired square root mod p.
      Fp2Ctx ctx{p, candidate_base};
      Fp2 rfp = fp2_pow(ctx, Fp2{b % p, Integer(1)}, power);
      return std::move(rfp.a);
    }
  }
  return llvm::failure();
}

// ---------------------------------------------------------------------------
// Coprime factorization helpers
// ---------------------------------------------------------------------------

using Factor = std::pair<Integer, Integer>;

/// Rewrite a product ∏ bᵢ^{kᵢ} over Z into:
///   u * ∏ cⱼ^{eⱼ}
/// where:
///   - u is a unit in Z (±1)
///   - each cⱼ is pairwise coprime with every other cₖ (gcd(cⱼ, cₖ) = ±1).
///
/// This is *not* full prime factorization. It only ensures the bases are
/// pairwise coprime, which is enough because †-decomposability factors over
/// coprime parts (Lemma C.19: if gcd(α, β) = 1, ξ = αβ is †-decomposable
/// iff α and β both are).
std::pair<Integer, std::vector<Factor>>
decompose_into_coprime_factors(const std::vector<Factor> &factors) {
  Integer unit = 1;

  // Work stack: factors still to be placed into the coprime list.
  std::vector<Factor> pending(factors.rbegin(), factors.rend());

  std::vector<Factor> coprime_factors;
  coprime_factors.reserve(factors.size());
  while (!pending.empty()) {
    // Get the next factor to process: b^k_b.
    auto [b, k_b] = pending.back();
    pending.pop_back();

    // If the current factor is ±1, it contributes only to the sign.
    if (b == 1)
      continue;
    if (b == -1 && (k_b & 1)) {
      unit = -unit;
      continue;
    }

    size_t i = 0;
    // Check each existing coprime factor for overlap with the current factor.
    while (true) {
      // If we have checked all existing coprime factors and found no overlap,
      // then add the current factor as a new coprime factor.
      if (i >= coprime_factors.size()) {
        coprime_factors.emplace_back(b, k_b);
        break;
      }

      // Take a reference to the existing factor: a^k_a.
      auto &&[a, k_a] = coprime_factors[i];

      // Case 1: bases are equal up to sign
      if (a == b || a == -b) {
        // If a = -b, then b^k_b = (-a)^k_b = (-1)^k_b * a^k_b.
        // An odd exponent contributes an extra -1 to the global unit.
        if (a == -b && (k_b & 1))
          unit = -unit;
        // Merge exponents: a^k_a * b^k_b = a^(k_a + k_b) (up to the sign above)
        k_a += k_b;
        break;
      }

      // Case 2: check whether a and b share any nontrivial common factor.
      Integer g = gcd(a, b);
      if (g == 1 || g == -1) {
        // They are already coprime: try the next existing factor.
        ++i;
        continue;
      }

      // Case 3: a and b share a nontrivial common divisor g.
      //
      // Write:
      //   a = g * a'
      //   b = g * b'
      // Then
      //   a^k_a * b^k_b = (a')^k_a * g^(k_a + k_b) * (b')^k_b.
      //
      // We want to:
      //   - replace a^k_a by a'^k_a * g^(k_a + k_b),
      //   - and later also account for b'^k_b.
      //
      // We recursively decompose the two-factor list:
      //   [ (a / g, k_a), (g, k_a + k_b) ]
      // into unit * pairwise-coprime bases

      std::vector<Factor> partial = {{a / g, k_a}, {g, k_a + k_b}};
      auto [partial_unit, partial_factors] =
          decompose_into_coprime_factors(partial);

      // Then we adjust unit and splice that back into coprime_factors
      unit *= partial_unit;

      // Replace the (now) old factor a^k_a by the first result partial factor,
      // and append any additional ones.
      coprime_factors[i] = partial_factors[0];
      coprime_factors.insert(coprime_factors.end(), partial_factors.begin() + 1,
                             partial_factors.end());

      // Finally, push (b / g, k_b) back to be processed.
      pending.emplace_back(b / g, k_b);
      break;
    }
  }
  return {unit, coprime_factors};
}

using ZSqrt2Factor = std::pair<ZSqrt2, Integer>;

/// Same as above but over Z[√2] elements.
///    — Uses ZSqrt2::are_associates (associate-equivalence, i.e. ξ ~ ξ') and
///    ZSqrt2::gcd.
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

// ===========================================================================
// †-decomposition of integer primes   (Lemma C.20)
// ===========================================================================

/// †-decompose a single positive integer prime p: find t ∈ Z[ω] with
/// t†t ~ p  (up to units in Z[√2]).
///
/// The case split by p mod 8 follows Lemma C.20:
///
///  p = 2:          δ†δ = λ√2 ~ √2, and √2·√2 = 2.
///                  Return δ·ω = ZOmega(-1,0,1,0). [*]
///
///  p ≡ 1 (mod 4):  −1 is a QR mod p (quadratic reciprocity).
///    [covers 1,5 mod 8]
///                  Find h with h² ≡ −1 (mod p), then t = gcd(h + i, p)
///                  in Z[ω] satisfies t†t ~ p   (or ~ the Z[√2]-prime above p).
///
///  p ≡ 3 (mod 8):  −2 is a QR mod p.
///                  Find h with h² ≡ −2 (mod p), then t = gcd(h + i√2, p)
///                  satisfies t†t ~ p.
///                  Here i√2 = ω + ω³ is represented as ZOmega(1,0,1,0).
///
///  p ≡ 7 (mod 8):  2 is a QR mod p (Lemma C.11), so p splits in Z[√2] as
///                  p ~ ξ•ξ.  But Lemma C.20 proves ξ is NOT †-decomposable.
///                  We check: if √2 exists mod p (confirming p ≡ 7 mod 8)
///                  → NoSolution.  If the `primality` test was wrong →
///                  Unsolved.
///
/// [*] Note: ZOmega(-1,0,1,0) = −ω³ + ω = i√2.  Since (i√2)†(i√2) = 2,
///     this is the correct representative for the p = 2 case.
///
/// If p is not actually prime (the probabilistic `primality` test was wrong),
/// returns Unsolved so the caller can attempt further factoring.
DiophantineResult adj_decompose_prime(Integer p) {
  if (p < 0)
    p = -p;
  if (p == 0 || p == 1)
    return ZOmega::from_int(p);
  if (p == 2)
    return ZOmega(-1, 0, 1, 0);

  LLVM_DEBUG(llvm::dbgs() << "[diophantine] adj_decompose_prime(int): p mod "
                             "8 = "
                          << static_cast<i64>(p & 7) << ", "
                          << num_decimal_digits(p)
                          << " digits, prime=" << is_probably_prime(p)
                          << '\n');

  if (is_probably_prime(p)) {
    if ((p & 0b11) == 1) {
      // p ≡ 1 (mod 4), i.e. p ≡ 1 or 5 (mod 8).
      // Solve h² ≡ −1 (mod p), then t = gcd(h + i, p) in Z[ω].
      llvm::FailureOr<Integer> h = sqrt_negative_one(p);
      if (llvm::failed(h))
        return NeedFactoring{};
      ZOmega t =
          gcd(ZOmega(0, 1, 0, 0) + ZOmega(0, 0, 0, *h), ZOmega::from_int(p));
      // Verify: t†t should be ±p (up to units).
      if (t.conj() * t == ZOmega::from_int(p) ||
          t.conj() * t == ZOmega::from_int(-p))
        return t;
      return NeedFactoring{};
    }

    if ((p & 0b111) == 3) {
      // p ≡ 3 (mod 8).
      // Solve h² ≡ −2 (mod p), then t = gcd(h + i√2, p) in Z[ω].
      // i√2 in Z[ω] basis: ω³ + ω = ZOmega(1,0,1,0).
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
      // p ≡ 7 (mod 8): not †-decomposable (Lemma C.20).
      // Confirm by checking that 2 is a QR mod p.
      llvm::FailureOr<Integer> h = root_mod(2, p);
      if (llvm::succeeded(h))
        return NoSolution{};
      return NeedFactoring{};
    }

    return NeedFactoring{};
  }

  // p is not prime — may need further factoring.
  if ((p & 0b111) == 7) {
    llvm::FailureOr<Integer> h = root_mod(2, p);
    if (llvm::succeeded(h))
      return NoSolution{};
    return NeedFactoring{};
  }
  return NeedFactoring{};
}

// ===========================================================================
// †-decomposition of prime powers   (Lemma C.21)
// ===========================================================================

/// †-decompose p^k for an integer prime p.
///
/// Lemma C.21: p^k is †-decomposable iff k is even, or p ≢ 7 (mod 8).
///  - k even:  t = p^(k/2)  (trivially, since p ∈ Z ⊂ Z[ω] and p† = p).
///  - k odd:   t = t_prime^k, where t_prime†·t_prime ~ p from Lemma C.20.
DiophantineResult adj_decompose_prime_power(const Integer &p,
                                            const Integer &k) {
  if (!(k & 1)) {
    // Even exponent: t = p^(k/2) ∈ Z, embedded in Z[ω].
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

  // Odd exponent: need the prime decomposition.
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

// ===========================================================================
// †-decomposition: integer n = ξ•ξ   (Proposition C.24, integer phase)
// ===========================================================================

/// †-decompose an integer n (obtained as ξ•ξ for some ξ ∈ Z[√2]).
///
/// Iteratively factors n into primes using Pollard-ρ, decomposes each
/// prime power via adj_decompose_int_prime_power, and multiplies the
/// partial solutions.  Coprime factors are handled independently
/// (Lemma C.19).
///
/// Returns NoSolution if any prime factor p ≡ 7 (mod 8) appears with
/// odd exponent (Lemma C.21).  Returns NoSolution on timeout (since
/// incomplete factoring cannot guarantee a solution exists).
DiophantineResult adj_decompose(Integer n, i32 diophantine_timeout_ms,
                                i32 factoring_timeout_ms,
                                std::chrono::steady_clock::time_point start) {
  if (n < 0)
    n = -n;
  LLVM_DEBUG(llvm::dbgs() << "[diophantine] adj_decompose(int): n has "
                          << num_decimal_digits(n) << " digits\n");
  std::vector<Factor> factors = {{n, 1}};
  ZOmega t = ZOmega::from_int(1);
  while (!factors.empty()) {
    auto [p, k] = factors.back();
    factors.pop_back();
    DiophantineResult t_p = adj_decompose_prime_power(p, k);
    if (is_no_solution(t_p))
      return NoSolution{};

    if (is_need_factoring(t_p)) {
      llvm::FailureOr<Integer> factor = find_factor(p, factoring_timeout_ms);
      if (llvm::failed(factor)) {
        factors.emplace_back(p, k);
        auto now = std::chrono::steady_clock::now();
        auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - start)
                .count();
        if (elapsed >= diophantine_timeout_ms)
          return NoSolution{};
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[diophantine] adj_decompose(int): found factor ("
                   << num_decimal_digits(*factor) << " digits) of "
                   << num_decimal_digits(p) << "-digit number\n");
        factors.emplace_back(p / *factor, k);
        factors.emplace_back(*factor, k);
        auto decomposed = decompose_into_coprime_factors(factors);
        factors = decomposed.second;
      }
      continue;
    }

    t = t * std::get<ZOmega>(t_p);
  }
  return t;
}

// ===========================================================================
// †-decomposition: self-associate part   (ξ ~ ξ•, Lemma C.10)
// ===========================================================================

/// †-decompose ξ ∈ Z[√2] under the assumption that ξ ~ ξ•  (i.e., ξ is
/// associate to its own √2-conjugate).
///
/// By Lemma C.10, such ξ satisfies either ξ ~ n (for some n ∈ Z) or
/// ξ ~ n√2.  This function extracts the integer content n = gcd(a, b)
/// where ξ = a + b√2, decomposes n via integer primes, and handles the
/// residual √2 factor (if present) using δ†δ ~ √2  (Remark 3.6).
///
/// @param xi Element of Z[√2] with xi ~ xi•.
/// @param diophantine_timeout_ms Overall timeout for the Diophantine solver.
/// @param factoring_timeout_ms Per-attempt timeout for Pollard-ρ.
/// @param start Clock reference for timeout checks.
DiophantineResult
adj_decompose_selfassociate(const ZSqrt2 &xi, i32 diophantine_timeout_ms,
                            i32 factoring_timeout_ms,
                            std::chrono::steady_clock::time_point start) {
  if (xi == ZSqrt2{0})
    return ZOmega::from_int(0);

  // Extract integer content: ξ = n · r  where n = gcd(a, b) ∈ Z
  // and r is either 1 or √2.
  Integer n = gcd(xi.a(), xi.b());
  ZSqrt2 r = xi / ZSqrt2{n};

  // †-decompose the integer part n.
  DiophantineResult t1 =
      adj_decompose(n, diophantine_timeout_ms, factoring_timeout_ms, start);

  // Handle the residual: if r ~ √2, use δ = 1 + ω with δ†δ = λ√2 ~ √2.
  // δ in Z[ω] basis: 1 + ω = ZOmega(0, 0, 1, 1).
  ZOmega t2 = ((r % ZSqrt2{0, 1}) == ZSqrt2{0}) ? ZOmega(0, 0, 1, 1)
                                                : ZOmega::from_int(1);

  if (!is_success(t1))
    return t1;
  return std::get<ZOmega>(t1) * t2;
}

// ===========================================================================
// †-decomposition of Z[√2]-primes   (Lemma C.20 for non-integer primes)
// ===========================================================================

/// †-decompose a single Z[√2]-prime η where η ≁ η•  (i.e., η is NOT
/// associate to its √2-conjugate, meaning η "splits" from an integer prime
/// p ≡ 1 or 7 mod 8 via Lemma C.11:  p ~ η•·η).
///
/// The norm p = |η•η| is an integer prime (or its square, if η ~ p).
/// The case split follows Lemma C.20 applied to η as a Z[√2]-prime:
///
///  p ≡ 1 (mod 4): find h² ≡ −1 (mod p), t = gcd(h + i, η) in Z[ω].
///  p ≡ 3 (mod 8): find h² ≡ −2 (mod p), t = gcd(h + i√2, η) in Z[ω].
///  p ≡ 7 (mod 8): NOT †-decomposable (Lemma C.20 final case).
///
/// The key difference from adj_decompose_int_prime is that the gcd is
/// computed against η ∈ Z[√2] (embedded in Z[ω]) rather than against p ∈ Z.
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

      // gcd in Z[ω] of (h + i) and η  (η embedded as a Z[ω] element).
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

/// †-decompose η^k for a Z[√2]-prime η with η ≁ η•.
///
/// Lemma C.21:
///  - k even: t = η^(k/2), embedded in Z[ω].
///  - k odd:  t = (t_prime)^k where t_prime†·t_prime ~ η.
DiophantineResult adj_decompose_prime_power(const ZSqrt2 &eta,
                                            const Integer &k) {
  if (!k.is_odd()) {
    // Even exponent: η^(k/2) ∈ Z[√2] ⊂ Z[ω].
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

  // Odd exponent: need the prime's †-decomposition.
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

// ===========================================================================
// †-decomposition: self-coprime part   (gcd(ξ, ξ•) = 1)
// ===========================================================================

/// †-decompose ξ ∈ Z[√2] under the assumption that gcd(ξ, ξ•) ~ 1.
///
/// All prime factors η of ξ satisfy η ≁ η• (they come from integer primes
/// p ≡ 1 or 7 mod 8 that split in Z[√2], per Lemma C.11).  We iteratively
/// factor n = |η•η| using Pollard-ρ, decompose each Z[√2]-prime power, and
/// combine via Lemma C.19.
DiophantineResult
adj_decompose_selfcoprime(const ZSqrt2 &xi, i32 diophantine_timeout_ms,
                          i32 factoring_timeout_ms,
                          std::chrono::steady_clock::time_point start) {
  LLVM_DEBUG(llvm::dbgs()
             << "[diophantine] adj_decompose_selfcoprime: xi=" << xi << '\n');
  std::vector<std::pair<ZSqrt2, Integer>> factors = {{xi, 1}};
  ZOmega t = ZOmega::from_int(1);
  while (!factors.empty()) {
    LLVM_DEBUG({
      std::string flist;
      for (const auto &[f, e] : factors)
        flist += f.to_string() + "^" + e.to_string() + " ";
      llvm::dbgs() << "[diophantine] adj_decompose_selfcoprime: "
                   << factors.size() << " pending factors: [" << flist
                   << "]\n";
    });

    auto [eta, k] = factors.back();
    factors.pop_back();
    DiophantineResult t_eta = adj_decompose_prime_power(eta, k);
    if (is_no_solution(t_eta))
      return NoSolution{};

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
        if (elapsed >= diophantine_timeout_ms)
          return NoSolution{};
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[diophantine] adj_decompose_selfcoprime: split eta via "
                   << num_decimal_digits(*fac_n) << "-digit factor\n");
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
  return t;
}

// ===========================================================================
// Main Z[√2] solver: adj_decompose   (Proposition C.24)
// ===========================================================================

/// Solve t†t ~ ξ  (up to units) for ξ ∈ Z[√2], by splitting ξ into:
///
///  d   = gcd(ξ, ξ•) — the "self-associate" part (d ~ d•), and
///  η   = ξ / d      — the "self-coprime" part (gcd(η, η•) ~ 1).
///
/// Each part is †-decomposed independently (Lemma C.19: coprime factors
/// decompose independently), and the results are multiplied.
///
/// This decomposition corresponds to Lemma C.23 / Proposition C.24.
DiophantineResult adj_decompose(const ZSqrt2 &xi, i32 diophantine_timeout_ms,
                                i32 factoring_timeout_ms,
                                std::chrono::steady_clock::time_point start) {
  if (xi == ZSqrt2{0})
    return ZOmega::from_int(0);

  ZSqrt2 xi_bullet = xi.conj_sq2(); // ξ•
  ZSqrt2 d = gcd(xi, xi_bullet);
  ZSqrt2 eta = xi / d;

  LLVM_DEBUG(llvm::dbgs() << "[diophantine] adj_decompose(ZSqrt2): "
                             "self-associate d="
                          << d << ", self-coprime eta=" << eta << '\n');

  DiophantineResult t1 = adj_decompose_selfassociate(
      d, diophantine_timeout_ms, factoring_timeout_ms, start);
  if (is_no_solution(t1))
    return t1;

  DiophantineResult t2 = adj_decompose_selfcoprime(eta, diophantine_timeout_ms,
                                                   factoring_timeout_ms, start);
  if (is_no_solution(t2))
    return t2;

  if (!is_success(t1) || !is_success(t2))
    return NeedFactoring{};

  return std::get<ZOmega>(t1) * std::get<ZOmega>(t2);
}

// ===========================================================================
// Z[√2] exact solver   (Theorem 6.2 / Lemma C.16)
// ===========================================================================

/// Solve t†t = ξ  (exact equality) for ξ ∈ Z[√2].
///
/// First solves the weaker problem t†t ~ ξ (up to units) via adj_decompose.
/// Then adjusts by the unit:  if t†t = u·ξ, then u is doubly positive
/// (since both t†t and ξ are), hence u = v² for some v ∈ Z[√2] (Lemma C.2).
/// The exact solution is t' = v · t  (Lemma C.16).
DiophantineResult diophantine(const ZSqrt2 &xi, i32 diophantine_timeout_ms,
                              i32 factoring_timeout_ms) {
  auto start = std::chrono::steady_clock::now();
  LLVM_DEBUG(llvm::dbgs() << "[diophantine] diophantine(ZSqrt2): xi=" << xi
                          << '\n');

  if (xi == ZSqrt2{0})
    return ZOmega::from_int(0);

  // Necessary conditions (Lemma 6.1): ξ ≥ 0 and ξ• ≥ 0.
  if (xi < ZSqrt2{0} || xi.conj_sq2() < ZSqrt2{0}) {
    LLVM_DEBUG(llvm::dbgs() << "[diophantine] diophantine: necessary "
                               "conditions failed (xi<0 or xi_bullet<0)\n");
    return NoSolution{};
  }

  DiophantineResult t =
      adj_decompose(xi, diophantine_timeout_ms, factoring_timeout_ms, start);
  if (!is_success(t))
    return t;

  // Unit adjustment (Lemma C.16).
  const ZOmega &t_val = std::get<ZOmega>(t);
  ZSqrt2 xi_approx = ZSqrt2::from_zomega(t_val.conj() * t_val); // t†t
  ZSqrt2 u = xi / xi_approx; // unit: ξ = u · (t†t)

  // u is doubly positive → u = v² (Lemma C.2).
  llvm::FailureOr<ZSqrt2> v_or = sqrt(u);
  if (llvm::failed(v_or))
    return NoSolution{};

  ZOmega v_zomega = ZOmega::from_zsqrt2(*v_or);
  LLVM_DEBUG(llvm::dbgs()
             << "[diophantine] diophantine: unit adjustment succeeded\n");
  return v_zomega * t_val;
}

} // namespace

// ===========================================================================
// Public API: D[√2] solver   (Theorem 6.2 / Lemma C.25)
// ===========================================================================

llvm::FailureOr<DOmega>
cudaq::synth::diophantine_dyadic(const DSqrt2 &xi, i32 diophantine_timeout,
                                 i32 factoring_timeout) {
  LLVM_DEBUG(llvm::dbgs() << "[diophantine] diophantine_dyadic: denom_exp="
                          << static_cast<i64>(xi.k())
                          << ", dioph_timeout=" << diophantine_timeout
                          << "ms, fact_timeout=" << factoring_timeout
                          << "ms\n");

  Integer k_div_2 = xi.k() >> 1;
  Integer k_mod_2 = xi.k() & 1;

  // We need to clear the √2 denominator so the solver can work in Z[√2].
  //
  // If k is even, then xi·(√2)^k = alpha ∈ Z[√2], so we can pass alpha
  // directly.
  //
  // If k is odd, write k = 2m+1. Multiplying xi by one extra √2 makes the total
  // exponent 2m+2, which is even. Algebraically we want:
  //
  //   xi' = xi · √2  = alpha · (√2)^(1−k) with an even exponent for √2.
  //
  // In the Z[√2] representation, multiplying by √2 corresponds to the map
  //   a + b√2  ↦  (a + 2b) + a√2,
  // which is implemented here as multiplication by ZSqrt2(1, 1).
  //
  // So:
  //   - if k is even:  use arg = alpha;
  //   - if k is odd:   use arg = alpha·√2 instead.
  ZSqrt2 arg = k_mod_2 ? (xi.alpha() * ZSqrt2(1, 1)) : xi.alpha();

  DiophantineResult t =
      diophantine(arg, diophantine_timeout, factoring_timeout);

  if (!is_success(t)) {
    LLVM_DEBUG(llvm::dbgs() << "[diophantine] diophantine_dyadic: no "
                               "solution found for denom_exp="
                            << static_cast<i64>(xi.k()) << '\n');
    return llvm::failure();
  }

  // Undo the "extra √2" if k was odd, using δ = 1 + ω.
  //
  // Lemma C.25 (δ = 1 + ω) gives δ† δ = λ√2, i.e. δ is a unit whose norm is √2
  // up to another unit λ.  Therefore:
  //
  //   dividing by √2  ≍  multiplying by δ / (δ†δ)
  //
  // and, in the Z[ω] basis, this corresponds to multiplying by a fixed element
  // represented here by ZOmega(0, -1, 1, 0).
  ZOmega z = std::get<ZOmega>(t);
  if (k_mod_2)
    z = z * ZOmega(0, -1, 1, 0);

  LLVM_DEBUG(llvm::dbgs() << "[diophantine] diophantine_dyadic: solution "
                             "found for denom_exp="
                          << static_cast<i64>(xi.k()) << '\n');

  return DOmega(z, k_div_2 + k_mod_2);
}
