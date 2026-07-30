/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include <cmath>
#include <string>

#include "cudaq/Synthesis/Math/Real.h"
#include "cudaq/Synthesis/Math/Unitary.h"
#include "cudaq/Synthesis/Synthesis/Gridsynth.h"
#include "llvm/Support/LogicalResult.h"

namespace {

using cudaq::synth::Circuit;
using cudaq::synth::Real;

// ============================================================
// Helpers
// ============================================================

// Theoretical T-count upper bound from Ross & Selinger §8 (Theorem 8.5):
//   T ≤ 4·log₂(1/ε) + K,  K ≈ 20 generous padding.
static int t_count_upper_bound(double epsilon) {
  return static_cast<int>(std::ceil(4.0 * std::log2(1.0 / epsilon))) + 20;
}

// ============================================================
// Parametrized accuracy test
// ============================================================

struct GridsynthCase {
  const char *theta;
  const char *epsilon;
};

class GridsynthApproxTest : public testing::TestWithParam<GridsynthCase> {};

TEST_P(GridsynthApproxTest, ErrorWithinEpsilonAndGatesAreValid) {
  const auto &tc = GetParam();
  std::string theta_str(tc.theta);
  std::string epsilon_str(tc.epsilon);

  Real theta(theta_str);
  Real epsilon(epsilon_str);

  llvm::FailureOr<Circuit> result = cudaq::synth::gridsynth(theta, epsilon);
  ASSERT_TRUE(llvm::succeeded(result))
      << "gridsynth failed for theta=" << tc.theta << " eps=" << tc.epsilon;

  const Circuit &circuit = *result;

  // Gate alphabet validity is structurally guaranteed by the Circuit type.

  // Verify the actual approximation error is within epsilon.
  std::string err_str =
      cudaq::synth::rz_gate_sequence_error(theta_str, circuit);
  Real err(err_str);
  EXPECT_LE(err, epsilon) << "error " << err_str << " exceeds epsilon "
                          << tc.epsilon << " for theta=" << tc.theta
                          << " circuit=" << circuit;

  // T-count must be at least 1 for a non-trivial approximation and
  // at most the Ross–Selinger theoretical bound plus generous padding.
  int tc_count = circuit.t_count();
  EXPECT_GT(tc_count, 0) << "expected T gates for theta=" << tc.theta;
  EXPECT_LE(tc_count, t_count_upper_bound(std::stod(epsilon_str)))
      << "T-count " << tc_count << " suspiciously large for eps=" << tc.epsilon;
}

// Angles chosen to exercise varied cases:
//   - generic irrational angles (0.5, 1.0, 2.0)
//   - π/8 (natural for the T gate circuit)
//   - π/6, π/3, π/5 (other rational multiples of π)
//   - π/32, π/64, π/128 (small angles, stress-test near-zero behaviour)
// Epsilons span six orders of magnitude: 1e-4 (fast), 1e-6 (moderate),
// 1e-10, 1e-12, 1e-15 (fine, exercises the full depth of the algorithm).
INSTANTIATE_TEST_SUITE_P(
    Angles, GridsynthApproxTest,
    testing::Values(
        // ε = 1e-4
        GridsynthCase{"0.5", "1e-4"}, GridsynthCase{"1.0", "1e-4"},
        GridsynthCase{"2.0", "1e-4"},

        // ε = 1e-6
        GridsynthCase{"0.5", "1e-6"}, GridsynthCase{"1.0", "1e-6"},
        GridsynthCase{"2.0", "1e-6"},
        // π/8 ≈ 0.39269908169872414
        GridsynthCase{"0.39269908169872414", "1e-6"},
        // π/6 ≈ 0.52359877559829882
        GridsynthCase{"0.52359877559829882", "1e-6"},
        // π/3 ≈ 1.04719755119659774
        GridsynthCase{"1.04719755119659774", "1e-6"},
        // π/5 ≈ 0.62831853071795865
        GridsynthCase{"0.62831853071795865", "1e-6"},
        // π/32 ≈ 0.09817477042468104
        GridsynthCase{"0.09817477042468104", "1e-6"},
        // π/64 ≈ 0.04908738521234052
        GridsynthCase{"0.04908738521234052", "1e-6"},
        // π/128 ≈ 0.02454369260617026
        GridsynthCase{"0.02454369260617026", "1e-6"},

        // ε = 1e-10  (regression cases)
        GridsynthCase{"0.5", "1e-10"},
        GridsynthCase{
            "0.392699081698724154807830422909937860524646174921888227621868074"
            "038477050785776124828353716294738443622192601661066882973382",
            "1e-10"},
        // π/32 and π/64 at higher precision
        GridsynthCase{
            "0.09817477042468103531623142992930937523388918688756445137",
            "1e-10"},
        GridsynthCase{
            "0.04908738521234051765811571496465468761694459344378222568",
            "1e-10"},

        // ε = 1e-12  (fine precision)
        GridsynthCase{"0.5", "1e-12"}, GridsynthCase{"1.0", "1e-12"},
        // π/32
        GridsynthCase{
            "0.09817477042468103531623142992930937523388918688756445137",
            "1e-12"},
        // π/64
        GridsynthCase{
            "0.04908738521234051765811571496465468761694459344378222568",
            "1e-12"},
        // π/128
        GridsynthCase{
            "0.02454369260617025882905785748232734380847229672189111284",
            "1e-12"},

        // ε = 1e-15  (very fine precision)
        GridsynthCase{"0.5", "1e-15"}, GridsynthCase{"1.0", "1e-15"},
        // π/32
        GridsynthCase{
            "0.09817477042468103531623142992930937523388918688756445137",
            "1e-15"},
        // π/64
        GridsynthCase{
            "0.04908738521234051765811571496465468761694459344378222568",
            "1e-15"},
        // π/128
        GridsynthCase{
            "0.02454369260617025882905785748232734380847229672189111284",
            "1e-15"}),
    [](const testing::TestParamInfo<GridsynthCase> &info) {
      // Build a readable test name from theta and epsilon
      std::string name;
      for (char c : std::string(info.param.theta))
        name += (c == '.' || c == '-') ? '_' : c;
      name += "__eps";
      for (char c : std::string(info.param.epsilon))
        name += (c == '-') ? 'n' : c;
      return name;
    });

// ============================================================
// Non-`parametrized` targeted tests
// ============================================================

// Verify that a smaller epsilon yields a circuit with more T gates
// (monotonicity of approximation quality vs. circuit depth).
TEST(GridsynthMonotonicityTest, FinerEpsilonMoreTGates) {
  Real theta("0.5");
  llvm::FailureOr<Circuit> r_coarse =
      cudaq::synth::gridsynth(theta, Real("1e-4"));
  llvm::FailureOr<Circuit> r_fine =
      cudaq::synth::gridsynth(theta, Real("1e-10"));

  ASSERT_TRUE(llvm::succeeded(r_coarse));
  ASSERT_TRUE(llvm::succeeded(r_fine));

  EXPECT_LT(r_coarse->t_count(), r_fine->t_count())
      << "coarse: " << *r_coarse << "\nfine: " << *r_fine;
}

// Verify the error function independently: for a zero-length gate sequence
// (identity), the error should equal 1 (max distance to a non-trivial
// rotation). R_z(θ) vs identity: ‖R_z(θ) - I‖ for θ≠0 is non-zero.
TEST(GridsynthErrorFuncTest, ErrorForIdentityCircuit) {
  // For an identity circuit, rz_gate_sequence_error(theta, Circuit{}) is the
  // distance from I to R_z(θ), which is non-zero for any θ ∉ {0, 2π, ...}.
  std::string err = cudaq::synth::rz_gate_sequence_error("0.5", Circuit{});
  Real e(err);
  EXPECT_GT(e, Real(0.0));
}

// Verify the error function returns ~0 for a synthesized circuit.
TEST(GridsynthErrorFuncTest, ErrorForSynthesizedCircuit) {
  std::string theta_str = "0.5";
  llvm::FailureOr<Circuit> result =
      cudaq::synth::gridsynth(Real(theta_str), Real("1e-6"));
  ASSERT_TRUE(llvm::succeeded(result));
  std::string err_str =
      cudaq::synth::rz_gate_sequence_error(theta_str, *result);
  Real err(err_str);
  EXPECT_LE(err, Real("1e-6"));
  EXPECT_GE(err, Real(0.0));
}

// Regression: the error metric used to return sqrt(|det(E)|), the geometric
// mean of the two singular values. That vanishes whenever E is singular, so
// T at theta = 0 reported 0 instead of its true distance
// ‖I - T‖ = |1 - e^{iπ/4}| = 2·sin(π/8) ≈ 0.7653668647301795.
TEST(GridsynthErrorFuncTest, SingularDifferenceUsesSpectralNorm) {
  Real err(cudaq::synth::rz_gate_sequence_error("0", std::string("T")));
  EXPECT_LT(abs(err - Real("0.76536686473017954345691996806")), Real("1e-25"))
      << "got " << err.to_string();
}

// The identity is exactly R_z(0), and S·W^7 is exactly R_z(π/2): both are
// zero-T Cliffords whose error must come out as (near) zero rather than
// merely small.
TEST(GridsynthErrorFuncTest, ExactCliffordRotationsHaveZeroError) {
  EXPECT_EQ(Real(cudaq::synth::rz_gate_sequence_error("0", Circuit{})),
            Real(0.0));

  // π/2 to well past double precision.
  Real err(cudaq::synth::rz_gate_sequence_error(
      "1.5707963267948966192313216916397514420985846996875529104874722961",
      std::string("SWWWWWWW")));
  EXPECT_LT(err, Real("1e-60")) << "got " << err.to_string();
}

// ============================================================
// Zero-T shortcut and epsilon validation
// ============================================================

// Regression: ‖R_z(0.5) - I‖ = 2·sin(0.5/4) ≈ 0.24935, so no gates at all
// already meet a tolerance of 0.3. The epsilon-region used to be built from
// the threshold sqrt(1 - ε²/4) rather than the Ross–Selinger 1 - ε²/2, which
// searched a strictly smaller region and returned a four-T circuit here.
TEST(GridsynthZeroTTest, LooseEpsilonNeedsNoGates) {
  struct Case {
    const char *epsilon;
  };
  // 0.3 is below the Lemma 7.2 threshold and exercises the corrected region;
  // 0.39018.. is the threshold itself; 2 and 3 are degenerate tolerances
  // (‖U - V‖ <= 2 for any two unitaries).
  for (const char *epsilon : {"0.25", "0.3", "0.39018", "0.5", "1", "2", "3"}) {
    llvm::FailureOr<Circuit> result =
        cudaq::synth::gridsynth(Real("0.5"), Real(epsilon));
    ASSERT_TRUE(llvm::succeeded(result)) << "epsilon=" << epsilon;
    EXPECT_EQ(result->t_count(), 0)
        << "epsilon=" << epsilon << " circuit=" << *result;
    EXPECT_EQ(result->to_string(), "I") << "epsilon=" << epsilon;
  }
}

// Just below the achievable error the shortcut must not fire, and the grid
// search has to do real work.
TEST(GridsynthZeroTTest, EpsilonJustBelowCliffordErrorNeedsTGates) {
  // ‖R_z(0.5) - I‖ ≈ 0.2493494667704553799.
  llvm::FailureOr<Circuit> result =
      cudaq::synth::gridsynth(Real("0.5"), Real("0.249"));
  ASSERT_TRUE(llvm::succeeded(result));
  EXPECT_GT(result->t_count(), 0) << "circuit=" << *result;
  EXPECT_LE(Real(cudaq::synth::rz_gate_sequence_error("0.5", *result)),
            Real("0.249"));
}

// Angles sitting on a multiple of π/2 are exactly Clifford, so the shortcut
// fires at any tolerance -- including very tight ones.
TEST(GridsynthZeroTTest, QuarterTurnsAreZeroTAtTightEpsilon) {
  const char *quarter_turns[] = {
      "0", "1.5707963267948966192313216916397514420985846996875529104874722961",
      "3.1415926535897932384626433832795028841971693993751058209749445923",
      "-1.5707963267948966192313216916397514420985846996875529104874722961"};
  for (const char *theta : quarter_turns) {
    llvm::FailureOr<Circuit> result =
        cudaq::synth::gridsynth(Real(theta), Real("1e-30"));
    ASSERT_TRUE(llvm::succeeded(result)) << "theta=" << theta;
    EXPECT_EQ(result->t_count(), 0)
        << "theta=" << theta << " circuit=" << *result;
    EXPECT_LE(Real(cudaq::synth::rz_gate_sequence_error(theta, *result)),
              Real("1e-30"))
        << "theta=" << theta;
  }
}

// A NaN threshold silently rejects every grid candidate, so an unvalidated
// non-finite epsilon spins the k-loop forever. These must fail fast instead.
TEST(GridsynthValidationTest, RejectsInvalidEpsilon) {
  for (const char *epsilon : {"0", "-0", "-1e-6", "-1", "nan", "inf", "-inf"}) {
    EXPECT_TRUE(llvm::failed(
        cudaq::synth::gridsynth(Real("0.5"), Real(std::string(epsilon)))))
        << "epsilon=" << epsilon << " should be rejected";
  }
}

TEST(GridsynthValidationTest, RejectsNonFiniteTheta) {
  for (const char *theta : {"nan", "inf", "-inf"}) {
    EXPECT_TRUE(llvm::failed(
        cudaq::synth::gridsynth(Real(std::string(theta)), Real("1e-6"))))
        << "theta=" << theta << " should be rejected";
  }
}

// ============================================================
// Search bound
// ============================================================

// The k-loop terminates because it is bounded by max_denominator_exponent.
// The bound has to sit above the k a real solution needs (or synthesis would
// fail on valid input) while staying finite for every admissible epsilon.
TEST(GridsynthBoundTest, ExceedsTheExpectedDenominatorExponent) {
  // Lemma 7.3: T-count is 2k-2 or 2k, so a solution is expected by
  // k ~ 2*log2(1/epsilon). The bound must leave headroom above that.
  for (const char *epsilon : {"1e-4", "1e-6", "1e-10", "1e-15", "1e-25"}) {
    const double expected_k = 2.0 * std::log2(1.0 / std::stod(epsilon));
    EXPECT_GT(cudaq::synth::details::max_denominator_exponent(
                  Real(std::string(epsilon))),
              static_cast<int64_t>(expected_k))
        << "bound too tight for epsilon=" << epsilon;
  }
}

// A loose epsilon drives log2(1/epsilon) to zero or negative; the bound must
// stay positive so the k-loop still runs its first iterations.
TEST(GridsynthBoundTest, StaysPositiveForLooseEpsilon) {
  for (const char *epsilon : {"0.5", "1", "10", "1e6"})
    EXPECT_GT(cudaq::synth::details::max_denominator_exponent(
                  Real(std::string(epsilon))),
              0)
        << "epsilon=" << epsilon;
}

// Epsilon far below the double range is the reason the bound reads the MPFR
// exponent instead of converting to a double: std::log2 of a flushed-to-zero
// epsilon would be infinite, and the k-loop would be unbounded again.
TEST(GridsynthBoundTest, FiniteAndMonotonicBelowTheDoubleRange) {
  const int64_t k_1e400 =
      cudaq::synth::details::max_denominator_exponent(Real("1e-400"));
  const int64_t k_1e4000 =
      cudaq::synth::details::max_denominator_exponent(Real("1e-4000"));

  EXPECT_GT(k_1e400, 2 * 400);
  EXPECT_LT(k_1e400, 2 * 400 * 4);
  EXPECT_GT(k_1e4000, k_1e400);
}

// Every entry point that takes an epsilon derives its working precision from
// this, so it has to clear log2(1/epsilon) by a healthy guard margin: running
// with barely enough bits is what stops the search converging.
TEST(GridsynthPrecisionTest, RequiredPrecisionExceedsTheBitsEpsilonNeeds) {
  for (const char *epsilon : {"1e-4", "1e-10", "1e-25", "1e-40"}) {
    const double needed = std::log2(1.0 / std::stod(epsilon));
    EXPECT_GT(
        cudaq::synth::details::required_precision(Real(std::string(epsilon))),
        static_cast<mpfr_prec_t>(2 * needed))
        << "too few guard bits for epsilon=" << epsilon;
  }
}

// Loose epsilon still gets a usable floor rather than a tiny (or negative)
// precision, and epsilon below the double range stays finite -- the reason
// this reads the MPFR exponent instead of calling std::log2 on a double.
TEST(GridsynthPrecisionTest, RequiredPrecisionIsBoundedAtBothExtremes) {
  for (const char *epsilon : {"0.5", "1", "10"})
    EXPECT_GE(
        cudaq::synth::details::required_precision(Real(std::string(epsilon))),
        64)
        << "epsilon=" << epsilon;

  EXPECT_GT(cudaq::synth::details::required_precision(Real("1e-400")),
            cudaq::synth::details::required_precision(Real("1e-40")));
}

// ============================================================
// Working-precision changes
// ============================================================

// Restores the default precision on scope exit so a test that raises it
// cannot leak the change into the rest of the suite.
class ScopedPrecision {
public:
  explicit ScopedPrecision(mpfr_prec_t prec)
      : saved_(Real::get_default_precision()) {
    Real::set_default_precision(prec);
  }
  ~ScopedPrecision() { Real::set_default_precision(saved_); }

private:
  mpfr_prec_t saved_;
};

// Synthesis caches precision-dependent constants (sqrt(2)/2, the powers of
// lambda) in function-local statics. A loose-epsilon run materializes them at
// a low working precision, a later tight-epsilon run must rebuild them rather
// than silently inheriting that precision as a ceiling on its own accuracy.
// CliffordTSynthesis derives the precision from epsilon, so a module holding
// two rotations at different epsilons hits exactly this ordering.
TEST(GridsynthPrecisionTest, TightEpsilonAfterLooseEpsilonStaysWithinEpsilon) {
  // Warm every cache at a precision far too low for the tight run below.
  {
    ScopedPrecision low(64);
    ASSERT_TRUE(
        llvm::succeeded(cudaq::synth::gridsynth(Real("0.5"), Real("1e-3"))));
  }

  // Same precision CliffordTSynthesis would pick for this epsilon:
  // max(64, ceil(-log2(eps) * 4 + 64)).
  const std::string epsilon_str("1e-25");
  ScopedPrecision high(396);

  const std::string theta_str("0.5");
  llvm::FailureOr<Circuit> result =
      cudaq::synth::gridsynth(Real(theta_str), Real(epsilon_str));
  ASSERT_TRUE(llvm::succeeded(result));

  std::string err_str =
      cudaq::synth::rz_gate_sequence_error(theta_str, *result);
  EXPECT_LE(Real(err_str), Real(epsilon_str))
      << "error " << err_str << " exceeds epsilon " << epsilon_str
      << " after a lower-precision run poisoned the constant caches";
}

} // namespace
