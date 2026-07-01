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

} // namespace
