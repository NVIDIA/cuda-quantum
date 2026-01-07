/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/builder/kernels.h"
#include <iostream>

namespace cudaq {
namespace details {
std::vector<std::string> grayCode(std::size_t);
}
} // namespace cudaq

CUDAQ_TEST(KernelsTester, checkGrayCode) {
  {
    auto test = cudaq::details::grayCode(2);
    std::vector<std::string> expected{"00", "01", "11", "10"};
    EXPECT_EQ(test.size(), expected.size());
    for (auto &t : test) {
      EXPECT_TRUE(std::find(expected.begin(), expected.end(), t) !=
                  expected.end());
    }
  }
  {
    std::vector<std::string> expected{
        "00000", "00001", "00011", "00010", "00110", "00111", "00101", "00100",
        "01100", "01101", "01111", "01110", "01010", "01011", "01001", "01000",
        "11000", "11001", "11011", "11010", "11110", "11111", "11101", "11100",
        "10100", "10101", "10111", "10110", "10010", "10011", "10001", "10000"};

    auto test = cudaq::details::grayCode(5);
    EXPECT_EQ(test.size(), expected.size());
    for (auto &t : test) {
      EXPECT_TRUE(std::find(expected.begin(), expected.end(), t) !=
                  expected.end());
    }
  }
}

CUDAQ_TEST(KernelsTester, checkGenCtrlIndices) {
  {
    auto test = cudaq::details::getControlIndices(2);
    std::vector<std::size_t> expected{0, 1, 0, 1};
    EXPECT_EQ(test.size(), expected.size());
    EXPECT_EQ(test, expected);
  }
  {
    std::vector<std::size_t> expected{0, 1, 0, 2, 0, 1, 0, 3, 0, 1, 0,
                                      2, 0, 1, 0, 4, 0, 1, 0, 2, 0, 1,
                                      0, 3, 0, 1, 0, 2, 0, 1, 0, 4};

    auto test = cudaq::details::getControlIndices(5);
    EXPECT_EQ(test.size(), expected.size());
    EXPECT_EQ(test, expected);
  }
}

CUDAQ_TEST(KernelsTester, checkGetAlphaY) {
  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 2);
    std::vector<double> expected{1.57079633};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }

  {
    std::vector<double> state{.70710678, 0., 0., 0.70710678};
    auto thetas = cudaq::details::getAlphaY(state, 2, 1);
    std::vector<double> expected{0.0, 3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
}

CUDAQ_TEST(KernelsTester, checkGetAlphaZ) {
  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 3);
    std::vector<double> expected{1.17809725};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 2);
    std::vector<double> expected{0., 0.78539816};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }

  {
    std::vector<double> omega{0., 0., 0., 0., 0., 1.57079633, 3.14159265, 0.};
    auto thetas = cudaq::details::getAlphaZ(omega, 3, 1);
    std::vector<double> expected{0., 0., 1.57079633, -3.14159265};
    for (std::size_t i = 0; auto t : thetas) {
      EXPECT_NEAR(t, expected[i++], 1e-3);
    }
  }
}

#if !defined(CUDAQ_BACKEND_DM) && !defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(KernelsTester, checkFromState) {
  {
    std::vector<std::complex<double>> state{.70710678, 0., 0., 0.70710678};
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(2);

    cudaq::from_state(kernel, qubits, state);

    std::cout << kernel << "\n";
    auto counts = cudaq::sample(kernel);
    counts.dump();
  }

  {
    std::vector<std::complex<double>> state{0., .292786, .956178, 0.};
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(2);
    cudaq::from_state(kernel, qubits, state);
    std::cout << kernel << "\n";

    auto H = 5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
             2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
             .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
    auto energy = cudaq::observe(kernel, H).expectation();
    EXPECT_NEAR(-1.748, energy, 1e-3);

    auto ss = cudaq::get_state(kernel);
    for (std::size_t i = 0; i < 4; i++)
      EXPECT_NEAR(ss[i].real(), state[i].real(), 1e-3);
  }

  {
    std::vector<std::complex<double>> state{.70710678, 0., 0., 0.70710678};

    // Return a kernel from the given state, this
    // comes back as a unique_ptr.
    auto kernel = cudaq::from_state(state);
    std::cout << *kernel << "\n";
    auto counts = cudaq::sample(*kernel);
    counts.dump();
  }

  {
    // Random unitary state
    const std::size_t numQubits = 2;
    auto randHam = cudaq::spin_op::random(numQubits, numQubits * numQubits,
                                          std::mt19937::default_seed);
    auto eigenVectors = randHam.to_matrix().eigenvectors();
    // Map the ground state to a cudaq::state
    std::vector<std::complex<double>> expectedData(eigenVectors.rows());
    for (std::size_t i = 0; i < eigenVectors.rows(); i++)
      expectedData[i] = eigenVectors(i, 0);
    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(numQubits);
    cudaq::from_state(kernel, qubits, expectedData);
    std::cout << kernel << "\n";
    auto ss = cudaq::get_state(kernel);
    const std::complex<double> globalPhase = [&]() {
      // find the first non-zero element to compute the global phase factor
      for (std::size_t i = 0; i < (1u << numQubits); i++)
        if (std::abs(ss[i]) > 1e-3)
          return expectedData[i] / ss[i];
      // Something wrong (state vector all zeros!)
      return std::complex<double>(0.0, 0.0);
    }();
    // Check the state (accounted for global phase)
    for (std::size_t i = 0; i < (1u << numQubits); i++)
      EXPECT_NEAR(std::abs(globalPhase * ss[i] - expectedData[i]), 0.0, 1e-6);
  }
}

CUDAQ_TEST(KernelsTester, checkSampleBug2937) {
  constexpr int qubit_count = 20;
  auto kernel = cudaq::make_kernel();
  auto qubits = kernel.qalloc(qubit_count);
  constexpr int depth = qubit_count / 5;
  for (int i = 0; i < depth; i++) {
    kernel.h(qubits[i * 5]);
    for (int j = 0; j < 4; j++) {
      kernel.x<cudaq::ctrl>(qubits[i * 5 + j], qubits[i * 5 + j + 1]);
    }
  }

  kernel.mz(qubits);
  auto counts = cudaq::sample(kernel);
  counts.dump();
  // Expect 16 unique bitstrings
  EXPECT_EQ(counts.size(), 16);
}

#endif

#if defined(CUDAQ_BACKEND_STIM)

/// Helper function to transpose the Measurement Syndrome Matrix.
std::vector<std::string> transpose_msm(const std::vector<std::string> &msm) {
  std::vector<std::string> transpose(msm[0].size(),
                                     std::string(msm.size(), '.'));
  for (std::size_t r = 0; r < msm.size(); r++)
    for (std::size_t c = 0; c < msm[r].size(); c++)
      if (msm[r][c] == '1')
        transpose[c][r] = '1';
  return transpose;
}

// This test tests the "msm_size" and "msm" execution contexts for Stim.
CUDAQ_TEST(KernelsTester, msmTester_mz_only) {
  struct multi_round_ghz {
    void operator()(int num_qubits, int num_rounds) __qpu__ {
      cudaq::qvector q(num_qubits);
      for (int round = 0; round < num_rounds; round++) {
        h(q[0]);
        for (int qi = 1; qi < num_qubits; qi++)
          x<cudaq::ctrl>(q[qi - 1], q[qi]);
        mz(q);
        for (int qi = 0; qi < num_qubits; qi++)
          reset(q[qi]);
      }
    }
  };

  int num_qubits = 5;
  int num_rounds = 3;
  double noise_bf_prob = 0.0625;

  cudaq::noise_model noise;
  cudaq::bit_flip_channel bf(noise_bf_prob);
  for (std::size_t i = 0; i < num_qubits; i++)
    noise.add_channel("mz", {i}, bf);
  cudaq::set_noise(noise);

  // Stage 1 - get the MSM size by running with "msm_size". The
  // result will be returned in ctx_msm_size.shots.
  cudaq::ExecutionContext ctx_msm_size("msm_size");
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_msm_size);
  multi_round_ghz{}(num_qubits, num_rounds);
  platform.reset_exec_ctx();

  // Stage 2 - get the MSM using the size calculated above
  // (ctx_msm_size.msm_dimensions).
  cudaq::ExecutionContext ctx_msm("msm");
  ctx_msm.noiseModel = &noise;
  ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
  platform.set_exec_ctx(&ctx_msm);
  multi_round_ghz{}(num_qubits, num_rounds);
  platform.reset_exec_ctx();

  // The MSM is now stored in ctx_msm.result. More precisely, the unfiltered
  // MSM is stored there, but some post-processing may be required to
  // eliminate duplicate columns.
  auto msm_as_strings = ctx_msm.result.sequential_data();
  printf("Columns of MSM:\n");
  for (int col = 0; auto x : msm_as_strings) {
    // For this multi_round_ghz, we expect a 15x15 identity matrix.
    std::string expected_string(num_qubits * num_rounds, '0');
    expected_string[col] = '1';
    EXPECT_EQ(expected_string, x);
    auto p = ctx_msm.msm_probabilities.value()[col];
    printf("Column %02d (Prob %.6f): %s\n", col, p, x.c_str());
    EXPECT_EQ(p, noise_bf_prob);
    col++;
  }
}

CUDAQ_TEST(KernelsTester, msmTester_mz_and_depol1_corr) {
  struct multi_round_ghz {
    void operator()(int num_qubits, int num_rounds,
                    double noise_probability) __qpu__ {
      cudaq::qvector q(num_qubits);
      for (int round = 0; round < num_rounds; round++) {
        h(q[0]);
        for (int qi = 0; qi < num_qubits; qi++)
          cudaq::apply_noise<cudaq::depolarization_channel>(noise_probability,
                                                            q[qi]);
        for (int qi = 1; qi < num_qubits; qi++)
          x<cudaq::ctrl>(q[qi - 1], q[qi]);
        mz(q);
        for (int qi = 0; qi < num_qubits; qi++)
          reset(q[qi]);
      }
    }
  };

  int num_qubits = 5;
  int num_rounds = 3;
  double noise_bf_prob = 0.0625;

  cudaq::noise_model noise;
  cudaq::bit_flip_channel bf(noise_bf_prob);
  for (std::size_t i = 0; i < num_qubits; i++)
    noise.add_channel("mz", {i}, bf);
  cudaq::set_noise(noise);

  // Stage 1 - get the MSM size by running with "msm_size". The
  // result will be returned in ctx_msm_size.shots.
  cudaq::ExecutionContext ctx_msm_size("msm_size");
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_msm_size);
  multi_round_ghz{}(num_qubits, num_rounds, noise_bf_prob);
  platform.reset_exec_ctx();

  // Stage 2 - get the MSM using the size calculated above
  // (ctx_msm_size.msm_dimensions).
  cudaq::ExecutionContext ctx_msm("msm");
  ctx_msm.noiseModel = &noise;
  ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
  platform.set_exec_ctx(&ctx_msm);
  multi_round_ghz{}(num_qubits, num_rounds, noise_bf_prob);
  platform.reset_exec_ctx();

  // The MSM is now stored in ctx_msm.result. More precisely, the unfiltered
  // MSM is stored there, but some post-processing may be required to
  // eliminate duplicate columns.
  auto msm_as_strings = ctx_msm.result.sequential_data();
  auto msm_transpose = transpose_msm(msm_as_strings);

  const std::vector<std::string> expected = {
      "11.............1............................................",
      "11.11...........1...........................................",
      "11.11.11.........1..........................................",
      "11.11.11.11.......1.........................................",
      "11.11.11.11.11.....1........................................",
      ".11.11..............11.............1........................",
      ".11.11..............11.11...........1.......................",
      ".11.11..............11.11.11.........1......................",
      ".11.11..............11.11.11.11.......1.....................",
      ".11.11..............11.11.11.11.11.....1....................",
      "....11.11............11.11..............11.............1....",
      "....11.11............11.11..............11.11...........1...",
      "....11.11............11.11..............11.11.11.........1..",
      "....11.11............11.11..............11.11.11.11.......1.",
      "....11.11............11.11..............11.11.11.11.11.....1"};

  EXPECT_EQ(msm_transpose, expected);

  std::vector<double> expected_probabilities{
      0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833,
      0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833,
      0.020833, 0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.020833,
      0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833,
      0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833,
      0.062500, 0.062500, 0.062500, 0.062500, 0.062500, 0.020833, 0.020833,
      0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833,
      0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.020833, 0.062500,
      0.062500, 0.062500, 0.062500, 0.062500};
  std::vector<std::size_t> expected_err_ids{
      0,  0,  0,  1,  1,  1,  2,  2,  2,  3,  3,  3,  4,  4,  4,
      5,  6,  7,  8,  9,  10, 10, 10, 11, 11, 11, 12, 12, 12, 13,
      13, 13, 14, 14, 14, 15, 16, 17, 18, 19, 20, 20, 20, 21, 21,
      21, 22, 22, 22, 23, 23, 23, 24, 24, 24, 25, 26, 27, 28, 29};
  for (std::size_t i = 0; i < expected_probabilities.size(); i++)
    EXPECT_NEAR(expected_probabilities[i], ctx_msm.msm_probabilities.value()[i],
                1e-5)
        << "Mismatch at index " << i;
  for (std::size_t i = 0; i < ctx_msm.msm_prob_err_id.value().size(); i++)
    EXPECT_EQ(ctx_msm.msm_prob_err_id.value()[i], expected_err_ids[i])
        << "Mismatch at index " << i;
}

/// This helper function is used in many tests below. It creates a simple kernel
/// that applies a noise operation, depending on the template parameters, and it
/// returns the MSM and the probabilities. NOTE: This is NOT intended to be a
/// true QEC-like kernel
template <typename NoiseType, int num_qubits>
std::tuple<std::vector<std::string>, std::vector<double>,
           std::vector<std::size_t>>
get_msm_test(double noise_probability) {
  // This simple kernel just creates qubits and applies one noise operation,
  // depending on the template parameters.
  struct simple_test {
    void operator()(double noise_probability) __qpu__ {
      cudaq::qvector q(num_qubits);
      if constexpr (std::is_same_v<NoiseType, cudaq::pauli1>) {
        double noise_prob_per_pauli = noise_probability / 3;
        cudaq::apply_noise<NoiseType>(noise_prob_per_pauli,
                                      noise_prob_per_pauli,
                                      noise_prob_per_pauli, q[0]);
      } else if constexpr (std::is_same_v<NoiseType, cudaq::pauli2>) {
        double tmp_prob = noise_probability / 15;
        cudaq::apply_noise<NoiseType>(tmp_prob, tmp_prob, tmp_prob, tmp_prob,
                                      tmp_prob, tmp_prob, tmp_prob, tmp_prob,
                                      tmp_prob, tmp_prob, tmp_prob, tmp_prob,
                                      tmp_prob, tmp_prob, tmp_prob, q[0], q[1]);
      } else if constexpr (num_qubits > 1) {
        cudaq::apply_noise<NoiseType>(noise_probability, q[0], q[1]);
      } else {
        cudaq::apply_noise<NoiseType>(noise_probability, q[0]);
      }
      mz(q);
    }
  };

  cudaq::noise_model noise;
  cudaq::set_noise(noise);

  // Stage 1 - get the MSM size by running with "msm_size". The
  // result will be returned in ctx_msm_size.shots.
  cudaq::ExecutionContext ctx_msm_size("msm_size");
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_msm_size);
  simple_test{}(noise_probability);
  platform.reset_exec_ctx();

  // Stage 2 - get the MSM using the size calculated above
  // (ctx_msm_size.msm_dimensions).
  cudaq::ExecutionContext ctx_msm("msm");
  ctx_msm.noiseModel = &noise;
  ctx_msm.msm_dimensions = ctx_msm_size.msm_dimensions;
  platform.set_exec_ctx(&ctx_msm);
  simple_test{}(noise_probability);
  platform.reset_exec_ctx();

  return {transpose_msm(ctx_msm.result.sequential_data()),
          ctx_msm.msm_probabilities.value(), ctx_msm.msm_prob_err_id.value()};
}

CUDAQ_TEST(KernelsTester, msmTester_depol2) {
  double noise_probability = 0.0625;
  auto [msm_transpose, msm_probabilities, msm_prob_err_id] =
      get_msm_test<cudaq::depolarization2, 2>(noise_probability);

  const std::vector<std::string> expected = {"...11111111....",
                                             "11..11..11..11."};

  EXPECT_EQ(msm_transpose, expected);

  std::vector<double> expected_probabilities(15, noise_probability / 15);
  for (std::size_t i = 0; i < expected_probabilities.size(); i++)
    EXPECT_NEAR(expected_probabilities[i], msm_probabilities[i], 1e-5)
        << "Mismatch at index " << i;
  for (std::size_t i = 0; i < msm_prob_err_id.size(); i++)
    EXPECT_EQ(msm_prob_err_id[i], 0) << "Mismatch at index " << i;
}

CUDAQ_TEST(KernelsTester, msmTester_x) {
  double noise_probability = 0.0625;
  auto [msm_transpose, msm_probabilities, msm_prob_err_id] =
      get_msm_test<cudaq::x_error, 1>(noise_probability);
  EXPECT_EQ(msm_transpose, std::vector<std::string>{"1"});
  EXPECT_NEAR(msm_probabilities[0], noise_probability, 1e-5);
  EXPECT_EQ(msm_prob_err_id[0], 0);
}

CUDAQ_TEST(KernelsTester, msmTester_y) {
  double noise_probability = 0.0625;
  auto [msm_transpose, msm_probabilities, msm_prob_err_id] =
      get_msm_test<cudaq::y_error, 1>(noise_probability);
  EXPECT_EQ(msm_transpose, std::vector<std::string>{"1"});
  EXPECT_NEAR(msm_probabilities[0], noise_probability, 1e-5);
  EXPECT_EQ(msm_prob_err_id[0], 0);
}

CUDAQ_TEST(KernelsTester, msmTester_z) {
  double noise_probability = 0.0625;
  auto [msm_transpose, msm_probabilities, msm_prob_err_id] =
      get_msm_test<cudaq::z_error, 1>(noise_probability);
  EXPECT_EQ(msm_transpose, std::vector<std::string>{"."});
  EXPECT_NEAR(msm_probabilities[0], noise_probability, 1e-5);
  EXPECT_EQ(msm_prob_err_id[0], 0);
}

CUDAQ_TEST(KernelsTester, msmTester_pauli1) {
  double noise_probability = 0.0625;
  auto [msm_transpose, msm_probabilities, msm_prob_err_id] =
      get_msm_test<cudaq::pauli1, 1>(noise_probability);
  EXPECT_EQ(msm_transpose, std::vector<std::string>{"11."});
  for (std::size_t i = 0; i < msm_probabilities.size(); i++)
    EXPECT_NEAR(msm_probabilities[i], noise_probability / 3, 1e-5);
  EXPECT_EQ(msm_prob_err_id[0], 0);
}

CUDAQ_TEST(KernelsTester, msmTester_pauli2) {
  double noise_probability = 0.0625;
  auto [msm_transpose, msm_probabilities, msm_prob_err_id] =
      get_msm_test<cudaq::pauli2, 2>(noise_probability);
  const std::vector<std::string> expected = {"...11111111....",
                                             "11..11..11..11."};
  EXPECT_EQ(msm_transpose, expected);
  for (std::size_t i = 0; i < msm_probabilities.size(); i++)
    EXPECT_NEAR(msm_probabilities[i], noise_probability / 15, 1e-5);
  EXPECT_EQ(msm_prob_err_id[0], 0);
}

#endif
