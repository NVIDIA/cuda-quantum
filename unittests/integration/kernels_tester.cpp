/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

#endif

#if defined(CUDAQ_BACKEND_STIM)

// This test tests the "pcm_size" and "pcm" execution contexts for Stim.
CUDAQ_TEST(KernelsTester, pcmTester) {
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
  double noise_bf_prob = 0.1;

  cudaq::noise_model noise;
  cudaq::bit_flip_channel bf(noise_bf_prob);
  for (std::size_t i = 0; i < num_qubits; i++)
    noise.add_channel("mz", {i}, bf);
  // noise.add_all_qubit_channel("x",
  // cudaq::qec::two_qubit_bitflip(noise_bf_prob),
  //                             /*num_controls=*/1);
  cudaq::set_noise(noise);

  // Stage 1 - get the PCM size by running with "pcm_size". The
  // result will be returned in ctx_pcm_size.shots.
  cudaq::ExecutionContext ctx_pcm_size("pcm_size");
  auto &platform = cudaq::get_platform();
  platform.set_exec_ctx(&ctx_pcm_size);
  multi_round_ghz{}(num_qubits, num_rounds);
  platform.reset_exec_ctx();

  // Stage 2 - get the PCM using the ctx_pcm_size.shots value.
  cudaq::ExecutionContext ctx_pcm("pcm");
  ctx_pcm.noiseModel = &noise;
  ctx_pcm.pcm_dimensions = ctx_pcm_size.pcm_dimensions;
  platform.set_exec_ctx(&ctx_pcm);
  multi_round_ghz{}(num_qubits, num_rounds);
  platform.reset_exec_ctx();

  // The PCM is now stored in ctx_pcm.result. More precisely, the unfiltered
  // PCM is stored there, but some post-processing may be required to
  // eliminate duplicate columns.
  auto pcm_as_strings = ctx_pcm.result.sequential_data();
  printf("Columns of PCM:\n");
  for (int col = 0; auto x : pcm_as_strings) {
    // For this multi_round_ghz, we expect a 15x15 identity matrix.
    std::string expected_string(num_qubits * num_rounds, '0');
    expected_string[col] = '1';
    EXPECT_EQ(expected_string, x);
    auto p = ctx_pcm.pcm_probabilities.value()[col];
    printf("Column %02d (Prob %.6f): %s\n", col, p, x.c_str());
    EXPECT_EQ(p, noise_bf_prob);
    col++;
  }
  // for (auto &[k, v] : ctx_pcm.result.to_map()) {
  //   if (v > 1) {
  //     printf("Key %s found with >1 entries (%lu)\n", k.c_str(), v);
  //   }
  // }
}
#endif
