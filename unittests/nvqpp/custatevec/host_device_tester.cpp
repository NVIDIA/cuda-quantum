/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cudaq.h>
#include <gtest/gtest.h>

TEST(HostDeviceTester, BasicCheck) {
  // This must be set before running this test to make sure host-device
  // migration is activated
  EXPECT_TRUE(std::getenv("CUDAQ_MAX_GPU_MEMORY_GB") != nullptr);
  const auto gpuMemGb = std::atoi(std::getenv("CUDAQ_MAX_GPU_MEMORY_GB"));
  // The threshold must be set to a small value.
  // 1GB ~ 26 qubits (fp64)/ 27 qubits (fp32)
  EXPECT_LE(gpuMemGb, 1);
  EXPECT_TRUE(std::getenv("CUDAQ_MAX_CPU_MEMORY_GB") != nullptr);
  const auto cpuMemGb = std::atoi(std::getenv("CUDAQ_MAX_CPU_MEMORY_GB"));
  // 32GB ~ 31 qubits (fp64)/ 32 qubits (fp32)
  EXPECT_GE(cpuMemGb, 32);
}

struct ghz {
  auto operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    h(q[0]);
    for (int i = 0; i < N - 1; ++i) {
      x<cudaq::ctrl>(q[i], q[i + 1]);
    }
    mz(q);
  }
};

TEST(HostDeviceTester, checkBell) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 29;
  auto counts = cudaq::sample(ghz{}, numQubits);
  counts.dump();
  int counter = 0;
  const std::string allZero(numQubits, '0');
  const std::string allOne(numQubits, '1');
  for (auto &[bits, count] : counts) {
    counter += count;
    EXPECT_TRUE(bits == allZero || bits == allOne);
  }
  EXPECT_EQ(counter, 1000);
}

template <int nrOfBits>
std::vector<bool> random_bits(int seed) {

  std::vector<bool> randomBits(nrOfBits);
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution(0.0, 1.0);

  for (size_t i = 0; i < nrOfBits; i++) {
    randomBits[i] = distribution(generator) >= 0.5;
  }
  return randomBits;
}

template <int nrOfBits>
struct oracle {
  auto operator()(std::vector<bool> bitvector, cudaq::qview<> qs,
                  cudaq::qubit &aux) __qpu__ {

    for (size_t i = 0; i < nrOfBits; i++) {
      if (bitvector[i] & 1) {
        x<cudaq::ctrl>(qs[nrOfBits - i - 1], aux);
      }
    }
  }
};

template <int nrOfBits>
struct bernstein_vazirani {
  auto operator()(std::vector<bool> bitvector) __qpu__ {

    cudaq::qarray<nrOfBits> qs;
    cudaq::qubit aux;
    h(aux);
    z(aux);
    h(qs);

    oracle<nrOfBits>{}(bitvector, qs, aux);
    h(qs);
    mz(qs);
  }
};

// Construct the bit vector such that the last bit has highest significance.
std::string asString(const std::vector<bool> &bitvector) {
  char *buffer = static_cast<char *>(alloca(bitvector.size() + 1));
  std::size_t N = bitvector.size();
  buffer[N] = '\0';
  for (std::size_t i = 0; i < N; ++i)
    buffer[N - 1 - i] = '0' + bitvector[i];
  return {buffer, N};
}

TEST(HostDeviceTester, checkBernsteinVazirani) {
  // The number of qubits should be large to test distribution
  const int nr_qubits = 28;
  const int nr_shots = 100;
  const int seed = 123;
  auto bitvector = random_bits<nr_qubits>(seed);
  auto kernel = bernstein_vazirani<nr_qubits>{};
  auto counts = cudaq::sample(nr_shots, kernel, bitvector);

  if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
    printf("Encoded bitstring:  %s\n", asString(bitvector).c_str());
    printf("Measured bitstring: %s\n\n", counts.most_probable().c_str());
    EXPECT_EQ(asString(bitvector), counts.most_probable());
    for (auto &[bits, count] : counts) {
      printf("observed %s with %.0f%% probability\n", bits.data(),
             100.0 * count / nr_shots);
    }
  }
}

TEST(HostDeviceTester, checkOverlap) {
  const int nQubits = 29;
  auto kernel1 = [&]() __qpu__ {
    cudaq::qvector q(nQubits);
    for (int i = 0; i < nQubits; ++i)
      x(q[i]);
  };

  auto kernel2 = [&]() __qpu__ {
    cudaq::qvector q(nQubits);
    h(q[0]);
    for (int i = 0; i < nQubits - 1; ++i)
      x<cudaq::ctrl>(q[i], q[i + 1]);
  };

  auto state1 = cudaq::get_state(kernel1);
  auto state2 = cudaq::get_state(kernel2);
  const auto overlap = state1.overlap(state2);
  std::cout << "Overlap = " << overlap << "\n";
  EXPECT_NEAR(std::abs(overlap), M_SQRT1_2, 1e-6);
}

TEST(HostDeviceTester, checkStateIndexing) {
  const int nQubits = 29;
  auto kernel = [&]() __qpu__ {
    cudaq::qvector q(nQubits);
    h(q[0]);
    for (int i = 0; i < nQubits - 1; ++i)
      x<cudaq::ctrl>(q[i], q[i + 1]);
  };

  auto state = cudaq::get_state(kernel);

  std::cout << "ampl(00..00) = " << state[0] << "\n";
  std::cout << "ampl(11..11) = " << state[(1ULL << nQubits) - 1] << "\n";
  EXPECT_NEAR(std::abs(state[0]), M_SQRT1_2, 1e-6);
  EXPECT_NEAR(std::abs(state[(1ULL << nQubits) - 1]), M_SQRT1_2, 1e-6);
}

// Make sure that we are finding the right amplitude.
TEST(HostDeviceTester, checkStateIndexingRandom) {
  const int nQubits = 29;
  const auto randomBitString = [nQubits]() {
    std::vector<int> bitStr;
    for (int i = 0; i < nQubits; ++i)
      bitStr.emplace_back((int)rand() % 2);
    return bitStr;
  };

  auto kernel = [&](std::vector<int> bitStr) __qpu__ {
    cudaq::qvector q(nQubits);
    for (int i = 0; i < nQubits; ++i)
      if (bitStr[i] == 1)
        x(q[i]);
  };

  constexpr int numTests = 10;
  for (int i = 0; i < numTests; ++i) {
    std::cout << "Test " << i << " with bitstring: ";
    const auto bitStr = randomBitString();
    for (const auto &b : bitStr)
      std::cout << b;
    std::cout << "\n";
    auto state = cudaq::get_state(kernel, bitStr);
    EXPECT_NEAR(std::abs(state.amplitude(bitStr)), 1.0, 1e-6);
  }
}

TEST(HostDeviceTester, checkNoise) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 27;
  // 2-qubit noise
  cudaq::kraus_op op0{cudaq::complex{0.99498743710662, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.99498743710662, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.99498743710662, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.99498743710662, 0.0}},
      op1{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0}},
      op2{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.05773502691896258},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.05773502691896258},
          {0.0, 0.05773502691896258},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.05773502691896258},
          {0.0, 0.0},
          {0.0, 0.0}},
      op3{cudaq::complex{0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.05773502691896258, 0.0},
          {-0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.0, 0.0},
          {-0.05773502691896258, 0.0}};
  cudaq::kraus_channel cnotNoise(
      std::vector<cudaq::kraus_op>{op0, op1, op2, op3});
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(cnotNoise, 1);
  cudaq::set_noise(noise);
  constexpr int num_shots = 50;
  auto counts = cudaq::sample(num_shots, ghz{}, numQubits);
  counts.dump();
  int counter = 0;
  const std::string allZero(numQubits, '0');
  const std::string allOne(numQubits, '1');
  for (auto &[bits, count] : counts)
    counter += count;
  EXPECT_EQ(counter, num_shots);

  // Check results
  // Noisy results: more than just the 2 Bell states
  EXPECT_GT(counts.size(), 2);
  // Noise is applied to all qubits.
  // This noise level is weak enough that we can still see the 2 Bell states
  // with relatively high probability.
  EXPECT_GT(counts.probability(allZero), 0.2);
  EXPECT_GT(counts.probability(allOne), 0.2);
  cudaq::unset_noise(); // clear for subsequent tests
}

struct initStateKernel {
  auto operator()(int N,
                  const std::vector<cudaq::complex> &initial_state) __qpu__ {
    cudaq::qvector q(N);
    cudaq::qubit q1(initial_state);
    for (int i = 0; i < N; ++i) {
      x<cudaq::ctrl>(q1, q[i]);
    }
    mz(q);
  }
};

TEST(HostDeviceTester, checkNoiseWithInitialState) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 26;
  // 2-qubit noise
  cudaq::kraus_op op0{cudaq::complex{0.99498743710662, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.99498743710662, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.99498743710662, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.99498743710662, 0.0}},
      op1{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0}},
      op2{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.05773502691896258},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.05773502691896258},
          {0.0, 0.05773502691896258},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.05773502691896258},
          {0.0, 0.0},
          {0.0, 0.0}},
      op3{cudaq::complex{0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.05773502691896258, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.05773502691896258, 0.0},
          {-0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.0, 0.0},
          {-0.05773502691896258, 0.0}};
  cudaq::kraus_channel cnotNoise(
      std::vector<cudaq::kraus_op>{op0, op1, op2, op3});
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(cnotNoise, 1);
  cudaq::set_noise(noise);
  constexpr int num_shots = 50;
  auto counts =
      cudaq::sample(num_shots, initStateKernel{}, numQubits,
                    std::vector<cudaq::complex>{M_SQRT1_2, M_SQRT1_2});
  counts.dump();
  int counter = 0;
  const std::string allZero(numQubits, '0');
  const std::string allOne(numQubits, '1');
  for (auto &[bits, count] : counts)
    counter += count;
  EXPECT_EQ(counter, num_shots);

  // Check results
  // Noisy results: more than just the 2 Bell states
  EXPECT_GT(counts.size(), 2);
  // Noise is applied to all qubits.
  // This noise level is weak enough that we can still see the 2 Bell states
  // with relatively high probability.
  EXPECT_GT(counts.probability(allZero), 0.2);
  EXPECT_GT(counts.probability(allOne), 0.2);
  cudaq::unset_noise(); // clear for subsequent tests
}

struct ExpPauliKernel {
  void operator()() __qpu__ {
    cudaq::qvector q(29);
    exp_pauli(1.234, q, "XYZXYZXYZXYZXYZXYZXYZXYZXYZXY");
  }
};

TEST(HostDeviceTester, checkExpPauli) {
  constexpr int nQubits = 29;
  auto ham = cudaq::spin_op::random(nQubits, 1, /*seed=*/123);
  auto result = cudaq::observe(ExpPauliKernel{}, ham);
  result.dump();
}
