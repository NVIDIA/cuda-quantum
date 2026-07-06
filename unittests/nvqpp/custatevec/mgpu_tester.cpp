/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/domains/chemistry.h"
#include <cudaq/algorithm.h>

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

struct prepFirstQubitOne {
  void operator()(int N) __qpu__ {
    cudaq::qvector q(N);
    x(q[0]);
  }
};

struct measureFirstQubitFromState {
  void operator()(const cudaq::state &state) __qpu__ {
    cudaq::qvector q(state);
    mz(q[0]);
  }
};

CUDAQ_TEST(MGPUTester, checkZeroQubitAmplitudeEncoding) {
  const auto encoded =
      cudaq::contrib::amplitude_encode(std::vector<double>{1.0});
  EXPECT_EQ(encoded.get_num_qubits(), 0u);
  EXPECT_NEAR(encoded[0].real(), 1.0, 1e-6);
  EXPECT_NEAR(encoded[0].imag(), 0.0, 1e-6);
}

CUDAQ_TEST(MGPUTester, checkNoisyStateInputPreservesInitialState) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 28;
  constexpr int numShots = 64;
  auto state = cudaq::get_state(prepFirstQubitOne{}, numQubits);

  cudaq::depolarization_channel depol(0.1);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::h>({0}, depol);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(numShots, measureFirstQubitFromState{}, state);
  counts.dump();

  EXPECT_EQ(counts.size(), 1);
  EXPECT_NEAR(counts.probability("1"), 1.0, 1e-12);
  cudaq::unset_noise();
}

CUDAQ_TEST(MGPUTester, checkBell) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 28;
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

CUDAQ_TEST(MGPUTester, checkBernsteinVazirani) {
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

CUDAQ_TEST(MGPUTester, checkObserve) {
  const int nQubits = 30;
  auto ansatz = [&](std::vector<double> thetas) __qpu__ {
    cudaq::qvector q(nQubits);
    for (int i = 0; i < nQubits; ++i)
      rx(thetas[i], q[i]);
  };

  constexpr int numTerms = 100;
  auto ham = cudaq::spin_op::random(nQubits, numTerms, /*seed=*/123);
  ham.dump();
  const std::vector<double> params(nQubits, 1.0);
  auto result = cudaq::observe(ansatz, ham, params);
  result.dump();
}

struct test_resizing {
  void operator()() __qpu__ {
    // Start with an initial allocation of 28 qubits.
    cudaq::qvector q(28);
    cudaq::x(q);
    auto result = mz(q);
    const bool allTrue = std::all_of(result.begin(), result.end(),
                                     [](const bool v) { return v; });
    if (allTrue) {
      // Allocate two more qubits mid-circuit.
      cudaq::qvector q2(2);
      mz(q2);
    }
  }
};

CUDAQ_TEST(MGPUTester, checkResizing) {
  auto counts = cudaq::sample(test_resizing{});
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == "111111111111111111111111111100");
  counts.dump();
}

struct test_resizing_1 {
  std::vector<bool> operator()() __qpu__ {
    // Start with an initial allocation less than mqpu threshold
    cudaq::qvector q(20);
    cudaq::x(q);
    if (cudaq::mz(q[0])) {
      // Put this in a mid-circuit block to force dynamic allocation (since
      // batching is not possible)
      cudaq::qvector q2(10);
      // X every qubit except the last one
      cudaq::x(q2.slice(0, 9));
      return cudaq::measure_result::to_bool_vector(mz(q, q2));
    }
    return cudaq::measure_result::to_bool_vector(mz(q));
  }
};

CUDAQ_TEST(MGPUTester, checkResizing1) {
  constexpr int shots = 5;
  auto results = cudaq::run(shots, test_resizing_1{});
  std::string expectedBitString(30, '1');
  expectedBitString.back() = '0'; // The last one is not flipped.
  std::map<std::string, std::size_t> counts;
  for (const auto &result : results) {
    EXPECT_EQ(result.size(), 30);
    std::string bits;
    bits.reserve(result.size());
    for (auto bit : result)
      bits.push_back(bit ? '1' : '0');
    ++counts[bits];
  }
  EXPECT_EQ(1, counts.size());
  EXPECT_TRUE(counts.begin()->first == expectedBitString);
}

struct test_resizing_2 {
  std::vector<bool> operator()() __qpu__ {
    // Start with an initial allocation less than mqpu threshold
    cudaq::qvector q(24);
    cudaq::x(q);
    if (cudaq::mz(q[0])) {
      // Just add one more qubit.
      // Note: need to run this with at least 4 GPUs to force the original state
      // being distributed to the first 2 GPUs.
      cudaq::qvector q2(1);
      cudaq::h(q2);
      return cudaq::measure_result::to_bool_vector(mz(q, q2));
    }
    return cudaq::measure_result::to_bool_vector(mz(q));
  }
};

CUDAQ_TEST(MGPUTester, checkResizing2) {
  constexpr int shots = 5;
  auto results = cudaq::run(shots, test_resizing_2{});
  std::map<std::string, std::size_t> counts;
  for (const auto &result : results) {
    EXPECT_EQ(result.size(), 25);
    std::string bits;
    bits.reserve(result.size());
    for (auto bit : result)
      bits.push_back(bit ? '1' : '0');
    ++counts[bits];
  }
  EXPECT_EQ(2, counts.size());
  for (auto &[bits, count] : counts) {
    // Last qubit is in superposition state (Hadamard)
    EXPECT_TRUE(bits == "1111111111111111111111111" ||
                bits == "1111111111111111111111110");
  }
}

CUDAQ_TEST(MGPUTester, checkNoise) {
  cudaq::set_random_seed(13);
  constexpr int numQubits = 30;
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

struct ExpPauliKernel {
  void operator()() __qpu__ {
    cudaq::qvector q(30);
    exp_pauli(1.234, q, "XYZXYZXYZXYZXYZXYZXYZXYZXYZXYZ");
  }
};

CUDAQ_TEST(MGPUTester, checkExpPauli) {
  constexpr int nQubits = 30;
  auto ham = cudaq::spin_op::random(nQubits, 1, /*seed=*/123);
  auto result = cudaq::observe(ExpPauliKernel{}, ham);
  result.dump();
}
