/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <set>
#include <stdio.h>

#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)
struct xOp {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

struct bell {
  void operator()() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q, r);
  }
};

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_channel specification.

CUDAQ_TEST(NoiseTest, checkSimple) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel depol({cudaq::complex{0.99498743710662, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {0.99498743710662, 0.0}},

                             {cudaq::complex{0.0, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.05773502691896258, 0.0},
                              {0.0, 0.0}},

                             {cudaq::complex{0.0, 0.0},
                              {0.0, -0.05773502691896258},
                              {0.0, 0.05773502691896258},
                              {0.0, 0.0}},

                             {cudaq::complex{0.05773502691896258, 0.0},
                              {0.0, 0.0},
                              {0.0, 0.0},
                              {-0.05773502691896258, 0.0}});
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);
  cudaq::set_noise(noise);

  auto counts = cudaq::sample(xOp{});
  counts.dump();

  // In a perfect would, we'd have a single 1,
  // in a noise world, we get a 0 too.
  EXPECT_EQ(2, counts.size());

  // Can unset the noise model.
  cudaq::unset_noise();

  counts = cudaq::sample(xOp{});
  counts.dump();

  EXPECT_EQ(1, counts.size());
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_channel specification.

CUDAQ_TEST(NoiseTest, checkAmplitudeDamping) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel amplitudeDamping{{1., 0., 0., .8660254037844386},
                                        {0., 0.5, 0.0, 0.}};
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, amplitudeDamping);
  cudaq::set_noise(noise);

  auto counts = cudaq::sample(xOp{});
  counts.dump();

  EXPECT_NEAR(counts.probability("0"), .25, .1);
  EXPECT_NEAR(counts.probability("1"), .75, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_op specification.

CUDAQ_TEST(NoiseTest, checkCNOT) {
  cudaq::set_random_seed(13);
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
  noise.add_channel<cudaq::types::x>({0, 1}, cnotNoise);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(10000, bell{});
  counts.dump();
  EXPECT_TRUE(counts.size() > 2);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_channel specification.

CUDAQ_TEST(NoiseTest, checkExceptions) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel amplitudeDamping{{1., 0., 0., .8660254037844386},
                                        {0., 0.5, 0.0, 0.}};
  cudaq::noise_model noise;
  EXPECT_ANY_THROW({
    noise.add_channel<cudaq::types::x>({0, 1}, amplitudeDamping);
  });
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkDepolType) {
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(.1);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkDepolTypeSimple) {
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(1.);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  EXPECT_NEAR(counts.probability("0"), .50, .2);
  EXPECT_NEAR(counts.probability("1"), .50, .2);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support cudaq::amplitude_damping_channel.

CUDAQ_TEST(NoiseTest, checkAmpDampType) {
  cudaq::set_random_seed(13);
  cudaq::amplitude_damping_channel ad(.25);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, ad);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  EXPECT_NEAR(counts.probability("0"), .25, .1);
  EXPECT_NEAR(counts.probability("1"), .75, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support cudaq::amplitude_damping_channel.

CUDAQ_TEST(NoiseTest, checkAmpDampTypeSimple) {
  cudaq::set_random_seed(13);
  cudaq::amplitude_damping_channel ad(1.);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, ad);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_NEAR(counts.probability("0"), 1., .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkBitFlipType) {
  cudaq::set_random_seed(13);
  cudaq::bit_flip_channel bf(.1);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, bf);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  EXPECT_NEAR(counts.probability("0"), .1, .1);
  EXPECT_NEAR(counts.probability("1"), .9, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkBitFlipTypeSimple) {
  cudaq::set_random_seed(13);
  cudaq::bit_flip_channel bf(1.);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, bf);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_NEAR(counts.probability("0"), 1., .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)
// Same as above but use alternate sample interface that specifies the number of
// shots and the noise model to use.
CUDAQ_TEST(NoiseTest, checkBitFlipTypeSimpleOptions) {
  cudaq::set_random_seed(13);
  cudaq::bit_flip_channel bf(1.);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, bf);
  const std::size_t shots = 252;
  auto counts = cudaq::sample({.shots = shots, .noise = noise}, xOp{});
  // Check results
  EXPECT_EQ(1, counts.size());
  EXPECT_NEAR(counts.probability("0"), 1., .1);
  std::size_t totalShots = 0;
  for (auto &[bitstr, count] : counts)
    totalShots += count;
  EXPECT_EQ(totalShots, shots);
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkPhaseFlipType) {
  cudaq::set_random_seed(13);

  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    z(q);
    h(q);
    mz(q);
  };

  cudaq::phase_flip_channel pf(1.);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::z>({0}, pf);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(kernel);
  counts.dump();
  EXPECT_EQ(1, counts.size());
  EXPECT_NEAR(counts.probability("0"), 1., .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

template <std::size_t N>
struct xOpAll {
  void operator()() __qpu__ {
    cudaq::qarray<N> q;
    x(q);
  }
};

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkAllQubitChannel) {
  cudaq::set_random_seed(13);
  cudaq::bit_flip_channel bf(1.);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(bf);
  const std::size_t shots = 252;
  auto counts = cudaq::sample({.shots = shots, .noise = noise}, xOpAll<3>{});
  // Check results
  EXPECT_EQ(1, counts.size());
  // Noise is applied to all qubits.
  EXPECT_NEAR(counts.probability("000"), 1., .1);
  std::size_t totalShots = 0;
  for (auto &[bitstr, count] : counts)
    totalShots += count;
  EXPECT_EQ(totalShots, shots);
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_op specification.

static cudaq::kraus_channel create2pNoiseChannel() {
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
  cudaq::kraus_channel noise2q(
      std::vector<cudaq::kraus_op>{op0, op1, op2, op3});
  return noise2q;
}

template <std::size_t N>
struct bellRandom {
  void operator()(int q, int r) __qpu__ {
    cudaq::qarray<N> qubits;
    h(qubits[q]);
    x<cudaq::ctrl>(qubits[q], qubits[r]);
  }
};

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_op specification.

CUDAQ_TEST(NoiseTest, checkAllQubitChannelWithControl) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(create2pNoiseChannel(),
                                               /*numControls=*/1);
  const std::size_t shots = 1024;
  constexpr std::size_t numQubits = 5;
  std::vector<int> qubitIds(numQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  std::set<std::pair<int, int>> runs;
  do {
    const auto pair = std::make_pair(qubitIds[0], qubitIds[1]);
    if (runs.contains(pair))
      continue;
    runs.insert(pair);
    std::cout << "Testing entangling b/w " << qubitIds[0] << " and "
              << qubitIds[1] << "\n";
    auto counts =
        cudaq::sample({.shots = shots, .noise = noise}, bellRandom<numQubits>{},
                      qubitIds[0], qubitIds[1]);
    // More than 2 entangled states due to the noise.
    EXPECT_GT(counts.size(), 2);
  } while (std::next_permutation(qubitIds.begin(), qubitIds.end()));
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support arbitrary cudaq::kraus_op specification.

CUDAQ_TEST(NoiseTest, checkAllQubitChannelWithControlPrefix) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("cx", create2pNoiseChannel());
  const std::size_t shots = 1024;
  constexpr std::size_t numQubits = 5;
  std::vector<int> qubitIds(numQubits);
  std::iota(qubitIds.begin(), qubitIds.end(), 0);
  std::set<std::pair<int, int>> runs;
  do {
    const auto pair = std::make_pair(qubitIds[0], qubitIds[1]);
    if (runs.contains(pair))
      continue;
    runs.insert(pair);
    std::cout << "Testing entangling b/w " << qubitIds[0] << " and "
              << qubitIds[1] << "\n";
    auto counts =
        cudaq::sample({.shots = shots, .noise = noise}, bellRandom<numQubits>{},
                      qubitIds[0], qubitIds[1]);
    // More than 2 entangled states due to the noise.
    EXPECT_GT(counts.size(), 2);
  } while (std::next_permutation(qubitIds.begin(), qubitIds.end()));
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM)

CUDAQ_TEST(NoiseTest, checkCallbackChannel) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>(
      [](const auto &qubits, const auto &params) -> cudaq::kraus_channel {
        if (qubits.size() == 1 && qubits[0] != 2)
          return cudaq::bit_flip_channel(1.);
        return cudaq::kraus_channel();
      });
  const std::size_t shots = 252;
  auto counts = cudaq::sample({.shots = shots, .noise = noise}, xOpAll<5>{});
  // Check results
  EXPECT_EQ(1, counts.size());
  counts.dump();
  // Noise is applied to all qubits.
  // All qubits, except q[2], are flipped.
  EXPECT_NEAR(counts.probability("00100"), 1., .1);
  std::size_t totalShots = 0;
  for (auto &[bitstr, count] : counts)
    totalShots += count;
  EXPECT_EQ(totalShots, shots);
}

struct rxOp {
  void operator()(double angle) __qpu__ {
    cudaq::qubit q;
    rx(angle, q);
  }
};

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support rx gate.

CUDAQ_TEST(NoiseTest, checkCallbackChannelWithParams) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::rx>(
      [](const auto &qubits, const auto &params) -> cudaq::kraus_channel {
        EXPECT_EQ(1, params.size());
        // For testing: only add noise if the angle is positive.
        if (params[0] > 0.0)
          return cudaq::bit_flip_channel(1.);
        return cudaq::kraus_channel();
      });
  const std::size_t shots = 252;
  {
    // Rx(pi) == X
    auto counts = cudaq::sample({.shots = shots, .noise = noise}, rxOp{}, M_PI);
    // Check results
    EXPECT_EQ(1, counts.size());
    counts.dump();
    // Due to 100% bit-flip, it becomes "0".
    EXPECT_NEAR(counts.probability("0"), 1., .1);
  }
  {
    // Rx(-pi) == X
    auto counts =
        cudaq::sample({.shots = shots, .noise = noise}, rxOp{}, -M_PI);
    // Check results
    EXPECT_EQ(1, counts.size());
    counts.dump();
    // Due to our custom setup, a negative angle will have no noise.
    EXPECT_NEAR(counts.probability("1"), 1., .1);
  }
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support custom operations.

CUDAQ_REGISTER_OPERATION(CustomX, 1, 0, {0, 1, 1, 0});
CUDAQ_TEST(NoiseTest, checkCustomOperation) {
  auto kernel = []() {
    cudaq::qubit q;
    CustomX(q);
  };

  // Add channel for custom operation using the (name + operand) API
  {
    cudaq::set_random_seed(13);
    cudaq::bit_flip_channel bf(1.);
    cudaq::noise_model noise;
    noise.add_channel("CustomX", {0}, bf);
    const std::size_t shots = 252;
    auto counts = cudaq::sample({.shots = shots, .noise = noise}, kernel);
    // Check results
    EXPECT_EQ(1, counts.size());
    // Due to bit-flip noise, it becomes "0".
    EXPECT_NEAR(counts.probability("0"), 1., .1);
    std::size_t totalShots = 0;
    for (auto &[bitstr, count] : counts)
      totalShots += count;
    EXPECT_EQ(totalShots, shots);
  }

  // Add channel for custom operation using the all-qubit API
  {
    cudaq::set_random_seed(13);
    cudaq::bit_flip_channel bf(1.);
    cudaq::noise_model noise;
    noise.add_all_qubit_channel("CustomX", bf);
    const std::size_t shots = 252;
    auto counts = cudaq::sample({.shots = shots, .noise = noise}, kernel);
    // Check results
    EXPECT_EQ(1, counts.size());
    // Due to bit-flip noise, it becomes "0".
    EXPECT_NEAR(counts.probability("0"), 1., .1);
    std::size_t totalShots = 0;
    for (auto &[bitstr, count] : counts)
      totalShots += count;
    EXPECT_EQ(totalShots, shots);
  }
  // Add channel for custom operation using the callback API
  {
    cudaq::set_random_seed(13);
    cudaq::noise_model noise;
    noise.add_channel(
        "CustomX",
        [](const auto &qubits, const auto &params) -> cudaq::kraus_channel {
          return cudaq::bit_flip_channel(1.);
        });
    const std::size_t shots = 252;
    auto counts = cudaq::sample({.shots = shots, .noise = noise}, kernel);
    // Check results
    EXPECT_EQ(1, counts.size());
    // Due to bit-flip noise, it becomes "0".
    EXPECT_NEAR(counts.probability("0"), 1., .1);
    std::size_t totalShots = 0;
    for (auto &[bitstr, count] : counts)
      totalShots += count;
    EXPECT_EQ(totalShots, shots);
  }
}
#endif
