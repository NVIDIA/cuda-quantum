/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <set>
#include <stdio.h>

#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)
struct xOp {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

struct xOp2 {
  void operator()() __qpu__ {
    cudaq::qvector q(2);
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

struct bell_depolarization2 {
  void operator()(double prob) __qpu__ {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q, r);
    cudaq::apply_noise<cudaq::depolarization2>(prob, q, r);
  }
};

template <typename T>
struct bell_error {
  void operator()(double prob) __qpu__ {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q, r);
    cudaq::apply_noise<T>(prob, q);
    cudaq::apply_noise<T>(prob, r);
  }
};

// FIXME is this supposed to work?
template <typename T>
struct bell_error_vec {
  void operator()(double prob) __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    cudaq::apply_noise<T>(prob, q);
  }
};

struct bell_depolarization2_vec {
  void operator()(double prob) __qpu__ {
    cudaq::qvector q(2);
    h(q[0]);
    x<cudaq::ctrl>(q[0], q[1]);
    cudaq::apply_noise<cudaq::depolarization2>(prob, q);
  }
};

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
// Stim does not support arbitrary cudaq::kraus_channel specification.

namespace test::hello {
struct hello_world : public ::cudaq::kraus_channel {
  static constexpr std::size_t num_parameters = 1;
  static constexpr std::size_t num_targets = 1;

  hello_world(const std::vector<cudaq::real> &params) {
    std::vector<cudaq::complex> k0v{std::sqrt(1 - params[0]), 0, 0,
                                    std::sqrt(1 - params[0])},
        k1v{0, std::sqrt(params[0]), std::sqrt(params[0]), 0};
    push_back(cudaq::kraus_op(k0v));
    push_back(cudaq::kraus_op(k1v));
    validateCompleteness();
    generateUnitaryParameters();
  }
  REGISTER_KRAUS_CHANNEL("test::hello::hello_world");
};
} // namespace test::hello

__qpu__ void test2(double p) {
  cudaq::qubit q;
  x(q);
  cudaq::apply_noise<test::hello::hello_world>({0.2}, q);
}

__qpu__ void test3(double p) {
  cudaq::qubit q;
  x(q);
  cudaq::apply_noise<test::hello::hello_world>(0.2, q);
}

__qpu__ int test4(double p) {
  cudaq::qubit q;
  x(q);
  cudaq::apply_noise<test::hello::hello_world>(0.2, q);
  return mz(q);
}

CUDAQ_TEST(NoiseTest, checkFineGrainArg) {
  {
    cudaq::noise_model noise;
    noise.register_channel<test::hello::hello_world>();

    auto counts = cudaq::sample({.noise = noise}, test3, .7);
    counts.dump();
    EXPECT_TRUE(counts.size() == 2);
  }
  {
    // test warning emitted / no noise applied for case where
    // no noise model is specified
    auto counts = cudaq::sample(test3, .7);
    counts.dump();
    EXPECT_TRUE(counts.size() == 1);
  }
  {
    // test warning emitted / no noise applied for case where
    // noise mnodel is provided but custom channel not registered.
    cudaq::noise_model noise;
    auto counts = cudaq::sample({.noise = noise}, test3, .7);
    counts.dump();
    EXPECT_TRUE(counts.size() == 1);
  }
  {
    // test noisy kernel invocation
    cudaq::noise_model noise;
    noise.register_channel<test::hello::hello_world>();
    cudaq::set_noise(noise);
    std::set<int> res;
    for (std::size_t i = 0; i < 100; i++)
      res.insert(test4(.7));
    EXPECT_EQ(res.size(), 2);
    cudaq::unset_noise();
  }
  {
    // test noisy kernel invocation,
    // channel not registered, should get no noise
    cudaq::noise_model noise;
    cudaq::set_noise(noise);
    std::set<int> res;
    for (std::size_t i = 0; i < 100; i++)
      res.insert(test4(.7));
    EXPECT_EQ(res.size(), 1);
    cudaq::unset_noise();
  }
  {
    // test noisy kernel invocation,
    // no noise, should get no application
    std::set<int> res;
    for (std::size_t i = 0; i < 100; i++)
      res.insert(test4(.7));
    EXPECT_EQ(res.size(), 1);
  }
}

CUDAQ_TEST(NoiseTest, checkFineGrainVec) {
  {
    cudaq::noise_model noise;
    noise.register_channel<test::hello::hello_world>();

    auto counts = cudaq::sample({.noise = noise}, test2, .7);
    counts.dump();
    EXPECT_TRUE(counts.size() == 2);
  }
  {
    // test warning emitted / no noise applied for case where
    // no noise model is specified
    auto counts = cudaq::sample(test2, .7);
    counts.dump();
    EXPECT_TRUE(counts.size() == 1);
  }
  {
    // test warning emitted / no noise applied for case where
    // noise mnodel is provided but custom channel not registered.
    cudaq::noise_model noise;
    auto counts = cudaq::sample({.noise = noise}, test2, .7);
    counts.dump();
    EXPECT_TRUE(counts.size() == 1);
  }
}

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

#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET_MPS)
CUDAQ_TEST(NoiseTest, checkAmplitudeDamping2) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel amplitudeDamping{{1., 0., 0., .8660254037844386},
                                        {0., 0.5, 0.0, 0.}};
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(amplitudeDamping);
  cudaq::set_noise(noise);

  auto counts = cudaq::sample(xOp2{});
  counts.dump();

  EXPECT_NEAR(counts.probability("00"), 0.0625, .1);
  EXPECT_NEAR(counts.probability("10"), 0.1875, .1);
  EXPECT_NEAR(counts.probability("01"), 0.1875, .1);
  EXPECT_NEAR(counts.probability("11"), 0.5625, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}
#endif

#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
// Stim does not support arbitrary cudaq::kraus_op specification.

CUDAQ_TEST(NoiseTest, checkCNOT) {
  cudaq::set_random_seed(13);
  // 1% depolarization
  cudaq::kraus_op op0{cudaq::complex{0.94868329805, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.94868329805, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.94868329805, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.94868329805, 0.0}},
      op1{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.18257418583, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.18257418583, 0.0},
          {0.18257418583, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.18257418583, 0.0},
          {0.0, 0.0},
          {0.0, 0.0}},
      op2{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.18257418583},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.18257418583},
          {0.0, 0.18257418583},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.18257418583},
          {0.0, 0.0},
          {0.0, 0.0}},
      op3{cudaq::complex{0.18257418583, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.18257418583, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.18257418583, 0.0},
          {-0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.0, 0.0},
          {-0.18257418583, 0.0}};
  cudaq::kraus_channel cnotNoise(
      std::vector<cudaq::kraus_op>{op0, op1, op2, op3});
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0, 1}, cnotNoise);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(100, bell{});
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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

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

CUDAQ_TEST(NoiseTest, checkApplyDepol2) {
  cudaq::set_random_seed(13);
  double probability = 0.1;
  cudaq::noise_model noise{};
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(bell_depolarization2{}, probability);
  counts.dump();
  EXPECT_EQ(4, counts.size());
  counts = cudaq::sample(bell_depolarization2_vec{}, probability);
  counts.dump();
  EXPECT_EQ(4, counts.size());
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(NoiseTest, checkApplySimplePauliErrors) {
  cudaq::set_random_seed(13);
  double probability = 0.1;
  cudaq::noise_model noise{};
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(bell_error<cudaq::x_error>{}, probability);
  counts.dump();
  EXPECT_EQ(4, counts.size());
  counts = cudaq::sample(bell_error<cudaq::y_error>{}, probability);
  counts.dump();
  EXPECT_EQ(4, counts.size());
  counts = cudaq::sample(bell_error<cudaq::z_error>{}, probability);
  counts.dump();
  EXPECT_EQ(2, counts.size());
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

CUDAQ_TEST(NoiseTest, checkDepolTypeSimple) {
  cudaq::set_random_seed(13);
  // Complete depolarizing channel
  // (https://en.wikipedia.org/wiki/Quantum_depolarizing_channel) to produce a
  // maximally-mixed state: lambda = 1.0 => p/3 == lambda/4 => p = 3/4 = 0.75
  cudaq::depolarization_channel depol(0.75);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  // maximally-mixed state (i.e., 50/50 distribution)
  EXPECT_NEAR(counts.probability("0"), .50, .2);
  EXPECT_NEAR(counts.probability("1"), .50, .2);
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif
#if defined(CUDAQ_BACKEND_DM)
// Stim does not support cudaq::amplitude_damping_channel.

CUDAQ_TEST(NoiseTest, checkAmpDampType) {
  cudaq::set_random_seed(13);
  {
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
  {
    cudaq::amplitude_damping ad(.25);
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
}

CUDAQ_TEST(NoiseTest, checkPhaseDampType) {
  struct phase_test {
    void operator()(double prob) __qpu__ {
      cudaq::qubit q;
      h(q);
      cudaq::apply_noise<cudaq::phase_damping>(prob, q);
      h(q);
      mz(q);
    }
  };

  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(phase_test{}, 0.25);
  counts.dump();
  // No errors would have counts.size() == 1, but errors would produce
  // counts.size() == 2.
  EXPECT_EQ(2, counts.size());
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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)
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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

CUDAQ_TEST(NoiseTest, checkPauli1) {
  cudaq::set_random_seed(13);

  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    cudaq::apply_noise<cudaq::pauli1>(0.1, 0.1, 0.1, q);
    cudaq::apply_noise<cudaq::pauli1>(0.1, 0.1, 0.1, r);
    mz(q);
    mz(r);
  };

  auto counts = cudaq::sample(cudaq::sample_options{}, kernel);
  counts.dump();
  EXPECT_EQ(4, counts.size());
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

CUDAQ_TEST(NoiseTest, checkPauli2) {
  cudaq::set_random_seed(13);

  auto kernel = [](std::vector<double> parms) __qpu__ {
    cudaq::qubit q, r;
    cudaq::apply_noise<cudaq::pauli2>(parms, q, r);
    mz(q);
    mz(r);
  };

  std::vector<double> probs(15, 0.9375 / 15);
  auto counts = cudaq::sample(cudaq::sample_options{}, kernel, probs);
  counts.dump();
  EXPECT_EQ(4, counts.size());
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

template <std::size_t N>
struct xOpAll {
  void operator()() __qpu__ {
    cudaq::qarray<N> q;
    x(q);
  }
};

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
// Stim does not support arbitrary cudaq::kraus_op specification.

static cudaq::kraus_channel create2pNoiseChannel() {
  // 20% depolarization
  cudaq::kraus_op op0{cudaq::complex{0.894427191, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.894427191, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.894427191, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.894427191, 0.0}},
      op1{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.25819888974, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.25819888974, 0.0},
          {0.25819888974, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.25819888974, 0.0},
          {0.0, 0.0},
          {0.0, 0.0}},
      op2{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.25819888974},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.25819888974},
          {0.0, 0.25819888974},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.25819888974},
          {0.0, 0.0},
          {0.0, 0.0}},
      op3{cudaq::complex{0.25819888974, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.25819888974, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.25819888974, 0.0},
          {-0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.0, 0.0},
          {-0.25819888974, 0.0}};
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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
// Stim does not support arbitrary cudaq::kraus_op specification.

CUDAQ_TEST(NoiseTest, checkAllQubitChannelWithControl) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(create2pNoiseChannel(),
                                               /*numControls=*/1);
  const std::size_t shots = 100;
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
    counts.dump();
    // More than 2 entangled states due to the noise.
    EXPECT_GT(counts.size(), 2);
  } while (std::next_permutation(qubitIds.begin(), qubitIds.end()));
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
// Stim does not support arbitrary cudaq::kraus_op specification.

CUDAQ_TEST(NoiseTest, checkAllQubitChannelWithControlPrefix) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("cx", create2pNoiseChannel());
  const std::size_t shots = 100;
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
    counts.dump();
    // More than 2 entangled states due to the noise.
    EXPECT_GT(counts.size(), 2);
  } while (std::next_permutation(qubitIds.begin(), qubitIds.end()));
}

#endif
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

CUDAQ_TEST(NoiseTest, checkCallbackChannel) {
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>(
      [](const auto &qubits, const auto &params) -> cudaq::kraus_channel {
        if (qubits.size() == 1 && qubits[0] != 2)
          return cudaq::bit_flip_channel(1.);
        return cudaq::kraus_channel();
      });
  const std::size_t shots = 100;
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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
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
#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
// Stim does not support custom operations.

CUDAQ_REGISTER_OPERATION(CustomXOp, 1, 0, {0, 1, 1, 0});
CUDAQ_TEST(NoiseTest, checkCustomOperation) {
  auto kernel = []() {
    cudaq::qubit q;
    CustomXOp(q);
  };

  // Add channel for custom operation using the (name + operand) API
  {
    cudaq::set_random_seed(13);
    cudaq::bit_flip_channel bf(1.);
    cudaq::noise_model noise;
    noise.add_channel("CustomXOp", {0}, bf);
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
    noise.add_all_qubit_channel("CustomXOp", bf);
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
        "CustomXOp",
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

#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_STIM) ||                \
    defined(CUDAQ_BACKEND_TENSORNET)

CUDAQ_TEST(NoiseTest, checkMeasurementNoise) {
  cudaq::set_random_seed(13);
  constexpr double bitFlipRate = 0.25;
  cudaq::bit_flip_channel bf(bitFlipRate);
  cudaq::noise_model noise;
  // 25% bit flipping during measurement
  noise.add_channel("mz", {0}, bf);
  cudaq::set_noise(noise);
  {
    auto kernel = []() {
      cudaq::qubit q;
      x(q);
      mz(q);
    };
    auto counts = cudaq::sample(1000, kernel);
    counts.dump();
    // Due to measurement errors, we have both 0/1 results.
    EXPECT_EQ(2, counts.size());
    EXPECT_NEAR(counts.probability("0"), bitFlipRate, 0.1);
    EXPECT_NEAR(counts.probability("1"), 1.0 - bitFlipRate, 0.1);
  }
  {
    auto kernel = []() {
      cudaq::qvector q(2);
      x(q);
      mz(q);
    };
    auto counts = cudaq::sample(1000, kernel);
    counts.dump();
    // We only have measurement noise on the first qubit.
    EXPECT_EQ(2, counts.size());
    EXPECT_NEAR(counts.probability("01"), bitFlipRate, 0.1);
    EXPECT_NEAR(counts.probability("11"), 1.0 - bitFlipRate, 0.1);
  }
  cudaq::unset_noise(); // clear for subsequent tests
}

#endif

#if defined(CUDAQ_BACKEND_DM) || defined(CUDAQ_BACKEND_TENSORNET)
CUDAQ_TEST(NoiseTest, checkObserveHamiltonianWithNoise) {

  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(0.1);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(depol);
  noise.add_all_qubit_channel<cudaq::types::ry>(depol);
  cudaq::set_noise(noise);

  auto ansatz = [](double theta) __qpu__ {
    cudaq::qubit q, r;
    x(q);
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  double result = cudaq::observe(ansatz, h, 0.59);
  printf("Energy value = %lf\n", result);
  EXPECT_GT(std::abs(result + 1.7487), 0.1);
  cudaq::unset_noise(); // clear for subsequent tests
}
#endif

#if defined(CUDAQ_BACKEND_TENSORNET)
CUDAQ_REGISTER_OPERATION(CustomIdOp, 1, 0, {1, 0, 0, 1});

CUDAQ_TEST(NoiseTest, checkNoiseMatrixRepresentation) {
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(
      1.0); // 1/3 probability of X, Y, Z noise op.
  cudaq::noise_model noise;
  noise.add_channel("CustomIdOp", {0}, depol);
  cudaq::set_noise(noise);
  const auto noisyCircuit = []() __qpu__ {
    cudaq::qubit q;
    h(q);
    CustomIdOp(q); // inject noise
  };

  // Referent states for X, Y, or Z noise
  std::vector<std::complex<double>> stateVecX(2), stateVecY(2), stateVecZ(2);

  cudaq::get_state([]() __qpu__ {
    cudaq::qubit q;
    h(q);
    x(q);
  }).to_host(stateVecX.data(), 2);
  cudaq::get_state([]() __qpu__ {
    cudaq::qubit q;
    h(q);
    y(q);
  }).to_host(stateVecY.data(), 2);
  cudaq::get_state([]() __qpu__ {
    cudaq::qubit q;
    h(q);
    z(q);
  }).to_host(stateVecZ.data(), 2);

  const auto checkEqualVec = [](const auto &vec1, const auto &vec2) {
    constexpr double tol = 1e-12;
    const auto vecSize = vec1.size();
    if (vec2.size() != vecSize)
      return false;
    for (std::size_t i = 0; i < vecSize; ++i) {
      if (std::abs(vec1[i] - vec2[i]) > tol)
        return false;
    }
    return true;
  };

  for (int i = 0; i < 10; ++i) {
    std::vector<std::complex<double>> noisyStateVec(2);
    auto noisyState = cudaq::get_state(noisyCircuit);
    noisyState.to_host(noisyStateVec.data(), 2);
    const auto equalX = checkEqualVec(noisyStateVec, stateVecX);
    const auto equalY = checkEqualVec(noisyStateVec, stateVecY);
    const auto equalZ = checkEqualVec(noisyStateVec, stateVecZ);
    // One of these expected output states
    EXPECT_TRUE(equalX || equalY || equalZ);
  }
  cudaq::unset_noise(); // clear for subsequent tests
}
#endif
