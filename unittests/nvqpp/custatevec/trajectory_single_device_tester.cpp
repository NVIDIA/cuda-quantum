#include "CUDAQTestUtils.h"
#include <cudaq.h>
#include <gtest/gtest.h>

#include <complex>

struct xOp {
  void operator()() __qpu__ {
    cudaq::qubit q;
    x(q);
  }
};

struct xOpMultiQubits {
  void operator()(int numQubits) __qpu__ {
    cudaq::qvector q(numQubits);
    x(q);
  }
};

struct xOpFirstQubit {
  void operator()(int numQubits) __qpu__ {
    cudaq::qvector q(numQubits);
    x(q[0]);
  }
};

struct xOpInitState {
  void operator()(const std::vector<cudaq::complex> &initial_state) __qpu__ {
    cudaq::qubit q(initial_state);
    x(q);
  }
};

struct xOpExplicitCudaqState {
  void operator()(const cudaq::state &initial_state) __qpu__ {
    cudaq::qvector q(initial_state);
    x(q[0]);
  }
};

struct bell {
  void operator()() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    x<cudaq::ctrl>(q, r);
  }
};

struct nonConditionalMeasurement {
  void operator()() __qpu__ {
    cudaq::qubit q;
    h(q);
    mz(q);
    x(q);
    mz(q);
  }
};

CUDAQ_TEST(TrajectoryNoiseTest, checkSimple) {
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
}

CUDAQ_TEST(TrajectoryNoiseTest, checkAmplitudeDamping) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkCNOT) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkExceptions) {
  cudaq::set_random_seed(13);
  cudaq::kraus_channel amplitudeDamping{{1., 0., 0., .8660254037844386},
                                        {0., 0.5, 0.0, 0.}};
  cudaq::noise_model noise;
  EXPECT_ANY_THROW(
      { noise.add_channel<cudaq::types::x>({0, 1}, amplitudeDamping); });
}

CUDAQ_TEST(TrajectoryNoiseTest, checkDepolType) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkDepolTypeSimple) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkAmpDampType) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkAmpDampTypeSimple) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkBitFlipType) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkBitFlipTypeSimple) {
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

// Same as above but use alternate sample interface that specifies the number of
// shots and the noise model to use.
CUDAQ_TEST(TrajectoryNoiseTest, checkBitFlipTypeSimpleOptions) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkPhaseFlipType) {
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

template <std::size_t N>
struct xOpAll {
  void operator()() __qpu__ {
    cudaq::qarray<N> q;
    x(q);
  }
};

CUDAQ_TEST(TrajectoryNoiseTest, checkAllQubitChannel) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkAllQubitChannelWithControl) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkAllQubitChannelWithControlPrefix) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkCallbackChannel) {
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

CUDAQ_TEST(TrajectoryNoiseTest, checkCallbackChannelWithParams) {
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

CUDAQ_REGISTER_OPERATION(CustomXGate, 1, 0, {0, 1, 1, 0});
CUDAQ_TEST(TrajectoryNoiseTest, checkCustomOperation) {
  auto kernel = []() {
    cudaq::qubit q;
    CustomXGate(q);
  };

  // Add channel for custom operation using the (name + operand) API
  {
    cudaq::set_random_seed(13);
    cudaq::bit_flip_channel bf(1.);
    cudaq::noise_model noise;
    noise.add_channel("CustomXGate", {0}, bf);
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
    noise.add_all_qubit_channel("CustomXGate", bf);
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
        "CustomXGate",
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

CUDAQ_TEST(TrajectoryNoiseTest, checkInitialState) {
  cudaq::set_random_seed(13);
  constexpr double errorProb = 0.25;
  cudaq::bit_flip_channel bf(errorProb);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, bf);
  cudaq::set_noise(noise);
  const std::vector<cudaq::complex> initState{0.0, 1.0};
  auto counts = cudaq::sample(xOpInitState{}, initState);
  counts.dump();

  EXPECT_NEAR(counts.probability("0"), 1.0 - errorProb, .1);
  EXPECT_NEAR(counts.probability("1"), errorProb, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkInitialStateExplicitCudaqState) {
  cudaq::set_random_seed(13);
  constexpr double errorProb = 0.25;
  cudaq::bit_flip_channel bf(errorProb);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, bf);
  cudaq::set_noise(noise);
  auto initState =
      cudaq::state::from_data(std::vector<cudaq::complex>{0.0, 1.0});
  auto counts = cudaq::sample(xOpExplicitCudaqState{}, initState);
  counts.dump();

  EXPECT_NEAR(counts.probability("0"), 1.0 - errorProb, .1);
  EXPECT_NEAR(counts.probability("1"), errorProb, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkTrajectoryObserve) {
  cudaq::set_random_seed(13);
  constexpr double errorProb = 0.25;
  cudaq::amplitude_damping_channel ad(.25);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, ad);
  cudaq::set_noise(noise);
  auto exp_val = cudaq::observe(xOp{}, cudaq::spin_op::z(0));
  constexpr double expectedProb0 = errorProb;
  constexpr double expectedProb1 = 1.0 - expectedProb0;
  // <Z> of |1> is -1; <Z> of |0> is +1
  constexpr double expectedResult = expectedProb0 - expectedProb1;
  exp_val.dump();
  std::cout << "Exp val = " << exp_val.expectation() << "\n";
  EXPECT_NEAR(exp_val.expectation(), expectedResult, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkTrajectoryObserveMultiTerm) {
  cudaq::set_random_seed(13);
  constexpr double errorProb = 0.25;
  constexpr int numQubits = 2;
  cudaq::amplitude_damping_channel ad(.25);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(ad);
  cudaq::set_noise(noise);
  auto exp_val = cudaq::observe(
      xOpMultiQubits{}, cudaq::spin_op::z(0) + cudaq::spin_op::z(1), numQubits);
  constexpr double expectedProb0 = errorProb;
  constexpr double expectedProb1 = 1.0 - expectedProb0;
  // <Z> of |1> is -1; <Z> of |0> is +1
  constexpr double expectedResult = expectedProb0 - expectedProb1;
  exp_val.dump();
  std::cout << "Exp val = " << exp_val.expectation() << "\n";
  EXPECT_NEAR(exp_val.expectation(), numQubits * expectedResult, .1);
  // Per-term access
  EXPECT_NEAR(exp_val.expectation(cudaq::spin_op::z(0)), expectedResult, .1);
  EXPECT_NEAR(exp_val.expectation(cudaq::spin_op::z(1)), expectedResult, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}

// Noisy identity gate for noise injection
CUDAQ_REGISTER_OPERATION(NoisyIdentity, 1, 0, {1, 0, 0, 1});
struct BitFlipCode {
  std::vector<int> operator()(int numCorrectionRounds) __qpu__ {
    cudaq::qvector dataQubits(3);
    cudaq::qvector syndromeQubits(2);
    for (int i = 0; i < dataQubits.size() - 1; ++i) {
      // Encode logical qubit
      cx(dataQubits[i], dataQubits[i + 1]);
    }

    for (int round = 0; round < numCorrectionRounds; ++round) {
      // Apply noise channels
      for (int i = 0; i < dataQubits.size(); ++i) {
        NoisyIdentity(dataQubits[i]);
      }

      // Syndrome measurement
      cx(dataQubits[0], syndromeQubits[0]);
      cx(dataQubits[1], syndromeQubits[0]);

      cx(dataQubits[1], syndromeQubits[1]);
      cx(dataQubits[2], syndromeQubits[1]);

      auto syndromes = mz(syndromeQubits);

      // Apply correction
      if (syndromes[0] && syndromes[1]) {
        // Bit-flip error in the dataQubits[1]
        x(dataQubits[1]);
        // Reset
        x(syndromeQubits[0]);
        x(syndromeQubits[1]);
      } else if (syndromes[0]) {
        // Bit-flip error in the dataQubits[0]
        x(dataQubits[0]);
        // Reset
        x(syndromeQubits[0]);
      } else if (syndromes[1]) {
        // Bit-flip error in the dataQubits[2]
        x(dataQubits[2]);
        // Reset
        x(syndromeQubits[1]);
      }
    }

    // Finish all rounds
    // Measure data qubits
    auto finalResults = mz(dataQubits);
    return {finalResults[0], finalResults[1], finalResults[2]};
  }
};

CUDAQ_TEST(TrajectoryNoiseTest, checkMidCircuitMeasurement) {
  cudaq::set_random_seed(13);
  constexpr double errorProb = 0.25;
  cudaq::bit_flip_channel bf(errorProb);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("NoisyIdentity", bf);
  constexpr int numRounds = 3;
  constexpr int shots = 1000;

  // Probability that there were more than 1 bit flip errors in a round
  constexpr double probMoreThanOneError =
      3.0 * errorProb * errorProb * (1.0 - errorProb) +
      errorProb * errorProb * errorProb;
  // Odd number of channel errors will give us a bit-flip
  const double logicalBitFlipProb =
      (1.0 - std::pow(1.0 - 2. * probMoreThanOneError,
                      static_cast<double>(numRounds))) /
      2.0;

  std::cout << "Logical bit-flip probability = " << logicalBitFlipProb << "\n";
  auto results = cudaq::run(shots, noise, BitFlipCode{}, numRounds);

  std::map<std::string, std::size_t> dataQubitCounts;
  for (const auto &result : results) {
    EXPECT_EQ(result.size(), 3);

    std::string bitString;
    bitString.reserve(3);
    for (auto bit : result)
      bitString.push_back(bit ? '1' : '0');
    ++dataQubitCounts[bitString];
  }

  for (const auto &[bits, count] : dataQubitCounts)
    std::cout << bits << ":" << count << " ";
  std::cout << "\n";

  const auto probability = [&](std::string_view bits) {
    auto iter = dataQubitCounts.find(std::string(bits));
    return iter == dataQubitCounts.end()
               ? 0.0
               : static_cast<double>(iter->second) / static_cast<double>(shots);
  };

  EXPECT_EQ(2, dataQubitCounts.size());
  EXPECT_NEAR(probability("000"), 1.0 - logicalBitFlipProb, 0.1);
  EXPECT_NEAR(probability("111"), logicalBitFlipProb, 0.1);
}

CUDAQ_TEST(TrajectoryNoiseTest,
           checkNonConditionalMidCircuitMeasurementReplay) {
  cudaq::set_random_seed(13);
  constexpr std::size_t shots = 1000;
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("h", cudaq::depolarization_channel(0.01));

  // Non-explicit `sample` supports only terminal measurements. This kernel
  // measures, then applies X and measures again -- a mid-circuit measurement,
  // which must be rejected unless explicit measurements are used.
  EXPECT_ANY_THROW(
      cudaq::sample(cudaq::sample_options{.shots = shots, .noise = noise},
                    nonConditionalMeasurement{}));

  const cudaq::sample_options options{
      .shots = shots, .noise = noise, .explicit_measurements = true};

  const auto explicitCounts =
      cudaq::sample(options, nonConditionalMeasurement{});
  std::size_t total = 0;
  for (const auto &[bits, count] : explicitCounts) {
    ASSERT_EQ(bits.size(), 2u);
    EXPECT_NE(bits[0], bits[1]);
    total += count;
  }
  EXPECT_EQ(total, shots);
  EXPECT_NEAR(explicitCounts.probability("01"), 0.5, 0.1);
  EXPECT_NEAR(explicitCounts.probability("10"), 0.5, 0.1);
}

CUDAQ_TEST(TrajectoryNoiseTest, checkObserveHamiltonianSmallNoise) {
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  // Inject a small-level of noise
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(1e-4);
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
  EXPECT_NEAR(result, -1.7487, 1e-3);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkObserveHamiltonianLargeNoise) {
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  // Inject a large-level of noise
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(0.1); // 10% error rate
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
  // We expect to see the value changes with this high-level of noise.
  EXPECT_GT(std::abs(result + -1.7487), 0.25);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkQubitReset) {
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(0.1);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(depol);
  cudaq::set_noise(noise);

  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    x(q);
    reset(q);
    mz(q);
  };

  auto result = cudaq::sample(kernel);
  result.dump();

  // Check results
  EXPECT_EQ(1, result.size());
  // Because of reset, the final state is 0.
  EXPECT_NEAR(result.probability("0"), 1., 1e-9);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkResetReplaysNoisyPrefix) {
  cudaq::set_random_seed(13);
  cudaq::bit_flip_channel flip(0.5);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({1}, flip);
  cudaq::set_noise(noise);

  auto kernel = []() __qpu__ {
    cudaq::qvector q(2);
    x(q[1]);
    reset(q[0]);
    mz(q[1]);
  };

  constexpr int shots = 1000;
  const auto result = cudaq::sample(shots, kernel);
  result.dump();

  // Resetting `q[0]` must not execute the noisy prefix on `q[1]` once.
  // Every final trajectory independently selects the bit-flip branch.
  EXPECT_EQ(result.size(), 2);
  EXPECT_NEAR(result.probability("0"), 0.5, 0.1);
  EXPECT_NEAR(result.probability("1"), 0.5, 0.1);
  cudaq::unset_noise();
}

CUDAQ_TEST(TrajectoryNoiseTest,
           checkObserveHamiltonianSmallNoiseWithInitialState) {
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  // Inject a small-level of noise
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(1e-4);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::ry>(depol);
  cudaq::set_noise(noise);

  auto ansatz = [](double theta,
                   const std::vector<cudaq::complex> &initial_state) __qpu__ {
    cudaq::qubit q(initial_state);
    cudaq::qubit r;
    ry(theta, r);
    x<cudaq::ctrl>(r, q);
  };

  // Init the state as |1> (rather than apply an X gate)
  double result =
      cudaq::observe(ansatz, h, 0.59, std::vector<cudaq::complex>{0.0, 1.0});
  printf("Energy value = %lf\n", result);
  EXPECT_NEAR(result, -1.7487, 1e-3);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest,
           checkObserveMultiTermRestoresCustomInitialState) {
  cudaq::bit_flip_channel flip(1.0);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, flip);
  cudaq::observe_options options;
  options.shots = 32;
  options.noise = noise;

  auto ansatz = [](const std::vector<cudaq::complex> &initial) __qpu__ {
    cudaq::qubit q(initial);
    x(q);
  };
  const auto initial = std::vector<cudaq::complex>{0.0, 1.0};
  const auto xTerm = cudaq::spin_op::x(0);
  const auto zTerm = cudaq::spin_op::z(0);
  cudaq::set_random_seed(13);
  auto multi = cudaq::observe(options, ansatz, xTerm + zTerm, initial);
  cudaq::set_random_seed(13);
  auto singleX = cudaq::observe(options, ansatz, xTerm, initial);
  cudaq::set_random_seed(13);
  auto singleZ = cudaq::observe(options, ansatz, zTerm, initial);

  EXPECT_NEAR(multi.expectation(xTerm), singleX.expectation(xTerm), 1e-9);
  EXPECT_NEAR(multi.expectation(zTerm), singleZ.expectation(zTerm), 1e-9);
}

CUDAQ_TEST(TrajectoryNoiseTest, checkGetState) {
  auto kernel = []() __qpu__ {
    cudaq::qubit q;
    x(q);
  };
  cudaq::set_random_seed(13);
  constexpr double bitFlipProb = 0.25;
  cudaq::bit_flip_channel bf(bitFlipProb);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel<cudaq::types::x>(bf);
  cudaq::set_noise(noise);
  constexpr int numRuns = 100;
  // If running as a trajectory simulation, get_state will return the final
  // state of a single trajectory. Under a bif-flip channel, this can be |0>
  // ([1, 0]) with the probability = bitFlipProb.
  int numOne = 0;
  int numZero = 0;
  for (int i = 0; i < numRuns; ++i) {
    auto state = cudaq::get_state(kernel);
    EXPECT_TRUE(state[0] == 1.0 || state[1] == 1.0);
    if (state[0] == 1.0)
      numZero++;
    else
      numOne++;
  }

  std::cout << "Number of zero states = " << numZero << "\n";
  EXPECT_NEAR(static_cast<double>(numZero) / numRuns, bitFlipProb, 0.1);
  cudaq::unset_noise(); // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkObserveHamiltonianWithShots) {
  cudaq::spin_op h =
      5.907 - 2.1433 * cudaq::spin_op::x(0) * cudaq::spin_op::x(1) -
      2.1433 * cudaq::spin_op::y(0) * cudaq::spin_op::y(1) +
      .21829 * cudaq::spin_op::z(0) - 6.125 * cudaq::spin_op::z(1);
  // Inject a small-level of noise
  cudaq::set_random_seed(13);
  cudaq::depolarization_channel depol(1e-5);
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
  constexpr int numShots = 8192;
  double result = cudaq::observe(numShots, ansatz, h, 0.59);
  printf("Energy value = %lf\n", result);
  EXPECT_NEAR(result, -1.7487, 0.1); // Large limit due to shot noise
  cudaq::unset_noise();              // clear for subsequent tests
}

CUDAQ_TEST(TrajectoryNoiseTest, checkObserveOptions) {
  constexpr double errorProb = 0.25;
  cudaq::amplitude_damping_channel ad(.25);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, ad);
  // After X the qubit is |1>; amplitude damping returns it to |0> with
  // probability `errorProb`, so the exact expectation is
  // <Z> = P(0) - P(1) = errorProb - (1 - errorProb).
  constexpr double expectedResult = errorProb - (1.0 - errorProb);

  // Trajectory sampling is a Monte Carlo estimator of <Z>: each trajectory
  // contributes z in {+1, -1}, so Var(z) = 1 - <Z>^2 and the estimator's
  // standard error is sqrt(Var/N). Comparing per-seed errors across two N is
  // only true in expectation, so instead assert each estimate lies within a
  // 5-sigma band of the exact value. That band shrinks as 1/sqrt(N) (checking
  // the num_trajectories knob is honored), keeps the false-failure probability
  // ~1e-6, and is independent of the seed and of the GPU floating-point/RNG.
  const double variance = 1.0 - expectedResult * expectedResult;
  const auto tolerance = [&](std::size_t trajectories) {
    return 5.0 * std::sqrt(variance / static_cast<double>(trajectories));
  };

  cudaq::observe_options opt;
  opt.noise = noise;

  cudaq::set_random_seed(13);
  opt.num_trajectories = 1024;
  const double result1 = cudaq::observe(opt, xOp{}, cudaq::spin_op::z(0));
  EXPECT_NEAR(result1, expectedResult, tolerance(1024));

  cudaq::set_random_seed(13);
  opt.num_trajectories = 8192;
  const double result2 = cudaq::observe(opt, xOp{}, cudaq::spin_op::z(0));
  EXPECT_NEAR(result2, expectedResult, tolerance(8192));
}

CUDAQ_TEST(TrajectoryNoiseTest, checkBatchedObserveSaturatedGpuMemory) {
  constexpr int numQubits = 15;
  constexpr std::size_t logicalStateBytes = std::size_t{100} << 30;
#ifdef CUDAQ_BACKEND_CUSTATEVEC_FP32
  constexpr std::size_t stateBytes =
      (std::size_t{1} << numQubits) * sizeof(std::complex<float>);
#else
  constexpr std::size_t stateBytes =
      (std::size_t{1} << numQubits) * sizeof(std::complex<double>);
#endif
  constexpr std::size_t numTrajectories = logicalStateBytes / stateBytes;

  cudaq::bit_flip_channel flip(1.0);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, flip);
  cudaq::observe_options options;
  options.noise = noise;
  options.num_trajectories = numTrajectories;

  auto result =
      cudaq::observe(options, xOpFirstQubit{}, cudaq::spin_op::z(0), numQubits);
  EXPECT_NEAR(result.expectation(), 1.0, 1.e-5);
}

CUDAQ_REGISTER_OPERATION(CustomId2Gate, 2, 0,
                         {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
CUDAQ_TEST(TrajectoryNoiseTest, checkTwoQubitNoiseBug) {
  auto kernel = []() {
    cudaq::qvector q(2);
    x(q);
    CustomId2Gate(q[0], q[1]);
  };

  cudaq::kraus_op op0{cudaq::complex{0.65465411, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.65465411, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.65465411, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.0, 0.0},
                      {0.65465411, 0.0}},
      op1{cudaq::complex{0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0}},
      op2{cudaq::complex{0.0, 0.0},
          {0.0, -0.37796428},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.37796428},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, -0.37796428},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.37796428},
          {0.0, 0.0}},
      op3{cudaq::complex{0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {-0.37796428, 0.0}},

      op4{cudaq::complex{0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.0, 0.0},
          {0.37796428, 0.0},
          {0.0, 0.0},
          {0.0, 0.0}};
  cudaq::kraus_channel noise2q(
      std::vector<cudaq::kraus_op>{op0, op1, op2, op3, op4});

  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  noise.add_all_qubit_channel("CustomId2Gate", noise2q);
  const std::size_t shots = 1000;
  auto counts = cudaq::sample({.shots = shots, .noise = noise}, kernel);
  counts.dump();
  // Check results
  // Noisy results (more than one count)
  EXPECT_GT(counts.size(), 1);
  std::size_t totalShots = 0;
  for (auto &[bitstr, count] : counts)
    totalShots += count;
  EXPECT_EQ(totalShots, shots);
}

CUDAQ_TEST(TrajectoryNoiseTest, checkApplyNoiseOrdering) {
  constexpr double errorProb = 0.1;
  // Check that applyNoise is ordered correctly w.r.t. gate dispatching.
  auto steane_code = [errorProb]() __qpu__ {
    cudaq::qvector data_qubits(7);
    cudaq::qvector ancilla_qubits(3);
    // Create a superposition over all possible combinations of parity check
    // bits
    h(data_qubits[4]);
    h(data_qubits[5]);
    h(data_qubits[6]);
    // Entangle states to enforce constraints of parity check matrix
    x<cudaq::ctrl>(data_qubits[0], data_qubits[1]);
    x<cudaq::ctrl>(data_qubits[0], data_qubits[2]);

    x<cudaq::ctrl>(data_qubits[4], data_qubits[0]);
    x<cudaq::ctrl>(data_qubits[4], data_qubits[1]);
    x<cudaq::ctrl>(data_qubits[4], data_qubits[3]);

    x<cudaq::ctrl>(data_qubits[5], data_qubits[0]);
    x<cudaq::ctrl>(data_qubits[5], data_qubits[2]);
    x<cudaq::ctrl>(data_qubits[5], data_qubits[3]);

    x<cudaq::ctrl>(data_qubits[6], data_qubits[1]);
    x<cudaq::ctrl>(data_qubits[6], data_qubits[2]);
    x<cudaq::ctrl>(data_qubits[6], data_qubits[3]);

    // Apply noise to the data qubits
    // Note: noise is applied to qubits 4-6, which have H gates applied.
    // This is to check the ordering of apply_noise w.r.t. gate dispatching.
    cudaq::apply_noise<cudaq::x_error>(errorProb, data_qubits[5]);

    // Detect X errors
    h(ancilla_qubits);
    z<cudaq::ctrl>(ancilla_qubits[0], data_qubits[0]);
    z<cudaq::ctrl>(ancilla_qubits[0], data_qubits[1]);
    z<cudaq::ctrl>(ancilla_qubits[0], data_qubits[3]);
    z<cudaq::ctrl>(ancilla_qubits[0], data_qubits[4]);

    z<cudaq::ctrl>(ancilla_qubits[1], data_qubits[0]);
    z<cudaq::ctrl>(ancilla_qubits[1], data_qubits[2]);
    z<cudaq::ctrl>(ancilla_qubits[1], data_qubits[3]);
    z<cudaq::ctrl>(ancilla_qubits[1], data_qubits[5]);

    z<cudaq::ctrl>(ancilla_qubits[2], data_qubits[1]);
    z<cudaq::ctrl>(ancilla_qubits[2], data_qubits[2]);
    z<cudaq::ctrl>(ancilla_qubits[2], data_qubits[3]);
    z<cudaq::ctrl>(ancilla_qubits[2], data_qubits[6]);
    h(ancilla_qubits);
    mz(ancilla_qubits);
  };
  cudaq::set_random_seed(13);
  cudaq::noise_model noise;
  const std::size_t shots = 1000;
  auto counts = cudaq::sample({.shots = shots, .noise = noise}, steane_code);
  counts.dump();
  // Check results
  // Noisy results (2 syndrome results for noise and no noise)
  EXPECT_EQ(counts.size(), 2);
  // Four standard deviations for p=0.1 and 1000 binomial samples is 0.038.
  EXPECT_NEAR(counts.probability("010"), errorProb, 0.04);
}

CUDAQ_TEST(TrajectoryNoiseTest, checkMultipleChannelsSameGate) {
  cudaq::set_random_seed(13);
  cudaq::amplitude_damping_channel ad1(0.25);
  cudaq::amplitude_damping_channel ad2(0.5);
  cudaq::noise_model noise;
  // Two amplitude damping channels on the same qubit for X gate
  // These channels will be applied sequentially.
  noise.add_channel<cudaq::types::x>({0}, ad1);
  noise.add_channel<cudaq::types::x>({0}, ad2);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  constexpr double expectedProb1 =
      (1.0 - 0.25) * (1.0 - 0.5); // No decay from |1> to |0>
  constexpr double expectedProb0 = 1.0 - expectedProb1;
  // Check results
  EXPECT_EQ(2, counts.size());
  EXPECT_NEAR(counts.probability("0"), expectedProb0, .1);
  cudaq::unset_noise(); // clear for subsequent tests
}
