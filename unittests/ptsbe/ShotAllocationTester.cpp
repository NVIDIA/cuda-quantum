/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/ShotAllocationStrategy.h"
#include <array>

using namespace cudaq;
using namespace cudaq::ptsbe;

// A simple trajectory for testing
static KrausTrajectory makeTrajectory(std::size_t id, double prob,
                                      std::size_t errors = 0,
                                      std::size_t multiplicity = 1) {
  std::vector<KrausSelection> selections;
  for (std::size_t i = 0; i < errors; ++i) {
    selections.push_back(KrausSelection(i, {0}, "h", 1, /*is_error=*/true));
  }
  KrausTrajectory traj(id, selections, prob, 0);
  traj.multiplicity = multiplicity;
  traj.weight = static_cast<double>(multiplicity);
  return traj;
}

CUDAQ_TEST(ShotAllocationTest, ProportionalBasic) {
  // PROPORTIONAL now weights by multiplicity, not probability.
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.5, 0, 5),
                                               makeTrajectory(1, 0.3, 0, 3),
                                               makeTrajectory(2, 0.2, 0, 2)};

  // Multinomial sampling with weights 5:3:2.
  // Tolerance is ~4*sigma (sigma = sqrt(n*p*(1-p))).
  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 1000, strategy);

  EXPECT_NEAR(trajectories[0].num_shots, 500, 65);
  EXPECT_NEAR(trajectories[1].num_shots, 300, 60);
  EXPECT_NEAR(trajectories[2].num_shots, 200, 55);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalWithEqualMultiplicity) {
  // Equal multiplicities -> equal allocation (like uniform).
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.333, 0, 1),
                                               makeTrajectory(1, 0.333, 0, 1),
                                               makeTrajectory(2, 0.334, 0, 1)};

  // Each trajectory gets ~333 shots; sigma ~14.9, tolerance is ~4*sigma.
  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 1000, strategy);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);

  for (const auto &traj : trajectories) {
    EXPECT_NEAR(traj.num_shots, 333, 60);
  }
}

CUDAQ_TEST(ShotAllocationTest, ProportionalSingleTrajectory) {
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 1.0)};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 5000, strategy);

  EXPECT_EQ(trajectories[0].num_shots, 5000);
}

CUDAQ_TEST(ShotAllocationTest, UniformBasic) {
  std::vector<KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.8), makeTrajectory(1, 0.1), makeTrajectory(2, 0.1)};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::UNIFORM);
  allocateShots(trajectories, 3000, strategy);

  // All get equal shots (ignoring probability)
  EXPECT_EQ(trajectories[0].num_shots, 1000);
  EXPECT_EQ(trajectories[1].num_shots, 1000);
  EXPECT_EQ(trajectories[2].num_shots, 1000);
}

CUDAQ_TEST(ShotAllocationTest, UniformWithRemainder) {
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.5),
                                               makeTrajectory(1, 0.5)};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::UNIFORM);
  allocateShots(trajectories, 1001, strategy);

  // Total must equal 1001
  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots;
  EXPECT_EQ(total, 1001);

  // Each should get approximately half
  EXPECT_GE(trajectories[0].num_shots, 500);
  EXPECT_GE(trajectories[1].num_shots, 500);
}

CUDAQ_TEST(ShotAllocationTest, LowWeightBiasBasic) {
  std::vector<KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.4, 0, 4), // No errors, mult=4
      makeTrajectory(1, 0.3, 1, 3), // 1 error, mult=3
      makeTrajectory(2, 0.3, 2, 3)  // 2 errors, mult=3
  };

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
                                  2.0);
  allocateShots(trajectories, 1000, strategy);

  // Trajectory 0 (no errors) should get most shots
  EXPECT_GT(trajectories[0].num_shots, trajectories[1].num_shots);
  EXPECT_GT(trajectories[1].num_shots, trajectories[2].num_shots);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, LowWeightBiasStrength) {
  std::vector<KrausTrajectory> trajectories1 = {makeTrajectory(0, 0.5, 0, 5),
                                                makeTrajectory(1, 0.5, 3, 5)};
  std::vector<KrausTrajectory> trajectories2 = trajectories1;

  // Weak bias (strength = 1.0)
  ShotAllocationStrategy weak_bias(
      ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 1.0);
  allocateShots(trajectories1, 1000, weak_bias);

  // Strong bias (strength = 3.0)
  ShotAllocationStrategy strong_bias(
      ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 3.0);
  allocateShots(trajectories2, 1000, strong_bias);

  // Strong bias should give even more shots to low-weight trajectory
  EXPECT_GT(trajectories2[0].num_shots, trajectories1[0].num_shots);
  EXPECT_LT(trajectories2[1].num_shots, trajectories1[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, HighWeightBiasBasic) {
  std::vector<KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.4, 0, 4), // No errors, mult=4
      makeTrajectory(1, 0.3, 1, 3), // 1 error, mult=3
      makeTrajectory(2, 0.3, 2, 3)  // 2 errors, mult=3
  };

  ShotAllocationStrategy strategy(
      ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0);
  allocateShots(trajectories, 1000, strategy);

  // Trajectory 2 (most errors) should get most shots
  EXPECT_LT(trajectories[0].num_shots, trajectories[1].num_shots);
  EXPECT_LT(trajectories[1].num_shots, trajectories[2].num_shots);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, EmptyTrajectoryList) {
  std::vector<KrausTrajectory> trajectories;
  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);

  EXPECT_ANY_THROW({ allocateShots(trajectories, 1000, strategy); });
}

CUDAQ_TEST(ShotAllocationTest, ProportionalThrowsOnZeroWeights) {
  std::vector<KrausTrajectory> trajectories;
  // Construct trajectories with default weight=probability.
  // Probability 0 means weight 0 from constructor default.
  trajectories.push_back(KrausTrajectory(0, {}, 0.0, 0));
  trajectories.push_back(KrausTrajectory(1, {}, 0.0, 0));

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  EXPECT_ANY_THROW({ allocateShots(trajectories, 100, strategy); });
}

CUDAQ_TEST(ShotAllocationTest, SingleShotDistribution) {
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.5),
                                               makeTrajectory(1, 0.5)};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 1, strategy);

  // One trajectory gets 1 shot, other gets 0
  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots;
  EXPECT_EQ(total, 1);
}

CUDAQ_TEST(ShotAllocationTest, LargeNumberOfTrajectories) {
  std::vector<KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 1000; ++i) {
    trajectories.push_back(makeTrajectory(i, 0.001));
  }

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 100000, strategy);

  std::size_t total = 0;
  for (const auto &traj : trajectories) {
    total += traj.num_shots;
  }
  EXPECT_EQ(total, 100000);

  // Check that no trajectory gets zero or an unreasonable amount
  for (const auto &traj : trajectories) {
    EXPECT_GE(traj.num_shots, 0);
    EXPECT_LE(traj.num_shots, 200);
  }
}

CUDAQ_TEST(ShotAllocationTest, CompareProportionalVsUniform) {
  // Multiplicities 9:1 drive proportional allocation.
  std::vector<KrausTrajectory> traj_prop = {makeTrajectory(0, 0.5, 0, 9),
                                            makeTrajectory(1, 0.5, 0, 1)};
  std::vector<KrausTrajectory> traj_unif = {makeTrajectory(0, 0.5, 0, 9),
                                            makeTrajectory(1, 0.5, 0, 1)};

  ShotAllocationStrategy proportional(
      ShotAllocationStrategy::Type::PROPORTIONAL);
  ShotAllocationStrategy uniform(ShotAllocationStrategy::Type::UNIFORM);

  allocateShots(traj_prop, 1000, proportional);
  allocateShots(traj_unif, 1000, uniform);

  // Proportional: high-multiplicity trajectory gets significantly more shots.
  EXPECT_GT(traj_prop[0].num_shots, traj_prop[1].num_shots);
  EXPECT_NEAR(traj_prop[0].num_shots, 900, 40);
  EXPECT_NEAR(traj_prop[1].num_shots, 100, 40);

  // Uniform: 500 / 500 (ignores multiplicity)
  EXPECT_EQ(traj_unif[0].num_shots, 500);
  EXPECT_EQ(traj_unif[1].num_shots, 500);
}

CUDAQ_TEST(ShotAllocationTest, CompareLowVsHighWeightBias) {
  std::vector<KrausTrajectory> low_bias = {
      makeTrajectory(0, 0.5, 0, 5), // No errors, mult=5
      makeTrajectory(1, 0.5, 3, 5)  // 3 errors, mult=5
  };
  std::vector<KrausTrajectory> high_bias = low_bias;

  ShotAllocationStrategy low_strategy(
      ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0);
  ShotAllocationStrategy high_strategy(
      ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0);

  allocateShots(low_bias, 1000, low_strategy);
  allocateShots(high_bias, 1000, high_strategy);

  // Low-weight bias: trajectory 0 gets more
  EXPECT_GT(low_bias[0].num_shots, low_bias[1].num_shots);

  // High-weight bias: trajectory 1 gets more
  EXPECT_LT(high_bias[0].num_shots, high_bias[1].num_shots);

  // Opposite directions
  EXPECT_GT(low_bias[0].num_shots, high_bias[0].num_shots);
  EXPECT_LT(low_bias[1].num_shots, high_bias[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, CountErrorsHelper) {
  // No errors
  KrausTrajectory traj0 = makeTrajectory(0, 0.5, 0);
  EXPECT_EQ(traj0.countErrors(), 0);

  // 1 error
  KrausTrajectory traj1 = makeTrajectory(1, 0.3, 1);
  EXPECT_EQ(traj1.countErrors(), 1);

  // 5 errors
  KrausTrajectory traj5 = makeTrajectory(2, 0.1, 5);
  EXPECT_EQ(traj5.countErrors(), 5);

  // Some identity, some errors
  std::vector<KrausSelection> mixed = {
      KrausSelection(0, {0}, "h", 0),       // No error
      KrausSelection(1, {0}, "x", 1, true), // Error
      KrausSelection(2, {0, 1}, "cx", 0),   // No error
      KrausSelection(3, {1}, "h", 2, true)  // Error
  };
  KrausTrajectory traj_mixed(3, mixed, 0.2, 0);
  EXPECT_EQ(traj_mixed.countErrors(), 2);
}

CUDAQ_TEST(ShotAllocationTest, VerySmallProbabilities) {
  std::vector<KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 10; ++i) {
    trajectories.push_back(makeTrajectory(i, 1e-6));
  }

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 1000, strategy);

  std::size_t total = 0;
  for (const auto &traj : trajectories) {
    total += traj.num_shots;
  }
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, ExtremelyUnequalMultiplicities) {
  std::vector<KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.999, 0, 999), // Dominant trajectory
      makeTrajectory(1, 0.001, 0, 1)    // Rare trajectory
  };

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 10000, strategy);

  EXPECT_GE(trajectories[0].num_shots, 9980);
  EXPECT_LE(trajectories[1].num_shots, 20);

  EXPECT_EQ(trajectories[0].num_shots + trajectories[1].num_shots, 10000);
}

CUDAQ_TEST(ShotAllocationTest, ManyTrajectoriesFewShots) {
  std::vector<KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 100; ++i) {
    trajectories.push_back(makeTrajectory(i, 0.01));
  }

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::UNIFORM);
  allocateShots(trajectories, 50, strategy);

  std::size_t total = 0;
  for (const auto &traj : trajectories) {
    total += traj.num_shots;
  }
  EXPECT_EQ(total, 50);

  std::size_t zero_shot_count = 0;
  for (const auto &traj : trajectories) {
    if (traj.num_shots == 0) {
      zero_shot_count++;
    }
  }
  EXPECT_GT(zero_shot_count, 0);
}

CUDAQ_TEST(ShotAllocationTest, TotalShotsInvariant) {
  std::vector<KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.4, 0, 4), makeTrajectory(1, 0.3, 1, 3),
      makeTrajectory(2, 0.2, 2, 2), makeTrajectory(3, 0.1, 3, 1)};

  std::vector<ShotAllocationStrategy::Type> strategies = {
      ShotAllocationStrategy::Type::PROPORTIONAL,
      ShotAllocationStrategy::Type::UNIFORM,
      ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
      ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS};

  for (auto strat_type : strategies) {
    auto traj_copy = trajectories;
    ShotAllocationStrategy strategy(strat_type, 2.0);
    allocateShots(traj_copy, 1000, strategy);

    std::size_t total = 0;
    for (const auto &traj : traj_copy) {
      total += traj.num_shots;
    }

    EXPECT_EQ(total, 1000) << "Strategy type failed to preserve total shots";
  }
}

CUDAQ_TEST(ShotAllocationTest, NonNegativeShots) {
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.5),
                                               makeTrajectory(1, 0.5)};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 1000, strategy);

  for (const auto &traj : trajectories) {
    // Always non-negative
    EXPECT_GE(traj.num_shots, 0);
  }
}

CUDAQ_TEST(ShotAllocationTest, SpanWithArray) {
  std::array<KrausTrajectory, 3> trajectories = {
      {makeTrajectory(0, 0.5, 0, 5), makeTrajectory(1, 0.3, 0, 3),
       makeTrajectory(2, 0.2, 0, 2)}};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 1000, strategy);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
  EXPECT_NEAR(trajectories[0].num_shots, 500, 65);
  EXPECT_NEAR(trajectories[1].num_shots, 300, 60);
  EXPECT_NEAR(trajectories[2].num_shots, 200, 55);
}

CUDAQ_TEST(ShotAllocationTest, SpanWithSubrange) {
  std::vector<KrausTrajectory> all_trajectories = {
      makeTrajectory(0, 0.25), makeTrajectory(1, 0.25), makeTrajectory(2, 0.25),
      makeTrajectory(3, 0.25)};

  std::span<KrausTrajectory> first_two(all_trajectories.data(), 2);
  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::UNIFORM);
  allocateShots(first_two, 600, strategy);

  EXPECT_EQ(all_trajectories[0].num_shots, 300);
  EXPECT_EQ(all_trajectories[1].num_shots, 300);
  EXPECT_EQ(all_trajectories[2].num_shots, 0);
  EXPECT_EQ(all_trajectories[3].num_shots, 0);
}

CUDAQ_TEST(ShotAllocationTest, RangesCountErrorsMultiple) {
  std::vector<KrausSelection> with_errors = {
      KrausSelection(0, {0}, "h", 1, true), KrausSelection(1, {0}, "x", 0),
      KrausSelection(2, {0}, "y", 2, true)};
  KrausTrajectory traj(1, with_errors, 0.5, 100);

  EXPECT_EQ(traj.countErrors(), 2);
}

CUDAQ_TEST(ShotAllocationTest, RangesCountErrorsEmpty) {
  KrausTrajectory empty_traj(0, {}, 1.0, 100);
  EXPECT_EQ(empty_traj.countErrors(), 0);
}

CUDAQ_TEST(ShotAllocationTest, RangesCountErrorsAllIdentity) {
  std::vector<KrausSelection> no_errors = {KrausSelection(0, {0}, "h", 0),
                                           KrausSelection(1, {0}, "x", 0),
                                           KrausSelection(2, {0}, "y", 0)};
  KrausTrajectory traj(0, no_errors, 1.0, 100);

  EXPECT_EQ(traj.countErrors(), 0);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalReproducibility) {
  std::vector<KrausTrajectory> t1 = {makeTrajectory(0, 0.5, 0, 5),
                                     makeTrajectory(1, 0.3, 0, 3),
                                     makeTrajectory(2, 0.2, 0, 2)};
  std::vector<KrausTrajectory> t2 = t1;

  ShotAllocationStrategy s1(ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
                            /*seed=*/42);
  ShotAllocationStrategy s2(ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
                            /*seed=*/42);
  allocateShots(t1, 1000, s1);
  allocateShots(t2, 1000, s2);

  EXPECT_EQ(t1[0].num_shots, t2[0].num_shots);
  EXPECT_EQ(t1[1].num_shots, t2[1].num_shots);
  EXPECT_EQ(t1[2].num_shots, t2[2].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, LowWeightBiasReproducibility) {
  std::vector<KrausTrajectory> t1 = {makeTrajectory(0, 0.5, 0, 5),
                                     makeTrajectory(1, 0.5, 2, 5)};
  std::vector<KrausTrajectory> t2 = t1;

  ShotAllocationStrategy s1(ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0,
                            /*seed=*/99);
  ShotAllocationStrategy s2(ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0,
                            /*seed=*/99);
  allocateShots(t1, 500, s1);
  allocateShots(t2, 500, s2);

  EXPECT_EQ(t1[0].num_shots, t2[0].num_shots);
  EXPECT_EQ(t1[1].num_shots, t2[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, HighWeightBiasReproducibility) {
  std::vector<KrausTrajectory> t1 = {makeTrajectory(0, 0.5, 0, 5),
                                     makeTrajectory(1, 0.5, 2, 5)};
  std::vector<KrausTrajectory> t2 = t1;

  ShotAllocationStrategy s1(ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0,
                            /*seed=*/77);
  ShotAllocationStrategy s2(ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0,
                            /*seed=*/77);
  allocateShots(t1, 500, s1);
  allocateShots(t2, 500, s2);

  EXPECT_EQ(t1[0].num_shots, t2[0].num_shots);
  EXPECT_EQ(t1[1].num_shots, t2[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalNoTruncationZeroShots) {
  std::vector<KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 100; ++i)
    trajectories.push_back(makeTrajectory(i, 0.01));

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL,
                                  2.0, /*seed=*/1234);
  allocateShots(trajectories, 50, strategy);

  std::size_t total = 0;
  for (const auto &traj : trajectories)
    total += traj.num_shots;
  EXPECT_EQ(total, 50);

  std::size_t nonzero = 0;
  for (const auto &traj : trajectories)
    if (traj.num_shots > 0)
      nonzero++;
  EXPECT_GT(nonzero, 0);
  EXPECT_LE(nonzero, 50);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalExactTotal) {
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.7, 0, 7),
                                               makeTrajectory(1, 0.2, 0, 2),
                                               makeTrajectory(2, 0.1, 0, 1)};

  for (std::size_t shots : {1, 3, 7, 100, 999, 10000}) {
    auto traj_copy = trajectories;
    ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
    allocateShots(traj_copy, shots, strategy);

    std::size_t total = 0;
    for (const auto &t : traj_copy)
      total += t.num_shots;
    EXPECT_EQ(total, shots) << "Total mismatch for shots=" << shots;
  }
}

// Calling allocateShots twice on the same trajectories should not accumulate
CUDAQ_TEST(ShotAllocationTest, DoubleAllocationDoesNotAccumulate) {
  std::vector<KrausTrajectory> trajectories = {makeTrajectory(0, 0.7, 0, 7),
                                               makeTrajectory(1, 0.3, 0, 3)};

  ShotAllocationStrategy strategy(ShotAllocationStrategy::Type::PROPORTIONAL);
  allocateShots(trajectories, 100, strategy);
  allocateShots(trajectories, 100, strategy);

  std::size_t total = 0;
  for (const auto &t : trajectories)
    total += t.num_shots;
  EXPECT_EQ(total, 100u);
}
