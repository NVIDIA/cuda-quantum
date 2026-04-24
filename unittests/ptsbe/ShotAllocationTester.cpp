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

// A simple trajectory for testing
static cudaq::KrausTrajectory makeTrajectory(std::size_t id, double prob,
                                             std::size_t errors = 0,
                                             std::size_t multiplicity = 1) {
  std::vector<cudaq::KrausSelection> selections;
  for (std::size_t i = 0; i < errors; ++i) {
    selections.push_back(
        cudaq::KrausSelection(i, {0}, "h", 1, /*is_error=*/true));
  }
  cudaq::KrausTrajectory traj(id, selections, prob, 0);
  traj.multiplicity = multiplicity;
  traj.weight = static_cast<double>(multiplicity);
  return traj;
}

CUDAQ_TEST(ShotAllocationTest, ProportionalBasic) {
  // PROPORTIONAL now weights by multiplicity, not probability.
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.5, 0, 5), makeTrajectory(1, 0.3, 0, 3),
      makeTrajectory(2, 0.2, 0, 2)};

  // Multinomial sampling with weights 5:3:2.
  // Tolerance is ~4*sigma (sigma = sqrt(n*p*(1-p))).
  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  EXPECT_NEAR(trajectories[0].num_shots, 500, 65);
  EXPECT_NEAR(trajectories[1].num_shots, 300, 60);
  EXPECT_NEAR(trajectories[2].num_shots, 200, 55);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalWithEqualMultiplicity) {
  // Equal multiplicities -> equal allocation (like uniform).
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.333, 0, 1), makeTrajectory(1, 0.333, 0, 1),
      makeTrajectory(2, 0.334, 0, 1)};

  // Each trajectory gets ~333 shots; sigma ~14.9, tolerance is ~4*sigma.
  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);

  for (const auto &traj : trajectories) {
    EXPECT_NEAR(traj.num_shots, 333, 60);
  }
}

CUDAQ_TEST(ShotAllocationTest, ProportionalSingleTrajectory) {
  std::vector<cudaq::KrausTrajectory> trajectories = {makeTrajectory(0, 1.0)};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 5000, strategy);

  EXPECT_EQ(trajectories[0].num_shots, 5000);
}

CUDAQ_TEST(ShotAllocationTest, UniformBasic) {
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.8), makeTrajectory(1, 0.1), makeTrajectory(2, 0.1)};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::UNIFORM);
  cudaq::ptsbe::allocateShots(trajectories, 3000, strategy);

  // All get equal shots (ignoring probability)
  EXPECT_EQ(trajectories[0].num_shots, 1000);
  EXPECT_EQ(trajectories[1].num_shots, 1000);
  EXPECT_EQ(trajectories[2].num_shots, 1000);
}

CUDAQ_TEST(ShotAllocationTest, UniformWithRemainder) {
  std::vector<cudaq::KrausTrajectory> trajectories = {makeTrajectory(0, 0.5),
                                                      makeTrajectory(1, 0.5)};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::UNIFORM);
  cudaq::ptsbe::allocateShots(trajectories, 1001, strategy);

  // Total must equal 1001
  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots;
  EXPECT_EQ(total, 1001);

  // Each should get approximately half
  EXPECT_GE(trajectories[0].num_shots, 500);
  EXPECT_GE(trajectories[1].num_shots, 500);
}

CUDAQ_TEST(ShotAllocationTest, LowWeightBiasBasic) {
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.4, 0, 4), // No errors, mult=4
      makeTrajectory(1, 0.3, 1, 3), // 1 error, mult=3
      makeTrajectory(2, 0.3, 2, 3)  // 2 errors, mult=3
  };

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  // Trajectory 0 (no errors) should get most shots
  EXPECT_GT(trajectories[0].num_shots, trajectories[1].num_shots);
  EXPECT_GT(trajectories[1].num_shots, trajectories[2].num_shots);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, LowWeightBiasStrength) {
  std::vector<cudaq::KrausTrajectory> trajectories1 = {
      makeTrajectory(0, 0.5, 0, 5), makeTrajectory(1, 0.5, 3, 5)};
  std::vector<cudaq::KrausTrajectory> trajectories2 = trajectories1;

  // Weak bias (strength = 1.0)
  cudaq::ptsbe::ShotAllocationStrategy weak_bias(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 1.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories1, 1000, weak_bias);

  // Strong bias (strength = 3.0)
  cudaq::ptsbe::ShotAllocationStrategy strong_bias(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 3.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories2, 1000, strong_bias);

  // Strong bias should give even more shots to low-weight trajectory
  EXPECT_GT(trajectories2[0].num_shots, trajectories1[0].num_shots);
  EXPECT_LT(trajectories2[1].num_shots, trajectories1[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, HighWeightBiasBasic) {
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.4, 0, 4), // No errors, mult=4
      makeTrajectory(1, 0.3, 1, 3), // 1 error, mult=3
      makeTrajectory(2, 0.3, 2, 3)  // 2 errors, mult=3
  };

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  // Trajectory 2 (most errors) should get most shots
  EXPECT_LT(trajectories[0].num_shots, trajectories[1].num_shots);
  EXPECT_LT(trajectories[1].num_shots, trajectories[2].num_shots);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, EmptyTrajectoryList) {
  std::vector<cudaq::KrausTrajectory> trajectories;
  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);

  EXPECT_ANY_THROW(
      { cudaq::ptsbe::allocateShots(trajectories, 1000, strategy); });
}

CUDAQ_TEST(ShotAllocationTest, ProportionalThrowsOnZeroWeights) {
  std::vector<cudaq::KrausTrajectory> trajectories;
  // Construct trajectories with default weight=probability.
  // Probability 0 means weight 0 from constructor default.
  trajectories.push_back(cudaq::KrausTrajectory(0, {}, 0.0, 0));
  trajectories.push_back(cudaq::KrausTrajectory(1, {}, 0.0, 0));

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  EXPECT_ANY_THROW(
      { cudaq::ptsbe::allocateShots(trajectories, 100, strategy); });
}

CUDAQ_TEST(ShotAllocationTest, SingleShotDistribution) {
  std::vector<cudaq::KrausTrajectory> trajectories = {makeTrajectory(0, 0.5),
                                                      makeTrajectory(1, 0.5)};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1, strategy);

  // One trajectory gets 1 shot, other gets 0
  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots;
  EXPECT_EQ(total, 1);
}

CUDAQ_TEST(ShotAllocationTest, LargeNumberOfTrajectories) {
  std::vector<cudaq::KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 1000; ++i) {
    trajectories.push_back(makeTrajectory(i, 0.001));
  }

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 100000, strategy);

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
  std::vector<cudaq::KrausTrajectory> traj_prop = {
      makeTrajectory(0, 0.5, 0, 9), makeTrajectory(1, 0.5, 0, 1)};
  std::vector<cudaq::KrausTrajectory> traj_unif = {
      makeTrajectory(0, 0.5, 0, 9), makeTrajectory(1, 0.5, 0, 1)};

  cudaq::ptsbe::ShotAllocationStrategy proportional(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::ShotAllocationStrategy uniform(
      cudaq::ptsbe::ShotAllocationStrategy::Type::UNIFORM);

  cudaq::ptsbe::allocateShots(traj_prop, 1000, proportional);
  cudaq::ptsbe::allocateShots(traj_unif, 1000, uniform);

  // Proportional: high-multiplicity trajectory gets significantly more shots.
  EXPECT_GT(traj_prop[0].num_shots, traj_prop[1].num_shots);
  EXPECT_NEAR(traj_prop[0].num_shots, 900, 40);
  EXPECT_NEAR(traj_prop[1].num_shots, 100, 40);

  // Uniform: 500 / 500 (ignores multiplicity)
  EXPECT_EQ(traj_unif[0].num_shots, 500);
  EXPECT_EQ(traj_unif[1].num_shots, 500);
}

CUDAQ_TEST(ShotAllocationTest, CompareLowVsHighWeightBias) {
  std::vector<cudaq::KrausTrajectory> low_bias = {
      makeTrajectory(0, 0.5, 0, 5), // No errors, mult=5
      makeTrajectory(1, 0.5, 3, 5)  // 3 errors, mult=5
  };
  std::vector<cudaq::KrausTrajectory> high_bias = low_bias;

  cudaq::ptsbe::ShotAllocationStrategy low_strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::ShotAllocationStrategy high_strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0,
      /*seed=*/42);

  cudaq::ptsbe::allocateShots(low_bias, 1000, low_strategy);
  cudaq::ptsbe::allocateShots(high_bias, 1000, high_strategy);

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
  cudaq::KrausTrajectory traj0 = makeTrajectory(0, 0.5, 0);
  EXPECT_EQ(traj0.countErrors(), 0);

  // 1 error
  cudaq::KrausTrajectory traj1 = makeTrajectory(1, 0.3, 1);
  EXPECT_EQ(traj1.countErrors(), 1);

  // 5 errors
  cudaq::KrausTrajectory traj5 = makeTrajectory(2, 0.1, 5);
  EXPECT_EQ(traj5.countErrors(), 5);

  // Some identity, some errors
  std::vector<cudaq::KrausSelection> mixed = {
      cudaq::KrausSelection(0, {0}, "h", 0),       // No error
      cudaq::KrausSelection(1, {0}, "x", 1, true), // Error
      cudaq::KrausSelection(2, {0, 1}, "cx", 0),   // No error
      cudaq::KrausSelection(3, {1}, "h", 2, true)  // Error
  };
  cudaq::KrausTrajectory traj_mixed(3, mixed, 0.2, 0);
  EXPECT_EQ(traj_mixed.countErrors(), 2);
}

CUDAQ_TEST(ShotAllocationTest, VerySmallProbabilities) {
  std::vector<cudaq::KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 10; ++i) {
    trajectories.push_back(makeTrajectory(i, 1e-6));
  }

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  std::size_t total = 0;
  for (const auto &traj : trajectories) {
    total += traj.num_shots;
  }
  EXPECT_EQ(total, 1000);
}

CUDAQ_TEST(ShotAllocationTest, ExtremelyUnequalMultiplicities) {
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.999, 0, 999), // Dominant trajectory
      makeTrajectory(1, 0.001, 0, 1)    // Rare trajectory
  };

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 10000, strategy);

  EXPECT_GE(trajectories[0].num_shots, 9980);
  EXPECT_LE(trajectories[1].num_shots, 20);

  EXPECT_EQ(trajectories[0].num_shots + trajectories[1].num_shots, 10000);
}

CUDAQ_TEST(ShotAllocationTest, ManyTrajectoriesFewShots) {
  std::vector<cudaq::KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 100; ++i) {
    trajectories.push_back(makeTrajectory(i, 0.01));
  }

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::UNIFORM);
  cudaq::ptsbe::allocateShots(trajectories, 50, strategy);

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
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.4, 0, 4), makeTrajectory(1, 0.3, 1, 3),
      makeTrajectory(2, 0.2, 2, 2), makeTrajectory(3, 0.1, 3, 1)};

  std::vector<cudaq::ptsbe::ShotAllocationStrategy::Type> strategies = {
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL,
      cudaq::ptsbe::ShotAllocationStrategy::Type::UNIFORM,
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS,
      cudaq::ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS};

  for (auto strat_type : strategies) {
    auto traj_copy = trajectories;
    cudaq::ptsbe::ShotAllocationStrategy strategy(strat_type, 2.0, /*seed=*/42);
    cudaq::ptsbe::allocateShots(traj_copy, 1000, strategy);

    std::size_t total = 0;
    for (const auto &traj : traj_copy) {
      total += traj.num_shots;
    }

    EXPECT_EQ(total, 1000) << "Strategy type failed to preserve total shots";
  }
}

CUDAQ_TEST(ShotAllocationTest, NonNegativeShots) {
  std::vector<cudaq::KrausTrajectory> trajectories = {makeTrajectory(0, 0.5),
                                                      makeTrajectory(1, 0.5)};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  for (const auto &traj : trajectories) {
    // Always non-negative
    EXPECT_GE(traj.num_shots, 0);
  }
}

CUDAQ_TEST(ShotAllocationTest, SpanWithArray) {
  std::array<cudaq::KrausTrajectory, 3> trajectories = {
      {makeTrajectory(0, 0.5, 0, 5), makeTrajectory(1, 0.3, 0, 3),
       makeTrajectory(2, 0.2, 0, 2)}};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 1000, strategy);

  std::size_t total = trajectories[0].num_shots + trajectories[1].num_shots +
                      trajectories[2].num_shots;
  EXPECT_EQ(total, 1000);
  EXPECT_NEAR(trajectories[0].num_shots, 500, 65);
  EXPECT_NEAR(trajectories[1].num_shots, 300, 60);
  EXPECT_NEAR(trajectories[2].num_shots, 200, 55);
}

CUDAQ_TEST(ShotAllocationTest, SpanWithSubrange) {
  std::vector<cudaq::KrausTrajectory> all_trajectories = {
      makeTrajectory(0, 0.25), makeTrajectory(1, 0.25), makeTrajectory(2, 0.25),
      makeTrajectory(3, 0.25)};

  std::span<cudaq::KrausTrajectory> first_two(all_trajectories.data(), 2);
  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::UNIFORM);
  cudaq::ptsbe::allocateShots(first_two, 600, strategy);

  EXPECT_EQ(all_trajectories[0].num_shots, 300);
  EXPECT_EQ(all_trajectories[1].num_shots, 300);
  EXPECT_EQ(all_trajectories[2].num_shots, 0);
  EXPECT_EQ(all_trajectories[3].num_shots, 0);
}

CUDAQ_TEST(ShotAllocationTest, RangesCountErrorsMultiple) {
  std::vector<cudaq::KrausSelection> with_errors = {
      cudaq::KrausSelection(0, {0}, "h", 1, true),
      cudaq::KrausSelection(1, {0}, "x", 0),
      cudaq::KrausSelection(2, {0}, "y", 2, true)};
  cudaq::KrausTrajectory traj(1, with_errors, 0.5, 100);

  EXPECT_EQ(traj.countErrors(), 2);
}

CUDAQ_TEST(ShotAllocationTest, RangesCountErrorsEmpty) {
  cudaq::KrausTrajectory empty_traj(0, {}, 1.0, 100);
  EXPECT_EQ(empty_traj.countErrors(), 0);
}

CUDAQ_TEST(ShotAllocationTest, RangesCountErrorsAllIdentity) {
  std::vector<cudaq::KrausSelection> no_errors = {
      cudaq::KrausSelection(0, {0}, "h", 0),
      cudaq::KrausSelection(1, {0}, "x", 0),
      cudaq::KrausSelection(2, {0}, "y", 0)};
  cudaq::KrausTrajectory traj(0, no_errors, 1.0, 100);

  EXPECT_EQ(traj.countErrors(), 0);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalReproducibility) {
  std::vector<cudaq::KrausTrajectory> t1 = {makeTrajectory(0, 0.5, 0, 5),
                                            makeTrajectory(1, 0.3, 0, 3),
                                            makeTrajectory(2, 0.2, 0, 2)};
  std::vector<cudaq::KrausTrajectory> t2 = t1;

  cudaq::ptsbe::ShotAllocationStrategy s1(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::ShotAllocationStrategy s2(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(t1, 1000, s1);
  cudaq::ptsbe::allocateShots(t2, 1000, s2);

  EXPECT_EQ(t1[0].num_shots, t2[0].num_shots);
  EXPECT_EQ(t1[1].num_shots, t2[1].num_shots);
  EXPECT_EQ(t1[2].num_shots, t2[2].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, LowWeightBiasReproducibility) {
  std::vector<cudaq::KrausTrajectory> t1 = {makeTrajectory(0, 0.5, 0, 5),
                                            makeTrajectory(1, 0.5, 2, 5)};
  std::vector<cudaq::KrausTrajectory> t2 = t1;

  cudaq::ptsbe::ShotAllocationStrategy s1(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0,
      /*seed=*/99);
  cudaq::ptsbe::ShotAllocationStrategy s2(
      cudaq::ptsbe::ShotAllocationStrategy::Type::LOW_WEIGHT_BIAS, 2.0,
      /*seed=*/99);
  cudaq::ptsbe::allocateShots(t1, 500, s1);
  cudaq::ptsbe::allocateShots(t2, 500, s2);

  EXPECT_EQ(t1[0].num_shots, t2[0].num_shots);
  EXPECT_EQ(t1[1].num_shots, t2[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, HighWeightBiasReproducibility) {
  std::vector<cudaq::KrausTrajectory> t1 = {makeTrajectory(0, 0.5, 0, 5),
                                            makeTrajectory(1, 0.5, 2, 5)};
  std::vector<cudaq::KrausTrajectory> t2 = t1;

  cudaq::ptsbe::ShotAllocationStrategy s1(
      cudaq::ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0,
      /*seed=*/77);
  cudaq::ptsbe::ShotAllocationStrategy s2(
      cudaq::ptsbe::ShotAllocationStrategy::Type::HIGH_WEIGHT_BIAS, 2.0,
      /*seed=*/77);
  cudaq::ptsbe::allocateShots(t1, 500, s1);
  cudaq::ptsbe::allocateShots(t2, 500, s2);

  EXPECT_EQ(t1[0].num_shots, t2[0].num_shots);
  EXPECT_EQ(t1[1].num_shots, t2[1].num_shots);
}

CUDAQ_TEST(ShotAllocationTest, ProportionalNoTruncationZeroShots) {
  std::vector<cudaq::KrausTrajectory> trajectories;
  for (std::size_t i = 0; i < 100; ++i)
    trajectories.push_back(makeTrajectory(i, 0.01));

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/1234);
  cudaq::ptsbe::allocateShots(trajectories, 50, strategy);

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
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.7, 0, 7), makeTrajectory(1, 0.2, 0, 2),
      makeTrajectory(2, 0.1, 0, 1)};

  for (std::size_t shots : {1, 3, 7, 100, 999, 10000}) {
    auto traj_copy = trajectories;
    cudaq::ptsbe::ShotAllocationStrategy strategy(
        cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
        /*seed=*/42);
    cudaq::ptsbe::allocateShots(traj_copy, shots, strategy);

    std::size_t total = 0;
    for (const auto &t : traj_copy)
      total += t.num_shots;
    EXPECT_EQ(total, shots) << "Total mismatch for shots=" << shots;
  }
}

// Calling cudaq::ptsbe::allocateShots twice on the same trajectories should not
// accumulate
CUDAQ_TEST(ShotAllocationTest, DoubleAllocationDoesNotAccumulate) {
  std::vector<cudaq::KrausTrajectory> trajectories = {
      makeTrajectory(0, 0.7, 0, 7), makeTrajectory(1, 0.3, 0, 3)};

  cudaq::ptsbe::ShotAllocationStrategy strategy(
      cudaq::ptsbe::ShotAllocationStrategy::Type::PROPORTIONAL, 2.0,
      /*seed=*/42);
  cudaq::ptsbe::allocateShots(trajectories, 100, strategy);
  cudaq::ptsbe::allocateShots(trajectories, 100, strategy);

  std::size_t total = 0;
  for (const auto &t : trajectories)
    total += t.num_shots;
  EXPECT_EQ(total, 100u);
}
