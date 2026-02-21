/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/NoiseModel.h"
#include "cudaq/ptsbe/PTSSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/ConditionalSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/ExhaustiveSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/OrderedSamplingStrategy.h"
#include "cudaq/ptsbe/strategies/ProbabilisticSamplingStrategy.h"
#include <cmath>
#include <gtest/gtest.h>
#include <set>

using namespace cudaq::ptsbe;

static cudaq::kraus_channel makeIYChannel(double pI, double pY) {
  const double sI = std::sqrt(pI);
  const double sY = std::sqrt(pY);
  const cudaq::complex i(0, 1);
  std::vector<cudaq::kraus_op> ops;
  ops.push_back(cudaq::kraus_op({sI, 0.0, 0.0, sI}));
  ops.push_back(cudaq::kraus_op({0.0, -i * sY, i * sY, 0.0}));
  return cudaq::kraus_channel(std::move(ops));
}

static cudaq::kraus_channel makeIXYChannel(double pI, double pX, double pY) {
  const double sI = std::sqrt(pI);
  const double sX = std::sqrt(pX);
  const double sY = std::sqrt(pY);
  const cudaq::complex i(0, 1);
  std::vector<cudaq::kraus_op> ops;
  ops.push_back(cudaq::kraus_op({sI, 0.0, 0.0, sI}));
  ops.push_back(cudaq::kraus_op({0.0, sX, sX, 0.0}));
  ops.push_back(cudaq::kraus_op({0.0, -i * sY, i * sY, 0.0}));
  return cudaq::kraus_channel(std::move(ops));
}

std::vector<NoisePoint> createSimpleNoisePoints() {
  std::vector<NoisePoint> noise_points;

  NoisePoint np1;
  np1.circuit_location = 0;
  np1.qubits = {0};
  np1.op_name = "h";
  np1.channel = cudaq::bit_flip_channel(0.1);
  noise_points.push_back(np1);

  NoisePoint np2;
  np2.circuit_location = 1;
  np2.qubits = {0};
  np2.op_name = "x";
  np2.channel = makeIYChannel(0.8, 0.2);
  noise_points.push_back(np2);

  return noise_points;
}

std::vector<NoisePoint> createThreeOperatorNoisePoints() {
  std::vector<NoisePoint> noise_points;

  NoisePoint np;
  np.circuit_location = 0;
  np.qubits = {0};
  np.op_name = "h";
  np.channel = makeIXYChannel(0.7, 0.2, 0.1);
  noise_points.push_back(np);

  return noise_points;
}

TEST(ProbabilisticSamplingStrategyTest, BasicGeneration) {
  auto noise_points = createSimpleNoisePoints();
  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 10);

  EXPECT_LE(trajectories.size(), 10);
  EXPECT_GT(trajectories.size(), 0);

  for (const auto &traj : trajectories) {
    EXPECT_EQ(traj.kraus_selections.size(), 2);
    EXPECT_GE(traj.probability, 0.0);
    EXPECT_LE(traj.probability, 1.0);
    EXPECT_EQ(traj.num_shots, 0);
  }
}

TEST(ProbabilisticSamplingStrategyTest, Uniqueness) {
  auto noise_points = createSimpleNoisePoints();
  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 4);

  for (std::size_t i = 0; i < trajectories.size(); ++i) {
    for (std::size_t j = i + 1; j < trajectories.size(); ++j) {
      bool same = true;
      for (std::size_t k = 0; k < trajectories[i].kraus_selections.size();
           ++k) {
        if (trajectories[i].kraus_selections[k].kraus_operator_index !=
            trajectories[j].kraus_selections[k].kraus_operator_index) {
          same = false;
          break;
        }
      }
      EXPECT_FALSE(same) << "Found duplicate trajectories at indices " << i
                         << " and " << j;
    }
  }
}

TEST(ProbabilisticSamplingStrategyTest, Reproducibility) {
  auto noise_points = createSimpleNoisePoints();

  ProbabilisticSamplingStrategy strategy1(123);
  ProbabilisticSamplingStrategy strategy2(123);

  auto trajectories1 = strategy1.generateTrajectories(noise_points, 5);
  auto trajectories2 = strategy2.generateTrajectories(noise_points, 5);

  EXPECT_EQ(trajectories1.size(), trajectories2.size());

  for (std::size_t i = 0; i < trajectories1.size(); ++i) {
    EXPECT_EQ(trajectories1[i].kraus_selections.size(),
              trajectories2[i].kraus_selections.size());
    for (std::size_t j = 0; j < trajectories1[i].kraus_selections.size(); ++j) {
      EXPECT_EQ(trajectories1[i].kraus_selections[j].kraus_operator_index,
                trajectories2[i].kraus_selections[j].kraus_operator_index);
    }
  }
}

TEST(ProbabilisticSamplingStrategyTest, EmptyNoisePoints) {
  std::vector<NoisePoint> empty_noise_points;
  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(empty_noise_points, 10);

  EXPECT_EQ(trajectories.size(), 0);
}

TEST(ProbabilisticSamplingStrategyTest, ProbabilityCalculation) {
  auto noise_points = createSimpleNoisePoints();
  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 4);

  for (const auto &traj : trajectories) {
    double expected_prob = 1.0;
    for (std::size_t i = 0; i < traj.kraus_selections.size(); ++i) {
      auto idx = static_cast<std::size_t>(
          traj.kraus_selections[i].kraus_operator_index);
      expected_prob *= noise_points[i].channel.probabilities[idx];
    }
    EXPECT_NEAR(traj.probability, expected_prob, 1e-9);
  }
}

TEST(ProbabilisticSamplingStrategyTest, StrategyName) {
  ProbabilisticSamplingStrategy strategy;
  EXPECT_STREQ(strategy.name(), "Probabilistic");
}

TEST(ProbabilisticSamplingStrategyTest, Clone) {
  ProbabilisticSamplingStrategy strategy(42);
  auto cloned = strategy.clone();

  EXPECT_NE(cloned.get(), nullptr);
  EXPECT_STREQ(cloned->name(), "Probabilistic");

  // Test that cloned strategy works
  auto noise_points = createSimpleNoisePoints();
  auto trajectories = cloned->generateTrajectories(noise_points, 5);
  EXPECT_GT(trajectories.size(), 0);
}

TEST(ProbabilisticSamplingStrategyTest, RequestMoreThanPossible) {
  std::vector<NoisePoint> noise_points;

  for (int i = 0; i < 2; ++i) {
    NoisePoint np;
    np.circuit_location = i;
    np.qubits = {0};
    np.op_name = "h";
    np.channel = cudaq::bit_flip_channel(0.5);
    noise_points.push_back(np);
  }

  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 100);

  EXPECT_EQ(trajectories.size(), 4);

  std::set<std::string> patterns;
  for (const auto &traj : trajectories) {
    std::string pattern;
    for (const auto &sel : traj.kraus_selections) {
      pattern +=
          std::to_string(static_cast<std::size_t>(sel.kraus_operator_index));
    }
    patterns.insert(pattern);
  }
  EXPECT_EQ(patterns.size(), 4);
}

TEST(ProbabilisticSamplingStrategyTest, FewPossibleTrajectoriesDiscoversAll) {
  std::vector<NoisePoint> noise_points;
  NoisePoint np;
  np.circuit_location = 0;
  np.qubits = {0};
  np.op_name = "x";
  np.channel = cudaq::depolarization_channel(0.001);
  noise_points.push_back(np);

  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 1000000);

  EXPECT_EQ(trajectories.size(), 4);

  std::set<std::size_t> operator_indices;
  for (const auto &traj : trajectories) {
    ASSERT_EQ(traj.kraus_selections.size(), 1);
    operator_indices.insert(static_cast<std::size_t>(
        traj.kraus_selections[0].kraus_operator_index));
  }
  EXPECT_EQ(operator_indices.size(), 4);
}

TEST(ProbabilisticSamplingStrategyTest,
     AccumulatesMultiplicityForAllTrajectories) {
  std::vector<NoisePoint> noise_points;

  NoisePoint np;
  np.circuit_location = 0;
  np.qubits = {0};
  np.op_name = "h";
  np.channel = makeIXYChannel(0.34, 0.33, 0.33);
  noise_points.push_back(np);

  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 1000);

  EXPECT_EQ(trajectories.size(), 3);

  std::set<std::size_t> operator_indices;
  std::size_t total_multiplicity = 0;
  for (const auto &traj : trajectories) {
    EXPECT_EQ(traj.kraus_selections.size(), 1);
    operator_indices.insert(static_cast<std::size_t>(
        traj.kraus_selections[0].kraus_operator_index));
    total_multiplicity += traj.multiplicity;
  }
  EXPECT_EQ(operator_indices.size(), 3);
}

TEST(ProbabilisticSamplingStrategyTest, LargeTrajectorySpace) {
  std::vector<NoisePoint> noise_points;

  auto channel = cudaq::depolarization_channel(0.75);
  for (int i = 0; i < 10; ++i) {
    NoisePoint np;
    np.circuit_location = i;
    np.qubits = {0};
    np.op_name = "h";
    np.channel = channel;
    noise_points.push_back(np);
  }

  ProbabilisticSamplingStrategy strategy(42);

  auto trajectories = strategy.generateTrajectories(noise_points, 50);

  EXPECT_EQ(trajectories.size(), 50);

  std::set<std::vector<std::size_t>> patterns;
  for (const auto &traj : trajectories) {
    std::vector<std::size_t> pattern;
    for (const auto &sel : traj.kraus_selections) {
      pattern.push_back(static_cast<std::size_t>(sel.kraus_operator_index));
    }
    patterns.insert(pattern);
  }
  EXPECT_EQ(patterns.size(), 50);
}

TEST(ExhaustiveSamplingStrategyTest, GeneratesAllTrajectories) {
  auto noise_points = createSimpleNoisePoints();
  ExhaustiveSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 100);

  EXPECT_EQ(trajectories.size(), 4);
}

TEST(ExhaustiveSamplingStrategyTest, LexicographicOrder) {
  auto noise_points = createSimpleNoisePoints();
  ExhaustiveSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 100);

  ASSERT_EQ(trajectories.size(), 4);

  EXPECT_EQ(trajectories[0].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(0));
  EXPECT_EQ(trajectories[0].kraus_selections[1].kraus_operator_index,
            cudaq::KrausOperatorType(0));

  EXPECT_EQ(trajectories[1].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(1));
  EXPECT_EQ(trajectories[1].kraus_selections[1].kraus_operator_index,
            cudaq::KrausOperatorType(0));

  EXPECT_EQ(trajectories[2].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(0));
  EXPECT_EQ(trajectories[2].kraus_selections[1].kraus_operator_index,
            cudaq::KrausOperatorType(1));

  EXPECT_EQ(trajectories[3].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(1));
  EXPECT_EQ(trajectories[3].kraus_selections[1].kraus_operator_index,
            cudaq::KrausOperatorType(1));
}

TEST(ExhaustiveSamplingStrategyTest, CapsAtMaxTrajectories) {
  auto noise_points = createSimpleNoisePoints();
  ExhaustiveSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 2);

  EXPECT_EQ(trajectories.size(), 2);
}

TEST(ExhaustiveSamplingStrategyTest, ThreeOperators) {
  auto noise_points = createThreeOperatorNoisePoints();
  ExhaustiveSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 100);

  EXPECT_EQ(trajectories.size(), 3);

  EXPECT_EQ(trajectories[0].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(0));
  EXPECT_EQ(trajectories[1].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(1));
  EXPECT_EQ(trajectories[2].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(2));
}

TEST(ExhaustiveSamplingStrategyTest, EmptyNoisePoints) {
  std::vector<NoisePoint> empty_noise_points;
  ExhaustiveSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(empty_noise_points, 10);

  EXPECT_EQ(trajectories.size(), 0);
}

TEST(ExhaustiveSamplingStrategyTest, StrategyName) {
  ExhaustiveSamplingStrategy strategy;
  EXPECT_STREQ(strategy.name(), "Exhaustive");
}

TEST(ExhaustiveSamplingStrategyTest, Clone) {
  ExhaustiveSamplingStrategy strategy;
  auto cloned = strategy.clone();

  EXPECT_NE(cloned.get(), nullptr);
  EXPECT_STREQ(cloned->name(), "Exhaustive");
}

TEST(OrderedSamplingStrategyTest, SortsByProbability) {
  auto noise_points = createSimpleNoisePoints();
  OrderedSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 4);

  EXPECT_EQ(trajectories.size(), 4);

  for (std::size_t i = 0; i < trajectories.size() - 1; ++i) {
    EXPECT_GE(trajectories[i].probability, trajectories[i + 1].probability)
        << "Probabilities not in descending order at index " << i;
  }
}

TEST(OrderedSamplingStrategyTest, HighestProbabilityFirst) {
  auto noise_points = createSimpleNoisePoints();
  OrderedSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 1);

  ASSERT_EQ(trajectories.size(), 1);

  EXPECT_EQ(trajectories[0].kraus_selections[0].kraus_operator_index,
            cudaq::KrausOperatorType(0));
  EXPECT_EQ(trajectories[0].kraus_selections[1].kraus_operator_index,
            cudaq::KrausOperatorType(0));
  EXPECT_NEAR(trajectories[0].probability, 0.72, 1e-9);
}

TEST(OrderedSamplingStrategyTest, TopKSelection) {
  auto noise_points = createSimpleNoisePoints();
  OrderedSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 2);

  EXPECT_EQ(trajectories.size(), 2);

  EXPECT_NEAR(trajectories[0].probability, 0.72, 1e-9);
  EXPECT_NEAR(trajectories[1].probability, 0.18, 1e-9);
}

TEST(OrderedSamplingStrategyTest, TrajectoryIDReassignment) {
  auto noise_points = createSimpleNoisePoints();
  OrderedSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(noise_points, 4);

  for (std::size_t i = 0; i < trajectories.size(); ++i) {
    EXPECT_EQ(trajectories[i].trajectory_id, i);
  }
}

TEST(OrderedSamplingStrategyTest, EmptyNoisePoints) {
  std::vector<NoisePoint> empty_noise_points;
  OrderedSamplingStrategy strategy;

  auto trajectories = strategy.generateTrajectories(empty_noise_points, 10);

  EXPECT_EQ(trajectories.size(), 0);
}

TEST(OrderedSamplingStrategyTest, StrategyName) {
  OrderedSamplingStrategy strategy;
  EXPECT_STREQ(strategy.name(), "Ordered");
}

TEST(OrderedSamplingStrategyTest, Clone) {
  OrderedSamplingStrategy strategy;
  auto cloned = strategy.clone();

  EXPECT_NE(cloned.get(), nullptr);
  EXPECT_STREQ(cloned->name(), "Ordered");
}

TEST(ConditionalSamplingStrategyTest, FilterByErrorCount) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &traj) {
    return traj.countErrors() == 1;
  };

  ConditionalSamplingStrategy strategy(predicate);
  auto trajectories = strategy.generateTrajectories(noise_points, 10);

  EXPECT_EQ(trajectories.size(), 2);

  for (const auto &traj : trajectories) {
    EXPECT_EQ(traj.countErrors(), 1);
  }
}

TEST(ConditionalSamplingStrategyTest, FilterByNoErrors) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &traj) {
    return traj.countErrors() == 0;
  };

  ConditionalSamplingStrategy strategy(predicate);
  auto trajectories = strategy.generateTrajectories(noise_points, 10);

  EXPECT_EQ(trajectories.size(), 1);
  EXPECT_EQ(trajectories[0].countErrors(), 0);
  EXPECT_NEAR(trajectories[0].probability, 0.72, 1e-9);
}

TEST(ConditionalSamplingStrategyTest, FilterByProbabilityThreshold) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &traj) {
    return traj.probability > 0.1;
  };

  ConditionalSamplingStrategy strategy(predicate);
  auto trajectories = strategy.generateTrajectories(noise_points, 10);

  EXPECT_EQ(trajectories.size(), 2);

  for (const auto &traj : trajectories) {
    EXPECT_GT(traj.probability, 0.1);
  }
}

TEST(ConditionalSamplingStrategyTest, FilterNonePass) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &traj) {
    return traj.probability > 1.0;
  };

  ConditionalSamplingStrategy strategy(predicate);
  auto trajectories = strategy.generateTrajectories(noise_points, 10);

  EXPECT_EQ(trajectories.size(), 0);
}

TEST(ConditionalSamplingStrategyTest, FilterAllPass) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &) { return true; };

  ConditionalSamplingStrategy strategy(predicate);
  auto trajectories = strategy.generateTrajectories(noise_points, 10);

  EXPECT_EQ(trajectories.size(), 4);
}

TEST(ConditionalSamplingStrategyTest, EarlyExit) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &) { return true; };

  ConditionalSamplingStrategy strategy(predicate);
  auto trajectories = strategy.generateTrajectories(noise_points, 2);

  EXPECT_EQ(trajectories.size(), 2);
}

TEST(ConditionalSamplingStrategyTest, EmptyNoisePoints) {
  std::vector<NoisePoint> empty_noise_points;

  auto predicate = [](const cudaq::KrausTrajectory &) { return true; };
  ConditionalSamplingStrategy strategy(predicate);

  auto trajectories = strategy.generateTrajectories(empty_noise_points, 10);

  EXPECT_EQ(trajectories.size(), 0);
}

TEST(ConditionalSamplingStrategyTest, ReproducibilityWithSeed) {
  auto noise_points = createSimpleNoisePoints();

  auto predicate = [](const cudaq::KrausTrajectory &) { return true; };

  ConditionalSamplingStrategy strategy1(predicate, 12345);
  ConditionalSamplingStrategy strategy2(predicate, 12345);
  ConditionalSamplingStrategy strategy3(predicate, 54321);

  auto trajectories1 = strategy1.generateTrajectories(noise_points, 4);
  auto trajectories2 = strategy2.generateTrajectories(noise_points, 4);
  auto trajectories3 = strategy3.generateTrajectories(noise_points, 4);

  EXPECT_EQ(trajectories1.size(), trajectories2.size());
  for (std::size_t i = 0; i < trajectories1.size(); ++i) {
    EXPECT_EQ(trajectories1[i].trajectory_id, trajectories2[i].trajectory_id);
    EXPECT_NEAR(trajectories1[i].probability, trajectories2[i].probability,
                cudaq::PROBABILITY_EPSILON);
  }

  EXPECT_EQ(trajectories3.size(), 4u);
  for (const auto &traj : trajectories3) {
    EXPECT_EQ(traj.kraus_selections.size(), 2u);
    EXPECT_GE(traj.probability, 0.0);
    EXPECT_LE(traj.probability, 1.0);
  }
}

TEST(ConditionalSamplingStrategyTest, StrategyName) {
  auto predicate = [](const cudaq::KrausTrajectory &) { return true; };
  ConditionalSamplingStrategy strategy(predicate);

  EXPECT_STREQ(strategy.name(), "Conditional");
}

TEST(ConditionalSamplingStrategyTest, Clone) {
  auto predicate = [](const cudaq::KrausTrajectory &traj) {
    return traj.countErrors() == 1;
  };
  ConditionalSamplingStrategy strategy(predicate);
  auto cloned = strategy.clone();

  EXPECT_NE(cloned.get(), nullptr);
  EXPECT_STREQ(cloned->name(), "Conditional");

  auto noise_points = createSimpleNoisePoints();
  auto trajectories = cloned->generateTrajectories(noise_points, 10);
  EXPECT_GT(trajectories.size(), 0);
}

TEST(ConditionalSamplingStrategyTest, UsesCUDAQGlobalRandomSeed) {
  auto predicate = [](const cudaq::KrausTrajectory &) { return true; };
  auto noise_points = createSimpleNoisePoints();

  cudaq::set_random_seed(12345);

  ConditionalSamplingStrategy strategy1(predicate);
  auto trajectories1 = strategy1.generateTrajectories(noise_points, 5);

  cudaq::set_random_seed(12345);

  ConditionalSamplingStrategy strategy2(predicate);
  auto trajectories2 = strategy2.generateTrajectories(noise_points, 5);

  ASSERT_EQ(trajectories1.size(), trajectories2.size());
  for (std::size_t i = 0; i < trajectories1.size(); ++i) {
    EXPECT_EQ(trajectories1[i].trajectory_id, trajectories2[i].trajectory_id);
  }

  cudaq::set_random_seed(0);
}

TEST(ProbabilisticSamplingStrategyTest, UsesCUDAQGlobalRandomSeed) {
  auto noise_points = createSimpleNoisePoints();

  cudaq::set_random_seed(54321);

  ProbabilisticSamplingStrategy strategy1;
  auto trajectories1 = strategy1.generateTrajectories(noise_points, 5);

  cudaq::set_random_seed(54321);

  ProbabilisticSamplingStrategy strategy2;
  auto trajectories2 = strategy2.generateTrajectories(noise_points, 5);

  ASSERT_EQ(trajectories1.size(), trajectories2.size());
  for (std::size_t i = 0; i < trajectories1.size(); ++i) {
    EXPECT_EQ(trajectories1[i].trajectory_id, trajectories2[i].trajectory_id);
  }

  cudaq::set_random_seed(0);
}

TEST(PTSSamplingStrategyTest, PolymorphicUsage) {
  auto noise_points = createSimpleNoisePoints();

  std::vector<std::unique_ptr<PTSSamplingStrategy>> strategies;
  strategies.push_back(std::make_unique<ProbabilisticSamplingStrategy>(42));
  strategies.push_back(std::make_unique<ExhaustiveSamplingStrategy>());
  strategies.push_back(std::make_unique<OrderedSamplingStrategy>());
  strategies.push_back(std::make_unique<ConditionalSamplingStrategy>(
      [](const cudaq::KrausTrajectory &) { return true; }));

  for (auto &strategy : strategies) {
    auto trajectories = strategy->generateTrajectories(noise_points, 5);
    EXPECT_GT(trajectories.size(), 0);
    EXPECT_NE(strategy->name(), nullptr);
  }
}

TEST(PTSSamplingStrategyTest, ClonePolymorphism) {
  auto noise_points = createSimpleNoisePoints();

  ProbabilisticSamplingStrategy concrete_strategy(42);
  PTSSamplingStrategy *base_ptr = &concrete_strategy;

  auto cloned = base_ptr->clone();

  EXPECT_NE(cloned.get(), nullptr);
  EXPECT_STREQ(cloned->name(), "Probabilistic");

  auto trajectories = cloned->generateTrajectories(noise_points, 5);
  EXPECT_GT(trajectories.size(), 0);
}

TEST(NoisePointTest, IsUnitaryMixture) {
  NoisePoint np;
  np.channel = makeIXYChannel(0.7, 0.2, 0.1);
  EXPECT_TRUE(np.channel.is_unitary_mixture());
}

TEST(NoisePointTest, IsNotUnitaryMixture) {
  NoisePoint np;
  np.channel = cudaq::amplitude_damping_channel(0.3);
  np.channel.generateUnitaryParameters();
  EXPECT_FALSE(np.channel.is_unitary_mixture());
}

TEST(NoisePointTest, IsUnitaryMixtureWithTolerance) {
  NoisePoint np;
  np.channel = cudaq::depolarization_channel(0.001);
  EXPECT_TRUE(np.channel.is_unitary_mixture());
}

TEST(NoisePointTest, FullUnitaryMixtureValidation) {
  NoisePoint np;
  np.circuit_location = 0;
  np.qubits = {0};
  np.op_name = "h";
  np.channel = cudaq::depolarization_channel(0.3);
  EXPECT_TRUE(np.channel.is_unitary_mixture());
}

TEST(NoisePointTest, NonUnitaryKrausOperators) {
  NoisePoint np;
  np.circuit_location = 0;
  np.qubits = {0};
  np.op_name = "h";
  np.channel = cudaq::amplitude_damping_channel(0.3);
  np.channel.generateUnitaryParameters();
  EXPECT_FALSE(np.channel.is_unitary_mixture());
}
