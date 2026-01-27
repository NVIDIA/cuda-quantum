/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "cudaq/ptsbe/TrajectoryMetadata.h"

using namespace cudaq;
using namespace cudaq::ptsbe;

CUDAQ_TEST(TrajectoryMetadataTest, DefaultConstruction) {
  TrajectoryMetadata meta;
  EXPECT_EQ(meta.trajectory_id, 0);
  EXPECT_TRUE(meta.kraus_selections.empty());
  EXPECT_EQ(meta.probability, 0.0);
  EXPECT_EQ(meta.num_shots, 0);
}

CUDAQ_TEST(TrajectoryMetadataTest, ParameterizedConstruction) {
  std::vector<KrausSelection> sels = {
      KrausSelection(0, {0}, "h", KrausOperatorIndex{2})};

  TrajectoryMetadata meta(10, sels, 0.25, 500);

  EXPECT_EQ(meta.trajectory_id, 10);
  EXPECT_EQ(meta.kraus_selections.size(), 1);
  EXPECT_NEAR(meta.probability, 0.25, 1e-9);
  EXPECT_EQ(meta.num_shots, 500);
}

CUDAQ_TEST(TrajectoryMetadataTest, ConstructionFromKrausTrajectory) {
  std::vector<KrausSelection> sels = {
      KrausSelection(0, {0}, "h", KrausOperatorIndex{2})};
  KrausTrajectory traj(10, sels, 0.25, 500);

  TrajectoryMetadata meta(traj);

  EXPECT_EQ(meta.trajectory_id, 10);
  EXPECT_EQ(meta.kraus_selections.size(), 1);
  EXPECT_NEAR(meta.probability, 0.25, 1e-9);
  EXPECT_EQ(meta.num_shots, 500);
}

CUDAQ_TEST(TrajectoryMetadataTest, Equality) {
  std::vector<KrausSelection> sels = {
      KrausSelection(0, {0}, "h", KrausOperatorIndex{1})};

  TrajectoryMetadata meta1(1, sels, 0.5, 100);
  TrajectoryMetadata meta2(1, sels, 0.5, 100);
  TrajectoryMetadata meta3(2, sels, 0.5, 100);

  EXPECT_TRUE(meta1 == meta2);
  EXPECT_FALSE(meta1 == meta3);
}

CUDAQ_TEST(TrajectoryMetadataTest, MultipleTrajectories) {
  std::vector<TrajectoryMetadata> metadata;

  for (std::size_t i = 0; i < 5; ++i) {
    metadata.push_back(TrajectoryMetadata(i,            // trajectory_id
                                          {},           // kraus_selections
                                          0.2,          // probability
                                          100 * (i + 1) // num_shots
                                          ));
  }

  EXPECT_EQ(metadata.size(), 5);

  for (std::size_t i = 0; i < 5; ++i) {
    EXPECT_EQ(metadata[i].trajectory_id, i);
    EXPECT_EQ(metadata[i].num_shots, 100 * (i + 1));
  }
}

CUDAQ_TEST(TrajectoryMetadataTest, ParallelArrays) {
  std::vector<TrajectoryMetadata> metadata;
  std::vector<std::size_t> shots_per_trajectory;

  for (std::size_t i = 0; i < 3; ++i) {
    metadata.push_back(TrajectoryMetadata(i, {}, 0.3, 100 * (i + 1)));
    shots_per_trajectory.push_back(100 * (i + 1));
  }

  EXPECT_EQ(metadata.size(), 3);
  EXPECT_EQ(shots_per_trajectory.size(), 3);

  for (std::size_t i = 0; i < 3; ++i) {
    EXPECT_EQ(metadata[i].num_shots, shots_per_trajectory[i]);
  }
}

CUDAQ_TEST(TrajectoryMetadataTest, ConstexprEquality) {
  TrajectoryMetadata meta1(1, {}, 0.5, 100);
  TrajectoryMetadata meta2(1, {}, 0.5, 100);
  TrajectoryMetadata meta3(2, {}, 0.5, 100);

  EXPECT_TRUE(meta1 == meta2);
  EXPECT_FALSE(meta1 == meta3);
}
