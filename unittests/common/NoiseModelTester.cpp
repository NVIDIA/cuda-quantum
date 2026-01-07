/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include "common/FmtCore.h"
#include "common/SampleResult.h"

using namespace cudaq;

CUDAQ_TEST(NoiseModelTester, checkConstruction) {
  // Amplitude damping, p = 0.5
  cudaq::kraus_channel simpleChannel{{1., 0., 0., .8660254037844386},
                                     {0., 0.5, 0., 0.}};

  cudaq::kraus_channel moreComplicatedChannel(
      {complex{0.99498743710662, 0.0},
       {0.0, 0.0},
       {0.0, 0.0},
       {0.99498743710662, 0.0}},

      {complex{0.0, 0.0},
       {0.05773502691896258, 0.0},
       {0.05773502691896258, 0.0},
       {0.0, 0.0}},

      {complex{0.0, 0.0},
       {0.0, -0.05773502691896258},
       {0.0, 0.05773502691896258},
       {0.0, 0.0}},

      {complex{0.05773502691896258, 0.0},
       {0.0, 0.0},
       {0.0, 0.0},
       {-0.05773502691896258, 0.0}});

  // Invalid number of matrix elements for a 2 qubit channel.
  EXPECT_ANY_THROW({ cudaq::kraus_channel bad({1, 2}, {1., 0., 0., 0.}); });

  cudaq::noise_model noise;
  // Can add by name
  noise.add_channel("x", {1}, moreComplicatedChannel);
  // Preferred - can add by type
  noise.add_channel<cudaq::types::h>({0}, simpleChannel);

  auto kraus_channels = noise.get_channels("x", {1});
  // We only add one channel
  EXPECT_EQ(1, kraus_channels.size());
  // That channel has 4 kraus_ops
  EXPECT_EQ(4, kraus_channels[0].size());

  // The first kraus op should be the following
  std::vector<complex> expected{complex{0.99498743710662, 0.0},
                                {0.0, 0.0},
                                {0.0, 0.0},
                                {0.99498743710662, 0.0}};
  EXPECT_EQ(expected, kraus_channels[0][0].data);

  // No channel for h on qubit 1, its on 0
  kraus_channels = noise.get_channels("h", {1});
  EXPECT_TRUE(kraus_channels.empty());

  // Preferred can get by type
  kraus_channels = noise.get_channels<cudaq::types::h>({0});
  EXPECT_EQ(1, kraus_channels.size());
  EXPECT_EQ(kraus_channels[0].size(), 2);

  noise.add_channel<cudaq::types::y, cudaq::types::z>({0}, simpleChannel);

  // Can only add channels for ops we know about.
  EXPECT_ANY_THROW({ noise.add_channel("invalid_op", {0}, simpleChannel); });
}

#if defined(CUDAQ_SIMULATION_SCALAR_FP64)
CUDAQ_TEST(NoiseModelTester, checkUnitaryDetection) {
  EXPECT_TRUE(amplitude_damping(0.5).unitary_ops.empty());
  EXPECT_FALSE(depolarization1(0.1).unitary_ops.empty());
  // Small probability depolarization
  EXPECT_FALSE(depolarization1(1e-6).unitary_ops.empty());
  EXPECT_EQ(depolarization1(1e-6).unitary_ops.size(), 4);
  EXPECT_EQ(depolarization1(1e-6).probabilities.size(), 4);
  EXPECT_NEAR(depolarization1(1e-6).probabilities[0], 1.0 - 1e-6, 1e-8);
  EXPECT_NEAR(depolarization1(1e-6).probabilities[1], 1e-6 / 3.0, 1e-8);
  EXPECT_NEAR(depolarization1(1e-6).probabilities[2], 1e-6 / 3.0, 1e-8);
  EXPECT_NEAR(depolarization1(1e-6).probabilities[3], 1e-6 / 3.0, 1e-8);

  // Depolarization 2-qubit
  EXPECT_FALSE(depolarization2(0.1).unitary_ops.empty());
  EXPECT_FALSE(depolarization2(1e-4).unitary_ops.empty());
}
#endif
