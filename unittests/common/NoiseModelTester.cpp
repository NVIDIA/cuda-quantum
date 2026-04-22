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

CUDAQ_TEST(NoiseModelTester, checkOpNames) {
  // Standard channel gets explicit names.
  auto depol = depolarization_channel(0.1);
  ASSERT_EQ(depol.op_names.size(), depol.size());
  EXPECT_EQ(depol.op_names[0], "id");
  EXPECT_EQ(depol.op_names[1], "x");
  EXPECT_EQ(depol.op_names[2], "y");
  EXPECT_EQ(depol.op_names[3], "z");

  // Non-unitary channel gets auto-generated default names.
  auto ad = amplitude_damping(0.5);
  ASSERT_EQ(ad.op_names.size(), ad.size());
  EXPECT_EQ(ad.op_names[0], "amplitude_damping[0]");
  EXPECT_EQ(ad.op_names[1], "amplitude_damping[1]");

  // Custom channel from raw kraus_ops gets "unknown[k]" defaults.
  cudaq::kraus_channel custom{{1., 0., 0., .8660254037844386},
                              {0., 0.5, 0., 0.}};
  ASSERT_EQ(custom.op_names.size(), 2u);
  EXPECT_EQ(custom.op_names[0], "unknown[0]");
  EXPECT_EQ(custom.op_names[1], "unknown[1]");

  // Copy preserves op_names.
  kraus_channel copy(depol);
  EXPECT_EQ(copy.op_names, depol.op_names);

  // push_back appends default name; explicit name overrides.
  auto channel = bit_flip_channel(0.1);
  channel.push_back(kraus_op({1., 0., 0., 1.}));
  EXPECT_EQ(channel.op_names.back(), "bit_flip_channel[2]");
  channel.push_back(kraus_op({1., 0., 0., 1.}), "custom_op");
  EXPECT_EQ(channel.op_names.back(), "custom_op");
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

  auto depol2_channel = depolarization2(0.3);
  EXPECT_EQ(depol2_channel.size(), 16);
  EXPECT_EQ(depol2_channel.unitary_ops.size(), 16);
  EXPECT_EQ(depol2_channel.probabilities.size(), 16);

  EXPECT_NEAR(depol2_channel.probabilities[0], 1.0 - 0.3, 1e-8);

  for (std::size_t i = 1; i < 16; ++i) {
    EXPECT_NEAR(depol2_channel.probabilities[i], 0.3 / 15.0, 1e-8);
  }

  auto depol2_small = depolarization2(1e-6);
  EXPECT_EQ(depol2_small.size(), 16);
  EXPECT_NEAR(depol2_small.probabilities[0], 1.0 - 1e-6, 1e-12);
  for (std::size_t i = 1; i < 16; ++i) {
    EXPECT_NEAR(depol2_small.probabilities[i], 1e-6 / 15.0, 1e-12);
  }
}
#endif
