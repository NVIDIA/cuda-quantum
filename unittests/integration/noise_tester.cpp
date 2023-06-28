/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <stdio.h>

#ifdef CUDAQ_BACKEND_DM
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

CUDAQ_TEST(NoiseTest, checkSimple) {
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

CUDAQ_TEST(NoiseTest, checkAmplitudeDamping) {
  cudaq::kraus_channel amplitudeDamping{{1., 0., 0., .8660254037844386},
                                        {0., 0.0, 0.5, 0.}};
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, amplitudeDamping);
  cudaq::set_noise(noise);

  auto counts = cudaq::sample(xOp{});
  counts.dump();

  EXPECT_NEAR(counts.probability("0"), .25, .1);
  EXPECT_NEAR(counts.probability("1"), .75, .1);
}

CUDAQ_TEST(NoiseTest, checkCNOT) {

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
}

CUDAQ_TEST(NoiseTest, checkExceptions) {
  cudaq::kraus_channel amplitudeDamping{{1., 0., 0., .8660254037844386},
                                        {0., 0.0, 0.5, 0.}};
  cudaq::noise_model noise;
  EXPECT_ANY_THROW({
    noise.add_channel<cudaq::types::x>({0, 1}, amplitudeDamping);
  });
}

CUDAQ_TEST(NoiseTest, checkDepolType) {
  cudaq::depolarization_channel depol(.1);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, depol);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
}

CUDAQ_TEST(NoiseTest, checkAmpDampType) {
  cudaq::amplitude_damping_channel ad(.25);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, ad);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  EXPECT_NEAR(counts.probability("0"), .25, .1);
  EXPECT_NEAR(counts.probability("1"), .75, .1);
}

CUDAQ_TEST(NoiseTest, checkBitFlipType) {
  cudaq::amplitude_damping_channel bf(.1);
  cudaq::noise_model noise;
  noise.add_channel<cudaq::types::x>({0}, bf);
  cudaq::set_noise(noise);
  auto counts = cudaq::sample(xOp{});
  counts.dump();
  EXPECT_EQ(2, counts.size());
  EXPECT_NEAR(counts.probability("0"), .1, .1);
  EXPECT_NEAR(counts.probability("1"), .9, .1);
}
#endif
