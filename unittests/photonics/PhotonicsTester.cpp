/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>

#include "cudaq.h"
#include "cudaq/photonics.h"

TEST(PhotonicsTester, checkSimple) {

  struct test {
    auto operator()() __qpu__ {
      cudaq::qvector<3> qumodes(2);
      create(qumodes[0]);
      create(qumodes[1]);
      create(qumodes[1]);
      return mz(qumodes);
    }
  };

  struct test2 {
    void operator()() __qpu__ {
      cudaq::qvector<3> qumodes(2);
      create(qumodes[0]);
      create(qumodes[1]);
      create(qumodes[1]);
      mz(qumodes);
    }
  };

  auto res = test{}();
  EXPECT_EQ(res[0], 1);
  EXPECT_EQ(res[1], 2);

  auto counts = cudaq::sample(test2{});
  for (auto &[k, v] : counts) {
    printf("Result / Count = %s : %lu\n", k.c_str(), v);
  }
}

TEST(PhotonicsTester, checkHOM) {

  struct HOM {
    // Hong–Ou–Mandel effect
    auto operator()(double theta) __qpu__ {

      constexpr std::array<std::size_t, 2> input_state{1, 1};

      cudaq::qvector<3> qumodes(2); // |00>
      for (std::size_t i = 0; i < 2; i++) {
        for (std::size_t j = 0; j < input_state[i]; j++) {
          create(qumodes[i]); // setting to  |11>
        }
      }

      beam_splitter(qumodes[0], qumodes[1], theta);
      mz(qumodes);
    }
  };

  auto counts = cudaq::sample(HOM{}, M_PI / 4);
  printf("Angle : %.2f \n", 180. / 4);
  for (auto &[k, v] : counts) {
    printf("Result / Count = %s : %lu\n", k.c_str(), v);
  }
  EXPECT_EQ(counts.size(), 2);

  auto counts2 = cudaq::sample(HOM{}, M_PI / 6);
  printf("Angle : %.2f \n", 180. / 6);
  for (auto &[k, v] : counts2) {
    printf("Result / Count = %s : %lu\n", k.c_str(), v);
  }
  EXPECT_EQ(counts2.size(), 3);

  auto counts3 = cudaq::sample(HOM{}, M_PI / 5);
  printf("Angle : %.2f \n", 180. / 5);
  for (auto &[k, v] : counts3) {
    printf("Result / Count = %s : %lu\n", k.c_str(), v);
  }
  EXPECT_EQ(counts3.size(), 3);
}

TEST(PhotonicsTester, checkMZI) {

  struct MZI {
    // Mach-Zendher Interferometer
    auto operator()() __qpu__ {

      constexpr std::array<std::size_t, 2> input_state{1, 0};

      cudaq::qvector<3> qumodes(2); // |00>
      for (std::size_t i = 0; i < 2; i++)
        for (std::size_t j = 0; j < input_state[i]; j++)
          create(qumodes[i]); // setting to  |10>

      beam_splitter(qumodes[0], qumodes[1], M_PI / 4);
      phase_shift(qumodes[0], M_PI / 3);

      beam_splitter(qumodes[0], qumodes[1], M_PI / 4);
      phase_shift(qumodes[0], M_PI / 3);

      mz(qumodes);
    }
  };

  cudaq::set_random_seed(13);
  std::size_t shots = 1000000;
  auto counts = cudaq::sample(shots, MZI{}); //
  counts.dump();
  EXPECT_NEAR(double(counts.count("10")) / shots, cos(M_PI / 3) * cos(M_PI / 3),
              1e-3);
}
