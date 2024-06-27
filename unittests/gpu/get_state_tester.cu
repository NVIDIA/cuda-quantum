/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CUDAQTestUtils.h"
#include <cudaq/algorithm.h>
#include <cudaq/optimizers.h>

#include <numeric>

#include <thrust/complex.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__qpu__ void bell() {
  cudaq::qubit q, r;
  h(q);
  cx(q, r);
}

CUDAQ_TEST(GetStateTester, checkOverlapFromHostVector) {
  auto state = cudaq::get_state(bell);
  state.dump();
  std::vector<std::complex<double>> hostStateData{M_SQRT1_2, 0, 0, M_SQRT1_2};
  auto hostState = cudaq::state::from_data(hostStateData);
  hostState.dump();
  // Check overlap with host vector
  EXPECT_NEAR(1.0, state.overlap(hostState).real(), 1e-3);
}

CUDAQ_TEST(GetStateTester, checkOverlapFromHostPointer) {
  auto state = cudaq::get_state(bell);
  state.dump();
  std::vector<std::complex<double>> hostStateData{M_SQRT1_2, 0, 0, M_SQRT1_2};
  auto hostState =
      cudaq::state::from_data(std::make_pair(hostStateData.data(), 4));
  hostState.dump();
  // Check overlap with host vector
  EXPECT_NEAR(1.0, state.overlap(hostState).real(), 1e-3);
}

CUDAQ_TEST(GetStateTester, checkOverlapPrecisionErrorThrows) {
  std::vector<std::complex<float>> hostStateData{M_SQRT1_2, 0, 0, M_SQRT1_2};
  EXPECT_ANY_THROW(auto hostState = cudaq::state::from_data(hostStateData));
}

CUDAQ_TEST(GetStateTester, checkOverlapFromGPUPointer) {
  // Generate a state on device
  auto state = cudaq::get_state(bell);
  state.dump();

  // Create a GPU device state vector
  thrust::host_vector<thrust::complex<double>> hostVector(4);
  hostVector[0] = M_SQRT1_2;
  hostVector[3] = M_SQRT1_2;
  thrust::device_vector<thrust::complex<double>> devState = hostVector;

  // Get the raw pointer.
  auto *devPtr = thrust::raw_pointer_cast(&devState[0]);

  // Create the cudaq state
  auto hostState = cudaq::state::from_data(
      std::make_pair(reinterpret_cast<std::complex<double> *>(devPtr), 4));
  hostState.dump();

  // Check overlap with host vector
  EXPECT_NEAR(1.0, state.overlap(hostState).real(), 1e-3);
}

CUDAQ_TEST(GetStateTester, checkCanGetElements) {
  // Generate a state on device
  auto state = cudaq::get_state(bell);
  state.dump();
  EXPECT_NEAR(M_SQRT1_2, state[0].real(), 1e-3);
  EXPECT_NEAR(M_SQRT1_2, state[3].real(), 1e-3);
  EXPECT_NEAR(0.0, state[1].real(), 1e-3);
  EXPECT_NEAR(0.0, state[2].real(), 1e-3);
}

CUDAQ_TEST(GetStateTester, checkTensors) {
  auto state = cudaq::get_state(bell);
  EXPECT_EQ(1, state.get_num_tensors());
  EXPECT_EQ(1, state.get_tensor().get_rank());
  EXPECT_EQ(4, state.get_tensor().get_num_elements());
}
