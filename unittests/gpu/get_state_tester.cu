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

CUDAQ_TEST(GetStateTester, checkGPUDeviceState) {

  // Generate a state.
  auto kernel = []() __qpu__ {
    cudaq::qubit q, r;
    h(q);
    cx(q, r);
  };

  auto state = cudaq::get_state(kernel);
  state.dump();
  std::vector<std::complex<double>> hostState{M_SQRT1_2, 0, 0, M_SQRT1_2};
  // Check overlap with host vector
  EXPECT_NEAR(1.0, state.overlap(hostState), 1e-3);

  // Copies host data to device
  thrust::host_vector<thrust::complex<double>> hostVector(4);
  hostVector[0] = M_SQRT1_2;
  hostVector[3] = M_SQRT1_2;
  thrust::device_vector<thrust::complex<double>> devState = hostVector;
  // FIXME in the future it'd be better to have thrust exposed in the API.
  auto *devPtr = thrust::raw_pointer_cast(&devState[0]);
  // check overlap with device vector
  EXPECT_NEAR(1.0,
              state.overlap(reinterpret_cast<std::complex<double> *>(devPtr)),
              1e-3);

  // check can get device to host
  auto *copiedFromDevice =
      reinterpret_cast<std::complex<double> *>(state.data_holder()->toHost());
  std::vector<std::complex<double>> copied(copiedFromDevice,
                                           copiedFromDevice + 4);
  EXPECT_NEAR(1. / std::sqrt(2.), state[0].real(), 1e-3);
  EXPECT_NEAR(0., state[1].real(), 1e-3);
  EXPECT_NEAR(0., state[2].real(), 1e-3);
  EXPECT_NEAR(1. / std::sqrt(2.), state[3].real(), 1e-3);
  free(copiedFromDevice);

  // Can use cudaMalloc and memcpy to test data. 
  void *devPtr2 = nullptr;
  cudaMalloc((void **)&devPtr2, 4 * sizeof(std::complex<double>));
  cudaMemcpy(devPtr2, hostState.data(), 4 * sizeof(std::complex<double>),
             cudaMemcpyHostToDevice);
  EXPECT_NEAR(1.0,
              state.overlap(reinterpret_cast<std::complex<double> *>(devPtr2)),
              1e-3);
  cudaFree(devPtr2);
}