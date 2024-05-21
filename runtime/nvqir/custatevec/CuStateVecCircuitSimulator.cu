/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma nv_diag_suppress = unsigned_compare_with_zero
#pragma nv_diag_suppress = unrecognized_gcc_pragma

#include "cuComplex.h"
#include "device_launch_parameters.h"
#include "CuStateVecCircuitSimulator.h"

namespace nvqir {
/// @brief Initialize the device state vector to the |0...0> state
template <typename CudaDataType>
__global__ void cudaInitializeDeviceStateVector(CudaDataType *sv, int64_t dim) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i == 0) {
    sv[i].x = 1.0;
    sv[i].y = 0.0;
  } else if (i < dim) {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

/// @brief Kernel to set the first N elements of the state vector sv equal to
/// the
// elements provided by the vector sv2. N is the number of elements to set.
// Size of sv must be greater than size of sv2.
template <typename CudaDataType>
__global__ void cudaSetFirstNElements(CudaDataType *sv, const CudaDataType *__restrict__ sv2, int64_t N) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < N) {
    sv[i].x = sv2[i].x;
    sv[i].y = sv2[i].y;
  } else {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

template <typename CudaDataType>
void setFirstNElements(uint32_t n_blocks, 
                       int32_t threads_per_block, 
                       void *newDeviceStateVector, 
                       void *deviceStateVector,
                       std::size_t previousStateDimension) {
  cudaSetFirstNElements<<<n_blocks, threads_per_block>>>(
    reinterpret_cast<CudaDataType *>(newDeviceStateVector),
    reinterpret_cast<CudaDataType *>(deviceStateVector),
    previousStateDimension);
}

template void
setFirstNElements<cuFloatComplex>(uint32_t n_blocks, 
                       int32_t threads_per_block, 
                       void *newDeviceStateVector, 
                       void *deviceStateVector,
                       std::size_t previousStateDimension);

template void
setFirstNElements<cuDoubleComplex>(uint32_t n_blocks, 
                       int32_t threads_per_block, 
                       void *newDeviceStateVector, 
                       void *deviceStateVector,
                       std::size_t previousStateDimension);

template <typename CudaDataType>
void initializeDeviceStateVector(uint32_t n_blocks, 
                                 int32_t threads_per_block, 
                                 void *deviceStateVector,
                                 std::size_t stateDimension) {
  cudaInitializeDeviceStateVector<<<n_blocks, threads_per_block>>>(
    reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension);
}

template void
initializeDeviceStateVector<cuFloatComplex>(uint32_t n_blocks, 
                                 int32_t threads_per_block, 
                                 void *deviceStateVector,
                                 std::size_t stateDimension);

template void
initializeDeviceStateVector<cuDoubleComplex>(uint32_t n_blocks, 
                                 int32_t threads_per_block, 
                                 void *deviceStateVector,
                                 std::size_t stateDimension);
}