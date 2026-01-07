/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <cstddef>
#include <stdint.h>
#include <thrust/complex.h>
#include <thrust/device_ptr.h>

namespace nvqir {

template <typename CudaDataType>
void setFirstNElements(uint32_t n_blocks, int32_t threads_per_block,
                       void *newDeviceStateVector, void *deviceStateVector,
                       size_t previousStateDimension);

template <typename CudaDataType>
void initializeDeviceStateVector(uint32_t n_blocks, int32_t threads_per_block,
                                 void *deviceStateVector,
                                 size_t stateDimension);

template <typename CudaDataType>
void kronprod(uint32_t n_blocks, int32_t threads_per_block, size_t tsize1,
              const void *arr1, size_t tsize2, const void *arr2, void *arr0);

#pragma pack(push, 4)
template <typename ScalarType>
struct complexValue {
  ScalarType real;
  ScalarType imaginary;
};
#pragma pack(pop)

template <typename ScalarType>
complexValue<ScalarType> innerProduct(void *devicePtr, void *otherPtr,
                                      std::size_t size, bool createDeviceAlloc);

} // namespace nvqir
