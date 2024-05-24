/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <stdint.h>

namespace nvqir {

template <typename CudaDataType>
void setFirstNElements(uint32_t n_blocks, int32_t threads_per_block,
                       void *newDeviceStateVector, void *deviceStateVector,
                       size_t previousStateDimension);

template <typename CudaDataType>
void initializeDeviceStateVector(uint32_t n_blocks, int32_t threads_per_block,
                                 void *deviceStateVector,
                                 size_t stateDimension);
} // namespace nvqir