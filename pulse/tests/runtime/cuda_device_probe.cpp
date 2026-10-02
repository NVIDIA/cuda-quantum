/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include <cuda_runtime_api.h>

int main() {
  int deviceCount = 0;
  const auto status = cudaGetDeviceCount(&deviceCount);
  return status == cudaSuccess && deviceCount > 0 ? 0 : 1;
}
