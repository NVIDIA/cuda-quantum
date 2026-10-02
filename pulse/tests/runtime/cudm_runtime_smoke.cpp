/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudm_runtime.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstdio>
#include <cstring>

int main(int argc, char **argv) {
  const auto version = cudm_runtime_version();
  if (version <= 0) {
    std::fprintf(stderr, "cuDensityMat returned an invalid version\n");
    return 1;
  }

  std::printf("cuDensityMat runtime version: %lld\n",
              static_cast<long long>(version));
  if (argc == 1 || std::strcmp(argv[1], "--gpu") != 0)
    return 0;

  int deviceCount = 0;
  const auto cudaStatus = cudaGetDeviceCount(&deviceCount);
  if (cudaStatus != cudaSuccess || deviceCount == 0) {
    std::fprintf(stderr, "No accessible NVIDIA GPU; skipping GPU smoke test\n");
    return 77;
  }

  CudmHandle handle = nullptr;
  if (cudm_init(&handle) != CUDM_SUCCESS)
    return 2;

  constexpr std::array<int64_t, 1> modeExtents = {2};
  CudmState state = nullptr;
  CudmWorkspace workspace = nullptr;
  CudmOperator op = nullptr;

  const bool created =
      cudm_state_alloc(handle, &state, modeExtents.data(), modeExtents.size(),
                       0, 16) == CUDM_SUCCESS &&
      cudm_workspace_create(handle, &workspace) == CUDM_SUCCESS &&
      cudm_operator_create(handle, &op, modeExtents.data(),
                           modeExtents.size()) == CUDM_SUCCESS;

  if (op)
    cudm_operator_destroy(op);
  if (workspace)
    cudm_workspace_destroy(workspace);
  if (state)
    cudm_state_destroy(state);
  cudm_destroy(handle);

  if (!created) {
    std::fprintf(stderr, "Failed to construct cuDensityMat GPU descriptors\n");
    return 3;
  }

  std::puts("cuDensityMat GPU descriptor smoke test passed");
  return 0;
}
