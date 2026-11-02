/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file driver_launch.h
/// @brief cuLaunchKernelEx for the tests' launchers of an application's own
/// dispatch kernel (cudaq_dispatcher_get_device_kernel).
///
/// The driver API takes that kernel on CUDA 12 and 13 alike.  An application
/// would simply link `libcuda` and call cuLaunchKernelEx.  The test binaries
/// must not have `libcuda` in NEEDED: `gtest` discovery runs them at build
/// time on CI runners without a GPU, and the test server also serves the
/// host-path tests.  So the function is taken from the `libcuda` the CUDA
/// runtime opens, as `libcudaq-realtime` does.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include <cstdio>
#include <cudaTypedefs.h>
#include <cuda_runtime.h>

namespace cudaq::realtime::testing {

/// cuLaunchKernelEx(config, kernel, args): CUDAQ_OK once launched, or
/// CUDAQ_ERR_CUDA after printing what failed.
inline cudaq_status_t launch_kernel_ex(const CUlaunchConfig *config,
                                       cudaKernel_t kernel, void **args) {
  static const PFN_cuLaunchKernelEx_v11060 launch_kernel_ex = [] {
    void *fn = nullptr;
    cudaDriverEntryPointQueryResult found = cudaDriverEntryPointSymbolNotFound;
    if (cudaGetDriverEntryPointByVersion("cuLaunchKernelEx", &fn, 12000,
                                         cudaEnableDefault,
                                         &found) != cudaSuccess ||
        found != cudaDriverEntryPointSuccess)
      fn = nullptr;
    return reinterpret_cast<PFN_cuLaunchKernelEx_v11060>(fn);
  }();
  if (!launch_kernel_ex) {
    std::fprintf(stderr, "ERROR: the CUDA driver has no cuLaunchKernelEx\n");
    return CUDAQ_ERR_CUDA;
  }
  const CUresult err = launch_kernel_ex(
      config, reinterpret_cast<CUfunction>(kernel), args, nullptr);
  if (err != CUDA_SUCCESS) {
    std::fprintf(stderr, "ERROR: cuLaunchKernelEx: CUresult %d\n",
                 static_cast<int>(err));
    return CUDAQ_ERR_CUDA;
  }
  return CUDAQ_OK;
}

} // namespace cudaq::realtime::testing
