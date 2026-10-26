/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file nvjitlink_linker.h
/// @brief Run-time device link of a DEVICE + UNIFIED dispatcher's module.
/// Internal to libcudaq-realtime.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/device_code/device_code.h"
#include <cstddef>
#include <cudaTypedefs.h>

namespace cudaq::realtime::detail {

/// The CUDA driver functions that load, query and launch the linked module.
/// The runtime's equivalents (cudaLibrary*) only exist from CUDA 12.8, and
/// calling the driver functions directly would link the `libcuda` driver
/// library, which libcudaq-realtime.so must load without.  So their addresses
/// are taken at run time from the runtime's cudaGetDriverEntryPointByVersion,
/// out of the `libcuda` the runtime itself opens.
///
/// `PFN_<name>_v<N>` (cudaTypedefs.h) is the type of `name` as of CUDA
/// version N, where it last changed.  The lookup requests version 12000
/// (CUDA 12.0), so each type must be the newest one no newer than 12000.
struct driver_library_api {
  PFN_cuLibraryLoadData_v12000 load_data;
  PFN_cuLibraryGetKernel_v12000 get_kernel;
  PFN_cuLibraryGetGlobal_v12000 get_global;
  PFN_cuLibraryUnload_v12000 unload;
  PFN_cuLaunchKernel_v4000 launch_kernel;
};

/// The driver's library API, looked up on the first call.  Null, after
/// printing which function is missing, if the driver lacks any of them.
const driver_library_api *driver_library();

/// One producer's images, named in diagnostics by `origin`.
struct device_code_input {
  const char *origin;
  const cudaq_device_image_t *images;
  std::size_t count;
};

/// Link `inputs`, in order, into one module for the current device with
/// nvJitLink (`-arch=sm_<major><minor> -lto`), and load it.  Each image may
/// hold several `fatbin`s back to back, as a binary's __nv_relfatbin section
/// does; each is added on its own.  On failure,
/// prints what failed together with the nvJitLink version and its log, and
/// leaves `*out_library` untouched.
cudaq_status_t link_device_code(const device_code_input *inputs,
                                std::size_t input_count,
                                CUlibrary *out_library);

} // namespace cudaq::realtime::detail
