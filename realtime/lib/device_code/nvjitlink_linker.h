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

namespace cudaq::realtime::detail {

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
                                cudaLibrary_t *out_library);

} // namespace cudaq::realtime::detail
