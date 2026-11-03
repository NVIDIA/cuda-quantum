/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file device_code.h
/// @brief Relocatable device code handed to a DEVICE + UNIFIED dispatcher.
///
/// The unified dispatch kernel, a bridge's device hooks and an application's
/// DEVICE_CALL handlers come from three different binaries.  The dispatcher
/// links them into one CUDA module at run time with nvJitLink, so each binary
/// hands over its relocatable device code as a list of images: the bridge
/// through `cudaq_gpu_dataplane_t::device_code`, the application when it
/// looks up a handler with `cudaq_dispatcher_get_device_handler`.  Both get
/// it from `cudaq_device_code_of_this_binary` in device_code_reader.h.
///
/// No CUDA dependency: a bridge or application includes this header from any
/// translation unit.  See realtime/docs/device_code_contract.md.

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/// Relocatable device code: one `fatbin`, as `nvcc -rdc` emits one per
/// translation unit, or several back to back, as in a binary's
/// `__nv_relfatbin` section.  The dispatcher splits an image into its
/// `fatbin`s when it links.
/// The bytes must stay valid for as long as the dispatcher that links them.
typedef struct {
  const void *data;
  size_t size;
} cudaq_device_image_t;

#ifdef __cplusplus
}
#endif

/// Prefix of the device pointer global through which
/// `cudaq_dispatcher_get_device_handler` finds an exported handler.
#define CUDAQ_REALTIME_DEVICE_FUNCTION_PREFIX "cudaq_rt_device_fn_"

/// Export the `__device__` function `fn` from the linked module, so that
/// `cudaq_dispatcher_get_device_handler(dispatcher, "fn", &ptr)` can return
/// its address.  CUDA looks up kernels and globals by name, never device
/// functions, so the address travels through a pointer global.
///
/// Use at namespace scope in a CUDA translation unit, after `fn` is declared.
/// `fn` itself may have internal linkage.  The braces are required: under
/// `-rdc`, `extern "C" __device__ T x = v;` only declares `x`.
#define CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(fn)                              \
  extern "C" {                                                                 \
  __device__ void *cudaq_rt_device_fn_##fn = (void *)&fn;                      \
  }
