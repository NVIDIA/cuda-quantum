/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file device_code_reader.h
/// @brief Header-only access to the relocatable device code of the binary
/// that includes this header.
///
/// `nvcc -rdc` places one relocatable `fatbin` per translation unit in the
/// `__nv_relfatbin` section of the final executable or shared library, and
/// the linker brackets that section with `__start___nv_relfatbin` and
/// `__stop___nv_relfatbin`.  Declared hidden, those resolve inside the binary
/// that includes this header, so a bridge and an application calling
/// `cudaq_device_code_of_this_binary` each get their own code.  A binary
/// without the section sees them as null.
///
/// The section is handed over whole, as one image; the dispatcher splits it
/// into its `fatbin`s when it links.
///
/// Everything here has internal linkage on purpose: an inline function would
/// be exported, and one binary could then end up running the copy in another
/// binary.

#include "cudaq/realtime/daemon/dispatcher/cudaq_status.h"
#include "cudaq/realtime/device_code/device_code.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

extern const char __start___nv_relfatbin[]
    __attribute__((weak, visibility("hidden")));
extern const char __stop___nv_relfatbin[]
    __attribute__((weak, visibility("hidden")));

/// Fill `*out_image` with the relocatable device code of the binary this call
/// is compiled into, its whole `__nv_relfatbin` section, as
/// `cudaq_gpu_dataplane_t::device_code` and
/// `cudaq_dispatcher_get_device_handler_in` take.  The bytes stay valid for
/// the life of the process.
///
/// Returns CUDAQ_ERR_UNSUPPORTED when the binary carries no relocatable device
/// code.
static inline cudaq_status_t
cudaq_device_code_of_this_binary(cudaq_device_image_t *out_image) {
  if (!out_image)
    return CUDAQ_ERR_INVALID_ARG;
  // As pointers: comparing the arrays themselves is -Warray-compare.
  const char *begin = __start___nv_relfatbin;
  const char *end = __stop___nv_relfatbin;
  if (!begin || !end || end == begin) {
    fprintf(stderr, "ERROR: cudaq_device_code_of_this_binary: this binary "
                    "carries no relocatable device code; build it with -rdc\n");
    return CUDAQ_ERR_UNSUPPORTED;
  }
  out_image->data = begin;
  out_image->size = (size_t)(end - begin);
  return CUDAQ_OK;
}

#ifdef __cplusplus
}
#endif
