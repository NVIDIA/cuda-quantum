/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file no_device_code.cpp
/// @brief A shared library with no device code, asking for its own.

#include "cudaq/realtime/device_code/device_code_reader.h"

extern "C" cudaq_status_t
gpu_dataplane_test_no_device_code(cudaq_device_image_t *image) {
  return cudaq_device_code_of_this_binary(image);
}
