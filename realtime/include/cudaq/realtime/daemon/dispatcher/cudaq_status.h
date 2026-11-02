/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file cudaq_status.h
/// @brief Status codes of the realtime C API.

// Error codes
typedef enum {
  CUDAQ_OK = 0,
  CUDAQ_ERR_INVALID_ARG = 1,
  CUDAQ_ERR_INTERNAL = 2,
  CUDAQ_ERR_CUDA = 3,
  // The component does not implement the requested optional capability
  // (e.g. a version-1 bridge provider asked for a version-2 interface field).
  CUDAQ_ERR_UNSUPPORTED = 4
} cudaq_status_t;
