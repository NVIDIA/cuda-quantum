/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file c_api_check.c
/// @brief Compiled as C, never run: the public headers must stay C-clean.

#include "cudaq/realtime/daemon/bridge/bridge_interface.h"
#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

cudaq_status_t cudaq_realtime_c_api_check(cudaq_dispatcher_t *dispatcher,
                                          void **out_fn) {
  return cudaq_dispatcher_get_device_handler(dispatcher, "handler", out_fn);
}
