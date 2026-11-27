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
#include "cudaq/realtime/validation/playback_provider_interface.h"

cudaq_status_t cudaq_realtime_c_api_check(cudaq_dispatcher_t *dispatcher,
                                          void **out_fn) {
  return cudaq_dispatcher_get_device_handler(dispatcher, "handler", out_fn);
}

// Check the playback status width and exported getter signature as C.
_Static_assert(sizeof(cudaq_playback_provider_status_t) == 4,
               "fixed playback status ABI");
cudaq_get_playback_provider_interface_fn_t
playback_provider_getter_type_check(void) {
  return &cudaq_get_playback_provider_interface;
}
