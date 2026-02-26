/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file bridge_interface.h
/// @brief Interface Bindings for transport layer providers (e.g. Hololink).
///

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef __cplusplus
extern "C" {
#endif

///@brief Opaque data structure storing the details of the transport layer
/// connection
typedef void *cudaq_realtime_bridge_handle_t;

typedef enum {
  CUDAQ_PROVIDER_HOLOLINK = 0, /// Hololink GPU-RoCE transceiver
  CUDAQ_PROVIDER_EXTERNAL = 1, /// Externally managed transport

} cudaq_realtime_transport_provider_t;

// Bridge lifecycle
cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv);
cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge);

// Retrieve the ringbuffer information for the given bridge.
cudaq_status_t
cudaq_bridge_get_ringbuffer(cudaq_realtime_bridge_handle_t bridge,
                            cudaq_ringbuffer_t *out_ringbuffer);

cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge);
cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge);
cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge);

#define CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION 1

typedef struct {
  int version;
  cudaq_status_t (*create)(cudaq_realtime_bridge_handle_t *, int, char **);
  cudaq_status_t (*destroy)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*get_ringbuffer)(cudaq_realtime_bridge_handle_t,
                                   cudaq_ringbuffer_t *);
  cudaq_status_t (*connect)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*launch)(cudaq_realtime_bridge_handle_t);
  cudaq_status_t (*disconnect)(cudaq_realtime_bridge_handle_t);

} cudaq_realtime_bridge_interface_t;

#ifdef __cplusplus
}
#endif
