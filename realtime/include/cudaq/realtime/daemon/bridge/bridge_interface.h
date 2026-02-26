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
/// Different transport providers can be loaded at runtime via dlopen, allowing
/// for dynamic selection and initialization of the desired transport layer.
/// Environment variable CUDAQ_REALTIME_BRIDGE_LIB must be set to the path of
/// the shared library implementing the desired transport provider (if not using
/// the built-in Hololink provider).

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#ifdef __cplusplus
extern "C" {
#endif

///@brief Opaque data structure storing the details of the transport layer
/// connection
typedef void *cudaq_realtime_bridge_handle_t;

typedef enum {
  CUDAQ_PROVIDER_HOLOLINK =
      0, /// Hololink GPU-RoCE transceiver (built-in provider)
  CUDAQ_PROVIDER_EXTERNAL = 1, /// Externally managed transport

} cudaq_realtime_transport_provider_t;

/// @brief Create and initialize a transport bridge for the specified provider.
/// For the built-in Hololink provider, this loads the Hololink shared library
/// and initializes the transceiver with the provided args.  For the EXTERNAL
/// provider, this loads the shared library specified by the
/// CUDAQ_REALTIME_BRIDGE_LIB environment variable and calls its create callback
/// to initialize the bridge.
cudaq_status_t
cudaq_bridge_create(cudaq_realtime_bridge_handle_t *out_bridge_handle,
                    cudaq_realtime_transport_provider_t provider, int argc,
                    char **argv);

/// @brief Destroy the transport bridge and release all associated resources.
cudaq_status_t cudaq_bridge_destroy(cudaq_realtime_bridge_handle_t bridge);

/// @brief Retrieve the ringbuffer information for the given bridge.
cudaq_status_t
cudaq_bridge_get_ringbuffer(cudaq_realtime_bridge_handle_t bridge,
                            cudaq_ringbuffer_t *out_ringbuffer);

/// @brief Connect the transport bridge.
cudaq_status_t cudaq_bridge_connect(cudaq_realtime_bridge_handle_t bridge);

/// @brief Launch the transport bridge's main processing loop (e.g. start
/// Hololink kernels).
cudaq_status_t cudaq_bridge_launch(cudaq_realtime_bridge_handle_t bridge);

/// @brief Disconnect the transport bridge (e.g. stop Hololink kernels and
/// disconnect).
cudaq_status_t cudaq_bridge_disconnect(cudaq_realtime_bridge_handle_t bridge);

#define CUDAQ_REALTIME_BRIDGE_INTERFACE_VERSION 1

/// @brief Interface struct for transport layer providers.  Each provider must
/// implement this interface and provide a getter function
/// (cudaq_realtime_get_bridge_interface) that returns a pointer to a statically
/// allocated instance of this struct with the function pointers set to the
/// provider's implementation.
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
