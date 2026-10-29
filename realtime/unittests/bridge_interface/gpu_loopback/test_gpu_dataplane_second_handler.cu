/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file test_gpu_dataplane_second_handler.cu
/// @brief A second handler in its own translation unit, so test_gpu_dataplane
/// carries two `fatbin`s, and a lookup made from this translation unit.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/device_code/device_code.h"
#include <cstddef>

namespace {

__device__ int gpu_dataplane_test_second_handler(const void *rx_slot,
                                                 void *tx_slot,
                                                 std::size_t slot_size) {
  return 0;
}

} // namespace

CUDAQ_REALTIME_EXPORT_DEVICE_FUNCTION(gpu_dataplane_test_second_handler)

extern "C" cudaq_status_t
gpu_dataplane_test_lookup_second_handler(cudaq_dispatcher_t *dispatcher,
                                         void **out_fn) {
  return cudaq_dispatcher_get_device_handler(
      dispatcher, "gpu_dataplane_test_second_handler", out_fn);
}
