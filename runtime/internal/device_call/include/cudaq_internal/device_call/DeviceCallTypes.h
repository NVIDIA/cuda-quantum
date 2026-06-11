/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime_api.h>

namespace cudaq_internal::device_call {

// Optional service-provided synchronizer used when the dispatch loop is owned
// by the service artifact rather than by a CUDA-Q-created stream.
using DeviceCallDispatchSynchronizeFn = cudaError_t (*)();

} // namespace cudaq_internal::device_call
