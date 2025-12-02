/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/dispatcher/cpu_dispatcher.h"
#ifdef __CUDACC__
#include "cudaq/nvqlink/daemon/dispatcher/gpu_dispatcher.h"
#endif

#include <stdexcept>

using namespace cudaq::nvqlink;

std::unique_ptr<Dispatcher>
cudaq::nvqlink::create_dispatcher(DatapathMode mode, Channel *channel,
                           FunctionRegistry *registry,
                           const ComputeConfig &compute_config) {
  switch (mode) {
  case DatapathMode::CPU:
    return std::make_unique<CPUDispatcher>(channel, registry, compute_config);
  case DatapathMode::GPU:
#ifdef __CUDACC__
    return std::make_unique<GPUDispatcher>(channel, registry, compute_config);
#else
    throw std::runtime_error("GPU mode requested but CUDA is not available");
#endif
  default:
    throw std::invalid_argument("Unknown datapath mode");
  }
}
