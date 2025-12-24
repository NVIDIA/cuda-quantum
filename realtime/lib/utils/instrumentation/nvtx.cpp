/*******************************************************************************
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/instrumentation/profiler.h"

#ifdef PROFILER_BACKEND_NVTX

namespace cudaq::nvqlink::nvtx {

// Define domain handles
nvtxDomainHandle_t domain_daemon = nullptr;
nvtxDomainHandle_t domain_dispatcher = nullptr;
nvtxDomainHandle_t domain_memory = nullptr;
nvtxDomainHandle_t domain_channel = nullptr;
nvtxDomainHandle_t domain_user = nullptr;
nvtxDomainHandle_t domain_gpu = nullptr;

void initialize() {
  domain_daemon = nvtxDomainCreateA("nvqlink::daemon");
  domain_dispatcher = nvtxDomainCreateA("nvqlink::dispatcher");
  domain_memory = nvtxDomainCreateA("nvqlink::memory");
  domain_channel = nvtxDomainCreateA("nvqlink::channel");
  domain_user = nvtxDomainCreateA("nvqlink::user");
  domain_gpu = nvtxDomainCreateA("nvqlink::gpu");
}

void shutdown() {
  if (domain_daemon)
    nvtxDomainDestroy(domain_daemon);
  if (domain_dispatcher)
    nvtxDomainDestroy(domain_dispatcher);
  if (domain_memory)
    nvtxDomainDestroy(domain_memory);
  if (domain_channel)
    nvtxDomainDestroy(domain_channel);
  if (domain_user)
    nvtxDomainDestroy(domain_user);
  if (domain_gpu)
    nvtxDomainDestroy(domain_gpu);
}

} // namespace cudaq::nvqlink::nvtx

namespace cudaq::nvqlink::profiler {

void initialize() { nvtx::initialize(); }

void shutdown() { nvtx::shutdown(); }

} // namespace cudaq::nvqlink::profiler

#endif // PROFILER_BACKEND_NVTX
