/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file nvtx.hpp
/// @brief NVTX/Nsight profiler backend implementation
///
/// This file should only be included via profiler.hpp when
/// PROFILER_BACKEND_NVTX is defined. It implements the unified profiling API
/// using NVIDIA NVTX3.

#include <cstdint>
#include <nvtx3/nvToolsExt.h>

// Note: profiler constants are defined in profiler.hpp
// This file assumes profiler.hpp has already been included

namespace cudaq::nvqlink::nvtx {

// NVTX domain handles (defined in nvtx.cpp)
extern nvtxDomainHandle_t domain_daemon;
extern nvtxDomainHandle_t domain_dispatcher;
extern nvtxDomainHandle_t domain_memory;
extern nvtxDomainHandle_t domain_channel;
extern nvtxDomainHandle_t domain_user;
extern nvtxDomainHandle_t domain_gpu;

/// Minimal overhead RAII range (no virtual, inlined destructor)
class ScopedRange {
  nvtxDomainHandle_t d_;

public:
  inline ScopedRange(nvtxDomainHandle_t d, const char *name, uint32_t cat,
                     uint32_t color)
      : d_(d) {
    nvtxEventAttributes_t a = {0};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.category = cat;
    a.colorType = NVTX_COLOR_ARGB;
    a.color = color;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = name;
    nvtxDomainRangePushEx(d_, &a);
  }

  inline ScopedRange(nvtxDomainHandle_t d, const char *name, uint32_t cat,
                     uint32_t color, uint64_t payload)
      : d_(d) {
    nvtxEventAttributes_t a = {0};
    a.version = NVTX_VERSION;
    a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    a.category = cat;
    a.colorType = NVTX_COLOR_ARGB;
    a.color = color;
    a.messageType = NVTX_MESSAGE_TYPE_ASCII;
    a.message.ascii = name;
    a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
    a.payload.ullValue = payload;
    nvtxDomainRangePushEx(d_, &a);
  }

  inline ~ScopedRange() {
    if (d_)
      nvtxDomainRangePop(d_);
  }
};

inline void mark(nvtxDomainHandle_t d, const char *msg) {
  nvtxEventAttributes_t a = {0};
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  a.messageType = NVTX_MESSAGE_TYPE_ASCII;
  a.message.ascii = msg;
  a.colorType = NVTX_COLOR_ARGB;
  a.color = profiler::COLOR_ERROR;
  nvtxDomainMarkEx(d, &a);
}

inline void counter(nvtxDomainHandle_t d, const char *name, uint64_t value) {
  nvtxEventAttributes_t a = {0};
  a.version = NVTX_VERSION;
  a.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  a.messageType = NVTX_MESSAGE_TYPE_ASCII;
  a.message.ascii = name;
  a.payloadType = NVTX_PAYLOAD_TYPE_UNSIGNED_INT64;
  a.payload.ullValue = value;
  a.colorType = NVTX_COLOR_ARGB;
  a.color = profiler::COLOR_METRICS;
  nvtxDomainMarkEx(d, &a);
}
} // namespace cudaq::nvqlink::nvtx

// Profiler namespace implementation (forward declared in nvtx.cpp)
namespace cudaq::nvqlink::profiler {
void initialize();
void shutdown();
} // namespace cudaq::nvqlink::profiler

//===----------------------------------------------------------------------===//
// Preprocessor-based domain resolution
//===----------------------------------------------------------------------===//

/// Helper for token pasting to directly map domain identifiers to handles at
/// compile-time.
#define _NVQLINK_NVTX_DOMAIN(domain) nvqlink::nvtx::domain_##domain

//===----------------------------------------------------------------------===//
// Unified API Macros (NVTX Implementation)
//===----------------------------------------------------------------------===//

#define NVQLINK_TRACE_SCOPE(domain, name)                                      \
  nvqlink::nvtx::ScopedRange _nvtx_s##__LINE__(                                \
      _NVQLINK_NVTX_DOMAIN(domain), name, nvqlink::profiler::CAT_FULL,         \
      nvqlink::profiler::COLOR_FULL)

#define NVQLINK_TRACE_SCOPE_COLOR(domain, name, color)                         \
  nvqlink::nvtx::ScopedRange _nvtx_sc##__LINE__(                               \
      _NVQLINK_NVTX_DOMAIN(domain), name, nvqlink::profiler::CAT_FULL, color)

#define NVQLINK_TRACE_HOTPATH(domain, name)                                    \
  nvqlink::nvtx::ScopedRange _nvtx_h##__LINE__(                                \
      _NVQLINK_NVTX_DOMAIN(domain), name, nvqlink::profiler::CAT_HOT_PATH,     \
      nvqlink::profiler::COLOR_HOTPATH)

#define NVQLINK_TRACE_HOTPATH_PAYLOAD(domain, name, payload)                   \
  nvqlink::nvtx::ScopedRange _nvtx_hp##__LINE__(                               \
      _NVQLINK_NVTX_DOMAIN(domain), name, nvqlink::profiler::CAT_HOT_PATH,     \
      nvqlink::profiler::COLOR_HOTPATH, payload)

#define NVQLINK_TRACE_FULL(domain, name)                                       \
  nvqlink::nvtx::ScopedRange _nvtx_f##__LINE__(                                \
      _NVQLINK_NVTX_DOMAIN(domain), name, nvqlink::profiler::CAT_FULL,         \
      nvqlink::profiler::COLOR_FULL)

#define NVQLINK_TRACE_MEMORY(name)                                             \
  nvqlink::nvtx::ScopedRange _nvtx_m##__LINE__(                                \
      nvqlink::nvtx::domain_memory, name, nvqlink::profiler::CAT_HOT_PATH,     \
      nvqlink::profiler::COLOR_MEMORY)

#define NVQLINK_TRACE_USER_RANGE(name)                                         \
  nvqlink::nvtx::ScopedRange _nvtx_u##__LINE__(                                \
      nvqlink::nvtx::domain_user, name, nvqlink::profiler::CAT_HOT_PATH,       \
      nvqlink::profiler::COLOR_USER)

#define NVQLINK_TRACE_COUNTER(name, value)                                     \
  nvqlink::nvtx::counter(nvqlink::nvtx::domain_memory, name, value)

#define NVQLINK_TRACE_MARK(domain, msg)                                        \
  nvqlink::nvtx::mark(_NVQLINK_NVTX_DOMAIN(domain), msg)

#define NVQLINK_TRACE_MARK_ERROR(domain, msg)                                  \
  nvqlink::nvtx::mark(_NVQLINK_NVTX_DOMAIN(domain), msg)

#define NVQLINK_TRACE_NAME_THREAD(name) nvtxNameOsThreadA(pthread_self(), name)

//===----------------------------------------------------------------------===//
// Tracy-specific features (no-op for NVTX)
//===----------------------------------------------------------------------===//

#define NVQLINK_ALLOC(ptr, size)
#define NVQLINK_FREE(ptr)
#define NVQLINK_TRACE_FRAME_MARK
