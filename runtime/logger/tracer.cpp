/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/runtime/logger/tracer.h"
#include "cudaq/runtime/logger/chrome_tracer.h"
#include "cudaq/runtime/logger/logger.h"
#include "cudaq/runtime/logger/spdlog_tracer.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <pthread.h>
#include <string>
#if defined(__linux__)
#include <sys/syscall.h>
#endif
#include <unistd.h>
#include <utility>

namespace cudaq {

namespace {

// Per-thread nesting depth used by the spdlog backend for indentation.
thread_local int tracerDepth = -1;

// Single active backend. Not mutex-guarded. Callers must quiesce span
// activity before swapping the backend.
std::shared_ptr<TraceBackend> &backendSlot() {
  static std::shared_ptr<TraceBackend> b;
  return b;
}

std::atomic<bool> programmaticCapture{false};

uint64_t nowMicroseconds() {
  auto now = std::chrono::system_clock::now().time_since_epoch();
  return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(now).count());
}

uint32_t currentThreadId() {
  // Cache per thread. gettid is a kernel round trip and we call this on
  // every admitted span.
  thread_local uint32_t cached = 0;
  thread_local bool cacheInitialized = false;
  if (!cacheInitialized) {
#if defined(__linux__)
    cached = static_cast<uint32_t>(syscall(SYS_gettid));
#elif defined(__APPLE__)
    uint64_t tid = 0;
    pthread_threadid_np(nullptr, &tid);
    cached = static_cast<uint32_t>(tid);
#else
    cached = static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pthread_self()));
#endif
    cacheInitialized = true;
  }
  return cached;
}

} // namespace

Tracer::Tracer() = default;

Tracer::~Tracer() = default;

Tracer &Tracer::instance() {
  static Tracer tracer;
  return tracer;
}

void Tracer::setBackend(std::shared_ptr<TraceBackend> backend) {
  backendSlot() = std::move(backend);
}

std::shared_ptr<TraceBackend> Tracer::getBackend() const {
  return backendSlot();
}

void Tracer::setCaptureEnabled(bool enabled) {
  programmaticCapture.store(enabled, std::memory_order_relaxed);
}

bool Tracer::isCaptureEnabled() const {
  return programmaticCapture.load(std::memory_order_relaxed);
}

SpanHandle Tracer::beginSpan(const TraceContext &ctx, std::string_view name,
                             int tag, std::string_view args,
                             std::string_view category) {
  const bool tagFound = (tag != 0) && cudaq::isTimingTagEnabled(tag);
  const bool traceOn =
      cudaq::details::should_log(cudaq::details::LogLevel::trace);
  const bool progCapture = programmaticCapture.load(std::memory_order_relaxed);
  if (!(tagFound || traceOn || progCapture))
    return {};

  auto *backend = backendSlot().get();
  if (!backend)
    return {};

  SpanHandle handle;
  handle.ctx = ctx;
  handle.name.assign(name);
  handle.category.assign(category);
  handle.args.assign(args);
  handle.tag = tag;
  handle.tid = currentThreadId();
  handle.beginTsUs = nowMicroseconds();
  ++tracerDepth;
  handle.depth = tracerDepth;
  handle.active = true;

  TraceEvent e{handle.ctx,  handle.name,      handle.category, handle.tag,
               handle.args, handle.beginTsUs, handle.tid,      handle.depth};
  backend->onBegin(e);
  return handle;
}

void Tracer::endSpan(SpanHandle handle) {
  if (!handle.active)
    return;

  auto *backend = backendSlot().get();
  if (!backend) {
    --tracerDepth;
    return;
  }

  const uint64_t endTs = nowMicroseconds();
  const uint64_t durUs =
      endTs > handle.beginTsUs ? endTs - handle.beginTsUs : 0;

  TraceEvent e{handle.ctx,  handle.name,      handle.category, handle.tag,
               handle.args, handle.beginTsUs, handle.tid,      handle.depth};
  backend->onEnd(e, durUs);

  --tracerDepth;
}

void configureTracerFromEnv() {
  const char *format = std::getenv("CUDAQ_TRACE_FORMAT");
  if (!format || format[0] == '\0' || std::strcmp(format, "spdlog") == 0) {
    Tracer::instance().setBackend(std::make_shared<SpdlogTraceBackend>());
    return;
  }

  if (std::strcmp(format, "chrome") == 0) {
    const char *path = std::getenv("CUDAQ_TRACE_PATH");
    if (!path || path[0] == '\0') {
      std::fprintf(
          stderr,
          "CUDAQ_TRACE_FORMAT=chrome requires CUDAQ_TRACE_PATH to be set\n");
      std::exit(1);
    }
    Tracer::instance().setBackend(std::make_shared<ChromeTraceBackend>(path));
    return;
  }

  std::fprintf(stderr,
               "CUDAQ_TRACE_FORMAT=%s is not recognized "
               "(expected 'spdlog' or 'chrome')\n",
               format);
  std::exit(1);
}

} // namespace cudaq
