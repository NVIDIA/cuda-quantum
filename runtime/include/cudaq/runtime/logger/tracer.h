/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace cudaq {

/// @brief Context information (function, file, and line) of a trace caller.
struct TraceContext {
  const char *funcName = nullptr;
  const char *fileName = nullptr;
  int lineNo = 0;

  TraceContext(const char *func = __builtin_FUNCTION(),
               const char *file = __builtin_FILE(), int line = __builtin_LINE())
      : funcName(func), fileName(file), lineNo(line) {}
};

struct TraceEvent {
  TraceContext ctx;
  std::string_view name;
  std::string_view category;
  int tag;
  std::string_view args;
  uint64_t tsUs;
  uint32_t tid;
  int depth;
};

// Backends must be constructed via std::make_shared so that C++ and Python
// can each hold independent shared_ptr references. The Python binding's
// nanobind::init factories enforce this.
class TraceBackend : public std::enable_shared_from_this<TraceBackend> {
public:
  virtual ~TraceBackend() = default;
  virtual void onBegin(const TraceEvent &e) = 0;
  virtual void onEnd(const TraceEvent &e, uint64_t durUs) = 0;
};

// Owns its string state so it is safe against dangling caller memory between
// beginSpan and endSpan. Move-only to avoid accidental copies.
struct SpanHandle {
  SpanHandle() = default;
  SpanHandle(const SpanHandle &) = delete;
  SpanHandle &operator=(const SpanHandle &) = delete;
  SpanHandle(SpanHandle &&) = default;
  SpanHandle &operator=(SpanHandle &&) = default;

  TraceContext ctx{};
  std::string name;
  std::string category;
  std::string args;
  uint64_t beginTsUs = 0;
  uint32_t tid = 0;
  int depth = 0;
  int tag = 0;
  bool active = false;
};

class Tracer {
public:
  static Tracer &instance();

  // Callers must quiesce span activity before swapping the backend.
  // In-flight beginSpan / endSpan calls dereference the raw backend
  // pointer without synchronization.
  void setBackend(std::shared_ptr<TraceBackend> backend);

  // Returns the currently installed backend, or nullptr if none.
  std::shared_ptr<TraceBackend> getBackend() const;

  // When enabled, beginSpan admits spans regardless of log level or tags.
  void setCaptureEnabled(bool enabled);
  bool isCaptureEnabled() const;

  SpanHandle beginSpan(const TraceContext &ctx, std::string_view name,
                       int tag = 0, std::string_view args = {},
                       std::string_view category = "scope");

  void endSpan(SpanHandle handle);

private:
  Tracer();
  ~Tracer();

  Tracer(const Tracer &) = delete;
  Tracer &operator=(const Tracer &) = delete;
};

// Called from initializeLogger in logger.cpp to apply CUDAQ_TRACE_FORMAT /
// CUDAQ_TRACE_PATH at library-load time. Exits the process with a diagnostic
// on invalid combinations.
void configureTracerFromEnv();

} // namespace cudaq
