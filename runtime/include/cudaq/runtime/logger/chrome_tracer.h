/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/runtime/logger/tracer.h"
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <sys/types.h>
#include <vector>

namespace cudaq {

// Captures Chrome Trace Event Format JSON in memory. If constructed with a
// non-empty path, the destructor writes that JSON to the path. Events
// accumulate in a member vector guarded by a mutex, so correctness does not
// depend on thread-local destruction ordering at process exit.
class ChromeTraceBackend : public TraceBackend {
public:
  explicit ChromeTraceBackend(std::string path = {});
  ~ChromeTraceBackend() override;

  void onBegin(const TraceEvent &e) override;
  void onEnd(const TraceEvent &e, uint64_t durUs) override;

  // Serialize the current event buffer as a Chrome Trace Event Format JSON
  // string. Non-destructive; safe to call mid-run.
  std::string toJson();

  // Write the current JSON to `path`, or to the ctor path if `path` is empty.
  // No-op if neither is provided. Non-destructive; the buffer stays intact.
  void writeFile(std::optional<std::string> path = {});

  // Drop all buffered events.
  void clear();

private:
  struct Event {
    std::string name;
    std::string category;
    std::string args;
    uint64_t tsUs;
    uint64_t durUs;
    uint32_t tid;
  };

  std::string outputPath;
  std::mutex mu;
  std::vector<Event> events;
  // pid at construction. If the backend's destructor runs in a forked
  // child, we skip writing so we don't clobber the parent's output at
  // the same path.
  pid_t ownerPid;
};

} // namespace cudaq
