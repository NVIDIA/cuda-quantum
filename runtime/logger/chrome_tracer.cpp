/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/runtime/logger/chrome_tracer.h"
#include "cudaq/runtime/logger/logger.h"

#include <nlohmann/json.hpp>

#include <fstream>
#include <unistd.h>
#include <utility>

namespace cudaq {

ChromeTraceBackend::ChromeTraceBackend(std::string path)
    : outputPath(std::move(path)), ownerPid(getpid()) {}

ChromeTraceBackend::~ChromeTraceBackend() {
  if (outputPath.empty())
    return;
  // After fork(), the child inherits a shared_ptr to this backend but must
  // not run our destructor's file write. Skip if our pid changed.
  if (getpid() != ownerPid)
    return;
  writeFile();
}

void ChromeTraceBackend::onBegin(const TraceEvent &) {}

void ChromeTraceBackend::onEnd(const TraceEvent &e, uint64_t durUs) {
  Event ev;
  ev.name.assign(e.name);
  ev.category.assign(e.category);
  ev.args.assign(e.args);
  ev.tsUs = e.tsUs;
  ev.durUs = durUs;
  ev.tid = e.tid;
  std::lock_guard<std::mutex> lock(mu);
  events.push_back(std::move(ev));
}

std::string ChromeTraceBackend::toJson() {
  std::vector<Event> snapshot;
  {
    std::lock_guard<std::mutex> lock(mu);
    snapshot = events;
  }

  const int pid = static_cast<int>(getpid());
  nlohmann::json traceEvents = nlohmann::json::array();
  for (const auto &e : snapshot) {
    nlohmann::json event = {
        {"name", e.name}, {"cat", e.category}, {"ph", "X"},    {"ts", e.tsUs},
        {"dur", e.durUs}, {"pid", pid},        {"tid", e.tid},
    };
    if (!e.args.empty())
      event["args"] = {{"detail", e.args}};
    traceEvents.push_back(std::move(event));
  }

  nlohmann::json doc = {
      {"displayTimeUnit", "ms"},
      {"traceEvents", std::move(traceEvents)},
  };
  return doc.dump();
}

void ChromeTraceBackend::writeFile(std::optional<std::string> path) {
  const std::string &target = path && !path->empty() ? *path : outputPath;
  if (target.empty()) {
    CUDAQ_WARN("Chrome trace backend has no output path; nothing written.");
    return;
  }

  const std::string json = toJson();
  std::ofstream os(target, std::ios::trunc);
  if (!os.is_open()) {
    CUDAQ_WARN("Chrome trace backend failed to open {} for writing.", target);
    return;
  }
  os << json;
}

void ChromeTraceBackend::clear() {
  std::lock_guard<std::mutex> lock(mu);
  events.clear();
}

} // namespace cudaq
