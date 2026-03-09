/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatUtils.h"
#include "cudaq.h"
#include <mutex>

namespace {
static std::unordered_map<std::string, cudaq::dynamics::PerfMetric>
    g_perfMetric;
static std::mutex g_metricMutex;
} // namespace

cudaq::dynamics::PerfMetricScopeTimer::PerfMetricScopeTimer(
    const std::string &name)
    : m_name(name), m_startTime(std::chrono::system_clock::now()) {}

cudaq::dynamics::PerfMetricScopeTimer::~PerfMetricScopeTimer() {
  if (!cudaq::details::should_log(cudaq::details::LogLevel::trace))
    return;

  auto duration =
      static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(
                              std::chrono::system_clock::now() - m_startTime)
                              .count() /
                          1e3);

  {
    std::scoped_lock lock(g_metricMutex);
    g_perfMetric[m_name].add(duration);
  }
}

void cudaq::dynamics::PerfMetric::add(double duration) {
  numCalls++;
  totalTimeMs = totalTimeMs + duration;
}

void cudaq::dynamics::dumpPerfTrace(std::ostream &os) {
  std::vector<std::pair<std::string, cudaq::dynamics::PerfMetric>> metrics(
      g_perfMetric.begin(), g_perfMetric.end());
  std::sort(
      metrics.begin(), metrics.end(),
      [](const std::pair<std::string, cudaq::dynamics::PerfMetric> &metricPair1,
         const std::pair<std::string, cudaq::dynamics::PerfMetric>
             &metricPair2) {
        return metricPair1.second.totalTimeMs > metricPair2.second.totalTimeMs;
      });

  if (!cudaq::mpi::is_initialized() || cudaq::mpi::rank() == 0) {
    for (const auto &[name, metric] : metrics) {
      os << name << ": number of calls = " << metric.numCalls
         << ", total time = " << metric.totalTimeMs << " [ms]. Time per call = "
         << metric.totalTimeMs / static_cast<double>(metric.numCalls)
         << "[ms].\n";
    }
  }
  g_perfMetric.clear();
}
