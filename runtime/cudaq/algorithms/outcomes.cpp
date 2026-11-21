/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Prototype outcome transport and histogram storage.
#include "cudaq/algorithms/sample_vnext.h"
#include <memory>

namespace {
struct HistogramStorage {
  cudaq::CountsDictionary counts;
  std::string readout;
};
struct LaunchStorage {
  cudaq::outcome_stream records;
  std::vector<std::unique_ptr<HistogramStorage>> histograms;
};
thread_local std::unique_ptr<LaunchStorage> active;
LaunchStorage &storage() {
  if (!active)
    throw std::logic_error("outcome operation requires cudaq::launch");
  return *active;
}
} // namespace
namespace cudaq::detail {
void begin_outcomes() {
  if (active)
    throw std::logic_error("nested cudaq::launch is unsupported");
  active = std::make_unique<LaunchStorage>();
}
outcome_stream take_outcomes() {
  auto records = std::move(storage().records);
  active.reset();
  return records;
}
void discard_outcomes() noexcept { active.reset(); }
void publish_histogram(sample_result histogram) {
  storage().records.emplace_back(std::move(histogram));
}
} // namespace cudaq::detail
extern "C" {
void *__cudaq_histogram_create(std::int64_t shots) {
  if (shots < 0)
    throw std::invalid_argument(
        "frequencies shot count exceeds the supported range");
  auto &histograms = storage().histograms;
  histograms.push_back(std::make_unique<HistogramStorage>());
  return histograms.back().get();
}
void __cudaq_histogram_assign(void *handle, void *sampled) {
  std::unique_ptr<cudaq::ExecutionResult> result(
      static_cast<cudaq::ExecutionResult *>(sampled));
  static_cast<HistogramStorage *>(handle)->counts = std::move(result->counts);
}
void __cudaq_histogram_bit(void *handle, bool bit) {
  static_cast<HistogramStorage *>(handle)->readout += bit ? '1' : '0';
}
void __cudaq_histogram_count(void *handle) {
  auto &histogram = *static_cast<HistogramStorage *>(handle);
  ++histogram.counts[histogram.readout];
  histogram.readout.clear();
}
void __cudaq_emit_histogram(void *handle) {
  const auto &histogram = *static_cast<HistogramStorage *>(handle);
  storage().records.emplace_back(
      cudaq::sample_result{cudaq::ExecutionResult{histogram.counts}});
}
void __cudaq_emit_integer(std::int64_t value) {
  storage().records.emplace_back(value);
}
}
