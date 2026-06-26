/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "KernelExecution.h"
#include "SampleResult.h"
#include "cudaq_json.h"
#include <string>
#include <vector>

namespace cudaq {

/// @brief One measured result's place in the user-visible output. resultIndex
/// is the compact provider result position, deviceQubit is the mapped device
/// qubit position, the name is the register it belongs to, and outputPosition
/// is its user-visible output order.
struct ResultOutputEntry {
  std::size_t resultIndex = 0;
  DeviceQubit deviceQubit = 0;
  std::string outputName;
  std::size_t outputPosition = 0;
};

/// @brief The single result representation consumed by reconstruction: the
/// per-result map read from enriched output_names. Replaces the typed execution
/// result layout.
struct ResultOutputMap {
  std::vector<ResultOutputEntry> outputs;
};

/// @brief Build the result map from enriched output_names metadata. Each
/// output-location tuple is [qubitNum, registerName, outputPosition];
/// resultIndex is read from the outer result id, deviceQubit is qubitNum, the
/// name is the register name, and the position is read from the third tuple
/// element. An old compiler that omits the third element falls back to dense
/// result-index order, matching the non-mapped reference order.
ResultOutputMap
makeResultOutputMapFromEnrichedOutputNames(const nlohmann::json &outputNames);
ResultOutputMap
makeResultOutputMapFromEnrichedOutputNames(const cudaq_json &outputNames);

/// Reconstruct a sample_result from per-shot flat bitstrings whose bits are
/// indexed by compact provider result index. Preserves per-shot sequential data
/// in the returned result.
sample_result reconstructSampleResultFromResultIndexedBitstringShots(
    const std::vector<std::string> &shots, const ResultOutputMap &resultMap);

/// Reconstruct a sample_result from per-shot flat bitstrings whose bits are
/// indexed by mapped device qubit. Preserves per-shot sequential data in the
/// returned result. Throws std::invalid_argument when a bitstring is shorter
/// than a mapped device-qubit index.
sample_result reconstructSampleResultFromDeviceIndexedBitstringShots(
    const std::vector<std::string> &shots, const ResultOutputMap &resultMap);

/// Reconstruct a sample_result from a counts dictionary whose bitstrings are
/// indexed by compact provider result index.
sample_result reconstructSampleResultFromResultIndexedMeasurements(
    const CountsDictionary &counts, const ResultOutputMap &resultMap);

/// Reconstruct a sample_result from a counts dictionary whose bitstrings are
/// indexed by mapped device qubit.
sample_result reconstructSampleResultFromDeviceIndexedMeasurements(
    const CountsDictionary &counts, const ResultOutputMap &resultMap);

} // namespace cudaq
