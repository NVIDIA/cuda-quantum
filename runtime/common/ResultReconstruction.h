/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "SampleResult.h"
#include "nlohmann/json_fwd.hpp"
#include <string>
#include <vector>

namespace cudaq {

/// Describes how one compact QIR result moves into a user-visible default
/// `sample` result.
struct ResultOutputEntry {
  /// Compact QIR result index. QIR-style backend bitstrings place this result
  /// at this source position, which commonly follows `mz` emission order.
  std::size_t resultIndex = 0;

  /// Name of the measurement register that receives this bit.
  std::string outputName;

  /// Dense destination position in the global result bitstring. For default
  /// `sample`, the compiler assigns this by measured-qubit allocation order,
  /// independently of the order in which `mz` operations execute.
  std::size_t outputPosition = 0;
};

/// Complete reconstruction map for the measured results of one kernel.
struct ResultOutputMap {
  std::vector<ResultOutputEntry> outputs;
};

/// Build a reconstruction map from QIR `output_names` metadata encoded as
/// `[[[resultIndex, [deviceQubit, outputName, outputPosition]], ...]]`.
///
/// The physical `deviceQubit` field remains part of the QIR schema, but compact
/// QIR responses are indexed by `resultIndex`. `outputPosition` is the
/// enriched field that separates this source coordinate from the
/// allocation-ordered destination coordinate. Legacy two-field output
/// locations omit it and fall back to dense metadata order.
ResultOutputMap
makeResultOutputMapFromEnrichedOutputNames(const nlohmann::json &outputNames);

/// Reconstruct aggregated QIR-style counts using compact result indices as
/// source positions and allocation order as the global destination order.
/// When provided, `sequentialData` contains per-shot bitstrings in the same
/// compact result order and is reconstructed for the global and named results.
sample_result reconstructSampleResultFromResultIndexedMeasurements(
    const CountsDictionary &counts, const ResultOutputMap &resultMap,
    const std::vector<std::string> &sequentialData = {});

} // namespace cudaq
