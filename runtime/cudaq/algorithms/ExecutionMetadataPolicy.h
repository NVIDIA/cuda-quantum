/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/KernelExecution.h"
#include "common/ResultReconstruction.h"

namespace cudaq {

/// @brief Shared base carrying the result-to-output map and the metadata setter
/// shared by all execution policies (sample, observe, ...). Local result
/// reconstruction reads this map, which carries the result data from the
/// enriched output_names channel.
struct ExecutionMetadataPolicy {
  /// @brief Result-to-output map for local execution policy handoff. It carries
  /// the per-result (bit index, output name, output position) read from the
  /// enriched output_names, and is populated only for mapped executions where
  /// local reconstruction must undo the placement shuffle.
  mutable ResultOutputMap resultOutputMap;

  /// @brief Active device qubits for interpreting local simulator bitstrings
  /// whose bits are ordered by local mapped slots.
  mutable ActiveDeviceQubits activeDeviceQubits;

  /// @brief Derive the result-to-output map for local reconstruction. Only
  /// mapped executions need it; non-mapped executions return their raw
  /// simulator order. The mapped signal is the active device qubit sidecar; the
  /// result data comes from the enriched output_names.
  void setKernelExecutionMetadata(const KernelExecution &execution) {
    if (execution.activeDeviceQubits.empty()) {
      resultOutputMap = {};
      activeDeviceQubits = {};
      return;
    }
    activeDeviceQubits = execution.activeDeviceQubits;
    resultOutputMap =
        makeResultOutputMapFromEnrichedOutputNames(execution.output_names);
  }
};

} // namespace cudaq
