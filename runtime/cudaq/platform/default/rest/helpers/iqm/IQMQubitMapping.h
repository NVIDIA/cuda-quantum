/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/KernelExecution.h"
#include "nlohmann/json.hpp"
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace cudaq {

/// @brief Natural-order comparison for qubit name strings ending in a number.
/// Assumes all strings share a prefix and end in a number. No checks on the
/// string composition are done for performance reasons.
struct IqmQubitOrder {
  bool operator()(const std::string &a, const std::string &b) const {
    if (a.size() < b.size())
      return true;
    if (a.size() > b.size())
      return false;
    return a.compare(b) < 0;
  }
};

/// @brief Dense IQM device-qubit view derived from a dynamic quantum
/// architecture. `qubitNameMap` maps each surviving provider qubit name to its
/// dense device-qubit index, `qubitAdjacencyMap` holds the `cz` adjacency over
/// dense indices, and `backendLabels` is the dense-index-ordered provider name
/// table that feeds the ServerHelper base label table.
struct IqmArchitectureMapping {
  std::map<std::string, uint, IqmQubitOrder> qubitNameMap;
  std::vector<std::set<uint>> qubitAdjacencyMap;
  std::vector<std::string> backendLabels;
};

/// @brief Build the dense device view from a parsed dynamic quantum
/// architecture JSON. A qubit is kept as a node when it can do single-qubit
/// rotations and readout, that is `prx` and `measure`. `cz` is required only to
/// participate in an edge, not to exist as a node, so a
/// `prx`-and-`measure`-only qubit survives as an isolated, edge-free node for
/// single-qubit tune-up. Adjacency is built from `cz_loci`, with an edge added
/// only when both endpoints survived the node filter.
IqmArchitectureMapping
buildIqmArchitectureMapping(const nlohmann::json &dynamicQuantumArchitecture);

/// @brief Build the IQM circuits-API `qubit_mapping` array for one emitted
/// execution. When `targetQubitMapping` is non-empty, emit a partial mapping
/// that resolves each target-code logical name and device qubit through
/// `backendLabel`, falling back to the dense IQM name when no dynamic label
/// table exists. Otherwise emit the full mapping derived from the dynamic
/// quantum architecture in `qubitNameMap`.
nlohmann::json buildIqmQubitMapping(
    const TargetQubitMapping &targetQubitMapping,
    const std::map<std::string, uint, IqmQubitOrder> &qubitNameMap,
    const std::function<std::optional<std::string>(DeviceQubit)> &backendLabel);

} // namespace cudaq
