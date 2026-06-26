/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 * Copyright 2025 IQM Quantum Computers                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "IQMQubitMapping.h"
#include <stdexcept>

cudaq::IqmArchitectureMapping cudaq::buildIqmArchitectureMapping(
    const nlohmann::json &dynamicQuantumArchitecture) {
  IqmArchitectureMapping mapping;
  auto &qubitNameMap = mapping.qubitNameMap;
  auto &qubitAdjacencyMap = mapping.qubitAdjacencyMap;

  // From the Dynamic Quantum Architecture we need the list of qubit names, the
  // list of qubit pairs which can form cz-gates, the lists of qubits which can
  // do prx-gates and the list of qubits which support measurement.
  auto &cz = dynamicQuantumArchitecture["gates"]["cz"];
  auto implementation = cz["default_implementation"];
  auto &cz_loci = cz["implementations"][implementation]["loci"];

  auto &prx = dynamicQuantumArchitecture["gates"]["prx"];
  implementation = prx["default_implementation"];
  auto prx_loci = prx["implementations"][implementation]["loci"];

  auto &measure = dynamicQuantumArchitecture["gates"]["measure"];
  implementation = measure["default_implementation"];
  auto &measure_loci = measure["implementations"][implementation]["loci"];

  // For each qubit set flags to indicate whether they can be used in `cz`,
  // `prx` or `measurement` operations. A qubit is usable as a node when it can
  // do single-qubit rotations and readout, that is `prx` and `measure`. `cz` is
  // required only to participate in an edge, not to exist as a node, so a
  // `prx`-and-`measure`-only qubit survives as an isolated, edge-free node for
  // single-qubit tune-up. Crop all qubits without both `prx` and `measure` and
  // enumerate the remaining ones.
  constexpr uint czCapable = (1 << 0);
  constexpr uint prxCapable = (1 << 1);
  constexpr uint measureCapable = (1 << 2);
  constexpr uint nodeCapable = prxCapable | measureCapable;

  for (auto qubit : dynamicQuantumArchitecture["qubits"]) {
    qubitNameMap[qubit] = 0; // initializing to zero meaning no capability
  }
  for (auto cz : cz_loci) {
    // each cz loci has 2 qubits - mark each qubit
    for (auto qubit : cz) { // cz is an array of strings
      qubitNameMap[qubit] |= czCapable;
    }
  }
  for (auto prx : prx_loci) {
    qubitNameMap[prx[0]] |= prxCapable;
  }
  for (auto measure : measure_loci) {
    qubitNameMap[measure[0]] |= measureCapable;
  }

  uint idx = 0; // enumeration counter
  for (auto qubit = qubitNameMap.begin(); qubit != qubitNameMap.end();) {
    if ((qubit->second & nodeCapable) == nodeCapable) {
      qubit->second = idx++; // replace flags with enumeration value
      // Feed the dense-index-ordered backend label table: this dense device
      // qubit carries the provider-native name qubit->first.
      mapping.backendLabels.push_back(qubit->first);
      qubit++;
    } else {
      qubit = qubitNameMap.erase(qubit);
    }
  }
  // From here on the qubitNameMap lists only qubits which can do `prx` and
  // `measure`. Starting with 0 each qubit in the list is enumerated.

  uint qubitCount = qubitNameMap.size();

  // Initialise the adjacency map with an empty set for each qubit
  qubitAdjacencyMap.reserve(qubitCount);
  for (uint i = 0; i < qubitCount; i++) {
    qubitAdjacencyMap.emplace_back();
  }

  // Iterate over all cz loci and add only those to the adjacency map for which
  // all qubits have passed the above tests.
  for (auto cz : cz_loci) {
    if (qubitNameMap.count(cz[0]) && qubitNameMap.count(cz[1])) {
      qubitAdjacencyMap[qubitNameMap[cz[0]]].insert(qubitNameMap[cz[1]]);
      qubitAdjacencyMap[qubitNameMap[cz[1]]].insert(qubitNameMap[cz[0]]);
    }
  } // for all cz loci

  return mapping;
}

nlohmann::json cudaq::buildIqmQubitMapping(
    const TargetQubitMapping &targetQubitMapping,
    const std::map<std::string, uint, IqmQubitOrder> &qubitNameMap,
    const std::function<std::optional<std::string>(DeviceQubit)>
        &backendLabel) {
  nlohmann::json qubitMapping = nlohmann::json::array();

  if (!targetQubitMapping.empty()) {
    // The IQM circuits REST API accepts a partial qubit_mapping containing only
    // the logical-to-physical assignments for qubits that the circuit actually
    // uses. Unmapped physical qubits are ignored by the backend. This partial
    // mapping covers exactly the target logical names and device qubits
    // produced by the mapper. Resolve each device qubit to its backend label
    // through the ServerHelper base table, which the dynamic architecture fetch
    // populated.
    for (const auto &entry : targetQubitMapping) {
      auto deviceQubit = entry.deviceQubit;
      auto physicalName = backendLabel(deviceQubit);
      // A static mapping_file target has no dynamic architecture label table.
      // In that path the architecture file already uses CUDA-Q's dense IQM
      // names, so the provider name is the dense device coordinate.
      std::string fallbackName = "QB" + std::to_string(deviceQubit + 1);
      nlohmann::json singleQubitMapping;
      singleQubitMapping["logical_name"] = entry.logicalName;
      singleQubitMapping["physical_name"] = physicalName.value_or(fallbackName);
      qubitMapping.push_back(singleQubitMapping);
    }
  } else {
    // Apply the mapping derived from the dynamic quantum architecture.
    for (auto &[key, value] : qubitNameMap) {
      nlohmann::json singleQubitMapping;
      singleQubitMapping["logical_name"] = "QB" + std::to_string(value + 1);
      singleQubitMapping["physical_name"] = key;
      qubitMapping.push_back(singleQubitMapping);
    }
  }

  return qubitMapping;
}
