/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nlohmann/json.hpp"
#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
class Operation;
}

namespace cudaq::opt {

/// Map of result id to its (qubit index, register name) pair.
using ResultQubitVals =
    std::map<std::size_t, std::pair<std::size_t, std::string>>;

/// Read an operation's `mapping_v2p` attribute.
///
/// \param operation Operation carrying mapping metadata.
/// \return A lookup indexed by virtual qubit id whose values are final
/// physical QIR qubit ids, or an empty lookup when the attribute is absent.
std::vector<std::size_t>
getVirtualToPhysicalMapping(mlir::Operation *operation);

/// Build output_names entries of the form
/// [resultId, [qubitNum, registerName, outputPosition]]. Output positions are
/// the dense ranks of the measured qubits' output order.
///
/// \param resultQubitVals Maps each QIR result id to the measured qubit id and
/// its output register name. The qubit id is preserved as qubitNum in the
/// emitted metadata.
/// \param qubitToOutputOrder Optional lookup from qubit id to its
/// user-visible logical order. An empty lookup, or a qubit id outside its
/// bounds, falls back to the qubit id itself.
/// \return The entries wrapped in the single-element outer array expected by
/// the runtime output_names parser.
nlohmann::json buildEnrichedOutputNamesJson(
    const ResultQubitVals &resultQubitVals,
    const std::vector<std::size_t> &qubitToOutputOrder);

/// Build `output_names` entries of the form
/// [resultId, [qubitNum, registerName, outputPosition]], accounting for a
/// virtual-to-physical qubit mapping. The physical qubit number remains in
/// qubitNum, while outputPosition follows the original virtual-qubit order.
///
/// \param resultQubitVals Maps each QIR result id to the measured physical
/// qubit id and its output register name.
/// \param virtualToPhysical Maps each original virtual qubit id to its final
/// physical QIR qubit id. An empty mapping preserves identity ordering.
/// \return The entries wrapped in the single-element outer array expected by
/// the runtime `output_names` parser.
nlohmann::json buildEnrichedOutputNamesJsonFromV2PMapping(
    const ResultQubitVals &resultQubitVals,
    const std::vector<std::size_t> &virtualToPhysical);

} // namespace cudaq::opt
