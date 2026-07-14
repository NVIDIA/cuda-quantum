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

namespace cudaq::opt {

/// Map of result id to its (qubit index, register name) pair.
using ResultQubitVals =
    std::map<std::size_t, std::pair<std::size_t, std::string>>;

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

} // namespace cudaq::opt
