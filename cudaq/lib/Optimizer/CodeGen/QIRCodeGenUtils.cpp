/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/QIRCodeGenUtils.h"
#include <algorithm>
#include <numeric>

nlohmann::json cudaq::opt::buildEnrichedOutputNamesJson(
    const ResultQubitVals &resultQubitVals,
    const std::vector<std::size_t> &qubitToOutputOrder) {
  std::vector<std::size_t> resultIds;
  std::vector<std::size_t> outputOrders;
  resultIds.reserve(resultQubitVals.size());
  outputOrders.reserve(resultQubitVals.size());
  // Resolve each measured qubit to its logical output order. Keep the result
  // ids in a parallel vector so equal output orders have a deterministic
  // result-id tie-breaker below.
  for (const auto &[resultId, qubitAndName] : resultQubitVals) {
    const auto qubit = qubitAndName.first;
    auto outputOrder = qubit;
    if (!qubitToOutputOrder.empty() && qubit < qubitToOutputOrder.size())
      outputOrder = qubitToOutputOrder[qubit];

    resultIds.push_back(resultId);
    outputOrders.push_back(outputOrder);
  }

  // Sort entry indices instead of the entries themselves. This preserves the
  // result-id order used when the JSON entries are emitted.
  std::vector<std::size_t> rank(resultIds.size());
  std::iota(rank.begin(), rank.end(), 0);
  std::sort(rank.begin(), rank.end(), [&](std::size_t lhs, std::size_t rhs) {
    if (outputOrders[lhs] != outputOrders[rhs])
      return outputOrders[lhs] < outputOrders[rhs];
    return resultIds[lhs] < resultIds[rhs];
  });

  // Invert the sorted index permutation. Its ordinal is the dense output
  // position, so partial measurements never leave gaps in the global result.
  std::vector<std::size_t> outputPositions(resultIds.size());
  for (std::size_t position = 0; position < rank.size(); ++position)
    outputPositions[rank[position]] = position;

  // Emit entries in result-id order, attaching the independently computed
  // user-visible position as the third output-location element.
  auto entries = nlohmann::json::array();
  std::size_t entryIndex = 0;
  for (const auto &[resultId, qubitAndName] : resultQubitVals) {
    entries.push_back({resultId,
                       {qubitAndName.first, qubitAndName.second,
                        outputPositions[entryIndex]}});
    ++entryIndex;
  }
  return nlohmann::json::array({std::move(entries)});
}
