/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ResultReconstruction.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace {

struct OutputTarget {
  std::string outputName;
  std::size_t globalPosition = 0;
  std::size_t localPosition = 0;
  std::size_t resultIndex = 0;
};

struct OutputLayout {
  std::vector<OutputTarget> targets;
  std::unordered_map<std::string, std::size_t> outputWidths;
  std::size_t globalWidth = 0;
};

OutputLayout buildOutputLayout(const cudaq::ResultOutputMap &resultMap) {
  OutputLayout layout;
  std::unordered_map<std::string, std::vector<std::size_t>> targetIdsByOutput;
  std::unordered_set<std::size_t> globalPositions;

  // Every measured bit must have one unambiguous destination in the global
  // allocation-ordered bitstring.
  for (const auto &entry : resultMap.outputs) {
    if (!globalPositions.insert(entry.outputPosition).second)
      throw std::invalid_argument(
          "result output map contains duplicate output positions");
    const auto targetId = layout.targets.size();

    layout.targets.push_back(
        {entry.outputName, entry.outputPosition, 0, entry.resultIndex});
    targetIdsByOutput[entry.outputName].push_back(targetId);
    layout.globalWidth = std::max(layout.globalWidth, entry.outputPosition + 1);
  }

  // A named register is a projection of the global result. Rank its bits by
  // global allocation order so metadata or `mz` traversal order cannot leak
  // into that register's user-visible bitstring.
  for (auto &[outputName, targetIds] : targetIdsByOutput) {
    std::sort(targetIds.begin(), targetIds.end(),
              [&](std::size_t lhs, std::size_t rhs) {
                return layout.targets[lhs].globalPosition <
                       layout.targets[rhs].globalPosition;
              });
    for (std::size_t localPosition = 0; localPosition < targetIds.size();
         ++localPosition)
      layout.targets[targetIds[localPosition]].localPosition = localPosition;
    layout.outputWidths[outputName] = targetIds.size();
  }

  return layout;
}

void validateBit(char bit) {
  if (bit != '0' && bit != '1')
    throw std::invalid_argument("returned measurement bit must be '0' or '1'");
}

std::unordered_map<std::string, std::string>
makeInitialNamedBits(const OutputLayout &layout) {
  std::unordered_map<std::string, std::string> namedBits;
  for (const auto &[name, width] : layout.outputWidths)
    namedBits.emplace(name, std::string(width, '0'));
  return namedBits;
}

cudaq::sample_result reconstructSampleResultFromCounts(
    const cudaq::CountsDictionary &counts,
    const cudaq::ResultOutputMap &resultMap,
    const std::vector<std::string> &sequentialData) {
  // Without enriched metadata there is no source-to-destination mapping, so
  // preserve the provider bitstrings and their existing ordering.
  if (resultMap.outputs.empty()) {
    cudaq::ExecutionResult result{counts};
    result.sequentialData = sequentialData;
    return cudaq::sample_result(std::move(result));
  }

  // Map each compact QIR result index to its allocation-ordered global
  // position and to its position within the corresponding named result.
  const auto layout = buildOutputLayout(resultMap);
  cudaq::CountsDictionary globalCounts;
  std::unordered_map<std::string, cudaq::CountsDictionary> namedCounts;

  // Apply exactly the same source-to-destination mapping to aggregated counts
  // and to each per-shot bitstring.
  const auto reconstructBits = [&](const std::string &bitstring) {
    std::string globalBits(layout.globalWidth, '0');
    auto namedBits = makeInitialNamedBits(layout);
    for (const auto &target : layout.targets) {
      if (target.resultIndex >= bitstring.size())
        throw std::invalid_argument(
            "bitstring is shorter than a mapped result index");
      const char bit = bitstring[target.resultIndex];
      validateBit(bit);
      globalBits[target.globalPosition] = bit;
      namedBits[target.outputName][target.localPosition] = bit;
    }
    return std::pair{std::move(globalBits), std::move(namedBits)};
  };

  for (const auto &[bitstring, count] : counts) {
    if (count == 0)
      continue;

    auto [globalBits, namedBits] = reconstructBits(bitstring);
    globalCounts[globalBits] += count;
    for (const auto &[name, bits] : namedBits)
      namedCounts[name][bits] += count;
  }

  std::vector<std::string> globalSequentialData;
  globalSequentialData.reserve(sequentialData.size());
  std::unordered_map<std::string, std::vector<std::string>> namedSequentialData;
  for (const auto &bitstring : sequentialData) {
    auto [globalBits, namedBits] = reconstructBits(bitstring);
    globalSequentialData.push_back(std::move(globalBits));
    for (auto &[name, bits] : namedBits)
      namedSequentialData[name].push_back(std::move(bits));
  }

  std::vector<cudaq::ExecutionResult> executionResults;
  cudaq::ExecutionResult globalResult{globalCounts, cudaq::GlobalRegisterName};
  globalResult.sequentialData = std::move(globalSequentialData);
  executionResults.push_back(std::move(globalResult));
  for (auto &[name, countsByBits] : namedCounts) {
    cudaq::ExecutionResult namedResult{std::move(countsByBits), name};
    if (auto iter = namedSequentialData.find(name);
        iter != namedSequentialData.end())
      namedResult.sequentialData = std::move(iter->second);
    executionResults.push_back(std::move(namedResult));
  }
  return cudaq::sample_result(executionResults);
}

} // namespace

cudaq::ResultOutputMap cudaq::makeResultOutputMapFromEnrichedOutputNames(
    const nlohmann::json &outputNames) {
  ResultOutputMap resultMap;
  if (outputNames.is_null() || outputNames.empty())
    return resultMap;

  // QIR stores output locations inside the first schema array. Each entry maps
  // a compact result id to a physical qubit, a register name, and, when
  // enriched, the allocation-ordered output position.
  resultMap.outputs.reserve(outputNames.at(0).size());
  std::size_t denseResultIndex = 0;
  for (const auto &entry : outputNames.at(0)) {
    const auto resultIndex = entry.at(0).get<std::size_t>();
    const auto &outputLocation = entry.at(1);
    auto outputName = outputLocation.at(1).get<std::string>();
    // Older metadata has no destination field. Dense result order preserves
    // its historical behavior; enriched producers provide the semantic order.
    std::size_t outputPosition = denseResultIndex;
    if (outputLocation.size() > 2)
      outputPosition = outputLocation.at(2).get<std::size_t>();
    resultMap.outputs.push_back(
        {resultIndex, std::move(outputName), outputPosition});
    ++denseResultIndex;
  }
  return resultMap;
}

cudaq::sample_result
cudaq::reconstructSampleResultFromResultIndexedMeasurements(
    const CountsDictionary &counts, const ResultOutputMap &resultMap,
    const std::vector<std::string> &sequentialData) {
  return reconstructSampleResultFromCounts(counts, resultMap, sequentialData);
}
