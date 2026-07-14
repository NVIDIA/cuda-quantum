/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ServerHelper.h"
#include "nlohmann/json.hpp"
#include "cudaq/platform/qpu_utils.h"
#include "cudaq/utils/owning_ptr.h"

namespace cudaq {

std::optional<ResultOutputMap>
ServerHelper::resultMapForJob(const std::string &jobId) const {
  const auto iter = outputNames.find(jobId);
  if (iter == outputNames.end() || iter->second.empty())
    return std::nullopt;

  ResultOutputMap resultMap;
  resultMap.outputs.reserve(iter->second.size());
  for (const auto &[result, info] : iter->second)
    resultMap.outputs.push_back({.resultIndex = result,
                                 .outputName = info.registerName,
                                 .outputPosition = info.outputPosition});
  return resultMap;
}

std::optional<sample_result>
ServerHelper::tryReconstructFromResultIndexedCounts(
    const std::string &jobId, const CountsDictionary &counts,
    const std::vector<std::string> &sequentialData) {
  const auto resultMap = resultMapForJob(jobId);
  if (!resultMap)
    return std::nullopt;
  return reconstructSampleResultFromResultIndexedMeasurements(
      counts, *resultMap, sequentialData);
}

void ServerHelper::parseConfigForCommonParams(const BackendConfig &config) {
  // Parse common parameters for each job and place into member variables
  for (auto &[key, val] : config) {
    // First Form a newKey with just the portion after the "." (i.e. jobId)
    auto ix = key.find_first_of('.');
    std::string newKey;
    if (ix != key.npos)
      newKey = key.substr(ix + 1);

    if (key.starts_with("output_names.")) {
      // Parse `val` into jobOutputNames.
      // Note: See `FunctionAnalysisData::resultQubitVals` of
      // LowerToQIRProfile.cpp for an example of how this was populated.
      OutputNamesType jobOutputNames;
      nlohmann::json outputNamesJSON = nlohmann::json::parse(val);
      std::size_t denseResultIndex = 0;
      for (const auto &el : outputNamesJSON[0]) {
        const auto result = el[0].get<std::size_t>();
        const auto &outputLocation = el[1];
        const auto qubitNum = outputLocation[0].get<std::size_t>();
        auto registerName = outputLocation[1].get<std::string>();
        std::size_t outputPosition = denseResultIndex;
        if (outputLocation.size() > 2)
          outputPosition = outputLocation[2].get<std::size_t>();
        jobOutputNames[result] = {qubitNum, std::move(registerName),
                                  outputPosition};
        ++denseResultIndex;
      }

      this->outputNames[newKey] = jobOutputNames;
    } else if (key.starts_with("reorderIdx.")) {
      nlohmann::json tmp = nlohmann::json::parse(val);
      this->reorderIdx[newKey] = tmp.get<std::vector<std::size_t>>();
    }
  }
}

// Out-of-line definition of the deleter for owning_ptr<ServerHelper>;
// defined here where ServerHelper is complete so headers holding an
// owning_ptr<ServerHelper> do not need to see the full type.
template <>
void opaque_deleter<ServerHelper>::operator()(ServerHelper *p) const {
  delete p;
}
} // namespace cudaq

CUDAQ_INSTANTIATE_REGISTRY(cudaq::ServerHelper::RegistryType)

// Bridge so the Python extension (which has hidden-visibility Head/Tail for
// Registry<ServerHelper>) can look up server helpers registered in this DSO's
// unique-symbol registry (populated by dlopen'd serverhelper .so files).
extern "C" cudaq::ServerHelper *cudaq_find_server_helper(const char *name) {
  auto helper = cudaq::registry::get<cudaq::ServerHelper>(std::string(name));
  return helper.release();
}

extern "C" bool cudaq_has_server_helper(const char *name) {
  return cudaq::registry::isRegistered<cudaq::ServerHelper>(std::string(name));
}
