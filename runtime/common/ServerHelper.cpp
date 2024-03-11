/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ServerHelper.h"

namespace cudaq {
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
      for (const auto &el : outputNamesJSON[0]) {
        auto result = el[0].get<std::size_t>();
        auto qubitNum = el[1][0].get<std::size_t>();
        auto registerName = el[1][1].get<std::string>();
        jobOutputNames[result] = {qubitNum, registerName};
      }

      this->outputNames[newKey] = jobOutputNames;
    } else if (key.starts_with("reorderIdx.")) {
      nlohmann::json tmp = nlohmann::json::parse(val);
      this->reorderIdx[newKey] = tmp.get<std::vector<std::size_t>>();
    }
  }
}
} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::ServerHelper::RegistryType)
