/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ServerHelper.h"

namespace cudaq {
void ServerHelper::parseConfigForOutputNames(const BackendConfig &config) {
  // Parse the output_names.* (for each job) and place it in outputNames[]
  for (auto &[key, val] : config) {
    if (key.starts_with("output_names.")) {
      // Parse `val` into jobOutputNames.
      // Note: See `FunctionAnalysisData::resultQubitVals` of
      // LowerToQIRProfile.cpp for an example of how this was populated.
      OutputNamesType jobOutputNames;
      nlohmann::json outputNamesJSON = nlohmann::json::parse(val);
      for (const auto &el : outputNamesJSON[0]) {
        auto result = el[0].get<std::size_t>();
        auto qirQubit = el[1][0].get<std::size_t>();
        auto userQubit = el[1][1].get<std::size_t>();
        auto registerName = el[1][2].get<std::string>();
        jobOutputNames[result] = {qirQubit, userQubit, registerName};
      }

      this->outputNames[key] = jobOutputNames;
    }
  }
}
} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::ServerHelper::RegistryType)
