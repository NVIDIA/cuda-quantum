/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "ServerHelper.h"
#include "nlohmann/json.hpp"

namespace cudaq {

KernelExecution::KernelExecution(const std::string &n, const std::string &c,
                                 std::optional<cudaq::JitEngine> jit,
                                 std::optional<Resources> rc,
                                 std::vector<std::size_t> &m)
    : name(n), code(c), jit(jit), resourceCounts(rc),
      output_names(nlohmann::json{}), mapping_reorder_idx(m),
      user_data(nlohmann::json{}) {}
KernelExecution::KernelExecution(const std::string &n, const std::string &c,
                                 std::optional<cudaq::JitEngine> jit,
                                 std::optional<Resources> rc, nlohmann::json &o,
                                 std::vector<std::size_t> &m)
    : name(n), code(c), jit(jit), resourceCounts(rc), output_names(o),
      mapping_reorder_idx(m), user_data(nlohmann::json{}) {}
KernelExecution::KernelExecution(const std::string &n, const std::string &c,
                                 std::optional<cudaq::JitEngine> jit,
                                 std::optional<Resources> rc, nlohmann::json &o,
                                 std::vector<std::size_t> &m,
                                 nlohmann::json &ud)
    : name(n), code(c), jit(jit), resourceCounts(rc), output_names(o),
      mapping_reorder_idx(m), user_data(ud) {}

KernelExecution::~KernelExecution() = default;
KernelExecution::KernelExecution(KernelExecution &&) noexcept = default;
KernelExecution &
KernelExecution::operator=(KernelExecution &&) noexcept = default;

KernelExecution::KernelExecution(const KernelExecution &other)
    : name(other.name), code(other.code), jit(other.jit),
      resourceCounts(other.resourceCounts), output_names(other.output_names),
      mapping_reorder_idx(other.mapping_reorder_idx),
      user_data(other.user_data) {}

KernelExecution &KernelExecution::operator=(const KernelExecution &other) {
  return *this = KernelExecution(other);
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

CUDAQ_INSTANTIATE_REGISTRY(cudaq::ServerHelper::RegistryType)
