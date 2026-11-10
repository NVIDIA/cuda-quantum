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
#include <algorithm>
#include <numeric>

namespace cudaq {

void ServerHelper::parseConfigForCommonParams(const BackendConfig &config) {
  // Parse common parameters for each job and place into member variables
  for (auto &[key, val] : config) {
    // First Form a newKey with just the portion after the "." (i.e. jobId)
    auto ix = key.find_first_of('.');
    std::string newKey;
    if (ix != key.npos)
      newKey = key.substr(ix + 1);

    if (key.starts_with("reorderIdx.")) {
      nlohmann::json tmp = nlohmann::json::parse(val);
      this->reorderIdx[newKey] = tmp.get<std::vector<std::size_t>>();
    } else if (key.starts_with("recordedQubits.")) {
      nlohmann::json tmp = nlohmann::json::parse(val);
      this->recordedQubits[newKey] = tmp.get<std::vector<std::size_t>>();
    }
  }
}

sample_result
ServerHelper::sliceToRecord(const sample_result &readout,
                            const std::vector<std::size_t> &recorded) {
  const auto &sliced =
      recorded.empty() ? readout : readout.get_marginal(recorded);
  return sample_result(
      std::vector<ExecutionResult>{ExecutionResult{sliced.to_map()}});
}

std::vector<std::size_t>
ServerHelper::reorderIdxFromRecord(const std::vector<std::size_t> &recorded) {
  std::vector<std::size_t> byPhysical(recorded.size());
  std::iota(byPhysical.begin(), byPhysical.end(), 0);
  std::sort(
      byPhysical.begin(), byPhysical.end(),
      [&](std::size_t a, std::size_t b) { return recorded[a] < recorded[b]; });
  // `byPhysical[p]` is the record position reported at readout position `p`;
  // invert it so the result is indexed by record position.
  std::vector<std::size_t> reorderIdx(recorded.size());
  for (std::size_t p = 0; p < byPhysical.size(); ++p)
    reorderIdx[byPhysical[p]] = p;
  return reorderIdx;
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
