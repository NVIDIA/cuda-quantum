/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CompiledModule.h"
#include "Resources.h"
#include "cudaq_json.h"
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

/// @brief Every kernel execution has a name, compiled code representation, and
/// (optionally) an output_names mapping showing how each Result maps back
/// to the original program's Qubits.
struct KernelExecution {
  std::string name;
  std::string code;
  std::optional<cudaq::JitEngine> jit;
  std::optional<Resources> resourceCounts;
  cudaq::cudaq_json output_names;
  std::vector<std::size_t> mapping_reorder_idx;
  cudaq::cudaq_json user_data;
  KernelExecution(const std::string &n, const std::string &c,
                  std::optional<cudaq::JitEngine> jit,
                  std::optional<Resources> rc, std::vector<std::size_t> &m);
  KernelExecution(const std::string &n, const std::string &c,
                  std::optional<cudaq::JitEngine> jit,
                  std::optional<Resources> rc, nlohmann::json &o,
                  std::vector<std::size_t> &m);
  KernelExecution(const std::string &n, const std::string &c,
                  std::optional<cudaq::JitEngine> jit,
                  std::optional<Resources> rc, nlohmann::json &o,
                  std::vector<std::size_t> &m, nlohmann::json &ud);
  ~KernelExecution();
  KernelExecution(const KernelExecution &other);
  KernelExecution &operator=(const KernelExecution &other);
  KernelExecution(KernelExecution &&) noexcept;
  KernelExecution &operator=(KernelExecution &&) noexcept;
};

} // namespace cudaq
