/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "KernelExecution.h"
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

} // namespace cudaq
