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
#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

using DeviceQubit = std::size_t;
using LocalQubitIndex = std::size_t;
using ActiveDeviceQubits = std::vector<DeviceQubit>;

struct TargetQubitMappingEntry {
  std::string logicalName;
  DeviceQubit deviceQubit = 0;
};

using TargetQubitMapping = std::vector<TargetQubitMappingEntry>;

/// @brief Check the execution metadata invariant conditions at the compile
/// boundary:
/// active device qubits are sorted and unique, and the user-visible output
/// positions carried in enriched output_names are dense and unique. Throws
/// std::invalid_argument when an invariant is violated.
void validateExecutionMetadata(const ActiveDeviceQubits &activeDeviceQubits,
                               const nlohmann::json &outputNames);

/// @brief Every kernel execution has a name, compiled code representation, and
/// (optionally) an output_names mapping showing how each Result maps back
/// to the original program's Qubits.
struct KernelExecution {
  std::string name;
  std::string code;
  std::optional<cudaq::JitEngine> jit;
  std::optional<Resources> resourceCounts;
  cudaq::cudaq_json output_names;
  bool hasConditionalsOnMeasureResults = false;
  cudaq::cudaq_json user_data;
  /// @brief Active device qubits used by this emitted execution, populated only
  /// when mapping ran. Unlike output_names, this includes unmeasured ancillary
  /// qubits and mapper-introduced wires, so it rides as a per-execution
  /// sidecar. Consumed by native-mapping remote backends to build hardware
  /// payloads, and used locally to detect that a mapped reconstruction is
  /// required.
  ActiveDeviceQubits activeDeviceQubits;
  /// @brief Target-code logical qubit names mapped to device qubits for this
  /// execution. Populated only when the target backend needs the mapped target
  /// code's logical names in its submission payload.
  TargetQubitMapping targetQubitMapping;
  KernelExecution(const std::string &n, const std::string &c,
                  std::optional<cudaq::JitEngine> jit,
                  std::optional<Resources> rc);
  KernelExecution(const std::string &n, const std::string &c,
                  std::optional<cudaq::JitEngine> jit,
                  std::optional<Resources> rc, nlohmann::json &o);
  ~KernelExecution();
  KernelExecution(const KernelExecution &) = default;
  KernelExecution &operator=(const KernelExecution &) = default;
  KernelExecution(KernelExecution &&) noexcept = default;
  KernelExecution &operator=(KernelExecution &&) noexcept = default;
};

} // namespace cudaq
