/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/algorithms/policies.h"
#include "cudaq/builder/kernel_builder.h"
#include "cudaq/platform/qpu_types.h"
#include "cudaq/platform/quantum_platform.h"

namespace cudaq {
quantum_platform *getQuantumPlatformInternal();

/// @brief Return the quantum platform provided by the linked platform library
/// @return
inline quantum_platform &get_platform() {
  return *getQuantumPlatformInternal();
}

/// @brief Return the number of QPUs (at runtime)
inline std::size_t platform_num_qpus() {
  return getQuantumPlatformInternal()->num_qpus();
}

/// @brief Return true if the quantum platform is remote.
inline bool is_remote_platform() {
  return getQuantumPlatformInternal()->is_remote();
}

/// @brief Return true if the quantum platform is emulated.
inline bool is_emulated_platform() {
  return getQuantumPlatformInternal()->is_emulated();
}

/// @brief Return true if the quantum platform is a simulator.
inline bool is_simulator_platform() {
  return getQuantumPlatformInternal()->is_simulator();
}

inline std::unique_ptr<cudaq::CompileTarget>
get_compile_target(const cudaq::sample_policy &policy) {
  return getQuantumPlatformInternal()->getCompileTarget(policy);
}

inline std::unique_ptr<cudaq::CompileTarget>
get_compile_target(const cudaq::observe_policy &policy) {
  return getQuantumPlatformInternal()->getCompileTarget(policy);
}

inline std::unique_ptr<cudaq::CompileTarget>
get_compile_target(const cudaq::other_policies &policy) {
  auto *ctx = cudaq::getExecutionContext();
  return getQuantumPlatformInternal()->getCompileTarget(policy, ctx);
}

/// Get the default compile target configuration used when JITing for Python.
std::unique_ptr<cudaq::CompileTarget>
getDefaultPythonCompileTarget(const sample_policy &policy);
std::unique_ptr<cudaq::CompileTarget>
getDefaultPythonCompileTarget(const observe_policy &policy);
std::unique_ptr<cudaq::CompileTarget>
getDefaultPythonCompileTarget(const other_policies &policy,
                              ExecutionContext *context);

// Declare this function, implemented elsewhere
std::string getQIR(const std::string &);

} // namespace cudaq
