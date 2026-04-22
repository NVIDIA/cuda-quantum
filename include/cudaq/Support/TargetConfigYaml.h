/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "TargetConfig.h"
#include "llvm/Support/YAMLTraits.h"

// These structs can be used in a vector.
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::TargetFeatureFlag)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::TargetArgument)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::ConditionalBuildConfig)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::BackendFeatureMap)
LLVM_YAML_IS_SEQUENCE_VECTOR(cudaq::config::MachineArchitectureConfig)

namespace llvm {
namespace yaml {
// YML serialization declarations.
template <>
struct ScalarBitSetTraits<cudaq::config::TargetFeatureFlag> {
  static void bitset(IO &io, cudaq::config::TargetFeatureFlag &value);
};

template <>
struct ScalarEnumerationTraits<cudaq::config::ArgumentType> {
  static void enumeration(IO &io, cudaq::config::ArgumentType &value);
};

template <>
struct MappingTraits<cudaq::config::TargetArgument> {
  static void mapping(IO &io, cudaq::config::TargetArgument &info);
  static std::string validate(IO &io, cudaq::config::TargetArgument &info);
};

template <>
struct BlockScalarTraits<cudaq::config::SimulationBackendSetting> {
  static void output(const cudaq::config::SimulationBackendSetting &Value,
                     void *Ctxt, llvm::raw_ostream &OS);
  static StringRef input(StringRef Scalar, void *Ctxt,
                         cudaq::config::SimulationBackendSetting &Value);
};
template <>
struct MappingTraits<cudaq::config::ConditionalBuildConfig> {
  static void mapping(IO &io, cudaq::config::ConditionalBuildConfig &info);
};

template <>
struct MappingTraits<cudaq::config::BackendEndConfigEntry> {
  static void mapping(IO &io, cudaq::config::BackendEndConfigEntry &info);
};
template <>
struct MappingTraits<cudaq::config::BackendFeatureMap> {
  static void mapping(IO &io, cudaq::config::BackendFeatureMap &info);
};

template <>
struct MappingTraits<cudaq::config::TargetConfig> {
  static void mapping(IO &io, cudaq::config::TargetConfig &info);
  static std::string validate(IO &io, cudaq::config::TargetConfig &info);
};

template <>
struct MappingTraits<cudaq::config::TargetArchitectureSettings> {
  static void mapping(IO &io, cudaq::config::TargetArchitectureSettings &info);
};

template <>
struct MappingTraits<cudaq::config::MachineArchitectureConfig> {
  static void mapping(IO &io, cudaq::config::MachineArchitectureConfig &info);
  static std::string validate(IO &io,
                              cudaq::config::MachineArchitectureConfig &info);
};
} // namespace yaml
} // namespace llvm
