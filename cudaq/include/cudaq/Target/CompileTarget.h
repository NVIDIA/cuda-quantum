/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/Profile/AllProfiles.h"
#include "cudaq/Target/Profile/ExplicitPipelineProfile.h"
#include <map>
#include <string>
#include <variant>
#include <vector>

namespace cudaq::config {
class TargetConfig;
} // namespace cudaq::config

namespace cudaq {

/// Target properties used to define the compilation pipeline.
struct CompileTarget {
  /// Whether to recompile the kernel in the presence of an AOT-compiled module.
  ///
  /// If this is `false` and an AOT-compiled kernel (in the form of a function
  /// pointer) is provided, then compilation will be skipped and all other
  /// options in this class will be ignored.
  ///
  /// If this is `true`, the AOT-compiled module (if it exists) will be
  /// discarded and compilation will start from scratch, according to the
  /// options in this class.
  bool overrideAOTCompilation = false;

  /// Ordered target IR profiles, defining the successive lowering stages during
  /// compilation.
  std::vector<config::Profile> profiles{};

  /// Code generation emission selected by the target.
  std::string codegenTranslation{};

  /// Whether branching on measurement results is supported.
  bool supportConditionalsOnMeasureResults = true;

  /// Whether device calls are supported by the target.
  bool supportDeviceCalls = false;

  /// Whether the target supports pre-computing resource counts for faster
  /// resource estimation.
  ///
  /// Only used for resource estimation. When set, the compiler will strip out
  /// all gates outside of control flow and produce gate counts statically.
  bool supportResourceCounts = true;

  /// Whether explicit measurements are supported by the target.
  bool supportExplicitMeasurements = true;

  /// Whether the target supports measuring arbitrary observables.
  ///
  /// If false, the compiler will implement observable measurements by splitting
  /// the hamiltonian into a sum of Pauli measurements and emit one lowered
  /// module per Pauli term.
  bool supportObservableMeasurements = true;

  /// Whether the target supports sampling without explicit measurements in the
  /// IR.
  ///
  /// If false, when compiling for `sample`, the compiler will ensure explicit
  /// measurements are present at the end of the kernel (or add them if not).
  bool supportSampleWithoutMeasurements = true;

  /// Whether to fully specialize the kernel.
  bool fullySpecialize = true;

  /// Set the `changeSemantics` flag for the argument synthesis pass.
  bool argumentSynthChangeSemantics = true;

  /// Access the i-th profile of type `T`, or `nullptr` if not present.
  template <typename T>
  const T *profile(std::size_t i = 0) const {
    if (i >= profiles.size())
      return nullptr;
    if (const auto *match = std::get_if<T>(&profiles[i]))
      return match;
    return nullptr;
  }

  /// Access the i-th profile of type `T`, or `nullptr` if not present.
  template <typename T>
  T *profile(std::size_t i = 0) {
    if (i >= profiles.size())
      return nullptr;
    if (auto *match = std::get_if<T>(&profiles[i]))
      return match;
    return nullptr;
  }

  /// Access the explicit pipeline profile.
  ///
  /// @throws std::runtime_error if there isn't exactly one profile or the first
  /// profile is not an explicit pipeline profile
  // TODO: Temporary utility function used while there is only one explicit
  // profile, to be replaced with more general profile handling
  const config::ExplicitPipelineProfile &getExplicitPipelineProfile() const {
    if (profiles.size() != 1)
      throw std::runtime_error(
          "Unsupported IR profile: only ExplicitPipelineProfile is "
          "currently supported");
    const auto *explicitProfile = profile<config::ExplicitPipelineProfile>();
    if (!explicitProfile)
      throw std::runtime_error(
          "Unsupported IR profile: only ExplicitPipelineProfile is "
          "currently supported");
    return *explicitProfile;
  }
  config::ExplicitPipelineProfile &getExplicitPipelineProfile() {
    if (profiles.size() != 1)
      throw std::runtime_error(
          "Unsupported IR profile: only ExplicitPipelineProfile is "
          "currently supported");
    auto *explicitProfile = profile<config::ExplicitPipelineProfile>();
    if (!explicitProfile)
      throw std::runtime_error(
          "Unsupported IR profile: only ExplicitPipelineProfile is "
          "currently supported");
    return *explicitProfile;
  }

  /// Construct a CompileTarget from static and runtime backend configurations.
  static CompileTarget createFromConfig(
      config::TargetConfig targetConfig,
      std::map<std::string, std::string> runtimeConfig,
      std::map<std::string, std::string> pipelineSubstitutions = {});

  bool operator==(const CompileTarget &other) const = default;
};

} // namespace cudaq

template <>
struct std::hash<cudaq::config::ExplicitPipelineProfile> {
  std::size_t operator()(
      const cudaq::config::ExplicitPipelineProfile &profile) const noexcept;
};
template <>
struct std::hash<std::vector<cudaq::config::Profile>> {
  std::size_t operator()(
      const std::vector<cudaq::config::Profile> &profiles) const noexcept;
};
template <>
struct std::hash<cudaq::CompileTarget> {
  std::size_t operator()(const cudaq::CompileTarget &t) const noexcept;
};
