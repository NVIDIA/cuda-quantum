/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/Hash.h"
#include <cstddef>
#include <optional>

namespace cudaq {

namespace detail {
// Helper to wrap std::optional<bool> with a baked-in default
template <bool DefaultValue>
struct DefaultedBool {
  std::optional<bool> opt = std::nullopt;

  // implicit conversions for clean assignment & reading
  DefaultedBool() = default;
  DefaultedBool(bool val) : opt(val) {}

  [[nodiscard]] bool get() const { return opt.value_or(DefaultValue); }
  operator bool() const { return get(); }
  [[nodiscard]] bool is_default() const { return !opt.has_value(); }
};
} // namespace detail

/// Boolean values that default to false or true if not set.
using default_false = detail::DefaultedBool<false>;
using default_true = detail::DefaultedBool<true>;

struct CompileOptions {
  /// Issue a warning if named measurements are contained in the kernel.
  default_false warnNamedMeasurements;

  /// Whether to retrieve mapping reorder indices from MLIR and store it as
  /// compiled metadata.
  default_false storeReorderIdx;

  /// Whether to generate resource counts.
  ///
  /// When true, the compiler will generate resource counts during compilation
  /// and simplify the IR to remove all quantum operations already accounted
  /// for in the counts.
  default_false emitResourceCounts;

  /// Whether to create local JIT artifacts even when not emulating the target.
  ///
  /// Analysis contexts that execute locally, but are entered through a remote
  /// platform, use this to run the kernel under the analysis simulator instead
  /// of submitting it to the remote executor.
  default_false emitJit;

  /// Whether to translate MLIR artifacts into target transport code.
  ///
  /// Local analysis contexts can set this to false when they only need the JIT
  /// artifact and do not need a QIR/QASM payload for the remote backend.
  default_true emitTargetCode;

  /// Whether to skip the target lowering compilation pipeline.
  ///
  /// Local analysis contexts set this to true: they JIT the kernel directly
  /// for an analysis simulator. The target lowering pipeline would otherwise
  /// erase operations such as noise or QEC, or fail to legalize them during
  /// code generation.
  default_false skipTargetLoweringPipeline;

  /// Whether to add measurements at the end of the kernel using the
  /// `add-measurements` pass.
  default_false addMeasurements;

  /// Throw a compilation error if the kernel uses conditionals on measurement
  /// results.
  default_false failOnConditionalsOnMeasureResults;

  /// Make the required adjustments to target an emulator rather than the real
  /// device.
  default_false emulate;
};

} // namespace cudaq

/// Hash of the effective option values. Compile options change the compiled
/// artifact, so any cache keyed on the compiler configuration must fold this
/// in alongside `std::hash<cudaq::CompileTarget>`.
template <>
struct std::hash<cudaq::CompileOptions> {
  std::size_t operator()(const cudaq::CompileOptions &o) const noexcept {
    return cudaq::hashVal(
        o.warnNamedMeasurements.get(), o.storeReorderIdx.get(),
        o.emitResourceCounts.get(), o.emitJit.get(), o.emitTargetCode.get(),
        o.skipTargetLoweringPipeline.get(), o.addMeasurements.get(),
        o.failOnConditionalsOnMeasureResults.get(), o.emulate.get());
  }
};
