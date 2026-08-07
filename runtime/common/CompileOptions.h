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

namespace cudaq {

struct CompileOptions {
  /// Issue a warning if named measurements are contained in the kernel.
  bool warnNamedMeasurements = false;

  /// Whether to retrieve mapping reorder indices from MLIR and store it as
  /// compiled metadata.
  bool storeReorderIdx = false;

  /// Whether to generate resource counts.
  ///
  /// When true, the compiler will generate resource counts during compilation
  /// and simplify the IR to remove all quantum operations already accounted
  /// for in the counts.
  bool emitResourceCounts = false;

  /// Whether to create local JIT artifacts even when not emulating the target.
  ///
  /// Analysis contexts that execute locally, but are entered through a remote
  /// platform, use this to run the kernel under the analysis simulator instead
  /// of submitting it to the remote executor.
  bool emitJit = false;

  /// Whether to translate MLIR artifacts into target transport code.
  ///
  /// Local analysis contexts can set this to false when they only need the JIT
  /// artifact and do not need a QIR/QASM payload for the remote backend.
  bool emitTargetCode = true;

  /// Whether to skip the target lowering compilation pipeline.
  ///
  /// Local analysis contexts set this to true: they JIT the kernel directly
  /// for an analysis simulator. The target lowering pipeline would otherwise
  /// erase operations such as noise or QEC, or fail to legalize them during
  /// code generation.
  bool skipTargetLoweringPipeline = false;

  /// Whether to add measurements at the end of the kernel using the
  /// `add-measurements` pass.
  bool addMeasurements = false;

  /// Throw a compilation error if the kernel uses conditionals on measurement
  /// results.
  bool failOnConditionalsOnMeasureResults = false;

  /// Make the required adjustments to target an emulator rather than the real
  /// device.
  bool emulate = false;
};

} // namespace cudaq

/// Hash of the effective option values. Compile options change the compiled
/// artifact, so any cache keyed on the compiler configuration must fold this
/// in alongside `std::hash<cudaq::CompileTarget>`.
template <>
struct std::hash<cudaq::CompileOptions> {
  std::size_t operator()(const cudaq::CompileOptions &o) const noexcept {
    return cudaq::detail::hashVal(
        o.warnNamedMeasurements, o.storeReorderIdx, o.emitResourceCounts,
        o.emitJit, o.emitTargetCode, o.skipTargetLoweringPipeline,
        o.addMeasurements, o.failOnConditionalsOnMeasureResults, o.emulate);
  }
};
