/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Support/Hash.h"
#include "cudaq/Target/CompileTarget.h"
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

  /// Whether to add measurements at the end of the kernel for `sample` using
  /// the `add-measurements` pass.
  bool addSampleMeasurements = false;

  /// Throw a compilation error if the kernel uses conditionals on measurement
  /// results.
  bool failOnConditionalsOnMeasureResults = false;

  /// Make the required adjustments to target an emulator rather than the real
  /// device.
  bool emulate = false;

  /// Whether to disable quantum optimization passes (e.g. value semantics
  /// lowering). Used in tracer mode.
  bool disableQuantumOpts = false;

  /// Whether to pack `i1` vectors as bit-packed `std::vector<bool>` (for local
  /// simulators)
  bool boolVecBitPacked = false;

  /// Whether to measure an observable at the end of the kernel, and if so,
  /// the observable to measure.
  std::optional<cudaq::spin_op> measureObservable;
};

} // namespace cudaq

/// Hash of the effective option values. Compile options change the compiled
/// artifact, so any cache keyed on the compiler configuration must fold this
/// in alongside `std::hash<cudaq::CompileTarget>`.
template <>
struct std::hash<cudaq::CompileOptions> {
  std::size_t operator()(const cudaq::CompileOptions &o) const noexcept {
    auto pauliStr = o.measureObservable ? o.measureObservable->to_string() : "";
    return cudaq::detail::hashVal(
        o.warnNamedMeasurements, o.storeReorderIdx, o.emitResourceCounts,
        o.emitJit, o.emitTargetCode, o.skipTargetLoweringPipeline,
        o.addSampleMeasurements, o.failOnConditionalsOnMeasureResults,
        o.emulate, o.disableQuantumOpts, o.boolVecBitPacked, pauliStr);
  }
};
