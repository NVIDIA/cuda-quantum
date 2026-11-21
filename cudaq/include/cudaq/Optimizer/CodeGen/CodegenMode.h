/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

namespace cudaq::opt {

/// Which of normalization and optimization `codegen` runs before it emits
/// code.
/// The compiler picks the mode for each kernel from the stages that ran.
// TODO: Remove this type once every target compiles through a stage that
// normalizes the kernel, because `codegen` then only emits. `normalize` and
// `normalize_and_optimize` serve the targets that do not yet.
enum class CodegenMode {
  /// The stages, prep, and finalize left the kernel ready for emission, so
  /// `codegen` only emits.
  emit_only,
  /// No stage normalized the kernel, so `codegen` does. It does not optimize,
  /// which is what level 0 means.
  normalize,
  /// No stage normalized the kernel, so `codegen` does, and it also applies
  /// the optimization level.
  normalize_and_optimize,
};

} // namespace cudaq::opt
