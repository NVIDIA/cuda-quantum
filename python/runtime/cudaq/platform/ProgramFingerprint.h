/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"
#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {
struct CompileTarget;
}

namespace cudaq::detail {

/// Compute a deterministic fingerprint of the exact program that compilation
/// would see: the module with every callable closure merged in, plus every
/// argument substitution generated for compile-time-bound (callable)
/// arguments. Two launches whose fingerprints match are guaranteed to compile
/// to the same artifact, so the digest validates reuse of the cached module.
///
/// Returns `std::nullopt` when the program has a dependency whose
/// implementation is not owned by the module — a declaration backed by the
/// C++ registered-kernel registry, or a `cc.device_call` — because the module
/// text cannot vouch for code that lives outside it. Callers must then treat
/// the launch cannot be cached (compile every call).
///
/// On return, \p resolvedModule holds the merged clone. On a cache miss the
/// caller can compile it directly.
///
/// Setting `CUDAQ_FINGERPRINT_STRUCTURE` summarizes the module's structure
/// instead of printing it, which skips the clone, the merge and the printing
/// and leaves \p resolvedModule empty, so a caller that compiles must build
/// the program with `resolveProgram`. Both modes answer the same question, but
/// their digests are not interchangeable, and the structural digest is valid
/// only within the process that computed it, since it mixes in the identity of
/// context-uniqued attributes and types.
std::optional<std::array<std::uint8_t, 32>>
createProgramFingerprint(const std::string &name, mlir::ModuleOp mod,
                         const std::vector<void *> &rawArgs,
                         mlir::OwningOpRef<mlir::ModuleOp> &resolvedModule);

/// Clone \p mod and merge in every callable closure that \p rawArgs binds:
/// the program that `createProgramFingerprint` describes.
///
/// Compile this rather than a plain clone of \p mod.
/// `Compiler::prepareModule` fills in registry definitions before it merges
/// closures, so a registry declaration that only a closure brings in would
/// otherwise never be defined.
mlir::OwningOpRef<mlir::ModuleOp>
resolveProgram(const std::string &name, mlir::ModuleOp mod,
               const std::vector<void *> &rawArgs);

} // namespace cudaq::detail
