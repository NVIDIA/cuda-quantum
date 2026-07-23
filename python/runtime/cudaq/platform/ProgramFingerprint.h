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
/// the launch as non-cacheable (compile every call).
///
/// On return, \p resolvedModule holds the merged clone. On a cache miss the
/// caller can compile it directly.
std::optional<std::array<std::uint8_t, 32>>
createProgramFingerprint(const std::string &name, mlir::ModuleOp mod,
                         const std::vector<void *> &rawArgs,
                         const cudaq::CompileTarget &target,
                         mlir::OwningOpRef<mlir::ModuleOp> &resolvedModule);

} // namespace cudaq::detail
