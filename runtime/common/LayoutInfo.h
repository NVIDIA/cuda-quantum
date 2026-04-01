/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <string>
#include <vector>

namespace mlir {
class ModuleOp;
class Type;
} // namespace mlir

namespace cudaq {
namespace cc {
class StructType;
} // namespace cc

using LayoutInfoType = std::pair<std::size_t, std::vector<std::size_t>>;

LayoutInfoType getLayoutInfo(const std::string &name,
                             void *opt_module = nullptr);

/// @brief Compute struct size and field offsets using the module's data layout.
LayoutInfoType getTargetLayout(mlir::ModuleOp mod, cc::StructType structTy);

/// @brief Compute the host-side buffer size (and struct field offsets, if
/// applicable) for the given MLIR kernel return type. Returns {0, {}} for
/// types that do not require a result buffer (e.g. `CallableType`). Throws on
/// unsupported types.
LayoutInfoType getResultBufferLayout(mlir::ModuleOp mod, mlir::Type resultTy);

} // namespace cudaq
