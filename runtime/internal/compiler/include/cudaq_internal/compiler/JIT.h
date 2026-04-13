/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/CompiledKernel.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>

namespace llvm {
class StringRef;
namespace orc {
class LLJIT;
}
} // namespace llvm

namespace mlir {
class ModuleOp;
class Type;
} // namespace mlir

namespace cudaq {
class CompiledKernel;
class ResultInfo;
} // namespace cudaq

namespace cudaq_internal::compiler {

/// Lower ModuleOp to QIR/LLVM IR and create a JIT execution engine.
cudaq::JitEngine createJITEngine(mlir::ModuleOp &moduleOp,
                                 llvm::StringRef convertTo);

/// @brief Create a `ResultInfo` from MLIR type and module.
///
/// When `resultType` is null or `isEntryPoint` is false, returns an empty
/// `ResultInfo`.
cudaq::ResultInfo createResultInfo(mlir::Type resultType, bool isEntryPoint,
                                   mlir::ModuleOp module);

} // namespace cudaq_internal::compiler
