/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/CompiledModule.h"
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
class PassManager;
class Type;
} // namespace mlir

namespace cudaq {
class CompiledModule;
} // namespace cudaq

namespace cudaq_internal::compiler {

/// Lower ModuleOp to QIR/LLVM IR and create a JIT execution engine.
cudaq::JitEngine createJITEngine(mlir::ModuleOp &moduleOp,
                                 llvm::StringRef convertTo);

} // namespace cudaq_internal::compiler
