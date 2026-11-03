/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "mlir/ExecutionEngine/ExecutionEngine.h"

namespace mlir {
class ModuleOp;
}

namespace cudaq_internal::compiler {

/// Create an ExecutionEngine that resolves CUDA-Q runtime symbols without
/// process-wide lookup.
llvm::Expected<std::unique_ptr<mlir::ExecutionEngine>>
createExecutionEngine(mlir::ModuleOp module,
                      mlir::ExecutionEngineOptions options);

} // namespace cudaq_internal::compiler
