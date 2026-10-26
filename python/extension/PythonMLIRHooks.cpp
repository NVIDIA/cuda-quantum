/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"
#include "runtime/cudaq/platform/PythonSignalCheck.h"

// FIXME: Declare this in a header file!
// Forward-declare the Python-aware helper so this translation unit does not
// pull in headers from python/. The symbol is defined in
// python/runtime/cudaq/platform/PythonSignalCheck.cpp, which is linked into
// the same Python extension.
namespace cudaq {
mlir::LogicalResult runPassManagerReleasingGIL(mlir::PassManager &pm,
                                               mlir::Operation *op);
}

static mlir::LogicalResult pythonRunPassManager(mlir::PassManager &pm,
                                                mlir::Operation *op) {
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  cudaq::addPythonSignalInstrumentation(pm);
  cudaq_internal::compiler::configurePassManagerFromEnv(pm);
  return cudaq::runPassManagerReleasingGIL(pm, op);
}

namespace cudaq_internal::compiler {
void installPythonMLIRHooks() { setRunPassManagerHook(&pythonRunPassManager); }
} // namespace cudaq_internal::compiler
