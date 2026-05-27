/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq_internal/compiler/TracePassInstrumentation.h"

// Pass registration is done through the 'register_dialect' python call.
// The native target initialization is built into the MLIR python extension.
void cudaq_internal::compiler::initializeLangMLIR() {}

// FIXME: Declare this in a header file!
// Forward-declare the Python-aware helper so this translation unit does not
// pull in headers from python/. The symbol is defined in
// python/runtime/cudaq/platform/PythonSignalCheck.cpp, which is linked into
// the same Python extension.
namespace cudaq {
mlir::LogicalResult runPassManagerReleasingGIL(mlir::PassManager &pm,
                                               mlir::Operation *op);
}

mlir::LogicalResult
cudaq_internal::compiler::runPassManager(mlir::PassManager &pm,
                                         mlir::Operation *op) {
  pm.addInstrumentation(std::make_unique<cudaq::TracePassInstrumentation>());
  return cudaq::runPassManagerReleasingGIL(pm, op);
}
