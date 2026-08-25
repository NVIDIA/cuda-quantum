/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_PYTHON_SIGNAL_CHECK_H
#define CUDAQ_PYTHON_SIGNAL_CHECK_H

#include "mlir/Support/LogicalResult.h"

namespace mlir {
class PassManager;
class Operation;
} // namespace mlir

namespace cudaq {

/// Add instrumentation that checks for pending Python signals between passes.
/// When a signal is pending, emits an MLIR error diagnostic to stop the
/// pipeline. The error message propagates through normal MLIR error handling.
void addPythonSignalInstrumentation(mlir::PassManager &pm);

/// Run `pm` on `op`, releasing the Python GIL for the duration of the run.
/// MLIR runs nested passes in parallel via its context thread pool. Workers
/// call PyGILState_Ensure via the signal-check instrumentation and would
/// otherwise deadlock against a main thread that still holds the GIL. Safe
/// (idempotent) when the GIL is already released by an outer caller.
mlir::LogicalResult runPassManagerReleasingGIL(mlir::PassManager &pm,
                                               mlir::Operation *op);

} // namespace cudaq

#endif // CUDAQ_PYTHON_SIGNAL_CHECK_H
