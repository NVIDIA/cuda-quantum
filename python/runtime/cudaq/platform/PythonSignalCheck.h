/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_PYTHON_SIGNAL_CHECK_H
#define CUDAQ_PYTHON_SIGNAL_CHECK_H

namespace mlir {
class PassManager;
}

namespace cudaq {

/// Add instrumentation that checks for pending Python signals between passes.
/// When a signal is pending, emits an MLIR error diagnostic to stop the
/// pipeline. The error message propagates through normal MLIR error handling.
void addPythonSignalInstrumentation(mlir::PassManager &pm);

} // namespace cudaq

#endif // CUDAQ_PYTHON_SIGNAL_CHECK_H
