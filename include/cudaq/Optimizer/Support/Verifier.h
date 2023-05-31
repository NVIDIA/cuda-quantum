/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

namespace cudaq {

/// A verification pass to verify the output from the bridge. This provides a
/// little bit of glue to run a verifier pass directly.
class VerifierPass
    : public mlir::PassWrapper<VerifierPass, mlir::OperationPass<>> {
  void runOnOperation() override final {
    if (mlir::failed(mlir::verify(getOperation())))
      signalPassFailure();
    markAllAnalysesPreserved();
  }
};

} // namespace cudaq
