/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/runtime/logger/tracer.h"

#include "mlir/Pass/PassInstrumentation.h"

#include <vector>

namespace cudaq {

class TracePassInstrumentation : public mlir::PassInstrumentation {
public:
  TracePassInstrumentation();

  void runBeforePass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPass(mlir::Pass *pass, mlir::Operation *op) override;
  void runAfterPassFailed(mlir::Pass *pass, mlir::Operation *op) override;

private:
  std::vector<SpanHandle> spanStack;
};

} // namespace cudaq
