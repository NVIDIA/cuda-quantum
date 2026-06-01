/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/TracePassInstrumentation.h"

#include "mlir/Pass/Pass.h"

#include <utility>

namespace cudaq {

TracePassInstrumentation::TracePassInstrumentation() {
  // Reserve for typical pass nesting depth to avoid per-push allocations.
  spanStack.reserve(8);
}

void TracePassInstrumentation::runBeforePass(mlir::Pass *pass,
                                             mlir::Operation *) {
  spanStack.push_back(Tracer::instance().beginSpan(
      TraceContext{}, pass->getName().str(), /*tag=*/0, {}, "mlir_pass"));
}

void TracePassInstrumentation::runAfterPass(mlir::Pass *, mlir::Operation *) {
  if (spanStack.empty())
    return;
  SpanHandle handle = std::move(spanStack.back());
  spanStack.pop_back();
  Tracer::instance().endSpan(std::move(handle));
}

void TracePassInstrumentation::runAfterPassFailed(mlir::Pass *pass,
                                                  mlir::Operation *op) {
  runAfterPass(pass, op);
}

} // namespace cudaq
