/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

// CAPI entry point for pulse-canonicalize: exposes MLIR's greedy pattern
// rewrite driver to Python callers via a thin C API. It runs the same
// canonicalization every loaded op registers (e.g. the Pulse ops declared with
// `hasCanonicalizer = 1`), gathered the way MLIR's own `-canonicalize` pass
// does.

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

extern "C" {

MLIR_CAPI_EXPORTED MlirLogicalResult
cudaqPulseRunCanonicalize(MlirOperation op) {
  mlir::Operation *cppOp = unwrap(op);
  mlir::MLIRContext *context = cppOp->getContext();

  // Collect the canonicalization patterns registered by every loaded op,
  // exactly as the built-in canonicalizer pass does, so pulse ops with
  // `hasCanonicalizer = 1` actually fold here instead of this being a no-op.
  mlir::RewritePatternSet patterns(context);
  for (mlir::RegisteredOperationName registeredOp :
       context->getRegisteredOperations())
    registeredOp.getCanonicalizationPatterns(patterns, context);

  mlir::GreedyRewriteConfig config;
  auto result = mlir::applyPatternsGreedily(cppOp, std::move(patterns), config);
  MlirLogicalResult mlirResult;
  mlirResult.value = mlir::succeeded(result) ? 1 : 0;
  return mlirResult;
}

} // extern "C"
