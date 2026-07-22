// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// CAPI entry point for pulse-canonicalize: exposes MLIR's
// applyPatternsAndFoldGreedily to Python callers via a thin C API.

#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

extern "C" {

MLIR_CAPI_EXPORTED MlirLogicalResult
cudaqPulseRunCanonicalize(MlirOperation op) {
  mlir::Operation *cppOp = unwrap(op);
  mlir::RewritePatternSet patterns(cppOp->getContext());

  mlir::GreedyRewriteConfig config;
  auto result = mlir::applyPatternsGreedily(cppOp, std::move(patterns), config);
  MlirLogicalResult mlirResult;
  mlirResult.value = mlir::succeeded(result) ? 1 : 0;
  return mlirResult;
}

} // extern "C"
