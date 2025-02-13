/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_WRITEAFTERWRITEELIMINATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "write-after-write-elimination"

using namespace mlir;

#include "WriteAfterWriteEliminationPatterns.inc"

namespace {
class WriteAfterWriteEliminationPass
    : public cudaq::opt::impl::WriteAfterWriteEliminationBase<
          WriteAfterWriteEliminationPass> {
public:
  using WriteAfterWriteEliminationBase::WriteAfterWriteEliminationBase;

  void runOnOperation() override {
    auto op = getOperation();
    DominanceInfo domInfo(op);

    LLVM_DEBUG(llvm::dbgs()
               << "Before write after write elimination: " << *op << '\n');

    auto analysis = SimplifyWritesAnalysis(domInfo, op);
    analysis.removeOverriddenStores();

    LLVM_DEBUG(llvm::dbgs()
               << "After write after write elimination: " << *op << '\n');
  }
};
} // namespace
