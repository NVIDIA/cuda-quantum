/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "llvm/ADT/SmallSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ERASEDETECTORS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "erase-detectors"

using namespace mlir;

/// \file
/// This pass exists simply to remove all the quake.detector Ops from the IR.

namespace {
class EraseDetectorPattern : public OpRewritePattern<quake::DetectorOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DetectorOp detector,
                                PatternRewriter &rewriter) const override {
    // clang-format off
    // Example IR produced by the bridge:
    // %2 = cc.alloca !cc.array<i64 x 2>
    // %3 = cc.cast %2 : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<!cc.array<i64 x ?>>
    // %4 = cc.cast %2 : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<i64>
    // cc.store %c-1_i64, %4 : !cc.ptr<i64>
    // %5 = cc.compute_ptr %2[1] : (!cc.ptr<!cc.array<i64 x 2>>) -> !cc.ptr<i64>
    // cc.store %c-2_i64, %5 : !cc.ptr<i64>
    // %6 = cc.stdvec_init %3, %c2_i64 : (!cc.ptr<!cc.array<i64 x ?>>, i64) -> !cc.stdvec<i64>
    // "quake.detector"(%6) : (!cc.stdvec<i64>) -> ()
    // clang-format on

    // TODO - decide whether or not to keep this path.
    // First determine if the arguments originate from cc.stdvec_init
    // operations.
    if (auto stdvecInit = dyn_cast<cudaq::cc::StdvecInitOp>(
            detector.getOperands()[0].getDefiningOp())) {
      Operation *op = stdvecInit;
      // Find the owning alloca operation. Walk up the operand chain until we
      // find an alloca operation or the operand chain is empty.
      while (op) {
        if (isa<cudaq::cc::AllocaOp>(op) || op->getNumOperands() == 0)
          break;
        op = op->getOperands()[0].getDefiningOp();
      }
      if (auto alloca = dyn_cast_if_present<cudaq::cc::AllocaOp>(op)) {
        // Erase the alloca operation and all its users.
        SmallVector<Value> worklist;
        SmallVector<Operation *> toErase;
        worklist.push_back(alloca);
        toErase.push_back(alloca);
        llvm::SmallSet<Operation *, 16> visited;
        bool contains_loads = false;
        while (!worklist.empty()) {
          auto value = worklist.pop_back_val();
          visited.insert(value.getDefiningOp());
          if (isa<cudaq::cc::LoadOp>(value.getDefiningOp())) {
            contains_loads = true;
            break;
          }
          for (auto user : value.getUsers()) {
            toErase.push_back(user);
            for (Value result : user->getResults())
              if (!visited.contains(result.getDefiningOp()))
                worklist.push_back(result);
          }
        }
        if (contains_loads) {
          // Just erase the detector operation, leaving other operations that
          // are still used.
          rewriter.eraseOp(detector);
        } else {
          // Process toErase in reverse order.
          for (auto op : llvm::reverse(toErase))
            rewriter.eraseOp(op);
        }
      }
    } else {
      rewriter.eraseOp(detector);
    }
    return success();
  }
};

class EraseDetectorsPass
    : public cudaq::opt::impl::EraseDetectorsBase<EraseDetectorsPass> {
public:
  using EraseDetectorsBase::EraseDetectorsBase;

  void runOnOperation() override {
    auto *op = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Before erasure:\n" << *op << "\n\n");
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    patterns.insert<EraseDetectorPattern>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
    LLVM_DEBUG(llvm::dbgs() << "After erasure:\n" << *op << "\n\n");
  }
};
} // namespace
