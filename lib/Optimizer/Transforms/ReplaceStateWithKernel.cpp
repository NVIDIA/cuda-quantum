/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <span>

namespace cudaq::opt {
#define GEN_PASS_DEF_REPLACESTATEWITHKERNEL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "replace-state-with-kernel"

using namespace mlir;

namespace {

static bool isCall(Operation *op, std::vector<const char *> &&names) {
  if (op) {
    if (auto callOp = dyn_cast<func::CallOp>(op)) {
      if (auto calleeAttr = callOp.getCalleeAttr()) {
        auto funcName = calleeAttr.getValue().str();
        if (std::find(names.begin(), names.end(), funcName) != names.end())
          return true;
      }
    }
  }
  return false;
}

static bool isNumberOfQubitsCall(Operation *op) {
  return isCall(op, {cudaq::getNumQubitsFromCudaqState});
}

// clang-format off
/// Replace `quake.init_state` by a call to a (modified) kernel that produced
/// the state.
///
/// ```
///  %0 = cc.get_state "__nvqpp__mlirgen__test_init_state.modified_0" : !cc.ptr<!cc.state>
///  %1 = call @__nvqpp_cudaq_state_numberOfQubits(%0) : (!cc.ptr<!cc.state>) -> i64
///  %2 = quake.alloca !quake.veq<?>[%1 : i64]
///  %3 = quake.init_state %2, %0 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
/// ───────────────────────────────────────────
/// ...
///  %5 = call @callee.modified_0() : () -> !quake.veq<?>
/// ```
// clang-format on
class ReplaceStateWithKernelPattern
    : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    auto *alloca = initState.getOperand(0).getDefiningOp();
    auto stateOp = initState.getOperand(1);

    if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(stateOp.getType())) {
      if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
        auto *numOfQubits = alloca->getOperand(0).getDefiningOp();
        stateOp.getDefiningOp()->dump();

        if (auto getState = stateOp.getDefiningOp<cudaq::cc::GetStateOp>()) {
          auto calleeName = getState.getCalleeName();
          rewriter.replaceOpWithNewOp<func::CallOp>(
              initState, initState.getType(), calleeName, mlir::ValueRange{});

          if (alloca->getUses().empty())
            rewriter.eraseOp(alloca);
          else {
            alloca->emitError(
                "Failed to remove `quake.alloca` in state synthesis");
            return failure();
          }
          if (isNumberOfQubitsCall(numOfQubits)) {
            if (numOfQubits->getUses().empty())
              rewriter.eraseOp(numOfQubits);
            else {
              numOfQubits->emitError("Failed to remove runtime call to get "
                                     "number of qubits in state synthesis");
              return failure();
            }
          }
          return success();
        }
        numOfQubits->emitError(
            "Failed to replace `quake.init_state` in state synthesis");
      }
    }
    return failure();
  }
};

class ReplaceStateWithKernelPass
    : public cudaq::opt::impl::ReplaceStateWithKernelBase<
          ReplaceStateWithKernelPass> {
public:
  using ReplaceStateWithKernelBase::ReplaceStateWithKernelBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceStateWithKernelPattern>(ctx);

    LLVM_DEBUG(llvm::dbgs()
               << "Before replace state with kernel: " << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "After replace state with kerenl: " << func << '\n');
  }
};
} // namespace
