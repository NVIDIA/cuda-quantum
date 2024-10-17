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
#define GEN_PASS_DEF_STATEINITIALIZATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "state-initialization"

using namespace mlir;

namespace {

static bool isCall(Operation *callOp, std::vector<const char *> &&names) {
  if (callOp) {
    if (auto createStateCall = dyn_cast<func::CallOp>(callOp)) {
      if (auto calleeAttr = createStateCall.getCalleeAttr()) {
        auto funcName = calleeAttr.getValue().str();
        if (std::find(names.begin(), names.end(), funcName) != names.end())
          return true;
      }
    }
  }
  return false;
}

static bool isGetStateCall(Operation *callOp) {
  return isCall(callOp, {cudaq::getCudaqState});
}

static bool isNumberOfQubitsCall(Operation *callOp) {
  return isCall(callOp, {cudaq::getNumQubitsFromCudaqState});
}

// clang-format off
/// Replace `quake.init_state` by a call to a (modified) kernel that produced the state.
/// ```
///  %0 = cc.string_literal "callee.modified_0" : !cc.ptr<!cc.array<i8 x 27>>
///  %1 = cc.cast %0 : (!cc.ptr<!cc.array<i8 x 27>>) -> !cc.ptr<i8>
///  %2 = call @__nvqpp_cudaq_state_get(%1) : (!cc.ptr<i8>) -> !cc.ptr<!cc.state>
///  %3 = call @__nvqpp_cudaq_state_numberOfQubits(%2) : (!cc.ptr<!cc.state>) -> i64
///  %4 = quake.alloca !quake.veq<?>[%3 : i64]
///  %5 = quake.init_state %4, %2 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
/// ───────────────────────────────────────────
/// ...
///  %5 = call @callee.modified_0() : () -> !quake.veq<?>
/// ```
// clang-format on
class StateInitPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    auto loc = initState.getLoc();
    auto allocaOp = initState.getOperand(0).getDefiningOp();
    auto getStateOp = initState.getOperand(1).getDefiningOp();
    auto numOfQubits = allocaOp->getOperand(0).getDefiningOp();

    if (isGetStateCall(getStateOp)) {
      auto calleeNameOp = getStateOp->getOperand(0);
      if (auto cast =
              dyn_cast<cudaq::cc::CastOp>(calleeNameOp.getDefiningOp())) {
        calleeNameOp = cast.getOperand();

        if (auto literal = dyn_cast<cudaq::cc::CreateStringLiteralOp>(
                calleeNameOp.getDefiningOp())) {
          auto calleeName = literal.getStringLiteral();

          Value result =
              rewriter
                  .create<func::CallOp>(loc, initState.getType(), calleeName,
                                        mlir::ValueRange{})
                  .getResult(0);
          rewriter.replaceAllUsesWith(initState, result);
          initState.erase();
          allocaOp->dropAllUses();
          rewriter.eraseOp(allocaOp);
          if (isNumberOfQubitsCall(numOfQubits)) {
            numOfQubits->dropAllUses();
            rewriter.eraseOp(numOfQubits);
          }
          getStateOp->dropAllUses();
          rewriter.eraseOp(getStateOp);
          cast->dropAllUses();
          rewriter.eraseOp(cast);
          literal->dropAllUses();
          rewriter.eraseOp(literal);
          return success();
        }
      }
    }
    return failure();
  }
};

class StateInitializationPass
    : public cudaq::opt::impl::StateInitializationBase<
          StateInitializationPass> {
public:
  using StateInitializationBase::StateInitializationBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<StateInitPattern>(ctx);

    LLVM_DEBUG(llvm::dbgs() << "Before state initialization: " << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs() << "After state initialization: " << func << '\n');
  }
};
} // namespace
