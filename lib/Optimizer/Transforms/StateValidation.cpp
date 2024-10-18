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

namespace cudaq::opt {
#define GEN_PASS_DEF_STATEVALIDATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "state-validation"

using namespace mlir;


/// Validate that quantum code does not contain runtime calls and remove runtime function definitions. 
namespace {

static bool isRuntimeStateCallName(llvm::StringRef funcName) {
  static std::vector<const char *> names = {
    cudaq::getCudaqState,
    cudaq::createCudaqStateFromDataFP32,
    cudaq::createCudaqStateFromDataFP64,
    cudaq::deleteCudaqState,
    cudaq::getNumQubitsFromCudaqState
  };
  if (std::find(names.begin(), names.end(), funcName) != names.end())
      return true; 
  return false;
}

static bool isRuntimeStateCall(Operation *callOp) {
  if (callOp) {
    if (auto call = dyn_cast<func::CallOp>(callOp)) {
      if (auto calleeAttr = call.getCalleeAttr()) {
        auto funcName = calleeAttr.getValue().str();
        if (isRuntimeStateCallName(funcName))
          return true;
      }
    }
  }
  return false;
}

class ValidateStateCallPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp,
                                PatternRewriter &rewriter) const override {
    if (isRuntimeStateCall(callOp)) {
      auto name = callOp.getCalleeAttr().getValue();
      callOp.emitError("Unsupported call for quantum platform: " + name);
    }
    return failure();
  }
};

class ValidateStateInitPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    auto stateOp = initState.getOperand(1);
    if (isa<cudaq::cc::StateType>(stateOp.getType())) 
      initState.emitError("Synthesis did not remove `quake.init_state <state>` instruction");
    
    return failure();
  }
};


class StateValidationPass
    : public cudaq::opt::impl::StateValidationBase<StateValidationPass> {
protected:
public:
  using StateValidationBase::StateValidationBase;

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto *ctx = &getContext();
    auto module = getModule();
    SmallVector<Operation *> toErase;

    for (Operation &op : *module.getBody()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func)
        continue;

      RewritePatternSet patterns(ctx);
      patterns.insert<ValidateStateCallPattern, ValidateStateInitPattern>(ctx);

      LLVM_DEBUG(llvm::dbgs()
                 << "Before state validation: " << func << '\n');

      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns))))
        signalPassFailure();

      // Delete runtime function definitions.
      if (func.getBody().empty() && isRuntimeStateCallName(func.getName()))
        toErase.push_back(func);

      LLVM_DEBUG(llvm::dbgs()
                 << "After state validation: " << func << '\n');
    }

    for (auto *op : toErase)
      op->erase();
  }
};

} // namespace
