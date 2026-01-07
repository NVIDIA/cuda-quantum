/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {

class ReplaceCallIndirect : public OpConversionPattern<func::CallIndirectOp> {
public:
  const SmallVector<StringRef> &names;
  // const llvm::DenseMap<std::size_t, std::size_t>& blockArgToNameMap;
  llvm::DenseMap<std::size_t, std::size_t> &blockArgToNameMap;

  ReplaceCallIndirect(MLIRContext *ctx,
                      const SmallVector<StringRef> &functionNames,
                      llvm::DenseMap<std::size_t, std::size_t> &map)
      : OpConversionPattern<func::CallIndirectOp>(ctx), names(functionNames),
        blockArgToNameMap(map) {}

  LogicalResult
  matchAndRewrite(func::CallIndirectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callableOperand = adaptor.getCallee();
    auto module = op->getParentOp()->getParentOfType<ModuleOp>();
    if (auto ccCallableFunc =
            callableOperand.getDefiningOp<cudaq::cc::CallableFuncOp>()) {
      if (auto blockArg =
              dyn_cast<BlockArgument>(ccCallableFunc.getOperand())) {
        auto argIdx = blockArg.getArgNumber();
        auto replacementName = names[blockArgToNameMap[argIdx]];
        auto replacement = module.lookupSymbol<func::FuncOp>(
            cudaq::runtime::cudaqGenPrefixName + replacementName.str());
        if (!replacement)
          return failure();

        rewriter.replaceOpWithNewOp<func::CallOp>(op, replacement,
                                                  adaptor.getCalleeOperands());
        rewriter.eraseOp(callableOperand.getDefiningOp());
        return success();
      }
    }
    return failure();
  }
};

class ReplaceCallCallable
    : public OpConversionPattern<cudaq::cc::CallCallableOp> {
public:
  const SmallVector<StringRef> &names;
  llvm::DenseMap<std::size_t, std::size_t> &blockArgToNameMap;

  ReplaceCallCallable(MLIRContext *ctx,
                      const SmallVector<StringRef> &functionNames,
                      llvm::DenseMap<std::size_t, std::size_t> &map)
      : OpConversionPattern<cudaq::cc::CallCallableOp>(ctx),
        names(functionNames), blockArgToNameMap(map) {}

  LogicalResult
  matchAndRewrite(cudaq::cc::CallCallableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callableOperand = adaptor.getCallee();
    auto module = op->getParentOp()->getParentOfType<ModuleOp>();
    if (auto blockArg = dyn_cast<BlockArgument>(callableOperand)) {
      auto argIdx = blockArg.getArgNumber();
      auto replacementName = names[blockArgToNameMap[argIdx]];
      auto replacement = module.lookupSymbol<func::FuncOp>(
          cudaq::runtime::cudaqGenPrefixName + replacementName.str());
      if (!replacement)
        return failure();

      rewriter.replaceOpWithNewOp<func::CallOp>(op, replacement,
                                                adaptor.getArgs());
      return success();
    }
    return failure();
  }
};

class UpdateQuakeApplyOp : public OpConversionPattern<quake::ApplyOp> {
public:
  const SmallVector<StringRef> &names;
  llvm::DenseMap<std::size_t, std::size_t> &blockArgToNameMap;
  UpdateQuakeApplyOp(MLIRContext *ctx,
                     const SmallVector<StringRef> &functionNames,
                     llvm::DenseMap<std::size_t, std::size_t> &map)
      : OpConversionPattern<quake::ApplyOp>(ctx), names(functionNames),
        blockArgToNameMap(map) {}

  LogicalResult
  matchAndRewrite(quake::ApplyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callableOperand = adaptor.getOperands().front();
    auto module = op->getParentOp()->getParentOfType<ModuleOp>();
    auto ctx = op.getContext();
    if (auto blockArg = dyn_cast<BlockArgument>(callableOperand)) {
      auto argIdx = blockArg.getArgNumber();
      auto replacementName = names[blockArgToNameMap[argIdx]];
      auto replacement = module.lookupSymbol<func::FuncOp>(
          cudaq::runtime::cudaqGenPrefixName + replacementName.str());
      if (!replacement)
        return failure();

      rewriter.replaceOpWithNewOp<quake::ApplyOp>(
          op, TypeRange{}, FlatSymbolRefAttr::get(ctx, replacement.getName()),
          adaptor.getIsAdj(), adaptor.getControls(), adaptor.getArgs());
      return success();
    }
    return failure();
  }
};

class PySynthCallableBlockArgs
    : public cudaq::opt::PySynthCallableBlockArgsBase<
          PySynthCallableBlockArgs> {
private:
  bool removeBlockArg = false;

public:
  SmallVector<StringRef> names;
  PySynthCallableBlockArgs(const SmallVector<StringRef> &_names, bool remove)
      : removeBlockArg(remove), names(_names) {}

  void runOnOperation() override {
    auto op = getOperation();
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);

    std::size_t numCallableBlockArgs = 0;
    // need to map blockArgIdx -> counter(0,1,2,...)
    llvm::DenseMap<std::size_t, std::size_t> blockArgToNamesMap;
    for (std::size_t i = 0, k = 0; auto ty : op.getFunctionType().getInputs()) {
      if (isa<cudaq::cc::CallableType>(ty)) {
        numCallableBlockArgs++;
        blockArgToNamesMap.insert({i, k++});
      }

      i++;
    }

    // Might not need to do any synthesis
    if (numCallableBlockArgs == 0)
      return;

    if (names.size() != numCallableBlockArgs) {
      emitError(op.getLoc(), "number of callable block arguments != number of "
                             "function names provided.");
      return;
    }

    patterns
        .insert<ReplaceCallIndirect, ReplaceCallCallable, UpdateQuakeApplyOp>(
            ctx, names, blockArgToNamesMap);
    ConversionTarget target(*ctx);
    // We should remove these operations
    target.addIllegalOp<func::CallIndirectOp>();
    target.addDynamicallyLegalOp<quake::ApplyOp>([](Operation *op) {
      if (auto apply = dyn_cast<quake::ApplyOp>(op)) {
        if (isa<BlockArgument>(apply.getOperand(0))) {
          return false;
        }
      }
      return true;
    });
    target.addLegalOp<func::CallOp>();
    if (failed(applyPartialConversion(op, target, std::move(patterns)))) {
      emitError(op.getLoc(),
                "error synthesizing callable functions for python.\n");
      signalPassFailure();
    }

    if (removeBlockArg) {
      auto numArgs = op.getNumArguments();
      BitVector argsToErase(numArgs);
      for (std::size_t argIndex = 0; argIndex < numArgs; ++argIndex)
        if (isa<cudaq::cc::CallableType>(op.getArgument(argIndex).getType()))
          argsToErase.set(argIndex);

      op.eraseArguments(argsToErase);
    }
  }
};
} // namespace

std::unique_ptr<Pass>
cudaq::opt::createPySynthCallableBlockArgs(const SmallVector<StringRef> &names,
                                           bool removeBlockArg) {
  return std::make_unique<PySynthCallableBlockArgs>(names, removeBlockArg);
}
