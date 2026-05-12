/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_RUNSEMANTICSHACKERY
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "run-semantics-hackery"

using namespace mlir;

namespace {
class RemoveVectorCopies : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp call,
                                PatternRewriter &rewriter) const override {
    if (call.getCallee() != "__nvqpp_vectorCopyCtor")
      return failure();
    rewriter.replaceOp(call, call.getOperand(0));
    return success();
  }
};

/// This pass will further break the semantics of return values for expedience.
/// Hopefully, this code will become obsolete and get removed.
///
/// When the return value has a dynamic size (and we do not mean the object
/// header information of the return value!), the data cannot be returned on the
/// stack as the activation frame unwinds. Thus the data must be allocated and
/// copied to the heap.
class RunSemanticsHackeryPass
    : public cudaq::opt::impl::RunSemanticsHackeryBase<
          RunSemanticsHackeryPass> {
public:
  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // 1. Analyze mod to see if we have run entry points. Those run entry points
    // are calling to original kernel. (The original kernel has not been inlined
    // already.) Finally, we're looking for original kernels that return
    // `std::vector` results.
    SmallVector<std::pair<func::CallOp, func::FuncOp>> worklist;
    mod.walk([&](func::FuncOp fn) {
      StringRef name = fn.getSymName();
      if (!name.ends_with(".run"))
        return;
      if (fn.getBody().empty())
        return;
      auto callOp = dyn_cast<func::CallOp>(fn.getBody().front().front());
      if (!callOp)
        return;
      StringRef calleeName = callOp.getCallee();
      if (calleeName.str() + ".run" != name)
        return;
      auto called = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          fn, callOp.getCalleeAttr());
      auto calledFnTy = called.getFunctionType();
      if (!(calledFnTy.getResults().size() == 1 &&
            isa<cudaq::cc::StdvecType>(calledFnTy.getResult(0))))
        return;
      worklist.emplace_back(callOp, called);
      LLVM_DEBUG(llvm::dbgs() << "adding kernel: " << name << '\n');
    });

    if (worklist.empty())
      return;

    // 2. For run entry points that meet the requirements of (1) above, we want
    // to evict the copy-to-heap operation in the original kernel body. This is
    // because run, in a deeply incorrect way, does not in fact return anything,
    // so copying the result to the stack turns out to be the wrong thing to do.
    // (beat our heads on a wall here) This implies that we need to call a clone
    // of the original kernel where the cloned code is rewritten to surgically
    // remove calls to `__nvqpp_vectorCopyCtor`. It is deeply incorrect to
    // remove that call from the original kernel as it is not semantically
    // broken and should not be warped in ad hoc ways.
    auto *ctx = mod.getContext();
    for (auto [callOp, origFn] : worklist) {
      std::string ruinedName = callOp.getCallee().str() + ".ruined";
      auto ruinedNameAttr = StringAttr::get(ctx, ruinedName);
      auto foundIt = SymbolTable::lookupNearestSymbolFrom<func::FuncOp>(
          origFn, ruinedNameAttr);
      if (foundIt)
        continue;
      auto ruinedFn = cast<func::FuncOp>(origFn->clone());
      ruinedFn.setName(ruinedName);
      mod.getBody()->push_back(ruinedFn);
      // Rewrite the body of the ruined function.
      RewritePatternSet patterns(ctx);
      patterns.insert<RemoveVectorCopies>(ctx);
      if (failed(applyPatternsGreedily(ruinedFn, std::move(patterns))))
        return;

      // Rewrite the call to call the new ruined function.
      callOp.setCallee(ruinedName);
      LLVM_DEBUG(llvm::dbgs() << "updated to call: " << ruinedName << '\n');
    }
  }
};
} // namespace
