/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_INJECTIMPLICITOUTPUT
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "inject-implicit-output"

using namespace mlir;

namespace {

class InjectImplicitOutputPass
    : public cudaq::opt::impl::InjectImplicitOutputBase<
          InjectImplicitOutputPass> {
public:
  using InjectImplicitOutputBase::InjectImplicitOutputBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;
    if (!funcOp->hasAttr(cudaq::entryPointAttrName))
      return;
    // If the kernel already contains any quake.log_output ops at the top level,
    // it has output: do not inject implicit output.
    for (Block &block : funcOp.getBody())
      for (Operation &op : block)
        if (isa<cudaq::quake::LogOutputOp, cudaq::quake::DeallocOp>(op)) {
          LLVM_DEBUG({
            if (isa<cudaq::quake::LogOutputOp>(op))
              llvm::dbgs() << "kernel already has log_output ops, skipping.\n";
            else
              llvm::dbgs() << "kernel already has deallocs, skipping.\n";
          });
          return;
        }

    // At this point, we have a function that we want to process.

    // 1. Collect quake.alloca ops at the top level of this func.func only.
    // Insertion order is preserved.
    SmallVector<cudaq::quake::AllocaOp> orderedAllocs;
    for (Block &block : funcOp.getBody())
      for (Operation &op : block)
        if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op))
          orderedAllocs.push_back(alloc);

    // If this is a degenerate kernel without qubits, exit now.
    if (orderedAllocs.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no top-level allocas, nothing to inject\n");
      return;
    }

    // λ to resolve the value to create a log_output for.
    auto logValue = [](cudaq::quake::AllocaOp alloc) -> Value {
      if (alloc.hasInitializedState())
        return alloc.getInitializedState().getResult();
      return alloc.getResult();
    };

    // λ to create a quake.log_output marked as compiler-generated.
    auto emitLogOutput = [&](OpBuilder &b, Location l, Value v) {
      auto op = cudaq::quake::LogOutputOp::create(b, l, ValueRange{v});
      op->setAttr(cudaq::runtime::compilerGeneratedOutput, b.getUnitAttr());
    };

    DominanceInfo dom(funcOp);
    auto loc = funcOp.getLoc();
    OpBuilder builder(funcOp.getContext());

    // 2. Create a new shared exit block. Its arguments match the function's
    // return type(s) and the original return(s) thread those values as block
    // arguments.
    auto returnTypes = funcOp.getFunctionType().getResults();
    SmallVector<Location> returnLocs(returnTypes.size(), loc);
    auto *exitBlock = new Block;
    exitBlock->addArguments(returnTypes, returnLocs);
    funcOp.getBody().push_back(exitBlock);

    // 3. Emit quake.log_output ops in declaration order into the new exit
    // block.
    builder.setInsertionPointToEnd(exitBlock);
    for (auto alloc : orderedAllocs)
      emitLogOutput(builder, alloc.getLoc(), logValue(alloc));
    func::ReturnOp::create(builder, loc, exitBlock->getArguments());

    // 3. Redirect all existing func.return ops to branch to the exit block,
    // provided the alloca set entirely dominates that return. Otherwise, emit
    // log_output ops for the dominating subset inline before the return.
    for (Block &block : funcOp.getBody()) {
      // Skip the exit block we just created.
      if (&block == exitBlock)
        continue;
      for (Operation &op : llvm::make_early_inc_range(block)) {
        auto ret = dyn_cast<func::ReturnOp>(op);
        if (!ret)
          continue;

        bool allDominate = llvm::all_of(orderedAllocs, [&](Operation *alloc) {
          return dom.dominates(alloc, ret.getOperation());
        });

        builder.setInsertionPoint(ret);
        if (allDominate) {
          cf::BranchOp::create(builder, ret.getLoc(), exitBlock,
                               ret.getOperands());
          ret.erase();
        } else {
          // Fall back case. Emit log_output only for the allocas that dominate
          // this return, preserving declaration order.
          for (auto alloc : orderedAllocs)
            if (dom.dominates(alloc.getOperation(), ret.getOperation()))
              emitLogOutput(builder, alloc.getLoc(), logValue(alloc));
        }
      }
    }
  }
};
} // namespace
