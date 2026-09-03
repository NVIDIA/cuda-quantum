/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
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
    // If the kernel already contains any quake.evince ops at the top level,
    // it has output: do not inject implicit output.
    //
    // Also collect quake.alloca ops at the top level of this func.func only.
    // Insertion order is preserved.
    SmallVector<cudaq::quake::AllocaOp> orderedAllocs;
    for (Block &block : funcOp.getBody())
      for (Operation &op : block) {
        if (auto alloc = dyn_cast<cudaq::quake::AllocaOp>(op)) {
          orderedAllocs.push_back(alloc);
        } else if (isa<cudaq::quake::EvinceOp, cudaq::quake::DeallocOp>(op)) {
          LLVM_DEBUG({
            if (isa<cudaq::quake::EvinceOp>(op))
              llvm::dbgs() << "kernel already has evince ops, skipping.\n";
            else
              llvm::dbgs() << "kernel already has deallocs, skipping.\n";
          });
          return;
        }
      }

    // If this is a degenerate kernel without qubits, exit now.
    if (orderedAllocs.empty()) {
      LLVM_DEBUG(llvm::dbgs() << "no top-level allocas, nothing to inject\n");
      return;
    }

    // 1. At this point, we have a function that we want to process.

    // λ to resolve the value to create an evince for.
    auto logValue = [](cudaq::quake::AllocaOp alloc) -> Value {
      if (alloc.hasInitializedState())
        return alloc.getInitializedState().getResult();
      return alloc.getResult();
    };

    // λ to create compiler-generated evince ops for a single value.
    // Passes /*compilerGenerated=*/true to the builder so the attribute is
    // stored as a first-class op attribute rather than a discardable side attr.
    auto emitEvince = [&](OpBuilder &b, Location l, Value v) {
      // Emit a single evince for the value regardless of whether it is a
      // ref or a veq.  Using evince %veq directly (rather than a per-
      // element loop) is essential: the DQE pass recognises evince as the
      // keep-alive signal for a veq alloca.  A loop of extract_ref + evince
      // %ref would hide the veq from DQE, causing the alloca to be treated as
      // dead and eliminated.  evince %veq also avoids the null-wire double-
      // use issue in memtoreg because EvinceOpPattern only fires on !ref
      // operands (hasNonVectorReference returns false for veq).
      cudaq::quake::EvinceOp::create(b, l, ValueRange{v},
                                     /*compilerGenerated=*/true);
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

    // 3. Emit quake.evince ops in declaration order into the new exit
    // block.
    builder.setInsertionPointToEnd(exitBlock);
    for (auto alloc : orderedAllocs)
      emitEvince(builder, alloc.getLoc(), logValue(alloc));
    func::ReturnOp::create(builder, loc, exitBlock->getArguments());

    // 3. Redirect all existing func.return ops to branch to the exit block,
    // provided the alloca set entirely dominates that return. Otherwise, emit
    // evince ops for the dominating subset inline before the return.
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
          // Fall back case. Emit evince only for the allocas that dominate
          // this return, preserving declaration order.
          for (auto alloc : orderedAllocs)
            if (dom.dominates(alloc.getOperation(), ret.getOperation()))
              emitEvince(builder, alloc.getLoc(), logValue(alloc));
        }
      }
    }
  }
};
} // namespace
