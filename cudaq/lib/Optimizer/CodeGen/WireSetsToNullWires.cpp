/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_WIRESETTONULLWIRES
#include "cudaq/Optimizer/CodeGen/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "wireset-to-null-wires"

using namespace mlir;

namespace {

class WireSetToNullWiresPass
    : public cudaq::opt::impl::WireSetToNullWiresBase<WireSetToNullWiresPass> {
public:
  using WireSetToNullWiresBase::WireSetToNullWiresBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto wireSets = llvm::to_vector(module.getOps<cudaq::quake::WireSetOp>());
    if (wireSets.empty())
      return;
    auto wireTy = cudaq::quake::WireType::get(&getContext());
    IRRewriter rewriter(&getContext());
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.empty())
        continue;
      // Move the entry block's borrows to its start in identity order. A
      // borrow inside control flow stays where it is, so it gets a fresh qubit
      // each time it runs.
      Block &entry = func.getBody().front();
      auto entryBorrows =
          llvm::to_vector(entry.getOps<cudaq::quake::BorrowWireOp>());
      llvm::stable_sort(entryBorrows, [](auto a, auto b) {
        return std::make_pair(a.getSetName(), a.getIdentity()) <
               std::make_pair(b.getSetName(), b.getIdentity());
      });
      for (auto borrow : llvm::reverse(entryBorrows))
        borrow->moveBefore(&entry, entry.begin());
      func.walk([&](Operation *op) {
        rewriter.setInsertionPoint(op);
        if (auto borrow = dyn_cast<cudaq::quake::BorrowWireOp>(op))
          rewriter.replaceOpWithNewOp<cudaq::quake::NullWireOp>(borrow, wireTy);
        else if (auto ret = dyn_cast<cudaq::quake::ReturnWireOp>(op))
          rewriter.replaceOpWithNewOp<cudaq::quake::SinkOp>(ret, TypeRange{},
                                                            ret.getTarget());
      });
    }
    for (auto wireSet : wireSets)
      rewriter.eraseOp(wireSet);
  }
};
} // namespace
