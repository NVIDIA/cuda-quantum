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
    auto wireTy = cudaq::quake::WireType::get(module.getContext());
    for (auto func : module.getOps<func::FuncOp>()) {
      if (func.empty())
        continue;
      SmallVector<cudaq::quake::BorrowWireOp> borrows;
      func.walk([&](cudaq::quake::BorrowWireOp borrow) {
        borrows.push_back(borrow);
      });
      llvm::stable_sort(borrows, [](auto a, auto b) {
        return std::make_pair(a.getSetName(), a.getIdentity()) <
               std::make_pair(b.getSetName(), b.getIdentity());
      });
      // Prepending the entry-block null wires in reverse leaves them in
      // identity order.
      Block &entry = func.getBody().front();
      for (auto borrow : llvm::reverse(borrows)) {
        OpBuilder builder(borrow);
        if (borrow->getBlock() == &entry)
          builder.setInsertionPointToStart(&entry);
        Value nullWire =
            cudaq::quake::NullWireOp::create(builder, borrow.getLoc(), wireTy);
        borrow.replaceAllUsesWith(nullWire);
        borrow.erase();
      }
      func.walk([](cudaq::quake::ReturnWireOp ret) {
        OpBuilder builder(ret);
        cudaq::quake::SinkOp::create(builder, ret.getLoc(), TypeRange{},
                                     ret.getTarget());
        ret.erase();
      });
    }
    for (auto wireSet : wireSets)
      wireSet.erase();
  }
};
} // namespace
