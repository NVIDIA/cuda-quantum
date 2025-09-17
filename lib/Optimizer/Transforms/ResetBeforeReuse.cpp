/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Todo.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

#define DEBUG_TYPE "reset-before-reuse"

namespace {

class QubitResetBeforeReusePass
    : public cudaq::opt::QubitResetBeforeReuseBase<QubitResetBeforeReusePass> {
public:
  QubitResetBeforeReusePass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;

    // Return the next use after the op
    const auto getNextUse = [](Value qubit, Operation *op) -> Operation * {
      Operation *result = nullptr;
      for (auto *useOp : qubit.getUsers()) {
        if (useOp == op)
          return result;
        result = useOp;
      }
      assert(false); // The anchor op is not using the qubit!
      return nullptr;
    };
    OpBuilder builder(funcOp);
    funcOp->walk([&](quake::MzOp mz) {
      SmallVector<Operation *> useOps;
      for (Value measuredQubit : mz.getTargets()) {
        auto *nextOp = getNextUse(measuredQubit, mz);
        if (nextOp) {
          // If the user is a reset op, nothing to do.
          if (isa<quake::ResetOp>(nextOp)) {
            continue;
          }
          // Insert reset
          Location loc = mz->getLoc();
          builder.setInsertionPointAfter(mz);
          builder.create<quake::ResetOp>(loc, TypeRange{}, measuredQubit);
          // Insert a conditional X to initialize qubit after reset.
          auto measOut = mz.getMeasOut();
          for (auto *out : measOut.getUsers()) {
            // A mz should be accompanied by a store op, find that op.
            if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out)) {
              auto bit = disc.getResult();
              for (auto *bitUser : bit.getUsers()) {
                if (auto store =
                        dyn_cast_if_present<cudaq::cc::StoreOp>(bitUser)) {
                  builder.setInsertionPointAfter(store);
                  auto conditionalVar = builder.create<cudaq::cc::LoadOp>(
                      loc, store.getPtrvalue());
                  builder.create<cudaq::cc::IfOp>(
                      loc, TypeRange{}, conditionalVar,
                      [&](OpBuilder &opBuilder, Location location,
                          Region &region) {
                        region.push_back(new Block{});
                        auto &bodyBlock = region.front();
                        OpBuilder::InsertionGuard guad(opBuilder);
                        opBuilder.setInsertionPointToStart(&bodyBlock);
                        opBuilder.create<quake::XOp>(location, measuredQubit);
                        opBuilder.create<cudaq::cc::ContinueOp>(location);
                      });
                }
              }
            }
          }
        }
      }

      return WalkResult::advance();
    });
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createQubitResetBeforeReuse() {
  return std::make_unique<QubitResetBeforeReusePass>();
}
