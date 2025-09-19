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

static SmallVector<Operation *, 8> sortUsers(const Value::user_range &users,
                                             const DominanceInfo &dom) {
  SmallVector<Operation *, 8> orderedUsers;

  for (auto *user : users) {
    if ([&]() {
          for (auto iter = orderedUsers.begin(), iterEnd = orderedUsers.end();
               iter != iterEnd; ++iter) {
            assert(*iter);
            if (dom.dominates(user, *iter)) {
              orderedUsers.insert(iter, user);
              return false;
            }
          }
          return true;
        }())
      orderedUsers.push_back(user);
  }
  return orderedUsers;
}

class QubitResetBeforeReusePass
    : public cudaq::opt::QubitResetBeforeReuseBase<QubitResetBeforeReusePass> {
public:
  QubitResetBeforeReusePass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (!funcOp || funcOp.empty())
      return;
    DominanceInfo dom(funcOp);

    // Return the next use after the op
    const auto getNextUse = [&dom](Value qubit, Operation *op) -> Operation * {
      {
        // Check direct use
        const auto orderedUsers = sortUsers(qubit.getUsers(), dom);
        assert(orderedUsers.size() > 0);
        for (auto v : llvm::enumerate(orderedUsers)) {
          if (v.value() == op && v.index() < (orderedUsers.size() - 1)) {
            return orderedUsers[v.index() + 1];
          }
        }
      }

      // No next use is found, check if this is an extracted qubit.
      if (isa<quake::RefType>(qubit.getType())) {
        if (auto extractOp = dyn_cast_if_present<quake::ExtractRefOp>(
                qubit.getDefiningOp())) {
          llvm::outs() << "Defining op: " << *extractOp << "\n";
          auto reg = extractOp.getVeq();
          std::optional<int64_t> index = extractOp.hasConstantIndex()
                                             ? extractOp.getConstantIndex()
                                             : std::optional<int64_t>();
          llvm::outs() << "Reg: " << reg << "; index = " << index.value_or(-1)
                       << "\n";

          const auto orderedUsers = sortUsers(reg.getUsers(), dom);
          // Find the next op on the register
          assert(orderedUsers.size() > 0);
          bool foundThisExtract = false;
          for (auto v : llvm::enumerate(orderedUsers)) {
            if (foundThisExtract) {
              // This is after the current extract.
              auto nextExtractOp =
                  dyn_cast_or_null<quake::ExtractRefOp>(v.value());
              assert(nextExtractOp);
              assert(nextExtractOp.getVeq() == reg);
              std::optional<int64_t> nextIndex =
                  nextExtractOp.hasConstantIndex()
                      ? nextExtractOp.getConstantIndex()
                      : std::optional<int64_t>();
              if ((!index.has_value() || !nextIndex.has_value()) ||
                  (index == nextIndex)) {
                // Either the previous index or this index is unknown, we assume
                // that they may be the same.
                const auto extractedQubit = nextExtractOp.getRef();
                const auto extractedQubitOrderedUsers =
                    sortUsers(extractedQubit.getUsers(), dom);
                assert(!extractedQubitOrderedUsers.empty());
                llvm::outs()
                    << "Next use: " << *extractedQubitOrderedUsers[0] << "\n";
                return extractedQubitOrderedUsers[0];
              }
            }
            if (v.value() == extractOp) {
              foundThisExtract = true;
            }
          }
          assert(foundThisExtract);
        }
      }
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
          mlir::Value measBit = [&]() {
            for (auto *out : measOut.getUsers()) {
              // A mz may be accompanied by a store op, find that op.
              if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out)) {
                builder.setInsertionPointAfter(disc);
                return disc.getResult();
              }
            }
            // No discriminate exists - create the discriminate Op
            auto discOp = builder.create<quake::DiscriminateOp>(
                loc, builder.getI1Type(), measOut);
            return discOp.getResult();
          }();
          builder.create<cudaq::cc::IfOp>(
              loc, TypeRange{}, measBit,
              [&](OpBuilder &opBuilder, Location location, Region &region) {
                region.push_back(new Block{});
                auto &bodyBlock = region.front();
                OpBuilder::InsertionGuard guad(opBuilder);
                opBuilder.setInsertionPointToStart(&bodyBlock);
                opBuilder.create<quake::XOp>(location, measuredQubit);
                opBuilder.create<cudaq::cc::ContinueOp>(location);
              });
        } else {
          llvm::outs() << "No next use\n";
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
