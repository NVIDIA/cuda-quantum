/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/CodeGen/Emitter.h"
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
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_QUBITRESETBEFOREREUSE
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "reset-before-reuse"

using namespace mlir;

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

// Track qubit register use chains.
// This is used to track if a qubit is reused after it has been measured across
// different extract ops. We cache the sorted order upfront for efficiency, but
// filter against current users at query time to handle operations that may be
// erased during pattern rewriting.
class RegUseTracker {
  mlir::DenseMap<mlir::Value, SmallVector<Operation *, 8>> regToOrderedUsers;
  DominanceInfo domInfo;

public:
  RegUseTracker(func::FuncOp func) : domInfo(func) {
    func->walk([&](quake::AllocaOp qalloc) {
      regToOrderedUsers[qalloc.getResult()] =
          sortUsers(qalloc.getResult().getUsers(), domInfo);
    });
  }

  // Returns users in dominance order by iterating through the cached sorted
  // list (regToOrderedUsers) and filtering out any operations that have been
  // erased during rewriting to avoid use-after-free bugs.
  SmallVector<Operation *, 8> getUsers(mlir::Value qreg) const {
    if (!isa<quake::VeqType>(qreg.getType()))
      mlir::emitError(qreg.getLoc(),
                      "Unexpected type used: expected a quake::VeqType.");

    auto iter = regToOrderedUsers.find(qreg);
    if (iter == regToOrderedUsers.end())
      return {};

    // Filter cached users against current users to handle erased operations.
    llvm::DenseSet<Operation *> currentUsers(qreg.getUsers().begin(),
                                             qreg.getUsers().end());
    SmallVector<Operation *, 8> validUsers;
    for (auto *op : iter->second) {
      if (currentUsers.contains(op))
        validUsers.push_back(op);
    }
    return validUsers;
  }
  DominanceInfo &getDominanceInfo() { return domInfo; }
  RegUseTracker(const RegUseTracker &) = delete;
  RegUseTracker(RegUseTracker &&) = delete;
  RegUseTracker &operator=(const RegUseTracker &) = delete;
};

class ResetAfterMeasurePattern : public OpRewritePattern<quake::MzOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  explicit ResetAfterMeasurePattern(MLIRContext *ctx, RegUseTracker &tracker)
      : OpRewritePattern(ctx), tracker(tracker) {}

  LogicalResult matchAndRewrite(quake::MzOp mz,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> useOps;
    bool modified = false;
    for (Value measuredQubit : mz.getTargets()) {
      auto *nextOp = getNextUse(measuredQubit, mz);
      if (nextOp) {
        // If the user is a reset/measure op, nothing to do.
        if (isa<quake::ResetOp>(nextOp) || isa<quake::MzOp>(nextOp)) {
          continue;
        }

        // If this is a dealloc op, nothing to do.
        if (isa<quake::DeallocOp>(nextOp)) {
          continue;
        }

        // Insert reset
        Location loc = mz->getLoc();
        rewriter.setInsertionPointAfter(mz);
        rewriter.create<quake::ResetOp>(loc, TypeRange{}, measuredQubit);
        // Insert a conditional X to initialize qubit after reset.
        auto measOut = mz.getMeasOut();
        mlir::Value measBit = [&]() {
          for (auto *out : measOut.getUsers()) {
            // A mz may be accompanied by a store op, find that op.
            if (auto disc = dyn_cast_if_present<quake::DiscriminateOp>(out)) {
              rewriter.setInsertionPointAfter(disc);
              return disc.getResult();
            }
          }
          // No discriminate exists - create the discriminate Op
          auto discOp = rewriter.create<quake::DiscriminateOp>(
              loc, rewriter.getI1Type(), measOut);
          return discOp.getResult();
        }();
        rewriter.create<cudaq::cc::IfOp>(
            loc, TypeRange{}, measBit,
            [&](OpBuilder &opBuilder, Location location, Region &region) {
              region.push_back(new Block{});
              auto &bodyBlock = region.front();
              OpBuilder::InsertionGuard guad(opBuilder);
              opBuilder.setInsertionPointToStart(&bodyBlock);
              opBuilder.create<quake::XOp>(location, measuredQubit);
              opBuilder.create<cudaq::cc::ContinueOp>(location);
            });
        modified = true;
      } else {
        LLVM_DEBUG(llvm::dbgs() << "No next use\n");
      }
    }

    return success(modified);
  }

private:
  Operation *getNextUse(Value qubit, Operation *op) const {
    auto &dom = tracker.getDominanceInfo();
    {
      // Check direct use
      const auto orderedUsers = sortUsers(qubit.getUsers(), dom);
      for (auto v : llvm::enumerate(orderedUsers))
        if (v.value() == op && v.index() < (orderedUsers.size() - 1) &&
            dom.dominates(op, orderedUsers[v.index() + 1]))
          return orderedUsers[v.index() + 1];
    }

    // No next use is found, check if this is an extracted qubit.
    if (isa<quake::RefType>(qubit.getType())) {
      if (auto extractOp =
              dyn_cast_if_present<quake::ExtractRefOp>(qubit.getDefiningOp())) {
        LLVM_DEBUG(llvm::dbgs() << "Defining op: " << *extractOp << "\n");
        auto reg = extractOp.getVeq();
        std::optional<int64_t> index =
            extractOp.hasConstantIndex()
                ? std::optional<int64_t>(extractOp.getConstantIndex())
                : cudaq::getIndexValueAsInt(extractOp.getIndex());
        LLVM_DEBUG(llvm::dbgs() << "Reg: " << reg
                                << "; index = " << index.value_or(-1) << "\n");
        if (isa<quake::AllocaOp>(reg.getDefiningOp())) {
          const auto orderedUsers = tracker.getUsers(reg);
          for (auto v : llvm::enumerate(orderedUsers)) {
            if (v.value() != extractOp) {
              // This is another extract.
              auto nextExtractOp =
                  dyn_cast_or_null<quake::ExtractRefOp>(v.value());
              if (nextExtractOp) {
                std::optional<int64_t> nextIndex =
                    nextExtractOp.hasConstantIndex()
                        ? nextExtractOp.getConstantIndex()
                        : cudaq::getIndexValueAsInt(nextExtractOp.getIndex());
                if ((!index.has_value() || !nextIndex.has_value()) ||
                    (index == nextIndex)) {
                  // Either the previous index or this index is unknown, we
                  // assume that they may be the same.
                  const auto extractedQubit = nextExtractOp.getRef();
                  const auto extractedQubitOrderedUsers =
                      sortUsers(extractedQubit.getUsers(), dom);
                  for (auto *user : extractedQubitOrderedUsers) {
                    // If the use is dominated by the original mz op,
                    // then this is the next use.
                    if (dom.dominates(op, user)) {
                      LLVM_DEBUG(llvm::dbgs() << "Next use: " << *user << "\n");
                      return user;
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    return nullptr;
  }

  RegUseTracker &tracker;
};

class QubitResetBeforeReusePass
    : public cudaq::opt::impl::QubitResetBeforeReuseBase<
          QubitResetBeforeReusePass> {
public:
  using QubitResetBeforeReuseBase::QubitResetBeforeReuseBase;
  QubitResetBeforeReusePass() = default;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.empty())
      return;
    auto *ctx = &getContext();
    RegUseTracker tracker(funcOp);
    RewritePatternSet patterns(ctx);
    patterns.insert<ResetAfterMeasurePattern>(ctx, tracker);
    if (failed(applyPatternsAndFoldGreedily(funcOp.getOperation(),
                                            std::move(patterns)))) {
      funcOp.emitOpError("Adding qubit reset before reuse pass failed");
      signalPassFailure();
    }
  }
};
} // namespace
