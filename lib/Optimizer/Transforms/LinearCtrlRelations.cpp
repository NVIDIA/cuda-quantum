/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LINEARCTRLRELATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "linear-ctrl-relations"

using namespace mlir;

namespace {
class ThreadControl : public OpRewritePattern<quake::ToControlOp> {
public:
  explicit ThreadControl(MLIRContext *ctx, DominanceInfo &di)
      : OpRewritePattern(ctx), dom(di) {}

  LogicalResult matchAndRewrite(quake::ToControlOp toCtrl,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "\n\n" << toCtrl << '\n');
    SmallVector<Operation *> users(toCtrl->getUsers().begin(),
                                   toCtrl->getUsers().end());
    auto numUsers = users.size();

    // Check the trivial cases.
    if (numUsers < 1)
      return failure();
    if (numUsers == 1) {
      // User must be a FromControlOp. We can erase them both.
      if (auto fromCtrl =
              dyn_cast_if_present<quake::FromControlOp>(users.front())) {
        fromCtrl.replaceAllUsesWith(toCtrl.getQubit());
        rewriter.eraseOp(fromCtrl);
        rewriter.eraseOp(toCtrl);
        return success();
      }
      // The IR must be broken.
      return failure();
    }

    // Now, we can rethread the more interesting case when a value of type
    // !quake.control is used in actual operations.
    //
    // 1. Get a list of the users in dominance order. We have to thread the
    // !quake.wire value respecting the dominance of the users.
    SmallVector<Operation *, 8> orderedUsers;
    for (auto *user : users) {
      assert(user);
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
    LLVM_DEBUG({
      for (std::size_t i = 0, j = orderedUsers.size(); i != j; ++i) {
        llvm::dbgs() << i << ' ';
        orderedUsers[i]->dump();
      }
    });

    // 2. Thread the wire value to each successive user.
    Value wireDef = toCtrl.getQubit();
    auto *ctx = rewriter.getContext();
    auto wireTy = quake::WireType::get(ctx);
    for (auto *user : orderedUsers) {
      assert(user);
      if (isa<quake::FromControlOp>(user)) {
        assert(user == orderedUsers.back() &&
               "FromControlOp must post-dominate all the users");
        rewriter.replaceOp(user, wireDef);
        continue;
      }
      const auto coarity = user->getResults().size();
      std::size_t position = 0; // Position of new wire in result tuple.
      constexpr std::size_t Uninitialized = ~0;
      std::size_t operandNumber = Uninitialized;
      for (auto iter : llvm::enumerate(user->getOperands())) {
        Value opnd = iter.value();
        if (isa<quake::WireType>(opnd.getType())) {
          position++;
          continue;
        }
        if (auto x = opnd.getDefiningOp<quake::ToControlOp>())
          if (x.getOperation() == toCtrl.getOperation()) {
            operandNumber = iter.index();
            break;
          }
      }
      assert(operandNumber != Uninitialized);
      LLVM_DEBUG(llvm::dbgs() << "position: " << position << '\n');

      // Add a !quake.wire to the return type of `user`.
      SmallVector<Type> wireTys{coarity + 1, wireTy};
      SmallVector<Value> operands = user->getOperands();
      // Replace the use of `toCtrl` with `wireDef` in `user`.
      operands[operandNumber] = wireDef;
      auto attrs = user->getAttrs();
      auto name = user->getName().getIdentifier();
      auto loc = user->getLoc();
      rewriter.setInsertionPoint(user);
      auto newUser = rewriter.create(loc, name, operands, wireTys, attrs);
      // Update `wireDef` with the new result just added to `user`.
      wireDef = newUser->getResult(position);
      SmallVector<Value> newUserResults = newUser->getResults();
      newUserResults.erase(newUserResults.begin() + position);
      // Replace `user` with `newUser`, omitting the new result which will be
      // threaded on the next iteration of this loop.
      rewriter.replaceOp(user, newUserResults);
    }
    return success();
  }

  DominanceInfo &dom;
};
} // namespace

namespace {
class LinearCtrlRelationsPass
    : public cudaq::opt::impl::LinearCtrlRelationsBase<
          LinearCtrlRelationsPass> {
public:
  using LinearCtrlRelationsBase::LinearCtrlRelationsBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    DominanceInfo domInfo(func);
    RewritePatternSet patterns(ctx);
    patterns.insert<ThreadControl>(ctx, domInfo);
    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace
