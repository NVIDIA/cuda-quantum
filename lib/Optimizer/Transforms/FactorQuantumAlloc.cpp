/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_FACTORQUANTUMALLOCATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "factor-quantum-alloc"

using namespace mlir;

static bool allocaOfVeqStruq(quake::AllocaOp alloc) {
  return isa<quake::VeqType, quake::StruqType>(alloc.getType());
}

static bool allocaOfUnspecifiedSize(quake::AllocaOp alloc) {
  if (auto veqTy = dyn_cast<quake::VeqType>(alloc.getType()))
    return !veqTy.hasSpecifiedSize();
  if (auto ty = dyn_cast<quake::StruqType>(alloc.getType()))
    return !ty.hasSpecifiedSize();
  return false;
}

static bool isUseConvertible(Operation *op) {
  if (isa<quake::DeallocOp>(op))
    return true;
  if (auto ext = dyn_cast<quake::ExtractRefOp>(op))
    if (ext.hasConstantIndex())
      return true;
  if (isa<quake::GetMemberOp>(op))
    return true;
  if (auto sub = dyn_cast<quake::SubVeqOp>(op)) {
    auto lowInt = [&]() -> std::optional<std::int32_t> {
      if (sub.hasConstantLowerBound())
        return {sub.getConstantLowerBound()};
      return cudaq::opt::factory::getIntIfConstant(sub.getLower());
    }();
    auto upInt = [&]() -> std::optional<std::int32_t> {
      if (sub.hasConstantUpperBound())
        return {sub.getConstantUpperBound()};
      return cudaq::opt::factory::getIntIfConstant(sub.getUpper());
    }();
    if (!(lowInt && upInt))
      return false;
    for (auto *subUser : sub->getUsers())
      if (!isUseConvertible(subUser))
        return false;
    return true;
  }
  return false;
}

namespace {
class AllocaPat : public OpRewritePattern<quake::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// If we are here, then all uses of \p allocOp are either an ExtractRefOp
  /// with a constant index or a DeallocOp. Any other user is assumed to block
  /// the factoring of the allocation.
  LogicalResult matchAndRewrite(quake::AllocaOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    // 0. Check necessary preconditions hold.
    if (!allocaOfVeqStruq(allocOp) || allocaOfUnspecifiedSize(allocOp) ||
        allocOp.hasInitializedState())
      return failure();
    auto usesAreConvertible = [&]() -> bool {
      for (auto *users : allocOp->getUsers()) {
        if (isUseConvertible(users))
          continue;
        return false;
      }
      return true;
    };
    if (isa<quake::VeqType>(allocOp.getType()) && !usesAreConvertible())
      return failure();

    if (auto stqTy = dyn_cast<quake::StruqType>(allocOp.getType())) {
      // allocOp is a struq.
      // 1. Convert the allocation into a member by member allocation.
      SmallVector<Value> memAllocs;
      for (auto memTy : stqTy.getMembers())
        memAllocs.emplace_back(
            rewriter.create<quake::AllocaOp>(loc, memTy).getResult());
      // 2. Create a value of the original struq type using quake.make_struq.
      auto aggregate =
          rewriter.create<quake::MakeStruqOp>(loc, stqTy, memAllocs);
      // 3. Walk all the uses. If they are quake.get_member operations, replace
      // them with direct uses.
      for (auto *user : allocOp->getUsers())
        if (auto getMem = dyn_cast<quake::GetMemberOp>(user)) {
          auto index = getMem.getIndex();
          rewriter.replaceOp(getMem, memAllocs[index]);
        }
      rewriter.replaceOp(allocOp, aggregate.getResult());
      return success();
    }

    // allocOp must be a veq.
    auto veqTy = cast<quake::VeqType>(allocOp.getType());
    std::size_t size = veqTy.getSize();
    SmallVector<quake::AllocaOp> newAllocs;
    auto *ctx = rewriter.getContext();
    auto refTy = quake::RefType::get(ctx);

    // 1. Split the aggregate veq into a sequence of distinct alloca of ref.
    for (std::size_t i = 0; i < size; ++i)
      newAllocs.emplace_back(rewriter.create<quake::AllocaOp>(loc, refTy));

    std::function<LogicalResult(Operation *, std::int64_t)> rewriteOpAndUsers =
        [&](Operation *op, std::int64_t start) -> LogicalResult {
      // First handle the users. Note that this can recurse.
      SmallVector<Operation *> users{op->getUsers().begin(),
                                     op->getUsers().end()};
      for (auto *user : users) {
        if (auto dealloc = dyn_cast<quake::DeallocOp>(user)) {
          rewriter.setInsertionPoint(dealloc);
          auto deloc = dealloc.getLoc();
          for (std::size_t i = 0; i < size - 1; ++i)
            rewriter.create<quake::DeallocOp>(deloc, newAllocs[i]);
          rewriter.replaceOpWithNewOp<quake::DeallocOp>(dealloc,
                                                        newAllocs[size - 1]);
          continue;
        }
        if (auto subveq = dyn_cast<quake::SubVeqOp>(user)) {
          auto lowInt = [&]() -> std::optional<std::int32_t> {
            if (subveq.hasConstantLowerBound())
              return {subveq.getConstantLowerBound()};
            return cudaq::opt::factory::getIntIfConstant(subveq.getLower());
          }();
          if (!lowInt)
            return failure();
          for (auto *subUser : subveq->getUsers())
            if (failed(rewriteOpAndUsers(subUser, *lowInt)))
              return failure();
          rewriter.eraseOp(subveq);
          continue;
        }
        if (auto ext = dyn_cast<quake::ExtractRefOp>(user)) {
          auto index = ext.getConstantIndex();
          rewriter.replaceOp(ext, newAllocs[start + index].getResult());
        }
      }
      // Now handle the base operation.
      if (isa<quake::SubVeqOp>(op)) {
        rewriter.eraseOp(op);
      } else if (auto ext = dyn_cast<quake::ExtractRefOp>(op)) {
        auto index = ext.getConstantIndex();
        rewriter.replaceOp(ext, newAllocs[start + index].getResult());
      }
      return success();
    };

    // 2. Visit all users and replace them accordingly.
    if (failed(rewriteOpAndUsers(allocOp, 0)))
      return failure();

    // 3. Remove the original alloca operation.
    rewriter.eraseOp(allocOp);

    return success();
  }
};

class DeallocPat : public OpRewritePattern<quake::DeallocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::DeallocOp dealloc,
                                PatternRewriter &rewriter) const override {
    // 0. Check necessary preconditions. Must be a Veq or Struq with a constant
    // size.
    if (dealloc.getReference().getDefiningOp<quake::InitializeStateOp>())
      return failure();
    if (auto ty = dyn_cast<quake::VeqType>(dealloc.getReference().getType())) {
      if (!ty.hasSpecifiedSize())
        return failure();
    } else if (auto ty = dyn_cast<quake::StruqType>(
                   dealloc.getReference().getType())) {
      if (!ty.hasSpecifiedSize())
        return failure();
    } else {
      // not a Veq or Struq.
      return failure();
    }

    auto alloc = dealloc.getReference();
    auto loc = dealloc.getLoc();
    // 1. Split the aggregate alloc into a sequence of distinct dealloc of
    // ref.
    if (auto veqTy = dyn_cast<quake::VeqType>(alloc.getType())) {
      generateDeallocs(veqTy, rewriter, loc, alloc);
    } else {
      auto stqTy = cast<quake::StruqType>(alloc.getType());
      for (auto iter : llvm::enumerate(stqTy.getMembers())) {
        Type memTy = iter.value();
        auto mem = rewriter.create<quake::GetMemberOp>(loc, memTy, alloc,
                                                       iter.index());
        if (auto veqTy = dyn_cast<quake::VeqType>(memTy))
          generateDeallocs(veqTy, rewriter, loc, mem);
        else
          rewriter.create<quake::DeallocOp>(loc, mem);
      }
    }

    // 2. Remove the original dealloc operation.
    rewriter.eraseOp(dealloc);
    return success();
  }

  static void generateDeallocs(quake::VeqType veqTy, PatternRewriter &rewriter,
                               Location loc, Value alloc) {
    assert(veqTy.hasSpecifiedSize());
    std::size_t size = veqTy.getSize();

    for (std::size_t i = 0; i < size; ++i) {
      Value r = rewriter.create<quake::ExtractRefOp>(loc, alloc, i);
      rewriter.create<quake::DeallocOp>(loc, r);
    }
  };
};

class FactorQuantumAllocationsPass
    : public cudaq::opt::impl::FactorQuantumAllocationsBase<
          FactorQuantumAllocationsPass> {
public:
  using FactorQuantumAllocationsBase::FactorQuantumAllocationsBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    LLVM_DEBUG(llvm::dbgs() << "Function before factoring quake alloca:\n"
                            << func << "\n\n");

    // 1) Factor any deallocations that are struqs or veqs of constant size. Do
    // this first to simplify preconditions for step 2.
    if (failed(factorDeallocations()))
      return;

    // 2) Factor any allocations that are struqs or veqs of constant size.
    if (failed(factorAllocations()))
      return;

    LLVM_DEBUG(llvm::dbgs() << "Function after factoring quake alloca:\n"
                            << func << "\n\n");
  }

  LogicalResult factorDeallocations() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<DeallocPat>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return failure();
    return success();
  }

  LogicalResult factorAllocations() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPat>(ctx);
    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns))))
      return failure();
    return success();
  }
};
} // namespace
