/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Factory.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_FACTORQUANTUMALLOCATIONS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "factor-quantum-alloc"

using namespace mlir;

static bool allocaOfVeqStruq(cudaq::quake::AllocaOp alloc) {
  return isa<cudaq::quake::VeqType, cudaq::quake::StruqType>(alloc.getType());
}

static bool allocaOfUnspecifiedSize(cudaq::quake::AllocaOp alloc) {
  if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(alloc.getType()))
    return !veqTy.hasSpecifiedSize();
  if (auto ty = dyn_cast<cudaq::quake::StruqType>(alloc.getType()))
    return !ty.hasSpecifiedSize();
  return false;
}

static bool isUseConvertible(Operation *op) {
  if (isa<cudaq::quake::DeallocOp, cudaq::quake::GetMemberOp>(op))
    return true;

  // extract_ref is ok, iff it has a constant offset
  if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(op))
    return ext.hasConstantIndex();

  auto sub = dyn_cast<cudaq::quake::SubVeqOp>(op);
  if (!sub)
    return false;

  // op must be a SubVeqOp
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

static bool usesAreConvertible(cudaq::quake::AllocaOp alloc) {
  for (auto *users : alloc->getUsers())
    if (!isUseConvertible(users))
      return false;
  return true;
}

namespace {
class AllocaPattern : public OpRewritePattern<cudaq::quake::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  /// If we are here, then all uses of \p allocOp are either an ExtractRefOp
  /// with a constant index or a DeallocOp. Any other user is assumed to block
  /// the factoring of the allocation.
  LogicalResult matchAndRewrite(cudaq::quake::AllocaOp allocOp,
                                PatternRewriter &rewriter) const override {
    auto loc = allocOp.getLoc();

    // 0. Check necessary preconditions hold.
    if (!allocaOfVeqStruq(allocOp) || allocaOfUnspecifiedSize(allocOp) ||
        allocOp.hasInitializedState())
      return failure();

    if (auto stqTy = dyn_cast<cudaq::quake::StruqType>(allocOp.getType())) {
      // allocOp is a struq.
      // 1. Convert the allocation into a member by member allocation.
      SmallVector<Value> memAllocs;
      for (auto memTy : stqTy.getMembers())
        memAllocs.emplace_back(
            cudaq::quake::AllocaOp::create(rewriter, loc, memTy).getResult());
      // 2. Create a value of the original struq type using quake.make_struq.
      auto aggregate =
          cudaq::quake::MakeStruqOp::create(rewriter, loc, stqTy, memAllocs);
      // 3. Walk all the uses. If they are quake.get_member operations, replace
      // them with direct uses.
      for (auto *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto getMem = dyn_cast<cudaq::quake::GetMemberOp>(user)) {
          auto index = getMem.getIndex();
          rewriter.replaceOp(getMem, memAllocs[index]);
        }
      rewriter.replaceOp(allocOp, aggregate.getResult());
      return success();
    }

    // allocOp must be a veq.
    auto veqTy = cast<cudaq::quake::VeqType>(allocOp.getType());
    std::size_t size = veqTy.getSize();
    SmallVector<cudaq::quake::AllocaOp> newAllocs;
    auto *ctx = rewriter.getContext();
    auto refTy = cudaq::quake::RefType::get(ctx);

    // Split the aggregate veq into a sequence of distinct alloca of ref.
    for (std::size_t i = 0; i < size; ++i)
      newAllocs.emplace_back(
          cudaq::quake::AllocaOp::create(rewriter, loc, refTy));

    if (usesAreConvertible(allocOp)) {
      // Visit all users and replace them accordingly.
      if (failed(rewriteOpAndUsers(allocOp, 0, rewriter, size, newAllocs)))
        return failure();
      // Remove the original alloca operation.
      rewriter.eraseOp(allocOp);
    } else {
      // Uses are more complex so just concat the refs together.
      SmallVector<Value> theRefs;
      std::for_each(
          newAllocs.begin(), newAllocs.end(),
          [&](cudaq::quake::AllocaOp a) { theRefs.push_back(a.getResult()); });
      rewriter.replaceOpWithNewOp<cudaq::quake::ConcatOp>(allocOp, veqTy,
                                                          theRefs);
    }
    return success();
  }

  static LogicalResult
  rewriteOpAndUsers(Operation *op, std::int64_t start,
                    PatternRewriter &rewriter, std::size_t size,
                    SmallVector<cudaq::quake::AllocaOp> &newAllocs) {
    // First handle the users. Note that this can recurse.
    SmallVector<Operation *> users{op->getUsers().begin(),
                                   op->getUsers().end()};
    for (auto *user : users) {
      if (auto dealloc = dyn_cast<cudaq::quake::DeallocOp>(user)) {
        rewriter.setInsertionPoint(dealloc);
        auto deloc = dealloc.getLoc();
        for (std::size_t i = 0; i < size - 1; ++i)
          cudaq::quake::DeallocOp::create(rewriter, deloc, newAllocs[i]);
        rewriter.replaceOpWithNewOp<cudaq::quake::DeallocOp>(
            dealloc, newAllocs[size - 1]);
        continue;
      }
      if (auto subveq = dyn_cast<cudaq::quake::SubVeqOp>(user)) {
        auto lowInt = [&]() -> std::optional<std::int32_t> {
          if (subveq.hasConstantLowerBound())
            return {subveq.getConstantLowerBound()};
          return cudaq::opt::factory::getIntIfConstant(subveq.getLower());
        }();
        if (!lowInt)
          return failure();
        SmallVector<Operation *> subUsers{subveq->getUsers().begin(),
                                          subveq->getUsers().end()};
        for (auto *subUser : subUsers)
          if (failed(rewriteOpAndUsers(subUser, *lowInt, rewriter, size,
                                       newAllocs)))
            return failure();
        rewriter.eraseOp(subveq);
        continue;
      }
      if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(user)) {
        auto index = ext.getConstantIndex();
        rewriter.replaceOp(ext, newAllocs[start + index].getResult());
      }
    }

    // Now handle the base operation.
    if (isa<cudaq::quake::SubVeqOp>(op)) {
      rewriter.eraseOp(op);
    } else if (auto ext = dyn_cast<cudaq::quake::ExtractRefOp>(op)) {
      auto index = ext.getConstantIndex();
      rewriter.replaceOp(ext, newAllocs[start + index].getResult());
    }
    return success();
  }
};

class DeallocPattern : public OpRewritePattern<cudaq::quake::DeallocOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::DeallocOp dealloc,
                                PatternRewriter &rewriter) const override {
    // 0. Check necessary preconditions. Must be a Veq or Struq with a constant
    // size.
    Value alloc = dealloc.getReference();
    if (!alloc)
      return failure();
    auto allocTy = alloc.getType();
    if (alloc.getDefiningOp<cudaq::quake::InitializeStateOp>())
      return failure();
    if (auto ty = dyn_cast<cudaq::quake::VeqType>(allocTy)) {
      if (!ty.hasSpecifiedSize())
        return failure();
    } else if (auto ty = dyn_cast<cudaq::quake::StruqType>(allocTy)) {
      if (!ty.hasSpecifiedSize())
        return failure();
    } else {
      // not a Veq or Struq, so nothing to do.
      return failure();
    }

    auto loc = dealloc.getLoc();
    if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(allocTy)) {
      generateDeallocs(veqTy, rewriter, loc, alloc);
    } else if (auto stqTy = dyn_cast<cudaq::quake::StruqType>(allocTy)) {
      for (auto iter : llvm::enumerate(stqTy.getMembers())) {
        Type memTy = iter.value();
        auto mem = cudaq::quake::GetMemberOp::create(rewriter, loc, memTy,
                                                     alloc, iter.index());
        if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(memTy))
          generateDeallocs(veqTy, rewriter, loc, mem);
        else
          cudaq::quake::DeallocOp::create(rewriter, loc, mem);
      }
    }

    // 2. Remove the original dealloc operation.
    rewriter.eraseOp(dealloc);
    return success();
  }

  static void generateDeallocs(cudaq::quake::VeqType veqTy,
                               PatternRewriter &rewriter, Location loc,
                               Value alloc) {
    assert(veqTy.hasSpecifiedSize());
    std::size_t size = veqTy.getSize();

    for (std::size_t i = 0; i < size; ++i) {
      Value r = cudaq::quake::ExtractRefOp::create(rewriter, loc, alloc, i);
      cudaq::quake::DeallocOp::create(rewriter, loc, r);
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
    if (failed(factorDeallocations())) {
      if (enableFailures)
        signalPassFailure();
      return;
    }

    // 2) Factor any allocations that are struqs or veqs of constant size.
    if (failed(factorAllocations())) {
      if (enableFailures)
        signalPassFailure();
      return;
    }

    LLVM_DEBUG(llvm::dbgs() << "Function after factoring quake alloca:\n"
                            << func << "\n\n");
  }

  LogicalResult factorDeallocations() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<DeallocPattern>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return failure();
    return success();
  }

  LogicalResult factorAllocations() {
    auto *ctx = &getContext();
    func::FuncOp func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPattern>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return failure();
    return success();
  }
};
} // namespace
