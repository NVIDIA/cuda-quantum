/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_SROA
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "cc-sroa"

using namespace mlir;

namespace {
class AllocaAggregate : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Nothing to do if this allocation is dynamically sized.
    if (allocOp.getSeqSize())
      return failure();
    // Nothing to do if this isn't an aggregate type.
    if (!isa<cudaq::cc::ArrayType, cudaq::cc::StructType>(
            allocOp.getElementType()))
      return failure();

    // If all the uses are cc.compute_ptr, then we can do a scalar replacement.
    for (auto *user : allocOp->getUsers()) {
      if (!user)
        continue;
      if (auto ptrOp = dyn_cast<cudaq::cc::ComputePtrOp>(user))
        if (ptrOp.getNumIndices() == 1 && ptrOp.getConstantIndex(0))
          continue;
      if (auto cast = dyn_cast<cudaq::cc::CastOp>(user))
        if (castMatches(allocOp.getElementType(), cast.getResult().getType()))
          continue;
      return failure();
    }

    // Go ahead with the scalar replacement.
    // Create the scalars.
    SmallVector<Value> scalars;
    auto loc = allocOp.getLoc();
    if (auto strTy =
            dyn_cast<cudaq::cc::StructType>(allocOp.getElementType())) {
      for (auto mTy : strTy.getMembers())
        scalars.push_back(rewriter.create<cudaq::cc::AllocaOp>(loc, mTy));
    } else if (auto arrTy =
                   dyn_cast<cudaq::cc::ArrayType>(allocOp.getElementType())) {
      Type vTy = arrTy.getElementType();
      for (cudaq::cc::ArrayType::SizeType i = 0; i < arrTy.getSize(); ++i)
        scalars.push_back(rewriter.create<cudaq::cc::AllocaOp>(loc, vTy));
    }

    // Replace the cc.compute_ptr ops with forwarding.
    for (auto *user : allocOp->getUsers()) {
      if (!user)
        continue;
      if (auto ptrOp = dyn_cast<cudaq::cc::ComputePtrOp>(user)) {
        auto pos = *ptrOp.getConstantIndex(0);
        rewriter.replaceOp(ptrOp, scalars[pos]);
      } else {
        auto castOp = cast<cudaq::cc::CastOp>(user);
        rewriter.replaceOp(castOp, scalars[0]);
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "scalar replacement on: " << allocOp << '\n');
    rewriter.eraseOp(allocOp);
    return success();
  }

  static bool castMatches(Type aggrTy, Type ty) {
    ty = [&]() -> Type {
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(ty))
        return ptrTy.getElementType();
      return {};
    }();
    // TODO: check compositions of these types.
    if (auto strTy = dyn_cast<cudaq::cc::StructType>(aggrTy))
      return strTy.getMember(0) == ty;
    if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(aggrTy))
      return arrTy.getElementType() == ty;
    return false;
  }
};

// Convert a store of an aggregate value to a series of stores of its elements.
// This primes the pump for the SROA step on the allocation itself.
class StoreAggregate : public OpRewritePattern<cudaq::cc::StoreOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::StoreOp storeOp,
                                PatternRewriter &rewriter) const override {
    auto val = storeOp.getValue();
    if (!isa<cudaq::cc::ArrayType, cudaq::cc::StructType>(val.getType()))
      return failure();

    // Can we roll back the composite value through cc.insert_value operations
    // to a cc.undef?
    SmallVector<cudaq::cc::InsertValueOp> stack;
    while (auto insVal = val.getDefiningOp<cudaq::cc::InsertValueOp>()) {
      stack.push_back(insVal);
      val = insVal.getContainer();
    }
    auto undefOp = val.getDefiningOp<cudaq::cc::UndefOp>();
    if (!undefOp) {
      LLVM_DEBUG(llvm::dbgs() << "chain did not end in undef: " << val << '\n');
      return failure();
    }

    // It's a match. We have a chain of insert_values back to a undef. Let's
    // replace this store.
    Value dest = storeOp.getPtrvalue();
    Type baseTy = dest.getType();
    for (auto insVal : stack) {
      // Each insert_value is converted to a compute_ptr and store.
      auto v = insVal.getValue();
      SmallVector<cudaq::cc::ComputePtrArg> args;
      for (std::int32_t off : insVal.getPosition())
        args.push_back(off);
      auto loc = insVal.getLoc();
      auto toAddr =
          rewriter.create<cudaq::cc::ComputePtrOp>(loc, baseTy, dest, args);
      rewriter.create<cudaq::cc::StoreOp>(loc, v, toAddr);
    }
    LLVM_DEBUG(llvm::dbgs() << "updated: " << storeOp << '\n');
    rewriter.eraseOp(storeOp);
    return success();
  }
};

class SROAPass : public cudaq::opt::impl::SROABase<SROAPass> {
public:
  using SROABase::SROABase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto *op = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaAggregate, StoreAggregate>(ctx);
    if (failed(applyPatternsAndFoldGreedily(op, std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace
