/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LIFTARRAYALLOC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lift-array-alloc"

using namespace mlir;

// Determine if \p alloc is a legit candidate for promotion to a constant array
// value.
static bool isGoodCandidate(cudaq::cc::AllocaOp alloc,
                            SmallVectorImpl<Operation *> &scoreboard) {
  LLVM_DEBUG(llvm::dbgs() << "checking candidate\n");
  if (alloc.getSeqSize())
    return false;
  auto arrTy = dyn_cast<cudaq::cc::ArrayType>(alloc.getElementType());
  if (!arrTy || arrTy.isUnknownSize())
    return false;
  auto arrEleTy = arrTy.getElementType();
  if (!isa<IntegerType, FloatType, ComplexType>(arrEleTy))
    return false;

  // There must be at least `size` uses to initialize the entire array.
  auto size = arrTy.getSize();
  if (std::distance(alloc->getUses().begin(), alloc->getUses().end()) < size)
    return false;

  // Keep a scoreboard for every element in the array. Every element *must* be
  // stored to with a constant exactly one time.
  scoreboard.resize(size);
  for (int i = 0; i < size; i++)
    scoreboard[i] = nullptr;

  auto getWriteOp = [](auto op) -> Operation * {
    Operation *theStore = nullptr;
    for (auto &use : op->getUses()) {
      Operation *u = use.getOwner();
      if (!u)
        return nullptr;
      if (auto store = dyn_cast<cudaq::cc::StoreOp>(u)) {
        if (op.getOperation() == store.getPtrvalue().getDefiningOp() &&
            isa_and_present<arith::ConstantOp, complex::ConstantOp>(
                store.getValue().getDefiningOp())) {
          if (theStore) {
            LLVM_DEBUG(llvm::dbgs()
                       << "more than 1 store to element of array\n");
            return nullptr;
          }
          theStore = u;
        }
        continue;
      }
      if (!isa<cudaq::cc::LoadOp, quake::InitializeStateOp>(u))
        return nullptr;
    }
    return theStore;
  };

  auto ptrArrEleTy = cudaq::cc::PointerType::get(arrTy.getElementType());
  for (auto &use : alloc->getUses()) {
    // All uses *must* be a degenerate cc.cast, cc.compute_ptr, or
    // cc.init_state.
    auto *op = use.getOwner();
    if (!op) {
      LLVM_DEBUG(llvm::dbgs() << "use was not an op\n");
      return false;
    }
    if (auto cptr = dyn_cast<cudaq::cc::ComputePtrOp>(op)) {
      if (auto index = cptr.getConstantIndex(0)) {
        if (auto w = getWriteOp(cptr)) {
          if (!scoreboard[*index]) {
            scoreboard[*index] = w;
          } else {
            return false;
          }
        }
      }
    } else if (auto cast = dyn_cast<cudaq::cc::CastOp>(op)) {
      if (cast.getType() == ptrArrEleTy) {
        if (auto w = getWriteOp(cast)) {
          if (!scoreboard[0]) {
            scoreboard[0] = w;
          } else {
            return false;
          }
        }
      } else {
        LLVM_DEBUG(llvm::dbgs() << "unexpected cast\n");
        return false;
      }
    } else {
      LLVM_DEBUG(llvm::dbgs() << "unexpected use: " << *op << '\n');
      return false;
    }
  }

  bool ok = std::all_of(scoreboard.begin(), scoreboard.end(),
                        [](bool b) { return b; });
  LLVM_DEBUG(llvm::dbgs() << "all elements of array are set: " << ok << '\n');
  return ok;
}

namespace {
class AllocaPattern : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> stores;
    if (!isGoodCandidate(alloc, stores))
      return failure();

    auto arrTy = cast<cudaq::cc::ArrayType>(alloc.getElementType());
    SmallVector<Attribute> values;
    // Every element of `stores` must be a cc::StoreOp with a ConstantOp as the
    // value argument. Build the array attr to attach to a cc.const_array.
    for (auto *op : stores) {
      auto store = cast<cudaq::cc::StoreOp>(op);
      auto *valOp = store.getValue().getDefiningOp();
      if (auto con = dyn_cast<arith::ConstantOp>(valOp))
        values.push_back(con.getValueAttr());
      else if (auto con = dyn_cast<complex::ConstantOp>(valOp))
        values.push_back(con.getValueAttr());
      else
        return alloc.emitOpError("could not fold");
    }

    // Create the cc.const_array.
    auto valuesAttr = rewriter.getArrayAttr(values);
    auto conArr = rewriter.create<cudaq::cc::ConstantArrayOp>(
        alloc.getLoc(), arrTy, valuesAttr);
    auto eleTy = arrTy.getElementType();

    // Rewalk all the uses of alloc, u, which must be cc.cast or cc.compute_ptr.
    // For each,u, remove a store and replace a load with a cc.extract_value.
    for (auto &use : alloc->getUses()) {
      auto *user = use.getOwner();
      std::int32_t offset = 0;
      if (auto cptr = dyn_cast<cudaq::cc::ComputePtrOp>(user))
        offset = cptr.getRawConstantIndices()[0];
      for (auto &useuse : user->getUses()) {
        auto *useuser = useuse.getOwner();
        if (auto ist = dyn_cast<quake::InitializeStateOp>(useuser)) {
          LLVM_DEBUG(llvm::dbgs() << "replaced init_state\n");
          rewriter.replaceOpWithNewOp<quake::InitializeStateOp>(
              ist, ist.getType(), ist.getTargets(), conArr);
          continue;
        }
        if (auto load = dyn_cast<cudaq::cc::LoadOp>(useuser)) {
          LLVM_DEBUG(llvm::dbgs() << "replaced load\n");
          rewriter.replaceOpWithNewOp<cudaq::cc::ExtractValueOp>(
              load, eleTy, conArr,
              ArrayRef<cudaq::cc::ExtractValueArg>{offset});
          continue;
        }
        assert(isa<cudaq::cc::StoreOp>(useuser));
        rewriter.eraseOp(useuser);
      }
      rewriter.eraseOp(user);
    }

    rewriter.eraseOp(alloc);
    return success();
  }
};

class LiftArrayAllocPass
    : public cudaq::opt::impl::LiftArrayAllocBase<LiftArrayAllocPass> {
public:
  using LiftArrayAllocBase::LiftArrayAllocBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<AllocaPattern>(ctx);

    LLVM_DEBUG(llvm::dbgs()
               << "Before lifting constant array: " << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "After lifting constant array: " << func << '\n');
  }
};
} // namespace
