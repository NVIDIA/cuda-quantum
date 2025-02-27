/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These patterns are used by the loft-array-alloc and cc-loop-unroll passes.

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

namespace {
class AllocaPattern : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  explicit AllocaPattern(MLIRContext *ctx, DominanceInfo &di, StringRef fn)
      : OpRewritePattern(ctx), dom(di), funcName(fn) {}

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> stores;
    if (!isGoodCandidate(alloc, stores, dom))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Candidate was found\n");
    auto allocTy = alloc.getElementType();
    auto arrTy = cast<cudaq::cc::ArrayType>(allocTy);
    auto eleTy = arrTy.getElementType();

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
    auto loc = alloc.getLoc();
    Value conArr =
        rewriter.create<cudaq::cc::ConstantArrayOp>(loc, arrTy, valuesAttr);

    assert(conArr && "must have created the constant array");
    LLVM_DEBUG(llvm::dbgs() << "constant array is:\n" << conArr << '\n');
    bool cannotEraseAlloc = false;

    // Collect all the stores, casts, and compute_ptr to be erased safely and in
    // topological order.
    SmallVector<Operation *> opsToErase;
    auto insertOpToErase = [&](Operation *op) {
      auto iter = std::find(opsToErase.begin(), opsToErase.end(), op);
      if (iter == opsToErase.end())
        opsToErase.push_back(op);
    };

    // Rewalk all the uses of alloc, u, which must be cc.cast or cc.compute_ptr.
    // For each u remove a store and replace a load with a cc.extract_value.
    for (auto *user : alloc->getUsers()) {
      if (!user)
        continue;
      std::int32_t offset = 0;
      if (auto cptr = dyn_cast<cudaq::cc::ComputePtrOp>(user))
        offset = cptr.getRawConstantIndices()[0];
      bool isLive = false;
      if (!isa<cudaq::cc::CastOp, cudaq::cc::ComputePtrOp>(user)) {
        cannotEraseAlloc = isLive = true;
      } else {
        for (auto *useuser : user->getUsers()) {
          if (!useuser)
            continue;
          if (auto load = dyn_cast<cudaq::cc::LoadOp>(useuser)) {
            rewriter.setInsertionPointAfter(useuser);
            LLVM_DEBUG(llvm::dbgs() << "replaced load\n");
            // rewriter.replaceOpWithNewOp<cudaq::cc::ExtractValueOp>(
            //     load, eleTy, conArr,
            //     ArrayRef<cudaq::cc::ExtractValueArg>{offset});

            auto extractValue = rewriter.create<cudaq::cc::ExtractValueOp>(
                loc, eleTy, conArr,
                ArrayRef<cudaq::cc::ExtractValueArg>{offset});
            rewriter.replaceAllUsesWith(load, extractValue);
            insertOpToErase(load);
            continue;
          }
          if (isa<cudaq::cc::StoreOp>(useuser)) {
            insertOpToErase(useuser);
            continue;
          }
          LLVM_DEBUG(llvm::dbgs() << "alloc is live\n");
          cannotEraseAlloc = isLive = true;
        }
      }
      if (!isLive)
        insertOpToErase(user);
    }

    for (auto *e : opsToErase)
      rewriter.eraseOp(e);

    if (cannotEraseAlloc) {
      rewriter.setInsertionPointAfter(alloc);
      rewriter.create<cudaq::cc::StoreOp>(loc, conArr, alloc);
      return success();
    }
    rewriter.eraseOp(alloc);
    return success();
  }

  // Determine if \p alloc is a legit candidate for promotion to a constant
  // array value. \p scoreboard is a vector of store operations. Each element of
  // the allocated array must be written to exactly 1 time, and the scoreboard
  // is used to track these stores. \p dom is the dominance info for this
  // function (to ensure the stores happen before uses).
  static bool isGoodCandidate(cudaq::cc::AllocaOp alloc,
                              SmallVectorImpl<Operation *> &scoreboard,
                              DominanceInfo &dom) {
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

    SmallVector<Operation *> toGlobalUses;
    SmallVector<SmallPtrSet<Operation *, 2>> loadSets(size);

    auto getWriteOp = [&](auto op, std::int32_t index) -> Operation * {
      Operation *theStore = nullptr;
      for (auto &use : op->getUses()) {
        Operation *u = use.getOwner();
        if (!u)
          return nullptr;

        if (auto store = dyn_cast<cudaq::cc::StoreOp>(u)) {
          if (op.getOperation() == store.getPtrvalue().getDefiningOp()) {
            if (theStore) {
              LLVM_DEBUG(llvm::dbgs()
                         << "more than 1 store to element of array\n");
              return nullptr;
            }
            LLVM_DEBUG(llvm::dbgs() << "found store: " << store << "\n");
            theStore = u;
          }
          continue;
        }
        if (isa<quake::InitializeStateOp>(u)) {
          toGlobalUses.push_back(u);
          continue;
        }
        if (isa<cudaq::cc::LoadOp>(u)) {
          loadSets[index].insert(u);
          continue;
        }
        return nullptr;
      }
      return theStore &&
                     isa_and_present<arith::ConstantOp, complex::ConstantOp>(
                         dyn_cast<cudaq::cc::StoreOp>(theStore)
                             .getValue()
                             .getDefiningOp())
                 ? theStore
                 : nullptr;
    };

    auto unsizedArrTy = cudaq::cc::ArrayType::get(arrEleTy);
    auto ptrUnsizedArrTy = cudaq::cc::PointerType::get(unsizedArrTy);
    auto ptrArrEleTy = cudaq::cc::PointerType::get(arrEleTy);
    for (auto &use : alloc->getUses()) {
      // All uses *must* be a degenerate cc.cast, cc.compute_ptr, or
      // cc.init_state.
      auto *op = use.getOwner();
      if (!op) {
        LLVM_DEBUG(llvm::dbgs() << "use was not an op\n");
        return false;
      }
      if (auto cptr = dyn_cast<cudaq::cc::ComputePtrOp>(op)) {
        if (auto index = cptr.getConstantIndex(0))
          if (auto w = getWriteOp(cptr, *index))
            if (!scoreboard[*index]) {
              scoreboard[*index] = w;
              continue;
            }
        return false;
      }
      if (auto cast = dyn_cast<cudaq::cc::CastOp>(op)) {
        // Process casts that are used in store ops.
        if (cast.getType() == ptrArrEleTy) {
          if (auto w = getWriteOp(cast, 0))
            if (!scoreboard[0]) {
              scoreboard[0] = w;
              continue;
            }
          return false;
        }
        // Process casts that are used in quake.init_state.
        if (cast.getType() == ptrUnsizedArrTy) {
          if (cast->hasOneUse()) {
            auto &use = *cast->getUses().begin();
            Operation *u = use.getOwner();
            if (isa_and_present<quake::InitializeStateOp>(u)) {
              toGlobalUses.push_back(op);
              continue;
            }
          }
          return false;
        }
        LLVM_DEBUG(llvm::dbgs() << "unexpected cast: " << *op << '\n');
        toGlobalUses.push_back(op);
        continue;
      }
      if (isa<quake::InitializeStateOp>(op)) {
        toGlobalUses.push_back(op);
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "unexpected use: " << *op << '\n');
      toGlobalUses.push_back(op);
    }

    bool ok = std::all_of(scoreboard.begin(), scoreboard.end(),
                          [](bool b) { return b; });
    LLVM_DEBUG(llvm::dbgs() << "all elements of array are set: " << ok << '\n');
    if (ok) {
      // Verify dominance relations.

      // For all stores, the store of an element $e$ must dominate all loads of
      // $e$.
      for (int i = 0; i < size; ++i) {
        for (auto *load : loadSets[i])
          if (!dom.dominates(scoreboard[i], load)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "store " << scoreboard[i]
                       << " doesn't dominate load: " << *load << '\n');
            return false;
          }
      }

      // For all global uses, all of the stores must dominate every use.
      for (auto *glob : toGlobalUses) {
        for (auto *store : scoreboard)
          if (!dom.dominates(store, glob)) {
            LLVM_DEBUG(llvm::dbgs()
                       << "store " << store << " doesn't dominate op: " << *glob
                       << '\n');
            return false;
          }
      }
    }
    return ok;
  }

  DominanceInfo &dom;
  StringRef funcName;
};
} // namespace
