/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_LIFTARRAYALLOC
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "lift-array-alloc"

using namespace mlir;

namespace {
class AllocaPattern : public OpRewritePattern<cudaq::cc::AllocaOp> {
public:
  explicit AllocaPattern(MLIRContext *ctx, DominanceInfo &di,
                         const std::string &fn, ModuleOp m)
      : OpRewritePattern(ctx), dom(di), funcName(fn), module(m) {}

  LogicalResult matchAndRewrite(cudaq::cc::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    SmallVector<Operation *> stores;
    bool toGlobal = false;
    if (!isGoodCandidate(alloc, stores, dom, toGlobal))
      return failure();

    LLVM_DEBUG(llvm::dbgs() << "Candidate was found\n");
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
    auto eleTy = arrTy.getElementType();
    auto valuesAttr = rewriter.getArrayAttr(values);
    auto loc = alloc.getLoc();
    Value conArr;
    Value conGlobal;
    if (toGlobal) {
      // static unsigned counter = 0;
      auto ptrTy = cudaq::cc::PointerType::get(arrTy);
      // Build a new name based on the kernel name.
      std::string name = funcName + ".rodata"; //_" + std::to_string(counter++);
      {
        OpBuilder::InsertionGuard guard(rewriter);
        if (auto complexTy = dyn_cast<ComplexType>(eleTy)) {
          // Transforming complex vectors is a bit more labor intensive. Use the
          // IRBuilder to create the object since we have to thread the needle
          // for the LLVM-IR to be lowered to LLVM correctly.
          auto transform = [&]<typename A>(SmallVectorImpl<Attribute> &vec)
              -> std::vector<std::complex<A>> {
            std::vector<std::complex<A>> result;
            for (auto a : vec) {
              auto v = cast<ArrayAttr>(a);
              if constexpr (std::is_same_v<A, double>) {
                result.emplace_back(
                    cast<FloatAttr>(v[0]).getValue().convertToDouble(),
                    cast<FloatAttr>(v[1]).getValue().convertToDouble());
              } else {
                result.emplace_back(
                    cast<FloatAttr>(v[0]).getValue().convertToFloat(),
                    cast<FloatAttr>(v[1]).getValue().convertToFloat());
              }
            }
            return result;
          };
          cudaq::IRBuilder irBuilder(rewriter.getContext());
          if (complexTy.getElementType() == rewriter.getF64Type()) {
            std::vector<std::complex<double>> vals =
                transform.template operator()<double>(values);
            irBuilder.genVectorOfComplexConstant(loc, module, name, vals);
          } else {
            std::vector<std::complex<float>> vals =
                transform.template operator()<float>(values);
            irBuilder.genVectorOfComplexConstant(loc, module, name, vals);
          }
        } else {
          OpBuilder::InsertionGuard guard(rewriter);
          rewriter.setInsertionPointToEnd(module.getBody());
          rewriter.create<cudaq::cc::GlobalOp>(loc, arrTy, name, valuesAttr,
                                               /*isConstant=*/true,
                                               /*isExternal=*/false);
        }
      }
      conGlobal = rewriter.create<cudaq::cc::AddressOfOp>(loc, ptrTy, name);
      conArr = rewriter.create<cudaq::cc::LoadOp>(loc, arrTy, conGlobal);
    } else {
      conArr =
          rewriter.create<cudaq::cc::ConstantArrayOp>(loc, arrTy, valuesAttr);
    }

    // Rewalk all the uses of alloc, u, which must be cc.cast or cc.compute_ptr.
    // For each,u, remove a store and replace a load with a cc.extract_value.
    for (auto &use : alloc->getUses()) {
      auto *user = use.getOwner();
      std::int32_t offset = 0;
      if (auto cptr = dyn_cast<cudaq::cc::ComputePtrOp>(user))
        offset = cptr.getRawConstantIndices()[0];
      bool isLive = false;
      for (auto &useuse : user->getUses()) {
        auto *useuser = useuse.getOwner();
        if (auto ist = dyn_cast<quake::InitializeStateOp>(useuser)) {
          LLVM_DEBUG(llvm::dbgs() << "replaced init_state\n");
          assert(conGlobal && "global must be defined");
          rewriter.replaceOpWithNewOp<quake::InitializeStateOp>(
              ist, ist.getType(), ist.getTargets(), conGlobal);
          continue;
        }
        if (auto load = dyn_cast<cudaq::cc::LoadOp>(useuser)) {
          LLVM_DEBUG(llvm::dbgs() << "replaced load\n");
          rewriter.replaceOpWithNewOp<cudaq::cc::ExtractValueOp>(
              load, eleTy, conArr,
              ArrayRef<cudaq::cc::ExtractValueArg>{offset});
          continue;
        }
        if (isa<cudaq::cc::StoreOp>(useuser))
          rewriter.eraseOp(useuser);
        isLive = true;
      }
      if (!isLive)
        rewriter.eraseOp(user);
    }
    if (toGlobal) {
      rewriter.replaceOp(alloc, conGlobal);
    } else {
      rewriter.eraseOp(alloc);
    }
    return success();
  }

  // Determine if \p alloc is a legit candidate for promotion to a constant
  // array value. \p scoreboard is a vector of store operations. Each element of
  // the allocated array must be written to exactly 1 time, and the scoreboard
  // is used to track these stores. \p dom is the dominance info for this
  // function (to ensure the stores happen before uses). \p toGlobal is returned
  // as a result. If it is `true`, then the constant array shall be lowered to a
  // global variable rather than an inline constant array.
  static bool isGoodCandidate(cudaq::cc::AllocaOp alloc,
                              SmallVectorImpl<Operation *> &scoreboard,
                              DominanceInfo &dom, bool &toGlobal) {
    LLVM_DEBUG(llvm::dbgs() << "checking candidate\n");
    toGlobal = false;
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
        if (isa<quake::InitializeStateOp>(u)) {
          toGlobalUses.push_back(u);
          toGlobal = true;
          continue;
        }
        if (isa<cudaq::cc::LoadOp>(u)) {
          loadSets[index].insert(u);
          continue;
        }
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
        if (auto index = cptr.getConstantIndex(0))
          if (auto w = getWriteOp(cptr, *index))
            if (!scoreboard[*index]) {
              scoreboard[*index] = w;
              continue;
            }
        return false;
      }
      if (auto cast = dyn_cast<cudaq::cc::CastOp>(op)) {
        if (cast.getType() == ptrArrEleTy) {
          if (auto w = getWriteOp(cast, 0))
            if (!scoreboard[0]) {
              scoreboard[0] = w;
              continue;
            }
          return false;
        }
        LLVM_DEBUG(llvm::dbgs() << "unexpected cast: " << *op << '\n');
        toGlobalUses.push_back(op);
        toGlobal = true;
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "unexpected use: " << *op << '\n');
      toGlobalUses.push_back(op);
      toGlobal = true;
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
  const std::string &funcName;
  mutable ModuleOp module;
};

// Fold complex.create ops if the arguments are constants.
class ComplexCreatePattern : public OpRewritePattern<complex::CreateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(complex::CreateOp create,
                                PatternRewriter &rewriter) const override {
    auto re = create.getReal();
    auto im = create.getImaginary();
    auto reCon = re.getDefiningOp<arith::ConstantOp>();
    auto imCon = im.getDefiningOp<arith::ConstantOp>();
    if (reCon && imCon) {
      auto aa = ArrayAttr::get(
          rewriter.getContext(),
          ArrayRef<Attribute>{reCon.getValue(), imCon.getValue()});
      rewriter.replaceOpWithNewOp<complex::ConstantOp>(create, create.getType(),
                                                       aa);
      return success();
    }
    return failure();
  }
};

class CustomUnitaryPattern
    : public OpRewritePattern<quake::CustomUnitarySymbolOp> {

public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::CustomUnitarySymbolOp customOp,
                                PatternRewriter &rewriter) const override {
    auto sref = customOp.getGenerator();
    StringRef generatorName = sref.getRootReference();
    auto parentModule = customOp->getParentOfType<ModuleOp>();

    auto ccGlobalOp = parentModule.lookupSymbol<cudaq::cc::GlobalOp>(
        generatorName.str() + ".rodata");

    if (ccGlobalOp) {
      /// ASKME: Is this okay or should we create a new operation?
      customOp.setGeneratorAttr(FlatSymbolRefAttr::get(ccGlobalOp));
      return success();
    }
    return failure();
  }
};

class LiftArrayAllocPass
    : public cudaq::opt::impl::LiftArrayAllocBase<LiftArrayAllocPass> {
public:
  using LiftArrayAllocBase::LiftArrayAllocBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();
    for (Operation &op : *module.getBody()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func)
        continue;
      DominanceInfo domInfo(func);
      std::string funcName = func.getName().str();
      RewritePatternSet patterns(ctx);
      patterns.insert<AllocaPattern>(ctx, domInfo, funcName, module);
      patterns.insert<ComplexCreatePattern>(ctx);
      patterns.insert<CustomUnitaryPattern>(ctx);

      LLVM_DEBUG(llvm::dbgs()
                 << "Before lifting constant array: " << func << '\n');

      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns))))
        signalPassFailure();

      LLVM_DEBUG(llvm::dbgs()
                 << "After lifting constant array: " << func << '\n');
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createLiftArrayAllocPass() {
  return std::make_unique<LiftArrayAllocPass>();
}