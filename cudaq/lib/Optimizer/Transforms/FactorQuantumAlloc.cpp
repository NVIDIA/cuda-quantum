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

// Resolves the inclusive [lower, upper] bounds of a `quake.subveq` if they
// are statically known. "Statically known" means either:
//   * the op carries a constant lower/upper bound attribute
//     (`hasConstantLowerBound`/`hasConstantUpperBound`), or
//   * the op's SSA lower/upper operand is a constant integer recognized by
//     `cudaq::opt::factory::getIntIfConstant`.
// Returns `std::nullopt` otherwise.
//
// `quake.subveq` bounds are inclusive on both ends; see
// `FixUnspecifiedSubveqPattern` and `CombineMeasurements` for the shared
// convention.
static std::optional<std::pair<std::int64_t, std::int64_t>>
subveqStaticBounds(cudaq::quake::SubVeqOp op) {
  std::int64_t lower;
  if (op.hasConstantLowerBound())
    lower = static_cast<std::int64_t>(op.getConstantLowerBound());
  else if (auto v = cudaq::opt::factory::getIntIfConstant(op.getLower()))
    lower = *v;
  else
    return std::nullopt;
  std::int64_t upper;
  if (op.hasConstantUpperBound())
    upper = static_cast<std::int64_t>(op.getConstantUpperBound());
  else if (auto v = cudaq::opt::factory::getIntIfConstant(op.getUpper()))
    upper = *v;
  else
    return std::nullopt;
  return std::make_pair(lower, upper);
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

  // A subveq is convertible only when its bounds are statically known and
  // every sub-user is itself convertible (so the per-element walk in
  // `rewriteOpAndUsers` can complete factoring of the parent veq alloca).
  if (!subveqStaticBounds(sub))
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
      if (newAllocs.empty()) {
        // The python bridge can generate quake.alloca with size 0, which has no
        // semantics. Check here rather than propagating the bad IR.
        return failure();
      }
      // Uses are more complex so just concat the refs together.
      SmallVector<Value> theRefs;
      std::for_each(
          newAllocs.begin(), newAllocs.end(),
          [&](cudaq::quake::AllocaOp a) { theRefs.push_back(a.getResult()); });
      auto concatOp = rewriter.replaceOpWithNewOp<cudaq::quake::ConcatOp>(
          allocOp, veqTy, theRefs);

      // Even in the fallback path, fold any constant-bounded `quake.subveq`
      // by replacing it with a narrow `quake.concat` composed of the
      // per-element alloca refs in the inclusive [lower, upper] range. Without
      // this, downstream passes -- cable-rough-in in particular -- see a
      // `subveq` whose defining op is opaque to them, so calls like
      // `wait(q[1:3], d)` (which lower to `subveq` feeding a `call`) are not
      // converted to wire form.
      //
      // Statically known bounds come from either the op's constant bound
      // attribute (`hasConstantLowerBound` / `hasConstantUpperBound`) or a
      // constant SSA integer recognized by
      // `cudaq::opt::factory::getIntIfConstant`. Bounds are inclusive on both
      // ends, per the shared convention with `FixUnspecifiedSubveqPattern`
      // and `CombineMeasurements`. The new narrow concat has the same sized
      // veq type as the source subveq produced (and is relaxed to an unsized
      // veq downstream if a `relax_size` was wrapping the source subveq --
      // `ForwardRelaxedSizePattern` cleans this up on the next canonicalize).
      //
      // We also re-process users of the new narrow concat, so chained
      // `quake.subveq` operations (subveq(subveq(%v))) fold all the way down
      // in a single pass; bounds of a chained subveq are relative to the
      // narrow concat it slices, not to the parent alloca.
      SmallVector<cudaq::quake::ConcatOp> concatsToProcess{concatOp};
      while (!concatsToProcess.empty()) {
        auto currentConcat = concatsToProcess.pop_back_val();
        auto targets = currentConcat.getTargets();
        auto viewSize = static_cast<std::int64_t>(targets.size());
        SmallVector<Operation *> users{currentConcat->getUsers().begin(),
                                      currentConcat->getUsers().end()};
        for (auto *user : users) {
          // Reach the underlying `quake.subveq`, peeling a `relax_size`
          // wrapper if present (the canonical shape produced by
          // `FixUnspecifiedSubveqPattern` for an unsized-veq subveq with
          // constant bound attributes).
          auto subveq = dyn_cast<cudaq::quake::SubVeqOp>(user);
          if (!subveq) {
            if (auto relax = dyn_cast<cudaq::quake::RelaxSizeOp>(user))
              subveq =
                  relax.getInputVec().getDefiningOp<cudaq::quake::SubVeqOp>();
            if (!subveq)
              continue;
          }

          auto bounds = subveqStaticBounds(subveq);
          if (!bounds)
            continue;
          auto low = bounds->first;
          auto up = bounds->second;
          // [low, up] are relative to the current concat's view of the parent
          // veq. For the parent concat itself this view spans the entire
          // allocation (`viewSize == size`); for a chained narrow concat it
          // is the narrow slice it was produced from. In all cases
          // `viewSize <= size`, so checking `up < viewSize` covers both.
          if (low < 0 || up < 0 || low > up ||
              low >= viewSize ||
              up >= viewSize)
            continue;
          std::size_t sliceSize = static_cast<std::size_t>(up - low + 1);

          SmallVector<Value> slicedRefs;
          slicedRefs.reserve(sliceSize);
          for (std::int64_t i = low; i <= up; ++i)
            slicedRefs.push_back(
                targets[static_cast<std::size_t>(i)]);
          auto narrowTy = cudaq::quake::VeqType::get(ctx, sliceSize);
          rewriter.setInsertionPoint(subveq);
          auto narrowConcat = cudaq::quake::ConcatOp::create(
              rewriter, subveq->getLoc(), narrowTy, slicedRefs);
          rewriter.replaceOp(subveq, narrowConcat.getResult());

          // Re-process users of the new narrow concat so chained subveqs
          // (subveq(subveq(%v))) fold all the way down.
          concatsToProcess.push_back(narrowConcat);
        }
      }
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
        auto bounds = subveqStaticBounds(subveq);
        if (!bounds)
          return failure();
        auto lowInt = bounds->first;
        SmallVector<Operation *> subUsers{subveq->getUsers().begin(),
                                          subveq->getUsers().end()};
        for (auto *subUser : subUsers)
          if (failed(rewriteOpAndUsers(subUser, lowInt, rewriter, size,
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
} // namespace

/// Compute the index within the `concat` result at which the operand at
/// position \p operandPos of \p concat lands. Returns `nullopt` if the index
/// cannot be determined statically (e.g., a preceding `veq` has an unspecified
/// size).
static std::optional<std::size_t>
concatResultIndex(cudaq::quake::ConcatOp concat, std::size_t operandPos) {
  std::size_t idx = 0;
  auto refTy = cudaq::quake::RefType::get(concat.getContext());
  for (std::size_t i = 0; i < operandPos; ++i) {
    auto t = concat.getTargets()[i].getType();
    if (t == refTy) {
      idx += 1;
    } else if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(t)) {
      if (!veqTy.hasSpecifiedSize())
        return std::nullopt;
      idx += veqTy.getSize();
    } else {
      return std::nullopt;
    }
  }
  return idx;
}

namespace {
/// When individual `!quake.ref` `alloca`s are aliased into a `veq` via
/// `quake.concat` and then also used directly by gate or measurement `ops` that
/// appear *after* the `concat` — either in the same block or inside a nested
/// region of an op that follows the concat — this pattern replaces each such
/// direct use with a fresh `quake.extract_ref` from the `concat` result at the
/// correct index, making every post-`concat` qubit access go through the `veq`
/// and eliminating use-def chain aliasing via the `concat` op.
class ConcatAllocaAliasPattern
    : public OpRewritePattern<cudaq::quake::ConcatOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    Block *concatBlock = concat->getBlock();
    Value concatResult = concat.getResult();
    bool modified = false;
    auto refTy = cudaq::quake::RefType::get(rewriter.getContext());

    // Returns true if `user` is dominated by `concat`.  This covers:
    //   (a) same-block uses that come after the concat, and
    //   (b) uses inside nested regions of ops that come after the concat.
    auto dominatedByConcat = [&](Operation *user) -> bool {
      // Walk up the parent-op chain until we reach concatBlock.
      Block *block = user->getBlock();
      while (block != concatBlock) {
        Operation *parentOp = block->getParentOp();
        if (!parentOp)
          return false;
        block = parentOp->getBlock();
        if (!block)
          return false;
        // If we found concatBlock, check that the containing op follows concat.
        if (block == concatBlock)
          return concat->isBeforeInBlock(parentOp);
      }
      // Same block as concat — just check ordering.
      return concat->isBeforeInBlock(user);
    };

    for (auto [pos, operand] : llvm::enumerate(concat.getTargets())) {
      if (operand.getType() != refTy ||
          !operand.getDefiningOp<cudaq::quake::AllocaOp>())
        continue;

      auto maybeIdx = concatResultIndex(concat, pos);
      if (!maybeIdx)
        continue;
      std::size_t veqIdx = *maybeIdx;

      // Replace every direct use of this alloca that (a) is dominated by the
      // concat (same block after, or nested in an op after), and (b) is not the
      // concat itself or a dealloc (deallocs manage lifetime, not qubit state).
      for (OpOperand &use : llvm::make_early_inc_range(operand.getUses())) {
        auto *user = use.getOwner();
        if (user == concat.getOperation())
          continue;
        if (isa<cudaq::quake::DeallocOp>(user))
          continue;
        if (!dominatedByConcat(user))
          continue;

        rewriter.setInsertionPoint(user);
        auto freshRef = cudaq::quake::ExtractRefOp::create(
            rewriter, user->getLoc(), concatResult, veqIdx);
        use.set(freshRef.getResult());
        modified = true;
      }
    }

    return success(modified);
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
    patterns.insert<ConcatAllocaAliasPattern>(ctx);
    if (failed(applyPatternsGreedily(func, std::move(patterns))))
      return failure();
    return success();
  }
};
} // namespace
