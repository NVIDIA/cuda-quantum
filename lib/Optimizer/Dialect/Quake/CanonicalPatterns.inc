/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// These canonicalization patterns are used by the canonicalize pass and not
// shared for other uses. Generally speaking, these patterns should be trivial
// peephole optimizations that reduce the size and complexity of the input IR.

// This file must be included after a `using namespace mlir;` as it uses bare
// identifiers from that namespace.

namespace {

struct AdjustAdjointExpPauliPattern
    : OpRewritePattern<cudaq::quake::ExpPauliOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExpPauliOp pauli,
                                PatternRewriter &rewriter) const override {
    if (!pauli.isAdj())
      return failure();
    SmallVector<Value> negp;
    if (!pauli.getParameters().empty())
      negp.push_back(arith::NegFOp::create(rewriter, pauli.getLoc(),
                                           pauli.getParameters()[0]));
    rewriter.replaceOpWithNewOp<cudaq::quake::ExpPauliOp>(
        pauli, pauli.getResultTypes(), UnitAttr{}, negp, pauli.getControls(),
        pauli.getTargets(), pauli.getNegatedQubitControlsAttr(),
        pauli.getPauli(), pauli.getPauliLiteralAttr());
    return success();
  }
};

// Bind an exp_pauli operation to its constant pauli word.
//
//   %a = cc.string_literal "IZ" : !cc.ptr<!cc.array<i8 x 3>>
//   %4 = cc.stdvec_init %a, %c3 : (!cc.ptr<!cc.array<i8 x 3>>, i64) ->
//           !cc.charspan
//   quake.exp_pauli (%c) %q to %4 : (f64, !quake.ref, !cc.charspan) -> ()
//   ─────────────────────────────────────────────────────────────────────
//   %a = cc.string_literal "IZ" : !cc.ptr<!cc.array<i8 x 3>>
//   %4 = cc.stdvec_init %a, %c3 : (!cc.ptr<!cc.array<i8 x 3>>, i64) ->
//           !cc.charspan  // DCE?
//   quake.exp_pauli (%c) %q to "IZ" : (f64, !quake.ref) -> ()
//
// or
//
//   %a = cc.string_literal "XX" : !cc.ptr<!cc.array<i8 x 3>>
//   quake.exp_pauli (%c) %q to %a : (f64, !quake.ref,
//           !cc.ptr<!cc.array<i8 x 3>>) -> ()
//   ─────────────────────────────────────────────────────────────────
//   %a = cc.string_literal "XX" : !cc.ptr<!cc.array<i8 x 3>>  // DCE?
//   quake.exp_pauli (%c) %q to "XX" : (f64, !quake.ref) -> ()
//
class BindExpPauliWord : public OpRewritePattern<cudaq::quake::ExpPauliOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExpPauliOp pauli,
                                PatternRewriter &rewriter) const override {
    auto pauliWord = pauli.getPauli();
    if (!pauliWord)
      return failure();
    auto vecInit = pauliWord.getDefiningOp<cudaq::cc::StdvecInitOp>();
    Value litVal = vecInit ? vecInit.getBuffer() : pauliWord;
    auto buffer = litVal.getDefiningOp<cudaq::cc::CreateStringLiteralOp>();
    if (!buffer)
      return failure();

    rewriter.replaceOpWithNewOp<cudaq::quake::ExpPauliOp>(
        pauli, pauli.getResultTypes(), pauli.getIsAdjAttr(),
        pauli.getParameters(), pauli.getControls(), pauli.getTargets(),
        pauli.getNegatedQubitControlsAttr(), Value{},
        buffer.getStringLiteralAttr());
    return success();
  }
};

// %4 = quake.veq_size %3 : (!quake.veq<10>) -> 164
// ────────────────────────────────────────────────
// %4 = constant 10 : i64
struct ForwardConstantVeqSizePattern
    : public OpRewritePattern<cudaq::quake::VeqSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::VeqSizeOp veqSize,
                                PatternRewriter &rewriter) const override {
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(veqSize.getVeq().getType());
    if (!veqTy)
      return failure();
    if (!veqTy.hasSpecifiedSize())
      return failure();
    auto resTy = veqSize.getType();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(veqSize, resTy,
                                                      veqTy.getSize());
    return success();
  }
};

// %2 = constant 10 : i32
// %3 = quake.alloca !quake.veq<?>[%2 : i32]
// ─────────────────────────────────────────
// %3 = quake.alloca !quake.veq<10>
struct FuseConstantToAllocaPattern
    : public OpRewritePattern<cudaq::quake::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    auto size = alloc.getSize();
    if (!size)
      return failure();
    auto intCon = cudaq::opt::factory::getIntIfConstant(size);
    if (!intCon)
      return failure();
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(alloc.getType());
    if (!veqTy)
      return failure();
    if (veqTy.hasSpecifiedSize())
      return failure();
    auto loc = alloc.getLoc();
    auto resTy = alloc.getType();
    auto newAlloc = cudaq::quake::AllocaOp::create(
        rewriter, loc, static_cast<std::size_t>(*intCon));
    rewriter.replaceOpWithNewOp<cudaq::quake::RelaxSizeOp>(alloc, resTy,
                                                           newAlloc);
    return success();
  }
};

// %2 = constant 10 : i32
// %3 = quake.extract_ref %1[%2] : (!quake.veq<?>, i32) -> !quake.ref
// ──────────────────────────────────────────────────────────────────
// %3 = quake.extract_ref %1[10] : (!quake.veq<?>) -> !quake.ref
struct FuseConstantToExtractRefPattern
    : public OpRewritePattern<cudaq::quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    auto index = extract.getIndex();
    if (!index)
      return failure();
    auto intCon = cudaq::opt::factory::getIntIfConstant(index);
    if (!intCon)
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::quake::ExtractRefOp>(
        extract, extract.getVeq(), static_cast<std::size_t>(*intCon));
    return success();
  }
};

// %4 = quake.concat %2, %3 : (!quake.ref, !quake.ref) -> !quake.veq<2>
// %7 = quake.extract_ref %4[0] : (!quake.veq<2>) -> !quake.ref
// ───────────────────────────────────────────
// replace all use with %2
struct ForwardConcatExtractPattern
    : public OpRewritePattern<cudaq::quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    auto veq = extract.getVeq();
    auto concatOp = veq.getDefiningOp<cudaq::quake::ConcatOp>();
    if (concatOp && extract.hasConstantIndex()) {
      // Don't run this canonicalization if any of the operands
      // to concat are of type veq.
      auto concatQubits = concatOp.getTargets();
      for (auto qOp : concatQubits)
        if (isa<cudaq::quake::VeqType>(qOp.getType()))
          return failure();

      // concat only has ref type operands.
      auto index = extract.getConstantIndex();
      if (index < concatQubits.size()) {
        auto qOpValue = concatQubits[index];
        if (isa<cudaq::quake::RefType>(qOpValue.getType())) {
          rewriter.replaceOp(extract, {qOpValue});
          return success();
        }
      }
    }
    return failure();
  }
};

// %2 = quake.concat %1 : (!quake.ref) -> !quake.veq<1>
// %3 = quake.extract_ref %2[0] : (!quake.veq<1>) -> !quake.ref
// quake.* %3 ...
// ───────────────────────────────────────────
// quake.* %1 ...
struct ForwardConcatExtractSingleton
    : public OpRewritePattern<cudaq::quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    if (auto concat = extract.getVeq().getDefiningOp<cudaq::quake::ConcatOp>())
      if (concat.getType().getSize() == 1 && extract.hasConstantIndex() &&
          extract.getConstantIndex() == 0) {
        assert(concat.getTargets().size() == 1 && concat.getTargets()[0]);
        extract.getResult().replaceUsesWithIf(
            concat.getTargets()[0], [&](OpOperand &use) {
              if (Operation *user = use.getOwner())
                return isQuakeOperation(user);
              return false;
            });
        return success();
      }
    return failure();
  }
};

static Value createCast(PatternRewriter &rewriter, Location loc, Value inVal) {
  auto i64Ty = rewriter.getI64Type();
  assert(inVal.getType() != rewriter.getIndexType() &&
         "use of index type is deprecated");
  return cudaq::cc::CastOp::create(rewriter, loc, i64Ty, inVal,
                                   cudaq::cc::CastOpMode::Unsigned);
}

class ExtractRefFromSubVeqPattern
    : public OpRewritePattern<cudaq::quake::ExtractRefOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  // Replace a pattern such as:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %2 = quake.subveq %1, %c2, %c3 : (!quake.veq<4>, i32, i32) ->
  //        !quake.veq<2>
  //   %3 = quake.extract_ref %2[0] : (!quake.veq<2>) -> !quake.ref
  // ```
  // with:
  // ```
  //   %1 = ... : !quake.veq<4>
  //   %3 = quake.extract_ref %1[2] : (!uwake.veq<4>) -> !quake.ref
  // ```
  LogicalResult matchAndRewrite(cudaq::quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    auto subveq = extract.getVeq().getDefiningOp<cudaq::quake::SubVeqOp>();
    if (!subveq)
      return failure();
    // Let the combining of back-to-back subveq ops happen first.
    auto subsub = subveq.getVeq().getDefiningOp<cudaq::quake::SubVeqOp>();
    if (subsub)
      return failure();

    Value offset;
    auto loc = extract.getLoc();
    auto low = [&]() -> Value {
      if (subveq.hasConstantLowerBound())
        return arith::ConstantIntOp::create(rewriter, loc,
                                            subveq.getConstantLowerBound(), 64);
      return subveq.getLower();
    }();
    if (extract.hasConstantIndex()) {
      Value cv = arith::ConstantIntOp::create(rewriter, loc, low.getType(),
                                              extract.getConstantIndex());
      offset = arith::AddIOp::create(rewriter, loc, cv, low);
    } else {
      auto cast1 = createCast(rewriter, loc, extract.getIndex());
      auto cast2 = createCast(rewriter, loc, low);
      offset = arith::AddIOp::create(rewriter, loc, cast1, cast2);
    }
    rewriter.replaceOpWithNewOp<cudaq::quake::ExtractRefOp>(
        extract, subveq.getVeq(), offset);
    return success();
  }
};

// %7 = quake.concat %4 : (!quake.veq<2>) -> !quake.veq<2>
// ───────────────────────────────────────────
// removed
struct ConcatNoOpPattern : public OpRewritePattern<cudaq::quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    // Remove concat veq<N> -> veq<N>
    // or
    // concat ref -> ref
    auto qubitsToConcat = concat.getTargets();
    if (qubitsToConcat.size() > 1)
      return failure();

    // We only want to handle veq -> veq here.
    if (isa<cudaq::quake::RefType>(qubitsToConcat.front().getType())) {
      return failure();
    }

    // Do not handle anything where we don't know the sizes.
    auto retTy = concat.getResult().getType();
    if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(retTy))
      if (!veqTy.hasSpecifiedSize())
        // This could be a folded quake.relax_size op.
        return failure();

    rewriter.replaceOp(concat, qubitsToConcat);
    return success();
  }
};

// %8 = quake.concat %4, %5, %6 : (!quake.ref, !quake.veq<4>,
//        !quake.veq<2>) -> !quake.veq<?>
// ───────────────────────────────────────────────────────────
// %.8 = quake.concat %4, %5, %6 : (!quake.ref, !quake.veq<4>,
//        !quake.veq<2>) -> !quake.veq<7>
// %8 = quake.relax_size %.8 : (!quake.veq<7>) -> !quake.veq<?>
struct ConcatSizePattern : public OpRewritePattern<cudaq::quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat.getType().hasSpecifiedSize())
      return failure();

    auto *ctx = rewriter.getContext();
    auto loc = concat.getLoc();
    auto refTy = cudaq::quake::RefType::get(ctx);
    // Walk the arguments and sum them, if possible.
    std::size_t sum = 0;
    SmallVector<Value> targets;
    for (auto opnd : concat.getTargets()) {
      if (auto veqTy = dyn_cast<cudaq::quake::VeqType>(opnd.getType())) {
        if (!veqTy.hasSpecifiedSize())
          return failure();
        sum += veqTy.getSize();
        targets.push_back(opnd);
        continue;
      }
      if (auto stqTy = dyn_cast<cudaq::quake::StruqType>(opnd.getType())) {
        if (!stqTy.hasSpecifiedSize())
          return failure();
        sum += cudaq::quake::getAllocationSize(stqTy);
        auto arity = stqTy.getArity();
        if (*arity) {
          // Get each member for IR legalization.
          for (auto [i, memTy] : llvm::enumerate(stqTy.getMembers())) {
            auto mem = cudaq::quake::GetMemberOp::create(rewriter, loc, memTy,
                                                         opnd, i);
            targets.push_back(mem);
          }
        }
        continue;
      }
      if (opnd.getType() == refTy) {
        sum++;
        targets.push_back(opnd);
        continue;
      }
      return failure();
    }

    // Leans into the relax_size canonicalization pattern.
    auto newTy = cudaq::quake::VeqType::get(ctx, sum);
    Value newOp = cudaq::quake::ConcatOp::create(rewriter, loc, newTy, targets);
    auto noSizeTy = cudaq::quake::VeqType::getUnsized(ctx);
    rewriter.replaceOpWithNewOp<cudaq::quake::RelaxSizeOp>(concat, noSizeTy,
                                                           newOp);
    return success();
  }
};

// %0 = quake.alloca !quake.veq<2>
// %1 = quake.extract_ref %0[0] : (!quake.veq<2>) -> !quake.ref
// %2 = quake.extract_ref %0[1] : (!quake.veq<2>) -> !quake.ref
// %3 = quake.concat %1, %2 : (!quake.ref, !quake.ref) -> !quake.veq<2>
// ────────────────────────────────────────────────────────────────
// [%0/%3] -- replace uses of %3 with %0
struct UselessConcatOpPattern
    : public OpRewritePattern<cudaq::quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    auto *ctx = rewriter.getContext();
    auto refTy = cudaq::quake::RefType::get(ctx);

    // Make sure this subgraph matches the pattern.
    cudaq::quake::AllocaOp alloc;
    for (auto iter : llvm::enumerate(concat.getTargets())) {
      auto arg = iter.value();
      if (arg.getType() != refTy)
        return failure();
      auto extract = arg.getDefiningOp<cudaq::quake::ExtractRefOp>();
      if (!extract || !extract.hasConstantIndex())
        return failure();
      auto i = iter.index();
      // The order of the operands must be exactly the interval $[0..n)$.
      if (extract.getConstantIndex() != i)
        return failure();
      if (alloc) {
        if (extract.getVeq() != alloc.getResult())
          return failure();
      } else {
        auto veqAlloc =
            extract.getVeq().getDefiningOp<cudaq::quake::AllocaOp>();
        if (!veqAlloc || veqAlloc.getSize())
          return failure();
        alloc = veqAlloc;
      }
    }
    auto allocSize = cudaq::quake::getAllocationSize(alloc.getType());
    if (concat.getTargets().size() != allocSize)
      return failure();

    // This quake.concat is entirely redundant with `alloc`.
    rewriter.replaceOp(concat, alloc.getResult());
    return success();
  }
};

// %7 = quake.make_struq %5, %6 : (!quake.veq<A>, !quake.veq<N>) ->
//                    !quake.struq<!quake.veq<A>, !quake.veq<N>>
// %8 = quake.get_member %7[1] : (!quake.struq<!quake.veq<A>,
//                                !quake.veq<N>>) -> !quake.veq<N>
// ────────────────────────────────────────────────────────────────
// [%6/%8] -- replace uses of %8 with %6
struct BypassMakeStruq : public OpRewritePattern<cudaq::quake::GetMemberOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::GetMemberOp getMem,
                                PatternRewriter &rewriter) const override {
    auto makeStruq =
        getMem.getStruq().getDefiningOp<cudaq::quake::MakeStruqOp>();
    if (!makeStruq)
      return failure();
    auto toStrTy = cast<cudaq::quake::StruqType>(getMem.getStruq().getType());
    std::uint32_t idx = getMem.getIndex();
    Value from = makeStruq.getOperand(idx);
    auto toTy = toStrTy.getMembers()[idx];
    if (from.getType() != toTy)
      rewriter.replaceOpWithNewOp<cudaq::quake::RelaxSizeOp>(getMem, toTy,
                                                             from);
    else
      rewriter.replaceOp(getMem, from);
    return success();
  }
};

// %22 = quake.init_state %1, %2 : (!quake.veq<k>, T) -> !quake.veq<?>
// ────────────────────────────────────────────────────────────────────
// %.22 = quake.init_state %1, %2 : (!quake.veq<k>, T) -> !quake.veq<k>
// %22 = quake.relax_size %.22 : (!quake.veq<k>) -> !quake.veq<?>
struct ForwardAllocaTypePattern
    : public OpRewritePattern<cudaq::quake::InitializeStateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    if (auto isTy = dyn_cast<cudaq::quake::VeqType>(initState.getType()))
      if (!isTy.hasSpecifiedSize()) {
        auto targ = initState.getTargets();
        if (auto targTy = dyn_cast<cudaq::quake::VeqType>(targ.getType()))
          if (targTy.hasSpecifiedSize()) {
            auto newInit = cudaq::quake::InitializeStateOp::create(
                rewriter, initState.getLoc(), targTy, targ,
                initState.getState());
            rewriter.replaceOpWithNewOp<cudaq::quake::RelaxSizeOp>(
                initState, isTy, newInit);
            return success();
          }
      }

    // Remove any intervening cast to !cc.ptr<!cc.array<T x ?>> ops.
    if (auto stateCast =
            initState.getState().getDefiningOp<cudaq::cc::CastOp>())
      if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(stateCast.getType())) {
        auto eleTy = ptrTy.getElementType();
        if (auto arrTy = dyn_cast<cudaq::cc::ArrayType>(eleTy))
          if (arrTy.isUnknownSize()) {
            rewriter.replaceOpWithNewOp<cudaq::quake::InitializeStateOp>(
                initState, initState.getTargets().getType(),
                initState.getTargets(), stateCast.getValue());
            return success();
          }
      }
    return failure();
  }
};

// %3 = quake.subveq %0, 4, 10 : (!quake.veq<12>, i64, i64) -> !quake.veq<?>
// ──────────────────────────────────────────────────────────────────────────
// %.3 = quake.subveq %0, 4, 10 : (!quake.veq<12>, i64, i64) -> !quake.veq<7>
// %3 = quake.relax_size %.3 : (!quake.veq<7>) -> !quake.veq<?>
struct FixUnspecifiedSubveqPattern
    : public OpRewritePattern<cudaq::quake::SubVeqOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::SubVeqOp subveq,
                                PatternRewriter &rewriter) const override {
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(subveq.getType());
    if (veqTy && veqTy.hasSpecifiedSize())
      return failure();
    if (!(subveq.hasConstantLowerBound() && subveq.hasConstantUpperBound()))
      return failure();
    auto *ctx = rewriter.getContext();
    std::size_t size =
        subveq.getConstantUpperBound() - subveq.getConstantLowerBound() + 1u;
    auto szVecTy = cudaq::quake::VeqType::get(ctx, size);
    auto loc = subveq.getLoc();
    auto subv = cudaq::quake::SubVeqOp::create(
        rewriter, loc, szVecTy, subveq.getVeq(), subveq.getLower(),
        subveq.getUpper(), subveq.getRawLower(), subveq.getRawUpper());
    rewriter.replaceOpWithNewOp<cudaq::quake::RelaxSizeOp>(subveq, veqTy, subv);
    return success();
  }
};

// %1 = constant 4 : i64
// %2 = constant 10 : i64
// %3 = quake.subveq %0, %1, %2 : (!quake.veq<12>, i64, i64) -> !quake.veq<?>
// ──────────────────────────────────────────────────────────────────────────
// %3 = quake.subveq %0, 4, 10 : (!quake.veq<12>, i64, i64) -> !quake.veq<7>
struct FuseConstantToSubveqPattern
    : public OpRewritePattern<cudaq::quake::SubVeqOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::SubVeqOp subveq,
                                PatternRewriter &rewriter) const override {
    if (subveq.hasConstantLowerBound() && subveq.hasConstantUpperBound())
      return failure();
    bool regen = false;
    std::int64_t lo = subveq.getConstantLowerBound();
    auto veqSize = [&]() -> std::int64_t {
      auto size =
          cast<cudaq::quake::VeqType>(subveq.getVeq().getType()).getSize();
      // Note: do not support more than 2^32 qubits.
      if (size == cudaq::quake::VeqType::kDynamicSize)
        return std::numeric_limits<std::int32_t>::max();
      return size - 1;
    }();
    Value loVal = subveq.getLower();
    if (!subveq.hasConstantLowerBound())
      if (auto olo = cudaq::opt::factory::getIntIfConstant(subveq.getLower())) {
        regen = true;
        loVal = nullptr;
        lo = *olo;
      }

    std::int64_t hi = subveq.getConstantUpperBound();
    Value hiVal = subveq.getUpper();
    if (!subveq.hasConstantUpperBound())
      if (auto ohi = cudaq::opt::factory::getIntIfConstant(subveq.getUpper())) {
        regen = true;
        hiVal = nullptr;
        hi = *ohi;
      }

    if (!regen)
      return failure();
    if ((!loVal && (lo < 0 || lo > veqSize)) ||
        (!hiVal && (hi < 0 || hi > veqSize)) || (!loVal && !hiVal && lo > hi)) {
      // If any invalid conditions with the constants then replace the subveq
      // with a poison value.
      auto poison = rewriter.replaceOpWithNewOp<cudaq::cc::PoisonOp>(
          subveq, subveq.getType());
      poison.emitWarning("subrange of qvector is invalid.");
    } else {
      rewriter.replaceOpWithNewOp<cudaq::quake::SubVeqOp>(
          subveq, subveq.getType(), subveq.getVeq(), loVal, hiVal, lo, hi);
    }
    return success();
  }
};

// Replace subveq operations that extract the entire original register with the
// original register.
struct RemoveSubVeqNoOpPattern
    : public OpRewritePattern<cudaq::quake::SubVeqOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::SubVeqOp subVeqOp,
                                PatternRewriter &rewriter) const override {
    auto origVeq = subVeqOp.getVeq();
    // The original veq size must be known
    auto veqType = dyn_cast<cudaq::quake::VeqType>(origVeq.getType());
    if (!veqType.hasSpecifiedSize())
      return failure();
    if (!(subVeqOp.hasConstantLowerBound() && subVeqOp.hasConstantUpperBound()))
      return failure();

    // If the subveq is the whole register, than the start value must be 0.
    if (subVeqOp.getConstantLowerBound() != 0)
      return failure();

    // If the sizes are equal, then replace
    if (veqType.getSize() != subVeqOp.getConstantUpperBound() + 1)
      return failure();

    // this subveq is the whole original register, hence a no-op
    rewriter.replaceOp(subVeqOp, origVeq);
    return success();
  }
};

// Combine back-to-back quake.subveq operations.
//
// %10 = quake.subveq %4, 1, 6 : (!quake.veq<?>) -> !quake.veq<7>
// %11 = quake.subveq %10, 0, 2 : (!quake.veq<7>) -> !quake.veq<3>
// ───────────────────────────────────────────────────────────────
// %11 = quake.subveq %4, 1, 3 : (!quake.veq<?>) -> !quake.veq<3>
class CombineSubVeqsPattern : public OpRewritePattern<cudaq::quake::SubVeqOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::SubVeqOp subveq,
                                PatternRewriter &rewriter) const override {
    auto prior = subveq.getVeq().getDefiningOp<cudaq::quake::SubVeqOp>();
    if (!prior)
      return failure();

    auto loc = subveq.getLoc();

    // Lambda to create a Value for the lower bound of `s`.
    auto lofunc = [&](cudaq::quake::SubVeqOp s) -> Value {
      if (s.hasConstantLowerBound())
        return arith::ConstantIntOp::create(rewriter, loc,
                                            s.getConstantLowerBound(), 64);
      return s.getLower();
    };
    auto priorlo = lofunc(prior);
    auto svlo = lofunc(subveq);

    // Lambda for creating the upper bound Value.
    auto svup = [&]() -> Value {
      if (subveq.hasConstantUpperBound())
        return arith::ConstantIntOp::create(rewriter, loc,
                                            subveq.getConstantUpperBound(), 64);
      return subveq.getUpper();
    }();
    auto cast1 = createCast(rewriter, loc, priorlo);
    auto cast2 = createCast(rewriter, loc, svlo);
    auto cast3 = createCast(rewriter, loc, svup);
    Value sum1 = arith::AddIOp::create(rewriter, loc, cast1, cast2);
    Value sum2 = arith::AddIOp::create(rewriter, loc, cast1, cast3);
    auto veqTy = subveq.getType();
    rewriter.replaceOpWithNewOp<cudaq::quake::SubVeqOp>(
        subveq, veqTy, prior.getVeq(), sum1, sum2);
    return success();
  }
};

// %11 = quake.init_state %_, %_ : (!quake.veq<2>, T1) -> !quake.veq<?>
// %12 = quake.veq_size %11 : (!quake.veq<?>) -> i64
// ────────────────────────────────────────────────────────────────────
// %11 = quake.init_state %_, %_ : (!quake.veq<2>, T1) -> !quake.veq<?>
// %12 = constant 2 : i64
struct FoldInitStateSizePattern
    : public OpRewritePattern<cudaq::quake::VeqSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::VeqSizeOp veqSize,
                                PatternRewriter &rewriter) const override {
    Value veq = veqSize.getVeq();
    if (auto initState = veq.getDefiningOp<cudaq::quake::InitializeStateOp>())
      if (auto veqTy =
              dyn_cast<cudaq::quake::VeqType>(initState.getTargets().getType()))
        if (veqTy.hasSpecifiedSize()) {
          std::size_t numQubits = veqTy.getSize();
          rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
              veqSize, veqSize.getType(), numQubits);
          return success();
        }
    return failure();
  }
};

// If there is no operation that modifies the wire after it gets unwrapped and
// before it is wrapped, then the wrap operation is a nop and can be
// eliminated.
struct KillDeadWrapPattern : public OpRewritePattern<cudaq::quake::WrapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::WrapOp wrap,
                                PatternRewriter &rewriter) const override {
    if (auto unwrap =
            wrap.getWireValue().getDefiningOp<cudaq::quake::UnwrapOp>())
      rewriter.eraseOp(wrap);
    return success();
  }
};

template <typename OP>
struct MergeRotationPattern : public OpRewritePattern<OP> {
  using Base = OpRewritePattern<OP>;
  using Base::Base;

  LogicalResult matchAndRewrite(OP rotate,
                                PatternRewriter &rewriter) const override {
    auto wireTy = cudaq::quake::WireType::get(rewriter.getContext());
    if (rotate.getTarget(0).getType() != wireTy ||
        !rotate.getControls().empty())
      return failure();
    assert(!rotate.getNegatedQubitControls());
    auto input = rotate.getTarget(0).template getDefiningOp<OP>();
    if (!input || !input.getControls().empty())
      return failure();
    assert(!input.getNegatedQubitControls());

    // At this point, we have
    //   %input  = quake.rotate %angle1, %wire
    //   %rotate = quake.rotate %angle2, %input
    // Replace those ops with
    //   %new    = quake.rotate (%angle1 + %angle2), %wire
    auto loc = rotate.getLoc();
    auto angle1 = input.getParameter(0);
    auto angle2 = rotate.getParameter(0);
    if (angle1.getType() != angle2.getType())
      return failure();
    auto adjAttr = rotate.getIsAdjAttr();
    auto newAngle = [&]() -> Value {
      if (input.isAdj() == rotate.isAdj())
        return arith::AddFOp::create(rewriter, loc, angle1, angle2);
      // One is adjoint, so it should be subtracted from the other.
      if (input.isAdj())
        return arith::SubFOp::create(rewriter, loc, angle2, angle1);
      adjAttr = input.getIsAdjAttr();
      return arith::SubFOp::create(rewriter, loc, angle1, angle2);
    }();
    rewriter.replaceOpWithNewOp<OP>(rotate, rotate.getResultTypes(), adjAttr,
                                    ValueRange{newAngle}, ValueRange{},
                                    ValueRange{input.getTarget(0)},
                                    rotate.getNegatedQubitControlsAttr());
    return success();
  }
};

// Forward the argument to a relax_size to the users for all users that are
// quake operations. All quake ops that take a sized veq argument are
// polymorphic on all veq types. If the op is not a quake op, then maintain
// strong typing.
struct ForwardRelaxedSizePattern
    : public OpRewritePattern<cudaq::quake::RelaxSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::RelaxSizeOp relax,
                                PatternRewriter &rewriter) const override {
    auto inpVec = relax.getInputVec();
    bool replaced = false;
    rewriter.replaceUsesWithIf(relax, inpVec, [&](OpOperand &use) {
      bool res = false;
      if (Operation *user = use.getOwner())
        res = isQuakeOperation(user) && !isa<cudaq::quake::ApplyOp>(user);
      replaced = replaced || res;
      return res;
    });
    // return success if and only if at least one use was replaced.
    return success(replaced);
  };
};

} // namespace
