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
    auto dict = pauli->getDiscardableAttrDictionary();
    auto newPauli = rewriter.replaceOpWithNewOp<cudaq::quake::ExpPauliOp>(
        pauli, pauli.getResultTypes(), UnitAttr{}, negp, pauli.getControls(),
        pauli.getTargets(), pauli.getNegatedQubitControlsAttr(),
        pauli.getPauli(), pauli.getPauliLiteralAttr());
    newPauli->setDiscardableAttrs(std::move(dict));
    return success();
  }
};

// Bind an exp_pauli operation to its constant pauli word.
//
//   %a = cc.string_literal "IZ" : !cc.ptr<!cc.array<i8 x 3>>
//   %4 = cc.sequence_init %a, %c3 : (!cc.ptr<!cc.array<i8 x 3>>, i64) ->
//           !cc.charspan
//   quake.exp_pauli (%c) %q to %4 : (f64, !quake.ref, !cc.charspan) -> ()
//   ─────────────────────────────────────────────────────────────────────
//   %a = cc.string_literal "IZ" : !cc.ptr<!cc.array<i8 x 3>>
//   %4 = cc.sequence_init %a, %c3 : (!cc.ptr<!cc.array<i8 x 3>>, i64) ->
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
    auto vecInit = pauliWord.getDefiningOp<cudaq::cc::SequenceInitOp>();
    Value litVal = vecInit ? vecInit.getBuffer() : pauliWord;
    auto buffer = litVal.getDefiningOp<cudaq::cc::CreateStringLiteralOp>();
    if (!buffer)
      return failure();

    auto dict = pauli->getDiscardableAttrDictionary();
    auto newPauli = rewriter.replaceOpWithNewOp<cudaq::quake::ExpPauliOp>(
        pauli, pauli.getResultTypes(), pauli.getIsAdjAttr(),
        pauli.getParameters(), pauli.getControls(), pauli.getTargets(),
        pauli.getNegatedQubitControlsAttr(), Value{},
        buffer.getStringLiteralAttr());
    newPauli->setDiscardableAttrs(std::move(dict));
    return success();
  }
};

// %4 = quake.veq_size %3 : (!quake.veq<10>) -> i64
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

// %3 = cc.undef !quake.veq<?>
// %4 = quake.veq_size %3 : (!quake.veq<?>) -> i64
// ────────────────────────────────────────────────
// %4 = constant 0 : i64
struct ForwardEmptyVeqSizePattern
    : public OpRewritePattern<cudaq::quake::VeqSizeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::VeqSizeOp veqSize,
                                PatternRewriter &rewriter) const override {
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(veqSize.getVeq().getType());
    if (!veqTy)
      return failure();
    if (veqTy.hasSpecifiedSize())
      return failure();
    auto undef = veqSize.getVeq().getDefiningOp<cudaq::cc::UndefOp>();
    if (!undef) {
      auto poison = veqSize.getVeq().getDefiningOp<cudaq::cc::PoisonOp>();
      if (!poison)
        return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(veqSize,
                                                      veqSize.getType(), 0);
    return success();
  }
};

// %0 = quake.alloca !quake.veq<0>
// ─────────────────────────────────
// %0 = cc.poison !quake.veq<0>
struct ReplaceZeroSizeAllocaPattern
    : public OpRewritePattern<cudaq::quake::AllocaOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::AllocaOp alloc,
                                PatternRewriter &rewriter) const override {
    auto veqTy = dyn_cast<cudaq::quake::VeqType>(alloc.getType());
    if (!veqTy || !veqTy.hasSpecifiedSize() || veqTy.getSize() != 0)
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::cc::PoisonOp>(alloc, veqTy);
    return success();
  }
};

// `!quake.veq<0>` has no semantics of its own: it is only permitted because a
// frontend may legitimately materialize an empty qubit vector. Once such an
// operand shows up as a control (or an apply control), it is pure baggage --
// there is nothing there to gate on -- so peel it out here rather than let
// every later pass special-case a zero-size veq.
static bool isEmptyVeqType(Value value) {
  auto veqTy = dyn_cast<cudaq::quake::VeqType>(value.getType());
  if (veqTy && veqTy.hasSpecifiedSize() && veqTy.getSize() == 0)
    return true;
  if (auto relax = value.getDefiningOp<cudaq::quake::RelaxSizeOp>())
    return isEmptyVeqType(relax.getInputVec());
  return false;
}

// quake.dealloc %empty : !quake.veq<0>
// ─────────────────────────────────────
// removed
//
// There are no qubits to free, so the op is dead.
struct EraseEmptyVeqDeallocPattern
    : public OpRewritePattern<cudaq::quake::DeallocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::DeallocOp dealloc,
                                PatternRewriter &rewriter) const override {
    if (!isEmptyVeqType(dealloc.getReference()))
      return failure();
    rewriter.eraseOp(dealloc);
    return success();
  }
};

// quake.x [%c, %empty] %t : (!quake.ref, !quake.veq<0>, !quake.ref) -> ()
// ─────────────────────────────────────────────────────────────────────
// quake.x [%c] %t : (!quake.ref, !quake.ref) -> ()
//
// Applies to any quantum operator (`quake.h`, `quake.x`, `quake.exp_pauli`,
// etc.) that carries a `controls` operand segment together with a matching
// `negated_qubit_controls` array.
template <typename ConcreteOp>
struct EraseEmptyVeqControlPattern : public OpRewritePattern<ConcreteOp> {
  using OpRewritePattern<ConcreteOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(ConcreteOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> emptyIndices;
    for (auto [index, control] : llvm::enumerate(op.getControls()))
      if (isEmptyVeqType(control))
        emptyIndices.push_back(index);
    if (emptyIndices.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      if (auto negated = op.getNegatedQubitControlsAttr()) {
        SmallVector<bool> polarity(negated.asArrayRef());
        for (unsigned index : llvm::reverse(emptyIndices))
          polarity.erase(polarity.begin() + index);
        op->setAttr("negated_qubit_controls",
                    rewriter.getDenseBoolArrayAttr(polarity));
      }
      for (unsigned index : llvm::reverse(emptyIndices))
        op.getControlsMutable().erase(index);
    });
    return success();
  }
};

// quake.apply @callee [%c, %empty] (%args) : (...) -> ()
// ────────────────────────────────────────────────────────
// quake.apply @callee [%c] (%args) : (...) -> ()
//
// `quake.apply` has no `negated_qubit_controls` attribute to keep in sync, so
// it gets its own (simpler) pattern rather than instantiating the templated
// one above.
struct EraseApplyEmptyVeqControlPattern
    : public OpRewritePattern<cudaq::quake::ApplyOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ApplyOp op,
                                PatternRewriter &rewriter) const override {
    SmallVector<unsigned> emptyIndices;
    for (auto [index, control] : llvm::enumerate(op.getControls()))
      if (isEmptyVeqType(control))
        emptyIndices.push_back(index);
    if (emptyIndices.empty())
      return failure();

    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned index : llvm::reverse(emptyIndices))
        op.getControlsMutable().erase(index);
    });
    return success();
  }
};

// quake.apply_noise @chan (%p) %r, %empty : (..., !quake.ref,
//     !quake.veq<0>) -> ()
// ─────────────────────────────────────────────────────────────────────
// quake.apply_noise @chan (%p) %r : (..., !quake.ref) -> ()
//
// A noise channel applied to an empty qubit register touches nothing, so the
// empty operand is dropped just like an empty control operand. `apply_noise`
// requires at least one qubit operand, so if every qubit is empty the whole
// op is dead and is erased outright instead of being left with zero operands.
struct EraseApplyNoiseEmptyVeqQubitPattern
    : public OpRewritePattern<cudaq::quake::ApplyNoiseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ApplyNoiseOp op,
                                PatternRewriter &rewriter) const override {
    auto qubits = op.getQubits();
    SmallVector<unsigned> emptyIndices;
    for (auto [index, qubit] : llvm::enumerate(qubits))
      if (isEmptyVeqType(qubit))
        emptyIndices.push_back(index);
    if (emptyIndices.empty())
      return failure();

    if (emptyIndices.size() == qubits.size()) {
      rewriter.eraseOp(op);
      return success();
    }

    rewriter.modifyOpInPlace(op, [&] {
      for (unsigned index : llvm::reverse(emptyIndices))
        op.getQubitsMutable().erase(index);
    });
    return success();
  }
};

// quake.log_output %r, %empty, %w : (..., !quake.veq<0>, ...) -> (...)
// ───────────────────────────────────────────────────────────────────
// quake.log_output %r, %w : (..., ...) -> (...)
//
// An empty veq has no observable qubit -- there is nothing there to log --
// so it contributes nothing to what's being observed. A veq is never a
// linear type, so dropping it never touches `outs`. If nothing observable
// remains, the whole op is dead and is erased.
struct DropEmptyVeqLogOutputArgsPattern
    : public OpRewritePattern<cudaq::quake::LogOutputOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::LogOutputOp log,
                                PatternRewriter &rewriter) const override {
    auto args = log.getArgs();
    SmallVector<Value> keep;
    for (Value arg : args)
      if (!isEmptyVeqType(arg))
        keep.push_back(arg);
    if (keep.size() == args.size())
      return failure();

    if (keep.empty()) {
      rewriter.eraseOp(log);
      return success();
    }

    rewriter.replaceOpWithNewOp<cudaq::quake::LogOutputOp>(
        log, keep, log.getCompilerGenerated());
    return success();
  }
};

// %3 = quake.extract_ref %v[%i] : (!quake.veq<0>, i64) -> !quake.ref
// ────────────────────────────────────────────────────────────────────
// %3 = cc.poison !quake.ref
//
// There is no valid index into an empty veq -- extracting from one is
// already an out-of-bounds access -- so replace it with a poison ref rather
// than let the invalid access linger in the IR.
struct ReplaceEmptyVeqExtractRefPattern
    : public OpRewritePattern<cudaq::quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    if (!isEmptyVeqType(extract.getVeq()))
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::cc::PoisonOp>(extract,
                                                     extract.getType());
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

// %0 = quake.concat %r0, %empty, %r1 : (!quake.ref, !quake.veq<0>,
//          !quake.ref) -> !quake.veq<2>
// ──────────────────────────────────────────────────────────────────────
// %0 = quake.concat %r0, %r1 : (!quake.ref, !quake.ref) -> !quake.veq<2>
//
// An empty veq contributes zero qubits to the concatenation, so dropping it
// changes nothing about the result.
struct DropEmptyVeqConcatOperandsPattern
    : public OpRewritePattern<cudaq::quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    SmallVector<Value> keep;
    for (Value target : concat.getTargets())
      if (!isEmptyVeqType(target))
        keep.push_back(target);
    if (keep.empty() || keep.size() == concat.getTargets().size())
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::quake::ConcatOp>(concat,
                                                        concat.getType(), keep);
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
    if (qubitsToConcat.size() != 1)
      return failure();

    // We only want to handle veq -> veq here.
    auto argTy = qubitsToConcat.front().getType();
    auto *ctx = rewriter.getContext();
    auto refTy = cudaq::quake::RefType::get(ctx);
    if (argTy == refTy)
      return failure();

    // Do not handle anything where we don't know the sizes.
    auto unsizedVeqTy = cudaq::quake::VeqType::getUnsized(ctx);
    auto retTy = concat.getResult().getType();
    if (retTy == unsizedVeqTy || argTy != retTy)
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

// %i = quake.concat %a, %b     : (!quake.ref, !quake.ref) -> !quake.veq<2>
// %o = quake.concat %x, %i, %y : (!quake.ref, !quake.veq<2>,
//                                 !quake.ref) -> !quake.veq<4>
// ─────────────────────────────────────────────────────────────────────────
// %o = quake.concat %x, %a, %b, %y : (!quake.ref, !quake.ref, !quake.ref,
//                                     !quake.ref) -> !quake.veq<4>
//
// Flattens one level of concat-of-concat. This enables downstream DCE of the
// inner concat once it has no remaining uses.
struct ConcatFlattenPattern : public OpRewritePattern<cudaq::quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    bool anyInner = false;
    SmallVector<Value> flatArgs;
    for (Value arg : concat.getTargets()) {
      auto inner = arg.getDefiningOp<cudaq::quake::ConcatOp>();
      if (!inner) {
        flatArgs.push_back(arg);
        continue;
      }
      auto innerTy = dyn_cast<cudaq::quake::VeqType>(inner.getType());
      if (!innerTy) {
        flatArgs.push_back(arg);
        continue;
      }
      flatArgs.append(inner.getTargets().begin(), inner.getTargets().end());
      anyInner = true;
    }
    if (!anyInner)
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::quake::ConcatOp>(
        concat, concat.getType(), flatArgs);
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

// %3 = quake.get_member %s[0] : (!quake.struq<!quake.veq<0>, ...>) ->
//                                !quake.veq<0>
// ──────────────────────────────────────────────────────────────────
// %3 = cc.poison !quake.veq<0>
//
// A struq member that is statically empty carries no qubits, so extracting
// it is just as meaningless as allocating it. Canonicalize to the same
// poison placeholder used for a zero-size alloca.
struct ReplaceEmptyVeqGetMemberPattern
    : public OpRewritePattern<cudaq::quake::GetMemberOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::GetMemberOp getMem,
                                PatternRewriter &rewriter) const override {
    if (!isEmptyVeqType(getMem.getResult()))
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::cc::PoisonOp>(getMem, getMem.getType());
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

// %0 = quake.init_state %targets, %state : (!quake.veq<0>, T) -> !quake.veq<?>
// ────────────────────────────────────────────────────────────────────────────
// %0 = quake.relax_size %targets : (!quake.veq<0>) -> !quake.veq<?>
//
// An empty state vector has nothing to initialize -- there are no qubits to
// carry the amplitude -- so the op is a no-op. Erase it and forward its
// `targets` operand directly to its uses.
struct ForwardEmptyVeqInitStatePattern
    : public OpRewritePattern<cudaq::quake::InitializeStateOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    if (!isEmptyVeqType(initState.getTargets()))
      return failure();
    Value targets = initState.getTargets();
    if (targets.getType() != initState.getType())
      targets = cudaq::quake::RelaxSizeOp::create(rewriter, initState.getLoc(),
                                                  initState.getType(), targets)
                    .getResult();
    rewriter.replaceOp(initState, targets);
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

// %0 = quake.subveq %v, %lo, %hi : (!quake.veq<0>, i64, i64) -> !quake.veq<?>
// ────────────────────────────────────────────────────────────────────────────
// %0 = cc.poison !quake.veq<?>
//
// There is no valid subrange of an empty veq -- any requested range is
// inherently out of bounds -- so replace it with a poison value instead of
// letting the invalid access linger in the IR.
struct ReplaceEmptyVeqSubVeqPattern
    : public OpRewritePattern<cudaq::quake::SubVeqOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::SubVeqOp subveq,
                                PatternRewriter &rewriter) const override {
    if (!isEmptyVeqType(subveq.getVeq()))
      return failure();
    rewriter.replaceOpWithNewOp<cudaq::cc::PoisonOp>(subveq, subveq.getType());
    return success();
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

// A wrap_new mints a fresh ref that stands in for a wire's identity. If that
// wire is itself the direct result of unwrapping some ref (with nothing in
// between to modify it), the fresh ref is redundant: the unwrapped ref
// already denotes that same identity. Forward it directly and drop the
// round trip through wrap_new (and the unwrap too, once it has no other
// uses).
struct ForwardUnwrappedRefPattern
    : public OpRewritePattern<cudaq::quake::WrapNewOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::WrapNewOp wrapNew,
                                PatternRewriter &rewriter) const override {
    auto unwrap =
        wrapNew.getWireValue().getDefiningOp<cudaq::quake::UnwrapOp>();
    if (!unwrap)
      return failure();
    rewriter.replaceOp(wrapNew, unwrap.getRefValue());
    if (unwrap->use_empty())
      rewriter.eraseOp(unwrap);
    return success();
  }
};

/// Return true when the two phase operations describe the same ordered
/// control predicate. An omitted polarity attribute is equivalent to an
/// all-positive control list.
static bool haveMatchingPhasePolarities(cudaq::quake::PhaseOp first,
                                        cudaq::quake::PhaseOp second) {
  auto firstControls = first.getControls();
  auto secondControls = second.getControls();
  if (firstControls.size() != secondControls.size())
    return false;

  auto firstNegations = first.getNegatedQubitControls();
  auto secondNegations = second.getNegatedQubitControls();
  for (std::size_t i = 0, e = firstControls.size(); i != e; ++i) {
    bool firstIsNegated = firstNegations && (*firstNegations)[i];
    bool secondIsNegated = secondNegations && (*secondNegations)[i];
    if (firstIsNegated != secondIsNegated)
      return false;
  }
  return true;
}

/// Check that all operands on the second operation are either the corresponding
/// wire result from the first, or the identical non-wire operand. Wire results
/// are ordered by controls followed by targets.
static bool isDirectlyThreadedPhase(cudaq::quake::PhaseOp first,
                                    cudaq::quake::PhaseOp second) {
  auto firstControls = first.getControls();
  auto secondControls = second.getControls();
  auto firstTargets = first.getTargets();
  auto secondTargets = second.getTargets();
  if (firstControls.size() != secondControls.size() ||
      firstTargets.size() != secondTargets.size())
    return false;

  unsigned resultIndex = 0;
  auto matchesOperand = [&](Value firstOperand, Value secondOperand) {
    if (isa<cudaq::quake::WireType>(firstOperand.getType())) {
      if (!isa<cudaq::quake::WireType>(secondOperand.getType()) ||
          resultIndex >= first.getNumResults() ||
          secondOperand != first.getResult(resultIndex))
        return false;
      ++resultIndex;
      return true;
    }
    return !isa<cudaq::quake::WireType>(secondOperand.getType()) &&
           firstOperand == secondOperand;
  };

  for (auto [firstControl, secondControl] :
       llvm::zip(firstControls, secondControls))
    if (!matchesOperand(firstControl, secondControl))
      return false;
  for (auto [firstTarget, secondTarget] :
       llvm::zip(firstTargets, secondTargets))
    if (!matchesOperand(firstTarget, secondTarget))
      return false;
  return resultIndex == first.getNumResults() &&
         llvm::all_of(first.getWires(),
                      [](Value wire) { return wire.hasOneUse(); });
}

struct EraseZeroPhasePattern : public OpRewritePattern<cudaq::quake::PhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::PhaseOp phase,
                                PatternRewriter &rewriter) const override {
    auto constant = phase.getParameter().getDefiningOp<arith::ConstantOp>();
    if (!constant)
      return failure();
    auto angle = dyn_cast<FloatAttr>(constant.getValue());
    if (!angle)
      return failure();

    llvm::APFloat remainder = angle.getValue();
    if (!remainder.isFinite())
      return failure();

    llvm::APFloat twoPi(
        remainder.getSemantics(),
        "6.283185307179586476925286766559005768394338798750211641949889");

    // Reducing in the source semantics alone can misclassify enormous values:
    // a rounded 2*pi approximation multiplied by a large power of two has an
    // exact zero remainder in that format even when its phase is far from one.
    // Require a candidate to be at least as close to identity as the source
    // format's canonical one-cycle representation.
    double phi = remainder.convertToDouble();
    double canonicalTwoPi = twoPi.convertToDouble();
    double tolerance =
        std::hypot(std::cos(canonicalTwoPi) - 1.0, std::sin(canonicalTwoPi));
    double phaseDistance = std::hypot(std::cos(phi) - 1.0, std::sin(phi));
    if (!std::isfinite(phi) || phaseDistance > tolerance)
      return failure();

    remainder.remainder(twoPi);
    if (!remainder.isZero())
      return failure();

    SmallVector<Value> wireInputs;
    for (Value control : phase.getControls())
      if (isa<cudaq::quake::WireType>(control.getType()))
        wireInputs.push_back(control);
    for (Value target : phase.getTargets())
      if (isa<cudaq::quake::WireType>(target.getType()))
        wireInputs.push_back(target);
    if (wireInputs.size() != phase.getNumResults())
      return failure();

    rewriter.replaceOp(phase, wireInputs);
    return success();
  }
};

struct MergeAdjacentPhasePattern
    : public OpRewritePattern<cudaq::quake::PhaseOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(cudaq::quake::PhaseOp phase,
                                PatternRewriter &rewriter) const override {
    auto *previous = phase->getPrevNode();
    if (!previous)
      return failure();
    auto input = dyn_cast<cudaq::quake::PhaseOp>(previous);
    if (!input || !haveMatchingPhasePolarities(input, phase) ||
        !isDirectlyThreadedPhase(input, phase))
      return failure();

    auto firstAngle = input.getParameter();
    auto secondAngle = phase.getParameter();
    if (firstAngle.getType() != secondAngle.getType())
      return failure();

    auto loc = phase.getLoc();
    auto adjoint = phase.getIsAdjAttr();
    Value mergedAngle;
    if (input.isAdj() == phase.isAdj()) {
      mergedAngle =
          arith::AddFOp::create(rewriter, loc, firstAngle, secondAngle);
    } else if (input.isAdj()) {
      mergedAngle =
          arith::SubFOp::create(rewriter, loc, secondAngle, firstAngle);
    } else {
      adjoint = input.getIsAdjAttr();
      mergedAngle =
          arith::SubFOp::create(rewriter, loc, firstAngle, secondAngle);
    }

    rewriter.replaceOpWithNewOp<cudaq::quake::PhaseOp>(
        phase, phase.getResultTypes(), adjoint, ValueRange{mergedAngle},
        input.getControls(), input.getTargets(),
        input.getNegatedQubitControlsAttr());
    rewriter.eraseOp(input);
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
    if (rotate.getTarget().getType() != wireTy ||
        !rotate.getControls().empty() || rotate.getNegatedQubitControls())
      return failure();
    auto input = rotate.getTarget().template getDefiningOp<OP>();
    if (!input || !input.getControls().empty() ||
        input.getNegatedQubitControls())
      return failure();

    // At this point, we have
    //   %input  = quake.rotate %angle1, %wire
    //   %rotate = quake.rotate %angle2, %input
    // Replace those ops with
    //   %new    = quake.rotate (%angle1 + %angle2), %wire
    auto loc = rotate.getLoc();
    auto angle1 = input.getParameter();
    auto angle2 = rotate.getParameter();
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
                                    ValueRange{input.getTarget()},
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
