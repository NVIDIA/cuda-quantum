/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

using namespace mlir;

namespace {
#include "cudaq/Optimizer/Dialect/Quake/Canonical.inc"
} // namespace

static bool isQuakeOperation(Operation *op) {
  if (auto *dialect = op->getDialect())
    return dialect->getNamespace().equals("quake");
  return false;
}

/// When a quake operation is in value form, the number of wire arguments (wire
/// arity) must be the same as the number of wires returned as results (wire
/// coarity). This function verifies that this property is true.
LogicalResult quake::verifyWireArityAndCoarity(Operation *op) {
  std::size_t arity = 0;
  std::size_t coarity = 0;
  auto getCounts = [&](auto op) {
    for (auto arg : op.getTargets())
      if (isa<quake::WireType>(arg.getType()))
        ++arity;
    coarity = op.getWires().size();
  };
  if (auto gate = dyn_cast<OperatorInterface>(op)) {
    for (auto arg : gate.getControls())
      if (isa<quake::WireType>(arg.getType()))
        ++arity;
    getCounts(gate);
  } else if (auto meas = dyn_cast<MeasurementInterface>(op)) {
    getCounts(meas);
  }
  if (arity == coarity)
    return success();
  return op->emitOpError("arity does not equal coarity of wires");
}

//===----------------------------------------------------------------------===//
// AllocaOp
//===----------------------------------------------------------------------===//

Value quake::createConstantAlloca(PatternRewriter &builder, Location loc,
                                  OpResult result, ValueRange args) {
  auto newAlloca = [&]() {
    if (result.getType().isa<quake::VeqType>() &&
        result.getType().cast<quake::VeqType>().hasSpecifiedSize()) {
      return builder.create<quake::AllocaOp>(
          loc, result.getType().cast<quake::VeqType>().getSize());
    }
    auto constOp = cast<arith::ConstantOp>(args[0].getDefiningOp());
    return builder.create<quake::AllocaOp>(
        loc, static_cast<std::size_t>(
                 constOp.getValue().cast<IntegerAttr>().getInt()));
  }();
  return builder.create<quake::RelaxSizeOp>(
      loc, quake::VeqType::getUnsized(builder.getContext()), newAlloca);
}

LogicalResult quake::AllocaOp::verify() {
  // Result must be RefType or VeqType by construction.
  if (auto resTy = dyn_cast<VeqType>(getResult().getType())) {
    if (resTy.hasSpecifiedSize()) {
      if (getSize())
        return emitOpError("unexpected size operand");
    } else {
      if (auto size = getSize()) {
        if (auto cnt =
                dyn_cast_or_null<arith::ConstantOp>(size.getDefiningOp())) {
          std::int64_t argSize = cnt.getValue().cast<IntegerAttr>().getInt();
          // TODO: This is a questionable check. We could have a very large
          // unsigned value that appears to be negative because of two's
          // complement. On the other hand, allocating 2^64 - 1 qubits isn't
          // going to go well.
          if (argSize < 0)
            return emitOpError("expected a non-negative integer size.");
        }
      } else {
        return emitOpError("size operand required");
      }
    }
  }
  return success();
}

void quake::AllocaOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  // Use a canonicalization pattern as folding the constant into the veq type
  // changes the type. Uses may still expect a veq with unspecified size.
  // Folding is strictly reductive and doesn't allow the creation of ops.
  patterns.add<FuseConstantToAllocaPattern>(context);
}

//===----------------------------------------------------------------------===//
// Apply
//===----------------------------------------------------------------------===//

void quake::ApplyOp::print(OpAsmPrinter &p) {
  if (getIsAdj())
    p << "<adj>";
  p << ' ';
  bool isDirect = getCallee().has_value();
  if (isDirect)
    p.printAttributeWithoutType(getCalleeAttr());
  else
    p << getIndirectCallee();
  p << ' ';
  if (!getControls().empty())
    p << '[' << getControls() << "] ";
  p << getArgs() << " : ";
  SmallVector<Type> operandTys{(*this)->getOperandTypes().begin(),
                               (*this)->getOperandTypes().end()};
  p.printFunctionalType(ArrayRef<Type>{operandTys}.drop_front(isDirect ? 0 : 1),
                        (*this)->getResultTypes());
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      {"operand_segment_sizes", "is_adj", getCalleeAttrNameStr()});
}

ParseResult quake::ApplyOp::parse(OpAsmParser &parser, OperationState &result) {
  if (succeeded(parser.parseOptionalLess())) {
    if (parser.parseKeyword("adj") || parser.parseGreater())
      return failure();
    result.addAttribute("is_adj", parser.getBuilder().getUnitAttr());
  }
  SmallVector<OpAsmParser::UnresolvedOperand> calleeOperand;
  if (parser.parseOperandList(calleeOperand))
    return failure();
  bool isDirect = calleeOperand.empty();
  if (calleeOperand.size() > 1)
    return failure();
  if (isDirect) {
    NamedAttrList attrs;
    SymbolRefAttr funcAttr;
    if (parser.parseCustomAttributeWithFallback(
            funcAttr, parser.getBuilder().getType<NoneType>(),
            getCalleeAttrNameStr(), attrs))
      return failure();
    result.addAttribute(getCalleeAttrNameStr(), funcAttr);
  }

  SmallVector<OpAsmParser::UnresolvedOperand> controlOperands;
  if (succeeded(parser.parseOptionalLSquare()))
    if (parser.parseOperandList(controlOperands) || parser.parseRSquare())
      return failure();

  SmallVector<OpAsmParser::UnresolvedOperand> miscOperands;
  if (parser.parseOperandList(miscOperands) || parser.parseColon())
    return failure();

  FunctionType applyTy;
  if (parser.parseType(applyTy) ||
      parser.parseOptionalAttrDict(result.attributes))
    return failure();
  result.addAttribute("operand_segment_sizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(calleeOperand.size()),
                           static_cast<int32_t>(controlOperands.size()),
                           static_cast<int32_t>(miscOperands.size())}));
  result.addTypes(applyTy.getResults());
  if (isDirect) {
    if (parser.resolveOperands(
            llvm::concat<const OpAsmParser::UnresolvedOperand>(
                calleeOperand, controlOperands, miscOperands),
            applyTy.getInputs(), parser.getNameLoc(), result.operands))
      return failure();
  } else {
    auto loc = parser.getNameLoc();
    auto fnTy = parser.getBuilder().getFunctionType(
        applyTy.getInputs().drop_front(controlOperands.size()),
        applyTy.getResults());
    auto callableTy = cudaq::cc::CallableType::get(parser.getContext(), fnTy);
    if (parser.resolveOperands(calleeOperand, callableTy, loc,
                               result.operands) ||
        parser.resolveOperands(
            llvm::concat<const OpAsmParser::UnresolvedOperand>(controlOperands,
                                                               miscOperands),
            applyTy.getInputs(), loc, result.operands))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Concat
//===----------------------------------------------------------------------===//

namespace {
// %7 = quake.concat %4 : (!quake.veq<2>) -> !quake.veq<2>
// ───────────────────────────────────────────
// removed
struct ConcatNoOpPattern : public OpRewritePattern<quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    // Remove concat veq<N> -> veq<N>
    // or
    // concat ref -> ref
    auto qubitsToConcat = concat.getQbits();
    if (qubitsToConcat.size() > 1)
      return failure();

    // We only want to handle veq -> veq here.
    if (isa<quake::RefType>(qubitsToConcat.front().getType())) {
      return failure();
    }

    // Do not handle anything where we don't know the sizes.
    auto retTy = concat.getResult().getType();
    if (auto veqTy = dyn_cast<quake::VeqType>(retTy))
      if (!veqTy.hasSpecifiedSize())
        // This could be a folded quake.relax_size op.
        return failure();

    rewriter.replaceOp(concat, qubitsToConcat);
    return success();
  }
};

struct ConcatSizePattern : public OpRewritePattern<quake::ConcatOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ConcatOp concat,
                                PatternRewriter &rewriter) const override {
    if (concat.getType().hasSpecifiedSize())
      return failure();

    // Walk the arguments and sum them, if possible.
    std::size_t sum = 0;
    for (auto opnd : concat.getQbits()) {
      if (auto veqTy = dyn_cast<quake::VeqType>(opnd.getType())) {
        if (!veqTy.hasSpecifiedSize())
          return failure();
        sum += veqTy.getSize();
        continue;
      }
      assert(isa<quake::RefType>(opnd.getType()));
      sum++;
    }

    // Leans into the relax_size canonicalization pattern.
    auto *ctx = rewriter.getContext();
    auto loc = concat.getLoc();
    auto newTy = quake::VeqType::get(ctx, sum);
    Value newOp =
        rewriter.create<quake::ConcatOp>(loc, newTy, concat.getQbits());
    auto noSizeTy = quake::VeqType::getUnsized(ctx);
    rewriter.replaceOpWithNewOp<quake::RelaxSizeOp>(concat, noSizeTy, newOp);
    return success();
  };
};
} // namespace

void quake::ConcatOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<ConcatSizePattern, ConcatNoOpPattern>(context);
}

//===----------------------------------------------------------------------===//
// ExtractRef
//===----------------------------------------------------------------------===//

static ParseResult
parseRawIndex(OpAsmParser &parser,
              std::optional<OpAsmParser::UnresolvedOperand> &index,
              IntegerAttr &rawIndex) {
  std::size_t constantIndex = quake::ExtractRefOp::kDynamicIndex;
  OptionalParseResult parsedInteger =
      parser.parseOptionalInteger(constantIndex);
  if (parsedInteger.has_value()) {
    if (failed(parsedInteger.value()))
      return failure();
    index = std::nullopt;
  } else {
    OpAsmParser::UnresolvedOperand operand;
    if (parser.parseOperand(operand))
      return failure();
    index = operand;
  }
  auto i64Ty = IntegerType::get(parser.getContext(), 64);
  rawIndex = IntegerAttr::get(i64Ty, constantIndex);
  return success();
}

static void printRawIndex(OpAsmPrinter &printer, quake::ExtractRefOp refOp,
                          Value index, IntegerAttr rawIndex) {
  if (rawIndex.getValue() == quake::ExtractRefOp::kDynamicIndex)
    printer.printOperand(index);
  else
    printer << rawIndex.getValue();
}

namespace {
// %4 = quake.concat %2, %3 : (!quake.ref, !quake.ref) -> !quake.veq<2>
// %7 = quake.extract_ref %4[0] : (!quake.veq<2>) -> !quake.ref
// ───────────────────────────────────────────
// replace all use with %2
struct ForwardConcatExtractPattern
    : public OpRewritePattern<quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    auto veq = extract.getVeq();
    auto concatOp = veq.getDefiningOp<quake::ConcatOp>();
    if (concatOp && extract.hasConstantIndex()) {
      // Don't run this canonicalization if any of the operands
      // to concat are of type veq.
      auto concatQubits = concatOp.getQbits();
      for (auto qOp : concatQubits)
        if (isa<quake::VeqType>(qOp.getType()))
          return failure();

      // concat only has ref type operands.
      auto index = extract.getConstantIndex();
      if (index < concatQubits.size()) {
        auto qOpValue = concatQubits[index];
        if (isa<quake::RefType>(qOpValue.getType()))
          rewriter.replaceOp(extract, {qOpValue});
      }
    }
    return success();
  }
};

// %2 = quake.concat %1 : (!quake.ref) -> !quake.veq<1>
// %3 = quake.extract_ref %2[0] : (!quake.veq<1>) -> !quake.ref
// quake.* %3 ...
// ───────────────────────────────────────────
// quake.* %1 ...
struct ForwardConcatExtractSingleton
    : public OpRewritePattern<quake::ExtractRefOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::ExtractRefOp extract,
                                PatternRewriter &rewriter) const override {
    if (auto concat = extract.getVeq().getDefiningOp<quake::ConcatOp>())
      if (concat.getType().getSize() == 1 && extract.hasConstantIndex() &&
          extract.getConstantIndex() == 0) {
        assert(concat.getQbits().size() == 1 && concat.getQbits()[0]);
        extract.getResult().replaceUsesWithIf(
            concat.getQbits()[0], [&](OpOperand &use) {
              if (Operation *user = use.getOwner())
                return isQuakeOperation(user);
              return false;
            });
        return success();
      }
    return failure();
  }
};
} // namespace

void quake::ExtractRefOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseConstantToExtractRefPattern, ForwardConcatExtractSingleton,
               ForwardConcatExtractPattern>(context);
}

LogicalResult quake::ExtractRefOp::verify() {
  if (getIndex()) {
    if (getRawIndex() != kDynamicIndex)
      return emitOpError(
          "must not have both a constant index and an index argument.");
  } else {
    if (getRawIndex() == kDynamicIndex) {
      return emitOpError("invalid constant index value");
    } else {
      auto veqSize = getVeq().getType().getSize();
      if (getVeq().getType().hasSpecifiedSize() && getRawIndex() >= veqSize)
        return emitOpError("invalid index [" + std::to_string(getRawIndex()) +
                           "] because >= size [" + std::to_string(veqSize) +
                           "]");
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// RelaxSizeOp
//===----------------------------------------------------------------------===//

LogicalResult quake::RelaxSizeOp::verify() {
  if (cast<quake::VeqType>(getType()).hasSpecifiedSize())
    emitOpError("return veq type must not specify a size");
  return success();
}

// Forward the argument to a relax_size to the users for all users that are
// quake operations. All quake ops that take a sized veq argument are
// polymorphic on all veq types. If the op is not a quake op, then maintain
// strong typing.
struct ForwardRelaxedSizePattern : public RewritePattern {
  ForwardRelaxedSizePattern(MLIRContext *context)
      : RewritePattern("quake.relax_size", 1, context, {}) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto relax = cast<quake::RelaxSizeOp>(op);
    auto inpVec = relax.getInputVec();
    Value result = relax.getResult();
    result.replaceUsesWithIf(inpVec, [&](OpOperand &use) {
      if (Operation *user = use.getOwner())
        return isQuakeOperation(user) && !isa<quake::ApplyOp>(user);
      return false;
    });
    return success();
  };
};

void quake::RelaxSizeOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<ForwardRelaxedSizePattern>(context);
}

//===----------------------------------------------------------------------===//
// SubVeqOp
//===----------------------------------------------------------------------===//

Value quake::createSizedSubVeqOp(PatternRewriter &builder, Location loc,
                                 OpResult result, Value inVec, Value lo,
                                 Value hi) {
  auto vecTy = result.getType().cast<quake::VeqType>();
  auto *ctx = builder.getContext();
  auto getVal = [&](Value v) {
    auto vCon = cast<arith::ConstantOp>(v.getDefiningOp());
    return static_cast<std::size_t>(
        vCon.getValue().cast<IntegerAttr>().getInt());
  };
  std::size_t size = getVal(hi) - getVal(lo) + 1u;
  auto szVecTy = quake::VeqType::get(ctx, size);
  auto subveq = builder.create<quake::SubVeqOp>(loc, szVecTy, inVec, lo, hi);
  return builder.create<quake::RelaxSizeOp>(loc, vecTy, subveq);
}

void quake::SubVeqOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FuseConstantToSubveqPattern>(context);
}

//===----------------------------------------------------------------------===//
// VeqSizeOp
//===----------------------------------------------------------------------===//

void quake::VeqSizeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *context) {
  patterns.add<ForwardConstantVeqSizePattern>(context);
}

//===----------------------------------------------------------------------===//
// WrapOp
//===----------------------------------------------------------------------===//

namespace {
// If there is no operation that modifies the wire after it gets unwrapped and
// before it is wrapped, then the wrap operation is a nop and can be
// eliminated.
struct KillDeadWrapPattern : public OpRewritePattern<quake::WrapOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::WrapOp wrap,
                                PatternRewriter &rewriter) const override {
    if (auto unwrap = wrap.getWireValue().getDefiningOp<quake::UnwrapOp>())
      rewriter.eraseOp(wrap);
    return success();
  }
};
} // namespace

void quake::WrapOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                MLIRContext *context) {
  patterns.add<KillDeadWrapPattern>(context);
}

//===----------------------------------------------------------------------===//
// Measurements (MxOp, MyOp, MzOp)
//===----------------------------------------------------------------------===//

// Common verification for measurement operations.
static LogicalResult verifyMeasurements(Operation *const op,
                                        TypeRange targetsType,
                                        const Type bitsType) {
  bool mustBeStdvec =
      targetsType.size() > 1 ||
      (targetsType.size() == 1 && targetsType[0].isa<quake::VeqType>());
  if (mustBeStdvec) {
    if (!op->getResult(0).getType().isa<cudaq::cc::StdvecType>())
      return op->emitOpError("must return `!cc.stdvec<i1>`, when measuring a "
                             "qreg, a series of qubits, or both");
  } else {
    if (!op->getResult(0).getType().isa<IntegerType>())
      return op->emitOpError(
          "must return `i1` when measuring exactly one qubit");
  }
  return success();
}

LogicalResult quake::MxOp::verify() {
  return verifyMeasurements(getOperation(), getTargets().getType(),
                            getBits().getType());
}

LogicalResult quake::MyOp::verify() {
  return verifyMeasurements(getOperation(), getTargets().getType(),
                            getBits().getType());
}

LogicalResult quake::MzOp::verify() {
  return verifyMeasurements(getOperation(), getTargets().getType(),
                            getBits().getType());
}

//===----------------------------------------------------------------------===//
// Operator interface
//===----------------------------------------------------------------------===//

// The following methods return to the operator's unitary matrix as a
// column-major array. For parametrizable operations, the matrix can only be
// built if the parameter can be computed at compilation time. These methods
// populate an empty array taken as a input. If the matrix was not successfuly
// computed, the array will be left empty.

/// If the parameter is known at compilation-time, set the result value and
/// returns success. Otherwise, returns failure.
static LogicalResult getParameterAsDouble(Value parameter, double &result) {
  auto paramDefOp = parameter.getDefiningOp();
  if (!paramDefOp)
    return failure();
  if (auto constOp = mlir::dyn_cast<mlir::arith::ConstantOp>(paramDefOp)) {
    if (auto value = dyn_cast<mlir::FloatAttr>(constOp.getValue())) {
      result = value.getValueAsDouble();
      return success();
    }
  }
  return failure();
}

void quake::HOp::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  matrix.assign({inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2});
}

void quake::PhasedRxOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;

  // Get parameters
  double theta;
  double phi;
  if (failed(getParameterAsDouble(getParameter(), theta)) ||
      failed(getParameterAsDouble(getParameter(1), phi)))
    return;

  if (getIsAdj())
    theta *= -1;

  matrix.assign(
      {std::cos(theta / 2.), -1i * std::exp(1i * phi) * std::sin(theta / 2.),
       -1i * std::exp(-1i * phi) * std::sin(theta / 2.), std::cos(theta / 2.)});
}

void quake::R1Op::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;
  if (getIsAdj())
    theta *= -1;
  matrix.assign({1, 0, 0, std::exp(theta * 1i)});
}

void quake::RxOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;
  if (getIsAdj())
    theta *= -1;
  matrix.assign({std::cos(theta / 2.), -1i * std::sin(theta / 2.),
                 -1i * std::sin(theta / 2.), std::cos(theta / 2.)});
}

void quake::RyOp::getOperatorMatrix(Matrix &matrix) {
  // Get parameter
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;

  if (getIsAdj())
    theta *= -1;

  matrix.assign({std::cos(theta / 2.), std::sin(theta / 2.),
                 -std::sin(theta / 2.), std::cos(theta / 2.)});
}

void quake::RzOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;

  // Get parameter
  double theta;
  if (failed(getParameterAsDouble(getParameter(), theta)))
    return;

  if (getIsAdj())
    theta *= -1;

  matrix.assign({std::exp(-1i * theta / 2.), 0, 0, std::exp(1i * theta / 2.)});
}

void quake::SOp::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  using namespace std::complex_literals;
  if (getIsAdj())
    matrix.assign({1, 0, 0, -1i});
  else
    matrix.assign({1, 0, 0, 1i});
}

void quake::SwapOp::getOperatorMatrix(Matrix &matrix) {
  matrix.assign({1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1});
}

void quake::TOp::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  if (getIsAdj())
    matrix.assign({1, 0, 0, {inv_sqrt2, -inv_sqrt2}});
  else
    matrix.assign({1, 0, 0, {inv_sqrt2, inv_sqrt2}});
}

void quake::U2Op::getOperatorMatrix(Matrix &matrix) {
  using namespace llvm::numbers;
  using namespace std::complex_literals;

  // Get parameters
  double phi;
  double lambda;
  if (failed(getParameterAsDouble(getParameter(), phi)) ||
      failed(getParameterAsDouble(getParameter(1), lambda)))
    return;

  if (getIsAdj()) {
    phi *= -1;
    lambda *= -1;
  }

  matrix.assign({inv_sqrt2, inv_sqrt2 * std::exp(phi * 1i),
                 -inv_sqrt2 * std::exp(lambda * 1i),
                 inv_sqrt2 * std::exp(1i * (phi + lambda))});
}

void quake::U3Op::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;

  // Get parameters
  double theta;
  double phi;
  double lambda;
  if (failed(getParameterAsDouble(getParameter(), theta)) ||
      failed(getParameterAsDouble(getParameter(1), phi)) ||
      failed(getParameterAsDouble(getParameter(2), lambda)))
    return;

  if (getIsAdj()) {
    theta *= -1;
    phi *= -1;
    lambda *= -1;
  }

  matrix.assign({std::cos(theta / 2.),
                 std::exp(phi * 1i) * std::sin(theta / 2.),
                 -std::exp(lambda * 1i) * std::sin(theta / 2.),
                 std::exp(1i * (phi + lambda)) * std::cos(theta / 2.)});
}

void quake::XOp::getOperatorMatrix(Matrix &matrix) {
  matrix.assign({0, 1, 1, 0});
}

void quake::YOp::getOperatorMatrix(Matrix &matrix) {
  using namespace std::complex_literals;
  matrix.assign({0, 1i, -1i, 0});
}

void quake::ZOp::getOperatorMatrix(Matrix &matrix) {
  matrix.assign({1, 0, 0, -1});
}

//===----------------------------------------------------------------------===//

/// Never inline a `quake.apply` of a variant form of a kernel. The apply
/// operation must be rewritten to a call before it is inlined when the apply
/// is a variant form.
bool cudaq::EnableInlinerInterface::isLegalToInline(Operation *call,
                                                    Operation *callable,
                                                    bool) const {
  if (auto applyOp = dyn_cast<quake::ApplyOp>(call))
    if (applyOp.applyToVariant())
      return false;
  return !(callable->hasAttr(cudaq::entryPointAttrName));
}

using EffectsVectorImpl =
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>;

/// For an operation with modeless effects, the operation always has effects
/// on the control and target quantum operands, whether those operands are in
/// reference or value form. A operation with modeless effects is not removed
/// when its result(s) is (are) unused.
[[maybe_unused]] inline static void
getModelessEffectsImpl(EffectsVectorImpl &effects, ValueRange controls,
                       ValueRange targets) {
  for (auto v : controls)
    effects.emplace_back(MemoryEffects::Read::get(), v,
                         SideEffects::DefaultResource::get());
  for (auto v : targets) {
    effects.emplace_back(MemoryEffects::Read::get(), v,
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(), v,
                         SideEffects::DefaultResource::get());
  }
}

/// For an operation with moded effects, the operation conditionally has
/// effects on the control and target quantum operands. If those operands are
/// in reference form, then the operation does have effects on those
/// references. Control operands have a read effect, while target operands
/// have both a read and write effect. If the operand is in value form, the
/// operation introduces no effects on that operand.
inline static void getModedEffectsImpl(EffectsVectorImpl &effects,
                                       ValueRange controls,
                                       ValueRange targets) {
  for (auto v : controls)
    if (isa<quake::RefType, quake::VeqType>(v.getType()))
      effects.emplace_back(MemoryEffects::Read::get(), v,
                           SideEffects::DefaultResource::get());
  for (auto v : targets)
    if (isa<quake::RefType, quake::VeqType>(v.getType())) {
      effects.emplace_back(MemoryEffects::Read::get(), v,
                           SideEffects::DefaultResource::get());
      effects.emplace_back(MemoryEffects::Write::get(), v,
                           SideEffects::DefaultResource::get());
    }
}

/// Quake reset has modeless effects.
void quake::getResetEffectsImpl(EffectsVectorImpl &effects,
                                ValueRange targets) {
  getModedEffectsImpl(effects, {}, targets);
}

/// Quake measurement operations have moded effects.
void quake::getMeasurementEffectsImpl(EffectsVectorImpl &effects,
                                      ValueRange targets) {
  getModedEffectsImpl(effects, {}, targets);
}

/// Quake quantum operators have moded effects.
void quake::getOperatorEffectsImpl(EffectsVectorImpl &effects,
                                   ValueRange controls, ValueRange targets) {
  getModedEffectsImpl(effects, controls, targets);
}

// This is a workaround for ODS generating these member function declarations
// but not having a way to define them in the ODS.
// clang-format off
#define GATE_OPS(MACRO) MACRO(XOp) MACRO(YOp) MACRO(ZOp) MACRO(HOp) MACRO(SOp) \
  MACRO(TOp) MACRO(SwapOp) MACRO(U2Op) MACRO(U3Op)                             \
  MACRO(R1Op) MACRO(RxOp) MACRO(RyOp) MACRO(RzOp) MACRO(PhasedRxOp)
#define MEASURE_OPS(MACRO) MACRO(MxOp) MACRO(MyOp) MACRO(MzOp)
#define QUANTUM_OPS(MACRO) MACRO(ResetOp) GATE_OPS(MACRO) MEASURE_OPS(MACRO)
// clang-format on
#define INSTANTIATE_CALLBACKS(Op)                                              \
  void quake::Op::getEffects(                                                  \
      SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>      \
          &effects) {                                                          \
    getEffectsImpl(effects);                                                   \
  }

QUANTUM_OPS(INSTANTIATE_CALLBACKS)

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

using namespace cudaq;

#define GET_OP_CLASSES
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.cpp.inc"
