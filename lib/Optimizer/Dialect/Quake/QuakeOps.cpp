/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Common/Ops.h"
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

namespace quake {
template <typename ConcreteType>
class QuantumTrait : public OpTrait::TraitBase<ConcreteType, QuantumTrait> {};
} // namespace quake

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

void quake::AllocaOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FuseConstantToAllocaPattern>(context);
}

LogicalResult quake::AllocaOp::verify() {
  auto resultType = dyn_cast<VeqType>(getResult().getType());
  if (auto size = getSize()) {
    std::int64_t argSize = 0;
    if (auto cnt = dyn_cast_or_null<arith::ConstantOp>(size.getDefiningOp())) {
      argSize = cnt.getValue().cast<IntegerAttr>().getInt();
      // TODO: This is a questionable check. We could have a very large unsigned
      // value that appears to be negative because of two's complement. On the
      // other hand, allocating 2^64 - 1 qubits isn't going to go well.
      if (argSize < 0)
        return emitOpError("expected a non-negative integer size.");
    }
    if (!resultType)
      return emitOpError(
          "must return a vector of qubits since a size was provided.");
    if (resultType.hasSpecifiedSize() &&
        (static_cast<std::size_t>(argSize) != resultType.getSize()))
      return emitOpError("expected operand size to match VeqType size.");
  } else if (resultType && !resultType.hasSpecifiedSize()) {
    return emitOpError("must return a veq with known size.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractRef
//===----------------------------------------------------------------------===//

static ParseResult
parseRawIndex(OpAsmParser &parser,
              std::optional<OpAsmParser::UnresolvedOperand> &index,
              IntegerAttr &rawIndex) {
  std::size_t constantIndex;
  OptionalParseResult parsedInteger =
      parser.parseOptionalInteger(constantIndex);
  if (parsedInteger.has_value()) {
    if (failed(parsedInteger.value()))
      return failure();
    index = std::nullopt;
  } else {
    constantIndex = quake::ExtractRefOp::kDynamicIndex;
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

#if 0
OpFoldResult quake::ExtractRefOp::fold(FoldAdaptor adaptor) {
  auto veq = getVeq();
  auto op = getOperation();
  for (auto user : veq.getUsers()) {
    if (user == op || op->getBlock() != user->getBlock() ||
        op->isBeforeInBlock(user))
      continue;
    if (auto extractRefOp = dyn_cast<quake::ExtractRefOp>(user)) {
      // Compare any constant extract_ref index values.
      auto getOffset =
          [&](quake::ExtractRefOp extract) -> std::optional<std::size_t> {
        if (static_cast<std::size_t>(extract.getRawIndex()) == kDynamicIndex) {
          if (auto val = extract.getIndex())
            if (auto defv = cast<arith::ConstantOp>(val.getDefiningOp()))
              if (auto intv = dyn_cast_or_null<IntegerAttr>(defv.getValue()))
                return intv.getValue().getLimitedValue();
          return {};
        }
        return extract.getRawIndex();
      };
      auto firstIdx = getOffset(extractRefOp);
      auto secondIdx = getOffset(*this);
      // Merge the two extract_ref ops if the indices are the same.
      if (firstIdx && secondIdx && firstIdx == secondIdx)
        return extractRefOp.getResult();
    }
  }
  return {};
}
#endif

void quake::ExtractRefOp::getCanonicalizationPatterns(
    RewritePatternSet &patterns, MLIRContext *context) {
  patterns.add<FuseConstantToExtractRefPattern>(context);
}

LogicalResult quake::ExtractRefOp::verify() {
  if (getIndex()) {
    if (getRawIndex() != kDynamicIndex)
      return emitOpError(
          "must not have both a constant index and an index argument.");
  } else {
    if (getRawIndex() == kDynamicIndex)
      return emitOpError("invalid constant index value");
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
        if (auto *dialect = user->getDialect())
          return dialect->getNamespace() == "quake";
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
// SubVecOp
//===----------------------------------------------------------------------===//

Value quake::createSizedSubVecOp(PatternRewriter &builder, Location loc,
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
  auto subvec = builder.create<quake::SubVecOp>(loc, szVecTy, inVec, lo, hi);
  return builder.create<quake::RelaxSizeOp>(loc, vecTy, subvec);
}

void quake::SubVecOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FuseConstantToSubvecPattern>(context);
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
// before it is wrapped, then the wrap operation is a nop and can be eliminated.
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

/// Never inline a `quake.apply` of a variant form of a kernel. The apply
/// operation must be rewritten to a call before it is inlined when the apply is
/// a variant form.
bool cudaq::EnableInlinerInterface::isLegalToInline(Operation *call,
                                                    Operation *callable,
                                                    bool) const {
  if (auto applyOp = dyn_cast<quake::ApplyOp>(call))
    if (applyOp.applyToVariant())
      return false;
  return !(callable->hasAttr(cudaq::entryPointAttrName));
}

void quake::getOperatorEffectsImpl(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects,
    ValueRange controls, ValueRange targets) {
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
