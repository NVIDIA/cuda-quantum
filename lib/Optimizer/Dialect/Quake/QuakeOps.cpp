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
    if (result.getType().isa<quake::QVecType>() &&
        result.getType().cast<quake::QVecType>().hasSpecifiedSize()) {
      return builder.create<quake::AllocaOp>(
          loc, result.getType().cast<quake::QVecType>().getSize());
    }
    auto constOp = cast<arith::ConstantOp>(args[0].getDefiningOp());
    return builder.create<quake::AllocaOp>(
        loc, static_cast<std::size_t>(
                 constOp.getValue().cast<IntegerAttr>().getInt()));
  }();
  return builder.create<quake::RelaxSizeOp>(
      loc, quake::QVecType::getUnsized(builder.getContext()), newAlloca);
}

void quake::AllocaOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FuseConstantToAllocaPattern>(context);
}

LogicalResult quake::AllocaOp::verify() {
  auto resultType = dyn_cast<QVecType>(getResult().getType());
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
      return emitOpError("expected operand size to match QVecType size.");
  } else if (resultType && !resultType.hasSpecifiedSize()) {
    return emitOpError("must return a qvec with known size.");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ExtractRef
//===----------------------------------------------------------------------===//

OpFoldResult quake::ExtractRefOp::fold(FoldAdaptor adaptor) {
  auto qvec = getQvec();
  auto op = getOperation();
  for (auto user : qvec.getUsers()) {
    if (user == op || op->getBlock() != user->getBlock() ||
        op->isBeforeInBlock(user))
      continue;
    if (auto extractRefOp = dyn_cast<quake::ExtractRefOp>(user)) {
      // Compare the constant extract index values
      // Get the first index and its defining op
      auto first = extractRefOp.getIndex();
      auto defFirst = first.getDefiningOp();

      // Get the second index and its defining op
      auto second = getIndex();
      auto defSecond = second.getDefiningOp();

      // We want to see if firstIdx == secondIdx
      std::optional<std::size_t> firstIdx = std::nullopt;
      if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(defFirst))
        if (auto isaIntValue = dyn_cast<IntegerAttr>(constOp.getValue()))
          firstIdx = isaIntValue.getValue().getLimitedValue();

      std::optional<std::size_t> secondIdx = std::nullopt;
      if (auto constOp = dyn_cast_or_null<arith::ConstantOp>(defSecond))
        if (auto isaIntValue = dyn_cast<IntegerAttr>(constOp.getValue()))
          secondIdx = isaIntValue.getValue().getLimitedValue();

      if (firstIdx.has_value() && secondIdx.has_value() &&
          firstIdx.value() == secondIdx.value())
        return extractRefOp.getResult();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// RelaxSizeOp
//===----------------------------------------------------------------------===//

LogicalResult quake::RelaxSizeOp::verify() {
  if (cast<quake::QVecType>(getType()).hasSpecifiedSize())
    emitOpError("return qvec type must not specify a size");
  return success();
}

// Forward the argument to a relax_size to the users for all users that are
// quake operations. All quake ops that take a sized qvec argument are
// polymorphic on all qvec types. If the op is not a quake op, then maintain
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
  auto vecTy = result.getType().cast<quake::QVecType>();
  auto *ctx = builder.getContext();
  auto getVal = [&](Value v) {
    auto vCon = cast<arith::ConstantOp>(v.getDefiningOp());
    return static_cast<std::size_t>(
        vCon.getValue().cast<IntegerAttr>().getInt());
  };
  std::size_t size = getVal(hi) - getVal(lo) + 1u;
  auto szVecTy = quake::QVecType::get(ctx, size);
  auto subvec = builder.create<quake::SubVecOp>(loc, szVecTy, inVec, lo, hi);
  return builder.create<quake::RelaxSizeOp>(loc, vecTy, subvec);
}

void quake::SubVecOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *context) {
  patterns.add<FuseConstantToSubvecPattern>(context);
}

//===----------------------------------------------------------------------===//
// QVecSizeOp
//===----------------------------------------------------------------------===//

void quake::QVecSizeOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                                    MLIRContext *context) {
  patterns.add<ForwardConstantQVecSizePattern>(context);
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
      (targetsType.size() == 1 && targetsType[0].isa<quake::QVecType>());
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
    if (isa<quake::RefType, quake::QVecType>(v.getType()))
      effects.emplace_back(MemoryEffects::Read::get(), v,
                           SideEffects::DefaultResource::get());
  for (auto v : targets)
    if (isa<quake::RefType, quake::QVecType>(v.getType())) {
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
