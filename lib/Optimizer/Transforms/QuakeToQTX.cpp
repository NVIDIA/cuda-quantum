/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "QuakeToQTXConverter.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

using namespace mlir;
using namespace cudaq;

namespace {
//===----------------------------------------------------------------------===//
// Quake converters
//===----------------------------------------------------------------------===//

class AllocaConverter final : public ConvertOpToQTXPattern<quake::AllocaOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(quake::AllocaOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    auto wireOrArray = rewriter.convertType(op.getResult().getType());
    if (!wireOrArray)
      return failure();
    auto newOp = rewriter.create<qtx::AllocaOp>(op.getLoc(), wireOrArray);

    // Map the Quake value to a QTX value
    rewriter.mapOrRemap(op.getResult(), newOp.getResult());

    rewriter.eraseOp(op);
    return success();
  }
};

class DeallocConverter final : public ConvertOpToQTXPattern<quake::DeallocOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(quake::DeallocOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    rewriter.create<qtx::DeallocOp>(op.getLoc(), adaptor.getQregOrVec());
    rewriter.eraseOp(op);
    return success();
  }
};

class ResetConverter final : public ConvertOpToQTXPattern<quake::ResetOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(quake::ResetOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    auto newOp =
        rewriter.create<qtx::ResetOp>(op.getLoc(), adaptor.getTargets());
    // `qtx.reset` operation can act on a list o values, in this case we know
    // it is only reseting one wire (or array), so it has only one result.
    rewriter.mapOrRemap(op.getTargets(), newOp.getResults()[0]);

    rewriter.eraseOp(op);
    return success();
  }
};

class ExtractRefConverter final
    : public ConvertOpToQTXPattern<quake::ExtractRefOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(quake::ExtractRefOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    // Fail when we are try to borrow from an array that has only dead wires
    auto arrayType = dyn_cast<qtx::WireArrayType>(adaptor.getQvec().getType());
    if (arrayType.getSize() == arrayType.getDead())
      return failure();

    auto newOp = rewriter.create<qtx::ArrayBorrowOp>(
        op.getLoc(), adaptor.getIndex(), adaptor.getQvec());

    // Map (or remap) Quake values to QTX values
    rewriter.mapOrRemap(op.getQvec(), newOp.getNewArray());
    rewriter.mapOrRemap(op.getResult(), newOp.getWires()[0]);
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename QuakeOp, typename QTXOp>
class MeasurementConverter final : public ConvertOpToQTXPattern<QuakeOp> {
public:
  using ConvertOpToQTXPattern<QuakeOp>::ConvertOpToQTXPattern;
  using OpAdaptor = typename ConvertOpToQTXPattern<QuakeOp>::OpAdaptor;

  LogicalResult matchAndRewrite(QuakeOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    auto newOp = rewriter.create<QTXOp>(op.getLoc(), adaptor.getOperands(),
                                        adaptor.getRegisterNameAttr());

    // The first result in QTX's corresponds to the measured bits, so when
    // remapping the qubit references, we need to shift the index by one.
    for (auto [i, quakeValue] : llvm::enumerate(op.getTargets())) {
      rewriter.mapOrRemap(quakeValue, newOp.getResult(i + 1));
    }
    rewriter.replaceAllUsesWith(op.getBits(), newOp.getBits());
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename QuakeOp, typename QTXOp>
class OperatorConverter final : public ConvertOpToQTXPattern<QuakeOp> {
public:
  using ConvertOpToQTXPattern<QuakeOp>::ConvertOpToQTXPattern;
  using OpAdaptor = typename ConvertOpToQTXPattern<QuakeOp>::OpAdaptor;

  LogicalResult matchAndRewrite(QuakeOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    auto newOp = rewriter.create<QTXOp>(
        op.getLoc(), adaptor.getIsAdj(), adaptor.getParameters(),
        adaptor.getControls(), adaptor.getTargets());

    // Map (or remap) Quake values to QTX values
    for (auto [quakeValue, qtxValue] :
         llvm::zip_equal(op.getTargets(), newOp.getResults()))
      rewriter.mapOrRemap(quakeValue, qtxValue);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// CC converters
//===----------------------------------------------------------------------===//

class CCScopeConverter final : public ConvertOpToQTXPattern<cc::ScopeOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(cc::ScopeOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    SmallVector<Value, 4> operands(adaptor.getOperands());
    rewriter.getRemapped(op, implicitUsedQuantumValues, operands);

    auto newOp = rewriter.create<cc::ScopeOp>(op.getLoc(),
                                              TypeRange(ValueRange(operands)));

    rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
                                newOp.getRegion().begin());

    // Map (or remap) Quake values to QTX values
    for (auto [quakeValue, qtxValue] :
         llvm::zip_equal(implicitUsedQuantumValues, newOp.getResults()))
      rewriter.mapOrRemap(quakeValue, qtxValue);

    rewriter.eraseOp(op);
    return success();
  }
};

class CCIfOpConverter final : public ConvertOpToQTXPattern<cc::IfOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(cc::IfOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    SmallVector<Value, 4> qtxValues;
    rewriter.getRemapped(op, implicitUsedQuantumValues, qtxValues);

    auto newOp = rewriter.create<cc::IfOp>(
        op.getLoc(), TypeRange(ValueRange(qtxValues)), adaptor.getCondition());

    // Take the regions from the original `cc.if`
    rewriter.inlineRegionBefore(op.getThenRegion(), newOp.getThenRegion(),
                                newOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(op.getElseRegion(), newOp.getElseRegion(),
                                newOp.getElseRegion().begin());

    if (newOp.getElseRegion().empty()) {
      // In this case, we need to create a `else` branch that simply return the
      // current wires (or wire array) corresponding to the
      // implicitUsedQuantumValues
      rewriter.createBlock(&newOp.getElseRegion());
      rewriter.create<cc::ContinueOp>(op.getLoc(), qtxValues);
    }

    // Map (or remap) Quake values to QTX values
    for (auto [quakeValue, qtxValue] :
         llvm::zip_equal(implicitUsedQuantumValues, newOp.getResults()))
      rewriter.mapOrRemap(quakeValue, qtxValue);

    rewriter.eraseOp(op);
    return success();
  }
};

class CCLoopConverter final : public ConvertOpToQTXPattern<cc::LoopOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(cc::LoopOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    SmallVector<Value, 4> qtxValues;
    rewriter.getRemapped(op, implicitUsedQuantumValues, qtxValues);

    SmallVector<Value, 4> operands(adaptor.getOperands());
    operands.append(qtxValues);
    auto newOp = rewriter.create<cc::LoopOp>(
        op.getLoc(), TypeRange(ValueRange(operands)), operands, op->getAttrs());

    // Move the regions
    rewriter.inlineRegionBefore(op.getWhileRegion(), newOp.getWhileRegion(),
                                newOp.getWhileRegion().begin());

    rewriter.inlineRegionBefore(op.getBodyRegion(), newOp.getBodyRegion(),
                                newOp.getBodyRegion().begin());

    // Operation inside the region were already converted, and thus are using
    // QTX values from the parent scope, which is wrong.  We add new arguments
    // that capture these values and replace the old values inside the region.
    auto fixArgs = [&](Value value, Block *block) {
      auto arg = rewriter.addArgument(block, value.getType(), value.getLoc());
      rewriter.replaceUseIf(value, arg, [&](auto &operand) {
        return operand.getOwner()->getParentRegion() == block->getParent();
      });
    };
    for (auto qtxValue : qtxValues) {
      fixArgs(qtxValue, newOp.getWhileBlock());
      fixArgs(qtxValue, newOp.getDoEntryBlock());
    }

    if (!op.getStepRegion().empty()) {
      rewriter.inlineRegionBefore(op.getStepRegion(), newOp.getStepRegion(),
                                  newOp.getStepRegion().begin());
      for (auto qtxValue : qtxValues)
        fixArgs(qtxValue, newOp.getStepBlock());
    }

    // Map (or remap) Quake values to QTX values
    for (auto [quakeValue, qtxValue] :
         llvm::zip_equal(implicitUsedQuantumValues, newOp.getResults()))
      rewriter.mapOrRemap(quakeValue, qtxValue);

    rewriter.eraseOp(op);
    return success();
  }
};

class CCConditionConverter final
    : public ConvertOpToQTXPattern<cc::ConditionOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(cc::ConditionOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    // We only need to rewrite `cc.continue` operations that have implicit used
    // quantum values
    if (implicitUsedQuantumValues.empty())
      return success();

    SmallVector<Value, 4> operands(adaptor.getOperands());
    rewriter.getRemapped(op, implicitUsedQuantumValues, operands);
    rewriter.create<cc::ConditionOp>(op.getLoc(), operands.front(),
                                     ValueRange(operands).drop_front());
    rewriter.eraseOp(op);
    return success();
  }
};

class CCContinueConverter final : public ConvertOpToQTXPattern<cc::ContinueOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(cc::ContinueOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    // We only need to rewrite `cc.continue` operations that have implicit used
    // quantum values
    if (implicitUsedQuantumValues.empty())
      return success();

    SmallVector<Value, 4> operands(adaptor.getOperands());
    rewriter.getRemapped(op, implicitUsedQuantumValues, operands);
    rewriter.create<cc::ContinueOp>(op.getLoc(), operands);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mlir::Func converters
//===----------------------------------------------------------------------===//

class FuncReturnConverter final : public ConvertOpToQTXPattern<func::ReturnOp> {
public:
  using ConvertOpToQTXPattern::ConvertOpToQTXPattern;

  LogicalResult matchAndRewrite(func::ReturnOp op,
                                ArrayRef<Value> implicitUsedQuantumValues,
                                OpAdaptor adaptor,
                                ConvertToQTXRewriter &rewriter) const override {
    SmallVector<Value, 4> operands;
    rewriter.getRemapped(op, implicitUsedQuantumValues, operands);
    rewriter.create<qtx::UnrealizedReturnOp>(op.getLoc(), adaptor.getOperands(),
                                             operands);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Quake to QTX pass
//===----------------------------------------------------------------------===//

class QuakeToQTXPass
    : public cudaq::opt::ConvertQuakeToQTXBase<QuakeToQTXPass> {
public:
  void runOnOperation() override final {
    RewritePatternSet patterns(&getContext());
    // clang-format off
    patterns.add<
      // Quake converters
      AllocaConverter,
      DeallocConverter,
      ResetConverter,
      ExtractRefConverter,
      OperatorConverter<quake::HOp, qtx::HOp>,
      OperatorConverter<quake::SOp, qtx::SOp>,
      OperatorConverter<quake::TOp, qtx::TOp>,
      OperatorConverter<quake::XOp, qtx::XOp>,
      OperatorConverter<quake::YOp, qtx::YOp>,
      OperatorConverter<quake::ZOp, qtx::ZOp>,
      OperatorConverter<quake::R1Op, qtx::R1Op>,
      OperatorConverter<quake::RxOp, qtx::RxOp>,
      OperatorConverter<quake::RyOp, qtx::RyOp>,
      OperatorConverter<quake::RzOp, qtx::RzOp>,
      OperatorConverter<quake::PhasedRxOp, qtx::PhasedRxOp>,
      OperatorConverter<quake::SwapOp, qtx::SwapOp>,
      MeasurementConverter<quake::MxOp, qtx::MxOp>,
      MeasurementConverter<quake::MyOp, qtx::MyOp>,
      MeasurementConverter<quake::MzOp, qtx::MzOp>,
      // CC converters
      CCScopeConverter,
      CCIfOpConverter,
      CCLoopConverter,
      CCConditionConverter,
      CCContinueConverter,
      FuncReturnConverter>(&getContext());
    // clang-format on

    if (failed(applyPartialQuakeToQTXConversion(getOperation(),
                                                std::move(patterns)))) {
      getOperation().emitWarning("couldn't convert kernel to QTX");
      return;
    }
    getOperation()->setAttr("convert-to-circuit", UnitAttr::get(&getContext()));
  }
};

std::unique_ptr<Pass> cudaq::opt::createConvertQuakeToQTXPass() {
  return std::make_unique<QuakeToQTXPass>();
}
