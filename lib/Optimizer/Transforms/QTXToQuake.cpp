/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// Alloca
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::AllocaOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  quake::AllocaOp quakeOp;
  if (auto array = dyn_cast<qtx::WireArrayType>(qtxOp.getResult().getType())) {
    auto size = array.getSize();
    quakeOp = builder.create<quake::AllocaOp>(size);
  } else
    quakeOp = builder.create<quake::AllocaOp>();
  qtxOp.replaceAllUsesWith(quakeOp.getResult());
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// DeallocOp
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::DeallocOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  for (auto operand : qtxOp.getOperands()) {
    builder.create<quake::DeallocOp>(operand);
  }
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Reset
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::ResetOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  for (auto [operand, result] :
       llvm::zip(qtxOp.getOperands(), qtxOp.getResults())) {
    result.replaceAllUsesWith(operand);
    builder.create<quake::ResetOp>(TypeRange{}, operand);
  }
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::ReturnOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  builder.create<func::ReturnOp>(qtxOp.getOperands());
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayBorrow
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::ArrayBorrowOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  // I need the adaptor to get the operands because such methods in `quakeOp`
  // will check the type, which I have already changed for QTX types, and throw
  // an error.
  qtx::ArrayBorrowOp::Adaptor adaptor(qtxOp.getOperands(),
                                      qtxOp->getAttrDictionary());

  for (auto [index, wire] : llvm::zip(adaptor.getIndices(), qtxOp.getWires())) {
    Value ref = builder.create<quake::ExtractRefOp>(adaptor.getArray(), index);
    wire.replaceAllUsesWith(ref);
  }
  qtxOp.getNewArray().replaceAllUsesWith(adaptor.getArray());
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ArrayCreate
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::ArrayCreateOp qtxOp) {
  // TODO
  return qtxOp.emitError("translation not implemented");
}

//===----------------------------------------------------------------------===//
// ArrayYield
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::ArrayYieldOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  qtx::ArrayYieldOp::Adaptor adaptor(qtxOp.getOperands(),
                                     qtxOp->getAttrDictionary());
  qtxOp.getNewArray().replaceAllUsesWith(adaptor.getArray());
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// ArraySplit
//===----------------------------------------------------------------------===//

LogicalResult convertOperation(qtx::ArraySplitOp qtxOp) {
  // TODO
  return qtxOp.emitError("translation not implemented");
}

//===----------------------------------------------------------------------===//
// Convert measurements
//===----------------------------------------------------------------------===//

template <typename QuakeOp, typename QTXOp>
LogicalResult convertMeasurement(QTXOp qtxOp) {
  using Adaptor = typename QTXOp::Adaptor;
  Adaptor adaptor(qtxOp.getOperands(), qtxOp->getAttrDictionary());
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  for (auto [input, result] :
       llvm::zip(adaptor.getTargets(), qtxOp.getNewTargets())) {
    result.replaceAllUsesWith(input);
  }
  Type resTy = qtxOp.getBits().getType();
  if (!resTy.isa<IntegerType>())
    resTy = cudaq::cc::StdvecType::get(builder.getI1Type());
  auto quakeOp = builder.create<QuakeOp>(resTy, adaptor.getTargets());
  qtxOp.getBits().replaceAllUsesWith(quakeOp.getBits());
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Convert quantum operators
//===----------------------------------------------------------------------===//

template <typename QuakeOp, typename QTXOp>
LogicalResult convertOperator(QTXOp qtxOp) {
  ImplicitLocOpBuilder builder(qtxOp.getLoc(), qtxOp);
  auto quakeOp =
      builder.create<QuakeOp>(qtxOp.getIsAdj(), qtxOp.getParameters(),
                              qtxOp.getControls(), qtxOp.getTargets());
  for (auto [tgt, ret] : llvm::zip(quakeOp.getTargets(), qtxOp.getResults()))
    ret.replaceAllUsesWith(tgt);
  qtxOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//

LogicalResult convertOperation(Operation &op) {
  using namespace qtx;
  return llvm::TypeSwitch<Operation *, LogicalResult>(&op)
      .Case<AllocaOp>([&](AllocaOp op) { return convertOperation(op); })
      .Case<DeallocOp>([&](DeallocOp op) { return convertOperation(op); })
      .Case<ResetOp>([&](ResetOp op) { return convertOperation(op); })
      // Structure Ops
      .Case<ReturnOp>([&](ReturnOp op) { return convertOperation(op); })
      // WireArray operations
      .Case<ArrayCreateOp>(
          [&](ArrayCreateOp op) { return convertOperation(op); })
      .Case<ArraySplitOp>([&](ArraySplitOp op) { return convertOperation(op); })
      .Case<ArrayBorrowOp>(
          [&](ArrayBorrowOp op) { return convertOperation(op); })
      .Case<ArrayYieldOp>([&](ArrayYieldOp op) { return convertOperation(op); })
      // Quantum operators
      .Case<HOp>([&](HOp op) { return convertOperator<quake::HOp>(op); })
      .Case<SOp>([&](SOp op) { return convertOperator<quake::SOp>(op); })
      .Case<TOp>([&](TOp op) { return convertOperator<quake::TOp>(op); })
      .Case<XOp>([&](XOp op) { return convertOperator<quake::XOp>(op); })
      .Case<YOp>([&](YOp op) { return convertOperator<quake::YOp>(op); })
      .Case<ZOp>([&](ZOp op) { return convertOperator<quake::ZOp>(op); })
      .Case<SwapOp>(
          [&](SwapOp op) { return convertOperator<quake::SwapOp>(op); })
      .Case<RxOp>([&](RxOp op) { return convertOperator<quake::RxOp>(op); })
      .Case<RyOp>([&](RyOp op) { return convertOperator<quake::RyOp>(op); })
      .Case<RzOp>([&](RzOp op) { return convertOperator<quake::RzOp>(op); })
      .Case<R1Op>([&](R1Op op) { return convertOperator<quake::R1Op>(op); })
      .Case<U2Op>([&](U2Op op) { return convertOperator<quake::U2Op>(op); })
      .Case<U3Op>([&](U3Op op) { return convertOperator<quake::U3Op>(op); })
      // Measurements
      .Case<MxOp>([&](MxOp op) { return convertMeasurement<quake::MxOp>(op); })
      .Case<MyOp>([&](MyOp op) { return convertMeasurement<quake::MyOp>(op); })
      .Case<MzOp>([&](MzOp op) { return convertMeasurement<quake::MzOp>(op); })
      .Default([&](Operation *) { return success(); });
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

/// Convert the types of the arguments. Add returns to the function for each
/// quantum input. Currently this handles only Refs and not Qvecs.
void fixArgumentsAndAddReturns(qtx::CircuitOp circuitOp) {
  auto context = circuitOp->getContext();
  auto refType = quake::RefType::get(context);

  // Iterate over the target arguments while converting types
  for (auto arg : circuitOp.getTargets()) {
    if (arg.getType().isa<qtx::WireType>()) {
      arg.setType(refType);
      continue;
    }
    auto type = dyn_cast<qtx::WireArrayType>(arg.getType());
    auto qvecType = quake::QVecType::get(context, type.getSize());
    arg.setType(qvecType);
  }

  auto terminator =
      dyn_cast<qtx::ReturnOp>(circuitOp.getBody().back().getTerminator());
  ImplicitLocOpBuilder builder = ImplicitLocOpBuilder::atBlockTerminator(
      circuitOp->getLoc(), &circuitOp.getBody().back());
  terminator->replaceAllUsesWith(
      builder.create<qtx::ReturnOp>(terminator.getClassical(), ValueRange{}));
  terminator->erase();
}

LogicalResult convertOperation(qtx::CircuitOp circuitOp) {
  fixArgumentsAndAddReturns(circuitOp);
  auto context = circuitOp->getContext();
  ImplicitLocOpBuilder builder(circuitOp.getLoc(), circuitOp);
  ValueRange args(circuitOp.getArguments());
  auto funcType = FunctionType::get(context, args.getTypes(),
                                    circuitOp.getClassicalResultTypes());
  auto funcOp =
      builder.create<func::FuncOp>(circuitOp.getSymNameAttr(), funcType);
  funcOp.getBody().takeBody(circuitOp.getBody());

  for (auto &op : llvm::make_early_inc_range(funcOp.getOps())) {
    if (failed(convertOperation(op)))
      return failure();
  }
  circuitOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//

class QTXToQuakePass
    : public cudaq::opt::ConvertQTXToQuakeBase<QTXToQuakePass> {
public:
  QTXToQuakePass() = default;

  void runOnOperation() override final {
    auto module = getOperation();

    // Insert an unrealized conversion for any arguments to the functions.
    for (auto op :
         llvm::make_early_inc_range(module.getOps<qtx::CircuitOp>())) {
      if (failed(convertOperation(op))) {
        emitError(module.getLoc(), "error converting from QTX to Quake\n");
        signalPassFailure();
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> cudaq::opt::createConvertQTXToQuakePass() {
  return std::make_unique<QTXToQuakePass>();
}
