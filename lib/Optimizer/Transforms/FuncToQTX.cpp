/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

using namespace mlir;

static LogicalResult convertOperation(func::FuncOp funcOp) {
  auto convertType = [](Type type) -> Type {
    if (type.isa<quake::QRefType>())
      return qtx::WireType::get(type.getContext());
    else if (auto qvec = type.dyn_cast_or_null<quake::QVecType>())
      return qtx::WireArrayType::get(type.getContext(), qvec.getSize(), 0);
    return nullptr;
  };
  auto entryBlock = &*funcOp.getBody().begin();
  // Iterate over the arguments while converting quantum types, and adding
  // those quantum arguments to the return list---in QTX all quantum arguments
  // in a circuit must be returned.
  auto cachedSize = entryBlock->getNumArguments();
  BitVector indices(cachedSize, false);
  for (auto i = 0u; i < cachedSize; ++i) {
    auto arg = entryBlock->getArgument(i);
    if (auto newType = convertType(arg.getType())) {
      auto newArg = entryBlock->addArgument(newType, arg.getLoc());
      arg.replaceAllUsesWith(newArg);
      indices.set(i);
    }
  }
  indices.resize(entryBlock->getNumArguments(), false);
  unsigned numQuantumArgs = indices.size() - cachedSize;
  entryBlock->eraseArguments(indices);
  auto classicalArgs = entryBlock->getArguments().drop_back(numQuantumArgs);
  auto quantumArgs = entryBlock->getArguments().take_back(numQuantumArgs);

  OpBuilder builder(funcOp);
  auto circuitOp = builder.create<qtx::CircuitOp>(
      funcOp.getLoc(), funcOp.getSymVisibilityAttr(), funcOp.getSymNameAttr(),
      classicalArgs, quantumArgs, funcOp.getResultTypes());
  circuitOp.getBody().takeBody(funcOp.getBody());

  // Pass along the cudaq-entrypoint tag...
  if (funcOp->hasAttr(cudaq::entryPointAttrName))
    circuitOp->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());

  for (auto &op : llvm::make_early_inc_range(circuitOp.getOps())) {
    if (isa<func::ReturnOp>(op))
      op.erase();
    else if (auto unrealized_return = dyn_cast<qtx::UnrealizedReturnOp>(op)) {
      builder.setInsertionPoint(unrealized_return);
      builder.create<qtx::ReturnOp>(op.getLoc(),
                                    unrealized_return.getClassical(),
                                    unrealized_return.getTargets());
      op.erase();
    }
  }
  funcOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Func to QTX pass
//===----------------------------------------------------------------------===//

class FuncToQTXPass : public cudaq::opt::ConvertFuncToQTXBase<FuncToQTXPass> {
public:
  void runOnOperation() override final {

    auto module = getOperation();

    // Insert an unrealized conversion for any arguments to the functions.
    for (auto op : llvm::make_early_inc_range(module.getOps<func::FuncOp>())) {
      if (!op->hasAttr("convert-to-circuit"))
        continue;
      if (failed(convertOperation(op))) {
        emitError(module.getLoc(), "error converting from func to QTX\n");
        signalPassFailure();
      }
    }
  }
};

std::unique_ptr<Pass> cudaq::opt::createConvertFuncToQTXPass() {
  return std::make_unique<FuncToQTXPass>();
}
