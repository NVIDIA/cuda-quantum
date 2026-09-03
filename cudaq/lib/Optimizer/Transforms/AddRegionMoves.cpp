/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_ADDREGIONMOVES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

using namespace mlir;

namespace {

static bool isWire(Value v) {
  return isa<cudaq::quake::WireType>(v.getType());
}

// Return the assigned region name for a lambda, or "" if not annotated.
static StringRef getRegionAttr(cudaq::cc::CreateLambdaOp lambda) {
  auto attr = lambda->getAttrOfType<FlatSymbolRefAttr>(
      StringAttr::get(lambda.getContext(), "region"));
  return attr ? attr.getValue() : StringRef{};
}

// At the entry of each subcircuit body, move wire block arguments into the
// region. At the exit, move returned wires out before cc.return.
static void addIntraBodyMoves(cudaq::cc::CreateLambdaOp lambda) {
  StringRef regionName = getRegionAttr(lambda);
  if (regionName.empty())
    return;

  MLIRContext *ctx = lambda.getContext();
  auto wireTy = cudaq::quake::WireType::get(ctx);
  auto symRef = FlatSymbolRefAttr::get(ctx, regionName);
  Location loc = lambda.getLoc();
  OpBuilder builder(ctx);
  Block &body = lambda.getBody().front();

  // Entry moves: wire arg → region slot (slot indices assigned in arg order).
  builder.setInsertionPointToStart(&body);
  int32_t slot = 0;
  for (BlockArgument arg : body.getArguments()) {
    if (!isWire(arg))
      continue;
    auto slotAttr = builder.getI32IntegerAttr(slot++);
    auto move = builder.create<cudaq::quake::MoveOp>(loc, wireTy, arg, symRef,
                                                     slotAttr);
    arg.replaceUsesWithIf(move.getResult(), [&](OpOperand &use) {
      return use.getOwner() != move.getOperation();
    });
  }
}

// In the enclosing block, insert a quake.move before each cc.call_callable
// for any wire argument that crosses a region boundary.
static void addInterSubcircuitMoves(func::FuncOp func) {
  MLIRContext *ctx = func.getContext();
  auto wireTy = cudaq::quake::WireType::get(ctx);
  OpBuilder builder(ctx);

  func.walk([&](cudaq::cc::CallCallableOp call) {
    auto destLambda =
        call.getCallee().getDefiningOp<cudaq::cc::CreateLambdaOp>();
    if (!destLambda)
      return;
    StringRef destRegion = getRegionAttr(destLambda);
    if (destRegion.empty())
      return;

    builder.setInsertionPoint(call);
    for (auto [i, arg] : llvm::enumerate(call.getArgs())) {
      if (!isWire(arg))
        continue;

      StringRef srcRegion;
      if (auto srcCall = arg.getDefiningOp<cudaq::cc::CallCallableOp>()) {
        auto srcLambda =
            srcCall.getCallee().getDefiningOp<cudaq::cc::CreateLambdaOp>();
        if (srcLambda)
          srcRegion = getRegionAttr(srcLambda);
      }

      if (srcRegion == destRegion)
        continue;

      auto symRef = FlatSymbolRefAttr::get(ctx, destRegion);
      auto move = builder.create<cudaq::quake::MoveOp>(
          call.getLoc(), wireTy, arg, symRef, IntegerAttr{});
      // Operand 0 is the callee; args start at operand 1.
      call->setOperand(1 + i, move.getResult());
    }
  });
}

class AddRegionMovesPass
    : public cudaq::opt::impl::AddRegionMovesBase<AddRegionMovesPass> {
  using Base = cudaq::opt::impl::AddRegionMovesBase<AddRegionMovesPass>;

public:
  using Base::Base;

  void runOnOperation() override {
    func::FuncOp func = getOperation();

    SmallVector<cudaq::cc::CreateLambdaOp> lambdas;
    func.walk([&](cudaq::cc::CreateLambdaOp lambda) {
      lambdas.push_back(lambda);
    });
    if (lambdas.empty())
      return;

    for (auto lambda : lambdas)
      addIntraBodyMoves(lambda);

    addInterSubcircuitMoves(func);
  }
};
} // namespace

namespace cudaq::opt {
std::unique_ptr<mlir::Pass> createAddRegionMovesPass() {
  return std::make_unique<AddRegionMovesPass>();
}
} // namespace cudaq::opt
