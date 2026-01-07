/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"
#include <span>

namespace cudaq::opt {
#define GEN_PASS_DEF_DELETESTATES
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "delete-states"

using namespace mlir;

namespace {
/// For a `quake.create_state`, get the number of qubits allocated.
static std::size_t getStateSize(Operation *op) {
  if (auto createStateOp = dyn_cast<quake::CreateStateOp>(op)) {
    auto sizeOperand = createStateOp.getOperand(1);
    auto defOp = sizeOperand.getDefiningOp();
    while (defOp && !dyn_cast<arith::ConstantIntOp>(defOp))
      defOp = defOp->getOperand(0).getDefiningOp();
    if (auto constOp = dyn_cast<arith::ConstantIntOp>(defOp))
      return constOp.getValue().cast<IntegerAttr>().getInt();
  }
  op->emitError("Cannot compute number of qubits from createStateOp");
  return 0;
}

// clang-format off
/// Replace `quake.get_number_of_qubits` by a constant.
/// ```
/// %c8_i64 = arith.constant 8 : i64
/// %2 = quake.create_state %3, %c8_i64 : (!cc.ptr<i8>, i64) -> !cc.ptr<!quake.state>
/// %3 = quake.get_number_of_qubits %2 : i64
/// ...
/// ───────────────────────────────────────────
/// %c8_i64 = arith.constant 8 : i64
/// %2 = quake.create_state %3, %c8_i64 : (!cc.ptr<i8>, i64) -> !cc.ptr<!quake.state>
/// %3 = arith.constant 3 : i64
/// ```
// clang-format on
class NumberOfQubitsPattern
    : public OpRewritePattern<quake::GetNumberOfQubitsOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::GetNumberOfQubitsOp op,
                                PatternRewriter &rewriter) const override {
    auto stateOp = op.getOperand();
    if (auto createStateOp = stateOp.getDefiningOp<quake::CreateStateOp>()) {
      auto size = getStateSize(createStateOp);
      rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(
          op, std::countr_zero(size), rewriter.getI64Type());
      return success();
    }
    return failure();
  }
};

// clang-format off
/// Remove `quake.create_state` instructions and pass their data directly to
/// the `quake.init_state` instruction instead.
/// ```
/// %2 = cc.cast %1 : (!cc.ptr<!cc.array<complex<f32> x 8>>) -> !cc.ptr<i8>
/// %3 = quake.create_state %3, %c8_i64 : (!cc.ptr<i8>, i64) -> !cc.ptr<!quake.state>
/// %4 = quake.alloca !quake.veq<?>[%0 : i64]
/// %5 = quake.init_state %4, %3 : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
/// ───────────────────────────────────────────
/// ...
/// %4 = quake.alloca !quake.veq<?>[%0 : i64]
/// %5 = quake.init_state %4, %1 : (!quake.veq<?>, !cc.ptr<!cc.array<complex<f32> x 8>>) -> !quake.veq<?>
/// ```
// clang-format on

class StateToDataPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    auto state = initState.getOperand(1);
    auto targets = initState.getTargets();

    if (auto createStateOp = state.getDefiningOp<quake::CreateStateOp>()) {
      auto dataOp = createStateOp->getOperand(0);
      if (auto cast = dataOp.getDefiningOp<cudaq::cc::CastOp>())
        dataOp = cast.getOperand();
      rewriter.replaceOpWithNewOp<quake::InitializeStateOp>(
          initState, targets.getType(), targets, dataOp);
      return success();
    }
    return failure();
  }
};

class DeleteStatesPass
    : public cudaq::opt::impl::DeleteStatesBase<DeleteStatesPass> {
public:
  using DeleteStatesBase::DeleteStatesBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto module = getOperation();
    for (Operation &op : *module.getBody()) {
      auto func = dyn_cast<func::FuncOp>(op);
      if (!func)
        continue;

      std::string funcName = func.getName().str();
      RewritePatternSet patterns(ctx);
      patterns.insert<NumberOfQubitsPattern, StateToDataPattern>(ctx);

      LLVM_DEBUG(llvm::dbgs() << "Before deleting states: " << func << '\n');

      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns))))
        signalPassFailure();

      // Remove unused states.
      llvm::SmallVector<Operation *> usedStates;

      func.walk([&](Operation *op) {
        if (isa<quake::CreateStateOp>(op)) {
          if (!op->getUses().empty())
            usedStates.push_back(op);
        }
      });

      // Call delete for used states on all function exits.
      if (!usedStates.empty()) {
        auto builder = OpBuilder::atBlockBegin(&func.getBody().front());
        cudaq::IRBuilder irBuilder(ctx);
        func.walk([&](Operation *op) {
          if (isa<func::ReturnOp>(op)) {
            auto loc = op->getLoc();
            auto result =
                irBuilder.loadIntrinsic(module, cudaq::deleteCudaqState);
            assert(succeeded(result) && "loading intrinsic should never fail");

            builder.setInsertionPoint(op);
            for (auto createStateOp : usedStates) {
              auto result = cast<quake::CreateStateOp>(createStateOp);
              builder.create<quake::DeleteStateOp>(loc, result);
            }
          }
        });
      }

      LLVM_DEBUG(llvm::dbgs() << "After deleting states: " << func << '\n');
    }
  }
};
} // namespace
