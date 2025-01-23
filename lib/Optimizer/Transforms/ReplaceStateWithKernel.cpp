/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
#define GEN_PASS_DEF_REPLACESTATEWITHKERNEL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "replace-state-with-kernel"

using namespace mlir;

namespace {
// clang-format off
/// Replace `quake.get_number_of_qubits` by a call to a a function
/// that computes the number of qubits for a state.
///
/// ```
///  %0 = quake.get_state "callee.num_qubits_0" "callee.init_0" : !cc.ptr<!cc.state>
///  %1 = quake.get_number_of_qubits %0 : (!cc.ptr<!cc.state>) -> i64
/// ───────────────────────────────────────────
/// ...
///  %1 = call @callee.num_qubits_0() : () -> i64
/// ```
// clang-format on
class ReplaceGetNumQubitsPattern
    : public OpRewritePattern<quake::GetNumberOfQubitsOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::GetNumberOfQubitsOp numQubits,
                                PatternRewriter &rewriter) const override {

    auto stateOp = numQubits.getOperand();
    if (auto getState = stateOp.getDefiningOp<quake::GetStateOp>()) {
      auto numQubitsName = getState.getNumQubitsFuncName();

      rewriter.setInsertionPoint(numQubits);
      rewriter.replaceOpWithNewOp<func::CallOp>(
          numQubits, numQubits.getType(), numQubitsName, mlir::ValueRange{});
      return success();
    }
    return numQubits->emitError(
        "ReplaceStateWithKernel: failed to replace `quake.get_num_qubits`");
  }
};

// clang-format off
/// Replace `quake.init_state` by a call to a (modified) kernel that produced
/// the state.
///
/// ```
///  %0 = quake.get_state "callee.num_qubits_0" "callee.init_0" : !cc.ptr<!cc.state>
///  %3 = quake.init_state %2, %0 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
/// ───────────────────────────────────────────
/// ...
/// %3 = call @callee.init_0(%2): (!quake.veq<?>) -> !quake.veq<?>
/// ```
// clang-format on
class ReplaceInitStatePattern
    : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    auto allocaOp = initState.getOperand(0);
    auto stateOp = initState.getOperand(1);

    if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(stateOp.getType())) {
      if (isa<cudaq::cc::StateType>(ptrTy.getElementType())) {
        if (auto getState = stateOp.getDefiningOp<quake::GetStateOp>()) {
          auto initName = getState.getInitFuncName();

          rewriter.setInsertionPoint(initState);
          rewriter.replaceOpWithNewOp<func::CallOp>(
              initState, initState.getType(), initName,
              mlir::ValueRange{allocaOp});

          return success();
        }

        return initState->emitError(
            "ReplaceStateWithKernel: failed to replace `quake.init_state`");
      }
    }
    return failure();
  }
};

class ReplaceStateWithKernelPass
    : public cudaq::opt::impl::ReplaceStateWithKernelBase<
          ReplaceStateWithKernelPass> {
public:
  using ReplaceStateWithKernelBase::ReplaceStateWithKernelBase;

  void runOnOperation() override {
    auto *ctx = &getContext();
    auto func = getOperation();
    RewritePatternSet patterns(ctx);
    patterns.insert<ReplaceGetNumQubitsPattern, ReplaceInitStatePattern>(ctx);

    LLVM_DEBUG(llvm::dbgs()
               << "Before replace state with kernel: " << func << '\n');

    if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                            std::move(patterns))))
      signalPassFailure();

    LLVM_DEBUG(llvm::dbgs()
               << "After replace state with kernel: " << func << '\n');
  }
};
} // namespace
