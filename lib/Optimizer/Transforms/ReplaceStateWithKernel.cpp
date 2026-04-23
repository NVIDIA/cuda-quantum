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

namespace cudaq::opt {
#define GEN_PASS_DEF_REPLACESTATEWITHKERNEL
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "replace-state-with-kernel"

using namespace mlir;

namespace {
// clang-format off
/// Replace `quake.get_number_of_qubits` by a call to a function
/// that computes the number of qubits for a state.
///
/// ```mlir
///  %0 = quake.materialize_state @callee.num_qubits_0, @callee.init_0 : !cc.ptr<!quake.state>
///  %1 = quake.get_number_of_qubits %0 : (!cc.ptr<!quake.state>) -> i64
/// ───────────────────────────────────────────
///  %1 = call @callee.num_qubits_0() : () -> i64
/// ```
// clang-format on
class ReplaceGetNumQubitsPattern
    : public OpRewritePattern<quake::GetNumberOfQubitsOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::GetNumberOfQubitsOp numQubits,
                                PatternRewriter &rewriter) const override {

    auto stateOp = numQubits.getState();
    auto materializeState = stateOp.getDefiningOp<quake::MaterializeStateOp>();
    if (!materializeState) {
      LLVM_DEBUG(llvm::dbgs() << "ReplaceStateWithKernel: failed to replace "
                                 "`quake.get_num_qubits`: "
                              << stateOp << '\n');
      return failure();
    }

    auto numQubitsFunc = materializeState.getNumQubitsFunc();
    rewriter.setInsertionPoint(numQubits);
    rewriter.replaceOpWithNewOp<func::CallOp>(
        numQubits, numQubits.getType(), numQubitsFunc, mlir::ValueRange{});
    return success();
  }
};

// clang-format off
/// Replace `quake.init_state` by a call to a (modified) kernel that produced
/// the state.
///
/// ```mlir
///  %0 = quake.materialize_state @callee.num_qubits_0, @callee.init_0 : !cc.ptr<!quake.state>
///  %3 = quake.init_state %2, %0 : (!quake.veq<?>, !cc.ptr<!quake.state>) -> !quake.veq<?>
/// ───────────────────────────────────────────
/// %3 = call @callee.init_0(%2): (!quake.veq<?>) -> !quake.veq<?>
/// ```
// clang-format on
class ReplaceInitStatePattern
    : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState,
                                PatternRewriter &rewriter) const override {
    auto allocaOp = initState.getTargets();
    auto stateOp = initState.getState();

    if (auto ptrTy = dyn_cast<cudaq::cc::PointerType>(stateOp.getType())) {
      if (isa<quake::StateType>(ptrTy.getElementType())) {
        auto materializeState =
            stateOp.getDefiningOp<quake::MaterializeStateOp>();
        if (!materializeState) {
          LLVM_DEBUG(llvm::dbgs() << "ReplaceStateWithKernel: failed to "
                                     "replace `quake.init_state`: "
                                  << stateOp << '\n');
          return failure();
        }

        auto initName = materializeState.getInitFunc();
        rewriter.setInsertionPoint(initState);
        rewriter.replaceOpWithNewOp<func::CallOp>(initState,
                                                  initState.getType(), initName,
                                                  mlir::ValueRange{allocaOp});

        return success();
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
