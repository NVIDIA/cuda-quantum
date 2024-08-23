/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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


/// Remove synthesized states from MLIR, or add a delete call.
/// For example:
///
/// Before DeleteStates (delete-states):
/// ```
/// func.func @foo() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
///   %c8_i64 = arith.constant 8 : i64
///   %0 = cc.address_of @function_test_state_param._Z16test_state_paramPN5cudaq5stateE.rodata_synth_0 : !cc.ptr<!cc.array<complex<f32> x 8>>
///   %1 = cc.load %0 : !cc.ptr<!cc.array<complex<f32> x 8>>
///   %2 = cc.alloca !cc.array<complex<f32> x 8>
///   cc.store %1, %2 : !cc.ptr<!cc.array<complex<f32> x 8>>

///   %3 = cc.cast %2 : (!cc.ptr<!cc.array<complex<f32> x 8>>) -> !cc.ptr<i8>
///   %4 = call @__nvqpp_cudaq_state_createFromData_fp32(%3, %c8_i64) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
///   %5 = call @__nvqpp_cudaq_state_numberOfQubits(%4) : (!cc.ptr<!cc.state>) -> i64
///   %6 = quake.alloca !quake.veq<?>[%5 : i64]
///   %7 = quake.init_state %6, %4 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>

///   return
/// }
/// ```
///
/// After DeleteStates (delete-states):
/// ```
/// module {
/// func.func @foo() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
///   %c8_i64 = arith.constant 8 : i64
///   %0 = cc.address_of @function_test_state_param._Z16test_state_paramPN5cudaq5stateE.rodata_synth_0 : !cc.ptr<!cc.array<complex<f32> x 8>>
///   %1 = cc.load %0 : !cc.ptr<!cc.array<complex<f32> x 8>>
///   %2 = cc.alloca !cc.array<complex<f32> x 8>
///   cc.store %1, %2 : !cc.ptr<!cc.array<complex<f32> x 8>>

///   %3 = quake.alloca !quake.veq<3>
///   %4 = quake.init_state %3, %2 : (!quake.veq<3>, !cc.ptr<!cc.array<complex<f32> x 4>>) -> !quake.veq<3>
///   return
/// }
/// ```


namespace {

static bool isCall(Operation* callOp, std::vector<const char*> &&names) {
  if (callOp) {
    if (auto createStateCall = dyn_cast<func::CallOp>(callOp)) {
      if (auto calleeAttr = createStateCall.getCalleeAttr()) {
        auto funcName = calleeAttr.getValue().str();
        if (std::find(names.begin(), names.end(), funcName) != names.end()) 
          return true;
      }
    }
  }
  return false;
}

static bool isCreateStateCall(Operation* callOp) {
  return isCall(callOp, {cudaq::createCudaqStateFromData64, cudaq::createCudaqStateFromData32});
}

static bool isNumberOfQubitsCall(Operation* callOp) {
  return isCall(callOp, {cudaq::getNumQubitsFromCudaqState});
}

/// For a call to `__nvqpp_cudaq_state_createFromData_fpXX`, get the number of
/// qubits allocated.
static std::size_t getStateSize(Operation* callOp) {
  if (isCreateStateCall(callOp)) {
    if (auto createStateCall = dyn_cast<func::CallOp>(callOp)) {
      auto sizeOperand = createStateCall.getOperand(1);
      auto defOp = sizeOperand.getDefiningOp();
      while (defOp && !dyn_cast<arith::ConstantIntOp>(defOp))
        defOp = defOp->getOperand(0).getDefiningOp();
      if (auto constOp = dyn_cast<arith::ConstantIntOp>(defOp))
        return constOp.getValue().cast<IntegerAttr>().getInt();
    }
  }
  callOp->emitError("Cannot compute number of qubits");
  return 0;
}

// Remove `__nvqpp_cudaq_state_numberOfQubits` calls.
/// ```
// %1 = arith.constant 8 : i64
// %2 = call @__nvqpp_cudaq_state_createFromData_fp32(%0, %1) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// %3 = call @__nvqpp_cudaq_state_numberOfQubits(%2) : (!cc.ptr<!cc.state>) -> i64
// ...
// ───────────────────────────────────────────
// %1 = arith.constant 8 : i64
// %2 = call @__nvqpp_cudaq_state_createFromData_fp32(%0, %1) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// %5 = arith.constant 3 : i64
/// ```
class NumberOfQubitsPattern : public OpRewritePattern<func::CallOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(func::CallOp callOp, PatternRewriter &rewriter) const override {
    if (isNumberOfQubitsCall(callOp)) {
      auto createStateOp = callOp.getOperand(0).getDefiningOp();
      if (isCreateStateCall(createStateOp)) {
        auto size = getStateSize(createStateOp);
        rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(callOp, std::countr_zero(size), rewriter.getI64Type());
        return success();
      }
    }
    return failure();
  }
};


// Replace calls to `__nvqpp_cudaq_state_numberOfQubits` by a constant.
// ```
// %2 = cc.cast %1 : (!cc.ptr<!cc.array<complex<f32> x 8>>) -> !cc.ptr<i8>
// %3 = call @__nvqpp_cudaq_state_createFromData_fp32(%2, %c8_i64) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// %4 = quake.alloca !quake.veq<?>[%0 : i64]
// %5 = quake.init_state %4, %3 : (!quake.veq<?>, !cc.ptr<!cc.state>) -> !quake.veq<?>
// ───────────────────────────────────────────
// ...
// %3 = call @__nvqpp_cudaq_state_createFromData_fp32(%2, %c8_i64) : (!cc.ptr<i8>, i64) -> !cc.ptr<!cc.state>
// %4 = quake.alloca !quake.veq<?>[%0 : i64]
// %5 = quake.init_state %4, %1 : (!quake.veq<?>, !cc.ptr<!cc.array<complex<f32> x 8>>) -> !quake.veq<?>
// ```
class StateToDataPattern : public OpRewritePattern<quake::InitializeStateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(quake::InitializeStateOp initState, PatternRewriter &rewriter) const override {
    auto stateOp = initState.getOperand(1).getDefiningOp();
    auto targets = initState.getTargets();

    if (isCreateStateCall(stateOp)) {
      auto dataOp = stateOp->getOperand(0);
      if (auto cast = dyn_cast<cudaq::cc::CastOp>(dataOp.getDefiningOp()))
        dataOp = cast.getOperand();
      rewriter.replaceOpWithNewOp<quake::InitializeStateOp>(initState, targets.getType(), targets, dataOp);
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

      LLVM_DEBUG(llvm::dbgs()
                 << "Before deleting states: " << func << '\n');

      if (failed(applyPatternsAndFoldGreedily(func.getOperation(),
                                              std::move(patterns))))
        signalPassFailure();

      // Remove unused states.
      llvm::SmallVector<Operation *> usedStates;

      func.walk([&](Operation *op) {
        if (isCreateStateCall(op)) {
          if(op->getUses().empty()) 
            op->erase();
          else 
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
            auto deleteState = cudaq::deleteCudaqState;
            auto result = irBuilder.loadIntrinsic(module, deleteState);
            assert(succeeded(result) && "loading intrinsic should never fail");
            
            builder.setInsertionPoint(op);
            for (auto createStateOp: usedStates) {
              auto results = cast<func::CallOp>(createStateOp).getResults();
              builder.create<func::CallOp>(loc, std::nullopt, deleteState, results);
            }
          }
        });
      }

      LLVM_DEBUG(llvm::dbgs()
                 << "After deleting states: " << func << '\n');
    }
  }
};
} // namespace
