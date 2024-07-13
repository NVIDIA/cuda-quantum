/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "StateDecomposer.h"
#include "cudaq/Optimizer/Builder/Runtime.h"
#include "cudaq/Optimizer/CodeGen/QIRFunctionNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Transforms/RegionUtils.h"
#include <span>

namespace cudaq::opt {
#define GEN_PASS_DEF_STATEPREPARATION
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "state-preparation"

using namespace mlir;

/// Replace a qubit initialization from vectors with quantum gates.
/// For example:
///
///
/// Before StatePreparation (state-prep):
///
/// module {
///   func.func @foo() attributes {
///     %0 = cc.address_of @foo.rodata_0 : !cc.ptr<!cc.array<complex<f32> x 4>>
///     %1 = quake.alloca !quake.veq<2>
///     %2 = quake.init_state %1, %0 : (!quake.veq<2>,
///       !cc.ptr<!cc.array<complex<f32> x 4>>) -> !quake.veq<2> return
///  }
///  cc.global constant @foo.rodata_0 (dense<[(0.707106769,0.000000e+00),
///      (0.707106769,0.000000e+00), (0.000000e+00,0.000000e+00),
///      (0.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f32>>) :
///    !cc.array<complex<f32> x 4>
/// }
///
/// After StatePreparation (state-prep):
///
/// module {
///   func.func @foo() attributes {
///     %0 = quake.alloca !quake.veq<2>
///     %c1_i64 = arith.constant 1 : i64
///     %1 = quake.extract_ref %0[%c1_i64] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst = arith.constant 0.000000e+00 : f64
///     quake.ry (%cst) %1 : (f64, !quake.ref) -> ()
///     %c0_i64 = arith.constant 0 : i64
///     %2 = quake.extract_ref %0[%c0_i64] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst_0 = arith.constant 0.78539816339744839 : f64
///     quake.ry (%cst_0) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     %cst_1 = arith.constant 0.78539816339744839 : f64
///     quake.ry (%cst_1) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     return
///   }
/// }

namespace {

std::vector<std::complex<double>>
readConstantArray(mlir::OpBuilder &builder, cudaq::cc::GlobalOp &global) {
  std::vector<std::complex<double>> result{};

  auto attr = global.getValue();
  auto type = global.getType().getElementType();

  auto arrayTy = dyn_cast<cudaq::cc::ArrayType>(type);
  assert(arrayTy);
  assert(attr.has_value());

  auto elementsAttr = dyn_cast<mlir::ElementsAttr>(attr.value());
  assert(elementsAttr);
  auto eleTy = elementsAttr.getElementType();
  auto values = elementsAttr.getValues<mlir::Attribute>();

  for (auto it = values.begin(); it != values.end(); ++it) {
    auto valAttr = *it;

    auto v = [&]() -> std::complex<double> {
      if (isa<FloatType>(eleTy))
        return {cast<FloatAttr>(valAttr).getValue().convertToDouble(),
                static_cast<double>(0.0)};
      if (isa<IntegerType>(eleTy))
        return {static_cast<double>(cast<IntegerAttr>(valAttr).getInt()),
                static_cast<double>(0.0)};
      assert(isa<ComplexType>(eleTy));
      auto arrayAttr = cast<mlir::ArrayAttr>(valAttr);
      auto real = cast<FloatAttr>(arrayAttr[0]).getValue().convertToDouble();
      auto imag = cast<FloatAttr>(arrayAttr[1]).getValue().convertToDouble();
      return {real, imag};
    }();

    result.push_back(v);
  }
  return result;
}

LogicalResult transform(ModuleOp module, func::FuncOp funcOp) {
  auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
  auto toErase = std::vector<mlir::Operation *>();
  auto hasInitState = false;
  auto replacedInitState = false;

  funcOp->walk([&](Operation *op) {
    if (auto initOp = dyn_cast<quake::InitializeStateOp>(op)) {
      toErase.push_back(initOp);
      hasInitState = true;
      auto loc = op->getLoc();
      builder.setInsertionPointAfter(initOp);
      // Find the qvector alloc.
      auto qubits = initOp.getOperand(0);
      if (auto alloc = dyn_cast<quake::AllocaOp>(qubits.getDefiningOp())) {

        // Find vector data.
        auto data = initOp.getOperand(1);
        if (auto cast = dyn_cast<cudaq::cc::CastOp>(data.getDefiningOp())) {
          data = cast.getOperand();
          toErase.push_back(cast);
        }
        if (auto addr =
                dyn_cast<cudaq::cc::AddressOfOp>(data.getDefiningOp())) {

          auto globalName = addr.getGlobalName();
          auto symbol = module.lookupSymbol(globalName);
          if (auto global = dyn_cast<cudaq::cc::GlobalOp>(symbol)) {
            // Read state initialization data from the global array.
            auto vec = readConstantArray(builder, global);

            // Prepare state from vector data.
            auto gateBuilder = StateGateBuilder(builder, loc, qubits);
            auto decomposer = StateDecomposer(gateBuilder, vec);
            decomposer.decompose();

            initOp.replaceAllUsesWith(qubits);
            toErase.push_back(addr);
            toErase.push_back(global);
            replacedInitState = true;
          }
        }
      }
    }
  });

  if (hasInitState && !replacedInitState) {
    funcOp.emitOpError("StatePreparation failed to replace quake.init_state");
    return failure();
  }

  for (auto &op : toErase) {
    if (op->getUses().empty()) {
      op->erase();
    } else {
      op->emitOpError("StatePreparation failed to remove quake.init_state "
                      "or its dependencies.");
      return failure();
    }
  }

  return success();
}

class StatePreparationPass
    : public cudaq::opt::impl::StatePreparationBase<StatePreparationPass> {
protected:
public:
  using StatePreparationBase::StatePreparationBase;

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto module = getModule();
    for (Operation &op : *module.getBody()) {
      auto funcOp = dyn_cast<func::FuncOp>(op);
      if (!funcOp)
        continue;
      std::string kernelName = funcOp.getName().str();

      auto result = transform(module, funcOp);
      if (result.failed()) {
        funcOp.emitOpError("Failed to prepare state for '" + kernelName);
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
