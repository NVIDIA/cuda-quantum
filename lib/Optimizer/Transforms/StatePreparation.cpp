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
#include "mlir/Transforms/RegionUtils.h"
#include <span>

#define DEBUG_TYPE "state-preparation"

using namespace mlir;

/// Replace a qubit initialization from vectors with quantum gates.
/// For example:
///
/// func.func
/// @__nvqpp__mlirgen__function_test._Z4testSt6vectorISt7complexIfESaIS1_EE()
/// attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
///   %0 = cc.address_of @__nvqpp_rodata_init_state.0 :
///   !cc.ptr<!cc.array<complex<f32> x 4>> %1 = cc.cast %0 :
///   (!cc.ptr<!cc.array<complex<f32> x 4>>) -> !cc.ptr<complex<f32>> %2 =
///   quake.alloca !quake.veq<2> %3 = quake.init_state %2, %1 : (!quake.veq<2>,
///   !cc.ptr<complex<f32>>) -> !quake.veq<2> return
/// }
///
/// is converted to:
///
///   func.func @foo(%arg0 : !cc.stdvec<complex<f32>>) {
///     %0 = quake.alloca !quake.veq<2>
///     %c0_i64 = arith.constant 0 : i64
///     %1 = quake.extract_ref %0[%c0_i64] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst = arith.constant 1.5707963267948968 : f64
///     quake.ry (%cst) %1 : (f64, !quake.ref) -> ()
///     %c1_i64 = arith.constant 1 : i64
///     %2 = quake.extract_ref %0[%c1_i64] : (!quake.veq<2>, i64) -> !quake.ref
///     %cst_0 = arith.constant 1.5707963267948966 : f64
///     quake.ry (%cst_0) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     %cst_1 = arith.constant -1.5707963267948966 : f64
///     quake.ry (%cst_1) %2 : (f64, !quake.ref) -> ()
///     quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
///     return
///   }

namespace {

std::vector<std::complex<double>>
readConstantArray(mlir::OpBuilder &builder, cudaq::cc::GlobalOp &global) {
  std::vector<std::complex<double>> result{};

  auto attr = global.getValue();
  auto type = global.getType().getElementType();

  if (auto arrayTy = dyn_cast<cudaq::cc::ArrayType>(type)) {
    auto eleTy = arrayTy.getElementType();

    if (attr.has_value()) {
      if (auto elementsAttr = dyn_cast<mlir::ElementsAttr>(attr.value())) {
        auto eleTy = elementsAttr.getElementType();
        if (isa<ComplexType>(eleTy)) {
          auto values = elementsAttr.getValues<mlir::ArrayAttr>();
          for (auto it = values.begin(); it != values.end(); ++it) {
            auto valueAttr = *it;
            auto real =
                cast<FloatAttr>(valueAttr[0]).getValue().convertToDouble();
            auto imag =
                cast<FloatAttr>(valueAttr[1]).getValue().convertToDouble();
            result.push_back({real, imag});
          }
        } else {
          auto values = elementsAttr.getValues<double>();
          for (auto it = values.begin(); it != values.end(); ++it) {
            result.push_back({*it, 0.0});
          }
        }
      } else if (auto values = dyn_cast<mlir::ArrayAttr>(attr.value())) {
        for (auto it = values.begin(); it != values.end(); ++it) {
          auto real = *it;
          // for (std::size_t idx = 0; idx < numConstants; idx += isComplex ? 2
          // : 1) {
          auto v = [&]() -> std::complex<double> {
            if (isa<FloatType>(eleTy))
              return {cast<FloatAttr>(real).getValue().convertToDouble(),
                      static_cast<double>(0.0)};
            if (isa<IntegerType>(eleTy))
              return {static_cast<double>(cast<IntegerAttr>(real).getInt()),
                      static_cast<double>(0.0)};
            assert(isa<ComplexType>(eleTy));
            it++;
            auto imag = *it;
            return {cast<FloatAttr>(real).getValue().convertToDouble(),
                    cast<FloatAttr>(imag).getValue().convertToDouble()};
          }();

          result.push_back(v);
        }
      }
    }
  }

  return result;
}

LogicalResult transform(OpBuilder &builder, ModuleOp module) {
  auto toErase = std::vector<mlir::Operation *>();
  module->walk([&](Operation *op) {
    if (auto initOp = dyn_cast<quake::InitializeStateOp>(op)) {
      toErase.push_back(initOp);
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
          }
        }
      }
    }
  });

  for (auto &op : toErase) {
    if (op->getUses().empty()) {
      op->erase();
    } else {
      module.emitOpError("StatePreparation failed to remove quake.init_state "
                         "or its dependencies.");
      return failure();
    }
  }

  return success();
}

class StatePreparation : public cudaq::opt::PrepareStateBase<StatePreparation> {
protected:
  // The name of the kernel to be synthesized
  std::string kernelName;

public:
  StatePreparation() = default;
  StatePreparation(std::string_view kernel) : kernelName(kernel) {}

  mlir::ModuleOp getModule() { return getOperation(); }

  void runOnOperation() override final {
    auto module = getModule();
    auto kernelNameInQuake = cudaq::runtime::cudaqGenPrefixName + kernelName;
    // Get the function we care about (the one with kernelName)
    auto funcOp = module.lookupSymbol<func::FuncOp>(kernelNameInQuake);
    if (!funcOp) {
      module.emitOpError("The kernel '" + kernelName +
                         "' was not found in the module.");
      signalPassFailure();
      return;
    }

    auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());
    auto result = transform(builder, module);
    if (result.failed()) {
      module.emitOpError("Failed to prepare state for '" + kernelName);
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createStatePreparation() {
  return std::make_unique<StatePreparation>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createStatePreparation(std::string_view kernelName) {
  return std::make_unique<StatePreparation>(kernelName);
}
