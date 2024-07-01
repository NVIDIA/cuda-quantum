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

#include <iostream>

#define DEBUG_TYPE "state-preparation2"

using namespace mlir;

/// Replace a qubit initialization from vectors with quantum gates.
/// For example:
///
/// func.func @__nvqpp__mlirgen__function_test._Z4testSt6vectorISt7complexIfESaIS1_EE() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
///   %0 = cc.address_of @__nvqpp_rodata_init_state.0 : !cc.ptr<!cc.array<complex<f32> x 4>>
///   %1 = cc.cast %0 : (!cc.ptr<!cc.array<complex<f32> x 4>>) -> !cc.ptr<complex<f32>>
///   %2 = quake.alloca !quake.veq<2>
///   %3 = quake.init_state %2, %1 : (!quake.veq<2>, !cc.ptr<complex<f32>>) -> !quake.veq<2>
///   return
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
///
/// Note: the following synthesis and const prop passes will replace
/// the argument by a constant and propagate the values and vector size
/// through other instructions.

namespace {

template <typename T>
concept IntegralType =
    std::is_same<T, bool>::value || std::is_same<T, std::int8_t>::value ||
    std::is_same<T, std::int16_t>::value ||
    std::is_same<T, std::int32_t>::value ||
    std::is_same<T, std::int64_t>::value;

template <typename T>
concept FloatingType = std::is_same<T, float>::value;

template <typename T>
concept DoubleType = std::is_same<T, double>::value;

template <typename T>
concept ComplexDataType = FloatingType<T> || DoubleType<T> || IntegralType<T>;

/// Input was complex<float> but we prefer
/// complex<double>. Make a copy, extending the values.
template <FloatingType From>
std::vector<std::complex<double>> convertToComplex(std::complex<From> *data,
                                                   std::uint64_t size) {
  auto convertData = std::vector<std::complex<double>>(size);
  for (std::size_t i = 0; i < size; ++i)
    convertData[i] = std::complex<double>{static_cast<double>(data[i].real()),
                                          static_cast<double>(data[i].imag())};
  return convertData;
}

template <DoubleType From>
std::vector<std::complex<double>> convertToComplex(std::complex<From> *data,
                                                   std::uint64_t size) {
  return std::vector<std::complex<From>>(data, data + size);
}

/// Input was float/double but we prefer complex<double>.
/// Make a copy, extending or truncating the values.
template <ComplexDataType From>
std::vector<std::complex<double>> convertToComplex(From *data,
                                                   std::uint64_t size) {
  auto convertData = std::vector<std::complex<double>>(size);
  for (std::size_t i = 0; i < size; ++i)
    convertData[i] = std::complex<double>{static_cast<double>(data[i]),
                                          static_cast<double>(0.0)};
  return convertData;
}

std::vector<std::complex<double>> readConstantArray(mlir::OpBuilder &builder, cudaq::cc::GlobalOp &global) {
  std::vector<std::complex<double>> result{};

  auto attr = global.getValue();
  auto type = global.getType().getElementType();
  
  if (auto arrayTy = dyn_cast<cudaq::cc::ArrayType>(type)) {
    auto eleTy = arrayTy.getElementType();
    std::cout << "Attribute element type:" << std::endl;
    eleTy.dump();

    if (attr.has_value()) {
      //  auto tensorTy = RankedTensorType::get(size, eleTy);
      // auto f64Attr = DenseElementsAttr::get(tensorTy, values);
      if (auto elementsAttr = dyn_cast<mlir::ElementsAttr>(attr.value())) {
        auto values = elementsAttr.getValues<double>();
        for (auto it = values.begin(); it != values.end(); ++it) {
          result.push_back({*it, 0.0});
        }
      }

      else if (auto values = dyn_cast<mlir::ArrayAttr>(attr.value())) {
        for (auto it = values.begin(); it != values.end(); ++it) {
          auto real = *it;
        // for (std::size_t idx = 0; idx < numConstants; idx += isComplex ? 2 : 1) {
          auto v = [&]() -> std::complex<double> {
            //auto val = constantValues[idx];
            
            if (isa<FloatType>(eleTy))
              return {
                cast<FloatAttr>(real).getValue().convertToDouble(), 
                static_cast<double>(0.0)
              };
            if (isa<IntegerType>(eleTy))
              return {
                static_cast<double>(cast<IntegerAttr>(real).getInt()), 
                static_cast<double>(0.0)
              };
            assert(isa<ComplexType>(eleTy));
            it++;
            auto imag = *it;
            return {
                cast<FloatAttr>(real).getValue().convertToDouble(),
                cast<FloatAttr>(imag).getValue().convertToDouble()
            };
          }();

          result.push_back(v);
        }
      }
    }
  }

  std::cout << "Results (" <<  result.size() << "):" << std::endl;
  for (auto &r: result) {
    std::cout << r << ", " << std::endl;
  }
  return result;
}

LogicalResult
transform(OpBuilder &builder, ModuleOp module) {
  //auto *ctx = builder.getContext();

  auto toErase = std::vector<mlir::Operation *>();

// Module after everything
// module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128", llvm.triple = "x86_64-unknown-linux-gnu", quake.mangled_name_map = {__nvqpp__mlirgen__function_test._Z4testSt6vectorISt7complexIfESaIS1_EE = "_Z4testSt6vectorISt7complexIfESaIS1_EE"}} {
//   func.func @__nvqpp__mlirgen__function_test._Z4testSt6vectorISt7complexIfESaIS1_EE() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
//     %0 = cc.address_of @__nvqpp_rodata_init_state.0 : !cc.ptr<!cc.array<complex<f32> x 4>>
//     %1 = cc.cast %0 : (!cc.ptr<!cc.array<complex<f32> x 4>>) -> !cc.ptr<complex<f32>>
//     %2 = quake.alloca !quake.veq<2>
//     %3 = quake.init_state %2, %1 : (!quake.veq<2>, !cc.ptr<complex<f32>>) -> !quake.veq<2>
//     return
//   }
//   cc.global constant @__nvqpp_rodata_init_state.0 ([0.707106769 : f32, 0.000000e+00 : f32, 0.707106769 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32, 0.000000e+00 : f32]) : !cc.array<complex<f32> x 4>
// }

// func.func @__nvqpp__mlirgen__function_f._Z1fv() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
//   %0 = cc.address_of @__nvqpp__rodata_init_0 : !cc.ptr<!cc.array<f64 x 4>>
//   %1 = quake.alloca !quake.veq<2>
//   %2 = quake.init_state %1, %0 : (!quake.veq<2>, !cc.ptr<!cc.array<f64 x 4>>) -> !quake.veq<2>
//   quake.dealloc %2 : !quake.veq<2>
//   return
// }

  
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
        if (auto addr = dyn_cast<cudaq::cc::AddressOfOp>(data.getDefiningOp())) {
          
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
    op->erase();
  }

  return success();
}

class StatePreparation2 : public cudaq::opt::PrepareState2Base<StatePreparation2> {
protected:
  // The name of the kernel to be synthesized
  std::string kernelName;

  // The raw pointer to the runtime arguments.
  void *args;

public:
  StatePreparation2() = default;
  StatePreparation2(std::string_view kernel, void *a)
      : kernelName(kernel), args(a) {}

  mlir::ModuleOp getModule() { return getOperation(); }


  void runOnOperation() override final {
    auto module = getModule();

    std::cout << "Module before state prep2" << std::endl;
    module.dump();

    auto kernelNameInQuake = cudaq::runtime::cudaqGenPrefixName + kernelName;
    // Get the function we care about (the one with kernelName)
    auto funcOp = module.lookupSymbol<func::FuncOp>(kernelNameInQuake);
    if (!funcOp) {
      module.emitOpError("The kernel '" + kernelName +
                         "' was not found in the module.");
      signalPassFailure();
      return;
    }

    // Create the builder.
    auto builder = OpBuilder::atBlockBegin(&funcOp.getBody().front());

    auto result = transform(builder, module);
    if (result.failed()) {
      module.emitOpError("Failed to prepare state for '" + kernelName);
      signalPassFailure();
      return;
    }
    
    std::cout << "Module after state prep2" << std::endl;
    module.dump();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createStatePreparation2() {
  return std::make_unique<StatePreparation2>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createStatePreparation2(std::string_view kernelName, void *a) {
  return std::make_unique<StatePreparation2>(kernelName, a);
}
