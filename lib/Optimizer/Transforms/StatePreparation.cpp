/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
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
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/TypeToLLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/RegionUtils.h"

#include <iostream>

#define DEBUG_TYPE "state-preparation"

using namespace mlir;

/// Replace a qubit initialization from vectors with quantum gates.
/// For example:
///
///   func.func @foo(%arg0 : !cc.stdvec<complex<f32>>) {
///     %0 = cc.stdvec_size %arg0 : (!cc.stdvec<complex<f32>>) -> i64
///     %1 = math.cttz %0 : i64
///     %2 = cc.stdvec_data %arg0 : (!cc.stdvec<complex<f32>>) -> !cc.ptr<complex<f32>>
///     %3 = quake.alloca !quake.veq<?>[%1 : i64]
///     %4 = quake.init_state %3, %2 : (!quake.veq<?>, !cc.ptr<complex<f32>>) -> !quake.veq<?>
///     return
///   }
///
/// on call that passes std::vector<cudaq::complex> vec{M_SQRT1_2, 0., 0., M_SQRT1_2} as arg0
/// will be updated to:
///
///   func.func @foo(%arg0 : !cc.stdvec<complex<f32>>) {
///     %0 = cc.stdvec_size %arg0 : (!cc.stdvec<complex<f32>>) -> i64
///     %c4_i64 = arith.constant 4 : i64
///     %3 = math.cttz %c4_i64 : i64
///     %5 = quake.alloca !quake.veq<?>[%3 : i64]
///     %6 = quake.extract_ref %5[0] : (!quake.veq<?>) -> !quake.ref
///     quake.h %6 : (!quake.ref) -> ()
///     %7 = quake.extract_ref %5[0] : (!quake.veq<?>) -> !quake.ref
///     %8 = quake.extract_ref %5[1] : (!quake.veq<?>) -> !quake.ref
///     quake.x [%7] %8 : (!quake.ref, !quake.ref) -> ()
///   }
///
/// Note: we rely on the later synthesis and const prop stages to replace
/// the argument by a constant and propagate the values and vector size
/// through those and other instructions.

namespace {
class StatePreparation
    : public cudaq::opt::StatePreparationBase<StatePreparation> {
protected:
  // The name of the kernel to be synthesized
  std::string kernelName;

  // The raw pointer to the runtime arguments.
  void *args;

public:
  StatePreparation() = default;
  StatePreparation(std::string_view kernel, void *a)
      : kernelName(kernel), args(a) {}

  mlir::ModuleOp getModule() { return getOperation(); }


  void runOnOperation() override final {
    std::cout << "Module before state prep " << std::endl;
    auto module = getModule();
    module.dump();
    if (args == nullptr || kernelName.empty()) {
      module.emitOpError("Synthesis requires a kernel and the values of the "
                         "arguments passed when it is called.");
      signalPassFailure();
      return;
    }

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
    
    std::cout << "Module after synthesis " << std::endl; 
    module.dump();
  }
};

} // namespace

std::unique_ptr<mlir::Pass> cudaq::opt::createStatePreparation() {
  return std::make_unique<StatePreparation>();
}

std::unique_ptr<mlir::Pass>
cudaq::opt::createStatePreparation(std::string_view kernelName, void *a) {
  return std::make_unique<StatePreparation>(kernelName, a);
}
