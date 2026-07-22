// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "cudaq-pulse/Dialect/Pulse/PulseDialect.h.inc"
#include "cudaq-pulse/Dialect/Pulse/PulseEnums.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/Pulse/PulseOps.h.inc"

#include "cudaq-pulse/Dialect/QOp/QOpDialect.h.inc"
#include "cudaq-pulse/Dialect/QOp/QOpEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/QOp/QOpOps.h.inc"

#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatDialect.h.inc"
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatEnums.h.inc"
#define GET_ATTRDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatAttrs.h.inc"
#define GET_TYPEDEF_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatTypes.h.inc"
#define GET_OP_CLASSES
#include "cudaq-pulse/Dialect/CuDensityMat/CuDensityMatOps.h.inc"

// Conversion passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseToQOpPass();
}
namespace qop {
std::unique_ptr<mlir::Pass> createQOpToCuDensityMatPass();
}
namespace cudm {
std::unique_ptr<mlir::Pass> createCuDensityMatToLLVMPass();
}
// Pulse Transforms passes
namespace pulse {
std::unique_ptr<mlir::Pass> createPulseVerifyPass();
std::unique_ptr<mlir::Pass> createVirtualZPass();
std::unique_ptr<mlir::Pass> createPulseFusionPass();
std::unique_ptr<mlir::Pass> createPulseScheduleAlapPass();
} // namespace pulse

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  registry.insert<pulse::PulseDialect>();
  registry.insert<qop::QOpDialect>();
  registry.insert<cudm::CuDensityMatDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  // Conversion passes
  mlir::registerPass(pulse::createPulseToQOpPass);
  mlir::registerPass(qop::createQOpToCuDensityMatPass);
  mlir::registerPass(cudm::createCuDensityMatToLLVMPass);

  // Pulse Transforms passes
  mlir::registerPass(pulse::createPulseVerifyPass);
  mlir::registerPass(pulse::createVirtualZPass);
  mlir::registerPass(pulse::createPulseFusionPass);
  mlir::registerPass(pulse::createPulseScheduleAlapPass);

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "cudaq-pulse MLIR optimizer\n", registry));
}
