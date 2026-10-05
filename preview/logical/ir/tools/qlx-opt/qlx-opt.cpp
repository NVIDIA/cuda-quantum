/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// An mlir-opt tool that loads the QLX dialect.
// Used for round-trip testing of QLX MLIR and for running passes.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Cflow/IR/CflowDialect.h"
#include "qlx/Dialect/Event/IR/EventDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/InitAllPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#ifdef QLX_HAS_CUDAQ_QUAKE
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#endif

int main(int argc, char **argv) {
  qlx::registerAllQLXPasses();

  mlir::DialectRegistry registry;
  registry.insert<qlx::QLXDialect>();
  registry.insert<qlx::lvm::LVMDialect>();
  registry.insert<qlx::phys::PhysDialect>();
  registry.insert<qlx::cflow::CflowDialect>();
  registry.insert<qlx::event::EventDialect>();
  registry.insert<qlx::fabric::FabricDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();
#ifdef QLX_HAS_CUDAQ_QUAKE
  registry.insert<cudaq::cc::CCDialect>();
  registry.insert<cudaq::quake::QuakeDialect>();
#endif
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "QLX dialect opt tool", registry));
}
