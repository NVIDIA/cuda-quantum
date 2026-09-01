//===- qlx-opt.cpp - QLX dialect opt tool -----------------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// An mlir-opt tool that loads the QLX dialect.
// Used for round-trip testing of QLX MLIR and for running passes.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/InitAllPasses.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  qlx::registerAllQLXPasses();

  mlir::DialectRegistry registry;
  registry.insert<qlx::QLXDialect>();
  registry.insert<qlx::lvm::LVMDialect>();
  registry.insert<qlx::fabric::FabricDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<cudaq::cc::CCDialect>();
  registry.insert<cudaq::quake::QuakeDialect>();
  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "QLX dialect opt tool", registry));
}
