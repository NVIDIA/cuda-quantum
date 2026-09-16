/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// An mlir-lsp-server that knows about the retained P0-P3 dialects so
// the VS Code MLIR extension can provide diagnostics, hover, and navigation
// for .mlir/.qlx sources in this project.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Cflow/IR/CflowDialect.h"
#include "qlx/Dialect/Event/IR/EventDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Tools/mlir-lsp-server/MlirLspServerMain.h"

int main(int argc, char **argv) {
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
  return mlir::failed(mlir::MlirLspServerMain(argc, argv, registry));
}
