/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CUDAQLogical/Interfaces/SemanticInterfaces.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"

namespace {

template <typename Op>
bool checkRoot(mlir::MLIRContext &context, cudaq::logical::Stage expectedStage,
               cudaq::logical::RootKind expectedKind) {
  mlir::OperationState state(mlir::UnknownLoc::get(&context),
                             Op::getOperationName());
  mlir::Operation *operation = mlir::Operation::create(state);
  auto root =
      llvm::dyn_cast<cudaq::logical::SemanticRootOpInterface>(operation);
  bool valid = root && root.getSemanticStage() == expectedStage &&
               root.getSemanticRootKind() == expectedKind;
  if (!valid)
    llvm::errs() << Op::getOperationName()
                 << " does not expose its expected semantic root contract\n";
  operation->destroy();
  return valid;
}

} // namespace

int main() {
  mlir::MLIRContext context;
  context.getOrLoadDialect<qlx::QLXDialect>();
  context.getOrLoadDialect<qlx::lvm::LVMDialect>();
  context.getOrLoadDialect<qlx::fabric::FabricDialect>();
  context.getOrLoadDialect<qlx::phys::PhysDialect>();

  bool valid = true;
  valid &= checkRoot<qlx::ProgramOp>(context, cudaq::logical::Stage::P0,
                                     cudaq::logical::RootKind::LogicalProgram);
  valid &=
      checkRoot<qlx::lvm::KernelOp>(context, cudaq::logical::Stage::P1,
                                    cudaq::logical::RootKind::PlacedKernel);
  valid &= checkRoot<qlx::fabric::CircuitOp>(
      context, cudaq::logical::Stage::P2, cudaq::logical::RootKind::QECCircuit);
  valid &= checkRoot<qlx::fabric::GadgetOp>(
      context, cudaq::logical::Stage::P2, cudaq::logical::RootKind::QECGadget);
  valid &=
      checkRoot<qlx::fabric::ProtocolOp>(context, cudaq::logical::Stage::P2,
                                         cudaq::logical::RootKind::QECProtocol);
  valid &=
      checkRoot<qlx::phys::GraphOp>(context, cudaq::logical::Stage::P3,
                                    cudaq::logical::RootKind::PhysicalGraph);

  valid &= cudaq::logical::stringifyStage(cudaq::logical::Stage::P4) == "p4";
  valid &= cudaq::logical::stringifyRootKind(
               cudaq::logical::RootKind::RealtimePlan) == "realtime_plan";
  return valid ? 0 : 1;
}
