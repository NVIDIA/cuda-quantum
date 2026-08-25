/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: test_resource_count_phase | FileCheck %s

#include "cudaq_internal/compiler/ResourceCount.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

static constexpr char phaseKernel[] = R"#(
func.func @phase_resource() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
  %theta = arith.constant 5.000000e-01 : f64
  %control0 = quake.alloca !quake.ref
  %control1 = quake.alloca !quake.ref
  %target = quake.alloca !quake.ref
  quake.phase (%theta) [%control0, %control1 neg [false, true]] %target
      : (f64, !quake.ref, !quake.ref, !quake.ref) -> ()
  return
}
)#";

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();
  auto module =
      mlir::parseSourceString<mlir::ModuleOp>(phaseKernel, context.get());
  if (!module)
    return 1;

  auto resources = cudaq::opt::countResourcesFromIR(*module);
  if (failed(resources))
    return 1;

  llvm::outs() << "x: " << resources->count("x") << '\n';
  llvm::outs() << "controlled r1: " << resources->count_controls("r1", 1)
               << '\n';
  llvm::outs() << "phase: " << resources->count("phase") << '\n';
  llvm::outs() << "qubits: " << resources->getNumQubits() << '\n';
  return 0;
}

// CHECK: x: 2
// CHECK: controlled r1: 1
// CHECK: phase: 0
// CHECK: qubits: 3
