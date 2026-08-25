/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: test_resource_count_wireset | FileCheck %s

#include "cudaq_internal/compiler/ResourceCount.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

// Output of `add-wireset,func.func(assign-wire-indices)`. The topology-agnostic
// wire set declares a cardinality of INT_MAX, so the qubit count has to come
// from the wires that are actually borrowed.
static constexpr char addWiresetKernel[] = R"#(
quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}
func.func @k() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
  %0 = quake.borrow_wire @wires[0] : !quake.wire
  %1 = quake.borrow_wire @wires[1] : !quake.wire
  %2 = quake.borrow_wire @wires[2] : !quake.wire
  %3 = quake.h %0 : (!quake.wire) -> !quake.wire
  %4:2 = quake.x [%3] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
  %5:2 = quake.x [%4#1] %2 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
  quake.return_wire %4#0 : !quake.wire
  quake.return_wire %5#0 : !quake.wire
  quake.return_wire %5#1 : !quake.wire
  return
}
)#";

// Output of the same pipeline followed by `qubit-mapping{device=path(5)}`.
// Mapping adds a second wire set sized to the device, and the topology-agnostic
// one it supersedes is still in the module. Neither cardinality describes how
// many qubits this kernel uses.
static constexpr char mappedKernel[] = R"#(
quake.wire_set @mapped_wireset[5] adjacency sparse<[[0, 1], [1, 2], [1, 0], [2, 3], [2, 1], [3, 4], [3, 2], [4, 3]], true> : tensor<5x5xi1> attributes {sym_visibility = "private"}
quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}
func.func @k() attributes {"cudaq-entrypoint", "cudaq-kernel", mapping_reorder_idx = [0, 1, 2], mapping_v2p = [0, 1, 2]} {
  %0 = quake.borrow_wire @mapped_wireset[0] : !quake.wire
  %1 = quake.borrow_wire @mapped_wireset[1] : !quake.wire
  %2 = quake.borrow_wire @mapped_wireset[2] : !quake.wire
  %3 = quake.h %0 : (!quake.wire) -> !quake.wire
  %4:2 = quake.x [%3] %1 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
  %5:2 = quake.x [%4#1] %2 : (!quake.wire, !quake.wire) -> (!quake.wire, !quake.wire)
  quake.return_wire %4#0 : !quake.wire
  quake.return_wire %5#0 : !quake.wire
  quake.return_wire %5#1 : !quake.wire
  return
}
)#";

// A wire returned to the set may be borrowed again under the same identity.
// That is one qubit reused, not two qubits.
static constexpr char reusedIdentityKernel[] = R"#(
quake.wire_set @wires[2147483647] attributes {sym_visibility = "private"}
func.func @k() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
  %0 = quake.borrow_wire @wires[0] : !quake.wire
  %1 = quake.h %0 : (!quake.wire) -> !quake.wire
  quake.return_wire %1 : !quake.wire
  %2 = quake.borrow_wire @wires[0] : !quake.wire
  %3 = quake.h %2 : (!quake.wire) -> !quake.wire
  quake.return_wire %3 : !quake.wire
  return
}
)#";

static int report(const char *label, const char *source,
                  mlir::MLIRContext *context) {
  auto module = mlir::parseSourceString<mlir::ModuleOp>(source, context);
  if (!module)
    return 1;
  auto resources = cudaq::opt::countResourcesFromIR(*module);
  if (failed(resources))
    return 1;
  llvm::outs() << label << " qubits: " << resources->getNumQubits() << '\n';
  llvm::outs() << label << " h: " << resources->count("h") << '\n';
  llvm::outs() << label
               << " controlled x: " << resources->count_controls("x", 1)
               << '\n';
  return 0;
}

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();
  if (report("add-wireset", addWiresetKernel, context.get()))
    return 1;
  if (report("mapped", mappedKernel, context.get()))
    return 1;
  if (report("reused", reusedIdentityKernel, context.get()))
    return 1;
  return 0;
}

// CHECK: add-wireset qubits: 3
// CHECK: add-wireset h: 1
// CHECK: add-wireset controlled x: 2
// CHECK: mapped qubits: 3
// CHECK: mapped h: 1
// CHECK: mapped controlled x: 2
// CHECK: reused qubits: 1
// CHECK: reused h: 2
// CHECK: reused controlled x: 0
