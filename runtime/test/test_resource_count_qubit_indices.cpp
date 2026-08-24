/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// RUN: test_resource_count_qubit_indices | FileCheck %s

#include "cudaq_internal/compiler/ResourceCount.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Parser/Parser.h"

// A control of type `!quake.control` threads no wire through the operation.
// The qubit it holds is reached through the `to_ctrl`/`from_ctrl` pair, which
// convert form without changing the qubit.
static constexpr char ctrlFormKernel[] = R"#(
func.func @k() attributes {"cudaq-entrypoint", "cudaq-kernel"} {
  %w0 = quake.null_wire
  %w1 = quake.null_wire
  %h = quake.h %w0 : (!quake.wire) -> !quake.wire
  %c = quake.to_ctrl %h : (!quake.wire) -> !quake.control
  %w1a = quake.x [%c] %w1 : (!quake.control, !quake.wire) -> !quake.wire
  %w0b = quake.from_ctrl %c : (!quake.control) -> !quake.wire
  %h2 = quake.h %w0b : (!quake.wire) -> !quake.wire
  quake.sink %h2 : !quake.wire
  quake.sink %w1a : !quake.wire
  return
}
)#";

// The control is a dynamic offset into a qvector, so its qubit index does not
// resolve. The gate is left in the IR to be counted
// at run time rather than attributed to the wrong qubit.
static constexpr char unresolvedIndexKernel[] = R"#(
func.func @k(%i: i64) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
  %0 = quake.alloca !quake.veq<2>
  %1 = quake.extract_ref %0[%i] : (!quake.veq<2>, i64) -> !quake.ref
  %2 = quake.extract_ref %0[1] : (!quake.veq<2>) -> !quake.ref
  quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
  return
}
)#";

// The control is a qvector of unknown size, so not even the number of controls
// is known. The gate is left in the IR to be counted at run time.
static constexpr char unknownArityKernel[] = R"#(
func.func @k(%v: !quake.veq<?>) attributes {"cudaq-entrypoint", "cudaq-kernel"} {
  %0 = quake.alloca !quake.ref
  quake.x [%v] %0 : (!quake.veq<?>, !quake.ref) -> ()
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
  llvm::outs() << label << " gates: " << resources->count() << '\n';
  llvm::outs() << label << " h: " << resources->count("h") << '\n';
  llvm::outs() << label
               << " controlled x: " << resources->count_controls("x", 1)
               << '\n';
  llvm::outs() << label
               << " uncontrolled x: " << resources->count_controls("x", 0)
               << '\n';
  llvm::outs() << label
               << " arity 1 gates: " << resources->getGateCountByArity(1)
               << '\n';
  llvm::outs() << label
               << " arity 2 gates: " << resources->getGateCountByArity(2)
               << '\n';
  llvm::outs() << label << " depth: " << resources->getCircuitDepth() << '\n';
  llvm::outs() << label << " arity 2 depth: " << resources->getDepthByArity(2)
               << '\n';
  llvm::outs() << label << " remaining IR:\n" << *module << '\n';
  return 0;
}

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();
  if (report("ctrl-form", ctrlFormKernel, context.get()))
    return 1;
  if (report("unresolved", unresolvedIndexKernel, context.get()))
    return 1;
  if (report("unknown-arity", unknownArityKernel, context.get()))
    return 1;
  return 0;
}

// The `x` sees the control qubit through `to_ctrl`, and the second `h` sees it
// again through `from_ctrl`, so both land on qubit 0 and the depth is 3.
// CHECK: ctrl-form gates: 3
// CHECK: ctrl-form h: 2
// CHECK: ctrl-form controlled x: 1
// CHECK: ctrl-form uncontrolled x: 0
// CHECK: ctrl-form arity 1 gates: 2
// CHECK: ctrl-form arity 2 gates: 1
// CHECK: ctrl-form depth: 3
// CHECK: ctrl-form arity 2 depth: 1
// CHECK: ctrl-form remaining IR:
// CHECK-NOT: quake.h
// CHECK-NOT: quake.x

// Not counted here; the operation survives for run time counting.
// CHECK: unresolved gates: 0
// CHECK: unresolved controlled x: 0
// CHECK: unresolved uncontrolled x: 0
// CHECK: unresolved remaining IR:
// CHECK: quake.x [

// Not counted here either: the number of controls is not even known.
// CHECK: unknown-arity gates: 0
// CHECK: unknown-arity controlled x: 0
// CHECK: unknown-arity uncontrolled x: 0
// CHECK: unknown-arity remaining IR:
// CHECK: quake.x [
