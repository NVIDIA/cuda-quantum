/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       *
 * All rights reserved.                                                      *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// RUN: profile_providers | FileCheck %s

// Test that the pipelines planned for catalog targets match the expected
// pipelines.

// CHECK-LABEL: ionq:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// CHECK-LABEL: fermioniq:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: expand-measurements
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// CHECK-LABEL: quantinuum:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=true}
// CHECK-NEXT: expand-measurements
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-NEXT: func.func(combine-quantum-alloc,qubit-reset-before-reuse)
// CHECK-EMPTY:

// `quantum_machines` sets high-, mid-, and low-level pipelines and uses `nop`
// codegen, which adds `no-loop-unroll=true` to prep and deploy and, without
// emulation, `lower-device-calls=false` to finalize.
// clang-format off
// CHECK-LABEL: quantum_machines:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false no-loop-unroll=true}
// CHECK-NEXT: func.func(add-measurements)
// CHECK-NEXT: jit-deploy-pipeline{no-loop-unroll=true}
// CHECK-NEXT: prepare-for-wireset{add-wireset=true unroll-only-aliasing-quantum-access-loops=true}
// CHECK-NEXT: decomposition{basis=h,s,t,r1,rx,ry,rz,x,y,z,z(1),x(1)}
// CHECK-NEXT: qubit-mapping{device=bypass}
// CHECK-NEXT: decomposition{basis=h,s,t,r1,rx,ry,rz,x,y,z,z(1),x(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=false}
// CHECK-NEXT: symbol-dce
// CHECK-EMPTY:

// CHECK-LABEL: quantum_machines emulated:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: emul-jit-prep-pipeline{erase-noise=true allow-early-exit=false no-loop-unroll=true}
// CHECK-NEXT: func.func(add-measurements)
// CHECK-NEXT: jit-deploy-pipeline{no-loop-unroll=true}
// CHECK-NEXT: prepare-for-wireset{add-wireset=true unroll-only-aliasing-quantum-access-loops=true}
// CHECK-NEXT: decomposition{basis=h,s,t,r1,rx,ry,rz,x,y,z,z(1),x(1)}
// CHECK-NEXT: qubit-mapping{device=bypass}
// CHECK-NEXT: decomposition{basis=h,s,t,r1,rx,ry,rz,x,y,z,z(1),x(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-NEXT: symbol-dce
// CHECK-EMPTY:
// clang-format on

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Target/TargetCatalog.h"
#include <iostream>
#include <string>

/// Plan the JIT pipeline of the actual catalog entry through the built-in
/// lowering handlers, as the production JIT does.
static std::string planPipeline(const cudaq::config::detail::TargetEntry &entry,
                                const cudaq::CompileOptions &options) {
  auto target = cudaq::CompileTarget::createFromConfig(*entry.config, {});
  auto handlers = cudaq_internal::compiler::makeBuiltinLoweringHandlers();
  std::string diagnostic;
  auto pipeline = cudaq_internal::compiler::getPassPipeline(
      target, options, handlers, diagnostic);
  return pipeline ? *pipeline : "FAILED: " + diagnostic;
}

static void printPipeline(const cudaq::config::TargetCatalog &catalog,
                          const char *name, bool emulate = false) {
  const auto *entry = catalog.lookup(name);
  cudaq::CompileOptions options;
  options.emulate = emulate;
  const std::string pipeline =
      entry ? planPipeline(*entry, options) : "not found in target catalog";
  std::cout << name << (emulate ? " emulated" : "") << ":\n";
  unsigned depth = 0;
  for (char c : pipeline) {
    if (c == '{' || c == '(')
      ++depth;
    else if ((c == '}' || c == ')') && depth > 0)
      --depth;
    if (c == ',' && depth == 0)
      std::cout << '\n';
    else
      std::cout << c;
  }
  std::cout << "\n\n";
}

int main() {
  cudaq::config::TargetCatalog catalog;
  printPipeline(catalog, "ionq");
  printPipeline(catalog, "fermioniq");
  printPipeline(catalog, "quantinuum");
  printPipeline(catalog, "quantum_machines");
  printPipeline(catalog, "quantum_machines", /*emulate=*/true);
  return 0;
}
