/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       *
 * All rights reserved.                                                      *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// RUN: profile_providers | FileCheck %s

// Test that new inferred pipelines match the expected pipelines.

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

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Target/TargetCatalog.h"
#include <iostream>
#include <string>

/// Plan the JIT pipeline of the actual catalog entry through the built-in
/// lowering handlers, as the production JIT does.
static std::string
planPipeline(const cudaq::config::detail::TargetEntry &entry) {
  auto target = cudaq::CompileTarget::createFromConfig(*entry.config, {});
  auto handlers = cudaq_internal::compiler::makeBuiltinLoweringHandlers(
      false, target.codegen.emissionFormat);
  std::string diagnostic;
  auto pipeline = cudaq_internal::compiler::getPassPipeline(
      target, {}, handlers, diagnostic);
  return pipeline ? *pipeline : "FAILED: " + diagnostic;
}

static void printPipeline(const cudaq::config::TargetCatalog &catalog,
                          const char *name) {
  const auto *entry = catalog.lookup(name);
  const std::string pipeline =
      entry ? planPipeline(*entry) : "not found in target catalog";
  std::cout << name << ":\n";
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
  return 0;
}
