/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       *
 * All rights reserved.                                                      *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// RUN: profile_providers | FileCheck %s

// Test that new inferred pipelines match the expected pipelines.

// CHECK: ionq: match
// CHECK: fermioniq: match
// CHECK: quantinuum: match

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Target/TargetCatalog.h"
#include <iostream>
#include <map>
#include <string>

/// The expected inferred pipelines. IonQ's measurements stay `any`
/// (no expand-measurements); Fermioniq requires the expanded form;
/// Quantinuum expands measurements, resets qubits before reuse, and keeps
/// the early-exit path of its adaptive-QIR emission enabled. Each opens with
/// the leading canonicalize, as the custom-pipeline lowering path does.
static const char *expectedIonqPipeline =
    "canonicalize,"
    "hw-jit-prep-pipeline{allow-early-exit=false},"
    "jit-deploy-pipeline,jit-finalize-pipeline{lower-device-calls=true},"
    "basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}";
static const char *expectedFermioniqPipeline =
    "canonicalize,"
    "hw-jit-prep-pipeline{allow-early-exit=false},"
    "expand-measurements,jit-deploy-pipeline,"
    "jit-finalize-pipeline{lower-device-calls=true},"
    "basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}";
static const char *expectedQuantinuumPipeline =
    "canonicalize,"
    "hw-jit-prep-pipeline{allow-early-exit=true},"
    "expand-measurements,jit-deploy-pipeline,"
    "jit-finalize-pipeline{lower-device-calls=true},"
    "func.func(combine-quantum-alloc,qubit-reset-before-reuse),"
    "basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}";

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

static void checkTarget(const cudaq::config::TargetCatalog &catalog,
                        const char *name, const std::string &expected) {
  const auto *entry = catalog.lookup(name);
  const std::string planned =
      entry ? planPipeline(*entry) : "not found in target catalog";
  const bool match = planned == expected;
  std::cout << name << ": " << (match ? "match" : "MISMATCH") << "\n";
  if (!match)
    std::cout << "  planned: " << planned << "\n"
              << "  expected: " << expected << "\n";
}

int main() {
  cudaq::config::TargetCatalog catalog;
  checkTarget(catalog, "ionq", expectedIonqPipeline);
  checkTarget(catalog, "fermioniq", expectedFermioniqPipeline);
  checkTarget(catalog, "quantinuum", expectedQuantinuumPipeline);
  return 0;
}
