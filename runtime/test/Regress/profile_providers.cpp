/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                       *
 * All rights reserved.                                                      *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// RUN: profile_providers %S/../../../python/tests/targets | FileCheck %s

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

// The FTQC targets list one `quake-unmapped` stage with the `t-count` cost
// model. The logical target's basis keeps rotations, so it runs no synthesis.
// CHECK-LABEL: compiler-bench-ftqc-logical level 2:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: decomposition{basis=h,s,t,rx,ry,rz,x,y,z,swap,x(1),y(1),z(1),x(2),z(2),r1}
// CHECK-NEXT: prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT: clifford-t-optimization-pipeline{level=2 epsilon=0}
// CHECK-NEXT: basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,swap,x(1),y(1),z(1),x(2),z(2)}
// CHECK-NEXT: verify-profile-quake-unmapped{basis=h,s,t,rx,ry,rz,x,y,z,swap,x(1),y(1),z(1),x(2),z(2)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// The Clifford+T target's basis has no rotation gate, so every level
// synthesizes. Level 0 stays on references and runs no optimization.
// CHECK-LABEL: compiler-bench-ftqc-clifford-t level 0:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: decomposition{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK-NEXT: clifford-t-synthesis-pipeline{epsilon=1e-10 seed=0}
// CHECK-NEXT: basis-conversion{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT: verify-profile-quake-unmapped{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// CHECK-LABEL: compiler-bench-ftqc-clifford-t level 1:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: decomposition{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK-NEXT: prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT: clifford-t-optimization-pipeline{level=1 epsilon=1e-10}
// CHECK-NEXT: clifford-t-synthesis-pipeline{epsilon=1e-10 seed=0}
// CHECK-NEXT: basis-conversion{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT: verify-profile-quake-unmapped{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// CHECK-LABEL: compiler-bench-ftqc-clifford-t level 2:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: decomposition{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK-NEXT: prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT: clifford-t-optimization-pipeline{level=2 epsilon=1e-10}
// CHECK-NEXT: clifford-t-synthesis-pipeline{epsilon=1e-10 seed=0}
// CHECK-NEXT: func.func(quake-simplify{threshold=1e-10},normalize-1q-clifford-t,phase-folding,normalize-phase-placement,lower-phase,canonicalize,dqe)
// CHECK-NEXT: basis-conversion{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT: verify-profile-quake-unmapped{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// Without a cost model, a `quake-unmapped` stage runs no optimization, even at
// level 2.
// CHECK-LABEL: no cost model level 2:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false}
// CHECK-NEXT: jit-deploy-pipeline
// CHECK-NEXT: basis-conversion{basis=h,x(1)}
// CHECK-NEXT: verify-profile-quake-unmapped{basis=h,x(1)}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:
// clang-format on

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Target/TargetCatalog.h"
#include <filesystem>
#include <iostream>
#include <string>

/// Plan the JIT pipeline of \p target through the built-in lowering handlers,
/// as the production JIT does, and print one pass per line under \p label.
static void printPipeline(const std::string &label,
                          const cudaq::CompileTarget &target,
                          const cudaq::CompileOptions &options) {
  auto handlers = cudaq_internal::compiler::makeBuiltinLoweringHandlers();
  std::string diagnostic;
  auto planned = cudaq_internal::compiler::getPassPipeline(
      target, options, handlers, diagnostic);
  const std::string pipeline = planned ? *planned : "FAILED: " + diagnostic;
  std::cout << label << ":\n";
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

/// Print the pipeline of the catalog entry \p name.
static void printPipeline(const cudaq::config::TargetCatalog &catalog,
                          const std::string &name,
                          const cudaq::CompileOptions &options = {},
                          const std::string &labelSuffix = "") {
  const auto *entry = catalog.lookup(name);
  if (!entry) {
    std::cout << name << ": not found in target catalog\n\n";
    return;
  }
  printPipeline(name + labelSuffix,
                cudaq::CompileTarget::createFromConfig(*entry->config, {}),
                options);
}

static cudaq::CompileOptions atLevel(cudaq::OptimizationLevel level) {
  cudaq::CompileOptions options;
  options.optimizationLevel = level;
  return options;
}

/// The FTQC benchmark targets live with the Python tests, in the directory
/// given as the first argument. Loading them from there keeps the test
/// independent of whether the Python project, which adds the Clifford+T target
/// to the built-in catalog, is built.
int main(int argc, char **argv) {
  cudaq::config::TargetCatalog catalog;
  if (argc > 1)
    for (const char *name :
         {"compiler-bench-ftqc-logical", "compiler-bench-ftqc-clifford-t"})
      catalog.addTargetConfigFile(std::filesystem::path(argv[1]) /
                                  (std::string(name) + ".yml"));

  printPipeline(catalog, "ionq");
  printPipeline(catalog, "fermioniq");
  printPipeline(catalog, "quantinuum");
  printPipeline(catalog, "quantum_machines");
  cudaq::CompileOptions emulated;
  emulated.emulate = true;
  printPipeline(catalog, "quantum_machines", emulated, " emulated");

  printPipeline(catalog, "compiler-bench-ftqc-logical",
                atLevel(cudaq::OptimizationLevel::O2), " level 2");
  for (auto level : {cudaq::OptimizationLevel::O0, cudaq::OptimizationLevel::O1,
                     cudaq::OptimizationLevel::O2})
    printPipeline(catalog, "compiler-bench-ftqc-clifford-t", atLevel(level),
                  " level " + std::to_string(static_cast<unsigned>(level)));

  cudaq::CompileTarget noCostModel;
  noCostModel.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{.basisGates = {"h", "x(1)"}},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("no cost model level 2", noCostModel,
                atLevel(cudaq::OptimizationLevel::O2));
  return 0;
}
