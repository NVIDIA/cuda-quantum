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

// The Clifford+T target lists one `quake-unmapped` stage with the `t-count`
// cost model, whose passes run between deploy and finalize. Its basis has no
// rotation gate, so the stage synthesizes. The stage lowers the `quake.phase`
// that decomposition leaves before the optimizer runs.
// CHECK-LABEL: compiler-bench-ftqc-clifford-t:
// CHECK:       jit-deploy-pipeline
// CHECK-NEXT:  decomposition{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK-NEXT:  prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT:  func.func(normalize-phase-placement,lower-phase)
// CHECK-NEXT:  clifford-t-optimization-pipeline{level=2 approximation-tolerance=1e-10}
// CHECK-NEXT:  clifford-t-synthesis-pipeline{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1) level=2 approximation-tolerance=1e-10}
// CHECK-NEXT:  basis-conversion{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// The level never changes the representation: at level 0 the stage still
// converts to wires, and only the optimization pipeline is left out.
// CHECK-LABEL: compiler-bench-ftqc-clifford-t level 0:
// CHECK:       jit-deploy-pipeline
// CHECK-NEXT:  decomposition{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK-NEXT:  prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT:  func.func(normalize-phase-placement,lower-phase)
// CHECK-NEXT:  clifford-t-synthesis-pipeline{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1) level=0 approximation-tolerance=1e-10}
// CHECK-NEXT:  basis-conversion{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// Without a cost model, a `quake-unmapped` stage runs no optimization.
// CHECK-LABEL: no cost model:
// CHECK:       jit-deploy-pipeline
// CHECK-NEXT:  basis-conversion{basis=h,x(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// A `two-qubit-gate-count` stage whose basis can't express a controlled `x`
// fails when its handler builds the pipeline.
// CHECK-LABEL: no two-qubit gate:
// CHECK-NEXT:  FAILED: compile target stage 1: the selected lowering handler
// CHECK-SAME:  failed to construct a pipeline from unchecked to quake-unmapped

// The NISQ target decomposes and optimizes in its `quake-unmapped` stage,
// keeping controlled `x` for phase folding until its basis conversion, and
// maps in its `quake-mapped` stage. Both stages convert to wires at every
// level, and level 1 adds only the two optimization pipelines.
// CHECK-LABEL: compiler-bench-nisq on path(4) level 0:
// CHECK:       jit-deploy-pipeline
// CHECK-NEXT:  decomposition{basis=h,rx,ry,rz,x,z(1),x(1)}
// CHECK-NEXT:  prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT:  basis-conversion{basis=h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  prepare-for-wireset{add-wireset=true maximum-iterations=1024}
// CHECK-NEXT:  qubit-mapping{device=path(4) raise-fatal-errors=true}
// CHECK-NEXT:  basis-conversion{basis=h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// CHECK-LABEL: compiler-bench-nisq on path(4) level 1:
// CHECK:       jit-deploy-pipeline
// CHECK-NEXT:  decomposition{basis=h,rx,ry,rz,x,z(1),x(1)}
// CHECK-NEXT:  prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT:  two-qubit-gate-count-optimization-pipeline{level=1}
// CHECK-NEXT:  basis-conversion{basis=h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  prepare-for-wireset{add-wireset=true maximum-iterations=1024}
// CHECK-NEXT:  qubit-mapping{device=path(4) raise-fatal-errors=true}
// CHECK-NEXT:  two-qubit-gate-count-optimization-pipeline{level=1}
// CHECK-NEXT:  basis-conversion{basis=h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// A stage that unrolls only index-use loops keeps prep and deploy from
// unrolling, and a target with erasure off keeps noise and QEC through prep.
// CHECK-LABEL: index-use, erasure off:
// CHECK-NEXT:  canonicalize
// CHECK-NEXT:  hw-jit-prep-pipeline{allow-early-exit=false no-loop-unroll=true erase-noise=false erase-qec=false}
// CHECK-NEXT:  jit-deploy-pipeline{no-loop-unroll=true}
// CHECK:       prepare-for-wireset{maximum-iterations=1024 unroll-only-index-use-loops=true}

// A stage whose unroll limit is below prep's 1024 iterations keeps prep and
// deploy from unrolling, even when it unrolls all loops.
// CHECK-LABEL: unroll limit 2:
// CHECK-NEXT:  canonicalize
// CHECK-NEXT:  hw-jit-prep-pipeline{allow-early-exit=false no-loop-unroll=true}
// CHECK-NEXT:  jit-deploy-pipeline{no-loop-unroll=true}
// CHECK:       prepare-for-wireset{maximum-iterations=2}
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

int main(int argc, char **argv) {
  cudaq::config::TargetCatalog catalog;
  // Only a build with the Python project adds the Clifford+T target to the
  // built-in catalog, and the NISQ target is never in it, so load both from
  // the YAML directory given as an argument.
  if (argc > 1) {
    catalog.addTargetConfigFile(std::filesystem::path(argv[1]) /
                                "compiler-bench-ftqc-clifford-t.yml");
    catalog.addTargetConfigFile(std::filesystem::path(argv[1]) /
                                "compiler-bench-nisq.yml");
  }

  printPipeline(catalog, "ionq");
  printPipeline(catalog, "fermioniq");
  printPipeline(catalog, "quantinuum");
  printPipeline(catalog, "quantum_machines");
  cudaq::CompileOptions emulated;
  emulated.emulate = true;
  printPipeline(catalog, "quantum_machines", emulated, " emulated");

  cudaq::CompileOptions level2;
  level2.optimizationLevel = cudaq::OptimizationLevel::O2;
  printPipeline(catalog, "compiler-bench-ftqc-clifford-t", level2);
  cudaq::CompileOptions level0;
  level0.optimizationLevel = cudaq::OptimizationLevel::O0;
  printPipeline(catalog, "compiler-bench-ftqc-clifford-t", level0, " level 0");
  cudaq::CompileTarget noCostModel;
  noCostModel.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{.basisGates = {"h", "x(1)"}},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("no cost model", noCostModel, level2);
  cudaq::CompileTarget noTwoQubitGate;
  noTwoQubitGate.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{
               .basisGates = {"h", "rz"},
               .costModel = cudaq::config::CostModel::two_qubit_gate_count},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("no two-qubit gate", noTwoQubitGate, {});

  if (const auto *nisq = catalog.lookup("compiler-bench-nisq")) {
    auto onPath = cudaq::CompileTarget::createFromConfig(*nisq->config, {});
    std::get<cudaq::config::QuakeMappedProfile>(
        onPath.stages.back().irOutputProfile)
        .device = "path(4)";
    printPipeline("compiler-bench-nisq on path(4) level 0", onPath, level0);
    printPipeline("compiler-bench-nisq on path(4) level 1", onPath, {});
  }

  const auto unmappedTarget = [](cudaq::config::QuakeUnmappedProfile profile) {
    cudaq::CompileTarget target;
    target.stages.push_back({.irOutputProfile = std::move(profile),
                             .lowering = cudaq::config::InferredLowering{}});
    return target;
  };
  auto indexUse =
      unmappedTarget({.basisGates = {"h", "rz", "x(1)"},
                      .costModel = cudaq::config::CostModel::t_count,
                      .unrollLoops = cudaq::config::LoopUnrolling::index_use});
  indexUse.eraseNoise = false;
  indexUse.eraseQEC = false;
  printPipeline("index-use, erasure off", indexUse, {});
  printPipeline("unroll limit 2",
                unmappedTarget({.basisGates = {"h", "rz", "x(1)"},
                                .costModel = cudaq::config::CostModel::t_count,
                                .maxUnrollIterations = 2}),
                {});
  return 0;
}
