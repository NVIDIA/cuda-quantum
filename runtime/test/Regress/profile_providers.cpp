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

// A simulator's YAML declares a `gate-count` stage. Prep keeps noise and QEC
// operations and leaves the gates that hardware lacks to the simulator, deploy
// does not run, and prep and finalize skip what the stage repeats. The stage
// converts to wires at level 0, with `exp_pauli` left on references because the
// stage has no basis, and the implicit output kept for `get_state`. From level
// 1 it runs `quake-simplify` and `dqe`.
// CHECK-LABEL: qpp-cpu level 0:
// CHECK-NEXT: canonicalize
// CHECK-NEXT: hw-jit-prep-pipeline{allow-early-exit=false erase-noise=false erase-qec=false lower-gates=false normalize=false}
// CHECK-NEXT: prepare-for-wireset{maximum-iterations=50 decompose-exp-pauli=false erase-implicit-output=false}
// CHECK-NEXT: jit-finalize-pipeline{lower-device-calls=true normalize=false}
// CHECK-EMPTY:

// CHECK-LABEL: qpp-cpu level 1:
// CHECK:       prepare-for-wireset{maximum-iterations=50 decompose-exp-pauli=false erase-implicit-output=false}
// CHECK-NEXT:  func.func(quake-simplify{threshold=0},dqe)
// CHECK-NEXT:  jit-finalize-pipeline{lower-device-calls=true normalize=false}

// `nvidia` declares its simulators in a configuration matrix, and every entry
// shares the target's stage.
// CHECK-LABEL: nvidia level 1:
// CHECK:       func.func(quake-simplify{threshold=0},dqe)

// The Clifford+T target lists one `quake-unmapped` stage with the `t-count`
// cost model, whose passes run between deploy and finalize. Its basis has no
// rotation gate, so the stage synthesizes. The stage lowers the `quake.phase`
// that decomposition leaves before the optimizer runs. Synthesis handles the
// rotations, so the fusion allows them as basis gates.
// CHECK-LABEL: compiler-bench-ftqc-clifford-t:
// CHECK:       jit-deploy-pipeline
// CHECK-NEXT:  decomposition{basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK-NEXT:  prepare-for-wireset{maximum-iterations=1024}
// CHECK-NEXT:  func.func(normalize-phase-placement,lower-phase)
// CHECK-NEXT:  clifford-t-optimization-pipeline{level=2 approximation-tolerance=1e-10}
// CHECK-NEXT:  func.func(fuse-1q-to-u3{cost=clifford-t-free epsilon=1e-10 basis=h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz},canonicalize)
// CHECK-NEXT:  decomposition{enable-patterns=U3ToRotations}
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

// A `t-count` stage whose basis has `u3` needs no synthesis, so it does not
// expand the `u3` again. The basis has no rotation, so the fusion turns every
// chain with a rotation into a `u3`, which the basis keeps.
// CHECK-LABEL: t-count to u3 level 2:
// CHECK:       clifford-t-optimization-pipeline{level=2 approximation-tolerance=1e-10}
// CHECK-NEXT:  func.func(fuse-1q-to-u3{cost=clifford-t-free epsilon=1e-10 basis=u3,h,x(1)},canonicalize)
// CHECK-NEXT:  basis-conversion{basis=u3,h,x(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// CHECK-LABEL: t-count to u3 level 1:
// CHECK:       clifford-t-optimization-pipeline{level=1 approximation-tolerance=1e-10}
// CHECK-NEXT:  func.func(fuse-1q-to-u3{cost=clifford-t-free epsilon=1e-10 basis=u3,h,x(1)},canonicalize)

// CHECK-LABEL: t-count to u3 level 0:
// CHECK-NOT:   fuse-1q-to-u3
// CHECK:       jit-finalize-pipeline

// Without a cost model or basis, the stage needs no gate lowering, so deploy
// does not run, and it adds no pass, so prep and finalize skip nothing.
// CHECK-LABEL: no cost model or basis:
// CHECK-NEXT:  canonicalize
// CHECK-NEXT:  hw-jit-prep-pipeline{allow-early-exit=false lower-gates=false}
// CHECK-NEXT:  jit-finalize-pipeline{lower-device-calls=true}
// CHECK-EMPTY:

// A `two-qubit-gate-count` stage whose basis can't express a controlled `x`
// fails when its handler builds the pipeline.
// CHECK-LABEL: no two-qubit gate:
// CHECK-NEXT:  FAILED: compile target stage 1: the selected lowering handler
// CHECK-SAME:  failed to construct a pipeline from unchecked to quake-unmapped

// A stage whose basis can't express a `u3` plans no single-qubit fusion, and
// warns instead.
// CHECK:       Single-qubit gate fusion skipped: no known decomposition lowers u3 to the basis gates [h, s, t, x, x(1)].
// CHECK-LABEL: no u3 decomposition level 1:
// CHECK:       two-qubit-gate-count-optimization-pipeline{level=1}
// CHECK-NEXT:  basis-conversion{basis=h,s,t,x,x(1)}

// The NISQ target decomposes and optimizes in its `quake-unmapped` stage,
// keeping controlled `x` for phase folding until its basis conversion, and
// maps in its `quake-mapped` stage. Both stages convert to wires at every
// level, and level 1 adds only the two optimization pipelines and the fusion
// of one-qubit chains.
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
// CHECK-NEXT:  func.func(fuse-1q-to-u3{cost=gate-count basis=h,rx,ry,rz,x,z(1)},canonicalize)
// CHECK-NEXT:  basis-conversion{basis=h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  prepare-for-wireset{add-wireset=true maximum-iterations=1024}
// CHECK-NEXT:  qubit-mapping{device=path(4) raise-fatal-errors=true}
// CHECK-NEXT:  two-qubit-gate-count-optimization-pipeline{level=1}
// CHECK-NEXT:  basis-conversion{basis=h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  jit-finalize-pipeline

// A target that preserves loops keeps prep and deploy from unrolling, so its
// stage decides which loops to unroll. A target that does not support noise
// rejects it in prep, and one with QEC erasure off keeps QEC through prep.
// CHECK-LABEL: loops preserved, noise unsupported, QEC kept:
// CHECK-NEXT:  canonicalize
// CHECK-NEXT:  hw-jit-prep-pipeline{allow-early-exit=false no-loop-unroll=true erase-noise=false reject-noise=true erase-qec=false}
// CHECK-NEXT:  jit-deploy-pipeline{no-loop-unroll=true}
// CHECK:       prepare-for-wireset{maximum-iterations=1024 unroll-only-index-use-loops=true}

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
  const std::string pipeline =
      planned ? planned->text : "FAILED: " + diagnostic;
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

  cudaq::CompileOptions level0;
  level0.optimizationLevel = cudaq::OptimizationLevel::O0;
  printPipeline(catalog, "qpp-cpu", level0, " level 0");
  cudaq::CompileOptions level1;
  level1.optimizationLevel = cudaq::OptimizationLevel::O1;
  printPipeline(catalog, "qpp-cpu", level1, " level 1");
  printPipeline(catalog, "nvidia", level1, " level 1");

  cudaq::CompileOptions level2;
  level2.optimizationLevel = cudaq::OptimizationLevel::O2;
  printPipeline(catalog, "compiler-bench-ftqc-clifford-t", level2);
  printPipeline(catalog, "compiler-bench-ftqc-clifford-t", level0, " level 0");
  cudaq::CompileTarget tCountToU3;
  tCountToU3.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{
               .basisGates = {"u3", "h", "x(1)"},
               .approximationTolerance = 1e-10,
               .costModel = cudaq::config::CostModel::t_count},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("t-count to u3 level 2", tCountToU3, level2);
  printPipeline("t-count to u3 level 1", tCountToU3, level1);
  printPipeline("t-count to u3 level 0", tCountToU3, level0);
  cudaq::CompileTarget noCostModelOrBasis;
  noCostModelOrBasis.stages.push_back(
      {.irOutputProfile = cudaq::config::QuakeUnmappedProfile{},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("no cost model or basis", noCostModelOrBasis, level2);
  cudaq::CompileTarget noTwoQubitGate;
  noTwoQubitGate.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{
               .basisGates = {"h", "rz"},
               .costModel = cudaq::config::CostModel::two_qubit_gate_count},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("no two-qubit gate", noTwoQubitGate, {});
  cudaq::CompileTarget noU3Decomposition;
  noU3Decomposition.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{
               .basisGates = {"h", "s", "t", "x", "x(1)"},
               .costModel = cudaq::config::CostModel::two_qubit_gate_count},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("no u3 decomposition level 1", noU3Decomposition, level1);

  if (const auto *nisq = catalog.lookup("compiler-bench-nisq")) {
    auto onPath = cudaq::CompileTarget::createFromConfig(*nisq->config, {});
    std::get<cudaq::config::QuakeMappedProfile>(
        onPath.stages.back().irOutputProfile)
        .device = "path(4)";
    printPipeline("compiler-bench-nisq on path(4) level 0", onPath, level0);
    printPipeline("compiler-bench-nisq on path(4) level 1", onPath, level1);
  }

  cudaq::CompileTarget target;
  target.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{
               .basisGates = {"h", "rz", "x(1)"},
               .costModel = cudaq::config::CostModel::t_count,
               .unrollLoops = cudaq::config::LoopUnrolling::index_use},
       .lowering = cudaq::config::InferredLowering{}});
  target.preserveLoops = true;
  target.noiseSupport = cudaq::NoiseSupport::unsupported;
  target.eraseQEC = false;
  printPipeline("loops preserved, noise unsupported, QEC kept", target, {});
  return 0;
}
