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

// clang-format off
// CHECK-LABEL: ionq:
// CHECK-NEXT:  jit options: default
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:  post-deploy:
// CHECK-NEXT:    basis-conversion{basis={h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-EMPTY:

// CHECK-LABEL: fermioniq:
// CHECK-NEXT:  jit options: default
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:    expand-measurements
// CHECK-NEXT:  post-deploy:
// CHECK-NEXT:    basis-conversion{basis={h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-EMPTY:

// CHECK-LABEL: quantinuum:
// CHECK-NEXT:  jit options: prep.allow-early-exit=true
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:    expand-measurements
// CHECK-NEXT:  post-deploy:
// CHECK-NEXT:    basis-conversion{basis={h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-NEXT:    func.func(combine-quantum-alloc,qubit-reset-before-reuse)
// CHECK-EMPTY:

// `quantum_machines` sets high-, mid-, and low-level pipelines and uses `nop`
// codegen, which keeps prep and deploy from unrolling loops and, without
// emulation, keeps finalize from lowering device calls.
// CHECK-LABEL: quantum_machines:
// CHECK-NEXT:  jit options: prep.no-loop-unroll=true deploy.no-loop-unroll=true finalize.lower-device-calls=false
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:    func.func(add-measurements{use-evince=false})
// CHECK-NEXT:  post-deploy:
// CHECK:         cc-loop-unroll{{.*}} unroll-only-aliasing-quantum-access-loops=true
// CHECK:         add-wireset
// CHECK:         decomposition{basis={h,s,t,r1,rx,ry,rz,x,y,z,z(1),x(1)}
// CHECK-NEXT:    qubit-mapping-prep{device=bypass
// CHECK-NEXT:    func.func(qubit-mapping-func{{.*}} device=bypass
// CHECK-NEXT:    decomposition{basis={h,s,t,r1,rx,ry,rz,x,y,z,z(1),x(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-NEXT:    symbol-dce
// CHECK-EMPTY:

// CHECK-LABEL: quantum_machines emulated:
// CHECK-NEXT:  jit options: prep.no-loop-unroll=true deploy.no-loop-unroll=true{{$}}
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:    func.func(add-measurements{use-evince=false})

// The Clifford+T target lists one `quake-unmapped` stage with the `t-count`
// cost model, whose passes run between deploy and finalize. It decomposes,
// converts to wires, optimizes, fuses one-qubit chains, and synthesizes,
// since its basis has no rotation gate. Synthesis handles the rotations, so
// the fusion allows them as basis gates.
// CHECK-LABEL: compiler-bench-ftqc-clifford-t:
// CHECK-NEXT:  jit options: default
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:  post-deploy:
// CHECK-NEXT:    decomposition{basis={h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz}
// CHECK:         cc-loop-unroll{{.*}} unroll-only-index-use-loops=false
// CHECK:         phase-folding
// CHECK:         func.func(fuse-1q-to-u3{basis={h,s,t,x,y,z,swap,x(1),y(1),z(1),r1,rx,ry,rz} cost=clifford-t-free epsilon=1.000000e-10}
// CHECK:         clifford-t-synthesis{epsilon=1.000000e-10
// CHECK:         basis-conversion{basis={h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-EMPTY:

// The level never changes the representation: at level 0 the stage still
// converts to wires and synthesizes, and only the optimizations are left out.
// CHECK-LABEL: compiler-bench-ftqc-clifford-t level 0:
// CHECK:         cc-loop-unroll
// CHECK-NOT:     phase-folding
// CHECK-NOT:     fuse-1q-to-u3
// CHECK:         clifford-t-synthesis
// CHECK:         basis-conversion{basis={h,s,t,x,y,z,swap,x(1),y(1),z(1)}
// CHECK-NEXT:  post-finalize:

// A `t-count` stage whose basis has `u3` needs no synthesis. The basis has no
// rotation, so the fusion turns every chain with a rotation into a `u3`,
// which the basis keeps. Level 1 keeps the fusion and drops phase folding.
// CHECK-LABEL: t-count to u3 level 2:
// CHECK:         phase-folding
// CHECK:         func.func(fuse-1q-to-u3{basis={u3,h,x(1)} cost=clifford-t-free epsilon=1.000000e-10}
// CHECK-NEXT:    basis-conversion{basis={u3,h,x(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-EMPTY:

// CHECK-LABEL: t-count to u3 level 1:
// CHECK-NOT:     phase-folding
// CHECK:         func.func(fuse-1q-to-u3{basis={u3,h,x(1)} cost=clifford-t-free epsilon=1.000000e-10}
// CHECK-NEXT:    basis-conversion{basis={u3,h,x(1)}

// CHECK-LABEL: t-count to u3 level 0:
// CHECK-NOT:     fuse-1q-to-u3
// CHECK-NOT:     clifford-t-synthesis
// CHECK:         basis-conversion{basis={u3,h,x(1)}
// CHECK-NEXT:  post-finalize:

// A `two-qubit-gate-count` stage whose basis can't express a controlled `x`
// fails when its handler builds the pipeline.
// CHECK-LABEL: no two-qubit gate:
// CHECK-NEXT:  FAILED: compile target stage 1: the selected lowering handler
// CHECK-SAME:  failed to construct a pipeline from unchecked to quake-unmapped

// The NISQ target decomposes and optimizes in its `quake-unmapped` stage,
// keeping controlled `x` for phase folding until its basis conversion, and
// maps in its `quake-mapped` stage. Both stages convert to wires at every
// level, and level 1 adds only the two optimization pipelines and the fusion
// of one-qubit chains.
// CHECK-LABEL: compiler-bench-nisq on path(4) level 0:
// CHECK-NEXT:  jit options: default
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:  post-deploy:
// CHECK-NEXT:    decomposition{basis={h,rx,ry,rz,x,z(1),x(1)}
// CHECK:         cc-loop-unroll
// CHECK-NOT:     quake-simplify
// CHECK:         basis-conversion{basis={h,rx,ry,rz,x,z(1)}
// CHECK:         cc-loop-unroll
// CHECK:         add-wireset
// CHECK:         qubit-mapping-prep{device=path(4) raise-fatal-errors=true}
// CHECK-NEXT:    func.func(qubit-mapping-func{{.*}} device=path(4)
// CHECK-NEXT:    basis-conversion{basis={h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-EMPTY:

// CHECK-LABEL: compiler-bench-nisq on path(4) level 1:
// CHECK-NEXT:  jit options: default
// CHECK-NEXT:  pre-deploy:
// CHECK-NEXT:  post-deploy:
// CHECK-NEXT:    decomposition{basis={h,rx,ry,rz,x,z(1),x(1)}
// CHECK:         cc-loop-unroll
// CHECK:         quake-simplify
// CHECK:         func.func(fuse-1q-to-u3{basis={h,rx,ry,rz,x,z(1)} cost=rotation-count
// CHECK-NEXT:    basis-conversion{basis={h,rx,ry,rz,x,z(1)}
// CHECK:         cc-loop-unroll
// CHECK:         add-wireset
// CHECK:         qubit-mapping-prep{device=path(4) raise-fatal-errors=true}
// CHECK-NEXT:    func.func(qubit-mapping-func{{.*}} device=path(4)
// CHECK-NEXT:    func.func({{.*}}quake-simplify
// CHECK-NEXT:    basis-conversion{basis={h,rx,ry,rz,x,z(1)}
// CHECK-NEXT:  post-finalize:
// CHECK-EMPTY:

// A target that preserves loops keeps prep and deploy from unrolling, so its
// stage decides which loops to unroll. A target that does not support noise
// rejects it in prep, and one with QEC erasure off keeps QEC through prep.
// CHECK-LABEL: loops preserved, noise unsupported, QEC kept:
// CHECK-NEXT:  jit options: prep.erase-noise=false prep.reject-noise=true prep.erase-qec=false prep.no-loop-unroll=true deploy.no-loop-unroll=true{{$}}
// CHECK:         cc-loop-unroll{{.*}} unroll-only-index-use-loops=true

// clang-format on

#include "common/CompileOptions.h"
#include "cudaq_internal/compiler/JITTargetPipeline.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include "cudaq/Optimizer/InitAllPasses.h"
#include "cudaq/Target/TargetCatalog.h"
#include "llvm/Support/raw_ostream.h"
#include <filesystem>
#include <iostream>
#include <string>

/// The textual form of \p pm without its `builtin.module(...)` anchor.
static std::string pipelineText(const mlir::OpPassManager &pm) {
  std::string text;
  llvm::raw_string_ostream os(text);
  pm.printAsTextualPipeline(os);
  const std::string anchor = "builtin.module(";
  if (text.starts_with(anchor) && text.ends_with(")"))
    text = text.substr(anchor.size(), text.size() - anchor.size() - 1);
  return text;
}

/// Print \p name, then the top-level passes of \p pm, one per line.
static void printSlot(const std::string &name, const mlir::OpPassManager &pm) {
  std::cout << name << ":\n";
  const auto text = pipelineText(pm);
  if (text.empty())
    return;
  std::cout << "  ";
  unsigned depth = 0;
  for (char c : text) {
    if (c == '{' || c == '(')
      ++depth;
    else if ((c == '}' || c == ')') && depth > 0)
      --depth;
    if (c == ',' && depth == 0)
      std::cout << "\n  ";
    else
      std::cout << c;
  }
  std::cout << '\n';
}

/// The options of prep, deploy, and finalize that differ from their defaults,
/// or `default` if none do.
static std::string
jitOptionsText(const cudaq_internal::compiler::JITPipelineLayout &layout) {
  const cudaq::opt::JITPrepPipelineOptions prep;
  const cudaq::opt::JITDeployPipelineOptions deploy;
  const cudaq::opt::JITFinalizePipelineOptions finalize;
  std::string text;
  const auto add = [&](const char *name, bool value, bool defaultValue) {
    if (value != defaultValue)
      text += std::string(" ") + name + "=" + (value ? "true" : "false");
  };
  add("prep.erase-noise", layout.prep.eraseNoise, prep.eraseNoise);
  add("prep.reject-noise", layout.prep.rejectNoise, prep.rejectNoise);
  add("prep.erase-qec", layout.prep.eraseQEC, prep.eraseQEC);
  add("prep.const-prop", layout.prep.applyConstProp, prep.applyConstProp);
  add("prep.allow-early-exit", layout.prep.allowEarlyExit, prep.allowEarlyExit);
  add("prep.no-loop-unroll", layout.prep.disableLoopUnrolling,
      prep.disableLoopUnrolling);
  add("deploy.no-loop-unroll", layout.deploy.disableLoopUnrolling,
      deploy.disableLoopUnrolling);
  add("finalize.lower-device-calls", layout.finalize.lowerDeviceCalls,
      finalize.lowerDeviceCalls);
  return text.empty() ? " default" : text;
}

/// Plan the JIT pipeline of \p target through the built-in lowering handlers,
/// as the production JIT does, and print under \p label the options of prep,
/// deploy, and finalize and the passes the stages add around them.
static void printPipeline(const std::string &label,
                          const cudaq::CompileTarget &target,
                          const cudaq::CompileOptions &options) {
  auto handlers = cudaq_internal::compiler::makeBuiltinLoweringHandlers();
  std::string diagnostic;
  auto layout = cudaq_internal::compiler::getPipelineLayout(
      target, options, handlers, diagnostic);
  std::cout << label << ":\n";
  if (!layout) {
    std::cout << "FAILED: " << diagnostic << "\n\n";
    return;
  }
  std::cout << "jit options:" << jitOptionsText(*layout) << '\n';
  using cudaq_internal::compiler::StagePasses;
  for (auto [name, place] :
       {std::pair{"pre-deploy", &StagePasses::preDeployPasses},
        {"post-deploy", &StagePasses::postDeployPasses},
        {"post-finalize", &StagePasses::postFinalizePasses}}) {
    mlir::OpPassManager pm("builtin.module");
    layout->addStagePasses(pm, place);
    printSlot(name, pm);
  }
  std::cout << '\n';
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
  cudaq::registerAllPasses();
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
  cudaq::CompileTarget tCountToU3;
  tCountToU3.stages.push_back(
      {.irOutputProfile =
           cudaq::config::QuakeUnmappedProfile{
               .basisGates = {"u3", "h", "x(1)"},
               .approximationTolerance = 1e-10,
               .costModel = cudaq::config::CostModel::t_count},
       .lowering = cudaq::config::InferredLowering{}});
  printPipeline("t-count to u3 level 2", tCountToU3, level2);
  cudaq::CompileOptions level1;
  level1.optimizationLevel = cudaq::OptimizationLevel::O1;
  printPipeline("t-count to u3 level 1", tCountToU3, level1);
  printPipeline("t-count to u3 level 0", tCountToU3, level0);
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
