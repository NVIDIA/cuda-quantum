/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                        *
 * All rights reserved.                                                       *
 *                                                                            *
 * This source code and the accompanying materials are made available under   *
 * the terms of the Apache License 2.0 which accompanies this distribution.   *
 ******************************************************************************/

// RUN: quake_reference_semantics_pipeline 2>&1 | FileCheck %s

// Tests for the built-in quake-reference-semantics lowering handler. Inferred
// pipelines are planned through the production built-in factory and stage
// composition (getPassPipeline); every pipeline produced is then actually run
// over the embedded kernel with cudaq-opt, so the checks inspect real lowered
// IR.

#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/LoweringHandler.h"
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

using HandlerVector =
    std::vector<std::unique_ptr<cudaq_internal::compiler::LoweringHandler>>;

// A kernel covering the profile options: a controlled X (needing basis
// decomposition when not native) and a veq measurement (expanded by
// expand-measurements).
static const char *inputKernel = R"MLIR(
module {
  func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %q = quake.alloca !quake.veq<2>
    %r0 = quake.extract_ref %q[0] : (!quake.veq<2>) -> !quake.ref
    %r1 = quake.extract_ref %q[1] : (!quake.veq<2>) -> !quake.ref
    quake.h %r0 : (!quake.ref) -> ()
    quake.t %r1 : (!quake.ref) -> ()
    quake.x [%r1] %r0 : (!quake.ref, !quake.ref) -> ()
    %meas = quake.mz %q : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
    return
  }
}
)MLIR";

// A kernel covering the reset-after-measure lowering: a measured qubit reused
// by a gate (reset inserted), a measured qubit measured again (no reset), a
// measured qubit reset by the kernel itself (no duplicate reset), and a
// terminal measurement (no reset).
static const char *reuseKernel = R"MLIR(
module {
  func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %qa = quake.alloca !quake.ref
    %qb = quake.alloca !quake.ref
    %qc = quake.alloca !quake.ref
    %qd = quake.alloca !quake.ref
    quake.h %qa : (!quake.ref) -> ()
    quake.t %qb : (!quake.ref) -> ()
    quake.h %qc : (!quake.ref) -> ()
    %m0 = quake.mz %qa : (!quake.ref) -> !quake.measure
    %d0 = quake.discriminate %m0 : (!quake.measure) -> i1
    quake.t %qa : (!quake.ref) -> ()
    %m1 = quake.mz %qb : (!quake.ref) -> !quake.measure
    %d1 = quake.discriminate %m1 : (!quake.measure) -> i1
    %m2 = quake.mz %qb : (!quake.ref) -> !quake.measure
    %d2 = quake.discriminate %m2 : (!quake.measure) -> i1
    %m3 = quake.mz %qc : (!quake.ref) -> !quake.measure
    %d3 = quake.discriminate %m3 : (!quake.measure) -> i1
    quake.reset %qc : (!quake.ref) -> ()
    quake.h %qc : (!quake.ref) -> ()
    %m4 = quake.mz %qd : (!quake.ref) -> !quake.measure
    %d4 = quake.discriminate %m4 : (!quake.measure) -> i1
    return
  }
}
)MLIR";

// A kernel reusing a qubit through extract_ref after a vector measurement:
// this could not be detected if we didn't force expand measurements whenever
// `reset-after-measure` is true.
static const char *veqReuseKernel = R"MLIR(
module {
  func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %q = quake.alloca !quake.veq<2>
    %r0 = quake.extract_ref %q[0] : (!quake.veq<2>) -> !quake.ref
    %r1 = quake.extract_ref %q[1] : (!quake.veq<2>) -> !quake.ref
    quake.h %r0 : (!quake.ref) -> ()
    quake.t %r1 : (!quake.ref) -> ()
    %m = quake.mz %q : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
    quake.h %r0 : (!quake.ref) -> ()
    return
  }
}
)MLIR";

// A kernel allocating inside a constant conditional. Test for leading
// canonicalize pass flattening the conditional.
static const char *conditionalAllocKernel = R"MLIR(
module {
  func.func @__nvqpp__mlirgen__kernel() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
    %true = arith.constant true
    cc.if (%true) {
      %q = quake.alloca !quake.ref
      quake.h %q : (!quake.ref) -> ()
      cc.continue
    }
    return
  }
}
)MLIR";

/// Pipelines planned so far, checked at the end for provider-named mappings.
static std::vector<std::string> plannedPipelines;

/// Plan the pipeline of \p target through the built-in lowering handlers.
static std::string planPipeline(const cudaq::CompileTarget &target,
                                const cudaq::CompileOptions &options) {
  HandlerVector handlers =
      cudaq_internal::compiler::makeBuiltinLoweringHandlers(
          options.emulate, target.codegen.emissionFormat);
  std::string diagnostic;
  auto pipeline = cudaq_internal::compiler::getPassPipeline(
      target, options, handlers, diagnostic);
  std::string result =
      pipeline ? std::move(*pipeline) : "FAILED: " + diagnostic;
  plannedPipelines.push_back(result);
  return result;
}

/// A target whose single stage lowers to \p profile by handler inference.
static cudaq::CompileTarget referenceSemanticsTarget(
    const cudaq::config::QuakeReferenceSemanticsProfile &profile,
    const char *emissionFormat) {
  cudaq::CompileTarget target;
  target.codegen.emissionFormat = emissionFormat;
  target.stages.push_back({.irOutputProfile = profile,
                           .lowering = cudaq::config::InferredLowering{}});
  return target;
}

/// Run \p pipeline over \p kernel with cudaq-opt; the lowered IR goes straight
/// to this process's stdout for FileCheck.
static void printLoweredIR(const char *kernel, const std::string &pipeline) {
  if (pipeline.starts_with("FAILED: "))
    return; // No pipeline was produced to lower with.
  std::cout.flush();
  std::string command =
      "cudaq-opt -pass-pipeline='builtin.module(" + pipeline + ")' 2>&1";
  FILE *stream = popen(command.c_str(), "w");
  if (!stream) {
    std::cout << "could not invoke cudaq-opt\n";
    return;
  }
  fwrite(kernel, 1, std::strlen(kernel), stream);
  if (pclose(stream) != 0)
    std::cout << "cudaq-opt pipeline failed\n";
  std::cout.flush();
}

// All gates of the embedded kernel are native to the full example basis, and
// `any` leaves the veq measurement untouched. The adaptive-QIR early-exit
// setting is captured from the codegen context.
void test_full_basis_any() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  std::cout << "case1: "
            << planPipeline(referenceSemanticsTarget(
                                profile, "qir-adaptive:0.1:int_computations"),
                            {})
            << std::endl;
  printLoweredIR(inputKernel, plannedPipelines.back());
}
// CHECK: case1: canonicalize,hw-jit-prep-pipeline{allow-early-exit=true},
// CHECK-SAME: jit-deploy-pipeline,jit-finalize-pipeline{lower-device-calls
// CHECK-SAME: =true},basis-conversion{basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK: quake.x [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK: quake.mz {{.*}} : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
// CHECK-NOT: cudaq-opt pipeline failed

// `single-qubit` runs expand-measurements: the veq measurement is replaced by
// one measurement per qubit.
void test_full_basis_expanded() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  profile.measurementType = cudaq::config::MeasurementForm::single_qubit;
  std::cout << "case2: "
            << planPipeline(referenceSemanticsTarget(profile, "qir-base"), {})
            << std::endl;
  printLoweredIR(inputKernel, plannedPipelines.back());
}
// CHECK: case2: canonicalize,hw-jit-prep-pipeline{allow-early-exit
// CHECK-SAME: =false},expand-measurements,jit-deploy-pipeline,jit-finalize-
// CHECK-SAME: pipeline{lower-device-calls=true},basis-conversion
// CHECK-SAME: {basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NOT: quake.mz {{.*}} : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
// CHECK: quake.mz {{.*}} : (!quake.ref) -> !quake.measure
// CHECK: quake.discriminate {{.*}} : (!quake.measure) -> i1
// CHECK-NOT: quake.mz {{.*}} : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
// CHECK-NOT: cudaq-opt pipeline failed

// Same input, restricted basis: the controlled X is decomposed into h/z(1).
void test_restricted_basis() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.basisGates = {"h", "t", "z(1)"};
  profile.measurementType = cudaq::config::MeasurementForm::single_qubit;
  std::cout << "case3: "
            << planPipeline(referenceSemanticsTarget(profile, "qir-base"), {})
            << std::endl;
  printLoweredIR(inputKernel, plannedPipelines.back());
}
// CHECK: case3: canonicalize,hw-jit-prep-pipeline{allow-early-exit
// CHECK-SAME: =false},expand-measurements,jit-deploy-pipeline,jit-finalize-
// CHECK-SAME: pipeline{lower-device-calls=true},basis-conversion
// CHECK-SAME: {basis=h,t,z(1)}
// CHECK-NOT: quake.x
// CHECK: quake.z [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK-NOT: cudaq-opt pipeline failed

// An empty basis imposes no gate-basis restriction: no basis-conversion in
// the pipeline. The nop codegen and emulation settings are captured; running
// the pipeline proves those emul/nop variants lower the kernel (loop
// preserving, gates untouched).
void test_unconstrained_basis() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.measurementType = cudaq::config::MeasurementForm::single_qubit;
  cudaq::CompileOptions options;
  options.emulate = true;
  std::cout << "case4: "
            << planPipeline(referenceSemanticsTarget(profile, "nop"), options)
            << std::endl;
  printLoweredIR(inputKernel, plannedPipelines.back());
}
// CHECK: case4: canonicalize,emul-jit-prep-pipeline{erase-noise=true
// CHECK-SAME: allow-early-exit=false no-loop-unroll=true},expand-measurements,
// CHECK-SAME: jit-deploy-pipeline{no-loop-unroll=true},jit-finalize-pipeline
// CHECK-SAME: {lower-device-calls=true}
// CHECK-NOT: basis-conversion
// CHECK: quake.x [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK: cc.loop while
// CHECK: quake.mz {{.*}} : (!quake.ref) -> !quake.measure
// CHECK-NOT: cudaq-opt pipeline failed

// A declared reference-semantics stage followed by an inferred
// reference-to-reference stage
// stage: the handler lowers the second stage from the first stage's output
// profile, changing the basis. The first stage's declared profile matches
// its actual `canonicalize` output (any-measurement, full basis), so the
// inferred stage truly starts from quake-reference-semantics input.
void test_reference_to_reference() {
  cudaq::config::QuakeReferenceSemanticsProfile full, restricted;
  full.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  restricted.basisGates = {"h", "t", "z(1)"};
  restricted.measurementType = cudaq::config::MeasurementForm::single_qubit;
  cudaq::CompileTarget target;
  target.codegen.emissionFormat = "qir-base";
  cudaq::config::CustomPipelineLowering custom;
  custom.overridePassPipeline = "canonicalize";
  target.stages.push_back({.irOutputProfile = full, .lowering = custom});
  target.stages.push_back({.irOutputProfile = restricted,
                           .lowering = cudaq::config::InferredLowering{}});
  std::cout << "case5: " << planPipeline(target, {}) << std::endl;
  printLoweredIR(inputKernel, plannedPipelines.back());
}
// CHECK: case5: canonicalize,canonicalize,hw-jit-prep-pipeline{allow-
// CHECK-SAME: early-exit=false},expand-measurements,jit-deploy-pipeline,
// CHECK-SAME: jit-finalize-pipeline{lower-device-calls=true},basis-
// CHECK-SAME: conversion{basis=h,t,z(1)}
// CHECK: quake.z [{{.*}}] {{.*}} : (!quake.ref, !quake.ref) -> ()
// CHECK-NOT: cudaq-opt pipeline failed

// Test `reset-after-measure=true`, which force expands measurements.
void test_reset_lowering() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  profile.resetAfterMeasure = true;
  std::string pipeline =
      planPipeline(referenceSemanticsTarget(profile, "qir-base"), {});
  std::cout << "case6: " << pipeline << std::endl;
  printLoweredIR(reuseKernel, pipeline);
}
// CHECK: case6: canonicalize,hw-jit-prep-pipeline{allow-early-exit
// CHECK-SAME: =false},expand-measurements,jit-deploy-pipeline,jit-finalize-
// CHECK-SAME: pipeline{lower-device-calls=true},func.func(combine-quantum-
// CHECK-SAME: alloc,qubit-reset-before-reuse),basis-conversion
// CHECK-SAME: {basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK: quake.alloca !quake.veq<4>
// CHECK: %[[M0:.*]] = quake.mz %[[QA:.*]] : (!quake.ref) -> !quake.measure
// CHECK-NEXT: quake.reset %[[QA]] : (!quake.ref) -> ()
// CHECK-NEXT: %[[B0:.*]] = quake.discriminate %[[M0]] : {{.*}} -> i1
// CHECK-NEXT: cc.if(%[[B0]]) {
// CHECK-NEXT: quake.x %[[QA]] : (!quake.ref) -> ()
// CHECK-NEXT: }
// CHECK-NEXT: quake.t %[[QA]] : (!quake.ref) -> ()
// CHECK: %[[M1:.*]] = quake.mz %[[QB:.*]] : (!quake.ref) -> !quake.measure
// CHECK-NEXT: %[[M2:.*]] = quake.mz %[[QB]] : {{.*}} !quake.measure
// CHECK-NOT: quake.reset
// CHECK: %[[M3:.*]] = quake.mz %[[QC:.*]] : (!quake.ref) -> !quake.measure
// CHECK-NEXT: quake.reset %[[QC]] : (!quake.ref) -> ()
// CHECK-NEXT: quake.h %[[QC]] : (!quake.ref) -> ()
// CHECK: %[[M4:.*]] = quake.mz %[[QD:.*]] : (!quake.ref) -> !quake.measure
// CHECK-NOT: quake.reset
// CHECK: quake.dealloc
// CHECK-NOT: cudaq-opt pipeline failed

// Same reset request under a basis without native X: the conditional X the
// reset pass inserts is decomposed into phased_rx by the final
// basis-conversion, so no quake.x remains in the lowered IR.
void test_reset_no_x_basis() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.basisGates = {"h", "t", "phased_rx"};
  profile.resetAfterMeasure = true;
  std::string pipeline =
      planPipeline(referenceSemanticsTarget(profile, "qir-base"), {});
  std::cout << "case7: " << pipeline << std::endl;
  printLoweredIR(reuseKernel, pipeline);
}
// CHECK: case7: canonicalize,hw-jit-prep-pipeline{allow-early-exit
// CHECK-SAME: =false},expand-measurements,jit-deploy-pipeline,jit-finalize-
// CHECK-SAME: pipeline{lower-device-calls=true},func.func(combine-quantum-
// CHECK-SAME: alloc,qubit-reset-before-reuse),basis-conversion
// CHECK-SAME: {basis=h,t,phased_rx}
// CHECK: quake.reset {{.*}} : (!quake.ref) -> ()
// CHECK: quake.phased_rx
// CHECK-NOT: quake.x
// CHECK-NOT: cudaq-opt pipeline failed

// A reset request under the default `any` measurement form expands the
// vector measurement, so the measured-and-reused qubit is protected even
// though the kernel measured it through the register, not the extract.
void test_reset_any_veq_reuse() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  profile.basisGates = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"};
  profile.resetAfterMeasure = true;
  std::string pipeline =
      planPipeline(referenceSemanticsTarget(profile, "qir-base"), {});
  std::cout << "case8: " << pipeline << std::endl;
  printLoweredIR(veqReuseKernel, pipeline);
}
// CHECK: case8: canonicalize,hw-jit-prep-pipeline{allow-early-exit
// CHECK-SAME: =false},expand-measurements,jit-deploy-pipeline,jit-finalize-
// CHECK-SAME: pipeline{lower-device-calls=true},func.func(combine-quantum-
// CHECK-SAME: alloc,qubit-reset-before-reuse),basis-conversion
// CHECK-SAME: {basis=h,s,t,rx,ry,rz,x,y,z,x(1)}
// CHECK-NOT: quake.mz {{.*}} !cc.sequence<!quake.measure>
// CHECK: %[[M0:.*]] = quake.mz %[[R0:.*]] : (!quake.ref) -> !quake.measure
// CHECK-NEXT: quake.reset %[[R0]] : (!quake.ref) -> ()
// CHECK-NEXT: %[[B0:.*]] = quake.discriminate %[[M0]] : {{.*}} -> i1
// CHECK-NEXT: cc.if(%[[B0]]) {
// CHECK-NEXT: quake.x %[[R0]] : (!quake.ref) -> ()
// CHECK-NEXT: }
// CHECK: quake.h %[[R0]] : (!quake.ref) -> ()
// CHECK-NOT: quake.reset
// CHECK: quake.dealloc
// CHECK-NOT: cudaq-opt pipeline failed

// The inferred pipeline's leading canonicalize is what makes prep's
// implicit-output/dealloc bookkeeping reach a conditionally allocated qubit:
// it flattens the constant cc.if first, so the allocation is top-level when
// the bookkeeping passes (which run before prep's own canonicalizer) look.
void test_conditional_alloc_bookkeeping() {
  cudaq::config::QuakeReferenceSemanticsProfile profile;
  std::string pipeline =
      planPipeline(referenceSemanticsTarget(profile, "qir-base"), {});
  std::cout << "case9: " << pipeline << std::endl;
  printLoweredIR(conditionalAllocKernel, pipeline);
}
// CHECK: case9: canonicalize,hw-jit-prep-pipeline{allow-early-exit=false},
// CHECK-SAME: jit-deploy-pipeline,jit-finalize-pipeline{lower-device-calls
// CHECK-SAME: =true}
// CHECK: quake.alloca !quake.ref
// CHECK: quake.evince
// CHECK: quake.dealloc
// CHECK-NOT: cudaq-opt pipeline failed

// No inferred pipeline contains provider-named gate-set mappings.
void test_no_provider_mappings() {
  for (const auto &pipeline : plannedPipelines)
    for (const char *mapping :
         {"ionq-gate-set-mapping", "fermioniq-gate-set-mapping",
          "quantinuum-gate-set-mapping"})
      if (pipeline.find(mapping) != std::string::npos) {
        std::cout << "unexpected provider mapping in: " << pipeline << "\n";
        return;
      }
  std::cout << "case10: no provider-named mappings\n";
}
// CHECK: case10: no provider-named mappings

int main() {
  test_full_basis_any();
  test_full_basis_expanded();
  test_restricted_basis();
  test_unconstrained_basis();
  test_reference_to_reference();
  test_reset_lowering();
  test_reset_no_x_basis();
  test_reset_any_veq_reuse();
  test_conditional_alloc_bookkeeping();
  test_no_provider_mappings();
  return 0;
}
