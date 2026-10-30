/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This test is compiled inside the runtime directory tree. We include it as a
// regression test and use FileCheck to verify the output.

// RUN: compile_target | FileCheck %s

#include "common/KernelArgs.h"
#include "common/KernelExecution.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
#include "cudaq/Optimizer/Builder/OptimizationLevel.h"
#include "cudaq/Target/CompileTarget.h"
#include "cudaq/Target/TargetConfig.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include <cstdint>
#include <memory>
#include <span>

using namespace cudaq_internal::compiler;

/// Compile \p quake and print the resulting MLIR module to stdout.
static void compileAndDump(mlir::MLIRContext *ctx,
                           const std::string &kernelName,
                           const std::string &quake,
                           cudaq::CompileTarget target,
                           cudaq::CompileOptions options = {},
                           cudaq::KernelArgs args = {}) {
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(quake, ctx);
  if (!mod) {
    llvm::outs() << "FAILED TO PARSE\n";
    return;
  }

  Compiler compiler(std::move(target), std::move(options));
  auto compiled = compiler.compile(
      kernelName, mod.release().getAsOpaquePointer(), args, /*isEntryPoint=*/
      true);

  auto mlirArtifact = compiled.getMlir();
  if (!mlirArtifact) {
    llvm::outs() << "NO MLIR ARTIFACT\n";
    return;
  }
  auto moduleOp = CompiledModuleHelper::getMlirModuleOp(*mlirArtifact);
  llvm::outs() << "Compiled module:\n" << moduleOp << "\n";
}

/// Compile \p quake and print selected resource metrics in a stable order
/// under \p heading.
static void compileAndDumpResources(mlir::MLIRContext *ctx,
                                    const std::string &heading,
                                    const std::string &kernelName,
                                    const std::string &quake,
                                    cudaq::CompileTarget target,
                                    cudaq::CompileOptions options) {
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(quake, ctx);
  if (!mod) {
    llvm::outs() << "FAILED TO PARSE\n";
    return;
  }

  Compiler compiler(std::move(target), std::move(options));
  auto compiled = compiler.compile(
      kernelName, mod.release().getAsOpaquePointer(), cudaq::KernelArgs{},
      /*isEntryPoint=*/true);

  auto *resources = compiled.getResources();
  if (!resources) {
    llvm::outs() << "NO RESOURCE ARTIFACT\n";
    return;
  }

  llvm::outs() << heading << " resource counts:\n"
               << "phase: " << resources->count("phase") << "\n"
               << "rz: " << resources->count("rz") << "\n"
               << "rz(0): " << resources->count_controls("rz", 0) << "\n"
               << "rz(1): " << resources->count_controls("rz", 1) << "\n"
               << "r1: " << resources->count("r1") << "\n"
               << "r1(0): " << resources->count_controls("r1", 0) << "\n"
               << "x: " << resources->count("x") << "\n"
               << "total: " << resources->count() << "\n"
               << "qubits: " << resources->getNumQubits() << "\n"
               << "used qubits: " << resources->getNumUsedQubits() << "\n"
               << "1Q gates: " << resources->getGateCountByArity(1) << "\n"
               << "2Q gates: " << resources->getGateCountByArity(2) << "\n";
}

/// Build a CompileTarget with no backend configuration
static cudaq::CompileTarget noBackendTarget() {
  cudaq::config::TargetConfig cfg;
  return cudaq::CompileTarget::createFromConfig(cfg, {});
}

/// Build a CompileTarget with a backend configuration with an empty pass
/// pipeline
static cudaq::CompileTarget emptyPipelineTarget() {
  cudaq::config::TargetConfig cfg;
  cfg.BackendConfig = cudaq::config::BackendEndConfigEntry{};
  return cudaq::CompileTarget::createFromConfig(cfg, {});
}

/// Build a CompileTarget with a backend configuration with a non-empty pass
/// pipeline
static cudaq::CompileTarget
nonEmptyPipelineTarget(const std::string &pipeline) {
  cudaq::config::TargetConfig cfg;
  cudaq::config::BackendEndConfigEntry backend;
  cudaq::config::CustomPipelineLowering lowering;
  lowering.overridePassPipeline = pipeline;
  backend.CompileTarget.stages.push_back(cudaq::config::CompilationStage{
      .irOutputProfile = cudaq::config::UncheckedProfile{},
      .lowering = std::move(lowering)});
  cfg.BackendConfig = backend;
  return cudaq::CompileTarget::createFromConfig(cfg, {});
}

/// Kernel with statically-sized qubit register
static const char *quantumKernel = R"#(
func.func @__nvqpp__mlirgen__foo() -> i64 attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %0 = quake.alloca !quake.veq<2>
  %1 = quake.veq_size %0 : (!quake.veq<2>) -> i64
  return %1 : i64
}
)#";

/// Kernel with a device call
static const char *deviceCallKernel = R"#(
func.func @__nvqpp__mlirgen__devKernel(%arg0: i64) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  cc.device_call @consume(%arg0) : (i64) -> ()
  return
}
func.func private @consume(i64) attributes {"cudaq-devicecall"}
)#";

/// R1ToRz emits an Rz and a Phase correction for each R1. The first Phase is
/// global and must disappear. The negative-controlled Phase must become an R1,
/// with the negative control expanded into surrounding X gates.
static const char *phaseResourceKernel = R"#(
func.func @__nvqpp__mlirgen__phaseResources() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %angle = arith.constant 5.000000e-01 : f64
  %q0 = quake.alloca !quake.ref
  %q1 = quake.alloca !quake.ref
  %q2 = quake.alloca !quake.ref
  quake.r1 (%angle) %q0 : (f64, !quake.ref) -> ()
  quake.r1 (%angle) [%q1 neg [true]] %q2 : (f64, !quake.ref, !quake.ref) -> ()
  return
}
)#";

void test_device_calls_supported(mlir::MLIRContext *ctx) {
  std::int64_t arg = 42;
  std::vector<void *> v = {static_cast<void *>(&arg)};
  cudaq::KernelArgs args(std::span<void *const>(v.data(), v.size()));

  auto target = cudaq::CompileTarget{};
  target.supportDeviceCalls = true;
  // Isolate the device-call lowering performed during argument synthesis from
  // the rest of the target lowering pipeline.
  cudaq::CompileOptions options;
  options.skipTargetLoweringPipeline = true;
  compileAndDump(ctx, "devKernel", deviceCallKernel, std::move(target),
                 std::move(options), args);
}

// clang-format off
// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__devKernel()
// CHECK:           %[[VAL_0:.*]] = arith.constant 42 : i64
// CHECK:           call @consume(%[[VAL_0]]) : (i64) -> ()
// CHECK-NOT:       cc.device_call
// CHECK:         func.func private @consume(i64) attributes {{.*}}cudaq-devicecall
// clang-format on

void test_device_calls_unsupported(mlir::MLIRContext *ctx) {
  std::int64_t arg = 42;
  std::vector<void *> v = {static_cast<void *>(&arg)};
  cudaq::KernelArgs args(std::span<void *const>(v.data(), v.size()));

  auto target = cudaq::CompileTarget{};
  target.supportDeviceCalls = false;
  cudaq::CompileOptions options;
  options.skipTargetLoweringPipeline = true;
  compileAndDump(ctx, "devKernel", deviceCallKernel, std::move(target),
                 std::move(options), args);
}

// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__devKernel()
// CHECK:           cc.device_call @consume

void test_no_backend_config(mlir::MLIRContext *ctx) {
  compileAndDump(ctx, "foo", quantumKernel, noBackendTarget());
}

// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__foo()
// CHECK:           quake.veq_size
// CHECK:           return

void test_empty_pipeline(mlir::MLIRContext *ctx) {
  compileAndDump(ctx, "foo", quantumKernel, emptyPipelineTarget());
}

// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__foo()
// CHECK:           quake.veq_size
// CHECK:           return

void test_non_empty_pipeline(mlir::MLIRContext *ctx) {
  compileAndDump(ctx, "foo", quantumKernel,
                 nonEmptyPipelineTarget("canonicalize"));
}

// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__foo()
// CHECK-NOT:       quake.veq_size
// CHECK:           %[[VAL_0:.*]] = arith.constant 2 : i64
// CHECK:           return %[[VAL_0]] : i64

void test_resource_count_completes_phase_lifecycle(mlir::MLIRContext *ctx) {
  auto target = nonEmptyPipelineTarget("decomposition{enable-patterns=R1ToRz}");

  cudaq::CompileOptions options;
  options.emitResourceCounts = true;
  options.emitTargetCode = false;
  compileAndDumpResources(ctx, "Phase", "phaseResources", phaseResourceKernel,
                          std::move(target), std::move(options));
}

// CHECK-LABEL: Phase resource counts:
// CHECK-NEXT: phase: 0
// CHECK-NEXT: rz: 2
// CHECK-NEXT: rz(0): 1
// CHECK-NEXT: rz(1): 1
// CHECK-NEXT: r1: 1
// CHECK-NEXT: r1(0): 1
// CHECK-NEXT: x: 4
// CHECK-NEXT: total: 7
// CHECK-NEXT: qubits: 3
// CHECK-NEXT: used qubits: 3
// CHECK-NEXT: 1Q gates: 6
// CHECK-NEXT: 2Q gates: 1

void test_resource_count_disabled_by_target(mlir::MLIRContext *ctx) {
  auto target = nonEmptyPipelineTarget("decomposition{enable-patterns=R1ToRz}");
  target.supportResourceCounts = false;

  cudaq::CompileOptions options;
  options.emitResourceCounts = true;
  options.emitTargetCode = false;
  compileAndDumpResources(ctx, "Phase", "phaseResources", phaseResourceKernel,
                          std::move(target), std::move(options));
}

// CHECK-LABEL: NO RESOURCE ARTIFACT

/// Handler lowering into quake-reference-semantics with a fixed marker pass
/// before deploy.
class FakeQuakeReferenceSemanticsHandler : public LoweringHandler {
public:
  bool supports(const cudaq::config::Profile &,
                const cudaq::config::Profile &output) const override {
    return std::holds_alternative<
        cudaq::config::QuakeReferenceSemanticsProfile>(output);
  }

  std::optional<StagePasses>
  getStagePasses(const cudaq::config::Profile &,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const override {
    if (!std::holds_alternative<cudaq::config::QuakeReferenceSemanticsProfile>(
            output))
      return std::nullopt;
    return StagePasses{.preDeployPasses = "func.func(add-measurements)"};
  }
};

/// Count of runPassManager invocations while the counting hook is installed.
static int targetPipelineRuns = 0;

static mlir::LogicalResult countTargetPipelineRun(mlir::PassManager &pm,
                                                  mlir::Operation *op) {
  ++targetPipelineRuns;
  return pm.run(op);
}

/// Target with a single inferred quake-reference-semantics stage.
static cudaq::CompileTarget inferredTarget() {
  cudaq::CompileTarget target;
  target.stages.push_back(
      {.irOutputProfile = cudaq::config::QuakeReferenceSemanticsProfile{},
       .lowering = cudaq::config::InferredLowering{}});
  return target;
}

void test_staged_inferred(mlir::MLIRContext *ctx) {
  std::vector<std::unique_ptr<LoweringHandler>> handlers;
  handlers.push_back(std::make_unique<FakeQuakeReferenceSemanticsHandler>());

  targetPipelineRuns = 0;
  cudaq_internal::compiler::setRunPassManagerHook(&countTargetPipelineRun);

  Compiler compiler(inferredTarget(), {}, std::move(handlers));
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(phaseResourceKernel, ctx);
  auto compiled = compiler.compile(
      "phaseResources", mod.release().getAsOpaquePointer(), cudaq::KernelArgs{},
      /*isEntryPoint=*/true);

  cudaq_internal::compiler::setRunPassManagerHook(nullptr);

  auto mlirArtifact = compiled.getMlir();
  auto moduleOp = CompiledModuleHelper::getMlirModuleOp(*mlirArtifact);
  llvm::outs() << "Staged module:\n" << moduleOp << "\n";
  llvm::outs() << "target pipeline runs: " << targetPipelineRuns << "\n";
}

// CHECK-LABEL: Staged module:
// CHECK: func.func @__nvqpp__mlirgen__phaseResources()
// CHECK: quake.r1
// CHECK: quake.mz
// CHECK: target pipeline runs: 1

/// A staged target with skipTargetLoweringPipeline neither runs nor plans any
/// target lowering.
void test_staged_skip(mlir::MLIRContext *ctx) {
  cudaq::CompileOptions options;
  options.skipTargetLoweringPipeline = true;
  Compiler compiler(inferredTarget(), std::move(options), {});
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(phaseResourceKernel, ctx);
  auto compiled = compiler.compile(
      "phaseResources", mod.release().getAsOpaquePointer(), cudaq::KernelArgs{},
      /*isEntryPoint=*/true);

  auto mlirArtifact = compiled.getMlir();
  auto moduleOp = CompiledModuleHelper::getMlirModuleOp(*mlirArtifact);
  llvm::outs() << "Skipped staged module:\n" << moduleOp << "\n";
}

// CHECK-LABEL: Skipped staged module:
// CHECK: quake.r1
// CHECK-NOT: quake.mz

/// Four one-qubit rotations that `fuse-1q-chains-for-clifford-t` rewrites as
/// three.
static const char *fusableRotationsKernel = R"#(
func.func @__nvqpp__mlirgen__fusable() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %a = arith.constant 3.000000e-01 : f64
  %b = arith.constant 5.000000e-01 : f64
  %c = arith.constant 7.000000e-01 : f64
  %d = arith.constant 2.000000e-01 : f64
  %q = quake.alloca !quake.ref
  quake.rz (%a) %q : (f64, !quake.ref) -> ()
  quake.rx (%b) %q : (f64, !quake.ref) -> ()
  quake.rz (%c) %q : (f64, !quake.ref) -> ()
  quake.rx (%d) %q : (f64, !quake.ref) -> ()
  return
}
)#";

/// `exp_pauli(pi/8, Z)` lowers to `rz(-pi/4)`, an exact angle that synthesis
/// rewrites as one `t<adj>` rather than a gridsynth sequence.
static const char *expPauliKernel = R"#(
func.func @__nvqpp__mlirgen__expPauli() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %angle = arith.constant 0.39269908169872414 : f64
  %q = quake.alloca !quake.veq<1>
  quake.exp_pauli (%angle) %q to "Z" : (f64, !quake.veq<1>) -> ()
  return
}
)#";

/// A loop longer than any unroll limit in the quake-unmapped stage.
static const char *longLoopKernel = R"#(
func.func @__nvqpp__mlirgen__longLoop() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c4096 = arith.constant 4096 : i64
  %q = quake.alloca !quake.ref
  %0 = cc.loop while ((%i = %c0) -> (i64)) {
    %1 = arith.cmpi slt, %i, %c4096 : i64
    cc.condition %1(%i : i64)
  } do {
  ^bb0(%i: i64):
    quake.h %q : (!quake.ref) -> ()
    cc.continue %i : i64
  } step {
  ^bb0(%i: i64):
    %2 = arith.addi %i, %c1 : i64
    cc.continue %2 : i64
  }
  return
}
)#";

/// A `quake-unmapped` stage fuses the rotations only from level 2, and a loop
/// it cannot unroll stays rolled and is still counted. Without a rotation in
/// the basis, every level synthesizes, and the strict basis conversion rejects
/// any rotation left over.
void test_quake_unmapped_levels(mlir::MLIRContext *ctx) {
  using cudaq::OptimizationLevel;
  const auto count = [&](const std::string &heading, const char *kernelName,
                         const char *kernel, std::vector<std::string> basis,
                         OptimizationLevel level) {
    cudaq::CompileTarget target;
    target.stages.push_back(
        {.irOutputProfile =
             cudaq::config::QuakeUnmappedProfile{
                 .basisGates = std::move(basis),
                 .approximationTolerance = 1e-3,
                 .costModel = cudaq::config::CostModel::t_count},
         .lowering = cudaq::config::InferredLowering{}});
    cudaq::CompileOptions options;
    options.emitResourceCounts = true;
    options.emitTargetCode = false;
    options.optimizationLevel = level;
    options.seed = 42;
    compileAndDumpResources(ctx, heading, kernelName, kernel, target,
                            std::move(options));
  };
  const std::vector<std::string> rotations = {"h",  "s",  "t", "rx",
                                              "ry", "rz", "x", "x(1)"};
  const std::vector<std::string> cliffordT = {
      "h", "s", "t", "x", "y", "z", "swap", "x(1)", "y(1)", "z(1)"};
  count("Rotations level 0", "fusable", fusableRotationsKernel, rotations,
        OptimizationLevel::O0);
  count("Rotations level 1", "fusable", fusableRotationsKernel, rotations,
        OptimizationLevel::O1);
  count("Rotations level 2", "fusable", fusableRotationsKernel, rotations,
        OptimizationLevel::O2);
  count("Long loop", "longLoop", longLoopKernel, rotations,
        OptimizationLevel::O1);
  count("Clifford+T level 0", "fusable", fusableRotationsKernel, cliffordT,
        OptimizationLevel::O0);
  count("Clifford+T level 2", "fusable", fusableRotationsKernel, cliffordT,
        OptimizationLevel::O2);
  count("Clifford+T exact angle", "expPauli", expPauliKernel, cliffordT,
        OptimizationLevel::O0);
}

// CHECK-LABEL: Rotations level 0 resource counts:
// CHECK: total: 4{{$}}
// CHECK-LABEL: Rotations level 1 resource counts:
// CHECK: total: 4{{$}}
// CHECK-LABEL: Rotations level 2 resource counts:
// CHECK: total: 3{{$}}
// CHECK-LABEL: Long loop resource counts:
// CHECK: total: 4096{{$}}
// CHECK-LABEL: Clifford+T level 0 resource counts:
// CHECK: rz: 0{{$}}
// CHECK: r1: 0{{$}}
// CHECK-LABEL: Clifford+T level 2 resource counts:
// CHECK: rz: 0{{$}}
// CHECK: r1: 0{{$}}
// CHECK-LABEL: Clifford+T exact angle resource counts:
// CHECK: total: 1{{$}}

/// A rotation whose angle is the kernel's argument.
static const char *argumentAngleKernel = R"#(
func.func @__nvqpp__mlirgen__argumentAngle(%angle: f64) attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %q = quake.alloca !quake.ref
  quake.rz (%angle) %q : (f64, !quake.ref) -> ()
  return
}
)#";

/// A target that can take arguments at run time, as a simulator can, still
/// substitutes the argument values when its stage synthesizes. A stage that
/// keeps `rz` leaves the argument for run time.
void test_quake_unmapped_specialization(mlir::MLIRContext *ctx) {
  const auto compile = [&](std::vector<std::string> basis,
                           cudaq::OptimizationLevel level) {
    cudaq::CompileTarget target;
    target.fullySpecialize = false;
    target.stages.push_back(
        {.irOutputProfile =
             cudaq::config::QuakeUnmappedProfile{
                 .basisGates = std::move(basis),
                 .approximationTolerance = 1e-3,
                 .costModel = cudaq::config::CostModel::t_count},
         .lowering = cudaq::config::InferredLowering{}});
    cudaq::CompileOptions options;
    options.optimizationLevel = level;
    options.seed = 42;
    double angle = 0.3;
    std::vector<void *> v = {static_cast<void *>(&angle)};
    cudaq::KernelArgs args(std::span<void *const>(v.data(), v.size()));
    compileAndDump(ctx, "argumentAngle", argumentAngleKernel, std::move(target),
                   std::move(options), args);
  };
  compile({"h", "s", "t", "x"}, cudaq::OptimizationLevel::O2);
  compile({"h", "s", "t", "x", "rz"}, cudaq::OptimizationLevel::O2);
}

// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__argumentAngle() attributes
// CHECK:           quake.t
// CHECK-LABEL: Compiled module:
// CHECK:         func.func @__nvqpp__mlirgen__argumentAngle(%[[ANGLE:.*]]: f64)
// CHECK:           quake.rz (%[[ANGLE]])

/// Kernel that the base QIR profile accepts.
static const char *measureKernel = R"#(
func.func @__nvqpp__mlirgen__measure() attributes {"cudaq-entrypoint", "cudaq-kernel", no_this} {
  %q = quake.alloca !quake.ref
  quake.h %q : (!quake.ref) -> ()
  %m = quake.mz %q : (!quake.ref) -> !quake.measure
  return
}
)#";

static mlir::LogicalResult printPipeline(mlir::PassManager &pm,
                                         mlir::Operation *op) {
  pm.printAsTextualPipeline(llvm::outs());
  llvm::outs() << "\n";
  return pm.run(op);
}

/// Level 0 must be honored all the way through codegen
void test_optimization_level_0(mlir::MLIRContext *ctx) {
  cudaq::CompileTarget target;
  target.codegen.emissionFormat = "qir-base";
  cudaq::CompileOptions options;
  options.optimizationLevel = cudaq::OptimizationLevel::O0;
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(measureKernel, ctx);
  llvm::outs() << "Optimization level 0 codegen:\n";
  Compiler compiler(target, std::move(options));
  cudaq_internal::compiler::setRunPassManagerHook(&printPipeline);
  auto compiled = compiler.compile(
      "measure", mod.release().getAsOpaquePointer(), cudaq::KernelArgs{},
      /*isEntryPoint=*/true);
  cudaq_internal::compiler::setRunPassManagerHook(nullptr);
}

// CHECK-LABEL: Optimization level 0 codegen:
// CHECK:         combine-quantum-alloc
// CHECK-NOT:     quake-simplify
// CHECK-NOT:     {{[(,]}}dqe{{[,)]}}
// CHECK:         cc-to-llvm

/// An optimization level specified as MLIR attribute overrides the level in the
/// compile options, through codegen
void test_optimization_level_attr(mlir::MLIRContext *ctx) {
  cudaq::CompileTarget target;
  target.codegen.emissionFormat = "qir-base";
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(measureKernel, ctx);
  cudaq::runtime::setOptimizationLevel(
      mod->lookupSymbol("__nvqpp__mlirgen__measure"), 0);
  llvm::outs() << "Kernel optimization level 0 codegen:\n";
  Compiler compiler(target, cudaq::CompileOptions{});
  cudaq_internal::compiler::setRunPassManagerHook(&printPipeline);
  auto compiled = compiler.compile(
      "measure", mod.release().getAsOpaquePointer(), cudaq::KernelArgs{},
      /*isEntryPoint=*/true);
  cudaq_internal::compiler::setRunPassManagerHook(nullptr);
}

// CHECK-LABEL: Kernel optimization level 0 codegen:
// CHECK:         combine-quantum-alloc
// CHECK-NOT:     quake-simplify
// CHECK-NOT:     {{[(,]}}dqe{{[,)]}}
// CHECK:         cc-to-llvm

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();

  test_device_calls_supported(context.get());
  test_device_calls_unsupported(context.get());
  test_no_backend_config(context.get());
  test_empty_pipeline(context.get());
  test_non_empty_pipeline(context.get());
  test_resource_count_completes_phase_lifecycle(context.get());
  test_resource_count_disabled_by_target(context.get());
  test_staged_inferred(context.get());
  test_staged_skip(context.get());
  test_quake_unmapped_levels(context.get());
  test_quake_unmapped_specialization(context.get());
  test_optimization_level_0(context.get());
  test_optimization_level_attr(context.get());
  return 0;
}
