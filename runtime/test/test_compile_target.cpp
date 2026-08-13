/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// This test is compiled inside the runtime directory tree. We include it as a
// regression test and use FileCheck to verify the output.

// RUN: test_compile_target | FileCheck %s

#include "common/KernelArgs.h"
#include "cudaq_internal/compiler/CompiledModuleHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "cudaq_internal/compiler/RuntimeMLIR.h"
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
  auto compiled = compiler.runPassPipeline(
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

/// Compile \p quake and print selected resource metrics in a stable order.
static void compileAndDumpResources(mlir::MLIRContext *ctx,
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
  auto compiled = compiler.runPassPipeline(
      kernelName, mod.release().getAsOpaquePointer(), cudaq::KernelArgs{},
      /*isEntryPoint=*/true);

  auto *resources = compiled.getResources();
  if (!resources) {
    llvm::outs() << "NO RESOURCE ARTIFACT\n";
    return;
  }

  llvm::outs() << "Phase resource counts:\n"
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
  return cudaq::CompileTarget(cfg, /*runtimeConfig=*/
                              std::map<std::string, std::string>{},
                              /*emulate=*/false);
}

/// Build a CompileTarget with a backend configuration with an empty pass
/// pipeline
static cudaq::CompileTarget emptyPipelineTarget() {
  cudaq::config::TargetConfig cfg;
  cfg.BackendConfig = cudaq::config::BackendEndConfigEntry{};
  return cudaq::CompileTarget(cfg, std::map<std::string, std::string>{},
                              /*emulate=*/false);
}

/// Build a CompileTarget with a backend configuration with a non-empty pass
/// pipeline
static cudaq::CompileTarget
nonEmptyPipelineTarget(const std::string &pipeline) {
  cudaq::config::TargetConfig cfg;
  cudaq::config::BackendEndConfigEntry backend;
  backend.TargetPassPipeline = pipeline;
  cfg.BackendConfig = backend;
  return cudaq::CompileTarget(cfg, std::map<std::string, std::string>{},
                              /*emulate=*/false);
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
  compileAndDumpResources(ctx, "phaseResources", phaseResourceKernel,
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

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();

  test_device_calls_supported(context.get());
  test_device_calls_unsupported(context.get());
  test_no_backend_config(context.get());
  test_empty_pipeline(context.get());
  test_non_empty_pipeline(context.get());
  test_resource_count_completes_phase_lifecycle(context.get());
  return 0;
}
