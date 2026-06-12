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
                           std::unique_ptr<cudaq::CompileTarget> target,
                           cudaq::KernelArgs args = {}) {
  auto mod = mlir::parseSourceString<mlir::ModuleOp>(quake, ctx);
  if (!mod) {
    llvm::outs() << "FAILED TO PARSE\n";
    return;
  }

  Compiler compiler(std::move(target));
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

/// Build a CompileTarget with no backend configuration
static std::unique_ptr<cudaq::CompileTarget> noBackendTarget() {
  cudaq::config::TargetConfig cfg;
  return std::make_unique<cudaq::CompileTarget>(
      cfg, /*runtimeConfig=*/
      std::map<std::string, std::string>{},
      /*emulate=*/false);
}

/// Build a CompileTarget with a backend configuration with an empty pass
/// pipeline
static std::unique_ptr<cudaq::CompileTarget> emptyPipelineTarget() {
  cudaq::config::TargetConfig cfg;
  cfg.BackendConfig = cudaq::config::BackendEndConfigEntry{};
  return std::make_unique<cudaq::CompileTarget>(
      cfg, std::map<std::string, std::string>{},
      /*emulate=*/false);
}

/// Build a CompileTarget with a backend configuration with a non-empty pass
/// pipeline
static std::unique_ptr<cudaq::CompileTarget>
nonEmptyPipelineTarget(const std::string &pipeline) {
  cudaq::config::TargetConfig cfg;
  cudaq::config::BackendEndConfigEntry backend;
  backend.TargetPassPipeline = pipeline;
  cfg.BackendConfig = backend;
  return std::make_unique<cudaq::CompileTarget>(
      cfg, std::map<std::string, std::string>{},
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

void test_device_calls_supported(mlir::MLIRContext *ctx) {
  std::int64_t arg = 42;
  std::vector<void *> v = {static_cast<void *>(&arg)};
  cudaq::KernelArgs args(std::span<void *const>(v.data(), v.size()));

  auto target = std::make_unique<cudaq::CompileTarget>();
  target->supportDeviceCalls = true;
  // Isolate the device-call lowering performed during argument synthesis from
  // the rest of the target lowering pipeline.
  target->pipelineConfig.skipTargetLoweringPipeline = true;
  compileAndDump(ctx, "devKernel", deviceCallKernel, std::move(target), args);
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

  auto target = std::make_unique<cudaq::CompileTarget>();
  target->supportDeviceCalls = false;
  target->pipelineConfig.skipTargetLoweringPipeline = true;
  compileAndDump(ctx, "devKernel", deviceCallKernel, std::move(target), args);
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

int main() {
  auto context = cudaq_internal::compiler::getOwningMLIRContext();

  test_device_calls_supported(context.get());
  test_device_calls_unsupported(context.get());
  test_no_backend_config(context.get());
  test_empty_pipeline(context.get());
  test_non_empty_pipeline(context.get());
  return 0;
}
