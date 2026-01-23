/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TargetPrepPipelineOptions
    : public PassPipelineOptions<TargetPrepPipelineOptions> {
  PassOptions::Option<bool> eraseNoise{
      *this, "erase-noise", llvm::cl::desc("Erase apply noise calls."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> applyConstProp{
      *this, "apply-const-prop",
      llvm::cl::desc("Enable constant propagation in apply specialization."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> allowEarlyExit{
      *this, "allow-early-exit",
      llvm::cl::desc(
          "Enable loop unrolling on loops with early exit conditions."),
      llvm::cl::init(false)};
};

struct TargetFinalizationPipelineOptions
    : public PassPipelineOptions<TargetFinalizationPipelineOptions> {
  PassOptions::Option<bool> allowBreaksInLoops{
      *this, "loops-may-have-break",
      llvm::cl::desc("Enable break statements in loops."),
      llvm::cl::init(true)};
};

struct PythonAOTOptions : public PassPipelineOptions<PythonAOTOptions> {
  PassOptions::Option<bool> autoGenRunStack{
      *this, "gen-run-stack",
      llvm::cl::desc("Autogenerate the cudaq::run dispatch stack."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> deferGKEToJIT{
      *this, "defer-gke-to-jit",
      llvm::cl::desc("Defer running GKE until JIT time."),
      llvm::cl::init(true)};
  PassOptions::Option<std::size_t> codegenKind{
      *this, "codegen-kind", llvm::cl::desc("GKE launch codegen kind."),
      llvm::cl::init(0)};
};
} // namespace

static void createTargetPrepPipeline(OpPassManager &pm,
                                     const TargetPrepPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  cudaq::opt::createClassicalOptimizationPipeline(pm, std::nullopt,
                                                  {options.allowEarlyExit});
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createUnitarySynthesis());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplySpecialization(
      {.constantPropagation = options.applyConstProp}));
  cudaq::opt::addAggressiveInlining(pm);
}

static void
createHardwareTargetPrepPipeline(OpPassManager &pm,
                                 const TargetPrepPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseNoise());
  createTargetPrepPipeline(pm, options);
}

/// Register the standard initial pipeline run for ALL target machines when
/// lowering to hardware. Either this preparation pipeline or the emulation
/// target preparation pipeline is always run between the JIT high-level and JIT
/// mid-level target-specific pipelines from the `.yml` file.
static void registerHardwareTargetPrepPipeline() {
  PassPipelineRegistration<TargetPrepPipelineOptions>(
      "hw-jit-prep-pipeline", "Prep for any hardware target.",
      [](OpPassManager &pm, const TargetPrepPipelineOptions &options) {
        createHardwareTargetPrepPipeline(pm, options);
      });
}

static void
createEmulationTargetPrepPipeline(OpPassManager &pm,
                                  const TargetPrepPipelineOptions &options) {
  if (options.eraseNoise)
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createEraseNoise());
  createTargetPrepPipeline(pm, options);
}

/// Register the standard initial pipeline run for ALL target machines when
/// emulation is enabled.
static void registerEmulationTargetPrepPipeline() {
  PassPipelineRegistration<TargetPrepPipelineOptions>(
      "emul-jit-prep-pipeline", "Prep for any emulation target.",
      [](OpPassManager &pm, const TargetPrepPipelineOptions &options) {
        createEmulationTargetPrepPipeline(pm, options);
      });
}

void cudaq::opt::addDecompositionPass(OpPassManager &pm,
                                      ArrayRef<std::string> enabledPats,
                                      ArrayRef<std::string> disabledPats) {
  // NB: Both of these ListOption *must* be set here or they may contain garbage
  // and the compiler may crash.
  cudaq::opt::DecompositionPassOptions opts;
  opts.disabledPatterns = disabledPats;
  opts.enabledPatterns = enabledPats;
  pm.addPass(cudaq::opt::createDecompositionPass(opts));
}

static void createTargetDeployPipeline(OpPassManager &pm) {
  cudaq::opt::createClassicalOptimizationPipeline(pm);
  cudaq::opt::addDecompositionPass(pm, {std::string("U3ToRotations")});
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      cudaq::opt::createMultiControlDecompositionPass());
}

/// Register the standard deployment pipeline run for ALL target machines. This
/// pipeline is run between the mid-level and low-level target-specific
/// pipelines.
static void registerTargetDeployPipeline() {
  PassPipelineRegistration<>(
      "jit-deploy-pipeline", "Standard deployment pipeline for all targets.",
      [](OpPassManager &pm) { ::createTargetDeployPipeline(pm); });
}

void cudaq::opt::createTargetFinalizePipeline(OpPassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addPass(createSymbolDCEPass());
}

static void createJITTargetFinalizePipeline(OpPassManager &pm) {
  pm.addPass(cudaq::opt::createDistributedDeviceCall());
  cudaq::opt::addAggressiveInlining(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createApplyControlNegations());
  cudaq::opt::createTargetFinalizePipeline(pm);
}

/// Register the standard finalization pipeline run for ALL target machines.
/// This pipeline is run after the low-level target-specific pipelines.
static void registerTargetFinalizePipeline() {
  PassPipelineRegistration<>(
      "jit-finalize-pipeline",
      "Standard JIT finalization pipeline for all targets.",
      [](OpPassManager &pm) { createJITTargetFinalizePipeline(pm); });
}

void cudaq::opt::registerJITPipelines() {
  registerHardwareTargetPrepPipeline();
  registerEmulationTargetPrepPipeline();
  registerTargetDeployPipeline();
  registerTargetFinalizePipeline();
}

/// This pipeline is defined to mirror the nvq++ driver's pipeline. It can be
/// used post front-end bridge to lower the Quake IR to a target agnostic
/// version of Quake code.
static void createPythonAOTPipeline(OpPassManager &pm,
                                    const PythonAOTOptions &options) {
  // NB: This pipeline should be kept in synch with the pipeline in nvq++.
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createVariableCoalesce());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addPass(cudaq::opt::createLambdaLifting());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplySpecialization());
  cudaq::opt::GenerateKernelExecutionOptions gkeOpts;
  gkeOpts.genRunStack = options.autoGenRunStack;
  gkeOpts.deferToJIT = options.deferGKEToJIT;
  gkeOpts.codegenKind = options.codegenKind;
  pm.addPass(cudaq::opt::createGenerateKernelExecution(gkeOpts));
  cudaq::opt::addAggressiveInlining(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createConstantPropagation());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLiftArrayAlloc());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createGetConcreteMatrix());
}

void cudaq::opt::createPythonAOTPipeline(OpPassManager &pm,
                                         bool autoGenRunStack) {
  PythonAOTOptions opts;
  opts.autoGenRunStack = autoGenRunStack;
  ::createPythonAOTPipeline(pm, opts);
}

static void registerPythonAOTPipeline() {
  PassPipelineRegistration<PythonAOTOptions>(
      "aot-prep-pipeline",
      "Pipeline to lower code for simulation or JIT compilation.",
      [](OpPassManager &pm, const PythonAOTOptions &options) {
        ::createPythonAOTPipeline(pm, options);
      });
}

void cudaq::opt::registerAOTPipelines() { registerPythonAOTPipeline(); }
