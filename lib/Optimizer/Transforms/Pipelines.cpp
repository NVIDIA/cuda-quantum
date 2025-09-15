/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
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
};

struct TargetFinalizationPipelineOptions
    : public PassPipelineOptions<TargetFinalizationPipelineOptions> {
  PassOptions::Option<bool> allowBreaksInLoops{
      *this, "loops-may-have-break",
      llvm::cl::desc("Enable break statements in loops."),
      llvm::cl::init(true)};
};

struct PreDeviceCodeLoaderOptions
    : public PassPipelineOptions<PreDeviceCodeLoaderOptions> {
  PassOptions::Option<bool> autoGenRunStack{
      *this, "gen-run-stack",
      llvm::cl::desc("Autogenerate the cudaq::run dispatch stack."),
      llvm::cl::init(true)};
};
} // namespace

static void createTargetPrepPipeline(OpPassManager &pm,
                                     const TargetPrepPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  cudaq::opt::createClassicalOptimizationPipeline(pm);
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createStatePreparation());
  pm.addPass(cudaq::opt::createUnitarySynthesis());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplySpecialization(
      {.constantPropagation = options.applyConstProp}));
  cudaq::opt::addAggressiveInlining(pm);
  cudaq::opt::createClassicalOptimizationPipeline(pm);
  cudaq::opt::DecompositionPassOptions opts;
  opts.enabledPatterns = {"U3ToRotations"};
  pm.addPass(cudaq::opt::createDecompositionPass(opts));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(
      cudaq::opt::createMultiControlDecompositionPass());
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

static void createTargetDeployPipeline(OpPassManager &pm) {
  pm.addPass(cudaq::opt::createDistributedDeviceCall());
  cudaq::opt::addAggressiveInlining(pm);
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
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createApplyControlNegations());
  pm.addPass(createSymbolDCEPass());
}

/// Register the standard finalization pipeline run for ALL target machines.
/// This pipeline is run after the low-level target-specific pipelines.
static void registerTargetFinalizePipeline() {
  PassPipelineRegistration<>(
      "jit-finalize-pipeline",
      "Standard finalization pipeline for all targets.",
      [](OpPassManager &pm) { cudaq::opt::createTargetFinalizePipeline(pm); });
}

void cudaq::opt::registerJITPipelines() {
  registerHardwareTargetPrepPipeline();
  registerEmulationTargetPrepPipeline();
  registerTargetDeployPipeline();
  registerTargetFinalizePipeline();
}

/// This pipeline is defined to mirror the nvq++ driver's pipeline up to and
/// including generation of the device code loader. It can be used post
/// front-end bridge to lower the Quake IR to the same GDCL level.
static void
createPreDeviceCodeLoaderPipeline(OpPassManager &pm,
                                  const PreDeviceCodeLoaderOptions &options) {
  // NB: This pipeline should be kept in synch with the pipeline in nvq++.
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createVariableCoalesce());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLambdaLiftingPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createApplySpecialization());
  cudaq::opt::GenerateKernelExecutionOptions gkeOpts;
  gkeOpts.genRunStack = options.autoGenRunStack;
  pm.addPass(cudaq::opt::createGenerateKernelExecution(gkeOpts));
  cudaq::opt::addAggressiveInlining(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createConstantPropagation());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLiftArrayAlloc());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  pm.addPass(cudaq::opt::createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createGetConcreteMatrix());
  pm.addPass(cudaq::opt::createGenerateDeviceCodeLoader());
}

void cudaq::opt::createPreDeviceCodeLoaderPipeline(OpPassManager &pm,
                                                   bool autoGenRunStack) {
  PreDeviceCodeLoaderOptions opts;
  opts.autoGenRunStack = autoGenRunStack;
  ::createPreDeviceCodeLoaderPipeline(pm, opts);
}

static void registerPreDeviceCodeLoaderPipeline() {
  PassPipelineRegistration<PreDeviceCodeLoaderOptions>(
      "aot-prep-pipeline",
      "Pipeline to lower code for simulation or JIT compilation.",
      [](OpPassManager &pm, const PreDeviceCodeLoaderOptions &options) {
        ::createPreDeviceCodeLoaderPipeline(pm, options);
      });
}

void cudaq::opt::registerAOTPipelines() {
  registerPreDeviceCodeLoaderPipeline();
}
