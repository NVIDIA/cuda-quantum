/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Conversion/MathToFuncs/MathToFuncs.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TargetCodegenPipelineOptions
    : public PassPipelineOptions<TargetCodegenPipelineOptions> {
  PassOptions::Option<bool> allowBreaksInLoops{
      *this, "loops-may-have-break",
      llvm::cl::desc("Enable break statements in loops."),
      llvm::cl::init(true)};
  PassOptions::Option<std::string> target{
      *this, "convert-to", llvm::cl::desc("Conversion target specifier."),
      llvm::cl::init("")};
};
} // namespace

static void addQIRConversionPipeline(OpPassManager &pm, StringRef convertTo) {
  auto convertFields = convertTo.split(':');
  if (convertFields.first == "qir" || convertFields.first == "qir-full") {
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "full:" +
                                                   convertFields.second.str());
  } else if (convertFields.first == "qir-base") {
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createDelayMeasurements());
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "base-profile:" +
                                                   convertFields.second.str());
  } else if (convertFields.first == "qir-adaptive") {
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "adaptive-profile:" +
                                                   convertFields.second.str());
  } else {
    [[maybe_unused]] auto droppedOnTheFloor = emitOptionalError(
        {}, "convert to QIR must be given a valid specification to use.");
  }
}

static void
createCommonTargetCodegenPipeline(OpPassManager &pm,
                                  const TargetCodegenPipelineOptions &options) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlNegations());
  cudaq::opt::addAggressiveInlining(pm);
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddMetadata());
  pm.addPass(cudaq::opt::createQuakePropagateMetadata());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
  cudaq::opt::LoopUnrollOptions luo;
  luo.allowBreak = options.allowBreaksInLoops;
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  // A final round of apply specialization after loop unrolling. This should
  // eliminate any residual control structures so the kernel specializations
  // can succeed.
  pm.addPass(cudaq::opt::createApplySpecialization());
  // If there was any specialization, we want another round in inlining to
  // inline the apply calls properly.
  cudaq::opt::addAggressiveInlining(pm);
  cudaq::opt::addLowerToCFG(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createStackFramePrealloc());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

static void
createTargetCodegenPipeline(OpPassManager &pm,
                            const TargetCodegenPipelineOptions &options,
                            bool useValueSemantics) {
  createCommonTargetCodegenPipeline(pm, options);
  if (useValueSemantics) {
    pm.addNestedPass<func::FuncOp>(
        cudaq::opt::createFactorQuantumAllocations());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandControlVeqs());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createCableRoughIn());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createMemToReg());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createRepairLinearType());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeSimplify());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createDeadQuantumElimination());
  }
  ::addQIRConversionPipeline(pm, options.target);
  // QIR conversion may introduce cc.loop, lower to cf.
  cudaq::opt::addLowerToCFG(pm);
  cudaq::opt::ReturnToOutputLogOptions opts;
  // Only allow dynamic results with full QIR (local simulator targets).
  auto tgt = StringRef(options.target).split(':').first;
  opts.allowDynamicResult = tgt == "qir" || tgt == "qir-full";
  pm.addPass(cudaq::opt::createReturnToOutputLog(opts));
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(cudaq::opt::createCCToLLVM());
}

static void createTargetCodegenPipeline(OpPassManager &pm,
                                        bool useValueSemantics,
                                        StringRef convertTo) {
  auto convertFields = convertTo.split(':');
  TargetCodegenPipelineOptions opts;
  opts.allowBreaksInLoops = convertFields.first == "qir-adaptive";
  opts.target = convertTo.str();
  createTargetCodegenPipeline(pm, opts, useValueSemantics);
}

void cudaq::opt::addAOTPipelineConvertToQIR(PassManager &pm,
                                            StringRef convertTo,
                                            bool useValueSemantics) {
  if (convertTo.empty())
    convertTo = "qir";
  ::createTargetCodegenPipeline(pm, useValueSemantics, convertTo);
}

namespace {
struct CodegenForQIRPipelineOptions
    : public PassPipelineOptions<CodegenForQIRPipelineOptions> {
  PassOptions::Option<std::string> convertTo{
      *this, "convert-to",
      llvm::cl::desc("option to specify what QIR profile to convert to."),
      llvm::cl::init("qir")};
  PassOptions::Option<bool> useValueSemantics{
      *this, "value-semantics",
      llvm::cl::desc(
          "lower to value semantics to enable quantum optimizations"),
      llvm::cl::init(false)};
};
} // namespace

void cudaq::opt::registerCodegenForQIRPipeline() {
  PassPipelineRegistration<CodegenForQIRPipelineOptions>(
      "codegen-for-qir", "Convert quake to one of the QIR APIs.",
      [](OpPassManager &pm, const CodegenForQIRPipelineOptions &opt) {
        ::createTargetCodegenPipeline(pm, opt.useValueSemantics, opt.convertTo);
      });
}

void cudaq::opt::createPipelineTransformsForPythonToOpenQASM(
    OpPassManager &pm) {
  pm.addPass(createLambdaLifting());
  // Run most of the passes from hardware pipelines.
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createLoopNormalize());
  pm.addNestedPass<func::FuncOp>(createLoopUnroll());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createLiftArrayAlloc());
  pm.addPass(createGlobalizeArrayValues());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createGetConcreteMatrix());
  pm.addPass(createUnitarySynthesis());
  pm.addPass(createApplySpecialization());
  addAggressiveInlining(pm);
  pm.addPass(createSymbolDCEPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createMultiControlDecomposition());
  pm.addPass(createDecomposition(
      {.basis = {"h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)"},
       .disabledPatterns = {},
       .enabledPatterns = {}}));
  pm.addPass(createQuakeToCCPrep());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createExpandControlVeqs());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addPass(createSymbolDCEPass());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createDeadStoreRemoval());
  pm.addPass(createSymbolDCEPass());
}

void cudaq::opt::addPipelineTranslateToIQMJson(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createLoopNormalize());
  LoopUnrollOptions luo;
  pm.addNestedPass<func::FuncOp>(createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  addLowerToCFG(pm);
  pm.addNestedPass<func::FuncOp>(createStackFramePrealloc());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
}
