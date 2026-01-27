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
#include "mlir/Transforms/Passes.h"

using namespace mlir;

namespace {
struct TargetCodegenPipelineOptions
    : public PassPipelineOptions<TargetCodegenPipelineOptions> {
  PassOptions::Option<bool> allowBreaksInLoops{
      *this, "loops-may-have-break",
      llvm::cl::desc("Enable break statements in loops."),
      llvm::cl::init(true)};
  PassOptions::Option<bool> appendDeprecatedVerifier{
      *this, "append-verifier",
      llvm::cl::desc("Append the QIR verifier pipeline."),
      llvm::cl::init(false)};
  PassOptions::Option<std::string> target{
      *this, "convert-to", llvm::cl::desc("Conversion target specifier."),
      llvm::cl::init("")};
};
} // namespace

static void addQIRConversionPipeline(PassManager &pm, StringRef convertTo) {
  auto convertFields = convertTo.split(':');
  if (convertFields.first == "qir" || convertFields.first == "qir-full") {
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "full:" +
                                                   convertFields.second.str());
  } else if (convertFields.first == "qir-base") {
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createDelayMeasurementsPass());
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "base-profile:" +
                                                   convertFields.second.str());
  } else if (convertFields.first == "qir-adaptive") {
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "adaptive-profile:" +
                                                   convertFields.second.str());
  } else {
    emitError(UnknownLoc::get(pm.getContext()),
              "convert to QIR must be given a valid specification to use.");
  }
}

template <bool isJIT>
void createCommonTargetCodegenPipeline(
    PassManager &pm, const TargetCodegenPipelineOptions &options) {
  if constexpr (isJIT) {
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandMeasurementsPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    // One last gasp of loop unrolling pass, primarily to catch the expanded
    // measurements.
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopNormalize());
    cudaq::opt::LoopUnrollOptions luo;
    luo.allowBreak = options.allowBreaksInLoops;
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createLoopUnroll(luo));
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  } else {
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createApplyControlNegations());
    cudaq::opt::addAggressiveInlining(pm);
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLowering());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createExpandMeasurementsPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
    pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createCSEPass());
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
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
  }
  cudaq::opt::addLowerToCFG(pm);
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createStackFramePrealloc());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

template <bool isJIT>
void createTargetCodegenPipeline(PassManager &pm,
                                 const TargetCodegenPipelineOptions &options) {
  createCommonTargetCodegenPipeline<isJIT>(pm, options);
  ::addQIRConversionPipeline(pm, options.target);
  pm.addPass(cudaq::opt::createReturnToOutputLog());
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(cudaq::opt::createCCToLLVM());
  if (options.appendDeprecatedVerifier)
    cudaq::opt::addQIRProfileVerify(pm, options.target);
}

template <bool isJIT>
void createTargetCodegenPipeline(PassManager &pm, StringRef convertTo) {
  auto convertFields = convertTo.split(':');
  TargetCodegenPipelineOptions opts;
  opts.allowBreaksInLoops = convertFields.first == "qir-adaptive";
  opts.appendDeprecatedVerifier =
      convertFields.first != "qir" && convertFields.first != "qir-full";
  opts.target = convertTo.str();
  createTargetCodegenPipeline<isJIT>(pm, opts);
}

void cudaq::opt::addJITPipelineConvertToQIR(PassManager &pm,
                                            StringRef convertTo) {
  ::createTargetCodegenPipeline</*JIT=*/true>(pm, convertTo);
}

void cudaq::opt::addAOTPipelineConvertToQIR(PassManager &pm,
                                            StringRef convertTo) {
  if (convertTo.empty())
    convertTo = "qir";
  ::createTargetCodegenPipeline</*JIT=*/false>(pm, convertTo);
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
  pm.addNestedPass<func::FuncOp>(createMultiControlDecompositionPass());
  pm.addPass(createDecompositionPass(
      {.enabledPatterns = {"SToR1", "TToR1", "R1ToU3", "U3ToRotations",
                           "CHToCX", "CCZToCX", "CRzToCX", "CRyToCX", "CRxToCX",
                           "CR1ToCX", "CCZToCX", "RxAdjToRx", "RyAdjToRy",
                           "RzAdjToRz"}}));
  pm.addPass(createQuakeToCCPrep());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createExpandControlVeqs());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addPass(createSymbolDCEPass());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm) {
  createCommonTargetCodegenPipeline</*isJIT=*/true>(pm, {});
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
