/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Pipelines.h"

using namespace mlir;

void cudaq::opt::commonLoweringPipeline(PassManager &pm, const StringRef &gateset, const std::optional<StringRef> &convertTo) {
  pm.addPass(createConstPropComplex());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addPass(createLiftArrayAlloc());
  pm.addPass(createStatePreparation());
  pm.addPass(createExpandMeasurementsPass());
  // Unrolling pipeline
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createLoopNormalize());
  pm.addPass(createCanonicalizerPass());
  LoopUnrollOptions luo;
  luo.allowBreak = convertTo && convertTo->equals("qir-adaptive");
  pm.addPass(cudaq::opt::createLoopUnroll(luo));
  pm.addPass(cudaq::opt::createUpdateRegisterNames());
  DecompositionPassOptions dpo;
  dpo.enabledPatterns = {"U3ToRotations"};
  pm.addPass(createDecompositionPass(dpo));
  pm.addNestedPass<func::FuncOp>(createLowerToCFGPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createMultiControlDecompositionPass());
  BasisConversionPassOptions options;
  // Empty basis as default will cause an error in BasisConversionPass
  options.basis = StringSwitch<llvm::ArrayRef<std::string>>(gateset)
                    .Case("oqc", OQCbasis)
                    .Case("iqm", IQMbasis)
                    .Case("quantinuum", Quantinuumbasis)
                    .Case("ionq", IonQbasis)
                    .Default({});
  pm.addPass(createBasisConversionPass(options));
  
}

void cudaq::opt::commonPipelineConvertToQIR(
    PassManager &pm, const std::optional<StringRef> &convertTo, const std::optional<StringRef> &mapping) {
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  addAggressiveEarlyInlining(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createApplyOpSpecializationPass());
  pm.addPass(createExpandMeasurementsPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createQuakeAddMetadata());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (convertTo && convertTo->equals("qir-base"))
    pm.addNestedPass<func::FuncOp>(createDelayMeasurementsPass());
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createConvertToQIR());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm, const std::optional<StringRef> &mapping) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (mapping) {
    pm.addNestedPass<func::FuncOp>(createFactorQuantumAllocations());
    pm.addNestedPass<func::FuncOp>(createMemToReg());
    MappingPassOptions mpo;
    mpo.device = mapping.value();
    pm.addNestedPass<func::FuncOp>(createMappingPass(mpo));
    pm.addNestedPass<func::FuncOp>(createRegToMem());
  }
}

void cudaq::opt::addPipelineTranslateToIQMJson(PassManager &pm, const std::optional<StringRef> &mapping) {
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addPass(createExpandMeasurementsPass());
  LoopUnrollOptions luo;
  pm.addPass(createLoopUnroll(luo));
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  if (mapping) {
    pm.addPass(createCanonicalizerPass());
    pm.addNestedPass<func::FuncOp>(createFactorQuantumAllocations());
    pm.addNestedPass<func::FuncOp>(createMemToReg());
    MappingPassOptions mpo;
    mpo.device = mapping.value();
    pm.addNestedPass<func::FuncOp>(createMappingPass(mpo));
    pm.addNestedPass<func::FuncOp>(createRegToMem());
    pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
    pm.addNestedPass<func::FuncOp>(createDelayMeasurementsPass());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}
