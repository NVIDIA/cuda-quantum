/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Pipelines.h"

using namespace mlir;

void cudaq::opt::commonPipelineConvertToQIR(
    PassManager &pm, const std::optional<StringRef> &convertTo) {
  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  addAggressiveEarlyInlining(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createApplyOpSpecializationPass());
  pm.addPass(createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createQuakeAddMetadata());
  pm.addPass(createLoopNormalize());
  LoopUnrollOptions luo;
  luo.allowBreak = convertTo && convertTo->equals("qir-adaptive");
  pm.addPass(createLoopUnroll(luo));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (convertTo && convertTo->equals("qir-base"))
    pm.addNestedPass<func::FuncOp>(createDelayMeasurementsPass());
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createConvertToQIR());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void cudaq::opt::addPipelineTranslateToIQMJson(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addPass(createExpandMeasurementsPass());
  LoopUnrollOptions luo;
  pm.addPass(createLoopUnroll(luo));
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}
