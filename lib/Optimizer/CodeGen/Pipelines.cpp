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
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createApplyControlNegations());
  cudaq::opt::addAggressiveEarlyInlining(pm);
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLoweringPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(cudaq::opt::createApplyOpSpecializationPass());
  pm.addPass(cudaq::opt::createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createClassicalMemToReg());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addPass(cudaq::opt::createLoopNormalize());
  cudaq::opt::LoopUnrollOptions luo;
  luo.allowBreak = convertTo && convertTo->equals("qir-adaptive");
  pm.addPass(cudaq::opt::createLoopUnroll(luo));
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  if (convertTo && convertTo->equals("qir-base"))
    pm.addNestedPass<func::FuncOp>(cudaq::opt::createDelayMeasurementsPass());
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(cudaq::opt::createConvertToQIRPass());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm) {
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}

void cudaq::opt::addPipelineTranslateToIQMJson(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createCombineQuantumAllocations());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
}
