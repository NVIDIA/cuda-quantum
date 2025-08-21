/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Pipelines.h"

using namespace mlir;

void cudaq::opt::commonPipelineConvertToQIR(PassManager &pm,
                                            StringRef codeGenFor,
                                            StringRef passConfigAs) {
  auto codeGenFields = codeGenFor.split(':');
  auto passConfigFields = passConfigAs.split(':');

  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  addAggressiveEarlyInlining(pm);
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createQuakeAddMetadata());
  pm.addPass(createQuakePropagateMetadata());
  pm.addNestedPass<func::FuncOp>(createLoopNormalize());
  LoopUnrollOptions luo;
  luo.allowBreak = passConfigAs == "qir-adaptive";
  pm.addNestedPass<func::FuncOp>(createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  // A final round of apply specialization after loop unrolling. This should
  // eliminate any residual control structures so the kernel specializations can
  // succeed.
  pm.addPass(createApplySpecialization());
  // If there was any specialization, we want another round in inlining to
  // inline the apply calls properly.
  addAggressiveEarlyInlining(pm);
  addLowerToCFG(pm);
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());

  if (passConfigFields.first == "qir-base")
    pm.addNestedPass<func::FuncOp>(createDelayMeasurementsPass());

  if (codeGenFields.first == "qir" || codeGenFields.first == "qir-full")
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "full:" +
                                                   codeGenFields.second.str());
  else if (codeGenFields.first == "qir-base")
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "base-profile:" +
                                                   codeGenFields.second.str());
  else if (codeGenFields.first == "qir-adaptive")
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "adaptive-profile:" +
                                                   codeGenFields.second.str());
  else
    emitError(UnknownLoc::get(pm.getContext()),
              "convert to QIR must be given a valid specification to use.");
  pm.addPass(createReturnToOutputLog());
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCCToLLVM());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createDeadStoreRemoval());
  pm.addPass(createSymbolDCEPass());
}

void cudaq::opt::addPipelineTranslateToIQMJson(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createUnwindLowering());
  pm.addNestedPass<func::FuncOp>(createExpandMeasurementsPass());
  LoopUnrollOptions luo;
  pm.addNestedPass<func::FuncOp>(createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  addLowerToCFG(pm);
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

void cudaq::opt::addPipelineConvertToQIR(PassManager &pm, StringRef convertTo) {
  commonPipelineConvertToQIR(pm, convertTo, convertTo);
  auto targetPair = convertTo.split(':');
  if (targetPair.first != "qir" && targetPair.first != "qir-full")
    addQIRProfileVerify(pm, convertTo);
}
