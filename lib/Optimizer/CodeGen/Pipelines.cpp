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
  pm.addNestedPass<func::FuncOp>(createApplyControlNegations());
  addAggressiveEarlyInlining(pm);
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createApplyOpSpecializationPass());
  pm.addNestedPass<func::FuncOp>(createExpandMeasurementsPass());
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createQuakeAddMetadata());
  pm.addNestedPass<func::FuncOp>(createLoopNormalize());
  LoopUnrollOptions luo;
  luo.allowBreak = passConfigAs == "qir-adaptive";
  pm.addNestedPass<func::FuncOp>(createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  if (passConfigAs == "qir-base")
    pm.addNestedPass<func::FuncOp>(createDelayMeasurementsPass());
  if (codeGenFor == "qir")
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "full");
  else if (codeGenFor == "qir-base")
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "base-profile");
  else if (codeGenFor == "qir-adaptive")
    cudaq::opt::addConvertToQIRAPIPipeline(pm, "adaptive-profile");
  else
    emitError(UnknownLoc::get(pm.getContext()),
              "convert to QIR must be given a valid specification to use.");
  pm.addPass(createConvertMathToFuncs());
  pm.addPass(createSymbolDCEPass());
  pm.addPass(createCCToLLVM());
}

void cudaq::opt::addPipelineTranslateToOpenQASM(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
  pm.addNestedPass<func::FuncOp>(createClassicalMemToReg());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addPass(createSymbolDCEPass());
}

void cudaq::opt::addPipelineTranslateToIQMJson(PassManager &pm) {
  pm.addNestedPass<func::FuncOp>(createUnwindLoweringPass());
  pm.addNestedPass<func::FuncOp>(createExpandMeasurementsPass());
  LoopUnrollOptions luo;
  pm.addNestedPass<func::FuncOp>(createLoopUnroll(luo));
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createLowerToCFGPass());
  pm.addNestedPass<func::FuncOp>(createQuakeAddDeallocs());
  pm.addNestedPass<func::FuncOp>(createCombineQuantumAllocations());
  pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(createCSEPass());
}

void cudaq::opt::addPipelineConvertToQIR(PassManager &pm, StringRef convertTo) {
  commonPipelineConvertToQIR(pm, convertTo, convertTo);
  if (convertTo != "qir")
    addQIRProfileVerify(pm, convertTo);
}
