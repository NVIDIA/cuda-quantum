/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Transforms/Passes.h"
#include "PassDetails.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

// Pipeline to convert Quake to QTX.
static void addPipelineFromQuakeToQTX(OpPassManager &pm) {
  pm.addPass(createInlinerPass());
  pm.addPass(createCanonicalizerPass());
  pm.addNestedPass<func::FuncOp>(cudaq::opt::createQuakeAddDeallocs());
  pm.addPass(cudaq::opt::createConvertQuakeToQTXPass());
  pm.addPass(cudaq::opt::createConvertFuncToQTXPass());
  pm.addPass(createCanonicalizerPass());
}

static void addPipelineFromQTXToQuake(OpPassManager &pm) {
  pm.addPass(cudaq::opt::createConvertQTXToQuakePass());
  pm.addPass(createCanonicalizerPass());
}

void cudaq::opt::registerConversionPipelines() {
  PassPipelineRegistration<>("quake-to-qtx",
                             "Convert quake dialect to qtx dialect",
                             addPipelineFromQuakeToQTX);
  PassPipelineRegistration<>("qtx-to-quake",
                             "Convert qtx dialect to quake dialect",
                             addPipelineFromQTXToQuake);
}
