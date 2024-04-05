/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;

static void addOQCPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      // TODO: make this our native gate set
      "h", "s", "t", "r1", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  pm.addPass(createBasisConversionPass(options));
}

static void addQuantinuumPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  pm.addPass(createBasisConversionPass(options));
}

static void addIQMPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "phased_rx",
      "z(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  pm.addPass(createBasisConversionPass(options));
}

static void addIonQPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "h",  "s", "t", "rx", "ry",
      "rz", "x", "y", "z",  "x(1)", // TODO set to ms, gpi, gpi2
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  pm.addPass(createBasisConversionPass(options));
}

void cudaq::opt::registerTargetPipelines() {
  PassPipelineRegistration<>("oqc-gate-set-mapping",
                             "Convert kernels to OQC gate set.",
                             addOQCPipeline);
  PassPipelineRegistration<>("iqm-gate-set-mapping",
                             "Convert kernels to IQM gate set.",
                             addIQMPipeline);
  PassPipelineRegistration<>("quantinuum-gate-set-mapping",
                             "Convert kernels to Quantinuum gate set.",
                             addQuantinuumPipeline);
  PassPipelineRegistration<>("ionq-gate-set-mapping",
                             "Convert kernels to IonQ gate set.",
                             addIonQPipeline);
}

void cudaq::opt::registerCodeGenDialect(DialectRegistry &registry) {
  registry.insert<cudaq::codegen::CodeGenDialect>();
}
