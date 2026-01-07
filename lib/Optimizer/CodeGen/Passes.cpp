/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

static SmallVector<std::string> z_disabledPatterns = {"R1ToU3"};

static void addAnyonPPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "z(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  options.disabledPatterns = z_disabledPatterns;
  pm.addPass(createBasisConversionPass(options));
}

static void addAnyonCPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  options.disabledPatterns = z_disabledPatterns;
  pm.addPass(createBasisConversionPass(options));
}

static void addOQCPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      // TODO: make this our native gate set
      "h", "s", "t", "r1", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  options.disabledPatterns = z_disabledPatterns;
  pm.addPass(createBasisConversionPass(options));
}

static void addQCIPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  // Note: QCI's basis gate set is "sx", "rz", "cz", but QCI currently has
  // a transpiler converting all other gates to that basis.
  // We use the gate set below so we can translate all gates to QIR.
  std::string basis[] = {
      "h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  options.disabledPatterns = z_disabledPatterns;
  pm.addPass(createBasisConversionPass(options));
}

static void addQuantinuumPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  options.disabledPatterns = z_disabledPatterns;
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
  options.disabledPatterns = z_disabledPatterns;
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
  options.disabledPatterns = z_disabledPatterns;
  pm.addPass(createBasisConversionPass(options));
}

static void addFermioniqPipeline(OpPassManager &pm) {
  using namespace cudaq::opt;
  std::string basis[] = {
      "h", "s", "t", "rx", "ry", "rz", "x", "y", "z", "x(1)",
  };
  BasisConversionPassOptions options;
  options.basis = basis;
  options.disabledPatterns = z_disabledPatterns;
  pm.addPass(createBasisConversionPass(options));
}

void cudaq::opt::registerTargetPipelines() {
  PassPipelineRegistration<>("anyon-cgate-set-mapping",
                             "Convert kernels to Anyon gate set.",
                             addAnyonCPipeline);
  PassPipelineRegistration<>("anyon-pgate-set-mapping",
                             "Convert kernels to Anyon gate set.",
                             addAnyonPPipeline);
  PassPipelineRegistration<>("oqc-gate-set-mapping",
                             "Convert kernels to OQC gate set.",
                             addOQCPipeline);
  PassPipelineRegistration<>("iqm-gate-set-mapping",
                             "Convert kernels to IQM gate set.",
                             addIQMPipeline);
  PassPipelineRegistration<>("qci-gate-set-mapping",
                             "Convert kernels to QCI gate set.",
                             addQCIPipeline);
  PassPipelineRegistration<>("quantinuum-gate-set-mapping",
                             "Convert kernels to Quantinuum gate set.",
                             addQuantinuumPipeline);
  PassPipelineRegistration<>("ionq-gate-set-mapping",
                             "Convert kernels to IonQ gate set.",
                             addIonQPipeline);
  PassPipelineRegistration<>("fermioniq-gate-set-mapping",
                             "Convert kernels to Fermioniq gate set.",
                             addFermioniqPipeline);
}

void cudaq::opt::registerCodeGenDialect(DialectRegistry &registry) {
  registry.insert<cudaq::codegen::CodeGenDialect>();
}
