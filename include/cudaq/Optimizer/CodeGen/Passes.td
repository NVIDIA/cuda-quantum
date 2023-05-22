/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
#define CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def QuakeToQIR : Pass<"quake-to-qir", "mlir::ModuleOp"> {
  let summary = "Lower Quake to QIR.";
  let description = [{
    Converts Quake to QIR (as LLVM-IR). The quake-add-deallocs pass should be
    run before this pass in order to properly generate deallocations for
    allocated qubits.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let constructor = "cudaq::opt::createConvertToQIRPass()";
}

def QIRToBaseQIRPrep : Pass<"qir-to-base-qir-prep", "mlir::ModuleOp"> {
  let summary = "Prepare the IR for rewriting to the base profile";
  let description = [{
    This is a (module) subpass of the pipeline to convert to QIR Base Profile.

    Before we can convert the functions to the base profile, we have to do
    a bit of bookkeeping on the module itself. That preparation is done in
    this pass. Specifically, we create all the function declarations that we
    may need and add them to the ModuleOp a priori. This avoids multi-threading
    conflicts when rewriting matching DAGs independently.
  }];

  let constructor = "cudaq::opt::createBaseProfilePreparationPass()";
}

def QIRToBaseQIRFunc : Pass<"quake-to-base-qir-func",
                            "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Analyze kernels and add attributes and record calls.";
  let description = [{
    This is a (function) subpass of the pipeline to convert to QIR Base Profile.

    Each function with a body is analyzed for qubit allocations and qubit
    measurements. Attributes for the total count of qubits are added to the
    LLVMFuncOp. Additionally, measurements are collected and recording marker
    functions are added to the final block in the function.
  }];

  let constructor = "cudaq::opt::createConvertToQIRFuncPass()";
}

def QIRToBaseQIR : Pass<"qir-to-base-qir"> {
  let summary =
    "After lowering a Quake kernel to QIR, lower further to the Base Profile.";
  let description = [{
    This is a subpass of the pipeline to convert to QIR Base Profile.

    This pass lowers various QIR DAGs to the QIR Base Profile. See
    https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/7_Profiles.md
  }];

  let constructor = "cudaq::opt::createQIRToBaseProfilePass()";

  let options = [
    Option<"topDownProcessingEnabled", "top-down", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"enableRegionSimplification", "region-simplify", "bool",
           /*default=*/"true",
           "Seed the worklist in general top-down order">,
    Option<"maxIterations", "max-iterations", "int64_t",
           /*default=*/"10",
           "Seed the worklist in general top-down order">
  ];
}

def IQMGateSetMapper : Pass<"iqm-gate-set-mapping", "mlir::func::FuncOp"> {
  let summary = "Convert gates to IQM mappings.";
  let description = [{
    This pass converts the standard gates to the IQM mapping gates. The
    supported gates are phased R and CZ.
  }];

  let constructor = "cudaq::opt::createIQMGateSetMapping()";
}

def R1DecompositionGateSetMapper : Pass<"r1-decomposition-gate-set-mapping",
                                        "mlir::func::FuncOp"> {
  let summary =
    "Decompose controlled-R1 gates to controlled-Z plus local rotations.";
  let description = [{
    Maps a controlled R1 on a single control qubit to a controlled Z rotation
    on a single control qubit.
  }];

  let constructor = "cudaq::opt::createR1DecompositionGateSetMapping()";
}

def RnDecompositionGateSetMapper : Pass<"rn-decomposition-gate-set-mapping",
                                        "mlir::func::FuncOp"> {
  let summary =
    "Decompose R[XYZ] gates to control-X and single qubit rotations.";
  let description = [{
    Maps controlled rotations with a single control qubit to CNOTs and single
    qubit rotations.
  }];

  let constructor = "cudaq::opt::createRnDecompositionGateSetMapping()";
}

def QuantinuumGateSetMapper : Pass<"quantinuum-gate-set-mapping",
                                   "mlir::func::FuncOp"> {
  let summary = "Convert gates to Quantinuum backend mappings.";
  let description = [{
    Maps a controlled R1 on a single control qubit to a controlled Z rotation
    on a single control qubit.

    Maps controlled rotations with a single control qubit to CNOTs and single
    qubit rotations.
  }];

  let constructor = "cudaq::opt::createQuantinuumGateSetMapping()";
}

#endif // CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
