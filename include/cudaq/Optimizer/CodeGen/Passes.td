/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
#define CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def PromoteRefToVeqAlloc : Pass<"promote-qubit-allocation"> {
  let summary = "Promote single qubit allocations.";
  let description = [{
    This pass converts all single qubit allocations in the quake dialect to
    allocations of vectors of qubits of length one. This conversion makes all
    allocations uniform for the conversion to QIR.
  }];
}

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

def VerifyBaseProfile : Pass<"verify-base-profile", "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Verify that the output conforms to the base profile";
  let description = [{
    This pass scans over functions in the LLVM-IR dialect to make sure they
    conform to the QIR base profile.
  }];

  let constructor = "cudaq::opt::verifyBaseProfilePass()";
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
}

def RemoveMeasurements : Pass<"remove-measurements"> {
  let summary =
    "Remove measurements and output recording calls from a QIR program";
  let description = [{
    Some backends perform implicit measurements on all qubits at the end of the
    kernel, which is a distinct problem for compiler-generated qubits (as is
    done in the quake-multicontrol-decomposition pass) because the user cannot
    tell the difference between measurement results for their qubits and
    measurement results for compiler-generated qubits. This pass enables the
    compiler to strip out the measurements and output recording calls from a QIR
    program to prevent that problem. This pass MUST be run after normal
    measurements and output recording calls have been generated because that
    metadata must be extracted from the program to allow reconstruction of the
    outputs from the backend.
  }];

  let constructor = "cudaq::opt::createRemoveMeasurementsPass()";
}


#endif // CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
