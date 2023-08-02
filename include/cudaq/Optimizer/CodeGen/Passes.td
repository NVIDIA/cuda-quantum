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

def QuakeCircuitCut : Pass<"cut-quake", "mlir::ModuleOp"> {
  let summary = "Cut the single Quake code into multiple Quake codes for circuit cutting use cases.";
  let description = [{
    Partitions a ModuleOp with a single FuncOp containing Quake code in the 
    value-semantic model (wires) to a ModuleOp with multiple FuncOps representing the 
    Quake code (quantum circuit) cut into k sub-circuits with inserted measure and 
    state preparation nodes. The resultant circuits can be used as part of a 
    circuit cutting simulation workflow. 
  }];

  let constructor = "cudaq::opt::createQuakeCircuitCutPass()";

  let options = [
    Option<"numPartitions", "k", "std::size_t",
      /*default=*/"2", "The number of partitions.">,
    Option<"graphPartitioner", "partitioner", "std::string",
      /*default=*/"\"metis\"", "The name of the GraphPartitioner plugin.">
  ];
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

#endif // CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
