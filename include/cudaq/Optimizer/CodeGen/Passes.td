/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
#define CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES

include "mlir/Pass/PassBase.td"

def QuakeToCC : Pass<"quake-to-cc", "mlir::ModuleOp"> {
  let summary = "Lower Quake to CC.";
  let description = [{
    Converts Quake to CC. The `add-dealloc` pass should be run before this pass
    in order to properly generate deallocations for allocated qubits.

    This is the second of 2 passes to lower quake code to cc code. The
    `quake-to-cc-prep` pass must be run first. After these passes the code will
    be in the CC dialect, with calls back to the runtime, specifically the
    execution manager.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];
}

def QuakeToCCPrep : Pass<"quake-to-cc-prep", "mlir::ModuleOp"> {
  let summary = "Prepare for lowering Quake to CC.";
  let description = [{
    Do the preparation work on the module. This pass must be run before the
    `quake-to-cc` pass is run.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect"];
}

def CCToLLVM : Pass<"cc-to-llvm", "mlir::ModuleOp"> {
  let summary = "Convert CC dialect (and others) to LLVM-IR.";
  let description = [{
    Complete the conversion of modules from 'CC dialect' to LLVM-IR dialect.
    This is the final step before generating LLVM code. It would typically be
    used after `-quake-to-cc` when a simulator is the target.
  }];

  let dependentDialects = [
    "cudaq::codegen::CodeGenDialect", "mlir::LLVM::LLVMDialect"
  ];
}

def ConvertToQIR : Pass<"quake-to-qir", "mlir::ModuleOp"> {
  let summary = "Lower Quake to QIR.";
  let description = [{
    Converts Quake to QIR (as LLVM-IR). The add-dealloc pass should be run
    before this pass in order to properly generate deallocations for allocated
    QIR qubits.
  }];

  let dependentDialects = [
    "cudaq::codegen::CodeGenDialect", "mlir::LLVM::LLVMDialect"
  ];
}

def LowerToCG : Pass<"lower-to-cg", "mlir::ModuleOp"> {
  let summary = "Lower Quake to CG dialect.";
  let description = [{
    Lower the Quake IR to the codegen dialect. The codegen dialect is used to
    fuse small DAGs of IR into larger macro operations just prior to the final
    codegen. This allows conversions to take place on the macro operations and
    avoid some of the limitations of an MLIR conversion pass.
  }];
  let dependentDialects = [ "cudaq::codegen::CodeGenDialect" ];
}

def QIRToQIRProfilePrep : Pass<"qir-profile-prep", "mlir::ModuleOp"> {
  let summary =
    "Prepare the IR for rewriting to the base profile or adaptive profile";
  let description = [{
    This is a (module) subpass of the pipeline to convert to a specific QIR
    Profile.

    Before we can convert the functions to the specific profile, we have to do
    a bit of bookkeeping on the module itself. That preparation is done in
    this pass. Specifically, we create all the function declarations that we
    may need and add them to the ModuleOp a priori. This avoids multi-threading
    conflicts when rewriting matching DAGs independently.
  }];

  let constructor = "cudaq::opt::createQIRProfilePreparationPass()";
}

def QIRToQIRProfileFunc : Pass<"quake-to-qir-func",
                            "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Analyze kernels and add attributes and record calls.";
  let description = [{
    This is a (function) subpass of the pipeline to convert to specific QIR
    Profile.

    Each function with a body is analyzed for qubit allocations and qubit
    measurements. Attributes for the total count of qubits are added to the
    LLVMFuncOp. Additionally, measurements are collected and recording marker
    functions are added to the final block in the function.
  }];

  let options = [
    Option<"convertTo", "convert-to", "std::string", /*default=*/"\"qir-base\"",
           "Which QIR profile to convert to (default is 'qir-base')">
  ];

  let constructor = "cudaq::opt::createConvertToQIRFuncPass(\"qir-base\")";
}

def QIRToQIRProfile : Pass<"convert-to-qir-profile"> {
  let summary =
    "Lower full QIR further to the specific QIR Profile.";
  let description = [{
    This is run after lowering a Quake kernel to full QIR.

    This is a subpass of the pipeline to convert to the specific QIR Profile.

    This pass lowers various QIR DAGs to the specific QIR Profile. See
    https://github.com/qir-alliance/qir-spec/blob/main/specification/v0.1/7_Profiles.md
  }];

  let options = [
    Option<"convertTo", "convert-to", "std::string", "\"qir-base\"",
           "Which QIR profile to convert to (default is 'qir-base')">
  ];

  let constructor = "cudaq::opt::createQIRToQIRProfilePass(\"qir-base\")";
}

def QirInsertArrayRecord : Pass<"qir-insert-array-record", "mlir::ModuleOp"> {
  let summary = "Analyze result recording calls and insert new instructions for QIR";
  let description = [{
    This pass runs after QIR prep and conversion passes, then inserts array 
    recording instruction based on the number of result recording calls.
  }];
}

def RemoveMeasurements : Pass<"remove-measurements"> {
  let summary =
    "Remove measurements and output recording calls from a QIR program";
  let description = [{
    Some backends perform implicit measurements on all qubits at the end of the
    kernel, which is a distinct problem for compiler-generated qubits (as is
    done in the multicontrol-decomposition pass) because the user cannot tell
    the difference between measurement results for their qubits and measurement
    results for compiler-generated qubits. This pass enables the compiler to
    strip out the measurements and output recording calls from a QIR program to
    prevent that problem. This pass MUST be run after normal measurements and
    output recording calls have been generated because that metadata must be
    extracted from the program to allow reconstruction of the outputs from the
    backend.
  }];
}

def VerifyNVQIRCallOps :
    Pass<"verify-nvqir-call-ops", "mlir::LLVM::LLVMFuncOp"> {
  let summary =
    "Verify that all LLVM CallOps' callees are NVQIR compliant";
  let description = [{
    This is a function pass to verify that all function calls within the
    function body are referring to pre-defined allowed functions or QIR
    functions, i.e., starting with a __quantum prefix, or NVQIR runtime
    functions.
  }];
  
  let options = [
    ListOption<"allowedFuncs", "allowed-funcs", "llvm::StringRef",
        "Allowed list of functions in addition to NVQIR-defined function.">
  ];

  let constructor = "cudaq::opt::createVerifyNVQIRCallOpsPass({})";
}

def VerifyQIRProfile : Pass<"verify-qir-profile", "mlir::LLVM::LLVMFuncOp"> {
  let summary = "Verify that the output conforms to the specific profile";
  let description = [{
    This pass scans over functions in the LLVM-IR dialect to make sure they
    conform to the QIR specific profile.
  }];

  let options = [
    Option<"convertTo", "convert-to", "std::string", "\"qir-base\"",
           "Which QIR profile to convert to (default is 'qir-base')">
  ];
}

def WireSetToProfileQIR : Pass<"wireset-to-profile-qir", "mlir::func::FuncOp"> {
  let summary = "Convert quake using wire sets to a profile of QIR";
  let description = [{
    This pass takes quake in "value semantics" form and after it has been
    converted to use wire sets (qubit management, mapping, etc.) and converts
    the code to CC dialect with QIR calls, etc.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];
  let options = [
    Option<"convertTo", "convert-to", "std::string", /*default=*/"\"qir-base\"",
           "Select the profile to convert wire sets to.">
  ];
}

def WireSetToProfileQIRPost :
    Pass<"wireset-to-profile-qir-post", "mlir::ModuleOp"> {
  let summary = "Post processing for lowering wire sets to a profile of QIR";
  let description = [{
    This pass should be run immediately after wireset-to-profile-qir.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::func::FuncDialect"];
}

def WireSetToProfileQIRPrep :
    Pass<"wireset-to-profile-qir-prep", "mlir::ModuleOp"> {
  let summary = "Prepare for lowering wire sets to a profile of QIR";
  let description = [{
    This pass should be run immediately before wireset-to-profile-qir.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::func::FuncDialect"];
}

def QuakeToQIRAPI : Pass<"quake-to-qir-api"> {
  let summary = "Convert the Quake dialect to the QIR API.";
  let description = [{
    This pass converts Quake operations to the QIR API as expressed in terms
    of function calls to QIR functions.

    Which QIR functions are to be used is parameterized on the `api` option.

    This pass can lower to either use the obsolete opaque structure types (per
    the QIR spec) or to use LLVM's currently supported opaque pointers. In the
    latter case, type information is fully understood from the function names
    themselves.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::arith::ArithDialect",
      "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
      "mlir::LLVM::LLVMDialect", "cudaq::codegen::CodeGenDialect"];

  let options = [
    Option<"api", "api", "std::string", /*default=*/"\"full\"",
           "Select the QIR API to use.">,
    Option<"opaquePtr", "opaque-pointer", "bool", /*default=*/"false",
           "Use opaque pointers.">
  ];
}

def QuakeToQIRAPIFinal : Pass<"quake-to-qir-api-final", "mlir::ModuleOp"> {
  let summary = "Convert the Quake dialect to the QIR API finalization.";
  let description = [{
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::arith::ArithDialect",
      "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
      "mlir::LLVM::LLVMDialect", "cudaq::codegen::CodeGenDialect"];

  let options = [
    Option<"api", "api", "std::string", /*default=*/"\"full\"",
           "Select the QIR API to use.">
  ];
}

def QuakeToQIRAPIPrep : Pass<"quake-to-qir-api-prep", "mlir::ModuleOp"> {
  let summary = "Convert the Quake dialect to the QIR API preparation.";
  let description = [{
  }];
  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::arith::ArithDialect",
      "mlir::cf::ControlFlowDialect", "mlir::func::FuncDialect",
      "mlir::LLVM::LLVMDialect", "cudaq::codegen::CodeGenDialect"];
  let options = [
    Option<"api", "api", "std::string", /*default=*/"\"full\"",
           "Select the QIR API to use.">,
    Option<"opaquePtr", "opaque-pointer", "bool", /*default=*/"false",
           "Use opaque pointers.">
  ];
}

def ReturnToOutputLog : Pass<"return-to-output-log", "mlir::ModuleOp"> {
  let summary = "Convert a kernel to be compatible with cudaq::run().";
  let description = [{
    When the target supports the cudaq::run() launch function, the kernel's
    return value(s) are translated into QIR output logging functions. This
    conversion allows the kernel to be executed as a group of shots on the QPU
    and a log file produced for the data produced by each kernel execution.
    Effectively, this allows for the benefit of running the kernel as a batch
    of executions and eliminating the overhead of executing the kernel one at
    a time with all the interprocessor overhead.
  }];
  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::func::FuncDialect"];
}


#endif // CUDAQ_OPT_OPTIMIZER_CODEGEN_PASSES
