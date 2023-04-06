/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
#define CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def GenerateKernelExecution : Pass<"kernel-execution", "mlir::ModuleOp"> {
  let summary = "Generate kernel execution code.";
  let description = [{
    Generate the kernel execution thunks. The kernel execution thunks allow
    the control side (C++ code) to launch quantum kernels. This pass
    generates the required glue code.
  }];

  let dependentDialects = ["::cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];
  let constructor = "cudaq::opt::createGenerateKernelExecution()";

  let options = [
    Option<"outputFilename", "output-filename", "std::string",
      /*default=*/"\"-\"", "Name of output file.">,
  ];
}

def ConvertToDirectCalls :
    Pass<"indirect-to-direct-calls", "mlir::func::FuncOp"> {
  let summary = "Convert calls to direct calls to Quake routines.";
  let description = [{
    Rewrite the calls in the IR so that they point to the generated code and not
    to the entry point thunks that call back to the runtime. After this pass
    quantum code will call other quantum code directly and without going
    indirectly through the launch kernel runtime.
  }];

  let constructor = "cudaq::opt::createConvertToDirectCalls()";
}

def GenerateDeviceCodeLoader : Pass<"device-code-loader", "mlir::ModuleOp"> {
  let summary = "Generate device code loader stubs.";
  let description = [{
    Generate device code loader stubs which are used for code introspection
    by the runtime.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
  let constructor = "cudaq::opt::createGenerateDeviceCodeLoader()";

  let options = [
    Option<"outputFilename", "output-filename", "std::string",
      /*default=*/"\"-\"", "Name of output file.">,
    Option<"generateAsQuake", "use-quake", "bool",
      /*default=*/"false", "Output should be module in Quake dialect.">
  ];
}

// ApplyOpSpecialization is a module pass because it may modify the ModuleOp
// and add new FuncOps.
def ApplySpecialization : Pass<"apply-op-specialization", "mlir::ModuleOp"> {
  let summary =
    "Replace quake.apply with conventional calls to specialized functions.";
  let description = [{
    The quake.apply op allows quake kernels to be called with implicit
    specialization of the function itself. For example, a user-defined kernel
    can be called with an optional qvec of control qubits. These extra control
    qubits implicitly create a new and distinct function that threads these
    control qubits to each quantum op in the function.

    The compute-action-opt command-line option is for debug and demonstration
    purposes only.
  }];

  let constructor = "cudaq::opt::createApplyOpSpecializationPass()";

  let options = [
    Option<"computeActionOptimization", "compute-action-opt", "bool",
      /*default=*/"true", "Enable the compute-action control optimization.">
  ];
}

// LambdaLifting is a module pass because it may modify the ModuleOp and add
// new FuncOps.
def LambdaLifting : Pass<"lambda-lifting", "mlir::ModuleOp"> {
  let summary = "Lift lambda expressions to global functions.";
  let description = [{
    This pass implements classical lambda lifting.

    1. Eliminate all free ssa-values in the function by adding arguments.
    2. Move the function to the global scope. (ModuleOp)
    3. Convert all uses to CallOps as needed.
  }];

  let constructor = "cudaq::opt::createLambdaLiftingPass()";
}

def LowerToCFG : Pass<"lower-to-cfg", "mlir::func::FuncOp"> {
  let summary = "Erase CLoop, CIf, etc. ops, replacing them with a CFG.";
  let description = [{
    This pass converts high-level control flow ops to a more primitive basic
    CFG structure.

    For example, a `quake.loop` model of a C `for` loop
    ```mlir
    quake.loop while {
      // while code
      quake.condition %cond
    } do {
      // loop body
    } step {
      // step body
    }
    // exit loop
    ```
    will be rewritten to
    ```mlir
    ^bb23:
      // while code
      cf.cond_br %cond, ^bb24, ^bb26
    ^bb24:
      // loop body
      cf.br ^bb25
    ^bb25:
      // step body
      cf.br ^bb23
    ^bb26:
      // exit loop
    ```

    Note that it may be the case that converting high-level control flow  Ops
    to CFG form will not be as amenable to lowering to QTX or QTX optimizations.
  }];

  let dependentDialects = [ "mlir::cf::ControlFlowDialect" ];
  let constructor = "cudaq::opt::createLowerToCFGPass()";
}

def QuakeSynthesize : Pass<"quake-synth", "mlir::ModuleOp"> {
  let summary =
    "Synthesize concrete quantum program from Quake code plus runtime values.";
  let description = [{
    Convert Quake representing a dynamic quantum kernel to Quake
    representing a concrete quantum program instance using known
     runtime values.
  }];

  let constructor = "cudaq::opt::createQuakeSynthesizer()";
}

def QuakeAddDeallocs : Pass<"quake-add-deallocs", "mlir::func::FuncOp"> {
 let summary = "Add quake deallocs to functions before they return.";
  let description = [{
    This pass ensures that a dealloc is inserted before functions return
    if the function contains a AllocaOp. It should be run before quake-to-qir.
  }];

  let constructor = "cudaq::opt::createQuakeAddDeallocs()";
}

def QuakeAddMetadata : Pass<"quake-add-metadata", "mlir::func::FuncOp"> {
 let summary = "Add various metadata attributes to a Quake function.";
  let description = [{
   This pass provides a hook for adding various metadata to a Quake
   function's list of compile-time attributes. This type of metadata
   exposure will enable quantum IR introspection in the CUDA Quantum
   runtime library.
  }];

  let constructor = "cudaq::opt::createQuakeAddMetadata()";
}

def QuakeObserveAnsatz : Pass<"quake-observe-ansatz", "mlir::func::FuncOp"> {
 let summary = "Given spin_op input, append measures to the Quake FuncOp";
  let description = [{
    Given an unmeasured Quake representation (i.e. a state prep
    ansatz), append measures based on the given spin_op specified in
    binary symplectic form.
  }];
  let constructor = "cudaq::opt::createQuakeObserveAnsatzPass()";
  let options = [
    ListOption<"termBSF", "term-bsf", "unsigned",
      "The measurement bases as a Pauli tensor product represented in binary symplectic form.">
  ];
}

def RaiseToAffine : Pass<"raise-to-affine", "mlir::func::FuncOp"> {
  let summary = "Convert CLoop ops to affine.for loops when possible.";
  let description = [{
    This pass converts quake.CLoop ops to a more structured loop operation,
    affine.for.

    For example, a `quake.loop` model of a C `for` loop
    ```mlir
    quake.scope {
      %i0 = constant 0 : index
      quake.loop while (%i1 = %i0) -> index {
        %c10 = constant 10 : index
        %cond = arith.cmpi %i1, %c10
        quake.condition %cond, %i1
      } do {
        ^bb1 (%i2 : index):
          %19 = quake.qubit_extract %qreg[%i2 : index] : !quake.qubit
          %c1 = costant 1 : index
          %i2p1 = arith.addi %i2, %c1 : index
          %23 = quake.qubit_extract %qreg[%i2p1 : index] : !quake.qubit
          quake.x [%19] %23
          quake.continue %i2 : index
      } step {
        ^bb2 (%i3 : index):
          %c1 = constant 1 : index
          %incr = arith.addi %i3, %c1
          quake.continue %incr : index
      }
    }
    // exit loop
    ```
    would be rewritten to
    ```mlir
    affine.for %arg1 = #map(%c0) to #map(%c9) {
      %19 = quake.qubit_extract %qreg[%arg1 : index] : !quake.qubit
      %21 = affine.apply affine_map<(i) -> (i+1)> (%arg1)
      %23 = quake.qubit_extract %qreg[%21 : index] : !quake.qubit
      quake.x [%19] %23 : !quake.qubit
    }
    ```
  }];
  let dependentDialects = [ "mlir::AffineDialect" ];
  let constructor = "cudaq::opt::createRaiseToAffinePass()";
}

def CCMemToReg : Pass<"cc-mem2reg", "mlir::func::FuncOp"> {
  let summary = "Converts classical compute memory to register SSA form.";
  let description = [{
    For cc.alloca ops on scalar values, this pass will convert the alloca op
    to register form when the memory reference does not escape.
  }];
  let constructor = "cudaq::opt::createCCMemToRegPass()";
}

def ExpandMeasurements : Pass<"expand-measurements"> {
  let summary = "Expand multi-ref measurements to series on single refs.";
  let description = [{
    The `mx`, `my`, `mz` ops can take a list of qubits and/or qvec arguments.
    The target may only support measuring a single qubit however. This pass
    expands these ops in list format into a series of measurements (including
    loops) on individual qubits and into a single `std::vector<bool>` result.
  }];

  let dependentDialects = ["::cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];

  let constructor = "cudaq::opt::createExpandMeasurementsPass()";
}

def LoopUnroll : Pass<"cc-loop-unroll"> {
  let summary = "Unroll classical compute (C++) loops.";
  let description = [{
    If a cc.loop op is a simple, constant counted loop, it can be fully
    unrolled into <i>n</i> copies of the body of the loop.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect"];

  let constructor = "cudaq::opt::createLoopUnrollPass()";

  let options = [
    Option<"threshold", "maximum-iterations", "std::size_t",
      /*default=*/"50", "Maximum iterations to unroll.">,
  ];
}

def UnwindLowering : Pass<"unwind-lowering", "mlir::func::FuncOp"> {
  let summary = "Lower global unwinding control-flow macros to a CFG.";
  let description = [{
    In C++, statements such as `return`, `break`, and `continue` are global
    transfers of control. The control-flow does not necessarily return to the
    next Operation in the Block containing the parent Op. Furthermore, this
    global jumps may imply additional operations depending on the context.
    These operations are stack unwinding operations (within an activation
    frame) that may include calling destructors, deallocations, automatic
    uncomputation, etc.

    `continue` and `break` may only be present in the context of a loop
    currently. The unwinding macros are matched to the nearest enclosing loop
    structure and the body of the loop is expanded to a CFG to expose the
    general control flow within the loop.

    `return` may be present in any context. The full context from the innermost
    structured op containing the unwinding return to the body of the function
    is lowered to a CFG form to expose the general control flow implied.

    The lower to CFG pass removed all structured operations from a function,
    lowering the body of the function completely to a primitive CFG.
  }];
  let constructor = "cudaq::opt::createUnwindLoweringPass()";
}

def OpCancellation : Pass<"qtx-op-cancellation", "qtx::CircuitOp"> {
  let summary = "Remove pairs of adjacent adjoint operations.";
  let description = [{
    This pass performs basic quantum operation cancellation: it traverses
    a 'circuit' and removes pairs of adjacent adjoint operations.  This
    optimization is purely structural, and its effectiveness is highly
    dependent on canonicalization.

    Note: This is a low-effort pass that only take into account commutations
    between operations that share the a subset of controls.
  }];
  let constructor = "cudaq::opt::createOpCancellationPass()";
}

def OpDecomposition: Pass<"qtx-op-decomposition", "qtx::CircuitOp"> {
  let summary = "Remove pairs of adjacent adjoint operations.";
  let description = [{
    Decomposition is the process of _systematically_ breaking down
    high-level instruction (operations) into a sequence of lower level ones.
    We emphasize "systematically" because it is the characteristic that
    differentiates decompostion from synthesis.  When doing deomposition, we
    generate the sequence of lower-level instructions by applying some
    construction rule(s).  Decomposition techniques are faster than synthesis
    and produce predictable results, i.e., we know in advance the cost in number
    of low-level instructions and in the number of qubits.

    NOTE: The implementation still preliminary.
  }];
  let constructor = "cudaq::opt::createOpDecompositionPass()";
}

def SplitArrays: Pass<"qtx-split-arrays", "qtx::CircuitOp"> {
  let summary = "Split arrays.";
  let description = [{
    There are some quantum languages, e.g. Quil, that do not have the notion of
    quantum register (array, vector).  Therefore, when targeting such languages,
    it is necessary to split all arrays.

    NOTE: We cannot split all arrays.  For example, we cannot split those which
    are indexed by runtime arguments.  In such cases, the default behavior is to
    leave those arrays alone and return success nevertheless.  This behavior can
    be configured using the `allow-partial-split` option, when this option is
    set to `false`, the pass will fail if cannot split all arrays.
  }];
  let constructor = "cudaq::opt::createSplitArraysPass()";
  let options = [
    Option<"allowPartialSplit", "allow-partial-split", "bool",
           /*default=*/"true", "Allow arrays that cannot be split">,
  ];
}

def ConvertQTXToQuake : Pass<"convert-qtx-to-quake", "mlir::ModuleOp">{
  let summary = "Convert QTX dialect to Quake dialect";
  let dependentDialects = ["::cudaq::cc::CCDialect", "quake::QuakeDialect",
                           "mlir::func::FuncDialect"];
  let constructor = "cudaq::opt::createConvertQTXToQuakePass()";
}

def ConvertQuakeToQTX : Pass<"convert-quake-to-qtx", "mlir::func::FuncOp">{
  let summary = "Convert Quake dialect to QTX dialect";
  let description = [{
    This pass assumes quake-synth and quake-add-deallocs have run.
  }];
  let dependentDialects = ["qtx::QTXDialect"];
  let constructor = "cudaq::opt::createConvertQuakeToQTXPass()";
}

def ConvertFuncToQTX : Pass<"convert-func-to-qtx", "mlir::ModuleOp">{
  let summary = "Convert Func dialect to QTX dialect";
  let description = [{
  }];
  let dependentDialects = ["qtx::QTXDialect"];
  let constructor = "cudaq::opt::createConvertFuncToQTXPass()";
}

#endif // CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
