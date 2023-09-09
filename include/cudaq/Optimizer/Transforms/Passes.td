/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
#define CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// ApplyOpSpecialization is a module pass because it may modify the ModuleOp
// and add new FuncOps.
def ApplySpecialization : Pass<"apply-op-specialization", "mlir::ModuleOp"> {
  let summary =
    "Replace quake.apply with conventional calls to specialized functions.";
  let description = [{
    The quake.apply op allows quake kernels to be called with implicit
    specialization of the function itself. For example, a user-defined kernel
    can be called with an optional veq of control qubits. These extra control
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

def CheckKernelCalls : Pass<"check-kernel-calls", "mlir::func::FuncOp"> {
  let summary = "Check calls between quantum kernels have been inlined.";
  let description = [{
    The aggressive inlining pass should fully inline any calls between quantum
    kernels. Any residual calls will be because the call graph contains cycles.
    This pass checks that there are no residual calls and print a diagnostic if
    any are found.
  }];
}

def CombineQuantumAllocations :
    Pass<"combine-quantum-alloc", "mlir::func::FuncOp"> {
  let summary = "Combines quake alloca operations.";
  let description = [{
    Quake code may contain several distinct `quake.alloca` operations prior to
    final code gen. This pass will combine them into a single allocation of
    type `!quake.veq` with appropriate `quake.extract_ref` operations. The
    combined allocation will be placed in the entry block and thus dominate all
    potential uses.

    This pass will only process `quake.alloca` operations at the top-level of
    a function. It assumes all calls have been inlined, loops unrolled, etc.

    If the function contains deallocations, these are combined as well. The
    combined deallocation will be added to each exit block.
  }];
  let dependentDialects = ["cudaq::cc::CCDialect", "quake::QuakeDialect"];
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
}

def ExpandMeasurements : Pass<"expand-measurements"> {
  let summary = "Expand multi-ref measurements to series on single refs.";
  let description = [{
    The `mx`, `my`, `mz` ops can take a list of qubits and/or veq arguments.
    The target may only support measuring a single qubit however. This pass
    expands these ops in list format into a series of measurements (including
    loops) on individual qubits and into a single `std::vector<bool>` result.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];

  let constructor = "cudaq::opt::createExpandMeasurementsPass()";
}

def FactorQuantumAllocations :
    Pass<"factor-quantum-alloc", "mlir::func::FuncOp"> {
  let summary = "Factors quake alloca operations.";
  let description = [{
    Quake code may contain `quake.alloca` operations that allocate vectors of
    qubits (`!quake.veq`). This pass will factor these single allocations of
    type `!quake.veq` and the associated `quake.extract_ref` operations into a
    series of single qubit allocations, thereby eliminating the
    `quake.extract_ref` operations.

    If the function contains deallocations of quantum vectors (`veq`), these
    will be replaced with a series of deallocations.
  }];
  let dependentDialects = ["quake::QuakeDialect"];
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

def GenerateKernelExecution : Pass<"kernel-execution", "mlir::ModuleOp"> {
  let summary = "Generate kernel execution code.";
  let description = [{
    Generate the kernel execution thunks. The kernel execution thunks allow
    the control side (C++ code) to launch quantum kernels. This pass
    generates the required glue code.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];
  let constructor = "cudaq::opt::createGenerateKernelExecution()";

  let options = [
    Option<"outputFilename", "output-filename", "std::string",
      /*default=*/"\"-\"", "Name of output file.">,
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

def LoopNormalize : Pass<"cc-loop-normalize"> {
  let summary = "Normalize classical compute (C++) loops.";
  let description = [{
    Transform a monotonic loop with constant step (slope) into an invariant loop
    or, if the bounds are constant, a simple counted loop.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def LoopUnroll : Pass<"cc-loop-unroll"> {
  let summary = "Unroll classical compute (C++) loops.";
  let description = [{
    If a cc.loop op is a simple, constant counted loop, it can be fully
    unrolled into <i>n</i> copies of the body of the loop.

    The signal-failure-if-any-loop-cannot-be-completely-unrolled option controls
    whether to signal a failure if all loops cannot be fully unrolled. This is
    necessary when synthesizing quantum circuits from CUDA Quantum kernels, such
    as when generating a QIR base profile. A quantum circuit requires all loops
    be completely unrolled.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::cf::ControlFlowDialect"];

  let options = [
    Option<"threshold", "maximum-iterations", "unsigned", /*default=*/"50",
      "Maximum iterations to unroll.">,
    Option<"signalFailure",
      "signal-failure-if-any-loop-cannot-be-completely-unrolled", "bool",
      /*default=*/"false", "Signal failure if pass can't unroll all loops.">
  ];
}

def UpdateRegisterNames : Pass<"update-register-names"> {
  let summary = "Update classical register names";
  let description = [{
    After loop unrolling, there may be duplicate registerName attributes. Run
    this pass to give them unique attribute values.
  }];
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
  }];

  let dependentDialects = [ "mlir::cf::ControlFlowDialect" ];
  let constructor = "cudaq::opt::createLowerToCFGPass()";
}

def MemToReg : Pass<"memtoreg", "mlir::func::FuncOp"> {
  let summary = "Converts memory-SSA to register-SSA form.";
  let description = [{
    When classical is enabled, loads and stores of simple scalars that are stack
    allocated are converted to register form. The values themselves are promoted
    out of memory and passed directly. Store to load forwarding and the addition
    of block arguments is performed to eliminate the loads and stores.

    When quantum is enabled, this pass converts values of type `!quake.ref` to
    values of type `!quake.wire`. A `ref` value has ordinary SSA semantics. It
    is possible to make copies of its value, pass the ref value, operate on the
    quantum information by reference, etc. A `wire` value does not have ordinary
    SSA semantics. A use can destroy the value itself, therefore it is required
    that `wire` values be threads in and out of each and every use.
  }];
  let dependentDialects = ["cudaq::cc::CCDialect", "quake::QuakeDialect"];

  let options = [
    Option<"classicalValues", "classical", "bool",
      /*default=*/"true", "Promote classical stack slots to values.">,
    Option<"quantumValues", "quantum", "bool",
      /*default=*/"true", "Promote of quantum values.">
  ];
}

def PruneCtrlRelations : Pass<"pruned-ctrl-form", "mlir::func::FuncOp"> {
  let summary = "Removes artifical control constraints between quantum ops.";
  let description = [{
    In the value semantics, quantum gates may be overly constrained in terms of
    control qubits. These constraints can be removed within the IR to get a more
    accurate representation of the dataflow of the *logical* qubits. This
    alternative representation may be better suited to certain transformations.

    The following example is overly constrained. The control wire, beginning
    with `%0` is imposing what appears to be a strict ordering of
    `[x -> h -> y]`.
    ```mlir
      %3:2 = x [%0] %1 : (!wire, !wire) -> (!wire, !wire)
      %4:2 = h [%3#0] %2 : (!wire, !wire) -> (!wire, !wire)
      %5:2 = y [%4#0] %3#1 : (!wire, !wire) -> (!wire, !wire)
    ```

    This snippet can be rewritten as the following IR to reflect that the
    control wire has an identity applied so is unchanged. We observe less strict
    ordering constraints of `[h | x -> y]`. This might allow a scheduling pass
    to swap the order of the `h` and `x` gates, for instance.
    ```mlir
      %ctrl = to_ctrl %old.0 : (!wire) -> !control
      %3 = x [%ctrl] %1 : (!control, !wire) -> !wire
      %4 = h [%ctrl] %2 : (!control, !wire) -> !wire
      %5 = y [%ctrl] %3 : (!control, !wire) -> !wire
      %new.0 = from_ctrl %ctrl : (!control) -> !wire
    ```
  }];
  let dependentDialects = ["quake::QuakeDialect"];
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

def RegToMem : Pass<"regtomem", "mlir::func::FuncOp"> {
  let summary = "Converts register-SSA to memory-SSA form.";
  let description = [{
    Perform the reverse transformation of the memtoreg pass.

    This pass converts values of type `!quake.wire` to be wrapped as references
    where possible. This is an greedy/exhaustive transformation in that wires
    will be removed from the Quake IR wherever possible. There may be cases
    where the wire cannot be wrapped uniquely. In those cases, this
    transformation will take no action and leave the IR (partially) with wire
    type values.
  }];
  let dependentDialects = ["cudaq::cc::CCDialect", "quake::QuakeDialect"];
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

def BasisConversionPass: Pass<"quake-basis-conversion", "mlir::ModuleOp"> {
  let summary = "Converts kernels to a set of basis operations.";
  let description = [{
    This pass takes as input a list of target (allowed) quantum operations.
    It builds a conversion target, and uses it, together with the dialect
    conversion driver, to try converting all operations into the defined basis.

    The `basis` option takes a comma-separated list of elements identifying
    specific quantum operations as input. Each element must have the
    following format:

    ```
    <op-name>(`(` [<number-of-controls> | `n`] `)` )?
    ```

    Examples:
    - `x` means targeting pauli-x operations without controls  (aka, `not`)
    - `x(1)` means targeting pauli-x operations with one control (aka, `cx`)
    - `x(n)` means targeting pauli-x operation with unbounded number of controls
    - `x,x(1)` means targeting both `not` and `cx` operations
  }];
  let options = [
    ListOption<"basis", "basis", "std::string", "Set of basis operations">,
    ListOption<"disabledPatterns", "disable-patterns", "std::string",
               "Labels of decomposition patterns that should be filtered out">,
    ListOption<"enabledPatterns", "enable-patterns", "std::string",
               "Labels of decomposition patterns that should be used, all "
               "other patterns are filtered out">,
  ];
}

def DecompositionPass: Pass<"quake-decomposition", "mlir::ModuleOp"> {
  let summary = "Break down quantum operations.";
  let description = [{
    This pass performs decomposition over a set of operations by iteratively
    applying decomposition patterns until either a fixpoint is reached or the
    maximum number of iterations/rewrites is exhausted. Decomposition is
    best-effort and does not guarantee that the entire IR is decomposed after
    running this pass.

    NOTE: The current implementation is conservative w.r.t global phase, which
    means no decomposition will take place under the presence of controlled
    `quake.apply` operations in the module.
  }];
  let options = [
    ListOption<"disabledPatterns", "disable-patterns", "std::string",
               "Labels of decomposition patterns that should be filtered out">,
    ListOption<"enabledPatterns", "enable-patterns", "std::string",
               "Labels of decomposition patterns that should be used, all "
               "other patterns are filtered out">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic patterns">,
  ];
}

def MultiControlDecompositionPass: Pass<"quake-multicontrol-decomposition",
                                        "mlir::func::FuncOp"> {
  let summary = "Break down multi-control quantum operations.";
  let description = [{
    This pass decomposes multi-control quantum operations. The decompostion
    involves allocating new qubits to hold intermediate results. The number of
    extra qubits depends on the particular operation being decomposed.
    Pauli-X and Pauli-Z operations add _N_ - 2 qubits, while other operations
    add _N_ - 1 qubits, where _N_ is the number of controls.

    Note: When a `veq` is used as control, we need to know its size to be able
    to decompose. In such cases, all qubits will be extracted. If the size is
    unknown at compilation-time, the pass leaves the operation as-is.
  }];
}

#endif // CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
