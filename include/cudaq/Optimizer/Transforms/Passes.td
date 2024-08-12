/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
#define CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def AddWireset : Pass<"add-wireset", "mlir::ModuleOp"> {
  let summary = "Adds a topology-less `quake.wire_set` to the module.";
  let description = [{
    Adds a `quake.wire_set` operation without any topological information to the
    ModuleOp.
  }];
}

def AssignWireIndices : Pass<"assign-wire-indices", "mlir::func::FuncOp"> {
  let summary = "Replaces `null_wire` with `borrow_wire` from a `wire_set`.";
  let description = [{
    Replaces all instances of `quake.null_wire` with `quake.borrow_wire` from a
    common `quake.wire_set` without any topological information.  Each wire is
    assigned a unique identifier (the index into the `quake.wire_set`) through
    this process.
  }];
}

def ApplyControlNegations :
    Pass<"apply-control-negations", "mlir::func::FuncOp"> {
  let summary =
    "Replace all `Ops` with negated controls with normal controls and `XOp`s.";

  let description = [{
    For every quantum operation with a negative control, replace that operation 
    with an X operation on the control qubit, the controlled operation with
    positive polarity, and a final X operation on the control qubit. 
  }];
}

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

def ArgumentSynthesis : Pass<"argument-synthesis", "mlir::func::FuncOp"> {
  let summary = "Specialize a function by replacing arguments with constants";
  let description = [{
    This pass takes a list of functions and argument substitutions. For each
    function in the list, the arguments to the function in the substitutions
    list will be erased and replaced with a computed value (e.g., a constant)
    provided in the substitution list. All arguments or some subset of arguments
    may be substituted in this way.

    To facilitate command-line testing, this pass can be run with the functions
    suboption using filenames containing the argument substitutions. For
    example, one might run
    ```console
      cudaq-opt input.qke \
        --argument-synthesis=functions="kernel1:subst1.qke,kernel2:subst2.qke"
    ```
    where `kernel1`, `kernel2` are the names of functions and `subst1.qke` and
    `subst2.qke` are quake source files contains arg_subst operations.

    For running this pass from code, one can build the substitution code in a
    std::string and use a prefix character '*' to indicate the text is inline
    and not in a file.
    ```
      kernel1:*"cc.arg_subst [0] { ... }"
    ```
  }];

  let options = [
    ListOption<"funcList", "functions", "std::string",
               "Function name and substitutions pairs (<func>:<filename>)">,
  ];
  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect",
    "mlir::cf::ControlFlowDialect"];
}

def BasisConversionPass : Pass<"basis-conversion", "mlir::ModuleOp"> {
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

def ConstPropComplex : Pass<"const-prop-complex", "mlir::ModuleOp"> {
  let summary = "Create and propagate complex constants.";
  let description = [{
    Rewrite the complex.CreateOp to complex.ConstantOp when possible.
    Replace array pointer casts with element pointer casts.
  }];
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

def DelayMeasurements : Pass<"delay-measurements", "mlir::func::FuncOp"> {
  let summary =
    "Move measurements as late as possible";

  let description = [{
    Move measurements as late as possible. This is useful for a Base Profile QIR program.
  }];

  let constructor = "cudaq::opt::createDelayMeasurementsPass()";
}

def DecompositionPass: Pass<"decomposition", "mlir::ModuleOp"> {
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

def EraseNopCalls : Pass<"erase-nop-calls", "mlir::func::FuncOp"> {
  let summary = "Erase calls to any builtin intrinsics that are NOPs.";
  let description = [{
    The code may contain marker function calls that do not generate any actual
    code. These calls are NOPs that will be erased by this pass.
  }];
}

def ExpandControlVeqs: Pass<"expand-control-veqs", "mlir::func::FuncOp"> {
  let summary = "Expands veqs used as controls into individual qubits.";
  let description = [{
    Given an operation of the form
    ```mlir
    quake.any [%veq] %r : (!quake.veq<n>, !quake.ref) -> ()
    ```
    this pass will extract each qubit from `%veq%` and explicitly provide them
    as qubits:
    ```mlir
    %arg0 = quake.extract_ref %veq[0] : (!quake.veq<n>) -> !quake.ref
    ...
    %argn = quake.extract_ref %veq[n] : (!quake.veq<n>) -> !quake.ref
    quake.any [%arg0, ..., %argn] %0 : (!quake.ref, ..., !quake.ref, !quake.ref) -> ()
    ```
  }];
}

def ExpandMeasurements : Pass<"expand-measurements"> {
  let summary = "Expand multi-ref measurements to series on single refs.";
  let description = [{
    The `mx`, `my`, `mz` ops can take a list of qubits and/or veq arguments.
    The target may only support measuring a single qubit however. This pass
    expands these ops in list format into a series of measurements (including
    loops) on individual qubits and into a single `std::vector<bool>` result.

    The `reset` op can also take a veq argument and this pass will also expand
    that to a series of `reset` operations on single qubits.
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

    Specifying the alt-launch=2 option will generate different code that makes
    use of library side argument conversion and the argument synthesis pass.
    More generally, this option can be used when JIT compiling kernels on the
    client/host/local processor.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"outputFilename", "output-filename", "std::string",
      /*default=*/"\"-\"", "Name of output file.">,
    Option<"startingArgIdx", "starting-arg-idx", "std::size_t", /*default=*/"0",
      "The starting argument index for the argsCreator.">,
    Option<"altLaunchVersion", "alt-launch", "std::size_t", /*default=*/"1",
      "Specify the version of altLaunchKernel to be used.">
  ];
}

def GetConcreteMatrix : Pass<"get-concrete-matrix", "mlir::func::FuncOp"> {
  let summary =
    "Replace the unitary matrix generator function with concrete matrix.";
  let description = [{
    Given a custom operation whose generator attribute is another function 
    within the module, such that if `LiftArrayAlloc` pass has run, there will
    be a global constant within the module which holds the concrete matrix 
    representation for the custom operation. This pass will find that global
    variable and update the custom operation to directly point to it. 

    Example:
    ```mlir
      module {
        func.func @__nvqpp__mlirgen__function_foo_generator_1.bar(%arg0: !cc.stdvec<f64>) -> !cc.stdvec<complex<f64>> {
          ...
          %0 = cc.address_of @__nvqpp__mlirgen__function_foo_generator_1.bar.rodata_0 : !cc.ptr<!cc.array<complex<f64> x 4>>
          ...
          return %3 : !cc.stdvec<complex<f64>>
        }

        func.func @__nvqpp__mlirgen__function_kernel_1._Z8kernel_1v() {
          %0 = quake.alloca !quake.ref
          quake.custom_op @__nvqpp__mlirgen__function_foo_generator_1.bar %0 : (!quake.ref) -> ()
          return
        }

        cc.global constant @__nvqpp__mlirgen__function_foo_generator_1.bar.rodata_0 ((dense<[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
      }
    ```

    The `quake.custom_op` call would be converted to

    ```mlir
      func.func @__nvqpp__mlirgen__function_kernel_1._Z8kernel_1v() {
        %0 = quake.alloca !quake.ref
        quake.custom_op @__nvqpp__mlirgen__function_foo_generator_1.bar.rodata_0 %0 : (!quake.ref) -> ()
        return
      }
    ```

  }];
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

def LiftArrayAlloc : Pass<"lift-array-value", "mlir::ModuleOp"> {
  let summary = "Convert constant arrays built on the stack to array values";
  let description = [{
    The bridge or other passes may generate inline code to build an array of
    arithmetic types. This construction can involve quite a few CC dialect
    operations and can "hide" what is really being done in the volume of that
    code. This pass folds and lifts those memory operations into a constant
    array value operation.

    Example:
    ```mlir
      %cst = arith.constant 5.000000e+00 : f64
      %cst_0 = arith.constant 6.000000e+00 : f64
      %cst_1 = arith.constant 7.000000e+00 : f64
      %cst_2 = arith.constant 8.000000e+00 : f64
      %0 = cc.alloca !cc.array<f64 x 4>
      %1 = cc.compute_ptr %0[0] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
      cc.store %cst, %1 : !cc.ptr<f64>
      %2 = cc.compute_ptr %0[1] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
      cc.store %cst_0, %2 : !cc.ptr<f64>
      %3 = cc.compute_ptr %0[2] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
      cc.store %cst_1, %3 : !cc.ptr<f64>
      %4 = cc.compute_ptr %0[3] : (!cc.ptr<!cc.array<f64 x 4>>) -> !cc.ptr<f64>
      cc.store %cst_2, %4 : !cc.ptr<f64>
    ```

    would be converted to

    ```mlir
      %0 = cc.const_array [5.0, 6.0, 7.0, 8.0] : !cc.array<f64 x 4>
    ```

    This converts a value in memory SSA form to an SSA value, so additional
    uses must also be considered. For example, if the array is subsequently
    updated or escapes the function, it cannot be replaced by a value. If
    it is elements are accessed in a read-only way, it is a legal transform
    and will enable further constant folding in other passes.
  }];

  let dependentDialects = ["mlir::complex::ComplexDialect"];
}

def LinearCtrlRelations : Pass<"linear-ctrl-form", "mlir::func::FuncOp"> {
  let summary = "Removes control type values between quantum ops.";
  let description = [{
    In the value semantics, quantum gates may be factored in terms of control
    qubits by the introduction of values of type `!quake.control`. These
    relaxed constraints can be removed within the IR to get a linear and overly
    constrained representation of the dataflow of the *logical* qubits. The
    overly constrained representation may be better suited to certain
    transformations.

    The following example is a factored value semantics. The wire type value is
    converted to a control type value, `%ctrl`, which is a proper SSA-value
    until it is converted back to a wire type, `%new.0`.
    ```mlir
      %ctrl = to_ctrl %old.0 : (!wire) -> !control
      %3 = x [%ctrl] %1 : (!control, !wire) -> !wire
      %4 = h [%ctrl] %2 : (!control, !wire) -> !wire
      %5 = y [%ctrl] %3 : (!control, !wire) -> !wire
      %new.0 = from_ctrl %ctrl : (!control) -> !wire
    ```

    Linearizing these control type values increases the coarity of the quantum
    operations and prevents reordering the operations without rewiring the
    use-def chains of wire values in control positions.
    ```mlir
      %3:2 = x [%0] %1 : (!wire, !wire) -> (!wire, !wire)
      %4:2 = h [%3#0] %2 : (!wire, !wire) -> (!wire, !wire)
      %5:2 = y [%4#0] %3#1 : (!wire, !wire) -> (!wire, !wire)
    ```
  }];
}

def LoopNormalize : Pass<"cc-loop-normalize"> {
  let summary = "Normalize classical compute (C++) loops.";
  let description = [{
    Transform a monotonic loop with constant step (slope) into an invariant loop
    or, if the bounds are constant, a simple counted loop.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect"];

  let options = [
    Option<"allowClosedInterval", "allow-closed-iterval", "bool",
      /*default=*/"true", "Allow loop iterations on a closed interval.">,
    Option<"allowBreak", "allow-early-exit", "bool", /*default=*/"true",
      "Allow unrolling of loop with early exit (i.e. break statement).">
  ];
}

def LoopPeeling : Pass<"cc-loop-peeling"> {
  let summary = "Peeling classical do-while loops.";
  let description = [{
    This loop peeling pass currently implements the transformation of a C++
    do-while loop into a peeled body and while sequence.

    ```c++
    do {
      body;
    } while (condition);
    ```
    is transformed to
    ```c++
    body;
    while (condition) {
      body;
    }
    ```

    Other cases of loop peeling may be implemented in the future.
  }];

  let dependentDialects = ["mlir::cf::ControlFlowDialect"];
}

def LoopUnroll : Pass<"cc-loop-unroll"> {
  let summary = "Unroll classical compute (C++) loops.";
  let description = [{
    If a cc.loop op is a simple, constant counted loop, it can be fully
    unrolled into <i>n</i> copies of the body of the loop.

    The signal-failure-if-any-loop-cannot-be-completely-unrolled option controls
    whether to signal a failure if all loops cannot be fully unrolled. This is
    necessary when synthesizing quantum circuits from CUDA-Q kernels, such
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
      /*default=*/"false", "Signal failure if pass can't unroll all loops.">,
    Option<"allowBreak", "allow-early-exit", "bool", /*default=*/"false",
      "Allow unrolling of loop with early exit (i.e. break statement).">
  ];
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

def MappingPass: Pass<"qubit-mapping", "mlir::func::FuncOp"> {
  let summary = "Perform qubit mapping to account for connectivity constraints";
  let description = [{
    Some backends cannot support any-to-any multi-qubit operations, so this pass
    performs mapping of the qubits by inserting the necessary swap operations
    into the Quake IR.

    Note 1: this pass requires strictly value semantics for any quantum
    operations. It will throw an error if any memory reference semantics are
    present on quantum operations.

    Note 2: this pass can introduce ancilla qubits.

    Note 3: as a result of note 2, if the IR contains no measurements, this pass
    will inject measurements so that the post-mapping measurements correspond
    to all of the input (user) qubits.
  }];

  let options = [
    Option<"device", "device", "std::string", /*default=*/"\"-\"",
      "Device topology: path(N), ring(N), star(N), star(N,c), grid(w,h), file(/path/to/file), bypass">,
    Option<"debug", "debug", "bool", /*default=*/"false", "Enable debug for qubit-mapping pass algorithm">,
    Option<"extendedLayerSize", "extendedLayerSize", "unsigned", /*default=*/"20", "Extended layer size">,
    Option<"extendedLayerWeight", "extendedLayerWeight", "float", /*default=*/"0.5", "Extended layer weight">,
    Option<"decayDelta", "decayDelta", "float", /*default=*/"0.5", "Decay delta">,
    Option<"roundsDecayReset", "roundsDecayReset", "unsigned", /*default=*/"5", "Number of rounds before decay is reset">
  ];
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

def MultiControlDecompositionPass: Pass<"multicontrol-decomposition",
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

def ObserveAnsatz : Pass<"observe-ansatz", "mlir::func::FuncOp"> {
 let summary = "Given spin_op input, append measures to the FuncOp";
  let description = [{
    Given an unmeasured Quake representation (i.e. a state prep ansatz), append
    measures based on the given spin_op specified in binary symplectic form.
  }];
  let options = [
    ListOption<"termBSF", "term-bsf", "unsigned",
      "The measurement bases as a Pauli tensor product represented in binary symplectic form.">
  ];
}

def StatePreparation : Pass<"state-prep", "mlir::ModuleOp"> {
  let summary =
    "Convert state vector data into gates";
  let description = [{
    Convert quake representation that includes qubit initialization
    from data into qubit initialization using gates.

    For example:

    ```mlir
    module {
      func.func @foo() attributes {
        %0 = cc.address_of @foo.rodata_0 : !cc.ptr<!cc.array<complex<f32> x 4>>
        %1 = quake.alloca !quake.veq<2>
        %2 = quake.init_state %1, %0 : (!quake.veq<2>, !cc.ptr<!cc.array<complex<f32> x 4>>) -> !quake.veq<2>
        return
      }
      cc.global constant @foo.rodata_0 (dense<[(0.707106769,0.000000e+00), (0.707106769,0.000000e+00), (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> : tensor<4xcomplex<f32>>) : !cc.array<complex<f32> x 4>
    }
    ```
    Will be rewritten to:
    ```mlir
    module {
      func.func @foo() attributes {
        %0 = quake.alloca !quake.veq<2>
        %c1_i64 = arith.constant 1 : i64
        %1 = quake.extract_ref %0[%c1_i64] : (!quake.veq<2>, i64) -> !quake.ref
        %cst = arith.constant 0.000000e+00 : f64
        quake.ry (%cst) %1 : (f64, !quake.ref) -> ()
        %c0_i64 = arith.constant 0 : i64
        %2 = quake.extract_ref %0[%c0_i64] : (!quake.veq<2>, i64) -> !quake.ref
        %cst_0 = arith.constant 0.78539816339744839 : f64
        quake.ry (%cst_0) %2 : (f64, !quake.ref) -> ()
        quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
        %cst_1 = arith.constant 0.78539816339744839 : f64
        quake.ry (%cst_1) %2 : (f64, !quake.ref) -> ()
        quake.x [%1] %2 : (!quake.ref, !quake.ref) -> ()
        return
      }
    }
    ```
  }];

  let options = [
    Option<"phaseThreshold", "threshold", "double",
      /*default=*/"1e-10", "Threshold to trigger phase equalization">,
  ];
}

def PromoteRefToVeqAlloc : Pass<"promote-qubit-allocation"> {
  let summary = "Promote single qubit allocations.";
  let description = [{
    This pass converts all single qubit allocations in the quake dialect to
    allocations of vectors of qubits of length one. This conversion makes all
    allocations uniform for the conversion to QIR.
  }];
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
}

def PySynthCallableBlockArgs :
    Pass<"py-synth-callable-block-args", "mlir::func::FuncOp"> {
  let summary = "Synthesize / Inline cc.callable_func on function block arguments.";
  let description = [{
    This pass is leveraged by the Python bindings to synthesize any 
    cc.callable block arguments. By synthesis we mean replace all uses of the 
    callable block argument with a specific in-Module function call (func.call) 
    retrieved at runtime (the name of the function passed to the kernel at the 
    cc.callable block argument index).
  }];
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

def QuakeAddDeallocs : Pass<"add-dealloc", "mlir::func::FuncOp"> {
 let summary = "Add quake deallocs to functions before they return.";
  let description = [{
    This pass ensures that a dealloc is inserted before functions return
    if the function contains a AllocaOp. It should be run before converting
    to QIR, for example, in order to generate correct code.
  }];

  let constructor = "cudaq::opt::createQuakeAddDeallocs()";
}

def QuakeAddMetadata : Pass<"quake-add-metadata", "mlir::func::FuncOp"> {
 let summary = "Add various metadata attributes to a Quake function.";
  let description = [{
   This pass provides a hook for adding various metadata to a Quake
   function's list of compile-time attributes. This type of metadata
   exposure will enable quantum IR introspection in the CUDA-Q
   runtime library.
  }];

  let constructor = "cudaq::opt::createQuakeAddMetadata()";
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

def UpdateRegisterNames : Pass<"update-register-names"> {
  let summary = "Update classical register names";
  let description = [{
    After loop unrolling, there may be duplicate registerName attributes. Run
    this pass to give them unique attribute values.
  }];
}

#endif // CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
