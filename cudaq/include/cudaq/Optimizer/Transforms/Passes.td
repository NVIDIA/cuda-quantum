/********************************************************** -*- tablegen -*- ***
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#ifndef CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
#define CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

// NB: Before approving changes to this file make sure the author understands
// the passes are listed in alphabetical order. Changes that add passes in
// non-alphabetical order should get a Request Changes.

def AddDeallocs : Pass<"add-dealloc", "mlir::func::FuncOp"> {
  let summary = "Add quake deallocs to functions before they return.";
  let description = [{
    This pass ensures that a dealloc is inserted before functions return
    if the function contains a AllocaOp. It should be run before converting
    to QIR, for example, in order to generate correct code.
  }];
}

def AddMeasurements : Pass<"add-measurements", "mlir::func::FuncOp"> {
  let summary = "Add measurement operations.";
  let description = [{
    This pass adds measurement operations to the kernels that have no 
    measurements.

    This pass should be run in the sampling context, so as to comply with the 
    spec - If no measurements appear in the kernel, then the `__global__` 
    register is formed with implicit measurements being added for all the 
    qubits defined in the kernel, and the measurements all occur at the end of 
    the kernel.

  }];
  
  let dependentDialects = ["cudaq::quake::QuakeDialect",
    "mlir::cf::ControlFlowDialect"];
}

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

  let options = [
    Option<"computeActionOptimization", "compute-action", "bool",
      /*default=*/"true", "Enable the compute-action control optimization.">,
    Option<"constantPropagation", "constant-prop", "bool", /*default=*/"false",
      "Enable specialization and constant propagation from apply call sites.">,
    Option<"legacyClassical", "legacy-classical", "bool", /*default=*/"false",
      "Enable legacy auto-generation of classical code in kernels. "
      "Classical code is run as it appears in the kernel modulo expressions "
      "derived from loop control variables as loops are reversed. Legacy mode "
      "may produce unexpected results.">
  ];

  let dependentDialects = ["cudaq::cc::CCDialect"];
}

def ArgumentSynthesis : Pass<"argument-synthesis", "mlir::ModuleOp"> {
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
    Option<"changeSemantics", "change-semantics", "bool", /*default=*/"true",
        "As a side-effect break the semantics of how to call the function(s).">
  ];
  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect",
    "mlir::cf::ControlFlowDialect"];
}

def BasisConversion : Pass<"basis-conversion", "mlir::ModuleOp"> {
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

    The pass automatically selects the set of rewrite patterns, ensuring every
    gate is decomposed to the specified basis in a unique way. Option
    `disable-patterns` can be used to filter the selected rewrite patterns.
    Option `enable-patterns` can be used to override the automatic pattern
    selection.

    If no `basis` is specified or the pass cannot decompose all operations to
    the specified basis, the pass application will fail.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let options = [
    ListOption<"basis", "basis", "std::string", "Set of basis operations">,
    ListOption<"disabledPatterns", "disable-patterns", "std::string",
               "Labels of decomposition patterns that should be filtered out">,
    ListOption<"enabledPatterns", "enable-patterns", "std::string",
               "Labels of decomposition patterns that should be used, all "
               "other patterns are filtered out">,
  ];
}

def CliffordTSynthesis : Pass<"clifford-t-synthesis", "mlir::ModuleOp"> {
  let summary =
    "Approximate constant-angle Rz rotations as Clifford+T sequences.";
  let description = [{
    Rewrites `quake.rz` operations with constant angles into sequences of
    `quake.h`, `quake.s`, `quake.t`, and `quake.x` (Clifford+T) gates. The
    synthesis algorithm is gridsynth (Ross & Selinger, arXiv:1403.2975)
    provided by the embedded `cudaq::synth` library.

    Only `quake.rz` is handled. `quake.rx`, `quake.ry`, and `quake.r1` must be
    reduced to Rz first via the `RxToRz`, `RyToRz`, and `R1ToRz` decomposition
    patterns; the `cudaq-fault-tolerant-target` pipeline runs that reduction
    ahead of this pass. Any other rotation left in the IR is passed through
    untouched.

    Both memory semantics (`!quake.ref` targets, in-place) and value
    semantics (`!quake.wire` targets, threaded) are handled. The synthesized
    per-angle helper mirrors the form of the rotation it replaces.

    Preconditions (must be enforced by upstream passes):
    - Rx/Ry/R1 rotations are reduced to Rz - run the `decomposition` pass with
      the `RxToRz`, `RyToRz`, `R1ToRz` patterns enabled first.
    - Rotation angles are folded to constants - run `ArgumentSynthesis` and
      constant-propagation passes first.
    - Controls and adjoints are materialized - run `ApplyOpSpecialization`
      first.

    The omega global phase of each synthesized word is emitted as an
    uncontrolled `quake.phase` in the helper body, so the helper is exactly
    equal to the rotation it replaces. Run `lower-phase` afterwards to
    complete the phase lifecycle; the `cudaq-fault-tolerant-target` pipeline
    does.

    Gridsynth is probabilistic, and its cost grows sharply as `epsilon`
    tightens: roughly 2 ms per distinct rotation at the default 1e-10, but
    orders of magnitude more below 1e-30. Repeated angles are cached, so the
    cost is per distinct angle rather than per operation. This pass is opt-in
    via the `cudaq-fault-tolerant-target` pipeline only. It is not added to
    the default target-prep pipeline.

    The work-budget options below cap the search work gridsynth may spend on
    each rotation. They trade compile time against T-count: raise
    `max-factoring-iterations` for shorter circuits, and the rest to widen the
    worst case a single angle is allowed to reach. See `GridsynthOptions.h`
    for what each one bounds in the algorithm and for the measurements behind
    the defaults.
  }];
  let options = [
    Option<"epsilon", "epsilon", "double", /*default=*/"1e-10",
      "Approximation tolerance epsilon for Clifford+T synthesis.">,
    Option<"maxFactoringIterations", "max-factoring-iterations", "int64_t",
      /*default=*/"500000",
      "Gridsynth tests a candidate circuit by factoring an integer derived "
      "from it (Algorithm 7.6, step 2b). This implementation factors with "
      "Pollard-Brent rho, a randomized method that either finds a factor or "
      "gives up, and this option caps the iterations one attempt may spend "
      "on its composite. Factoring is where nearly all synthesis time goes, "
      "so a lower cap is faster on every call; the cost is that abandoning a "
      "candidate pushes the answer onto a later one, whose denominator "
      "exponent -- and so T-count -- is at least as large. Raise this one to "
      "trade compile time for shorter circuits.">,
    Option<"maxCandidateIterations", "max-candidate-iterations", "int64_t",
      /*default=*/"2000000",
      "Total factoring iterations allowed for one candidate circuit, summed "
      "over its attempts. This is the cap on a whole candidate, and so on the "
      "worst-case time one rotation can take: max-factoring-iterations "
      "applies to a single attempt, and max-factoring-restarts to a single "
      "composite within it. Doubling this costs ~1.8x compile time for the "
      "same T-count, so it bounds the tail rather than tuning quality.">,
    Option<"maxFactoringRestarts", "max-factoring-restarts", "int32_t",
      /*default=*/"8",
      "Consecutive failed factoring attempts allowed on one composite before "
      "the candidate is abandoned. Each attempt re-rolls a random parameter, "
      "so a retry searches differently rather than repeating. Measured "
      "restart chains succeed within a depth of 3, so the default of 8 leaves "
      "ample margin.">,
    Option<"maxOdgpScanSteps", "max-odgp-scan-steps", "int64_t",
      /*default=*/"65536",
      "Steps one candidate-enumeration line scan may take without producing a "
      "candidate. Gridsynth finds candidate circuits by scanning along grid "
      "lines, and a line carrying no candidate can otherwise be scanned "
      "indefinitely. This governs the supply of candidates rather than the "
      "effort spent on each, so starving it makes synthesis fail outright "
      "rather than return a longer circuit.">,
    Option<"retryCount", "retry-count", "int32_t", /*default=*/"3",
      "Extra synthesis attempts for an angle gridsynth failed to synthesize. "
      "Geometric backoff: the budget options above are the baseline, and "
      "retry N re-runs the whole synthesis with them scaled by 2^N, so the "
      "last retry runs at 2^retry-count times the baseline and the "
      "worst-case total work for one angle is (2^(retry-count+1) - 1) times "
      "it. Scaled: max-factoring-iterations, max-candidate-iterations and "
      "max-odgp-scan-steps. Unscaled: max-factoring-restarts, which counts "
      "consecutive failures on a single composite and holds its configured "
      "value, since each attempt already re-rolls its own random parameter. "
      "Set to 0 to attempt each angle once.">,
    Option<"onDynamicAngle", "on-dynamic-angle", "std::string",
      /*default=*/"\"error\"",
      "Behavior on non-constant rotation angles: 'error' or 'skip'.">,
    // TODO: Remove this option once controlled Clifford+T synthesis preserves
    // the synthesis global phase.
    Option<"failOnControlledRotation", "fail-on-controlled-rotation", "bool",
      /*default=*/"false",
      "Fail instead of preserving a controlled rotation that cannot be "
      "synthesized without global-phase support.">,
    Option<"skipBelow", "skip-below", "double", /*default=*/"1e-12",
      "Erase (treat as identity) rotations whose |theta| is below this "
      "threshold, rather than synthesizing a Clifford+T approximation for "
      "them. Set to 0 to disable and synthesize every constant-angle "
      "rotation.">,
  ];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "quake::QuakeDialect"];
}

def OptimizeSingleQubitCliffordT
    : Pass<"optimize-1q-clifford-t", "mlir::ModuleOp"> {
  let summary =
    "Rewrite exact one-qubit Clifford+T chains to Matsumoto-Amano normal form.";
  let description = [{
    Consumes and produces the scalar linear SSA `!quake.wire` form created by
    `convert-to-linear-values`.

    Rewrites exact one-qubit Clifford+T chains on scalar `!quake.wire` values
    into Matsumoto-Amano normal form while preserving the exact unitary.
    Controlled chains are supported when their controls and polarities remain
    unchanged, including through single-block `cc.scope` operations. The pass
    optimizes the one-qubit gate sequence itself, not any subsequent
    decomposition of its controlled gates.

    The normal form is defined by K. Matsumoto and K. Amano, "Representation
    of Quantum Circuits with Clifford and pi/8 Gates," arXiv:0806.3834,
    https://arxiv.org/abs/0806.3834.

  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "quake::QuakeDialect"];
}

def PruneNegligibleRotations
    : Pass<"prune-negligible-rotations", "mlir::func::FuncOp"> {
  let summary = "Erase constant-angle rotations that are effectively identity.";
  let description = [{
    Erases `quake.rz`, `quake.rx`, `quake.ry`, and `quake.r1` operations whose
    constant angle has magnitude below `threshold`, treating them as the
    identity. In memory semantics (`!quake.ref` targets) the op is simply
    dropped and in value semantics (`!quake.wire` targets) the input wire is
    threaded straight through to the users of the result.

    Only uncontrolled rotations are pruned: a controlled rotation with a
    negligible angle is left in place (materialize controls with
    `ApplyOpSpecialization` first if pruning them matters). Non-constant and
    non-finite angles are also left untouched.

    This is the negligible-rotation pruning used internally by
    `clifford-t-synthesis`, exposed as a standalone pass so it can be run
    independently of Clifford+T synthesis.
  }];
  let options = [
    Option<"threshold", "threshold", "double", /*default=*/"1e-12",
      "Erase (treat as identity) rotations whose |theta| is below this "
      "threshold. Set to 0 to disable.">,
  ];
  let dependentDialects = ["quake::QuakeDialect"];
}

def CableRoughIn : Pass<"cable-rough-in", "mlir::func::FuncOp"> {
  let summary = "Perform the rough-in cabling for value semantics conversion.";
  let description = [{
    Some higher-level aspects of the IR, such as function calls, may not be
    obviously reducible to quake's value semantics form. This can impede other
    passes and analyses.

    This prerequisite pass will "rough-in" cabling abstractions, which are part
    of value semantics to reduce the switching cost of having both value and
    reference semantics present at the same time in the IR.
  }];
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

def ClassicalOptimization : Pass<"classical-optimization"> {
  let summary = "Perform classical optimizations until a fix point is reached.";
  let description = [{
    Performs a number of classical optimizations greedily until a fix point
    is reached:
    - canonicalization
    - simplify regions
    - write-after-write-elimination
    - lift-array-alloc
    - cc-loop-normalize
    - cc-loop-unroll

    Running these together to a fix point unrolls loops that a fixed sequence
    of the individual passes cannot, such as one whose trip count is a constant
    array element indexed by an outer induction variable.

    The two selective unrolling options mirror cc-loop-unroll and are mutually
    exclusive. Unlike cc-loop-unroll, this pass never reports a loop it could
    not unroll.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect", "cudaq::cc::CCDialect",
    "mlir::cf::ControlFlowDialect"];

  let options = [
    Option<"threshold", "maximum-iterations", "unsigned", /*default=*/"50",
      "Maximum iterations to unroll.">,
    Option<"allowClosedInterval", "allow-closed-interval", "bool",
      /*default=*/"true", "Allow loop iterations on a closed interval.">,
    Option<"allowBreak", "allow-early-exit", "bool", /*default=*/"false",
      "Allow unrolling of loop with early exit (i.e. break statement).">,
    Option<"disableLoopUnrolling", "no-loop-unroll", "bool",
      /*default=*/"false",
      "Disable loop unrolling and preserve cc.loop operations.">,
    Option<"unrollOnlyAliasingQuantumAccessLoops",
      "unroll-only-aliasing-quantum-access-loops", "bool",
      /*default=*/"false",
      "Unroll only loops containing aliasing quantum accesses or exp_pauli "
      "operands that must be resolved for wire-set lowering. See "
      "cc-loop-unroll.">,
    Option<"unrollOnlyIndexUseLoops", "unroll-only-index-use-loops", "bool",
      /*default=*/"false",
      "Unroll only index use loops, i.e. loops whose induction variable is "
      "used in the loop body. See cc-loop-unroll.">
  ];
}

def CombineMeasurements :
    Pass<"combine-measurements", "mlir::func::FuncOp"> {
  let summary = "Extends measurements on subveqs adds output names";
  let description = [{
    Replace a pattern such as:
    ```
    func.func @kernel() attributes {"cudaq-entrypoint"} {
      %1 = ... : !quake.veq<4>
      %2 = quake.subveq %1, %c2, %c3 : (!quake.veq<4>, i32, i32) ->
            !quake.veq<2>
      %measOut = quake.mz %2 : (!quake.veq<2>) -> !cc.sequence<!quake.measure>
    }
    ```
    with:
    ```
    func.func @kernel() attributes {"cudaq-entrypoint", ["output_names",
    "[[[0,[1,\22q0\22]],[1,[2,\22q1\22]]]]"]} {
      %1 = ... : !quake.veq<4>
      %measOut = quake.mz %1 : (!quake.veq<4>) -> !cc.sequence<!quake.measure>
    }
    ```
  }];
  let dependentDialects = ["cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect"];
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

  let dependentDialects = ["cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect"];
}

def ConsolidateBroadcasts :
    Pass<"consolidate-broadcasts", "mlir::func::FuncOp"> {
  let summary = "Roll loops over a veq back into broadcast form.";
  let description = [{
    A counted loop that applies one-target operators to every element of a
    `veq` in order is replaced by those operators applied to the `veq` itself.
    ```mlir
    %0 = quake.alloca !quake.veq<2>
    cc.loop while ((%i = %c0) -> (i64)) {
      ...
    } do {
    ^bb0(%i: i64):
      %r = quake.extract_ref %0[%i] : (!quake.veq<2>, i64) -> !quake.ref
      quake.h %r : (!quake.ref) -> ()
      ...
    }
    ────────────────────────────────────────────────────────────────────────
    %0 = quake.alloca !quake.veq<2>
    quake.h %0 : (!quake.veq<2>) -> ()
    ```
    The loop is only rolled if it covers the entire vector and its body does
    nothing but extract the element and operate on it. Only an uncontrolled
    operator broadcasts: given a control, the last qubit is the sole target of
    a single operation.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect"];
}

def ConstantPropagation : Pass<"constant-propagation", "mlir::func::FuncOp"> {
  let summary = "Propagate constants to their uses.";
  let description = [{
    This is a greedy pass that will push constants from single and
    multidimensional array constants to their ultimate uses.
  }];
}

def ConvertToDirectCalls : Pass<"indirect-to-direct-calls", "mlir::ModuleOp"> {
  let summary = "Convert calls to direct calls to Quake routines.";
  let description = [{
    Rewrite the calls in the IR so that they point to the generated code and not
    to the entry point thunks that call back to the runtime. After this pass
    quantum code will call other quantum code directly and without going
    indirectly through the launch kernel runtime.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect"];
}

def DeadQuantumElimination : Pass<"dqe"> {
  let summary = "Dead quantum elimination (DQE).";
  let description = [{
    Eliminate any clearly dead quantum (`!quake.ref` or `!quake.wire`)
    allocations from the IR. (quantum here means some flavor of qudit.)
  }];

  let statistics = [
    Statistic<"numRefsEliminated", "refs-eliminated",
              "Number of dead quantum ref allocations eliminated">,
    Statistic<"numWiresEliminated", "wires-eliminated",
              "Number of dead quantum wires eliminated">
  ];
}

def DeadStoreRemoval : Pass<"dead-store-removal"> {
  let summary = "Dead store removal (DSR).";
  let description = [{
    This pass performs classic dead store removal to clean up code that may
    contain residual dead stores.
  }];
}

def Decomposition : Pass<"decomposition", "mlir::ModuleOp"> {
  let summary = "Break down quantum operations.";
  let description = [{
    This pass decomposes quantum operations by iteratively applying rewrite
    patterns until all quantum operations are in basis, a fixpoint is reached or
    the maximum number of iterations is exhausted.

    When `basis` is specified, the pass automatically selects the set of rewrite
    patterns, ensuring every gate is decomposed to the specified basis in a
    unique way.
    
    #### Pattern selection

    Pattern selection can be controlled with the following optional settings:

    - `disable-patterns`: used to filter out specific rewrite patterns from the
       selection.
    - `enable-patterns`: when set, overrides the automatic pattern selection.
    - `basis`: takes a comma-separated list of quantum operations with the
      format: `<op-name>([<number-of-controls> | n])?`

    If no `basis` is specified, as many patterns as possible are applied.

    Examples of valid `basis` values:
    - `x` — Pauli-X without controls (aka `not`)
    - `x(1)` — Pauli-X with one control (aka `cx`)
    - `x(n)` — Pauli-X with unbounded controls
    - `x,x(1)` — Both `not` and `cx`

    Every successful decomposition preserves the complete unitary, including
    any scalar phase represented by compiler bookkeeping. This lets the pass
    run before a later controlled `quake.apply` specialization; unsupported
    forms remain unchanged rather than taking an implicit phase-relaxed path.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
  let options = [
    ListOption<"basis", "basis", "std::string", "Set of basis operations">,
    ListOption<"disabledPatterns", "disable-patterns", "std::string",
               "Labels of decomposition patterns that should be filtered out">,
    ListOption<"enabledPatterns", "enable-patterns", "std::string",
               "Labels of decomposition patterns that should be used, all "
               "other patterns are filtered out">,
    Option<"testConvergence", "test-convergence", "bool", /*default=*/"false",
           "Test only: Fail pass on non-convergence to detect cyclic patterns">,
  ];
}

def DependencyAnalysis : Pass<"dep-analysis", "mlir::ModuleOp"> {
  let summary =
    "Maps qubits and reorders operations based on dependency graph.";
  let description = [{
    A dependency graph is a Directed Acyclic Graph (DAG) where each node
    represents an operation, and each edge represents a "depends on" relation
    between that operation and another operation. For example, in the following
    snippet the `x` operation depends on the `h` operation because it is applied
    to the same qubit (`q`), so the `h` operation must happen before the `x`
    operation:
    ```c++
      cudaq::qubit q;
      x(q);
      h(q);
    ```

    Once a dependency graph is created, it is then scheduled, assigning each
    operation a virtual cycle. Operations that don't depend on each other may
    be scheduled at the same cycle. However, an operation that depends on a
    second operation must be scheduled after the second operation. The
    scheduling algorithm tries to pack operations as densely as possible,
    minimizing the number of cycles between operations.

    From this dependency graph, we can calculate the lifetime of a qubit: from
    the cycle in which it is first used through the cycle in which it is last
    used. If two virtual qubits have non-overlapping lifetimes, they can be
    assigned to the same physical qubit, as every virtual qubit is assumed to be
    fully reset before release. Failure to fully reset virtual qubits before
    release is undefined behavior and will likely lead to incorrect output
    when running DependencyAnalysis.
  }];

  let dependentDialects = ["cudaq::quake::QuakeDialect"];

  let statistics = [
    Statistic<"numVirtualQubits", "num-virtual-qubits",
              "Number of virtual qubits used">,
    Statistic<"numPhysicalQubits", "num-physical-qubits",
              "Number of physical qubits used">,
    Statistic<"numCycles", "num-cycles",
              "Length of kernel in cycles">,
  ];
}

def DesugarCableOps : Pass<"desugar-cable-ops"> {
  let summary = "Desugar some of the cable ops";
  let description = [{
    The `quake.attach_wire` and `quake.detach_wire` operations are syntactic
    sugar for a sequence of more primitive operations. This pass desugars the
    IR and uses the primitives instead.

    Specifically, we make the following expansions.
    ```mlir
    %wire, %cab = quake.detach_wire %0[4] : (!quake.cable<10>) -> (!quake.wire,
        !quake.cable<9>)
    ```
    becomes
    ```mlir
    %1:10 = quake.split_cable %0 : (!quake.cable<10>) -> (!quake.wire, ...)
    %cab = quake.bundle_cable %1#0, ... %1#3, %1#5, ... %1#9 :
        (!quake.wire ...) -> !quake.cable<9>
    // %wire is substituted with %1#4
    ```

    The converse operation
    ```mlir
    %cab = quake.attach_wire %w, %c[8] : (!quake.wire, !quake.cable<8>) ->
             !quake.cable<9>)
    ```
    becomes
    ```mlir
    %1:8 = quake.split_cable %c : (!quake.cable<8>) -> (!quake.wire, ...)
    %cab = quake.bundle_cable %1#0, ... %1#7, %w : (!quake.wire ...) ->
           !quake.cable<9>
    ```
  }];

  let dependentDialects = ["cudaq::quake::QuakeDialect"];
}

def DistributedDeviceCall : Pass<"distributed-device-call", "mlir::ModuleOp"> {
  let summary = "Translate cc.device_call operations";

  let description = [{
    Perform the translation of cc.device_call to a set of substeps for a
    distributed memory model.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"insertTrapImplementation", "insert-trap", "bool",
      /*default=*/"false",
      "Enable pass to insert a default implementation (runtime trap) for "
      "device call (default=false). The default implementation has weak "
      "linkage, i.e., allowed to be overridden at link time.">
  ];
}

def EraseNoise : Pass<"erase-noise"> {
  let summary = "Erase the injection of noise via Kraus channels.";
  let description = [{
    Although CUDA-Q allows the user to specify the application of noise via
    Kraus channels, these are not needed and must be removed if the code is to
    run on quantum hardware, for example.
  }];
}

def EraseNopCalls : Pass<"erase-nop-calls"> {
  let summary = "Erase calls to any builtin intrinsics that are NOPs.";
  let description = [{
    The code may contain marker function calls that do not generate any actual
    code. These calls are NOPs that will be erased by this pass.
  }];
}

def EraseQEC : Pass<"erase-qec"> {
  let summary = "Erase QEC detector and observable operations.";
  let description = [{
    Removes `qec.detector`, `qec.observable`, and `qec.pair_detectors`
    operations from the IR. These are side-effecting declaration ops: they
    record a parity-check or logical-observable relationship over prior
    measurement results so a detector error model can be built from the kernel.
    This pass strips them.
  }];
}

def EraseVectorCopyCtor : Pass<"erase-vector-copy-ctor"> {
  let summary = "Erase unneeded vector copy ctor idiom after inlining.";
  let description = [{
    After inlining, we may be left with a peculiar idiom of the callee creating
    a vector on the stack, copying that vector to a temporary on the heap, and
    the calling function copying the heap data to a new vector on the stack.
    Since the callee is inlined, none of this is really required. The callee and
    caller share a common stack frame, so one copy is plenty.
  }];
}

def ExpandBroadcasts : Pass<"expand-broadcasts", "mlir::func::FuncOp"> {
  let summary = "Expands veqs used as targets into individual qubits.";
  let description = [{
    A one-target operator applied to a `veq` broadcasts that operator over
    every element of the vector. Given a vector of constant size `n`, this
    pass rewrites
    ```mlir
    quake.* %veq : (!quake.veq<n>) -> ()
    ```
    into the `n` operations it stands for:
    ```mlir
    %arg0 = quake.extract_ref %veq[0] : (!quake.veq<n>) -> !quake.ref
    quake.* %arg0 : (!quake.ref) -> ()
    ...
    %argn = quake.extract_ref %veq[n-1] : (!quake.veq<n>) -> !quake.ref
    quake.* %argn : (!quake.ref) -> ()
    ```
    Parameters and attributes are replicated on each operation. Multi-qubit
    operators (`swap`, `exp_pauli`, custom unitaries) are left alone: for those
    a `veq` target is the operand list of a single N-qubit gate rather than a
    broadcast. Controlled operators are left alone as well: only an
    uncontrolled operator broadcasts.
  }];
}

def ExpandControlNegations :
    Pass<"expand-control-negations", "mlir::func::FuncOp"> {
  let summary =
    "Replace all `Ops` with negated controls with normal controls and `XOp`s.";

  let description = [{
    For every quantum operation with a negated control, replace that operation 
    with an X operation on the control qubit, the controlled operation with
    positive polarity, and a final X operation on the control qubit. 
  }];
}

def ExpandControlVeqs: Pass<"expand-control-veqs", "mlir::func::FuncOp"> {
  let summary = "Expands veqs used as controls into individual qubits.";
  let description = [{
    Given an operation of the form
    ```mlir
    quake.* [%veq] %r : (!quake.veq<n>, !quake.ref) -> ()
    ```
    this pass will extract each qubit from `%veq%` and explicitly provide them
    as qubits:
    ```mlir
    %arg0 = quake.extract_ref %veq[0] : (!quake.veq<n>) -> !quake.ref
    ...
    %argn = quake.extract_ref %veq[n] : (!quake.veq<n>) -> !quake.ref
    quake.* [%arg0, ..., %argn] %0 : (!quake.ref, ..., !quake.ref,
        !quake.ref) -> ()
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
  let dependentDialects = ["cudaq::quake::QuakeDialect",
    "cudaq::cc::CCDialect"];

  let options = [
    Option<"enableFailures", "enable-failures", "bool", /*default=*/"false",
      "Enable pass to raise errors when factoring fails (default=false).">
  ];
}

def GenerateDeviceCodeLoader : Pass<"device-code-loader", "mlir::ModuleOp"> {
  let summary = "Generate device code loader stubs.";
  let description = [{
    Generate device code loader stubs which are used for code introspection
    by the runtime.
  }];

  let dependentDialects = ["mlir::LLVM::LLVMDialect"];

  let options = [
    Option<"outputFilename", "output-filename", "std::string",
      /*default=*/"\"-\"", "Name of output file.">,
    Option<"generateAsQuake", "use-quake", "bool",
      /*default=*/"true", "Output should be module in Quake dialect.">,
    Option<"jitTime", "jit-compile", "bool",
      /*default=*/"false", "Running pass at JIT compile time (default=false).">
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

    There are multiple code generation kinds that are supported for flexibility
    and streamlining the kernel launch process. These tend to be related to the
    target and runtime environment the compiler is being run in and can involve
    some technical issues that require deeper understanding of the entire
    process. In general, it is not recommended for user's to change this value.

    ```
    codegen kind   description

      0            Hybrid. A combination of 1 and 2 that allowed early and
                   streamlined JIT compilation but also supports return values
                   and dynamic parameters.
      1            Client-server interchange format. Supports kernels that
                   return results and dynamic parameters.
      2            Streamlined for JIT. The kernel will be converted to a
                   nullary function with no results. Return values from the
                   kernel are ignored, if present. All parameter values are to
                   be inlined by the JIT compiler, so this codegen kind does not
                   support any dynamic parameters.
    ```

    The `generate-run-stack` option must be used whenever `cudaq::run` is being
    used to launch kernels. This flag will enable the generation of a second
    launch stack for entry-point kernels that duplicates the functionality of
    the entry-point kernel's body while omitting its return value. While this
    clearly changes the semantics of the kernel call, some transport layers
    require that return values be dropped on the floor.

    The `posit-nullary` option works with the `change-semantics` option of
    argument synthesis. If argument synthesis does not change the semantics (by
    blindly removing its arguments) of the kernel, then this pass should set the
    posit nullary option so that it can assume there are no arguments (the
    kernel is specialized such that its actual arguments are inlined as
    constants) regardless of the kernel's (semantically correct) signature.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::LLVM::LLVMDialect",
    "mlir::func::FuncDialect", "mlir::cf::ControlFlowDialect",
    "mlir::arith::ArithDialect"];

  let options = [
    Option<"outputFilename", "output-filename", "std::string",
      /*default=*/"\"-\"", "Name of output file.">,
    Option<"startingArgIdx", "starting-arg-idx", "std::size_t", /*default=*/"0",
      "The starting argument index for the argsCreator.">,
    Option<"codegenKind", "codegen", "std::size_t", /*default=*/"0",
      "Set the kind of code to generate for the launches.">,
    Option<"genRunStack", "generate-run-stack", "bool", /*default=*/"false",
      "Generate a cudaq::run stack for the entry-point kernels.">,
    Option<"deferToJIT", "deferred", "bool", /*default=*/"false",
      "Defer generating kernel execution until JIT time.">,
    Option<"positNullary", "posit-nullary", "bool", /*default=*/"false",
      "Posit that the entry-point is nullary regardless of its signature.">,
    Option<"ignoreHostFunction", "ignore-host-function", "bool",
      /*default=*/"false",
      "Assume that the kernel doesn't need a host function.">
  ];
}

def GetConcreteMatrix : Pass<"get-concrete-matrix", "mlir::ModuleOp"> {
  let summary =
    "Replace the unitary matrix generator function with a constant matrix.";
  let description = [{
    Given a custom operation whose generator attribute is another function 
    within the module, such that if `LiftArrayAlloc` pass has run, there will
    be a global constant within the module which holds the constant matrix 
    representation for the custom operation. This pass will find that global
    variable and update the custom operation to directly point to it. 

    Example:
    ```mlir
      module {
        func.func @__nvqpp__mlirgen__function_foo_generator_1.bar(%arg0:
            !cc.sequence<f64>) -> !cc.sequence<complex<f64>> {
          ...
          %0 = cc.address_of
              @__nvqpp__mlirgen__function_foo_generator_1.bar.rodata_0 :
              !cc.ptr<!cc.array<complex<f64> x 4>>
          ...
          return %3 : !cc.sequence<complex<f64>>
        }

        func.func @__nvqpp__mlirgen__function_kernel_1._Z8kernel_1v() {
          %0 = quake.alloca !quake.ref
          quake.custom_op @__nvqpp__mlirgen__function_foo_generator_1.bar %0 :
              (!quake.ref) -> ()
          return
        }

        cc.global constant private
          @__nvqpp__mlirgen__function_foo_generator_1.bar.rodata_0
          ((dense<[(0.000000e+00,0.000000e+00), (1.000000e+00,0.000000e+00),
            (1.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> :
              tensor<4xcomplex<f64>>) : !cc.array<complex<f64> x 4>
      }
    ```

    The `quake.custom_op` call would be converted to

    ```mlir
      func.func @__nvqpp__mlirgen__function_kernel_1._Z8kernel_1v() {
        %0 = quake.alloca !quake.ref
        quake.custom_op
            @__nvqpp__mlirgen__function_foo_generator_1.bar.rodata_0 %0 :
              (!quake.ref) -> ()
        return
      }
    ```

  }];
}

// GlobalizeArrayValues must be a module pass because it may promoted array
// constants from functions to global constants (changes their scope).
def GlobalizeArrayValues : Pass<"globalize-array-values", "mlir::ModuleOp"> {
  let summary = "Convert const_array ops to globals.";
  let description = [{
    Often a `const_array` op can be canonicalized into scalar constants that
    are then constant propagated to their uses in the quake ops. When this
    happens, the `const_array` may become unused and can be eliminated.

    However, there can also be cases where the `const_array` remains alive, such
    as when it is used in a `state_init` op. In such cases, we may be able to go
    ahead and replace the `const_array` with a global constant. This pass makes
    such conversions.

    A `const_array` may also be used as an image for a tree of spans which has
    yet to be reified. In this case, the inner most array may be lowered to a
    global object, but the tree of spans working outward from those objects
    must be constructed. This pass can replace the reify_span operation with
    smaller granularity operations that build these trees of spans.
  }];
}

def InjectImplicitOutput : Pass<"inject-implicit-output", "mlir::func::FuncOp"> {
  let summary = "Inject quake.evince ops for implicitly output quantum values.";
  let description = [{
    This pass inserts `quake.evince` operations for each `quake.alloca` at
    the top level of a `func.func`. The evince ops are added to a new exit
    block that all return ops are redirected through, preserving the declaration
    order of the allocations. Unlike `add-dealloc`, this pass does not recurse
    into nested regions.
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

    The `constantPropagation` option can be used to eliminate constants being
    passed in closures. This option defaults to off to avoid phase ordering
    issues with memtoreg altering the structure of cc.loop too aggressively
    before the loop is analyzed.
  }];

  let dependentDialects = ["mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect", "cudaq::quake::QuakeDialect"];

  let options = [
    Option<"constantPropagation", "constant-prop", "bool", /*default=*/"false",
      "Enable specialization and constant propagation into lifted lambdas.">
  ];
}

def LiftArrayAlloc : Pass<"lift-array-alloc", "mlir::func::FuncOp"> {
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

    See the globalize array values pass for converting `const_array` values
    to global constants. Conversion to globals is intentionally deferred to
    allow constant propagation to take place correctly.
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

    The unroll-only-aliasing-quantum-access-loops option opts in to selective
    unrolling for wire-set lowering. Quantum references and slices accessed
    through dynamic indices are aliasing quantum accesses and must be
    statically resolved. An `exp_pauli` also requires both its quantum register
    and Pauli word to be resolved. Loops containing these accesses or operands
    are unrolled; other counted loops remain rolled. A required loop that
    cannot be unrolled (e.g. a non-constant trip count) is reported as an
    error. This option defaults to false, leaving the pass behavior unchanged
    for existing pipelines.

    The unroll-only-index-use-loops option unrolls only index use loops:
    loops whose induction variable is used in the loop body. A loop that
    ignores its induction variable performs identical work on each iteration,
    so leaving it rolled keeps the IR small. An index use loop that cannot be
    unrolled (e.g. a non-constant trip count) is reported as an error. This
    option also defaults to false.

    The two selective unrolling options are mutually exclusive; enabling both
    is an error.
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
      "Allow unrolling of loop with early exit (i.e. break statement).">,
    Option<"unrollOnlyAliasingQuantumAccessLoops",
      "unroll-only-aliasing-quantum-access-loops", "bool",
      /*default=*/"false",
      "Unroll only loops containing aliasing quantum accesses or exp_pauli "
      "operands that must be resolved for wire-set lowering.">,
    Option<"unrollOnlyIndexUseLoops", "unroll-only-index-use-loops", "bool",
      /*default=*/"false",
      "Unroll only index use loops, i.e. loops whose induction variable is "
      "used in the loop body; leave loops that merely repeat identical work "
      "rolled.">
  ];
}

def LoopInductionFusion : Pass<"cc-loop-induction-fusion"> {
  let summary = "Fuse secondary loop inductions into the primary induction.";
  let description = [{
    For a normalized cc.loop (one that has already been through
    cc-loop-normalize so the primary control induction starts at 0 and steps by
    1), any additional loop-carried integer values that step by a loop-invariant
    amount on every iteration are *secondary inductions*.  They are linear
    functions of the primary induction variable `i`:

    ```
      j(i) = j_initial ± i * step_j
    ```

    This pass replaces every use of a secondary induction's block argument in
    the loop body with the closed-form expression above, then removes the
    secondary from the loop's carried-value set.  The final value of each
    secondary after the loop is replaced in the parent block by substituting
    the loop's primary result:

    ```
      j_final = j_initial ± loop_result_i * step_j
    ```

    The loop must be in invariant form: primary starts at 0 and steps by 1.
    Loops stamped with the `normalized` attribute by `cc-loop-normalize`
    satisfy this, but the pass also accepts any loop already in that form
    without the attribute.  If the loop is not in invariant form, or has no
    secondary inductions, the pass is a no-op for that loop.
  }];

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "cudaq::cc::CCDialect"];
}

def LowerPhase : Pass<"lower-phase", "mlir::func::FuncOp"> {
  let summary = "Lower phase bookkeeping operations to ordinary Quake gates";
  let description = [{
    Lowers controlled `quake.phase` operations to exact `quake.r1`/`quake.rz`
    gate sequences and erases uncontrolled phases after forwarding their wire
    results. This pass is intentionally exposed in isolation until the global
    phase lifecycle is integrated into production pipelines.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "cudaq::quake::QuakeDialect"];
}

def ConvertToCFG : Pass<"convert-to-cfg"> {
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

  let options = [
    Option<"unsafeQuantumAlloc", "unsafe-quantum-alloc", "bool",
           /*default=*/"false",
           "When true, suppress adding the cc.no_qubit_combine attribute when "
           "lowering a cc.scope that contains a quantum allocation.  This is "
           "unsafe in general (the subsequent combine-quantum-alloc pass may "
           "combine allocations across scope boundaries), but useful when "
           "lower-to-cfg is followed immediately by combine-quantum-alloc and "
           "factor-quantum-alloc, as in the quake_fake mid-level pipeline.">
  ];

  let dependentDialects = [ "mlir::cf::ControlFlowDialect" ];
}

def ConvertToCFGPrep : Pass<"convert-to-cfg-prep", "mlir::ModuleOp"> {
  let summary = "Prepare the Module to be converted to a CFG.";
  let description = [{
    This pass must be run before the ConvertToCFG pass.
  }];
}

def MappingFunc: Pass<"qubit-mapping-func", "mlir::func::FuncOp"> {
  let summary = "Perform qubit mapping to account for connectivity constraints";
  let description = [{
    Some backends cannot support any-to-any multi-qubit operations, so this pass
    performs mapping of the qubits by inserting the necessary swap operations
    into the Quake IR.

    The mapping pass emits Mapped Quake IR, in which kernels borrow device wires
    from `@mapped_wireset`.

    Note 1: this pass requires strictly value semantics (in the form of wire
    sets) for any quantum operations. It will throw an error if any memory
    reference semantics are present on quantum operations.

    Note 2: this pass can introduce ancilla qubits.

    Note 3: as a result of note 2, if the IR contains no measurements, this pass
    will inject measurements so that the post-mapping measurements correspond
    to all of the input (user) qubits.

    Control flow: structured `cc.if` and `cc.loop` operations that carry
    quantum wires are routed. Each branch and loop body is routed recursively
    against the layout that is live where the region is entered, and the
    placements that meet at each control-flow join must be reconciled to a
    single layout so the block after the join has an unambiguous entry.

    Single-block `cc.scope` operations, including nested scopes, are routed as
    ordered lexical segments. The mapper preserves their attributes and
    classical operations, adding passthrough results when routing requires
    captured wires to leave the scope. Other consumed input wires must already
    be scope results.

    The selected outer scope of a `run` entry is treated as the function body.
    It must contain one block, but the restrictions on scope contents below do
    not apply to it.

    Reconciliation is currently done by reverting each predecessor to the
    layout it entered with (its swaps are replayed in reverse before the region
    terminator):

    - `cc.if`: both branches share the same entry layout, so restoring each to
      its entry makes them agree at the if-join with no extra search.
    - `cc.loop`: the body is restored to its entry layout, making that layout
      the loop invariant. This needs no swaps before the loop, but pays the
      full restoration cost on every iteration.

    This handles two-qubit gates and swaps inside branches and loop bodies
    (unlike a scheme that forbids them), but always collapses back to the entry
    layout rather than searching for a cheaper common layout. A future strategy
    may pick a different common exit layout and add the swaps needed to reach it
    on each predecessor (including before a loop).

    The following control flow is not yet supported and is rejected (a hard
    error under `raise-fatal-errors`, otherwise the function is skipped):

    - in non-`run` functions, measurements inside a scope, branch, or loop.
    - a mid-circuit measurement whose measured wire feeds a non-terminal user,
      or measurement-dependent (adaptive) control flow, unless the function is
      a `run` entry point with structured control flow.
    - loops with a multi-block body, a Python `else`, or a `break`.
    - quantum gates in a loop's condition or step region.
    - a scope inside structured control flow, or structured control flow inside
      a scope.
    - multi-block scopes, or scopes containing quantum allocations, borrows, or
      terminal wire operations.
    - more than one block at the function's top level.
  }];

  let options = [
    Option<"device", "device", "std::string", /*default=*/"\"-\"",
      "Device topology: path(N), ring(N), star(N), star(N,c), grid(w,h), "
      "file(/path/to/file), bypass">,
    Option<"extendedLayerSize", "extended-layer-size", "unsigned",
           /*default=*/"20", "Extended layer size">,
    Option<"extendedLayerWeight", "extended-layer-weight", "float",
           /*default=*/"0.5", "Extended layer weight">,
    Option<"decayDelta", "decay-delta", "float", /*default=*/"0.5",
           "Decay delta">,
    Option<"roundsDecayReset", "rounds-decay-reset", "unsigned",
           /*default=*/"5", "Number of rounds before decay is reset">,
    Option<"minStallSwapBudget", "min-stall-swap-budget", "unsigned",
           /*default=*/"64",
           "Release valve floor: minimum number of consecutive swaps that "
           "route no gate before a stuck front layer is force-routed along a "
           "shortest path (advanced)">,
    Option<"stallSwapBudgetPerQubit", "stall-swap-budget-per-qubit", "unsigned",
           /*default=*/"4",
           "Release valve scaling: per-device-qubit stall budget. The budget "
           "used is max(min-stall-swap-budget, stall-swap-budget-per-qubit * "
           "numQubits) (advanced)">,
    Option<"placement", "placement", "std::string",
           /*default=*/"\"identity\"",
           "Initial placement strategy: auto (propose identity and greedy "
           "seeds, select by routed swap count), identity, greedy">,
    Option<"search", "search", "std::string", /*default=*/"\"none\"",
           "Layout search strategy: sabre (SABRE forward-backward-forward "
           "reverse-traversal refinement of each seed), none (single forward "
           "route per seed; legacy mapping is placement=identity search=none)">,
    Option<"islandPackingAttemptBudget", "island-packing-attempt-budget",
           "unsigned", /*default=*/"1048576",
           "For disconnected devices with greedy placement: the maximum number "
           "of placement attempts the exhaustive island-packing fallback may "
           "make before giving up. The fallback only runs when the fast "
           "best-fit-decreasing packer fails, and feasible packings settle in a "
           "handful of attempts; this bound caps the cost of proving "
           "infeasibility on pathological topologies. Exhausting it emits a "
           "diagnostic suggesting a larger value. 0 disables the fallback "
           "(best-fit-decreasing only) (advanced)">,
    Option<"nonComposable", "raise-fatal-errors", "bool", /*default=*/"false",
           "Run the pass in a non-composable way, which may cause immediate "
           "internal compiler errors">
  ];
}

def MappingPrep: Pass<"qubit-mapping-prep", "mlir::ModuleOp"> {
  let summary = "Prepare module for qubit mapping.";
  let description = [{
    Insert the required topology-aware `quake.wire_set` operation that
    corresponds to the requested device.
  }];

  let options = [
    Option<"device", "device", "std::string", /*default=*/"\"-\"",
      "Device topology: path(N), ring(N), star(N), star(N,c), grid(w,h), "
      "file(/path/to/file), bypass">,
    Option<"nonComposable", "raise-fatal-errors", "bool", /*default=*/"false",
           "Run the pass in a non-composable way, which may cause immediate "
           "internal compiler errors">
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
  let dependentDialects = ["cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect"];

  let options = [
    Option<"classicalValues", "classical", "bool",
      /*default=*/"true", "Promote classical stack slots to values.">,
    Option<"quantumValues", "quantum", "bool",
      /*default=*/"true", "Promote quantum references to wire values.">
  ];
  let statistics = [
    Statistic<"numProcessOpWithRegionsCalls", "process-op-with-regions-calls",
      "Number of times processOpWithRegions was called">,
    Statistic<"numWorklistIterations", "worklist-iterations",
      "Number of worklist loop iterations in processOpWithRegions">
  ];
}

def MultiControlDecomposition :
    Pass<"multicontrol-decomposition", "mlir::func::FuncOp"> {
  let summary = "Break down multi-control quantum operations.";
  let description = [{
    This pass decomposes multi-control quantum operations. The decomposition
    involves allocating new qubits to hold intermediate results. The number of
    extra qubits depends on the particular operation being decomposed.
    Pauli-X and Pauli-Z operations add _N_ - 2 qubits, while other operations
    add _N_ - 1 qubits, where _N_ is the number of controls.

    Note: When a `veq` is used as control, we need to know its size to be able
    to decompose. In such cases, all qubits will be extracted. If the size is
    unknown at compilation-time, the pass leaves the operation as-is.
  }];
}

def NormalizePhasePlacement :
    Pass<"normalize-phase-placement", "mlir::func::FuncOp"> {
  let summary = "Normalize block-local placement of phase corrections";
  let description = [{
    Conservatively sinks `quake.phase` operations within safe straight-line
    portions of a basic block, reconnects value-semantics wire chains, and
    combines corrections with identical ordered control predicates. Movement
    stops at control-changing gates, calls, regions, terminators, non-unitary
    operations, unknown effects, or whenever wire reconnection is ambiguous.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "cudaq::quake::QuakeDialect"];
}

def ObserveAnsatz : Pass<"observe-ansatz", "mlir::func::FuncOp"> {
 let summary = "Given spin_op input, append measures to the FuncOp";
  let description = [{
    Given an unmeasured Quake representation (i.e. a state prep ansatz), append
    measures based on the given spin_op specified in binary symplectic form.
  }];
  let options = [
    ListOption<"termBSF", "term-bsf", "unsigned",
      "The measurement bases as a Pauli tensor product represented in binary "
      "symplectic form.">
  ];
}

def PhaseFolding: Pass<"phase-folding", "mlir::func::FuncOp"> {
  let summary = "Perform phase based rotation merging.";

  let description = [{
    Multiple `quake.rz` ops can be merged into a single gate if they occur
    at the same phase. In a circuit with only Swap, NOT, CNOT, and Rz gates, a
    phase represents the state of a qubit based on the Swap, NOT, and CNOT
    operations applied to it.

    At the beginning of each subcircuit, each qubit is assigned a unique phase
    variable. A phase is an exclusive sum of the phase variables determining
    the current state of a qubit, calculated as follows. While walking the
    subcircuit:
    * A Swap gate swaps the phases for two qubits.
    * A NOT gate inverts the phase for a qubit by (un)setting a flag.
    * A CNOT gate sets the phase of the target to the exclusive sum of the
      phase variables in the phases for the control and target qubits.
    * Compatible Z-axis rotations at the same phase are combined.

    Operates on `!quake.wire` SSA values. Traverses def-use chains directly.
  }];

  let options = [
    Option<"minimumBlockLength", "min-length", "unsigned", /*default=*/"20",
    "Minimum subcircuit length to run phase folding">,
    Option<"minimumrzWeight", "min-rz-weight", "double", /*default=*/"0.2",
    "Minimum percentage of rz ops in subcircuit to run phase folding">,
  ];

  let dependentDialects = ["cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect"];
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
  let summary = "Removes artificial control constraints between quantum ops.";
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

def PyFrontendCleanup : Pass<"py-frontend-cleanup", "mlir::func::FuncOp"> {
  let summary = "Clean up IR residue specific to the Python AST bridge.";
  let description = [{
    This pass tidies up IR constructs produced only by the Python front-end
    (the AST bridge). It runs as part of the Python AOT pipeline and must be
    followed by `canonicalize`, which completes the folds it exposes.

    Currently it folds `quake.veq_size` of a `cc.if` result whose branches may
    yield an empty veq (`cc.undef`/`cc.poison`). When the `cc.if` condition is
    not a compile-time constant the `cc.if` cannot be folded away, leaving a
    `quake.veq_size` that reads a possibly-undef veq. The pass hoists the
    `quake.veq_size` into both branches as an extra `i64` result so each
    computes the size of its own branch-local veq. It only exposes those
    branch-local veqs; the resulting folds (`quake.veq_size` of an undef/poison
    veq to 0, or of a sized veq to a constant) are performed by the Quake
    canonicalization patterns.
  }];
}

def PySynthCallableBlockArgs :
    Pass<"py-synth-callable-block-args", "mlir::func::FuncOp"> {
  let summary =
    "Synthesize / Inline cc.callable_func on function block arguments.";
  let description = [{
    The Python bindings use this pass to synthesize any
    cc.callable block arguments. Synthesis replaces all uses of the callable
    block argument with a specific in-module function call (`func.call`)
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

  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::complex::ComplexDialect",
                           "cudaq::cc::CCDialect", "mlir::math::MathDialect"];

  let constructor = "cudaq::opt::createQuakeSynthesizer()";
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

def QuakePropagateMetadata :
    Pass<"quake-propagate-metadata", "mlir::ModuleOp"> {
  let summary = "Propagate metadata from callees to callers when needed.";
  let description = [{
   Propagates function metadata attributes added by `quake-add-metadata` pass,
   such as `qubitMeasurementFeedback`, to all the callers of the function. This
   insures these attributes are not missing after inlining.
  }];
}

def QuakeSimplify : Pass<"quake-simplify"> {
  let summary = "Perform simplifications on the quake ops.";
  let description = [{
    This pass performs several simplifications on Quake operations. The
    IR is expected to be in value-semantics form.

    Commutation-aware rewrites are block local and support participating
    operations whose controls and targets are scalar `!quake.wire` values.
    Other valid control representations conservatively do not match. Direct
    rewrites require exactly threaded, distinct wire operands. A rewrite that
    crosses another operation requires the block-local analysis to prove
    commutation.

    Among these, a constant-angle rotation that is the identity is erased. The
    angle is first folded into the period after which the rotation is exactly
    the identity (`4*pi` for the axis rotations, which are spinors, and
    `2*pi` for `r1`) so a full turn is pruned just like a zero angle.
    Non-constant and non-finite angles are left untouched.

    A rotation is erased when its folded angle has magnitude at or below
    `threshold`.

    The optional `rotations-to-clifford-t` pattern family recognizes constant
    `r1`, `rx`, `ry`, and `rz` angles at multiples of `pi/4` and replaces them
    with named-gate sequences instead of leaving them for approximate synthesis.
    The replacement is exact, with no global-phase difference from the original
    rotation. When needed, this includes injecting a `quake.phase` bookkeeping
    operation.
    With a positive `clifford-t-epsilon`, nearby angles are accepted only when
    their operator-norm distance from the selected exact rotation is within that
    epsilon. The family is disabled by default because targets with native
    arbitrary rotations do not benefit from this legalization.
  }];
  let options = [
    Option<"threshold", "threshold", "double", /*default=*/"1e-12",
      "Erase (treat as identity) rotations whose angle, folded into its "
      "identity period, has magnitude at or below this value.">,
    Option<"rotationsToCliffordT", "rotations-to-clifford-t", "bool",
      /*default=*/"false",
      "Enable canonicalization of pi/4 rotation classes to Clifford+T gates.">,
    Option<"cliffordTEpsilon", "clifford-t-epsilon", "double",
      /*default=*/"0.0",
      "Operator-norm epsilon for snapping rotations to a neighboring pi/4 "
      "class. Zero enables bit-exact matching only.">,
  ];
  let statistics = [
    Statistic<"numHermitianEliminations", "hermitian-eliminations",
              "Number of commutation-compatible Hermitian gate pairs eliminated">,
    Statistic<"numAdjointEliminations", "adjoint-eliminations",
              "Number of commutation-compatible adjoint gate pairs eliminated">,
    Statistic<"numZeroRotationsEliminated", "zero-rotations-eliminated",
              "Number of identity rotations eliminated">,
    Statistic<"numRotationsCombined", "rotations-combined",
              "Number of rotation pairs combined">,
    Statistic<"numDoubleSRewrites", "double-s-rewrites",
              "Number of SS -> Z rewrites">,
    Statistic<"numDoubleTRewrites", "double-t-rewrites",
              "Number of TT -> S rewrites">,
    Statistic<"numReduceYSXRewrites", "reduce-ysx-rewrites",
              "Number of YSX -> S rewrites">,
    Statistic<"numResetsErased", "resets-erased",
              "Number of redundant resets erased (double-reset or reset-before-sink)">,
    Statistic<"numCliffordTRotations", "clifford-t-rotations",
              "Number of rotations canonicalized to Clifford+T gates">,
  ];
}

def QubitResetBeforeReuse :
    Pass<"qubit-reset-before-reuse", "mlir::func::FuncOp"> {
  let summary = "After measurement of a ref, add reset (and initialization).";
  let description = [{
   Adds a `quake.reset` and conditionally applies an X gate if the measurement
   result is 1 to initialize the reference back into the just measured state
   afterward. This is only activated when the measured reference is to be
   reused.
   
   Note: if the measurement is already accompanied by a reset, this pass does
   not add any additional operations.
  }];

}

def RealtimeDeviceCall : Pass<"realtime-device-call", "mlir::ModuleOp"> {
  let summary = "Lower cc.device_call operations to the realtime frame ABI.";

  let description = [{
    Lower `cc.device_call` operations to the realtime device-call frame ABI.
    Each device call is rewritten into a sequence of runtime calls operating
    on a leased request/response frame: acquire a frame from the selected
    device channel, marshal the operands into the request payload, dispatch
    the frame, and read the response back before releasing the frame.

    The generated code calls
    `__cudaq_device_call_acquire_realtime_frame`,
    `__cudaq_device_call_dispatch_realtime_frame`, and
    `__cudaq_device_call_safely_release_realtime_frame`, so the final
    executable must be linked against a runtime that provides this leased
    frame API.

    Boolean vector payloads are marshaled through the private
    `__nvqpp_RealtimePackBits`, `__nvqpp_RealtimePackMeasurements`, and
    `__nvqpp_RealtimeUnpackBits` intrinsic functions. Run the `inline`
    pass afterwards to expand these helpers at their call sites.
  }];

  // The lowering materializes `cc` and `arith` ops for payload sizing and
  // marshaling, `func` calls/declarations for the runtime ABI, and may
  // materialize `quake.discriminate` (within the loaded packing intrinsics)
  // when converting measurement-handle vectors to boolean payload bytes.
  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "cudaq::quake::QuakeDialect"];
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

    Note that, it is entirely possible for quake to obtain a cyclic structure
    and otherwise reuse values of type `wire`. In the general case, there is
    not necessarily a way to recover a DAG from a graph with arbitrary cycles.
    However, if we assume the quake code fully enumerates a finite set of wires
    (which is currently true by construction), it is theoretically possible to
    use cloning to recover such a finite DAG representation. This pass does not,
    however, attempt to perform such heroics.
  }];
  let dependentDialects = ["cudaq::cc::CCDialect",
    "cudaq::quake::QuakeDialect"];
}

def RepairLinearType : Pass<"repair-linear-type", "mlir::func::FuncOp"> {
  let summary = "Repair linear type constraints.";
  let description = [{
    The desired linear type constraint is that a value of linear type must be
    used exactly once. That is \e not SSA. Unfortunately, MLIR assumes SSA
    semantics at a very deep level and can and will destroy the linear type
    constraints in its own default optimizations.

    Running this pass will repair the damage in such a way that the linear type
    value is lifted to a reference type, which is a valid SSA value.
  }];
  let dependentDialects = ["cudaq::quake::QuakeDialect"];
  let statistics = [
    Statistic<"numRepairs", "repairs",
              "Number of linear type violations repaired with wrap/unwrap">,
  ];
}

def ReplaceStateWithKernel : Pass<"replace-state-with-kernel",
    "mlir::func::FuncOp"> {
  let summary = "Replace `init_state` operations with calls to a kernel.";
  let description = [{
    Replaces `quake.init_state`, `quake.get_number_of_qubits`, and
    `quake.materialize_state` operations invoked on state pointers during
    argument synthesis for quantum devices with appropriate calls to support
    routines and/or kernels to build the state.

    Before this optimization, argument synthesis for state pointers for quantum
    devices substituted a state created from the `quake.materialize_state`
    operation for the state argument.

    The `quake.materialize_state` operation accepts symbols for the synthesized
    kernels `@num_qubits` and `@init` that argument synthesis generated from
    the original kernel call that generated the state, e.g.,
    the `cudaq::get_state` call that refers to the result of a specific quantum
    kernel being invoked with a set of parameters

    For example, for the user code:
      ```
      state = cudaq::get_state(callee, args)
      caller(state)
      ```

    The argument synthesis generated the following new kernels from the `callee`
    and synthesized them to substitute their arguments with `args`:
      ```
      func.func @callee_num_qubits(arguments) -> i64
      func.func @callee_init(qubits: !quake.veq<?>, arguments) -> !quake.veq<?>
      ```

    The argument synthesis also substituted the state argument in the `caller`
    with:
      ```
      quake.materialize_state @callee_num_qubits @callee_init:
          !cc.ptr<!quake.state>
      ```

    This optimization performs the replacements for the the following
    operations that use a state produced by  `quake.materialize_state
    @num_qubits, @init` operation:

    - Replace `quake.get_number_of_qubits` operation by call to `@num_qubits`
    - Replace `quake.init_state` operation by call to `@init`
    - Clean up unused `quake.materialize_state` operation

    For example:

    Before ReplaceStateWithKernel (replace-state-with-kernel):
    ```
    func.func @foo() {
      %0 = quake.materialize_state @callee.num_qubits_0, @callee.init_0:
              !cc.ptr<!quake.state>
      %1 = quake.get_number_of_qubits %0 : (!cc.ptr<!quake.state>) -> i64
      %2 = quake.alloca !quake.veq<?>[%1 : i64]
      %3 = quake.init_state %2, %0 : (!quake.veq<?>, !cc.ptr<!quake.state>)
              -> !quake.veq<?>
      return
    }
    ```

    After ReplaceStateWithKernel (replace-state-with-kernel):
    ```
    func.func @foo()  {
      %1 = call @callee.num_qubits_0() : () -> i64
      %2 = quake.alloca !quake.veq<?>[%1 : i64]
      %3 = call @callee.init_0(%2) : (!quake.veq<?>) -> !quake.veq<?>
      return
    }
    ```
  }];
}

def ResourceCountPreprocess :
    Pass<"resource-count-preprocess", "mlir::func::FuncOp"> {
  let summary = "Performs preprocessing to optimize for resource counter.";
  let description = [{
    This pass performs preprocessing to optimize the code for the resource
    counter by removing as many invariant gates as possible during compilation,
    with the aim of reducing the program size for further compilation, and
    speeding up resource counting by the simulator at runtime.

    For example, in the following program:

    ```c++
      cudaq::qarray<10> q;
      h(q[0]);
      for (size_t i = 0; i < 10; i++) {
        y(q[i]);
        if (mz(q[i]))
          x(q[i]);
      }
    ```

    The initial `h(q[0])` will always be run once, so we can count it in advance
    and remove it. Similarly, it is trivial to see that the loop will run 10
    times, so we can count `y(q[i])` ten times in advance as well. However,
    because `x(q[i])` is gated by `mz(q[i])`, it is not invariant, so it will
    remain and be counted during simulation.

    Currently, the detection of "invariance" is purposefully dumb.

    This preprocessing path may interfere with detecting downstream
    optimizations, as preprocessed gates are counted and removed early in the
    pipeline.
  }];

  let options = [
    Option<"countGate", "count-gate",
      "std::function<void(std::string,std::vector<std::size_t>,std::vector"
      "<std::size_t>,size_t)>",
      /*default=*/"[](std::string,std::vector<std::size_t>,std::vector"
      "<std::size_t>,size_t){}",
      "Closure that receives gate name, control qubit indices, target qubit "
      "indices, and multiplicity">,
    Option<"dumpPreprocessed", "dump-preprocessed", "bool", /*default=*/"false",
      "Dump preprocessed gates instead of counting them (useful for testing)">,
  ];
}

def RunSemanticsHackery : Pass<"run-semantics-hackery", "mlir::ModuleOp"> {
  let summary = "Adapt vector results for cudaq::run entry points.";
  let description = [{
    `cudaq::run` entry points log kernel results instead of returning them
    through the host ABI. After the `kernel-execution` pass creates the `.run`
    wrapper, this pass finds wrappers that call kernels returning `!cc.sequence`,
    clones those kernels, removes calls to `__nvqpp_vectorCopyCtor` from the
    clones, and redirects the wrappers to call the clones. It must run after
    `kernel-execution`, while both the generated wrapper and original kernel are
    present.
  }];

  let dependentDialects = ["cudaq::cc::CCDialect", "mlir::func::FuncDialect"];
}

def SROA : Pass<"cc-sroa"> {
  let summary = "Scalar replacement of aggregates.";
  let description = [{
    Simple scalar replacement of aggregates. This pass will rewrite simple use
    cases by replacing aggregates with scalars.

    To be implemented: handling recursive compositions of aggregate types.
  }];
  let dependentDialects = ["cudaq::cc::CCDialect"];
}

def StackFramePrealloc : Pass<"stack-frame-prealloc", "mlir::func::FuncOp"> {
  let summary = "Stack frame preallocation.";
  let description = [{
    This pass is intended to be run on the CFG (low-level semantics) level.

    Stack slots may end up being allocated across the body of a function. It
    is generally beneficial however to collect them in the function preamble
    (or entry block).

    This pass is similar to variable coalescing, which can merge allocations
    based on their scopes, but works at a lower semantic level of the IR. This
    pass aims to optimize patterns such as
    ```mlir
      %10 = call @llvm.stacksave.p0()
      %24 = cc.alloca !T
      call @llvm.stackrestore.p0(%10)
    ```
    such that the allocation, %24, is moved to the entry block and the two
    stack guards, if they guard no allocations, can be correctly removed.

    Not all allocations can be moved in this way, which implies that not all
    stack guards can be safely removed.  Transport layer conformity tools may
    unsafely remove operations, but the compiler should never do so, as it
    would mean the compiler is generating incorrect code.
  }];
}

def StatePreparation : Pass<"state-prep", "mlir::func::FuncOp"> {
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
        %2 = quake.init_state %1, %0 : (!quake.veq<2>,
          !cc.ptr<!cc.array<complex<f32> x 4>>) -> !quake.veq<2>
        return
      }
      cc.global constant private @foo.rodata_0
        (dense<[(0.707106769,0.000000e+00), (0.707106769,0.000000e+00),
          (0.000000e+00,0.000000e+00), (0.000000e+00,0.000000e+00)]> :
            tensor<4xcomplex<f32>>) : !cc.array<complex<f32> x 4>
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

// UnitarySynthesis is a module pass because it may modify the `ModuleOp` by
// adding new `FuncOp`(s).
def UnitarySynthesis : Pass<"unitary-synthesis", "mlir::ModuleOp"> {
  let summary = 
    "Convert a custom quantum operation into sequence of native quantum gates.";
  let description = [{
    Given a custom quantum operation specified in the form of its concrete
    unitary matrix, convert it into a sequence of native quantum gates.

    For example,
    ```mlir
    quake.custom_op @__nvqpp__mlirgen__custom_x_generator_1.rodata %0 :
      (!quake.ref) -> ()
    ```
    This call will be converted to an `ApplyOp`.
    ```mlir
    quake.apply @__nvqpp__mlirgen__custom_x_generator_1.kernel %[[VAL_0]] :
      (!quake.ref) -> ()
    ```
    The function has sequence of decomposed gates.
    ```mlir
    func.func private @__nvqpp__mlirgen__custom_x_generator_1.kernel(%arg0:
      !quake.ref) {
        %cst = arith.constant 1.5707963267948966 : f64        
        quake.ry (%cst) %arg0 : (f64, !quake.ref) -> ()
        ...
        return
      }
    ```
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"];
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

  let dependentDialects = ["mlir::cf::ControlFlowDialect",
    "cudaq::quake::QuakeDialect", "cudaq::cc::CCDialect"];
}

def UnwindByDataFlow : Pass<"unwind-by-dataflow", "mlir::func::FuncOp"> {
  let summary = "Lower unwind control-flow macros to data-flow flag variables.";
  let description = [{
    An alternative to `unwind-lowering` that converts `cc.unwind_return`,
    `cc.unwind_break`, and `cc.unwind_continue` into data-flow form instead of
    CFG form.  Structured operations (`cc.scope`, `cc.if`, `cc.loop`) are
    preserved; flag variables threaded through them carry the unwind signal.

    For `cc.unwind_return`: a boolean `returnship` alloca and one alloca per
    return-value type are created in the function entry block.  Each
    `cc.unwind_return` is replaced by stores into those allocas.  Working
    outward through every enclosing structured op, subsequent code is guarded
    by `cc.if (!returnship) { ... }` until the function body is reached, where
    an explicit `if (returnship) { func.return (load returnity) }` is inserted.
    `cc.loop` while-conditions are also ANDed with `!returnship` so the loop
    exits on the next check.

    For `cc.unwind_break` / `cc.unwind_continue`: per-loop `breakship` /
    `continueship` flag allocas (and matching value allocas) are created just
    before the enclosing loop.  Replacement and outward guarding proceed as for
    return, but propagation stops at the enclosing loop, where a
    `cc.if (breakship) { cc.break ... }` or `cc.if (continueship) {
    cc.continue ... }` is appended to the loop body.  Additionally,
    `continueship` is reset to `false` at the top of the loop body on every
    iteration so that a continue from one iteration does not affect the next.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect", "cudaq::cc::CCDialect",
    "mlir::func::FuncDialect"];
}

def UpdateRegisterNames : Pass<"update-register-names"> {
  let summary = "Update classical register names";
  let description = [{
    After loop unrolling, there may be duplicate registerName attributes. Run
    this pass to give them unique attribute values.
  }];
}

def VariableCoalesce : Pass<"variable-coalesce", "mlir::func::FuncOp"> {
  let summary = "Coalesce variables in functions.";
  let description = [{
    This pass will move temporaries from inner scopes to the outermost,
    function scope, and coalesce variables of the same type and disjoint
    live-ranges. It relies on the correct and proper construction and use of
    `cc.scope` ops.
  }];

  let options = [
    Option<"hoistOnly", "hoist-only", "bool", /*default=*/"false",
      "Hoist the variables only.">
  ];
}

def VerifyAtomicQuantumRegions :
    Pass<"verify-atomic-quantum-regions", "mlir::ModuleOp"> {
  let summary =
      "Reject measurement operations and qubit allocations inside an atomic "
      "quantum region";
  let description = [{
    This pass rejects measurement operations and qubit allocations inside a
    marked `cc.scope` and inside a function carrying the
    `atomic_quantum_region` attribute. It follows a call whose callee it can
    resolve. It does not follow a `quake.apply`.

    When it runs before inlining, the pass covers a violation written in a
    marked kernel and a violation in a helper that the kernel calls. Every
    operation still carries the location of the user source that produced it,
    so the diagnostic names that line.

    When it runs after inlining and `apply-op-specialization`, the pass also
    checks the kernels it could not reach earlier.

    Run the pass before the conversion to value semantics, where a qubit
    allocation stops being a `quake.alloca`.
  }];
}

def WriteAfterWriteElimination : Pass<"write-after-write-elimination"> {
  let summary = "Remove stores that are overridden by subsequent store";
  let description = [{
    Remove stores to a location on the stack that have a subsequent store
    to the same location without a use between them:

    Example:
    ```mlir
    %1 = cc.alloca !cc.array<i64 x 1>
    %2 = cc.cast %1 : (!cc.ptr<!cc.array<i64 x 1>>) -> !cc.ptr<i64>
    cc.store %c0_i64, %2 : !cc.ptr<i64>
    // nothing using %2 until the next instruction
    cc.store %c1_i64, %2 : !cc.ptr<i64>
    ```

    would be converted to

    ```mlir
    %1 = cc.alloca !cc.array<i64 x 1>
    %2 = cc.cast %1 : (!cc.ptr<!cc.array<i64 x 1>>) -> !cc.ptr<i64>
    cc.store %c1_i64, %2 : !cc.ptr<i64>
    ```
  }];
}

#endif // CUDAQ_OPT_OPTIMIZER_TRANSFORMS_PASSES
