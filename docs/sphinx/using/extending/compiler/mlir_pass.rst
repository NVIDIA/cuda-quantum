:orphan:

Developing compiler passes
**************************

CUDA-Q represents its compiler IR using MLIR and uses the MLIR pass manager to
schedule work over the mixed-dialect IR described in :doc:`cudaq_ir`. A compiler
pass is a scheduled unit of work over an MLIR operation. It should run at a
defined IR boundary and clearly document the valid input IR and the IR it leaves
for the next compiler step. This includes the required operations, types,
attributes, and structural properties, along with the output guarantees and
failure conditions.

The `MLIR pass infrastructure <https://mlir.llvm.org/docs/PassManagement/>`_
defines the scheduling, nesting, analysis, failure, and registration rules used
by CUDA-Q passes.

Decide whether a new pass is needed
===================================

Before adding a pass, review the :doc:`pass catalog <available_passes>` and
check whether the change belongs in an existing pass that already works at the
same point in the pipeline and on the same form of IR. Extend the existing pass
when the new behavior is part of the same transformation and should run whenever
that pass runs. Add a separate pass when the work needs independent scheduling,
has its own options or diagnostics, is useful in other pipelines, or expects a
different form of IR. A local simplification that is valid wherever an operation
appears may be better expressed as folding or canonicalization.

Identify where and when the pass will run
=========================================

Start with the producer and consumer of the IR rather than a pass name. Capture
the IR immediately before the proposed transformation and identify the next
component that depends on its result. Existing pipeline definitions and lit
tests provide useful inputs, but they do not replace checking the operations,
types, attributes, symbol relationships, and control flow that are present at
that boundary.

General-purpose optimization passes should stay within a single block when
possible and use SSA use-def chains, operation traits, and interfaces to
understand dependencies. Reading operations in textual order is not enough once
the IR contains branches, loops, nested regions, or operations with unknown
effects. Moving, combining, or canceling operations across those boundaries
requires explicit reasoning about dominance, control flow, and effects. Such a
pass should document the forms it supports and test branches, loops, joins, and
multiple exits.

Compilation stages
------------------

The following sections describe the current working boundaries between
compiler stages. The CUDA-Q compiler team is still formalizing these stages and
expanding their documentation.

Frontend IR construction and early cleanup
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passes in this stage remove language-specific details and produce the
mixed-dialect IR expected by the shared compiler pipelines. Frontend-specific
passes may handle differences in IR construction, but passes after this
boundary should accept equivalent C++ and Python kernel forms.

High-level normalization and specialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passes in this stage expose calls, arguments, loops, and kernel variants that
later transformations need. Each pass should state whether it requires calls to
be inlined, arguments to be constant, loops to be normalized, or a particular
kernel variant to have been selected.

Quantum transformation and optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passes in this stage change gates, measurements, allocations, and other quantum
semantics. A pass should state whether it is target-independent or depends on a
gate set, device topology, execution profile, or another target constraint.

Every quantum optimization must preserve the computation represented by its
input IR. A transformation with this property is *semantics-preserving*. A
pass can produce valid IR and reduce gate counts or depth while still changing
what the program does, so correctness must be considered separately from IR
validity and optimization quality.

The pass description should explain when the transformation is valid and what
behavior it preserves. The implementation should apply the transformation only
when it can establish those conditions from the IR. Other cases should remain
unchanged or produce a diagnostic when the pipeline requires the
transformation.

For unitary IR, preserving the computation normally means implementing the same
unitary. If a pass allows a different relation, such as equivalence up to global
phase, equivalence under a known qubit mapping, or a bounded approximation, it
should state that explicitly. Passes involving measurement or other
non-unitary behavior should describe the observable behavior they preserve.

Quantum optimizations that reason about gate order or quantum dependencies
should use Quake's value form unless the transformation specifically needs
Quake's reference semantics, for example to reason about allocation, lifetime,
or aliasing. ``!quake.wire`` and ``!quake.cable`` are linear types, so their
use-def chains expose how quantum state flows between operations without
reconstructing aliases between ``!quake.ref`` and ``!quake.veq`` values. The
:doc:`Quake semantic specification <../../../specification/quake-dialect>`
explains the reference and value models and the boundaries formed by
``quake.unwrap`` and ``quake.wrap``.

Quantum optimization passes should use these value chains and Quake operation
interfaces instead of reconstructing a circuit by scanning operations. They
should preserve the exactly-once use of linear values. A pass that crosses a
branch or region should thread quantum values through the relevant block
arguments and region results and use the appropriate dominance, control-flow,
and effect analyses.

Pipeline integration should keep related quantum optimizations together in a
value-form portion of the pipeline. A quantum optimization should not require
its own ``memtoreg`` and ``regtomem`` round trip. If a downstream path requires
reference form, the pipeline should convert after the value-form work is
complete. A pass that intentionally accepts reference or mixed form should
document why and test the relevant aliasing and conversion boundaries.

IR form conversion and structural lowering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Conversion passes move between structured control flow and control-flow graphs,
memory and value forms, and CUDA-Q or upstream dialects. They should make both
sides of the conversion explicit and keep semantic optimization separate from
representation changes unless one requires the other.

Target preparation, lowering, and emission
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Passes in this stage adapt the module to the selected target or execution
profile and prepare the representation for final code emission. A pass should
state which target constraints it depends on and what representation it
produces for the next compiler step or emitter.

.. _compiler-pass-input-output-ir:

Identify the input and output IR
================================

Write a small before-and-after example that includes the operation containing
the transformation. Prefer to scope the pass to one operation and its nested
IR, such as a ``func.func``. Use a ``builtin.module`` pass only when the
transformation must coordinate symbols or changes across several functions or
other module-level operations. MLIR operation passes must not inspect or modify
sibling operations outside their current operation.

Mixed-dialect input is normal. The pass description should identify the
compiler stage and IR form it expects, the assumptions that affect correctness
or pipeline placement, and the form it produces. For a quantum optimization,
this might include whether it consumes Quake reference or value semantics and
whether it is target-independent. Focus on meaningful boundaries rather than
enumerating every dialect, operation, or program feature that may appear.

Define expected behavior
========================

The pass's TableGen ``description`` and regression tests should name the
accepted input requirements, guaranteed output properties, behavior for input
outside the match, and conditions that produce a diagnostic. Options need
defaults and descriptions that make separate pass instances understandable in
a textual pipeline.

List a dialect in ``dependentDialects`` when the pass may create its operations,
types, or attributes and that dialect is not otherwise guaranteed to be loaded.
This lets the pass manager load dependencies before a multithreaded pipeline
starts. Signal pass failure after emitting a diagnostic when the pass cannot
produce its stated output. Do not leave invalid IR for a later pass to diagnose.

Analyses are invalidated when a transformation changes IR unless the pass marks
them preserved. A pass that mutates symbols, regions, types, or control flow
should make that effect clear in its implementation and tests. Running
``cudaq-opt`` with ``--verify-each`` is useful while establishing these output
conditions.

Choose the MLIR transformation mechanism
========================================

Use an operation's fold hook or canonicalization patterns for a local canonical
form that is valid whenever the operation appears. For example, the
``cc.cast`` canonicalizer removes a cast when its operand and result have the
same type. This rule is local and always valid, so it does not need a dedicated
pass.

Use a rewrite pattern for a local DAG transformation whose match can decline
without making the IR illegal. The
`pattern rewriting framework <https://mlir.llvm.org/docs/PatternRewriter/>`_
provides greedy and walk-based drivers. Mutate matched IR through the supplied
``PatternRewriter`` so the driver can maintain its work queue and listeners.

Use
`dialect conversion <https://mlir.llvm.org/docs/DialectConversion/>`_ when the
expected output is naturally expressed as legal and illegal operations or when
types and region signatures must be converted. Choose full, partial, or
analysis conversion according to the promised output. A full conversion is
appropriate only when every illegal operation must be removed.

Use an analysis when the work computes information without changing IR. Use an
`operation or dialect interface <https://mlir.llvm.org/docs/Interfaces/>`_ when
an algorithm needs behavior supplied by several operation kinds without a
central type switch. Use a dedicated operation pass when the work must be
scheduled in a pipeline, expose options, emit diagnostics, or coordinate a
bounded set of rewrites, conversions, and analyses. These mechanisms are often
components of the pass rather than alternatives to it.

Implement and register a built-in pass
======================================

General transformations are declared in
``cudaq/include/cudaq/Optimizer/Transforms/Passes.td``. Passes whose primary
responsibility is final lowering and code generation are declared in
``cudaq/include/cudaq/Optimizer/CodeGen/Passes.td``.

Add a TableGen definition for the pass. The definition identifies its
command-line name and operation anchor, documents the expected input and output,
and declares any dependent dialects or options. This example defines a
``func.func`` pass that can create Quake and control-flow operations:

.. :spellcheck-disable:

.. code:: tablegen

   def ExampleRewrite : Pass<"example-rewrite", "mlir::func::FuncOp"> {
     let summary = "Rewrite the example form.";
     let description = [{
       Describe the accepted input, the guaranteed output, and failure cases.
     }];
     let dependentDialects = ["cudaq::quake::QuakeDialect",
                              "mlir::cf::ControlFlowDialect"];
     let options = [
       Option<"strict", "strict", "bool", /*default=*/"false",
              "Diagnose input outside the supported form.">
     ];
   }

.. :spellcheck-enable:

TableGen generates the base class, factories, option storage, registration
functions, and catalog entry. The implementation selects its generated base in
the corresponding ``.cpp`` file:

.. :spellcheck-disable:

.. code:: cpp

   #include "PassDetails.h"
   #include "cudaq/Optimizer/Transforms/Passes.h"

   namespace cudaq::opt {
   #define GEN_PASS_DEF_EXAMPLEREWRITE
   #include "cudaq/Optimizer/Transforms/Passes.h.inc"
   } // namespace cudaq::opt

   namespace {
   class ExampleRewritePass
       : public cudaq::opt::impl::ExampleRewriteBase<ExampleRewritePass> {
   public:
     using ExampleRewriteBase::ExampleRewriteBase;
     void runOnOperation() override;
   };

   void ExampleRewritePass::runOnOperation() {
     // Check the input requirements, perform the rewrite, and signal failure when
     // the promised output cannot be produced.
   }
   } // namespace

.. :spellcheck-enable:

Add the implementation file to ``OptTransforms`` or ``OptCodeGen`` in the
corresponding ``cudaq/lib/Optimizer`` CMake file. A normal declarative pass does
not need a handwritten registration call. ``cudaq-opt`` calls
``cudaq::registerAllPasses()``, which registers both generated CUDA-Q pass sets
and the registered CUDA-Q pipelines. Add an explicit declaration to
``Passes.h`` only when callers need a helper or factory overload that TableGen
does not generate.

Build and run the pass
======================

Rebuild the optimizer after changing a TableGen record or implementation:

.. :spellcheck-disable:

.. code:: bash

   cmake --build build --target cudaq-opt -j8

.. :spellcheck-enable:

Run the pass on the smallest IR that exercises its expected behavior. An
explicit pipeline makes the operation nesting and options visible:

.. :spellcheck-disable:

.. code:: bash

   build/bin/cudaq-opt \
     --pass-pipeline='builtin.module(func.func(example-rewrite{strict=true}))' \
     input.qke -o -

.. :spellcheck-enable:

Nest the pass under the operation named by its definition. A module pass runs
directly under ``builtin.module``. A function pass must run under the matching
``func.func`` or ``llvm.func`` operation manager. Use
``--dump-pass-pipeline`` to inspect the resulting pipeline, ``--verify-each``
to find the first pass that produces invalid IR, and
``--mlir-print-ir-before=<pass-argument>`` or
``--mlir-print-ir-after=<pass-argument>`` to examine the IR around a selected
transformation. The ``--mlir-print-ir-before-all`` and
``--mlir-print-ir-after-all`` variants print around every pass. ``cudaq-opt
--help`` lists the passes and pipelines available in the current build. The
:doc:`available pass catalog <available_passes>` documents CUDA-Q's built-in
passes and their options.

Test pass behavior
==================

Compiler pass regressions normally belong under ``cudaq/test/Transforms`` as
small textual IR tests. Use ``cudaq-opt`` in a ``RUN`` line and ``FileCheck`` to
verify the relevant IR structure. Include the shortest positive case, important
input outside the match that must remain unchanged, option-dependent behavior,
and the boundary forms described by the pass. Check diagnostics with
``-verify-diagnostics``. ``-split-input-file`` keeps independent positive or
negative cases in one file when that improves readability.

Validate quantum transformations
--------------------------------

Tests for a quantum transformation should show both that the intended rewrite
occurred and that the computation was preserved. Use ``FileCheck`` to check the
resulting IR. For bounded unitary IR supported by ``CircuitCheck``, use
``CircuitCheck`` to compare the original and transformed functions. Use
``--up-to-global-phase`` or ``--up-to-mapping`` only when the pass explicitly
claims that form of equivalence.

Include representative transformations, the conditions on which they depend,
and nearby inputs that must remain unchanged. ``CircuitCheck`` does not support
every form of quantum IR, so transformations outside its scope need tests
appropriate to the behavior they preserve. Simulation and sampling can expose
defects, but agreement on selected inputs does not establish correctness for
every supported input.

Test the pass directly before adding a pipeline test. Add a pipeline-level test
when correctness depends on a predecessor, successor, option forwarding, or
operation nesting. A target test is justified when behavior depends on target
configuration or final emission rather than the pass's textual IR
boundary. These choices follow the distinction in the
`MLIR testing guide <https://mlir.llvm.org/getting_started/TestingGuide/>`_
between focused transformation tests and more expensive integration tests.

The configured build exposes the full lit target and supports filtered local
runs. Activate the Python environment used to configure CMake before invoking
``llvm-lit`` directly:

.. :spellcheck-disable:

.. code:: bash

   cmake --build build --target check-cudaq
   python "$LLVM_INSTALL_PREFIX/bin/llvm-lit" -sv \
     --filter example-rewrite build/cudaq/test

.. :spellcheck-enable:

Integrate with a pipeline
=========================

Keep a pass independently runnable even when its production use is in a
pipeline. Add it to a shared pipeline only when every target using that pipeline
produces the pass's expected input and needs its output. Test the pass directly
before adding a pipeline-level test for its surrounding transformations and any
option forwarding. When placement depends on a target or output format, also
test the relevant target configuration or final emission path.

Develop an external pass plugin
===============================

Passes maintained outside the CUDA-Q source tree follow the same design and
testing guidance, but are built and registered as shared-library plugins. See
:doc:`External compiler pass plugins <pass_plugins>` for the plugin interface,
the maintained example, and the ``cudaq-opt`` loading workflow.
