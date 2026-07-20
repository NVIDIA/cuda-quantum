CUDA-Q compiler IR
******************

MLIR organizes compiler IR into dialects, each of which defines operations,
types, attributes, and interfaces for a particular abstraction. A single MLIR
module can combine those abstractions. This lets a transformation replace one
part of a program without first lowering every other part to the same level.
The `MLIR language reference <https://mlir.llvm.org/docs/LangRef/>`_ describes
the common operation and region structure shared by all dialects.

The C++ and Python frontends create a ``builtin.module`` whose kernel functions
can contain Quake quantum operations, CC classical operations, and upstream
MLIR operations for functions and arithmetic. A kernel can also contain QEC
declarations. Later transformations can introduce ``CodeGen`` helpers or lower
selected operations to the LLVM dialect.
The dialects present in a module therefore depend on where the module was
captured and which target-selected transformations have run.

Changes to IR semantics can affect the language frontends, compiler passes,
pipelines, translations, and downstream consumers. Discuss a proposed change
with the CUDA-Q compiler team before implementation, including the problem it
solves and why the existing IR is insufficient. Reusing existing semantics or
adding a focused transformation is often simpler than extending the IR.

For an agreed IR change, state the operations and forms that the transformation
accepts, the properties it preserves, and the IR it produces. Record these
expectations in the implementation and its tests.

CUDA-Q dialects
===============

Quake
-----

Quake is CUDA-Q's hardware-independent quantum dialect. It represents quantum
allocation and lifetime, quantum references and registers, gates,
measurements, and calls between quantum kernels. Frontend IR commonly uses
reference-oriented types such as ``!quake.ref`` and ``!quake.veq``. Quake also
defines value-oriented forms such as ``!quake.wire`` and ``!quake.control`` so
that transformations can expose quantum data flow explicitly. A pass that
requires one of these forms must state that requirement explicitly.

The authored
:doc:`Quake semantic specification <../../../specification/quake-dialect>`
explains the reference and value models and the reasoning behind them.

.. only:: compiler_developer_docs

   See the :doc:`generated Quake dialect documentation
   </_mdgen/Dialects/Quake>` for operation and type details.

CC
--

CC represents the classical computation that remains inside a CUDA-Q kernel.
Its types cover source-level concepts such as pointers, arrays, structures,
vectors, callables, and measurement handles. Its operations retain structured
scope, loop, conditional, memory, and callable behavior before those concepts
are lowered to target forms. Both CUDA-Q frontends use CC alongside Quake, so
classical control and quantum operations can remain interleaved while their
relationship is still visible.

.. only:: compiler_developer_docs

   See the :doc:`generated CC dialect documentation
   </_mdgen/Dialects/CC>` for operation and type details.

QEC
---

QEC holds quantum error-correction declarations that are distinct from circuit
operations. Its detector and logical-observable operations describe parity
relationships over measurement results. These operations consume CC
measurement handles, which is one concrete example of CUDA-Q dialects sharing
types and semantics within the same module.

.. only:: compiler_developer_docs

   See the :doc:`generated QEC dialect documentation
   </_mdgen/Dialects/QEC>` for operation details.

Code generation helpers
-----------------------

``CodeGen`` contains temporary helper operations used during code generation.
The helpers preserve information needed while Quake and CC operations are
being converted to lower-level target forms, including QIR and the LLVM
dialect. ``CodeGen`` is not frontend IR or a general-purpose optimization
dialect. A pass should produce ``CodeGen`` operations only when they are
required by that code generation path.

.. only:: compiler_developer_docs

   See the generated ``CodeGen`` :doc:`dialect documentation
   </_mdgen/Dialects/CodeGen>` for operation details.

Upstream MLIR dialects
----------------------

CUDA-Q reuses upstream MLIR dialects instead of duplicating common compiler
concepts. The built-in and ``Func`` dialects provide module and function
structure. ``Arith``, ``Math``, and ``Complex`` operations represent common
computations. ``ControlFlow`` and ``LLVM`` operations appear as transformations
lower structured source semantics and prepare for translation. Individual
conversions can declare additional dependent dialects when they introduce
their operations.

The `upstream dialect documentation <https://mlir.llvm.org/docs/Dialects/>`_
defines these operations and types. Which upstream forms are valid depends on
the current compiler boundary.

Source and tests
================

Quake, CC, and QEC declarations are under
``cudaq/include/cudaq/Optimizer/Dialect``. ``CodeGen`` declarations are under
``cudaq/include/cudaq/Optimizer/CodeGen``, and their implementations are under
``cudaq/lib/Optimizer``. Focused C++ tests for dialect operations and interfaces
are under ``cudaq/unittests/Optimizer``. Textual IR coverage is under
``cudaq/test/Transforms``, ``cudaq/test/AST-Quake``, and ``python/tests/mlir``.
