******************
CUDA-Q Compiler IR
******************

CUDA-Q uses MLIR dialects to represent quantum programs and the classical code
that controls them. The generated references in this section come from the
TableGen definitions in the CUDA-Q source tree. Edit the TableGen descriptions
when an operation or type reference needs correction.

Quake
=====

Quake represents quantum operations, quantum data, and measurement. The
:doc:`Quake specification </specification/quake-dialect>` documents semantics
that apply across operations.

Sources: `dialect <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/Quake/QuakeDialect.td>`__,
`operations <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/Quake/QuakeOps.td>`__,
`types <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/Quake/QuakeTypes.td>`__, and
`implementation <https://github.com/NVIDIA/cuda-quantum/tree/main/cudaq/lib/Optimizer/Dialect/Quake>`__.

CC
==

CC represents the classical computation in a CUDA-Q kernel before lowering to
LLVM IR.

Sources: `dialect <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/CC/CCDialect.td>`__,
`operations <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/CC/CCOps.td>`__,
`types <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/CC/CCTypes.td>`__, and
`implementation <https://github.com/NVIDIA/cuda-quantum/tree/main/cudaq/lib/Optimizer/Dialect/CC>`__.

QEC
===

QEC represents declarations that associate measurement results with detectors
and logical observables.

Sources: `dialect <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/QEC/QECDialect.td>`__,
`operations <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/Dialect/QEC/QECOps.td>`__, and
`implementation <https://github.com/NVIDIA/cuda-quantum/tree/main/cudaq/lib/Optimizer/Dialect/QEC>`__.

CodeGen
========

CodeGen is an internal dialect used while lowering CUDA-Q IR toward target
representations. Its definitions and implementation are the current source of
truth. CUDA-Q does not publish a generated CodeGen reference.

Sources: `dialect <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/CodeGen/CodeGenDialect.td>`__,
`operations <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/CodeGen/CodeGenOps.td>`__,
`types <https://github.com/NVIDIA/cuda-quantum/blob/main/cudaq/include/cudaq/Optimizer/CodeGen/CodeGenTypes.td>`__, and
`implementation <https://github.com/NVIDIA/cuda-quantum/tree/main/cudaq/lib/Optimizer/CodeGen>`__.

.. toctree::
   :caption: Generated References
   :maxdepth: 1

      Quake Operation and Type Reference <quake-reference>
      CC Operation and Type Reference <cc-reference>
      QEC Operation Reference <qec-reference>
