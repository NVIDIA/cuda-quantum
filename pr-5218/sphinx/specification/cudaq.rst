Language Specification 
***********************************

NVIDIA CUDA-Q is a programming model in C++ for heterogeneous 
quantum-classical computing. As such, the model provides primitive types, concepts, syntax, and
semantics that enable the integration of quantum processing units (QPUs) with
existing NVIDIA multi-node, multi-GPU parallel programming models and architectures. 
The language is designed for C++. Extensions to other languages is fully supported 
via appropriate bindings. 

CUDA-Q implementations are intended to be enabled via library-based language 
extensions that define quantum code as callables in the classical language being extended. 
These callables are meant to be compiled with standard compilers and runtimes 
and run via quantum computer simulators or available physical quantum architectures. The 
language callables intended for quantum coprocessing must be annotated for 
compilers and runtimes to process. These annotations should directly enable quantum-specific 
optimization and transformation for optimal execution on simulators and QPUs. 

The rest of this document will detail all aspects of the language specification - its machine model, 
primitive types and operations, core concepts and abstractions, and algorithmic primitives. All code 
specification and demonstrative examples are provided in C++ (the native CUDA-Q classical language). 
Where possible, bindings for higher-level languages (e.g. Python) will be displayed to aid in understanding.

.. toctree::
  :numbered:
  :caption: CUDA-Q
  :maxdepth: 1

  Machine Model <cudaq/machine_model.rst>
  Namespace and Standard <cudaq/namespace.rst>
  Quantum Types <cudaq/types.rst>
  Quantum Operators <cudaq/operators.rst>
  Quantum Operations <cudaq/operations.rst>
  Quantum Kernels <cudaq/kernels.rst>
  Sub-circuit Synthesis <cudaq/synthesis.rst>
  Control Flow <cudaq/control_flow.rst>
  Just-in-Time Kernel Creation <cudaq/dynamic_kernels.rst>
  Quantum Patterns <cudaq/patterns.rst>
  Platform <cudaq/platform.rst>
  Algorithmic Primitives <cudaq/algorithmic_primitives.rst>
  Example Programs <cudaq/examples.rst>
