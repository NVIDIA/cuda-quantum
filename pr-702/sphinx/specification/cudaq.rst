Language Specification 
***********************************

NVIDIA CUDA Quantum is a single-source
programming model in C++ for heterogeneous quantum-classical computing.
As such, the model provides primitive types, concepts, syntax, and
semantics that enable the integration of quantum processors with
existing NVIDIA multi-GPU parallel programming models and architectures. 

CUDA Quantum is a library-based language extension that defines quantum code as typed
callables in standard C++. These callables are meant to be compiled with
standard C++ compilers and run via quantum computer simulators or physical
quantum architectures. When annotated with a unique quantum function
attribute, these functions can be enhanced with standard circuit
optimization and transformation passes. 

.. toctree::
  :numbered:
  :caption: CUDA Quantum
  :maxdepth: 1

  Machine Model <cudaq/machine_model.rst>
  Namespace <cudaq/namespace.rst>
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
