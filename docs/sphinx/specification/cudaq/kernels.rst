Quantum Kernels
***************
**[1]** To differentiate between host and quantum device code, the CUDA-Q programming
model defines the concept of a **quantum kernel**. A quantum kernel is any callable
in C++ annotated to indicate compilation and execution on an available quantum coprocessor.

**[2]** All quantum kernels must be annotated to indicate they are to be compiled for, and executed
on, a specified quantum coprocessor. CUDA-Q requires the :code:`__qpu__` function
attribute for quantum kernel declarations. Other language bindings may opt to use other language
features to enable function annotation or decoration (e.g. a :code:`@cudaq.kernel()` function
decorator in Python).

**[3]** CUDA-Q specifically differentiates between kernels invoked from host code and those invoked
from within another quantum kernel. The former are denoted **entry-point**
quantum kernels, the latter are **pure device** quantum kernels. Entry point kernels can
only take classical types as input and return classical output. Pure-device kernels
can take quantum types as input.

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/kernels/kernels_types_example.cpp
     :language: cpp
     :start-after: [Begin Kernel Types C++]
     :end-before: [End Kernel Types C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/kernels/kernels_types_example.py
     :language: python
     :start-after: [Begin Kernel Types Python]
     :end-before: [End Kernel Types Python]

**[4]** Quantum kernel function bodies are programmed in a subset of the parent classical language.
Kernels can be composed of the following:

  * Quantum intrinsic operations and measurements
  * In-scope kernel calls with or without any kernel modifiers (e.g. :code:`cudaq::control`)
  * Classical control flow constructs from the classical language (:code:`if`, :code:`for`, :code:`while`, etc.)
  * Stack variable declarations for supported types.
  * Arithmetic operations on integer and floating point stack variables
  * Coherent conditional execution - :code:`if ( boolExprFromQubitMeasurement ) { x (another_qubit); }`
  * Syntax for common quantum programming patterns (e.g. compute-action-uncompute).


**[5]** CUDA-Q defines a set of allowed types that can be leveraged in quantum kernel
function signatures (input and return types), in-function variable declaration or construction,
and in variable capture from parent scope.

The allowed types are as follows:

  * :code:`T` such that :code:`std::is_arithmetic_v<T> == true`
  * :code:`std::vector<T>` such that :code:`std::is_arithmetic_v<T> == true`
  * :code:`std::span<T>` such that :code:`std::is_arithmetic_v<T> == true`
  * :code:`std::vector<V>` such that type :code:`V` is a valid :code:`std::vector<T>` (possibly recursively)
  * :code:`struct` types composed of any valid CUDA-Q type.
  * :code:`std::tuple<Ts...>` such that T is any allowed type.
  * :code:`std::pair<T,U>` such that T is any allowed type.

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/kernels/kernels_allowed_types.cpp
     :language: cpp
     :start-after: [Begin Allowed Types C++]
     :end-before: [End Allowed Types C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/kernels/kernels_allowed_types.py
     :language: python
     :start-after: [Begin Allowed Types Python]
     :end-before: [End Allowed Types Python]

**[6]** Any variable with an allowed CUDA-Q type can be allocated on the stack within
CUDA-Q kernels. Variables of type :code:`std::vector<T>` for any allowed type
:code:`T` can only be constructed with known size. Vector-like variables cannot be
default constructed and later filled with type :code:`T` data (i.e. no dynamic memory allocation).

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/kernels/kernels_variable_allocation.cpp
     :language: cpp
     :start-after: [Begin Variable Allocation C++]
     :end-before: [End Variable Allocation C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/kernels/kernels_variable_allocation.py
     :language: python
     :start-after: [Begin Variable Allocation Python]
     :end-before: [End Variable Allocation Python]

**[7]** All entry-point kernel arguments adhere to pass-by-value semantics.

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/kernels/kernels_pass_by_value.cpp
     :language: cpp
     :start-after: [Begin Pass By Value C++]
     :end-before: [End Pass By Value C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/kernels/kernels_pass_by_value.py
     :language: python
     :start-after: [Begin Pass By Value Python]
     :end-before: [End Pass By Value Python]


.. FIXME Pass by value vs reference, should we mandate pass by reference for inter-kernel calls

**[8]** CUDA-Q kernel lambdas in C++ can capture variables of allowed type
by value. CUDA-Q kernels defined as custom callable types can define non-reference type
class members of any allowed type. These member variables can be set with
pass-by-value semantics at kernel construction.

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/kernels/kernels_variable_capture.cpp
     :language: cpp
     :start-after: [Begin Variable Capture C++]
     :end-before: [End Variable Capture C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/kernels/kernels_variable_capture.py
     :language: python
     :start-after: [Begin Variable Capture Python]
     :end-before: [End Variable Capture Python]

**[9]** All quantum kernel invocations are synchronous calls by default.

**[10]** CUDA-Q kernels can serve as input to other quantum kernels and invoked by kernel
function body code. To support CUDA-Q kernel parameterization on callable quantum kernel
code, programmers can leverage standard C++ template definitions or dynamic typing in language bindings such as Python:

.. tab:: C++

  .. literalinclude:: /../snippets/cpp/kernels/kernels_composition_templates.cpp
     :language: cpp
     :start-after: [Begin Kernel Composition Templates C++]
     :end-before: [End Kernel Composition Templates C++]

.. tab:: Python

  .. literalinclude:: /../snippets/python/kernels/kernels_composition_callable.py
     :language: python
     :start-after: [Begin Kernel Composition Callable Python]
     :end-before: [End Kernel Composition Callable Python]

CUDA-Q kernel inputs can also be `constrained <https://en.cppreference.com/w/cpp/language/constraints>`_.

.. literalinclude:: /../snippets/cpp/kernels/kernels_composition_constraints.cpp
   :language: cpp
   :start-after: [Begin Kernel Composition Constraints C++]
   :end-before: [End Kernel Composition Constraints C++]

This approach enables the development of generic libraries of quantum
algorithms that are parameterized on sub-units of the global circuit representation.