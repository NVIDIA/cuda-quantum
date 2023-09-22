Quantum Kernels
***************
To differentiate between host and quantum device code, the CUDA Quantum programming
model defines the concept of a **quantum kernel**. CUDA Quantum specifically 
differentiates between kernels invoked from host code and those invoked
from within another quantum kernel. The former are denoted **entry-point**
quantum kernels, the latter are **pure device** quantum kernels. All quantum
kernels must be annotated to indicate they are to be compiled to and executed
on a specified quantum coprocessor. CUDA Quantum requires the **__qpu__** function
attribute for quantum kernel declarations. 

Quantum kernel function bodies are programmed in a subset of C++. They can be composed of the following: 

  * quantum intrinsic operations and measurements
  * in-scope kernel calls with or without any kernel modifiers
  * classical control flow constructs (:code:`if`, :code:`for`, :code:`while`, etc.)
  * primitive variable declarations and arithmetic manipulations (:code:`std::is_arithmetic<T>` types)
  * coherent conditional execution ( :code:`if ( qubit ) { x (another_qubit); }` ) 
  * novel syntax for common quantum programming patterns (e.g. compute-action-uncompute).
  * kernels may reference and call previously defined quantum kernels, but cannot operate on global data.

An entry-point quantum kernel must be defined as a `typed callable <https://en.cppreference.com/w/cpp/utility/functional>`_
(i.e. a lambda, or struct with :code:`R operator()(Args...)` implemented)
that can be annotated with an appropriate function attribute. This requirement
on typed-callables directly enables the implementation of generic libraries
of quantum algorithms parameterized on user-specified quantum kernels. CUDA Quantum
requires these typed quantum callables to be annotated with the :code:`__qpu__`
attribute preceding the left brace that opens the function body declaration: 

.. code-block:: cpp

    auto my_first_kernel = [](double x) __qpu__ { ... quantum code ... };
    struct my_second_kernel { 
      void operator()(double x) __qpu__ { 
        ... quantum code ...
      }
    };

Entry-point quantum kernels expressed as structs or classes with an :code:`operator()(...)`
overload may leverage primitive class members within the kernel body, 
specifically any type by which :code:`std::is_arithmetic` evaluates to :code:`true`. 

All quantum kernels can specify a return type from the set 
:code:`{void, T : std::is_arithmetic<T> == true, std::vector<bool>}`. 
All quantum kernels can take as input any type in the set 
:code:`{T : std::is_arithmetic<T> == true, std::vector<T>, std::span<T>}`. All 
kernels can take :code:`cudaq:spin_op` instances as input. 
Entry-point quantum kernels cannot take quantum input arguments because quantum
memory cannot be allocated from within host code. 

Pure device quantum kernels can be expressed as typed-callables, but can also
be represented as annotated free functions. Pure device quantum kernels can take
:code:`cudaq::qudit<N>` specializations and containers (e.g. 
:code:`cudaq::qspan`, :code:`cudaq::qreg`) as input. 

.. code-block:: cpp

    auto my_first_device_kernel = [](cudaq::qreg<>& q) __qpu__ { 
       ... quantum code using q ...
       };
    struct my_second_device_kernel { 
      void operator()(cudaq::qubit& q, double x) __qpu__ { 
        ... quantum code ...
      } 
    };
    __qpu__ void my_third_device_kernel(cudaq::qubit& qb) { 
        ... quantum code using qb ...
    }

Classical arithmetic data can be instantiated and manipulated within
any quantum kernel and is modeled implicitly using the quantum device
classical control memory space. Returning classical data requires an
implicit data transfer from device to host, and this should be configured
by the compiler implementation. 

CUDA Quantum kernels expressed as lambda expressions can capture simple arithmetic
variables by value. Specifically, any valid input type for a CUDA Quantum kernel 
function argument can also be provided as a variable captured by value. 

All quantum kernel invocations are synchronous calls by default. 

Kernel Composability
--------------------
CUDA Quantum kernels can also serve as input to other quantum kernels. This is a typical 
pattern in quantum computing, whereby you have an algorithm that is dependent on 
some input sub-circuit, e.g. for state preparation, oracle invocation, etc. In order to
to support this pattern, CUDA Quantum kernels can be passed as arguments for indirect invocation.

CUDA Quantum builds upon C++ to enable this capability. To support CUDA Quantum kernel parameterization 
on callable quantum kernel code, programmers can leverage standard C++ template definitions:

.. code-block:: cpp 

  struct MyStatePrep {
    void operator()(cudaq::qspan<> qubits) __qpu__ {
      ... apply state prep operations on qubits ...
    }
  };

  struct MyGenericAlgorithm {
    template<typename StatePrep>
    void operator()(StatePrep&& statePrep) __qpu__ {
      cudaq::qreg<10> q;
      statePrep(q);
      ...
    }
  };

  // -or- with placeholder type specifiers
  struct MyGenericAlgorithm2 {
    void operator()(auto&& statePrep) __qpu__ {
      cudaq::qreg<10> q;
      statePrep(q);
      ...
    }
  };

  MyGenericAlgorithm algorithm;
  algorithm(MyStatePrep{});

  MyGenericAlgorithm2 anotherVersion;
  anotherVersion(MyStatePrep{});

CUDA Quantum kernel inputs can also be `constrained <https://en.cppreference.com/w/cpp/language/constraints>`_. 

.. code-block:: cpp 

  namespace cudaq {

    // Generic constraint on Kernel Function Signatures
    template <typename Kernel, typename Signature>
    concept signature = std::is_convertible_v<Kernel, std::function<Signature>>; 

    // Specialized for taking a single qubit
    template<typename Kernel>
    concept takes_qubit = signature<Kernel, void(qubit&)>;
  }

  struct MyGenericAlgorithmOnQreg {
    void operator()(cudaq::signature<void(cudaq::qreg&)> auto&& statePrep) __qpu__ {
      cudaq::qreg<10> q;
      statePrep(q);
      ...
    }
  };

  struct MyGenericAlgorithmOnQubit {
    void operator()(cudaq::takes_qubit auto&& statePrep) __qpu__ {
      cudaq::qreg<10> q;
      statePrep(q[0]);
      ...
    }
  };

This approach enables the development of generic libraries of quantum 
algorithms that are parameterized on sub-units of the global circuit representation. 

Allowed Kernel Classical Function Invocations
---------------------------------------------
TODO