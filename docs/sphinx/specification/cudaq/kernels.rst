Quantum Kernels
***************
**[1]** To differentiate between host and quantum device code, the CUDA Quantum programming
model defines the concept of a **quantum kernel**. A quantum kernel is any callable 
in C++ annotated to indicate compilation and execution on an available quantum coprocessor. 

**[2]** All quantum kernels must be annotated to indicate they are to be compiled to and executed
on a specified quantum coprocessor. CUDA Quantum requires the :code:`__qpu__` function
attribute for quantum kernel declarations. 

**[3]** CUDA Quantum specifically differentiates between kernels invoked from host code and those invoked
from within another quantum kernel. The former are denoted **entry-point**
quantum kernels, the latter are **pure device** quantum kernels. Entry point kernels can 
only take classical types as input and return classical output. Pure-device kernels 
can take quantum types as input.

.. tab:: C++ 
  
  .. code-block:: cpp
  
      // Entry point lambda kernel
      auto my_first_entry_point_kernel = [](double x) __qpu__ { ... quantum code ... };
      
      // Entry point typed callable kernel
      struct my_second_entry_point_kernel { 
        void operator()(double x, std::vector<double> params) __qpu__ { 
          ... quantum code ...
        }
      };

      // Pure device free function kernel
      __qpu__ void my_first_pure_device_kernel(cudaq::qview<> qubits) {
         ... quantum code ... 
      }

.. tab:: Python 

  .. code-block:: python 

    @cudaq.kernel()
    def my_first_entry_point_kernel(x : float):
       ... quantum code ... 
    
    @cudaq.kernel()
    def my_second_entry_point_kernel(x : float, params : list[float]):
       ... quantum code ... 

    @cudaq.kernel()
    def my_first_pure_device_kernel(qubits : cudaq.qview):
       ... quantum code ... 
    
**[4]** Quantum kernel function bodies are programmed in a subset of C++. Kernels can be composed of the following: 

  * Quantum intrinsic operations and measurements
  * In-scope kernel calls with or without any kernel modifiers (e.g. :code:`cudaq::control`)
  * Classical control flow constructs from the classical language (:code:`if`, :code:`for`, :code:`while`, etc.)
  * Stack variable declarations for supported types. 
  * Arithmetic operations on integer and floating point stack variables
  * Coherent conditional execution ( :code:`if ( boolExprFromQubitMeasurement ) { x (another_qubit); }` ) 
  * Syntax for common quantum programming patterns (e.g. compute-action-uncompute).

**[5]** Entry-point quantum kernels expressed as structs or classes with an :code:`operator()(...)`
overload may leverage primitive class members within the kernel body, 
specifically any type by which :code:`std::is_arithmetic_v` evaluates to :code:`true`. 

**[6]** All quantum kernels can specify a return type from the set 
:code:`{void, T : std::is_arithmetic_v<T> == true, std::vector<T>}` as 
well as aggregate :code:`struct` types composed of types in this set. 

**[7]** All quantum kernels can take as input any type in the set 
:code:`{T : std::is_arithmetic_v<T> == true, std::vector<T>, std::span<T>}` 
as well as aggregate :code:`struct` types composed of types in this set. Type :code:`T` 
in :code:`std::vector<T>` can recursively contain :code:`std::vector<S>` on any type :code:`S` 
that is a valid kernel input type (e.g. :code:`std::vector<std::vector<std::size_t>>`). 

**[8]** Pure device kernels can take :code:`cudaq::qudit<N>` specializations and containers (e.g. 
:code:`cudaq::qview`, :code:`cudaq::qvector`) as input. 

**[9]** CUDA Quantum kernels expressed as lambda expressions can capture any valid 
entry-point kernel input type by value. 

**[10]** All quantum kernel invocations are synchronous calls by default. 

**[11]** CUDA Quantum kernels can serve as input to other quantum kernels and invoked by kernel function body code. 

**[12]** To support CUDA Quantum kernel parameterization on callable quantum kernel code, programmers can leverage 
standard C++ template definitions or dynamic typing in language bindings such as Python:

.. tab:: C++ 

  .. code-block:: cpp 

      __qpu__ void MyStatePrep(cudaq::qview<> qubits) {
          ... apply state prep operations on qubits ...
      }

      struct MyGenericAlgorithm {
        template<typename StatePrep>
        void operator()(const StatePrep& statePrep) __qpu__ {
          cudaq::qarray<10> q;
          statePrep(q);
          ...
        }
      };

      // -or- with placeholder type specifiers
      struct MyGenericAlgorithm2 {
        void operator()(const auto& statePrep) __qpu__ {
          cudaq::qarray<10> q;
          statePrep(q);
          ...
        }
      };

      void callKernels() {

        MyGenericAlgorithm algorithm;
        algorithm(MyStatePrep);

        MyGenericAlgorithm2 anotherVersion;
        anotherVersion(MyStatePrep);
      }

.. tab:: Python 

  .. code-block:: python 

    @cudaq.kernel()
    def MyStatePrep(qubits : cudaq.qview):
        ... apply state prep operations on qubits ... 
    
    @cudaq.kernel()
    def MyGenericAlgorithm(statePrep : typing.Callable[[cudaq.qview], None]):
        q = cudaq.qvector(10)
        statePrep(q)
        ...
    
    MyGenericAlgorithm(MyStatePrep)

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

    struct MyGenericAlgorithmOnQarray {
      void operator()(cudaq::signature<void(cudaq::qarray&)> auto&& statePrep) __qpu__ {
        cudaq::qarray<10> q;
        statePrep(q);
        ...
      }
    };

    struct MyGenericAlgorithmOnQubit {
      void operator()(cudaq::takes_qubit auto&& statePrep) __qpu__ {
        cudaq::qarray<10> q;
        statePrep(q[0]);
        ...
      }
    };

This approach enables the development of generic libraries of quantum 
algorithms that are parameterized on sub-units of the global circuit representation. 
