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

  .. code-block:: cpp

    struct MyCustomSimpleStruct {
       int i = 0;
       int j = 0;
       std::vector<double> angles;
    }; 

    // Valid CUDA-Q Types used in Kernels
    auto kernel = [](int N, bool flag, float angle, std::vector<std::size_t> layers,
             std::vector<double> parameters, std::vector<std::vector<float>> recursiveVec, 
             MyCustomSimpleStruct var) __qpu__ { ... }
    
    __qpu__ double kernelThatReturns() { 
       ... 
       return M_PI_2;
    }
    
.. tab:: Python 

  .. code-block:: python 

    class MySimpleStruct(object):
        def __init__(self):
            self.i = 0 
            self.j = 0

    @cudaq.kernel
    def kernel(N : int, flag : bool, angle : float, layers : list[int], 
                parameters : list[float], recursiveList : list[list[float]], 
                var : MySimpleStruct): ... 
    
    @cudaq.kernel 
    def kernelThatReturns() -> float:
        ... 
        return np.pi / 2.0

**[6]** Any variable with an allowed CUDA-Q type can be allocated on the stack within 
CUDA-Q kernels. Variables of type :code:`std::vector<T>` for any allowed type 
:code:`T` can only be constructed with known size. Vector-like variables cannot be 
default constructed and later filled with type :code:`T` data (i.e. no dynamic memory allocation). 

.. tab:: C++ 

  .. code-block:: cpp

    // Valid CUDA-Q Types used in Kernels
    auto kernel = []() __qpu__ {
      
      // Not Allowed. 
      // std::vector<int> i; 
      // i.push_back(1);

      // Valid variable declarations

      std::vector<int> i(5);
      i[2] = 3; 

      std::vector<float> f {1.0, 2.0, 3.0};

      int k = 0;

      double pi = 3.1415926
    };
    
.. tab:: Python 

  .. code-block:: python 

    @cudaq.kernel
    def kernel(): 
       # Not Allowed 
       # i = [] 
       # i.append(1)

       i = [0 for k in range(5)]
       i[2] = 3 

       f = [1., 2., 3.]

       k = 0 

       pi = 3.1415926

**[7]** All entry-point kernel arguments adhere to pass-by-value semantics. 

.. tab:: C++ 

  .. code-block:: cpp 

    auto kernel = [](int i, std::vector<double> v) __qpu__ {
       // i == 2, allocate 2 qubits 
       cudaq::qvector q(i); 
       // v[1] == 2.0, angle here is 2.0
       ry(v[1], q[0]);

       // Change the variables, caller does not see this
       i = 5; 
       v[0] = 3.0;
    };

    int k = 2;
    std::vector<double> d {1.0, 2.0};
    
    kernel(k, d);

    // k is still 2, pass by value 
    // d is still {1.0, 2.0}, pass by value 
  
.. tab:: Python 

  .. code-block:: python 

    @cudaq.kernel 
    def kernel(i : int, v : list[float]): 
        # i = 2, allocate 2 qubits 
        q = cudaq.qvector(i)
        # v[1] == 2.0, angle here is 2.0 
        ry(v[1], q[0])

        # Change the variables, caller does not see this 
        i = 5 
        v[0] = 3.0 

    k, d = 2, [1., 2.]
    kernel(i, d)

    # k is still 2, pass by value 
    # d is still {1.0, 2.0}, pass by value 


.. FIXME Pass by value vs reference, should we mandate pass by reference for inter-kernel calls

**[8]** CUDA-Q kernel lambdas in C++ can capture variables of allowed type 
by value. CUDA-Q kernels defined as custom callable types can define non-reference type 
class members of any allowed type. These member variables can be set with 
pass-by-value semantics at kernel construction. 

.. tab:: C++ 

  .. code-block:: C++ 

    struct kernel {
      int i; 
      float f; 

      void operator()() __qpu__ {
         cudaq::qvector q(i); 
         ry(f, q[0]);
      }
    };

    kernel{2, 2.2}();

    int i = 2; 
    double f = 2.2;

    auto kernelLambda = [=]() __qpu__ {
      // Use captured variables 
      cudaq::qvector q(i); 
      ry(f, q[0]);
      i = 5; 
      return i; 
    };

    auto k = kernelLambda(); 

    // Pass by value semantics 
    assert(k != i);

.. tab:: Python 

  .. code-block:: python 

    i = 2 
    f = np.pi / 2. 

    @cudaq.kernel
    def kernel() -> int:
       # Use captured variables 
       q = cudaq.qvector(i) 
       ry(f, q[0])

       i = 5 
       return i 
    
    k = kernel() 

    # Pass by value semantics 
    assert k != i 

**[9]** All quantum kernel invocations are synchronous calls by default. 

**[10]** CUDA-Q kernels can serve as input to other quantum kernels and invoked by kernel 
function body code. To support CUDA-Q kernel parameterization on callable quantum kernel 
code, programmers can leverage standard C++ template definitions or dynamic typing in language bindings such as Python:

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

CUDA-Q kernel inputs can also be `constrained <https://en.cppreference.com/w/cpp/language/constraints>`_. 

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
