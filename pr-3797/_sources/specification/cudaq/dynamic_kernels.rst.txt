Just-in-Time Kernel Creation
****************************

**[1]** CUDA-Q provides a set of programming abstractions for dynamically constructing
quantum kernel code at runtime. 

**[2]** The callable :code:`cudaq::kernel_builder` abstraction
facilitates the dynamic definition of quantum kernels that can optionally
be parameterized by user-defined input arguments. 

The :code:`kernel_builder` takes the following structure

.. code-block:: cpp 

    namespace cudaq {
      // Type wrapping a value in the kernel IR
      struct Value;
      template <typename... Args>
      class kernel_builder {
        private:
          std::vector<Value> arguments;

        public:
          std::vector<Value>& getArguments();
          std::string name() const;
          std::size_t getNumParams() const;

          Value qalloc(std::size_t nQubits = 1);
          Value qalloc(Value size);

          void h(Value& qubit);
          //... all other quantum operations ... 

          // if (mz(q)) thenFunctor()
          void c_if(Value& result, std::function<void()>&& thenFunctor);
          
          // Invoke a predefined kernel
          template<typename OtherKernelBuilder, typename... Values>
          void call(OtherKernelBuilder&& kernelToCall, Values&... args);
          
          // General multi-control on a predefined kernel
          // models cudaq::control(...)
          template<typename OtherKernelBuilder, typename... Values>
          void control(OtherKernelBuilder&& kernelToControl, Value& ctrl, Values&... values);
          
          // General adjoint on a predefined kernel
          // models cudaq::adjoint(...)
          template<typename OtherKernelBuilder, typename... Values>
          void adjoint(OtherKernelBuilder&& kernelToAdjoint, Values&... values);
       
          // The constructed kernel is callable
          void operator()(Args... args);
          void operator()(void** argsArray);

          // Enable structured bindings
          template <std::size_t N>
          decltype(auto) get() {
            if constexpr (N == 0)
              return *this;
            else
              return arguments[N - 1];
          }
      };
    }

    /// Enable structured bindings on the kernel_builder type.
    /// auto [kernel, theta, phi] = std::make_kernel<double,double>();
    namespace std {
      template <typename... Args> 
      struct tuple_size<cudaq::kernel_builder<Args...>>
        : std::integral_constant<std::size_t, sizeof...(Args) + 1> {};

      template <std::size_t N, typename... Args>
      struct tuple_element<N, cudaq::kernel_builder<Args...>> {
        using type = std::conditional_t<N == 0, cudaq::kernel_builder<Args...>,
                                  cudaq::QuakeValue>;
      };
    } // namespace std

    namespace cudaq {
      kernel_builder<> make_kernel();
      template<typename... Args>
      kernel_builder<Args...> make_kernel();
    }

**[3]** The structure above allows one to leverage the provided factory functions (:code:`make_kernel`)
to construct an empty CUDA-Q kernel with defined argument signature. 

**[4]** For each type in the signature, the programmer is returned a new :code:`Value` instance which 
can be leveraged in the construction of the kernel. The intended mechanism for 
kernel creation and argument value extraction is via standard C++ 
`structured bindings <https://en.cppreference.com/w/cpp/language/structured_binding>`_.

**[5]** Once the kernel is created, the programmer is free to build up the kernel expression 
using the exposed API. There are methods for qubit allocation, quantum operation 
invocation, control and adjoint synthesis, and conditional branching based on 
`boolean` values. 

Here is a simple example how one might build a CUDA-Q kernel dynamically. 

.. tab:: C++ 

  .. code-block:: cpp 

    auto kernel = cudaq::make_kernel();
    auto qubits = kernel.qalloc(2);
    kernel.h(qubits[0]);
    kernel.x<cudaq::ctrl>(qubits[0], qubits[1]);
    kernel.mz(qubits);
    
    // See algorithmic primitives section for more on sample
    auto counts = cudaq::sample(kernel);

.. tab:: Python 

  .. code-block:: python 

    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(2)
    kernel.h(qubits[0])
    kernel.cx(qubits[0], qubits[1])
    kernel.mz(qubits)

Here is an example demonstrating how one may build a dynamic set of 
CUDA-Q kernels for executing the standard Hadamard test. 

.. tab:: C++ 

  .. code-block:: cpp

      auto [xPrep, qubitIn] = cudaq::make_kernel<cudaq::qubit>();
      xPrep.x(qubitIn);

      // Compute <1|X|1> = 0
      auto hadamardTest = cudaq::make_kernel();
      auto q = hadamardTest.qalloc();
      auto ancilla = hadamardTest.qalloc();
      hadamardTest.call(xPrep, q);
      hadamardTest.h(ancilla);
      hadamardTest.control(xPrep, ancilla, q);
      hadamardTest.h(ancilla);
      hadamardTest.mz(ancilla);

      // See algorithmic primitives section for more on sample
      auto counts = cudaq::sample(hadamardTest);

.. tab:: Python 

  .. code-block:: python 

    xPrep, qubitIn = cudaq.make_kernel(cudaq.qubit)

    # Compute <1|X|1> = 0
    hadamardTest = cudaq.make_kernel()
    q = hadamardTest.qalloc()
    hadamardTest.call(xPrep, q)
    hadamardTest.h(ancilla)
    hadamardTest.control(xPrep, ancilla, q)
    hadamardTest.h(ancilla)
    hadamardTest.mz(ancilla)

    # See algorithmic primitives section for more on sample
    counts = cudaq.sample(hadamardTest)
