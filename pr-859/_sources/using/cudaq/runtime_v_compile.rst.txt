Runtime Versus Compile-time Kernels
-----------------------------------
The structure of the :code:`cudaq::qreg` allows programmers to reason about
the definition of kernels in a couple of ways. Programmers can define 
quantum code that is generic and depends on runtime parameters, or they can
define kernel expressions that are static and can be reasoned about and
analyzed at compile time. Let's consider a kernel that generates 
a maximally-entangled GHZ state:

.. code-block:: cpp 

    // Define Kernels that generate circuits 
    // at compile time
    template <std::size_t N> 
    struct ghz_compile_time {
      auto operator()() __qpu__ {
        cudaq::qreg<N> q;
        h(q[0]);
        for (int i = 0; i < N - 1; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    };

    // Define Kernels that require runtime input
    // and therefore cannot be reasoned about at compile time
    struct ghz_runtime {
      auto operator()(int N) __qpu__ {
        cudaq::qreg q(N);
        h(q[0]);
        for (int i = 0; i < N - 1; i++) {
          x<cudaq::ctrl>(q[i], q[i + 1]);
        }
        mz(q);
      }
    }; 
    
    // GHZ on 30 qubits, known at compile time
    ghz_compile_time<30>{}();

    // GHZ on 30 qubits, only known at runtime
    ghz_runtime{}(30);

This is a trivial example, and there is not really anything we can do
with regards to compile-time optimization. But this snippet should
demonstrate how programmers can reason about the definition of quantum 
code with CUDA Quantum, and that compile-time optimizations will be more effective 
with code that relies on :code:`cudaq::qreg<N>`-like semantics.