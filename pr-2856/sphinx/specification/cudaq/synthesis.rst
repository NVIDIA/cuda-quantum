Sub-circuit Synthesis
*********************
**[1]** Execution of pure device quantum kernels can be modified via explicit intrinsic
library calls that provide multi-qubit controlled application and adjoint,
or circuit reversal, semantics. 

**[2]** CUDA-Q defines the following library functions
for enabling multi-controlled and adjoint operations on a general pure
device kernel:

.. code-block:: cpp

    template <typename QuantumKernel, typename... Args>
    void control(QuantumKernel &&kernel,
                 cudaq::qubit& ctrl_qubit, Args &... args);
 
    template <typename QuantumKernel, typename QuantumRegister, typename... Args>
      requires(std::ranges::range<QuantumRegister>)
    void control(QuantumKernel &&kernel,
                 QuantumRegister& ctrl_qubits, Args &... args);
 
    template <typename QuantumKernel, typename... Args>
    void adjoint(QuantumKernel &&kernel, Args &... args);

These functions can be leveraged in quantum kernel code in the following way

.. tab:: C++ 

  .. code-block:: cpp

      __qpu__ void x_gate(cudaq::qubit& q) { x(q); }
      
      struct kernel {
        void operator() () __qpu__ {
          cudaq::qarray<3> q;
          ...
          // Create Toffoli gate
          auto ctrl_bits = q.front(2);
          cudaq::control(x_gate, ctrl_bits, q[2]);
          ...
        }
      };
      
      void rx_and_h_gate(double x, cudaq::qubit& q) __qpu__ { rx(x,q); h(q); }
      
      __qpu__ kernel(int N) {
        cudaq::qvector q(N);
        ...
        // apply h(q[2]); rx(-pi, q[2]);
        cudaq::adjoint(rx_and_h_gate{}, M_PI, q[2]);
        ...
      }

.. tab:: Python 

  .. code-block:: python 

    @cudaq.kernel()
    def x_gate(q : cudaq.qubit):
        x(q)
    
    @cudaq.kernel()
    def kernelTestControl():
        q = cudaq.qvector(3)
        ...
        ctrl_bits = q.front(2)
        cudaq.control(x_gate, ctrl_bits, q[2])
        ...
    
    @cudaq.kernel()
    def rx_and_h_gate(x : float, q : cudaq.qubit):
        rx(x, q)
        h(q)
    
    @cudaq.kernel()
    def kernelTestAdjoint(N : int):
        q = cudaq.qvector(N)
        ...
        # apply h(q[2]); rx(-pi, q[2])
        cudaq.adjoint(rx_and_h_gate, np.pi, q[2])
        ...


**[3]** The :code:`cudaq::control(...)` function takes as input an instantiated pure
device quantum kernel, a std::range of control qubits (:code:`cudaq::qvector`
or :code:`cudaq::qview`), and the remaining arguments for the kernel itself.

**[4]** Compiler implementations are free to synthesize multi-controlled operations
using any pertinent synthesis strategy available. Qubits may be aggregated into
a range of control qubits with or without the use of the :code:`operator!`
:doc:`negated polarity operator <operations>`.

.. tab:: C++ 
  
  .. code-block:: cpp

      cudaq::control(kernel{}, {qubit0, !qubit1}, kernel_arg);

.. tab:: Python 

  .. code-block:: python 

    cudaq.control(kernel, [qubit0, ~qubit1], kernel_arg)
  
**[5]** The :code:`cudaq::adjoint(...)` function takes as input an
pure device quantum kernel and the remaining arguments for the kernel.
