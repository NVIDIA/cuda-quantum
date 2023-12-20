Sub-circuit Synthesis
*********************
Execution of pure device quantum kernels can be modified via explicit intrinsic
library calls that provide multi-qubit controlled application and adjoint,
or circuit reversal, semantics. CUDA Quantum defines the following library functions
for enabling multi-controlled and adjoint operations on a general pure
device kernel

.. code-block:: cpp

    template <typename QuantumKernel, typename... Args>
    void control(QuantumKernel &&kernel,
                 cudaq::qubit& ctrl_qubit, Args &... args) { ... }
 
    template <typename QuantumKernel, typename QuantumRegister, typename... Args>
    requires(std::ranges::range<QuantumRegister>)
    void control(QuantumKernel &&kernel,
                 QuantumRegister& ctrl_qubits, Args &... args) { ... }
 
    template <typename QuantumKernel, typename... Args>
    void adjoint(QuantumKernel &&kernel, Args &... args) { ... }

These functions can be leveraged in quantum kernel code in the following way

.. code-block:: cpp

    struct x_gate {
      void operator()(cudaq::qubit& q) __qpu__ { x(q); }
    };
    struct kernel {
      void operator() () __qpu__ {
        cudaq::qreg q(3);
        ...
        // Create Toffoli gate
        auto ctrl_bits = q.front(2);
        cudaq::control(x_gate{}, ctrl_bits, q[2]);
        ...
      }
    };
    struct rx_and_h_gate {
      void operator()(double x, cudaq::qubit& q) __qpu__ { rx(x,q); h(q); }
    };
    struct kernel {
      void operator(int N) () __qpu__ {
        cudaq::qreg q(N);
        ...
        // apply h(q[2]); rx(-pi, q[2]);
        cudaq::adjoint(rx_and_h_gate{}, M_PI, q[2]);
        ...
      }
    };

The :code:`cudaq::control(...)` function takes as input an instantiated pure
device quantum kernel, an std::range of control qubits (:code:`cudaq::qreg`
or :code:`cudaq::qspan`), and the remaining arguments for the kernel itself.
Compiler implementations are free to synthesize multi-controlled operations
using any pertinent synthesis strategy available. Qubits may be aggregated into
a range of control qubits with or without the use of the :code:`operator!`
:doc:`negated polarity operator <operations>`.

.. code-block:: cpp

    cudaq::control(kernel{}, {qubit0, !qubit1}, kernel_arg);

The :code:`cudaq::adjoint(...)` function takes as input an
instantiated pure device quantum kernel (or specified template type)
and the remaining arguments for the kernel.
