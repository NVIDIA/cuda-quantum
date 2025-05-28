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

These functions can be leveraged in quantum kernel code in the following way:

.. tab:: C++ 

  .. literalinclude:: /../snippets/cpp/synthesis/synthesis_examples.cpp
     :language: cpp
     :start-after: [Begin CPP ControlAndAdjointCombined]
     :end-before: [End CPP ControlAndAdjointCombined]

.. tab:: Python 

  .. literalinclude:: /../snippets/python/synthesis/synthesis_examples.py
     :language: python
     :start-after: [Begin PY ControlAndAdjointCombined]
     :end-before: [End PY ControlAndAdjointCombined]

**[3]** The :code:`cudaq::control(...)` function takes as input an instantiated pure
device quantum kernel, a std::range of control qubits (:code:`cudaq::qvector`
or :code:`cudaq::qview`), and the remaining arguments for the kernel itself.

**[4]** Compiler implementations are free to synthesize multi-controlled operations
using any pertinent synthesis strategy available. Qubits may be aggregated into
a range of control qubits with or without the use of the :code:`operator!`
:doc:`negated polarity operator <operations>`.

.. tab:: C++ 
  
  .. literalinclude:: /../snippets/cpp/synthesis/synthesis_examples.cpp
     :language: cpp
     :start-after: [Begin CPP NegatedControlRSTLine]
     :end-before: [End CPP NegatedControlRSTLine]

.. tab:: Python 

  .. literalinclude:: /../snippets/python/synthesis/synthesis_examples.py
     :language: python
     :start-after: [Begin PY NegatedControlRSTLine]
     :end-before: [End PY NegatedControlRSTLine]
  
**[5]** The :code:`cudaq::adjoint(...)` function takes as input an
pure device quantum kernel and the remaining arguments for the kernel.