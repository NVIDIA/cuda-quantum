Quantum Intrinsic Operations
****************************
In an effort to support low-level quantum programming and build foundational
quantum kernels for large-scale applications, CUDA Quantum defines the quantum
intrinsic operation as an abstraction
for device-specific single-qudit unitary operations. A quantum intrinsic
operation is modeled via a standard C++ function with a unique instruction name and
general function operands. These operands can model classical rotation
parameters or units of quantum information (e.g. the :code:`cudaq::qudit`).
The syntax for these operations is 

.. code-block:: cpp 

    void INST_NAME( PARAM...?, qudit<N>&...);

where :code:`INST_NAME` is the name of the instruction, :code:`qudit<N>&...` indicates one many
:code:`cudaq::qudit` instances, and :code:`PARAM...?` indicates optional parameters of 
floating point type (e.g. :code:`double`, :code:`float`). All intrinsic operations should 
start with a base declaration targeting a single :code:`cudaq::qudit`, and overloads
should be provided that take more than one :code:`cudaq::qudit` instances to model the application
of that instruction on all provided :code:`cudaq::qudits`, e.g. :code:`void x(cudaq::qubit&)` and
:code:`x(cudaq::qubit&, cudaq::qubit&, cudaq::qubit&)`, modeling the NOT operation on a single 
:code:`cudaq::qubit` or on multiple :code:`cudaq::qubit`. 

Implementations should provide overloads to support broadcasting of single-qubit
intrinsic operations across a register of :code:`cudaq::qudit`.
For example, :code:`x(cudaq::qreg<>&)` should apply a NOT operation on all :code:`cudaq::qubit` in the provided :code:`cudaq::qreg`. 
A set of quantum intrinsic operations for the :code:`cudaq::qubit` then for example looks as follows, where :code:`NAME`, :code:`ROTATION_NAME`, and :code:`MEASURE_OP` stand for the names of single-qubit operations, single-qubit rotations, and measurement operations respectively: 

.. code-block:: cpp 

    namespace cudaq {
      struct base;
      struct ctrl;
      struct adj;
  
      // Single qubit operations, ctrl / adj variants, and broadcasting
      template<typename mod = base, typename... QubitArgs>
      void NAME(QubitArgs&... args) noexcept { ... }
  
      template<typename mod = base>
      void NAME(const qreg& qr) noexcept { ... }
  
      template<typename mod = ctrl>
      void NAME(qreg& ctrls, qubit& target) noexcept { ... }
 
      // Single qubit rotation operations and ctrl / adj variants
      template <typename mod = base, typename ScalarAngle, typename... QubitArgs> 
      void ROTATION_NAME(ScalarAngle angle, QubitArgs &...args) noexcept { ... }
 
      bool MEASURE_OP(qubit &q) noexcept;
      std::vector<bool> MEASURE_OP(qreg &q) noexcept;
      double measure(cudaq::spin_op & term) noexcept { ... }
  }

The set of gates that the official CUDA Quantum implementation supports can be found in the :doc:`API documentation </api/api>`.