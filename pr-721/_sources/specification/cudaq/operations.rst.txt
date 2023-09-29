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

The specific set of quantum intrinsic operations available to the programmer
will be platform specific, e.g. the standard Clifford+T gate set on
:code:`cudaq::qubit` versus a continuous variable (photonic) gate set on 
:code:`cudaq::qudit<N>`. 

Implementations should provide overloads to support broadcasting of an
operation across a register of :code:`cudaq::qudit`, e.g. :code:`x(cudaq::qreg<>&)`
to apply a NOT operation on all :code:`cudaq::qubit` in the provided :code:`cudaq::qreg`. 

Programmers can further modify quantum intrinsic operations via an extra specified template
parameter, and CUDA Quantum leverages this syntax for synthesizing control and adjoint variants of the operation.
Here is an example of how one might modify an intrinsic operation for multi-control
and adjoint operations. 

.. code-block:: cpp

    cudaq::qubit q, r, s;
    // Apply T operation
    t(q);
    // Apply Tdg operation
    t<cudaq::adj>(q);
    // Apply control hadamard operation
    h<cudaq::ctrl>(q,r,s);
    // Error, ctrl requires > 1 qubit operands
    // h<cudaq::ctrl>(r);

Operations on :code:`cudaq::qubit`
----------------------------------
The default set of quantum intrinsic operations for the
:code:`cudaq::qubit` type is as follows: 

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
 
      // General swap with control variants
      // must take at least 2 qubits
      template<typename... QubitArgs>
      void swap(QubitArgs&... args) { ... }
 
      bool MEASURE_OP(qubit &q) noexcept;
      std::vector<bool> MEASURE_OP(qreg &q) noexcept;
      double measure(cudaq::spin_op & term) noexcept { ... }
  }

For the default implementation of the :code:`cudaq::qubit` intrinsic operations, we
let :code:`NAME` be any operation name in the set :code:`{x, y, z, h, t, s}`
and :code:`ROTATION_NAME` be any operation in :code:`{rx, ry, rz, r1 (phase)}`. 
Measurements (:code:`MEASURE_OP`) can be general qubit measurements in the x, y, or z 
direction (:code:`mx, my, mz`). 
Implementations may provide appropriate function implementations using the
above foundational functions to enable other common operations
(e.g. :code:`cnot` -> :code:`x<ctrl>`).

Control qubits can be specified with positive or negative polarity. By this we mean
that a control qubit can specify that a target operation is applied if the control 
qubit state is a :code:`|0>` (positive polarity) or :code:`|1>` (negative polarity). 
By default all control qubits are assumed to convey positive polarity. 
The syntax for negating the polarity is the not operator preceeding the
control qubit (e.g., :code:`x<cudaq::ctrl>(!q, r)`, 
for :code:`cudaq::qubits` :code:`q` and :code:`r`). Negating the polarity of
control qubits is supported in :code:`swap` and the gates in sets :code:`NAME`
or :code:`ROTATION_NAME`. The negate notation is only supported on control
qubits and not target qubits. So negating either of the target qubits in the
:code:`swap` operation is not allowed.
