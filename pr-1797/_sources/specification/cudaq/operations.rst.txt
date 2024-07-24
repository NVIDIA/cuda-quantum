Quantum Intrinsic Operations
****************************
**[1]** To support low-level quantum programming and build foundational
quantum kernels for large-scale applications, CUDA-Q specifies the *quantum
intrinsic operation* abstraction for device-specific single-qudit unitary operations. 

**[2]** A quantum intrinsic operation is modeled via a standard free function in the native classical language. It 
has a unique instruction name and general specified function operands. 

**[3]** These operands can model classical rotation parameters or units of quantum information (e.g. the :code:`cudaq::qudit`).
The syntax for these operations is 

.. code-block:: cpp 

    void INST_NAME( PARAM...?, cudaq::qudit<N>&...);

where :code:`INST_NAME` is the name of the instruction, :code:`qudit<N>&...` indicates one many
:code:`cudaq::qudit` instances, and :code:`PARAM...?` indicates optional parameters of 
floating point type (e.g. :code:`double`, :code:`float`). Quantum intrinsic operations return :code:`void`.

**[4]** All intrinsic operations should start with a base declaration targeting a single :code:`cudaq::qudit`, and overloads
should be provided that take more than one :code:`cudaq::qudit` instances to model the application
of that instruction on all provided :code:`cudaq::qudits`, e.g. :code:`void x(cudaq::qubit&)` and
:code:`x(cudaq::qubit&, cudaq::qubit&, cudaq::qubit&)`, modeling the NOT operation on a single 
:code:`cudaq::qubit` or on multiple :code:`cudaq::qubit`. 

**[5]** The specific set of quantum intrinsic operations available to the programmer can 
be platform specific and provided via platform specific header files. 

**[6]** Implementations should provide overloads to support broadcasting of single-qubit
intrinsic operations across containers of :code:`cudaq::qudit`. For example,
:code:`x(cudaq::qvector<>&)` should apply a NOT operation on all
:code:`cudaq::qubit` in the provided :code:`cudaq::qvector`. 

**[7]** Programmers can further modify quantum intrinsic operations via 
an extra specified template parameter, and CUDA-Q leverages this syntax 
for synthesizing control and adjoint variants of the operation. Specifically CUDA-Q 
provides the `cudaq::ctrl`, and `cudaq::adj` type modifiers for synthesizing control and 
adjoint forms of the operation. For language bindings that do not support template 
parameterization, implementations may rely on static method calls (e.g. :code:`x.ctrl(q, r)`)

Here is an example of how one might modify an intrinsic operation for multi-control and adjoint operations.

.. tab:: C++ 
  
  .. code-block:: cpp 

    cudaq::qubit q, r, s;
    // Apply T operation
    t(q);
    // Apply Tdg operation
    t<cudaq::adj>(q);
    // Apply control Hadamard operation
    h<cudaq::ctrl>(q,r,s);
    // Error, ctrl requires > 1 qubit operands
    // h<cudaq::ctrl>(r);

.. tab:: Python 

  .. code-block:: python 

    q, r, s = cudaq.qubit(), cudaq.qubit(), cudaq.qubit()
    # Apply T operation
    t(q)
    # Apply Tdg operation
    t.adj(q)
    # Apply control Hadamard operation
    h.ctrl(q,r,s)
    # Error, ctrl requires > 1 qubit operands
    # h.ctrl(r);

Operations on :code:`cudaq::qubit`
----------------------------------
The default set of quantum intrinsic operations for the cudaq::qubit type is as follows: 

.. code-block:: cpp 

    namespace cudaq {
      struct base;
      struct ctrl;
      struct adj;
  
      // Single qubit operations, ctrl / adj variants, and broadcasting
      template<typename mod = base, typename... QubitArgs>
      void NAME(QubitArgs&... args) noexcept { ... }
  
      template<typename mod = base>
      void NAME(const qvector& qr) noexcept { ... }
  
      template<typename mod = ctrl>
      void NAME(qvector& ctrls, qubit& target) noexcept { ... }
 
      // Single qubit rotation operations and ctrl / adj variants
      template <typename mod = base, typename ScalarAngle, typename... QubitArgs> 
      void ROTATION_NAME(ScalarAngle angle, QubitArgs &...args) noexcept { ... }
 
      bool MEASURE_OP(qubit &q) noexcept;
      std::vector<bool> MEASURE_OP(qvector &q) noexcept;
      double measure(cudaq::spin_op & term) noexcept { ... }
  }

**[1]** For the default implementation of the :code:`cudaq::qubit` intrinsic operations, we let 
NAME be any operation name in the set :code:`{h, x, y, z, t, s}` and :code:`ROTATION_NAME` be any 
operation in :code:`{rx, ry, rz, r1 (phase)}`. Implementations may provide appropriate 
function implementations using the above foundational functions to enable other 
common operations (e.g. :code:`cnot -> x<cudaq::ctrl>`).

**[2]** Control qubits can be specified with positive or negative polarity. 
By this we mean that a control qubit can specify that a target operation is 
applied if the control qubit state is a :code:`|0>` (positive polarity) or :code:`|1>` (negative polarity). 
By default all control qubits are assumed to convey positive polarity. The 
syntax for negating the polarity is the not operator preceding the control 
qubit 

.. tab:: C++ 

  .. code-block:: cpp 

    x<cudaq::ctrl>(!q, r);

.. tab:: Python 

  .. code-block:: python 

    x.ctrl(~q, r)
  
The set of gates that the official CUDA-Q implementation supports can be found in the :doc:`API documentation </api/api>`.
