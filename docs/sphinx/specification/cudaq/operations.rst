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
 
      // Deferred-discrimination measurement (see `measure_handle` proposal).
      class measure_handle {
      public:
        measure_handle() = default;
        operator bool() const;            // non-explicit; discriminates
      };
      measure_handle MEASURE_OP(qubit &q) noexcept;
      template <typename QubitRange>
      std::vector<measure_handle> MEASURE_OP(QubitRange &q) noexcept;
      std::vector<bool> to_bools(const std::vector<measure_handle> &h) noexcept;
      std::int64_t to_integer(const std::vector<bool> &b) noexcept;
      double measure(const cudaq::spin_op & term) noexcept { ... }
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

**[3]** :code:`MEASURE_OP` denotes the measurement family
(:code:`mz`, :code:`mx`, :code:`my`). Each call records a measurement event
and returns an opaque :code:`cudaq::measure_handle` (a
:code:`std::vector<cudaq::measure_handle>` for range overloads); the
classical bit is produced on demand by the non-explicit
:code:`measure_handle::operator bool()` whenever the handle is used in a
bool-requiring context. The single-API surface defined by the
:code:`measure_handle` proposal collapses what used to be two parallel
families (the bit-returning :code:`mz` and the handle-returning
:code:`mz_handle`) into one: every call site that previously returned a
:code:`bool` continues to compile because the implicit conversion runs at
the use site. The classical bit is therefore *deferred*: the compiler is
free to reorder, batch, or eliminate the discriminate as long as program
behavior is preserved.

**[4]** Bulk discrimination of a vector of handles is explicit, via
:code:`cudaq::to_bools`. The scalar :code:`operator bool()` does not
propagate through :code:`std::vector` (unrelated C++ types), so any code
that needs a :code:`std::vector<bool>` from a
:code:`std::vector<cudaq::measure_handle>` must call :code:`to_bools`
explicitly. :code:`cudaq::to_integer` packs a
:code:`std::vector<bool>` into the low bits of an :code:`std::int64_t`;
to bit-pack a handle vector, compose them: :code:`to_integer(to_bools(h))`.

**[5]** Handles must not cross the host-device boundary. The frontend
rejects :code:`measure_handle` in any entry-point parameter or return
position with the diagnostic :code:`measure_handle cannot cross the
host-device boundary; entry-point kernels must discriminate first`.
Pure-device :code:`__qpu__` callees may take :code:`measure_handle` by
:code:`const&`; the bridge passes them through as
:code:`!cc.ptr<!cc.measure_handle>`. Default-constructed (unbound) handles
that reach a bool-coercion context produce the diagnostic
:code:`discriminating an unbound measure_handle`. See the
:code:`measure_handle` proposal for the full normative requirements.
