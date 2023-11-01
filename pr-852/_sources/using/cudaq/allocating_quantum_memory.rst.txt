Allocating and Using Quantum Memory in CUDA Quantum
---------------------------------------------------
CUDA Quantum provides a quantum memory model that enables one to think about 
general qudits of information, dynamic and static registers of those qudits, 
and whether those registers are owning or non-owning. The latter point 
very much follows the same pattern one sees in modern C++, where we have 
generic owning container types like the :code:`std::vector<T>` and 
:code:`std::array<T>`, as well as non-owning container types like
the :code:`std::span<T>`. 

To this end, CUDA Quantum defines a non-copyable unit of quantum information, 
the :code:`cudaq::qudit<Levels>` template type. Because it is non-copyable, 
instances of this type cannot be passed by value and must always be
passed by reference, in an effort to avoid copying, or cloning, the underlying
quantum information. 

.. note:: 

  Thus far (as of this beta release), the majority of CUDA Quantum
  development work has focused on :code:`cudaq::qudit<2>` (which we
  :code:`typedef` as :code:`cudaq::qubit`) but the demonstrations and
  discussions that follow are meant to be general on qudits.

The CUDA Quantum quantum memory container types are the 
:code:`cudaq::qreg<NQudits = dyn, Levels>` and the 
:code:`cudaq::qspan<NQudits = dyn, Levels>`, representing owning and non-owning
semantics, respectively. Notice that the first template parameter represents
the number of qudits contained and is defaulted to a special symbol in CUDA Quantum
(:code:`dyn`) indicating that this is a runtime-known, dynamic register of
qudits, very much akin to something like :code:`std::vector<T> v(N)` in 
classical C++. One can alternatively specify this template parameter to pick 
up :code:`std::array<T, N> a` like semantics whereby the size of the register is 
known at compile time. The trade-off here is important - there may be quantum
circuit optimizations that can be enabled at compile time for CUDA Quantum kernels 
solely employing compile-time-known :code:`cudaq::qreg` allocations. 

These quantum memory types are specifically designed to throw compile-time errors 
when they are incorrectly used. An example of this for quantum memory and 
its underlying ownership model can be seen in this snippet 

.. code-block:: cpp 

    __qpu__ void fooBad(cudaq::qubit q) { ... };
    __qpu__ void fooGood(cudaq::qubit& q) { ... };
    __qpu__ void barBad(cudaq::qreg<> q) { ... };
    __qpu__ void barGood(cudaq::qreg<>& q) { ... };
    __qpu__ void barGoodWithSpan(cudaq::qspan q) { ... };

    struct myEntryPointKernel {
      void operator()(int runtimeKnownInteger) __qpu__ { 
        // Allocate array-like compile-time-known
        // register of 2 qubits. Owns the qubits. 
        cudaq::qreg<2> a;
        // fooBad (a[0]); // Compile Error, cannot pass qubits by value (no copy)
        // auto alias = a[0]; // Compile Error, cannot copy (auto defaults to by-value)
        auto& alias = a[0]; // Must alias by reference
        fooGood (a[0]); // Can pass by reference, no copy

        // barBad(a); // Compile Error, cannot pass qreg by value
        barGood(a); // Can pass by reference, no copy

        // Allocate vector-like register of qubits
        // Owns the qubits
        cudaq::qreg b(runtimeKnownInteger);
        
        // Get the front 2 qubits, which returns 
        // a cudaq::span<>, it does not own the qubits. 
        auto sub_view = b.front(2);

        // cudaq::span is non-owning, it can be passed by value
        barGoodWithSpan(sub_view);

        // cudaq::qreg can also be passed to a kernel that accepts a span
        barGoodWithSpan(a);

        // Front with no size provided will 
        // return a reference to the first qubit. 
        // You must define this as a reference variable. 
        auto& frontQubit = b.front();

        // a, b go out of scope, qubits deallocated
        // returned to infinite global register of qubits
        // NOTE automated uncomputation not currently implemented.
      }
    };

:code:`cudaq::qubit` and :code:`cudaq::qreg` types are owning types and
therefore cannot be passed by value to invoked pure quantum device kernels. 
In order to share allocated registers with other quantum function calls, 
one must pass by reference or define the invoked kernel to take the qubits 
as a :code:`cudaq::span`. Note that all slicing operations intended to 
extract a sub-register from a given :code:`cudaq::qreg` will return a 
non-owning :code:`qspan`. 
