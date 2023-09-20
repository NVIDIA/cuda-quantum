Quantum Types
*************
This language extension provides certain fundamental types that are pertinent
to quantum-classical computing. These types are provided via defined library
types in the :code:`cudaq` namespace. 

:code:`cudaq::qudit<Levels>`
----------------------------
The :code:`cudaq::qudit` models a :math:`D`-level unit of quantum information. The state of
this system (for :math:`N` qudits) can be described by a :math:`D`\ :sup:`N`\-dimensional vector in
Hilbert space with the absolute square of all elements summing to 1. The
:code:`cudaq::qudit` encapsulates a unique :code:`std::size_t` modeling the index of the
qudit in the underlying quantum memory space (assuming an infinite register
of available qudits). To adhere to the no-cloning theorem of quantum mechanics,
the :code:`cudaq::qudit` is non-copyable and non-movable. Therefore, all :code:`cudaq::qudit` 
instances must be passed by reference, and the no-cloning theorem is satisfied
at compile-time. :code:`cudaq::qudit` instances can only be allocated within CUDA Quantum quantum
kernel code and can never be allocated from classical host code.

The :code:`cudaq::qudit` takes on the following structure

.. code-block:: cpp

    template <std::size_t Levels>
    class qudit {
      protected: 
        const std::size_t idx = 0;
      public:
        qudit();
        qudit(const qudit&) = delete;
        qudit(qudit &&) = delete;
        std::size_t id() const;
        static constexpr std::size_t n_levels() { return Levels; }
    };

:code:`cudaq::qubit`
--------------------
The specification provides a primitive :code:`cudaq::qubit` type which models a
single quantum bit (2-level) in the discrete quantum memory space.
:code:`cudaq::qubit` is an alias for :code:`cudaq::qudit<2>` 

.. code-block:: cpp
    
    namespace cudaq {
      using qubit = qudit<2>;
    }

The :code:`qubit` type can only be allocated within quantum kernel code and cannot
be instantiated in host classical code. All instantiated :code:`cudaq::qubit` instances start
in the :code:`|0>` computational basis state and serve as the primary input argument
for quantum intrinsic operations modeling typical logical quantum gate operations. 

.. code-block:: cpp

    {
      // Allocate a qubit in the |0> state
      cudaq::qubit q;
      // Put the qubit in a superposition of |0> and |1>
      h(q); // cudaq::h == hadamard, ADL leveraged
      printf("ID = %lu\n", q.id()); // prints 0
      cudaq::qubit r;
      printf("ID = %lu\n", r.id()); // prints 1
      // qubit out of scope, implicit deallocation
    }
    cudaq::qubit q;
    printf("ID = %lu\n", q.id()); // prints 0 (previous deallocated)

Quantum Containers
------------------
CUDA Quantum provides abstractions for dealing with groups of :code:`cudaq::qudit` instances in the
form of familiar, std-like C++ containers. The underlying
connectivity of the :code:`cudaq::qudit` instances stored in these containers is opaque to
the programmer and any logical-to-physical program connectivity mapping
should be done by compiler implementations. 

:code:`cudaq::qspan<N, Levels>`
+++++++++++++++++++++++++++++++
:code:`cudaq::qspan` is a non-owning reference to a part of the discrete quantum
memory space. It's a :code:`std::span`-like C++ range of :code:`cudaq::qudit` 
(see C++ `span <https://en.cppreference.com/w/cpp/container/span>`_). It does not
own its elements. It takes a single template parameter indicating the levels for 
the underlying qudits that it stores. This parameter defaults to 2 for qubits. 
It takes on the following structure:

.. code-block:: cpp
    
    namespace cudaq {
      template <std::size_t Levels = 2>
      class qspan {
        private:
          std::span<qudit<Levels>> qubits;
        public:
          // Construct a span that refers to the qudits in `other`.
          qspan(std::ranges::range<qudit<Levels>> auto& other);
          qspan(qspan const& other);
 
          // Iterator interface.
          auto begin();
          auto end();
 
          // Returns the qudit at `idx`.
          qudit<Levels>& operator[](const std::size_t idx);
 
          // Returns the `[0, count)` qudits.
          qspan<Levels> front(std::size_t count);
          // Returns the first qudit.
          qudit<Levels>& front();
          // Returns the `[count, size())` qudits.
          qspan<Levels> back(std::size_t count);
          // Returns the last qudit.
          qudit<Levels>& back();
 
          // Returns the `[start, start+count)` qudits.
          qspan<Levels>
          slice(std::size_t start, std::size_t count);

          // Returns the number of contained qudits.
          std::size_t size() const;
      };
    }

:code:`cudaq::qreg<N, Levels>`
++++++++++++++++++++++++++++++
:code:`cudaq::qreg<N, Levels>` models a register of the discrete quantum memory space - a
C++ container of :code:`cudaq::qudit`.  As a container, it owns its elements and
their storage. :code:`qreg<dyn, Levels>` is a dynamically allocated container
(:code:`std::vector`-like, see C++ `vector <https://en.cppreference.com/w/cpp/container/vector>`_).
:code:`cudaq::qreg<N, Levels>` (where N is an integral
constant) is a statically allocated container (:code:`std::array`-like, 
see `array <https://en.cppreference.com/w/cpp/container/array>`_). 
Its template parameters default to dynamic allocation and :code:`cudaq::qudit<2>`.

.. code-block:: cpp

    namespace cudaq {
      template <std::size_t N = dyn, std::size_t Levels = 2>
      class qreg {
        private:
          std::conditional_t<
            N == dyn,
            std::vector<qudit<Levels>>,
            std::array<qudit<Levels>, N>
          > qudits;
        public:
          // Construct a qreg with `size` qudits in the |0> state.
          qreg(std::size_t size) requires (N == dyn);
          qreg(qreg const&) = delete;
 
          // Iterator interface.
          auto begin();
          auto end();
 
          // Returns the qudit at `idx`.
          qudit<Levels>& operator[](const std::size_t idx);
 
          // Returns the `[0, count)` qudits.
          qspan<dyn, Levels> front(std::size_t count);
          // Returns the first qudit.
          qudit<Levels>& front();
          // Returns the `[count, size())` qudits.
          qspan<dyn, Levels> back(std::size_t count);
          // Returns the last qudit.
          qudit<Levels>& back();
 
          // Returns the `[start, start+count)` qudits.
          qspan<dyn, Levels>
          slice(std::size_t start, std::size_t count);

          // Returns the number of contained qudits.
          std::size_t size() const;
 
          // Destroys all contained qudits. Postcondition: `size() == 0`.
          void clear();
      };
    } 

:code:`qreg` instances can only be instantiated from within quantum kernels,
they cannot be instantiated in host code. All qubits in the :code:`qreg` 
start in the :code:`|0>` computational basis state. 

.. code-block:: cpp

    // Allocate 20 qubits, std::vector-like semantics
    cudaq::qreg q(20);
    auto first = q.front();
    auto first_5 = q.front(5);
    auto last = q.back();
    for (int i = 0; i < q.size(); i++) {
      ... do something with q[i] ...
    }
    for (auto & qb : q) {
      ... do something with qb ...
    }
 
    // std::array-like semantics
    cudaq::qreg<5> fiveCompileTimeQubits;