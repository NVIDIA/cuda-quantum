Quantum Types
*************
The CUDA-Q language extension provides certain fundamental types that are pertinent
to quantum-classical computing. These types are provided via defined classical library
types in C++ in the :code:`cudaq` namespace. 

Qudit
=====

A qudit is a model to represent a :math:`d`-level (or :math:`d`-dimensional)
quantum system. Mathematically, we describe the state of a qudit using a
:math:`d`-dimensional vector of unit form, living in the vector space
:math:`\mathbb{C}^d` (a Hilbert space). Qudits with known values for
:math:`d` have specific name. A qubit has dimension 2, a qutrit has dimension 3,
and so on.

In a CUDA Quantum implementation, objects of type :code:`qudit` must adhere to
the following constraints:

**[1]** Objects of type :code:`qudit` shall be parameterized with :math:`d`,
to indicate its dimension.

**[2]** Objects of type :code:`qudit` shall not carry a quantum state.

**[3]** Objects of type :code:`qudit` shall identify a qudit in the underlying
quantum system. The identifier shall be unique and valid during the
lifetime of the :code:`qudit` object.

**[4]** Objects of type :code:`qudit` and any directly or indirectly derived
type shall only be used within quantum kernel code.

**[5]** The default constructor shall create an object of type :code:`qudit`
that references a qudit in the :math:`|0\rangle` computational basis state.

**[6]** Objects of type :code:`qudit` may be non-copyable.

**[7]** Objects of type :code:`qudit` may be non-movable.

**[8]** Objects of type :code:`qudit` may have a mechanism that indicates the
presence of implicit Pauli-X operators surrounding the qudit use as a control.
For example (pseudocode):

.. code-block:: cpp

    cudaq::qudit control;
    cudaq::qudit target;

    op(!control, target) // Implies: x(control)
                         //          op(control, target)
                         //          x(control)

C++ Implementation :code:`cudaq::qudit<Levels>`
-----------------------------------------------

The C++ implementation of CUDA Quantum defines the templated type
:code:`cudaq::qudit` to reference qudits. The template parameter defines
the dimension, :math:`d`, of the qudit, also known as the number of logical
levels allowed by the qudit (hence use of `Levels` as parameter name).

The :code:`cudaq::qudit` type encapsulates a unique identifier of type
:code:`std::size_t` modeling the index of the qudit in the underlying quantum
memory space (assuming an infinite register of available qudits). This
identifier is unique and valid only through the lifetime of the object.

To facilitate memory handling and hide internal details from the developer,
instances of :code:`cudaq::qudit` are non-copyable and non-movable. Therefore,
all objects of this type must be passed by reference. Qudits are implicitly
deallocated upon the destruction of :code:`cudaq::qudit` object that references
it.

:code:`cudaq::qudit` instances can be negated when leveraged as controls in 
quantum operators. The mechanism for negation is via overloading of
:code:`qudit<Levels>::operator!()`.

In library mode, it is the runtime responsibilities to:

**[1]** Guarantee that all :code:`cudaq::qudit` instances are allocated within
quantum kernel code (and can never from classical host code).

**[2]** Guarantee that all :code:`cudaq::qudit` instances used directly or
indirectly within quantum kernel code have the same dimension.

**[3]** Guarantee the implicit deallocation the qudit referenced by a
:code:`cudaq::qudit` instance upon its destruction.

**[4]** Guarantee that all :code:`cudaq::qudit` default constructed instances
are referencing a qudit in the :math:`|0\rangle` computational basis
state.

**[5]** Properly handle uses :code:`cudaq::qudit` instances as negated controls.

In compilation mode, these responsibilities fall under the compiler.

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
        static constexpr std::size_t n_levels();
        qudit<Levels> &operator!(); 
    };

:code:`cudaq::qubit`
--------------------
The C++ implementation of CUDA Quantum provides a primitive :code:`cudaq::qubit`
type which references a single quantum bit (:math:`2`-level) in the discrete
quantum memory space.

:code:`cudaq::qubit` is an alias for :code:`cudaq::qudit<2>` 

.. code-block:: cpp
    
    namespace cudaq {
      using qubit = qudit<2>;
    }

.. tab:: C++ 

  .. code-block:: cpp

      {
        cudaq::qubit q; // Allocate a qubit in the |0> state
        cudaq::h(q);    // Put the qubit in a superposition of |0> and |1>
      } // Qubit `q` goes out of scope, implicit deallocation.

      // Allocate a new qubit in the |0> state
      cudaq::qubit q;

.. tab:: Python 

  .. code-block:: python 

    # Allocate a qubit in the |0> state
    q = cudaq.qubit()
    # Put the qubit in a superposition of |0> and |1>
    h(q)
    # Qubits `q` goes out of scope, implicit deallocation
    
Quantum Containers
------------------
**[1]** CUDA-Q specifies abstractions for dealing with groups of :code:`cudaq::qudit` instances in the
form of familiar C++ containers. 

**[2]** The underlying connectivity of the :code:`cudaq::qudit` instances stored in these containers is opaque to
the programmer and any logical-to-physical program connectivity mapping should be done by compiler implementations. 

:code:`cudaq::qview<Levels = 2>`
++++++++++++++++++++++++++++++++
**[1]** :code:`cudaq::qview<N>` is a non-owning reference to a subset of the discrete quantum memory space, 
and as such, it is a :code:`std::span`-like C++ range of :code:`cudaq::qudit`.

**[2]** The :code:`cudaq::qview` does not own its elements and can therefore be passed by value or reference. 

**[3]** The :code:`cudaq::qview` is templated on the dimensionality of the contained quantum information unit, 
and defaults to :math:`2` for qubit systems.

**[4]** The :code:`cudaq::qview` provides an API for individual qubit extraction and sub-register slicing. 
Programmers can extract the front :math:`N` :code:`qudits`, the back :math:`N` :code:`qudits`, and the 
inner slice starting at a given index and including user-specified :code:`count` :code:`qudits`.

The :code:`cudaq::qview` should take on the following structure:

.. code-block:: cpp
    
    namespace cudaq { 
      template <std::size_t Levels = 2>
      class qview {
        private:
          std::span<qudit<Levels>> qudits;
        public:
          // Construct a span that refers to the qudits in `other`.
          template <typename R>
          requires(std::ranges::range<R>)
          qview(R&& other);
          qview(const qview& other);

          // Iterator interface.
          auto begin();
          auto end();

          // Returns the qudit at `idx`.
          qudit<Levels>& operator[](const std::size_t idx);

          // Returns the `[0, count)` qudits.
          qview<Levels> front(std::size_t count);
          // Returns the first qudit.
          qudit<Levels>& front();
          // Returns the `[count, size())` qudits.
          qview<Levels> back(std::size_t count);
          // Returns the last qudit.
          qudit<Levels>& back();


          // Returns the `[start, start+count)` qudits.
          qview<Levels>
          slice(std::size_t start, std::size_t count);

          // Returns the number of contained qudits.
          std::size_t size() const;
      };
    }

:code:`cudaq::qvector<Levels = 2>`
++++++++++++++++++++++++++++++++++
**[1]** :code:`cudaq::qvector<Levels>` is a container of elements from the discrete quantum memory space - a C++ container of :code:`cuda::qudit`.  

**[2]** The :code:`cudaq::qvector` is a dynamically constructed owning container for :code:`cuda::qudit` (:code:`std::vector`-like), 
and since it owns the quantum memory, it cannot be copied or moved. 

**[3]** The :code:`cudaq::qvector` is templated on the dimensionality of the contained 
quantum information unit, and defaults to :math:`2` for qubit systems.

**[4]** The :code:`cudaq::qvector` can only be instantiated within CUDA-Q kernels

**[5]** All qudits in the :code:`cudaq::qvector` start in the :code:`|0>` computational basis state. 

**[6]** The :code:`cudaq::qvector` provides an API for individual qubit extraction and sub-register slicing. 
Programmers can extract the front :math:`N` :code:`qudits`, the back :math:`N` :code:`qudits`, and the 
inner slice starting at a given index and including user-specified :code:`count` :code:`qudits`.

The :code:`cudaq::qview` should take on the following structure:

.. code-block:: cpp
    
    namespace cudaq { 
      template <std::size_t Levels = 2>
      class qvector {
        private:
          std::vector<qudit<Levels>> qudits;

        public:
          // Construct a qvector with `size` qudits in the |0> state.
          qvector(std::size_t size);
          qvector(const qvector&) = delete;

          // Iterator interface.
          auto begin();
          auto end();

          // Returns the qudit at `idx`.
          qudit<Levels>& operator[](const std::size_t idx);

          // Returns the `[0, count)` qudits.
          qview<Levels> front(std::size_t count);
          // Returns the first qudit.
          qudit<Levels>& front();
          // Returns the `[count, size())` qudits.
          qview<Levels> back(std::size_t count);
          // Returns the last qudit.
          qudit<Levels>& back();
 
          // Returns the `[start, start+count)` qudits.
          qview<Levels>
          slice(std::size_t start, std::size_t count);

          // Returns the `{start, start + stride, ...}` qudits.
          qview<Levels>
          slice(std::size_t start, std::size_t stride, std::size_t end);

          // Returns the number of contained qudits.
          std::size_t size() const;

          // Destroys all contained qudits. Postcondition: `size() == 0`.
          void clear();
      };
    }

.. tab:: C++ 

  .. code-block:: cpp 

    // Allocate 20 qubits, std::vector-like semantics
    cudaq::qvector q(20);
    // Get first qubit
    auto first = q.front();
    // Get first 5 qubits
    auto first_5 = q.front(5);
    // Get last qubit 
    auto last = q.back();
    // Can loop over qubits with size() method
    for (int i = 0; i < q.size(); i++) {
      ... do something with q[i] ...
    }
    // Range based for loop supported 
    for (auto & qb : q) {
      ... do something with qb ...
    }

.. tab:: Python 

  .. code-block:: python 

    # Allocate 20 qubits, vector-like semantics
    q = cudaq.qvector(20)
    # Get the first qubit 
    first = q.front()
    # Get the first 5 qubits 
    first_5 = q.front(5)
    # Get the last qubit 
    last = q.back()
    # Can loop over qubits with size or len function 
    for i in range(len(q)):
      .. do something with q[i] ..
    # Range based for loop 
    for qb in q:
      .. do something with qb .. 


:code:`cudaq::qarray<N, Levels = 2>`
++++++++++++++++++++++++++++++++++++
**[1]** :code:`cudaq::qarray<N, Levels>` (where :code:`N` is an integral constant) is a statically 
allocated container (:code:`std::array`-like). The utility of this type is in the compile-time 
knowledge of allocated containers of qudits that may directly enable ahead-of-time quantum 
optimization and synthesis. 

**[2]** The second template parameter defaults to :math:`2`-level :code:`cudaq::qudit`.

**[3]** The :code:`cudaq::qarray` owns the quantum memory it contains, and therefore cannot be copied or moved.

**[4]** The :code:`cudaq::qarray` can only be instantiated within CUDA-Q kernels

**[5]** All qudits in the :code:`cudaq::qarray` start in the :code:`|0>` computational basis state. 

**[6]** The :code:`cudaq::qarray` provides an API for individual qubit extraction and sub-register slicing. 
Programmers can extract the front :math:`N` :code:`qudits`, the back :math:`N` :code:`qudits`, and the 
inner slice starting at a given index and including user-specified :code:`count` :code:`qudits`.

The :code:`cudaq::qarray` should take on the following structure:

.. code-block:: cpp 

    namespace cudaq {
      template <std::size_t N, std::size_t Levels = 2>
      class qarray {
        private:
          std::array<qudit<Levels>, N> qudits;

        public:
          // Construct an qarray with `size` qudits in the |0> state.
          qarray();
          qarray(const qvector&) = delete;
          qarray(qarray &&) = delete;

          qarray& operator=(const qarray &) = delete;

          // Iterator interface.
          auto begin();
          auto end();

          // Returns the qudit at `idx`.
          qudit<Levels>& operator[](const std::size_t idx);

          // Returns the `[0, count)` qudits.
          qview<Levels> front(std::size_t count);
          // Returns the first qudit.
          qudit<Levels>& front();
          // Returns the `[count, size())` qudits.
          qview<Levels> back(std::size_t count);
          // Returns the last qudit.
          qudit<Levels>& back();

          // Returns the `[start, start+count)` qudits.
          qview<Levels>
          slice(std::size_t start, std::size_t count);

          // Returns the `{start, start + stride, ...}` qudits.
          qview<Levels>
          slice(std::size_t start, std::size_t stride, std::size_t end);

          // Returns the number of contained qudits.
          std::size_t size() const;

          // Destroys all contained qudits. Postcondition: `size() == 0`.
          void clear();
      };
    }

:code:`cudaq::qspan<N, Levels>` (Deprecated. Use :code:`cudaq::qview<Levels>` instead.)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
**[1]** :code:`cudaq::qspan` is a non-owning reference to a part of the discrete quantum
memory space, a :code:`std::span`-like C++ range of :code:`cudaq::qudit` 
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

:code:`cudaq::qreg<N, Levels>` (Deprecated. Use :code:`cudaq::qvector<Levels>` instead.)
++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
**[1]** :code:`cudaq::qreg<N, Levels>` models a register of the discrete quantum memory space - a
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
