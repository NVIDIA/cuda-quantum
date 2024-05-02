Quantum Operators
*****************

:code:`cudaq::spin_op`
----------------------
**[1]** CUDA-Q provides a native :code:`spin_op` data type in the :code:`cudaq` namespace for the
expression of quantum mechanical spin operators. 

**[2]** The :code:`spin_op` provides an abstraction for a general tensor product of Pauli
spin operators, and sums thereof:

.. math:: 

    H = \sum_{i=0}^M P_i, P_i = \prod_{j=0}^N \sigma_j^a

for :math:`a = {x,y,z}`, :math:`j` the qubit index, and :math:`N` the number of qubits.

**[3]** The :code:`spin_op` exposes common C++ operator overloads for algebraic expressions. 

**[4]** CUDA-Q defines convenience functions in :code:`cudaq::spin` namespace that produce
the primitive X, Y, and Z Pauli operators on specified qubit indices
which can subsequently be used in algebraic expressions to build up
more complicated Pauli tensor products and their sums.

.. tab:: C++ 

    .. code-block:: cpp

        using namespace cudaq::spin;
        auto h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + \
                 .21829 * z(0) - 6.125 * z(1);

.. tab:: Python

    .. code-block:: python 

        from cudaq import spin 
        h = 5.907 - 2.1433 * spin.x(0) * spin.x(1) - 2.1433 * spin.y(0) * spin.y(1) + \
                 .21829 * spin.z(0) - 6.125 * spin.z(1)


**[5]** The :code:`spin_op` also provides a mechanism for the expression of circuit
synthesis tasks within quantum kernel code. Specifically, operations
that encode :math:`N`\ :sup:`th`\ order trotterization of exponentiated :code:`spin_op`
rotations, e.g. :math:`U = \exp(-i H t)`, where :math:`H` is the provided :code:`spin_op`.

**[6]** The :code:`spin_op` can be created within classical host code and quantum kernel
code, and can also be passed by value to quantum kernel code from host code. 

The :code:`spin_op` should take on the following structure: 

.. code-block:: cpp

    namespace cudaq {
    class spin_op {
      public:
        spin_op();
        spin_op(const spin_op&);
        bool empty() const;
        std::size_t num_qubits() const;
        std::size_t num_terms() const;
        std::complex<double> get_coefficient();
        bool is_identity() const;
        void for_each_term(std::function<void(spin_op &)> &&) const;
        void for_each_pauli(std::function<void(pauli, std::size_t)> &&) const;
        spin_op& operator=(const spin_op&);
        spin_op& operator+=(const spin_op&);
        spin_op& operator-=(const spin_op&);
        spin_op& operator*=(const spin_op&);
        bool operator==(const spin_op&);
        spin_op& operator*=(const double);
        spin_op& operator*=(const std::complex<double>)
    };

    namespace spin {
      spin_op x(const std::size_t);
      spin_op y(const std::size_t);
      spin_op z(const std::size_t);
    }
  }

