Quantum Operators
*****************

:code:`cudaq::spin_op`
----------------------
CUDA Quantum provides a native :code:`spin_op` data type in the :code:`cudaq` namespace for the
expression of quantum mechanical spin operators. These operators
should provide an abstraction for a general tensor product of Pauli
spin operators, and sums thereof:

.. math:: 

    H = \sum_{i=0}^M P_i, P_i = \prod_{j=0}^N \sigma_j^a

for :math:`a = {x,y,z}`, :math:`j` the qubit index, and :math:`N` the number of qubits.

Critically, the :code:`spin_op` exposes common C++ operator overloads
for algebraic expressions. 

CUDA Quantum defines convenience functions in :code:`cudaq::spin` namespace that produce
the primitive X, Y, and Z Pauli operators on specified qubit indices
which can subsequently be used in algebraic expressions to build up
more complicated Pauli tensor products and their sums.

.. code-block:: cpp

    using namespace cudaq::spin;
    auto h = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) + \
             .21829 * z(0) - 6.125 * z(1);

The :code:`spin_op` also provides a mechanism for the expression of circuit
synthesis tasks within quantum kernel code. Specifically, operations
that encode :math:`N`\ :sup:`th`\ order trotterization of exponentiated :code:`spin_op`
rotations, e.g. :math:`U = \exp(-i H t)`, where :math:`H` is the provided :code:`spin_op`.

The :code:`spin_op` can be created within classical host code and quantum kernel
code, and can also be passed by value to quantum kernel code from host code. 

.. code-block:: cpp

    namespace cudaq {
      enum class pauli { I, X, Y, Z };

      class spin_op {
        public:
          using binary_symplectic_form = std::vector<std::vector<bool>>;
          
          spin_op();
          spin_op(const spin_op&);
          static spin_op from_binary_symplectic(binary_symplectic_form& data, 
                            std::vector<std::complex<double>>& coeffs);

          std::size_t n_qubits() const;
          std::size_t n_terms() const;
          std::complex<double> get_term_coefficient(std::size_t idx);
          std::vector<std::complex<double>> get_coefficients();
          bool is_identity();
          std::string to_string() const;

          // Extract a set of terms
          spin_op slice(std::size_t startIdx, std::size_t count);

          // Custom operations on terms and paulis
          void for_each_term(std::function<void(spin_op&)>&&);
          void for_each_pauli(std::function<void(pauli, std::size_t)>&&);

          // Common algebraic overloads
          spin_op& operator=(const spin_op&);
          spin_op& operator+=(const spin_op&);
          spin_op& operator-=(const spin_op&);
          spin_op& operator*=(const spin_op&);
          bool operator==(const spin_op&);
          spin_op& operator*=(const double);
          spin_op& operator*=(const std::complex<double>)
          // ... other algebraic overloads ...
          
          spin_op operator[](std::size_t);
      };
 
      namespace spin {
        spin_op i(size_t);
        spin_op x(std::size_t);
        spin_op y(std::size_t);
        spin_op z(std::size_t);
      }
    }

The :code:`spin_op` is intended only for two-level qudit (qubit) CUDA Quantum programs.