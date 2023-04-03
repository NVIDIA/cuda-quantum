/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once
#include <complex>

#include "matrix.h"
#include "utils/cudaq_utils.h"
#include <functional>
#include <map>

// Define friend functions for operations between spin_op and scalars.
#define CUDAQ_SPIN_SCALAR_OPERATIONS(op, U)                                    \
  friend spin_op operator op(const spin_op &lhs, const U &rhs) noexcept {      \
    spin_op nrv(lhs);                                                          \
    nrv op## = rhs;                                                            \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op operator op(const spin_op &lhs, U &&rhs) noexcept {           \
    spin_op nrv(lhs);                                                          \
    nrv op## = std::move(rhs);                                                 \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op &&operator op(spin_op &&lhs, const U &rhs) noexcept {         \
    lhs op## = rhs;                                                            \
    return std::move(lhs);                                                     \
  }                                                                            \
  friend spin_op &&operator op(spin_op &&lhs, U &&rhs) noexcept {              \
    lhs op## = std::move(rhs);                                                 \
    return std::move(lhs);                                                     \
  }                                                                            \
  friend spin_op operator op(const U &lhs, const spin_op &rhs) noexcept {      \
    spin_op nrv(rhs);                                                          \
    nrv op## = lhs;                                                            \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op &&operator op(const U &lhs, spin_op &&rhs) noexcept {         \
    rhs op## = lhs;                                                            \
    return std::move(rhs);                                                     \
  }                                                                            \
  friend spin_op operator op(U &&lhs, const spin_op &rhs) noexcept {           \
    spin_op nrv(rhs);                                                          \
    nrv op## = std::move(lhs);                                                 \
    return nrv;                                                                \
  }                                                                            \
  friend spin_op &&operator op(U &&lhs, spin_op &&rhs) noexcept {              \
    rhs op## = std::move(lhs);                                                 \
    return std::move(rhs);                                                     \
  }

// Define friend functions for operations between two spin_ops
#define CUDAQ_SPIN_OPERATORS(op)                                               \
  friend spin_op operator op(const spin_op &lhs,                               \
                             const spin_op &rhs) noexcept {                    \
    spin_op nrv(lhs);                                                          \
    nrv op## = rhs;                                                            \
    return nrv;                                                                \
  }                                                                            \
                                                                               \
  friend spin_op &&operator op(const spin_op &lhs, spin_op &&rhs) noexcept {   \
    rhs op## = lhs;                                                            \
    return std::move(rhs);                                                     \
  }                                                                            \
                                                                               \
  friend spin_op &&operator op(spin_op &&lhs, const spin_op &rhs) noexcept {   \
    lhs op## = rhs;                                                            \
    return std::move(lhs);                                                     \
  }                                                                            \
                                                                               \
  friend spin_op &&operator op(spin_op &&lhs, spin_op &&rhs) noexcept {        \
    lhs op## = std::move(rhs);                                                 \
    return std::move(lhs);                                                     \
  }

namespace cudaq {
class spin_op;

/// @brief Utility enum representing Paulis.
enum class pauli { I, X, Y, Z };

namespace spin {

/// @brief Return a spin_op == to I on the idx qubit
spin_op i(const std::size_t idx);

/// @brief Return a spin_op == X on the idx qubit
spin_op x(const std::size_t idx);

/// @brief Return a spin_op == Y on the idx qubit
spin_op y(const std::size_t idx);

/// @brief Return a spin_op == Z on the idx qubit
spin_op z(const std::size_t idx);
} // namespace spin

/// @brief The spin_op represents a general sum of pauli tensor products.
/// It exposes the typical algebraic operations that allow programmers to
/// define primitive pauli operators and use them to compose larger, more
/// complex pauli tensor products and sums thereof.
class spin_op {
private:
  /// We want these creation functions to have access to
  /// spin_op constructors that programmers don't need
  friend spin_op spin::i(const std::size_t);
  friend spin_op spin::x(const std::size_t);
  friend spin_op spin::y(const std::size_t);
  friend spin_op spin::z(const std::size_t);

  /// @brief We represent the spin_op in binary symplectic form,
  /// i.e. each term is a vector of 1s and 0s of size 2 * nQubits,
  /// where the first n elements represent X, the next n elements
  /// represent Z, and X=Z=1 -> Y on site i, X=1, Z=0 -> X on site i,
  /// and X=0, Z=1 -> Z on site i.
  using BinarySymplecticForm = std::vector<std::vector<bool>>;

  /// @brief The spin_op representation
  BinarySymplecticForm data;

  /// @brief The coefficients for each term in the spin_op
  std::vector<std::complex<double>> coefficients;

  /// @brief The number of qubits this spin_op is on
  std::size_t m_n_qubits = 1;

  /// @brief Utility map that takes the pauli enum to a string representation
  std::map<pauli, std::string> pauli_to_str{
      {pauli::I, "I"}, {pauli::X, "X"}, {pauli::Y, "Y"}, {pauli::Z, "Z"}};

  /// @brief Expand this spin_op binary symplectic representation to
  /// a larger number of qubits.
  void expandToNQubits(const std::size_t nQubits);

  /// @brief Internal constructor, takes the Pauli type, the qubit site, and the
  /// term coefficient. Constructs a spin_op of one pauli on one qubit.
  spin_op(pauli, const std::size_t id, std::complex<double> coeff = 1.0);

  /// @brief Internal constructor, constructs from existing binary symplectic
  /// form data and term coefficients.
  spin_op(BinarySymplecticForm bsf, std::vector<std::complex<double>> coeffs);

public:
  /// @brief Return a new spin_op from the user-provided binary symplectic data.
  static spin_op
  from_binary_symplectic(BinarySymplecticForm &data,
                         std::vector<std::complex<double>> &coeffs) {
    return spin_op(data, coeffs);
  }

  /// @brief Return a random spin_op on nQubits composed of nTerms.
  static spin_op random(std::size_t nQubits, std::size_t nTerms);

  /// @brief Constructor, creates the identity term
  spin_op();

  /// @brief Copy constructor
  spin_op(const spin_op &o);

  /// @brief Construct this spin_op from a serialized representation.
  /// Specifically, this encoding is via a vector of doubles. The encoding is
  /// as follows: for each term, a list of doubles where the ith element is
  /// a 3.0 for a Y, a 1.0 for a X, and a 2.0 for a Z on qubit i, followed by
  /// the real and imag part of the coefficient. Each term is appended to the
  /// array forming one large 1d array of doubles. The array is ended with the
  /// total number of terms represented as a double.
  spin_op(std::vector<double> &data_rep, std::size_t nQubits);

  /// The destructor
  ~spin_op() = default;

  /// @brief Set the provided spin_op equal to this one and return *this.
  spin_op &operator=(const spin_op &);

  /// @brief Add the given spin_op to this one and return *this
  spin_op &operator+=(const spin_op &v) noexcept;

  /// @brief Subtract the given spin_op from this one and return *this
  spin_op &operator-=(const spin_op &v) noexcept;

  /// @brief Multiply the given spin_op with this one and return *this
  spin_op &operator*=(const spin_op &v) noexcept;

  /// @brief Return true if this spin_op is equal to the given one. Equality
  /// here does not consider the coefficients.
  bool operator==(const spin_op &v) const noexcept;

  /// @brief Multiply this spin_op by the given double.
  spin_op &operator*=(const double v) noexcept;

  /// @brief Multiply this spin_op by the given complex value
  spin_op &operator*=(const std::complex<double> v) noexcept;

  CUDAQ_SPIN_SCALAR_OPERATIONS(*, double)
  CUDAQ_SPIN_SCALAR_OPERATIONS(*, std::complex<double>)
  CUDAQ_SPIN_OPERATORS(+)
  CUDAQ_SPIN_OPERATORS(*)

  // Define the subtraction operators
  friend spin_op operator-(const spin_op &lhs, const spin_op &rhs) noexcept {
    spin_op nrv(lhs);
    nrv -= rhs;
    return nrv;
  }

  friend spin_op operator-(const spin_op &lhs, spin_op &&rhs) noexcept {
    spin_op nrv(lhs);
    nrv -= std::move(rhs);
    return nrv;
  }

  friend spin_op &&operator-(spin_op &&lhs, const spin_op &rhs) noexcept {
    lhs -= rhs;
    return std::move(lhs);
  }

  friend spin_op &&operator-(spin_op &&lhs, spin_op &&rhs) noexcept {
    lhs -= std::move(rhs);
    return std::move(lhs);
  }

  /// @brief Return the ith term of this spin_op (by value).
  spin_op operator[](const std::size_t termIdx) const;

  /// @brief Return the number of qubits this spin_op is on
  std::size_t n_qubits() const;

  /// @brief Return the number of terms in this spin_op
  std::size_t n_terms() const;

  /// @brief Return the coefficient on the ith term in this spin_op
  std::complex<double> get_term_coefficient(const std::size_t idx) const;

  /// @brief Return the binary symplectic form data
  BinarySymplecticForm get_bsf() const;

  /// @brief Is this spin_op == to the identity
  bool is_identity();

  /// @brief Dump a string representation of this spin_op to standard out.
  void dump() const;

  /// @brief Return a string representation of this spin_op
  std::string to_string(bool printCoefficients = true) const;

  /// @brief Return the vector<double> serialized representation of this
  /// spin_op. (see the constructor for the encoding)
  std::vector<double> getDataRepresentation();

  /// @brief Return all term coefficients in this spin_op
  std::vector<std::complex<double>> get_coefficients() const;

  /// @brief Return a new spin_op made up of a sum of spin_op terms
  /// where the first term is the one at startIdx, and the remaining terms
  /// are the next count terms.
  spin_op slice(const std::size_t startIdx, const std::size_t count);

  /// @brief Apply the give functor on each term of this spin_op. This method
  /// can enable general reductions via lambda capture variables.
  void for_each_term(std::function<void(spin_op &)> &&) const;

  /// @brief Apply the functor on each pauli in this 1-term spin_op. An
  /// exception is thrown if there are more than 1 terms. Users should pass a
  /// functor that takes the pauli type and the qubit index.
  void for_each_pauli(std::function<void(pauli, std::size_t)> &&) const;

  /// @brief Return a dense matrix representation of this
  /// spin_op.
  complex_matrix to_matrix() const;
};

/// @brief Add a double and a spin_op
spin_op operator+(double coeff, spin_op op);

/// @brief Add a double and a spin_op
spin_op operator+(spin_op op, double coeff);

/// @brief Subtract a double and a spin_op
spin_op operator-(double coeff, spin_op op);

/// @brief Subtract a spin_op and a double
spin_op operator-(spin_op op, double coeff);

class spin_op_reader {
public:
  virtual ~spin_op_reader() = default;
  virtual spin_op read(const std::string &data_filename) = 0;
};

class binary_spin_op_reader : public spin_op_reader {
public:
  spin_op read(const std::string &data_filename) override;
};
} // namespace cudaq
