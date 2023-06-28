/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "matrix.h"
#include "utils/cudaq_utils.h"
#include <complex>
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

/// @brief Return a spin_op == to I on the `idx` qubit
spin_op i(const std::size_t idx);

/// @brief Return a spin_op == X on the `idx` qubit
spin_op x(const std::size_t idx);

/// @brief Return a spin_op == Y on the `idx` qubit
spin_op y(const std::size_t idx);

/// @brief Return a spin_op == Z on the `idx` qubit
spin_op z(const std::size_t idx);
} // namespace spin

/// @brief The spin_op represents a general sum of Pauli tensor products.
/// It exposes the typical algebraic operations that allow programmers to
/// define primitive Pauli operators and use them to compose larger, more
/// complex Pauli tensor products and sums thereof.
class spin_op {
public:
  /// @brief We represent the spin_op terms in binary symplectic form,
  /// i.e. each term is a vector of 1s and 0s of size 2 * nQubits,
  /// where the first n elements represent X, the next n elements
  /// represent Z, and X=Z=1 -> Y on site i, X=1, Z=0 -> X on site i,
  /// and X=0, Z=1 -> Z on site i.
  using spin_op_term = std::vector<bool>;
  using key_type = spin_op_term;
  using mapped_type = std::complex<double>;

  bool empty() const { return terms.empty(); }

  template <typename QualifiedSpinOp>
  struct iterator {

    using _iter_type =
        std::unordered_map<spin_op_term, std::complex<double>>::iterator;
    using _const_iter_type =
        std::unordered_map<spin_op_term, std::complex<double>>::const_iterator;
    using iter_type =
        std::conditional_t<std::is_same_v<QualifiedSpinOp, spin_op>, _iter_type,
                           _const_iter_type>;
    iterator(iterator &&) = default;

    iterator(iter_type i) : iter(i) {}
    ~iterator() {
      for (auto &c : created) {
        auto *ptr = c.release();
        delete ptr;
      }
      created.clear();
    }

    QualifiedSpinOp &operator*() {
      // We have to store pointers to spin_op terms here
      // so that we can return references or pointers to them
      // based on the current state of the unordered_map iterator.
      created.emplace_back(std::make_unique<spin_op>(*iter));
      return *created.back();
    }

    QualifiedSpinOp *operator->() {
      created.emplace_back(std::make_unique<spin_op>(*iter));
      return created.back().get();
    }

    iterator &operator++() {
      iter++;
      return *this;
    }
    iterator &operator++(int) {
      iterator &tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const iterator &a, const iterator &b) {
      return a.iter == b.iter;
    };
    friend bool operator!=(const iterator &a, const iterator &b) {
      return a.iter != b.iter;
    };

  private:
    iter_type iter;
    std::vector<std::unique_ptr<spin_op>> created;
  };

private:
  /// We want these creation functions to have access to
  /// spin_op constructors that programmers don't need
  friend spin_op spin::i(const std::size_t);
  friend spin_op spin::x(const std::size_t);
  friend spin_op spin::y(const std::size_t);
  friend spin_op spin::z(const std::size_t);

  /// @brief The spin_op representation. The spin_op is equivalent
  /// to a mapping of unique terms to their term coefficient.
  std::unordered_map<spin_op_term, std::complex<double>> terms;

  /// @brief Utility map that takes the Pauli enum to a string representation
  std::map<pauli, std::string> pauli_to_str{
      {pauli::I, "I"}, {pauli::X, "X"}, {pauli::Y, "Y"}, {pauli::Z, "Z"}};

  /// @brief Expand this spin_op binary symplectic representation to
  /// a larger number of qubits.
  void expandToNQubits(const std::size_t nQubits);

public:
  /// @brief The constructor, takes a single term / coefficient pair
  spin_op(std::pair<const spin_op_term, std::complex<double>> &termData);

  /// @brief The constructor, takes a single term / coefficient constant pair
  spin_op(const std::pair<const spin_op_term, std::complex<double>> &termData);

  /// @brief Constructor, takes the Pauli type, the qubit site, and the
  /// term coefficient. Constructs a `spin_op` of one Pauli on one qubit.
  spin_op(pauli, const std::size_t id, std::complex<double> coeff = 1.0);

  /// @brief Constructor, takes the binary representation of a single term and
  /// its coefficient.
  spin_op(const spin_op_term &term, const std::complex<double> &coeff);

  /// @brief Constructor, takes a full set of terms for the composite spin op
  /// as an unordered_map mapping individual terms to their coefficient.
  spin_op(const std::unordered_map<spin_op_term, std::complex<double>> &_terms);

  /// @brief Construct from a vector of term data.
  spin_op(const std::vector<spin_op_term> &bsf,
          const std::vector<std::complex<double>> &coeffs);

  /// @brief Return a random spin operator acting on the specified number of
  /// qubits and composed of the given number of terms.
  static spin_op random(std::size_t nQubits, std::size_t nTerms);

  /// @brief Constructor, creates the identity term
  spin_op();

  /// @brief Construct the identity term on the given number of qubits.
  spin_op(std::size_t numQubits);

  /// @brief Copy constructor
  spin_op(const spin_op &o);

  /// @brief Construct this spin_op from a serialized representation.
  /// Specifically, this encoding is via a vector of doubles. The encoding is
  /// as follows: for each term, a list of doubles where element `i` is
  /// a 3.0 for a Y, a 1.0 for a X, and a 2.0 for a Z on qubit i, followed by
  /// the real and imaginary part of the coefficient. Each term is appended to
  /// the array forming one large 1d array of doubles. The array is ended with
  /// the total number of terms represented as a double.
  spin_op(std::vector<double> &data_rep, std::size_t nQubits);

  /// The destructor
  ~spin_op() = default;

  /// @brief Return iterator to start of spin_op terms.
  iterator<spin_op> begin();

  /// @brief Return iterator to end of spin_op terms.
  iterator<spin_op> end();

  /// @brief Return constant iterator to start of `spin_op` terms.
  iterator<const spin_op> begin() const;

  /// @brief Return constant iterator to end of `spin_op` terms.
  iterator<const spin_op> end() const;

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

  /// @brief Return the number of qubits this spin_op is on
  std::size_t num_qubits() const;

  /// @brief Return the number of terms in this spin_op
  std::size_t num_terms() const;

  /// @brief For a spin_op with 1 term, get that terms' coefficient.
  /// Throws an exception for spin_ops with > 1 terms.
  std::complex<double> get_coefficient() const;

  /// @brief Return the binary symplectic form data
  std::pair<std::vector<spin_op_term>, std::vector<std::complex<double>>>
  get_raw_data() const;

  /// @brief Is this spin_op == to the identity
  bool is_identity() const;

  /// @brief Dump a string representation of this spin_op to standard out.
  void dump() const;

  /// @brief Return a string representation of this spin_op
  std::string to_string(bool printCoefficients = true) const;

  /// @brief Return the vector<double> serialized representation of this
  /// spin_op. (see the constructor for the encoding)
  std::vector<double> getDataRepresentation();

  /// @brief Return a vector of spin_op representing a distribution of the
  /// terms in this spin_op into equally sized chunks.
  std::vector<spin_op> distribute_terms(std::size_t numChunks) const;

  /// @brief Apply the give functor on each term of this spin_op. This method
  /// can enable general reductions via lambda capture variables.
  void for_each_term(std::function<void(spin_op &)> &&) const;

  /// @brief Apply the functor on each Pauli in this 1-term spin_op. An
  /// exception is thrown if there are more than 1 terms. Users should pass a
  /// functor that takes the `pauli` type and the qubit index.
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
