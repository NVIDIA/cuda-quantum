/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <random>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "operators/evaluation.h"
#include "operators/operator_leafs.h"
#include "operators/templates.h"
#include "utils/matrix.h"

namespace cudaq {

class spin_handler;
enum class pauli;

#define HANDLER_SPECIFIC_TEMPLATE(ConcreteTy)                                  \
  template <typename T = HandlerTy,                                            \
            std::enable_if_t<std::is_same<T, ConcreteTy>::value &&             \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool> = true>

// utility functions for backward compatibility

#define SPIN_OPS_BACKWARD_COMPATIBILITY(deprecation_message)                   \
  template <typename T = HandlerTy,                                            \
            std::enable_if_t<std::is_same<HandlerTy, spin_handler>::value &&   \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool> = true>                                     \
  [[deprecated(deprecation_message)]]

/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy>
class sum_op {
  template <typename T>
  friend class sum_op;
  template <typename T>
  friend class product_op;

private:
  // inserts a new term combining it with an existing one if possible
  void insert(product_op<HandlerTy> &&other);
  void insert(const product_op<HandlerTy> &other);

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(product_op<HandlerTy> &&head, Args &&...args);

  template <typename EvalTy>
  EvalTy transform(operator_arithmetics<EvalTy> arithmetics) const;

protected:
  std::unordered_map<std::string, std::size_t>
      term_map; // quick access to term index given its id (used for aggregating
                // terms)
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;
  bool is_default = true;

  constexpr sum_op(bool is_default) : is_default(is_default){};
  sum_op(const sum_op<HandlerTy> &other, bool is_default, std::size_t size);
  sum_op(sum_op<HandlerTy> &&other, bool is_default, std::size_t size);

public:
  // called const_iterator because it will *not* modify the sum,
  // regardless of what is done with the products/iterator
  struct const_iterator {
  private:
    const sum_op<HandlerTy> *sum;
    product_op<HandlerTy> current_val;
    std::size_t current_idx;

    const_iterator(const sum_op<HandlerTy> *sum, std::size_t idx,
                   product_op<HandlerTy> &&value)
        : sum(sum), current_val(std::move(value)), current_idx(idx) {}

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = product_op<HandlerTy>;
    using pointer = product_op<HandlerTy> *;
    using reference = product_op<HandlerTy> &;

    const_iterator(const sum_op<HandlerTy> *sum) : const_iterator(sum, 0) {}

    const_iterator(const sum_op<HandlerTy> *sum, std::size_t idx)
        : sum(sum), current_val(1.), current_idx(idx) {
      if (sum && current_idx < sum->num_terms())
        current_val = product_op<HandlerTy>(sum->coefficients[current_idx],
                                            sum->terms[current_idx]);
    }

    bool operator==(const const_iterator &other) const {
      return sum == other.sum && current_idx == other.current_idx;
    }

    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    reference operator*() {
      return current_val;
    } // not const - allow to move current_value
    pointer operator->() { return &current_val; }

    // prefix
    const_iterator &operator++() {
      if (++current_idx < sum->num_terms())
        current_val = product_op<HandlerTy>(sum->coefficients[current_idx],
                                            sum->terms[current_idx]);
      return *this;
    }

    // postfix
    const_iterator operator++(int) {
      auto iter = const_iterator(sum, current_idx, std::move(current_val));
      ++(*this);
      return iter;
    }
  };

  /// @brief Get iterator to beginning of operator terms
  const_iterator begin() const { return const_iterator(this); }

  /// @brief Get iterator to end of operator terms
  const_iterator end() const { return const_iterator(this, this->num_terms()); }

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on.
  /// The order of degrees is from smallest to largest and reflect
  /// the ordering of the matrix returned by `to_matrix`.
  /// Specifically, the indices of a statevector with two qubits are {00, 01,
  /// 10, 11}. An ordering of degrees {0, 1} then indicates that a state where
  /// the qubit with index 0 equals 1 with probability 1 is given by
  /// the vector {0., 1., 0., 0.}.
  std::vector<std::size_t> degrees() const;
  std::size_t min_degree() const;
  std::size_t max_degree() const;

  /// @brief Return the number of operator terms that make up this operator sum.
  std::size_t num_terms() const;

  std::unordered_map<std::string, std::string>
  get_parameter_descriptions() const;

  // constructors and destructors

  // A default initialized sum will act as both the additive
  // and multiplicative identity. To construct a true "0" value
  // (neutral element for addition only), use sum_op<T>::empty().
  constexpr sum_op() = default;

  sum_op(std::size_t size);

  template <typename... Args,
            std::enable_if_t<std::conjunction<std::is_same<
                                 product_op<HandlerTy>, Args>...>::value &&
                                 sizeof...(Args),
                             bool> = true>
  sum_op(Args &&...args);

  sum_op(const product_op<HandlerTy> &other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op(const sum_op<T> &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_handler>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op(const sum_op<T> &other,
         const matrix_handler::commutation_behavior &behavior);

  // copy constructor
  sum_op(const sum_op<HandlerTy> &other) = default;

  // move constructor
  sum_op(sum_op<HandlerTy> &&other) = default;

  ~sum_op() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op<HandlerTy> &operator=(const product_op<T> &other);

  sum_op<HandlerTy> &operator=(const product_op<HandlerTy> &other);

  sum_op<HandlerTy> &operator=(product_op<HandlerTy> &&other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op<HandlerTy> &operator=(const sum_op<T> &other);

  // assignment operator
  sum_op<HandlerTy> &operator=(const sum_op<HandlerTy> &other) = default;

  // move assignment operator
  sum_op<HandlerTy> &operator=(sum_op<HandlerTy> &&other) = default;

  // evaluations

  /// @brief Return the matrix representation of the operator.
  /// The matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by `degrees`. See
  /// the documentation for `degrees` for more detail.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  /// @arg `invert_order`: if set to true, the ordering convention is reversed.
  complex_matrix
  to_matrix(std::unordered_map<std::size_t, int64_t> dimensions = {},
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {},
            bool invert_order = false) const;

  // comparisons

  /// @brief True, if the other value is an sum_op<HandlerTy> with
  /// equivalent terms, and False otherwise.
  /// The equality takes into account that operator
  /// addition is commutative, as is the product of two operators if they
  /// act on different degrees of freedom.
  /// The equality comparison does *not* take commutation relations into
  /// account, and does not try to reorder terms `blockwise`; it may hence
  /// evaluate to False, even if two operators in reality are the same.
  /// If the equality evaluates to True, on the other hand, the operators
  /// are guaranteed to represent the same transformation for all arguments.
  bool operator==(const sum_op<HandlerTy> &other) const;

  // unary operators

  sum_op<HandlerTy> operator-() const &;
  sum_op<HandlerTy> operator-() &&;
  sum_op<HandlerTy> operator+() const &;
  sum_op<HandlerTy> operator+() &&;

  // right-hand arithmetics

  sum_op<HandlerTy> operator*(const scalar_operator &other) const &;
  sum_op<HandlerTy> operator*(const scalar_operator &other) &&;
  sum_op<HandlerTy> operator/(const scalar_operator &other) const &;
  sum_op<HandlerTy> operator/(const scalar_operator &other) &&;
  sum_op<HandlerTy> operator+(scalar_operator &&other) const &;
  sum_op<HandlerTy> operator+(scalar_operator &&other) &&;
  sum_op<HandlerTy> operator+(const scalar_operator &other) const &;
  sum_op<HandlerTy> operator+(const scalar_operator &other) &&;
  sum_op<HandlerTy> operator-(scalar_operator &&other) const &;
  sum_op<HandlerTy> operator-(scalar_operator &&other) &&;
  sum_op<HandlerTy> operator-(const scalar_operator &other) const &;
  sum_op<HandlerTy> operator-(const scalar_operator &other) &&;
  sum_op<HandlerTy> operator*(const product_op<HandlerTy> &other) const;
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator*(const sum_op<HandlerTy> &other) const;
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) &&;

  sum_op<HandlerTy> &operator*=(const scalar_operator &other);
  sum_op<HandlerTy> &operator/=(const scalar_operator &other);
  sum_op<HandlerTy> &operator+=(scalar_operator &&other);
  sum_op<HandlerTy> &operator+=(const scalar_operator &other);
  sum_op<HandlerTy> &operator-=(scalar_operator &&other);
  sum_op<HandlerTy> &operator-=(const scalar_operator &other);
  sum_op<HandlerTy> &operator*=(const product_op<HandlerTy> &other);
  sum_op<HandlerTy> &operator+=(const product_op<HandlerTy> &other);
  sum_op<HandlerTy> &operator+=(product_op<HandlerTy> &&other);
  sum_op<HandlerTy> &operator-=(const product_op<HandlerTy> &other);
  sum_op<HandlerTy> &operator-=(product_op<HandlerTy> &&other);
  sum_op<HandlerTy> &operator*=(const sum_op<HandlerTy> &other);
  sum_op<HandlerTy> &operator+=(const sum_op<HandlerTy> &other);
  sum_op<HandlerTy> &operator+=(sum_op<HandlerTy> &&other);
  sum_op<HandlerTy> &operator-=(const sum_op<HandlerTy> &other);
  sum_op<HandlerTy> &operator-=(sum_op<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other,
                             const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other, sum_op<T> &&self);

  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             product_op<T> &&self);

  // common operators

  static sum_op<HandlerTy> empty();
  static product_op<HandlerTy> identity();
  static product_op<HandlerTy> identity(std::size_t target);

  // handler specific operators

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> number(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> parity(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> position(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> momentum(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> squeeze(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> displace(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> i(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> x(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> y(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> z(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static sum_op<T> plus(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static sum_op<T> minus(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static product_op<T> create(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static product_op<T> annihilate(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static product_op<T> number(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static sum_op<T> position(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static sum_op<T> momentum(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(fermion_handler)
  static product_op<T> create(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(fermion_handler)
  static product_op<T> annihilate(std::size_t target);

  HANDLER_SPECIFIC_TEMPLATE(fermion_handler)
  static product_op<T> number(std::size_t target);

  // general utility functions

  /// @brief Return the string representation of the operator.
  std::string to_string() const;

  /// @brief Print the string representation of the operator to the standard
  /// output.
  void dump() const;

  /// Removes all terms from the sum for which the absolute value of
  /// the coefficient is below the given tolerance
  sum_op<HandlerTy> &
  trim(double tol = 0.0,
       const std::unordered_map<std::string, std::complex<double>> &parameters =
           {});

  /// Removes all identity operators from the operator.
  sum_op<HandlerTy> &canonicalize();
  static sum_op<HandlerTy> canonicalize(const sum_op<HandlerTy> &orig);

  /// Expands the operator to act on all given degrees, applying identities as
  /// needed. If an empty set is passed, canonicalizes all terms in the sum to
  /// act on the same degrees of freedom.
  sum_op<HandlerTy> &canonicalize(const std::set<std::size_t> &degrees);
  static sum_op<HandlerTy> canonicalize(const sum_op<HandlerTy> &orig,
                                        const std::set<std::size_t> &degrees);

  std::vector<sum_op<HandlerTy>> distribute_terms(std::size_t numChunks) const;

  // handler specific utility functions

  HANDLER_SPECIFIC_TEMPLATE(spin_handler) // naming is not very general
  std::size_t num_qubits() const;

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  sum_op(const std::vector<double> &input_vec);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<HandlerTy> from_word(const std::string &word);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static sum_op<HandlerTy> random(std::size_t nQubits, std::size_t nTerms,
                                  unsigned int seed = std::random_device{}());

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by `degrees`. See
  /// the documentation for `degrees` for more detail.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  /// @arg `invert_order`: if set to true, the ordering convention is reversed.
  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  csr_spmatrix
  to_sparse_matrix(std::unordered_map<std::size_t, int64_t> dimensions = {},
                   const std::unordered_map<std::string, std::complex<double>>
                       &parameters = {},
                   bool invert_order = false) const;

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  std::vector<double> get_data_representation() const;

  // utility functions for backward compatibility
  /// @cond

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "serialization format changed - use the constructor without a size_t "
      "argument to create a spin_op from the new format")
  sum_op(const std::vector<double> &input_vec, std::size_t nQubits);

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "construction from binary symplectic form will no longer be supported")
  sum_op(const std::vector<std::vector<bool>> &bsf_terms,
         const std::vector<std::complex<double>> &coeffs);

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "serialization format changed - use get_data_representation instead")
  std::vector<double> getDataRepresentation() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "data tuple is no longer used for serialization - use "
      "get_data_representation instead")
  std::tuple<std::vector<double>, std::size_t> getDataTuple() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY("raw data access will no longer be supported")
  std::pair<std::vector<std::vector<bool>>, std::vector<std::complex<double>>>
  get_raw_data() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "use to_string(), get_term_id or get_pauli_word depending on your use "
      "case - see release notes for more detail")
  std::string to_string(bool printCoeffs) const;

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "iterate over the operator instead to access each term")
  void for_each_term(std::function<void(sum_op<HandlerTy> &)> &&functor) const;

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "iterate over each term in the operator instead and use as_pauli to "
      "access each pauli")
  void for_each_pauli(std::function<void(pauli, std::size_t)> &&functor) const;

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "is_identity will no longer be supported on an entire sum_op, but will "
      "continue to be supported on each term")
  bool is_identity() const;

  /// @endcond
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy>
class product_op {
  template <typename T>
  friend class product_op;
  template <typename T>
  friend class sum_op;

private:
  // template defined as long as T implements an in-place multiplication -
  // won't work if the in-place multiplication was inherited from a base class
  template <typename T>
  static decltype(std::declval<T>().inplace_mult(std::declval<T>()))
  handler_mult(int);
  template <typename T>
  static std::false_type handler_mult(
      ...); // ellipsis ensures the template above is picked if it exists
  static constexpr bool supports_inplace_mult =
      !std::is_same<decltype(handler_mult<HandlerTy>(0)),
                    std::false_type>::value;

  typename std::vector<HandlerTy>::const_iterator
  find_insert_at(const HandlerTy &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                                 !product_op<T>::supports_inplace_mult,
                             std::false_type> = std::false_type()>
  void insert(T &&other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                                 product_op<T>::supports_inplace_mult,
                             std::true_type> = std::true_type()>
  void insert(T &&other);

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(HandlerTy &&head, Args &&...args);

  template <typename EvalTy>
  EvalTy transform(operator_arithmetics<EvalTy> arithmetics) const;

protected:
  std::vector<HandlerTy> operators;
  scalar_operator coefficient;

  template <typename... Args,
            std::enable_if_t<
                std::conjunction<std::is_same<HandlerTy, Args>...>::value,
                bool> = true>
  product_op(scalar_operator coefficient, Args &&...args);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_op(scalar_operator coefficient,
             const std::vector<HandlerTy> &atomic_operators,
             std::size_t size = 0);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_op(scalar_operator coefficient,
             std::vector<HandlerTy> &&atomic_operators, std::size_t size = 0);

public:
  struct const_iterator {
  private:
    const product_op<HandlerTy> *prod;
    std::size_t current_idx;

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = const HandlerTy;
    using pointer = const HandlerTy *;
    using reference = const HandlerTy &;

    const_iterator(const product_op<HandlerTy> *prod, std::size_t idx = 0)
        : prod(prod), current_idx(idx) {}

    bool operator==(const const_iterator &other) const {
      return prod == other.prod && current_idx == other.current_idx;
    }

    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    reference operator*() const { return prod->operators[current_idx]; }
    pointer operator->() { return &(prod->operators[current_idx]); }

    // prefix
    const_iterator &operator++() {
      ++current_idx;
      return *this;
    }
    const_iterator &operator--() {
      --current_idx;
      return *this;
    }

    // postfix
    const_iterator operator++(int) {
      return const_iterator(prod, current_idx++);
    }
    const_iterator operator--(int) {
      return const_iterator(prod, current_idx--);
    }
  };

  /// @brief Get iterator to beginning of operator terms
  const_iterator begin() const { return const_iterator(this); }

  /// @brief Get iterator to end of operator terms
  const_iterator end() const {
    return const_iterator(this, this->operators.size());
  }

  // read-only properties

#if !defined(NDEBUG)
  bool is_canonicalized() const;
#endif

  /// @brief The degrees of freedom that the operator acts on.
  /// The order of degrees is from smallest to largest and reflects
  /// the ordering of the matrix returned by `to_matrix`.
  /// Specifically, the indices of a statevector with two qubits are {00, 01,
  /// 10, 11}. An ordering of degrees {0, 1} then indicates that a state where
  /// the qubit with index 0 equals 1 with probability 1 is given by
  /// the vector {0., 1., 0., 0.}.
  std::vector<std::size_t> degrees() const;
  std::size_t min_degree() const;
  std::size_t max_degree() const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  std::size_t num_ops() const;

  // Public since it is used by the CUDA-Q compiler and runtime
  // to retrieve expectation values for specific terms.
  // The term id uniquely identifies the operators and targets
  // (degrees) that they act on, but does not include information
  // about the coefficient.
  std::string get_term_id() const;

  scalar_operator get_coefficient() const;

  std::unordered_map<std::string, std::string>
  get_parameter_descriptions() const;

  // constructors and destructors

  constexpr product_op() {}

  constexpr product_op(std::size_t first_degree, std::size_t last_degree) {
    static_assert(std::is_constructible_v<HandlerTy, std::size_t>,
                  "operator handlers must have a constructor that take a "
                  "single degree of "
                  "freedom and returns the identity operator on that degree.");
    if (last_degree > first_degree) // being a bit permissive here
      this->operators.reserve(last_degree - first_degree);
    for (auto degree = first_degree; degree < last_degree; ++degree)
      this->operators.push_back(HandlerTy(degree));
  }

  product_op(double coefficient);

  product_op(std::complex<double> coefficient);

  product_op(HandlerTy &&atomic);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op(const product_op<T> &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_handler>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op(const product_op<T> &other,
             const matrix_handler::commutation_behavior &behavior);

  // copy constructor
  product_op(const product_op<HandlerTy> &other, std::size_t size = 0);

  // move constructor
  product_op(product_op<HandlerTy> &&other, std::size_t size = 0);

  ~product_op() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op<HandlerTy> &operator=(const product_op<T> &other);

  // assignment operator
  product_op<HandlerTy> &
  operator=(const product_op<HandlerTy> &other) = default;

  // move assignment operator
  product_op<HandlerTy> &operator=(product_op<HandlerTy> &&other) = default;

  // evaluations

  std::complex<double> evaluate_coefficient(
      const std::unordered_map<std::string, std::complex<double>> &parameters =
          {}) const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by `degrees`. See
  /// the documentation for `degrees` for more detail.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  /// @arg `invert_order`: if set to true, the ordering convention is reversed.
  complex_matrix
  to_matrix(std::unordered_map<std::size_t, int64_t> dimensions = {},
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {},
            bool invert_order = false) const;

  // comparisons

  /// @brief True, if the other value is an sum_op<HandlerTy> with
  /// equivalent terms, and False otherwise.
  /// The equality takes into account that operator
  /// addition is commutative, as is the product of two operators if they
  /// act on different degrees of freedom.
  /// The equality comparison does *not* take commutation relations into
  /// account, and does not try to reorder terms `blockwise`; it may hence
  /// evaluate to False, even if two operators in reality are the same.
  /// If the equality evaluates to True, on the other hand, the operators
  /// are guaranteed to represent the same transformation for all arguments.
  bool operator==(const product_op<HandlerTy> &other) const;

  // unary operators

  product_op<HandlerTy> operator-() const &;
  product_op<HandlerTy> operator-() &&;
  product_op<HandlerTy> operator+() const &;
  product_op<HandlerTy> operator+() &&;

  // right-hand arithmetics

  product_op<HandlerTy> operator*(scalar_operator &&other) const &;
  product_op<HandlerTy> operator*(scalar_operator &&other) &&;
  product_op<HandlerTy> operator*(const scalar_operator &other) const &;
  product_op<HandlerTy> operator*(const scalar_operator &other) &&;
  product_op<HandlerTy> operator/(scalar_operator &&other) const &;
  product_op<HandlerTy> operator/(scalar_operator &&other) &&;
  product_op<HandlerTy> operator/(const scalar_operator &other) const &;
  product_op<HandlerTy> operator/(const scalar_operator &other) &&;
  sum_op<HandlerTy> operator+(scalar_operator &&other) const &;
  sum_op<HandlerTy> operator+(scalar_operator &&other) &&;
  sum_op<HandlerTy> operator+(const scalar_operator &other) const &;
  sum_op<HandlerTy> operator+(const scalar_operator &other) &&;
  sum_op<HandlerTy> operator-(scalar_operator &&other) const &;
  sum_op<HandlerTy> operator-(scalar_operator &&other) &&;
  sum_op<HandlerTy> operator-(const scalar_operator &other) const &;
  sum_op<HandlerTy> operator-(const scalar_operator &other) &&;
  product_op<HandlerTy> operator*(const product_op<HandlerTy> &other) const &;
  product_op<HandlerTy> operator*(const product_op<HandlerTy> &other) &&;
  product_op<HandlerTy> operator*(product_op<HandlerTy> &&other) const &;
  product_op<HandlerTy> operator*(product_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator*(const sum_op<HandlerTy> &other) const;
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) &&;
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) const &;
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) &&;
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) const &;
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) &&;

  product_op<HandlerTy> &operator*=(const scalar_operator &other);
  product_op<HandlerTy> &operator/=(const scalar_operator &other);
  product_op<HandlerTy> &operator*=(const product_op<HandlerTy> &other);
  product_op<HandlerTy> &operator*=(product_op<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend product_op<T> operator*(scalar_operator &&other,
                                 const product_op<T> &self);
  template <typename T>
  friend product_op<T> operator*(scalar_operator &&other, product_op<T> &&self);
  template <typename T>
  friend product_op<T> operator*(const scalar_operator &other,
                                 const product_op<T> &self);
  template <typename T>
  friend product_op<T> operator*(const scalar_operator &other,
                                 product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, product_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const product_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             product_op<T> &&self);

  template <typename T>
  friend sum_op<T> operator*(scalar_operator &&other, const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator*(scalar_operator &&other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other,
                             const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, sum_op<T> &&self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const sum_op<T> &self);
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other, sum_op<T> &&self);

  // general utility functions

  // Checks if all operators in the product are the identity.
  // Note: this function returns true regardless of the value
  // of the coefficient.
  bool is_identity() const;

  /// @brief Return the string representation of the operator.
  std::string to_string() const;

  /// @brief Print the string representation of the operator to the standard
  /// output.
  void dump() const;

  /// Removes all identity operators from the operator.
  product_op<HandlerTy> &canonicalize();
  static product_op<HandlerTy> canonicalize(const product_op<HandlerTy> &orig);

  /// Expands the operator to act on all given degrees, applying identities as
  /// needed.
  product_op<HandlerTy> &canonicalize(const std::set<std::size_t> &degrees);
  static product_op<HandlerTy>
  canonicalize(const product_op<HandlerTy> &orig,
               const std::set<std::size_t> &degrees);

  // handler specific utility functions

  HANDLER_SPECIFIC_TEMPLATE(spin_handler) // naming is not very general
  std::size_t num_qubits() const;

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  std::string get_pauli_word(std::size_t pad_identities = 0) const;

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  std::vector<bool> get_binary_symplectic_form() const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by `degrees`. See
  /// the documentation for `degrees` for more detail.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  /// @arg `invert_order`: if set to true, the ordering convention is reversed.
  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  csr_spmatrix
  to_sparse_matrix(std::unordered_map<std::size_t, int64_t> dimensions = {},
                   const std::unordered_map<std::string, std::complex<double>>
                       &parameters = {},
                   bool invert_order = false) const;

  // utility functions for backward compatibility
  /// @cond

  SPIN_OPS_BACKWARD_COMPATIBILITY(
      "use to_string(), get_term_id or get_pauli_word depending on your use "
      "case - see release notes for more detail")
  std::string to_string(bool printCoeffs) const;

  /// @endcond
};

/// @brief Representation of a time-dependent Hamiltonian for Rydberg system
class rydberg_hamiltonian {
public:
  using coordinate = std::pair<double, double>;

  /// @brief Constructor.
  /// @param atom_sites List of 2D coordinates for trap sites.
  /// @param amplitude Time-dependent driving amplitude, Omega(t).
  /// @param phase Time-dependent driving phase, phi(t).
  /// @param delta_global Time-dependent driving detuning, Delta_global(t).
  /// @param atom_filling Optional. Marks occupied trap sites (1) and empty
  /// sites (0). Defaults to all sites occupied.
  /// @param delta_local Optional. A tuple of Delta_local(t) and site dependent
  /// local detuning factors.
  rydberg_hamiltonian(
      const std::vector<coordinate> &atom_sites,
      const scalar_operator &amplitude, const scalar_operator &phase,
      const scalar_operator &delta_global,
      const std::vector<int> &atom_filling = {},
      const std::optional<std::pair<scalar_operator, std::vector<double>>>
          &delta_local = std::nullopt);

  /// @brief Get atom sites.
  const std::vector<coordinate> &get_atom_sites() const;

  /// @brief Get atom filling.
  const std::vector<int> &get_atom_filling() const;

  /// @brief Get amplitude operator.
  const scalar_operator &get_amplitude() const;

  /// @brief Get phase operator.
  const scalar_operator &get_phase() const;

  /// @brief Get global detuning operator.
  const scalar_operator &get_delta_global() const;

private:
  std::vector<coordinate> atom_sites;
  std::vector<int> atom_filling;
  scalar_operator amplitude;
  scalar_operator phase;
  scalar_operator delta_global;
  std::optional<std::pair<scalar_operator, std::vector<double>>> delta_local;
};

// type aliases for convenience
typedef std::unordered_map<std::string, std::complex<double>> parameter_map;
typedef std::unordered_map<std::size_t, int64_t> dimension_map;
typedef sum_op<matrix_handler> matrix_op;
typedef product_op<matrix_handler> matrix_op_term;
typedef sum_op<spin_handler> spin_op;
typedef product_op<spin_handler> spin_op_term;
typedef sum_op<boson_handler> boson_op;
typedef product_op<boson_handler> boson_op_term;
typedef sum_op<fermion_handler> fermion_op;
typedef product_op<fermion_handler> fermion_op_term;

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
extern template class product_op<matrix_handler>;
extern template class product_op<spin_handler>;
extern template class product_op<boson_handler>;
extern template class product_op<fermion_handler>;

extern template class sum_op<matrix_handler>;
extern template class sum_op<spin_handler>;
extern template class sum_op<boson_handler>;
extern template class sum_op<fermion_handler>;
#endif

} // namespace cudaq
