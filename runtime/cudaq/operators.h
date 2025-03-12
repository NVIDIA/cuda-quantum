/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <set>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "dynamics/evaluation.h"
#include "dynamics/operator_leafs.h"
#include "dynamics/templates.h"
#include "utils/cudaq_utils.h"
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

#define SPIN_OPS_BACKWARD_COMPATIBILITY                                        \
  template <typename T = HandlerTy,                                            \
            std::enable_if_t<std::is_same<HandlerTy, spin_handler>::value &&   \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool> = true>

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
  EvalTy evaluate(operator_arithmetics<EvalTy> arithmetics) const;

protected:
  std::unordered_map<std::string, int>
      term_map; // quick access to term index given its id (used for aggregating
                // terms)
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

  constexpr sum_op(){};
  sum_op(const sum_op<HandlerTy> &other, bool sized, int size);
  sum_op(sum_op<HandlerTy> &&other, bool sized, int size);

public:
  // called const_iterator because it will *not* modify the sum,
  // regardless of what is done with the products/iterator
  struct const_iterator {
  private:
    const sum_op<HandlerTy> *sum;
    typename std::unordered_map<std::string, int>::const_iterator iter;
    product_op<HandlerTy> current_val;

  public:
    using iterator_category = std::forward_iterator_tag;
    using difference_type = std::ptrdiff_t;
    using value_type = product_op<HandlerTy>;
    using pointer = product_op<HandlerTy> *;
    using reference = product_op<HandlerTy> &;

    const_iterator(const sum_op<HandlerTy> *sum)
        : const_iterator(sum, sum->term_map.begin()) {}

    const_iterator(const sum_op<HandlerTy> *sum,
                   std::unordered_map<std::string, int>::const_iterator &&it)
        : sum(sum), iter(std::move(it)), current_val(1.) {
      if (iter != sum->term_map.end())
        current_val = product_op<HandlerTy>(sum->coefficients[iter->second],
                                            sum->terms[iter->second]);
    }

    bool operator==(const const_iterator &other) const {
      return sum == other.sum && iter == other.iter;
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
      if (++iter != sum->term_map.end())
        current_val = product_op<HandlerTy>(sum->coefficients[iter->second],
                                            sum->terms[iter->second]);
      return *this;
    }

    // postfix
    const_iterator operator++(int) { return const_iterator(sum, iter++); }
  };

  /// @brief Get iterator to beginning of operator terms
  const_iterator begin() const { return const_iterator(this); }

  /// @brief Get iterator to end of operator terms
  const_iterator end() const {
    return const_iterator(this, this->term_map.cend());
  }

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on.
  /// By default, degrees reflect the ordering convention (endianness) used in
  /// CUDA-Q, and the ordering of the matrix returned by default by `to_matrix`.
  std::vector<std::size_t> degrees(bool application_order = true) const;

  /// @brief Return the number of operator terms that make up this operator sum.
  std::size_t num_terms() const;

  // constructors and destructors

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
  sum_op(const sum_op<HandlerTy> &other);

  // move constructor
  sum_op(sum_op<HandlerTy> &&other);

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
  sum_op<HandlerTy> &operator=(const sum_op<HandlerTy> &other);

  // move assignment operator
  sum_op<HandlerTy> &operator=(sum_op<HandlerTy> &&other);

  // evaluations

  /// @brief Return the sum_op<HandlerTy> as a string.
  std::string to_string() const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by default by `degrees`.
  /// @arg `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  complex_matrix
  to_matrix(std::unordered_map<int, int> dimensions = {},
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {},
            bool application_order = true) const;

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
  static product_op<HandlerTy> identity(int target);

  // handler specific operators

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> number(int target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> parity(int target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> position(int target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> momentum(int target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> squeeze(int target);

  HANDLER_SPECIFIC_TEMPLATE(matrix_handler)
  static product_op<T> displace(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> i(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> x(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> y(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static product_op<T> z(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static sum_op<T> plus(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  static sum_op<T> minus(int target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static product_op<T> create(int target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static product_op<T> annihilate(int target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static product_op<T> number(int target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static sum_op<T> position(int target);

  HANDLER_SPECIFIC_TEMPLATE(boson_handler)
  static sum_op<T> momentum(int target);

  HANDLER_SPECIFIC_TEMPLATE(fermion_handler)
  static product_op<T> create(int target);

  HANDLER_SPECIFIC_TEMPLATE(fermion_handler)
  static product_op<T> annihilate(int target);

  HANDLER_SPECIFIC_TEMPLATE(fermion_handler)
  static product_op<T> number(int target);

  // general utility functions

  void dump() const;

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
                                  unsigned int seed);

  // utility functions for backward compatibility

  SPIN_OPS_BACKWARD_COMPATIBILITY
  sum_op(const std::vector<double> &input_vec, std::size_t nQubits);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  sum_op(const std::vector<std::vector<bool>> &bsf_terms,
         const std::vector<std::complex<double>> &coeffs);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::vector<double> getDataRepresentation() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::pair<std::vector<std::vector<bool>>, std::vector<std::complex<double>>>
  get_raw_data() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::string to_string(bool printCoeffs) const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  void for_each_term(std::function<void(sum_op<HandlerTy> &)> &&functor) const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  void for_each_pauli(std::function<void(pauli, std::size_t)> &&functor) const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  bool is_identity() const;
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

#if !defined(NDEBUG)
  bool is_canonicalized() const;
#endif

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
  EvalTy evaluate(operator_arithmetics<EvalTy> arithmetics) const;

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
             const std::vector<HandlerTy> &atomic_operators, int size = 0);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_op(scalar_operator coefficient,
             std::vector<HandlerTy> &&atomic_operators, int size = 0);

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

  /// @brief The degrees of freedom that the operator acts on.
  /// By default, degrees reflect the ordering convention (endianness) used in
  /// CUDA-Q, and the ordering of the matrix returned by default by `to_matrix`.
  ///
  /// Specifically, the indices of a statevector with two qubits are {00, 01,
  /// 10, 11}. An ordering of degrees {0, 1} then indicates that a state where
  /// the qubit with index 0 equals 1 with probability 1 is given by
  /// the vector {0., 1., 0., 0.}.
  std::vector<std::size_t> degrees(bool application_order = true) const;

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

  // constructors and destructors

  constexpr product_op() {}

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
  product_op(const product_op<HandlerTy> &other, int size = 0);

  // move constructor
  product_op(product_op<HandlerTy> &&other, int size = 0);

  ~product_op() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op<HandlerTy> &operator=(const product_op<T> &other);

  // assignment operator
  product_op<HandlerTy> &operator=(const product_op<HandlerTy> &other);

  // move assignment operator
  product_op<HandlerTy> &operator=(product_op<HandlerTy> &&other);

  // evaluations

  /// @brief Return the `product_op<HandlerTy>` as a string.
  std::string to_string() const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by default by `degrees`.
  /// @arg  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @arg `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  complex_matrix
  to_matrix(std::unordered_map<int, int> dimensions = {},
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {},
            bool application_order = true) const;

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

  void dump() const;

  // handler specific utility functions

  HANDLER_SPECIFIC_TEMPLATE(spin_handler) // naming is not very general
  std::size_t num_qubits() const;

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  std::string get_pauli_word() const;

  HANDLER_SPECIFIC_TEMPLATE(spin_handler)
  std::vector<bool> get_binary_symplectic_form() const;

  // utility functions for backward compatibility

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::string to_string(bool printCoeffs) const;
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
typedef std::unordered_map<int, int> dimension_map;
typedef sum_op<matrix_handler> matrix_op;
typedef product_op<matrix_handler> matrix_op_term;
// commented out since this will require the complete replacement of spin ops
// everywhere
// typedef sum_op<spin_handler> spin_op;
// typedef product_op<spin_handler> spin_op_term;
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
