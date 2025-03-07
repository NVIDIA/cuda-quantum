/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

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

// Here primarily for backward compatibility
class spin_operator;

#define HANDLER_SPECIFIC_TEMPLATE(ConcreteTy)                                             \
  template <typename T = HandlerTy, std::enable_if_t<                                     \
                                      std::is_same<HandlerTy, ConcreteTy>::value &&       \
                                      std::is_same<HandlerTy, T>::value, bool> = true>

  // utility functions for backward compatibility

#define SPIN_OPS_BACKWARD_COMPATIBILITY                                                   \
  template <typename T = HandlerTy, std::enable_if_t<                                     \
                                      std::is_same<HandlerTy, spin_operator>::value &&    \
                                      std::is_same<HandlerTy, T>::value, bool> = true>


/// @brief Represents an operator expression consisting of a sum of terms, where
/// each term is a product of elementary and scalar operators. Operator
/// expressions cannot be used within quantum kernels, but they provide methods
/// to convert them to data types that can.
template <typename HandlerTy>
class operator_sum {
  template <typename T>
  friend class operator_sum;
  template <typename T>
  friend class product_operator;

private:
  // inserts a new term combining it with an existing one if possible
  void insert(product_operator<HandlerTy> &&other);
  void insert(const product_operator<HandlerTy> &other);

  void aggregate_terms();

  template <typename... Args>
  void aggregate_terms(product_operator<HandlerTy> &&head, Args &&...args);

  template <typename EvalTy>
  EvalTy evaluate(operator_arithmetics<EvalTy> arithmetics) const;

protected:
  std::unordered_map<std::string, int>
      term_map; // quick access to term index given its id (used for aggregating
                // terms)
  std::vector<std::vector<HandlerTy>> terms;
  std::vector<scalar_operator> coefficients;

  template <typename... Args,
            std::enable_if_t<std::conjunction<std::is_same<
                                 product_operator<HandlerTy>, Args>...>::value,
                             bool> = true>
  operator_sum(Args &&...args);

public:
  // read-only properties

  /// @brief The degrees of freedom that the operator acts on.
  /// By default, degrees reflect the ordering convention (endianness) used in
  /// CUDA-Q, and the ordering of the matrix returned by default by `to_matrix`.
  std::vector<std::size_t> degrees(bool application_order = true) const;

  /// @brief Return the number of operator terms that make up this operator sum.
  std::size_t num_terms() const;

  /// FIXME: GET RID OF THIS (MAKE ITERABLE INSTEAD)
  std::vector<product_operator<HandlerTy>> get_terms() const;

  // constructors and destructors

  operator_sum(const product_operator<HandlerTy> &other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum(const operator_sum<T> &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_operator>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum(const operator_sum<T> &other,
               const matrix_operator::commutation_behavior &behavior);

  // copy constructor
  operator_sum(const operator_sum<HandlerTy> &other, int size = 0);

  // move constructor
  operator_sum(operator_sum<HandlerTy> &&other, int size = 0);

  ~operator_sum() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum<HandlerTy> &operator=(const product_operator<T> &other);

  operator_sum<HandlerTy> &operator=(const product_operator<HandlerTy> &other);

  operator_sum<HandlerTy> &operator=(product_operator<HandlerTy> &&other);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  operator_sum<HandlerTy> &operator=(const operator_sum<T> &other);

  // assignment operator
  operator_sum<HandlerTy> &operator=(const operator_sum<HandlerTy> &other);

  // move assignment operator
  operator_sum<HandlerTy> &operator=(operator_sum<HandlerTy> &&other);

  // evaluations

  /// @brief Return the operator_sum<HandlerTy> as a string.
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
  complex_matrix to_matrix(std::unordered_map<int, int> dimensions = {},
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {},
                     bool application_order = true) const;

  // unary operators

  operator_sum<HandlerTy> operator-() const &;
  operator_sum<HandlerTy> operator-() &&;
  operator_sum<HandlerTy> operator+() const &;
  operator_sum<HandlerTy> operator+() &&;

  // right-hand arithmetics

  operator_sum<HandlerTy> operator*(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator*(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator/(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator/(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) &&;
  operator_sum<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator+(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator-(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) &&;

  operator_sum<HandlerTy> &operator*=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator/=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator+=(scalar_operator &&other);
  operator_sum<HandlerTy> &operator+=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator-=(scalar_operator &&other);
  operator_sum<HandlerTy> &operator-=(const scalar_operator &other);
  operator_sum<HandlerTy> &operator*=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(product_operator<HandlerTy> &&other);
  operator_sum<HandlerTy> &operator-=(const product_operator<HandlerTy> &other);
  operator_sum<HandlerTy> &operator-=(product_operator<HandlerTy> &&other);
  operator_sum<HandlerTy> &operator*=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator+=(operator_sum<HandlerTy> &&other);
  operator_sum<HandlerTy> &operator-=(const operator_sum<HandlerTy> &other);
  operator_sum<HandlerTy> &operator-=(operator_sum<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   operator_sum<T> &&self);

  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   product_operator<T> &&self);

  // common operators

  template <typename T>
  friend operator_sum<T> operator_handler::empty();

  // handler specific operators

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static operator_sum<HandlerTy> empty();

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static product_operator<HandlerTy> i(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static product_operator<HandlerTy> x(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static product_operator<HandlerTy> y(int target);

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static product_operator<HandlerTy> z(int target);

  // general utility functions

  std::vector<operator_sum<HandlerTy>> distribute_terms(std::size_t numChunks) const;

  // utility functions for backward compatibility

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::vector<std::vector<bool>> _get_binary_symplectic_form() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  size_t num_qubits() const {
    return this->degrees().size();
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  static product_operator<HandlerTy> from_word(const std::string &word);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  static operator_sum<HandlerTy> random(std::size_t nQubits, std::size_t nTerms, unsigned int seed);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  operator_sum(const std::vector<double> &input_vec, std::size_t nQubits);

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::vector<double> getDataRepresentation() const {
    // FIXME: this is an imperfect representation because it does not capture targets accurately
    std::vector<double> dataVec;
    for(std::size_t i = 0; i < this->terms.size(); ++i) {
      for(std::size_t j = 0; j < this->terms[i].size(); ++j) {
        auto op_str = this->terms[i][j].to_string(false);
        // FIXME: align numbering with op codes
        // FIXME: compare to pauli instead
        if (op_str == "X")
          dataVec.push_back(1.);
        else if (op_str == "Z")
          dataVec.push_back(2.);
        else if (op_str == "Y")
          dataVec.push_back(3.);
        else
          dataVec.push_back(0.);
      }
      auto coeff = this->coefficients[i].evaluate(); // fails if we have params
      dataVec.push_back(coeff.real());
      dataVec.push_back(coeff.imag());
    }
    dataVec.push_back(this->terms.size());
    return dataVec;
  }
};

/// @brief Represents an operator expression consisting of a product of
/// elementary and scalar operators. Operator expressions cannot be used within
/// quantum kernels, but they provide methods to convert them to data types
/// that can.
template <typename HandlerTy>
class product_operator {
  template <typename T>
  friend class product_operator;
  template <typename T>
  friend class operator_sum;

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
                                 !product_operator<T>::supports_inplace_mult,
                             std::false_type> = std::false_type()>
  void insert(T &&other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, T>::value &&
                                 product_operator<T>::supports_inplace_mult,
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
  product_operator(scalar_operator coefficient, Args &&...args);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_operator(scalar_operator coefficient,
                   const std::vector<HandlerTy> &atomic_operators,
                   int size = 0);

  // keep this constructor protected (otherwise it needs to ensure canonical
  // order)
  product_operator(scalar_operator coefficient,
                   std::vector<HandlerTy> &&atomic_operators, int size = 0);

public:
  // iterator subclass
  struct const_iterator {
  private:
    const product_operator<HandlerTy> *prod;
    std::size_t current_idx;

  public:
    using iterator_category = std::bidirectional_iterator_tag;
    using difference_type   = std::ptrdiff_t;
    using value_type        = const HandlerTy;
    using pointer           = const HandlerTy*;
    using reference         = const HandlerTy&;

    const_iterator(const product_operator<HandlerTy> *prod, std::size_t idx = 0)
    : prod(prod), current_idx(idx) {}

    bool operator==(const const_iterator &other) const {
      return prod == other.prod && current_idx == other.current_idx;
    }

    bool operator!=(const const_iterator &other) const { return !(*this == other); }

    reference operator*() const { return prod->operators[current_idx]; }
    pointer operator->() { return &(prod->operators[current_idx]); }

    // prefix
    const_iterator& operator++() { ++current_idx; return *this; }  
    const_iterator& operator--() { --current_idx; return *this; }  

    // postfix
    const_iterator operator++(int) { return const_iterator(prod, current_idx++); }
    const_iterator operator--(int) { return const_iterator(prod, current_idx--); }
  };

  /// @brief Get iterator to beginning of operator terms
  const_iterator begin() const { return const_iterator(this); }

  /// @brief Get iterator to end of operator terms
  const_iterator end() const { return const_iterator(this, this->operators.size()); }

  // read-only properties

  /// @brief The degrees of freedom that the operator acts on.
  /// By default, degrees reflect the ordering convention (endianness) used in
  /// CUDA-Q, and the ordering of the matrix returned by default by `to_matrix`.
  ///
  /// Specifically, the indices of a statevector with two qubits are {00, 01, 10, 11}.
  /// An ordering of degrees {0, 1} then indicates that a state where
  /// the qubit with index 0 equals 1 with probability 1 is given by 
  /// the vector {0., 1., 0., 0.}.
  std::vector<std::size_t> degrees(bool application_order = true) const;

  /// @brief Return the number of operator terms that make up this product
  /// operator.
  std::size_t num_terms() const;

  // Public since it is used by the CUDA-Q compiler and runtime 
  // to retrieve expectation values for specific terms.
  // The term id uniquely identifies the operators and targets
  // (degrees) that they act on, but does not include information 
  // about the coefficient. 
  std::string get_term_id() const;

  scalar_operator get_coefficient() const;

  // constructors and destructors

  product_operator(double coefficient);

  product_operator(std::complex<double> coefficient);

  product_operator(HandlerTy &&atomic);

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_operator(const product_operator<T> &other);

  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_operator>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_operator(const product_operator<T> &other,
                   const matrix_operator::commutation_behavior &behavior);

  // copy constructor
  product_operator(const product_operator<HandlerTy> &other, int size = 0);

  // move constructor
  product_operator(product_operator<HandlerTy> &&other, int size = 0);

  ~product_operator() = default;

  // assignments

  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_operator<HandlerTy> &operator=(const product_operator<T> &other);

  // assignment operator
  product_operator<HandlerTy> &
  operator=(const product_operator<HandlerTy> &other);

  // move assignment operator
  product_operator<HandlerTy> &operator=(product_operator<HandlerTy> &&other);

  // evaluations

  /// @brief Return the `product_operator<HandlerTy>` as a string.
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
  complex_matrix to_matrix(std::unordered_map<int, int> dimensions = {},
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {},
                     bool application_order = true) const;

  // comparisons

  /// @brief True, if the other value is an operator_sum<HandlerTy> with
  /// equivalent terms,
  ///  and False otherwise. The equality takes into account that operator
  ///  addition is commutative, as is the product of two operators if they
  ///  act on different degrees of freedom.
  ///  The equality comparison does *not* take commutation relations into
  ///  account, and does not try to reorder terms `blockwise`; it may hence
  ///  evaluate to False, even if two operators in reality are the same.
  ///  If the equality evaluates to True, on the other hand, the operators
  ///  are guaranteed to represent the same transformation for all arguments.
  bool operator==(const product_operator<HandlerTy> &other) const;

  // unary operators

  product_operator<HandlerTy> operator-() const &;
  product_operator<HandlerTy> operator-() &&;
  product_operator<HandlerTy> operator+() const &;
  product_operator<HandlerTy> operator+() &&;

  // right-hand arithmetics

  product_operator<HandlerTy> operator*(scalar_operator &&other) const &;
  product_operator<HandlerTy> operator*(scalar_operator &&other) &&;
  product_operator<HandlerTy> operator*(const scalar_operator &other) const &;
  product_operator<HandlerTy> operator*(const scalar_operator &other) &&;
  product_operator<HandlerTy> operator/(scalar_operator &&other) const &;
  product_operator<HandlerTy> operator/(scalar_operator &&other) &&;
  product_operator<HandlerTy> operator/(const scalar_operator &other) const &;
  product_operator<HandlerTy> operator/(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator+(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator+(const scalar_operator &other) &&;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) const &;
  operator_sum<HandlerTy> operator-(scalar_operator &&other) &&;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) const &;
  operator_sum<HandlerTy> operator-(const scalar_operator &other) &&;
  product_operator<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) const &;
  product_operator<HandlerTy>
  operator*(const product_operator<HandlerTy> &other) &&;
  product_operator<HandlerTy>
  operator*(product_operator<HandlerTy> &&other) const &;
  product_operator<HandlerTy> operator*(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator+(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator+(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) const &;
  operator_sum<HandlerTy>
  operator-(const product_operator<HandlerTy> &other) &&;
  operator_sum<HandlerTy>
  operator-(product_operator<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(product_operator<HandlerTy> &&other) &&;
  operator_sum<HandlerTy> operator*(const operator_sum<HandlerTy> &other) const;
  operator_sum<HandlerTy>
  operator+(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator+(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator+(operator_sum<HandlerTy> &&other) &&;
  operator_sum<HandlerTy>
  operator-(const operator_sum<HandlerTy> &other) const &;
  operator_sum<HandlerTy> operator-(const operator_sum<HandlerTy> &other) &&;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) const &;
  operator_sum<HandlerTy> operator-(operator_sum<HandlerTy> &&other) &&;

  product_operator<HandlerTy> &operator*=(const scalar_operator &other);
  product_operator<HandlerTy> &operator/=(const scalar_operator &other);
  product_operator<HandlerTy> &
  operator*=(const product_operator<HandlerTy> &other);
  product_operator<HandlerTy> &operator*=(product_operator<HandlerTy> &&other);

  // left-hand arithmetics

  // Being a bit permissive here, since otherwise the explicit template
  // instantiation is a nightmare.
  template <typename T>
  friend product_operator<T> operator*(scalar_operator &&other,
                                       const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(scalar_operator &&other,
                                       product_operator<T> &&self);
  template <typename T>
  friend product_operator<T> operator*(const scalar_operator &other,
                                       const product_operator<T> &self);
  template <typename T>
  friend product_operator<T> operator*(const scalar_operator &other,
                                       product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   product_operator<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const product_operator<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   product_operator<T> &&self);

  template <typename T>
  friend operator_sum<T> operator*(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator*(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator+(const scalar_operator &other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(scalar_operator &&other,
                                   operator_sum<T> &&self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   const operator_sum<T> &self);
  template <typename T>
  friend operator_sum<T> operator-(const scalar_operator &other,
                                   operator_sum<T> &&self);

  // common operators

  // FIXME: remove
  template <typename T>
  friend product_operator<T> operator_handler::identity();
  template <typename T>
  friend product_operator<T> operator_handler::identity(int target);

  // handler specific operators

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  std::string get_pauli_word() const;

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static product_operator<HandlerTy> identity();

  HANDLER_SPECIFIC_TEMPLATE(spin_operator)
  static product_operator<HandlerTy> identity(int target);

  // utility functions for backward compatibility

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::vector<bool> get_binary_symplectic_form() const;

  SPIN_OPS_BACKWARD_COMPATIBILITY
  bool is_identity() const {
    // ignores the coefficients (according to the old behavior)
    for (const auto &op : this->operators)
      if (op.to_string(false) != "I") return false; // fixme: use pauli instead
    return true;
  }

  SPIN_OPS_BACKWARD_COMPATIBILITY
  std::string _to_string(bool printCoeffs) const {
    std::unordered_map<int, int> dims;
    auto degrees = this->degrees(false); // degrees in canonical order to match the evaluation
    auto evaluated = this->evaluate(
              operator_arithmetics<operator_handler::canonical_evaluation>(
                  dims, {})); // fails if operator is parameterized
    assert(evaluated.terms.size() == 1);
    auto term = std::move(evaluated.terms[0]);

    std::stringstream ss;
    if (printCoeffs) {
      auto coeff = term.first;
      ss << "[" << coeff.real() << (coeff.imag() < 0.0 ? "-" : "+") << std::fabs(coeff.imag()) << "j] ";
    }
    
    // For compatibility with existing code, the ordering for the term string always
    // needs to be from smallest to largest degree, and it necessarily must include 
    // all consecutive degrees starting from 0 (even if the operator doesn't act on them).
    if (degrees.size() > 0) {
      auto max_target = operator_handler::canonical_order(0, 1) ? degrees.back() : degrees[0];
      std::string term_str(max_target + 1, 'I');
      for (std::size_t i = 0; i < degrees.size(); ++i)
        term_str[degrees[i]] = term.second[i];
      ss << term_str;  
    }
    return ss.str();
  }
};

// type aliases for convenience
typedef std::unordered_map<std::string, std::complex<double>> parameter_map;
typedef std::unordered_map<int, int> dimension_map;
typedef operator_sum<spin_operator> spin_op;
typedef product_operator<spin_operator> spin_op_term;

#ifndef CUDAQ_INSTANTIATE_TEMPLATES
extern template class product_operator<matrix_operator>;
extern template class product_operator<spin_operator>;
extern template class product_operator<boson_operator>;
extern template class product_operator<fermion_operator>;

extern template class operator_sum<matrix_operator>;
extern template class operator_sum<spin_operator>;
extern template class operator_sum<boson_operator>;
extern template class operator_sum<fermion_operator>;
#endif

} // namespace cudaq
