/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
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

#define PROPERTY_SPECIFIC_TEMPLATE(property)                                   \
  template <typename T = HandlerTy,                                            \
            std::enable_if_t<std::is_same<HandlerTy, T>::value && property,    \
                             std::true_type> = std::true_type()>

#define PROPERTY_AGNOSTIC_TEMPLATE(property)                                   \
  template <typename T = HandlerTy,                                            \
            std::enable_if_t<std::is_same<HandlerTy, T>::value && !property,   \
                             std::false_type> = std::false_type()>

#define SPIN_OPS_BACKWARD_COMPATIBILITY(deprecation_message)                   \
  template <typename T = HandlerTy,                                            \
            std::enable_if_t<std::is_same<HandlerTy, spin_handler>::value &&   \
                                 std::is_same<HandlerTy, T>::value,            \
                             bool> = true>                                     \
  [[deprecated(deprecation_message)]]

/// @brief Represents a sum of operator products in a quantum operator algebra.
///
/// The sum_op class is a templated container that encapsulates a linear
/// combination of product operators, where each term is defined by a specific
/// configuration of operator components paired with a scalar coefficient.
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
  struct const_iterator {
  private:
    /// @brief A pointer to the sum_op instance whose terms are being iterated
    /// over.
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

    const_iterator(const sum_op<HandlerTy> *sum) : const_iterator(sum, 0) {}

    const_iterator(const sum_op<HandlerTy> *sum, std::size_t idx)
        : sum(sum), current_val(1.), current_idx(idx) {
      if (sum && current_idx < sum->num_terms())
        current_val = product_op<HandlerTy>(sum->coefficients[current_idx],
                                            sum->terms[current_idx]);
    }

    /// @brief Equality operator which compares iterators
    bool operator==(const const_iterator &other) const {
      return sum == other.sum && current_idx == other.current_idx;
    }

    /// @brief Non-equality operator which compares iterators
    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    /// @brief `Dereferences` the iterator to yield a reference to current_val,
    /// allowing access to the current product_op.
    product_op<HandlerTy> &operator*() { return current_val; }
    /// @brief Provides pointer access to the current product_op.
    product_op<HandlerTy> *operator->() { return &current_val; }

    /// @brief Advances the iterator to the next term in the term_map and
    /// updates current_val accordingly.
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

  /// @brief Operator to get the product term at a particular index.
  product_op<HandlerTy> operator[](std::size_t idx) const {
    if (idx >= this->num_terms())
      throw std::out_of_range("Index out of range in sum_op::operator[]");
    return product_op<HandlerTy>(this->coefficients[idx], this->terms[idx]);
  }

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

  /// @brief Constructs a new sum operator instance from product operator
  /// arguments.
  /// @tparam `Args` Variadic template parameters representing
  /// product_op<HandlerTy> types.
  /// @param args One or more product operator objects used in the summation
  /// operation.
  template <typename... Args,
            std::enable_if_t<std::conjunction<std::is_same<
                                 product_op<HandlerTy>, Args>...>::value &&
                                 sizeof...(Args),
                             bool> = true>
  sum_op(Args &&...args);

  /// @brief Constructs a sum_op instance from a given product_op instance.
  /// @param other A reference to the product_op object used to construct this
  /// sum_op.
  sum_op(const product_op<HandlerTy> &other);

  /// @brief Copy constructor for sum_op that enables conversion from a sum_op
  /// instantiated with a different type.
  /// @tparam T The type of the other sum_op object, which must not be HandlerTy
  /// and must be constructible to HandlerTy.
  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op(const sum_op<T> &other);

  /// @brief Constructs a new sum_op object from an existing sum_op of a
  /// different, constructible type.
  /// @param other The source sum_op object whose contents are used for
  /// construction.
  /// @param behavior The commutation behavior to be applied during
  /// construction.
  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_handler>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op(const sum_op<T> &other,
         const matrix_handler::commutation_behavior &behavior);

  /// @brief Copy constructor for sum_op.
  sum_op(const sum_op<HandlerTy> &other) = default;

  /// @brief Move constructor for sum_op.
  sum_op(sum_op<HandlerTy> &&other) = default;

  /// @brief Default destructor.
  ~sum_op() = default;

  // assignments

  /// @brief Assigns a product_op to a sum_op.
  /// This operator overload enables assignment from a product_op<T> to a
  /// sum_op<HandlerTy>. It is only enabled when T is not the same as HandlerTy
  /// and when HandlerTy is constructible from T. This constraint ensures that
  /// only compatible types are allowed in the assignment operation.
  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op<HandlerTy> &operator=(const product_op<T> &other);

  /// @brief Assign a product_op object to a sum_op object.
  /// @param other A constant reference to the product_op object to be assigned.
  /// @return A reference to the sum_op object after the assignment.
  sum_op<HandlerTy> &operator=(const product_op<HandlerTy> &other);

  /// @brief Assigns the contents of a product operation to the current sum
  /// operation using move semantics.
  /// @param other An rvalue reference to a product_op instance whose resources
  /// will be transferred.
  /// @return A reference to the updated sum_op instance.
  sum_op<HandlerTy> &operator=(product_op<HandlerTy> &&other);

  /// @brief Template assignment operator allowing conversion between different
  /// sum_op types.
  /// @tparam T The type of the sum_op object being assigned from.
  /// @param other The sum_op object with type T to be assigned.
  /// @return A reference to the current sum_op object after assignment.
  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  sum_op<HandlerTy> &operator=(const sum_op<T> &other);

  /// @brief Performs a copy assignment of one sum_op to another.
  /// @param other A constant reference to the sum_op object whose data is to be
  /// copied.
  /// @return A reference to the current sum_op object after the assignment.
  sum_op<HandlerTy> &operator=(const sum_op<HandlerTy> &other) = default;

  /// @brief Move assignment operator.
  /// @param other A rvalue reference to a sum_op object whose resources will be
  /// moved.
  /// @return A reference to this sum_op instance after the move.
  sum_op<HandlerTy> &operator=(sum_op<HandlerTy> &&other) = default;

  // evaluations

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by default by `degrees`.
  /// @param `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @param `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  /// @arg `invert_order`: if set to true, the ordering convention is reversed.
  complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> dimensions = {},
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

  /// @brief Returns a new sum_op instance representing the negation of this
  /// object.
  sum_op<HandlerTy> operator-() const &;
  /// @brief Applies the unary negation to a sum_op object using move semantics.
  sum_op<HandlerTy> operator-() &&;
  /// @brief Returns a new sum_op instance using the unary plus operator.
  /// @return A new instance of sum_op<HandlerTy> representing the unary plus
  /// operation.
  sum_op<HandlerTy> operator+() const &;
  /// @brief Applies the unary plus operator to a rvalue instance.
  sum_op<HandlerTy> operator+() &&;

  // right-hand arithmetics

  /// @brief Multiplies the current operator with a scalar operator.
  /// @param other The scalar_operator to multiply with.
  /// @return A new sum_op instance encapsulating the result of the
  /// multiplication.
  sum_op<HandlerTy> operator*(const scalar_operator &other) const &;
  /// @brief Overloads the multiplication operator to combine a rvalue sum
  /// operator with a scalar operator.
  /// @param other A constant reference to a scalar_operator that will be
  /// multiplied with the current rvalue sum operator.
  /// @return A new sum_op<HandlerTy> representing the resulting operator after
  /// applying the multiplication.
  sum_op<HandlerTy> operator*(const scalar_operator &other) &&;
  /// @brief Performs division of the current sum operator by a scalar operator.
  /// @param other The scalar operator serving as the divisor.
  /// @return A new sum operator resulting from the division operation.
  sum_op<HandlerTy> operator/(const scalar_operator &other) const &;
  /// @brief Overloaded division operator for a rvalue instance of the
  /// operator.
  /// @param other The scalar_operator divisor used in the division.
  /// @return A sum_op representing the result of the division operation.
  sum_op<HandlerTy> operator/(const scalar_operator &other) &&;
  /// @brief Combines the current operator with a scalar operator to form a new
  /// sum operator.
  /// @param other The scalar operator to be incorporated into the sum.
  /// @return A sum_op instance representing the combined operator.
  sum_op<HandlerTy> operator+(scalar_operator &&other) const &;
  /// @brief Combines the current sum_op with an additional scalar_operator.
  /// @param other An rvalue reference to a scalar_operator to be added.
  /// @return A new sum_op representing the sum of the current operator and the
  /// provided scalar_operator.
  sum_op<HandlerTy> operator+(scalar_operator &&other) &&;
  /// @brief Overloads the addition operator to combine the current sum operator
  /// with a scalar operator.
  /// @param other The scalar_operator instance to be added to the sum operator.
  /// @return A sum_op<HandlerTy> representing the result of combining the
  /// current operator with the specified scalar operator.
  sum_op<HandlerTy> operator+(const scalar_operator &other) const &;
  /// @brief Adds a scalar operator to the current operator instance.
  /// @param other The scalar operator to be added to this operator.
  /// @return A new sum_op<HandlerTy> representing the sum of the current
  /// operator and the provided scalar operator.
  sum_op<HandlerTy> operator+(const scalar_operator &other) &&;
  /// @brief Subtracts a scalar operator from the current operator
  /// instance.
  /// @param other An rvalue reference to a scalar_operator that will be
  /// subtracted from the current instance.
  /// @return A sum_op<HandlerTy> object that represents the result of the
  /// subtraction.
  sum_op<HandlerTy> operator-(scalar_operator &&other) const &;
  /// @brief Overloads the subtraction operator to subtract a scalar_operator
  /// from an rvalue instance.
  /// @param other A rvalue reference to a scalar_operator object to subtract.
  /// @return A sum_op object representing the result of the subtraction.
  sum_op<HandlerTy> operator-(scalar_operator &&other) &&;
  /// @brief Subtracts a scalar operator from the current operator.
  /// @param other The scalar_operator instance to subtract.
  /// @return A new sum_op object representing the result of the subtraction.
  sum_op<HandlerTy> operator-(const scalar_operator &other) const &;
  /// @brief Subtract a scalar operator from the current operator (rvalue
  /// context).
  /// @param other The scalar operator to subtract.
  /// @return sum_op<HandlerTy> The resultant operator after performing the
  /// subtraction.
  sum_op<HandlerTy> operator-(const scalar_operator &other) &&;
  /// @brief Multiplies this operator with a product operator, yielding a sum
  /// operator.
  /// @param other The product operator to be multiplied with this operator.
  /// @return A sum operator that represents the result of the multiplication.
  sum_op<HandlerTy> operator*(const product_op<HandlerTy> &other) const;
  /// @brief Adds a product operation to the current sum operation.
  /// @param other A `const` reference to the product operation to be added.
  /// @return A new sum operation that encapsulates the result of the addition.
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) const &;
  /// @brief Constructs a sum operation by adding a product operation to the
  /// current rvalue product operation.
  /// @param other A constant reference to a product_op instance to be added.
  /// @return A sum_op instance representing the result of the addition.
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) &&;
  /// @brief Overloads the addition operator to combine a product operation with
  /// the current sum operation.
  /// @param other A rvalue reference to the product operation to be added.
  /// @return A new sum operation representing the result of adding the provided
  /// product operation.
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) const &;
  /// @brief Overloads the addition operator to combine a product operation into
  /// a sum operation.
  /// @param other An rvalue reference to a product operation to be added.
  /// @return A sum operation resulting from the addition of the current
  /// instance and the given product operation.
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) &&;
  /// @brief Subtracts a product operator from the current sum operator.
  /// @param other A constant reference to the product operator to subtract.
  /// @return A new sum operator resulting from the subtraction.
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) const &;
  /// @brief Subtracts a product operation from a sum operation.
  /// @param other The product operation to subtract from the current sum
  /// operation.
  /// @return A new sum_op instance representing the result of the subtraction.
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) &&;
  /// @brief Subtracts a product operator from the current object to form a sum
  /// operator.
  /// @param other The product operator to subtract, provided as an rvalue to
  /// enable move semantics.
  /// @return A sum operator resulting from the subtraction operation.
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) const &;
  /// @brief Subtracts a product operator from a sum operator.
  /// @param other An rvalue reference to the product operator to be subtracted.
  /// @return A new sum operator representing the result of the subtraction.
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) &&;
  /// @brief Multiplies two sum_op objects.
  /// @param other The sum_op object to multiply with.
  /// @return A new sum_op object representing the result of the multiplication.
  sum_op<HandlerTy> operator*(const sum_op<HandlerTy> &other) const;
  /// @brief Computes the sum of the current and another sum_op instance.
  /// @param other A constant reference to another sum_op instance to be added.
  /// @return A new sum_op instance representing the result of the addition.
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) const &;
  /// @brief Combines the current sum_op with another sum_op.
  /// @param other A constant reference to another sum_op to be added.
  /// @return A new sum_op representing the result of the addition.
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) &&;
  /// @brief Adds the current sum_op with another sum_op object.
  /// @param other An rvalue reference to another sum_op object to be added.
  /// @return A new sum_op instance representing the sum of the two operands.
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) const &;
  /// @brief Overloads the addition operator for a sum_op object.
  /// @param other An rvalue reference to a sum_op object that is to be combined
  /// with the current object.
  /// @return A new sum_op object representing the result of the summation.
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) &&;
  /// @brief Subtracts the specified sum_op from this instance.
  /// @param other The sum_op to be subtracted from this instance.
  /// @return A new sum_op representing the result of the subtraction.
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) const &;
  /// @brief Subtracts another sum_op instance from the current rvalue sum_op.
  /// @param other A constant reference to another sum_op instance to subtract.
  /// @return A new sum_op instance resulting from the subtraction of other.
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) &&;
  /// @brief Subtracts the provided sum_op from this sum_op instance.
  /// @param other An rvalue reference to a sum_op object that will be
  /// subtracted from this instance.
  /// @return A new sum_op representing the result of the subtraction.
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) const &;
  /// @brief Subtracts another sum_op from this instance.
  /// @param other A rvalue sum_op instance to subtract.
  /// @return A new sum_op instance containing the difference.
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) &&;

  /// @brief Multiplies the current sum operator by a scalar operator.
  /// @param other The scalar operator to multiply with.
  /// @return A reference to the modified sum operator.
  sum_op<HandlerTy> &operator*=(const scalar_operator &other);
  /// @brief Performs an in-place division of this operator by the given scalar
  /// operator.
  /// @param other The scalar operator to divide by.
  /// @return A reference to this sum_op after the division.
  sum_op<HandlerTy> &operator/=(const scalar_operator &other);
  /// @brief Adds a scalar operator to the current sum operator.
  /// @param other The scalar operator to add, provided as an rvalue reference.
  /// @return A reference to the updated sum operator.
  sum_op<HandlerTy> &operator+=(scalar_operator &&other);
  /// @brief Adds a scalar operator to the current sum_op instance.
  /// @param other The scalar_operator to be added.
  /// @return A reference to the updated sum_op instance.
  sum_op<HandlerTy> &operator+=(const scalar_operator &other);
  /// @brief Subtracts a scalar operator from the current sum operator.
  /// @param other An rvalue reference to a scalar_operator that will be
  /// subtracted.
  /// @return A reference to the updated sum_op instance.
  sum_op<HandlerTy> &operator-=(scalar_operator &&other);
  /// @brief Subtracts a scalar operator from this sum operator.
  /// @param other The scalar operator to subtract from this instance.
  /// @return A reference to this sum operator after subtraction.
  sum_op<HandlerTy> &operator-=(const scalar_operator &other);
  /// @brief Updates the current sum operator by multiplying it with a given
  /// product operator.
  /// @param other A constant reference to the product operator to multiply
  /// with.
  /// @return A reference to the updated sum operator instance.
  sum_op<HandlerTy> &operator*=(const product_op<HandlerTy> &other);
  /// @brief Adds the specified product operation to the current sum operation.
  /// @param other The product operation to be incorporated into the sum
  /// operation.
  /// @return A reference to the updated sum operation.
  sum_op<HandlerTy> &operator+=(const product_op<HandlerTy> &other);
  /// @brief Adds the given product operation to the current sum operation.
  /// @param other The product operation to be added (provided as an rvalue
  /// reference).
  /// @return A reference to the updated sum operation.
  sum_op<HandlerTy> &operator+=(product_op<HandlerTy> &&other);
  /// @brief Subtracts the specified product operator from the current sum
  /// operator.
  /// @param other The product operator to subtract.
  /// @return A reference to the modified sum operator.
  sum_op<HandlerTy> &operator-=(const product_op<HandlerTy> &other);
  /// @brief Subtracts a product operator from this sum operator using move
  /// semantics.
  /// @param other The product operator to subtract (rvalue reference).
  /// @return A reference to the modified sum operator.
  sum_op<HandlerTy> &operator-=(product_op<HandlerTy> &&other);
  /// @brief Performs compound assignment multiplication with another sum_op
  /// object.
  /// @param other The sum_op instance to multiply with.
  /// @return A reference to the modified sum_op instance.
  sum_op<HandlerTy> &operator*=(const sum_op<HandlerTy> &other);
  /// @brief Adds the contents of the given sum_op instance to this sum_op.
  /// @param other The sum_op instance to be added.
  /// @return A reference to this sum_op after combining with the provided
  /// instance.
  sum_op<HandlerTy> &operator+=(const sum_op<HandlerTy> &other);
  /// @brief Adds the state of another sum_op object into the current instance
  /// using move semantics.
  /// @param other An rvalue reference to a sum_op object whose state is to be
  /// merged into the current instance.
  /// @return A reference to the modified sum_op object.
  sum_op<HandlerTy> &operator+=(sum_op<HandlerTy> &&other);
  /// @brief Subtracts the provided sum operator from this instance.
  /// @param other A constant reference to the sum operator to subtract.
  /// @return A reference to this sum operator after subtraction.
  sum_op<HandlerTy> &operator-=(const sum_op<HandlerTy> &other);
  /// @brief Subtracts the contents of the provided sum_op from this sum_op.
  /// @param other An rvalue reference to the sum_op whose contents will be
  /// subtracted from this sum_op.
  /// @return A reference to the current sum_op after subtraction.
  sum_op<HandlerTy> &operator-=(sum_op<HandlerTy> &&other);

  // left-hand arithmetics

  /// @brief Multiplies a scalar_operator with a sum_op.
  /// @param other The scalar_operator to multiply.
  /// @param self The sum_op to be multiplied.
  /// @return A new sum_op that is the result of the multiplication.
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other,
                             const sum_op<T> &self);
  /// @brief Multiplies a scalar operator with a sum operator.
  /// @param other The scalar operator to multiply.
  /// @param self The sum operator to be multiplied.
  /// @return A new sum operator that is the result of the multiplication.
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other, sum_op<T> &&self);
  /// @brief Overloads the + operator to add a scalar_operator to a sum_op.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op to which the scalar_operator is added.
  /// @return A new sum_op that is the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, const sum_op<T> &self);
  /// @brief Overloads the addition operator for combining a scalar_operator
  /// with a sum_op.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op to which the scalar_operator is added.
  /// @return A new sum_op that is the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, sum_op<T> &&self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// sum_op.
  /// @tparam T The type of the elements in the sum_op.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op to which the scalar_operator is added.
  /// @return A new sum_op that is the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const sum_op<T> &self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// sum_op object.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op object to which the scalar_operator is added.
  /// @return A new sum_op object that is the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other, sum_op<T> &&self);
  /// @brief Overloads the subtraction operator for a scalar_operator and a
  /// sum_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The sum_op from which the scalar_operator is subtracted.
  /// @return A new sum_op<T> representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, const sum_op<T> &self);

  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, sum_op<T> &&self);

  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const sum_op<T> &self);

  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other, sum_op<T> &&self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// sum_op.
  /// @tparam T The type of the elements in the sum_op.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op to which the scalar_operator is added.
  /// @return A new sum_op that is the result of adding the scalar_operator to
  /// the sum_op.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other,
                             const product_op<T> &self);
  /// @brief Overloads the + operator to add a scalar_operator and a product_op.
  /// @param other The scalar_operator to be added.
  /// @param self The product_op to be added.
  /// @return A sum_op object representing the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, product_op<T> &&self);
  /// @brief Overloads the addition operator to add a scalar_operator and a
  /// product_op.
  /// @param other The scalar_operator to be added.
  /// @param self The product_op to be added.
  /// @return A sum_op object representing the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const product_op<T> &self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// product_op.
  /// @tparam T The type of the product_op.
  /// @param other The scalar_operator to be added.
  /// @param self The product_op to which the scalar_operator is added.
  /// @return A sum_op object representing the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             product_op<T> &&self);
  /// @brief Overloads the subtraction operator for scalar_operator and
  /// product_op.
  /// @param other A scalar_operator object to be subtracted.
  /// @param self A constant reference to a product_op object.
  /// @return A sum_op object representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other,
                             const product_op<T> &self);
  /// @brief Overloads the subtraction operator for a scalar_operator and a
  /// product_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The product_op from which the scalar_operator is subtracted.
  /// @return A sum_op object representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, product_op<T> &&self);
  /// @brief Subtracts a product operator from a scalar operator.
  /// @tparam T The type of the product operator.
  /// @param other The scalar operator to subtract from.
  /// @param self The product operator to be subtracted.
  /// @return The result of the subtraction as a sum_op<T> object.
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const product_op<T> &self);
  /// @brief Overloads the subtraction operator for a scalar_operator and a
  /// product_op.
  /// @param other The scalar_operator on the left-hand side of the subtraction.
  /// @param self The product_op on the right-hand side of the subtraction,
  /// passed as an rvalue reference.
  /// @return A sum_op<T> representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             product_op<T> &&self);

  // common operators

  /// @brief Creates and returns an empty sum operator.
  /// @return An empty sum operator of type sum_op<HandlerTy>.
  static sum_op<HandlerTy> empty();
  /// @brief Creates and returns an identity product operator.
  /// @return A product_op<HandlerTy> representing the identity operator.
  static product_op<HandlerTy> identity();
  /// @brief Creates an identity product operator for the specified target.
  /// @param target The target index for which the identity operator is created.
  /// @return A product_op object representing the identity operator for the
  /// given target.
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

  /// @brief Distributes the terms into a specified number of chunks.
  /// @param numChunks The number of chunks to distribute the terms into.
  /// @return A vector of sum_op<HandlerTy> representing the distributed terms.
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
  PROPERTY_SPECIFIC_TEMPLATE(product_op<T>::supports_inplace_mult)
  csr_spmatrix to_sparse_matrix(
      std::unordered_map<std::size_t, std::int64_t> dimensions = {},
      const std::unordered_map<std::string, std::complex<double>> &parameters =
          {},
      bool invert_order = false) const;

  /// @brief Return the multi-diagonal matrix representation of the operator.
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
  PROPERTY_SPECIFIC_TEMPLATE(product_op<T>::supports_inplace_mult)
  mdiag_sparse_matrix to_diagonal_matrix(
      std::unordered_map<std::size_t, std::int64_t> dimensions = {},
      const std::unordered_map<std::string, std::complex<double>> &parameters =
          {},
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

  PROPERTY_AGNOSTIC_TEMPLATE(product_op<T>::supports_inplace_mult)
  void insert(T &&other);

  PROPERTY_SPECIFIC_TEMPLATE(product_op<T>::supports_inplace_mult)
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

    /// @brief Constructs a const_iterator for a given product operator.
    /// @param prod Pointer to the product operator containing the operators.
    /// @param idx Starting index for the iterator (default is 0).
    const_iterator(const product_op<HandlerTy> *prod, std::size_t idx = 0)
        : prod(prod), current_idx(idx) {}

    /// @brief Compares this iterator with another for equality.
    /// @param other Another const_iterator to compare with.
    /// @return True if both iterators refer to the same product operator and
    /// current index, false otherwise.
    bool operator==(const const_iterator &other) const {
      return prod == other.prod && current_idx == other.current_idx;
    }

    /// @brief Compares this iterator with another for inequality.
    /// @param other Another const_iterator to compare with.
    /// @return True if the iterators do not refer to the same product operator
    /// or index.
    bool operator!=(const const_iterator &other) const {
      return !(*this == other);
    }

    /// @brief `Dereferences` the iterator to access the current operator.
    /// @return A constant reference to the current operator.
    const HandlerTy &operator*() const { return prod->operators[current_idx]; }

    /// @brief Provides pointer-like access to the current operator.
    /// @return A pointer to the current operator.
    const HandlerTy *operator->() { return &(prod->operators[current_idx]); }

    /// @brief Advances the iterator to the next operator (prefix increment).
    /// @return Reference to the updated iterator.
    const_iterator &operator++() {
      ++current_idx;
      return *this;
    }

    /// @brief Moves the iterator to the previous operator (prefix decrement).
    /// @return Reference to the updated iterator.
    const_iterator &operator--() {
      --current_idx;
      return *this;
    }

    /// @brief Advances the iterator (`postfix` increment) and returns the
    /// iterator state before increment.
    /// @return A const_iterator representing the state prior to increment.
    const_iterator operator++(int) {
      return const_iterator(prod, current_idx++);
    }

    /// @brief Moves the iterator (`postfix` decrement) and returns the iterator
    /// state before decrement.
    /// @return A const_iterator representing the state prior to decrement.
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

  /// @brief Operator to get the operator at a particular index.
  HandlerTy operator[](std::size_t idx) const {
    if (idx >= this->operators.size())
      throw std::out_of_range("Index out of range in product_op::operator[]");
    return this->operators[idx];
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
  /// @brief The term id uniquely identifies the operators and targets
  /// (degrees) that they act on, but does not include information
  /// about the coefficient.
  std::string get_term_id() const;

  /// @brief Retrieves the coefficient associated with this operator instance.
  /// @return A scalar_operator representing the operator's coefficient.
  scalar_operator get_coefficient() const;

  std::unordered_map<std::string, std::string>
  get_parameter_descriptions() const;

  // constructors and destructors

  /// @brief Default constructor for the product_op class.
  constexpr product_op() {}

  /// @brief Constructor instantiates a product that applies an identity
  /// operator to all targets in the open range [first_degree, last_degree).
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

  /// @brief Constructs a product operator with the given coefficient.
  /// @param coefficient A double representing the scaling factor for the
  /// product operator.
  product_op(double coefficient);

  /// @brief Constructs a product operator with the given coefficient.
  /// @param coefficient A complex of double representing the scaling factor for
  /// the product operator.
  product_op(std::complex<double> coefficient);

  /// @brief Constructs a product operator from a given atomic operator handler.
  /// @tparam HandlerTy The type of the underlying atomic operator handler.
  /// @param atomic An rvalue reference to the atomic operator handler.
  product_op(HandlerTy &&atomic);

  /// @brief Constructs a product_op from another product_op instance.
  /// This constructor is enabled only if T is not equivalent to HandlerTy and
  /// if HandlerTy can be constructed from T. It allows implicit conversion
  /// between different instantiations of product_op.
  /// @param other The product_op instance to copy from.
  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op(const product_op<T> &other);

  /// @brief Constructs a product operator from an existing product operator
  /// with a different type. This constructor enables the creation of a new
  /// product_op object when the HandlerTy is a matrix_handler, provided that
  /// the type T is not the same as HandlerTy but is convertible to HandlerTy.
  /// It allows for proper handling of commutation behavior during the
  /// construction.
  /// @tparam T The type of the operand from which the new product_op is
  /// constructed.
  /// @param other The source product_op instance from which to create this new
  /// object.
  /// @param behavior The commutation behavior to be used with the
  /// matrix_handler.
  template <typename T,
            std::enable_if_t<std::is_same<HandlerTy, matrix_handler>::value &&
                                 !std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op(const product_op<T> &other,
             const matrix_handler::commutation_behavior &behavior);

  /// @brief Constructs a new product_op by copying an existing product_op
  /// instance.
  /// @param other The product_op instance to be copied.
  /// @param size An optional parameter to specify how many operator elements to
  /// reserve space for.
  product_op(const product_op<HandlerTy> &other, std::size_t size = 0);

  /// @brief Constructs a product_op by moving the resources from an existing
  /// product_op instance.
  /// @param other An rvalue reference to the product_op to move from.
  /// @param size An optional size parameter indicating how many operator
  /// elements to reserve space for. to adjust or specify internal dimensions.
  product_op(product_op<HandlerTy> &&other, std::size_t size = 0);

  /// @brief Default destructor for product_op.
  /// @details The explicit default destructor allowing the compiler for proper
  /// resource cleanup.
  ~product_op() = default;

  // assignments

  /// @brief Templated assignment operator that enables assigning from a
  /// product_op with a different but convertible type. This operator is only
  /// enabled when the template parameter T is not the same as HandlerTy and is
  /// constructible as a HandlerTy. It allows for copying or converting a
  /// product_op instance of type T into one of type HandlerTy.
  /// @tparam T The type of the product_op to be assigned from, which must
  /// satisfy that it is not HandlerTy and is constructible as HandlerTy.
  template <typename T,
            std::enable_if_t<!std::is_same<T, HandlerTy>::value &&
                                 std::is_constructible<HandlerTy, T>::value,
                             bool> = true>
  product_op<HandlerTy> &operator=(const product_op<T> &other);

  /// @brief Assignment operator for the product_op class.
  /// @param other The product_op instance to be assigned.
  /// @return A reference to the assigned product_op instance.
  product_op<HandlerTy> &
  operator=(const product_op<HandlerTy> &other) = default;

  /// @brief Move assignment operator for product_op.
  /// @param other The product_op instance to move from.
  /// @return A reference to the assigned product_op instance.
  product_op<HandlerTy> &operator=(product_op<HandlerTy> &&other) = default;

  // evaluations

  std::complex<double> evaluate_coefficient(
      const std::unordered_map<std::string, std::complex<double>> &parameters =
          {}) const;

  /// @brief Return the matrix representation of the operator.
  /// By default, the matrix is ordered according to the convention (endianness)
  /// used in CUDA-Q, and the ordering returned by default by `degrees`.
  /// @param  `dimensions` : A mapping that specifies the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0:2, 1:2}`.
  /// @param `parameters` : A map of the parameter names to their concrete,
  /// complex values.
  /// @arg `invert_order`: if set to true, the ordering convention is reversed.
  complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> dimensions = {},
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

  /// @brief Unary negation operator for the product_op class.
  /// @return A new product_op object with the negated value.
  product_op<HandlerTy> operator-() const &;
  /// @brief Unary negation operator for product_op.
  /// @return A new product_op object with the negated value.
  product_op<HandlerTy> operator-() &&;
  /// @brief Unary plus operator for the product_op class.
  /// @return A new product_op object.
  product_op<HandlerTy> operator+() const &;
  /// @brief Overloaded unary plus operator for product_op.
  /// @return A product_op object with the current handler type.
  product_op<HandlerTy> operator+() &&;

  // right-hand arithmetics

  /// @brief Overloaded multiplication operator for combining a scalar operator
  /// with a product operator.
  /// @param other The scalar operator to be multiplied.
  /// @return A new product operator resulting from the multiplication.
  product_op<HandlerTy> operator*(scalar_operator &&other) const &;
  /// @brief Multiplies the current operator with a scalar operator to yield a
  /// product operator.
  /// @param other An rvalue reference to the scalar_operator that is multiplied
  /// with the current operator.
  /// @return A product_op containing the resulting operator after
  /// multiplication.
  product_op<HandlerTy> operator*(scalar_operator &&other) &&;
  /// @brief Multiplies this operator with a scalar operator, yielding a product
  /// operator.
  /// @param other The scalar operator to multiply with.
  /// @return A product_op representing the product of the current operator and
  /// the provided scalar_operator.
  product_op<HandlerTy> operator*(const scalar_operator &other) const &;
  /// @brief Multiplies the current operator (as an rvalue) with a scalar
  /// operator.
  /// @param other The scalar operator to multiply with.
  /// @return A product_op<HandlerTy> representing the product of the
  /// multiplication.
  product_op<HandlerTy> operator*(const scalar_operator &other) &&;
  /// @brief Divides the current operator by a scalar operator.
  /// @param other A scalar operator to be divided from the current operator.
  /// @return A product operator representing the result of the division
  /// operation.
  product_op<HandlerTy> operator/(scalar_operator &&other) const &;
  /// @brief Divides the current rvalue operator by a scalar operator.
  /// @param other An rvalue reference to the scalar operator to be used in the
  /// division.
  /// @return A product_op representing the result of the division.
  product_op<HandlerTy> operator/(scalar_operator &&other) &&;
  /// @brief Divides this operator by a scalar operator, yielding a product
  /// operator.
  /// @param other The scalar operator used as the divisor.
  /// @return A product_op representing the result of the division.
  product_op<HandlerTy> operator/(const scalar_operator &other) const &;
  /// @brief Divides an rvalue product operator by a scalar operator.
  /// @param other The scalar operator by which the product operator is divided.
  /// @return A new product operator representing the result of the division.
  product_op<HandlerTy> operator/(const scalar_operator &other) &&;
  /// @brief Adds a scalar operator to the current operator, yielding a new
  /// aggregated sum operator.
  /// @param other An rvalue reference to a scalar operator which is to be
  /// added.
  /// @return A new sum_op instance representing the combined effect of the
  /// current operator and the provided scalar operator.
  sum_op<HandlerTy> operator+(scalar_operator &&other) const &;
  /// @brief Combines the current operator with a provided scalar operator.
  /// @param other An rvalue reference to a scalar_operator to be added.
  /// @return A sum_op<HandlerTy> representing the resulting sum of the
  /// operators.
  sum_op<HandlerTy> operator+(scalar_operator &&other) &&;
  /// @brief Adds the current scalar_operator with the given scalar_operator,
  /// yielding a new sum_op instance that encapsulates the resulting operation.
  /// @param other The scalar_operator instance to be added.
  /// @return A sum_op instance representing the sum of the two operators.
  sum_op<HandlerTy> operator+(const scalar_operator &other) const &;
  /// @brief Overloads the binary + operator to add a scalar operator to a
  /// temporary operator instance.
  /// @param other The scalar_operator instance to be added.
  /// @return A sum_op object representing the combined operator after
  /// performing the addition.
  sum_op<HandlerTy> operator+(const scalar_operator &other) &&;
  /// @brief Subtracts a scalar operator from this instance (l-value reference).
  /// @param other R-value reference to scalar_operator.
  /// @return A sum_op instance representing the subtraction.
  sum_op<HandlerTy> operator-(scalar_operator &&other) const &;
  /// @brief Subtracts a scalar operator from this instance (r-value reference).
  /// @param other R-value reference to scalar_operator.
  /// @return A sum_op instance representing the subtraction.
  sum_op<HandlerTy> operator-(scalar_operator &&other) &&;
  /// @brief Subtracts a scalar operator (by `const` reference) from this
  /// instance (l-value reference).
  /// @param other Constant reference to scalar_operator.
  /// @return A sum_op instance representing the subtraction.
  sum_op<HandlerTy> operator-(const scalar_operator &other) const &;
  /// @brief Subtracts a scalar operator (by `const` reference) from this
  /// instance (r-value reference).
  /// @param other Constant reference to scalar_operator.
  /// @return A sum_op instance representing the subtraction.
  sum_op<HandlerTy> operator-(const scalar_operator &other) &&;
  /// @brief Multiplies this instance (l-value reference) by another product_op
  /// instance.
  /// @param other Constant reference to another product_op.
  /// @return A product_op representing the multiplication.
  product_op<HandlerTy> operator*(const product_op<HandlerTy> &other) const &;
  /// @brief Multiplies this instance (r-value reference) by another product_op
  /// instance.
  /// @param other Constant reference to another product_op.
  /// @return A product_op representing the multiplication.
  product_op<HandlerTy> operator*(const product_op<HandlerTy> &other) &&;
  /// @brief Multiplies this instance (l-value reference) by another product_op
  /// instance (r-value).
  /// @param other R-value reference to another product_op.
  /// @return A product_op representing the multiplication.
  product_op<HandlerTy> operator*(product_op<HandlerTy> &&other) const &;
  /// @brief Multiplies this instance (r-value reference) by another product_op
  /// instance (r-value).
  /// @param other R-value reference to another product_op.
  /// @return A product_op representing the multiplication.
  product_op<HandlerTy> operator*(product_op<HandlerTy> &&other) &&;
  /// @brief Adds a product_op (l-value) to this instance (l-value).
  /// @param other Constant reference to product_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) const &;
  /// @brief Adds a product_op (l-value) to this instance (r-value).
  /// @param other Constant reference to product_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(const product_op<HandlerTy> &other) &&;
  /// @brief Adds a product_op (r-value) to this instance (l-value).
  /// @param other R-value reference to product_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) const &;
  /// @brief Adds a product_op (r-value) to this instance (r-value).
  /// @param other R-value reference to product_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(product_op<HandlerTy> &&other) &&;
  /// @brief Subtracts a product_op (l-value) from this instance (l-value).
  /// @param other Constant reference to product_op.
  /// @return A sum_op representing the subtraction.
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) const &;
  /// @brief Subtracts a product_op (l-value) from this instance (r-value).
  /// @param other Constant reference to product_op.
  /// @return A sum_op representing the subtraction.
  sum_op<HandlerTy> operator-(const product_op<HandlerTy> &other) &&;
  /// @brief Subtracts a product_op (r-value) from this instance (l-value).
  /// @param other R-value reference to product_op.
  /// @return A sum_op representing the subtraction.
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) const &;
  /// @brief Subtracts a product_op (r-value) from this instance (r-value).
  /// @param other R-value reference to product_op.
  /// @return A sum_op representing the subtraction.
  sum_op<HandlerTy> operator-(product_op<HandlerTy> &&other) &&;
  /// @brief Multiplies this sum_op instance with another sum_op.
  /// @param other Constant reference to sum_op.
  /// @return A sum_op representing the multiplication result.
  sum_op<HandlerTy> operator*(const sum_op<HandlerTy> &other) const;
  /// @brief Adds a sum_op (l-value) to this instance (l-value).
  /// @param other Constant reference to sum_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) const &;
  /// @brief Adds a sum_op (l-value) to this instance (r-value).
  /// @param other Constant reference to sum_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(const sum_op<HandlerTy> &other) &&;
  /// @brief Adds a sum_op (r-value) to this instance (l-value).
  /// @param other R-value reference to sum_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) const &;
  /// @brief Adds a sum_op (r-value) to this instance (r-value).
  /// @param other R-value reference to sum_op.
  /// @return A sum_op representing the addition.
  sum_op<HandlerTy> operator+(sum_op<HandlerTy> &&other) &&;
  /// @brief Subtracts a sum_op (l-value) from this instance (l-value).
  /// @param other Constant reference to sum_op.
  /// @return A sum_op representing the subtraction.
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) const &;
  /// @brief Subtracts a sum_op (l-value) from this instance (r-value).
  /// @param other Constant reference to sum_op.
  /// @return A sum_op representing the subtraction.
  sum_op<HandlerTy> operator-(const sum_op<HandlerTy> &other) &&;
  /// @brief Subtracts another sum_op object from this one.
  /// @param other The sum_op object to subtract.
  /// @return A new sum_op object that is the result of the subtraction.
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) const &;
  /// @brief Subtracts another sum_op object from this sum_op object.
  /// @param other The sum_op object to be subtracted.
  /// @return A new sum_op object that is the result of the subtraction.
  sum_op<HandlerTy> operator-(sum_op<HandlerTy> &&other) &&;

  /// @brief Multiplies the current product operator by a scalar operator.
  /// @param other The scalar operator to multiply with.
  /// @return A reference to the modified product operator.
  product_op<HandlerTy> &operator*=(const scalar_operator &other);
  /// @brief Overloads the division assignment operator to divide the current
  /// product operator by a scalar operator.
  /// @param other The scalar operator to divide by.
  /// @return A reference to the modified product operator.
  product_op<HandlerTy> &operator/=(const scalar_operator &other);
  /// @brief Compound assignment operator that multiplies this product_op with
  /// another product_op.
  /// @param other The product_op to multiply with.
  /// @return A reference to the modified product_op.
  product_op<HandlerTy> &operator*=(const product_op<HandlerTy> &other);
  /// @brief Multiplies this product operator with another product operator and
  /// assigns the result to this operator.
  /// @param other The product operator to multiply with.
  /// @return A reference to the modified product operator.
  product_op<HandlerTy> &operator*=(product_op<HandlerTy> &&other);

  // left-hand arithmetics

  /// @brief Overloads the multiplication operator to multiply a scalar operator
  /// with a product operator. This function enables left-sided multiplication
  /// of a scalar operator (provided as an rvalue reference) with an existing
  /// product operator, producing a new product operator that encapsulates the
  /// resultant state.
  /// @tparam T The type parameter used within the product operator.
  /// @param other The scalar operator to be multiplied (rvalue reference).
  /// @param self The product operator to be multiplied.
  /// @return A new product operator that is the result of multiplying the
  /// scalar operator with the product operator.
  template <typename T>
  friend product_op<T> operator*(scalar_operator &&other,
                                 const product_op<T> &self);
  /// @brief Overloads the multiplication operator to multiply a scalar_operator
  /// with a product_op.
  /// @tparam T The type of the elements in the product_op.
  /// @param other The scalar_operator to be multiplied.
  /// @param self The product_op to be multiplied.
  /// @return A new product_op resulting from the multiplication of the
  /// scalar_operator and the product_op.
  template <typename T>
  friend product_op<T> operator*(scalar_operator &&other, product_op<T> &&self);
  /// @brief Overloads the multiplication operator to allow multiplication
  /// between a scalar_operator and a product_op<T>.
  /// @param other The scalar_operator to be multiplied.
  /// @param self The product_op<T> to be multiplied.
  /// @return A new product_op<T> resulting from the multiplication of
  /// the scalar_operator and the product_op<T>.
  template <typename T>
  friend product_op<T> operator*(const scalar_operator &other,
                                 const product_op<T> &self);
  /// @brief Overloaded multiplication operator for combining a scalar operator
  /// with a product operator.
  /// @param other The scalar operator to be multiplied.
  /// @param self The product operator to be multiplied, passed as an rvalue
  /// reference.
  /// @return A new product operator resulting from the multiplication of the
  /// scalar operator and the product operator.
  template <typename T>
  friend product_op<T> operator*(const scalar_operator &other,
                                 product_op<T> &&self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// product_op.
  /// @param other The scalar_operator to be added.
  /// @param self The product_op to which the scalar_operator is added.
  /// @return A sum_op object representing the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other,
                             const product_op<T> &self);
  /// @brief Overloads the addition operator to add a scalar_operator and a
  /// product_op.
  /// @param other The scalar_operator to be added.
  /// @param self The product_op to be added.
  /// @return A sum_op object representing the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, product_op<T> &&self);
  /// @brief Overloads the addition operator to combine a scalar operator with a
  /// product operator.
  /// @param other The scalar operator to be added.
  /// @param self The product operator to be added.
  /// @return A sum operator representing the combined result of the scalar and
  /// product operators.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const product_op<T> &self);
  /// @brief Overloads the + operator to add a scalar_operator to a product_op.
  /// @tparam T The type of the elements in the product_op.
  /// @param other The scalar_operator to be added.
  /// @param self The product_op to which the scalar_operator is added.
  /// @return A sum_op<T> representing the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             product_op<T> &&self);
  /// @brief Overloads the subtraction operator to subtract a product_op object
  /// from a scalar_operator object.
  /// @tparam T The type of the elements in the product_op.
  /// @param other The scalar_operator object to be subtracted from.
  /// @param self The product_op object to subtract.
  /// @return A sum_op object representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other,
                             const product_op<T> &self);
  /// @brief Overloads the subtraction operator for a scalar_operator and a
  /// product_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The product_op from which the scalar_operator is subtracted.
  /// @return A sum_op<T> representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, product_op<T> &&self);
  /// @brief Overloads the subtraction operator to subtract a product operator
  /// from a scalar operator.
  /// @param other The scalar operator to subtract from.
  /// @param self The product operator to be subtracted.
  /// @return A sum_op object representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const product_op<T> &self);
  /// @brief Overloads the subtraction operator to subtract a scalar_operator
  /// from a product_op.
  /// @tparam T The type of the elements in the product_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The product_op from which the scalar_operator is subtracted.
  /// @return A sum_op<T> representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             product_op<T> &&self);
  /// @brief Overloaded multiplication operator for scalar_operator and sum_op.
  /// @param other The scalar_operator to be multiplied.
  /// @param self The sum_op to be multiplied.
  /// @return A new sum_op resulting from the multiplication of the
  /// scalar_operator and the sum_op.
  template <typename T>
  friend sum_op<T> operator*(scalar_operator &&other, const sum_op<T> &self);
  /// @brief Overloaded multiplication operator for combining a scalar operator
  /// with a sum operator.
  /// @param other The scalar operator to be multiplied.
  /// @param self The sum operator to be multiplied.
  /// @return A new sum_op<T> resulting from the multiplication of the scalar
  /// operator and the sum operator.
  template <typename T>
  friend sum_op<T> operator*(scalar_operator &&other, sum_op<T> &&self);
  /// @brief Overloads the multiplication operator to allow multiplication of a
  /// scalar_operator with a sum_op.
  /// @param other The scalar_operator to be multiplied.
  /// @param self The sum_op to be multiplied.
  /// @return A new sum_op resulting from the multiplication of the
  /// scalar_operator and the sum_op.
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other,
                             const sum_op<T> &self);
  /// @brief Multiplies a scalar_operator with a sum_op object.
  /// @param other The scalar_operator to multiply.
  /// @param self The sum_op object to be multiplied.
  /// @return A new sum_op object that is the result of the multiplication.
  template <typename T>
  friend sum_op<T> operator*(const scalar_operator &other, sum_op<T> &&self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// sum_op object.
  /// @param other The scalar_operator object to be added.
  /// @param self The sum_op object to which the scalar_operator is added.
  /// @return A new sum_op object that represents the sum of the scalar_operator
  /// and the sum_op.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, const sum_op<T> &self);
  /// @brief Overloads the addition operator for combining a scalar_operator and
  /// a sum_op.
  /// @tparam T The type of the elements in the sum_op.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op to which the scalar_operator will be added.
  /// @return A new sum_op<T> that is the result of adding the scalar_operator
  /// to the sum_op.
  template <typename T>
  friend sum_op<T> operator+(scalar_operator &&other, sum_op<T> &&self);
  /// @brief Overloads the addition operator to allow adding a scalar_operator
  /// to a sum_op object.
  /// @tparam T The type of the elements in the sum_op.
  /// @param other The scalar_operator object to be added.
  /// @param self The sum_op object to which the scalar_operator is added.
  /// @return A new sum_op object that is the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other,
                             const sum_op<T> &self);
  /// @brief Overloads the addition operator to add a scalar_operator to a
  /// sum_op.
  /// @tparam T The type of the elements in the sum_op.
  /// @param other The scalar_operator to be added.
  /// @param self The sum_op to which the scalar_operator is added.
  /// @return A new sum_op that is the result of the addition.
  template <typename T>
  friend sum_op<T> operator+(const scalar_operator &other, sum_op<T> &&self);
  /// @brief Subtracts a sum_op object from a scalar_operator object.
  /// @param other The scalar_operator object to be subtracted.
  /// @param self The sum_op object from which the scalar_operator is
  /// subtracted.
  /// @return A new sum_op object representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, const sum_op<T> &self);
  /// @brief Overloads the subtraction operator for a scalar_operator and a
  /// sum_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The sum_op from which the scalar_operator is subtracted.
  /// @return A new sum_op resulting from the subtraction.
  template <typename T>
  friend sum_op<T> operator-(scalar_operator &&other, sum_op<T> &&self);
  /// @brief Overloads the subtraction operator to subtract a scalar_operator
  /// from a sum_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The sum_op from which the scalar_operator is subtracted.
  /// @return A new sum_op representing the result of the subtraction.
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other,
                             const sum_op<T> &self);
  /// @brief Overloads the subtraction operator for scalar_operator and sum_op.
  /// @param other The scalar_operator to be subtracted.
  /// @param self The sum_op object from which the scalar_operator is
  /// subtracted.
  /// @return A new sum_op object resulting from the subtraction.
  template <typename T>
  friend sum_op<T> operator-(const scalar_operator &other, sum_op<T> &&self);

  // general utility functions

  /// @brief Checks if all operators in the product are the identity.
  /// Note: this function returns true regardless of the value
  /// of the coefficient.
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
  PROPERTY_SPECIFIC_TEMPLATE(product_op<T>::supports_inplace_mult)
  csr_spmatrix to_sparse_matrix(
      std::unordered_map<std::size_t, std::int64_t> dimensions = {},
      const std::unordered_map<std::string, std::complex<double>> &parameters =
          {},
      bool invert_order = false) const;

  /// @brief Return the multi-diagonal matrix representation of the operator.
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
  PROPERTY_SPECIFIC_TEMPLATE(product_op<T>::supports_inplace_mult)
  mdiag_sparse_matrix to_diagonal_matrix(
      std::unordered_map<std::size_t, std::int64_t> dimensions = {},
      const std::unordered_map<std::string, std::complex<double>> &parameters =
          {},
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

// https://en.wikipedia.org/wiki/Superoperator
// 'super_op' is a linear operator acting on a vector space of linear
// operators.
/// @brief Representation of generic operator action on the state.
// For example, a given operator might be applied to the density matrix as a
// left multiplication or a right multiplication.
class super_op {
public:
  // A super_op term is a pair of left/right multiplication operators.
  // If the first (second) operator of the pair is missing (null), it represents
  // the right (left) multiplication. If both are present, it represents a
  // multiplication on both side (`A * rho * B`, `A` and `B` are the first and
  // second operators in the pair).
  using term =
      std::pair<std::optional<cudaq::product_op<cudaq::matrix_handler>>,
                std::optional<cudaq::product_op<cudaq::matrix_handler>>>;
  /// @brief Default constructor
  super_op() = default;

  /// @brief Combine the given super-operator into this
  /// @param superOp Input super-operator to combine
  /// @return This super-operator after accumulating terms from the other
  /// super-operator
  super_op &operator+=(const super_op &superOp);

  /// @brief Multiply this super-operator by a scalar
  /// @tparam T Scalar type
  /// @param coeff Multiplication coefficient
  /// @return This super-operator after being scaled by the input scalar
  template <typename T>
  super_op &operator*=(T coeff) {
    for (auto &[l_op, r_op] : m_terms) {
      if (l_op.has_value())
        l_op->operator*=(coeff);

      assert(r_op.has_value());
      r_op->operator*=(coeff);
    }
    return *this;
  }

  /// @brief Create a super-operator that represents the left multiplication of
  /// the input product operator
  /// @param op Product operator to be applied to the left
  /// @return Super-operator
  static super_op
  left_multiply(const cudaq::product_op<cudaq::matrix_handler> &op);

  /// @brief Create a super-operator that represents the right multiplication of
  /// the input product operator
  /// @param op Product operator to be applied to the right
  /// @return Super-operator
  static super_op
  right_multiply(const cudaq::product_op<cudaq::matrix_handler> &op);

  /// @brief Create a super-operator that represents the simultaneous left and
  /// right multiplication action
  /// @param leftOp Operator to be applied on the left
  /// @param rightOp Operator to be applied on the right
  /// @return Super-operator
  static super_op
  left_right_multiply(const cudaq::product_op<cudaq::matrix_handler> &leftOp,
                      const cudaq::product_op<cudaq::matrix_handler> &rightOp);

  /// @brief Create a super-operator that represents the left multiplication of
  /// the input sum operator
  /// @param op Sum operator to be applied to the left
  /// @return Super-operator
  static super_op left_multiply(const cudaq::sum_op<cudaq::matrix_handler> &op);

  /// @brief Create a super-operator that represents the right multiplication of
  /// the input sum operator
  /// @param op Sum operator to be applied to the right
  /// @return Super-operator
  static super_op
  right_multiply(const cudaq::sum_op<cudaq::matrix_handler> &op);

  /// @brief Create a super-operator that represents the simultaneous left and
  /// right multiplication action
  /// @param leftOp Operator to be applied on the left
  /// @param rightOp Operator to be applied on the right
  /// @return Super-operator
  static super_op
  left_right_multiply(const cudaq::sum_op<cudaq::matrix_handler> &leftOp,
                      const cudaq::sum_op<cudaq::matrix_handler> &rightOp);

  /// @brief Super-operator term iterator
  using const_iterator = std::vector<term>::const_iterator;

  /// @brief Get iterator to beginning of operator terms
  const_iterator begin() const;

  /// @brief Get iterator to end of operator terms
  const_iterator end() const;

  /// @brief Get a reference to a specific term in the super-operator
  /// @param idx Index of the term to retrieve
  /// @return Reference to the specified term
  const term &operator[](std::size_t idx) const { return m_terms[idx]; }

  /// @brief Get the number of terms in the super-operator
  /// @return Number of terms
  std::size_t num_terms() const { return m_terms.size(); }

private:
  /// @brief Construct a super-operator from a term
  /// @param term Super-operator term
  super_op(term &&term);
  /// @brief Construct a super-operator from a list of terms
  /// @param terms Super-operator term
  super_op(std::vector<term> &&terms);

private:
  std::vector<term> m_terms;
};

// type aliases for convenience
/// @brief Typedef for a map of parameters.
/// This typedef defines `parameter_map` as a map of strings to complex
/// numbers, which is used to store the parameters for the operators.
typedef std::unordered_map<std::string, std::complex<double>> parameter_map;
/// @brief Typedef for a map of dimensions.
/// This typedef defines `dimension_map` as a map of integers to 64-bit
/// integers, which defines the number of levels (i.e., the dimension) for each
/// degree of freedom an operator targets.
typedef std::unordered_map<std::size_t, std::int64_t> dimension_map;
/// @brief Typedef for a sum operation using a matrix handler.
/// This typedef defines `matrix_op` as a sum operation that utilizes
/// the `matrix_handler` for its operations.
typedef sum_op<matrix_handler> matrix_op;
/// @brief Typedef for a product operator with a matrix handler.
/// This typedef defines a matrix operation term using the product_op template
/// with a matrix_handler as the template parameter.
typedef product_op<matrix_handler> matrix_op_term;
/// @brief Typedef for a sum operator specialized with a spin handler.
/// This typedef creates a shorthand for a sum operator that uses the
/// spin_handler.
typedef sum_op<spin_handler> spin_op;
/// @brief Typedef for a product operator term specific to spin handling.
/// This typedef defines spin_op_term as a product_op with a spin_handler,
/// which is used to represent terms in spin operator expressions.
typedef product_op<spin_handler> spin_op_term;
/// @brief Typedef for a sum operator specialized with a boson handler.
/// This typedef creates a shorthand for a sum operator that uses the
/// boson_handler.
typedef sum_op<boson_handler> boson_op;
/// @brief Typedef for a product operator term specific to boson handling.
/// This typedef defines boson_op_term as a product_op with a boson_handler,
/// which is used to represent terms in boson operator expressions.
typedef product_op<boson_handler> boson_op_term;
/// @brief Typedef for a sum operator specialized with a `fermion` handler.
/// This typedef creates an alias `fermion_op` for a `sum_op` that is
/// specialized to handle fermionic operations.
typedef sum_op<fermion_handler> fermion_op;
/// @brief Typedef for a product operator term using a `fermion` handler.
/// This typedef defines `fermion_op_term` as a product_op with a
/// `fermion_handler`.
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
