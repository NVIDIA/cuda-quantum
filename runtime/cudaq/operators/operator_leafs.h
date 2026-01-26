/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <complex>
#include <functional>
#include <map>
#include <type_traits>
#include <unordered_map>
#include <variant>
#include <vector>

#include "callback.h"
#include "cudaq/utils/matrix.h"

namespace cudaq {

class scalar_operator {

private:
  // If someone gave us a constant value, we will just return that
  // directly to them when they call `evaluate`.
  std::variant<std::complex<double>, scalar_callback> value = 1.;
  std::unordered_map<std::string, std::string> param_desc = {};

public:
  // read-only properties

  /// @brief Checks if the scalar operator represents a constant value.
  /// @return True if the operator is constant, false otherwise.
  bool is_constant() const;

  /// @brief A map that contains the documentation the parameters of
  /// the operator, if available. The operator may use parameters that
  /// are not represented in this dictionary.
  const std::unordered_map<std::string, std::string> &
  get_parameter_descriptions() const;

  // constructors and destructors
  /// @brief Default constructor that initializes the scalar operator
  constexpr scalar_operator() = default;

  /// @brief Constructs a scalar operator with a double value.
  scalar_operator(double value);

  /// @brief Constructs a scalar operator with a complex double value.
  scalar_operator(std::complex<double> value);

  /// @brief Constructs a scalar operator from a scalar callback.
  scalar_operator(const scalar_callback &create,
                  std::unordered_map<std::string, std::string>
                      &&parameter_descriptions = {});

  /// @brief Constructs a scalar operator from an rvalue scalar callback.
  scalar_operator(scalar_callback &&create,
                  std::unordered_map<std::string, std::string>
                      &&parameter_descriptions = {});

  /// @brief Copy constructor.
  scalar_operator(const scalar_operator &other) = default;

  /// @brief Move constructor.
  scalar_operator(scalar_operator &&other) = default;

  /// @brief Default destructor.
  ~scalar_operator() = default;

  // assignments

  /// @brief Copy assignment operator.
  scalar_operator &operator=(const scalar_operator &other) = default;

  /// @brief Move assignment operator.
  scalar_operator &operator=(scalar_operator &&other) = default;

  /// @brief Evaluates the scalar operator using the given parameters.
  /// @param parameters A map of parameter names to complex values.
  /// @return The evaluated complex value.
  std::complex<double>
  evaluate(const std::unordered_map<std::string, std::complex<double>>
               &parameters = {}) const;

  /// @brief Converts the scalar operator to a 1x1 matrix.
  /// @param parameters A map of parameter names to complex values.
  /// @return The 1x1 complex matrix representation.
  complex_matrix
  to_matrix(const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const;

  /// @brief Returns a string representation of the scalar operator.
  /// @return The string representation.
  std::string to_string() const;

  /// @brief Compares two scalar operators for equality.
  /// @param other The scalar operator to compare against.
  /// @return True if both operators are equal, false otherwise.
  bool operator==(const scalar_operator &other) const;

  // unary operators

  /// @brief Unary minus operator (lvalue).
  /// @return A new scalar operator representing the negation.
  scalar_operator operator-() const &;

  /// @brief Unary minus operator (rvalue).
  /// @return A new scalar operator representing the negation.
  scalar_operator operator-() &&;

  /// @brief Unary plus operator (lvalue).
  /// @return A copy of the current scalar operator.
  scalar_operator operator+() const &;

  /// @brief Unary plus operator (rvalue).
  /// @return A moved copy of the current scalar operator.
  scalar_operator operator+() &&;

  // right-hand arithmetics

  /// @brief Multiplies the scalar operator by a double (lvalue).
  scalar_operator operator*(double other) const &;

  /// @brief Multiplies the scalar operator by a double (rvalue).
  scalar_operator operator*(double other) &&;

  /// @brief Divides the scalar operator by a double (lvalue).
  scalar_operator operator/(double other) const &;

  /// @brief Divides the scalar operator by a double (rvalue).
  scalar_operator operator/(double other) &&;

  /// @brief Adds a double to the scalar operator (lvalue).
  scalar_operator operator+(double other) const &;

  /// @brief Adds a double to the scalar operator (rvalue).
  scalar_operator operator+(double other) &&;

  /// @brief Subtracts a double from the scalar operator (lvalue).
  scalar_operator operator-(double other) const &;

  /// @brief Subtracts a double from the scalar operator (rvalue).
  scalar_operator operator-(double other) &&;

  /// @brief Multiplies the scalar operator by a double in-place.
  scalar_operator &operator*=(double other);

  /// @brief Divides the scalar operator by a double in-place.
  scalar_operator &operator/=(double other);

  /// @brief Adds a double to the scalar operator in-place.
  scalar_operator &operator+=(double other);

  /// @brief Subtracts a double from the scalar operator in-place.
  scalar_operator &operator-=(double other);

  /// @brief Multiplies the scalar operator by a complex number (lvalue).
  scalar_operator operator*(std::complex<double> other) const &;

  /// @brief Multiplies the scalar operator by a complex number (rvalue).
  scalar_operator operator*(std::complex<double> other) &&;

  /// @brief Divides the scalar operator by a complex number (lvalue).
  scalar_operator operator/(std::complex<double> other) const &;

  /// @brief Divides the scalar operator by a complex number (rvalue).
  scalar_operator operator/(std::complex<double> other) &&;

  /// @brief Adds a complex number to the scalar operator (lvalue).
  scalar_operator operator+(std::complex<double> other) const &;

  /// @brief Adds a complex number to the scalar operator (rvalue).
  scalar_operator operator+(std::complex<double> other) &&;

  /// @brief Subtracts a complex number from the scalar operator (lvalue).
  scalar_operator operator-(std::complex<double> other) const &;

  /// @brief Subtracts a complex number from the scalar operator (rvalue).
  scalar_operator operator-(std::complex<double> other) &&;

  /// @brief Multiplies the scalar operator by a complex number in-place.
  scalar_operator &operator*=(std::complex<double> other);

  /// @brief Divides the scalar operator by a complex number in-place.
  scalar_operator &operator/=(std::complex<double> other);

  /// @brief Adds a complex number to the scalar operator in-place.
  scalar_operator &operator+=(std::complex<double> other);

  /// @brief Subtracts a complex number from the scalar operator in-place.
  scalar_operator &operator-=(std::complex<double> other);

  /// @brief Multiplies two scalar operators (lvalue).
  scalar_operator operator*(const scalar_operator &other) const &;

  /// @brief Multiplies two scalar operators (rvalue).
  scalar_operator operator*(const scalar_operator &other) &&;

  /// @brief Divides two scalar operators (lvalue).
  scalar_operator operator/(const scalar_operator &other) const &;

  /// @brief Divides two scalar operators (rvalue).
  scalar_operator operator/(const scalar_operator &other) &&;

  /// @brief Adds two scalar operators (lvalue).
  scalar_operator operator+(const scalar_operator &other) const &;

  /// @brief Adds two scalar operators (rvalue).
  scalar_operator operator+(const scalar_operator &other) &&;

  /// @brief Subtracts two scalar operators (lvalue).
  scalar_operator operator-(const scalar_operator &other) const &;

  /// @brief Subtracts two scalar operators (rvalue).
  scalar_operator operator-(const scalar_operator &other) &&;

  /// @brief Multiplies two scalar operators in-place.
  scalar_operator &operator*=(const scalar_operator &other);

  /// @brief Divides two scalar operators in-place.
  scalar_operator &operator/=(const scalar_operator &other);

  /// @brief Adds two scalar operators in-place.
  scalar_operator &operator+=(const scalar_operator &other);

  /// @brief Subtracts two scalar operators in-place.
  scalar_operator &operator-=(const scalar_operator &other);

  // left-hand arithmetics

  /// @brief Multiplies a double with a scalar operator (lvalue).
  friend scalar_operator operator*(double other, const scalar_operator &self);

  /// @brief Multiplies a double with a scalar operator (rvalue).
  friend scalar_operator operator*(double other, scalar_operator &&self);

  /// @brief Divides a double by a scalar operator (lvalue).
  friend scalar_operator operator/(double other, const scalar_operator &self);

  /// @brief Divides a double by a scalar operator (rvalue).
  friend scalar_operator operator/(double other, scalar_operator &&self);

  /// @brief Adds a double to a scalar operator (lvalue).
  friend scalar_operator operator+(double other, const scalar_operator &self);

  /// @brief Adds a double to a scalar operator (rvalue).
  friend scalar_operator operator+(double other, scalar_operator &&self);

  /// @brief Subtracts a scalar operator from a double (lvalue).
  friend scalar_operator operator-(double other, const scalar_operator &self);

  /// @brief Subtracts a scalar operator from a double (rvalue).
  friend scalar_operator operator-(double other, scalar_operator &&self);

  /// @brief Multiplies a complex number with a scalar operator (lvalue).
  friend scalar_operator operator*(std::complex<double> other,
                                   const scalar_operator &self);

  /// @brief Multiplies a complex number with a scalar operator (rvalue).
  friend scalar_operator operator*(std::complex<double> other,
                                   scalar_operator &&self);

  /// @brief Divides a complex number by a scalar operator (lvalue).
  friend scalar_operator operator/(std::complex<double> other,
                                   const scalar_operator &self);

  /// @brief Divides a complex number by a scalar operator (rvalue).
  friend scalar_operator operator/(std::complex<double> other,
                                   scalar_operator &&self);

  /// @brief Adds a complex number to a scalar operator (lvalue).
  friend scalar_operator operator+(std::complex<double> other,
                                   const scalar_operator &self);

  /// @brief Adds a complex number to a scalar operator (rvalue).
  friend scalar_operator operator+(std::complex<double> other,
                                   scalar_operator &&self);

  /// @brief Subtracts a scalar operator from a complex number (lvalue).
  friend scalar_operator operator-(std::complex<double> other,
                                   const scalar_operator &self);

  /// @brief Subtracts a scalar operator from a complex number (rvalue).
  friend scalar_operator operator-(std::complex<double> other,
                                   scalar_operator &&self);
};

// Generally speaking, degrees of freedom can (and should) be grouped
// into particles/states of different kind. For example, a system may
// consist of both boson and fermion particles. Regardless of how an
// operator is composed, the particle kind of each degree should always
// remain fixed. Since "degrees" are runtime information, we approximate
// the distinction of particles via distinguishing operator types. This
// distinction at the type system level is only possible when we have
// a single kind of particle in an operator. As soon as we have different
// kinds, the operators get converted to a general "matrix operator" type,
// and we rely on runtime tracking to enforce the correct particle-kind
// specific behavior for subsequent manipulations of the operator.
// The commutation relations declared below store the information about
// what kind of particles an operator acts on. Each "kind" of particles
// is assigned a unique id, as well as a complex value that reflects the
// factor acquired when two operator that act on the same group of particles
// are exchanged (e.g. 1 for bosons, and -1 for fermions).
struct commutation_relations {
  friend class operator_handler;

private:
  // The factor that should be applied when exchanging two operators with the
  // same group id that act on different degrees of freedom. E.g. for fermion
  // relations {a†(k), a(q)} = 0 for k != q, the exchange factor is -1.
  static std::unordered_map<uint, std::complex<double>> exchange_factors;

  // The id for the "commutation set" an operator class or instance belongs to.
  // If the id is negative, it indicates that operators of this kind
  // always commute with all other operators.
  int id;

  constexpr commutation_relations(int group_id) : id(group_id) {}

public:
  // Negative ids are reserved for the operator classes that CUDA-Q defines.
  void define(uint group_id, std::complex<double> exchange_factor);

  constexpr commutation_relations(const commutation_relations &other)
      : id(other.id) {}

  constexpr commutation_relations &
  operator=(const commutation_relations &other) {
    if (this != &other) {
      this->id = other.id;
    }
    return *this;
  }

  std::complex<double> commutation_factor() const;

  bool operator==(const commutation_relations &other) const;
};

template <typename HandlerTy>
class product_op;

template <typename HandlerTy>
class sum_op;

using csr_spmatrix =
    std::tuple<std::vector<std::complex<double>>, std::vector<std::size_t>,
               std::vector<std::size_t>>;

class operator_handler {
  template <typename T>
  friend class product_op;
  template <typename T>
  friend class sum_op;
  template <typename T>
  friend class operator_arithmetics;

private:
  // Validate or populate the dimension defined for the degree(s) of freedom the
  // operator acts on, and return a string that identifies the operator but not
  // what degrees it acts on. Use for canonical evaluation and not expected to
  // be user friendly.
  virtual std::string
  canonical_form(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                 std::vector<std::int64_t> &relevant_dims) const = 0;

  // data storage classes for evaluation

  class matrix_evaluation {
    template <typename T>
    friend class product_op;
    template <typename T>
    friend class sum_op;
    template <typename T>
    friend class operator_arithmetics;

  private:
    std::vector<std::size_t> degrees;
    complex_matrix matrix;

  public:
    matrix_evaluation();
    matrix_evaluation(std::vector<std::size_t> &&degrees,
                      complex_matrix &&matrix);
    matrix_evaluation(matrix_evaluation &&other);
    matrix_evaluation &operator=(matrix_evaluation &&other);
    // delete copy constructor and copy assignment to avoid unnecessary copies
    matrix_evaluation(const matrix_evaluation &other) = delete;
    matrix_evaluation &operator=(const matrix_evaluation &other) = delete;
  };

  class canonical_evaluation {
    template <typename T>
    friend class product_op;
    template <typename T>
    friend class sum_op;
    template <typename T>
    friend class operator_arithmetics;

  private:
    struct term_data {
      std::string encoding;
      std::complex<double> coefficient;
      std::vector<int64_t> relevant_dimensions;
    };
    std::vector<term_data> terms;

  public:
    canonical_evaluation();
    canonical_evaluation(std::complex<double> &&coefficient);
    canonical_evaluation(std::string &&encoding,
                         std::vector<int64_t> &&relevant_dims);
    canonical_evaluation(canonical_evaluation &&other);
    canonical_evaluation &operator=(canonical_evaluation &&other);
    // delete copy constructor and copy assignment to avoid unnecessary copies
    canonical_evaluation(const canonical_evaluation &other) = delete;
    canonical_evaluation &operator=(const canonical_evaluation &other) = delete;
    void push_back(const std::string &op,
                   const std::vector<int64_t> &relevant_dimensions);
  };

public:
#if !defined(NDEBUG)
  static bool can_be_canonicalized; // whether a canonical order can be defined
                                    // for operator expressions
#endif

  // Individual handlers should *not* override this but rather adhere to it.
  // The canonical ordering is the ordering used internally by the operator
  // classes. It is the order in which custom matrix operators are defined,
  // the order returned by to_matrix and degree, and the order in which a user
  // would define a state vector.
  static constexpr auto canonical_order = std::less<std::size_t>();

  /// Default commutation relations mean that two operator always commute as
  /// long as they act on different degrees of freedom.
  static constexpr commutation_relations default_commutation_relations =
      commutation_relations(-1);
  static constexpr commutation_relations fermion_commutation_relations =
      commutation_relations(-2);
  static constexpr commutation_relations boson_commutation_relations =
      default_commutation_relations;
  static commutation_relations custom_commutation_relations(uint id);

  static constexpr commutation_relations commutation_group =
      default_commutation_relations;
  // Indicates whether operators of this type commute with any other operator,
  // as long as both operators don't act on the same degree.
  // Handlers that require non-trivial commutation relations across different
  // degrees should define an instance variable with the same name and set it
  // to true or false as appropriate for the concrete instance (e.g. the
  // identity operator will always commute regardless of what kind of particle
  // it acts on).
  static constexpr bool commutes_across_degrees = true;

  virtual ~operator_handler() = default;

  // returns a unique string id for the operator
  /// @brief Generates a unique identifier for the derived class.
  /// @return A string representing the unique identifier.
  virtual std::string unique_id() const = 0;

  virtual std::vector<std::size_t> degrees() const = 0;

  /// @brief Return the `matrix_handler` as a matrix.
  /// @param  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  virtual complex_matrix
  to_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
            const std::unordered_map<std::string, std::complex<double>>
                &parameters = {}) const = 0;

  /// @brief Converts the object to a string representation.
  /// @param include_degrees A boolean flag indicating whether to include
  /// degrees in the string representation.
  /// @return A string representation of the object.
  virtual std::string to_string(bool include_degrees = true) const = 0;
};

// An adapter for operator handler classes that can generate multi-diagonal
// representation.
// Subclasses to implement the `to_diagonal_matrix` method
class mdiag_operator_handler {
public:
  /// @brief Return the `matrix_handler` as a multi-diagonal matrix.
  /// @param  `dimensions` : A map specifying the number of levels,
  ///                      that is, the dimension of each degree of freedom
  ///                      that the operator acts on. Example for two, 2-level
  ///                      degrees of freedom: `{0 : 2, 1 : 2}`.
  /// @param  `parameters` : A map specifying runtime parameter values.
  virtual mdiag_sparse_matrix
  to_diagonal_matrix(std::unordered_map<std::size_t, std::int64_t> &dimensions,
                     const std::unordered_map<std::string, std::complex<double>>
                         &parameters = {}) const = 0;
  /// @brief Default destructor
  virtual ~mdiag_operator_handler() = default;
};
} // namespace cudaq
