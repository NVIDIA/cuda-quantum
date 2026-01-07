/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/state.h"
#include "cudaq/utils/matrix.h"

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {

/// @brief A callback wrapper that encapsulates a user-provided function taking
/// a set of complex parameters and returning a complex result.
class scalar_callback {
private:
  // The user provided callback function that takes a map of complex
  // parameters.
  std::function<std::complex<double>(
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  /// @brief Constructs a scalar callback from a callable object.
  /// @tparam Callable The type of the callable object. It must satisfy the
  /// signature: std::complex<double>(`const` std::unordered_map<std::string,
  /// std::complex<double>> &).
  /// @param callable The callable object to be wrapped and stored for later
  /// invocation.
  template <typename Callable,
            std::enable_if_t<
                std::is_invocable_r_v<std::complex<double>, Callable,
                                      const std::unordered_map<
                                          std::string, std::complex<double>> &>,
                bool> = true>
  scalar_callback(Callable &&callable) {
    callback_func = std::forward<Callable>(callable);
  }

  /// @brief Default copy constructor for scalar_callback.
  /// Creates a new scalar_callback instance as a copy of an existing instance.
  scalar_callback(const scalar_callback &other) = default;
  /// @brief Move constructor for the scalar_callback class.
  /// Transfers ownership from the provided source instance to the new instance.
  scalar_callback(scalar_callback &&other) = default;

  /// @brief Default copy assignment operator for scalar_callback.
  scalar_callback &operator=(const scalar_callback &other) = default;
  /// @brief Default move assignment operator for scalar_callback.
  scalar_callback &operator=(scalar_callback &&other) = default;

  /// @brief Evaluates the callback with the provided parameters.
  /// @param parameters A mapping from parameter names to their complex values.
  /// @return std::complex<double> The computed complex result based on the
  /// input parameters.
  std::complex<double> operator()(
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      const;
};

class matrix_callback {
private:
  // The user provided callback function that takes a vector defining the
  // dimension for each degree of freedom it acts on, and a map of complex
  // parameters.
  std::function<complex_matrix(
      const std::vector<std::int64_t> &,
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  template <
      typename Callable,
      std::enable_if_t<
          std::is_invocable_r_v<
              complex_matrix, Callable, const std::vector<std::int64_t> &,
              const std::unordered_map<std::string, std::complex<double>> &>,
          bool> = true>
  matrix_callback(Callable &&callable) {
    callback_func = std::forward<Callable>(callable);
  }

  matrix_callback(const matrix_callback &other) = default;
  matrix_callback(matrix_callback &&other) = default;

  matrix_callback &operator=(const matrix_callback &other) = default;
  matrix_callback &operator=(matrix_callback &&other) = default;

  complex_matrix
  operator()(const std::vector<std::int64_t> &relevant_dimensions,
             const std::unordered_map<std::string, std::complex<double>>
                 &parameters) const;
};

using mdiag_sparse_matrix =
    std::pair<std::vector<std::complex<double>>, std::vector<std::int64_t>>;

class diag_matrix_callback {
private:
  // The user provided callback function that takes a vector defining the
  // dimension for each degree of freedom it acts on, and a map of complex
  // parameters.
  std::function<mdiag_sparse_matrix(
      const std::vector<std::int64_t> &,
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  template <
      typename Callable,
      std::enable_if_t<
          std::is_invocable_r_v<
              mdiag_sparse_matrix, Callable, const std::vector<std::int64_t> &,
              const std::unordered_map<std::string, std::complex<double>> &>,
          bool> = true>
  diag_matrix_callback(Callable &&callable) {
    callback_func = std::forward<Callable>(callable);
  }

  diag_matrix_callback(const diag_matrix_callback &other) = default;
  diag_matrix_callback(diag_matrix_callback &&other) = default;

  diag_matrix_callback &operator=(const diag_matrix_callback &other) = default;
  diag_matrix_callback &operator=(diag_matrix_callback &&other) = default;

  mdiag_sparse_matrix
  operator()(const std::vector<std::int64_t> &relevant_dimensions,
             const std::unordered_map<std::string, std::complex<double>>
                 &parameters) const;
};

/// @brief Object used to store the definition of a custom matrix operator.
class Definition {
private:
  std::string id;
  matrix_callback generator;
  std::optional<diag_matrix_callback> diag_generator;
  std::vector<std::int64_t> required_dimensions;

public:
  const std::vector<std::int64_t> &expected_dimensions =
      this->required_dimensions;
  const std::unordered_map<std::string, std::string> parameter_descriptions;

  Definition(
      std::string operator_id, const std::vector<int64_t> &expected_dimensions,
      matrix_callback &&create,
      std::unordered_map<std::string, std::string> &&parameter_descriptions);
  Definition(
      std::string operator_id, const std::vector<int64_t> &expected_dimensions,
      matrix_callback &&create, diag_matrix_callback &&diag_create,
      std::unordered_map<std::string, std::string> &&parameter_descriptions);
  Definition(Definition &&def);
  ~Definition();

  // To call the generator function
  complex_matrix
  generate_matrix(const std::vector<std::int64_t> &relevant_dimensions,
                  const std::unordered_map<std::string, std::complex<double>>
                      &parameters) const;
  bool has_dia_generator() const { return diag_generator.has_value(); }
  mdiag_sparse_matrix generate_dia_matrix(
      const std::vector<std::int64_t> &relevant_dimensions,
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      const;
};
} // namespace cudaq
