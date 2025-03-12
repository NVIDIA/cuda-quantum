/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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

class scalar_callback {
private:
  // The user provided callback function that takes a map of complex
  // parameters.
  std::function<std::complex<double>(
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  template <typename Callable,
            std::enable_if_t<
                std::is_invocable_r_v<std::complex<double>, Callable,
                                      const std::unordered_map<
                                          std::string, std::complex<double>> &>,
                bool> = true>
  scalar_callback(Callable &&callable) {
    callback_func = std::forward<Callable>(callable);
  }

  scalar_callback(const scalar_callback &other) = default;
  scalar_callback(scalar_callback &&other) = default;

  scalar_callback &operator=(const scalar_callback &other) = default;
  scalar_callback &operator=(scalar_callback &&other) = default;

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
      const std::vector<int> &,
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  template <
      typename Callable,
      std::enable_if_t<
          std::is_invocable_r_v<
              complex_matrix, Callable, const std::vector<int> &,
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
  operator()(const std::vector<int> &relevant_dimensions,
             const std::unordered_map<std::string, std::complex<double>>
                 &parameters) const;
};

/// @brief Object used to store the definition of a custom matrix operator.
class Definition {
private:
  std::string id;
  matrix_callback generator;
  std::vector<int> required_dimensions;

public:
  const std::vector<int> &expected_dimensions = this->required_dimensions;

  Definition(std::string operator_id,
             const std::vector<int> &expected_dimensions,
             matrix_callback &&create);
  Definition(Definition &&def);
  ~Definition();

  // To call the generator function
  complex_matrix
  generate_matrix(const std::vector<int> &relevant_dimensions,
                  const std::unordered_map<std::string, std::complex<double>>
                      &parameters) const;
};
} // namespace cudaq
