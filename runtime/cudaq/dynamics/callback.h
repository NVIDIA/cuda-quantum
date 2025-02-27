/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/qis/state.h"
#include "cudaq/utils/tensor.h"

#include <complex>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace cudaq {

class ScalarCallbackFunction {
private:
  // The user provided callback function that takes a map of complex
  // parameters.
  std::function<std::complex<double>(
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  template <typename Callable>
  ScalarCallbackFunction(Callable &&callable) {
    static_assert(
        std::is_invocable_r_v<
            std::complex<double>, Callable,
            const std::unordered_map<std::string, std::complex<double>> &>,
        "Invalid callback function. Must have signature std::complex<double>("
        "const std::unordered_map<std::string, std::complex<double>>&)");
    callback_func = std::forward<Callable>(callable);
  }

  ScalarCallbackFunction(const ScalarCallbackFunction &other) = default;
  ScalarCallbackFunction(ScalarCallbackFunction &&other) = default;

  ScalarCallbackFunction &
  operator=(const ScalarCallbackFunction &other) = default;
  ScalarCallbackFunction &operator=(ScalarCallbackFunction &&other) = default;

  std::complex<double> operator()(
      const std::unordered_map<std::string, std::complex<double>> &parameters)
      const;
};

class MatrixCallbackFunction {
private:
  // The user provided callback function that takes a vector defining the
  // dimension for each degree of freedom it acts on, and a map of complex
  // parameters.
  std::function<matrix_2(
      const std::vector<int> &,
      const std::unordered_map<std::string, std::complex<double>> &)>
      callback_func;

public:
  template <typename Callable>
  MatrixCallbackFunction(Callable &&callable) {
    static_assert(
        std::is_invocable_r_v<
            matrix_2, Callable, const std::vector<int> &,
            const std::unordered_map<std::string, std::complex<double>> &>,
        "Invalid callback function. Must have signature "
        "matrix_2(const std::vector<int>&, const "
        "std::unordered_map<std::string, std::complex<double>>&)");
    callback_func = std::forward<Callable>(callable);
  }

  MatrixCallbackFunction(const MatrixCallbackFunction &other) = default;
  MatrixCallbackFunction(MatrixCallbackFunction &&other) = default;

  MatrixCallbackFunction &
  operator=(const MatrixCallbackFunction &other) = default;
  MatrixCallbackFunction &operator=(MatrixCallbackFunction &&other) = default;

  matrix_2
  operator()(const std::vector<int> &relevant_dimensions,
             const std::unordered_map<std::string, std::complex<double>>
                 &parameters) const;
};

/// @brief Object used to store the definition of a custom matrix operator.
class Definition {
private:
  std::string id;
  MatrixCallbackFunction generator;
  std::vector<int> required_dimensions;

public:
  const std::vector<int> &expected_dimensions = this->required_dimensions;

  Definition(std::string operator_id,
             const std::vector<int> &expected_dimensions,
             MatrixCallbackFunction &&create);
  Definition(Definition &&def);
  ~Definition();

  // To call the generator function
  matrix_2
  generate_matrix(const std::vector<int> &relevant_dimensions,
                  const std::unordered_map<std::string, std::complex<double>>
                      &parameters) const;
};
} // namespace cudaq
