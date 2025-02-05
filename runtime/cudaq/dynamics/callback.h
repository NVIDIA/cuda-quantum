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
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace cudaq {

class ScalarCallbackFunction {
private:
  // The user provided callback function that takes a map of complex 
  // parameters.
  std::function<std::complex<double>(std::map<std::string, std::complex<double>>)> _callback_func;

public:
  template <typename Callable>
  ScalarCallbackFunction(Callable &&callable) {
    static_assert(
        std::is_invocable_r_v<std::complex<double>, Callable,
                              std::map<std::string, std::complex<double>>>,
        "Invalid callback function. Must have signature std::complex<double>("
        "std::map<std::string, std::complex<double>>)");
    _callback_func = std::forward<Callable>(callable);
  }

  // copy constructor
  ScalarCallbackFunction(const ScalarCallbackFunction &other);

  // move constructor.
  ScalarCallbackFunction(ScalarCallbackFunction &&other);

  // assignment operator
  ScalarCallbackFunction& operator=(const ScalarCallbackFunction &other);

  // move assignment operator
  ScalarCallbackFunction& operator=(ScalarCallbackFunction &&other);

  std::complex<double>
  operator()(std::map<std::string, std::complex<double>> parameters) const;
};


class MatrixCallbackFunction {
private:
  // The user provided callback function that takes a vector defining the 
  // dimension for each degree of freedom it acts on, and a map of complex 
  // parameters.
  std::function<matrix_2(std::vector<int>, std::map<std::string, std::complex<double>>)> _callback_func;

public:
  template <typename Callable>
  MatrixCallbackFunction(Callable &&callable) {
    static_assert(
        std::is_invocable_r_v<matrix_2, Callable, std::vector<int>,
                              std::map<std::string, std::complex<double>>>,
        "Invalid callback function. Must have signature "
        "matrix_2("
        "std::map<int,int>, "
        "std::map<std::string, std::complex<double>>)");
    _callback_func = std::forward<Callable>(callable);
  }

  // copy constructor
  MatrixCallbackFunction(const MatrixCallbackFunction &other);

  // move constructor.
  MatrixCallbackFunction(MatrixCallbackFunction &&other);

  // assignment operator
  MatrixCallbackFunction& operator=(const MatrixCallbackFunction &other);

  // move assignment operator
  MatrixCallbackFunction& operator=(MatrixCallbackFunction &&other);

  matrix_2
  operator()(std::vector<int> relevant_dimensions,
             std::map<std::string, std::complex<double>> parameters) const;
};


/// @brief Object used to store the definition of a custom matrix operator.
class Definition {
private:
  std::string id;
  MatrixCallbackFunction generator;
  std::vector<int> m_expected_dimensions;

public:
  const std::vector<int>& expected_dimensions = this->m_expected_dimensions;

  Definition(const std::string &operator_id, std::vector<int> expected_dimensions, MatrixCallbackFunction &&create);
  Definition(Definition &&def);
  ~Definition();

  // To call the generator function
  matrix_2 generate_matrix(
      const std::vector<int> &relevant_dimensions,
      const std::map<std::string, std::complex<double>> &parameters) const;
};
} // namespace cudaq
