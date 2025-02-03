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

using Func = std::function<matrix_2(
    std::vector<int>, std::map<std::string, std::complex<double>>)>;

class CallbackFunction {
private:
  // The user provided callback function that takes a vector defining the 
  // dimension for each degree of freedom it acts on, and a map of complex 
  // parameters.
  Func _callback_func;

public:
  CallbackFunction() = default;

  template <typename Callable>
  CallbackFunction(Callable &&callable) {
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
  CallbackFunction(const CallbackFunction &other) {
    _callback_func = other._callback_func;
  }

  // move constructor.
  CallbackFunction(CallbackFunction &&other) {
    _callback_func = std::move(other._callback_func);
  }

  // assignment operator
  CallbackFunction& operator=(const CallbackFunction &other) {
    if (this != &other) {
      _callback_func = other._callback_func;
    }
    return *this;
  }

  // move assignment operator
  CallbackFunction& operator=(CallbackFunction &&other) {
    if (this != &other) {
      _callback_func = std::move(other._callback_func);
    }
    return *this;
  }

  bool operator!() { return (!_callback_func); }

  matrix_2
  operator()(std::vector<int> relevant_dimensions,
             std::map<std::string, std::complex<double>> parameters) const {
    return _callback_func(std::move(relevant_dimensions), std::move(parameters));
  }
};

using ScalarFunc = std::function<std::complex<double>(
    std::map<std::string, std::complex<double>>)>;

// A scalar callback function does not need to accept the dimensions,
// therefore we will use a different function type for this specific class.
class ScalarCallbackFunction : CallbackFunction {
private:
  // The user provided callback function that takes a vector of parameters.
  ScalarFunc _callback_func;

public:
  ScalarCallbackFunction() = default;

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
  ScalarCallbackFunction(const ScalarCallbackFunction &other) {
    _callback_func = other._callback_func;
  }

  // move constructor.
  ScalarCallbackFunction(ScalarCallbackFunction &&other) {
    _callback_func = std::move(other._callback_func);
  }

  // assignment operator
  ScalarCallbackFunction& operator=(const ScalarCallbackFunction &other) {
    if (this != &other) {
      _callback_func = other._callback_func;
    }
    return *this;
  }

  // move assignment operator
  ScalarCallbackFunction& operator=(ScalarCallbackFunction &&other) {
    if (this != &other) {
      _callback_func = std::move(other._callback_func);
    }
    return *this;
  }

  bool operator!() { return (!_callback_func); }

  std::complex<double>
  operator()(std::map<std::string, std::complex<double>> parameters) const {
    return _callback_func(std::move(parameters));
  }
};

/// @brief Object used to give an error if a Definition of an elementary
/// or scalar operator is instantiated by other means than the `define`
/// class method.
class Definition {
private: 
  std::string id;
  CallbackFunction generator;
  std::vector<int> m_expected_dimensions;

public:

  Definition(const std::string &operator_id, std::vector<int> expected_dimensions, CallbackFunction &&create);
  Definition(Definition &&def);
  ~Definition();

  // To call the generator function
  matrix_2 generate_matrix(
      const std::vector<int> &relevant_dimensions,
      const std::map<std::string, std::complex<double>> &parameters) const;
};
} // namespace cudaq
