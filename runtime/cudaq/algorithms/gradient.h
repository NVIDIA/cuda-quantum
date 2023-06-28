/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "observe.h"
#include <cudaq/builder.h>
#include <cudaq/spin_op.h>
#include <functional>

namespace cudaq {

///
/// \brief The cudaq::gradient represents a base type for all gradient
/// strategies leveraged by variational algorithms.
///
/// \details The cudaq::gradient tracks a std::function with signature
/// void(std::vector<double>) representing the parameterized kernel
/// ansatz. For ansatzae that do not follow this signature, a separate
/// Argument Mapper must be provided which takes std::vector<double> to
/// a tuple of custom function arguments. All gradient subtypes should
/// inherit the base class constructors (using gradient::gradient), but
/// provide a concrete implementation of gradient::compute(). The compute
/// method takes as input the current iteration parameters, a reference to
/// the gradient vector which compute() must update, the cudaq::spin_op
/// for this variational algorithm, and the expected value of the spin_op
/// at the current iteration's parameters. Subtypes can leverage the
/// protected gradient::getExpectedValue() method to compute
/// <psi(x) | H | psi(x)> at the provided set of variational parameters.
///
class gradient {
protected:
  /// The parameterized ansatz, a quantum kernel expression
  std::function<void(std::vector<double>)> ansatz_functor;

  // Given the parameters x and the spin_op h, compute the
  // expected value with respect to the ansatz.
  double getExpectedValue(std::vector<double> &x, spin_op h) {
    return cudaq::observe(ansatz_functor, h, x);
  }

public:
  /// Constructor, takes the quantum kernel with prescribed signature
  gradient(std::function<void(std::vector<double>)> &&kernel)
      : ansatz_functor(kernel) {}

  /// Empty constructor.
  gradient() = default;

  /// Constructor, takes a callable with non-standard signature
  /// the Argument Mapper function object that maps the parameters to
  /// concrete arguments for the quantum kernel.
  template <typename KernelT, typename ArgsMapper>
  gradient(KernelT &kernel, ArgsMapper &&argsMapper) {
    ansatz_functor = [&](std::vector<double> x) {
      auto as_args = argsMapper(x);
      std::apply([&](auto &&...new_args) { kernel(new_args...); }, as_args);
    };
  }

  /// Constructor, takes a callable that must have the
  /// prescribed call signature (void(std::vector<double>))
  template <typename KernelT>
  gradient(KernelT &kernel) {
    if (kernel.getNumParams() != 1)
      throw std::invalid_argument(
          "Callable kernel from cudaq::make_kernel must "
          "have 1 std::vector<double> argument. Provide an ArgMapper if not.");
    ansatz_functor = [&](std::vector<double> x) { return kernel(x); };
  }

  /// Constructor, takes the quantum kernel with non-standard signature
  /// the Argument Mapper function object that maps the parameters to
  /// concrete arguments for the quantum kernel.
  template <typename QuantumKernel, typename ArgsMapper>
  gradient(QuantumKernel &&kernel, ArgsMapper &&argsMapper) {
    ansatz_functor = [&](std::vector<double> x) {
      auto as_args = argsMapper(x);
      std::apply([&](auto &&...new_args) { kernel(new_args...); }, as_args);
    };
  }

  /// Compute the current iterations gradient vector and update the
  /// provided vector<double reference (dx).
  virtual void compute(const std::vector<double> &x, std::vector<double> &dx,
                       spin_op &h, double funcAtX) = 0;

  /// Compute the gradient vector for the provided objective function, `func`,
  /// at the given set of parameters, `x`.
  virtual std::vector<double>
  compute(const std::vector<double> &x,
          const std::function<double(std::vector<double>)> &func,
          double funcAtX) = 0;

  virtual ~gradient() = default;
};
} // namespace cudaq
