/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <pybind11/functional.h>
#include <pybind11/stl.h>

#include "py_observe.h"
#include "py_vqe.h"

#include "cudaq/algorithms/gradient.h"
#include "cudaq/algorithms/optimizer.h"

namespace cudaq {

/// @brief Run `cudaq.vqe()` without a gradient strategy.
optimization_result pyVQE(kernel_builder<> &kernel, spin_op &hamiltonian,
                          cudaq::optimizer &optimizer, const int n_params,
                          const int shots = -1) {
  if (kernel.getNumParams() != 1)
    throw std::runtime_error(
        "Kernels with signature other than "
        "`void(List[float])` must provide an `argument_mapper`.");

  if (!kernel.isArgStdVec(0))
    throw std::runtime_error(
        "Kernels with signature other than "
        "`void(List[float])` must provide an `argument_mapper`.");

  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    py::args params = py::make_tuple(x);
    observe_result result = pyObserve(kernel, hamiltonian, params, shots);
    double energy = result.exp_val_z();
    return energy;
  });
}

/// @brief Run `cudaq.vqe()` without a gradient strategy, using the
/// user provided `argument_mapper`.
optimization_result pyVQE(kernel_builder<> &kernel, spin_op &hamiltonian,
                          cudaq::optimizer &optimizer, const int n_params,
                          py::function &argumentMapper, const int shots = -1) {
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    py::args params;
    auto hasToBeTuple = argumentMapper(x);
    if (py::isinstance<py::tuple>(hasToBeTuple))
      params = hasToBeTuple;
    else
      params = py::make_tuple(hasToBeTuple);
    observe_result result = pyObserve(kernel, hamiltonian, params, shots);
    double energy = result.exp_val_z();
    return energy;
  });
}

/// @brief Run `cudaq.vqe()` with the provided gradient strategy.
optimization_result pyVQE(kernel_builder<> &kernel, cudaq::gradient &gradient,
                          spin_op &hamiltonian, cudaq::optimizer &optimizer,
                          const int n_params, const int shots = -1) {
  if (kernel.getNumParams() != 1)
    throw std::runtime_error(
        "Kernels with signature other than "
        "`void(List[float])` must provide an `argument_mapper`.");

  if (!kernel.isArgStdVec(0))
    throw std::runtime_error(
        "Kernels with signature other than "
        "`void(List[float])` must provide an `argument_mapper`.");

  // Get the expected value of the system, <H> at the provided
  // vector of parameters. This is passed to `cudaq::gradient::compute`
  // to allow for the calculation of the gradient vector with the
  // provided gradient strategy.
  std::function<double(std::vector<double>)> get_expected_value =
      [&](std::vector<double> x) {
        py::args params = py::make_tuple(x);
        observe_result result = pyObserve(kernel, hamiltonian, params, shots);
        double energy = result.exp_val_z();
        return energy;
      };
  auto requires_grad = optimizer.requiresGradients();
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double energy = get_expected_value(x);
    if (requires_grad) {
      grad_vec = gradient.compute(x, get_expected_value, energy);
    }
    return energy;
  });
}

/// @brief Run `cudaq.vqe()` with the provided gradient strategy,
/// using the provided `argument_mapper`.
optimization_result pyVQE(kernel_builder<> &kernel, cudaq::gradient &gradient,
                          spin_op &hamiltonian, cudaq::optimizer &optimizer,
                          const int n_params, py::function &argumentMapper,
                          const int shots = -1) {

  // Get the expected value of the system, <H> at the provided
  // vector of parameters. This is passed to `cudaq::gradient::compute`
  // to allow for the calculation of the gradient vector with the
  // provided gradient strategy.
  std::function<double(std::vector<double>)> get_expected_value =
      [&](std::vector<double> x) {
        py::args params;
        auto hasToBeTuple = argumentMapper(x);
        if (py::isinstance<py::tuple>(hasToBeTuple))
          params = hasToBeTuple;
        else
          params = py::make_tuple(hasToBeTuple);
        observe_result result = pyObserve(kernel, hamiltonian, params, shots);
        double energy = result.exp_val_z();
        return energy;
      };
  auto requires_grad = optimizer.requiresGradients();
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double energy = get_expected_value(x);
    if (requires_grad) {
      grad_vec = gradient.compute(x, get_expected_value, energy);
    }
    return energy;
  });
}

void bindVQE(py::module &mod) {
  /// @brief Gradient-Free `cudaq.optimizer` overloads:
  mod.def(
      "vqe",
      [](kernel_builder<> &kernel, cudaq::spin_op &spin_operator,
         cudaq::optimizer &optimizer, const int parameter_count,
         const int shots) {
        auto requires_grad = optimizer.requiresGradients();
        if (requires_grad) {
          std::runtime_error("Provided optimizer requires a gradient strategy "
                             "but none was given.\n");
        }
        return pyVQE(kernel, spin_operator, optimizer, parameter_count, shots);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::arg("optimizer"),
      py::arg("parameter_count"), py::arg("shots") = -1, "");

  // With a provided `argument_mapper`.
  mod.def(
      "vqe",
      [](kernel_builder<> &kernel, cudaq::spin_op &spin_operator,
         cudaq::optimizer &optimizer, const int parameter_count,
         py::function &argumentMapper, const int shots) {
        auto requires_grad = optimizer.requiresGradients();
        if (requires_grad) {
          std::runtime_error("Provided optimizer requires a gradient strategy "
                             "but none was given.\n");
        }
        return pyVQE(kernel, spin_operator, optimizer, parameter_count,
                     argumentMapper, shots);
      },
      py::arg("kernel"), py::arg("spin_operator"), py::arg("optimizer"),
      py::arg("parameter_count"), py::arg("argument_mapper"),
      py::arg("shots") = -1, "");

  /// @brief Gradient based `cudaq.optimizers` overloads:
  mod.def(
      "vqe",
      [](kernel_builder<> &kernel, cudaq::gradient &gradient,
         cudaq::spin_op &spin_operator, cudaq::optimizer &optimizer,
         const int parameter_count, const int shots) {
        return pyVQE(kernel, gradient, spin_operator, optimizer,
                     parameter_count, shots);
      },
      py::arg("kernel"), py::arg("gradient_strategy"), py::arg("spin_operator"),
      py::arg("optimizer"), py::arg("parameter_count"), py::arg("shots") = -1,
      "");

  // With a provided `argument_mapper`.
  mod.def(
      "vqe",
      [](kernel_builder<> &kernel, cudaq::gradient &gradient,
         cudaq::spin_op &spin_operator, cudaq::optimizer &optimizer,
         const int parameter_count, py::function &argumentMapper,
         const int shots) {
        return pyVQE(kernel, gradient, spin_operator, optimizer,
                     parameter_count, argumentMapper, shots);
      },
      py::arg("kernel"), py::arg("gradient_strategy"), py::arg("spin_operator"),
      py::arg("optimizer"), py::arg("parameter_count"),
      py::arg("argument_mapper"), py::arg("shots") = -1, "");
}

} // namespace cudaq
