/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cudaq/gradients.h"
#include "gradient.h"
#include "observe.h"
#include "optimizer.h"
#include <stdio.h>

namespace cudaq {

namespace __internal__ {
/// \brief This is an internal helper function to reduce duplicated code in the
/// user-facing `vqe()` functions below. Users should not directly call this
/// function.
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>>>
static inline optimization_result
remote_vqe(cudaq::quantum_platform &platform, QuantumKernel &&kernel,
           const cudaq::spin_op &H, cudaq::optimizer &optimizer,
           cudaq::gradient *gradient, const int n_params,
           const std::size_t shots, Args &&...args) {
  auto ctx = std::make_unique<ExecutionContext>("observe", shots);
  ctx->kernelName = cudaq::getKernelName(kernel);
  ctx->spin = cudaq::spin_op::canonicalize(H);
  platform.set_exec_ctx(ctx.get());
  auto serializedArgsBuffer = serializeArgs(args...);
  platform.launchVQE(ctx->kernelName, serializedArgsBuffer.data(), gradient, H,
                     optimizer, n_params, shots);
  platform.reset_exec_ctx();
  return ctx->optResult.value_or(optimization_result{});
}

static inline void print_arg_mapper_warning() {
  printf(
      "WARNING: Usage of ArgMapper type on this platform will result in "
      "suboptimal performance. Consider updating your code to update your "
      "kernel to use this signature (std::function<void(std::vector<double>, "
      "arg1, arg2, ...)>) and pass concrete arguments to cudaq::vqe() for "
      "the non-variational arguments.\n");
}

} // namespace __internal__

///
/// \brief Compute the minimal eigenvalue of \p H with VQE.
///
/// \param kernel The ansatz, a quantum kernel callable, must have
///        callable-type void(std::vector<double>) and no measures.
/// \param H The hermitian cudaq::spin_op to compute the minimal eigenvalue for.
/// \param optimizer The cudaq::optimizer to use for iteratively searching for
///        the minimal eigenvalue of \p H. This optimizer does not require
///        gradients.
/// \param n_params The number of variational parameters in the ansatz quantum
///        kernel callable.
/// \param args Non-variational arguments to \p kernel that will be passed to
///        \p kernel on each invocation during VQE.
/// \returns The optimal value and corresponding parameters as a
///        cudaq::optimization_result (std::tuple<double,std::vector<double>>)
///
/// \details Given a quantum kernel of type void(std::vector<double>),
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian cudaq::spin_op
/// \p H.
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/algorithm.h>
/// #include <cudaq/optimizers.h>
/// ...
/// struct ansatz {
///   void operator(std::vector<double> x) __qpu__ {
///     cudaq::qreg q(2);
///     x(q[0]);
///     ry(x[0], q[1]);
///     x<ctrl>(q[1],q[0]);
///   }
/// };
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// cudaq::optimizers::cobyla optimizer;
/// auto [val, params] = cudaq::vqe(ansatz{}, H, optimizer, 1);
/// \endcode
///
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>>>
optimization_result vqe(QuantumKernel &&kernel, cudaq::spin_op H,
                        cudaq::optimizer &optimizer, const int n_params,
                        Args &&...args) {
  static_assert(
      std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>,
      "Invalid parameterized quantum kernel expression. Must have "
      "void(std::vector<double>, <Args...>) signature, or provide "
      "std::tuple<Args...>(std::vector<double>) ArgMapper function object.");
  if (optimizer.requiresGradients()) {
    throw std::invalid_argument("Provided cudaq::optimizer requires gradients. "
                                "Please provide a cudaq::gradient instance.");
  }

  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe)
    return __internal__::remote_vqe(platform, kernel, H, optimizer,
                                    /*gradient=*/nullptr, n_params, /*shots=*/0,
                                    args...);

  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double e = cudaq::observe(kernel, H, x, args...);
    return e;
  });
}

///
/// \brief Compute the minimal eigenvalue of \p H with VQE.
///
/// \param shots The number samples to collect.
/// \param kernel The ansatz, a quantum kernel callable, must have
///        callable-type void(std::vector<double>) and no measures.
/// \param H The hermitian cudaq::spin_op to compute the minimal eigenvalue for.
/// \param optimizer The cudaq::optimizer to use for iteratively searching for
///        the minimal eigenvalue of \p H. This optimizer does not require
///        gradients.
/// \param n_params The number of variational parameters in the ansatz quantum
///        kernel callable.
/// \param args Non-variational arguments to \p kernel that will be passed to
///        \p kernel on each invocation during VQE.
/// \returns The optimal value and corresponding parameters as a
///        cudaq::optimization_result (std::tuple<double,std::vector<double>>)
///
/// \details Given a quantum kernel of type void(std::vector<double>),
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian cudaq::spin_op
/// \p H.
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/algorithm.h>
/// #include <cudaq/optimizers.h>
/// ...
/// struct ansatz {
///   void operator(std::vector<double> x) __qpu__ {
///     cudaq::qreg q(2);
///     x(q[0]);
///     ry(x[0], q[1]);
///     x<ctrl>(q[1],q[0]);
///   }
/// };
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// cudaq::optimizers::cobyla optimizer;
/// auto [val, params] = cudaq::vqe(/*shots*/ 100, ansatz{}, H, optimizer, 1);
/// \endcode
///
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>>>
optimization_result vqe(std::size_t shots, QuantumKernel &&kernel,
                        cudaq::spin_op H, cudaq::optimizer &optimizer,
                        const int n_params, Args &&...args) {
  static_assert(
      std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>,
      "Invalid parameterized quantum kernel expression. Must have "
      "void(std::vector<double>, <Args...>) signature, or provide "
      "std::tuple<Args...>(std::vector<double>) ArgMapper function object.");
  if (optimizer.requiresGradients()) {
    throw std::invalid_argument("Provided cudaq::optimizer requires gradients. "
                                "Please provide a cudaq::gradient instance.");
  }

  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe)
    return __internal__::remote_vqe(platform, kernel, H, optimizer,
                                    /*gradient=*/nullptr, n_params, shots,
                                    args...);

  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double e = cudaq::observe(shots, kernel, H, x, args...);
    return e;
  });
}

///
/// \brief Compute the minimal eigenvalue of \p H using VQE with a specific
///        gradient strategy.
///
/// \param kernel The ansatz, a quantum kernel callable, must have
///        callable-type void(std::vector<double>) and no measures.
/// \param gradient a sub-type of cudaq::gradient that uses extra QPU calls
///        to compute all elements of the input grad std::vector<double>
///        reference.
/// \param H The hermitian cudaq::spin_op to compute the minimal eigenvalue for.
/// \param optimizer The cudaq::optimizer to use for iteratively searching for
///        the minimal eigenvalue of \p H.
/// \param n_params The number of variational parameters in the ansatz quantum
///        kernel callable.
/// \param args Non-variational arguments to \p kernel that will be passed to
///        \p kernel on each invocation during VQE.
/// \returns The optimal value and corresponding parameters as a
///        cudaq::optimization_result (std::tuple<double,std::vector<double>>)
///
/// \details Given a quantum kernel of type void(std::vector<double>),
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian cudaq::spin_op
/// \p H. If the provided optimizer does not require gradients the gradient
/// input is quietly ignored.
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/algorithm.h>
/// #include <cudaq/optimizers.h>
/// ...
/// auto ansatz = [](std::vector<double> x) __qpu__ {
///   cudaq::qreg q(2);
///   x(q[0]);
///   ry(x[0], q[1]);
///   x<ctrl>(q[1],q[0]);
/// };
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// cudaq::gradients::central_difference gradient(ansatz);
/// cudaq::optimizers::lbfgs optimizer;
/// auto [val, params] =
///     cudaq::vqe(ansatz, gradient, H, optimizer, 1);
/// \endcode
///
template <typename QuantumKernel, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>>>
optimization_result vqe(QuantumKernel &&kernel, cudaq::gradient &gradient,
                        cudaq::spin_op H, cudaq::optimizer &optimizer,
                        const int n_params, Args &&...args) {
  static_assert(
      std::is_invocable_v<QuantumKernel, std::vector<double>> ||
          std::is_invocable_v<QuantumKernel, std::vector<double>, Args...>,
      "Invalid parameterized quantum kernel expression. Must have "
      "void(std::vector<double>, <Args...>) signature, or provide "
      "std::tuple<Args...>(std::vector<double>) ArgMapper function object.");

  auto &platform = cudaq::get_platform();
  if (platform.get_remote_capabilities().vqe)
    return __internal__::remote_vqe(platform, kernel, H, optimizer, &gradient,
                                    n_params,
                                    /*shots=*/0, args...);

  auto requires_grad = optimizer.requiresGradients();
  // If there are additional arguments, we need to clone the gradient and
  // provide it the concrete arguments.
  std::unique_ptr<cudaq::gradient> newGrad = [&]() {
    if (requires_grad) {
      auto newGrad_ = gradient.clone();
      if constexpr (sizeof...(args) > 0)
        newGrad_->setArgs(kernel, args...);
      return newGrad_;
    }
    return std::unique_ptr<cudaq::gradient>();
  }();
  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    double e = cudaq::observe(kernel, H, x, args...);
    if (requires_grad)
      newGrad->compute(x, grad_vec, H, e);
    return e;
  });
}

///
/// \brief Compute the minimal eigenvalue of \p H with VQE with a kernel
///        callable with non-trivial (not `std::vector<double>`) argument
///        structure.
///
/// \param kernel The CUDA-Q callable with non-trivial function
/// signature. \param H The hermitian cudaq::spin_op to compute the minimal
/// eigenvalue for. \param optimizer The cudaq::optimizer to use for iteratively
/// searching for
///        the minimal eigenvalue of \p H.
/// \param n_params The number of variational parameters in the ansatz quantum
///        kernel callable.
/// \param argsMapper A custom callable type with function type
///        `std::tuple<Args...>(std::vector<double>&)` that takes the parameter
///        vector as input and returns a tuple representing the arguments
///        required for evaluation of the quantum kernel.
/// \returns The optimal value and corresponding parameters as a
///        cudaq::optimization_result (std::tuple<double,std::vector<double>>)
///
/// \details Given a quantum kernel of type `void(Args...)`,
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian cudaq::spin_op
/// \p H. This function will use the custom ArgMapper to map input variational
/// parameters to a tuple for use in evaluating the kernel function.
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/builder.h>
/// #include <cudaq/algorithm.h>
/// #include <cudaq/optimizers.h>
/// ...
/// auto [kernel, theta, phi] = cudaq::make_kernel<double, double>();
/// auto q = kernel.qalloc(3);
/// kernel.x(q[0]);
/// kernel.ry(theta, q[1]);
/// kernel.ry(phi, q[2]);
/// kernel.x<ctrl>(q[2], q[0]);
/// kernel.x<ctrl>(q[0], q[1]);
/// kernel.ry(-theta, q[1]);
/// kernel.x<ctrl>(q[0], q[1]);
/// kernel.x<ctrl>(q[1], q[0]);
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///            .21829 * z(0) - 6.125 * z(1) + 9.625 - 9.625 * z(2) -
///            3.913119 * x(1) * x(2) - 3.913119 * y(1) * y(2);
/// cudaq::optimizers::cobyla optimizer;
/// auto [val, params] = cudaq::vqe(kernel, H, optimizer, 1,
///                       [](std::vector<double> x){
///                           return std::make_tuple(x[0], x[1]);
///                      });
/// \endcode
///
template <typename QuantumKernel, typename ArgMapper, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<ArgMapper, std::vector<double>> ||
              std::is_invocable_v<ArgMapper, std::vector<double> &> ||
              std::is_invocable_v<ArgMapper, const std::vector<double> &>>>
optimization_result vqe(QuantumKernel &&kernel, cudaq::spin_op H,
                        cudaq::optimizer &optimizer, const int n_params,
                        ArgMapper &&argsMapper) {
  if (optimizer.requiresGradients()) {
    throw std::invalid_argument(
        "Provided cudaq::optimizer requires gradients. "
        "Please provide a cudaq::gradient instance. Make sure the gradient is "
        "aware of the ArgMapper.");
  }
  if (cudaq::get_platform().get_remote_capabilities().vqe)
    __internal__::print_arg_mapper_warning();

  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    auto args = argsMapper(x);
    double energy = std::apply(
        [&](auto &&...arg) -> double {
          return cudaq::observe(kernel, H, arg...);
        },
        args);
    return energy;
  });
}

///
/// \brief Compute the minimal eigenvalue of \p H with VQE with a kernel
///        callable with non-trivial (not `std::vector<double>`) argument
///        structure.
///
/// \param shots The number samples to collect.
/// \param kernel The quantum callable with non-trivial function
/// signature. \param H The hermitian cudaq::spin_op to compute the minimal
/// eigenvalue for. \param optimizer The cudaq::optimizer to use for iteratively
/// searching for
///        the minimal eigenvalue of \p H.
/// \param n_params The number of variational parameters in the ansatz quantum
///        kernel callable.
/// \param argsMapper A custom callable type with function type
///        `std::tuple<Args...>(std::vector<double>&)` that takes the parameter
///        vector as input and returns a tuple representing the arguments
///        required for evaluation of the quantum kernel.
/// \returns The optimal value and corresponding parameters as a
///        cudaq::optimization_result (std::tuple<double,std::vector<double>>)
///
/// \details Given a quantum kernel of type `void(Args...)`,
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian cudaq::spin_op
/// \p H. This function will use the custom ArgMapper to map input variational
/// parameters to a tuple for use in evaluating the kernel function.
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/builder.h>
/// #include <cudaq/algorithm.h>
/// #include <cudaq/optimizers.h>
/// ...
/// auto [kernel, theta, phi] = cudaq::make_kernel<double, double>();
/// auto q = kernel.qalloc(3);
/// kernel.x(q[0]);
/// kernel.ry(theta, q[1]);
/// kernel.ry(phi, q[2]);
/// kernel.x<ctrl>(q[2], q[0]);
/// kernel.x<ctrl>(q[0], q[1]);
/// kernel.ry(-theta, q[1]);
/// kernel.x<ctrl>(q[0], q[1]);
/// kernel.x<ctrl>(q[1], q[0]);
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///            .21829 * z(0) - 6.125 * z(1) + 9.625 - 9.625 * z(2) -
///            3.913119 * x(1) * x(2) - 3.913119 * y(1) * y(2);
/// cudaq::optimizers::cobyla optimizer;
/// auto [val, params] = cudaq::vqe(kernel, H, optimizer, 1,
///                       [](std::vector<double> x){
///                           return std::make_tuple(x[0], x[1]);
///                      });
/// \endcode
///
template <typename QuantumKernel, typename ArgMapper, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<ArgMapper, std::vector<double>> ||
              std::is_invocable_v<ArgMapper, std::vector<double> &> ||
              std::is_invocable_v<ArgMapper, const std::vector<double> &>>>
optimization_result vqe(std::size_t shots, QuantumKernel &&kernel,
                        cudaq::spin_op H, cudaq::optimizer &optimizer,
                        const int n_params, ArgMapper &&argsMapper) {
  if (optimizer.requiresGradients()) {
    throw std::invalid_argument(
        "Provided cudaq::optimizer requires gradients. "
        "Please provide a cudaq::gradient instance. Make sure the gradient is "
        "aware of the ArgMapper.");
  }
  if (cudaq::get_platform().get_remote_capabilities().vqe)
    __internal__::print_arg_mapper_warning();

  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    auto args = argsMapper(x);
    double energy = std::apply(
        [&](auto &&...arg) -> double {
          return cudaq::observe(shots, kernel, H, arg...);
        },
        args);
    return energy;
  });
}

///
/// \brief Compute the minimal eigenvalue of \p H with VQE with a kernel
///        callable with non-trivial (not `std::vector<double>`) argument
///        structure and a specified gradient strategy.
///
/// \param kernel The quantum callable with non-trivial function
/// signature. \param gradient a sub-type of cudaq::gradient that uses extra QPU
/// calls
///        to compute all elements of the input grad std::vector<double>
///        reference.
/// \param H The hermitian cudaq::spin_op to compute the minimal eigenvalue for.
/// \param optimizer The cudaq::optimizer to use for iteratively searching for
///        the minimal eigenvalue of \p H.
/// \param n_params The number of variational parameters in the ansatz quantum
///        kernel callable.
/// \param argsMapper A custom callable type with function type
///        `std::tuple<Args...>(std::vector<double>&)` that takes the parameter
///        vector as input and returns a tuple representing the arguments
///        required for evaluation of the quantum kernel.
/// \returns The optimal value and corresponding parameters as a
///        cudaq::optimization_result (std::tuple<double,std::vector<double>>)
///
/// \details Given a quantum kernel of type `void(Args...)`,
/// run the variational quantum eigensolver routine to compute
/// the minimum eigenvalue of the specified hermitian cudaq::spin_op
/// \p H. This function will use the custom ArgMapper to map input variational
/// parameters to a tuple for use in evaluating the kernel function.
///
template <typename QuantumKernel, typename ArgMapper, typename... Args,
          typename = std::enable_if_t<
              std::is_invocable_v<ArgMapper, std::vector<double>> ||
              std::is_invocable_v<ArgMapper, std::vector<double> &> ||
              std::is_invocable_v<ArgMapper, const std::vector<double> &>>>
optimization_result vqe(QuantumKernel &&kernel, cudaq::gradient &gradient,
                        cudaq::spin_op H, cudaq::optimizer &optimizer,
                        const int n_params, ArgMapper &&argsMapper) {
  bool requiresGrad = optimizer.requiresGradients();
  if (cudaq::get_platform().get_remote_capabilities().vqe)
    __internal__::print_arg_mapper_warning();

  return optimizer.optimize(n_params, [&](const std::vector<double> &x,
                                          std::vector<double> &grad_vec) {
    auto args = argsMapper(x);
    double energy = std::apply(
        [&](auto &&...arg) -> double {
          return cudaq::observe(kernel, H, arg...);
        },
        args);
    if (requiresGrad) {
      gradient.compute(x, grad_vec, H, energy);
    }
    return energy;
  });
}

} // namespace cudaq
