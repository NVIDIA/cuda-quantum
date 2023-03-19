/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include <cudaq/spin_op.h>

#include <functional>
#include <type_traits>
#include <vector>

#include "common/ExecutionContext.h"
#include "common/ObserveResult.h"
#include "cudaq/concepts.h"
#include "cudaq/platform.h"
#include "cudaq/platform/quantum_platform.h"

namespace cudaq {

/// @brief Return type for asynchronous observation.
using async_observe_result = async_result<observe_result>;

/// @brief Define a combined sample function validation concept.
/// These concepts provide much better error messages than old-school SFINAE
template <typename QuantumKernel, typename... Args>
concept ObserveCallValid =
    ValidArgumentsPassed<QuantumKernel, Args...> &&
    HasVoidReturnType<std::invoke_result_t<QuantumKernel, Args...>>;

namespace details {

/// @brief Take the input KernelFunctor (a lambda that captures runtime args and
/// invokes the quantum kernel) and invoke the spin_op observation process.
template <typename KernelFunctor>
std::optional<observe_result>
runObservation(KernelFunctor &&k, cudaq::spin_op &h, quantum_platform &platform,
               int shots, std::size_t qpu_id = 0,
               details::future *futureResult = nullptr) {
  auto ctx = std::make_unique<ExecutionContext>("observe", shots);
  ctx->spin = &h;
  if (shots > 0)
    ctx->shots = shots;

  // Indicate that this is an async exec
  ctx->asyncExec = futureResult != nullptr;

  platform.set_current_qpu(qpu_id);
  platform.set_exec_ctx(ctx.get(), qpu_id);

  k();

  // If this is an async execution, we need
  // to store the cudaq::details::future
  if (futureResult) {
    *futureResult = ctx->futureResult;
    return std::nullopt;
  }

  platform.reset_exec_ctx(qpu_id);

  // Extract the results
  sample_result data;
  double expectationValue;
  data = ctx->result;

  // It is possible for the expectation value to be
  // pre computed, if so grab it and set it so the client gets it
  if (ctx->expectationValue.has_value())
    expectationValue = ctx->expectationValue.value_or(0.0);
  else {
    // If not, we have everything we need to compute it.
    double sum = 0.0;
    for (std::size_t i = 0; i < h.n_terms(); i++) {
      auto term = h[i];
      if (term.is_identity())
        sum += term.get_coefficients()[0].real();
      else
        sum += data.exp_val_z(term.to_string(false)) *
               term.get_coefficients()[0].real();
    }
    expectationValue = sum;
  }

  return observe_result(expectationValue, h, data);
}

/// @brief Take the input KernelFunctor (a lambda that captures runtime args and
/// invokes the quantum kernel) and invoke the spin_op observation process
/// asynchronously
template <typename KernelFunctor>
auto runObservationAsync(KernelFunctor &&wrappedKernel, spin_op &H,
                         quantum_platform &platform, int shots,
                         std::size_t qpu_id = 0) {

  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");
  }

  // Could be that the platform we are running on is
  // remotely hosted, if so, we can't do async execution with a
  // separate thread, the separate thread is the remote server invocation
  if (platform.is_remote(qpu_id)) {
    // In this case, everything we need can be dumped into a details::future
    // type. Just return that wrapped in an async_result
    details::future futureResult;
    details::runObservation(std::forward<KernelFunctor>(wrappedKernel), H,
                            platform, shots, qpu_id, &futureResult);
    return async_observe_result(std::move(futureResult), &H);
  }

  // If the platform is not remote, then we can handle async execution via
  // a new worker thread.
  KernelExecutionTask task(
      [&, qpu_id, shots,
       kernel = std::forward<KernelFunctor>(wrappedKernel)]() mutable {
        return details::runObservation(kernel, H, platform, shots, qpu_id)
            .value()
            .raw_data();
      });

  return async_observe_result(
      details::future(platform.enqueueAsyncTask(qpu_id, task)), &H);
}

/// @brief Distribute the expectation value computations amongst the
/// available platform QPUs. The asyncLauncher functor takes as input the
/// qpu index and the spin_op chunk and returns an async_observe_result.
inline auto distributeComputations(
    std::function<async_observe_result(std::size_t, spin_op &)> &&asyncLauncher,
    spin_op &H, std::size_t nQpus) {

  // Calculate how many terms we can equally divide amongst the qpus
  auto nTermsPerQPU = H.n_terms() / nQpus + (H.n_terms() % nQpus != 0);

  // Slice the given spin_op into subsets for each QPU
  std::vector<spin_op> spins;
  for (auto uniqueQpuId : cudaq::range(nQpus)) {
    auto lowerBound = uniqueQpuId * nTermsPerQPU;
    spins.emplace_back(H.slice(lowerBound, nTermsPerQPU));
  }

  // Observe each sub-spin_op asynchronously
  std::vector<async_observe_result> asyncResults;
  for (std::size_t i = 0; auto &op : spins) {
    asyncResults.emplace_back(asyncLauncher(i, op));
    i++;
  }

  // Wait for the results, should be executing
  // in parallel on the available QPUs.
  double result = 0.0;
  sample_result data;
  for (auto &asyncResult : asyncResults) {
    auto res = asyncResult.get();
    result += res.exp_val_z();
    auto incomingData = res.raw_data();
    data += incomingData;
  }

  return observe_result(result, H, data);
}

} // namespace details

///
/// \brief Compute the expected value of \p H with respect to kernel(Args...).
///
/// \tparam Args The variadic list of argument types for this kernel. Usually
///         can be deduced by the compiler.
/// \param kernel The instantiated ansatz callable, a CUDA Quantum kernel,
///         cannot contain measure statements.
/// \param H The hermitian cudaq::spin_op to compute the expected value for.
/// \param args The variadic concrete arguments for evaluation of the kernel.
/// \returns exp The expected value <ansatz(args...)|H|ansatz<args...)>.
///
/// \details Given a CUDA Quantum kernel of general callable type
///          void(Args...), compute the expectation value of \p H at the
///          concrete kernel parameter args...
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/algorithm.h>
/// ...
/// struct ansatz {
///   void operator(double  x) __qpu__ {
///     cudaq::qreg q(2);
///     x(q[0]);
///     ry(x, q[1]);
///     x<ctrl>(q[1],q[0]);
///   }
/// };
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// double theta = .59;
/// auto exp_val = cudaq::observe(ansatz{}, H, theta);
/// \endcode
///
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(QuantumKernel &&kernel, spin_op H, Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);

  // Does this platform expose more than 1 QPU
  // If so, let's distribute the work amongst the QPUs
  if (auto nQpus = platform.num_qpus(); nQpus > 1)
    return details::distributeComputations(
        [&kernel, ... args = std::forward<Args>(args)](std::size_t i,
                                                       spin_op &op) mutable {
          return observe_async(i, std::forward<QuantumKernel>(kernel), op,
                               std::forward<Args>(args)...);
        },
        H, nQpus);

  return details::runObservation(
             [&kernel, ... args = std::forward<Args>(args)]() mutable {
               kernel(args...);
             },
             H, platform, shots)
      .value();
}

///
/// \brief Compute the expected value of \p H with respect to kernel(Args...).
///
/// \tparam Args The variadic list of argument types for this kernel. Usually
///         can be deduced by the compiler.
/// \param shots The number of samples to collect
/// \param kernel The instantiated ansatz callable, a CUDA Quantum kernel,
///         cannot contain measure statements.
/// \param H The hermitian cudaq::spin_op to compute the expected value for.
/// \param args The variadic concrete arguments for evaluation of the kernel.
/// \returns exp The expected value <ansatz(args...)|H|ansatz<args...)>.
///
/// \details Given a CUDA Quantum kernel of general callable type
///          void(Args...), compute the expectation value of \p H at the
///          concrete kernel parameter args...
///
/// Usage:
/// \code{.cpp}
/// #include <cudaq.h>
/// #include <cudaq/algorithm.h>
/// ...
/// struct ansatz {
///   void operator(double  x) __qpu__ {
///     cudaq::qreg q(2);
///     x(q[0]);
///     ry(x, q[1]);
///     x<ctrl>(q[1],q[0]);
///   }
/// };
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// double theta = .59;
/// auto exp_val = cudaq::observe(ansatz{}, H, theta);
/// \endcode
///
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(std::size_t shots, QuantumKernel &&kernel, spin_op H,
                       Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();

  // Does this platform expose more than 1 QPU
  // If so, let's distribute the work amongst the QPUs
  if (auto nQpus = platform.num_qpus(); nQpus > 1)
    return details::distributeComputations(
        [&kernel, ... args = std::forward<Args>(args)](std::size_t i,
                                                       spin_op &op) mutable {
          return observe_async(i, std::forward<QuantumKernel>(kernel), op,
                               std::forward<Args>(args)...);
        },
        H, nQpus);

  return details::runObservation(
             [&kernel, ... args = std::forward<Args>(args)]() mutable {
               kernel(args...);
             },
             H, platform, shots)
      .value();
}

///
/// \brief Asynchronously compute the expected value of \p H with respect to
/// kernel(Args...).
///
/// \tparam Args The variadic list of argument types for this kernel. Usually
///         can be deduced by the compiler.
/// \param qpu_id The QPU id to run asynchronously on.
/// \param kernel The instantiated ansatz callable, a CUDA Quantum kernel,
///         cannot contain measure statements.
/// \param H The hermitian cudaq::spin_op to compute the expected value for.
/// \param args The variadic concrete arguments for evaluation of the kernel.
/// \returns exp The expected value <ansatz(args...)|H|ansatz<args...)> as a
/// std::future.
///
/// \details Given a CUDA Quantum kernel of general callable type
///          void(Args...), compute the expectation value of \p H at the
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// double theta = .59;
/// auto exp_val = cudaq::observe(ansatz{}, H, theta);
///
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(const std::size_t qpu_id, QuantumKernel &&kernel, spin_op &H,
                   Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);

  return details::runObservationAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      H, platform, shots, qpu_id);
}

///
/// \brief Asynchronously compute the expected value of \p H with respect to
/// kernel(Args...).
///
/// \tparam Args The variadic list of argument types for this kernel. Usually
///         can be deduced by the compiler.
/// \param qpu_id The QPU id to run asynchronously on.
/// \param kernel The instantiated ansatz callable, a CUDA Quantum kernel,
///         cannot contain measure statements.
/// \param H The hermitian cudaq::spin_op to compute the expected value for.
/// \param args The variadic concrete arguments for evaluation of the kernel.
/// \returns exp The expected value <ansatz(args...)|H|ansatz<args...)> as a
/// std::future.
///
/// \details Given a CUDA Quantum kernel of general callable type
///          void(Args...), compute the expectation value of \p H at the
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// double theta = .59;
/// auto exp_val = cudaq::observe(ansatz{}, H, theta);
///
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(std::size_t shots, std::size_t qpu_id,
                   QuantumKernel &&kernel, spin_op &H, Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();

  return details::runObservationAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      H, platform, shots, qpu_id);
}

///
/// \brief Asynchronously compute the expected value of \p H with respect to
/// kernel(Args...). Default to the 0th QPU
///
/// \tparam Args The variadic list of argument types for this kernel. Usually
///         can be deduced by the compiler.
/// \param kernel The instantiated ansatz callable, a CUDA Quantum kernel,
///         cannot contain measure statements.
/// \param H The hermitian cudaq::spin_op to compute the expected value for.
/// \param args The variadic concrete arguments for evaluation of the kernel.
/// \returns exp The expected value <ansatz(args...)|H|ansatz<args...)> as a
/// std::future.
///
/// \details Given a CUDA Quantum kernel of general callable type
///          void(Args...), compute the expectation value of \p H at the
/// auto H = 5.907 - 2.1433 * x(0) * x(1) - 2.1433 * y(0) * y(1) +
///           .21829 * z(0) - 6.125 * z(1);
/// double theta = .59;
/// auto exp_val = cudaq::observe(ansatz{}, H, theta);
///
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(QuantumKernel &&kernel, spin_op &H, Args &&...args) {
  return observe_async(0, std::forward<QuantumKernel>(kernel), H,
                       std::forward<Args>(args)...);
}
} // namespace cudaq
