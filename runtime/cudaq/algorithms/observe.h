/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/spin_op.h"

#include <functional>
#include <type_traits>
#include <vector>

#include "common/ExecutionContext.h"
#include "common/ObserveResult.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"

namespace cudaq {

namespace mpi {
int rank();
int num_ranks();
bool is_initialized();
template <typename T, typename Func>
T all_reduce(const T &, const Func &);
} // namespace mpi

/// @brief Return type for asynchronous observation.
using async_observe_result = async_result<observe_result>;

namespace par {
/// @brief Multi-GPU Multi-Node (MPI)
/// Distribution Type for observe
struct mpi {};

/// @brief Single node, multi-GPU
struct thread {};

} // namespace par

/// @brief Define a combined sample function validation concept.
/// These concepts provide much better error messages than old-school SFINAE
template <typename QuantumKernel, typename... Args>
concept ObserveCallValid =
    ValidArgumentsPassed<QuantumKernel, Args...> &&
    HasVoidReturnType<std::invoke_result_t<QuantumKernel, Args...>>;

namespace details {

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the `spin_op`
/// observation process.
template <typename KernelFunctor>
std::optional<observe_result>
runObservation(KernelFunctor &&k, cudaq::spin_op &h, quantum_platform &platform,
               int shots, const std::string &kernelName, std::size_t qpu_id = 0,
               details::future *futureResult = nullptr,
               std::size_t batchIteration = 0,
               std::size_t totalBatchIters = 0) {
  auto ctx = std::make_unique<ExecutionContext>("observe", shots);
  ctx->kernelName = kernelName;
  ctx->spin = &h;
  if (shots > 0)
    ctx->shots = shots;

  ctx->batchIteration = batchIteration;
  ctx->totalIterations = totalBatchIters;

  // Indicate that this is an asynchronous execution
  ctx->asyncExec = futureResult != nullptr;

  platform.set_current_qpu(qpu_id);
  platform.set_exec_ctx(ctx.get(), qpu_id);

  k();

  // If this is an asynchronous execution, we need
  // to store the `cudaq::details::future`
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
  // precomputed, if so grab it and set it so the client gets it
  if (ctx->expectationValue.has_value())
    expectationValue = ctx->expectationValue.value_or(0.0);
  else {
    // If not, we have everything we need to compute it.
    double sum = 0.0;
    h.for_each_term([&](spin_op &term) {
      if (term.is_identity())
        sum += term.get_coefficient().real();
      else
        sum += data.exp_val_z(term.to_string(false)) *
               term.get_coefficient().real();
    });

    expectationValue = sum;
  }

  return observe_result(expectationValue, h, data);
}

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the `spin_op`
/// observation process asynchronously
template <typename KernelFunctor>
auto runObservationAsync(KernelFunctor &&wrappedKernel, spin_op &H,
                         quantum_platform &platform, int shots,
                         const std::string &kernelName,
                         std::size_t qpu_id = 0) {

  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument(
        "Provided qpu_id is invalid (must be <= to platform.num_qpus()).");
  }

  // Could be that the platform we are running on is
  // remotely hosted, if so, we can't do asynchronous execution with a
  // separate thread, the separate thread is the remote server invocation
  if (platform.is_remote(qpu_id)) {
    // In this case, everything we need can be dumped into a details::future
    // type. Just return that wrapped in an async_result
    details::future futureResult;
    details::runObservation(std::forward<KernelFunctor>(wrappedKernel), H,
                            platform, shots, kernelName, qpu_id, &futureResult);
    return async_observe_result(std::move(futureResult), &H);
  }

  // If the platform is not remote, then we can handle asynchronous execution
  // via a new worker thread.
  KernelExecutionTask task(
      [&, qpu_id, shots, kernelName,
       kernel = std::forward<KernelFunctor>(wrappedKernel)]() mutable {
        return details::runObservation(kernel, H, platform, shots, kernelName,
                                       qpu_id)
            .value()
            .raw_data();
      });

  return async_observe_result(
      details::future(platform.enqueueAsyncTask(qpu_id, task)), &H);
}

/// @brief Distribute the expectation value computations among the
/// available platform QPUs. The `asyncLauncher` functor takes as input the
/// qpu index and the `spin_op` chunk and returns an `async_observe_result`.
inline auto distributeComputations(
    std::function<async_observe_result(std::size_t, spin_op &)> &&asyncLauncher,
    spin_op &H, std::size_t nQpus) {

  // Distribute the given spin_op into subsets for each QPU
  auto spins = H.distribute_terms(nQpus);

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
    auto incomingData = res.raw_data();
    result += incomingData.exp_val_z();
    data += incomingData;
  }

  return observe_result(result, H, data);
}

} // namespace details

/// \brief Compute the expected value of `H` with respect to `kernel(Args...)`.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(QuantumKernel &&kernel, spin_op H, Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runObservation(
             [&kernel, ... args = std::forward<Args>(args)]() mutable {
               kernel(args...);
             },
             H, platform, shots, kernelName)
      .value();
}

/// @brief Compute the expected value of `H` with respect to `kernel(Args...)`.
/// Distribute the work amongst available QPUs on the platform in parallel. This
/// distribution can occur on multi-gpu multi-node platforms, multi-gpu
/// single-node platforms, or multi-node no-gpu platforms. Programmers must
/// indicate the distribution type via the corresponding template types
/// (cudaq::mgmn, cudaq::mgsn, cudaq::mn).
template <typename DistributionType, typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(std::size_t shots, QuantumKernel &&kernel, spin_op H,
                       Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  // Does platform support parallelism? Need a check here
  if (!platform.supports_task_distribution())
    throw std::runtime_error(
        "The current quantum_platform does not support parallel distribution "
        "of observe() expectation value computations.");

  auto nQpus = platform.num_qpus();
  if constexpr (std::is_same_v<DistributionType, par::thread>) {
    if (nQpus == 1)
      printf(
          "[cudaq::observe warning] distributed observe requested but only 1 "
          "QPU available. no speedup expected.\n");
    // Let's distribute the work among the QPUs on this node
    return details::distributeComputations(
        [&kernel, ... args = std::forward<Args>(args)](std::size_t i,
                                                       spin_op &op) mutable {
          return observe_async(i, std::forward<QuantumKernel>(kernel), op,
                               std::forward<Args>(args)...);
        },
        H, nQpus);
  } else if (std::is_same_v<DistributionType, par::mpi>) {

    // This is an MPI distribution, where each node has N GPUs.
    if (!mpi::is_initialized())
      throw std::runtime_error("Cannot use mgmn multi-node observe() without "
                               "MPI (did you initialize MPI?).");

    // Note - For MGMN, we assume that nQpus == num visible GPUs for this local
    // rank.

    // FIXME, how do we handle an mpi run where each rank
    // is targeting the same GPU? Should we even allow that?

    // Get the rank and the number of ranks
    auto rank = mpi::rank();
    auto nRanks = mpi::num_ranks();

    // Each rank gets a subset of the spin terms
    auto spins = H.distribute_terms(nRanks);

    // Get this rank's set of spins to compute
    auto localH = spins[rank];

    // Distribute locally, i.e. to the local nodes QPUs
    auto localRankResult = details::distributeComputations(
        [&kernel, ... args = std::forward<Args>(args)](std::size_t i,
                                                       spin_op &op) mutable {
          return observe_async(i, std::forward<QuantumKernel>(kernel), op,
                               std::forward<Args>(args)...);
        },
        localH, nQpus);

    // combine all the data via an all_reduce
    auto exp_val = localRankResult.exp_val_z();
    auto globalExpVal = mpi::all_reduce(exp_val, std::plus<double>());
    return observe_result(globalExpVal, H);

  } else
    throw std::runtime_error("Invalid cudaq::par execution type.");
}

template <typename DistributionType, typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(QuantumKernel &&kernel, spin_op H, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);
  return observe<DistributionType>(shots, std::forward<QuantumKernel>(kernel),
                                   H, std::forward<Args>(args)...);
}

/// \brief Compute the expected value of `H` with respect to `kernel(Args...)`.
/// Specify the number of shots.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(std::size_t shots, QuantumKernel &&kernel, spin_op H,
                       Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

  // Does this platform expose more than 1 QPU
  // If so, let's distribute the work among the QPUs
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
             H, platform, shots, kernelName)
      .value();
}

/// \brief Asynchronously compute the expected value of `H` with respect to
/// `kernel(Args...)`.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(const std::size_t qpu_id, QuantumKernel &&kernel, spin_op &H,
                   Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);
  auto kernelName = cudaq::getKernelName(kernel);

  return details::runObservationAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      H, platform, shots, kernelName, qpu_id);
}

/// \brief Asynchronously compute the expected value of `H` with respect to
/// `kernel(Args...)`. Specify the shots.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(std::size_t shots, std::size_t qpu_id,
                   QuantumKernel &&kernel, spin_op &H, Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

  return details::runObservationAsync(
      [&kernel, ... args = std::forward<Args>(args)]() mutable {
        kernel(std::forward<Args>(args)...);
      },
      H, platform, shots, kernelName, qpu_id);
}

/// \brief Asynchronously compute the expected value of \p H with respect to
/// `kernel(Args...)`. Default to the `0-th` QPU.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(QuantumKernel &&kernel, spin_op &H, Args &&...args) {
  return observe_async(0, std::forward<QuantumKernel>(kernel), H,
                       std::forward<Args>(args)...);
}

/// @brief Run the standard observe functionality over a set of `N`
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the `i-th` element of each vector is used `i-th`
/// execution of the standard observe function. Results are collected
/// from the execution of every argument set and returned.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
std::vector<observe_result> observe_n(QuantumKernel &&kernel, spin_op H,
                                      ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of quantum computers
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the observations across
  // all requested argument sets provided.
  details::BroadcastFunctorType<observe_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> observe_result {
    auto shots = platform.get_shots().value_or(-1);
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret =
        details::runObservation(
            [&kernel, ... args = std::forward<decltype(singleIterParameters)>(
                          singleIterParameters)]() mutable { kernel(args...); },
            H, platform, shots, kernelName, qpuId, nullptr, counter, N)
            .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<observe_result, Args...>(
      numQpus, platform, functor, params);
}

/// @brief Run the standard observe functionality over a set of N
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the `i-th` element of each vector is used `i-th`
/// execution of the standard observe function. Results are collected
/// from the execution of every argument set and returned. This overload
/// allows the number of circuit executions (shots) to be specified.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
std::vector<observe_result> observe_n(std::size_t shots, QuantumKernel &&kernel,
                                      spin_op H,
                                      ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of quantum computers
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();

  // Create the functor that will broadcast the observations across
  // all requested argument sets provided.
  details::BroadcastFunctorType<observe_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> observe_result {
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret =
        details::runObservation(
            [&kernel, ... args = std::forward<decltype(singleIterParameters)>(
                          singleIterParameters)]() mutable { kernel(args...); },
            H, platform, shots, kernelName, qpuId, nullptr, counter, N)
            .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<observe_result, Args...>(
      numQpus, platform, functor, params);
}
} // namespace cudaq
