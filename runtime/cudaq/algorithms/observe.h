/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/ObserveResult.h"
#include "cudaq/algorithms/broadcast.h"
#include "cudaq/concepts.h"
#include "cudaq/host_config.h"
#include "cudaq/operators.h"
#include <functional>
#include <ranges>
#include <type_traits>
#include <vector>

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

namespace parallel {
/// @brief Multi-GPU Multi-Node (MPI)
/// Distribution Type for observe
struct mpi {};

/// @brief Single node, multi-GPU
struct thread {};

} // namespace parallel

/// @brief Define a combined sample function validation concept.
/// These concepts provide much better error messages than old-school SFINAE
template <typename QuantumKernel, typename... Args>
concept ObserveCallValid =
    ValidArgumentsPassed<QuantumKernel, Args...> &&
    HasVoidReturnType<std::invoke_result_t<QuantumKernel, Args...>>;

/// @brief Observe options to provide as an argument to the `observe()`,
/// `async_observe()` functions.
/// @param shots number of shots to run for the given kernel, or -1 if not
/// applicable.
/// @param noise noise model to use for the sample operation
/// @param num_trajectories is the optional number of trajectories to be used
/// when computing the expectation values in the presence of noise. This
/// parameter is only applied to simulation backends that support noisy
/// simulation of trajectories.
struct observe_options {
  int shots = -1;
  cudaq::noise_model noise;
  std::optional<std::size_t> num_trajectories;
};

namespace details {

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the `spin_op`
/// observation process.
template <typename KernelFunctor>
std::optional<observe_result>
runObservation(KernelFunctor &&k, const cudaq::spin_op &H,
               quantum_platform &platform, int shots,
               const std::string &kernelName, std::size_t qpu_id = 0,
               details::future *futureResult = nullptr,
               std::size_t batchIteration = 0, std::size_t totalBatchIters = 0,
               std::optional<std::size_t> numTrajectories = {}) {
  auto ctx = std::make_unique<ExecutionContext>("observe", shots, qpu_id);
  ctx->kernelName = kernelName;
  ctx->spin = cudaq::spin_op::canonicalize(H);
  if (shots > 0)
    ctx->shots = shots;

  if (numTrajectories.has_value())
    ctx->numberTrajectories = *numTrajectories;

  ctx->batchIteration = batchIteration;
  ctx->totalIterations = totalBatchIters;

  // Indicate that this is an asynchronous execution
  ctx->asyncExec = futureResult != nullptr;

  platform.set_exec_ctx(ctx.get());
  try {
    k();
  } catch (...) {
    platform.reset_exec_ctx();
    throw;
  }
  platform.reset_exec_ctx();

  // If this is an asynchronous execution, we need
  // to store the `cudaq::details::future`
  if (futureResult) {
    *futureResult = ctx->futureResult;
    return std::nullopt;
  }

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
    for (const auto &term : ctx->spin.value()) {
      if (term.is_identity())
        sum += term.evaluate_coefficient().real();
      else
        sum += data.expectation(term.get_term_id()) *
               term.evaluate_coefficient().real();
    }
    expectationValue = sum;
  }

  return observe_result(expectationValue, ctx->spin.value(), data);
}

/// @brief Take the input KernelFunctor (a lambda that captures runtime
/// arguments and invokes the quantum kernel) and invoke the `spin_op`
/// observation process asynchronously
template <typename KernelFunctor>
auto runObservationAsync(KernelFunctor &&wrappedKernel, const spin_op &H,
                         quantum_platform &platform, int shots,
                         const std::string &kernelName,
                         std::size_t qpu_id = 0) {

  if (qpu_id >= platform.num_qpus()) {
    throw std::invalid_argument("Provided qpu_id " + std::to_string(qpu_id) +
                                " is invalid (must be < " +
                                std::to_string(platform.num_qpus()) +
                                " i.e. platform.num_qpus())");
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
      [&, H, qpu_id, shots, kernelName,
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
/// QPU index and the `spin_op` chunk and returns an `async_observe_result`.
inline auto distributeComputations(
    std::function<async_observe_result(std::size_t, const spin_op &)>
        &&asyncLauncher,
    const spin_op &H, std::size_t nQpus) {

  auto op = cudaq::spin_op::canonicalize(H);
  // Distribute the given spin_op into subsets for each QPU
  auto spins = op.distribute_terms(nQpus);

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
    result += incomingData.expectation();
    data += incomingData;
  }

  return observe_result(result, op, data);
}

} // namespace details

/// \overload
/// \brief Compute the expected value of `H` with respect to `kernel(Args...)`.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(QuantumKernel &&kernel, const spin_op &H,
                       Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);
  auto kernelName = cudaq::getKernelName(kernel);
  return details::runObservation(
             [&kernel, &args...]() mutable {
               kernel(std::forward<Args>(args)...);
             },
             H, platform, shots, kernelName)
      .value();
}

/// @brief Compute the expected value of every `spin_op` provided in
/// `SpinOpContainer` (a range concept) with respect to `kernel(Args...)`.
/// Return a `std::vector<observe_result>`.
template <typename QuantumKernel, typename SpinOpContainer, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...> &&
           std::ranges::range<SpinOpContainer>
std::vector<observe_result> observe(QuantumKernel &&kernel,
                                    const SpinOpContainer &termList,
                                    Args &&...args) {
  // Here to give a more comprehensive error if the container does not contain
  // values of type spin_op_term.
  typedef typename SpinOpContainer::value_type value_type;
  static_assert(std::is_same_v<spin_op_term, value_type>,
                "term list must be a container of spin_op_term");

  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);
  auto kernelName = cudaq::getKernelName(kernel);

  // Convert all spin_ops to a single summed spin_op
  auto op = cudaq::spin_op::empty();
  for (auto &o : termList)
    op += cudaq::spin_op_term::canonicalize(o);

  // Run the observation
  auto result = details::runObservation(
                    [&kernel, &args...]() mutable {
                      kernel(std::forward<Args>(args)...);
                    },
                    op, platform, shots, kernelName)
                    .value();

  // Convert back to a vector of results
  std::vector<observe_result> results;
  for (const auto &term : op)
    results.emplace_back(result.expectation(term), term, result.counts(term));

  return results;
}

// Doxygen: ignore overloads with `DistributionType`s, preferring the simpler
// ones
/// @cond
/// @brief Compute the expected value of `H` with respect to `kernel(Args...)`.
/// Distribute the work `amongst` available QPUs on the platform in parallel.
/// This distribution can occur on multi-GPU multi-node platforms, multi-GPU
/// single-node platforms, or multi-node no-GPU platforms. Programmers must
/// indicate the distribution type via the corresponding template types
/// (cudaq::mgmn, cudaq::mgsn, cudaq::mn).
template <typename DistributionType, typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(std::size_t shots, QuantumKernel &&kernel,
                       const spin_op &H, Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  // Does platform support parallelism? Need a check here
  if (!platform.supports_task_distribution())
    throw std::runtime_error(
        "The current quantum_platform does not support parallel distribution "
        "of observe() expectation value computations.");

  auto nQpus = platform.num_qpus();
  if constexpr (std::is_same_v<DistributionType, parallel::thread>) {
    if (nQpus == 1)
      printf(
          "[cudaq::observe warning] distributed observe requested but only 1 "
          "QPU available. no speedup expected.\n");
    // Let's distribute the work among the QPUs on this node.
    return details::distributeComputations(
        [&kernel, shots, ... args = std::forward<Args>(args)](
            std::size_t i, const spin_op &op) mutable {
          return observe_async(shots, i, std::forward<QuantumKernel>(kernel),
                               op, std::forward<Args>(args)...);
        },
        H, nQpus);
  } else if (std::is_same_v<DistributionType, parallel::mpi>) {

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
    auto localH = spins[rank].canonicalize();

    // Distribute locally, i.e. to the local nodes QPUs
    auto localRankResult = details::distributeComputations(
        [&kernel, shots, ... args = std::forward<Args>(args)](
            std::size_t i, const spin_op &op) mutable {
          return observe_async(shots, i, std::forward<QuantumKernel>(kernel),
                               op, std::forward<Args>(args)...);
        },
        localH, nQpus);

    // combine all the data via an all_reduce
    auto exp_val = localRankResult.expectation();
    auto globalExpVal = mpi::all_reduce(exp_val, std::plus<double>());
    // we need the canonicalized version of H -
    // maybe we can get it from the context instead?
    cudaq::spin_op canonH;
    for (auto &&terms : spins)
      canonH += std::move(terms);
    return observe_result(globalExpVal, canonH);

  } else
    throw std::runtime_error("Invalid cudaq::par execution type.");
}

template <typename DistributionType, typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(QuantumKernel &&kernel, const spin_op &H,
                       Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto shots = platform.get_shots().value_or(-1);
  return observe<DistributionType>(shots, std::forward<QuantumKernel>(kernel),
                                   H, std::forward<Args>(args)...);
}
/// \endcond

/// \overload
/// \brief Compute the expected value of `H` with respect to `kernel(Args...)`.
/// Specify the number of shots.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(std::size_t shots, QuantumKernel &&kernel,
                       const spin_op &H, Args &&...args) {
  // Run this SHOTS times
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);

  // Does this platform expose more than 1 QPU
  // If so, let's distribute the work among the QPUs
  if (auto nQpus = platform.num_qpus(); nQpus > 1)
    return details::distributeComputations(
        [&kernel, shots, ... args = std::forward<Args>(args)](
            std::size_t i, const spin_op &op) mutable {
          return observe_async(shots, i, std::forward<QuantumKernel>(kernel),
                               op, std::forward<Args>(args)...);
        },
        H, nQpus);

  return details::runObservation(
             [&kernel, &args...]() mutable {
               kernel(std::forward<Args>(args)...);
             },
             H, platform, shots, kernelName)
      .value();
}

/// \brief Compute the expected value of `H` with respect to `kernel(Args...)`.
/// Specify the observation options
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
observe_result observe(const observe_options &options, QuantumKernel &&kernel,
                       const spin_op &H, Args &&...args) {
  auto &platform = cudaq::get_platform();
  auto kernelName = cudaq::getKernelName(kernel);
  auto shots = options.shots;

  platform.set_noise(&options.noise);

  auto ret = details::runObservation(
                 [&kernel, &args...]() mutable {
                   kernel(std::forward<Args>(args)...);
                 },
                 H, platform, shots, kernelName, /*qpu_id=*/0,
                 /*futureResult=*/nullptr,
                 /*batchIteration=*/0,
                 /*totalBatchIters=*/0, options.num_trajectories)
                 .value();

  platform.reset_noise();
  return ret;
}

/// \brief Asynchronously compute the expected value of `H` with respect to
/// `kernel(Args...)`.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
auto observe_async(const std::size_t qpu_id, QuantumKernel &&kernel,
                   const spin_op &H, Args &&...args) {
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
                   QuantumKernel &&kernel, const spin_op &H, Args &&...args) {
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
auto observe_async(QuantumKernel &&kernel, const spin_op &H, Args &&...args) {
  return observe_async(0, std::forward<QuantumKernel>(kernel), H,
                       std::forward<Args>(args)...);
}

/// @overload
/// @brief Run the standard observe functionality over a set of `N`
/// argument packs. For a kernel with signature `void(Args...)`, this
/// function takes as input a set of `vector<Arg>...`, a vector for
/// each argument type in the kernel signature. The vectors must be of
/// equal length, and the `i-th` element of each vector is used `i-th`
/// execution of the standard observe function. Results are collected
/// from the execution of every argument set and returned.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
std::vector<observe_result> observe(QuantumKernel &&kernel, const spin_op &H,
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
    auto ret = details::runObservation(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   H, platform, shots, kernelName, qpuId, nullptr, counter, N)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  return details::broadcastFunctionOverArguments<observe_result, Args...>(
      numQpus, platform, functor, params);
}

/// @overload
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
std::vector<observe_result> observe(std::size_t shots, QuantumKernel &&kernel,
                                    const spin_op &H,
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
    auto ret = details::runObservation(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
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
/// allows the `observe_options` to be specified.
template <typename QuantumKernel, typename... Args>
  requires ObserveCallValid<QuantumKernel, Args...>
std::vector<observe_result> observe(cudaq::observe_options &options,
                                    QuantumKernel &&kernel, const spin_op &H,
                                    ArgumentSet<Args...> &&params) {
  // Get the platform and query the number of quantum computers
  auto &platform = cudaq::get_platform();
  auto numQpus = platform.num_qpus();
  auto shots = options.shots;

  platform.set_noise(&options.noise);

  // Create the functor that will broadcast the observations across
  // all requested argument sets provided.
  details::BroadcastFunctorType<observe_result, Args...> functor =
      [&](std::size_t qpuId, std::size_t counter, std::size_t N,
          Args &...singleIterParameters) -> observe_result {
    auto kernelName = cudaq::getKernelName(kernel);
    auto ret = details::runObservation(
                   [&kernel, &singleIterParameters...]() mutable {
                     kernel(std::forward<Args>(singleIterParameters)...);
                   },
                   H, platform, shots, kernelName, qpuId, nullptr, counter, N,
                   options.num_trajectories)
                   .value();
    return ret;
  };

  // Broadcast the executions and return the results.
  auto ret = details::broadcastFunctionOverArguments<observe_result, Args...>(
      numQpus, platform, functor, params);

  platform.reset_noise();
  return ret;
}
} // namespace cudaq
