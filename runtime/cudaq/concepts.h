/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/platform.h"
#include <type_traits>

namespace cudaq {

/// @brief Sample or observe calls need to have valid trailing runtime arguments
/// Define a concept that ensures the given kernel can be called with
/// the provided runtime arguments types.
template <typename QuantumKernel, typename... Args>
concept ValidArgumentsPassed = std::is_invocable_v<QuantumKernel, Args...>;

/// @brief All kernels passed to sample or observe must have a void return type.
template <typename ReturnType>
concept HasVoidReturnType = std::is_void_v<ReturnType>;

/// @brief An ArgumentSet is a tuple of vectors of general
/// arguments to a CUDA Quantum kernel. The ith vector of the tuple
/// corresponds to the ith argument of the kernel. The jth element of
/// the ith vector corresponds to the jth batch of arguments to evaluate
/// the kernel at.
template <typename... Args>
using ArgumentSet = std::tuple<std::vector<Args>...>;

/// @brief Create a new ArgumentSet from a variadic list of
/// vectors of general args.
template <typename... Args>
auto make_argset(const std::vector<Args> &...args) {
  return std::make_tuple(args...);
}

namespace details {
template <typename ReturnType, typename... Args>
using BroadcastFunctorType = const std::function<ReturnType(
    std::size_t, std::size_t, std::size_t, Args &...)>;

/// @brief
template <typename R, typename... Args>
std::vector<R>
broadcastFunctionOverArguments(std::size_t numQpus, quantum_platform &pk,
                               BroadcastFunctorType<R, Args...> &apply,
                               ArgumentSet<Args...> &params) {
  // std::size_t numQpus = 1;

  // Assert all arg vectors are the same size
  auto N = std::get<0>(params).size();
  cudaq::tuple_for_each(params, [&](auto &&element) {
    if (element.size() != N)
      throw std::runtime_error("Invalid argument set to broadcast function "
                               "over - vector sizes not the same.");
  });

  std::vector<std::future<std::vector<R>>> futures;
  auto nExecsPerQpu = N / numQpus + (N % numQpus != 0);
  for (std::size_t qpuId = 0; qpuId < numQpus; qpuId++) {
    std::promise<std::vector<R>> promise;
    futures.emplace_back(promise.get_future());
    std::function<void()> functor = detail::make_copyable_function(
        [&params, &apply, &pk, qpuId, nExecsPerQpu,
         p = std::move(promise)]() mutable {
          auto lowerBound = qpuId * nExecsPerQpu;
          auto upperBound = lowerBound + nExecsPerQpu;
          std::vector<R> results;

          // Loop over all sets of arguments, the ith element of each vector
          // in the ArgumentSet tuple
          for (std::size_t i = lowerBound, counter = 0; i < upperBound; i++) {
            // Construct the current set of arguments as a new tuple
            // We want a tuple so we can use std::apply with the
            // existing observe() functions.
            std::tuple<std::size_t, std::size_t, std::size_t, Args...>
                currentArgs;
            std::get<0>(currentArgs) = qpuId;
            std::get<1>(currentArgs) = counter;
            std::get<2>(currentArgs) = nExecsPerQpu;
            counter++;

            cudaq::tuple_for_each_with_idx(
                params, [&]<typename IDX_TYPE>(auto &&element, IDX_TYPE &&idx) {
                  std::get<IDX_TYPE::value + 3>(currentArgs) = element[i];
                });

            // Call observe with the current set of arguments (provided as a
            // tuple)
            auto result = std::apply(apply, currentArgs);

            // Store the result.
            results.push_back(result);
          }
          // printf("Computed %lu results\n", results.size());
          p.set_value(results);
        });

    // printf("Enqueue on qpu %lu\n", qpuId);
    pk.enqueueAsyncTask(qpuId, functor);
  }

  std::vector<R> allResults;
  for (auto &f : futures) {
    auto res = f.get();
    // printf("Get future of size %lu\n", res.size());
    allResults.insert(allResults.end(), res.begin(), res.end());
  }

  return allResults;
}
} // namespace details
} // namespace cudaq
