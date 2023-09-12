/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/platform.h"

namespace cudaq {

void set_random_seed(std::size_t);
std::size_t get_random_seed();

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

/// @brief Given the input BroadcastFunctorType, apply it to all
/// argument sets in the provided ArgumentSet `params`. Distribute the
/// work over the provided number of QPUs.
template <typename ResType, typename... Args>
std::vector<ResType>
broadcastFunctionOverArguments(std::size_t numQpus, quantum_platform &platform,
                               BroadcastFunctorType<ResType, Args...> &apply,
                               ArgumentSet<Args...> &params) {
  using FutureCollection = std::vector<std::future<std::vector<ResType>>>;

  // Assert all arg vectors are the same size
  auto N = std::get<0>(params).size();
  auto nExecsPerQpu = N / numQpus + (N % numQpus != 0);

  // Validate the input deck
  cudaq::tuple_for_each(params, [&](auto &&element) {
    if (element.size() != N)
      throw std::runtime_error("Invalid argument set to broadcast function "
                               "over - vector sizes not the same.");
  });

  // Fetch the thread-specific seed outside the functor and then pass it inside.
  std::size_t seed = cudaq::get_random_seed();

  FutureCollection futures;
  for (std::size_t qpuId = 0; qpuId < numQpus; qpuId++) {
    std::promise<std::vector<ResType>> _promise;
    futures.emplace_back(_promise.get_future());
    std::function<void()> functor = detail::make_copyable_function(
        [&params, &apply, qpuId, nExecsPerQpu, seed,
         promise = std::move(_promise)]() mutable {
          // Compute the lower and upper bounds of the
          // argument set that should be computed on the current QPU
          auto lowerBound = qpuId * nExecsPerQpu;
          auto upperBound = lowerBound + nExecsPerQpu;

          // Store the results
          std::vector<ResType> results;

          // Loop over all sets of arguments, the ith element of each vector
          // in the ArgumentSet tuple
          for (std::size_t i = lowerBound, counter = 0; i < upperBound; i++) {
            // Construct the current set of arguments as a new tuple
            // We want a tuple so we can use std::apply with the
            // existing sample()/observe() functions.
            std::tuple<std::size_t, std::size_t, std::size_t, Args...>
                currentArgs;

            // Fill the argument tuple with the QPU id, current argument
            // iteration, and the total number of arguments that will be applied
            // on this QPU.
            std::get<0>(currentArgs) = qpuId;
            std::get<1>(currentArgs) = counter;
            std::get<2>(currentArgs) = nExecsPerQpu;
            counter++;

            // If seed is 0, then it has not been set.
            if (seed > 0)
              cudaq::set_random_seed(seed);

            // Fill the argument tuple with the actual arguments.
            cudaq::tuple_for_each_with_idx(
                params, [&]<typename IDX_TYPE>(auto &&element, IDX_TYPE &&idx) {
                  std::get<IDX_TYPE::value + 3>(currentArgs) = element[i];
                });

            // Call observe/sample with the current set of arguments
            // (provided as a tuple)
            auto result = std::apply(apply, currentArgs);

            // Store the result.
            results.push_back(result);
          }

          // Set the promised results.
          promise.set_value(results);
        });

    platform.enqueueAsyncTask(qpuId, functor);
  }

  // Get all the async-generated results and return.
  std::vector<ResType> allResults;
  for (auto &f : futures) {
    auto res = f.get();
    allResults.insert(allResults.end(), res.begin(), res.end());
  }

  return allResults;
}
} // namespace details
} // namespace cudaq