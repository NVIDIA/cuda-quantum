/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

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

template <typename TypeToCheck, typename TypeToCheckAgainst>
concept type_is =
    std::same_as<std::remove_cvref_t<TypeToCheck>, TypeToCheckAgainst>;

template <typename... Args>
using ArgumentSet = std::tuple<std::vector<Args>...>;

template <typename... Args>
auto make_argset(const std::vector<Args> &...args) {
  return std::make_tuple(args...);
}

namespace details {

template <typename ApplyFunctor, typename... Args,
          typename ReturnType = std::invoke_result_t<ApplyFunctor>>
std::vector<ReturnType>
broadcastFunctionOverArguments(ApplyFunctor &&apply,
                               ArgumentSet<Args...> &params) {
  std::vector<ReturnType> results;

  // Assert all arg vectors are the same size
  auto N = std::get<0>(params).size();
  cudaq::tuple_for_each(params, [&](auto &&element) {
    if (element.size() != N)
      throw std::runtime_error("Invalid argument set to broadcast function "
                               "over - vector sizes not the same.");
  });

  // Loop over all sets of arguments, the ith element of each vector
  // in the ArgumentSet tuple
  for (std::size_t i = 0; i < std::get<0>(params).size(); i++) {
    // Construct the current set of arguments as a new tuple
    // We want a tuple so we can use std::apply with the
    // existing observe() functions.
    std::tuple<Args...> currentArgs;
    cudaq::tuple_for_each_with_idx(
        params, [&]<typename IDX_TYPE>(auto &&element, IDX_TYPE &&idx) {
          std::get<IDX_TYPE::value>(currentArgs) = element[i];
        });

    // Call observe with the current set of arguments (provided as a tuple)
    auto result = std::apply(apply, currentArgs);

    // Store the result.
    results.push_back(result);
  }

  return results;
}
} // namespace details

} // namespace cudaq
