/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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

} // namespace cudaq
