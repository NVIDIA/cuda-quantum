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
