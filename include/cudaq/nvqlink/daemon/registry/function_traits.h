/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace cudaq::nvqlink {

//===----------------------------------------------------------------------===//
// Serialization Concepts
//===----------------------------------------------------------------------===//

/// Enforces POD (Plain Old Data) requirement for RPC serialization. Types must
// be trivially copyable and standard layout for zero-copy transfer.
template <typename T>
concept Serializable =
    std::is_trivially_copyable_v<T> && std::is_standard_layout_v<T>;

/// Validates return types for RPC (POD or void)
template <typename T>
concept SerializableReturn = std::is_void_v<T> || Serializable<T>;

//===----------------------------------------------------------------------===//
// Function Traits for Signature Introspection
//===----------------------------------------------------------------------===//

/// Primary template (undefined - will cause compile error for unsupported
/// types)
template <typename T>
struct function_traits;

/// Specialization for function types (e.g., int(int, int))
template <typename R, typename... Args>
struct function_traits<R(Args...)> {
  using return_type = R;
  using arg_tuple = std::tuple<std::remove_cvref_t<Args>...>;
  static constexpr std::size_t arity = sizeof...(Args);

  // Compile-time validation: all arguments must be Serializable
  static_assert((Serializable<std::remove_cvref_t<Args>> && ...),
                "All function arguments must be POD types (trivially copyable "
                "and standard layout)");
  static_assert(SerializableReturn<R>, "Return type must be void or POD type");
};

/// Specialization for function pointers (e.g., int(*)(int, int))
template <typename R, typename... Args>
struct function_traits<R (*)(Args...)> : function_traits<R(Args...)> {};

/// Specialization for member function pointers (non-const)
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...)> : function_traits<R(Args...)> {};

/// Specialization for member function pointers (const)
template <typename C, typename R, typename... Args>
struct function_traits<R (C::*)(Args...) const> : function_traits<R(Args...)> {
};

/// Specialization for functors and lambdas (via operator())
/// This uses SFINAE to only match types that have an operator()
template <typename F>
struct function_traits : function_traits<decltype(&F::operator())> {};

//===----------------------------------------------------------------------===//
// Compile-Time Size Calculation
//===----------------------------------------------------------------------===//

/// Calculate serialized size of a type
template <typename T>
constexpr std::size_t serialized_size() noexcept {
  if constexpr (std::is_void_v<T>) {
    return 0;
  } else {
    static_assert(Serializable<T>, "Type must be serializable (POD)");
    return sizeof(T);
  }
}

/// Calculate max result size for a function's return type
template <typename F>
constexpr std::size_t result_size() noexcept {
  using Traits = function_traits<std::remove_cvref_t<F>>;
  using Return = typename Traits::return_type;
  return serialized_size<Return>();
}

//===----------------------------------------------------------------------===//
// FNV-1a Hash for Automatic ID Generation
//===----------------------------------------------------------------------===//

/// FNV-1a hash algorithm for generating stable function IDs from names.
/// This is a constexpr-friendly hash with good distribution properties.
///
/// @param name Fully-qualified function name (e.g., "namespace::function")
/// @return 32-bit hash value
///
constexpr std::uint32_t hash_name(std::string_view name) noexcept {
  // FNV-1a constants
  constexpr std::uint32_t FNV_OFFSET_BASIS = 2166136261u;
  constexpr std::uint32_t FNV_PRIME = 16777619u;

  std::uint32_t hash = FNV_OFFSET_BASIS;
  for (char c : name) {
    hash ^= static_cast<std::uint32_t>(static_cast<unsigned char>(c));
    hash *= FNV_PRIME;
  }
  return hash;
}

} // namespace cudaq::nvqlink
