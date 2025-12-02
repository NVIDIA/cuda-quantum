/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/daemon/registry/function_registry.h"
#include "cudaq/nvqlink/daemon/registry/function_traits.h"
#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"

#include <tuple>
#include <type_traits>
#include <utility>

namespace cudaq::nvqlink {

/// Macro to capture both function pointer and its name as a string.
/// Expands to: function_pointer, "namespace::function_name"
///
/// @example
/// ```cpp
/// daemon->register_function(NVQLINK_RPC_HANDLE(math::add));
/// // Expands to: daemon->register_function(math::add, "math::add")
/// ```
#define NVQLINK_RPC_HANDLE(f) f, #f

//===----------------------------------------------------------------------===//
// Argument Deserialization
//===----------------------------------------------------------------------===//

/// Deserialize arguments from InputStream into a tuple.
/// Uses fold expressions and index_sequence for compile-time iteration.
///
/// @tparam Tuple Tuple type containing argument types
/// @tparam Is Index sequence (0, 1, 2, ...)
/// @param in Input stream to read from
/// @return Tuple containing deserialized arguments
///
template <typename Tuple, std::size_t... Is>
Tuple deserialize_args_impl(InputStream &in, std::index_sequence<Is...>) {
  // Read each argument in sequence using fold expression
  return Tuple{in.read<std::tuple_element_t<Is, Tuple>>()...};
}

/// Deserialize all function arguments from stream.
///
/// @tparam Args Variadic template of argument types
/// @param in Input stream to read from
/// @return Tuple containing all deserialized arguments
///
template <typename... Args>
std::tuple<Args...> deserialize_args(InputStream &in) {
  return deserialize_args_impl<std::tuple<Args...>>(
      in, std::index_sequence_for<Args...>{});
}

//===----------------------------------------------------------------------===//
// Result Serialization
//===----------------------------------------------------------------------===//

/// Serialize function result to OutputStream.
/// Handles both void and non-void return types.
///
/// @tparam R Return type
/// @param out Output stream to write to
/// @param result Result value to serialize
///
template <typename R>
void serialize_result(OutputStream &out, R &&result) {
  if constexpr (!std::is_void_v<R>)
    out.write(std::forward<R>(result));
  // For void, do nothing
}

//===----------------------------------------------------------------------===//
// Function Wrapper Generator
//===----------------------------------------------------------------------===//

/// Generate a CPUFunction wrapper from a normal C++ function.
///
/// This is the core of the magic - it takes a regular function like:
///   `int32_t add(int32_t a, int32_t b)`
///
/// And generates a wrapper compatible with the RPC system:
///   `int wrapper(InputStream& in, OutputStream& out)`
///
/// All template work happens at registration time (control path).
/// The generated lambda has zero overhead on the hot path.
///
/// @tparam F Function type (deduced)
/// @param func Function to wrap
/// @return CPUFunction compatible with the dispatcher
///
template <typename F>
CPUFunction make_wrapper(F &&func) {
  using Traits = function_traits<std::remove_cvref_t<F>>;
  using Args = typename Traits::arg_tuple;
  using Return = typename Traits::return_type;

  // The wrapper lambda captures the function and performs serialization
  return [func = std::forward<F>(func)](InputStream &in,
                                        OutputStream &out) -> int {
    try {
      // Deserialize arguments from input stream into tuple
      auto args = deserialize_args_impl<Args>(
          in, std::make_index_sequence<std::tuple_size_v<Args>>{});

      // Call the function with unpacked arguments
      if constexpr (std::is_void_v<Return>) {
        std::apply(func, std::move(args));
        // No result to serialize
      } else {
        auto result = std::apply(func, std::move(args));
        serialize_result(out, std::move(result));
      }

      return 0; // Success
    } catch (const std::exception &e) {
      // TODO: Consider logging the error
      return -1; // Error
    }
  };
}

} // namespace cudaq::nvqlink
