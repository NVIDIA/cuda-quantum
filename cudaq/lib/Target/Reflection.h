/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Compile-time reflection of the target types (`cudaq::CompileTarget` and the
/// `struct`s and `enum`s it contains).
///
/// Private header: reflect-cpp is not a public dependency of `CUDAQTarget`, so
/// this header must not be exported with the public installed headers.

#include <optional>
#include <rfl/enums.hpp>
#include <rfl/internal/get_type_name.hpp>
#include <rfl/to_view.hpp>
#include <string_view>

namespace cudaq::reflection {

namespace detail {
/// The fully qualified name of `T`, e.g. `cudaq::config::Codegen`.
// Unfortunately, this has to use a private API of reflect-cpp. If this turns
// out to be brittle, we could roll our own implementation.
template <typename T>
constexpr std::string_view qualifiedName =
    rfl::internal::get_type_name_str_view<T>();
} // namespace detail

/// The name of `T` with the `cudaq::` or `cudaq::config::` namespaces stripped,
/// e.g. `Codegen`.
template <typename T>
constexpr std::string_view typeName = [] {
  static_assert(detail::qualifiedName<T>.find("cudaq::") !=
                    std::string_view::npos,
                "expected a type declared within the cudaq namespace");
  return detail::qualifiedName<T>.substr(detail::qualifiedName<T>.rfind("::") +
                                         2);
}();

/// Calls `f(name, field)` on each field of the struct `value`.
template <typename T, typename F>
void forEachField(T &value, F &&f) {
  rfl::to_view(value).apply(
      [&](const auto &field) { f(field.name(), *field.value()); });
}

/// The name of the enumerator `value`.
template <typename E>
std::string_view enumName(E value) {
  for (auto [name, enumerator] : rfl::get_enumerator_array<E>())
    if (enumerator == value)
      return name;
  return {};
}

/// The enumerator of `E` called `name`.
template <typename E>
std::optional<E> enumFromName(std::string_view name) {
  for (auto [enumeratorName, enumerator] : rfl::get_enumerator_array<E>())
    if (enumeratorName == name)
      return enumerator;
  return std::nullopt;
}

} // namespace cudaq::reflection
