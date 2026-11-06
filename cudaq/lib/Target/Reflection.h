/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// Compile-time reflection of the target types (`cudaq::CompileTarget` and the
/// structs and enums it contains).
///
/// Internal to the target library: reflect-cpp is a private dependency, so this
/// header must not be included from installed headers.

#include <optional>
#include <rfl/enums.hpp>
#include <rfl/internal/get_type_name.hpp>
#include <rfl/to_view.hpp>
#include <string_view>
#include <type_traits>

namespace cudaq::reflection {

/// The fully qualified name of `T`, e.g. `cudaq::config::Codegen`.
template <typename T>
constexpr std::string_view qualifiedName =
    rfl::internal::get_type_name_str_view<T>();

/// The unqualified name of `T`, e.g. `Codegen`.
template <typename T>
constexpr std::string_view typeName =
    qualifiedName<T>.substr(qualifiedName<T>.rfind("::") + 2);

/// Whether `T` is a struct or an enum of the `cudaq` namespace.
template <typename T>
constexpr bool isReflected = (std::is_aggregate_v<T> || std::is_enum_v<T>) &&
                             qualifiedName<T>.starts_with("cudaq::");

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
