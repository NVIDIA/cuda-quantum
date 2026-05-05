/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

//
// This header provides a wrapper around some of the `fmtlib` functions, so that
// `fmtlib` headers, which are not distributed, do not leak into user code.
//
// It introduces a cudaq_fmt namespace to make it easy to distinguish between
// what would otherwise be `fmt::format` and `cudaq::format`.
//

namespace cudaq_fmt {

/// Type-erased reference to a format argument.
///
/// This intentionally stores only a pointer and an out-of-line appending
/// callback. The callback is instantiated in logger.cpp, where heavier headers
/// such as `fmtlib` and `chrono` can be included without forcing
/// every formatting caller to parse them.
struct FormatArgument {
  using Appender = void (*)(void *store, const void *value);

  template <typename T>
  FormatArgument(const T &value)
      : value(std::addressof(value)), append(&appendArgument<std::decay_t<T>>) {
  }

  FormatArgument(const char *value) : value(value), append(&appendCString) {}
  FormatArgument(char *value) : value(value), append(&appendCString) {}

  const void *value = nullptr;
  Appender append = nullptr;

  template <typename T>
  static void appendArgument(void *store, const void *value);
  static void appendCString(void *store, const void *value);
};

namespace details {
std::string format_packed(const std::string_view message,
                          const std::span<const FormatArgument> &arr);

void print_packed(const std::string_view message,
                  const std::span<const FormatArgument> &arr);
} // namespace details

//
// Functions substitutes for fmt::format and fmt::print
//
template <typename... Args>
std::string format(const std::string_view message, Args &&...args) {
  auto array = std::array<FormatArgument, sizeof...(Args)>{
      FormatArgument(std::forward<Args>(args))...};
  return details::format_packed(
      message, std::span<const FormatArgument>(array.data(), array.size()));
}

template <typename... Args>
void print(const std::string_view message, Args &&...args) {
  auto array = std::array<FormatArgument, sizeof...(Args)>{
      FormatArgument(std::forward<Args>(args))...};
  return details::print_packed(
      message, std::span<const FormatArgument>(array.data(), array.size()));
}

//
// Substitute for fmt::underlying
//
// Converts `e` to the underlying type.
//
// Example:
//
//      enum class color { red, green, blue };
//      auto s = cudaq_fmt::format("{}", cudaq_fmt::underlying(color::red));  //
//      s == "0"
//
template <typename Enum>
constexpr auto underlying(Enum e) noexcept -> std::underlying_type_t<Enum> {
  return static_cast<std::underlying_type_t<Enum>>(e);
}

} // namespace cudaq_fmt
