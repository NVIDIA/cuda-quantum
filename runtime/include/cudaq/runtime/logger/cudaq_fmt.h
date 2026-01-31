/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <chrono>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <span>
#include <string>
#include <type_traits>
#include <variant>
#include <vector>

//
// This header provides a wrapper around some of the fmtlib functions, so that
// fmtlib headers, which are not distributed, do not bleed into user code.
//
// It introduces a cudaq_fmt namespace to make it easy to distinguish between
// what would otherwise be fmt::format and cudaq::fmt::format.
//

namespace cudaq_fmt {
struct fmt_arg;
namespace details {
//
// Concepts to check if T is in the variant
//
template <typename T, typename... Types>
concept one_of = ((std::same_as<T, Types> ||
                   std::same_as<std::reference_wrapper<const T>, Types>) ||
                  ...);

template <typename T, typename Variant>
struct is_variant_member;

template <typename T, typename... Types>
struct is_variant_member<T, std::variant<Types...>>
    : std::bool_constant<one_of<T, Types...>> {};

template <typename T, typename Variant>
concept variant_alternative =
    is_variant_member<std::decay_t<T>, Variant>::value;

//
// Packed versions of format and print, implemented in Logger.cpp
//
std::string format_packed(const std::string_view message,
                          const std::span<fmt_arg> &arr);

void print_packed(const std::string_view message,
                  const std::span<fmt_arg> &arr);
} // namespace details

//
// Packed parameter type passing arguments to fmt
// Built-in types need to be passed by value.
// Please store large types (like vectors, strings) as reference_wrappers
//
struct fmt_arg {
  using storage_t = std::variant<
      bool, char, uint32_t, int32_t, uint64_t, int64_t, float, double,
      std::complex<float>, std::complex<double>, std::string_view, const char *,
      char *, void *, std::chrono::milliseconds,
      std::chrono::system_clock::time_point,
      std::reference_wrapper<const std::vector<int32_t>>,
      std::reference_wrapper<const std::string>,
      std::reference_wrapper<const std::vector<uint32_t>>,
      std::reference_wrapper<const std::vector<uint64_t>>,
      std::reference_wrapper<const std::vector<float>>,
      std::reference_wrapper<const std::vector<double>>,
      std::reference_wrapper<const std::vector<std::string>>,
      std::reference_wrapper<const std::map<std::string, std::string>>,
      std::reference_wrapper<const std::vector<std::complex<float>>>,
      std::reference_wrapper<const std::vector<std::complex<double>>>>;
  storage_t value;

  template <typename T>
    requires details::variant_alternative<T, storage_t>
  fmt_arg(const T &v) : value(std::cref(v)) {}
};

//
// Functions substitutes for fmt::format and fmt::print
//
template <typename... Args>
std::string format(const std::string_view message, Args &&...args) {
  auto array = std::array<fmt_arg, sizeof...(Args)>{
      fmt_arg(std::forward<Args>(args))...};
  return details::format_packed(message, array);
}

template <typename... Args>
void print(const std::string_view message, Args &&...args) {
  auto array = std::array<fmt_arg, sizeof...(Args)>{
      fmt_arg(std::forward<Args>(args))...};
  return details::print_packed(message, array);
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
