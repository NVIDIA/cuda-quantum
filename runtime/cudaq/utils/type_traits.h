/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <complex>
#include <string>
#include <tuple>

namespace cudaq {
/// @brief A template struct for mapping related types
/// @tparam T The base type
template <typename T>
struct RelatedTypesMap {
  using types = std::tuple<>;
};

/// @brief Specialization of RelatedTypesMap for int
template <>
struct RelatedTypesMap<int> {
  using types = std::tuple<std::size_t, long, short>;
};

/// @brief Specialization of RelatedTypesMap for std::size_t
template <>
struct RelatedTypesMap<std::size_t> {
  using types = std::tuple<int, long, short>;
};

/// @brief Specialization of RelatedTypesMap for long
template <>
struct RelatedTypesMap<long> {
  using types = std::tuple<int, std::size_t, short>;
};

/// @brief Specialization of RelatedTypesMap for short
template <>
struct RelatedTypesMap<short> {
  using types = std::tuple<int, long, std::size_t>;
};

/// @brief Specialization of RelatedTypesMap for double
template <>
struct RelatedTypesMap<double> {
  using types = std::tuple<float>;
};

/// @brief Specialization of RelatedTypesMap for float
template <>
struct RelatedTypesMap<float> {
  using types = std::tuple<double>;
};

/// @brief Specialization of RelatedTypesMap for std::string
template <>
struct RelatedTypesMap<std::string> {
  using types = std::tuple<const char *>;
};

/// @brief Type trait to check if a type is a bounded char array
template <class>
struct is_bounded_char_array : std::false_type {};

/// @brief Specialization for bounded char arrays
template <std::size_t N>
struct is_bounded_char_array<char[N]> : std::true_type {};

/// @brief Type trait to check if a type is a bounded array
template <class>
struct is_bounded_array : std::false_type {};

/// @brief Specialization for bounded arrays
template <class T, std::size_t N>
struct is_bounded_array<T[N]> : std::true_type {};

// Primary template (for unsupported types)
template <typename T>
constexpr std::string_view type_to_string() {
  return "unknown";
}

// Specializations for common scalar types
template <>
constexpr std::string_view type_to_string<int>() {
  return "int";
}

template <>
constexpr std::string_view type_to_string<double>() {
  return "double";
}

template <>
constexpr std::string_view type_to_string<float>() {
  return "float";
}

template <>
constexpr std::string_view type_to_string<long>() {
  return "long";
}
template <>
constexpr std::string_view type_to_string<std::size_t>() {
  return "stdsizet";
}
template <>
constexpr std::string_view type_to_string<std::complex<double>>() {
  return "complex<double>";
}
template <>
constexpr std::string_view type_to_string<std::complex<float>>() {
  return "complex<float>";
}

} // namespace cudaq