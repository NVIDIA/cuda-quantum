/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace cudaq::detail {

/// True when @c T is one of the types in @c Tuple.
template <typename T, typename Tuple>
struct is_in_tuple : std::false_type {};

template <typename T, typename... Ts>
struct is_in_tuple<T, std::tuple<Ts...>>
    : std::bool_constant<(std::is_same_v<T, Ts> || ...)> {};

template <typename T, typename Tuple>
inline constexpr bool is_in_tuple_v = is_in_tuple<T, Tuple>::value;

/// Compile-time position of type @c T within @c Tuple.
template <typename T, typename Tuple>
struct find_pos;

template <typename T, typename... Ts>
struct find_pos<T, std::tuple<T, Ts...>>
    : std::integral_constant<std::size_t, 0> {};

template <typename T, typename U, typename... Ts>
struct find_pos<T, std::tuple<U, Ts...>>
    : std::integral_constant<std::size_t,
                             1 + find_pos<T, std::tuple<Ts...>>::value> {};

template <typename T, typename Tuple>
inline constexpr std::size_t find_pos_v = find_pos<T, Tuple>::value;

} // namespace cudaq::detail
