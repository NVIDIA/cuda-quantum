/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <type_traits>

namespace cudaq {
#if CUDAQ_USE_STD20
// Primary trait: defaults to false.
template <typename T>
struct is_operator_type : std::false_type {};

// product_operator
template <typename HandlerTy>
struct is_operator_type<product_operator<HandlerTy>> : std::true_type {};

// operator_sum
template <typename HandlerTy>
struct is_operator_type<operator_sum<HandlerTy>> : std::true_type {};

// Satisfied if T (after decay) is one of operator types.
template <typename T>
concept operator_type = is_operator_type<std::decay_t<T>>::value;
#endif
}
