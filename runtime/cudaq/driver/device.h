/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <functional>
#include <type_traits>
#include <utility>

namespace cudaq {

template <typename DeviceCode, typename... Args>
auto device_call(DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}

template <typename DeviceCode, typename... Args>
auto device_call(std::size_t device_id, DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}

// --- GPU Overloads ---
template <std::size_t BlockSize, std::size_t GridSize, typename DeviceCode,
          typename... Args>
auto device_call(DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}
template <std::size_t BlockSize, std::size_t GridSize, typename DeviceCode,
          typename... Args>
auto device_call(std::size_t device_id, DeviceCode &&code, Args &&...args)
    -> std::invoke_result_t<DeviceCode, Args...> {
  return std::invoke(std::forward<DeviceCode>(code),
                     std::forward<Args>(args)...);
}

} // namespace cudaq
