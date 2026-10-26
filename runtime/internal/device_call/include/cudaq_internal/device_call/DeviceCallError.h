/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>

namespace cudaq_internal::device_call {

// Status values carried across the device_call C ABI and RPC response frames.
enum class DeviceCallStatus : std::int32_t {
  Success = 0,
  InvalidArgument = 1,
  NotInitialized = 2,
  CudaError = 3,
  Timeout = 4,
  ResponseTooLarge = 5,
  RemoteError = 6,
};

// Convert the typed status to integer.
constexpr std::int32_t toAbiStatus(DeviceCallStatus status) noexcept {
  return static_cast<std::int32_t>(status);
}

// Exception type for channel failures that need to preserve a status code.
class DeviceCallError : public std::runtime_error {
public:
  DeviceCallError(DeviceCallStatus status, std::string message)
      : std::runtime_error(std::move(message)), statusCode(status) {}

  DeviceCallStatus status() const noexcept { return statusCode; }

private:
  DeviceCallStatus statusCode;
};

} // namespace cudaq_internal::device_call
