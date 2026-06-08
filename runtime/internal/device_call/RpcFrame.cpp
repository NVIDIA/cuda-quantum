/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/device_call/RpcFrame.h"
#include "cudaq_internal/device_call/DeviceCallError.h"

#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include "cudaq/runtime/logger/logger.h"

#include <limits>
#include <stdexcept>
#include <string>

namespace cudaq_internal::device_call {

// Keep the local request offset constant synchronized with CUDA-Q realtime.
static_assert(sizeof(cudaq::realtime::RPCHeader) == CUDAQ_RPC_HEADER_SIZE);

std::byte *requestPayload(void *frame) {
  // `device_call` argument bytes immediately follow the realtime request
  // prefix.
  return reinterpret_cast<std::byte *>(static_cast<std::uint8_t *>(frame) +
                                       CUDAQ_RPC_HEADER_SIZE);
}

std::byte *responsePayload(void *frame) {
  // Realtime writes `RPCResponse` first; result bytes begin after that prefix.
  return reinterpret_cast<std::byte *>(static_cast<std::uint8_t *>(frame) +
                                       sizeof(cudaq::realtime::RPCResponse));
}

std::uint64_t validateResponseFrame(const void *frame, std::uint32_t requestId,
                                    std::uint64_t responseCapacity,
                                    std::uint64_t availableFrameBytes) {
  CUDAQ_DBG("[device-call] validate response frame requestId={} "
            "responseCapacity={} availableFrameBytes={}",
            requestId, responseCapacity, availableFrameBytes);
  if (!frame || availableFrameBytes < sizeof(cudaq::realtime::RPCResponse))
    throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                          "invalid device_call response frame");

  // The TX slot is owned by realtime until dispatch completes. Validate both
  // the magic and request_id before reading result metadata so a stale or
  // partially written slot cannot be mistaken for this response.
  const auto *response =
      reinterpret_cast<const cudaq::realtime::RPCResponse *>(frame);
  if (response->magic != cudaq::realtime::RPC_MAGIC_RESPONSE ||
      response->request_id != requestId)
    throw std::runtime_error("mismatched device_call response frame");

  CUDAQ_DBG("[device-call] response frame status={} resultLen={}",
            response->status, response->result_len);

  if (response->status != toAbiStatus(DeviceCallStatus::Success)) {
    const std::string message = "device_call remote endpoint: ";
    switch (response->status) {
    case toAbiStatus(DeviceCallStatus::InvalidArgument):
      throw DeviceCallError(DeviceCallStatus::InvalidArgument,
                            message + "invalid request");
    case toAbiStatus(DeviceCallStatus::NotInitialized):
      throw DeviceCallError(DeviceCallStatus::NotInitialized,
                            message + "endpoint is not initialized");
    case toAbiStatus(DeviceCallStatus::CudaError):
      throw DeviceCallError(DeviceCallStatus::CudaError,
                            message + "CUDA error");
    case toAbiStatus(DeviceCallStatus::Timeout):
      throw DeviceCallError(DeviceCallStatus::Timeout, message + "timed out");
    case toAbiStatus(DeviceCallStatus::ResponseTooLarge):
      throw DeviceCallError(DeviceCallStatus::ResponseTooLarge,
                            message + "response exceeds caller capacity");
    case toAbiStatus(DeviceCallStatus::RemoteError):
      throw DeviceCallError(DeviceCallStatus::RemoteError,
                            message + "remote endpoint error");
    default:
      throw DeviceCallError(DeviceCallStatus::RemoteError,
                            message + "returned device_call status " +
                                std::to_string(response->status));
    }
  }

  const std::uint64_t responseFrameBytes =
      sizeof(cudaq::realtime::RPCResponse) + response->result_len;
  // First check the bytes realtime actually produced in this frame, then check
  // the caller result buffer capacity before the caller copies payload bytes.
  if (responseFrameBytes > availableFrameBytes)
    throw std::runtime_error("truncated device_call response frame");
  if (response->result_len > responseCapacity)
    throw DeviceCallError(DeviceCallStatus::ResponseTooLarge,
                          "device_call response exceeds caller capacity");

  return response->result_len;
}

} // namespace cudaq_internal::device_call
