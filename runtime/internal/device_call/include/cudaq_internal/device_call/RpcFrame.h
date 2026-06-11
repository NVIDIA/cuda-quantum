/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

namespace cudaq_internal::device_call {

// Helpers for CUDA-Q realtime RPC wire frames.  A request frame starts with an
// RPCHeader followed by the serialized argument payload; a response frame
// starts with an RPCResponse followed by the serialized result payload.

// Return the first byte after the request header.
std::byte *requestPayload(void *frame);

// Return the first byte after the response header.
std::byte *responsePayload(void *frame);

// Validate a response frame for the expected request and caller capacity, then
// return the number of result payload bytes.  availableFrameBytes bounds the
// frame bytes actually present, allowing callers to detect truncated frames.
std::uint64_t validateResponseFrame(const void *frame, std::uint32_t requestId,
                                    std::uint64_t responseCapacity,
                                    std::uint64_t availableFrameBytes);

} // namespace cudaq_internal::device_call
