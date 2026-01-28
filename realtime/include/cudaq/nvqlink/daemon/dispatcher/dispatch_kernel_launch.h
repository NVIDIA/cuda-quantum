/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace cudaq::nvqlink {

//==============================================================================
// RPC Protocol Structures (Wire Format)
//==============================================================================

/// @brief RPC request header - wire format for function dispatch.
/// Must be wire-compatible with cuda-quantum RPC protocol.
struct __attribute__((packed)) RPCHeader {
  std::uint32_t magic;       ///< Magic value to validate message framing
  std::uint32_t function_id; ///< Hash of function name (FNV-1a)
  std::uint32_t arg_len;     ///< Length of argument data in bytes
};

/// @brief RPC response header - returned to caller.
struct __attribute__((packed)) RPCResponse {
  std::uint32_t magic;      ///< Magic value to validate message framing
  std::int32_t status;      ///< Return status (0 = success)
  std::uint32_t result_len; ///< Length of result data in bytes
};

//==============================================================================
// Device Function Type
//==============================================================================

/// @brief Device RPC function signature.
/// @param buffer Pointer to argument/result buffer
/// @param arg_len Length of argument data
/// @param max_result_len Maximum result buffer size
/// @param result_len Output: actual result length
/// @return Status code (0 = success)
using DeviceRPCFunction = int (*)(void *buffer, std::uint32_t arg_len,
                                  std::uint32_t max_result_len,
                                  std::uint32_t *result_len);

//==============================================================================
// Function ID Hashing
//==============================================================================

/// @brief Compute FNV-1a hash of a string (for function_id).
/// @param str Null-terminated string to hash
/// @return 32-bit hash value
constexpr std::uint32_t fnv1a_hash(const char *str) {
  std::uint32_t hash = 2166136261u;
  while (*str) {
    hash ^= static_cast<std::uint32_t>(*str++);
    hash *= 16777619u;
  }
  return hash;
}

// RPC framing magic values (ASCII: CUQ?).
constexpr std::uint32_t RPC_MAGIC_REQUEST = 0x43555152;  // 'CUQR'
constexpr std::uint32_t RPC_MAGIC_RESPONSE = 0x43555153; // 'CUQS'

//==============================================================================
// Schema-Driven Type System
//==============================================================================

/// @brief Standardized payload type identifiers for RPC arguments/results.
enum PayloadTypeID : std::uint8_t {
  TYPE_UINT8 = 0x10,
  TYPE_INT32 = 0x11,
  TYPE_INT64 = 0x12,
  TYPE_FLOAT32 = 0x13,
  TYPE_FLOAT64 = 0x14,
  TYPE_ARRAY_UINT8 = 0x20,
  TYPE_ARRAY_INT32 = 0x21,
  TYPE_ARRAY_FLOAT32 = 0x22,
  TYPE_ARRAY_FLOAT64 = 0x23,
  TYPE_BIT_PACKED = 0x30
};

/// @brief Type descriptor for a single argument or result.
struct __attribute__((packed)) cudaq_type_desc_t {
  std::uint8_t type_id;       ///< PayloadTypeID value
  std::uint8_t reserved[3];   ///< Padding for alignment
  std::uint32_t size_bytes;   ///< Total size in bytes
  std::uint32_t num_elements; ///< Number of elements (for arrays)
};

/// @brief Handler schema describing argument and result types.
struct __attribute__((packed)) cudaq_handler_schema_t {
  std::uint8_t num_args;         ///< Number of arguments
  std::uint8_t num_results;      ///< Number of results
  std::uint16_t reserved;        ///< Padding for alignment
  cudaq_type_desc_t args[8];     ///< Argument type descriptors (max 8)
  cudaq_type_desc_t results[4];  ///< Result type descriptors (max 4)
};

} // namespace cudaq::nvqlink
