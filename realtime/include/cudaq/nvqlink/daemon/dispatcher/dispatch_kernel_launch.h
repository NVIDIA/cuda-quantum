/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>

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
///
/// The handler reads arguments from the input buffer and writes results
/// directly to the output buffer. The two buffers never overlap, which
/// enables the dispatch kernel to point `output` straight into the TX
/// ring-buffer slot, eliminating a post-handler copy.
///
/// @param input  Pointer to argument data (RX buffer, read-only)
/// @param output Pointer to result buffer (TX buffer, write-only)
/// @param arg_len Length of argument data in bytes
/// @param max_result_len Maximum result buffer size in bytes
/// @param result_len Output: actual result length written
/// @return Status code (0 = success)
using DeviceRPCFunction = int (*)(const void *input, void *output,
                                  std::uint32_t arg_len,
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
// Graph IO Context (for CUDAQ_DISPATCH_GRAPH_LAUNCH)
//==============================================================================

/// @brief IO context passed to graph-launched RPC handlers via pointer
/// indirection.
///
/// The dispatch kernel fills this context before each fire-and-forget graph
/// launch so the graph kernel knows where to read input, where to write the
/// response, and how to signal completion.  The graph kernel is responsible
/// for writing the RPCResponse header to `tx_slot` and then setting
/// `*tx_flag = tx_flag_value` after a `__threadfence_system()`.
struct GraphIOContext {
  void *rx_slot;                   ///< Input: RX slot (RPCHeader + `args`)
  std::uint8_t *tx_slot;           ///< Output: TX slot for RPCResponse
  volatile std::uint64_t *tx_flag; ///< Pointer to TX flag for this slot
  std::uint64_t tx_flag_value;     ///< Value to write to tx_flag when done
  std::size_t tx_stride_sz;        ///< TX slot size (for max_result_len)
};

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
  std::uint8_t num_args;        ///< Number of arguments
  std::uint8_t num_results;     ///< Number of results
  std::uint16_t reserved;       ///< Padding for alignment
  cudaq_type_desc_t args[8];    ///< Argument type descriptors (max 8)
  cudaq_type_desc_t results[4]; ///< Result type descriptors (max 4)
};

} // namespace cudaq::nvqlink
