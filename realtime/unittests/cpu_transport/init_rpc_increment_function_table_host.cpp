/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file init_rpc_increment_function_table_host.cpp
/// @brief Host-side (pure-C++) increment RPC handler + function-table
///        initialiser for CUDAQ_DISPATCH_HOST_CALL.
///
/// Parallels init_rpc_increment_function_table.cu (device-side, dispatch
/// mode CUDAQ_DISPATCH_DEVICE_CALL) but compiled by the C++ host compiler
/// — no nvcc, no CUDA, no graph.  Used by hsb_bridge_cpu (Phase 1) which
/// is GPU-less by design.

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"

#include <cstdint>
#include <cstring>

using cudaq::realtime::fnv1a_hash;
using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;

namespace {

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID = fnv1a_hash("rpc_increment");

/// Host-side increment handler.  Two-pointer `cudaq_host_rpc_fn_t`:
///   void (*)(const void *rx_slot, void *tx_slot, size_t slot_size)
///
/// Reads the RPCHeader + payload from `rx_slot` (the inbound request) and
/// writes the RPCResponse + incremented payload into `tx_slot` (the outbound
/// slot the transport sends).  RPCHeader and RPCResponse are both 24 bytes, so
/// the payload offset is the same (24) on both sides.
///
/// Wire-compatible with the device-side handler: same function_id, same
/// payload semantics (add 1 to each byte), and the request_id / ptp_timestamp
/// (first-8-bytes PTP) are echoed from the request into the response.
extern "C" void rpc_increment_handler_host(const void *rx_slot, void *tx_slot,
                                           std::size_t slot_size) {
  const auto *header = static_cast<const RPCHeader *>(rx_slot);
  if (header->magic != RPC_MAGIC_REQUEST)
    return; // dispatcher already filtered, but be defensive

  const std::uint32_t arg_len = header->arg_len;
  const std::uint32_t request_id = header->request_id;
  const std::uint64_t ptp_timestamp = header->ptp_timestamp;
  const std::uint8_t *payload_in =
      static_cast<const std::uint8_t *>(rx_slot) + sizeof(RPCHeader);

  // Bound the work by the slot size so a malformed arg_len can't write past
  // the TX slot.  This mirrors the dispatcher's max_result_len guard for the
  // device-side handler.
  const std::size_t max_payload =
      slot_size > sizeof(RPCResponse) ? slot_size - sizeof(RPCResponse) : 0;
  const std::uint32_t len =
      arg_len < max_payload ? arg_len : static_cast<std::uint32_t>(max_payload);

  // Read each input byte from the RX slot, write the incremented value into
  // the TX slot.  The two slots are distinct buffers, so there is no aliasing.
  std::uint8_t *payload_out =
      static_cast<std::uint8_t *>(tx_slot) + sizeof(RPCResponse);
  for (std::uint32_t i = 0; i < len; ++i)
    payload_out[i] = static_cast<std::uint8_t>(payload_in[i] + 1);

  auto *response = static_cast<RPCResponse *>(tx_slot);
  response->magic = RPC_MAGIC_RESPONSE;
  response->status = 0;
  response->result_len = len;
  response->request_id = request_id;
  response->ptp_timestamp = ptp_timestamp;
}

} // anonymous namespace

//==============================================================================
// Setup function for hsb_bridge_cpu (and future host-call test binaries)
//==============================================================================

/// Populate one cudaq_function_entry_t with the host-side increment handler.
/// Unlike the device-side variant this takes an entry pointer that lives
/// in plain host memory (the function table is in host memory for
/// HOST_CALL).
extern "C" void
setup_rpc_increment_function_table_host(cudaq_function_entry_t *h_entries) {
  std::memset(h_entries, 0, sizeof(cudaq_function_entry_t));
  h_entries[0].handler.host_fn = &rpc_increment_handler_host;
  h_entries[0].function_id = RPC_INCREMENT_FUNCTION_ID;
  h_entries[0].dispatch_mode = CUDAQ_DISPATCH_HOST_CALL;

  // Schema mirrors the device-side entry: 1 array argument (uint8), 1
  // array result (uint8).  Sizes left at 0 because the handler is
  // dynamically bounded by arg_len.
  h_entries[0].schema.num_args = 1;
  h_entries[0].schema.num_results = 1;
  h_entries[0].schema.args[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
  h_entries[0].schema.results[0].type_id = CUDAQ_TYPE_ARRAY_UINT8;
}
