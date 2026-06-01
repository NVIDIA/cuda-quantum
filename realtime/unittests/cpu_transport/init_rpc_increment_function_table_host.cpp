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
using cudaq::realtime::RPCHeader;
using cudaq::realtime::RPCResponse;
using cudaq::realtime::RPC_MAGIC_REQUEST;
using cudaq::realtime::RPC_MAGIC_RESPONSE;

namespace {

constexpr std::uint32_t RPC_INCREMENT_FUNCTION_ID =
    fnv1a_hash("rpc_increment");

/// Host-side increment handler.  Signature matches `cudaq_host_rpc_fn_t`:
///   void (*)(void *slot_host, size_t slot_size)
///
/// The host dispatcher's HOST_CALL path pre-copies the RX slot bytes into
/// the TX slot and passes us a pointer to the TX slot.  We:
///   1. Read the RPCHeader (in place at slot[0..24]).
///   2. Read the payload at slot[24..24+arg_len].
///   3. Overwrite slot[0..24] with the RPCResponse (RPCHeader and
///      RPCResponse are both 24 bytes packed; the layout overlaps
///      magic/status/result_len/request_id/ptp_timestamp safely).
///   4. Write the incremented bytes back to slot[24..24+arg_len].
///
/// Wire-compatible with the device-side handler: same function_id, same
/// payload semantics (add 1 to each byte), same first-8-bytes PTP echo
/// (preserved by the RPCResponse.ptp_timestamp copy from the header).
extern "C" void
rpc_increment_handler_host(void *slot_host, std::size_t slot_size) {
  auto *header = static_cast<RPCHeader *>(slot_host);
  if (header->magic != RPC_MAGIC_REQUEST)
    return; // dispatcher already filtered, but be defensive

  const std::uint32_t arg_len = header->arg_len;
  const std::uint32_t request_id = header->request_id;
  const std::uint64_t ptp_timestamp = header->ptp_timestamp;
  std::uint8_t *payload_in =
      static_cast<std::uint8_t *>(slot_host) + sizeof(RPCHeader);

  // Bound the work by the slot size so a malformed arg_len can't write
  // past the slot.  This mirrors what the dispatcher's max_result_len
  // guard does for the device-side handler.
  const std::size_t max_payload =
      slot_size > sizeof(RPCResponse) ? slot_size - sizeof(RPCResponse) : 0;
  const std::uint32_t len =
      arg_len < max_payload ? arg_len : static_cast<std::uint32_t>(max_payload);

  // Increment in place.  Header bytes will be overwritten next; payload
  // bytes live entirely after sizeof(RPCResponse) which is the same as
  // sizeof(RPCHeader), so the indices don't shift.
  for (std::uint32_t i = 0; i < len; ++i)
    payload_in[i] = static_cast<std::uint8_t>(payload_in[i] + 1);

  // Write the RPCResponse over the RPCHeader.  Same byte layout for the
  // common fields (request_id, ptp_timestamp); magic/function_id slot is
  // overwritten with magic/status; arg_len slot is overwritten with
  // result_len.  All four fields are read above into locals first to
  // avoid reading after partial overwrite.
  auto *response = static_cast<RPCResponse *>(slot_host);
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
extern "C" void setup_rpc_increment_function_table_host(
    cudaq_function_entry_t *h_entries) {
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
