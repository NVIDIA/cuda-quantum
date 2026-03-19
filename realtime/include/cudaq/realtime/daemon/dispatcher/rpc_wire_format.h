/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

/// @file rpc_wire_format.h
/// @brief Single source of truth for RPC wire-format constants.
///
/// This header is C-compatible so it can be included by both the public C API
/// (cudaq_realtime.h) and the C++ kernel header (dispatch_kernel_launch.h).

#pragma once

// RPC framing magic values (ASCII: CUQ?).
#define CUDAQ_RPC_MAGIC_REQUEST  0x43555152u  /* 'CUQR' */
#define CUDAQ_RPC_MAGIC_RESPONSE 0x43555153u  /* 'CUQS' */

// sizeof(RPCHeader): 4 x uint32_t + 1 x uint64_t = 24 bytes.
#define CUDAQ_RPC_HEADER_SIZE    24u

// TX flag sentinel values used by the host dispatcher and ring buffer helpers.
#define CUDAQ_TX_FLAG_IN_FLIGHT  0xEEEEEEEEEEEEEEEEULL
#define CUDAQ_TX_FLAG_ERROR_TAG  0xDEADULL
