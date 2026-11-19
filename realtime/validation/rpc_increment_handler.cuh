/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

/// @file rpc_increment_handler.cuh
/// @brief Declaration of the device increment RPC handler, defined in
/// rpc_increment_handler.cu.

#include <cstddef>

/// Increments each byte of the request's payload into the response; see
/// rpc_increment_handler.cu.
__device__ int rpc_increment_handler(const void *rx_slot, void *tx_slot,
                                     std::size_t slot_size);
