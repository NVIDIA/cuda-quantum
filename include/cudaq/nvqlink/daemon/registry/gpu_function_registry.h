/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// NEW: NVQLink-specific component (not adapted from Hololink)
// PURPOSE: Device-side function registry for RPC dispatch in GPU kernels

#pragma once

#include "cudaq/nvqlink/network/serialization/gpu_input_stream.h"
#include "cudaq/nvqlink/network/serialization/gpu_output_stream.h"

#include <cstdint>

namespace cudaq::nvqlink {

/**
 * @brief Device-side function wrapper for GPU RPC dispatch
 * 
 * Wraps a GPU-callable function that processes RPC requests.
 * The function receives serialized arguments via GPUInputStream
 * and writes results via GPUOutputStream.
 */
struct GPUFunctionWrapper {
    using InvokeFunc = std::int32_t (*)(GPUInputStream&, GPUOutputStream&);
    
    InvokeFunc invoke;
    std::uint32_t function_id;
};

/**
 * @brief Device-side function registry (GPU-accessible)
 * 
 * Simple linear search registry for GPU kernels. Optimized for
 * small function counts (< 256 functions). For larger registries,
 * consider hash table or binary search.
 */
struct GPUFunctionRegistry {
    static constexpr std::uint32_t MAX_FUNCTIONS = 256;
    
    GPUFunctionWrapper functions[MAX_FUNCTIONS];
    std::uint32_t num_functions;
    
    /**
     * @brief Look up function by ID (device-side)
     * 
     * @param function_id RPC function ID
     * @return Function wrapper or nullptr if not found
     */
    __device__ GPUFunctionWrapper* lookup(std::uint32_t function_id) {
        // Linear search - sufficient for small registries
        for (std::uint32_t i = 0; i < num_functions; ++i) {
            if (functions[i].function_id == function_id) {
                return &functions[i];
            }
        }
        return nullptr;
    }
};

} // namespace cudaq::nvqlink

