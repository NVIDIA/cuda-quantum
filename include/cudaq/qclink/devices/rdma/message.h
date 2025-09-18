/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once 

#include <cstddef>
#include <stdint.h>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error("CUDA error: " +                                \
                               std::string(cudaGetErrorString(err)));          \
    }                                                                          \
  } while (0)

namespace cudaq::qclink {

// Enhanced message protocol
struct rdma_message_header {
  uint32_t magic;         // 0xDEADBEEF for validation
  uint32_t message_type;  // 0=function_call, 1=shutdown, 2=get_result
  uint32_t function_id;   // ID of function to call
  uint32_t num_args;      // Number of arguments
  uint32_t total_size;    // Total message size including this header
  uint32_t result_offset; // Offset where result should be written
  uint32_t result_size;   // Size of expected result
  uint32_t reserved;      // Padding for alignment
};

using dispatch_func_t = void (*)(void *, void *);

} // namespace cudaq::qclink
