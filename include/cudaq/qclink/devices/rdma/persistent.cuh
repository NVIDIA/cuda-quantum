/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "message.h"
#include <cstddef>
#include <stdio.h>

namespace cudaq::qclink {
// Persistent CUDA kernel that processes messages
__global__ void persistent_rdma_dispatcher_kernel(
    unsigned char *message_buffer, size_t buffer_size,
    volatile bool *trigger_flag, volatile bool *completion_flag,
    dispatch_func_t *functions, std::size_t num_funcs) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  __shared__ bool should_shutdown, apply_func;
  __shared__ uint32_t fid, resultOffset;
  if (tid == 0) {
    should_shutdown = false;
    apply_func = false;
  }

  __syncthreads();

  while (true) {

    if (*trigger_flag && tid == 0) {
      // Parse message
      rdma_message_header *msg =
          reinterpret_cast<rdma_message_header *>(message_buffer);
      if (msg->message_type == 1) { // Shutdown
        printf("THERE IS A SHUTDOWN REQUEST\n");
        *completion_flag = true;
        should_shutdown = true;
        break;
      }

      if (msg->message_type == 0) {
        printf("THERE IS A FUNC REQUEST %d\n", msg->function_id);

        // Get function pointer
        auto localFid = msg->function_id;
        if (localFid >= num_funcs) {
          printf("BAD FUNCTION\n");
        }

        fid = localFid;
        resultOffset = msg->result_offset;
        apply_func = true;
      }

      *trigger_flag = false;
    }

    __syncthreads();

    if (apply_func) {
      auto *arg_data = message_buffer + sizeof(rdma_message_header);
      auto *result = message_buffer + resultOffset;
      auto *func = functions[fid];
      (*func)(arg_data, result);
      if (tid == 0) {
        apply_func = false;
        *completion_flag = true;
      }
      __syncthreads();
    }

    // ALL threads check the shared flag and exit together
    if (should_shutdown) {
      break;
    }

    // Small delay to prevent busy waiting
    __nanosleep(1000);
  }
}
} // namespace cudaq::qclink
