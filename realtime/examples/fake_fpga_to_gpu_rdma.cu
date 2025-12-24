/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <cassert>
#include <stdio.h>

#include "cudaq/nvqlink/devices/extensible_rdma_device.cuh"
#include "cudaq/nvqlink/nvqlink.h"

// Mock the FPGA on the CPU. Write to "Fake FPGA memory" to trigger RDMA
// data transfer to persistent CUDA kernel on RDMA device

// clang-format off
// Compile with 
//
// nvcc -forward-unknown-to-host-compiler -gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90 --compiler-options -fPIC -O3 -DNDEBUG fake_fpga_to_gpu_rdma.cu -I /path/to/libs/nvqlink/include/ -L /path/to/nvqlink/lib -lcudaq-nvqlink -Wl,-rpath,$PWD/lib 
// ./a.out
//
// clang-format on

// ------ Example of Internal Library Code for Device Functions and Devices ----

__device__ void add_op(void *args, void *res) {
  // printf("Test here\n");
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  int *a = reinterpret_cast<int *>(args);
  int *b = reinterpret_cast<int *>((char *)args + sizeof(int));
  if (tid == 0) {
    *reinterpret_cast<int *>(res) = *a + *b;
    printf("adding numbers %d + %d = %d\n", *a, *b,
           *reinterpret_cast<int *>(res));
  }

  // All threads synchronize before returning
  __syncthreads();
}

using namespace cudaq::nvqlink;

__device__ dispatch_func_t d_add_ptr = add_op;

class concrete_rdma_test : public cpu_gpu_rdma_device {
protected:
  void build_device_function_table() override {
    host_func_table.resize(1);
    cudaMemcpyFromSymbol(&host_func_table[0], d_add_ptr,
                         sizeof(dispatch_func_t));
    cudaMalloc(&device_func_table,
               host_func_table.size() * sizeof(dispatch_func_t));
    cudaMemcpy(device_func_table, host_func_table.data(),
               sizeof(dispatch_func_t), cudaMemcpyHostToDevice);
  }
};
// ------------------------------------------------------------------------------

// --- Example for user code -----

using namespace cudaq::nvqlink;

int main() {
  // Create the rdma device
  concrete_rdma_test dev;
  // connect to it
  dev.connect();

  // Get the internal RDMA connection details
  auto &rdma_details = dev.get_rdma_connection_data();
  char *args_ptr = static_cast<char *>(rdma_details.get_raw_source()) +
                   sizeof(rdma_message_header);

  // Set some rdma device add_op() function arguments
  int arg1 = 42, arg2 = 24;

  {
    // Test automatic RDMA by writing to the CPU buffer
    // The monitoring thread should automatically detect this and transfer to
    // GPU

    // Prepare a function call message
    rdma_message_header msg = {
        .magic = 0xDEADBEEF,
        .message_type = 0, // function call
        .function_id = 0,  // add function
        .num_args = 2,
        .total_size = sizeof(rdma_message_header) + 2 * sizeof(int) +
                      sizeof(int), // header + args + result
        .result_offset = sizeof(rdma_message_header) + 2 * sizeof(int),
        .result_size = sizeof(int),
        .reserved = 0};

    std::memcpy(args_ptr, &arg1, sizeof(int));
    std::memcpy(args_ptr + sizeof(int), &arg2, sizeof(int));

    // Write message to CPU buffer (this should trigger automatic RDMA)
    std::memcpy(rdma_details.get_raw_source(), &msg, sizeof(msg));

    // Give the monitoring thread time to detect and process the change
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  // The result should be automatically computed and available in the result
  // buffer
  int *result_ptr = reinterpret_cast<int *>(args_ptr + 2 * sizeof(int));
  printf("Result: %d (expected: %d)\n", *result_ptr, arg1 + arg2);
  assert(*result_ptr == arg1 + arg2 && "result not correct");
  dev.disconnect();
}
// ------------------------------
