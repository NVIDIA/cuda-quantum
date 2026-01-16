/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>
#include <cstring>

#include "cudaq/nvqlink/daemon/dispatcher/kernel_types.h"
#include "cudaq/nvqlink/daemon/dispatcher/dispatch_modes.h"

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err = call;                                                    \
    ASSERT_EQ(err, cudaSuccess) << "CUDA error: " << cudaGetErrorString(err);  \
  } while (0)

namespace {

//==============================================================================
// Test Handler: Simple noop that copies input to output
//==============================================================================

/// @brief Simple test handler that echoes input to output.
__device__ int noop_handler(void* buffer, std::uint32_t arg_len,
                            std::uint32_t max_result_len, 
                            std::uint32_t* result_len) {
  // Echo: copy input to output (they share the same buffer)
  *result_len = arg_len;
  return 0;  // Success
}

/// @brief Test handler that adds 1 to each byte.
__device__ int increment_handler(void* buffer, std::uint32_t arg_len,
                                  std::uint32_t max_result_len,
                                  std::uint32_t* result_len) {
  std::uint8_t* data = static_cast<std::uint8_t*>(buffer);
  for (std::uint32_t i = 0; i < arg_len && i < max_result_len; ++i) {
    data[i] = data[i] + 1;
  }
  *result_len = arg_len;
  return 0;
}

//==============================================================================
// Test Kernel for DeviceCallMode
//==============================================================================

using HandlerFunc = int (*)(void*, std::uint32_t, std::uint32_t, std::uint32_t*);

__device__ HandlerFunc d_noop_handler = noop_handler;
__device__ HandlerFunc d_increment_handler = increment_handler;

/// @brief Test kernel that dispatches to a handler using DeviceCallMode.
template <typename KernelType>
__global__ void test_dispatch_kernel(
    HandlerFunc handler,
    void* buffer,
    std::uint32_t arg_len,
    std::uint32_t max_result_len,
    std::uint32_t* result_len,
    int* status) {
  
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *status = handler(buffer, arg_len, max_result_len, result_len);
  }
  
  KernelType::sync();
}

//==============================================================================
// Test Fixture
//==============================================================================

class DispatchKernelTest : public ::testing::Test {
protected:
  void SetUp() override {
    CUDA_CHECK(cudaMalloc(&d_buffer_, 1024));
    CUDA_CHECK(cudaMalloc(&d_result_len_, sizeof(std::uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_status_, sizeof(int)));
  }
  
  void TearDown() override {
    if (d_buffer_) cudaFree(d_buffer_);
    if (d_result_len_) cudaFree(d_result_len_);
    if (d_status_) cudaFree(d_status_);
  }
  
  void* d_buffer_ = nullptr;
  std::uint32_t* d_result_len_ = nullptr;
  int* d_status_ = nullptr;
};

//==============================================================================
// Tests
//==============================================================================

TEST_F(DispatchKernelTest, NoopHandlerBasic) {
  // Prepare test data
  std::vector<uint8_t> input = {1, 2, 3, 4, 5};
  CUDA_CHECK(cudaMemcpy(d_buffer_, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  // Get device function pointer
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_noop_handler, 
                                   sizeof(HandlerFunc)));
  
  // Launch kernel
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 32>>>(
      h_handler, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Check results
  int status;
  std::uint32_t result_len;
  CUDA_CHECK(cudaMemcpy(&status, d_status_, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result_len, d_result_len_, sizeof(std::uint32_t), 
                        cudaMemcpyDeviceToHost));
  
  EXPECT_EQ(status, 0) << "Handler should return success";
  EXPECT_EQ(result_len, input.size()) << "Result length should match input";
  
  // Verify data unchanged (noop)
  std::vector<uint8_t> output(input.size());
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(input, output) << "Noop handler should not modify data";
}

TEST_F(DispatchKernelTest, IncrementHandlerBasic) {
  // Prepare test data
  std::vector<uint8_t> input = {0, 1, 2, 3, 4};
  std::vector<uint8_t> expected = {1, 2, 3, 4, 5};
  CUDA_CHECK(cudaMemcpy(d_buffer_, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  // Get device function pointer
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler, 
                                   sizeof(HandlerFunc)));
  
  // Launch kernel
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 32>>>(
      h_handler, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  // Check results
  int status;
  std::uint32_t result_len;
  CUDA_CHECK(cudaMemcpy(&status, d_status_, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(&result_len, d_result_len_, sizeof(std::uint32_t), 
                        cudaMemcpyDeviceToHost));
  
  EXPECT_EQ(status, 0) << "Handler should return success";
  EXPECT_EQ(result_len, input.size()) << "Result length should match input";
  
  // Verify data incremented
  std::vector<uint8_t> output(input.size());
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(expected, output) << "Increment handler should add 1 to each byte";
}

TEST_F(DispatchKernelTest, MultipleBlocks) {
  // Test with multiple blocks to verify RegularKernel sync works
  std::vector<uint8_t> input = {10, 20, 30, 40, 50};
  CUDA_CHECK(cudaMemcpy(d_buffer_, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_noop_handler, 
                                   sizeof(HandlerFunc)));
  
  // Launch with multiple blocks
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<4, 64>>>(
      h_handler, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  int status;
  CUDA_CHECK(cudaMemcpy(&status, d_status_, sizeof(int), cudaMemcpyDeviceToHost));
  EXPECT_EQ(status, 0) << "Handler should return success with multiple blocks";
}

TEST_F(DispatchKernelTest, LargeBuffer) {
  // Test with larger data
  const std::size_t size = 512;
  std::vector<uint8_t> input(size);
  for (std::size_t i = 0; i < size; ++i) {
    input[i] = static_cast<uint8_t>(i & 0xFF);
  }
  
  CUDA_CHECK(cudaMemcpy(d_buffer_, input.data(), input.size(), 
                        cudaMemcpyHostToDevice));
  
  HandlerFunc h_handler;
  CUDA_CHECK(cudaMemcpyFromSymbol(&h_handler, d_increment_handler, 
                                   sizeof(HandlerFunc)));
  
  test_dispatch_kernel<cudaq::realtime::RegularKernel><<<1, 256>>>(
      h_handler, d_buffer_, input.size(), 1024, d_result_len_, d_status_);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  
  std::uint32_t result_len;
  CUDA_CHECK(cudaMemcpy(&result_len, d_result_len_, sizeof(std::uint32_t), 
                        cudaMemcpyDeviceToHost));
  EXPECT_EQ(result_len, size) << "Should process all bytes";
  
  // Verify all bytes incremented
  std::vector<uint8_t> output(size);
  CUDA_CHECK(cudaMemcpy(output.data(), d_buffer_, output.size(), 
                        cudaMemcpyDeviceToHost));
  
  for (std::size_t i = 0; i < size; ++i) {
    uint8_t expected = static_cast<uint8_t>((i + 1) & 0xFF);
    EXPECT_EQ(output[i], expected) << "Mismatch at index " << i;
  }
}

} // namespace
