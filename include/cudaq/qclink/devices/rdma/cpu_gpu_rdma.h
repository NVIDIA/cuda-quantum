/*******************************************************************************
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "message.h"
#include "rdma_data.h"

#include "cudaq/qclink/device.h"
#include <condition_variable>
#include <cstring>
#include <memory>
#include <thread>

namespace cudaq::qclink {

class cpu_gpu_rdma_data : public rdma_data<cpu_gpu_rdma_data> {
private:
  //   void *gpu_message_buffer = nullptr;
  void *cpu_mapped_buffer = nullptr;
  size_t buffer_size = 64 * 1024; // 64KB buffer
  std::unique_ptr<char[]> buffer_shadow_copy;
  std::mutex buffer_mutex;
  std::atomic<uint64_t> buffer_version{0};
  std::atomic<uint64_t> last_processed_version{0};

public:
  void connect(persistent_kernel_data &data) {
    buffer_shadow_copy = std::make_unique<char[]>(buffer_size);
    printf("cpu_gpu_rdma_data connecting\n");
    // Allocate GPU message buffer
    CUDA_CHECK(cudaMalloc(&data.gpu_message_buffer, buffer_size));
    CUDA_CHECK(cudaMemset(data.gpu_message_buffer, 0, buffer_size));

    // Allocate trigger and completion flags
    CUDA_CHECK(cudaMalloc(&data.trigger_flag, sizeof(bool)));
    CUDA_CHECK(cudaMalloc(&data.completion_flag, sizeof(bool)));
    CUDA_CHECK(cudaMemset(data.trigger_flag, 0, sizeof(bool)));
    CUDA_CHECK(cudaMemset(data.completion_flag, 0, sizeof(bool)));

    // Allocate host-accessible buffer
    CUDA_CHECK(cudaMallocHost(&cpu_mapped_buffer, buffer_size));
    memset(cpu_mapped_buffer, 0, buffer_size);

    // Initialize shadow copy
    std::memcpy(buffer_shadow_copy.get(), cpu_mapped_buffer, buffer_size);
  }

  void disconnect(persistent_kernel_data &data) {

    // Send shutdown message
    rdma_message_header shutdown_msg = {
        0xDEADBEEF, 1, 0, 0, 0, sizeof(rdma_message_header)};
    std::memcpy(cpu_mapped_buffer, &shutdown_msg, sizeof(shutdown_msg));

    CUDA_CHECK(cudaMemcpyAsync(data.gpu_message_buffer, cpu_mapped_buffer,
                               sizeof(shutdown_msg), cudaMemcpyHostToDevice,
                               reinterpret_cast<cudaStream_t &>(data.stream)));

    // 3. Asynchronously set the trigger flag
    bool trigger = true;
    CUDA_CHECK(cudaMemcpyAsync(data.trigger_flag, &trigger, sizeof(bool),
                               cudaMemcpyHostToDevice, data.stream));

    // 4. Wait for the stream to finish all tasks (copies and kernel
    // termination)
    CUDA_CHECK(cudaStreamSynchronize(data.stream));
    if (cpu_mapped_buffer)
      cudaFreeHost(cpu_mapped_buffer);
  }

  void *get_raw_source() { return cpu_mapped_buffer; }

  bool detected_buffer_change() {

    std::lock_guard<std::mutex> lock(buffer_mutex);

    // Compare current buffer with shadow copy
    if (std::memcmp(cpu_mapped_buffer, buffer_shadow_copy.get(), buffer_size) !=
        0) {
      // Update shadow copy
      std::memcpy(buffer_shadow_copy.get(), cpu_mapped_buffer, buffer_size);
      buffer_version.fetch_add(1);
      return true;
    }
    return false;
  }
  void process_buffer_change(persistent_kernel_data &data) {

    // Validate the message header first
    rdma_message_header *header =
        reinterpret_cast<rdma_message_header *>(cpu_mapped_buffer);

    if (header->magic != 0xDEADBEEF) {
      // Not a valid message, ignore
      return;
    }

    size_t copy_size =
        std::min(static_cast<size_t>(header->total_size), buffer_size);

    printf("RDMA Monitor: Detected buffer change, copying %zu bytes to GPU\n",
           copy_size);

    // Copy changed data to GPU
    CUDA_CHECK(cudaMemcpyAsync(data.gpu_message_buffer, cpu_mapped_buffer,
                               copy_size, cudaMemcpyHostToDevice, data.stream));

    // Set trigger flag to notify the persistent kernel
    bool trigger = true;
    CUDA_CHECK(cudaMemcpyAsync(data.trigger_flag, &trigger, sizeof(bool),
                               cudaMemcpyHostToDevice, data.stream));

    // Ensure the copy operations complete
    CUDA_CHECK(cudaStreamSynchronize(data.stream));

    // Poll the completion flag until the kernel signals it's done
    bool completed = false;
    while (!completed) {
      CUDA_CHECK(cudaMemcpyAsync(&completed, data.completion_flag, sizeof(bool),
                                 cudaMemcpyDeviceToHost, data.stream));
      if (!completed)
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    printf("RDMA Monitor: Kernel processing completed, copying result back\n");

    // Copy the result back from GPU to CPU mapped buffer
    if (header->result_size > 0) {
      void *gpu_result_ptr =
          static_cast<char *>(data.gpu_message_buffer) + header->result_offset;
      void *cpu_result_ptr =
          static_cast<char *>(cpu_mapped_buffer) + header->result_offset;

      CUDA_CHECK(cudaMemcpyAsync(cpu_result_ptr, gpu_result_ptr,
                                 header->result_size, cudaMemcpyDeviceToHost,
                                 data.stream));
    }

    // Reset completion flag for next operation
    completed = false;
    CUDA_CHECK(cudaMemcpyAsync(data.completion_flag, &completed, sizeof(bool),
                               cudaMemcpyHostToDevice, data.stream));
    CUDA_CHECK(cudaStreamSynchronize(data.stream));

    last_processed_version.store(buffer_version.load());
    printf("RDMA Monitor: GPU transfer completed for message type %u\n",
           header->message_type);
  }
};

} // namespace cudaq::qclink
