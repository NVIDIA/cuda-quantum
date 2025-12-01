/****************************************************************-*- C++ -*-****
 * Copyright (c) 2024 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "rdma/cpu_gpu_rdma.h"
#include "rdma/persistent.cuh"

#include "cudaq/nvqlink/device.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <fstream>
#include <functional>
#include <memory>
#include <thread>
#include <unordered_map>
#include <vector>

#include "cuda_runtime.h"

namespace cudaq::nvqlink {

void launch_persistent_kernel_impl(persistent_kernel_data &data,
                                   std::size_t buffer_size,
                                   dispatch_func_t *&function_table,
                                   std::size_t num_funcs) {
  // Launch persistent kernel
  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));

  persistent_rdma_dispatcher_kernel<<<1, 32, 0, stream>>>(
      static_cast<unsigned char *>(data.gpu_message_buffer), buffer_size,
      data.trigger_flag, data.completion_flag, function_table, num_funcs);

  CUDA_CHECK(cudaGetLastError());
  printf("Persistent RDMA dispatcher kernel launched\n");
}

template <typename RDMADataT>
class extensible_rdma_device : public device_mixin<rdma_trait<RDMADataT>> {
protected:
  dispatch_func_t *device_func_table;
  std::vector<dispatch_func_t> host_func_table;

  virtual void build_device_function_table() = 0;

private:
  RDMADataT m_rdma_data;
  persistent_kernel_data m_data;
  size_t buffer_size = 64 * 1024; // 64KB buffer

  // RDMA emulation thread components
  std::unique_ptr<std::thread> rdma_monitor_thread;
  std::atomic<bool> monitor_thread_active{false};
  std::condition_variable buffer_cv;

  cudaStream_t stream;

  void start_rdma_monitor_thread() {
    monitor_thread_active.store(true);
    rdma_monitor_thread =
        std::make_unique<std::thread>([this]() { this->rdma_monitor_loop(); });
    printf("RDMA monitor thread started\n");
  }

  void stop_rdma_monitor_thread() {
    if (monitor_thread_active.load()) {
      monitor_thread_active.store(false);
      buffer_cv.notify_all(); // Wake up the monitoring thread
      if (rdma_monitor_thread && rdma_monitor_thread->joinable()) {
        rdma_monitor_thread->join();
      }
      printf("RDMA monitor thread stopped\n");
    }
  }

  void rdma_monitor_loop() {
    const auto poll_interval = std::chrono::microseconds(100); // 100μs polling

    while (monitor_thread_active.load()) {
      try {
        if (m_rdma_data.detected_buffer_change()) {
          m_rdma_data.process_buffer_change(m_data);
        }

        // Small sleep to prevent excessive CPU usage
        std::this_thread::sleep_for(poll_interval);

      } catch (const std::exception &e) {
        printf("Error in RDMA monitor thread: %s\n", e.what());
        // Continue monitoring despite errors
      }
    }
    printf("RDMA monitor thread exiting\n");
  }

  void cleanup_gpu_resources() {
    m_data.cleanup();

    if (device_func_table)
      cudaFree(device_func_table);
  }

public:
  extensible_rdma_device() {}
  ~extensible_rdma_device() {
    if (monitor_thread_active.load()) {
      stop_rdma_monitor_thread();
    }
    cudaStreamDestroy(m_data.stream);
  }

  void connect() override {
    printf("Connecting RDMA persistent device...\n");
    CUDA_CHECK(cudaStreamCreate(&m_data.stream));
    build_device_function_table();
    printf("we have %lu functions on this device.\n", host_func_table.size());

    m_rdma_data.connect(m_data);
    launch_persistent_kernel_impl(m_data, buffer_size, device_func_table,
                                  host_func_table.size());

    // Start the RDMA monitoring thread
    start_rdma_monitor_thread();

    printf("RDMA persistent device connected successfully\n");
  }

  void disconnect() override {
    printf("Disconnecting RDMA persistent device...\n");
    // Stop the monitoring thread first
    stop_rdma_monitor_thread();

    m_rdma_data.disconnect(m_data);

    cleanup_gpu_resources();

    printf("RDMA persistent device disconnected\n");
  }

  RDMADataT &get_rdma_connection_data() { return m_rdma_data; }
};

using cpu_gpu_rdma_device = extensible_rdma_device<cpu_gpu_rdma_data>;
// Add more...

} // namespace cudaq::nvqlink
