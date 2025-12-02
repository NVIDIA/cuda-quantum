/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - Present NVIDIA Corporation & Affiliates.               *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/nvqlink/daemon/dispatcher/dispatcher.h"

#include <atomic>
#include <cuda_runtime.h>

namespace cudaq::nvqlink {

/// GPU-mode dispatcher using persistent CUDA kernel.
///
class GPUDispatcher : public Dispatcher {
public:
  GPUDispatcher(Channel *channel, FunctionRegistry *registry,
                const ComputeConfig &config);
  ~GPUDispatcher() override;

  void start() override;
  void stop() override;
  uint64_t get_packets_processed() const override;
  uint64_t get_packets_sent() const override;

private:
  void launch_persistent_kernel();

  Channel *channel_;
  FunctionRegistry *registry_;
  ComputeConfig config_;

  // GPU resources
  cudaStream_t stream_{nullptr};
  void *device_shutdown_flag_{nullptr};
  void *device_stats_{nullptr};

  std::atomic<bool> running_{false};
};

/// CUDA kernel declaration
///
/// @param rx_queue RX queue
/// @param tx_queue TX queue
/// @param buffer_pool Buffer pool
/// @param function_table Function table
/// @param function_ids Function IDs
/// @param func_count Function count
/// @param shutdown_flag Shutdown flag
/// @param stats Stats
/// @param num_blocks Number of blocks
/// @param threads_per_block Threads per block
/// @param stream Stream
///
void launch_daemon_kernel(void *rx_queue, void *tx_queue, void *buffer_pool,
                          void **function_table, uint32_t *function_ids,
                          size_t func_count, volatile int *shutdown_flag,
                          uint64_t *stats, uint32_t num_blocks,
                          uint32_t threads_per_block, cudaStream_t stream);

} // namespace cudaq::nvqlink
