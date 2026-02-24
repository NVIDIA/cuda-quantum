/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"

#include <cuda_runtime.h>
#include <cuda/std/atomic>
#include <cstddef>
#include <cstdint>
#include <vector>

#ifndef QEC_CPU_RELAX
#if defined(__x86_64__)
#include <immintrin.h>
#define QEC_CPU_RELAX() _mm_pause()
#elif defined(__aarch64__)
#define QEC_CPU_RELAX() __asm__ volatile("yield" ::: "memory")
#else
#define QEC_CPU_RELAX() do { } while (0)
#endif
#endif

namespace cudaq::realtime {

using atomic_uint64_sys = cuda::std::atomic<uint64_t>;
using atomic_int_sys = cuda::std::atomic<int>;

struct HostDispatchWorker {
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    uint32_t function_id;  // matches table entry; used to assign slot to this worker
};

struct HostDispatcherConfig {
    atomic_uint64_sys* rx_flags;
    atomic_uint64_sys* tx_flags;
    uint8_t* rx_data_host;
    uint8_t* rx_data_dev;
    uint8_t* tx_data_host;
    uint8_t* tx_data_dev;
    size_t tx_stride_sz;
    void** h_mailbox_bank;
    size_t num_slots;
    size_t slot_size;
    std::vector<HostDispatchWorker> workers;
    /// Host-visible function table for lookup by function_id (GRAPH_LAUNCH or HOST_CALL).
    cudaq_function_entry_t* function_table = nullptr;
    size_t function_table_count = 0;
    atomic_int_sys* shutdown_flag;
    uint64_t* stats_counter;
    /// Optional: atomic counter incremented on each dispatch (for progress diagnostics).
    atomic_uint64_sys* live_dispatched = nullptr;

    /// Dynamic worker pool (graph workers only; host calls run inline)
    atomic_uint64_sys* idle_mask;   ///< 1 = free, 0 = busy; bit index = worker_id
    int* inflight_slot_tags;        ///< worker_id -> origin FPGA slot for tx_flags routing
};

/// Run the host-side dispatcher loop. Blocks until *config.shutdown_flag
/// becomes non-zero. Call from a dedicated thread.
/// Uses dynamic worker pool: allocates via idle_mask, tags with inflight_slot_tags.
void host_dispatcher_loop(const HostDispatcherConfig& config);

} // namespace cudaq::realtime
