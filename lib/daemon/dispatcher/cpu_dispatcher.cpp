/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/nvqlink/daemon/dispatcher/cpu_dispatcher.h"
#include "cudaq/nvqlink/utils/instrumentation/logger.h"
#include "cudaq/nvqlink/utils/instrumentation/profiler.h"
#include "cudaq/nvqlink/daemon/registry/function_registry.h"
#include "cudaq/nvqlink/network/serialization/input_stream.h"
#include "cudaq/nvqlink/network/serialization/output_stream.h"

#include <cstring>

using namespace cudaq::nvqlink;

// Simple RPC header format
struct __attribute__((packed)) RPCHeader {
  std::uint32_t function_id;
  std::uint32_t arg_len;
};

struct __attribute__((packed)) RPCResponse {
  int32_t status;
  std::uint32_t result_len;
};

CPUDispatcher::CPUDispatcher(Channel *channel, FunctionRegistry *registry,
                             const ComputeConfig &config)
    : channel_(channel), registry_(registry), config_(config) {
  // Detect backend execution model
  model_ = channel_->get_execution_model();
}

CPUDispatcher::~CPUDispatcher() { stop(); }

void CPUDispatcher::start() {
  NVQLINK_TRACE_FULL(DOMAIN_DISPATCHER, "CPUDispatcher::start");
  if (running_.exchange(true)) {
    return;
  }

  NVQLINK_LOG_INFO(
      DOMAIN_DISPATCHER, "Starting CPU dispatcher with {} cores in {} mode",
      config_.cpu_cores.size(),
      (model_ == ChannelModel::POLLING ? "POLLING" : "EVENT-DRIVEN"));

  // For event-driven model, register our callback with backend
  if (model_ == ChannelModel::EVENT_DRIVEN)
    channel_->register_packet_callback(packet_received_callback, this);

  // Launch worker threads
  for (auto core : config_.cpu_cores) {
    if (model_ == ChannelModel::POLLING)
      threads_.emplace_back(&CPUDispatcher::polling_worker_thread, this, core);
    else
      threads_.emplace_back(&CPUDispatcher::event_driven_worker_thread, this,
                            core);
  }
}

void CPUDispatcher::stop() {
  if (!running_.exchange(false))
    return;

  for (auto &thread : threads_) {
    if (thread.joinable())
      thread.join();
  }
  threads_.clear();

  NVQLINK_LOG_INFO(DOMAIN_DISPATCHER,
                   "CPU dispatcher stopped. Processed {} packets",
                   packets_processed_.load());
}

std::uint64_t CPUDispatcher::get_packets_processed() const {
  return packets_processed_.load();
}

std::uint64_t CPUDispatcher::get_packets_sent() const {
  return packets_sent_.load();
}

void CPUDispatcher::polling_worker_thread(std::uint32_t core_id) {
  NVQLINK_TRACE_FULL(DOMAIN_DISPATCHER, "worker_thread_init");
  NVQLINK_TRACE_NAME_THREAD(("Dispatcher-" + std::to_string(core_id)).c_str());

  // TODO: set CPU affinity to core_id

  constexpr std::uint32_t BURST_SIZE = 32;
  Buffer *rx_buffers[BURST_SIZE];

  while (running_.load()) {
    // Poll for packets (batch-level instrumentation)
    std::uint32_t received;
    {
      NVQLINK_TRACE_HOTPATH(DOMAIN_DISPATCHER, "rx_batch");
      received = channel_->receive_burst(rx_buffers, BURST_SIZE);
    }

    if (received == 0) {
      // No packets, yield
      std::this_thread::yield();
      continue;
    }

    // Process batch (one range for entire batch)
    {
      NVQLINK_TRACE_HOTPATH(DOMAIN_DISPATCHER, "process_batch");
      for (std::uint32_t i = 0; i < received; ++i) {
        process_packet(rx_buffers[i]);
        packets_processed_.fetch_add(1);
      }
    }
  }
}

void CPUDispatcher::event_driven_worker_thread(std::uint32_t core_id) {
  // TODO: set CPU affinity to core_id

  while (running_.load()) {
    channel_->process_events();

    // Brief yield to prevent tight loop
    std::this_thread::yield();
  }
}

void CPUDispatcher::packet_received_callback(Buffer **buffers,
                                             std::uint32_t count,
                                             void *user_data) {
  auto *dispatcher = static_cast<CPUDispatcher *>(user_data);

  // Backend pushed packets to us via callback
  for (std::uint32_t i = 0; i < count; i++) {
    dispatcher->process_packet(buffers[i]);
    dispatcher->packets_processed_.fetch_add(1);
  }
}

void CPUDispatcher::process_packet(Buffer *buffer) {
  // Parse RPC header (zero-copy: read directly from buffer)
  auto *header = reinterpret_cast<RPCHeader *>(buffer->get_data());

  // Lookup function
  const auto *func_meta = registry_->lookup(header->function_id);
  if (!func_meta) {
    // Unknown function, drop packet
    // Copy packed field to avoid binding issue
    std::uint32_t fid = header->function_id;
    NVQLINK_LOG_ERROR(DOMAIN_DISPATCHER,
                      "CPUDispatcher: Unknown function_id={}", fid);
    channel_->release_buffer(buffer); // Release buffer (we still own it)
    return;
  }

  // Create type-safe streams for the user function
  // Both streams point to the same buffer (zero-copy)
  std::uint32_t arg_offset = sizeof(RPCHeader);
  void *arg_start = static_cast<char *>(buffer->get_data()) + arg_offset;

  // Input stream: read arguments from buffer
  InputStream input_stream(arg_start, header->arg_len);

  // Output stream: write results at BEGINNING of buffer
  // (will overwrite RPCHeader + arguments, which is fine - we're done reading
  // them)
  OutputStream output_stream(buffer->get_data(), func_meta->max_result_size);

  // Execute user function with type-safe streams
  int status = func_meta->cpu_function(input_stream, output_stream);

  // Update buffer length to reflect result data (overwrote RPCHeader + args)
  buffer->set_data_length(output_stream.bytes_written());

  // Prepare response (reuse buffer, prepend response header)
  auto *response =
      reinterpret_cast<RPCResponse *>(buffer->prepend(sizeof(RPCResponse)));
  response->status = status;
  response->result_len = output_stream.bytes_written();

  // Final length: RPCResponse + result
  buffer->set_data_length(sizeof(RPCResponse) + output_stream.bytes_written());

  // Send response (backend takes ownership and will deallocate after TX)
  uint32_t sent = channel_->send_burst(&buffer, 1);
  packets_sent_.fetch_add(sent);

  if (sent == 0)
    NVQLINK_TRACE_MARK_ERROR(DOMAIN_DISPATCHER, "TX_FAIL");
}
