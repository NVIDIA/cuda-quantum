/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/realtime/daemon/dispatcher/cudaq_realtime.h"
#include "cudaq/realtime/daemon/dispatcher/dispatch_kernel_launch.h"
#include <cstdint>
#include <cstring>
#include <thread>
#include <vector>

inline void write_rpc_request(cudaq_ringbuffer_t &ringbuffer,
                              const std::vector<std::uint8_t> &payload,
                              int num_repeats = 10) {
  constexpr std::size_t slot_size = 256;
  constexpr std::size_t slot = 0;
  constexpr uint32_t FUNC_ID = cudaq::realtime::fnv1a_hash("rpc_increment");
  // Write RPC request asynchronously (on another thread)
  std::thread rpc_thread([&]() {
    for (int i = 0; i < num_repeats; i++) {
      printf("Writing RPC request %d...\n", i + 1);
      std::uint8_t *slot_data =
          const_cast<std::uint8_t *>(ringbuffer.rx_data) + slot * slot_size;
      auto *header = reinterpret_cast<cudaq::realtime::RPCHeader *>(slot_data);
      header->magic = cudaq::realtime::RPC_MAGIC_REQUEST;
      header->function_id = FUNC_ID;
      header->arg_len = static_cast<std::uint32_t>(payload.size());
      cudaMemcpy(slot_data + sizeof(cudaq::realtime::RPCHeader), payload.data(),
                 payload.size(), cudaMemcpyHostToDevice);
      __sync_synchronize();
      // Cuda memcpy to rx_flags to mark the slot as ready for processing by the
      // dispatch kernel
      const uint64_t flag_val =
          1; // Any non-zero value indicates the slot is ready
      cudaMemcpy(const_cast<uint64_t *>(ringbuffer.rx_flags) + slot, &flag_val,
                 sizeof(uint64_t), cudaMemcpyHostToDevice);
      std::this_thread::sleep_for(std::chrono::milliseconds(
          10)); // Sleep to simulate time between requests
    }
  });
  rpc_thread.detach(); // Detach the thread to allow it to run independently
}

inline void read_rpc_response(cudaq_ringbuffer_t &ringbuffer,
                              const std::vector<std::uint8_t> &payload,
                              int num_repeats = 10) {
  constexpr std::size_t slot = 0;
  constexpr std::size_t slot_size = 256;
  int count = 0;
  std::vector<std::uint8_t> expected(payload.size());
  for (size_t i = 0; i < payload.size(); i++) {
    expected[i] = static_cast<std::uint8_t>((payload[i] + 1) & 0xFF);
  }
  std::thread rpc_thread([&]() {
    while (true && count < num_repeats) {
      uint64_t flag_val = 0;
      cudaMemcpy(&flag_val, (void*)(ringbuffer.tx_flags + slot), sizeof(uint64_t),
                 cudaMemcpyDeviceToHost);
      if (flag_val != 0) {
        printf("Received RPC response for run %d\n", ++count);
        __sync_synchronize();

        // Read from TX buffer (dispatch kernel writes response to symmetric TX)
        const std::uint8_t *slot_data =
            const_cast<std::uint8_t *>(ringbuffer.tx_data) + slot * slot_size;
        cudaq::realtime::RPCResponse response;
        cudaMemcpy(&response, slot_data, sizeof(cudaq::realtime::RPCResponse),
                   cudaMemcpyDeviceToHost);

        if (response.magic != cudaq::realtime::RPC_MAGIC_RESPONSE)
          printf("  Invalid magic: 0x%08x\n", response.magic);
        else if (response.status != 0)
          printf("  Error status: %d\n", response.status);

        std::vector<std::uint8_t> result(response.result_len);
        memcpy(result.data(), slot_data + sizeof(cudaq::realtime::RPCResponse),
               response.result_len);
        if (result != expected) {
          printf("  Incorrect result:");
          for (size_t i = 0; i < result.size(); i++) {
            printf(" %02x", result[i]);
          }
          printf("\n  Expected:");
          for (size_t i = 0; i < expected.size(); i++) {
            printf(" %02x", expected[i]);
            printf("\n ");
          }
        } else {
          printf("  Result is correct!\n");
        }
      }

      std::this_thread::sleep_for(
          std::chrono::milliseconds(1)); // Sleep to avoid busy-waiting
    }
  });
  rpc_thread.detach();
}
