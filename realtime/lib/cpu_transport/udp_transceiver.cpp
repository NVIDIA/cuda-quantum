/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/// @file udp_transceiver.cpp
/// @brief UDP ring transceiver: the plain-UDP counterpart of
/// CpuRoceTransceiver.
///
/// See udp_wrapper.h for the ring contract. The RX thread places one inbound
/// datagram into one RX ring slot in strict ring order (matching the FIFO
/// ordering the RoCE recv-WQE path provides); the TX thread ships any
/// published TX slot (tx_flag[slot] == slot data address) to the peer as one
/// full-stride datagram and clears the flag.

#include "cudaq/realtime/cpu_transport/udp_wrapper.h"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

namespace {

std::uint64_t load_flag(const volatile std::uint64_t *flag) {
  return __atomic_load_n(const_cast<const std::uint64_t *>(flag),
                         __ATOMIC_ACQUIRE);
}

void store_flag(volatile std::uint64_t *flag, std::uint64_t value) {
  __atomic_store_n(const_cast<std::uint64_t *>(flag), value, __ATOMIC_RELEASE);
}

std::uint8_t *aligned_ring(std::size_t bytes) {
  void *ptr = nullptr;
  if (posix_memalign(&ptr, 256, bytes) != 0)
    return nullptr;
  std::memset(ptr, 0, bytes);
  return static_cast<std::uint8_t *>(ptr);
}

class UdpTransceiver {
public:
  UdpTransceiver(std::size_t page_size, unsigned num_pages)
      : page_size(page_size), num_pages(num_pages) {
    rx_flags = reinterpret_cast<volatile std::uint64_t *>(
        aligned_ring(num_pages * sizeof(std::uint64_t)));
    tx_flags = reinterpret_cast<volatile std::uint64_t *>(
        aligned_ring(num_pages * sizeof(std::uint64_t)));
    rx_data = aligned_ring(num_pages * page_size);
    tx_data = aligned_ring(num_pages * page_size);
  }

  ~UdpTransceiver() {
    close();
    std::free(const_cast<std::uint64_t *>(rx_flags));
    std::free(const_cast<std::uint64_t *>(tx_flags));
    std::free(rx_data);
    std::free(tx_data);
  }

  bool valid() const { return rx_flags && tx_flags && rx_data && tx_data; }

  bool bind(const char *host, std::uint16_t port) {
    if (!openSocket())
      return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    if (!host || !*host)
      addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    else if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1)
      return false;
    addr.sin_port = htons(port);
    if (::bind(fd, reinterpret_cast<const sockaddr *>(&addr), sizeof(addr)) !=
        0)
      return false;
    socklen_t addrlen = sizeof(addr);
    ::getsockname(fd, reinterpret_cast<sockaddr *>(&addr), &addrlen);
    local_port = ntohs(addr.sin_port);
    return true;
  }

  bool connect(const char *host, std::uint16_t port) {
    if (!openSocket())
      return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    if (::inet_pton(AF_INET, host, &addr.sin_addr) != 1)
      return false;
    if (::connect(fd, reinterpret_cast<const sockaddr *>(&addr),
                  sizeof(addr)) != 0)
      return false;
    {
      std::lock_guard<std::mutex> lock(peer_mutex);
      peer_addr = addr;
      have_peer = true;
    }
    sockaddr_in local{};
    socklen_t locallen = sizeof(local);
    ::getsockname(fd, reinterpret_cast<sockaddr *>(&local), &locallen);
    local_port = ntohs(local.sin_port);
    return true;
  }

  bool start() {
    if (fd < 0 || running)
      return false;
    running = true;
    rx_thread = std::thread([this] { rxLoop(); });
    tx_thread = std::thread([this] { txLoop(); });
    return true;
  }

  void close() {
    running = false;
    if (rx_thread.joinable())
      rx_thread.join();
    if (tx_thread.joinable())
      tx_thread.join();
    if (fd >= 0) {
      ::close(fd);
      fd = -1;
    }
  }

  std::uint16_t port() const { return local_port; }
  volatile std::uint64_t *rxFlags() const { return rx_flags; }
  volatile std::uint64_t *txFlags() const { return tx_flags; }
  std::uint8_t *rxData() const { return rx_data; }
  std::uint8_t *txData() const { return tx_data; }

private:
  bool openSocket() {
    if (fd >= 0)
      return true;
    fd = ::socket(AF_INET, SOCK_DGRAM, 0);
    if (fd < 0)
      return false;
    // Bounded recv wait so rxLoop can observe shutdown.
    const timeval rx_poll_interval{0, 100000}; // 100 ms
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rx_poll_interval,
                 sizeof(rx_poll_interval));
    return true;
  }

  // One inbound datagram -> one RX slot, in strict ring order. Back-pressure
  // until the in-order slot is fully recycled: rx flag cleared by the consumer
  // AND the slot's response drained from the TX ring.
  void rxLoop() {
    std::vector<std::uint8_t> buffer(65536);
    unsigned next_slot = 0;
    while (running) {
      sockaddr_in from{};
      socklen_t fromlen = sizeof(from);
      const ssize_t got =
          ::recvfrom(fd, buffer.data(), buffer.size(), 0,
                     reinterpret_cast<sockaddr *>(&from), &fromlen);
      if (got <= 0)
        continue; // timeout/EINTR: re-check `running`
      if (static_cast<std::size_t>(got) > page_size) {
        // Stride mismatch with the peer; drop. Warn once -- a mismatch drops
        // EVERY request, which otherwise looks like a silent hang upstream.
        if (!oversize_warned.exchange(true))
          std::fprintf(stderr,
                       "[cudaq-udp-transport] dropping %zd-byte datagram: "
                       "exceeds this end's page_size (%zu); both ends must "
                       "use the same slot stride (further drops not logged)\n",
                       got, page_size);
        continue;
      }

      {
        std::lock_guard<std::mutex> lock(peer_mutex);
        peer_addr = from;
        have_peer = true;
      }

      const unsigned slot = next_slot;
      while (running && (load_flag(&rx_flags[slot]) != 0 ||
                         load_flag(&tx_flags[slot]) != 0))
        std::this_thread::sleep_for(std::chrono::microseconds(50));
      if (!running)
        return;

      std::uint8_t *rx_slot = rx_data + slot * page_size;
      std::memset(rx_slot, 0, page_size);
      std::memcpy(rx_slot, buffer.data(), static_cast<std::size_t>(got));
      store_flag(&rx_flags[slot], reinterpret_cast<std::uint64_t>(rx_slot));
      next_slot = (slot + 1) % num_pages;
    }
  }

  // Ship published TX slots to the peer as one full-stride datagram each and
  // clear them (this transport has no per-message frame size; see the "Wire
  // behavior" note in udp_wrapper.h).
  //
  // Slots are consumed in FIFO cursor order, NOT by index scan: both
  // producers (the udp device_call channel and a daemon mirroring request
  // slots to response slots) publish slots strictly round-robin, so cursor
  // order equals publish order. An index scan reorders any publish burst
  // that spans the ring wrap (slot N-1 and slot 0 pending together get sent
  // 0 first) -- fire-and-forget device_calls (enqueue_syndromes,
  // reset_decoder) then arrive at the peer out of program order, violating
  // the decoding server's in-order message contract
  // (decoder_server_runtime.md, Message ordering).
  void txLoop() {
    unsigned cursor = 0;
    while (running) {
      const std::uint64_t value = load_flag(&tx_flags[cursor]);
      if (value == 0) {
        std::this_thread::sleep_for(std::chrono::microseconds(50));
        continue;
      }
      const std::uint8_t *tx_slot = tx_data + cursor * page_size;
      if (value == reinterpret_cast<std::uint64_t>(tx_slot)) {
        std::lock_guard<std::mutex> lock(peer_mutex);
        if (have_peer)
          ::sendto(fd, tx_slot, page_size, 0,
                   reinterpret_cast<const sockaddr *>(&peer_addr),
                   sizeof(peer_addr));
      }
      // Non-address values (in-flight/error sentinels) are recycled too;
      // this transport has no side channel to report them.
      store_flag(&tx_flags[cursor], 0);
      cursor = (cursor + 1) % num_pages;
    }
  }

  const std::size_t page_size;
  const unsigned num_pages;
  volatile std::uint64_t *rx_flags = nullptr;
  volatile std::uint64_t *tx_flags = nullptr;
  std::uint8_t *rx_data = nullptr;
  std::uint8_t *tx_data = nullptr;
  int fd = -1;
  std::uint16_t local_port = 0;
  std::atomic<bool> running{false};
  std::atomic<bool> oversize_warned{false};
  std::thread rx_thread;
  std::thread tx_thread;
  std::mutex peer_mutex;
  sockaddr_in peer_addr{};
  bool have_peer = false;
};

UdpTransceiver *cast(cpu_udp_transceiver_t handle) {
  return static_cast<UdpTransceiver *>(handle);
}

} // namespace

extern "C" {

cpu_udp_transceiver_t cpu_udp_create_transceiver(size_t page_size,
                                                 unsigned num_pages) {
  if (page_size == 0 || num_pages == 0)
    return nullptr;
  auto *xcvr = new (std::nothrow) UdpTransceiver(page_size, num_pages);
  if (!xcvr || !xcvr->valid()) {
    delete xcvr;
    return nullptr;
  }
  return xcvr;
}

void cpu_udp_destroy_transceiver(cpu_udp_transceiver_t handle) {
  delete cast(handle);
}

int cpu_udp_bind_to(cpu_udp_transceiver_t handle, const char *host,
                    uint16_t port) {
  return handle && cast(handle)->bind(host, port) ? 1 : 0;
}

int cpu_udp_bind(cpu_udp_transceiver_t handle, uint16_t port) {
  return cpu_udp_bind_to(handle, /*host=*/nullptr, port);
}

int cpu_udp_connect(cpu_udp_transceiver_t handle, const char *host,
                    uint16_t port) {
  return handle && host && cast(handle)->connect(host, port) ? 1 : 0;
}

uint16_t cpu_udp_get_port(cpu_udp_transceiver_t handle) {
  return handle ? cast(handle)->port() : 0;
}

int cpu_udp_start(cpu_udp_transceiver_t handle) {
  return handle && cast(handle)->start() ? 1 : 0;
}

void cpu_udp_close(cpu_udp_transceiver_t handle) {
  if (handle)
    cast(handle)->close();
}

uint64_t cpu_udp_get_rx_ring_flag_addr(cpu_udp_transceiver_t handle) {
  return handle ? reinterpret_cast<uint64_t>(cast(handle)->rxFlags()) : 0;
}

uint64_t cpu_udp_get_rx_ring_data_addr(cpu_udp_transceiver_t handle) {
  return handle ? reinterpret_cast<uint64_t>(cast(handle)->rxData()) : 0;
}

uint64_t cpu_udp_get_tx_ring_flag_addr(cpu_udp_transceiver_t handle) {
  return handle ? reinterpret_cast<uint64_t>(cast(handle)->txFlags()) : 0;
}

uint64_t cpu_udp_get_tx_ring_data_addr(cpu_udp_transceiver_t handle) {
  return handle ? reinterpret_cast<uint64_t>(cast(handle)->txData()) : 0;
}

} // extern "C"
