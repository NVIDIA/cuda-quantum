/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/cpu_transport/roce_transceiver.hpp"

#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

namespace {

using cudaq::realtime::bridge::CpuRoceTransceiver;
using cudaq::realtime::bridge::CpuRoceTxMode;
using namespace std::chrono_literals;

struct Config {
  std::string role;
  std::string device = "rxe_cudaq0";
  std::string ip = "10.88.0.1";
  std::string rendezvous_host = "127.0.0.1";
  uint16_t rendezvous_port = 39017;
  std::size_t frame_size = 64;
  std::size_t page_size = 4096;
  unsigned pages = 8;
  unsigned iterations = 4;
};

void usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " --role server|client [options]\n"
      << "\n"
      << "Options:\n"
      << "  --role ROLE         server or client\n"
      << "  --device NAME       RDMA device name (default: rxe_cudaq0)\n"
      << "  --ip ADDR           Local/peer RoCE IPv4 address (default: "
         "10.88.0.1)\n"
      << "  --host ADDR         TCP rendezvous host (default: 127.0.0.1)\n"
      << "  --port N            TCP rendezvous port (default: 39017)\n"
      << "  --frame-size N      Logical frame size in bytes (default: 64)\n"
      << "  --page-size N       Ring slot stride in bytes (default: 4096)\n"
      << "  --pages N           Ring slots, power of two (default: 8)\n"
      << "  --iterations N      Request/response exchanges, <= pages (default: "
         "4)\n"
      << "  -h, --help          Show this help\n";
}

bool parse_unsigned(const char *text, unsigned &out) {
  try {
    std::size_t consumed = 0;
    const unsigned long value = std::stoul(text, &consumed, 0);
    if (consumed != std::strlen(text) || value == 0 ||
        value > std::numeric_limits<unsigned>::max())
      return false;
    out = static_cast<unsigned>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_port(const char *text, uint16_t &out) {
  unsigned value = 0;
  if (!parse_unsigned(text, value) || value > 65535)
    return false;
  out = static_cast<uint16_t>(value);
  return true;
}

bool parse_size(const char *text, std::size_t &out) {
  try {
    std::size_t consumed = 0;
    const unsigned long long value = std::stoull(text, &consumed, 0);
    if (consumed != std::strlen(text) || value == 0)
      return false;
    out = static_cast<std::size_t>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_args(int argc, char **argv, Config &cfg) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "ERROR: missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--role") {
      if (const char *value = need_value("--role"))
        cfg.role = value;
      else
        return false;
    } else if (arg == "--device") {
      if (const char *value = need_value("--device"))
        cfg.device = value;
      else
        return false;
    } else if (arg == "--ip") {
      if (const char *value = need_value("--ip"))
        cfg.ip = value;
      else
        return false;
    } else if (arg == "--host") {
      if (const char *value = need_value("--host"))
        cfg.rendezvous_host = value;
      else
        return false;
    } else if (arg == "--port") {
      const char *value = need_value("--port");
      if (!value || !parse_port(value, cfg.rendezvous_port)) {
        std::cerr << "ERROR: invalid --port\n";
        return false;
      }
    } else if (arg == "--frame-size") {
      const char *value = need_value("--frame-size");
      if (!value || !parse_size(value, cfg.frame_size)) {
        std::cerr << "ERROR: invalid --frame-size\n";
        return false;
      }
    } else if (arg == "--page-size") {
      const char *value = need_value("--page-size");
      if (!value || !parse_size(value, cfg.page_size)) {
        std::cerr << "ERROR: invalid --page-size\n";
        return false;
      }
    } else if (arg == "--pages") {
      const char *value = need_value("--pages");
      if (!value || !parse_unsigned(value, cfg.pages)) {
        std::cerr << "ERROR: invalid --pages\n";
        return false;
      }
    } else if (arg == "--iterations") {
      const char *value = need_value("--iterations");
      if (!value || !parse_unsigned(value, cfg.iterations)) {
        std::cerr << "ERROR: invalid --iterations\n";
        return false;
      }
    } else if (arg == "-h" || arg == "--help") {
      usage(argv[0]);
      std::exit(0);
    } else {
      std::cerr << "ERROR: unknown option: " << arg << "\n";
      return false;
    }
  }

  if (cfg.role != "server" && cfg.role != "client") {
    std::cerr << "ERROR: --role must be server or client\n";
    return false;
  }
  if ((cfg.pages & (cfg.pages - 1)) != 0) {
    std::cerr << "ERROR: --pages must be a power of two\n";
    return false;
  }
  if (cfg.iterations > cfg.pages) {
    std::cerr << "ERROR: --iterations must be <= --pages so the probe does "
                 "not wrap receive WQEs\n";
    return false;
  }
  if (cfg.frame_size > cfg.page_size) {
    std::cerr << "ERROR: --frame-size must be <= --page-size\n";
    return false;
  }
  return true;
}

class UniqueFd {
public:
  explicit UniqueFd(int fd = -1) : fd_(fd) {}
  ~UniqueFd() { reset(); }

  UniqueFd(const UniqueFd &) = delete;
  UniqueFd &operator=(const UniqueFd &) = delete;

  UniqueFd(UniqueFd &&other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd &operator=(UniqueFd &&other) noexcept {
    if (this != &other) {
      reset();
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }

  int get() const { return fd_; }
  int release() {
    const int fd = fd_;
    fd_ = -1;
    return fd;
  }
  void reset(int fd = -1) {
    if (fd_ >= 0)
      ::close(fd_);
    fd_ = fd;
  }

private:
  int fd_ = -1;
};

[[noreturn]] void throw_errno(const std::string &what) {
  throw std::runtime_error(what + ": " + std::strerror(errno));
}

void send_all(int fd, const void *buffer, std::size_t bytes) {
  const auto *ptr = static_cast<const std::uint8_t *>(buffer);
  while (bytes > 0) {
    const ssize_t n = ::send(fd, ptr, bytes, MSG_NOSIGNAL);
    if (n < 0)
      throw_errno("send");
    if (n == 0)
      throw std::runtime_error("send returned 0");
    ptr += n;
    bytes -= static_cast<std::size_t>(n);
  }
}

void recv_all(int fd, void *buffer, std::size_t bytes) {
  auto *ptr = static_cast<std::uint8_t *>(buffer);
  while (bytes > 0) {
    const ssize_t n = ::recv(fd, ptr, bytes, 0);
    if (n < 0)
      throw_errno("recv");
    if (n == 0)
      throw std::runtime_error("peer closed rendezvous socket");
    ptr += n;
    bytes -= static_cast<std::size_t>(n);
  }
}

UniqueFd listen_on_loopback(uint16_t port) {
  UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
  if (fd.get() < 0)
    throw_errno("socket");

  int one = 1;
  if (::setsockopt(fd.get(), SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one)) != 0)
    throw_errno("setsockopt(SO_REUSEADDR)");

  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  if (::bind(fd.get(), reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0)
    throw_errno("bind");
  if (::listen(fd.get(), 1) != 0)
    throw_errno("listen");
  return fd;
}

UniqueFd accept_one(int listen_fd) {
  UniqueFd fd(::accept(listen_fd, nullptr, nullptr));
  if (fd.get() < 0)
    throw_errno("accept");
  return fd;
}

UniqueFd connect_with_retry(const std::string &host, uint16_t port) {
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port);
  if (::inet_pton(AF_INET, host.c_str(), &addr.sin_addr) != 1)
    throw std::runtime_error("bad rendezvous host: " + host);

  const auto deadline = std::chrono::steady_clock::now() + 10s;
  while (std::chrono::steady_clock::now() < deadline) {
    UniqueFd fd(::socket(AF_INET, SOCK_STREAM, 0));
    if (fd.get() < 0)
      throw_errno("socket");
    if (::connect(fd.get(), reinterpret_cast<sockaddr *>(&addr),
                  sizeof(addr)) == 0)
      return fd;
    std::this_thread::sleep_for(100ms);
  }
  throw std::runtime_error("timed out connecting to rendezvous server");
}

void send_peer_info(int fd, std::uint32_t qp, std::uint32_t rkey) {
  std::uint32_t wire[2] = {htonl(qp), htonl(rkey)};
  send_all(fd, wire, sizeof(wire));
}

void recv_peer_info(int fd, std::uint32_t &qp, std::uint32_t &rkey) {
  std::uint32_t wire[2] = {};
  recv_all(fd, wire, sizeof(wire));
  qp = ntohl(wire[0]);
  rkey = ntohl(wire[1]);
}

std::uint64_t load_flag(std::uint64_t *flag) {
  return __atomic_load_n(flag, __ATOMIC_ACQUIRE);
}

void store_flag(std::uint64_t *flag, std::uint64_t value) {
  __atomic_store_n(flag, value, __ATOMIC_RELEASE);
}

template <typename Predicate>
bool wait_for(const std::string &what, Predicate pred,
              std::chrono::milliseconds timeout = 5s) {
  const auto deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline) {
    if (pred())
      return true;
    std::this_thread::sleep_for(1ms);
  }
  std::cerr << "ERROR: timed out waiting for " << what << "\n";
  return false;
}

void write_message(std::uint8_t *slot, std::size_t frame_size,
                   const std::string &message) {
  std::memset(slot, 0, frame_size);
  const std::size_t copy_bytes =
      std::min(frame_size - 1, static_cast<std::size_t>(message.size()));
  std::memcpy(slot, message.data(), copy_bytes);
}

bool expect_message(const char *label, const std::uint8_t *slot,
                    const std::string &expected) {
  const char *actual = reinterpret_cast<const char *>(slot);
  if (std::strncmp(actual, expected.c_str(), expected.size() + 1) == 0)
    return true;

  std::cerr << "ERROR: " << label << " payload mismatch\n"
            << "  expected: " << expected << "\n"
            << "  actual:   " << actual << "\n";
  return false;
}

std::string request_text(unsigned iter) {
  std::ostringstream os;
  os << "request-" << iter;
  return os.str();
}

std::string response_text(unsigned iter) {
  std::ostringstream os;
  os << "response-" << iter;
  return os.str();
}

void start_monitor(CpuRoceTransceiver &xcvr, std::thread &monitor) {
  monitor = std::thread([&]() { xcvr.blocking_monitor(); });
  std::this_thread::sleep_for(100ms);
}

void stop_monitor(CpuRoceTransceiver &xcvr, std::thread &monitor) {
  xcvr.close();
  if (monitor.joinable())
    monitor.join();
}

int run_server(const Config &cfg) {
  std::cout << "=== CpuRoceTransceiver two-process server ===\n"
            << "device=" << cfg.device << " ip=" << cfg.ip
            << " port=" << cfg.rendezvous_port
            << " iterations=" << cfg.iterations << "\n";

  CpuRoceTransceiver responder(cfg.device.c_str(), 1, 0, cfg.frame_size,
                               cfg.page_size, cfg.pages, cfg.ip.c_str(), false,
                               false, false, false, CpuRoceTxMode::kRdmaSend, 0,
                               0);
  responder.set_local_ip(cfg.ip.c_str());
  if (!responder.setup()) {
    std::cerr << "ERROR: responder setup() failed\n";
    return 1;
  }

  UniqueFd listener = listen_on_loopback(cfg.rendezvous_port);
  std::cout << "TWO_PROCESS_SERVER_READY port=" << cfg.rendezvous_port
            << " qp=" << responder.get_qp_number()
            << " rkey=" << responder.get_rkey() << std::endl;

  UniqueFd conn = accept_one(listener.get());
  std::uint32_t peer_qp = 0;
  std::uint32_t peer_rkey = 0;
  recv_peer_info(conn.get(), peer_qp, peer_rkey);
  send_peer_info(conn.get(), responder.get_qp_number(), responder.get_rkey());
  std::cout << "server peer qp=" << peer_qp << " rkey=" << peer_rkey << "\n";

  if (!responder.connect(peer_qp, cfg.ip.c_str(), 0)) {
    std::cerr << "ERROR: responder connect() failed\n";
    return 1;
  }

  auto *rx_data = responder.get_rx_ring_data_addr();
  auto *rx_flags = responder.get_rx_ring_flag_addr();
  auto *tx_data = responder.get_tx_ring_data_addr();
  auto *tx_flags = responder.get_tx_ring_flag_addr();

  std::thread monitor;
  start_monitor(responder, monitor);

  for (unsigned iter = 0; iter < cfg.iterations; ++iter) {
    const unsigned slot = iter & (cfg.pages - 1);
    if (!wait_for("server rx flag",
                  [&]() { return load_flag(rx_flags + slot) != 0; })) {
      stop_monitor(responder, monitor);
      return 1;
    }
    if (!expect_message("server rx",
                        rx_data +
                            static_cast<std::size_t>(slot) * cfg.page_size,
                        request_text(iter))) {
      stop_monitor(responder, monitor);
      return 1;
    }
    store_flag(rx_flags + slot, 0);

    if (!wait_for("server tx flag to clear",
                  [&]() { return load_flag(tx_flags + slot) == 0; })) {
      stop_monitor(responder, monitor);
      return 1;
    }

    auto *tx_slot = tx_data + static_cast<std::size_t>(slot) * cfg.page_size;
    write_message(tx_slot, cfg.frame_size, response_text(iter));
    store_flag(tx_flags + slot, reinterpret_cast<std::uint64_t>(tx_slot));

    if (!wait_for("server tx flag post claim",
                  [&]() { return load_flag(tx_flags + slot) == 0; })) {
      stop_monitor(responder, monitor);
      return 1;
    }
    std::cout << "server exchange " << iter << " passed on slot " << slot
              << "\n";
  }

  stop_monitor(responder, monitor);
  std::cout << "CpuRoceTransceiver two-process server passed\n";
  return 0;
}

int run_client(const Config &cfg) {
  std::cout << "=== CpuRoceTransceiver two-process client ===\n"
            << "device=" << cfg.device << " ip=" << cfg.ip
            << " host=" << cfg.rendezvous_host
            << " port=" << cfg.rendezvous_port
            << " iterations=" << cfg.iterations << "\n";

  CpuRoceTransceiver requester(cfg.device.c_str(), 1, 0, cfg.frame_size,
                               cfg.page_size, cfg.pages, cfg.ip.c_str(), false,
                               false, false, false,
                               CpuRoceTxMode::kRdmaWriteWithImm, 0, 0);
  requester.set_local_ip(cfg.ip.c_str());
  if (!requester.setup()) {
    std::cerr << "ERROR: requester setup() failed\n";
    return 1;
  }

  UniqueFd conn = connect_with_retry(cfg.rendezvous_host, cfg.rendezvous_port);
  send_peer_info(conn.get(), requester.get_qp_number(), requester.get_rkey());
  std::uint32_t peer_qp = 0;
  std::uint32_t peer_rkey = 0;
  recv_peer_info(conn.get(), peer_qp, peer_rkey);
  std::cout << "client peer qp=" << peer_qp << " rkey=" << peer_rkey << "\n";

  if (!requester.connect(peer_qp, cfg.ip.c_str(), peer_rkey)) {
    std::cerr << "ERROR: requester connect() failed\n";
    return 1;
  }

  auto *rx_data = requester.get_rx_ring_data_addr();
  auto *rx_flags = requester.get_rx_ring_flag_addr();
  auto *tx_data = requester.get_tx_ring_data_addr();
  auto *tx_flags = requester.get_tx_ring_flag_addr();

  std::thread monitor;
  start_monitor(requester, monitor);

  for (unsigned iter = 0; iter < cfg.iterations; ++iter) {
    const unsigned slot = iter & (cfg.pages - 1);
    if (!wait_for("client tx flag to clear",
                  [&]() { return load_flag(tx_flags + slot) == 0; }) ||
        !wait_for("client rx flag to clear",
                  [&]() { return load_flag(rx_flags + slot) == 0; })) {
      stop_monitor(requester, monitor);
      return 1;
    }

    auto *tx_slot = tx_data + static_cast<std::size_t>(slot) * cfg.page_size;
    write_message(tx_slot, cfg.frame_size, request_text(iter));
    store_flag(tx_flags + slot, reinterpret_cast<std::uint64_t>(tx_slot));

    if (!wait_for("client rx flag",
                  [&]() { return load_flag(rx_flags + slot) != 0; })) {
      stop_monitor(requester, monitor);
      return 1;
    }
    if (!expect_message("client rx",
                        rx_data +
                            static_cast<std::size_t>(slot) * cfg.page_size,
                        response_text(iter))) {
      stop_monitor(requester, monitor);
      return 1;
    }
    store_flag(rx_flags + slot, 0);
    std::cout << "client exchange " << iter << " passed on slot " << slot
              << "\n";
  }

  stop_monitor(requester, monitor);
  std::cout << "CpuRoceTransceiver two-process client passed\n";
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  Config cfg;
  if (!parse_args(argc, argv, cfg)) {
    usage(argv[0]);
    return 2;
  }

  try {
    return cfg.role == "server" ? run_server(cfg) : run_client(cfg);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
