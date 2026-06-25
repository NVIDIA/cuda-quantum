/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/realtime/cpu_transport/roce_transceiver.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

namespace {

using cudaq::realtime::bridge::CpuRoceTransceiver;
using cudaq::realtime::bridge::CpuRoceTxMode;
using namespace std::chrono_literals;

struct Config {
  std::string device = "rxe_cudaq0";
  std::string ip = "10.88.0.1";
  std::size_t frame_size = 64;
  std::size_t page_size = 4096;
  unsigned pages = 8;
  unsigned iterations = 4;
};

void usage(const char *argv0) {
  std::cout
      << "Usage: " << argv0 << " [options]\n"
      << "\n"
      << "Options:\n"
      << "  --device NAME       RDMA device name (default: rxe_cudaq0)\n"
      << "  --ip ADDR           Local/peer RoCE IPv4 address (default: "
         "10.88.0.1)\n"
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
    unsigned long value = std::stoul(text, &consumed, 0);
    if (consumed != std::strlen(text) || value == 0 ||
        value > static_cast<unsigned long>(UINT32_MAX))
      return false;
    out = static_cast<unsigned>(value);
    return true;
  } catch (...) {
    return false;
  }
}

bool parse_size(const char *text, std::size_t &out) {
  try {
    std::size_t consumed = 0;
    unsigned long long value = std::stoull(text, &consumed, 0);
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
    std::string arg = argv[i];
    auto need_value = [&](const char *name) -> const char * {
      if (i + 1 >= argc) {
        std::cerr << "ERROR: missing value for " << name << "\n";
        return nullptr;
      }
      return argv[++i];
    };

    if (arg == "--device") {
      if (const char *value = need_value("--device"))
        cfg.device = value;
      else
        return false;
    } else if (arg == "--ip") {
      if (const char *value = need_value("--ip"))
        cfg.ip = value;
      else
        return false;
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

int run_probe(const Config &cfg) {
  std::cout << "=== CpuRoceTransceiver SoftRoCE self-loop probe ===\n"
            << "device=" << cfg.device << " ip=" << cfg.ip
            << " frame_size=" << cfg.frame_size
            << " page_size=" << cfg.page_size << " pages=" << cfg.pages
            << " iterations=" << cfg.iterations << "\n";

  CpuRoceTransceiver requester(cfg.device.c_str(), 1, 0, cfg.frame_size,
                               cfg.page_size, cfg.pages, cfg.ip.c_str(), false,
                               false, false, false,
                               CpuRoceTxMode::kRdmaWriteWithImm, 0, 0);
  CpuRoceTransceiver responder(cfg.device.c_str(), 1, 0, cfg.frame_size,
                               cfg.page_size, cfg.pages, cfg.ip.c_str(), false,
                               false, false, false, CpuRoceTxMode::kRdmaSend, 0,
                               0);

  requester.set_local_ip(cfg.ip.c_str());
  responder.set_local_ip(cfg.ip.c_str());

  if (!requester.setup()) {
    std::cerr << "ERROR: requester setup() failed\n";
    return 1;
  }
  if (!responder.setup()) {
    std::cerr << "ERROR: responder setup() failed\n";
    return 1;
  }

  std::cout << "requester qp=" << requester.get_qp_number()
            << " rkey=" << requester.get_rkey() << "\n"
            << "responder qp=" << responder.get_qp_number()
            << " rkey=" << responder.get_rkey() << "\n";

  if (!requester.connect(responder.get_qp_number(), cfg.ip.c_str(),
                         responder.get_rkey())) {
    std::cerr << "ERROR: requester connect() failed\n";
    return 1;
  }
  if (!responder.connect(requester.get_qp_number(), cfg.ip.c_str(), 0)) {
    std::cerr << "ERROR: responder connect() failed\n";
    return 1;
  }

  auto *requester_rx_data = requester.get_rx_ring_data_addr();
  auto *requester_rx_flags = requester.get_rx_ring_flag_addr();
  auto *requester_tx_data = requester.get_tx_ring_data_addr();
  auto *requester_tx_flags = requester.get_tx_ring_flag_addr();

  auto *responder_rx_data = responder.get_rx_ring_data_addr();
  auto *responder_rx_flags = responder.get_rx_ring_flag_addr();
  auto *responder_tx_data = responder.get_tx_ring_data_addr();
  auto *responder_tx_flags = responder.get_tx_ring_flag_addr();

  std::thread requester_monitor;
  std::thread responder_monitor;
  bool started_monitors = false;
  auto stop_monitors = [&]() {
    requester.close();
    responder.close();
    if (requester_monitor.joinable())
      requester_monitor.join();
    if (responder_monitor.joinable())
      responder_monitor.join();
  };

  try {
    requester_monitor = std::thread([&]() { requester.blocking_monitor(); });
    responder_monitor = std::thread([&]() { responder.blocking_monitor(); });
    started_monitors = true;
  } catch (const std::exception &e) {
    std::cerr << "ERROR: failed to start monitor threads: " << e.what() << "\n";
    stop_monitors();
    return 1;
  }

  std::this_thread::sleep_for(100ms);

  for (unsigned iter = 0; iter < cfg.iterations; ++iter) {
    const unsigned slot = iter & (cfg.pages - 1);
    const std::uint64_t requester_tx_addr = reinterpret_cast<std::uint64_t>(
        requester_tx_data + static_cast<std::size_t>(slot) * cfg.page_size);
    const std::uint64_t responder_tx_addr = reinterpret_cast<std::uint64_t>(
        responder_tx_data + static_cast<std::size_t>(slot) * cfg.page_size);

    std::ostringstream req;
    req << "request-" << iter;
    std::ostringstream rsp;
    rsp << "response-" << iter;

    if (!wait_for(
            "requester tx flag to clear",
            [&]() { return load_flag(requester_tx_flags + slot) == 0; }) ||
        !wait_for("responder rx flag to clear", [&]() {
          return load_flag(responder_rx_flags + slot) == 0;
        })) {
      stop_monitors();
      return 1;
    }

    write_message(requester_tx_data +
                      static_cast<std::size_t>(slot) * cfg.page_size,
                  cfg.frame_size, req.str());
    store_flag(requester_tx_flags + slot, requester_tx_addr);

    if (!wait_for("responder rx flag", [&]() {
          return load_flag(responder_rx_flags + slot) != 0;
        })) {
      stop_monitors();
      return 1;
    }
    if (!expect_message("responder rx",
                        responder_rx_data +
                            static_cast<std::size_t>(slot) * cfg.page_size,
                        req.str())) {
      stop_monitors();
      return 1;
    }
    store_flag(responder_rx_flags + slot, 0);

    if (!wait_for(
            "responder tx flag to clear",
            [&]() { return load_flag(responder_tx_flags + slot) == 0; }) ||
        !wait_for("requester rx flag to clear", [&]() {
          return load_flag(requester_rx_flags + slot) == 0;
        })) {
      stop_monitors();
      return 1;
    }

    write_message(responder_tx_data +
                      static_cast<std::size_t>(slot) * cfg.page_size,
                  cfg.frame_size, rsp.str());
    store_flag(responder_tx_flags + slot, responder_tx_addr);

    if (!wait_for("requester rx flag", [&]() {
          return load_flag(requester_rx_flags + slot) != 0;
        })) {
      stop_monitors();
      return 1;
    }
    if (!expect_message("requester rx",
                        requester_rx_data +
                            static_cast<std::size_t>(slot) * cfg.page_size,
                        rsp.str())) {
      stop_monitors();
      return 1;
    }
    store_flag(requester_rx_flags + slot, 0);

    std::cout << "exchange " << iter << " passed on slot " << slot << "\n";
  }

  if (started_monitors)
    stop_monitors();

  std::cout << "CpuRoceTransceiver self-loop probe passed\n";
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
    return run_probe(cfg);
  } catch (const std::exception &e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
