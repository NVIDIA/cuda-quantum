/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MQPUUtils.h"
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "common/RestClient.h"
#include "cudaq/utils/cudaq_utils.h"
#include "llvm/Support/Program.h"
#include <arpa/inet.h>
#include <execinfo.h>
#include <filesystem>
#include <numeric>
#include <random>
#include <set>
#include <signal.h>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#ifdef CUDAQ_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif

// Check if a TCP/IP port is available for use
bool portAvailable(int port) {
  struct sockaddr_in servAddr;
  ::bzero((char *)&servAddr, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = INADDR_ANY;
  servAddr.sin_port = port;
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0)
    return false;
  bool available =
      (::bind(sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) == 0);

  // Close the socket to avoid leaks
  if (::close(sock) != 0) {
    perror("Failed to close socket");
    return false;
  }

  return available;
}

// Util to pick (at random) an available TCP/IP port for auto-launching a server
// instance.
static std::optional<std::string> getRandomAvailablePort(int seed) {
  std::mt19937 gen(seed);
  std::uniform_int_distribution<> portDistr(49152,
                                            65534); // TCP/IP dynamic port range
  std::uniform_int_distribution<> delayDistr(1,
                                             100); // 1 - 100 ms
  // Try for a maximum of 100 times
  constexpr std::size_t MAX_RETRIES = 100;
  for (std::size_t i = 0; i < MAX_RETRIES; ++i) {
    const auto randomPort = portDistr(gen);
    if (portAvailable(randomPort)) {
      // Randomly delay some duration, then recheck the port again.
      // This is to mitigate potential race condition between port checking and
      // REST server process claiming the port.
      const int delayMs = delayDistr(gen);
      std::this_thread::sleep_for(std::chrono::milliseconds(delayMs));
      // If this is still available, save to use.
      if (portAvailable(randomPort))
        return std::to_string(randomPort);
    }
  }
  CUDAQ_INFO("Failed to find a random TCP/IP port after {} trials.",
             MAX_RETRIES);
  return {};
}

cudaq::AutoLaunchRestServerProcess::AutoLaunchRestServerProcess(
    int seed_offset) {
  CUDAQ_INFO("Auto launch REST server");
  const std::string serverExeName = "cudaq-qpud";
  const std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
  const auto binPath = cudaqLibPath.parent_path().parent_path() / "bin";
  CUDAQ_INFO("Search for {} in {} directory.", serverExeName, binPath.c_str());
  auto serverApp =
      llvm::sys::findProgramByName(serverExeName.c_str(), {binPath.c_str()});
  if (!serverApp)
    throw std::runtime_error("Unable to find CUDA-Q REST server to launch.");

  // If the CUDAQ_DYNLIBS env var is set (typically from the Python
  // environment), add these to the LD_LIBRARY_PATH.
  std::string libPaths;
  std::optional<llvm::SmallVector<llvm::StringRef>> Env;
  if (auto *ch = getenv("CUDAQ_DYNLIBS")) {
    Env = llvm::SmallVector<llvm::StringRef>();
    libPaths = ch;
    if (libPaths.size() > 0)
      libPaths += ":";

    std::set<std::string> libDirs;
    while (libPaths.size() > 0) {
      auto next_delim = libPaths.find(":");
      if (next_delim < libPaths.size()) {
        std::filesystem::path lib(libPaths.substr(0, next_delim));
        libPaths = libPaths.substr(next_delim + 1);
        auto libDir = lib.remove_filename().string();
        if (libDir.size() > 0)
          libDirs.insert(libDir);
      }
    }

    std::string dynLibs = std::accumulate(
        std::begin(libDirs), std::end(libDirs), std::string{},
        [](const std::string &a, const std::string &b) { return a + b + ':'; });
    dynLibs.pop_back();
    if (auto *p = getenv("LD_LIBRARY_PATH")) {
      std::string envLibs = p;
      if (envLibs.size() > 0)
        dynLibs += ":" + envLibs;
    }
    for (char **env = environ; *env != nullptr; ++env) {
      if (!std::string(*env).starts_with("LD_LIBRARY_PATH="))
        Env->push_back(*env);
    }
    // Cache the string as a member var to keep the pointer alive.
    m_ldLibPathEnv = "LD_LIBRARY_PATH=" + dynLibs;
    Env->push_back(m_ldLibPathEnv);
  }

  constexpr std::size_t PORT_MAX_RETRIES = 10;
  for (std::size_t j = 0; j < PORT_MAX_RETRIES; j++) {
    /// Step 1: Look up a port
    // Use process Id to seed the random port search to minimize collision.
    // For example, multiple processes trying to auto-launch server app on the
    // same machine.
    // Also, prevent collision when a single process (same PID) constructing
    // multiple AutoLaunchRestServerProcess in a loop by allowing to pass an
    // offset for the seed.
    static std::mt19937 gen(::getpid() * 100 + seed_offset);
    const auto port = getRandomAvailablePort(gen());
    if (!port.has_value())
      throw std::runtime_error("Unable to find a TCP/IP port on the local "
                               "machine for auto-launch a REST server.");

    /// Step 2: Create process and set URL
    llvm::StringRef argv[] = {serverApp.get(), "--port", port.value()};
    std::string errorMsg;
    bool executionFailed = false;
    auto processInfo = llvm::sys::ExecuteNoWait(serverApp.get(), argv, Env, {},
                                                0, &errorMsg, &executionFailed);
    if (executionFailed)
      throw std::runtime_error("Failed to launch " + serverExeName +
                               " at port " + port.value() + ": " + errorMsg);
    CUDAQ_INFO("Auto launch REST server at http://localhost:{} (PID {})",
               port.value(), processInfo.Pid);
    m_pid = processInfo.Pid;
    m_url = fmt::format("localhost:{}", port.value());

    /// Step 3: Ping to verify availability
    // Throttle the delay to ping the server (checking if it's ready).
    // The idea is to gradually increase the delay so that
    // (1) on a 'fast' system, we don't wait too much.
    // (2) on a 'slow' system (heavily loaded), we wait enough for the server
    // process to be launched, while not spamming the systems with unnecessary
    // ping requests.
    constexpr std::size_t MAX_RETRIES = 100;
    constexpr std::size_t POLL_INTERVAL_MAX_MS = 1000;
    constexpr std::size_t POLL_INTERVAL_MIN_MS = 10;
    const auto throttledDelay = [&](int i) {
      // Gradually prolong the delay:
      return POLL_INTERVAL_MIN_MS +
             (POLL_INTERVAL_MAX_MS - POLL_INTERVAL_MIN_MS) * i / MAX_RETRIES;
    };
    int totalWaitTimeMs = 0;
    cudaq::RestClient restClient;
    for (std::size_t i = 0; i < MAX_RETRIES; ++i) {
      try {
        std::map<std::string, std::string> headers;
        [[maybe_unused]] auto pingResult = restClient.get(m_url, "", headers);
        CUDAQ_INFO("Successfully connected to the REST server at "
                   "http://localhost:{} (PID {}) after {} milliseconds.",
                   port.value(), processInfo.Pid, totalWaitTimeMs);
        return;
      } catch (...) {
        // Wait and retry
        const auto delay = throttledDelay(i);
        totalWaitTimeMs += delay;
        std::this_thread::sleep_for(std::chrono::milliseconds(delay));
      }
    }
    CUDAQ_INFO(
        "Timeout Error: No response from the server. Look for another port...");
    CUDAQ_INFO("Shutting down REST server process {}", m_pid);
    ::kill(m_pid, SIGKILL);
  }
  throw std::runtime_error("No usable ports available");
}

cudaq::AutoLaunchRestServerProcess::~AutoLaunchRestServerProcess() {
  CUDAQ_INFO("Shutting down REST server process {}", m_pid);
  ::kill(m_pid, SIGKILL);
}

std::string cudaq::AutoLaunchRestServerProcess::getUrl() const { return m_url; }

int cudaq::getCudaGetDeviceCount() {
#ifdef CUDAQ_ENABLE_CUDA
  int nDevices{0};
  const auto status = cudaGetDeviceCount(&nDevices);
  return status != cudaSuccess ? 0 : nDevices;
#else
  return 0;
#endif
}
