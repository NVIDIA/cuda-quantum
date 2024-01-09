/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MQPUUtils.h"
#include "common/Logger.h"
#include "llvm/Support/Program.h"
#include <arpa/inet.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/socket.h>

#ifdef CUDAQ_ENABLE_CUDA
#include "cuda_runtime_api.h"
#endif

// Util to query an available TCP/IP port for auto-launching a server
// instance.
static std::optional<std::string> getAvailablePort() {
  int sock = ::socket(AF_INET, SOCK_STREAM, 0);
  if (sock < 0)
    return {};
  struct sockaddr_in servAddr;
  ::bzero((char *)&servAddr, sizeof(servAddr));
  servAddr.sin_family = AF_INET;
  servAddr.sin_addr.s_addr = INADDR_ANY;
  // sin_port = 0 => auto assign
  servAddr.sin_port = 0;
  if (::bind(sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
    return {};
  socklen_t len = sizeof(servAddr);
  if (::getsockname(sock, (struct sockaddr *)&servAddr, &len) == -1)
    return {};
  if (close(sock) < 0)
    return {};
  return std::to_string(::ntohs(servAddr.sin_port));
}

cudaq::AutoLaunchRestServerProcess::AutoLaunchRestServerProcess() {
  cudaq::info("Auto launch REST server");
  const std::string serverExeName = "cudaq-qpud";
  auto serverApp = llvm::sys::findProgramByName(serverExeName.c_str());
  if (!serverApp)
    throw std::runtime_error(
        "Unable to find CUDA Quantum REST server to launch.");
  const auto port = getAvailablePort();
  if (!port.has_value())
    throw std::runtime_error("Unable to find a TCP/IP port on the local "
                             "machine for auto-launch a REST server.");
  llvm::StringRef argv[] = {serverApp.get(), "--port", port.value()};
  [[maybe_unused]] auto processInfo =
      llvm::sys::ExecuteNoWait(serverApp.get(), argv, std::nullopt);
  cudaq::info("Auto launch REST server at http://localhost:{} (PID {})",
              port.value(), processInfo.Pid);
  m_pid = processInfo.Pid;
  m_url = fmt::format("localhost:{}", port.value());
}

cudaq::AutoLaunchRestServerProcess::~AutoLaunchRestServerProcess() {
  cudaq::info("Shutting down REST server process {}", m_pid);
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
