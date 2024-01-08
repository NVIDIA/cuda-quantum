/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "RemoteKernelExecutor.h"
#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/RuntimeMLIR.h"
#include "cudaq.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "nvqir/CircuitSimulator.h"
#include "llvm/Support/Program.h"
#include <arpa/inet.h>
#include <execinfo.h>
#include <signal.h>
#include <sys/socket.h>

namespace {
using namespace mlir;

// Remote QPU: delegating the execution to a remotely-hosted server, which can
// reinstate the execution context and JIT-invoke the kernel.
class RemoteSimulatorQPU : public cudaq::QPU {
private:
  std::string m_simName;
  std::unordered_map<std::thread::id, cudaq::ExecutionContext *> m_contexts;
  std::mutex m_contextMutex;
  MLIRContext &m_mlirContext;
  std::unique_ptr<cudaq::RemoteRuntimeClient> m_client;

public:
  RemoteSimulatorQPU(MLIRContext &mlirContext, const std::string &url,
                     std::size_t id, const std::string &simName)
      : QPU(id), m_simName(simName), m_mlirContext(mlirContext),
        m_client(cudaq::registry::get<cudaq::RemoteRuntimeClient>("rest")) {
    cudaq::info("Create a remote QPU: id={}; url={}; simulator={}", qpu_id, url,
                m_simName);
    m_client->setConfig({{"url", url}});
  }

  std::thread::id getExecutionThreadId() const {
    return execution_queue->getExecutionThreadId();
  }

  void enqueue(cudaq::QuantumTask &task) override {
    cudaq::info("RemoteSimulatorQPU: Enqueue Task on QPU {}", qpu_id);
    execution_queue->enqueue(task);
  }

  void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset) override {
    cudaq::info("RemoteSimulatorQPU: Launch kernel named '{}' remote QPU {} "
                "(simulator = {})",
                name, qpu_id, m_simName);

    cudaq::ExecutionContext *executionContext = [&]() {
      std::scoped_lock<std::mutex> lock(m_contextMutex);
      const auto iter = m_contexts.find(std::this_thread::get_id());
      if (iter == m_contexts.end())
        throw std::runtime_error("Internal error: invalid execution context");
      return iter->second;
    }();
    std::string errorMsg;
    const bool requestOkay =
        m_client->sendRequest(m_mlirContext, *executionContext, m_simName, name,
                              kernelFunc, args, voidStarSize, &errorMsg);
    if (!requestOkay)
      throw std::runtime_error("Failed to launch kernel. Error: " + errorMsg);
  }

  void setExecutionContext(cudaq::ExecutionContext *context) override {
    cudaq::info("RemoteSimulatorQPU::setExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts[std::this_thread::get_id()] = context;
  }

  void resetExecutionContext() override {
    cudaq::info("RemoteSimulatorQPU::resetExecutionContext QPU {}", qpu_id);
    std::scoped_lock<std::mutex> lock(m_contextMutex);
    m_contexts.erase(std::this_thread::get_id());
  }
};

// Util to query an available TCP/IP port for auto-launching a server instance.
std::optional<std::string> getAvailablePort() {
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

// Multi-QPU platform for remotely-hosted simulator QPUs
class RemoteSimulatorQuantumPlatform : public cudaq::quantum_platform {
  std::unique_ptr<MLIRContext> m_mlirContext;
  std::vector<llvm::sys::ProcessInfo> m_serverProcesses;

public:
  RemoteSimulatorQuantumPlatform() : m_mlirContext(cudaq::initializeMLIR()) {
    platformNumQPUs = 0;
  }

  ~RemoteSimulatorQuantumPlatform() {
    for (auto &process : m_serverProcesses) {
      cudaq::info("Shutting down REST server process {}", process.Pid);
      ::kill(process.Pid, SIGKILL);
    }
  }
  bool supports_task_distribution() const override { return true; }
  void setTargetBackend(const std::string &description) override {
    const auto getOpt = [](const std::string &str,
                           const std::string &prefix) -> std::string {
      const auto prefixPos = str.find(prefix);
      if (prefixPos == std::string::npos)
        return "";
      const auto endPos = str.find_first_of(";", prefixPos);
      return (endPos == std::string::npos)
                 ? str.substr(prefixPos + prefix.size() + 1)
                 : cudaq::split(str.substr(prefixPos + prefix.size() + 1),
                                ';')[0];
    };

    auto urls = cudaq::split(getOpt(description, "url"), ',');
    auto sims = cudaq::split(getOpt(description, "backend"), ',');
    const bool autoLaunch =
        description.find("auto_launch") != std::string::npos;

    const auto formatUrl = [](const std::string &url) -> std::string {
      auto formatted = url;
      if (formatted.rfind("http", 0) != 0)
        formatted = std::string("http://") + formatted;
      if (formatted.back() != '/')
        formatted += '/';
      return formatted;
    };

    if (autoLaunch) {
      urls.clear();
      if (sims.empty())
        sims.emplace_back("qpp");
      const int numInstances = std::stoi(getOpt(description, "auto_launch"));
      cudaq::info("Auto launch {} REST servers", numInstances);
      const std::string serverExeName = "cudaq_rest_server";
      auto serverApp = llvm::sys::findProgramByName(serverExeName.c_str());
      if (!serverApp)
        throw std::runtime_error(
            "Unable to find CUDA Quantum REST server to launch.");

      for (int i = 0; i < numInstances; ++i) {
        const auto port = getAvailablePort();
        if (!port.has_value())
          throw std::runtime_error("Unable to find a TCP/IP port on the local "
                                   "machine for auto-launch a REST server.");
        urls.emplace_back(std::string("localhost:") + port.value());
        llvm::StringRef argv[] = {serverApp.get(), "--port", port.value()};
        [[maybe_unused]] auto processInfo =
            llvm::sys::ExecuteNoWait(serverApp.get(), argv, std::nullopt);
        cudaq::info("Auto launch REST server at http://localhost:{} (PID {})",
                    port.value(), processInfo.Pid);
        m_serverProcesses.emplace_back(processInfo);
      }
      // Allows some time for the servers to start
      std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }

    // List of simulator names must either be one or the same length as the URL
    // list. If one simulator name is provided, assuming that all the URL should
    // be using the same simulator.
    if (sims.size() > 1 && sims.size() != urls.size())
      throw std::runtime_error(
          fmt::format("Invalid number of remote backend simulators provided: "
                      "receiving {}, expecting {}.",
                      sims.size(), urls.size()));
    for (std::size_t qId = 0; qId < urls.size(); ++qId) {
      const auto simName = sims.size() == 1 ? sims.front() : sims[qId];
      // Populate the information and add the QPUs
      auto qpu = std::make_unique<RemoteSimulatorQPU>(
          *m_mlirContext, formatUrl(urls[qId]), qId, simName);
      threadToQpuId[std::hash<std::thread::id>{}(qpu->getExecutionThreadId())] =
          qId;
      platformQPUs.emplace_back(std::move(qpu));
    }

    platformNumQPUs = platformQPUs.size();
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(RemoteSimulatorQuantumPlatform, remote_sim)
