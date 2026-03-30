/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "MPI/QpuProcessGroup.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "common/NoiseModel.h"
#include "common/RuntimeTarget.h"
#include "cudaq/Support/TargetConfigYaml.h"
#include "cudaq/platform/qpu.h"
#include "cudaq/platform/quantum_platform.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/runtime/logger/logger.h"
#include "helpers/MQPUUtils.h"
#include "utils/cudaq_utils.h"
#include "llvm/Support/Base64.h"
#include <filesystem>
#include <fstream>

LLVM_INSTANTIATE_REGISTRY(cudaq::QPU::RegistryType)

namespace {

class DefaultQPU : public cudaq::QPU {
public:
  DefaultQPU() = default;
  virtual ~DefaultQPU() = default;

  void enqueue(cudaq::QuantumTask &task) override {
    execution_queue->enqueue(task);
  }

  cudaq::KernelThunkResultType
  launchKernel(const std::string &name, cudaq::KernelThunkType kernelFunc,
               void *args, std::uint64_t argsSize, std::uint64_t resultOffset,
               const std::vector<void *> &rawArgs) override {
    ScopedTraceWithContext(cudaq::TIMING_LAUNCH, "QPU::launchKernel");
    return kernelFunc(args, /*isRemote=*/false);
  }

  void
  configureExecutionContext(cudaq::ExecutionContext &context) const override {
    ScopedTraceWithContext("DefaultPlatform::prepareExecutionContext",
                           context.name);
    if (noiseModel)
      context.noiseModel = noiseModel;

    context.executionManager = cudaq::getDefaultExecutionManager();
    context.executionManager->configureExecutionContext(context);
  }

  void beginExecution() override {
    cudaq::getExecutionContext()->executionManager->beginExecution();
  }

  void endExecution() override {
    cudaq::getExecutionContext()->executionManager->endExecution();
  }

  void
  finalizeExecutionContext(cudaq::ExecutionContext &context) const override {
    ScopedTraceWithContext(
        context.name == "observe" ? cudaq::TIMING_OBSERVE : 0,
        "DefaultPlatform::finalizeExecutionContext", context.name);
    handleObservation(context);

    cudaq::getExecutionContext()->executionManager->finalizeExecutionContext(
        context);
  }
};

class MultiQPUQuantumPlatform : public cudaq::quantum_platform {
  std::vector<cudaq::details::QpuProcessGroup> m_qpuProcessGroups;
  bool m_broadcastKernelResults = true;
public:
  ~MultiQPUQuantumPlatform() {
    // Make sure that we clean up the client QPUs first before cleaning up the
    // remote servers.
    platformQPUs.clear();
    m_qpuProcessGroups.clear();
  }

  MultiQPUQuantumPlatform() {
    // Environment variable CUDAQ_MQPU_BROADCAST_RESULTS controls whether QPUs
    // on different ranks should broadcast their kernel results to each other.
    // This might be useful for users (all ranks have the same results), but can
    // be turned off for platforms where this is not necessary to avoid the
    // overhead of broadcasting.
    const char *broadcastEnvVal = std::getenv("CUDAQ_MQPU_BROADCAST_RESULTS");
    if (broadcastEnvVal != nullptr) {
      const std::string valStr = broadcastEnvVal;
      if (valStr == "0" || valStr == "false" || valStr == "False" ||
          valStr == "FALSE" || valStr == "off" || valStr == "OFF") {
        m_broadcastKernelResults = false;
      }
    }

    //   if (cudaq::registry::isRegistered<cudaq::QPU>("GPUEmulatedQPU")) {
    //     int nDevices = cudaq::getCudaDeviceCount();
    //     // Skipped if CUDA-Q was built with CUDA but no devices present at
    //     // runtime.
    //     if (nDevices > 0) {
    //       const char *envVal = std::getenv("CUDAQ_MQPU_NGPUS");
    //       if (envVal != nullptr) {
    //         int specifiedNDevices = 0;
    //         try {
    //           specifiedNDevices = std::stoi(envVal);
    //         } catch (...) {
    //           throw std::runtime_error("Invalid CUDAQ_MQPU_NGPUS environment
    //           "
    //                                    "variable, must be integer.");
    //         }

    //         if (specifiedNDevices < nDevices)
    //           nDevices = specifiedNDevices;
    //       }

    //       if (nDevices == 0)
    //         throw std::runtime_error(
    //             "No GPUs available to instantiate platform.");

    //       // Add a QPU for each GPU.
    //       for (int i = 0; i < nDevices; i++) {
    //         platformQPUs.emplace_back(
    //             cudaq::registry::get<cudaq::QPU>("GPUEmulatedQPU"));
    //         platformQPUs.back()->setId(i);
    //       }
    //     }
    //   }
  }

  bool supports_task_distribution() const override { return true; }

  std::future<cudaq::sample_result>
  enqueueAsyncTask(const std::size_t qpu_id,
                   cudaq::KernelExecutionTask &t) override {
    if (!m_qpuProcessGroups.empty()) {
      const int localRank = cudaq::details::QpuProcessGroup::getGlobalMpiRank();
      const auto& qpuProcessGroup = m_qpuProcessGroups[qpu_id];
      if (!qpuProcessGroup.contains(localRank)) {
        if (!m_broadcastKernelResults) {
          // Add a no-op task to keep the promise alive for this rank, even
          // though it won't do any work since it's not assigned to this QPU.
          cudaq::KernelExecutionTask emptyTask =
              cudaq::detail::make_copyable_function([]() mutable {
                // No-op task
                return cudaq::sample_result();
              });
          return quantum_platform::enqueueAsyncTask(qpu_id, emptyTask);
        } else {
          cudaq::KernelExecutionTask waitForBroadcastTask =
              cudaq::detail::make_copyable_function([&qpuProcessGroup]() mutable {
                cudaq::sample_result result;
                // Wait for the result to be broadcasted from the ranks that execute the task.
                cudaq::details::QpuProcessGroup::broadcast(result, qpuProcessGroup);
                return result;
              });
          return quantum_platform::enqueueAsyncTask(qpu_id, waitForBroadcastTask);
        }
      } else {
        // Execute then broadcast results among ranks in the same QPU process group.
        auto future = quantum_platform::enqueueAsyncTask(qpu_id, t);
        if (m_broadcastKernelResults) {
          // As std::future doesn't support continuations (i.e., `then()`), we
          // need to wrap the future in another one to broadcast the results
          // after the task is done.
          cudaq::KernelExecutionTask wrappedTask =
              cudaq::detail::make_copyable_function(
                  [&qpuProcessGroup, future = std::move(future)]() mutable {
                    auto result = future.get(); // Wait for the original task to
                                                // complete and get the result.
                    // Broadcast the result to other ranks in the same QPU
                    // process group.
                    cudaq::details::QpuProcessGroup::broadcast(result, qpuProcessGroup);
                    return result;
                  });
          return quantum_platform::enqueueAsyncTask(qpu_id, wrappedTask);
        } else {
          return future;
        }
      }
    }    
    return quantum_platform::enqueueAsyncTask(qpu_id, t);
  }

private:
  static std::string getQpuType(const std::string &description) {
    // Target name is the first one in the target config string
    // or the whole string if this is the only config.
    const auto targetName = description.find(";") != std::string::npos
                                ? cudaq::split(description, ';').front()
                                : description;
    std::filesystem::path cudaqLibPath{cudaq::getCUDAQLibraryPath()};
    auto platformPath = cudaqLibPath.parent_path().parent_path() / "targets";
    std::string targetConfigFileName = targetName + std::string(".yml");
    auto configFilePath = platformPath / targetConfigFileName;
    CUDAQ_INFO("Config file path for target {} = {}", targetName,
               configFilePath.string());
    // Don't try to load something that doesn't exist.
    if (!std::filesystem::exists(configFilePath))
      return "";
    std::ifstream configFile(configFilePath.string());
    std::string configContents((std::istreambuf_iterator<char>(configFile)),
                               std::istreambuf_iterator<char>());
    cudaq::config::TargetConfig config;
    llvm::yaml::Input Input(configContents.c_str());
    Input >> config;

    if (config.BackendConfig.has_value() &&
        !config.BackendConfig->PlatformQpu.empty()) {
      return config.BackendConfig->PlatformQpu;
    }

    return "";
  }
  static std::string getOption(const std::string &str,
                               const std::string &prefix) {
    // Return the first key-value configuration option found in the format:
    // "<prefix>;<option>".
    // Note: This expects an exact match of the prefix and the option value is
    // the next one.
    auto splitParts = cudaq::split(str, ';');
    if (splitParts.empty())
      return "";
    for (std::size_t i = 0; i < splitParts.size() - 1; ++i) {
      if (splitParts[i] == prefix) {
        CUDAQ_DBG(
            "Retrieved option '{}' for the key '{}' from input string '{}'",
            splitParts[i + 1], prefix, str);
        if (splitParts[i + 1].starts_with("base64_")) {
          splitParts[i + 1].erase(0, 7); // erase "base64_"
          std::vector<char> decoded_vec;
          if (auto err = llvm::decodeBase64(splitParts[i + 1], decoded_vec))
            throw std::runtime_error("DecodeBase64 error");
          std::string decodedStr(decoded_vec.data(), decoded_vec.size());
          CUDAQ_INFO("Decoded {} parameter from '{}' to '{}'", splitParts[i],
                     splitParts[i + 1], decodedStr);
          return decodedStr;
        }
        return splitParts[i + 1];
      }
    }
    return "";
  }

  static std::string formatUrl(const std::string &url) {
    auto formatted = url;
    // Default to http:// if none provided.
    if (!formatted.starts_with("http"))
      formatted = std::string("http://") + formatted;
    if (!formatted.empty() && formatted.back() != '/')
      formatted += '/';
    return formatted;
  }

  static cudaqDistributedCommunicator_t *getMpiCommWrapper() {
    auto mpiPlugin = cudaq::mpi::getMpiPlugin();
    if (!mpiPlugin)
      throw std::runtime_error("Failed to retrieve MPI plugin");
    cudaqDistributedCommunicator_t *comm = mpiPlugin->getComm();
    if (!comm)
      throw std::runtime_error("Invalid MPI distributed plugin encountered");
    return comm;
  }

  void setTargetBackend(const std::string &description) override {
    std::cout << "Configuring MultiQPU platform with target description: "
              << description << std::endl;
    const auto qpuSubType = getQpuType(description);
    if (!qpuSubType.empty()) {
      if (!cudaq::registry::isRegistered<cudaq::QPU>(qpuSubType))
        throw std::runtime_error(
            fmt::format("Unable to retrieve {} QPU implementation. Please "
                        "check your installation.",
                        qpuSubType));
      if (qpuSubType == "orca") {
        auto urls = cudaq::split(getOption(description, "url"), ',');
        platformQPUs.clear();
        for (std::size_t qId = 0; qId < urls.size(); ++qId) {
          // Populate the information and add the QPUs
          platformQPUs.emplace_back(cudaq::registry::get<cudaq::QPU>("orca"));
          platformQPUs.back()->setId(qId);
          const std::string configStr =
              fmt::format("orca;url;{}", formatUrl(urls[qId]));
          platformQPUs.back()->setTargetBackend(configStr);
        }
        return;
      } else {
        throw std::runtime_error(
            fmt::format("Unsupported platform QPU sub-type '{}' specified in "
                        "target config. Currently only 'orca' is supported.",
                        qpuSubType));
      }
    }
    platformQPUs.clear();
    // No QPU sub-type, i.e., simulators
    // Check for MPI first, i.e., if we're running with mpirun/mpiexec.
    const auto numMpiRanks = cudaq::details::QpuProcessGroup::getNumMpiRanks();
    std::cout << "Detected " << numMpiRanks
              << " MPI ranks in the environment. Configuring platform QPUs for "
                 "MPI-based distributed execution."
              << std::endl;
    if (numMpiRanks > 1) {
      CUDAQ_INFO("MPI environment detected with {} ranks, configuring platform "
                 "QPUs for distributed execution.",
                 numMpiRanks);
      // Default to using 1 QPU per MPI rank, but allow user to specify
      // otherwise.
      auto numQpus = numMpiRanks;
      // Determine the number of QPUs based on user configurations
      const auto numQpusStr = getOption(description, "nqpus");
      if (!numQpusStr.empty()) {
        try {
          int numQpus = std::stoi(numQpusStr);
          if (numQpus <= 0)
            throw std::runtime_error(
                "Invalid number of QPUs specified in target config, must be "
                "positive integer.");
          // Number of QPUs cannot exceed number of MPI ranks
          if (numQpus > numMpiRanks)
            throw std::runtime_error(
                "Number of QPUs specified in target config cannot exceed "
                "number of MPI ranks.");
          // If we have fewer QPUs than MPI ranks, we will assign multiple
          // ranks to each QPU. Required that this is evenly divisible.
          if (numMpiRanks % numQpus != 0)
            throw std::runtime_error("Number of MPI ranks must be evenly "
                                     "divisible by the number of QPUs.");
        } catch (const std::exception &e) {
          throw std::runtime_error(
              fmt::format("Invalid number of QPUs specified in target config: "
                          "{}. Error: {}",
                          numQpusStr, e.what()));
        }

        CUDAQ_INFO("Configuring platform with {} QPUs on MPI platform with "
                   "{} ranks.",
                   numQpus, numMpiRanks);
      }

      platformQPUs.clear();
      // Split the comunicator evenly across all QPUs.
      // For example, if we have 4 MPI ranks and 2 QPUs, ranks 0 and 1 will be
      // assigned to QPU 0 and ranks 2 and 3 will be assigned to QPU 1.
      std::vector<std::vector<int>> qpuRankAssignments(numQpus);
      std::unordered_map<int, int> localRankToQpuId;
      for (int rank = 0; rank < numMpiRanks; ++rank) {
        qpuRankAssignments[rank % numQpus].push_back(rank);
        localRankToQpuId[rank] = rank % numQpus;
      }
      for (std::size_t qId = 0; qId < qpuRankAssignments.size(); ++qId) {
        m_qpuProcessGroups.emplace_back(qpuRankAssignments[qId]);
        platformQPUs.emplace_back(std::make_unique<DefaultQPU>());
        platformQPUs.back()->setId(qId);
        platformQPUs.back()->setTargetBackend(description);
      }

      return;
    }
  }
};
} // namespace

CUDAQ_REGISTER_PLATFORM(MultiQPUQuantumPlatform, mqpu)
