/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecConfig.h"
#include "cudaq/runtime/logger/logger.h"

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {

std::optional<std::string_view> getEnvironment(std::string_view name) {
  if (const char *const value = std::getenv(std::string(name).c_str()))
    return value;
  return std::nullopt;
}

std::string upper(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::toupper(c); });
  return result;
}

std::string lower(std::string_view value) {
  std::string result(value);
  std::transform(result.begin(), result.end(), result.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return result;
}

int64_t parseInteger(std::string_view value, std::string_view name) {
  int64_t result = 0;
  const auto *const begin = value.data();
  const auto *const end = begin + value.size();
  const auto [position, error] = std::from_chars(begin, end, result);
  if (error != std::errc{} || position != end)
    throw std::invalid_argument("Invalid " + std::string(name) + " value: '" +
                                std::string(value) + "'.");
  return result;
}

int32_t parseInt32(std::string_view value, std::string_view name) {
  const int64_t result = parseInteger(value, name);
  if (result < std::numeric_limits<int32_t>::min() ||
      result > std::numeric_limits<int32_t>::max())
    throw std::out_of_range(std::string(name) + " is outside int32 range.");
  return static_cast<int32_t>(result);
}

int32_t parsePositive(std::string_view value, std::string_view name) {
  const int32_t result = parseInt32(value, name);
  if (result <= 0)
    throw std::invalid_argument(std::string(name) + " must be positive.");
  return result;
}

int32_t parseNonNegative(std::string_view value, std::string_view name) {
  const int32_t result = parseInt32(value, name);
  if (result < 0)
    throw std::invalid_argument(std::string(name) + " must be non-negative.");
  return result;
}

int32_t clampMaximum(int32_t value, int32_t maximum, std::string_view name) {
  if (value <= maximum)
    return value;
  CUDAQ_WARN("{}={} exceeds cuStateVecEx maximum {}; clamping to {}.", name,
             value, maximum, maximum);
  return maximum;
}

std::size_t parsePositiveSize(std::string_view value, std::string_view name) {
  return static_cast<std::size_t>(parsePositive(value, name));
}

std::optional<int32_t> parseOptionalNonNegative(std::string_view value,
                                                std::string_view name) {
  const std::string normalized = upper(value);
  if (normalized == "NONE" || normalized == "NULL")
    return std::nullopt;

  const int32_t result = parseInt32(value, name);
  if (result < 0)
    throw std::invalid_argument(std::string(name) +
                                " must be non-negative, NONE, or NULL.");
  return result;
}

bool parseBool(std::string_view name, bool defaultValue) {
  const auto value = getEnvironment(name);
  if (!value)
    return defaultValue;
  const std::string normalized = lower(*value);
  return normalized == "1" || normalized == "on" || normalized == "true" ||
         (!normalized.empty() && normalized.front() == 'y');
}

std::vector<int32_t> parsePositiveList(std::string_view value,
                                       std::string_view name) {
  std::vector<int32_t> result;
  while (true) {
    const std::size_t comma = value.find(',');
    const std::string_view item = value.substr(0, comma);
    result.push_back(parsePositive(item, name));
    if (comma == std::string_view::npos)
      return result;
    value.remove_prefix(comma + 1);
  }
}

cudaq::cusv::CommunicatorPlugin
parseCommunicatorPlugin(std::string_view value) {
  const std::string normalized = lower(value);
  if (normalized == "custateveccommpluginmpiauto" || normalized == "auto")
    return cudaq::cusv::CommunicatorPlugin::Auto;
  if (normalized == "custateveccommpluginexternal" || normalized == "external")
    return cudaq::cusv::CommunicatorPlugin::External;
  if (normalized == "custateveccommpluginself" || normalized == "self")
    return cudaq::cusv::CommunicatorPlugin::Self;
  if (normalized == "custateveccommpluginopenmpi" || normalized == "openmpi")
    return cudaq::cusv::CommunicatorPlugin::OpenMPI;
  if (normalized == "custateveccommpluginmpich" || normalized == "mpich")
    return cudaq::cusv::CommunicatorPlugin::MPICH;
  throw std::invalid_argument(
      "Invalid CUDAQ_MGPU_COMM_PLUGIN_TYPE environment variable.");
}

} // namespace

namespace cudaq::cusv {

CuStateVecConfig CuStateVecConfig::fromEnvironment() {
  CuStateVecConfig config;

  if (const auto value = getEnvironment("CUDAQ_MGPU_FUSE"))
    config.denseFusionQubits = clampMaximum(
        parseInt32(*value, "CUDAQ_MGPU_FUSE"), 10, "CUDAQ_MGPU_FUSE");
  // CUDAQ_FUSION_MAX_QUBITS is the generic alias and takes precedence over the
  // historical multi-GPU name.
  if (const auto value = getEnvironment("CUDAQ_FUSION_MAX_QUBITS"))
    config.denseFusionQubits =
        clampMaximum(parseInt32(*value, "CUDAQ_FUSION_MAX_QUBITS"), 10,
                     "CUDAQ_FUSION_MAX_QUBITS");
  config.gateMode = config.denseFusionQubits > 0 ? GateExecutionMode::Fused
                                                 : GateExecutionMode::Direct;

  if (const auto value =
          getEnvironment("CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS")) {
    config.diagonalFusionQubits =
        parseInt32(*value, "CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS");
    if (config.diagonalFusionQubits < -1)
      throw std::invalid_argument(
          "CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS must be at least -1.");
    config.diagonalFusionQubits =
        clampMaximum(config.diagonalFusionQubits, 20,
                     "CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS");
  }

  if (const auto value = getEnvironment("CUDAQ_MGPU_NUM_HOST_THREADS"))
    config.hostThreads =
        clampMaximum(parsePositive(*value, "CUDAQ_MGPU_NUM_HOST_THREADS"), 32,
                     "CUDAQ_MGPU_NUM_HOST_THREADS");
  if (const auto value = getEnvironment("CUDAQ_FUSION_NUM_HOST_THREADS"))
    config.hostThreads =
        clampMaximum(parsePositive(*value, "CUDAQ_FUSION_NUM_HOST_THREADS"), 32,
                     "CUDAQ_FUSION_NUM_HOST_THREADS");

  if (const auto value = getEnvironment("CUDAQ_MAX_GPU_MEMORY_GB")) {
    config.maxGpuMemoryGb =
        parseOptionalNonNegative(*value, "CUDAQ_MAX_GPU_MEMORY_GB");
    if (config.maxGpuMemoryGb == 0)
      throw std::invalid_argument("Setting GPU memory to zero is not allowed.");
  }
  if (const auto value = getEnvironment("CUDAQ_MAX_CPU_MEMORY_GB"))
    config.maxCpuMemoryGb =
        parseOptionalNonNegative(*value, "CUDAQ_MAX_CPU_MEMORY_GB");

  if (const auto value = getEnvironment("CUDAQ_OBSERVE_NUM_TRAJECTORIES"))
    config.observeTrajectories =
        parsePositiveSize(*value, "CUDAQ_OBSERVE_NUM_TRAJECTORIES");
  if (const auto value = getEnvironment("CUDAQ_BATCH_SIZE")) {
    config.trajectoryBatchSize =
        parseOptionalNonNegative(*value, "CUDAQ_BATCH_SIZE");
    if (config.trajectoryBatchSize == 0)
      throw std::invalid_argument(
          "CUDAQ_BATCH_SIZE must be positive, NONE, or NULL.");
  }
  if (const auto value = getEnvironment("CUDAQ_BATCHED_SIM_MAX_BRANCHES"))
    config.batchedMaxBranches =
        parsePositiveSize(*value, "CUDAQ_BATCHED_SIM_MAX_BRANCHES");
  if (const auto value = getEnvironment("CUDAQ_BATCHED_SIM_MAX_QUBITS"))
    config.batchedMaxQubits =
        parsePositiveSize(*value, "CUDAQ_BATCHED_SIM_MAX_QUBITS");
  if (const auto value = getEnvironment("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE"))
    config.batchedMinBatchSize =
        parsePositiveSize(*value, "CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE");
  if (const auto value = getEnvironment("CUDAQ_GPU_RNG_THRESHOLD")) {
    const int64_t threshold = parseInteger(*value, "CUDAQ_GPU_RNG_THRESHOLD");
    if (threshold < 0)
      throw std::invalid_argument(
          "CUDAQ_GPU_RNG_THRESHOLD must be non-negative.");
    config.gpuRngThreshold = static_cast<std::size_t>(threshold);
  }
  config.enableMemPool =
      parseBool("CUDAQ_ENABLE_MEMPOOL", config.enableMemPool);
  config.allowFp32Emulation =
      parseBool("CUDAQ_ALLOW_FP32_EMULATED", config.allowFp32Emulation);
  config.forceExpPauliDecomposition = parseBool(
      "CUDAQ_FORCE_EXP_PAULI_DECOMPOSE", config.forceExpPauliDecomposition);
  config.forceAllocateState =
      parseBool("CUDAQ_FORCE_ALLOCATE_STATE", config.forceAllocateState);
  config.logTrajectorySampling =
      parseBool("CUDAQ_LOG_TRAJECTORY_SAMPLING", config.logTrajectorySampling);
  config.ptsbeBatchSampleShotByShot =
      parseBool("CUDAQ_PTSBE_BATCH_SAMPLE_SHOT_BY_SHOT",
                config.ptsbeBatchSampleShotByShot);

  return config;
}

void CuStateVecConfig::applyDistributedEnvironment() {
  if (const auto value = getEnvironment("CUDAQ_HOST_DEVICE_MIGRATION_LEVEL"))
    migrationLevel =
        parseOptionalNonNegative(*value, "CUDAQ_HOST_DEVICE_MIGRATION_LEVEL");
  if (const auto value = getEnvironment("CUDAQ_GLOBAL_INDEX_BITS"))
    globalIndexBits = parsePositiveList(*value, "CUDAQ_GLOBAL_INDEX_BITS");
  if (const auto value = getEnvironment("CUDAQ_MGPU_P2P_DEVICE_BITS"))
    p2pDeviceBits = parseNonNegative(*value, "CUDAQ_MGPU_P2P_DEVICE_BITS");
  if (const auto value = getEnvironment("CUDAQ_DATA_TRANSFER_BUFFER_BITS")) {
    dataTransferBufferBits =
        parsePositive(*value, "CUDAQ_DATA_TRANSFER_BUFFER_BITS");
    if (dataTransferBufferBits < 24)
      throw std::invalid_argument(
          "CUDAQ_DATA_TRANSFER_BUFFER_BITS must be at least 24.");
  }
  if (const auto value = getEnvironment("CUDAQ_MGPU_NQUBITS_THRESH"))
    mgpuQubitThreshold = parsePositive(*value, "CUDAQ_MGPU_NQUBITS_THRESH");
  if (const auto value = getEnvironment("CUDAQ_GPU_FABRIC"))
    gpuFabric = upper(*value);
  if (const auto value = getEnvironment("CUDAQ_MGPU_LIB_MPI"))
    mpiLibrary = *value;
  if (const auto value = getEnvironment("CUDAQ_MGPU_COMM_PLUGIN_TYPE"))
    communicatorPlugin = parseCommunicatorPlugin(*value);
}

CuStateVecConfig
CuStateVecConfig::fromEnvironment(int32_t computeCapabilityMajor,
                                  int32_t computeCapabilityMinor, bool isFp32) {
  auto config = fromEnvironment();
  if (getEnvironment("CUDAQ_MGPU_FUSE") ||
      getEnvironment("CUDAQ_FUSION_MAX_QUBITS"))
    return config;

  int32_t fusionQubits = 0;
  if (computeCapabilityMinor == 0) {
    if (computeCapabilityMajor == 8)
      fusionQubits = isFp32 ? 4 : 5;
    else if (computeCapabilityMajor == 9)
      fusionQubits = isFp32 ? 5 : 6;
    else if (computeCapabilityMajor == 10)
      fusionQubits = isFp32 ? 5 : 4;
  } else if (computeCapabilityMajor == 10 && computeCapabilityMinor == 3) {
    // B300 has limited FP64 throughput, so a smaller FP64 fusion size performs
    // better than the generic Blackwell setting.
    fusionQubits = isFp32 ? 5 : 1;
  } else if (computeCapabilityMajor == 12 && computeCapabilityMinor == 1) {
    // GB10 uses its own empirically selected fusion sizes.
    fusionQubits = isFp32 ? 5 : 3;
  }
  if (fusionQubits > 0)
    config.denseFusionQubits = fusionQubits;
  return config;
}

} // namespace cudaq::cusv
