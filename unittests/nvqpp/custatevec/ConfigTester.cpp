/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecConfig.h"
#include "CuStateVecDevice.h"
#include "CuStateVecGateEngine.h"

#include <gtest/gtest.h>

namespace {

class Environment {
public:
  Environment(const char *name, const char *value) : name_(name) {
    if (const char *old = std::getenv(name))
      old_ = old;
    setenv(name, value, 1);
  }
  ~Environment() {
    if (old_)
      setenv(name_, old_->c_str(), 1);
    else
      unsetenv(name_);
  }

private:
  const char *name_;
  std::optional<std::string> old_;
};
TEST(CuStateVecConfigTester, AppliesGpuFamilyFusionDefaults) {
  using cudaq::cusv::CuStateVecConfig;
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(8, 0, true).denseFusionQubits, 4);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(8, 0, false).denseFusionQubits,
            5);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(9, 0, true).denseFusionQubits, 5);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(9, 0, false).denseFusionQubits,
            6);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(10, 0, true).denseFusionQubits,
            5);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(10, 0, false).denseFusionQubits,
            4);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(10, 3, true).denseFusionQubits,
            5);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(10, 3, false).denseFusionQubits,
            1);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(12, 1, true).denseFusionQubits,
            5);
  EXPECT_EQ(CuStateVecConfig::fromEnvironment(12, 1, false).denseFusionQubits,
            3);
}

TEST(CuStateVecConfigTester, EnvironmentOverridesGpuFamilyFusionDefault) {
  Environment fusion("CUDAQ_FUSION_MAX_QUBITS", "2");
  const auto config =
      cudaq::cusv::CuStateVecConfig::fromEnvironment(9, 0, false);
  EXPECT_EQ(config.denseFusionQubits, 2);
}

TEST(CuStateVecConfigTester, ClampsCuStateVecExUpdaterLimits) {
  Environment fusion("CUDAQ_FUSION_MAX_QUBITS", "11");
  Environment diagonal("CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS", "21");
  Environment threads("CUDAQ_FUSION_NUM_HOST_THREADS", "33");
  const auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  EXPECT_EQ(config.denseFusionQubits, 10);
  EXPECT_EQ(config.diagonalFusionQubits, 20);
  EXPECT_EQ(config.hostThreads, 32);
}

} // namespace

TEST(CuStateVecDeviceTester, AccountsForReusableMemoryPoolBytes) {
  cudaq::cusv::DeviceMemoryCapacity capacity;
  capacity.cudaFreeBytes = 4;
  capacity.poolReservedBytes = 7;
  capacity.poolUsedBytes = 2;
  EXPECT_EQ(capacity.usableBytes(), 9);
}

TEST(CuStateVecDeviceTester, UsesSystemAvailableMemoryOnlyForSpark) {
  cudaq::cusv::DeviceMemoryCapacity capacity;
  capacity.cudaFreeBytes = 4;
  capacity.systemAvailableBytes = 16;
  capacity.computeCapabilityMajor = 9;
  capacity.computeCapabilityMinor = 0;
  EXPECT_EQ(capacity.usableBytes(), 4);

  capacity.computeCapabilityMajor = 12;
  capacity.computeCapabilityMinor = 1;
  EXPECT_EQ(capacity.usableBytes(), 16);
}

TEST(CuStateVecDeviceTester, MatchesLegacyMigrationWireCapacity) {
  using cudaq::cusv::migrationWireCapacity;
  constexpr std::size_t deviceBytes = 16;
  EXPECT_EQ(migrationWireCapacity(0, deviceBytes), 0);
  EXPECT_EQ(migrationWireCapacity(16, deviceBytes), 0);
  EXPECT_EQ(migrationWireCapacity(32, deviceBytes), 1);
  EXPECT_EQ(migrationWireCapacity(48, deviceBytes), 1);
  EXPECT_EQ(migrationWireCapacity(64, deviceBytes), 2);
  EXPECT_EQ(migrationWireCapacity(128, deviceBytes), 3);
  EXPECT_EQ(migrationWireCapacity(1024, deviceBytes), 3);
}

TEST(CuStateVecDeviceTester, MatchesLegacyTrajectoryBatchSizing) {
  using cudaq::cusv::trajectoryBatchSize;
  EXPECT_EQ(trajectoryBatchSize(48, 16, 100, std::nullopt), 2);
  EXPECT_EQ(trajectoryBatchSize(128, 16, 5, std::nullopt), 8);
  EXPECT_EQ(trajectoryBatchSize(16, 16, 100, 8), 8);
  EXPECT_EQ(trajectoryBatchSize(128, 16, 5, 3), 3);
  EXPECT_EQ(trajectoryBatchSize(128, 16, 2, 8), 2);
}
TEST(CuStateVecConfigTester, NonPositiveFusionDisablesUpdater) {
  for (const char *value : {"0", "-1"}) {
    Environment setting("CUDAQ_FUSION_MAX_QUBITS", value);
    const auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
    EXPECT_EQ(config.gateMode, cudaq::cusv::GateExecutionMode::Direct);
    EXPECT_NE(dynamic_cast<cudaq::cusv::DirectGateEngine<double> *>(
                  cudaq::cusv::createGateEngine<double>(config).get()),
              nullptr);
  }
}

TEST(CuStateVecConfigTester, FusionAliasPrecedenceMatchesUserSurface) {
  Environment legacy("CUDAQ_MGPU_FUSE", "7");
  Environment generic("CUDAQ_FUSION_MAX_QUBITS", "3");
  const auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  EXPECT_EQ(config.denseFusionQubits, 3);
}

TEST(CuStateVecConfigTester, ParsesTrajectoryControls) {
  Environment batch("CUDAQ_BATCH_SIZE", "8");
  Environment branches("CUDAQ_BATCHED_SIM_MAX_BRANCHES", "4");
  Environment shotByShot("CUDAQ_PTSBE_BATCH_SAMPLE_SHOT_BY_SHOT", "1");
  const auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  EXPECT_EQ(config.trajectoryBatchSize, 8);
  EXPECT_EQ(config.batchedMaxBranches, 4);
  EXPECT_TRUE(config.ptsbeBatchSampleShotByShot);
}
TEST(CuStateVecConfigTester, ParsesCompleteUserSurface) {
  Environment fusion("CUDAQ_FUSION_MAX_QUBITS", "6");
  Environment diagonal("CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS", "10");
  Environment legacyThreads("CUDAQ_MGPU_NUM_HOST_THREADS", "4");
  Environment threads("CUDAQ_FUSION_NUM_HOST_THREADS", "12");
  Environment gpuMemory("CUDAQ_MAX_GPU_MEMORY_GB", "16");
  Environment cpuMemory("CUDAQ_MAX_CPU_MEMORY_GB", "64");
  Environment migrationLevel("CUDAQ_HOST_DEVICE_MIGRATION_LEVEL", "2");
  Environment globalBits("CUDAQ_GLOBAL_INDEX_BITS", "1,2");
  Environment p2pBits("CUDAQ_MGPU_P2P_DEVICE_BITS", "1");
  Environment transferBits("CUDAQ_DATA_TRANSFER_BUFFER_BITS", "24");
  Environment trajectories("CUDAQ_OBSERVE_NUM_TRAJECTORIES", "17");
  Environment batchSize("CUDAQ_BATCH_SIZE", "8");
  Environment maxBranches("CUDAQ_BATCHED_SIM_MAX_BRANCHES", "5");
  Environment maxQubits("CUDAQ_BATCHED_SIM_MAX_QUBITS", "19");
  Environment minBatch("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE", "3");
  Environment rngThreshold("CUDAQ_GPU_RNG_THRESHOLD", "0");
  Environment mgpuThreshold("CUDAQ_MGPU_NQUBITS_THRESH", "9");
  Environment memPool("CUDAQ_ENABLE_MEMPOOL", "off");
  Environment fp32Emulation("CUDAQ_ALLOW_FP32_EMULATED", "false");
  Environment expPauli("CUDAQ_FORCE_EXP_PAULI_DECOMPOSE", "yes");
  Environment allocate("CUDAQ_FORCE_ALLOCATE_STATE", "on");
  Environment logTrajectories("CUDAQ_LOG_TRAJECTORY_SAMPLING", "true");
  Environment shotByShot("CUDAQ_PTSBE_BATCH_SAMPLE_SHOT_BY_SHOT", "1");
  Environment fabric("CUDAQ_GPU_FABRIC", "nvl");
  Environment mpiLibrary("CUDAQ_MGPU_LIB_MPI", "/tmp/libmpi-test.so");
  Environment communicator("CUDAQ_MGPU_COMM_PLUGIN_TYPE", "OpenMPI");

  auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  config.applyDistributedEnvironment();
  EXPECT_EQ(config.gateMode, cudaq::cusv::GateExecutionMode::Fused);
  EXPECT_EQ(config.denseFusionQubits, 6);
  EXPECT_EQ(config.diagonalFusionQubits, 10);
  EXPECT_EQ(config.hostThreads, 12);
  EXPECT_EQ(config.maxGpuMemoryGb, 16);
  EXPECT_EQ(config.maxCpuMemoryGb, 64);
  EXPECT_EQ(config.migrationLevel, 2);
  EXPECT_EQ(config.globalIndexBits, (std::vector<int32_t>{1, 2}));
  EXPECT_EQ(config.p2pDeviceBits, 1);
  EXPECT_EQ(config.dataTransferBufferBits, 24);
  EXPECT_EQ(config.observeTrajectories, 17);
  EXPECT_EQ(config.trajectoryBatchSize, 8);
  EXPECT_EQ(config.batchedMaxBranches, 5);
  EXPECT_EQ(config.batchedMaxQubits, 19);
  EXPECT_EQ(config.batchedMinBatchSize, 3);
  EXPECT_EQ(config.gpuRngThreshold, 0);
  EXPECT_EQ(config.mgpuQubitThreshold, 9);
  EXPECT_FALSE(config.enableMemPool);
  EXPECT_FALSE(config.allowFp32Emulation);
  EXPECT_TRUE(config.forceExpPauliDecomposition);
  EXPECT_TRUE(config.forceAllocateState);
  EXPECT_TRUE(config.logTrajectorySampling);
  EXPECT_TRUE(config.ptsbeBatchSampleShotByShot);
  EXPECT_EQ(config.gpuFabric, "NVL");
  EXPECT_EQ(config.mpiLibrary, "/tmp/libmpi-test.so");
  EXPECT_EQ(config.communicatorPlugin,
            cudaq::cusv::CommunicatorPlugin::OpenMPI);
}

TEST(CuStateVecConfigTester, ParsesOptionalMemorySentinels) {
  Environment gpuMemory("CUDAQ_MAX_GPU_MEMORY_GB", "NONE");
  Environment cpuMemory("CUDAQ_MAX_CPU_MEMORY_GB", "null");
  Environment batchSize("CUDAQ_BATCH_SIZE", "NULL");
  const auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  EXPECT_FALSE(config.maxGpuMemoryGb);
  EXPECT_FALSE(config.maxCpuMemoryGb);
  EXPECT_FALSE(config.trajectoryBatchSize);
}

TEST(CuStateVecConfigTester, DefersDistributedEnvironmentParsing) {
  Environment p2pBits("CUDAQ_MGPU_P2P_DEVICE_BITS", "invalid");
  Environment threshold("CUDAQ_MGPU_NQUBITS_THRESH", "invalid");
  Environment communicator("CUDAQ_MGPU_COMM_PLUGIN_TYPE", "invalid");

  EXPECT_NO_THROW(cudaq::cusv::CuStateVecConfig::fromEnvironment());
  auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  EXPECT_THROW(config.applyDistributedEnvironment(), std::exception);
}

TEST(CuStateVecConfigTester, ZeroDisablesDistributedP2PBits) {
  Environment p2pBits("CUDAQ_MGPU_P2P_DEVICE_BITS", "0");
  auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  EXPECT_NO_THROW(config.applyDistributedEnvironment());
  EXPECT_EQ(config.p2pDeviceBits, 0);
}

TEST(CuStateVecConfigTester, RejectsInvalidControls) {
  const auto rejects = [](const char *name, const char *value) {
    Environment setting(name, value);
    EXPECT_THROW(cudaq::cusv::CuStateVecConfig::fromEnvironment(),
                 std::exception)
        << name << '=' << value;
  };
  const auto rejectsDistributed = [](const char *name, const char *value) {
    Environment setting(name, value);
    auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
    EXPECT_THROW(config.applyDistributedEnvironment(), std::exception)
        << name << '=' << value;
  };

  rejects("CUDAQ_FUSION_DIAGONAL_GATE_MAX_QUBITS", "-2");
  rejects("CUDAQ_FUSION_NUM_HOST_THREADS", "0");
  rejects("CUDAQ_MAX_GPU_MEMORY_GB", "0");
  rejects("CUDAQ_MAX_CPU_MEMORY_GB", "-1");
  rejects("CUDAQ_OBSERVE_NUM_TRAJECTORIES", "0");
  rejects("CUDAQ_BATCH_SIZE", "0");
  rejects("CUDAQ_BATCHED_SIM_MAX_BRANCHES", "0");
  rejects("CUDAQ_BATCHED_SIM_MAX_QUBITS", "0");
  rejects("CUDAQ_BATCHED_SIM_MIN_BATCH_SIZE", "0");
  rejects("CUDAQ_GPU_RNG_THRESHOLD", "-1");
  rejectsDistributed("CUDAQ_GLOBAL_INDEX_BITS", "1,0");
  rejectsDistributed("CUDAQ_DATA_TRANSFER_BUFFER_BITS", "23");
  rejectsDistributed("CUDAQ_MGPU_NQUBITS_THRESH", "0");
  rejectsDistributed("CUDAQ_MGPU_P2P_DEVICE_BITS", "-1");
  rejectsDistributed("CUDAQ_MGPU_COMM_PLUGIN_TYPE", "unknown");
}
