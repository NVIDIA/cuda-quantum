/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace cudaq::cusv {

/// Selects direct gate application or the `cuStateVecEx` fused updater.
enum class GateExecutionMode { Fused, Direct };

/// Selects the communicator provider used by the multi-process simulator.
enum class CommunicatorPlugin { Auto, External, Self, OpenMPI, MPICH };

/// @brief Configuration shared by all `cuStateVecEx` simulator components.
///
/// Defaults preserve the NVIDIA target behavior. `fromEnvironment()` applies
/// the supported CUDA-Q environment-variable overrides and validates them.
struct CuStateVecConfig {
  /// A non-positive fusion size selects direct gate-by-gate execution.
  GateExecutionMode gateMode = GateExecutionMode::Fused;

  /// Maximum number of qubits in a fused dense or diagonal operation. A
  /// diagonal value of -1 lets cuStateVecEx select a performant default.
  int32_t denseFusionQubits = 4;
  int32_t diagonalFusionQubits = -1;

  /// Number of host threads available to the fused updater.
  int32_t hostThreads = 8;

  /// GPU and host-memory limits used to size the state vector. An unset GPU
  /// limit uses available device memory. A zero host limit disables migration,
  /// while an unset host limit uses all available host memory.
  std::optional<int32_t> maxGpuMemoryGb;
  std::optional<int32_t> maxCpuMemoryGb = 0;

  /// Insertion point for the host-migration layer among global-index layers.
  std::optional<int32_t> migrationLevel;

  /// Sizes of the network communication layers, ordered from the local state
  /// outward. Their sum must equal log2 of the communicator size.
  std::vector<int32_t> globalIndexBits;

  /// Number of the innermost global index bits that can use GPU-direct P2P.
  /// For example, three bits cover the eight GPUs connected by NVLink or
  /// NVSwitch within a DGX node.
  int32_t p2pDeviceBits = 0;

  /// Base-two logarithm of the distributed transfer workspace size. The
  /// minimum supported value is 24, corresponding to 16 MiB.
  int32_t dataTransferBufferBits = 26;

  /// Number of noise trajectories averaged for an exact observe call. Unlike
  /// shots-based observe, these samples contain only Kraus-branch randomness,
  /// not additional bit-string measurement randomness.
  std::size_t observeTrajectories = 1000;

  /// Explicit trajectory batch size, or no value to derive it from memory.
  std::optional<int32_t> trajectoryBatchSize;

  /// Limits controlling when trajectory and PTSBE batching is profitable.
  /// States larger than batchedMaxQubits generally provide no batching benefit;
  /// batches smaller than batchedMinBatchSize use sequential trajectories.
  std::size_t batchedMaxBranches = 16;
  std::size_t batchedMaxQubits = 20;
  std::size_t batchedMinBatchSize = 4;

  /// Minimum random-number count that uses GPU generation and sorting.
  std::size_t gpuRngThreshold = 100'000;

  /// Minimum circuit width that uses a distributed state vector.
  int32_t mgpuQubitThreshold = 25;

  /// Retain allocations in the default CUDA pool to avoid repeated allocation
  /// overhead across simulation loops.
  bool enableMemPool = true;

  /// Allow cuStateVec floating-point emulation kernels.
  bool allowFp32Emulation = true;

  /// Decompose exp-Pauli operations instead of using the native rotation API.
  bool forceExpPauliDecomposition = false;

  /// Disable state reuse by forcing deallocation between simulations.
  bool forceAllocateState = false;

  /// Log the randomly selected Kraus branch for each trajectory.
  bool logTrajectorySampling = false;

  /// Sample all PTSBE batch members together one shot at a time instead of
  /// sampling each completed trajectory with its exact shot count.
  bool ptsbeBatchSampleShotByShot = false;

  /// Optional GPU-fabric topology and MPI provider configuration.
  std::optional<std::string> gpuFabric;
  std::string mpiLibrary = "libmpi.so";
  CommunicatorPlugin communicatorPlugin = CommunicatorPlugin::Auto;

  static CuStateVecConfig fromEnvironment();

  /// Applies overrides that are meaningful only for a distributed simulator.
  /// Keeping these separate lets single-GPU targets ignore MGPU shell
  /// configuration.
  void applyDistributedEnvironment();

  /// Applies GPU-family fusion defaults before honoring environment overrides.
  static CuStateVecConfig fromEnvironment(int32_t computeCapabilityMajor,
                                          int32_t computeCapabilityMinor,
                                          bool isFp32);
};

} // namespace cudaq::cusv
