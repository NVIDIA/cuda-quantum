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

namespace cudaq::cusv {

/// Captures the memory-capacity inputs used to size an Ex state vector.
///
/// CUDA free memory excludes reusable bytes held by the default memory pool.
/// On the unified-memory GB10 platform, system-available memory can also exceed
/// the value reported by CUDA.
struct DeviceMemoryCapacity {
  std::size_t cudaFreeBytes = 0;
  std::size_t poolReservedBytes = 0;
  std::size_t poolUsedBytes = 0;
  std::optional<std::size_t> systemAvailableBytes;
  int32_t computeCapabilityMajor = 0;
  int32_t computeCapabilityMinor = 0;

  std::size_t usableBytes() const;
};

/// Queries the active CUDA device and host for state-vector memory capacity.
DeviceMemoryCapacity queryDeviceMemoryCapacity();

/// Reads `MemAvailable` from `/proc/meminfo` -- the kernel's estimate of memory
/// usable for new (pageable) allocations, accounting for reclaimable page
/// cache. Returns nullopt when it cannot be read (e.g., old Linux kernel or
/// non-Linux).
std::optional<std::size_t> systemMemAvailableBytes();

/// Returns the number of host-migration wires supported by the memory ratio,
/// capped at the three migration wires accepted by cuStateVecEx.
int32_t migrationWireCapacity(std::size_t hostBytes, std::size_t deviceBytes);

/// Selects the trajectory batch size from available memory or an explicit
/// user value, then applies the legacy power-of-two work-item cap.
std::size_t trajectoryBatchSize(std::size_t availableBytes,
                                std::size_t stateBytes, std::size_t workItems,
                                std::optional<int32_t> configuredBatchSize);

} // namespace cudaq::cusv
