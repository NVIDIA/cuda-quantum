/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecConfig.h"
#include "CuStateVecDevice.h"

#include <cudaq.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <sstream>
#include <string>

/// State-vector capacity probes for the host-device migration tests.
///
/// These mirror sizing the simulator performs in `ensureState()`, but
/// intersect the configured `CUDAQ_MAX_*_MEMORY_GB` budgets with the memory the
/// machine actually has free. The tests are launched with budgets larger than
/// some validation nodes provide, so the environment variables alone cannot
/// tell whether a case will fit.
namespace state_vector_memory {

#ifdef CUDAQ_SIMULATION_SCALAR_FP32
using SimulationScalar = float;
#else
using SimulationScalar = double;
#endif

namespace detail {

/// The migration tests place every rank on a single node, so ranks share one
/// host memory pool.
inline int32_t localRanks() {
  return cudaq::mpi::is_initialized() ? cudaq::mpi::num_ranks() : 1;
}

inline std::size_t usableGpuBytes(const cudaq::cusv::CuStateVecConfig &config) {
  const std::size_t freeBytes =
      cudaq::cusv::queryDeviceMemoryCapacity().usableBytes();
  if (!config.maxGpuMemoryGb)
    return freeBytes;
  return std::min(freeBytes, static_cast<std::size_t>(*config.maxGpuMemoryGb)
                                 << 30);
}

/// Mirrors `maximumMigrationWires()`: `MemAvailable` with a 10% margin, capped
/// by the configured budget, then divided among the ranks sharing the node.
inline std::size_t
usableHostBytes(const cudaq::cusv::CuStateVecConfig &config) {
  if (config.maxCpuMemoryGb == 0)
    return 0;
  const auto memAvailable = cudaq::cusv::systemMemAvailableBytes();
  const std::size_t availableBytes =
      memAvailable ? (*memAvailable / 10) * 9 : 0;
  const std::size_t hostBytes = [&] {
    if (!config.maxCpuMemoryGb)
      return availableBytes;
    const std::size_t requested =
        static_cast<std::size_t>(*config.maxCpuMemoryGb) << 30;
    return availableBytes == 0 ? requested
                               : std::min(requested, availableBytes);
  }();
  return hostBytes / localRanks();
}

} // namespace detail

/// Widest state vector this rank can hold, in qubits, or -1 when not even a
/// single amplitude fits.
inline int32_t stateVectorCapacityQubits() {
  const auto config = cudaq::cusv::CuStateVecConfig::fromEnvironment();
  const std::size_t amplitudes =
      detail::usableGpuBytes(config) / sizeof(std::complex<SimulationScalar>);
  if (amplitudes == 0)
    return -1;

  const int32_t deviceWires =
      static_cast<int32_t>(std::bit_width(amplitudes)) - 1;
  const std::size_t deviceBytes =
      (std::size_t{1} << deviceWires) * sizeof(std::complex<SimulationScalar>);
  const int32_t migrationWires = cudaq::cusv::migrationWireCapacity(
      detail::usableHostBytes(config), deviceBytes);
  // A distributed run splits the state across ranks, so each process bit buys
  // one more qubit on top of this rank's device and host tiers.
  const int32_t processBits =
      static_cast<int32_t>(
          std::bit_width(static_cast<unsigned>(detail::localRanks()))) -
      1;
  return deviceWires + migrationWires + processBits;
}

/// Under MPI every rank returns the same answer, so all ranks skip or all run.
inline bool systemHasEnoughMemory(int numQubits) {
  const bool fits = stateVectorCapacityQubits() >= numQubits;
  if (!cudaq::mpi::is_initialized())
    return fits;
  // No boolean reduction is exposed; multiplying 0/1 gives a logical AND.
  return cudaq::mpi::all_reduce(fits ? 1.0 : 0.0, std::multiplies<double>()) !=
         0.0;
}

inline std::string memoryShortfallMessage(int numQubits) {
  const double gibibytes =
      std::ldexp(sizeof(std::complex<SimulationScalar>), numQubits - 30);
  std::ostringstream message;
  message << "Skipping because the system doesn't have enough memory: a "
          << numQubits << "-qubit state vector needs " << gibibytes
          << " GiB, but the free GPU and CPU memory holds at most "
          << stateVectorCapacityQubits() << " qubits.";
  return message.str();
}

} // namespace state_vector_memory

/// Skip the enclosing test when a `numQubits` state vector does not fit in the
/// free GPU and CPU memory of this machine.
#define SKIP_IF_INSUFFICIENT_MEMORY(numQubits)                                 \
  do {                                                                         \
    if (!state_vector_memory::systemHasEnoughMemory(numQubits))                \
      GTEST_SKIP() << state_vector_memory::memoryShortfallMessage(numQubits);  \
  } while (0)
