/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecCircuitSimulatorEx.h"
#include "CuStateVecCommunicator.h"
#include "cudaq/cudaq_mpi.h"
#include "cudaq/runtime/logger/logger.h"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <unistd.h>
#include <vector>

namespace cudaq::cusv {

/// @brief Multi-process extension of the `cuStateVecEx` circuit simulator.
///
/// This class configures the communicator, GPU fabric, global-index-bit
/// layout, and host-device migration policy for a distributed state vector.
/// Circuits below the configured multi-GPU threshold continue to use the base
/// single-process state representation.
template <typename Scalar>
class CuStateVecMpiCircuitSimulator : public CuStateVecCircuitSimulator<Scalar>,
                                      public nvqir::MpiCircuitSimulator {
  using Base = CuStateVecCircuitSimulator<Scalar>;

  using Base::m_config;
  using Base::m_engine;
  using Base::m_pendingRandomNumbers;
  using Base::m_state;
  using Base::maximumDeviceWires;
  using Base::maximumMigrationWires;
  using Base::nQubitsAllocated;
  using Base::summaryData;

public:
  CuStateVecMpiCircuitSimulator()
      : m_communicator([this] {
          m_config.applyDistributedEnvironment();
          return std::make_shared<CuStateVecCommunicator>(
              m_config.communicatorPlugin, m_config.mpiLibrary);
        }()),
        m_requestedGlobalIndexBits(m_config.globalIndexBits),
        m_requestedP2pDeviceBits(m_config.p2pDeviceBits),
        m_worldRank(m_communicator->provider() == CommunicatorProvider::Cudaq
                        ? cudaq::mpi::rank()
                        : m_communicator->rank()) {
    initializeCommunicatorState();
  }

  ~CuStateVecMpiCircuitSimulator() override {
    m_engine.reset();
    m_state.reset();
    m_communicator.reset();
  }
  std::string name() const override {
    if constexpr (std::is_same_v<Scalar, float>)
      return "custatevec-mgpu-fp32";
    return "custatevec-mgpu-fp64";
  }

  nvqir::CircuitSimulator *clone() override { return this; }

  void setRandomSeed(std::size_t seed) override {
    m_userSeed = seed;
    Base::setRandomSeed(seed);
  }

  /// Replace the default world communicator before state allocation. Python
  /// (`mpi4py`) passes `MPI._addressof(comm)`; C++ passes the address of its
  /// `MPI_Comm`, with communicatorSize describing that handle representation.
  bool setMpiCommunicator(void *communicator, int communicatorSize) override {
    if (nQubitsAllocated != 0)
      throw std::runtime_error(
          "Cannot replace the MPI communicator after state allocation.");
    if (!communicator || communicatorSize <= 0)
      throw std::invalid_argument("Invalid MPI communicator.");
    // A normal deallocation retains the reusable state allocation. Drop it
    // before changing communicators so the descriptor cannot outlive the
    // communicator it was created with.
    m_state.reset();
    m_communicator->setCommunicator(communicator, communicatorSize);
    initializeCommunicatorState();
    return true;
  }

  void tearDownBeforeMPIFinalize() override {
    m_engine.reset();
    m_state.reset();
    m_communicator.reset();
  }

protected:
  // Distributed sampling requires every rank to consume identical
  // probabilities. All ranks advance their GPU streams, while rank zero
  // provides the sequence that is broadcast to the other ranks. Small CPU
  // requests remain identical because all ranks share the same seed.
  std::vector<double> generateRandomNumbers(std::size_t count) override {
    if (count < m_config.gpuRngThreshold)
      return Base::generateRandomNumbers(count);
    if (count > static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
      throw std::overflow_error("Random-number count exceeds MPI range.");
    // Advance every rank's GPU RNG stream. Keep the CPU engines aligned as
    // well: a rank that falls back consumes values directly, while a successful
    // rank skips the same number without materializing a second result vector.
    std::vector<double> result;
    try {
      result = this->m_gpuRandom.generate(count);
      this->m_randomEngine.discard(count);
    } catch (const std::exception &error) {
      CUDAQ_WARN("GPU random-number generation failed for {} values ({}); "
                 "falling back to CPU generation.",
                 count, error.what());
      result = this->randomVector(count);
    }
    // Rank zero owns the observable sequence; broadcasting its values keeps
    // distributed sampling deterministic.
    if (m_communicatorRank != 0)
      std::fill(result.begin(), result.end(), 0.0);
    m_communicator->broadcast(result.data(), static_cast<int32_t>(count),
                              CUDA_R_64F, 0);
    return result;
  }

  bool supportsBatchedTrajectories() const override { return false; }

  void addQubitsToState(std::size_t count, const void *stateData) override {
    Base::addQubitsToState(count, stateData);
    // The initial ensureState() may defer distribution until the newly
    // allocated wires make a valid distributed state.
    ensureState();
  }

  /// Lazily (re)build this rank's state descriptor. Below the multi-GPU
  /// threshold every rank keeps the same full (replicated) single-process state
  /// through the base class; once the circuit is large enough to distribute,
  /// this promotes that replicated state to a multi-process descriptor whose
  /// wires are split across ranks, carrying the amplitudes across the change.
  void ensureState() override {
    // The snapshot below reads through the base class, which re-enters
    // ensureState(); skip that recursive call while the transition is running.
    if (m_transitioningToDistributed)
      return;
    // Below the distribution threshold: keep the replicated single-process
    // state that every rank holds in full.
    if (!shouldDistribute()) {
      Base::ensureState();
      return;
    }
    if (m_communicatorSize > 1 &&
        !std::has_single_bit(static_cast<unsigned>(m_communicatorSize)))
      throw std::runtime_error(
          "cuStateVecEx requires a power-of-two communicator size for "
          "distributed state creation.");
    const int32_t processBits = totalProcessBits();
    if (m_state) {
      // Already distributed -- nothing to promote.
      if (distributionType() == CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS)
        return;
      // A distributed descriptor cannot represent fewer logical wires than
      // process bits. Let this allocation grow the replicated state first.
      if (m_state->numWires() < processBits)
        return;
    }

    // Below the distribution threshold every rank holds the same complete
    // state. Preserve that replicated state while replacing its descriptor with
    // a distributed layout, then write only each rank's assigned sub-states.
    std::vector<std::complex<Scalar>> replicatedState;
    int32_t existingWires = 0;
    if (m_state) {
      existingWires = m_state->numWires();
      if (existingWires != 0) {
        // Set the re-entry guard: readState() flushes through the base class,
        // which would otherwise recurse back into ensureState().
        detail::ScopedFlag transition(m_transitioningToDistributed);
        replicatedState = Base::readState();
      }
    }

    // Capacity of each memory tier -- device (local) wires and host (migration)
    // wires -- and the resulting per-layer index-bit classes and sizes.
    const int32_t maxLocalWires = maximumDeviceWires();
    const int32_t maxMigrationWires = maximumMigrationWires(maxLocalWires);
    const auto [classes, bits] = communicationLayers(maxMigrationWires);
    // Select how peer ranks share sub-state device memory, consistent with the
    // layers `communicationLayers` produced: cuStateVecEx requires a sharing
    // method whenever a global-index layer is `INTERPROC_P2P`, and `NONE` when
    // none is. Both are governed by `p2pDeviceBits`, so they always agree.
    // `FABRIC_HANDLE` is the method required to enable a multi-node NVLink
    // fabric (MNNVL); for any other (intra-node) P2P domain -- NVLink or
    // NVSwitch within a host -- `AUTODETECT` lets cuStateVecEx pick the
    // mechanism itself.
    const auto memorySharing =
        m_config.p2pDeviceBits == 0 ? CUSTATEVEC_EX_MEMORY_SHARING_METHOD_NONE
        : m_config.gpuFabric == "MNNVL"
            ? CUSTATEVEC_EX_MEMORY_SHARING_METHOD_FABRIC_HANDLE
            : CUSTATEVEC_EX_MEMORY_SHARING_METHOD_AUTODETECT;
    const std::size_t workspace = std::size_t{1}
                                  << m_config.dataTransferBufferBits;
    // Build the empty distributed descriptor with the computed layout.
    m_state.emplace(CuStateVecState<Scalar>::createMultiProcess(
        maxLocalWires + maxMigrationWires + processBits, maxLocalWires,
        m_deviceId, memorySharing, classes, bits, workspace, m_communicator,
        m_config.allowFp32Emulation));

    if (existingWires != 0) {
      // Re-materialize the saved amplitudes: partition the existing wires into
      // device-local, host-migration, and cross-rank (global) domains, then
      // write the replicated state back into the distributed layout.
      const int32_t processLocalWires = existingWires - processBits;
      const int32_t localWires = std::min(processLocalWires, maxLocalWires);
      m_state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, localWires);
      m_state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION,
                        processLocalWires - localWires);
      m_state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE,
                        processBits);
      writeState(replicatedState);
    }
  }

  void addZeroWires(std::size_t count) override {
    if (!shouldDistribute() ||
        distributionType() != CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS) {
      Base::addZeroWires(count);
      return;
    }

    if (m_state->numWires() == 0) {
      const int32_t processBits = totalProcessBits();
      if (count < static_cast<std::size_t>(processBits))
        throw std::runtime_error(
            "Distributed state requires at least log2(ranks) qubits.");
      const int32_t processLocalWires =
          static_cast<int32_t>(count) - processBits;
      const int32_t localWires =
          std::min(processLocalWires, m_state->maxLocalWires());
      m_state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, localWires);
      m_state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION,
                        processLocalWires - localWires);
      m_state->addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_GLOBAL_DEVICE,
                        processBits);
      return;
    }
    Base::addZeroWires(count);
  }

  std::vector<std::complex<Scalar>> readState() override {
    // Return a logical, globally ordered state rather than rank-ordered
    // storage. cuStateVecEx may assign non-contiguous sub-state indices to each
    // rank.
    this->normalizeStateWireOrdering();
    if (distributionType() != CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS)
      return Base::readState();
    const int32_t localWires = m_state->numLocalWires();
    const std::size_t localSize = std::size_t{1} << localWires;
    const std::vector<int32_t> localIndices = localSubStateIndices();
    const std::size_t localElements = localSize * localIndices.size();
    std::vector<std::complex<Scalar>> local(localElements);
    for (std::size_t index = 0; index < localIndices.size(); ++index) {
      const std::size_t begin =
          static_cast<std::size_t>(localIndices[index]) * localSize;
      m_state->getState(local.data() + index * localSize, begin,
                        begin + localSize);
    }
    m_state->synchronize();

    if (localElements >
        static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
      throw std::overflow_error("Local state vector exceeds MPI count range.");

    const std::size_t globalSubStates =
        localIndices.size() * m_communicatorSize;
    std::vector<int32_t> subStateIndices(globalSubStates);
    m_communicator->allGather(localIndices.data(), subStateIndices.data(),
                              static_cast<int32_t>(localIndices.size()),
                              CUDA_R_32I);

    std::vector<std::complex<Scalar>> rankOrdered(localElements *
                                                  m_communicatorSize);
    m_communicator->allGather(local.data(), rankOrdered.data(),
                              static_cast<int32_t>(localElements),
                              complexDataType<Scalar>());

    // allGather yields rank-ordered data; scatter each rank's sub-states into
    // their global sub-state positions, checking every index is in range and
    // assigned exactly once.
    std::vector<std::complex<Scalar>> global(rankOrdered.size());
    std::vector<bool> assigned(globalSubStates, false);
    for (int32_t rank = 0; rank < m_communicatorSize; ++rank) {
      for (std::size_t localIndex = 0; localIndex < localIndices.size();
           ++localIndex) {
        const std::size_t rankIndex =
            static_cast<std::size_t>(rank) * localIndices.size() + localIndex;
        const int32_t subStateIndex = subStateIndices[rankIndex];
        if (subStateIndex < 0 ||
            subStateIndex >= static_cast<int32_t>(globalSubStates) ||
            assigned[subStateIndex])
          throw std::runtime_error(
              "Invalid cuStateVecEx distributed sub-state assignment.");
        assigned[subStateIndex] = true;
        std::copy_n(rankOrdered.data() + rankIndex * localSize, localSize,
                    global.data() +
                        static_cast<std::size_t>(subStateIndex) * localSize);
      }
    }
    return global;
  }

  void writeState(const std::complex<Scalar> *values,
                  std::size_t size) override {
    this->normalizeStateWireOrdering();
    if (distributionType() != CUSTATEVEC_EX_SV_DISTRIBUTION_MULTI_PROCESS) {
      Base::writeState(values, size);
      return;
    }
    const int32_t localWires = m_state->numLocalWires();
    const std::size_t localSize = std::size_t{1} << localWires;
    const std::vector<int32_t> localIndices = localSubStateIndices();
    const std::size_t globalSubStates =
        localIndices.size() * m_communicatorSize;
    if (!values || size != localSize * globalSubStates)
      throw std::invalid_argument("Invalid distributed state-vector size.");
    for (const int32_t subStateIndex : localIndices) {
      const std::size_t begin =
          static_cast<std::size_t>(subStateIndex) * localSize;
      m_state->setState(values + begin, begin, begin + localSize);
    }
    m_state->synchronize();
  }

  void writeState(const std::vector<std::complex<Scalar>> &values) override {
    writeState(values.data(), values.size());
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    // Normalize deferred work before transferring descriptor ownership. The
    // SimulationState adapter preserves distributed ownership and collectives.
    if (!m_state)
      throw std::runtime_error("No cuStateVec state is allocated.");
    this->flushGateQueue();
    if (this->isNoisySimulation())
      this->replayDeferredFromCurrentState();
    this->synchronize();
    auto result = std::make_unique<CuStateVecSimulationState<Scalar>>(
        std::move(*m_state));
    m_state.reset();
    return result;
  }

private:
  void initializeCommunicatorState() {
    // Fabric discovery and rank-width clamping derive effective values from
    // the active communicator. Restore the parsed user settings before each
    // rebinding so a previous communicator cannot permanently alter them.
    m_config.globalIndexBits = m_requestedGlobalIndexBits;
    m_config.p2pDeviceBits = m_requestedP2pDeviceBits;
    // Device placement follows world rank even when simulation uses a custom
    // subcommunicator, keeping round-robin GPU selection stable.
    m_communicatorSize = m_communicator->size();
    m_communicatorRank = m_communicator->rank();
    if (m_communicatorSize < 1 || m_communicatorRank < 0 ||
        m_communicatorRank >= m_communicatorSize)
      throw std::runtime_error("Invalid distributed communicator topology.");
    initializeGpuFabric();
    const int32_t processBits = totalProcessBits();
    if (m_config.p2pDeviceBits > processBits) {
      CUDAQ_WARN("CUDAQ_MGPU_P2P_DEVICE_BITS={} exceeds log2(ranks)={}; "
                 "clamping to {}.",
                 m_config.p2pDeviceBits, processBits, processBits);
      m_config.p2pDeviceBits = processBits;
    }
    int32_t deviceCount = 0;
    HANDLE_CUDA_ERROR(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1)
      throw std::runtime_error("No CUDA device is available.");
    m_deviceId = m_worldRank % deviceCount;
    HANDLE_CUDA_ERROR(cudaSetDevice(m_deviceId));

    // Start every rank with the same seed so CPU-generated sampling and noise
    // choices remain synchronized across the distributed state.
    std::uint64_t seed = m_userSeed.value_or(0);
    if (!m_userSeed && m_communicatorRank == 0)
      seed = std::random_device{}();
    m_communicator->broadcast(&seed, 1, CUDA_R_64I, 0);
    Base::setRandomSeed(seed);
    summaryData.name = name();
  }

  int32_t sharedMemoryDomainSize() const {
    // The NVL P2P domain is the group of ranks on the same physical host, so
    // gather every rank's host name to count how many share each host.
    constexpr std::size_t maxHostnameLength = 256;
    std::array<char, maxHostnameLength> hostname{};
    const int32_t hostnameAvailable =
        gethostname(hostname.data(), hostname.size()) == 0 ? 1 : 0;
    hostname.back() = '\0';

    std::vector<char> hostnames(maxHostnameLength * m_communicatorSize);
    m_communicator->allGather(hostname.data(), hostnames.data(),
                              static_cast<int32_t>(maxHostnameLength),
                              CUDA_R_8U);
    std::vector<int32_t> hostnameAvailability(m_communicatorSize);
    m_communicator->allGather(&hostnameAvailable, hostnameAvailability.data(),
                              1, CUDA_R_32I);
    if (std::find(hostnameAvailability.begin(), hostnameAvailability.end(),
                  0) != hostnameAvailability.end())
      throw std::runtime_error("Unable to query a distributed host name.");

    // Domain size = number of ranks sharing a host name; it must be identical
    // on every host.
    int32_t domainSize = 0;
    for (int32_t rank = 0; rank < m_communicatorSize; ++rank) {
      const std::string_view rankHostname(hostnames.data() +
                                          rank * maxHostnameLength);
      int32_t rankDomainSize = 0;
      for (int32_t peer = 0; peer < m_communicatorSize; ++peer) {
        const std::string_view peerHostname(hostnames.data() +
                                            peer * maxHostnameLength);
        if (peerHostname == rankHostname)
          ++rankDomainSize;
      }
      if (domainSize != 0 && domainSize != rankDomainSize)
        throw std::invalid_argument(
            "CUDAQ_GPU_FABRIC=NVL requires the same number of ranks on every "
            "host.");
      domainSize = rankDomainSize;
    }
    // Each host's ranks must form a contiguous block so the innermost P2P index
    // bits address exactly one shared-memory node.
    for (int32_t first = 0; first < m_communicatorSize; first += domainSize) {
      const std::string_view blockHostname(hostnames.data() +
                                           first * maxHostnameLength);
      for (int32_t rank = first + 1; rank < first + domainSize; ++rank) {
        const std::string_view rankHostname(hostnames.data() +
                                            rank * maxHostnameLength);
        if (rankHostname != blockHostname)
          throw std::invalid_argument(
              "CUDAQ_GPU_FABRIC=NVL requires ranks on each host to occupy "
              "a contiguous rank block.");
      }
    }
    return domainSize;
  }

  void initializeGpuFabric() {
    // CUDAQ_GPU_FABRIC is communicator-dependent and is therefore interpreted
    // only after communicator initialization. MNNVL spans the communicator,
    // NVL spans the shared-memory node, an integer gives an explicit NVLink
    // domain size, and NONE disables P2P global bits.
    if (!m_config.gpuFabric)
      return;
    const std::string &fabric = *m_config.gpuFabric;
    int32_t domainSize = 1;
    if (fabric == "MNNVL") {
      domainSize = m_communicatorSize;
    } else if (fabric == "NVL") {
      domainSize = sharedMemoryDomainSize();
    } else if (fabric != "NONE") {
      const char *const begin = fabric.data();
      const char *const end = begin + fabric.size();
      const auto [position, error] = std::from_chars(begin, end, domainSize);
      if (error != std::errc{} || position != end || domainSize < 1)
        throw std::invalid_argument(
            "CUDAQ_GPU_FABRIC must be MNNVL, NVL, NONE, or a positive "
            "integer domain size.");
    }

    domainSize = std::min(domainSize, m_communicatorSize);
    if (!std::has_single_bit(static_cast<unsigned>(domainSize)) ||
        m_communicatorSize % domainSize != 0)
      throw std::invalid_argument(
          "CUDAQ_GPU_FABRIC domain size must be a power-of-two factor of "
          "the communicator size.");
    m_config.p2pDeviceBits = static_cast<int32_t>(
        std::countr_zero(static_cast<unsigned>(domainSize)));
    // Unless the user supplied explicit communication layers, separate the
    // P2P domain from the outer communicator layer.
    if (m_config.globalIndexBits.empty() && domainSize > 1 &&
        domainSize < m_communicatorSize)
      m_config.globalIndexBits = {m_config.p2pDeviceBits,
                                  totalProcessBits() - m_config.p2pDeviceBits};
  }

  bool shouldDistribute() const {
    return m_communicatorSize > 1 &&
           nQubitsAllocated >=
               static_cast<std::size_t>(m_config.mgpuQubitThreshold) &&
           nQubitsAllocated >= static_cast<std::size_t>(totalProcessBits());
  }

  int32_t totalProcessBits() const {
    if (m_communicatorSize == 1)
      return 0;
    return static_cast<int32_t>(
        std::countr_zero(static_cast<unsigned>(m_communicatorSize)));
  }

  /// Build the global-index-bit layer layout of the distributed descriptor: the
  /// per-layer class list and the matching per-layer bit counts, ordered from
  /// the innermost (fastest) layer outward.
  ///
  /// Layer sizes come from `CUDAQ_GLOBAL_INDEX_BITS` when set, otherwise they
  /// are derived as an inner peer-to-peer layer of `p2pDeviceBits` followed by
  /// an outer communicator layer for the remaining process bits. A layer within
  /// the innermost `p2pDeviceBits` is classified `INTERPROC_P2P` (peer GPUs
  /// reachable by GPU-direct P2P) and `COMMUNICATOR` otherwise, so an
  /// `INTERPROC_P2P` layer exists exactly when `p2pDeviceBits > 0` -- the
  /// invariant the memory-sharing method relies on. A non-zero `migrationWires`
  /// inserts a host-memory `MIGRATION` layer at `migrationLevel`, which
  /// defaults to the outermost position.
  ///
  /// Examples (layers innermost-first):
  /// ```
  /// 32 ranks (5 process bits), p2pDeviceBits = 3  // 4 nodes x 8 P2P GPUs
  ///   bits    = {3, 2}
  ///   classes = {INTERPROC_P2P, COMMUNICATOR}
  /// 8 ranks (3 process bits), p2pDeviceBits = 0   // no P2P domain
  ///   bits    = {3}
  ///   classes = {COMMUNICATOR}
  /// 16 ranks (4 process bits), p2pDeviceBits = 4  // GH200, MNNVL -> all P2P
  ///   bits    = {4, 2}                            // 2 migration bits (host)
  ///   classes = {INTERPROC_P2P, MIGRATION}         // 480/64 -> 2 host wires
  /// ```
  /// (`MNNVL` places all ranks in one P2P domain, so `p2pDeviceBits` =
  /// log2(ranks) and the `FABRIC_HANDLE` sharing method is used.)
  std::pair<std::vector<custatevecExGlobalIndexBitClass_t>,
            std::vector<int32_t>>
  communicationLayers(int32_t migrationWires) const {
    const int32_t processBits = totalProcessBits();
    std::vector<int32_t> bits = m_config.globalIndexBits;
    // Derive the layout when the user did not pin it: an inner P2P layer, then
    // a communicator layer for whatever process bits remain.
    if (bits.empty()) {
      if (m_config.p2pDeviceBits > 0)
        bits.push_back(m_config.p2pDeviceBits);
      if (m_config.p2pDeviceBits < processBits)
        bits.push_back(processBits - m_config.p2pDeviceBits);
    }
    if (std::accumulate(bits.begin(), bits.end(), int32_t{0}) != processBits)
      throw std::invalid_argument(
          "CUDAQ_GLOBAL_INDEX_BITS must sum to log2(ranks).");
    if (m_config.p2pDeviceBits < 0 || m_config.p2pDeviceBits > processBits)
      throw std::invalid_argument(
          "CUDAQ_MGPU_P2P_DEVICE_BITS exceeds log2(ranks).");

    // Walk the layers outward, tagging each as P2P while it sits inside the
    // innermost p2pDeviceBits and communicator once past that boundary.
    std::vector<custatevecExGlobalIndexBitClass_t> classes;
    int32_t assigned = 0;
    for (const int32_t layerBits : bits) {
      // A single layer cannot span both sides of the P2P boundary.
      if (assigned < m_config.p2pDeviceBits &&
          assigned + layerBits > m_config.p2pDeviceBits)
        throw std::invalid_argument(
            "P2P device bits must end at a global-index layer boundary.");
      classes.push_back(
          assigned < m_config.p2pDeviceBits
              ? CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_INTERPROC_P2P
              : CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_COMMUNICATOR);
      assigned += layerBits;
    }
    // Where to slot the host-memory layer. By default it is appended as the
    // outermost (slowest) layer, so host-device migration surrounds the faster
    // GPU-GPU exchanges; `CUDAQ_HOST_DEVICE_MIGRATION_LEVEL` overrides the
    // index.
    const int32_t migrationLevel =
        m_config.migrationLevel.value_or(static_cast<int32_t>(classes.size()));
    if (migrationLevel < 0 ||
        migrationLevel > static_cast<int32_t>(classes.size()))
      throw std::invalid_argument(
          "CUDAQ_HOST_DEVICE_MIGRATION_LEVEL exceeds the number of "
          "global-index layers.");
    if (migrationWires != 0) {
      // Host memory is represented as another global-index layer. Its position
      // controls which communication layers surround a migration operation.
      classes.insert(classes.begin() + migrationLevel,
                     CUSTATEVEC_EX_GLOBAL_INDEX_BIT_CLASS_MIGRATION);
      bits.insert(bits.begin() + migrationLevel, migrationWires);
    }
    return {std::move(classes), std::move(bits)};
  }

  custatevecExStateVectorDistributionType_t distributionType() const {
    return m_state ? m_state->distributionType()
                   : CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE;
  }

  std::vector<int32_t> localSubStateIndices() const {
    const auto indices = queryInt32ArrayProperty(
        m_state->descriptor(), CUSTATEVEC_EX_SV_PROP_NUM_SUBSVS,
        CUSTATEVEC_EX_SV_PROP_SUBSV_INDICES);
    if (indices.empty())
      throw std::runtime_error("No distributed sub-state vector is available.");
    return indices;
  }

  // Communicator (MPI or plugin) owning this simulator's process group.
  std::shared_ptr<CuStateVecCommunicator> m_communicator;
  // User-requested layout, kept verbatim so it can be re-resolved each time the
  // communicator (and its fabric-derived config) is rebound.
  const std::vector<int32_t> m_requestedGlobalIndexBits;
  const int32_t m_requestedP2pDeviceBits;
  // Rank in the world communicator; used to assign this process's GPU.
  const int32_t m_worldRank;
  // Size and rank within the active (possibly split) communicator.
  int32_t m_communicatorSize = 1;
  int32_t m_communicatorRank = 0;
  // CUDA device this rank drives.
  int32_t m_deviceId = 0;
  // Preserve an explicit user seed across communicator rebinding.
  std::optional<std::size_t> m_userSeed;
  // Guards ensureState() against re-entry while promoting the replicated state
  // to a distributed one.
  bool m_transitioningToDistributed = false;
};

} // namespace cudaq::cusv
