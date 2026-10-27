/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CustabilizerErrorHandling.h"
#include "../stim/StimCircuitSimulator.h"

#include <cassert>
#include <iomanip>
#include <limits>
#include <locale>
#include <numeric>
#include <optional>
#include <sstream>
#include <unordered_set>

using namespace cudaq;

namespace nvqir {

namespace {

struct CustabilizerCircuit {
  custabilizerHandle_t handle = nullptr;
  custabilizerCircuit_t circuit = nullptr;
  void *buffer_d = nullptr;
  int64_t buffer_size = 0;

  explicit CustabilizerCircuit(const std::string &stim_circuit_str) {
    HANDLE_CUST_ERROR(custabilizerCreate(&handle));
    HANDLE_CUST_ERROR(custabilizerCircuitSizeFromString(
        handle, stim_circuit_str.c_str(), &buffer_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&buffer_d, static_cast<size_t>(buffer_size)));
    HANDLE_CUST_ERROR(custabilizerCreateCircuitFromString(
        handle, stim_circuit_str.c_str(), buffer_d, buffer_size, &circuit));
  }

  ~CustabilizerCircuit() {
    if (buffer_d)
      cudaFree(buffer_d);
    if (circuit)
      custabilizerDestroyCircuit(circuit);
    if (handle)
      custabilizerDestroy(handle);
  }

  CustabilizerCircuit(const CustabilizerCircuit &) = delete;
  CustabilizerCircuit &operator=(const CustabilizerCircuit &) = delete;
};

struct CustabilizerFrameRunner {
  // cuStabilizer context owned by this runner.
  custabilizerHandle_t handle = nullptr;
  // Frame simulator configured for the allocated table dimensions.
  custabilizerFrameSimulator_t frame_simulator = nullptr;
  // Packed GPU X flips, with one row per qubit and one bit per shot.
  custabilizerBitInt_t *x_table_d = nullptr;
  // Packed GPU Z flips, with one row per qubit and one bit per shot.
  custabilizerBitInt_t *z_table_d = nullptr;
  // Packed GPU measurement flips, with one row per measurement.
  custabilizerBitInt_t *m_table_d = nullptr;
  // Number of qubits represented by the frame tables.
  int64_t num_qubits = 0;
  // Shot capacity of the allocated tables.
  int64_t allocated_shots = 0;
  // Row capacity of the measurement table.
  int64_t allocated_measurements = 0;
  // Byte offset between adjacent rows in each GPU table.
  int64_t stride_bytes = 0;
  // Circuit already converted for cuStabilizer, replayed to generate flips.
  stim::Circuit circuitCache;
  // Whether the recorded circuit has run and its measurement table is valid.
  bool cache_executed = false;

  explicit CustabilizerFrameRunner(int64_t num_qubits)
      : num_qubits(num_qubits) {
    if (num_qubits <= 0)
      throw std::runtime_error(
          "cuStabilizer frame simulator requires num_qubits > 0");
    HANDLE_CUST_ERROR(custabilizerCreate(&handle));
  }

  ~CustabilizerFrameRunner() {
    deallocate();
    if (handle)
      custabilizerDestroy(handle);
  }

  CustabilizerFrameRunner(const CustabilizerFrameRunner &) = delete;
  CustabilizerFrameRunner &operator=(const CustabilizerFrameRunner &) = delete;

  // Append converted operations for replay, keeping each instruction's targets
  // disjoint because cuStabilizer processes them on parallel GPU threads.
  // cuStabilizer requires non-overlapping instruction targets:
  // https://docs.nvidia.com/cuda/cuquantum/26.09.0/custabilizer/overview.html#circuit
  void apply(const stim::Circuit &circuit) {
    circuit.for_each_operation([&](const stim::CircuitInstruction &op) {
      const auto flags = stim::GATE_DATA[op.gate_type].flags;
      const std::size_t width = flags & stim::GATE_TARGETS_PAIRS ? 2 : 1;
      // Preserve operations such as TICK and circuit annotations directly;
      // their targets do not represent parallel qubit updates to split.
      if (!(flags & (stim::GATE_IS_UNITARY | stim::GATE_IS_NOISY |
                     stim::GATE_IS_RESET | stim::GATE_PRODUCES_RESULTS))) {
        circuitCache.safe_append(op);
        lastInstructionQubits.clear();
        return;
      }
      for (std::size_t i = 0; i < op.targets.size(); i += width) {
        auto instruction = op;
        instruction.targets = {op.targets.begin() + i,
                               op.targets.begin() + i + width};
        // Start a fresh target set for the first instruction or a gate that
        // cannot fuse with the previous one. Reusing qubits across separate
        // instructions is safe.
        if (circuitCache.operations.empty() ||
            !circuitCache.operations.back().can_fuse(instruction)) {
          lastInstructionQubits.clear();
        } else {
          bool overlaps = false;
          for (const auto &target : instruction.targets)
            overlaps |= lastInstructionQubits.count(target.qubit_value()) != 0;
          if (overlaps) {
            // TICK also prevents cuStabilizer from fusing these targets again
            // when it parses the serialized circuit.
            circuitCache.safe_append_u("TICK", {});
            lastInstructionQubits.clear();
          }
        }
        circuitCache.safe_append(instruction);
        for (const auto &target : instruction.targets)
          lastInstructionQubits.insert(target.qubit_value());
      }
    });
    cache_executed = false;
  }

  // Discard the recorded circuit and cached results, retaining GPU buffers.
  void clear() {
    circuitCache.clear();
    lastInstructionQubits.clear();
    cache_executed = false;
  }

  // More qubits require larger frame tables. Retain the recorded circuit so
  // the next replay reconstructs the state, including earlier measurements.
  void growQubits(int64_t new_qubits) {
    if (new_qubits > num_qubits) {
      deallocate();
      num_qubits = new_qubits;
      cache_executed = false;
    }
  }

  /// @brief Replay the recorded circuit as needed and copy measurement flips
  /// from GPU memory to the host. Results are cached until apply() or clear().
  /// @param[in] num_shots Positive number of shots to simulate.
  /// @param[in] seed Random seed for replay; unused when returning cached
  /// results.
  /// @return Host vector of packed measurement flips relative to the noiseless
  /// tableau reference. Each measurement has a row of (num_shots + 31) / 32
  /// words; shot s occupies bit (s % 32) of word (s / 32) in that row. XOR a
  /// flip with the corresponding reference bit to obtain the sampled outcome.
  std::vector<uint32_t> getMeasurementTable(int64_t num_shots, uint64_t seed) {
    if (num_shots <= 0)
      throw std::runtime_error("num_shots must be > 0");
    // Number of measurement bits produced by one shot of the entire circuit.
    // For example, "M 0 1" followed by "M 0" counts as three measurement bits.
    int64_t measurement_count = circuitCache.count_measurements();

    // Reuse existing allocations unless more shots or measurement bits per shot
    // are needed than the GPU tables can currently hold.
    if (num_shots > allocated_shots ||
        measurement_count > allocated_measurements) {
      // Recreate the GPU tables and frame simulator for the required
      // dimensions.
      reallocate(num_shots, measurement_count);
    }

    if (!cache_executed) {
      // Preserve double precision and the decimal point cuStabilizer expects.
      std::ostringstream text;
      text.imbue(std::locale::classic());
      text << std::setprecision(std::numeric_limits<double>::max_digits10)
           << circuitCache;
      CustabilizerCircuit circuit(text.str());
      HANDLE_CUST_ERROR(custabilizerFrameSimulatorApplyCircuit(
          handle, frame_simulator, circuit.circuit,
          /*randomizeFrameAfterMeasurement=*/1, seed, x_table_d, z_table_d,
          m_table_d, nullptr));
      cache_executed = true;
    }

    const size_t words_per_row = static_cast<size_t>((num_shots + 31) / 32);
    const size_t total_words =
        words_per_row * static_cast<size_t>(measurement_count);
    std::vector<uint32_t> out(total_words);
    const size_t total_bytes = total_words * sizeof(uint32_t);
    if (total_bytes > 0)
      HANDLE_CUDA_ERROR(cudaMemcpy(out.data(), m_table_d, total_bytes,
                                   cudaMemcpyDeviceToHost));
    return out;
  }

private:
  // Qubits used by the last recorded instruction, including earlier appends.
  std::unordered_set<std::uint32_t> lastInstructionQubits;

  void deallocate() {
    if (x_table_d) {
      cudaFree(x_table_d);
      x_table_d = nullptr;
    }
    if (z_table_d) {
      cudaFree(z_table_d);
      z_table_d = nullptr;
    }
    if (m_table_d) {
      cudaFree(m_table_d);
      m_table_d = nullptr;
    }
    if (frame_simulator) {
      custabilizerDestroyFrameSimulator(frame_simulator);
      frame_simulator = nullptr;
    }
    allocated_shots = 0;
    allocated_measurements = 0;
  }

  /// @brief Replace the GPU tables and frame simulator with new allocations.
  /// Existing GPU contents are discarded; the recorded circuit is retained.
  /// X/Z tables have num_qubits rows, and the measurement table has
  /// new_measurements rows. All tables are zeroed, with each row padded to
  /// whole 32-bit words containing one bit per shot.
  /// @param[in] new_shots Number of shots to allocate in each table row.
  /// @param[in] new_measurements Number of measurement bits produced per shot.
  void reallocate(int64_t new_shots, int64_t new_measurements) {
    // Debug-only checks; callers ensure positive dimensions.
    assert(num_qubits > 0);
    assert(new_shots > 0);
    assert(new_measurements > 0);
    deallocate();

    allocated_shots = new_shots;
    allocated_measurements = new_measurements;
    // Each row stores one bit per shot. Round up to whole 32-bit words, then
    // convert the row length to bytes for cuStabilizer's table stride.
    stride_bytes =
        ((new_shots + 31) / 32) * static_cast<int64_t>(sizeof(uint32_t));

    // Each X/Z table has one row per qubit; this is the size of either table.
    const size_t xz_bytes =
        static_cast<size_t>(stride_bytes) * static_cast<size_t>(num_qubits);
    // The measurement table has one row per measurement bit produced per shot.
    const size_t m_bytes = static_cast<size_t>(stride_bytes) *
                           static_cast<size_t>(new_measurements);

    HANDLE_CUDA_ERROR(cudaMalloc(&x_table_d, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMalloc(&z_table_d, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMalloc(&m_table_d, m_bytes));
    HANDLE_CUDA_ERROR(cudaMemset(x_table_d, 0, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMemset(z_table_d, 0, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMemset(m_table_d, 0, m_bytes));

    HANDLE_CUST_ERROR(custabilizerCreateFrameSimulator(
        handle, num_qubits, new_shots, new_measurements, stride_bytes,
        &frame_simulator));
  }
};

/// @brief The CustabilizerCircuitSimulator extends StimCircuitSimulator
/// to use cuStabilizer for GPU-accelerated frame simulation, while keeping
/// Stim's TableauSimulator for the noiseless reference.
class CustabilizerCircuitSimulator : public StimCircuitSimulator {
protected:
  /// @brief Persistent frame runner for circuit accumulation and execution
  std::unique_ptr<CustabilizerFrameRunner> frameRunner;
  // Replays must preserve earlier measurement outcomes within a shot.
  std::optional<std::uint64_t> frameSeed;

  /// @brief Override: reject MSM mode, initialize frame runner
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    auto *executionContext = getExecutionContext();
    if (stateDataIn)
      throw std::runtime_error("The cuStabilizer backend does not support "
                               "initialization of qubits from state data.");
    if (executionContext && executionContext->name == "msm")
      throw std::runtime_error(
          "MSM mode is not supported by custabilizer backend.");
    StimCircuitSimulator::addQubitsToState(qubitCount, stateDataIn);
    if (nQubitsAllocated > 0 && !frameRunner) {
      frameRunner = std::make_unique<CustabilizerFrameRunner>(
          static_cast<int64_t>(nQubitsAllocated));
    } else if (frameRunner) {
      frameRunner->growQubits(static_cast<int64_t>(nQubitsAllocated));
    }
    // Allocated qubits start in |0>. Record resets so every replay initializes
    // their X frames to zero and Z frames randomly, as Stim does for new
    // qubits. Random Z frames supply measurement randomness after gates such as
    // H.
    std::vector<std::uint32_t> targets(qubitCount);
    std::iota(targets.begin(), targets.end(), nQubitsAllocated - qubitCount);
    applyOpToSims("R", targets);
  }

  /// @brief Override: also clear the frame runner
  void deallocateStateImpl() override {
    StimCircuitSimulator::deallocateStateImpl();
    if (frameRunner) {
      frameRunner->clear();
      frameRunner.reset();
    }
    frameSeed.reset();
  }

  /// @brief Apply original gates to the tableau and record supported GPU gates.
  void applyOpToSims(const std::string &gate_name,
                     const std::vector<uint32_t> &targets) override {
    if (targets.empty())
      return;
    stim::Circuit tempCircuit;
    CUDAQ_INFO("Calling applyOpToSims {} - {}", gate_name, targets);
    tempCircuit.safe_append_u(gate_name, targets);
    tableau->safe_do_circuit(tempCircuit);
    if (!frameRunner)
      return;

    // cuStabilizer supports a smaller set of Clifford gates than Stim. Replace
    // unsupported gates with supported gates that propagate the same X/Z
    // frames. The CPU tableau above keeps the original gates for reference
    // outcomes.
    const auto gate_type = tempCircuit.operations.front().gate_type;
    if (gate_type == stim::GateType::S_DAG) {
      // S and S_DAG map an X Pauli flip to Y and -Y, respectively. These have
      // identical X/Z frame bits; the CPU tableau retains the sign difference
      // needed for reference measurement outcomes.
      tempCircuit.operations.front().gate_type = stim::GateType::S;
    } else if (gate_type == stim::GateType::CY ||
               gate_type == stim::GateType::SWAP) {
      tempCircuit.clear();
      for (std::size_t i = 0; i < targets.size(); i += 2) {
        const auto a = targets[i], b = targets[i + 1];
        if (gate_type == stim::GateType::CY) {
          // CY(a,b): apply S_DAG(b), CX(a,b), then S(b).
          // Use S for S_DAG because their X/Z-frame updates are identical.
          tempCircuit.safe_append_u("S", {b});
          tempCircuit.safe_append_u("CX", {a, b});
          tempCircuit.safe_append_u("S", {b});
        } else {
          // SWAP(a,b): apply CX(a,b), CX(b,a), then CX(a,b).
          tempCircuit.safe_append_u("CX", {a, b});
          tempCircuit.safe_append_u("CX", {b, a});
          tempCircuit.safe_append_u("CX", {a, b});
        }
      }
    }
    frameRunner->apply(tempCircuit);
  }

  /// @brief Override: accumulate noise in frameRunner
  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::uint32_t> &qubits) override {
    CUDAQ_INFO("[custabilizer] apply kraus channel {}, is_msm_mode = {}",
               channel.get_type_name(), is_msm_mode);
    if (is_msm_mode)
      throw std::runtime_error(
          "MSM mode is not supported by custabilizer backend.");

    if (auto res = isValidStimNoiseChannel(channel)) {
      stim::Circuit noiseOps;
      noiseOps.safe_append_u(res.value().stim_name, qubits, channel.parameters);
      if (frameRunner)
        frameRunner->apply(noiseOps);
      msm_err_count += res->params.size();
    }
  }

  /// @brief Reset the actual qubits and GPU circuit between trajectories.
  void setToZeroState() override {
    flushPendingQubits();
    if (!tableau) {
      deallocateState();
      return;
    }
    recordedCircuit.clear();
    if (frameRunner)
      frameRunner->clear();
    const auto qubits =
        std::max(tableau->inv_state.num_qubits, nQubitsAllocated);
    if (qubits > 0) {
      std::vector<std::uint32_t> targets(qubits);
      std::iota(targets.begin(), targets.end(), 0);
      applyOpToSims("R", targets);
    }
    tableau->measurement_record.clear();
    frameSeed.reset();
    num_measurements = 0;
  }

  /// @brief Override: use frame runner for mid-circuit measurements
  bool measureQubit(const std::size_t index) override {
    auto *executionContext = getExecutionContext();
    applyOpToSims(
        "M", std::vector<std::uint32_t>{static_cast<std::uint32_t>(index)});
    num_measurements++;

    const std::vector<bool> &v = tableau->measurement_record.storage;
    const bool tableauBit = *v.crbegin();

    if (executionContext && executionContext->name == "sample" &&
        !executionContext->hasConditionalsOnMeasureResults) {
      return tableauBit;
    }

    if (!frameRunner)
      throw std::runtime_error("Frame runner not initialized");
    if (!frameSeed)
      frameSeed = randomEngine();
    const auto m_table = frameRunner->getMeasurementTable(
        /*num_shots=*/1, *frameSeed);
    const size_t words_per_row = 1;
    const size_t last_meas = num_measurements - 1;
    const bool sampleFlip = (m_table[last_meas * words_per_row] & 0x1u) != 0;
    return tableauBit ^ sampleFlip;
  }

  /// @brief Override: use frame runner for batch sampling
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    auto *executionContext = getExecutionContext();
    if (executionContext->explicitMeasurements && qubits.empty() &&
        num_measurements == 0)
      throw std::runtime_error(
          "The sampling option `explicit_measurements` is not supported on a "
          "kernel without any measurement operation.");

    bool populateResult = [&]() {
      if (executionContext->explicitMeasurements)
        return qubits.empty();
      return true;
    }();

    if (nQubitsAllocated == 0)
      throw std::runtime_error(
          "custabilizer backend state is not initialized.");

    const auto batch_size = static_cast<int64_t>(getBatchSize());
    if (shots > batch_size)
      throw std::runtime_error(
          "Requested shots exceed custabilizer batch size.");

    std::vector<std::uint32_t> stimTargetQubits(qubits.begin(), qubits.end());
    applyOpToSims("M", stimTargetQubits);
    num_measurements += stimTargetQubits.size();

    if (!populateResult)
      return cudaq::ExecutionResult();

    const std::vector<bool> &v = tableau->measurement_record.storage;
    if (v.size() != num_measurements)
      throw std::runtime_error(
          "Internal error: tableau measurement record size mismatch.");

    if (!frameRunner)
      throw std::runtime_error("Frame runner not initialized");
    if (!frameSeed)
      frameSeed = randomEngine();
    const auto m_table =
        frameRunner->getMeasurementTable(batch_size, *frameSeed);

    const size_t words_per_row = static_cast<size_t>((batch_size + 31) / 32);
    auto get_flip = [&](size_t shot, size_t meas) -> bool {
      const uint32_t word = m_table[meas * words_per_row + (shot >> 5)];
      return ((word >> (shot & 31)) & 1u) != 0;
    };

    const size_t bits_per_sample = num_measurements;
    assert(bits_per_sample >= qubits.size());
    const std::size_t first_bit_to_save = executionContext->explicitMeasurements
                                              ? 0
                                              : bits_per_sample - qubits.size();

    CountsDictionary counts;
    std::vector<std::string> sequentialData;
    if (includeSequentialData)
      sequentialData.reserve(shots);
    for (std::size_t shot = 0; shot < static_cast<std::size_t>(shots); shot++) {
      std::string aShot(bits_per_sample - first_bit_to_save, '0');
      for (std::size_t b = first_bit_to_save; b < bits_per_sample; b++) {
        const bool bit = get_flip(shot, b) ^ v[b];
        aShot[b - first_bit_to_save] = bit ? '1' : '0';
      }
      counts[aShot]++;
      if (includeSequentialData)
        sequentialData.push_back(std::move(aShot));
    }

    ExecutionResult result(counts);
    if (includeSequentialData)
      result.sequentialData = std::move(sequentialData);
    return result;
  }

public:
  CustabilizerCircuitSimulator() { summaryData.name = name(); }

  void setRandomSeed(std::size_t seed) override {
    StimCircuitSimulator::setRandomSeed(seed);
    frameSeed.reset();
  }

  std::string name() const override { return "custabilizer"; }
  NVQIR_SIMULATOR_CLONE_IMPL(CustabilizerCircuitSimulator)
};

} // namespace

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
NVQIR_REGISTER_SIMULATOR(nvqir::CustabilizerCircuitSimulator, custabilizer)
#endif
