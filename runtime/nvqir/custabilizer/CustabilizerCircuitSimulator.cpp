/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CustabilizerErrorHandling.h"
#include "../stim/StimCircuitSimulator.h"

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
    HANDLE_CUST_ERROR(custabilizerCircuitSizeFromString(handle, stim_circuit_str.c_str(),
                                                &buffer_size));
    HANDLE_CUDA_ERROR(cudaMalloc(&buffer_d, static_cast<size_t>(buffer_size)));
    HANDLE_CUST_ERROR(custabilizerCreateCircuitFromString(handle, stim_circuit_str.c_str(),
                                                  buffer_d, buffer_size,
                                                  &circuit));
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
  custabilizerHandle_t handle = nullptr;
  custabilizerFrameSimulator_t frame_simulator = nullptr;
  custabilizerBitInt_t *x_table_d = nullptr;
  custabilizerBitInt_t *z_table_d = nullptr;
  custabilizerBitInt_t *m_table_d = nullptr;
  int64_t num_qubits = 0;
  int64_t allocated_shots = 0;
  int64_t allocated_measurements = 0;
  int64_t stride_bytes = 0;
  stim::Circuit circuitCache;
  bool cache_executed = false;

  explicit CustabilizerFrameRunner(int64_t num_qubits)
      : num_qubits(num_qubits) {
    if (num_qubits <= 0)
      throw std::runtime_error("cuStabilizer frame simulator requires num_qubits > 0");
    HANDLE_CUST_ERROR(custabilizerCreate(&handle));
  }

  ~CustabilizerFrameRunner() {
    deallocate();
    if (handle)
      custabilizerDestroy(handle);
  }

  CustabilizerFrameRunner(const CustabilizerFrameRunner &) = delete;
  CustabilizerFrameRunner &operator=(const CustabilizerFrameRunner &) = delete;

  void apply(const stim::Circuit &circuit) {
    circuitCache += circuit;
    cache_executed = false;
  }

  void clear() {
    circuitCache.clear();
    cache_executed = false;
  }

  std::vector<uint32_t> getMeasurementTable(int64_t num_shots,
                                            bool randomize_after_measurement,
                                            uint64_t seed) {
    if (num_shots <= 0)
      throw std::runtime_error("num_shots must be > 0");
    int64_t measurement_count = circuitCache.count_measurements();

    if (num_shots > allocated_shots || measurement_count > allocated_measurements) {
      reallocate(num_shots, measurement_count);
    }

    if (!cache_executed) {
      CustabilizerCircuit circuit(circuitCache.str());
      HANDLE_CUST_ERROR(custabilizerFrameSimulatorApplyCircuit(
          handle, frame_simulator, circuit.circuit,
          randomize_after_measurement ? 1 : 0, seed, x_table_d, z_table_d,
          m_table_d, 0));
      cache_executed = true;
    }

    const size_t words_per_row = static_cast<size_t>((num_shots + 31) / 32);
    const size_t total_words = words_per_row * static_cast<size_t>(measurement_count);
    std::vector<uint32_t> out(total_words);
    const size_t total_bytes = total_words * sizeof(uint32_t);
    if (total_bytes > 0)
      HANDLE_CUDA_ERROR(cudaMemcpy(out.data(), m_table_d, total_bytes, cudaMemcpyDeviceToHost));
    return out;
  }

private:
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

  void reallocate(int64_t new_shots, int64_t new_measurements) {
    deallocate();

    allocated_shots = new_shots;
    allocated_measurements = new_measurements;
    stride_bytes = ((new_shots + 31) / 32) * static_cast<int64_t>(sizeof(uint32_t));

    const size_t xz_bytes =
        static_cast<size_t>(stride_bytes) * static_cast<size_t>(num_qubits);
    const size_t m_bytes =
        static_cast<size_t>(stride_bytes) * static_cast<size_t>(new_measurements);

    HANDLE_CUDA_ERROR(cudaMalloc(&x_table_d, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMalloc(&z_table_d, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMalloc(&m_table_d, m_bytes));
    HANDLE_CUDA_ERROR(cudaMemset(x_table_d, 0, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMemset(z_table_d, 0, xz_bytes));
    HANDLE_CUDA_ERROR(cudaMemset(m_table_d, 0, m_bytes));

    HANDLE_CUST_ERROR(custabilizerCreateFrameSimulator(handle, num_qubits, new_shots,
                                               new_measurements, stride_bytes,
                                               &frame_simulator));
  }
};

} // namespace

/// @brief The CustabilizerCircuitSimulator extends StimCircuitSimulator
/// to use cuStabilizer for GPU-accelerated frame simulation, while keeping
/// Stim's TableauSimulator for the noiseless reference.
class CustabilizerCircuitSimulator : public StimCircuitSimulator {
protected:
  /// @brief Persistent frame runner for circuit accumulation and execution
  std::unique_ptr<CustabilizerFrameRunner> frameRunner;

  /// @brief Override: reject MSM mode, initialize frame runner
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (stateDataIn)
      throw std::runtime_error(
          "The cuStabilizer backend does not support initialization of qubits from state data.");
    if (executionContext && executionContext->name == "msm")
      throw std::runtime_error("MSM mode is not supported by custabilizer backend.");
    StimCircuitSimulator::addQubitsToState(qubitCount, stateDataIn);
    if (nQubitsAllocated > 0 && !frameRunner) {
      frameRunner = std::make_unique<CustabilizerFrameRunner>(
          static_cast<int64_t>(nQubitsAllocated));
    }
  }

  /// @brief Override: also clear the frame runner
  void deallocateStateImpl() override {
    StimCircuitSimulator::deallocateStateImpl();
    if (frameRunner) {
      frameRunner->clear();
      frameRunner.reset();
    }
  }

  /// @brief Override: apply to tableau and accumulate in frameRunner
  void applyOpToSims(const std::string &gate_name,
                     const std::vector<uint32_t> &targets) override {
    if (targets.empty())
      return;
    stim::Circuit tempCircuit;
    CUDAQ_INFO("Calling applyOpToSims {} - {}", gate_name, targets);
    tempCircuit.safe_append_u(gate_name, targets);
    tableau->safe_do_circuit(tempCircuit);
    if (frameRunner)
      frameRunner->apply(tempCircuit);
  }

  /// @brief Override: accumulate noise in frameRunner
  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::uint32_t> &qubits) override {
    CUDAQ_INFO("[custabilizer] apply kraus channel {}, is_msm_mode = {}",
               channel.get_type_name(), is_msm_mode);
    if (is_msm_mode)
      throw std::runtime_error("MSM mode is not supported by custabilizer backend.");

    if (auto res = isValidStimNoiseChannel(channel)) {
      stim::Circuit noiseOps;
      noiseOps.safe_append_u(res.value().stim_name, qubits, channel.parameters);
      if (frameRunner)
        frameRunner->apply(noiseOps);
      msm_err_count += res->params.size();
    }
  }

  /// @brief Override: use frame runner for mid-circuit measurements
  bool measureQubit(const std::size_t index) override {
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
    const uint64_t seed = randomEngine();
    const auto m_table = frameRunner->getMeasurementTable(
        /*num_shots=*/1, /*randomize_after_measurement=*/true, seed);
    const size_t words_per_row = 1;
    const size_t last_meas = num_measurements - 1;
    const bool sampleFlip = (m_table[last_meas * words_per_row] & 0x1u) != 0;
    return tableauBit ^ sampleFlip;
  }

  /// @brief Override: use frame runner for batch sampling
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots) override {
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
      throw std::runtime_error("custabilizer backend state is not initialized.");

    const auto batch_size = static_cast<int64_t>(getBatchSize());
    if (shots > batch_size)
      throw std::runtime_error("Requested shots exceed custabilizer batch size.");

    std::vector<std::uint32_t> stimTargetQubits(qubits.begin(), qubits.end());
    applyOpToSims("M", stimTargetQubits);
    num_measurements += stimTargetQubits.size();

    if (!populateResult)
      return cudaq::ExecutionResult();

    const std::vector<bool> &v = tableau->measurement_record.storage;
    if (v.size() != num_measurements)
      throw std::runtime_error("Internal error: tableau measurement record size mismatch.");

    if (!frameRunner)
      throw std::runtime_error("Frame runner not initialized");
    const uint64_t seed = randomEngine();
    const auto m_table = frameRunner->getMeasurementTable(
        batch_size, /*randomize_after_measurement=*/true, seed);

    const size_t words_per_row = static_cast<size_t>((batch_size + 31) / 32);
    auto get_flip = [&](size_t shot, size_t meas) -> bool {
      const uint32_t word = m_table[meas * words_per_row + (shot >> 5)];
      return ((word >> (shot & 31)) & 1u) != 0;
    };

    const size_t bits_per_sample = num_measurements;
    assert(bits_per_sample >= qubits.size());
    const std::size_t first_bit_to_save =
        executionContext->explicitMeasurements ? 0 : bits_per_sample - qubits.size();

    CountsDictionary counts;
    std::vector<std::string> sequentialData;
    sequentialData.reserve(shots);
    for (std::size_t shot = 0; shot < static_cast<std::size_t>(shots); shot++) {
      std::string aShot(bits_per_sample - first_bit_to_save, '0');
      for (std::size_t b = first_bit_to_save; b < bits_per_sample; b++) {
        const bool bit = get_flip(shot, b) ^ v[b];
        aShot[b - first_bit_to_save] = bit ? '1' : '0';
      }
      counts[aShot]++;
      sequentialData.push_back(std::move(aShot));
    }

    ExecutionResult result(counts);
    result.sequentialData = std::move(sequentialData);
    return result;
  }

public:
  std::string name() const override { return "custabilizer"; }
  NVQIR_SIMULATOR_CLONE_IMPL(CustabilizerCircuitSimulator)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
NVQIR_REGISTER_SIMULATOR(nvqir::CustabilizerCircuitSimulator, custabilizer)
#endif
