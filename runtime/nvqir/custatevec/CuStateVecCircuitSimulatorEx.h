/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CircuitSimulator.h"
#include "CuStateVecBatch.h"
#include "CuStateVecConfig.h"
#include "CuStateVecDevice.h"
#include "CuStateVecError.h"
#include "CuStateVecGateEngine.h"
#include "CuStateVecRandom.h"
#include "CuStateVecSimulationState.h"
#include "CuStateVecSimulatorDetail.h"
#include "Gates.h"
#include "cudaq/ptsbe/PTSBESamplerImpl.h"

#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <complex>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace cudaq::cusv {

/// @brief Single-process `cuStateVecEx` circuit simulator.
template <typename Scalar>
class CuStateVecCircuitSimulator : public nvqir::CircuitSimulatorBase<Scalar>,
                                   public cudaq::ptsbe::BatchSimulator {
protected:
  using Base = nvqir::CircuitSimulatorBase<Scalar>;
  using GateApplicationTask = typename Base::GateApplicationTask;

  using Base::flushGateQueue;
  using Base::nQubitsAllocated;
  using Base::previousStateDimension;
  using Base::shouldObserveFromSampling;
  using Base::stateDimension;
  using Base::summaryData;

public:
  CuStateVecCircuitSimulator()
      : m_config(detail::deviceConfig<Scalar>()),
        m_engine(createGateEngine<Scalar>(m_config)),
        m_randomEngine(m_randomDevice()) {
    HANDLE_CUDA_ERROR(cudaFree(nullptr));
    if (m_config.enableMemPool) {
      int32_t device = 0;
      int32_t supported = 0;
      HANDLE_CUDA_ERROR(cudaGetDevice(&device));
      HANDLE_CUDA_ERROR(cudaDeviceGetAttribute(
          &supported, cudaDevAttrMemoryPoolsSupported, device));
      if (supported) {
        cudaMemPool_t pool = nullptr;
        HANDLE_CUDA_ERROR(cudaDeviceGetDefaultMemPool(&pool, device));
        std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
        HANDLE_CUDA_ERROR(cudaMemPoolSetAttribute(
            pool, cudaMemPoolAttrReleaseThreshold, &threshold));
      }
    }
    summaryData.name = name();
  }

  std::string name() const override {
    if constexpr (std::is_same_v<Scalar, float>)
      return "custatevec-fp32";
    return "custatevec-fp64";
  }

  NVQIR_SIMULATOR_CLONE_IMPL(CuStateVecCircuitSimulator<Scalar>)

  bool isStateVectorSimulator() const override { return true; }

  /// Run PTSBE trajectories in batches when they have compatible gate
  /// structure and fit on one device; otherwise use the generic sequential
  /// implementation, which also supports migrated and distributed states.
  std::vector<cudaq::sample_result>
  sampleWithPTSBE(const cudaq::ptsbe::PTSBatch &batch) override {
    if (batch.trajectories.empty())
      return {};
    const auto plan = analyzePTSBEBatch(batch);
    if (!plan)
      return cudaq::ptsbe::detail::samplePTSBEGeneric<Scalar>(*this, batch);
    // A plan with no active trajectories means every trajectory requested zero
    // shots; the result is one empty entry per trajectory.
    if (plan->active.empty())
      return std::vector<cudaq::sample_result>(batch.trajectories.size());
    return runPTSBEBatch(batch, *plan);
  }

  void setRandomSeed(std::size_t seed) override {
    m_randomEngine.seed(seed);
    m_gpuRandom.setSeed(seed);
  }

  void synchronize() override {
    // A noisy circuit must be replayed once per trajectory, so leave its queued
    // gates and channels deferred until the trajectory loop executes them.
    if (isNoisySimulation())
      return;
    if (!m_state)
      return;
    m_engine->flush(*m_state, m_pendingRandomNumbers);
    m_pendingRandomNumbers.clear();
    m_state->synchronize();
  }

protected:
  void addQubitToState() override { addQubitsToState(1, nullptr); }

  void addQubitsToState(std::size_t count, const void *stateData) override {
    if (count == 0)
      return;
    const bool firstAllocation = nQubitsAllocated == count;
    if (firstAllocation)
      m_measurementRecorded = false;
    if (stateData)
      m_hasCustomInitialState = true;
    else if (firstAllocation)
      m_hasCustomInitialState = false;
    if (firstAllocation && m_state) {
      if (m_state->numWires() == static_cast<int32_t>(count)) {
        // Reuse the same-size allocation (config can force realloc instead). A
        // prior run may leave a non-identity wire ordering -- from
        // host-migration bits here, or distribution bits in the MGPU subclass.
        // Since we zero the state next, just relabel the ordering to identity
        // (metadata-only) rather than physically permuting the discarded
        // amplitudes; |0..0> is permutation-invariant.
        m_deferredTasks.clear();
        m_pendingRandomNumbers.clear();
        m_engine = createGateEngine<Scalar>(m_config);
        std::vector<int32_t> identity(count);
        for (std::size_t index = 0; index < count; ++index)
          identity[index] = static_cast<int32_t>(index);
        m_state->reassignWireOrdering(identity);
        m_state->setZeroState();
        if (stateData) {
          const std::size_t size = std::size_t{1} << count;
          if (!m_state->setStateFromDevicePointer(stateData, size)) {
            writeState(static_cast<const std::complex<Scalar> *>(stateData),
                       size);
          }
        }
        return;
      }
      m_state.reset();
      m_engine = createGateEngine<Scalar>(m_config);
    }
    ensureState();
    synchronize();

    const std::size_t oldWires = m_state->numWires();
    const std::size_t addedSize = std::size_t{1} << count;
    // Fast path: append `state (x) stateData` directly on device via a single
    // cuBLAS rank-1 outer product. Only works when the resized state stays
    // single-device.
    if (stateData && oldWires != 0 &&
        m_state->appendState(stateData, addedSize))
      return;

    // Direct Kronecker not possible (migrated/distributed): grow the register
    // with zeroed wires, then initialize them via the gate-based extension.
    addZeroWires(count);
    if (!stateData)
      return;

    if (oldWires == 0) {
      if (!m_state->setStateFromDevicePointer(stateData, addedSize)) {
        writeState(static_cast<const std::complex<Scalar> *>(stateData),
                   addedSize);
      }
      return;
    }

    const auto addedState =
        detail::copyPointerToHost<Scalar>(stateData, addedSize);
    // The new wires are initially |0...0>. A matrix whose first column is the
    // requested state therefore performs the required Kronecker extension
    // without copying the existing state vector through host memory.
    MatrixTask<Scalar> initialization;
    initialization.matrix.assign(addedSize * addedSize, std::complex<Scalar>{});
    for (std::size_t row = 0; row < addedSize; ++row)
      initialization.matrix[row * addedSize] = addedState[row];
    initialization.targets.reserve(count);
    for (std::size_t wire = 0; wire < count; ++wire)
      initialization.targets.push_back(static_cast<int32_t>(oldWires + wire));
    DirectGateEngine<Scalar> direct;
    direct.apply(*m_state, initialization);
    m_state->synchronize();
  }

  void addQubitsToState(const cudaq::SimulationState &input) override {
    if (input.getPrecision() != simulationPrecision())
      throw std::invalid_argument("Initial-state precision mismatch.");
    if (const auto *const exState =
            dynamic_cast<const CuStateVecSimulationState<Scalar> *>(&input)) {
      const bool firstAllocation = nQubitsAllocated == input.getNumQubits();
      if (!m_state || firstAllocation) {
        addQubitsToState(input.getNumQubits(), nullptr);
        m_hasCustomInitialState = true;
        synchronize();
        // Direct state-to-state copy: copyFrom transfers each sub-state slice
        // in place -- device-to-device, and host-to-host for migrated slices --
        // with no host materialization. It throws if the source layout is
        // incompatible (runs under the same configuration always match).
        m_state->copyFrom(exState->state());
        return;
      }
      if (exState->state().distributionType() ==
              CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE &&
          exState->state().numMigrationWires() == 0) {
        const auto tensor = exState->getTensor();
        addQubitsToState(input.getNumQubits(), tensor.data);
        return;
      }
      // Appending (Kronecker) a migrated/distributed source onto an existing
      // register
      std::vector<std::complex<Scalar>> host(std::size_t{1}
                                             << input.getNumQubits());
      input.toHost(host.data(), host.size());
      addQubitsToState(input.getNumQubits(), host.data());
      return;
    }
    // Only a custatevec simulation state can initialize this backend; importing
    // another simulator's state is not supported (they are not layout- or
    // representation-compatible).
    throw std::invalid_argument(
        "Incompatible initial state: expected a custatevec simulation state.");
  }

  void deallocateStateImpl() override {
    if (m_config.forceAllocateState)
      m_state.reset();
    m_deferredTasks.clear();
    m_measurementRecorded = false;
    m_pendingRandomNumbers.clear();
    m_engine = createGateEngine<Scalar>(m_config);
  }

  void applyGate(const GateApplicationTask &task) override {
    ensureState();
    rejectOperationAfterMeasurement();
    if (isNoisySimulation()) {
      // Defer gates together with their following noise channels so the entire
      // circuit can be replayed independently for every trajectory.
      MatrixTask<Scalar> operation;
      operation.matrix = task.matrix;
      operation.targets = detail::toInt32(task.targets);
      operation.controls = detail::toInt32(task.controls);
      compactMatrixTask(operation);
      m_deferredTasks.emplace_back(std::move(operation));
      return;
    }

    MatrixTask<Scalar> operation;
    operation.matrix = task.matrix;
    operation.targets = detail::toInt32(task.targets);
    operation.controls = detail::toInt32(task.controls);
    compactMatrixTask(operation);
    m_engine->apply(*m_state, operation);
  }

  void setToZeroState() override {
    ensureState();
    synchronize();
    m_state->setZeroState();
    m_deferredTasks.clear();
    m_measurementRecorded = false;
  }

  bool mz(std::size_t qubit) override { return mz(qubit, ""); }

  bool mz(std::size_t qubit, const std::string &registerName) override {
    const auto *const context = cudaq::getExecutionContext();
    if (!isNoisySimulation() || !context || context->name != "sample" ||
        context->hasConditionalsOnMeasureResults)
      return Base::mz(qubit, registerName);

    flushGateQueue();
    if (this->getNoiseModel() && !this->getNoiseModel()->empty())
      applyNoiseChannel("mz", {}, {qubit}, {});

    if (context->explicitMeasurements) {
      const bool result = measureQubit(qubit);
      cudaq::ExecutionResult measurement;
      measurement.appendResult(result ? "1" : "0", 1);
      this->internalResult.append(measurement, true);
      return result;
    }

    // Non-explicit `sample` supports only terminal measurements. Flag that a
    // measurement has been recorded (so any later quantum operation is rejected
    // as an unsupported mid-circuit measurement) and register the wire for
    // terminal sampling; the bit is recovered from terminal sampling of the
    // final state, so no collapse is applied here.
    m_measurementRecorded = true;
    recordMeasuredRegister(qubit, cudaq::GlobalRegisterName);
    if (!registerName.empty())
      recordMeasuredRegister(qubit, registerName);
    return false;
  }

  /// Register `qubit` under measurement register `name` so terminal sampling
  /// reconstructs that register's bit-string.
  void recordMeasuredRegister(std::size_t qubit, const std::string &name) {
    auto &qubits = this->registerNameToMeasuredQubit[name];
    if (std::find(qubits.begin(), qubits.end(), qubit) == qubits.end())
      qubits.push_back(qubit);
  }

  /// Non-explicit `sample` supports only terminal measurements, so once a
  /// measurement has been recorded no further quantum operation may run. Throw
  /// otherwise -- that would be an unsupported mid-circuit measurement.
  void rejectOperationAfterMeasurement() const {
    if (m_measurementRecorded)
      throw std::runtime_error(
          "Mid-circuit measurement is not supported in `sample` mode: a "
          "measurement must be the final operation on the qubits. Use "
          "`explicit_measurements=true` or `cudaq::run` instead.");
  }

  bool measureQubit(std::size_t qubit) override {
    replayDeferredFromCurrentState();
    synchronize();
    const int32_t wire = static_cast<int32_t>(qubit);
    custatevecIndex_t result = 0;
    HANDLE_CUSTATEVEC_ERROR(custatevecExMeasure(
        /*stateVector=*/state().descriptor(), /*bitString=*/&result,
        /*bitStringOrdering=*/&wire, /*bitStringOrderingLen=*/1,
        /*randnum=*/randomNumber(),
        /*collapse=*/CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO,
        /*reserved=*/nullptr));
    return result != 0;
  }

  void resetQubit(std::size_t qubit) override {
    flushGateQueue();
    this->flushAnySamplingTasks();
    rejectOperationAfterMeasurement();
    if (isNoisySimulation()) {
      // Reset is the channel K0 = |0><0|, K1 = |0><1|. Defer it with the
      // surrounding noisy circuit so every trajectory replays the prefix and
      // reset independently.
      NoiseTask<Scalar> reset;
      reset.matrices = {
          {1.0, 0.0, 0.0, 0.0},
          {0.0, 1.0, 0.0, 0.0},
      };
      reset.wires = {static_cast<int32_t>(qubit)};
      compactNoiseMatrices(reset);
      m_deferredTasks.emplace_back(std::move(reset));
      return;
    }
    replayDeferredFromCurrentState();
    synchronize();
    const int32_t wire = static_cast<int32_t>(qubit);
    custatevecIndex_t result = 0;
    HANDLE_CUSTATEVEC_ERROR(custatevecExMeasure(
        /*stateVector=*/state().descriptor(), /*bitString=*/&result,
        /*bitStringOrdering=*/&wire, /*bitStringOrderingLen=*/1,
        /*randnum=*/randomNumber(), /*collapse=*/CUSTATEVEC_COLLAPSE_RESET,
        /*reserved=*/nullptr));
  }

  void applyExpPauli(double theta, const std::vector<std::size_t> &controls,
                     const std::vector<std::size_t> &qubits,
                     const cudaq::spin_op_term &term) override {
    if (cudaq::isInTracerMode()) {
      nvqir::CircuitSimulator::applyExpPauli(theta, controls, qubits, term);
      return;
    }
    if (isNoisySimulation() || m_config.forceExpPauliDecomposition) {
      // Noise models are specified on individual gates, so noisy exp-Pauli
      // operations must use the decomposed circuit form.
      Base::applyExpPauli(theta, controls, qubits, term);
      return;
    }
    flushGateQueue();
    if (term.num_ops() != qubits.size())
      throw std::runtime_error("Incorrect exp-Pauli qubit count.");

    PauliRotationTask operation;
    operation.angle = theta;
    operation.targets = detail::toInt32(qubits);
    operation.controls = detail::toInt32(controls);
    for (const auto &termOp : term)
      operation.paulis.push_back(detail::toCuStateVecPauli(termOp.as_pauli()));
    m_engine->apply(state(), operation);
  }

  bool canHandleObserve() override {
    // Shots-based observe must use term-by-term sampling. Without shots, keep
    // observe here so all Pauli terms are evaluated in one cuStateVecEx call.
    const auto *const context = cudaq::getExecutionContext();
    if (context && context->shots > 0 &&
        context->shots != static_cast<std::size_t>(-1))
      return false;
    return !shouldObserveFromSampling(false);
  }

  // Shot-based term-by-term observe must simulate each noisy term from the
  // base ansatz independently. Reusing a post-measurement state would correlate
  // trajectories, while reversing basis changes could introduce extra noise.
  cudaq::SpinMeasureResult measureSpinOp(const cudaq::spin_op &op) override {
    if (!isNoisySimulation())
      return Base::measureSpinOp(op);

    flushGateQueue();
    auto *const context = cudaq::getExecutionContext();
    if (context->canHandleObserve) {
      auto result = observe(context->spin.value());
      return cudaq::SpinMeasureResult(result.expectation(), result.raw_data());
    }
    if (op.num_terms() != 1)
      throw std::invalid_argument(
          "Shot-based measurement requires one spin term.");

    // Preserve the base ansatz queue while basis-change gates are appended for
    // this term, then restore it for the next term.
    auto circuitTasks = m_deferredTasks;
    std::vector<std::size_t> measuredWires;
    std::vector<std::function<void()>> basisChanges;
    for (const auto &termOp : *op.begin()) {
      const auto pauli = termOp.as_pauli();
      const std::size_t target = termOp.target();
      if (pauli != cudaq::pauli::I)
        measuredWires.push_back(target);
      if (pauli == cudaq::pauli::Y)
        basisChanges.emplace_back([this, target] { this->rx(M_PI_2, target); });
      else if (pauli == cudaq::pauli::X)
        basisChanges.emplace_back([this, target] { this->h(target); });
    }
    for (auto &basisChange : basisChanges)
      basisChange();
    flushGateQueue();

    const int shots = context->shots > 0 ? context->shots : 0;
    try {
      auto result = sample(measuredWires, shots);
      m_deferredTasks = std::move(circuitTasks);
      return cudaq::SpinMeasureResult(result.expectationValue.value_or(0.0),
                                      cudaq::sample_result(std::move(result)));
    } catch (...) {
      m_deferredTasks = std::move(circuitTasks);
      throw;
    }
  }

  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    flushGateQueue();
    if (isNoisySimulation())
      return observeTrajectories(op);

    synchronize();
    if (op.num_terms() == 0)
      return {};

    // cuStateVecEx evaluates every term in one call from parallel arrays that
    // point into the stable per-term Pauli and wire storage owned by `terms`.
    const auto terms = detail::extractPauliTerms(op);
    std::vector<const custatevecPauli_t *> pauliArrays;
    std::vector<const int32_t *> wireArrays;
    std::vector<int32_t> wireCounts;
    pauliArrays.reserve(op.num_terms());
    wireArrays.reserve(op.num_terms());
    wireCounts.reserve(op.num_terms());
    for (std::size_t term = 0; term < op.num_terms(); ++term) {
      pauliArrays.push_back(dataOrNull(terms.paulis[term]));
      wireArrays.push_back(dataOrNull(terms.targets[term]));
      wireCounts.push_back(static_cast<int32_t>(terms.targets[term].size()));
    }

    std::vector<double> expectations(op.num_terms());
    HANDLE_CUSTATEVEC_ERROR(custatevecExComputeExpectationOnPauliBasis(
        /*stateVector=*/state().descriptor(),
        /*expectationValues=*/expectations.data(),
        /*pauliOperatorArrays=*/pauliArrays.data(),
        /*numPauliOperatorArrays=*/static_cast<int32_t>(pauliArrays.size()),
        /*basisWiresArray=*/wireArrays.data(),
        /*numBasisWiresArray=*/wireCounts.data()));
    double expectation = 0.0;
    std::vector<cudaq::ExecutionResult> termResults;
    termResults.reserve(op.num_terms());
    std::size_t index = 0;
    for (const auto &term : op) {
      expectation += (term.evaluate_coefficient() * expectations[index]).real();
      termResults.emplace_back(
          cudaq::ExecutionResult({}, term.get_term_id(), expectations[index]));
      ++index;
    }
    return cudaq::observe_result(
        expectation, op,
        cudaq::sample_result(expectation, std::move(termResults)));
  }

  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                int shots,
                                bool includeSequentialData = true) override {
    flushGateQueue();
    if (isNoisySimulation())
      return sampleTrajectories(measuredBits, shots, includeSequentialData);

    synchronize();
    const auto wires = detail::toInt32(measuredBits);
    if (shots < 1) {
      std::vector<custatevecPauli_t> paulis(wires.size(), CUSTATEVEC_PAULI_Z);
      const custatevecPauli_t *pauliArrays[] = {paulis.data()};
      const int32_t *wireArrays[] = {wires.data()};
      const int32_t wireCounts[] = {static_cast<int32_t>(wires.size())};
      double expectation = 0.0;
      HANDLE_CUSTATEVEC_ERROR(custatevecExComputeExpectationOnPauliBasis(
          /*stateVector=*/state().descriptor(),
          /*expectationValues=*/&expectation,
          /*pauliOperatorArrays=*/pauliArrays, /*numPauliOperatorArrays=*/1,
          /*basisWiresArray=*/wireArrays, /*numBasisWiresArray=*/wireCounts));
      return cudaq::ExecutionResult{expectation};
    }

    auto randomNumbers = generateRandomNumbers(shots);
    std::vector<custatevecIndex_t> bitStrings(shots);
    HANDLE_CUSTATEVEC_ERROR(custatevecExSample(
        /*stateVector=*/state().descriptor(), /*bitStrings=*/bitStrings.data(),
        /*bitStringOrdering=*/wires.data(),
        /*bitStringOrderingLen=*/static_cast<int32_t>(wires.size()),
        /*randnums=*/randomNumbers.data(), /*numShots=*/shots,
        /*output=*/CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER,
        /*abs2Sums=*/nullptr));

    cudaq::ExecutionResult result;
    // cuStateVecEx returns ascending bit strings. Run-length encoding avoids a
    // string conversion for every shot and instead performs one per unique
    // sampled bit string.
    for (int index = 0; index < shots;) {
      const custatevecIndex_t value = bitStrings[index];
      int runLength = 1;
      while (index + runLength < shots &&
             bitStrings[index + runLength] == value)
        ++runLength;
      std::string bitString = formatBitString(value, measuredBits.size());
      if (includeSequentialData)
        result.appendResult(bitString, runLength);
      else
        result.counts[bitString] += runLength;
      index += runLength;
    }
    result.expectationValue = detail::parityExpectation(result, shots);
    return result;
  }

  bool isValidNoiseChannel(const cudaq::noise_model_type &) const override {
    // Support all noise channels
    return true;
  }

  void applyNoise(const cudaq::kraus_channel &channel,
                  const std::vector<std::size_t> &qubits) override {
    flushGateQueue();
    applyNoiseTask(channel, qubits);
  }

  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &controls,
                         const std::vector<std::size_t> &targets,
                         const std::vector<double> &parameters) override {
    auto *const context = cudaq::getExecutionContext();
    if (!context || !context->noiseModel)
      return;
    const auto channels = context->noiseModel->get_channels(
        std::string(gateName), targets, controls, parameters);
    if (channels.empty())
      return;
    std::vector<std::size_t> wires(controls);
    wires.insert(wires.end(), targets.begin(), targets.end());
    for (const auto &channel : channels)
      applyNoiseTask(channel, wires);
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    if (isNoisySimulation())
      replayDeferredFromCurrentState();
    synchronize();
    if (!m_state)
      throw std::runtime_error("No cuStateVec state is allocated.");
    auto result = std::make_unique<CuStateVecSimulationState<Scalar>>(
        std::move(*m_state));
    m_state.reset();
    return result;
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    std::size_t size = 0;
    const std::complex<Scalar> *pointer = nullptr;
    if constexpr (std::is_same_v<Scalar, float>) {
      if (const auto *values =
              std::get_if<std::vector<std::complex<float>>>(&data)) {
        size = values->size();
        pointer = values->data();
      } else if (const auto *values =
                     std::get_if<std::pair<std::complex<float> *, std::size_t>>(
                         &data)) {
        size = values->second;
        pointer = values->first;
      } else {
        throw std::runtime_error("[sim-state] invalid data precision.");
      }
    } else {
      if (const auto *values =
              std::get_if<std::vector<std::complex<double>>>(&data)) {
        size = values->size();
        pointer = values->data();
      } else if (const auto *values = std::get_if<
                     std::pair<std::complex<double> *, std::size_t>>(&data)) {
        size = values->second;
        pointer = values->first;
      } else {
        throw std::runtime_error("[sim-state] invalid data precision.");
      }
    }
    if (!pointer || size == 0 || (size & (size - 1)) != 0)
      throw std::invalid_argument("Invalid initial state-vector data.");
    return CuStateVecSimulationState<Scalar>::create(
        size, pointer, m_config.allowFp32Emulation);
  }

protected:
  virtual bool supportsBatchedTrajectories() const { return true; }

  cudaq::SimulationState::precision simulationPrecision() const {
    if constexpr (std::is_same_v<Scalar, float>)
      return cudaq::SimulationState::precision::fp32;
    return cudaq::SimulationState::precision::fp64;
  }

  double randomNumber() {
    return std::generate_canonical<double, std::numeric_limits<double>::digits>(
        m_randomEngine);
  }

  /// Draw `count` CPU pseudo-random numbers from the trajectory engine.
  /// Trajectory branch selection must stay on this seeded CPU engine so a
  /// replay is reproducible; it deliberately bypasses the GPU RNG path used for
  /// terminal sampling.
  std::vector<double> randomVector(std::size_t count) {
    std::vector<double> values(count);
    std::generate(values.begin(), values.end(),
                  [this] { return randomNumber(); });
    return values;
  }

  /// Generate probabilities on the GPU above the configured threshold;
  /// small requests stay on the CPU to avoid GPU setup overhead. Sampling
  /// output ordering is delegated to cuStateVec.
  virtual std::vector<double> generateRandomNumbers(std::size_t count) {
    if (count >= m_config.gpuRngThreshold) {
      try {
        return m_gpuRandom.generate(count);
      } catch (const std::exception &error) {
        CUDAQ_WARN("GPU random-number generation failed for {} values ({}); "
                   "falling back to CPU generation.",
                   count, error.what());
      }
    }
    return randomVector(count);
  }

  std::size_t availableGpuBytes() const {
    return m_config.maxGpuMemoryGb
               ? static_cast<std::size_t>(*m_config.maxGpuMemoryGb) << 30
               : queryDeviceMemoryCapacity().usableBytes();
  }

  int32_t maximumDeviceWires() const {
    const std::size_t availableBytes = availableGpuBytes();
    const std::size_t amplitudes =
        availableBytes / sizeof(std::complex<Scalar>);
    if (amplitudes == 0)
      throw std::runtime_error("Insufficient GPU memory for a state vector.");
    return static_cast<int32_t>(std::bit_width(amplitudes) - 1);
  }

  int32_t maximumMigrationWires(int32_t maxDeviceWires) const {
    if (m_config.maxCpuMemoryGb == 0)
      return 0;

    // cuStateVecEx host-migration buffer is pageable, so it can
    // reclaim clean page cache; MemAvailable is the usable ceiling. Keep a
    // margin (90%) for safety.
    const std::optional<std::size_t> memAvailable = systemMemAvailableBytes();
    const std::size_t availableBytes =
        memAvailable ? (*memAvailable / 10) * 9 : 0;
    const std::size_t hostBytes = [&] {
      if (!m_config.maxCpuMemoryGb)
        return availableBytes;
      const std::size_t requested =
          static_cast<std::size_t>(*m_config.maxCpuMemoryGb) << 30;
      return availableBytes == 0 ? requested
                                 : std::min(requested, availableBytes);
    }();
    if (hostBytes == 0)
      return 0;

    const std::size_t deviceBytes =
        (std::size_t{1} << maxDeviceWires) * sizeof(std::complex<Scalar>);
    // Each host-memory block equal to device capacity adds one migration wire.
    return migrationWireCapacity(hostBytes, deviceBytes);
  }

  void normalizeStateWireOrdering() { state().normalizeWireOrdering(); }
  virtual std::vector<std::complex<Scalar>> readState() {
    normalizeStateWireOrdering();
    std::vector<std::complex<Scalar>> result(std::size_t{1}
                                             << state().numWires());
    state().getState(result.data(), 0, result.size());
    state().synchronize();
    return result;
  }

  virtual void writeState(const std::complex<Scalar> *values,
                          std::size_t size) {
    normalizeStateWireOrdering();
    if (!values || size != (std::size_t{1} << state().numWires()))
      throw std::invalid_argument("Invalid full state-vector size.");
    state().setState(values, 0, size);
    state().synchronize();
  }

  virtual void writeState(const std::vector<std::complex<Scalar>> &values) {
    writeState(values.data(), values.size());
  }

  virtual void addZeroWires(std::size_t count) {
    const int32_t localHeadroom =
        state().maxLocalWires() - state().numLocalWires();
    const int32_t localCount =
        std::min(static_cast<int32_t>(count), localHeadroom);
    state().addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_LOCAL, localCount);

    const int32_t migrationCount = static_cast<int32_t>(count) - localCount;
    if (migrationCount >
        state().maxMigrationWires() - state().numMigrationWires())
      throw std::runtime_error(
          "State vector exceeds configured GPU and CPU memory.");
    state().addWires(CUSTATEVEC_EX_INDEX_BIT_DOMAIN_MIGRATION, migrationCount);
  }

  virtual void ensureState() {
    if (m_state)
      return;
    int32_t device = 0;
    HANDLE_CUDA_ERROR(cudaGetDevice(&device));
    const int32_t maxDeviceWires = maximumDeviceWires();
    const int32_t maxMigrationWires = maximumMigrationWires(maxDeviceWires);
    if (nQubitsAllocated >
        static_cast<std::size_t>(maxDeviceWires + maxMigrationWires))
      throw std::runtime_error(
          "State vector exceeds configured GPU and CPU memory.");
    m_state.emplace(CuStateVecState<Scalar>::createSingleDevice(
        maxDeviceWires + maxMigrationWires, maxDeviceWires, device,
        m_config.allowFp32Emulation));
  }

  CuStateVecState<Scalar> &state() {
    ensureState();
    return *m_state;
  }

  NoiseTask<Scalar> makeNoiseTask(const cudaq::kraus_channel &channel,
                                  const std::vector<std::size_t> &wires) {
    NoiseTask<Scalar> task;
    task.wires = detail::toInt32(wires);
    if (channel.is_unitary_mixture()) {
      task.kind = NoiseChannelKind::MixedUnitary;
      task.probabilities = channel.probabilities;
      for (const auto &matrix : channel.unitary_ops) {
        auto &converted = task.matrices.emplace_back();
        converted.reserve(matrix.size());
        converted.assign(matrix.begin(), matrix.end());
      }
    } else {
      task.kind = NoiseChannelKind::General;
      for (const auto &operation : channel.get_ops()) {
        auto &converted = task.matrices.emplace_back();
        // Python-defined channels may carry a precision different from the
        // simulator, so inspect each operation before converting its matrix.
        converted.reserve(operation.data.size());
        if (operation.precision == cudaq::simulation_precision::fp32) {
          const auto *const values =
              reinterpret_cast<const std::complex<float> *>(
                  operation.data.data());
          converted.assign(values, values + operation.data.size());
        } else {
          const auto *const values =
              reinterpret_cast<const std::complex<double> *>(
                  operation.data.data());
          converted.assign(values, values + operation.data.size());
        }
      }
    }
    compactNoiseMatrices(task);
    return task;
  }

  void applyNoiseTask(const cudaq::kraus_channel &channel,
                      const std::vector<std::size_t> &wires) {
    auto task = makeNoiseTask(channel, wires);
    if (isNoisySimulation()) {
      m_deferredTasks.emplace_back(std::move(task));
      return;
    }
    if (m_config.gateMode == GateExecutionMode::Direct) {
      const double random = randomNumber();
      m_engine->apply(state(), task, std::span<const double>(&random, 1));
      return;
    }
    m_pendingRandomNumbers.push_back(randomNumber());
    m_engine->apply(state(), task);
  }

  void logTrajectorySelection(const NoiseTask<Scalar> &task,
                              double random) const {
    if (!m_config.logTrajectorySampling)
      return;
    if (task.kind == NoiseChannelKind::General) {
      cudaq::log("Use random number {} for general channel", random);
      return;
    }

    double cumulative = 0.0;
    std::size_t selected = task.probabilities.size() - 1;
    for (std::size_t branch = 0; branch < task.probabilities.size(); ++branch) {
      cumulative += task.probabilities[branch];
      if (random < cumulative) {
        selected = branch;
        break;
      }
    }
    cudaq::log("Randomly pick channel {} in unitary channel", selected);
  }

  bool isNoisySimulation() const {
    if (m_replayingTrajectory)
      return false;
    const auto *const context = cudaq::getExecutionContext();
    // A one-shot call has no execution context but may still invoke explicit
    // noise operations, so conservatively use trajectory semantics.
    return !context || context->noiseModel;
  }

  void executeDeferredTasks(bool clearTasks) {
    // Each replay gets a fresh engine because a fused updater owns per-circuit
    // queue state. Random numbers are accumulated in the same order as the
    // queued noise tasks and consumed when the updater is flushed.
    auto replayEngine = createGateEngine<Scalar>(m_config);
    std::vector<double> randomNumbers;
    for (const auto &task : m_deferredTasks) {
      const auto *noise = std::get_if<NoiseTask<Scalar>>(&task);
      if (!noise) {
        replayEngine->apply(state(), task, randomNumbers);
        continue;
      }

      const double random = randomNumber();
      logTrajectorySelection(*noise, random);
      if (m_config.gateMode == GateExecutionMode::Direct) {
        replayEngine->apply(state(), task, std::span<const double>(&random, 1));
      } else {
        randomNumbers.push_back(random);
        replayEngine->apply(state(), task, randomNumbers);
      }
    }
    replayEngine->flush(state(), randomNumbers);
    state().synchronize();
    if (clearTasks)
      m_deferredTasks.clear();
  }

  void runDeferredTrajectory(const CuStateVecState<Scalar> *initialState,
                             bool clearTasks) {
    if (initialState)
      m_state->copyFrom(*initialState);
    else
      m_state->setZeroState();
    executeDeferredTasks(clearTasks);
  }

  void restoreTrajectoryState(const CuStateVecState<Scalar> *initialState) {
    if (initialState)
      m_state->copyFrom(*initialState);
    else
      m_state->setZeroState();
    m_state->synchronize();
  }

  void replayDeferredFromCurrentState() {
    if (!m_deferredTasks.empty())
      executeDeferredTasks(true);
  }

  std::size_t selectedTrajectoryBatchSize(std::size_t workItems,
                                          bool isObserve = false) const {
    // Batched observe needs an equally sized state copy to compute inner
    // products without modifying the original trajectory states.
    const std::size_t stateBytes = (std::size_t{1} << m_state->numWires()) *
                                   sizeof(std::complex<Scalar>) *
                                   (isObserve ? 2 : 1);
    const std::size_t availableBytes = availableGpuBytes();
    return trajectoryBatchSize(availableBytes, stateBytes, workItems,
                               m_config.trajectoryBatchSize);
  }

  bool canBatchDeferredTrajectories(std::size_t workItems) const {
    const auto *const context = cudaq::getExecutionContext();
    if ((context && context->hasConditionalsOnMeasureResults) ||
        !supportsBatchedTrajectories() ||
        selectedTrajectoryBatchSize(workItems) < m_config.batchedMinBatchSize ||
        m_state->numMigrationWires() != 0 ||
        m_state->numWires() > static_cast<int32_t>(m_config.batchedMaxQubits))
      return false;
    for (const auto &task : m_deferredTasks) {
      if (std::holds_alternative<PauliRotationTask>(task))
        return false;
      if (const auto *noise = std::get_if<NoiseTask<Scalar>>(&task)) {
        if (noise->matrices.size() > m_config.batchedMaxBranches) {
          CUDAQ_INFO("Noise channel with {} branches exceeds the batched "
                     "trajectory limit {}; using sequential replay.",
                     noise->matrices.size(), m_config.batchedMaxBranches);
          return false;
        }
      }
    }
    return true;
  }

  const void *batchedInitialState() {
    state().synchronize();
    state().exposeResources();
    const auto indices = state().deviceSubStateIndices();
    if (indices.size() != 1)
      throw std::runtime_error(
          "Batched execution requires one device sub-state vector.");
    return state().deviceSubStateVector(indices.front()).data;
  }

  /// The plan for running one PTSBE batch on a single device: the merged gate
  /// list per trajectory, the active (non-zero-shot) indices ordered
  /// longest-first, per-gate uniformity flags, and the device batch size.
  struct PTSBEPlan {
    std::vector<std::vector<GateApplicationTask>> trajectories;
    std::vector<std::size_t> active;
    std::vector<bool> uniformGates;
    std::size_t numGates = 0;
    std::size_t batchSize = 0;
  };

  /// Decide whether `batch` is eligible for the batched PTSBE path and, if so,
  /// build its plan. Returns `nullopt` when the batch must fall back to the
  /// generic sequential path (unsupported layout, non-uniform gate structure,
  /// too many branches, or too small a batch). A returned plan may still have
  /// no active trajectories when every trajectory requested zero shots.
  std::optional<PTSBEPlan>
  analyzePTSBEBatch(const cudaq::ptsbe::PTSBatch &batch) {
    if (!supportsBatchedTrajectories() || state().numMigrationWires() != 0 ||
        state().numWires() > static_cast<int32_t>(m_config.batchedMaxQubits))
      return std::nullopt;

    PTSBEPlan plan;
    plan.trajectories.resize(batch.trajectories.size());
    // Trajectories with zero shots remain empty results and need not be
    // simulated. Keep their original indices so result ordering is unchanged.
    for (std::size_t index = 0; index < batch.trajectories.size(); ++index) {
      if (batch.trajectories[index].num_shots == 0)
        continue;
      plan.trajectories[index] = cudaq::ptsbe::mergeTasksWithTrajectory<Scalar>(
          batch.trace, batch.trajectories[index], /*includeIdentity=*/true);
      plan.active.push_back(index);
    }
    if (plan.active.empty())
      return plan;

    // Put the longest sampling jobs first. In shot-by-shot mode this keeps
    // later batches from running to a large maximum shot count for only one
    // trajectory, while the original indices preserve result ordering.
    std::stable_sort(plan.active.begin(), plan.active.end(),
                     [&](std::size_t left, std::size_t right) {
                       return batch.trajectories[left].num_shots >
                              batch.trajectories[right].num_shots;
                     });

    // Batched matrix application requires every trajectory to have the same
    // number of gates and the same target/control structure at each position.
    plan.numGates = plan.trajectories[plan.active.front()].size();
    plan.uniformGates.assign(plan.numGates, false);
    for (const std::size_t trajectory : plan.active)
      if (plan.trajectories[trajectory].size() != plan.numGates)
        return std::nullopt;
    for (std::size_t gate = 0; gate < plan.numGates; ++gate) {
      const auto &first = plan.trajectories[plan.active.front()][gate];
      std::vector<std::vector<std::complex<Scalar>>> uniqueMatrices;
      for (const std::size_t trajectory : plan.active) {
        const auto &task = plan.trajectories[trajectory][gate];
        if (task.targets != first.targets || task.controls != first.controls)
          return std::nullopt;
        if (std::find(uniqueMatrices.begin(), uniqueMatrices.end(),
                      task.matrix) == uniqueMatrices.end()) {
          if (uniqueMatrices.size() >= m_config.batchedMaxBranches)
            return std::nullopt;
          uniqueMatrices.push_back(task.matrix);
        }
      }
      plan.uniformGates[gate] = uniqueMatrices.size() == 1;
    }

    plan.batchSize = selectedTrajectoryBatchSize(plan.active.size());
    if (plan.batchSize < m_config.batchedMinBatchSize)
      return std::nullopt;
    return plan;
  }

  /// Execute an eligible PTSBE `plan`, filling one sample_result per
  /// trajectory. Requires at least one active trajectory (checked by the
  /// caller).
  std::vector<cudaq::sample_result>
  runPTSBEBatch(const cudaq::ptsbe::PTSBatch &batch, const PTSBEPlan &plan) {
    assert(!plan.active.empty() &&
           "runPTSBEBatch requires at least one active trajectory.");
    const auto measuredWires = detail::toInt32(batch.measureQubits);
    std::vector<cudaq::sample_result> results(batch.trajectories.size());
    CuStateVecBatch<Scalar> batchedState(state().numWires(), plan.batchSize,
                                         m_config.allowFp32Emulation, false);
    for (std::size_t offset = 0; offset < plan.active.size();
         offset += plan.batchSize) {
      const std::size_t count =
          std::min(plan.batchSize, plan.active.size() - offset);
      batchedState.resize(count);
      batchedState.setZeroState();
      applyPTSBEGates(batchedState, plan, offset, count);
      if (m_config.ptsbeBatchSampleShotByShot)
        recordPTSBEShotByShot(batchedState, batch, plan, offset, count,
                              measuredWires, results);
      else
        recordPTSBEPerState(batchedState, batch, plan, offset, count,
                            measuredWires, results);
    }
    return results;
  }

  /// Apply the merged gate sequence to the batch: a gate shared by every active
  /// trajectory is applied once, while a diverging gate is applied per member
  /// through the indexed matrix map.
  void applyPTSBEGates(CuStateVecBatch<Scalar> &batchedState,
                       const PTSBEPlan &plan, std::size_t offset,
                       std::size_t count) {
    const auto toMatrixTask = [](const GateApplicationTask &source) {
      MatrixTask<Scalar> task;
      task.matrix = source.matrix;
      task.targets = detail::toInt32(source.targets);
      task.controls = detail::toInt32(source.controls);
      return task;
    };
    for (std::size_t gate = 0; gate < plan.numGates; ++gate) {
      if (plan.uniformGates[gate]) {
        batchedState.apply(
            toMatrixTask(plan.trajectories[plan.active[offset]][gate]));
        continue;
      }
      std::vector<MatrixTask<Scalar>> gateBatch;
      gateBatch.reserve(count);
      for (std::size_t local = 0; local < count; ++local)
        gateBatch.emplace_back(
            toMatrixTask(plan.trajectories[plan.active[offset + local]][gate]));
      batchedState.apply(gateBatch);
    }
  }

  /// Sample the whole batch together, one shot at a time, stopping each
  /// trajectory once it reaches its requested shot count.
  void recordPTSBEShotByShot(CuStateVecBatch<Scalar> &batchedState,
                             const cudaq::ptsbe::PTSBatch &batch,
                             const PTSBEPlan &plan, std::size_t offset,
                             std::size_t count,
                             const std::vector<int32_t> &measuredWires,
                             std::vector<cudaq::sample_result> &results) {
    std::vector<cudaq::ExecutionResult> batchResults(count);
    std::vector<std::size_t> shotsPerState(count);
    for (std::size_t local = 0; local < count; ++local)
      shotsPerState[local] =
          batch.trajectories[plan.active[offset + local]].num_shots;
    const std::size_t maxShots =
        *std::max_element(shotsPerState.begin(), shotsPerState.end());
    for (std::size_t shot = 0; shot < maxShots; ++shot) {
      const auto bitStrings =
          batchedState.measure(measuredWires, randomVector(count));
      for (std::size_t local = 0; local < count; ++local) {
        if (shot >= shotsPerState[local])
          continue;
        std::string bits =
            formatBitString(bitStrings[local], measuredWires.size());
        if (batch.includeSequentialData)
          batchResults[local].appendResult(bits, 1);
        else
          batchResults[local].counts[bits]++;
      }
    }
    for (std::size_t local = 0; local < count; ++local)
      results[plan.active[offset + local]] =
          cudaq::sample_result(std::move(batchResults[local]));
  }

  /// Sample each completed trajectory separately with its exact shot count.
  /// This is preferred when shot counts differ substantially because each
  /// trajectory performs only the number of samples it needs.
  void recordPTSBEPerState(CuStateVecBatch<Scalar> &batchedState,
                           const cudaq::ptsbe::PTSBatch &batch,
                           const PTSBEPlan &plan, std::size_t offset,
                           std::size_t count,
                           const std::vector<int32_t> &measuredWires,
                           std::vector<cudaq::sample_result> &results) {
    for (std::size_t local = 0; local < count; ++local) {
      const std::size_t trajectoryIndex = plan.active[offset + local];
      const std::size_t shots = batch.trajectories[trajectoryIndex].num_shots;
      auto randomNumbers = generateRandomNumbers(shots);
      results[trajectoryIndex] = cudaq::sample_result(
          batchedState.sample(local, measuredWires, std::move(randomNumbers),
                              batch.includeSequentialData));
    }
  }

  /// Snapshot the current state when it was seeded from a custom initial state
  /// so each trajectory can restart from it. Returns `nullopt` for zero-state
  /// inputs, which are cheaper to reconstruct directly.
  std::optional<CuStateVecState<Scalar>> cloneInitialStateIfCustom() {
    std::optional<CuStateVecState<Scalar>> initialState;
    if (m_hasCustomInitialState) {
      initialState.emplace(m_state->cloneEmpty());
      initialState->copyFrom(*m_state);
    }
    return initialState;
  }

  /// Resolve how many noisy trajectories to run: the execution context's
  /// explicit request when present, otherwise the configured default. Throws
  /// when the resolved count is zero.
  std::size_t resolveTrajectoryCount() const {
    const auto *const context = cudaq::getExecutionContext();
    const std::size_t trajectoryCount = context && context->numberTrajectories
                                            ? *context->numberTrajectories
                                            : m_config.observeTrajectories;
    if (trajectoryCount == 0)
      throw std::invalid_argument(
          "numberTrajectories must be greater than zero.");
    return trajectoryCount;
  }

  /// Replay the deferred noisy circuit onto every active member of `batch`,
  /// selecting an independent Kraus branch per member.
  void applyDeferredTasksToBatch(CuStateVecBatch<Scalar> &batch,
                                 std::size_t count) {
    for (const auto &simulationTask : m_deferredTasks) {
      if (const auto *matrix =
              std::get_if<MatrixTask<Scalar>>(&simulationTask)) {
        batch.apply(*matrix);
        continue;
      }
      const auto &noise = std::get<NoiseTask<Scalar>>(simulationTask);
      batch.applyNoise(noise, randomVector(count));
    }
  }

  cudaq::ExecutionResult
  sampleTrajectories(const std::vector<std::size_t> &measuredBits, int shots,
                     bool includeSequentialData) {
    if (m_deferredTasks.empty()) {
      detail::ScopedFlag replay(m_replayingTrajectory);
      return sample(measuredBits, shots, includeSequentialData);
    }

    if (shots > 0 && canBatchDeferredTrajectories(shots))
      return sampleBatchedTrajectories(batchedInitialState(), measuredBits,
                                       shots, includeSequentialData);

    auto initialState = cloneInitialStateIfCustom();
    const auto *const initialStatePointer =
        initialState ? &*initialState : nullptr;
    // shots < 1: exact noisy observe -- average the expectation value over
    // independent trajectories.
    if (shots < 1) {
      const std::size_t trajectoryCount = resolveTrajectoryCount();
      double expectation = 0.0;
      for (std::size_t trajectory = 0; trajectory < trajectoryCount;
           ++trajectory) {
        runDeferredTrajectory(initialStatePointer, false);
        detail::ScopedFlag replay(m_replayingTrajectory);
        expectation +=
            sample(measuredBits, shots, false).expectationValue.value_or(0.0);
      }
      restoreTrajectoryState(initialStatePointer);
      m_deferredTasks.clear();
      return cudaq::ExecutionResult{expectation /
                                    static_cast<double>(trajectoryCount)};
    }

    // shots >= 1: run one independent trajectory per shot and record its single
    // measured bit-string.
    cudaq::ExecutionResult result;
    for (int shot = 0; shot < shots; ++shot) {
      runDeferredTrajectory(initialStatePointer, false);
      detail::ScopedFlag replay(m_replayingTrajectory);
      const auto trajectoryResult = sample(measuredBits, 1, true);
      if (trajectoryResult.counts.empty())
        throw std::runtime_error("Trajectory sampling returned no result.");
      const std::string &bits = trajectoryResult.counts.begin()->first;
      if (includeSequentialData)
        result.appendResult(bits, 1);
      else
        result.counts[bits]++;
    }
    restoreTrajectoryState(initialStatePointer);
    m_deferredTasks.clear();

    result.expectationValue = detail::parityExpectation(result, shots);
    return result;
  }

  /// Simulate multiple noisy shots as independent state vectors in one
  /// cuStateVec batched allocation.
  cudaq::ExecutionResult
  sampleBatchedTrajectories(const void *initialState,
                            const std::vector<std::size_t> &measuredBits,
                            int shots, bool includeSequentialData) {
    const std::size_t batchSize =
        selectedTrajectoryBatchSize(static_cast<std::size_t>(shots));

    const auto wires = detail::toInt32(measuredBits);
    cudaq::ExecutionResult result;
    CuStateVecBatch<Scalar> batch(state().numWires(), batchSize,
                                  m_config.allowFp32Emulation, false);
    for (std::size_t offset = 0; offset < static_cast<std::size_t>(shots);
         offset += batchSize) {
      const std::size_t count =
          std::min(batchSize, static_cast<std::size_t>(shots) - offset);
      batch.resize(count);
      batch.setState(initialState);
      applyDeferredTasksToBatch(batch, count);

      const auto bitStrings = batch.measure(wires, randomVector(count));
      for (std::size_t stateIndex = 0; stateIndex < bitStrings.size();
           ++stateIndex) {
        const custatevecIndex_t value = bitStrings[stateIndex];
        std::string bits = formatBitString(value, measuredBits.size());
        if (includeSequentialData)
          result.appendResult(bits, 1);
        else
          result.counts[bits]++;
      }
    }
    m_deferredTasks.clear();
    result.expectationValue = detail::parityExpectation(result, shots);
    return result;
  }

  /// Average Pauli-term expectations across a batch of noisy trajectories.
  /// This does not add bit-string measurement randomness.
  cudaq::observe_result
  observeBatchedTrajectories(const void *initialState, const cudaq::spin_op &op,
                             std::size_t trajectoryCount) {
    const std::size_t batchSize =
        selectedTrajectoryBatchSize(trajectoryCount, /*isObserve=*/true);

    const auto terms = detail::extractPauliTerms(op);
    std::vector<double> averaged(op.num_terms(), 0.0);
    CuStateVecBatch<Scalar> batch(state().numWires(), batchSize,
                                  m_config.allowFp32Emulation, false);
    for (std::size_t offset = 0; offset < trajectoryCount;
         offset += batchSize) {
      const std::size_t count = std::min(batchSize, trajectoryCount - offset);
      batch.resize(count);
      batch.setState(initialState);
      applyDeferredTasksToBatch(batch, count);

      const auto values = batch.expectationPauli(terms.paulis, terms.targets);
      for (std::size_t stateIndex = 0; stateIndex < count; ++stateIndex)
        for (std::size_t termIndex = 0; termIndex < op.num_terms(); ++termIndex)
          averaged[termIndex] +=
              values[stateIndex * op.num_terms() + termIndex];
    }
    m_deferredTasks.clear();

    return detail::makeObserveResult(op, std::move(averaged), trajectoryCount);
  }

  cudaq::observe_result observeTrajectories(const cudaq::spin_op &op) {
    if (op.num_terms() == 0)
      return {};
    if (m_deferredTasks.empty()) {
      detail::ScopedFlag replay(m_replayingTrajectory);
      return observe(op);
    }

    const std::size_t trajectoryCount = resolveTrajectoryCount();
    if (trajectoryCount <=
            static_cast<std::size_t>(std::numeric_limits<int>::max()) &&
        canBatchDeferredTrajectories(trajectoryCount)) {
      return observeBatchedTrajectories(batchedInitialState(), op,
                                        trajectoryCount);
    }
    // Exact noisy observe also starts every trajectory from the same custom
    // state before averaging its independently selected Kraus branches.
    auto initialState = cloneInitialStateIfCustom();
    const auto *const initialStatePointer =
        initialState ? &*initialState : nullptr;
    std::vector<double> averaged(op.num_terms(), 0.0);
    for (std::size_t trajectory = 0; trajectory < trajectoryCount;
         ++trajectory) {
      runDeferredTrajectory(initialStatePointer, false);
      detail::ScopedFlag replay(m_replayingTrajectory);
      auto trajectoryResult = observe(op);
      std::size_t termIndex = 0;
      for (const auto &term : op)
        averaged[termIndex++] += trajectoryResult.expectation(term);
    }
    restoreTrajectoryState(initialStatePointer);
    m_deferredTasks.clear();

    return detail::makeObserveResult(op, std::move(averaged), trajectoryCount);
  }

  CuStateVecConfig m_config;
  std::optional<CuStateVecState<Scalar>> m_state;
  std::unique_ptr<GateEngine<Scalar>> m_engine;
  std::random_device m_randomDevice;
  std::mt19937 m_randomEngine;
  CuStateVecRandom m_gpuRandom;
  std::vector<SimulationTask<Scalar>> m_deferredTasks;
  std::vector<double> m_pendingRandomNumbers;
  // Set once a measurement is recorded on the non-explicit noisy `sample` path;
  // used to reject unsupported non-terminal (mid-circuit) measurements.
  bool m_measurementRecorded = false;
  bool m_replayingTrajectory = false;
  bool m_hasCustomInitialState = false;
};

} // namespace cudaq::cusv
