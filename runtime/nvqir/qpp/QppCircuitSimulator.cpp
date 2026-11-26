/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QppInPlaceGates.h"
#include "common/FmtCore.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include <algorithm>
#include <bit>
#include <cmath>
#include <iostream>
#include <map>
#include <numeric>
#include <qpp.h>
#include <random>
#include <set>
#include <span>

using namespace cudaq;

namespace nvqir {

/// @brief QppState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Qpp Circuit Simulator.
struct QppState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  QppState(qpp::ket &&data) : state(std::move(data)) {}
  QppState(const std::vector<std::size_t> &shape,
           const std::vector<std::complex<double>> &data) {
    if (shape.size() != 1)
      throw std::runtime_error(
          "QppState must be created from data with 1D shape.");

    state = Eigen::Map<qpp::ket>(
        const_cast<std::complex<double> *>(data.data()), shape[0]);
  }

  std::size_t getNumQubits() const override { return std::log2(state.size()); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 ||
        (other.getTensor().extents != getTensor().extents))
      throw std::runtime_error("[qpp-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    std::span<std::complex<double>> otherState(
        reinterpret_cast<std::complex<double> *>(other.getTensor().data),
        other.getTensor().extents[0]);
    return std::abs(std::inner_product(
        state.begin(), state.end(), otherState.begin(), complex{0., 0.},
        [](auto a, auto b) { return a + b; },
        [](auto a, auto b) { return a * std::conj(b); }));
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(fmt::format(
          "[qpp-state] getAmplitude with an invalid number of bits in the "
          "basis state: expected {}, provided {}.",
          getNumQubits(), basisState.size()));
    if (std::any_of(basisState.begin(), basisState.end(),
                    [](int x) { return x != 0 && x != 1; }))
      throw std::runtime_error(
          "[qpp-state] getAmplitude with an invalid basis state: only "
          "qubit state (0 or 1) is supported.");

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    return state[idx];
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-state] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(
            const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
        getPrecision()};
  }

  // /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  // /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-state] invalid tensor requested.");
    if (indices.size() != 1)
      throw std::runtime_error("[qpp-state] invalid element extraction.");

    return state[indices[0]];
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    if (!ptr || size == 0)
      throw std::runtime_error(
          "[createFromSizeAndPtr] invalid null pointer or zero size");
    return std::make_unique<QppState>(Eigen::Map<qpp::ket>(
        reinterpret_cast<std::complex<double> *>(ptr), size));
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
};

namespace details {

/// @brief Draw `shots` basis states from a state vector, with probability
/// proportional to the squared magnitude of each amplitude.
///
/// Returns (basis state index, count) pairs in ascending index order. The
/// state does not need to be normalized. Unlike `qpp::sample`, this allocates
/// nothing the size of the state: the sorted random draws are resolved in one
/// pass over the per-chunk weights, and only chunks that contain a draw are
/// scanned amplitude by amplitude. The result does not depend on the number
/// of threads.
template <typename Generator>
std::vector<std::pair<std::size_t, std::size_t>>
sampleBasisStates(const std::complex<double> *state, std::size_t dimension,
                  std::size_t shots, Generator &generator) {
  constexpr std::size_t chunkSize = 1ULL << 12;
  const std::size_t numChunks = (dimension + chunkSize - 1) / chunkSize;

  // Total weight of each fixed-size chunk of the state.
  std::vector<double> chunkWeights(numChunks);
  [[maybe_unused]] const int numThreads = parallelThreads(dimension);
#if defined(_OPENMP)
#pragma omp parallel for num_threads(numThreads) if (numThreads > 1)
#endif
  for (std::size_t chunk = 0; chunk < numChunks; ++chunk) {
    const std::size_t begin = chunk * chunkSize;
    const std::size_t end = std::min(begin + chunkSize, dimension);
    double weight = 0.0;
    for (std::size_t i = begin; i < end; ++i)
      weight += std::norm(state[i]);
    chunkWeights[chunk] = weight;
  }
  const double total =
      std::accumulate(chunkWeights.begin(), chunkWeights.end(), 0.0);
  if (!(total > 0.0) || !std::isfinite(total))
    throw std::runtime_error("[qpp] cannot sample from a state whose norm is "
                             "zero or not finite.");

  // One sorted point in [0, total) per shot.
  std::vector<double> draws(shots);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);
  for (auto &draw : draws)
    draw = uniform(generator) * total;
  std::sort(draws.begin(), draws.end());

  std::vector<std::pair<std::size_t, std::size_t>> counts;
  const auto record = [&](std::size_t index, std::size_t count) {
    if (!counts.empty() && counts.back().first == index)
      counts.back().second += count;
    else
      counts.emplace_back(index, count);
  };

  // Walk the cumulative distribution; every draw below `chunkStart` has
  // already been assigned.
  std::size_t next = 0;
  double chunkStart = 0.0;
  for (std::size_t chunk = 0; chunk < numChunks && next < shots; ++chunk) {
    const double chunkEnd = chunkStart + chunkWeights[chunk];
    if (draws[next] < chunkEnd) {
      const std::size_t begin = chunk * chunkSize;
      const std::size_t end = std::min(begin + chunkSize, dimension);
      double cumulative = chunkStart;
      std::size_t lastNonZero = begin;
      for (std::size_t i = begin;
           i < end && next < shots && draws[next] < chunkEnd; ++i) {
        const double weight = std::norm(state[i]);
        if (weight == 0.0)
          continue;
        cumulative += weight;
        lastNonZero = i;
        std::size_t count = 0;
        for (; next < shots && draws[next] < cumulative; ++next)
          ++count;
        if (count > 0)
          record(i, count);
      }
      // Draws left in this chunk by floating-point rounding.
      std::size_t count = 0;
      for (; next < shots && draws[next] < chunkEnd; ++next)
        ++count;
      if (count > 0)
        record(lastNonZero, count);
    }
    chunkStart = chunkEnd;
  }

  // Draws at or beyond the total (rounding) go to the last possible outcome.
  if (next < shots) {
    std::size_t last = dimension - 1;
    while (last > 0 && std::norm(state[last]) == 0.0)
      --last;
    record(last, shots - next);
  }
  return counts;
}

} // namespace details

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
template <typename StateType>
class QppCircuitSimulator : public nvqir::CircuitSimulatorBase<double> {
protected:
  /// The QPP state representation (qpp::ket or qpp::cmat)
  StateType state;

  /// @brief Convert internal qubit index to Q++ qubit index.
  ///
  /// In Q++, qubits are indexed from left to right, and thus q0 is the leftmost
  /// qubit. Internally, in CUDA-Q, qubits are index from right to left,
  /// hence q0 is the rightmost qubit. Example:
  /// ```
  ///   Q++ indices:  0  1  2  3
  ///                |0>|0>|0>|0>
  ///                 3  2  1  0 : CUDA-Q indices
  /// ```
  std::size_t convertQubitIndex(std::size_t qubitIndex) {
    assert(stateDimension > 0 && "The state is empty, and thus has no qubits");
    return std::log2(stateDimension) - qubitIndex - 1;
  }

  /// @brief Compute the expectation value <Z...Z> over the given qubit indices.
  double calculateExpectationValue(const std::vector<std::size_t> &qubits) {
    std::size_t bitmask = 0;
    for (auto q : qubits)
      bitmask |= (1ULL << q);
    const auto hasEvenParity = [&bitmask](std::size_t x) -> bool {
      return std::popcount(x & bitmask) % 2 == 0;
    };

    std::vector<double> result;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      result.resize(stateDimension);
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (std::size_t i = 0; i < stateDimension; ++i)
        result[i] = (hasEvenParity(i) ? 1.0 : -1.0) * std::norm(state[i]);
    } else if constexpr (std::is_same_v<StateType, qpp::cmat>) {
      Eigen::VectorXcd diag = state.diagonal();
      result.resize(state.rows());
#if defined(_OPENMP)
#pragma omp parallel for
#endif
      for (Eigen::Index i = 0; i < state.rows(); ++i)
        result[i] = hasEvenParity(i) ? diag(i).real() : -diag(i).real();
    }

    // Accumulate outside the for loop to ensure repeatability
    return std::accumulate(result.begin(), result.end(), 0.0);
  }

  qpp::cmat toQppMatrix(const std::vector<std::complex<double>> &data,
                        std::size_t nTargets) {
    auto nRows = (1UL << nTargets);
    assert(data.size() == nRows * nRows &&
           "Invalid number of gate matrix elements passed to toQppMatrix");

    // we represent row major, they represent column major
    return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>>(
        const_cast<std::complex<double> *>(data.data()), nRows, nRows);
  }

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  void addQubitsToState(std::size_t qubitCount,
                        const void *stateDataIn = nullptr) override {
    if (qubitCount == 0)
      return;

    auto *stateData = reinterpret_cast<std::complex<double> *>(
        const_cast<void *>(stateDataIn));

    if (state.size() == 0) {
      // Size from qubitCount: with deferred allocations stateDimension
      // can already count qubits this call is not materializing.
      const auto dimension = calculateStateDim(qubitCount);
      if (stateData == nullptr) {
        state = qpp::ket::Zero(dimension);
        state(0) = 1.0;
      } else
        state = qpp::ket::Map(stateData, dimension);
      return;
    }
    // If we are resizing an existing, allocate
    // a zero state on a n qubit, and Kron-prod
    // that with the existing state.
    if (stateData == nullptr) {
      qpp::ket zero_state = qpp::ket::Zero((1UL << qubitCount));
      zero_state(0) = 1.0;
      state = qpp::kron(zero_state, state);
    } else {
      qpp::ket initState = qpp::ket::Map(stateData, (1UL << qubitCount));
      state = qpp::kron(initState, state);
    }
    return;
  }

  void addQubitsToState(const cudaq::SimulationState &in_state) override {
    const QppState *const casted = dynamic_cast<const QppState *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[QppCircuitSimulator] Incompatible state input");

    if (state.size() == 0)
      state = casted->state;
    else
      state = qpp::kron(casted->state, state);
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    StateType tmp;
    state = tmp;
  }

  void applyGate(const GateApplicationTask &task) override {
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      // State vector: apply the gate in place, which is much faster than
      // `qpp::apply` / `qpp::applyCTRL`.
      const std::size_t numQubits =
          std::bit_width(static_cast<std::size_t>(state.size())) - 1;
      details::applyGateInPlace(state.data(), numQubits, task.matrix,
                                task.controls, task.targets);
      return;
    }

    // Density matrix: delegate to Q++.
    auto matrix = toQppMatrix(task.matrix, task.targets.size());
    // First, convert all of the qubit indices to big endian.
    std::vector<std::size_t> controls;
    for (auto index : task.controls) {
      controls.push_back(convertQubitIndex(index));
    }
    std::vector<std::size_t> targets;
    for (auto index : task.targets) {
      targets.push_back(convertQubitIndex(index));
    }

    if (controls.empty()) {
      state = qpp::apply(state, matrix, targets);
      return;
    }
    state = qpp::applyCTRL(state, matrix, controls, targets);
  }

  /// @brief Set the current state back to the |0> state.
  void setToZeroState() override {
    // `stateDimension` already counts queued allocations.
    flushPendingQubits();
    state = qpp::ket::Zero(stateDimension);
    state(0) = 1.0;
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t index) override {
    const auto qubitIdx = convertQubitIndex(index);
    // If here, then we care about the result bit, so compute it.
    const auto measurement_tuple =
        qpp::measure(state, qpp::cmat::Identity(2, 2), {qubitIdx},
                     /*qudit dimension=*/2, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      state = Eigen::Map<const StateType>(collapsed_state.data(),
                                          collapsed_state.size());
    } else {
      state = Eigen::Map<const StateType>(collapsed_state.data(),
                                          collapsed_state.rows(),
                                          collapsed_state.cols());
    }
    CUDAQ_INFO("Measured qubit {} -> {}", qubitIdx, measurement_result);
    return measurement_result == 1 ? true : false;
  }

  QubitOrdering getQubitOrdering() const override { return QubitOrdering::msb; }

public:
  QppCircuitSimulator() {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
  }
  virtual ~QppCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override {
    qpp::RandomDevices::get_instance().get_prng().seed(seed);
  }

  bool canHandleObserve() override {
    auto executionContext = cudaq::getExecutionContext();

    // Do not compute <H> from matrix if shots based sampling requested
    if (executionContext &&
        executionContext->shots != static_cast<std::size_t>(-1)) {
      return false;
    }

    return !shouldObserveFromSampling();
  }

  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    assert(cudaq::spin_op::canonicalize(op) == op);
    flushGateQueue();

    // The op is on the following target bits.
    auto targets = op.degrees();

    // Get the matrix as an Eigen matrix
    auto matrix = op.to_matrix();
    qpp::cmat asEigen = matrix.as_eigen();

    // Compute the expected value
    double ee = 0.0;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      qpp::ket k = qpp::apply(state, asEigen, targets, 2);
      ee = state.dot(k).real();
    } else {
      ee = qpp::apply(asEigen, state, targets).trace().real();
    }

    return cudaq::observe_result(
        ee, op,
        cudaq::sample_result(cudaq::ExecutionResult({}, op.to_string(), ee)));
  }

  /// @brief Reset the qubit
  /// @param index 0-based index of qubit to reset
  void resetQubit(const std::size_t index) override {
    flushGateQueue();
    flushAnySamplingTasks();
    const auto qubitIdx = convertQubitIndex(index);
    state = qpp::reset(state, {qubitIdx});
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    if (shots < 1) {
      double expectationValue = calculateExpectationValue(qubits);
      CUDAQ_INFO("Computed expectation value = {}", expectationValue);
      return cudaq::ExecutionResult{{}, expectationValue};
    }

    // Bitstring of the measured qubits -> number of shots, in the
    // (lexicographic) order the results are reported in.
    std::map<std::string, std::size_t> sampleResult;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
      // State vector: sample without the state-sized temporaries of
      // `qpp::sample`. Qubit `q` is bit `q` of the basis state index.
      const auto sampled = details::sampleBasisStates(
          state.data(), state.size(), shots,
          qpp::RandomDevices::get_instance().get_prng());
      std::string bitstring(qubits.size(), '0');
      for (const auto &[basisState, count] : sampled) {
        for (std::size_t i = 0; i < qubits.size(); ++i)
          bitstring[i] = ((basisState >> qubits[i]) & 1) ? '1' : '0';
        sampleResult[bitstring] += count;
      }
    } else {
      // Density matrix: delegate to Q++.
      std::vector<std::size_t> measuredBits;
      for (auto index : qubits)
        measuredBits.push_back(convertQubitIndex(index));
      for (const auto &[result, count] :
           qpp::sample(shots, state, measuredBits, 2)) {
        std::string bitstring;
        for (const auto &bit : result)
          bitstring.push_back(bit ? '1' : '0');
        sampleResult[bitstring] += count;
      }
    }

    // Convert to what we expect, with the expectation value from the counts.
    cudaq::ExecutionResult counts;
    double expVal = 0.0;
    for (const auto &[bitstring, count] : sampleResult) {
      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      if (includeSequentialData)
        counts.appendResult(bitstring, count);
      else
        counts.counts[bitstring] += count;
      auto par = cudaq::sample_result::has_even_parity(bitstring);
      auto p = count / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
    }

    counts.expectationValue = expVal;
    return counts;
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    return std::make_unique<QppState>(std::move(state));
  }

  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    return std::make_unique<QppState>(qpp::ket{})->createFromData(data);
  }

  bool isStateVectorSimulator() const override {
    return std::is_same_v<StateType, qpp::ket>;
  }

  std::string name() const override { return "qpp"; }
  NVQIR_SIMULATOR_CLONE_IMPL(QppCircuitSimulator<StateType>)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::QppCircuitSimulator<qpp::ket>, qpp)
#endif
