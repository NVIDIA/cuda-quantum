/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/FmtCore.h"
#include "nvqir/CircuitSimulator.h"
#include "nvqir/Gates.h"
#include <bit>
#include <iostream>
#include <qpp/qpp.hpp>
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
    const auto measurement_result = std::get<qpp::measure_idx::res>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::measure_idx::st>(measurement_tuple);
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

  /// @brief Draw `shots` basis states from the state vector and return their
  /// amplitude indices in ascending order.
  ///
  /// One uniform number is drawn per shot and located in the running sum of
  /// the probabilities. The numbers are sorted first, so one pass over the
  /// state places all of them and no table of probabilities is stored.
  std::vector<std::size_t> drawBasisStates(std::size_t shots) {
    const std::size_t dimension = state.size();
    double total = 0.0;
    for (std::size_t i = 0; i < dimension; ++i)
      total += std::norm(state[i]);

    auto &generator = qpp::RandomDevices::get_instance().get_prng();
    std::vector<double> thresholds(shots);
    for (auto &threshold : thresholds)
      threshold = std::generate_canonical<double, 53>(generator) * total;
    std::sort(thresholds.begin(), thresholds.end());

    std::vector<std::size_t> indices;
    indices.reserve(shots);
    double runningSum = 0.0;
    std::size_t lastPossible = 0;
    for (std::size_t i = 0; i < dimension && indices.size() < shots; ++i) {
      const double probability = std::norm(state[i]);
      if (probability == 0.0)
        continue;
      runningSum += probability;
      lastPossible = i;
      while (indices.size() < shots && thresholds[indices.size()] < runningSum)
        indices.push_back(i);
    }
    // Rounding can leave the largest numbers at or above the final sum. They
    // belong to the last basis state that can occur.
    indices.resize(shots, lastPossible);
    return indices;
  }

  /// @brief Sample the given qubits of a state vector.
  cudaq::ExecutionResult
  sampleStateVector(const std::vector<std::size_t> &qubits, const int shots,
                    bool includeSequentialData) {
    if (state.size() == 0)
      throw std::runtime_error("cannot sample an empty state");
    const std::size_t numQubits = std::log2(state.size());
    for (const auto qubit : qubits)
      if (qubit >= numQubits)
        throw std::runtime_error("qubit index " + std::to_string(qubit) +
                                 " is out of range");

    // Pack the measured bits of each drawn basis state into an integer, the
    // first qubit in the highest bit, so that the integers sort in the order
    // of the bitstrings.
    const auto indices = drawBasisStates(shots);
    std::vector<std::uint64_t> outcomes;
    outcomes.reserve(indices.size());
    for (const auto index : indices) {
      std::uint64_t outcome = 0;
      for (const auto qubit : qubits)
        outcome = (outcome << 1) | ((index >> qubit) & 1);
      outcomes.push_back(outcome);
    }
    std::sort(outcomes.begin(), outcomes.end());

    cudaq::ExecutionResult counts;
    double expVal = 0.0;
    std::string bitstring(qubits.size(), '0');
    for (std::size_t first = 0; first < outcomes.size();) {
      const auto outcome = outcomes[first];
      std::size_t last = first;
      while (last < outcomes.size() && outcomes[last] == outcome)
        ++last;
      const std::size_t count = last - first;
      first = last;

      for (std::size_t j = 0; j < qubits.size(); ++j)
        bitstring[j] = '0' + ((outcome >> (qubits.size() - 1 - j)) & 1);
      // in mid-circ sampling mode this will append 1 bitstring
      if (includeSequentialData)
        counts.appendResult(bitstring, count);
      else
        counts.counts[bitstring] += count;

      // Expectation value from the counts
      auto p = count / (double)shots;
      if (std::popcount(outcome) % 2 != 0)
        p = -p;
      expVal += p;
    }

    counts.expectationValue = expVal;
    return counts;
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

    if constexpr (std::is_same_v<StateType, qpp::ket>)
      if (qubits.size() <= 64)
        return sampleStateVector(qubits, shots, includeSequentialData);

    std::vector<std::size_t> measuredBits;
    for (auto index : qubits) {
      measuredBits.push_back(convertQubitIndex(index));
    }

    auto sampleResult = qpp::sample(shots, state, measuredBits, 2);
    // Convert to what we expect
    std::stringstream bitstream;
    cudaq::ExecutionResult counts;

    // Expectation value from the counts
    double expVal = 0.0;
    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of bits to the bitstring.
      for (const auto &bit : result) {
        bitstream << bit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      auto bitstring = bitstream.str();
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
      // Reset the state.
      bitstream.str("");
      bitstream.clear();
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
