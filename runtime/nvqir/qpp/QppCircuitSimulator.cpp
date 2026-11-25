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
#include <qpp.h>
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

/// @brief Insert a 0 bit at each of the given (ascending) bit positions.
inline std::size_t insertZeroBits(std::size_t value,
                                  const std::vector<std::size_t> &sortedBits) {
  for (auto bit : sortedBits) {
    const std::size_t low = value & ((1ULL << bit) - 1);
    value = ((value >> bit) << (bit + 1)) | low;
  }
  return value;
}

/// @brief Complex product without the NaN/Inf handling of `operator*`, which
/// is much slower in the hot loops.
inline std::complex<double> multiply(const std::complex<double> &a,
                                     const std::complex<double> &b) {
  return {a.real() * b.real() - a.imag() * b.imag(),
          a.real() * b.imag() + a.imag() * b.real()};
}

/// @brief Apply a (controlled) gate to a state vector, in place.
///
/// Qubit `q` is bit `q` of the amplitude index. `matrix` is row-major, with
/// `targets[0]` as the most significant bit of its index. Only amplitudes
/// whose control bits are all 1 are touched.
inline void applyGateInPlace(std::complex<double> *state, std::size_t numQubits,
                             const std::vector<std::complex<double>> &matrix,
                             const std::vector<std::size_t> &controls,
                             const std::vector<std::size_t> &targets) {
  const std::size_t numTargets = targets.size();
  if (numTargets == 0)
    throw std::invalid_argument("[qpp] gate applied to no target qubits.");
  if (numTargets >= numQubits + 1)
    throw std::invalid_argument("[qpp] gate has more targets than qubits.");
  const std::size_t dim = 1ULL << numTargets;
  if (matrix.size() != dim * dim)
    throw std::invalid_argument(
        "[qpp] gate matrix size does not match the number of targets.");

  // Collect and validate the control and target qubits.
  std::size_t controlMask = 0, involvedMask = 0;
  std::vector<std::size_t> involved;
  involved.reserve(controls.size() + numTargets);
  const auto addQubit = [&](std::size_t qubit) {
    if (qubit >= numQubits)
      throw std::invalid_argument("[qpp] qubit index out of range.");
    const std::size_t bit = 1ULL << qubit;
    if (involvedMask & bit)
      throw std::invalid_argument(
          "[qpp] control and target qubits must be distinct.");
    involvedMask |= bit;
    involved.push_back(qubit);
  };
  for (auto qubit : controls) {
    addQubit(qubit);
    controlMask |= 1ULL << qubit;
  }
  for (auto qubit : targets)
    addQubit(qubit);
  std::sort(involved.begin(), involved.end());

  // One group of amplitudes per assignment of the remaining qubits.
  const std::size_t numGroups = 1ULL << (numQubits - involved.size());
  [[maybe_unused]] const bool runParallel = numGroups >= (1ULL << 12);

  if (numTargets == 1) {
    const std::size_t targetBit = 1ULL << targets[0];
    const auto m00 = matrix[0], m01 = matrix[1], m10 = matrix[2],
               m11 = matrix[3];
#if defined(_OPENMP)
#pragma omp parallel for if (runParallel)
#endif
    for (std::size_t i = 0; i < numGroups; ++i) {
      const std::size_t i0 = insertZeroBits(i, involved) | controlMask;
      const std::size_t i1 = i0 | targetBit;
      const auto a0 = state[i0], a1 = state[i1];
      state[i0] = multiply(m00, a0) + multiply(m01, a1);
      state[i1] = multiply(m10, a0) + multiply(m11, a1);
    }
    return;
  }

  // Index offset of each target basis state within a group.
  std::vector<std::size_t> offsets(dim, 0);
  for (std::size_t m = 0; m < dim; ++m)
    for (std::size_t j = 0; j < numTargets; ++j)
      if (m & (1ULL << (numTargets - 1 - j)))
        offsets[m] |= 1ULL << targets[j];

#if defined(_OPENMP)
#pragma omp parallel if (runParallel)
#endif
  {
    std::vector<std::complex<double>> amplitudes(dim);
#if defined(_OPENMP)
#pragma omp for
#endif
    for (std::size_t i = 0; i < numGroups; ++i) {
      const std::size_t base = insertZeroBits(i, involved) | controlMask;
      for (std::size_t m = 0; m < dim; ++m)
        amplitudes[m] = state[base | offsets[m]];
      for (std::size_t row = 0; row < dim; ++row) {
        std::complex<double> sum = 0.0;
        const auto *matrixRow = matrix.data() + row * dim;
        for (std::size_t col = 0; col < dim; ++col)
          sum += multiply(matrixRow[col], amplitudes[col]);
        state[base | offsets[row]] = sum;
      }
    }
  }
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
