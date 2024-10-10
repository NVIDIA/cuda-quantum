/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "nvqir/photonics/PhotonicCircuitSimulator.h"
#include <qpp.h>
#include <set>
#include <span>

using namespace cudaq;

namespace nvqir {

/// @brief QppPhotonicState provides an implementation of `PhotonicState` that
/// encapsulates the state data for the Qpp Circuit Simulator.
struct QppPhotonicState : public cudaq::PhotonicState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  /// @brief The levels of the qudits
  std::size_t levels;

  QppPhotonicState(qpp::ket &&data, std::size_t lvl)
      : state(std::move(data)), levels(lvl) {}
  QppPhotonicState(const std::vector<std::size_t> &shape,
                   const std::vector<std::complex<double>> &data,
                   std::size_t lvl) {
    if (shape.size() != 1)
      throw std::runtime_error(
          "QppPhotonicState must be created from data with 1D shape.");

    state = Eigen::Map<qpp::ket>(
        const_cast<std::complex<double> *>(data.data()), shape[0]);

    levels = lvl;
  }

  std::size_t getNumQudits() const override {
    return (std::log2(state.size()) / std::log2(levels));
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQudits() != basisState.size())
      throw std::runtime_error(
          fmt::format("[photonic-state] getAmplitude with an invalid number "
                      "of qudits in the basis state: expected{}, provided{}.",
                      getNumQudits(), basisState.size()));

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [&](std::size_t acc, int digit) { return (acc * levels) + digit; });
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

  std::unique_ptr<PhotonicState>
  createPSFromSizeAndPtr(std::size_t size, void *ptr,
                         std::size_t dataType) override {
    return std::make_unique<QppPhotonicState>(
        Eigen::Map<qpp::ket>(reinterpret_cast<std::complex<double> *>(ptr),
                             size),
        levels);
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    return std::make_unique<QppPhotonicState>(
        Eigen::Map<qpp::ket>(reinterpret_cast<std::complex<double> *>(ptr),
                             size),
        levels);
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
}; // QppPhotonicState

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
template <typename StateType>
class QppPhotonicCircuitSimulator
    : public PhotonicCircuitSimulatorBase<double> {
protected:
  /// The QPP state representation (qpp::ket or qpp::cmat)
  StateType state;

  std::size_t convertQuditIndex(std::size_t quditIndex) {
    assert(stateDimension > 0 && "The state is empty, and thus has no qudits");
    return quditIndex;
  }

  qpp::cmat toQppMatrix(const std::vector<std::complex<double>> &data,
                        std::size_t nTargets) {

    auto nRows = std::pow(levels, nTargets);
    assert(data.size() == nRows * nRows &&
           "Invalid number of gate matrix elements passed to toQppMatrix");

    // we represent row major, they represent column major
    return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>>(
        const_cast<std::complex<double> *>(data.data()), nRows, nRows);
  }

  /// @brief Grow the state vector by one qudit.
  void addQuditToState() override { addQuditsToState(1); }

  void addQuditsToState(std::size_t quditCount,
                        const void *stateDataIn = nullptr) override {
    if (quditCount == 0)
      return;

    auto *stateData = reinterpret_cast<std::complex<double> *>(
        const_cast<void *>(stateDataIn));

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      if (stateData == nullptr) {
        state = qpp::ket::Zero(stateDimension);
        state(0) = 1.0;
      } else
        state = qpp::ket::Map(stateData, stateDimension);
      return;
    }
    // If we are resizing an existing, allocate
    // a zero state on a n qudit, and Kron-prod
    // that with the existing state.
    if (stateData == nullptr) {
      qpp::ket zero_state = qpp::ket::Zero(calculateStateDim(quditCount));
      zero_state(0) = 1.0;
      state = qpp::kron(zero_state, state);
    } else {
      qpp::ket initState =
          qpp::ket::Map(stateData, calculateStateDim(quditCount));
      state = qpp::kron(initState, state);
    }
    return;
  }

  void addQuditsToState(const cudaq::PhotonicState &in_state) override {
    const QppPhotonicState *const casted =
        dynamic_cast<const QppPhotonicState *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[PhotonicCircuitSimulator] Incompatible state input");

    if (state.size() == 0)
      state = casted->state;
    else
      state = qpp::kron(casted->state, state);
  }

  void deallocateStateImpl() override {
    qpp::ket tmp;
    state = tmp;
  }

  void applyGate(const GateApplicationTask &task) override {
    auto matrix = toQppMatrix(task.matrix, task.targets.size());
    // First, convert all of the qudit indices to big endian.
    std::vector<std::size_t> controls;
    for (auto index : task.controls) {
      controls.push_back(convertQuditIndex(index));
    }
    std::vector<std::size_t> targets;
    for (auto index : task.targets) {
      targets.push_back(convertQuditIndex(index));
    }

    if (controls.empty()) {
      state = qpp::apply(state, matrix, targets, levels);
      return;
    }
    state = qpp::applyCTRL(state, matrix, controls, targets, levels);
  }

  void setToZeroState() override {
    state = qpp::ket::Zero(stateDimension);
    state(0) = 1.0;
  }

  int measureQudit(const std::size_t index) override {
    const auto quditIdx = convertQuditIndex(index);
    // If here, then we care about the result digit, so compute it.
    const auto measurement_tuple =
        qpp::measure(state, qpp::cmat::Identity(levels, levels), {quditIdx},
                     /*qudit dimension=*/levels, /*destructive measmt=*/false);
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
    cudaq::info("Measured qudit {} -> {}", quditIdx, measurement_result);
    return measurement_result;
  }


public:
  QppPhotonicCircuitSimulator() {
    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();
  }

  virtual ~QppPhotonicCircuitSimulator() = default;

  void setRandomSeed(std::size_t seed) override {
    qpp::RandomDevices::get_instance().get_prng().seed(seed);
  }

  bool canHandleObserve() override { return false; }

  void resetQudit(const std::size_t index) override {
    flushGateQueue();
    const auto quditIdx = convertQuditIndex(index);
    state = qpp::reset(state, {quditIdx});
  }

  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qudits,
                                const int shots) override {

    std::vector<std::size_t> measuredDigits;
    for (auto index : qudits) {
      measuredDigits.push_back(convertQuditIndex(index));
    }

    auto sampleResult = qpp::sample(shots, state, measuredDigits, levels);
    // Convert to what we expect
    std::stringstream digit_stream;
    cudaq::ExecutionResult counts;

    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of digits to the digit_stream.
      for (const auto &digit : result) {
        digit_stream << digit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 digit_stream
      counts.appendResult(digit_stream.str(), count);
      // Reset the state.
      digit_stream.str("");
      digit_stream.clear();
    }

    return counts;
  }

  std::unique_ptr<cudaq::PhotonicState> getSimulationState() override {
    flushGateQueue();
    return std::make_unique<QppPhotonicState>(std::move(state), levels);
  }

  bool isStateVectorSimulator() const override {
    return std::is_same_v<StateType, qpp::ket>;
  }

  /// @brief Primarily used for testing.
  auto getStateVector() {
    flushGateQueue();
    return state;
  }

  std::string name() const override { return "qpp-photonics"; }

  NVQIR_PHOTONIC_SIMULATOR_CLONE_IMPL(QppPhotonicCircuitSimulator<StateType>)

}; // QppPhotonicCircuitSimulator

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_PHOTONIC_SIMULATOR(nvqir::QppPhotonicCircuitSimulator<qpp::ket>,
                                  photonics)
#endif