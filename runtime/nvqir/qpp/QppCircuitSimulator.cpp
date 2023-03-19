/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "CircuitSimulator.h"
#include "Gates.h"
#include "qpp.h"
#include <iostream>

namespace nvqir {

/// @brief The QppCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
template <typename StateType>
class QppCircuitSimulator : public nvqir::CircuitSimulator {
protected:
  /// The QPP state representation (qpp::ket or qpp::cmat)
  StateType state;

  /// @brief Provide a base-class method that can be invoked
  /// after every gate application and will apply any noise
  /// channels after the gate invocation based on a user-provided noise
  /// model. Unimplemented on the base class, sub-types can implement noise
  /// modeling.
  virtual void applyNoiseChannel(const std::string_view gateName,
                                 const std::vector<std::size_t> &qubits) {}

  /// @brief Compute the expectation value <Z...Z> over the given qubit indices.
  /// @param qubit_indices
  /// @return expectation
  double
  calculateExpectationValue(const std::vector<std::size_t> &qubit_indices) {
    const auto hasEvenParity =
        [](std::size_t x,
           const std::vector<std::size_t> &in_qubitIndices) -> bool {
      size_t count = 0;
      for (const auto &bitIdx : in_qubitIndices) {
        if (x & (1ULL << bitIdx)) {
          count++;
        }
      }
      return (count % 2) == 0;
    };

    // First, convert all of the qubit indices to big endian.
    std::vector<std::size_t> casted_qubit_indices;
    for (auto index : qubit_indices) {
      casted_qubit_indices.push_back(bigEndian(nQubitsAllocated, index));
    }

    double result = 0.0;
    if constexpr (std::is_same_v<StateType, qpp::ket>) {
#pragma omp parallel for reduction(+ : result)
      for (std::size_t i = 0; i < stateDimension; ++i) {
        result += (hasEvenParity(i, casted_qubit_indices) ? 1.0 : -1.0) *
                  std::norm(state[i]);
      }
    } else if constexpr (std::is_same_v<StateType, qpp::cmat>) {
      Eigen::VectorXcd diag = state.diagonal();
#pragma omp parallel for reduction(+ : result)
      for (Eigen::Index i = 0; i < state.rows(); i++) {
        auto element = diag(i).real();
        if (!hasEvenParity(i, casted_qubit_indices))
          element *= -1.;
        result += element;
      }
    }

    return result;
  }

  qpp::cmat toQppMatrix(std::vector<std::complex<double>> &&data) {
    assert(data.size() == 4 &&
           "Invalid number of gate matrix elements passed to toQppMatrix");

    // we represent row major, they represent column major
    return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                    Eigen::Dynamic, Eigen::RowMajor>>(
        data.data(), 2, 2);
  }

  /// @brief Utility function for applying one-target-qubit operations with
  /// optional control qubits
  /// @tparam GateT The instruction type, must be QppInstruction derived
  /// @param controls The control qubits, can be empty
  /// @param qubitIdx The target qubit
  template <typename GateT>
  void oneQubitApply(const std::vector<std::size_t> &controls,
                     const std::size_t qubitIdx) {
    GateT gate;
    cudaq::info(gateToString(gate.name(), controls, {}, {qubitIdx}));
    auto matrix = toQppMatrix(gate.getGate());
    state = qpp::applyCTRL(state, matrix, controls, {qubitIdx});
    std::vector<std::size_t> noiseQubits{controls.begin(), controls.end()};
    noiseQubits.push_back(qubitIdx);
    applyNoiseChannel(gate.name(), noiseQubits);
  }

  /// @brief Utility function for applying one-target-qubit rotation operations
  /// @tparam RotationGateT The instruction type, must be QppInstruction derived
  /// @param angle The rotation angle
  /// @param controls The control qubits, can be empty
  /// @param qubitIdx The target qubit
  template <typename RotationGateT>
  void oneQubitOneParamApply(const double angle,
                             const std::vector<std::size_t> &controls,
                             const std::size_t qubitIdx) {
    RotationGateT gate;
    cudaq::info(gateToString(gate.name(), controls, {angle}, {qubitIdx}));
    auto matrix = toQppMatrix(gate.getGate(angle));
    state = qpp::applyCTRL(state, matrix, controls, {qubitIdx});
    std::vector<std::size_t> noiseQubits{controls.begin(), controls.end()};
    noiseQubits.push_back(qubitIdx);
    applyNoiseChannel(gate.name(), noiseQubits);
  }

  /// @brief Grow the state vector by one qubit.
  void addQubitToState() override {
    // Update the state vector
    if (state.size() == 0) {
      // If this is the first time, allocate the state
      state = qpp::ket::Zero(stateDimension);
      state(0) = 1.0;
    } else {
      // If we are resizing an existing, allocate
      // a zero state on a single qubit, and Kron-prod
      // that with the existing state.
      qpp::ket zero_state = qpp::ket::Zero(2);
      zero_state(0) = 1.0;
      state = qpp::kron(state, zero_state);
    }
  }

  /// @brief Reset the qubit state.
  void resetQubitStateImpl() override {
    StateType tmp;
    state = tmp;
  }

public:
  QppCircuitSimulator() = default;
  virtual ~QppCircuitSimulator() = default;

/// The one-qubit overrides
#define QPP_ONE_QUBIT_METHOD_OVERRIDE(NAME)                                    \
  using CircuitSimulator::NAME;                                                \
  virtual void NAME(const std::vector<std::size_t> &controls,                  \
                    const std::size_t qubitIdx) override {                     \
    oneQubitApply<nvqir::NAME<double>>(controls, qubitIdx);                    \
  }

  QPP_ONE_QUBIT_METHOD_OVERRIDE(x)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(y)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(z)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(h)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(s)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(t)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(sdg)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(tdg)

/// The one-qubit parameterized overrides
#define QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(NAME)                          \
  using CircuitSimulator::NAME;                                                \
  virtual void NAME(const double angle,                                        \
                    const std::vector<std::size_t> &controls,                  \
                    const std::size_t qubitIdx) override {                     \
    oneQubitOneParamApply<nvqir::NAME<double>>(angle, controls, qubitIdx);     \
  }

  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(rx)
  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(ry)
  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(rz)
  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(r1)
  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(u1)

  /// @brief U2 operation
  using CircuitSimulator::u2;
  void u2(const double phi, const double lambda,
          const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    cudaq::info(gateToString("u3", controls, {phi, lambda}, {qubitIdx}));
    qpp::cmat matrix = qpp::cmat::Zero(2, 2);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = -1.0 * std::exp(nvqir::im<> * lambda);
    matrix(1, 0) = std::exp(nvqir::im<> * phi);
    matrix(1, 1) = std::exp(nvqir::im<> * (phi + lambda));
    state = qpp::applyCTRL(state, matrix, controls, {qubitIdx});
    std::vector<std::size_t> noiseQubits{controls.begin(), controls.end()};
    noiseQubits.push_back(qubitIdx);
    applyNoiseChannel("u2", noiseQubits);
  }

  /// @brief U3 operation
  using CircuitSimulator::u3;
  void u3(const double theta, const double phi, const double lambda,
          const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    cudaq::info(gateToString("u3", controls, {theta, phi, lambda}, {qubitIdx}));
    qpp::cmat matrix = qpp::cmat::Zero(2, 2);
    matrix(0, 0) = std::cos(theta / 2);
    matrix(0, 1) = std::exp(nvqir::im<> * phi) * std::sin(theta / 2);
    matrix(1, 0) = -1. * std::exp(nvqir::im<> * lambda) * std::sin(theta / 2);
    matrix(1, 1) = std::exp(nvqir::im<> * (phi + lambda)) * std::cos(theta / 2);
    state = qpp::applyCTRL(state, matrix, controls, {qubitIdx});
    std::vector<std::size_t> noiseQubits{controls.begin(), controls.end()};
    noiseQubits.push_back(qubitIdx);
    applyNoiseChannel("u3", noiseQubits);
  }

  /// @brief Swap operation
  using CircuitSimulator::swap;
  void swap(const std::vector<std::size_t> &ctrlBits, const std::size_t srcIdx,
            const std::size_t tgtIdx) override {
    cudaq::info(gateToString("swap", ctrlBits, {}, {srcIdx, tgtIdx}));
    state = qpp::applyCTRL(state, qpp::Gates::get_instance().SWAP, ctrlBits,
                           {srcIdx, tgtIdx});
    std::vector<std::size_t> noiseQubits{ctrlBits.begin(), ctrlBits.end()};
    noiseQubits.push_back(srcIdx);
    noiseQubits.push_back(tgtIdx);
    applyNoiseChannel("swap", noiseQubits);
  }

  /// @brief Measure the qubit and return the result. Collapse the
  /// state vector.
  bool measureQubit(const std::size_t qubitIdx) override {
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
    cudaq::info("Measured qubit {} -> {}", qubitIdx, measurement_result);
    return measurement_result == 1 ? true : false;
  }

  /// @brief Reset the qubit
  /// @param qubitIdx
  void resetQubit(const std::size_t qubitIdx) override {
    state = qpp::reset(state, {qubitIdx});
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
    double expectationValue = calculateExpectationValue(measuredBits);
    if (shots < 1) {
      cudaq::info("Computed expectation value = {}", expectationValue);
      return cudaq::ExecutionResult{{}, expectationValue};
    }

    auto sampleResult = qpp::sample(shots, state, measuredBits, 2);
    // Convert to what we expect
    std::stringstream bitstring;
    cudaq::ExecutionResult counts(expectationValue);

    for (auto [result, count] : sampleResult) {
      // Push back each term in the vector of bits to the bitstring.
      for (const auto &bit : result) {
        bitstring << bit;
      }

      // Add to the sample result
      // in mid-circ sampling mode this will append 1 bitstring
      counts.appendResult(bitstring.str(), count);

      // Reset the state.
      bitstring.str("");
      bitstring.clear();
    }

    return counts;
  }

  cudaq::State getStateData() override {
    // There has to be at least one copy
    return cudaq::State{{stateDimension},
                        {state.data(), state.data() + state.size()}};
  }

  /// @brief Primarily used for testing.
  auto getStateVector() { return state; }
  std::string name() const override { return "qpp"; }
  NVQIR_SIMULATOR_CLONE_IMPL(QppCircuitSimulator<StateType>)
};

} // namespace nvqir

#ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(nvqir::QppCircuitSimulator<qpp::ket>, qpp)
#endif
