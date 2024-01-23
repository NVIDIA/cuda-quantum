/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define __NVQIR_QPP_TOGGLE_CREATE
#include "QppCircuitSimulator.cpp"

namespace {

/// @brief QppState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Qpp Circuit Simulator.
struct QppDmState : public cudaq::SimulationState {
  /// @brief The state.
  qpp::cmat state;

  QppDmState(qpp::cmat &&data) : state(std::move(data)) {}
  QppDmState(const std::vector<std::size_t> &shape,
             const std::vector<std::complex<double>> &data) {
    if (shape.size() != 2)
      throw std::runtime_error(
          "QppDmState must be created from data with 2D shape.");

    state = Eigen::Map<qpp::ket>(const_cast<cudaq::complex *>(data.data()),
                                 shape[0], shape[1]);
  }
  std::size_t getNumQubits() const override { return std::log2(state.rows()); }

  std::vector<std::size_t> getDataShape() const override {
    return {static_cast<std::size_t>(state.rows()),
            static_cast<std::size_t>(state.cols())};
  }

  double overlap(const cudaq::SimulationState &other) override {
    if (other.getDataShape() != getDataShape())
      throw std::runtime_error("[qpp-dm-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
        reinterpret_cast<cudaq::complex *>(other.ptr()),
        other.getDataShape()[0], other.getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(const std::vector<cudaq::complex> &other) override {
    if (other.size() != getDataShape()[0] * getDataShape()[1])
      throw std::runtime_error("[qpp-dm-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma =
        Eigen::Map<Eigen::MatrixXcd>(const_cast<cudaq::complex *>(other.data()),
                                     getDataShape()[0], getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(void *other) override {

    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma =
        Eigen::Map<Eigen::MatrixXcd>(reinterpret_cast<cudaq::complex *>(other),
                                     getDataShape()[0], getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  cudaq::complex matrixElement(std::size_t i, std::size_t j) override {
    return state(i, j);
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void *ptr() const override {
    return reinterpret_cast<void *>(const_cast<cudaq::complex *>(state.data()));
  }

  void destroyState() override {
    cudaq::info("qpp-dm-state destroying state vector handle.");
    qpp::cmat k;
    state = k;
  }
};

/// @brief The QppNoiseCircuitSimulator further specializes the
/// QppCircuitSimulator to use a density matrix representation of the state.
/// This class directly enables a simple noise modeling capability for CUDA
/// Quantum.
class QppNoiseCircuitSimulator : public nvqir::QppCircuitSimulator<qpp::cmat> {

protected:
  /// @brief If we have a noise model, apply any user-specified
  /// kraus_channels for the given gate name on the provided qubits.
  /// @param gateName
  /// @param qubits
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &qubits) override {
    // Do nothing if no execution context
    if (!executionContext)
      return;

    // Do nothing if no noise model
    if (!executionContext->noiseModel)
      return;

    // Get the name as a string
    std::string gName(gateName);

    std::vector<std::size_t> casted_qubits;
    for (auto index : qubits) {
      casted_qubits.push_back(convertQubitIndex(index));
    }

    // Get the Kraus channels specified for this gate and qubits
    auto krausChannels =
        executionContext->noiseModel->get_channels(gName, qubits);

    // If none, do nothing
    if (krausChannels.empty())
      return;

    cudaq::info("Applying {} kraus channels to qubits {}", krausChannels.size(),
                qubits);

    for (auto &channel : krausChannels) {
      // Map our kraus ops to the qpp::cmat
      std::vector<qpp::cmat> K;
      auto ops = channel.get_ops();
      std::transform(
          ops.begin(), ops.end(), std::back_inserter(K), [&](auto &el) {
            return Eigen::Map<qpp::cmat>(el.data.data(), el.nRows, el.nCols);
          });

      // Apply K rho Kdag
      state = qpp::apply(state, K, casted_qubits);
    }
  }

  /// @brief Grow the density matrix by one qubit.
  void addQubitToState() override { addQubitsToState(1); }

  void addQubitsToState(std::size_t count) override {
    if (count == 0)
      return;

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      state = qpp::cmat::Zero(stateDimension, stateDimension);
      state(0, 0) = 1.0;
      return;
    }

    qpp::cmat zero_state = qpp::cmat::Zero(1 << count, 1 << count);
    zero_state(0, 0) = 1.0;
    state = qpp::kron(zero_state, state);
  }

  void setToZeroState() override {
    state = qpp::cmat::Zero(stateDimension, stateDimension);
    state(0, 0) = 1.0;
  }

public:
  QppNoiseCircuitSimulator() = default;
  virtual ~QppNoiseCircuitSimulator() = default;
  std::string name() const override { return "dm"; }

  std::unique_ptr<cudaq::SimulationState> createSimulationState(
      const std::vector<std::size_t> &shape,
      const std::vector<std::complex<double>> &data) const override {
    return std::make_unique<QppDmState>(shape, data);
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    return std::make_unique<QppDmState>(std::move(state));
  }

  NVQIR_SIMULATOR_CLONE_IMPL(QppNoiseCircuitSimulator)
};

} // namespace

/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(QppNoiseCircuitSimulator, dm)
#undef __NVQIR_QPP_TOGGLE_CREATE
