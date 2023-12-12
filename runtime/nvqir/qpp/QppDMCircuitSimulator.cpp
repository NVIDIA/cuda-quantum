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
      state = qpp::apply(state, K, qubits);
    }
  }

  /// @brief Grow the density matrix by one qubit.
  void addQubitToState() override {
    // Update the state vector
    if (state.size() == 0) {
      state = qpp::cmat::Zero(stateDimension, stateDimension);
      state(0, 0) = 1.0;
      return;
    }

    state.conservativeResize(stateDimension, stateDimension);
    for (std::size_t i = previousStateDimension; i < stateDimension; i++) {
      state.col(i).setZero();
      state.row(i).setZero();
    }
  }

  void addQubitsToState(std::size_t count) override {
    if (count == 0)
      return;

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      state = qpp::cmat::Zero(stateDimension, stateDimension);
      state(0, 0) = 1.0;
      return;
    }

    state.conservativeResize(stateDimension, stateDimension);
    for (std::size_t i = previousStateDimension; i < stateDimension; i++) {
      state.col(i).setZero();
      state.row(i).setZero();
    }

    return;
  }

  void setToZeroState() override {
    state = qpp::cmat::Zero(stateDimension, stateDimension);
    state(0, 0) = 1.0;
  }

public:
  QppNoiseCircuitSimulator() = default;
  virtual ~QppNoiseCircuitSimulator() = default;
  std::string name() const override { return "dm"; }

  cudaq::State getStateData() override {
    flushGateQueue();
    // There has to be at least one copy
    return cudaq::State{{stateDimension, stateDimension},
                        {state.data(), state.data() + state.size()}};
  }

  NVQIR_SIMULATOR_CLONE_IMPL(QppNoiseCircuitSimulator)
};

} // namespace

/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(QppNoiseCircuitSimulator, dm)
#undef __NVQIR_QPP_TOGGLE_CREATE
