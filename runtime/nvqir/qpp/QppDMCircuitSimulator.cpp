/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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

    auto prevDim = 1UL << (nQubitsAllocated - 1);
    state.conservativeResize(stateDimension, stateDimension);
    for (std::size_t i = prevDim; i < stateDimension; i++) {
      state.col(i).setZero();
      state.row(i).setZero();
    }
  }

public:
  QppNoiseCircuitSimulator() = default;
  virtual ~QppNoiseCircuitSimulator() = default;
  std::string name() const override { return "dm"; }

  /// @brief Override the default sized allocation of qubits
  /// here to be a bit more efficient than the default implementation
  std::vector<std::size_t> allocateQubits(const std::size_t count) override {
    std::vector<std::size_t> qubits;
    for (std::size_t i = 0; i < count; i++)
      qubits.emplace_back(tracker.getNextIndex());

    if (state.size() == 0) {
      // If this is the first time, allocate the state
      nQubitsAllocated += count;
      stateDimension = calculateStateDim(nQubitsAllocated);
      state = qpp::cmat::Zero(stateDimension, stateDimension);
      state(0, 0) = 1.0;
      return qubits;
    }

    auto oldNQ = nQubitsAllocated;
    nQubitsAllocated += count;
    stateDimension = calculateStateDim(nQubitsAllocated);

    auto prevDim = 1UL << oldNQ;
    state.conservativeResize(stateDimension, stateDimension);
    for (std::size_t i = prevDim; i < stateDimension; i++) {
      state.col(i).setZero();
      state.row(i).setZero();
    }
    return qubits;
  }

  cudaq::State getStateData() override {
    flushGateQueue();
    // There has to be at least one copy
    return cudaq::State{{stateDimension, stateDimension},
                        {state.data(), state.data() + state.size()}};
  }
};

} // namespace

/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(QppNoiseCircuitSimulator, dm)
#undef __NVQIR_QPP_TOGGLE_CREATE
