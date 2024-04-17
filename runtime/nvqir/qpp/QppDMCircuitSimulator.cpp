/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define __NVQIR_QPP_TOGGLE_CREATE

#include "QppCircuitSimulator.cpp"

using namespace cudaq;

namespace {

/// @brief QppDmState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Qpp Density Matrix Circuit Simulator.
struct QppDmState : public cudaq::SimulationState {
  /// @brief The state.
  qpp::cmat state;

  QppDmState(qpp::cmat &&data) : state(std::move(data)) {}
  QppDmState(const std::vector<std::size_t> &shape,
             const std::vector<std::complex<double>> &data) {
    if (shape.size() != 2)
      throw std::runtime_error(
          "QppDmState must be created from data with 2D shape.");

    state = Eigen::Map<qpp::ket>(
        const_cast<std::complex<double> *>(data.data()), shape[0], shape[1]);
  }
  std::size_t getNumQubits() const override { return std::log2(state.rows()); }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (other.getNumTensors() != 1 ||
        (other.getTensor().extents != getTensor().extents))
      throw std::runtime_error("[qpp-dm-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        state.data(), getTensor().extents[0], getTensor().extents[1]);
    Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
        reinterpret_cast<std::complex<double> *>(other.getTensor().data),
        other.getTensor().extents[0], other.getTensor().extents[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          fmt::format("[qpp-dm-state] getAmplitude with an invalid number "
                      "of bits in the "
                      "basis state: expected {}, provided {}.",
                      getNumQubits(), basisState.size()));
    if (std::any_of(basisState.begin(), basisState.end(),
                    [](int x) { return x != 0 && x != 1; }))
      throw std::runtime_error(
          "[qpp-dm-state] getAmplitude with an invalid basis state: only "
          "qubit state (0 or 1) is supported.");

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    // Returns the diagonal element.
    // Notes:
    //  (1) This is considered a 'probability amplitude' in the (generally)
    //  mixed state context represented by this density matrix.
    //  (2) For pure states (i.e., Tr(rho^2) == 1), e.g., using the density
    //  matrix simulator without noise,
    // we can refactor the density matrix back to the state vector if needed.
    // This is however not a common use case for the density matrix simulator.
    return state(idx, idx);
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[qpp-dm-state] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(
            const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.rows()),
                                 static_cast<std::size_t>(state.cols())},
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
      throw std::runtime_error("[qpp-dm-state] invalid tensor requested.");
    if (indices.size() != 2)
      throw std::runtime_error("[qpp-dm-state] invalid element extraction.");

    return state(indices[0], indices[1]);
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    return std::make_unique<QppDmState>(
        Eigen::Map<qpp::cmat>(reinterpret_cast<std::complex<double> *>(ptr),
                              std::sqrt(size), std::sqrt(size)));
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
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

  void addQubitsToState(std::size_t count,
                        const void *data = nullptr) override {
    if (count == 0)
      return;

    if (data != nullptr)
      throw std::runtime_error("init state not implemented for dm sim");

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
