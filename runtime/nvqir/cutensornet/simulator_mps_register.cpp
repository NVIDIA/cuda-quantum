/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "simulator_cutensornet.h"
#include <charconv>
namespace nvqir {

class SimulatorMPS : public SimulatorTensorNetBase {
  // Default max bond dim
  int64_t m_maxBond = 64;
  // Default absolute cutoff
  double m_absCutoff = 1e-5;
  // Default relative cutoff
  double m_relCutoff = 1e-5;
  std::vector<void *> m_mpsTensors_d;
  std::vector<std::size_t> m_auxQubitsForGateDecomp;

public:
  SimulatorMPS() : SimulatorTensorNetBase() {
    if (auto *maxBondEnvVar = std::getenv("CUDAQ_MPS_MAX_BOND")) {
      const std::string maxBondStr(maxBondEnvVar);
      int maxBond;
      auto [ptr, ec] = std::from_chars(
          maxBondStr.data(), maxBondStr.data() + maxBondStr.size(), maxBond);
      if (ec != std::errc{} || maxBond < 1)
        throw std::runtime_error("Invalid CUDAQ_MPS_MAX_BOND setting. Expected "
                                 "a positive number. Got: " +
                                 maxBondStr);

      m_maxBond = maxBond;
      cudaq::info("Setting MPS max bond dimension to {}.", m_maxBond);
    }
    // Cutoff values
    if (auto *absCutoffEnvVar = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) {
      const std::string absCutoffStr(absCutoffEnvVar);
      double absCutoff;
      auto [ptr, ec] =
          std::from_chars(absCutoffStr.data(),
                          absCutoffStr.data() + absCutoffStr.size(), absCutoff);
      if (ec != std::errc{} || absCutoff <= 0.0 || absCutoff >= 1.0)
        throw std::runtime_error(
            "Invalid CUDAQ_MPS_ABS_CUTOFF setting. Expected "
            "a number in range (0.0, 1.0). Got: " +
            absCutoffStr);

      m_absCutoff = absCutoff;
      cudaq::info("Setting MPS absolute cutoff to {}.", m_absCutoff);
    }
    if (auto *relCutoffEnvVar = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) {
      const std::string relCutoffStr(relCutoffEnvVar);
      double relCutoff;
      auto [ptr, ec] =
          std::from_chars(relCutoffStr.data(),
                          relCutoffStr.data() + relCutoffStr.size(), relCutoff);
      if (ec != std::errc{} || relCutoff <= 0.0 || relCutoff >= 1.0)
        throw std::runtime_error(
            "Invalid CUDAQ_MPS_RELATIVE_CUTOFF setting. Expected "
            "a number in range (0.0, 1.0). Got: " +
            relCutoffStr);

      m_relCutoff = relCutoff;
      cudaq::info("Setting MPS relative cutoff to {}.", m_relCutoff);
    }
  }

  virtual void prepareQubitTensorState() override {
    LOG_API_TIME();
    // Clean up previously factorized MPS tensors
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor));
    }
    m_mpsTensors_d.clear();
    // Factorize the state:
    if (m_state->getNumQubits() > 1)
      m_mpsTensors_d =
          m_state->factorizeMPS(m_maxBond, m_absCutoff, m_relCutoff);
  }

  virtual void applyGate(const GateApplicationTask &task) override {
    // Check that we don't apply gates on 3+ qubits (not supported in MPS)
    if (task.controls.size() + task.targets.size() > 2) {
      const std::string gateDesc = task.operationName +
                                   containerToString(task.controls) +
                                   containerToString(task.targets);
      throw std::runtime_error("MPS simulator: Gates on 3 or more qubits are "
                               "unsupported. Encountered: " +
                               gateDesc);
    }
    SimulatorTensorNetBase::applyGate(task);
  }

  virtual std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  virtual std::string name() const override { return "tensornet-mps"; }
  CircuitSimulator *clone() override {
    thread_local static auto simulator = std::make_unique<SimulatorMPS>();
    return simulator.get();
  }
  virtual ~SimulatorMPS() noexcept {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor));
    }
    m_mpsTensors_d.clear();
  }

  void resetExecutionContext() override {
    SimulatorTensorNetBase::resetExecutionContext();
    m_auxQubitsForGateDecomp.clear();
  }

  /// @brief Return the state vector data
  cudaq::State getStateData() override {
    LOG_API_TIME();
    if (m_state->getNumQubits() - m_auxQubitsForGateDecomp.size() > 64)
      throw std::runtime_error("State vector data is too large.");
    // Handle empty state (e.g., no qubit allocation)
    if (!m_state)
      return cudaq::State{{0}, {}};
    const uint64_t svDim =
        (1ull << (m_state->getNumQubits() - m_auxQubitsForGateDecomp.size()));
    const std::vector<int32_t> projectedModes(m_auxQubitsForGateDecomp.begin(),
                                              m_auxQubitsForGateDecomp.end());
    return cudaq::State{{svDim}, m_state->getStateVector(projectedModes)};
  }

  size_t addAuxQubit() {
    if (m_state->isDirty())
      throw std::runtime_error(
          "[MPS Simulator] Unable to perform multi-control gate decomposition "
          "due to dynamical circuits.");
    m_state = std::make_unique<TensorNetState>(m_state->getNumQubits() + 1,
                                               m_cutnHandle);
    return m_state->getNumQubits() - 1;
  }

  template <typename QuantumOperation>
  void
  decomposeMultiControlledInstruction(const std::vector<double> &params,
                                      const std::vector<std::size_t> &controls,
                                      const std::vector<std::size_t> &targets) {
    if (controls.size() <= 1) {
      enqueueQuantumOperation<QuantumOperation>(params, controls, targets);
      return;
    }

    const auto ccnot = [&](std::size_t a, std::size_t b, std::size_t c) {
      enqueueQuantumOperation<nvqir::h<double>>({}, {}, {c});
      enqueueQuantumOperation<nvqir::x<double>>({}, {b}, {c});
      enqueueQuantumOperation<nvqir::tdg<double>>({}, {}, {c});
      enqueueQuantumOperation<nvqir::x<double>>({}, {a}, {c});
      enqueueQuantumOperation<nvqir::t<double>>({}, {}, {c});
      enqueueQuantumOperation<nvqir::x<double>>({}, {b}, {c});
      enqueueQuantumOperation<nvqir::tdg<double>>({}, {}, {c});
      enqueueQuantumOperation<nvqir::x<double>>({}, {a}, {c});
      enqueueQuantumOperation<nvqir::t<double>>({}, {}, {b});
      enqueueQuantumOperation<nvqir::t<double>>({}, {}, {c});
      enqueueQuantumOperation<nvqir::h<double>>({}, {}, {c});
      enqueueQuantumOperation<nvqir::x<double>>({}, {a}, {b});
      enqueueQuantumOperation<nvqir::t<double>>({}, {}, {a});
      enqueueQuantumOperation<nvqir::tdg<double>>({}, {}, {b});
      enqueueQuantumOperation<nvqir::x<double>>({}, {a}, {b});
    };

    const auto collectControls = [&](const std::vector<std::size_t> &ctls,
                                     const std::vector<std::size_t> &aux,
                                     int adjustment) {
      for (int i = 0; i < static_cast<int>(ctls.size()) - 1; i += 2) {
        ccnot(ctls[i], ctls[i + 1], aux[i / 2]);
      }
      for (int i = 0; i < static_cast<int>(ctls.size()) / 2 - 1 - adjustment;
           ++i) {
        ccnot(aux[i * 2], aux[(i * 2) + 1], aux[i + ctls.size() / 2]);
      }
    };
    const auto adjustForSingleControl =
        [&](const std::vector<std::size_t> &ctls,
            const std::vector<std::size_t> &aux) {
          if (ctls.size() % 2 != 0)
            ccnot(ctls[ctls.size() - 1], aux[ctls.size() - 3],
                  aux[ctls.size() - 2]);
        };

    std::vector<std::size_t> aux;
    for (std::size_t i = 0; i < controls.size() - 1; ++i) {
      const auto auxQubit = addAuxQubit();
      m_auxQubitsForGateDecomp.emplace_back(auxQubit);
      aux.emplace_back(auxQubit);
    }

    collectControls(controls, aux, 0);
    adjustForSingleControl(controls, aux);

    // Add to the singly-controlled instruction queue
    enqueueQuantumOperation<QuantumOperation>(
        params, {aux[controls.size() - 2]}, targets);

    adjustForSingleControl(controls, aux);
    collectControls(controls, aux, 0);
  }

#define CIRCUIT_SIMULATOR_ONE_QUBIT(NAME)                                      \
  using CircuitSimulator::NAME;                                                \
  void NAME(const std::vector<std::size_t> &controls,                          \
            const std::size_t qubitIdx) override {                             \
    decomposeMultiControlledInstruction<nvqir::NAME<double>>(                  \
        {}, controls, std::vector<std::size_t>{qubitIdx});                     \
  }

#define CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(NAME)                            \
  using CircuitSimulator::NAME;                                                \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::size_t qubitIdx) override {                             \
    decomposeMultiControlledInstruction<nvqir::NAME<double>>(                  \
        {angle}, controls, std::vector<std::size_t>{qubitIdx});                \
  }

  /// @brief The X gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(x)
  /// @brief The Y gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(y)
  /// @brief The Z gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(z)
  /// @brief The H gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(h)
  /// @brief The S gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(s)
  /// @brief The T gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(t)
  /// @brief The Sdg gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(sdg)
  /// @brief The Tdg gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(tdg)
  /// @brief The RX gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(rx)
  /// @brief The RY gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(ry)
  /// @brief The RZ gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(rz)
  /// @brief The Phase gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(r1)
// Undef those preprocessor defines.
#undef CIRCUIT_SIMULATOR_ONE_QUBIT
#undef CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM

  using CircuitSimulator::swap;
  void swap(const std::vector<std::size_t> &ctrlBits, const std::size_t srcIdx,
            const std::size_t tgtIdx) override {
    if (ctrlBits.empty())
      return SimulatorTensorNetBase::swap(ctrlBits, srcIdx, tgtIdx);
    {
      std::vector<std::size_t> ctls = ctrlBits;
      ctls.emplace_back(tgtIdx);
      decomposeMultiControlledInstruction<nvqir::x<double>>({}, ctls, {srcIdx});
    }
    {
      std::vector<std::size_t> ctls = ctrlBits;
      ctls.emplace_back(srcIdx);
      decomposeMultiControlledInstruction<nvqir::x<double>>({}, ctls, {tgtIdx});
    }
    {
      std::vector<std::size_t> ctls = ctrlBits;
      ctls.emplace_back(tgtIdx);
      decomposeMultiControlledInstruction<nvqir::x<double>>({}, ctls, {srcIdx});
    }
  }

  void applyExpPauli(double theta, const std::vector<std::size_t> &controls,
                     const std::vector<std::size_t> &qubitIds,
                     const cudaq::spin_op &op) override {
    if (op.is_identity()) {
      if (controls.empty()) {
        // exp(i*theta*Id) is noop if this is not a controlled gate.
        return;
      } else {
        // Throw an error if this exp_pauli(i*theta*Id) becomes a non-trivial
        // gate due to control qubits.
        // FIXME: revisit this once
        // https://github.com/NVIDIA/cuda-quantum/issues/483 is implemented.
        throw std::logic_error("Applying controlled global phase via exp_pauli "
                               "of identity operator is not supported");
      }
    }
    std::vector<std::size_t> qubitSupport;
    std::vector<std::function<void(bool)>> basisChange;
    op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
      if (type != cudaq::pauli::I)
        qubitSupport.push_back(qubitIds[qubitIdx]);

      if (type == cudaq::pauli::Y)
        basisChange.emplace_back([&, qubitIdx](bool reverse) {
          rx(!reverse ? M_PI_2 : -M_PI_2, qubitIds[qubitIdx]);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { h(qubitIds[qubitIdx]); });
    });

    if (!basisChange.empty())
      for (auto &basis : basisChange)
        basis(false);

    std::vector<std::pair<std::size_t, std::size_t>> toReverse;
    for (std::size_t i = 0; i < qubitSupport.size() - 1; i++) {
      x({qubitSupport[i]}, qubitSupport[i + 1]);
      toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
    }

    // Perform multi-control decomposition.
    decomposeMultiControlledInstruction<nvqir::rz<double>>(
        {-2.0 * theta}, controls, {qubitSupport.back()});

    std::reverse(toReverse.begin(), toReverse.end());
    for (auto &[i, j] : toReverse)
      x({i}, j);

    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);
    }
  }
};
} // end namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS, tensornet_mps)
