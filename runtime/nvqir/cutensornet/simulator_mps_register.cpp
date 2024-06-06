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

  virtual std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  static std::vector<std::complex<double>> generateXX(double theta) {
    const auto halfTheta = theta / 2.;
    const std::complex<double> cos = std::cos(halfTheta);
    const std::complex<double> isin = {0., std::sin(halfTheta)};
    // Row-major
    return {cos, 0.,    0.,  -isin, 0.,    cos, -isin, 0.,
            0.,  -isin, cos, 0.,    -isin, 0.,  0.,    cos};
  };

  static std::vector<std::complex<double>> generateYY(double theta) {
    const auto halfTheta = theta / 2.;
    const std::complex<double> cos = std::cos(halfTheta);
    const std::complex<double> isin = {0., std::sin(halfTheta)};
    // Row-major
    return {cos, 0.,    0.,  isin, 0.,   cos, -isin, 0.,
            0.,  -isin, cos, 0.,   isin, 0.,  0.,    cos};
  };

  static std::vector<std::complex<double>> generateZZ(double theta) {
    const std::complex<double> itheta2 = {0., theta / 2.0};
    const std::complex<double> exp_itheta2 = std::exp(itheta2);
    const std::complex<double> exp_minus_itheta2 = std::exp(-1.0 * itheta2);
    // Row-major
    return {exp_minus_itheta2, 0., 0., 0., 0., exp_itheta2,      0., 0., 0., 0.,
            exp_itheta2,       0., 0., 0., 0., exp_minus_itheta2};
  };

  virtual void applyExpPauli(double theta,
                             const std::vector<std::size_t> &controls,
                             const std::vector<std::size_t> &qubitIds,
                             const cudaq::spin_op &op) override {
    // Special handling for equivalence of Rxx(theta), Ryy(theta), Rzz(theta)
    // expressed as exp_pauli.
    //  Note: for MPS, the runtime is ~ linear with the number of 2-body gates
    //  (gate split procedure).
    // Hence, we check if this is a Rxx(theta), Ryy(theta), or Rzz(theta), which
    // are commonly-used gates and apply the operation directly (the base
    // decomposition will result in 2 CNOT gates).
    const auto shouldHandlePauliOp =
        [](const cudaq::spin_op &opToCheck) -> bool {
      const std::string opStr = opToCheck.to_string(false);
      return opStr == "XX" || opStr == "YY" || opStr == "ZZ";
    };
    if (controls.empty() && qubitIds.size() == 2 && shouldHandlePauliOp(op)) {
      flushGateQueue();
      cudaq::info("[SimulatorMPS] (apply) exp(i*{}*{}) ({}, {}).", theta,
                  op.to_string(false), qubitIds[0], qubitIds[1]);
      const GateApplicationTask task = [&]() {
        const std::string opStr = op.to_string(false);
        // Note: Rxx(angle) ==  exp(-i*angle/2 XX)
        // i.e., exp(i*theta XX) == Rxx(-2 * theta)
        if (opStr == "XX") {
          // Note: use a special name so that the gate matrix caching procedure
          // works properly.
          return GateApplicationTask("Rxx", generateXX(-2.0 * theta), {},
                                     qubitIds, {theta});
        } else if (opStr == "YY") {
          return GateApplicationTask("Ryy", generateYY(-2.0 * theta), {},
                                     qubitIds, {theta});
        } else if (opStr == "ZZ") {
          return GateApplicationTask("Rzz", generateZZ(-2.0 * theta), {},
                                     qubitIds, {theta});
        }
        __builtin_unreachable();
      }();
      applyGate(task);
      return;
    }
    // Let the base class to handle this Pauli rotation
    SimulatorTensorNetBase::applyExpPauli(theta, controls, qubitIds, op);
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

  /// @brief Return the state vector data
  cudaq::State getStateData() override {
    LOG_API_TIME();
    if (m_state->getNumQubits() > 64)
      throw std::runtime_error("State vector data is too large.");
    // Handle empty state (e.g., no qubit allocation)
    if (!m_state)
      return cudaq::State{{0}, {}};
    const std::uint64_t svDim = (1ull << m_state->getNumQubits());
    // Returns the main qubit register state (auxiliary qubits are projected to
    // zero state)
    return cudaq::State{{svDim}, m_state->getStateVector()};
  }
};
} // end namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS, tensornet_mps)
