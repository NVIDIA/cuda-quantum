/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
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

  virtual ~SimulatorMPS() noexcept {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor));
    }
    m_mpsTensors_d.clear();
  }
};
} // end namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS, tensornet_mps)
