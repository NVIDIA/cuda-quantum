/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mps_simulation_state.h"
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
  std::vector<MPSTensor> m_mpsTensors_d;

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
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
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

  virtual void
  addQubitsToState(const cudaq::SimulationState &in_state) override {
    throw std::runtime_error("Not yet supported.");
  }

  virtual std::string name() const override { return "tensornet-mps"; }

  CircuitSimulator *clone() override {
    thread_local static auto simulator = std::make_unique<SimulatorMPS>();
    return simulator.get();
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    LOG_API_TIME();

    if (!m_state || m_state->getNumQubits() == 0)
      return std::make_unique<MPSSimulationState>(
          std::move(m_state), std::vector<MPSTensor>{}, m_cutnHandle);

    if (m_state->getNumQubits() > 1) {
      std::vector<MPSTensor> tensors =
          m_state->factorizeMPS(m_maxBond, m_absCutoff, m_relCutoff);
      return std::make_unique<MPSSimulationState>(std::move(m_state), tensors,
                                                  m_cutnHandle);
    }

    auto [d_tensor, numElements] = m_state->contractStateVectorInternal({});
    assert(numElements == 2);
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = {static_cast<int64_t>(numElements)};

    return std::make_unique<MPSSimulationState>(
        std::move(m_state), std::vector<MPSTensor>{stateTensor}, m_cutnHandle);
  }

  virtual ~SimulatorMPS() noexcept {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }
    m_mpsTensors_d.clear();
  }
};

} // end namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS, tensornet_mps)
