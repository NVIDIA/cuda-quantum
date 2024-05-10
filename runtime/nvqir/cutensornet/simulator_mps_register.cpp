/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mps_simulation_state.h"
#include "simulator_cutensornet.h"

namespace nvqir {

class SimulatorMPS : public SimulatorTensorNetBase {
  MPSSettings m_settings;
  std::vector<MPSTensor> m_mpsTensors_d;

public:
  SimulatorMPS() : SimulatorTensorNetBase() {}

  virtual void prepareQubitTensorState() override {
    LOG_API_TIME();
    // Clean up previously factorized MPS tensors
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }
    m_mpsTensors_d.clear();
    // Factorize the state:
    if (m_state->getNumQubits() > 1)
      m_mpsTensors_d = m_state->factorizeMPS(
          m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff);
  }

  virtual std::size_t calculateStateDim(const std::size_t numQubits) override {
    return numQubits;
  }

  virtual void
  addQubitsToState(const cudaq::SimulationState &in_state) override {
    LOG_API_TIME();
    const MPSSimulationState *const casted =
        dynamic_cast<const MPSSimulationState *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[SimulatorMPS simulator] Incompatible state input");
    if (!m_state) {
      m_state = casted->reconstructBackendState();
    } else {
      // Expand an existing state:
      // Append MPS tensors
      throw std::runtime_error(
          "[SimulatorMPS simulator] Expanding state is not supported");
    }
  }

  virtual std::string name() const override { return "tensornet-mps"; }

  CircuitSimulator *clone() override {
    thread_local static auto simulator = std::make_unique<SimulatorMPS>();
    return simulator.get();
  }

  void addQubitsToState(std::size_t numQubits, const void *ptr) override {
    LOG_API_TIME();
    if (!m_state) {
      if (!ptr) {
        m_state = std::make_unique<TensorNetState>(numQubits, m_cutnHandle);
      } else {
        auto [state, mpsTensors] = MPSSimulationState::createFromStateVec(
            m_cutnHandle, 1ULL << numQubits,
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr)),
            m_settings.maxBond);
        m_state = std::move(state);
      }
    } else {
      // FIXME: expand the MPS tensors to the max extent
      if (!ptr) {
        auto tensors = m_state->factorizeMPS(
            m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff);
        // The right most MPS tensor needs to have one more extra leg (no longer
        // the boundary tensor).
        tensors.back().extents.emplace_back(1);
        // The newly added MPS tensors are in zero state
        constexpr std::complex<double> tensorBody[2]{1.0, 0.0};
        constexpr auto tensorSizeBytes = 2 * sizeof(std::complex<double>);
        for (std::size_t i = 0; i < numQubits; ++i) {
          const std::vector<int64_t> extents =
              (i != numQubits - 1) ? std::vector<int64_t>{1, 2, 1}
                                   : std::vector<int64_t>{1, 2};
          void *mpsTensor{nullptr};
          HANDLE_CUDA_ERROR(cudaMalloc(&mpsTensor, tensorSizeBytes));
          HANDLE_CUDA_ERROR(cudaMemcpy(mpsTensor, tensorBody, tensorSizeBytes,
                                       cudaMemcpyHostToDevice));
          tensors.emplace_back(MPSTensor(mpsTensor, extents));
        }
        m_state = TensorNetState::createFromMpsTensors(tensors, m_cutnHandle);
      } else {
        // Non-zero state needs to be factorized and appended.
        auto [state, mpsTensors] = MPSSimulationState::createFromStateVec(
            m_cutnHandle, 1ULL << numQubits,
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr)),
            m_settings.maxBond);
        auto tensors = m_state->factorizeMPS(
            m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff);
        // Adjust the extents of the last tensor in the original state
        tensors.back().extents.emplace_back(1);

        // Adjust the extents of the first tensor in the state to be appended.
        auto extents = mpsTensors.front().extents;
        extents.insert(extents.begin(), 1);
        mpsTensors.front().extents = extents;
        // Combine the list
        tensors.insert(tensors.end(), mpsTensors.begin(), mpsTensors.end());
        m_state = TensorNetState::createFromMpsTensors(tensors, m_cutnHandle);
      }
    }
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    LOG_API_TIME();

    if (!m_state || m_state->getNumQubits() == 0)
      return std::make_unique<MPSSimulationState>(
          std::move(m_state), std::vector<MPSTensor>{}, m_cutnHandle);

    if (m_state->getNumQubits() > 1) {
      std::vector<MPSTensor> tensors = m_state->factorizeMPS(
          m_settings.maxBond, m_settings.absCutoff, m_settings.relCutoff);
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
