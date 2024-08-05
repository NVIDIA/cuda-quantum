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
#include <errno.h>

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
      m_mpsTensors_d =
          m_state->factorizeMPS(m_settings.maxBond, m_settings.absCutoff,
                                m_settings.relCutoff, m_settings.svdAlgo);
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
      m_state = TensorNetState::createFromMpsTensors(casted->getMpsTensors(),
                                                     scratchPad, m_cutnHandle);
    } else {
      // Expand an existing state: Append MPS tensors
      // Factor the existing state
      auto tensors =
          m_state->factorizeMPS(m_settings.maxBond, m_settings.absCutoff,
                                m_settings.relCutoff, m_settings.svdAlgo);
      // The right most MPS tensor needs to have one more extra leg (no longer
      // the boundary tensor).
      tensors.back().extents.emplace_back(1);
      auto mpsTensors = casted->getMpsTensors();
      for (std::size_t i = 0; i < mpsTensors.size(); ++i) {
        auto &tensor = mpsTensors[i];
        std::vector<int64_t> extents = tensor.extents;
        if (i == 0) {
          // First tensor: add a bond (dim 1 since no entanglement) to the
          // existing state
          extents.insert(extents.begin(), 1);
        }
        const auto numElements =
            std::reduce(extents.begin(), extents.end(), 1, std::multiplies());
        const auto tensorSizeBytes = sizeof(std::complex<double>) * numElements;
        void *mpsTensor{nullptr};
        HANDLE_CUDA_ERROR(cudaMalloc(&mpsTensor, tensorSizeBytes));
        HANDLE_CUDA_ERROR(cudaMemcpy(mpsTensor, tensor.deviceData,
                                     tensorSizeBytes, cudaMemcpyDefault));
        tensors.emplace_back(MPSTensor(mpsTensor, extents));
      }
      m_state = TensorNetState::createFromMpsTensors(tensors, scratchPad,
                                                     m_cutnHandle);
    }
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

  void addQubitsToState(std::size_t numQubits, const void *ptr) override {
    LOG_API_TIME();
    if (!m_state) {
      if (!ptr) {
        m_state = std::make_unique<TensorNetState>(numQubits, scratchPad,
                                                   m_cutnHandle);
      } else {
        auto [state, mpsTensors] = MPSSimulationState::createFromStateVec(
            m_cutnHandle, scratchPad, 1ULL << numQubits,
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr)),
            m_settings.maxBond);
        m_state = std::move(state);
      }
    } else {
      if (!ptr) {
        auto tensors =
            m_state->factorizeMPS(m_settings.maxBond, m_settings.absCutoff,
                                  m_settings.relCutoff, m_settings.svdAlgo);
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
        m_state = TensorNetState::createFromMpsTensors(tensors, scratchPad,
                                                       m_cutnHandle);
      } else {
        // Non-zero state needs to be factorized and appended.
        auto [state, mpsTensors] = MPSSimulationState::createFromStateVec(
            m_cutnHandle, scratchPad, 1ULL << numQubits,
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr)),
            m_settings.maxBond);
        auto tensors =
            m_state->factorizeMPS(m_settings.maxBond, m_settings.absCutoff,
                                  m_settings.relCutoff, m_settings.svdAlgo);
        // Adjust the extents of the last tensor in the original state
        tensors.back().extents.emplace_back(1);

        // Adjust the extents of the first tensor in the state to be appended.
        auto extents = mpsTensors.front().extents;
        extents.insert(extents.begin(), 1);
        mpsTensors.front().extents = extents;
        // Combine the list
        tensors.insert(tensors.end(), mpsTensors.begin(), mpsTensors.end());
        m_state = TensorNetState::createFromMpsTensors(tensors, scratchPad,
                                                       m_cutnHandle);
      }
    }
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    LOG_API_TIME();

    if (!m_state || m_state->getNumQubits() == 0)
      return std::make_unique<MPSSimulationState>(std::move(m_state),
                                                  std::vector<MPSTensor>{},
                                                  scratchPad, m_cutnHandle);

    if (m_state->getNumQubits() > 1) {
      std::vector<MPSTensor> tensors =
          m_state->factorizeMPS(m_settings.maxBond, m_settings.absCutoff,
                                m_settings.relCutoff, m_settings.svdAlgo);
      return std::make_unique<MPSSimulationState>(std::move(m_state), tensors,
                                                  scratchPad, m_cutnHandle);
    }

    auto [d_tensor, numElements] = m_state->contractStateVectorInternal({});
    assert(numElements == 2);
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = {static_cast<int64_t>(numElements)};

    return std::make_unique<MPSSimulationState>(
        std::move(m_state), std::vector<MPSTensor>{stateTensor}, scratchPad,
        m_cutnHandle);
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
