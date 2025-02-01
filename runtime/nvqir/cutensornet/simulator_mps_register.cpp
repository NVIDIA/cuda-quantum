/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
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
      m_state = TensorNetState::createFromMpsTensors(
          casted->getMpsTensors(), scratchPad, m_cutnHandle, m_randomEngine);
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
      m_state = TensorNetState::createFromMpsTensors(
          tensors, scratchPad, m_cutnHandle, m_randomEngine);
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

  // Helper to compute expectation value from a bit string distribution
  static double computeExpValFromDistribution(
      const std::unordered_map<std::string, std::size_t> &distribution,
      int shots) {
    double expVal = 0.0;
    // Compute the expectation value from the distribution
    for (auto &kv : distribution) {
      auto par = cudaq::sample_result::has_even_parity(kv.first);
      auto p = kv.second / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
    }
    return expVal;
  };

  // Set up the MPS factorization before trajectory simulation run loop.
  // We only need to do cutensornetStateFinalizeMPS once
  void setUpFactorizeForTrajectoryRuns() {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }
    m_mpsTensors_d.clear();
    m_mpsTensors_d =
        m_state->setupMPSFactorize(m_settings.maxBond, m_settings.absCutoff,
                                   m_settings.relCutoff, m_settings.svdAlgo);
  }

  /// @brief Sample a subset of qubits
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
    const bool hasNoise = executionContext && executionContext->noiseModel;
    if (!hasNoise || shots < 1)
      return SimulatorTensorNetBase::sample(measuredBits, shots);

    LOG_API_TIME();
    cudaq::ExecutionResult counts;
    std::vector<int32_t> measuredBitIds(measuredBits.begin(),
                                        measuredBits.end());

    setUpFactorizeForTrajectoryRuns();
    std::map<std::vector<int64_t>, std::pair<cutensornetStateSampler_t,
                                             cutensornetWorkspaceDescriptor_t>>
        samplerCache;
    for (int i = 0; i < shots; ++i) {
      // As the Kraus operator sampling may change the MPS state, we need to
      // re-compute the factorization in each trajectory.
      m_state->computeMPSFactorize(m_mpsTensors_d);
      std::vector<int64_t> samplerKey;
      for (const auto &tensor : m_mpsTensors_d)
        samplerKey.insert(samplerKey.end(), tensor.extents.begin(),
                          tensor.extents.end());

      auto iter = samplerCache.find(samplerKey);
      if (iter == samplerCache.end()) {
        const auto [itInsert, success] = samplerCache.insert(
            {samplerKey, m_state->prepareSample(measuredBitIds)});
        assert(success);
        iter = itInsert;
      }

      assert(iter != samplerCache.end());
      auto &[sampler, workDesc] = iter->second;
      const auto samples = m_state->executeSample(
          sampler, workDesc, measuredBitIds, 1, requireCacheWorkspace());
      assert(samples.size() == 1);
      for (const auto &[bitString, count] : samples)
        counts.appendResult(bitString, count);
    }

    for (const auto &[k, v] : samplerCache) {
      auto &[sampler, workDesc] = v;
      // Destroy the workspace descriptor
      HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
      // Destroy the quantum circuit sampler
      HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
    }

    counts.expectationValue =
        computeExpValFromDistribution(counts.counts, shots);

    return counts;
  }

  // Helper to prepare term-by-term data from a spin op
  static std::tuple<std::vector<std::string>,
                    std::vector<cudaq::spin_op::spin_op_term>,
                    std::vector<std::complex<double>>>
  prepareSpinOpTermData(const cudaq::spin_op &ham) {
    std::vector<std::string> termStrs;
    std::vector<cudaq::spin_op::spin_op_term> terms;
    std::vector<std::complex<double>> coeffs;
    termStrs.reserve(ham.num_terms());
    terms.reserve(ham.num_terms());
    coeffs.reserve(ham.num_terms());

    // Note: as the spin_op terms are stored as an unordered map, we need to
    // iterate in one loop to collect all the data (string, symplectic data, and
    // coefficient).
    ham.for_each_term([&](cudaq::spin_op &term) {
      termStrs.emplace_back(term.to_string(false));
      auto [symplecticRep, coeff] = term.get_raw_data();
      if (symplecticRep.size() != 1 || coeff.size() != 1)
        throw std::runtime_error(fmt::format(
            "Unexpected data encountered when iterating spin operator terms: "
            "expecting a single term, got {} terms.",
            symplecticRep.size()));
      terms.emplace_back(symplecticRep[0]);
      coeffs.emplace_back(coeff[0]);
    });
    return std::make_tuple(termStrs, terms, coeffs);
  }

  cudaq::observe_result observe(const cudaq::spin_op &ham) override {
    LOG_API_TIME();
    const bool hasNoise = executionContext && executionContext->noiseModel;
    // If no noise, just use base class implementation.
    if (!hasNoise)
      return SimulatorTensorNetBase::observe(ham);

    setUpFactorizeForTrajectoryRuns();
    const std::size_t numObserveTrajectories =
        this->executionContext->numberTrajectories.has_value()
            ? this->executionContext->numberTrajectories.value()
            : TensorNetState::g_numberTrajectoriesForObserve;

    auto [termStrs, terms, coeffs] = prepareSpinOpTermData(ham);
    std::vector<std::complex<double>> termExpVals(terms.size(), 0.0);

    for (std::size_t i = 0; i < numObserveTrajectories; ++i) {
      // As the Kraus operator sampling may change the MPS state, we need to
      // re-compute the factorization in each trajectory.
      m_state->computeMPSFactorize(m_mpsTensors_d);
      // We run a single trajectory for MPS as the final MPS form depends on the
      // randomly-selected noise op.
      const auto trajTermExpVals = m_state->computeExpVals(terms, 1);

      for (std::size_t idx = 0; idx < terms.size(); ++idx) {
        termExpVals[idx] += (trajTermExpVals[idx] /
                             static_cast<double>(numObserveTrajectories));
      }
    }
    std::complex<double> expVal = 0.0;
    // Construct per-term data in the final observe_result
    std::vector<cudaq::ExecutionResult> results;
    results.reserve(terms.size());

    for (std::size_t i = 0; i < terms.size(); ++i) {
      expVal += (coeffs[i] * termExpVals[i]);
      results.emplace_back(
          cudaq::ExecutionResult({}, termStrs[i], termExpVals[i].real()));
    }

    cudaq::sample_result perTermData(expVal.real(), results);
    return cudaq::observe_result(expVal.real(), ham, perTermData);
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
        m_state = std::make_unique<TensorNetState>(
            numQubits, scratchPad, m_cutnHandle, m_randomEngine);
      } else {
        auto [state, mpsTensors] = MPSSimulationState::createFromStateVec(
            m_cutnHandle, scratchPad, 1ULL << numQubits,
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr)),
            m_settings.maxBond, m_randomEngine);
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
        m_state = TensorNetState::createFromMpsTensors(
            tensors, scratchPad, m_cutnHandle, m_randomEngine);
      } else {
        // Non-zero state needs to be factorized and appended.
        auto [state, mpsTensors] = MPSSimulationState::createFromStateVec(
            m_cutnHandle, scratchPad, 1ULL << numQubits,
            reinterpret_cast<std::complex<double> *>(const_cast<void *>(ptr)),
            m_settings.maxBond, m_randomEngine);
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
        m_state = TensorNetState::createFromMpsTensors(
            tensors, scratchPad, m_cutnHandle, m_randomEngine);
      }
    }
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    LOG_API_TIME();

    if (!m_state || m_state->getNumQubits() == 0)
      return std::make_unique<MPSSimulationState>(
          std::move(m_state), std::vector<MPSTensor>{}, scratchPad,
          m_cutnHandle, m_randomEngine);

    if (m_state->getNumQubits() > 1) {
      std::vector<MPSTensor> tensors =
          m_state->factorizeMPS(m_settings.maxBond, m_settings.absCutoff,
                                m_settings.relCutoff, m_settings.svdAlgo);
      return std::make_unique<MPSSimulationState>(std::move(m_state), tensors,
                                                  scratchPad, m_cutnHandle,
                                                  m_randomEngine);
    }

    auto [d_tensor, numElements] = m_state->contractStateVectorInternal({});
    assert(numElements == 2);
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = {static_cast<int64_t>(numElements)};

    return std::make_unique<MPSSimulationState>(
        std::move(m_state), std::vector<MPSTensor>{stateTensor}, scratchPad,
        m_cutnHandle, m_randomEngine);
  }

  bool requireCacheWorkspace() const override { return false; }

  virtual ~SimulatorMPS() noexcept {
    for (auto &tensor : m_mpsTensors_d) {
      HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
    }
    m_mpsTensors_d.clear();
  }
};
} // end namespace nvqir

NVQIR_REGISTER_SIMULATOR(nvqir::SimulatorMPS, tensornet_mps)
