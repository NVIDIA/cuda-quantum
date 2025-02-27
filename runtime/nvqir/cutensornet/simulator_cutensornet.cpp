/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "simulator_cutensornet.h"
#include "cudaq.h"
#include "cutensornet.h"
#include "tensornet_spin_op.h"

namespace nvqir {

SimulatorTensorNetBase::SimulatorTensorNetBase()
    : m_randomEngine(std::random_device()()) {
  int numDevices{0};
  HANDLE_CUDA_ERROR(cudaGetDeviceCount(&numDevices));
  // we assume that the processes are mapped to nodes in contiguous chunks
  const int deviceId =
      cudaq::mpi::is_initialized() ? cudaq::mpi::rank() % numDevices : 0;
  HANDLE_CUDA_ERROR(cudaSetDevice(deviceId));
  HANDLE_CUTN_ERROR(cutensornetCreate(&m_cutnHandle));
  // The scratch pad must be allocated after we have selected the device.
  scratchPad.allocate();

  // Check whether observe path reuse is enabled.
  m_reuseContractionPathObserve =
      cudaq::getEnvBool("CUDAQ_TENSORNET_OBSERVE_CONTRACT_PATH_REUSE", false);
}

static std::vector<std::complex<double>>
generateFullGateTensor(std::size_t num_control_qubits,
                       const std::vector<std::complex<double>> &target_gate) {
  const auto mat_size = target_gate.size();
  // Must be square matrix (n x n)
  assert(std::ceil(std::sqrt(mat_size)) == std::floor(std::sqrt(mat_size)) &&
         "Input matrix is not a square matrix.");
  // Dim == rows == cols
  const std::size_t target_gate_dim = std::ceil(std::sqrt(mat_size));
  // Number of qubits
  const std::size_t num_target_qubits = std::bit_width(target_gate_dim) - 1;
  assert(target_gate_dim == (1u << num_target_qubits) &&
         "Gate matrix dimension is not 2^N");
  // No control => return the input matrix
  if (num_control_qubits == 0)
    return target_gate;
  // Expand the matrix
  const std::size_t full_dim =
      (1UL << (num_control_qubits + num_target_qubits));
  std::vector<std::complex<double>> gate_tensor(full_dim * full_dim,
                                                {0.0, 0.0});
  std::size_t offset = 0;
  // Set the diagonal elements:
  for (int i = 0; i < static_cast<int>(full_dim - target_gate_dim); ++i) {
    gate_tensor[offset] = {1.0, 0.0};
    offset += (full_dim + 1);
  }
  // Set the target gate matrix:
  for (std::size_t row = 0; row < target_gate_dim; ++row) {
    for (std::size_t col = 0; col < target_gate_dim; ++col) {
      const auto org_idx = row * target_gate_dim + col;
      // The anchor point of the gate matrix inside the expanded matrix (lower
      // right, i.e., shift up and left by target_gate_dim)
      const auto block_anchor = full_dim - target_gate_dim;
      // Row and column idxs in the expanded matrix
      const auto block_row_idx = block_anchor + row;
      const auto block_col_idx = block_anchor + col;
      const auto expanded_idx = block_row_idx * full_dim + block_col_idx;
      gate_tensor[expanded_idx] = target_gate[org_idx];
    }
  }
  return gate_tensor;
}

/// @brief Provide a unique hash code for the input vector of complex values.
std::size_t vecComplexHash(const std::vector<std::complex<double>> &vec) {
  std::size_t seed = vec.size();
  for (auto &i : vec) {
    seed ^= std::hash<double>{}(i.real()) + std::hash<double>{}(i.imag()) +
            0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

void SimulatorTensorNetBase::applyGate(const GateApplicationTask &task) {
  const auto &controls = task.controls;
  const auto &targets = task.targets;
  // Cache name lookup key:
  // <GateName>_<Param>_<Matrix>
  const std::string gateKey = task.operationName + "_" + [&]() {
    std::stringstream paramsSs;
    for (const auto &param : task.parameters) {
      paramsSs << param << "_";
    }
    return paramsSs.str() + "__" + std::to_string(vecComplexHash(task.matrix));
  }();

  if (controls.size() <= m_maxControlledRankForFullTensorExpansion) {
    // If the number of controlled qubits is less than the threshold, expand the
    // full matrix and apply it as a single tensor operation.
    // Qubit operands are now both control and target qubits.
    std::vector<std::int32_t> qubitOperands(controls.begin(), controls.end());
    qubitOperands.insert(qubitOperands.end(), targets.begin(), targets.end());
    // Use a different key for expanded gate matrix (reflecting the number of
    // control qubits)
    const auto expandedMatKey =
        gateKey + "_c(" + std::to_string(controls.size()) + ")";
    const auto iter = m_gateDeviceMemCache.find(expandedMatKey);
    if (iter != m_gateDeviceMemCache.end()) {
      m_state->applyGate(/*controlQubits=*/{}, qubitOperands, iter->second);
    } else {
      // If this is the first time seeing this (gate + number of control qubits)
      // compo, compute the expanded matrix.
      const auto expandedGateMat =
          generateFullGateTensor(controls.size(), task.matrix);
      void *dMem = allocateGateMatrix(expandedGateMat);
      m_gateDeviceMemCache[expandedMatKey] = dMem;
      m_state->applyGate(/*controlQubits=*/{}, qubitOperands, dMem);
    }
  } else {
    // Propagates control qubits to cutensornet.
    const auto iter = m_gateDeviceMemCache.find(gateKey);
    // This is the first time we see this gate, allocate device mem and cache
    // it.
    if (iter == m_gateDeviceMemCache.end()) {
      void *dMem = allocateGateMatrix(task.matrix);
      m_gateDeviceMemCache[gateKey] = dMem;
    }
    // Type conversion
    const std::vector<std::int32_t> ctrlQubits(controls.begin(),
                                               controls.end());
    const std::vector<std::int32_t> targetQubits(targets.begin(),
                                                 targets.end());
    m_state->applyGate(ctrlQubits, targetQubits, m_gateDeviceMemCache[gateKey]);
  }
}

// Helper to look up a device memory pointer from a cache.
// If not found, allocate a new device memory buffer and put it to the cache.
static void *
getOrCacheMat(const std::string &key,
              const std::vector<std::complex<double>> &mat,
              std::unordered_map<std::string, void *> &gateDeviceMemCache) {
  const auto iter = gateDeviceMemCache.find(key);

  if (iter == gateDeviceMemCache.end()) {
    void *dMem = allocateGateMatrix(mat);
    gateDeviceMemCache[key] = dMem;
    return dMem;
  }
  return iter->second;
};

void SimulatorTensorNetBase::applyKrausChannel(
    const std::vector<int32_t> &qubits,
    const cudaq::kraus_channel &krausChannel) {
  LOG_API_TIME();
  if (krausChannel.is_unitary_mixture()) {
    std::vector<void *> channelMats;
    for (const auto &mat : krausChannel.unitary_ops)
      channelMats.emplace_back(
          getOrCacheMat("ScaledUnitary_" + std::to_string(vecComplexHash(mat)),
                        mat, m_gateDeviceMemCache));
    m_state->applyUnitaryChannel(qubits, channelMats,
                                 krausChannel.probabilities);
  } else {
    throw std::runtime_error("Non-unitary noise channels are not supported.");
  }
}

void SimulatorTensorNetBase::applyNoiseChannel(
    const std::string_view gateName, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const std::vector<double> &params) {
  LOG_API_TIME();
  // Do nothing if no execution context
  if (!executionContext)
    return;

  // Do nothing if no noise model
  if (!executionContext->noiseModel)
    return;

  // Get the name as a string
  std::string gName(gateName);
  std::vector<int32_t> qubits{controls.begin(), controls.end()};
  qubits.insert(qubits.end(), targets.begin(), targets.end());

  // Get the Kraus channels specified for this gate and qubits
  auto krausChannels = executionContext->noiseModel->get_channels(
      gName, targets, controls, params);

  // If none, do nothing
  if (krausChannels.empty())
    return;

  cudaq::info(
      "[SimulatorTensorNetBase] Applying {} kraus channels on qubits: {}",
      krausChannels.size(), qubits);

  for (const auto &krausChannel : krausChannels)
    applyKrausChannel(qubits, krausChannel);
}

/// @brief Reset the state of a given qubit to zero
void SimulatorTensorNetBase::resetQubit(const std::size_t qubitIdx) {
  flushGateQueue();
  flushAnySamplingTasks();
  LOG_API_TIME();
  // Prepare the state before RDM calculation
  prepareQubitTensorState();
  const auto rdm = m_state->computeRDM({static_cast<int32_t>(qubitIdx)});
  assert(rdm.size() == 4);
  const double prob0 = rdm[0].real();
  cudaq::info("Reset qubit {} with prob(|0>) = {}", qubitIdx, prob0);
  // If this is a zero state, no need to do anything.
  if (std::abs(1.0 - prob0) < 1e-9)
    return;

  // One state => flip
  if (prob0 < 1e-9) {
    x(qubitIdx);
    return;
  }

  // Otherwise, perform projection
  // Reset == project back to 0
  const std::vector<std::complex<double>> projected0Mat{
      {1.0 / std::sqrt(prob0), 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::string projKey = std::string("Project") + "_" +
                              std::to_string(false) + "_Prob" +
                              std::to_string(prob0);
  const auto iter = m_gateDeviceMemCache.find(projKey);
  if (iter == m_gateDeviceMemCache.end()) {
    void *d_gateProj{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_gateProj, 4 * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_gateProj, projected0Mat.data(),
                                 4 * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    m_gateDeviceMemCache[projKey] = d_gateProj;
  }

  m_state->applyQubitProjector(m_gateDeviceMemCache[projKey],
                               {static_cast<int32_t>(qubitIdx)});
}

/// @brief Device synchronization
void SimulatorTensorNetBase::synchronize() {
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

/// @brief Perform a measurement on a given qubit
bool SimulatorTensorNetBase::measureQubit(const std::size_t qubitIdx) {
  LOG_API_TIME();
  // Prepare the state before RDM calculation
  prepareQubitTensorState();
  const auto rdm = m_state->computeRDM({static_cast<int32_t>(qubitIdx)});
  assert(rdm.size() == 4);
  const double prob0 = rdm[0].real();
  const double prob1 = rdm[3].real();
  assert(std::abs(1.0 - (prob0 + prob1)) < 1e-9);
  const double rand = randomValues(1, 1.0, m_randomEngine)[0];
  const bool resultBool = (rand > prob0);
  const std::vector<std::complex<double>> projected0Mat{
      {1.0 / std::sqrt(prob0), 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
  const std::vector<std::complex<double>> projected1Mat{
      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0 / std::sqrt(prob1), 0.0}};

  const std::string projKey = std::string("Project") + "_" +
                              std::to_string(resultBool) + "_Prob" +
                              std::to_string(resultBool ? prob1 : prob0);
  const auto iter = m_gateDeviceMemCache.find(projKey);
  if (iter == m_gateDeviceMemCache.end()) {
    void *d_gateProj{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_gateProj, 4 * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(
        d_gateProj, resultBool ? projected1Mat.data() : projected0Mat.data(),
        4 * sizeof(std::complex<double>), cudaMemcpyHostToDevice));
    m_gateDeviceMemCache[projKey] = d_gateProj;
  }
  m_state->applyQubitProjector(m_gateDeviceMemCache[projKey],
                               {static_cast<int32_t>(qubitIdx)});
  return resultBool;
}

/// @brief Sample a subset of qubits
cudaq::ExecutionResult
SimulatorTensorNetBase::sample(const std::vector<std::size_t> &measuredBits,
                               const int shots) {
  LOG_API_TIME();
  std::vector<int32_t> measuredBitIds(measuredBits.begin(), measuredBits.end());
  if (shots < 1) {
    cudaq::spin_op::spin_op_term allZTerm(2 * m_state->getNumQubits(), 0);
    for (const auto &m : measuredBits)
      allZTerm.at(m_state->getNumQubits() + m) = 1;
    cudaq::spin_op allZ(allZTerm, 1.0);
    // Just compute the expected value on <Z...Z>
    return cudaq::ExecutionResult({}, observe(allZ).expectation());
  }

  prepareQubitTensorState();
  const auto samples =
      m_state->sample(measuredBitIds, shots, requireCacheWorkspace());
  cudaq::ExecutionResult counts(samples);
  double expVal = 0.0;
  std::size_t sum_counts = 0;
  // Compute the expectation value from the counts
  for (auto &kv : counts.counts) {
    auto par = cudaq::sample_result::has_even_parity(kv.first);
    auto p = kv.second / (double)shots;
    if (!par) {
      p = -p;
    }
    expVal += p;
    sum_counts += kv.second;
  }

  counts.expectationValue = expVal;
  counts.sequentialData.resize(sum_counts);
  std::size_t s = 0;
  for (auto &kv : counts.counts)
    for (std::size_t c = 0; c < kv.second; c++)
      counts.sequentialData[s++] = kv.first;

  return counts;
}

bool SimulatorTensorNetBase::canHandleObserve() {
  // Do not compute <H> from matrix if shots based sampling requested
  // i.e., a valid shots count value was set.
  // Note: -1 is also used to denote non-sampling execution. Hence, we need to
  // check for this particular -1 value as being casted to an unsigned type.
  if (executionContext && executionContext->shots > 0 &&
      executionContext->shots != ~0ull) {
    // This 'shots' mode is very slow for tensor network.
    // However, we need to respect the shots option.
    cudaq::info("[SimulatorTensorNetBase] Shots mode expectation calculation "
                "is requested with {} shots.",
                executionContext->shots);

    return false;
  }
  // Otherwise, perform exact expectation value calculation/contraction.
  return true;
}

/// @brief Evaluate the expectation value of a given observable
cudaq::observe_result
SimulatorTensorNetBase::observe(const cudaq::spin_op &ham) {
  LOG_API_TIME();
  prepareQubitTensorState();
  if (!m_reuseContractionPathObserve) {
    // If contraction path reuse is disabled, convert spin_op to
    // cutensornetNetworkOperator_t and compute the expectation value.
    TensorNetworkSpinOp spinOp(ham, m_cutnHandle);
    std::complex<double> expVal =
        m_state->computeExpVal(spinOp.getNetworkOperator(),
                               this->executionContext->numberTrajectories);
    expVal += spinOp.getIdentityTermOffset();
    return cudaq::observe_result(expVal.real(), ham,
                                 cudaq::sample_result(cudaq::ExecutionResult(
                                     {}, ham.to_string(false), expVal.real())));
  }

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

  // Compute the expectation value for all terms
  const auto termExpVals = m_state->computeExpVals(
      terms, this->executionContext->numberTrajectories);
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

nvqir::CircuitSimulator *SimulatorTensorNetBase::clone() { return nullptr; }

void SimulatorTensorNetBase::addQubitToState() { addQubitsToState(1); }

/// @brief Destroy the entire qubit register
void SimulatorTensorNetBase::deallocateStateImpl() {
  if (m_state) {
    m_state.reset();
    // Reset cuTensorNet library
    HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
    HANDLE_CUTN_ERROR(cutensornetCreate(&m_cutnHandle));
  }
}

/// @brief Reset all qubits to zero
void SimulatorTensorNetBase::setToZeroState() {
  LOG_API_TIME();
  const auto numQubits = m_state->getNumQubits();
  m_state.reset();
  // Re-create a zero state of the same size
  m_state = std::make_unique<TensorNetState>(numQubits, scratchPad,
                                             m_cutnHandle, m_randomEngine);
}

void SimulatorTensorNetBase::swap(const std::vector<std::size_t> &ctrlBits,
                                  const std::size_t srcIdx,
                                  const std::size_t tgtIdx) {
  if (ctrlBits.empty())
    return nvqir::CircuitSimulatorBase<double>::swap(ctrlBits, srcIdx, tgtIdx);
  // Controlled swap gate: using cnot decomposition of swap gate to perform
  // decomposition.
  // Note: cutensornetStateApplyControlledTensorOperator can only handle
  // single-target.
  const auto size = ctrlBits.size();
  std::vector<std::size_t> ctls(size + 1);
  std::copy(ctrlBits.begin(), ctrlBits.end(), ctls.begin());
  {
    ctls[size] = tgtIdx;
    nvqir::CircuitSimulatorBase<double>::x(ctls, srcIdx);
  }
  {
    ctls[size] = srcIdx;
    nvqir::CircuitSimulatorBase<double>::x(ctls, tgtIdx);
  }
  {
    ctls[size] = tgtIdx;
    nvqir::CircuitSimulatorBase<double>::x(ctls, srcIdx);
  }
}

void SimulatorTensorNetBase::setRandomSeed(std::size_t randomSeed) {
  m_randomEngine = std::mt19937(randomSeed);
}

SimulatorTensorNetBase::~SimulatorTensorNetBase() {
  m_state.reset();
  for (const auto &[key, dMem] : m_gateDeviceMemCache)
    HANDLE_CUDA_ERROR(cudaFree(dMem));

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
}
} // end namespace nvqir
