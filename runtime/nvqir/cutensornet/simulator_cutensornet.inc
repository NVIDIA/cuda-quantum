/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq.h"
#include "cutensornet.h"
#include "tensornet_spin_op.h"

namespace nvqir {
template <typename ScalarType>
SimulatorTensorNetBase<ScalarType>::SimulatorTensorNetBase()
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

  bool limit_pathfinding = cudaq::getEnvBool("CUDAQ_TENSORNET_FIND_LIMIT", true);
  if (!limit_pathfinding) {
    CUDAQ_INFO("Disabling cutensornet smart opt");
    setenv("CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SMART_OPTION","0",1);
  }

  bool deterministic = cudaq::getEnvBool("CUDAQ_TENSORNET_FIND_DETERMINISTIC",
                                         false);
  if (deterministic) {
    CUDAQ_INFO("Enabling deterministic tensornet with 1 pathfinding thread");
    setenv("CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS",
           "1",1);
  }

  if (auto *finder_threads = std::getenv("CUDAQ_TENSORNET_FIND_THREADS")) {
    const std::string nthreads_str(finder_threads);
    int nthreads = -1;
    try {
      nthreads = std::stoi(nthreads_str);
    } catch (const std::invalid_argument &e) {
      nthreads = -1;
    } catch (const std::out_of_range &e) {
      nthreads = -1;
    }
    if (nthreads < 1)
      throw std::runtime_error("Invalid CUDAQ_TENSORNET_FIND_THREADS setting" +
                               nthreads_str);
    if (deterministic && nthreads > 1)
      CUDAQ_WARN("Ignoring CUDAQ_TENSORNET_FIND_THREADS due to CUDAQ_TENSORNET_FIND_DETERMINISTIC");
    else {
      CUDAQ_INFO("Setting the # of pathfinding threads to {}", finder_threads);
      setenv("CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_HYPER_NUM_THREADS", finder_threads, 1);
    }
  }
}

template <typename T>
std::vector<std::complex<T>>
generateFullGateTensor(std::size_t num_control_qubits,
                       const std::vector<std::complex<T>> &target_gate) {
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
  std::vector<std::complex<T>> gate_tensor(full_dim * full_dim, {0.0, 0.0});
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
template <typename T>
std::size_t vecComplexHash(const std::vector<std::complex<T>> &vec) {
  std::size_t seed = vec.size();
  for (auto &i : vec) {
    seed ^= std::hash<T>{}(i.real()) + std::hash<T>{}(i.imag()) + 0x9e3779b9 +
            (seed << 6) + (seed >> 2);
  }
  return seed;
}

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::applyGate(
    const GateApplicationTask &task) {
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
template <typename T>
void *
getOrCacheMat(const std::string &key, const std::vector<std::complex<T>> &mat,
              std::unordered_map<std::string, void *> &gateDeviceMemCache) {
  const auto iter = gateDeviceMemCache.find(key);

  if (iter == gateDeviceMemCache.end()) {
    void *dMem = allocateGateMatrix(mat);
    gateDeviceMemCache[key] = dMem;
    return dMem;
  }
  return iter->second;
};

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::applyKrausChannel(
    const std::vector<int32_t> &qubits,
    const cudaq::kraus_channel &krausChannel) {
  LOG_API_TIME();
  if (krausChannel.is_unitary_mixture()) {
    std::vector<void *> channelMats;
    for (const auto &mat : krausChannel.unitary_ops) {
      std::vector<std::complex<ScalarType>> casted(mat.begin(), mat.end());
      channelMats.emplace_back(
          getOrCacheMat("ScaledUnitary_" + std::to_string(vecComplexHash(mat)),
                        casted, m_gateDeviceMemCache));
    }
    m_state->applyUnitaryChannel(qubits, channelMats,
                                 krausChannel.probabilities);
  } else {
    if (!canHandleGeneralNoiseChannel())
      throw std::runtime_error(
          "General noise channels are not supported on simulator " +
          this->name());

    std::vector<void *> channelMats;
    for (const auto &op : krausChannel.get_ops()) {
      void *channelMatPtr_d = [&]() {
        const auto targetPrecision = std::is_same_v<ScalarType, float>
                                         ? cudaq::simulation_precision::fp32
                                         : cudaq::simulation_precision::fp64;
        const std::string cacheKey = cudaq_fmt::format(
            "GeneralKrausMat_{}", std::to_string(vecComplexHash(op.data)));
        if (op.precision == targetPrecision)
          return getOrCacheMat(cacheKey, op.data, m_gateDeviceMemCache);
        // The channel data in a different precision.
        return op.precision == cudaq::simulation_precision::fp32
                   ? getOrCacheMat(cacheKey,
                                   std::vector<std::complex<double>>(
                                       op.data.begin(), op.data.end()),
                                   m_gateDeviceMemCache)
                   : getOrCacheMat(cacheKey,
                                   std::vector<std::complex<float>>(
                                       op.data.begin(), op.data.end()),
                                   m_gateDeviceMemCache);
      }();
      channelMats.emplace_back(channelMatPtr_d);
    }
    m_state->applyGeneralChannel(qubits, channelMats);
  }
}

template <typename ScalarType>
bool SimulatorTensorNetBase<ScalarType>::isValidNoiseChannel(
    const cudaq::noise_model_type &type) const {
  switch (type) {
  case cudaq::noise_model_type::depolarization_channel:
  case cudaq::noise_model_type::bit_flip_channel:
  case cudaq::noise_model_type::phase_flip_channel:
  case cudaq::noise_model_type::x_error:
  case cudaq::noise_model_type::y_error:
  case cudaq::noise_model_type::z_error:
  case cudaq::noise_model_type::pauli1:
  case cudaq::noise_model_type::pauli2:
  case cudaq::noise_model_type::depolarization1:
  case cudaq::noise_model_type::depolarization2:
  case cudaq::noise_model_type::phase_damping:
  case cudaq::noise_model_type::unknown: // may be unitary, so return true
    return true;
  // These are explicitly non-unitary
  case cudaq::noise_model_type::amplitude_damping_channel:
  case cudaq::noise_model_type::amplitude_damping:
  default:
    return canHandleGeneralNoiseChannel();
  }
}

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::applyNoise(
    const cudaq::kraus_channel &channel,
    const std::vector<std::size_t> &targets) {
  LOG_API_TIME();

  // Apply all prior gates before applying noise.
  std::vector<int32_t> qubits{targets.begin(), targets.end()};
  CUDAQ_INFO(
      "[SimulatorTensorNetBase] Applying kraus channel {} on qubits: {}",
      cudaq::get_noise_model_type_name(channel.noise_type), qubits);

  this->flushGateQueue();
  applyKrausChannel(qubits, channel);
}

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::applyNoiseChannel(
    const std::string_view gateName, const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const std::vector<double> &params) {
  LOG_API_TIME();
  // Do nothing if no execution context
  if (!this->executionContext)
    return;

  // Do nothing if no noise model
  if (!this->executionContext->noiseModel)
    return;

  // Get the name as a string
  std::string gName(gateName);
  std::vector<int32_t> qubits{controls.begin(), controls.end()};
  qubits.insert(qubits.end(), targets.begin(), targets.end());

  // Get the Kraus channels specified for this gate and qubits
  auto krausChannels = this->executionContext->noiseModel->get_channels(
      gName, targets, controls, params);

  // If none, do nothing
  if (krausChannels.empty())
    return;

  CUDAQ_INFO(
      "[SimulatorTensorNetBase] Applying {} kraus channels on qubits: {}",
      krausChannels.size(), qubits);

  for (const auto &krausChannel : krausChannels)
    applyKrausChannel(qubits, krausChannel);
}

/// @brief Reset the state of a given qubit to zero
template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::resetQubit(
    const std::size_t qubitIdx) {
  this->flushGateQueue();
  this->flushAnySamplingTasks();
  LOG_API_TIME();
  // Prepare the state before RDM calculation
  prepareQubitTensorState();
  const auto rdm = m_state->computeRDM({static_cast<int32_t>(qubitIdx)});
  assert(rdm.size() == 4);
  const ScalarType prob0 = rdm[0].real();
  CUDAQ_INFO("Reset qubit {} with prob(|0>) = {}", qubitIdx, prob0);
  // If this is a zero state, no need to do anything.
  if (std::abs(1.0 - prob0) < 1e-9)
    return;

  // One state => flip
  if (prob0 < 1e-9) {
    this->x(qubitIdx);
    return;
  }

  // Otherwise, perform projection
  // Reset == project back to 0
  const std::vector<std::complex<ScalarType>> projected0Mat{
      {static_cast<ScalarType>(1.0) / std::sqrt(prob0), 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0}};
  const std::string projKey = std::string("Project") + "_" +
                              std::to_string(false) + "_Prob" +
                              std::to_string(prob0);
  const auto iter = m_gateDeviceMemCache.find(projKey);
  if (iter == m_gateDeviceMemCache.end()) {
    void *d_gateProj{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_gateProj, 4 * sizeof(std::complex<ScalarType>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_gateProj, projected0Mat.data(),
                                 4 * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyHostToDevice));
    m_gateDeviceMemCache[projKey] = d_gateProj;
  }

  m_state->applyQubitProjector(m_gateDeviceMemCache[projKey],
                               {static_cast<int32_t>(qubitIdx)});
}

/// @brief Device synchronization
template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::synchronize() {
  HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
}

/// @brief Perform a measurement on a given qubit
template <typename ScalarType>
bool SimulatorTensorNetBase<ScalarType>::measureQubit(
    const std::size_t qubitIdx) {
  LOG_API_TIME();
  // Prepare the state before RDM calculation
  prepareQubitTensorState();
  const auto rdm = m_state->computeRDM({static_cast<int32_t>(qubitIdx)});
  assert(rdm.size() == 4);
  const ScalarType prob0 = rdm[0].real();
  const ScalarType prob1 = rdm[3].real();
  assert(std::abs(1.0 - (prob0 + prob1)) < 1e-3);
  const ScalarType rand = randomValues(1, 1.0, m_randomEngine)[0];
  const bool resultBool = (rand > prob0);
  const std::vector<std::complex<ScalarType>> projected0Mat{
      {static_cast<ScalarType>(1.0) / std::sqrt(prob0), 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0}};
  const std::vector<std::complex<ScalarType>> projected1Mat{
      {0.0, 0.0},
      {0.0, 0.0},
      {0.0, 0.0},
      {static_cast<ScalarType>(1.0) / std::sqrt(prob1), 0.0}};

  const std::string projKey = std::string("Project") + "_" +
                              std::to_string(resultBool) + "_Prob" +
                              std::to_string(resultBool ? prob1 : prob0);
  const auto iter = m_gateDeviceMemCache.find(projKey);
  if (iter == m_gateDeviceMemCache.end()) {
    void *d_gateProj{nullptr};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_gateProj, 4 * sizeof(std::complex<ScalarType>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(
        d_gateProj, resultBool ? projected1Mat.data() : projected0Mat.data(),
        4 * sizeof(std::complex<ScalarType>), cudaMemcpyHostToDevice));
    m_gateDeviceMemCache[projKey] = d_gateProj;
  }
  m_state->applyQubitProjector(m_gateDeviceMemCache[projKey],
                               {static_cast<int32_t>(qubitIdx)});
  return resultBool;
}

/// @brief Sample a subset of qubits
template <typename ScalarType>
cudaq::ExecutionResult SimulatorTensorNetBase<ScalarType>::sample(
    const std::vector<std::size_t> &measuredBits, const int shots) {
  LOG_API_TIME();
  std::vector<int32_t> measuredBitIds(measuredBits.begin(), measuredBits.end());
  if (shots < 1) {
    auto allZ = cudaq::spin_op::identity();
    for (std::size_t t = 0; t < m_state->getNumQubits(); ++t)
      allZ *= cudaq::spin_op::z(t);
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

template <typename ScalarType>
bool SimulatorTensorNetBase<ScalarType>::canHandleObserve() {
  // Do not compute <H> from matrix if shots based sampling requested
  // i.e., a valid shots count value was set.
  // Note: -1 is also used to denote non-sampling execution. Hence, we need to
  // check for this particular -1 value as being casted to an unsigned type.
  if (this->executionContext && this->executionContext->shots > 0 &&
      this->executionContext->shots != ~0ull) {
    // This 'shots' mode is very slow for tensor network.
    // However, we need to respect the shots option.
    CUDAQ_INFO("[SimulatorTensorNetBase] Shots mode expectation calculation "
                "is requested with {} shots.",
                this->executionContext->shots);

    return false;
  }
  // Otherwise, perform exact expectation value calculation/contraction.
  return true;
}

/// @brief Evaluate the expectation value of a given observable
template <typename ScalarType>
cudaq::observe_result
SimulatorTensorNetBase<ScalarType>::observe(const cudaq::spin_op &ham) {
  assert(cudaq::spin_op::canonicalize(ham) == ham);
  LOG_API_TIME();
  prepareQubitTensorState();
  if (!m_reuseContractionPathObserve) {
    // If contraction path reuse is disabled, convert spin_op to
    // cutensornetNetworkOperator_t and compute the expectation value.
    TensorNetworkSpinOp<ScalarType> spinOp(ham, m_cutnHandle);
    std::complex<ScalarType> expVal =
        m_state->computeExpVal(spinOp.getNetworkOperator(),
                               this->executionContext->numberTrajectories);
    expVal += spinOp.getIdentityTermOffset();
    return cudaq::observe_result(expVal.real(), ham,
                                 cudaq::sample_result(cudaq::ExecutionResult(
                                     {}, ham.to_string(), expVal.real())));
  }

  std::vector<std::string> termStrs;
  std::vector<cudaq::spin_op_term> prods;
  termStrs.reserve(ham.num_terms());
  prods.reserve(ham.num_terms());
  for (auto &&term : ham) {
    termStrs.emplace_back(term.get_term_id());
    prods.push_back(std::move(term));
  }

  // Compute the expectation value for all terms
  const auto termExpVals = m_state->computeExpVals(
      prods, this->executionContext->numberTrajectories);
  std::complex<double> expVal = 0.0;
  // Construct per-term data in the final observe_result
  std::vector<cudaq::ExecutionResult> results;
  results.reserve(prods.size());

  for (std::size_t i = 0; i < prods.size(); ++i) {
    expVal += termExpVals[i];
    results.emplace_back(
        cudaq::ExecutionResult({}, termStrs[i], termExpVals[i].real()));
  }

  cudaq::sample_result perTermData(expVal.real(), results);
  return cudaq::observe_result(expVal.real(), ham, perTermData);
}

template <typename ScalarType>
nvqir::CircuitSimulator *SimulatorTensorNetBase<ScalarType>::clone() {
  return nullptr;
}

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::addQubitToState() {
  this->addQubitsToState(1);
}

/// @brief Destroy the entire qubit register
template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::deallocateStateImpl() {
  if (m_state) {
    m_state.reset();
    // Reset cuTensorNet library
    HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
    HANDLE_CUTN_ERROR(cutensornetCreate(&m_cutnHandle));
  }
}

/// @brief Reset all qubits to zero
template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::setToZeroState() {
  LOG_API_TIME();
  const auto numQubits = m_state->getNumQubits();
  m_state.reset();
  // Re-create a zero state of the same size
  m_state = std::make_unique<TensorNetState<ScalarType>>(
      numQubits, scratchPad, m_cutnHandle, m_randomEngine);
}

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::swap(
    const std::vector<std::size_t> &ctrlBits, const std::size_t srcIdx,
    const std::size_t tgtIdx) {
  if (ctrlBits.empty())
    return nvqir::CircuitSimulatorBase<ScalarType>::swap(ctrlBits, srcIdx,
                                                         tgtIdx);
  // Controlled swap gate: using cnot decomposition of swap gate to perform
  // decomposition.
  // Note: cutensornetStateApplyControlledTensorOperator can only handle
  // single-target.
  const auto size = ctrlBits.size();
  std::vector<std::size_t> ctls(size + 1);
  std::copy(ctrlBits.begin(), ctrlBits.end(), ctls.begin());
  {
    ctls[size] = tgtIdx;
    nvqir::CircuitSimulatorBase<ScalarType>::x(ctls, srcIdx);
  }
  {
    ctls[size] = srcIdx;
    nvqir::CircuitSimulatorBase<ScalarType>::x(ctls, tgtIdx);
  }
  {
    ctls[size] = tgtIdx;
    nvqir::CircuitSimulatorBase<ScalarType>::x(ctls, srcIdx);
  }
}

template <typename ScalarType>
void SimulatorTensorNetBase<ScalarType>::setRandomSeed(std::size_t randomSeed) {
  m_randomEngine = std::mt19937(randomSeed);
  // Although we set the pathfinder seed here, it is only deterministic when
  // the number of pathfinding threads is 1. Note that this does not override a
  // user setting for this environment variable. cutensornet uses int32_t but
  // requires the environment variable to be positive
  int32_t rndSeed = m_randomEngine();
  rndSeed = std::abs(rndSeed);
  if (rndSeed == 0) rndSeed++;
  setenv("CUTENSORNET_CONTRACTION_OPTIMIZER_CONFIG_SEED",
         std::to_string(rndSeed).c_str(),0);
}

template <typename ScalarType>
SimulatorTensorNetBase<ScalarType>::~SimulatorTensorNetBase() {
  m_state.reset();
  for (const auto &[key, dMem] : m_gateDeviceMemCache)
    HANDLE_CUDA_ERROR(cudaFree(dMem));

  // Finalize the cuTensorNet library
  HANDLE_CUTN_ERROR(cutensornetDestroy(m_cutnHandle));
}
} // end namespace nvqir
