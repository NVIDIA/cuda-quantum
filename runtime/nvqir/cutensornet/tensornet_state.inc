/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EigenDense.h"
#include "common/Environment.h"
#include <algorithm>
#include <bitset>
#include <cassert>

namespace nvqir {
template <typename ScalarType>
std::int32_t TensorNetState<ScalarType>::numHyperSamples = []() {
  // Path optimization hyper samples are multithreaded.
  // Default to 8 samples, reasonable for most systems (trade-off between
  // prepare/path optimization time and contraction time). On systems with more
  // CPU cores and/or larger circuits, users may consider increasing this number
  // to improve the path quality.
  constexpr int32_t defaultNumHyperSamples = 8;
  if (auto envVal = std::getenv("CUDAQ_TENSORNET_NUM_HYPER_SAMPLES")) {
    int32_t specifiedNumHyperSamples = 0;
    try {
      specifiedNumHyperSamples = std::stoi(envVal);
      if (specifiedNumHyperSamples <= 0) {
        // for the 'catch' below to handle.
        throw std::invalid_argument("must be a positive number");
      }
      CUDAQ_INFO("Update number of hyper samples from {} to {}.",
                 defaultNumHyperSamples, specifiedNumHyperSamples);
      return specifiedNumHyperSamples;
    } catch (...) {
      throw std::runtime_error(
          "Invalid CUDAQ_TENSORNET_NUM_HYPER_SAMPLES environment "
          "variable, must be a positive integer.");
    }
  }
  return defaultNumHyperSamples;
}();

template <typename ScalarType>
bool TensorNetState<ScalarType>::m_deterministic = []() {
  return cudaq::getEnvBool("CUDAQ_TENSORNET_FIND_DETERMINISTIC", false);
}();

template <typename ScalarType>
TensorNetState<ScalarType>::TensorNetState(std::size_t numQubits,
                                           ScratchDeviceMem &inScratchPad,
                                           cutensornetHandle_t handle,
                                           std::mt19937 &randomEngine)
    : m_numQubits(numQubits), m_cutnHandle(handle), scratchPad(inScratchPad),
      m_randomEngine(randomEngine) {
  const std::vector<int64_t> qubitDims(m_numQubits, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubits,
      qubitDims.data(), cudaDataType, &m_quantumState));
}

template <typename ScalarType>
TensorNetState<ScalarType>::TensorNetState(const std::vector<int> &basisState,
                                           ScratchDeviceMem &inScratchPad,
                                           cutensornetHandle_t handle,
                                           std::mt19937 &randomEngine)
    : TensorNetState(basisState.size(), inScratchPad, handle, randomEngine) {
  constexpr std::complex<ScalarType> h_xGate[4] = {0.0, 1.0, 1.0, 0.0};
  constexpr auto sizeBytes = 4 * sizeof(std::complex<ScalarType>);
  void *d_gate{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, sizeBytes));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_gate, h_xGate, sizeBytes, cudaMemcpyHostToDevice));
  m_tempDevicePtrs.emplace_back(d_gate, TempDevicePtrDeleter{});
  for (int32_t qId = 0; const auto &bit : basisState) {
    if (bit == 1) {
      applyGate({}, {qId}, d_gate);
    }
    ++qId;
  }
}

template <typename ScalarType>
std::unique_ptr<TensorNetState<ScalarType>>
TensorNetState<ScalarType>::clone() const {
  return createFromOpTensors(m_numQubits, m_tensorOps, scratchPad, m_cutnHandle,
                             m_randomEngine);
}

template <typename ScalarType>
void TensorNetState<ScalarType>::applyGate(
    const std::vector<int32_t> &controlQubits,
    const std::vector<int32_t> &targetQubits, void *gateDeviceMem,
    bool adjoint) {
  ScopedTraceWithContext("TensorNetState<ScalarType>::applyGate",
                         controlQubits.size(), targetQubits.size());
  if (controlQubits.empty()) {
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
        m_cutnHandle, m_quantumState, targetQubits.size(), targetQubits.data(),
        gateDeviceMem, nullptr, /*immutable*/ 1,
        /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
  } else {
    HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
        m_cutnHandle, m_quantumState, /*numControlModes=*/controlQubits.size(),
        /*stateControlModes=*/controlQubits.data(),
        /*stateControlValues=*/nullptr,
        /*numTargetModes*/ targetQubits.size(),
        /*stateTargetModes*/ targetQubits.data(), gateDeviceMem, nullptr,
        /*immutable*/ 1,
        /*adjoint*/ static_cast<int32_t>(adjoint), /*unitary*/ 1, &m_tensorId));
  }
  m_tensorOps.emplace_back(AppliedTensorOp{gateDeviceMem, targetQubits,
                                           controlQubits, adjoint, true});
}

template <typename ScalarType>
void TensorNetState<ScalarType>::applyUnitaryChannel(
    const std::vector<int32_t> &qubits, const std::vector<void *> &krausOps,
    const std::vector<double> &probabilities) {
  LOG_API_TIME();
  HANDLE_CUTN_ERROR(cutensornetStateApplyUnitaryChannel(
      m_cutnHandle, m_quantumState, /*numStateModes=*/qubits.size(),
      /*stateModes=*/qubits.data(),
      /*numTensors=*/krausOps.size(),
      /*tensorData=*/const_cast<void **>(krausOps.data()),
      /*tensorModeStrides=*/nullptr,
      /*probabilities=*/probabilities.data(), &m_tensorId));
  m_tensorOps.emplace_back(AppliedTensorOp{qubits, krausOps, probabilities});
  m_hasNoiseChannel = true;
}

template <typename ScalarType>
void TensorNetState<ScalarType>::applyGeneralChannel(
    const std::vector<int32_t> &qubits, const std::vector<void *> &krausOps) {
  LOG_API_TIME();
  HANDLE_CUTN_ERROR(cutensornetStateApplyGeneralChannel(
      m_cutnHandle, m_quantumState, /*numStateModes=*/qubits.size(),
      /*stateModes=*/qubits.data(),
      /*numTensors=*/krausOps.size(),
      /*tensorData=*/const_cast<void **>(krausOps.data()),
      /*tensorModeStrides=*/nullptr, &m_tensorId));
  m_tensorOps.emplace_back(AppliedTensorOp{qubits, krausOps, {}});
  m_hasNoiseChannel = true;
}

template <typename ScalarType>
void TensorNetState<ScalarType>::applyQubitProjector(
    void *proj_d, const std::vector<int32_t> &qubitIdx) {
  LOG_API_TIME();
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
      m_cutnHandle, m_quantumState, qubitIdx.size(), qubitIdx.data(), proj_d,
      nullptr,
      /*immutable*/ 1,
      /*adjoint*/ 0, /*unitary*/ 0, &m_tensorId));
  m_tensorOps.emplace_back(AppliedTensorOp{proj_d, qubitIdx, {}, false, false});
}

template <typename ScalarType>
void TensorNetState<ScalarType>::addQubits(std::size_t numQubits) {
  LOG_API_TIME();
  // Destroy the current quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
  m_numQubits += numQubits;
  const std::vector<int64_t> qubitDims(m_numQubits, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubits,
      qubitDims.data(), cudaDataType, &m_quantumState));
  // Append any previously-applied gate tensors.
  // These tensors will only be appending to those existing qubit wires, i.e.,
  // the new wires are all empty (zero state).
  int64_t tensorId = 0;
  for (auto &op : m_tensorOps)
    if (op.deviceData) {
      if (op.controlQubitIds.empty()) {
        HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
            m_cutnHandle, m_quantumState, op.targetQubitIds.size(),
            op.targetQubitIds.data(), op.deviceData, nullptr, /*immutable*/ 1,
            /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
            /*unitary*/ static_cast<int32_t>(op.isUnitary), &tensorId));
      } else {
        HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
            m_cutnHandle, m_quantumState,
            /*numControlModes=*/op.controlQubitIds.size(),
            /*stateControlModes=*/op.controlQubitIds.data(),
            /*stateControlValues=*/nullptr,
            /*numTargetModes*/ op.targetQubitIds.size(),
            /*stateTargetModes*/ op.targetQubitIds.data(), op.deviceData,
            nullptr,
            /*immutable*/ 1,
            /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
            /*unitary*/ static_cast<int32_t>(op.isUnitary), &m_tensorId));
      }
    } else if (op.noiseChannel.has_value()) {
      const bool isGeneralChannel = op.noiseChannel->tensorData.size() !=
                                    op.noiseChannel->probabilities.size();
      if (isGeneralChannel) {
        HANDLE_CUTN_ERROR(cutensornetStateApplyGeneralChannel(
            m_cutnHandle, m_quantumState,
            /*numStateModes=*/op.targetQubitIds.size(),
            /*stateModes=*/op.targetQubitIds.data(),
            /*numTensors=*/op.noiseChannel->tensorData.size(),
            /*tensorData=*/op.noiseChannel->tensorData.data(),
            /*tensorModeStrides=*/nullptr, &m_tensorId));
      } else {
        HANDLE_CUTN_ERROR(cutensornetStateApplyUnitaryChannel(
            m_cutnHandle, m_quantumState,
            /*numStateModes=*/op.targetQubitIds.size(),
            /*stateModes=*/op.targetQubitIds.data(),
            /*numTensors=*/op.noiseChannel->tensorData.size(),
            /*tensorData=*/op.noiseChannel->tensorData.data(),
            /*tensorModeStrides=*/nullptr,
            /*probabilities=*/op.noiseChannel->probabilities.data(),
            &m_tensorId));
      }
    } else {
      throw std::runtime_error("Invalid AppliedTensorOp encountered.");
    }
}

template <typename ScalarType>
void TensorNetState<ScalarType>::addQubits(
    std::span<std::complex<ScalarType>> stateVec) {
  LOG_API_TIME();
  const std::size_t numQubits = std::log2(stateVec.size());
  auto ket =
      Eigen::Map<const Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>>(
          stateVec.data(), stateVec.size());
  Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic> initState =
      Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>::Zero(
          stateVec.size());
  initState(0) = std::complex<ScalarType>{1.0, 0.0};
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
      stateVecProj = ket * initState.transpose();
  assert(static_cast<std::size_t>(stateVecProj.size()) ==
         stateVec.size() * stateVec.size());
  stateVecProj.transposeInPlace();
  void *d_proj{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_proj, stateVecProj.size() *
                                            sizeof(std::complex<ScalarType>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_proj, stateVecProj.data(),
                 stateVecProj.size() * sizeof(std::complex<ScalarType>),
                 cudaMemcpyHostToDevice));

  std::vector<int32_t> qubitIdx(numQubits);
  std::iota(qubitIdx.begin(), qubitIdx.end(), m_numQubits);
  // Add qubits in zero state
  addQubits(numQubits);

  // Project the state of those new qubits to the input state.
  applyQubitProjector(d_proj, qubitIdx);
  m_tempDevicePtrs.emplace_back(d_proj, TempDevicePtrDeleter{});
}

template <typename ScalarType>
std::pair<cutensornetStateSampler_t, cutensornetWorkspaceDescriptor_t>
TensorNetState<ScalarType>::prepareSample(
    const std::vector<int32_t> &measuredBitIds) {
  LOG_API_TIME();
  // Create the quantum circuit sampler
  cutensornetStateSampler_t sampler;
  {
    ScopedTraceWithContext("cutensornetCreateSampler");
    HANDLE_CUTN_ERROR(cutensornetCreateSampler(
        m_cutnHandle, m_quantumState, measuredBitIds.size(),
        measuredBitIds.data(), &sampler));
  }

  {
    ScopedTraceWithContext("cutensornetSamplerConfigure");
    HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
        m_cutnHandle, sampler, CUTENSORNET_SAMPLER_CONFIG_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));
  }
  // Generate a random seed from the backend simulator's random engine to
  // configure cutn for deterministic path-finding. Here the configure is
  // before the prepare call so setting the deterministic attribute will
  // limit # of pathfinding threads to 1 and can therefore cause significant
  // performance impact
  if (m_deterministic) {
    const int32_t rndSeed = m_randomEngine();
    HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
        m_cutnHandle, sampler, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC,
        &rndSeed, sizeof(rndSeed)));
  }

  // Prepare the quantum circuit sampler
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetSamplerPrepare");
    HANDLE_CUTN_ERROR(cutensornetSamplerPrepare(m_cutnHandle, sampler,
                                                scratchPad.scratchSize,
                                                workDesc, /*cudaStream*/ 0));
  }
  // Attach the workspace buffer
  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  // This should not happen (cutensornetWorkspaceGetMemorySize would have
  // returned an error code).
  if (worksize <= 0)
    throw std::runtime_error(
        "INTERNAL ERROR: Invalid workspace size encountered.");

  if (worksize <= static_cast<int64_t>(scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }

  return std::make_pair(sampler, workDesc);
}

template <typename ScalarType>
std::unordered_map<std::string, size_t>
TensorNetState<ScalarType>::executeSample(
    cutensornetStateSampler_t &sampler,
    cutensornetWorkspaceDescriptor_t &workDesc,
    const std::vector<int32_t> &measuredBitIds, int32_t shots,
    bool enableCacheWorkspace) {
  int64_t reqCacheSize{0};
  void *d_cache{nullptr};
  if (enableCacheWorkspace) {
    ScopedTraceWithContext("Allocate Cache Workspace");
    HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
        m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
        CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_CACHE,
        &reqCacheSize));

    // Query the GPU memory capacity
    std::size_t freeSize{0}, totalSize{0};
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeSize, &totalSize));
    // Compute the minimum of [required size, or 90% of the free memory (to
    // avoid oversubscribing)] (see cutensornet examples)
    const std::size_t cacheSizeAvailable =
        std::min(static_cast<size_t>(reqCacheSize),
                 size_t(freeSize * 0.9) - (size_t(freeSize * 0.9) % 4096));
    CUDAQ_INFO("Cache size = {} bytes", cacheSizeAvailable);
    const auto errCode = cudaMalloc(&d_cache, cacheSizeAvailable);
    if (errCode == cudaSuccess) {
      HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
          m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
          CUTENSORNET_WORKSPACE_CACHE, d_cache, cacheSizeAvailable));
    } else {
      CUDAQ_INFO("Failed to allocate cache workspace memory.");
      d_cache = nullptr;
    }
  }
  // Sample the quantum circuit state
  std::unordered_map<std::string, size_t> counts;
  // If this is a trajectory simulation, each shot needs an independent
  // trajectory sampling.
  const int64_t MAX_SHOTS_PER_RUNS = m_hasNoiseChannel ? 1 : shots;
  int64_t shotsToRun = shots;
  while (shotsToRun > 0) {
    const int64_t numShots = std::min(shotsToRun, MAX_SHOTS_PER_RUNS);
    std::vector<int64_t> samples(measuredBitIds.size() * numShots);
    {
      // Generate a random seed from the backend simulator's random engine.
      // Note: Even after a random seed setting at the user's level,
      // consecutive `cudaq::sample` calls will still return different results
      // (yet deterministic), i.e., the seed that we send to cutensornet should
      // not be the user's seed. Here the configure is after the prepare call so
      // setting the deterministic attribute won't impact # of pathfinding
      // threads. If m_deterministic is not set, there is some possibility of a
      // small deviation in sampling results from finite precision on a
      // different contraction path
      const int32_t rndSeed = m_randomEngine();
      HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
          m_cutnHandle, sampler, CUTENSORNET_SAMPLER_CONFIG_DETERMINISTIC,
          &rndSeed, sizeof(rndSeed)));
    }

    {
      ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                             "cutensornetSamplerSample");
      HANDLE_CUTN_ERROR(cutensornetSamplerSample(
          m_cutnHandle, sampler, numShots, workDesc, samples.data(),
          /*cudaStream*/ 0));
    }

    const auto numMeasuredQubits = measuredBitIds.size();
    std::string bitstring(numMeasuredQubits, '0');
    for (int64_t i = 0; i < numShots; ++i) {
      constexpr char digits[2] = {'0', '1'};
      for (std::size_t j = 0; j < numMeasuredQubits; ++j)
        bitstring[j] = digits[samples[i * numMeasuredQubits + j]];
      counts[bitstring] += 1;
    }
    shotsToRun -= numShots;
  }

  if (enableCacheWorkspace && d_cache) {
    HANDLE_CUDA_ERROR(cudaFree(d_cache));
  }
  return counts;
}

template <typename ScalarType>
std::unordered_map<std::string, size_t>
TensorNetState<ScalarType>::sample(const std::vector<int32_t> &measuredBitIds,
                                   int32_t shots, bool enableCacheWorkspace) {
  LOG_API_TIME();
  auto [sampler, workDesc] = prepareSample(measuredBitIds);
  std::unordered_map<std::string, size_t> counts = executeSample(
      sampler, workDesc, measuredBitIds, shots, enableCacheWorkspace);
  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  // Destroy the quantum circuit sampler
  HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
  return counts;
}

template <typename ScalarType>
std::pair<void *, std::size_t>
TensorNetState<ScalarType>::contractStateVectorInternal(
    const std::vector<int32_t> &projectedModes,
    const std::vector<int64_t> &in_projectedModeValues) {
  // Make sure that we don't overflow the memory size calculation.
  // Note: the actual limitation will depend on the system memory.
  if ((m_numQubits - projectedModes.size()) > 64 ||
      (1ull << (m_numQubits - projectedModes.size())) >
          std::numeric_limits<uint64_t>::max() /
              sizeof(std::complex<ScalarType>))
    throw std::runtime_error(
        "Too many qubits are requested for full state vector contraction.");
  LOG_API_TIME();
  void *d_sv{nullptr};
  const uint64_t svDim = 1ull << (m_numQubits - projectedModes.size());
  {
    ScopedTraceWithContext(
        "TensorNetState<ScalarType>::contractStateVectorInternal "
        "State vector allocation");
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_sv, svDim * sizeof(std::complex<ScalarType>)));
  }
  // Create the quantum state amplitudes accessor
  cutensornetStateAccessor_t accessor;
  {
    ScopedTraceWithContext("cutensornetCreateAccessor");
    HANDLE_CUTN_ERROR(cutensornetCreateAccessor(
        m_cutnHandle, m_quantumState, projectedModes.size(),
        projectedModes.data(), nullptr, &accessor));
  }

  {
    ScopedTraceWithContext("cutensornetAccessorConfigure");
    HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(
        m_cutnHandle, accessor, CUTENSORNET_ACCESSOR_CONFIG_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));
  }
  // Prepare the quantum state amplitudes accessor
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetAccessorPrepare");
    HANDLE_CUTN_ERROR(cutensornetAccessorPrepare(
        m_cutnHandle, accessor, scratchPad.scratchSize, workDesc, 0));
  }
  // Attach the workspace buffer
  int64_t worksize = 0;
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  if (worksize <= static_cast<int64_t>(scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }

  // Compute the quantum state amplitudes
  std::complex<ScalarType> stateNorm{0.0, 0.0};
  if (!in_projectedModeValues.empty() &&
      in_projectedModeValues.size() != projectedModes.size())
    throw std::invalid_argument(cudaq_fmt::format(
        "The number of projected modes ({}) must equal the number of "
        "projected values ({}).",
        projectedModes.size(), in_projectedModeValues.size()));
  // All projected modes are assumed to be projected to 0 if none provided.
  std::vector<int64_t> projectedModeValues =
      in_projectedModeValues.empty()
          ? std::vector<int64_t>(projectedModes.size(), 0)
          : in_projectedModeValues;
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetAccessorCompute");
    HANDLE_CUTN_ERROR(cutensornetAccessorCompute(
        m_cutnHandle, accessor, projectedModeValues.data(), workDesc, d_sv,
        static_cast<void *>(&stateNorm), 0));
  }
  // Free resources
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));

  return std::make_pair(d_sv, svDim);
}

template <typename ScalarType>
std::vector<MPSTensor> TensorNetState<ScalarType>::setupMPSFactorize(
    int64_t maxExtent, double absCutoff, double relCutoff,
    cutensornetTensorSVDAlgo_t algo,
    const std::optional<cutensornetStateMPSGaugeOption_t> &gauge) {
  LOG_API_TIME();
  if (m_numQubits == 0)
    return {};
  if (m_numQubits == 1) {
    // Single tensor
    MPSTensor tensor;
    tensor.extents = {2};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&tensor.deviceData, 2 * sizeof(std::complex<ScalarType>)));

    return {tensor};
  }

  std::vector<MPSTensor> mpsTensors(m_numQubits);
  std::vector<int64_t *> extentsPtr(m_numQubits);
  for (std::size_t i = 0; i < m_numQubits; ++i) {
    if (i == 0) {
      mpsTensors[i].extents = {2, maxExtent};
      HANDLE_CUDA_ERROR(
          cudaMalloc(&mpsTensors[i].deviceData,
                     2 * maxExtent * sizeof(std::complex<ScalarType>)));
    } else if (i == m_numQubits - 1) {
      mpsTensors[i].extents = {maxExtent, 2};
      HANDLE_CUDA_ERROR(
          cudaMalloc(&mpsTensors[i].deviceData,
                     2 * maxExtent * sizeof(std::complex<ScalarType>)));
    } else {
      mpsTensors[i].extents = {maxExtent, 2, maxExtent};
      HANDLE_CUDA_ERROR(cudaMalloc(&mpsTensors[i].deviceData,
                                   2 * maxExtent * maxExtent *
                                       sizeof(std::complex<ScalarType>)));
    }
    extentsPtr[i] = mpsTensors[i].extents.data();
  }
  {
    ScopedTraceWithContext("cutensornetStateFinalizeMPS");
    // Specify the final target MPS representation (use default fortran strides)
    HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
        m_cutnHandle, m_quantumState, CUTENSORNET_BOUNDARY_CONDITION_OPEN,
        extentsPtr.data(), /*strides=*/nullptr));
  }
  // Set up the SVD method for truncation.
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      m_cutnHandle, m_quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_ALGO,
      &algo, sizeof(algo)));
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      m_cutnHandle, m_quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_ABS_CUTOFF,
      &absCutoff, sizeof(absCutoff)));
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      m_cutnHandle, m_quantumState, CUTENSORNET_STATE_CONFIG_MPS_SVD_REL_CUTOFF,
      &relCutoff, sizeof(relCutoff)));
  if (gauge.has_value()) {
    cutensornetStateMPSGaugeOption_t gaugeOption = gauge.value();
    HANDLE_CUTN_ERROR(cutensornetStateConfigure(
        m_cutnHandle, m_quantumState, CUTENSORNET_STATE_CONFIG_MPS_GAUGE_OPTION,
        &gaugeOption, sizeof(gaugeOption)));
  }
  return mpsTensors;
}

template <typename ScalarType>
void TensorNetState<ScalarType>::computeMPSFactorize(
    std::vector<MPSTensor> &mpsTensors) {
  LOG_API_TIME();
  if (mpsTensors.empty())
    return;
  if (mpsTensors.size() == 1) {
    MPSTensor &tensor = mpsTensors[0];
    // Just contract all the gates to the tensor.
    // Note: if none gates, don't call `getStateVector`, which performs a
    // contraction (`Flop count is zero` error).
    const std::vector<std::complex<ScalarType>> stateVec =
        isDirty() ? getStateVector()
                  : std::vector<std::complex<ScalarType>>{1.0, 0.0};
    assert(stateVec.size() == 2);
    HANDLE_CUDA_ERROR(cudaMemcpy(tensor.deviceData, stateVec.data(),
                                 2 * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyHostToDevice));
    return;
  }

  std::vector<int64_t *> extentsPtr(mpsTensors.size());
  std::vector<void *> allData(mpsTensors.size());
  for (std::size_t i = 0; auto &tensor : mpsTensors) {
    allData[i] = tensor.deviceData;
    extentsPtr[i] = tensor.extents.data();
    ++i;
  }

  // Prepare the MPS computation and attach workspace
  cutensornetWorkspaceDescriptor_t workDesc;

  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET, "cutensornetStatePrepare");
    HANDLE_CUTN_ERROR(cutensornetStatePrepare(
        m_cutnHandle, m_quantumState, scratchPad.scratchSize, workDesc, 0));
  }
  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  if (worksize <= static_cast<int64_t>(scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }
  int64_t hostWorkspaceSize;
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_HOST, CUTENSORNET_WORKSPACE_SCRATCH,
      &hostWorkspaceSize));

  void *hostWork = nullptr;
  if (hostWorkspaceSize > 0) {
    hostWork = malloc(hostWorkspaceSize);
    if (!hostWork) {
      throw std::runtime_error("Unable to allocate " +
                               std::to_string(hostWorkspaceSize) +
                               " bytes for cuTensorNet host workspace.");
    }
  }

  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
      m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_HOST,
      CUTENSORNET_WORKSPACE_SCRATCH, hostWork, hostWorkspaceSize));

  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET, "cutensornetStateCompute");
    // Execute MPS computation
    HANDLE_CUTN_ERROR(cutensornetStateCompute(
        m_cutnHandle, m_quantumState, workDesc, extentsPtr.data(),
        /*strides=*/nullptr, allData.data(), 0));
  }

  if (hostWork)
    free(hostWork);
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
TensorNetState<ScalarType>::getStateVector(
    const std::vector<int32_t> &projectedModes,
    const std::vector<int64_t> &projectedModeValues) {
  auto [d_sv, svDim] =
      contractStateVectorInternal(projectedModes, projectedModeValues);
  std::vector<std::complex<ScalarType>> h_sv(svDim);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_sv.data(), d_sv,
                               svDim * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToHost));
  // Free resources
  HANDLE_CUDA_ERROR(cudaFree(d_sv));

  return h_sv;
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
TensorNetState<ScalarType>::computeRDM(const std::vector<int32_t> &qubits) {
  // Make sure that we don't overflow the memory size calculation.
  // Note: the actual limitation will depend on the system memory.
  if (qubits.size() >= 32 ||
      (1ull << (2 * qubits.size())) > std::numeric_limits<uint64_t>::max() /
                                          sizeof(std::complex<ScalarType>))
    throw std::runtime_error("Too many qubits are requested for reduced "
                             "density matrix contraction.");
  LOG_API_TIME();
  void *d_rdm{nullptr};
  const uint64_t rdmSize = 1ull << (2 * qubits.size());
  const uint64_t rdmSizeBytes = rdmSize * sizeof(std::complex<ScalarType>);
  HANDLE_CUDA_ERROR(cudaMalloc(&d_rdm, rdmSizeBytes));

  cutensornetStateMarginal_t marginal;
  {
    ScopedTraceWithContext("cutensornetCreateMarginal");
    HANDLE_CUTN_ERROR(cutensornetCreateMarginal(
        m_cutnHandle, m_quantumState, qubits.size(), qubits.data(),
        /*numProjectedModes*/ 0, /*projectedModes*/ nullptr,
        /*marginalTensorStrides*/ nullptr, &marginal));
  }

  {
    ScopedTraceWithContext("cutensornetMarginalConfigure");
    HANDLE_CUTN_ERROR(cutensornetMarginalConfigure(
        m_cutnHandle, marginal, CUTENSORNET_MARGINAL_CONFIG_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));
  }

  // Prepare the specified quantum circuit reduced density matrix (marginal)
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetMarginalPrepare");
    HANDLE_CUTN_ERROR(cutensornetMarginalPrepare(
        m_cutnHandle, marginal, scratchPad.scratchSize, workDesc, 0));
  }
  // Attach the workspace buffer
  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  if (worksize <= static_cast<int64_t>(scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetMarginalCompute");
    // Compute the specified quantum circuit reduced density matrix (marginal)
    HANDLE_CUTN_ERROR(cutensornetMarginalCompute(m_cutnHandle, marginal,
                                                 nullptr, workDesc, d_rdm, 0));
  }
  std::vector<std::complex<ScalarType>> h_rdm(rdmSize);
  HANDLE_CUDA_ERROR(
      cudaMemcpy(h_rdm.data(), d_rdm, rdmSizeBytes, cudaMemcpyDeviceToHost));

  // Clean up
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyMarginal(marginal));
  HANDLE_CUDA_ERROR(cudaFree(d_rdm));

  return h_rdm;
}

// Returns MPS tensors (device mems)
// Note: user needs to clean up these tensors
template <typename ScalarType>
std::vector<MPSTensor> TensorNetState<ScalarType>::factorizeMPS(
    int64_t maxExtent, double absCutoff, double relCutoff,
    cutensornetTensorSVDAlgo_t algo,
    const std::optional<cutensornetStateMPSGaugeOption_t> &gauge) {
  LOG_API_TIME();
  auto mpsTensors =
      setupMPSFactorize(maxExtent, absCutoff, relCutoff, algo, gauge);
  computeMPSFactorize(mpsTensors);
  return mpsTensors;
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
TensorNetState<ScalarType>::computeExpVals(
    const std::vector<cudaq::spin_op_term> &product_terms,
    const std::optional<std::size_t> &numberTrajectories) {
  LOG_API_TIME();
  if (product_terms.empty())
    return {};

  const std::size_t numQubits = getNumQubits();

  constexpr int ALIGNMENT_BYTES = 256;
  const int placeHolderArraySize = ALIGNMENT_BYTES * numQubits;

  void *pauliMats_h = malloc(placeHolderArraySize);
  void *pauliMats_d{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&pauliMats_d, placeHolderArraySize));
  std::vector<const void *> pauliTensorData;
  std::vector<std::vector<int32_t>> stateModes;

  for (std::size_t i = 0; i < numQubits; ++i) {
    pauliTensorData.emplace_back(static_cast<char *>(pauliMats_d) +
                                 ALIGNMENT_BYTES * i);
    stateModes.emplace_back(std::vector<int32_t>{static_cast<int32_t>(i)});
  }

  const std::vector<int64_t> qubitDims(numQubits, 2);

  // Initialize device mem for Pauli matrices
  constexpr std::complex<ScalarType> PauliI_h[4] = {
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};

  constexpr std::complex<ScalarType> PauliX_h[4]{
      {0.0, 0.0}, {1.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}};

  constexpr std::complex<ScalarType> PauliY_h[4]{
      {0.0, 0.0}, {0.0, -1.0}, {0.0, 1.0}, {0.0, 0.0}};

  constexpr std::complex<ScalarType> PauliZ_h[4]{
      {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {-1.0, 0.0}};

  cutensornetNetworkOperator_t cutnNetworkOperator;

  HANDLE_CUTN_ERROR(cutensornetCreateNetworkOperator(
      m_cutnHandle, numQubits, qubitDims.data(), cudaDataType,
      &cutnNetworkOperator));

  const std::vector<int32_t> numModes(pauliTensorData.size(), 1);
  int64_t id;
  std::vector<const int32_t *> dataStateModes;
  for (const auto &stateMode : stateModes) {
    dataStateModes.emplace_back(stateMode.data());
  }
  const cuDoubleComplex termCoeff{1.0, 0.0};
  HANDLE_CUTN_ERROR(cutensornetNetworkOperatorAppendProduct(
      m_cutnHandle, cutnNetworkOperator, termCoeff, pauliTensorData.size(),
      numModes.data(), dataStateModes.data(),
      /*tensorModeStrides*/ nullptr, pauliTensorData.data(), &id));

  // Step 1: create
  cutensornetStateExpectation_t tensorNetworkExpectation;
  {
    ScopedTraceWithContext("cutensornetCreateExpectation");
    HANDLE_CUTN_ERROR(cutensornetCreateExpectation(m_cutnHandle, m_quantumState,
                                                   cutnNetworkOperator,
                                                   &tensorNetworkExpectation));
  }
  // Step 2: configure
  {
    ScopedTraceWithContext("cutensornetExpectationConfigure");
    HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(
        m_cutnHandle, tensorNetworkExpectation,
        CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples,
        sizeof(numHyperSamples)));
  }

  // Step 3: Prepare
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetExpectationPrepare");
    HANDLE_CUTN_ERROR(
        cutensornetExpectationPrepare(m_cutnHandle, tensorNetworkExpectation,
                                      scratchPad.scratchSize, workDesc,
                                      /*cudaStream*/ 0));
  }

  if (::cudaq::details::should_log(::cudaq::details::LogLevel::info)) {
    double flops = 0.0;
    HANDLE_CUTN_ERROR(cutensornetExpectationGetInfo(
        m_cutnHandle, tensorNetworkExpectation,
        CUTENSORNET_EXPECTATION_INFO_FLOPS, &flops, sizeof(flops)));
    CUDAQ_INFO("Total flop count = {} GFlop.", (flops / 1e9));
  }

  // Attach the workspace buffer
  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  if (worksize <= static_cast<int64_t>(scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }

  // Step 4: Compute
  const std::size_t numObserveTrajectories = [&]() -> std::size_t {
    if (!m_hasNoiseChannel)
      return 1;
    if (numberTrajectories.has_value())
      return numberTrajectories.value();
    return g_numberTrajectoriesForObserve;
  }();

  std::vector<std::complex<ScalarType>> allExpVals;
  allExpVals.reserve(product_terms.size());

  // NOTE: The logic in the loop below relies on the following:
  // Spin operator terms are canonically ordered. Specifically, we can
  // assume that every operator does not act on the same target more than
  // once. That assumption is only checked via an assertion.
  // Additionally, the loops that inject identities rely on the ordering
  // starting with the smallest index/degree. We could write it agnostic
  // by querying cudaq::operator_handler::canonical_order, but I kept it
  // at putting an assert in for that one, too.
  assert(cudaq::operator_handler::canonical_order(0, 1));
  constexpr int PAULI_ARRAY_SIZE_BYTES = 4 * sizeof(std::complex<ScalarType>);
  for (const auto &prod : product_terms) {
    assert(prod.is_canonicalized());
    bool allIdOps = true;
    auto offset = 0;
    for (const auto &p : prod) {
      // The Pauli matrix data that we want to load to this slot.
      // Default is the Identity matrix.
      const std::complex<ScalarType> *pauliMatrixPtr = PauliI_h;
      // We need to make sure to populate the identity for all qubits
      // that are not part of this term
      while (offset < p.target()) {
        auto *address =
            static_cast<char *>(pauliMats_h) + offset++ * ALIGNMENT_BYTES;
        std::memcpy(address, pauliMatrixPtr, PAULI_ARRAY_SIZE_BYTES);
      }
      // Memory address of this Pauli term in the placeholder array.
      auto *address =
          static_cast<char *>(pauliMats_h) + offset++ * ALIGNMENT_BYTES;
      auto pauli = p.as_pauli();
      if (pauli == cudaq::pauli::Y) {
        allIdOps = false;
        pauliMatrixPtr = PauliY_h;
      } else if (pauli == cudaq::pauli::X) {
        allIdOps = false;
        pauliMatrixPtr = PauliX_h;
      } else if (pauli == cudaq::pauli::Z) {
        allIdOps = false;
        pauliMatrixPtr = PauliZ_h;
      }
      // Copy the Pauli matrix data to the placeholder array at the appropriate
      // slot.
      std::memcpy(address, pauliMatrixPtr, PAULI_ARRAY_SIZE_BYTES);
    }
    // Populate the remaining identities.
    const std::complex<ScalarType> *pauliMatrixPtr = PauliI_h;
    while (offset < numQubits) {
      auto *address =
          static_cast<char *>(pauliMats_h) + offset++ * ALIGNMENT_BYTES;
      std::memcpy(address, pauliMatrixPtr, PAULI_ARRAY_SIZE_BYTES);
    }
    if (allIdOps) {
      allExpVals.emplace_back(prod.evaluate_coefficient());
    } else {
      HANDLE_CUDA_ERROR(cudaMemcpy(pauliMats_d, pauliMats_h,
                                   placeHolderArraySize,
                                   cudaMemcpyHostToDevice));
      std::complex<ScalarType> expVal = 0.0;
      for (std::size_t trajId = 0; trajId < numObserveTrajectories; ++trajId) {
        std::complex<ScalarType> result;
        ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                               "cutensornetExpectationCompute");
        HANDLE_CUTN_ERROR(cutensornetExpectationCompute(
            m_cutnHandle, tensorNetworkExpectation, workDesc, &result, nullptr,
            /*cudaStream*/ 0));
        expVal += (result / static_cast<ScalarType>(numObserveTrajectories));
      }
      const std::complex<double> coeff = prod.evaluate_coefficient();
      allExpVals.emplace_back(
          expVal * std::complex<ScalarType>(coeff.real(), coeff.imag()));
    }
  }

  // Clean up cutensornet resources
  HANDLE_CUTN_ERROR(cutensornetDestroyExpectation(tensorNetworkExpectation));
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkOperator(cutnNetworkOperator));

  free(pauliMats_h);
  HANDLE_CUDA_ERROR(cudaFree(pauliMats_d));

  return allExpVals;
}

template <typename ScalarType>
std::complex<ScalarType> TensorNetState<ScalarType>::computeExpVal(
    cutensornetNetworkOperator_t tensorNetworkOperator,
    const std::optional<std::size_t> &numberTrajectories) {
  LOG_API_TIME();
  cutensornetStateExpectation_t tensorNetworkExpectation;
  // Step 1: create
  {
    ScopedTraceWithContext("cutensornetCreateExpectation");
    HANDLE_CUTN_ERROR(cutensornetCreateExpectation(m_cutnHandle, m_quantumState,
                                                   tensorNetworkOperator,
                                                   &tensorNetworkExpectation));
  }
  // Step 2: configure
  {
    ScopedTraceWithContext("cutensornetExpectationConfigure");
    HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(
        m_cutnHandle, tensorNetworkExpectation,
        CUTENSORNET_EXPECTATION_CONFIG_NUM_HYPER_SAMPLES, &numHyperSamples,
        sizeof(numHyperSamples)));
  }

  // Step 3: Prepare
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetExpectationPrepare");
    HANDLE_CUTN_ERROR(cutensornetExpectationPrepare(
        m_cutnHandle, tensorNetworkExpectation, scratchPad.scratchSize,
        workDesc, /*cudaStream*/ 0));
  }

  if (::cudaq::details::should_log(::cudaq::details::LogLevel::info)) {
    double flops = 0.0;
    HANDLE_CUTN_ERROR(cutensornetExpectationGetInfo(
        m_cutnHandle, tensorNetworkExpectation,
        CUTENSORNET_EXPECTATION_INFO_FLOPS, &flops, sizeof(flops)));
    CUDAQ_INFO("Total flop count = {} GFlop.", (flops / 1e9));
  }

  // Attach the workspace buffer
  int64_t worksize{0};
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  if (worksize <= static_cast<int64_t>(scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }

  // Step 4: Compute
  const std::size_t numObserveTrajectories = [&]() -> std::size_t {
    if (!m_hasNoiseChannel)
      return 1;
    if (numberTrajectories.has_value())
      return numberTrajectories.value();
    return g_numberTrajectoriesForObserve;
  }();

  std::complex<ScalarType> expVal = 0.0;
  for (std::size_t trajId = 0; trajId < numObserveTrajectories; ++trajId) {
    std::complex<ScalarType> result;
    ScopedTraceWithContext(cudaq::TIMING_TENSORNET,
                           "cutensornetExpectationCompute");
    HANDLE_CUTN_ERROR(cutensornetExpectationCompute(
        m_cutnHandle, tensorNetworkExpectation, workDesc, &result,
        /*stateNorm*/ nullptr,
        /*cudaStream*/ 0));
    expVal += (result / static_cast<ScalarType>(numObserveTrajectories));
  }
  // Step 5: clean up
  HANDLE_CUTN_ERROR(cutensornetDestroyExpectation(tensorNetworkExpectation));
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  return expVal;
}

template <typename ScalarType>
std::unique_ptr<TensorNetState<ScalarType>>
TensorNetState<ScalarType>::createFromMpsTensors(
    const std::vector<MPSTensor> &in_mpsTensors, ScratchDeviceMem &inScratchPad,
    cutensornetHandle_t handle, std::mt19937 &randomEngine) {
  LOG_API_TIME();
  if (in_mpsTensors.empty())
    throw std::invalid_argument("Empty MPS tensor list");
  auto state = std::make_unique<TensorNetState>(
      in_mpsTensors.size(), inScratchPad, handle, randomEngine);
  std::vector<const int64_t *> extents;
  std::vector<void *> tensorData;
  for (const auto &tensor : in_mpsTensors) {
    extents.emplace_back(tensor.extents.data());
    tensorData.emplace_back(tensor.deviceData);
  }
  HANDLE_CUTN_ERROR(cutensornetStateInitializeMPS(
      handle, state->m_quantumState, CUTENSORNET_BOUNDARY_CONDITION_OPEN,
      extents.data(), nullptr, tensorData.data()));
  return state;
}

/// Reconstruct/initialize a tensor network state from a list of tensor
/// operators.
template <typename ScalarType>
std::unique_ptr<TensorNetState<ScalarType>>
TensorNetState<ScalarType>::createFromOpTensors(
    std::size_t numQubits, const std::vector<AppliedTensorOp> &opTensors,
    ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle,
    std::mt19937 &randomEngine) {
  LOG_API_TIME();
  auto state = std::make_unique<TensorNetState>(numQubits, inScratchPad, handle,
                                                randomEngine);
  for (const auto &op : opTensors)
    if (op.isUnitary)
      state->applyGate(op.controlQubitIds, op.targetQubitIds, op.deviceData,
                       op.isAdjoint);
    else
      state->applyQubitProjector(op.deviceData, op.targetQubitIds);

  return state;
}

template <typename ScalarType>
std::vector<std::complex<ScalarType>>
TensorNetState<ScalarType>::reverseQubitOrder(
    std::span<std::complex<ScalarType>> stateVec) {
  std::vector<std::complex<ScalarType>> ket(stateVec.size());
  const std::size_t numQubits = std::log2(stateVec.size());
  for (std::size_t i = 0; i < stateVec.size(); ++i) {
    std::bitset<64> bs(i);
    std::string bitStr = bs.to_string();
    std::reverse(bitStr.begin(), bitStr.end());
    bitStr = bitStr.substr(0, numQubits);
    ket[std::stoull(bitStr, nullptr, 2)] = stateVec[i];
  }
  return ket;
}

template <typename ScalarType>
bool TensorNetState<ScalarType>::hasGeneralChannelApplied() const {
  for (const auto &op : m_tensorOps)
    if (op.noiseChannel.has_value() && op.noiseChannel->probabilities.empty())
      return true;

  return false;
}

template <typename ScalarType>
void TensorNetState<ScalarType>::applyCachedOps() {
  int64_t tensorId = 0;
  for (auto &op : m_tensorOps)
    if (op.deviceData) {
      if (op.controlQubitIds.empty()) {
        HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
            m_cutnHandle, m_quantumState, op.targetQubitIds.size(),
            op.targetQubitIds.data(), op.deviceData, nullptr, /*immutable*/ 1,
            /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
            /*unitary*/ static_cast<int32_t>(op.isUnitary), &tensorId));
      } else {
        HANDLE_CUTN_ERROR(cutensornetStateApplyControlledTensorOperator(
            m_cutnHandle, m_quantumState,
            /*numControlModes=*/op.controlQubitIds.size(),
            /*stateControlModes=*/op.controlQubitIds.data(),
            /*stateControlValues=*/nullptr,
            /*numTargetModes*/ op.targetQubitIds.size(),
            /*stateTargetModes*/ op.targetQubitIds.data(), op.deviceData,
            nullptr,
            /*immutable*/ 1,
            /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
            /*unitary*/ static_cast<int32_t>(op.isUnitary), &m_tensorId));
      }
    } else if (op.noiseChannel.has_value()) {
      const bool isGeneralChannel = op.noiseChannel->tensorData.size() !=
                                    op.noiseChannel->probabilities.size();
      if (isGeneralChannel) {
        HANDLE_CUTN_ERROR(cutensornetStateApplyGeneralChannel(
            m_cutnHandle, m_quantumState,
            /*numStateModes=*/op.targetQubitIds.size(),
            /*stateModes=*/op.targetQubitIds.data(),
            /*numTensors=*/op.noiseChannel->tensorData.size(),
            /*tensorData=*/op.noiseChannel->tensorData.data(),
            /*tensorModeStrides=*/nullptr, &m_tensorId));
      } else {
        HANDLE_CUTN_ERROR(cutensornetStateApplyUnitaryChannel(
            m_cutnHandle, m_quantumState,
            /*numStateModes=*/op.targetQubitIds.size(),
            /*stateModes=*/op.targetQubitIds.data(),
            /*numTensors=*/op.noiseChannel->tensorData.size(),
            /*tensorData=*/op.noiseChannel->tensorData.data(),
            /*tensorModeStrides=*/nullptr,
            /*probabilities=*/op.noiseChannel->probabilities.data(),
            &m_tensorId));
      }
    } else {
      throw std::runtime_error("Invalid AppliedTensorOp encountered.");
    }
}

template <typename ScalarType>
void TensorNetState<ScalarType>::setZeroState() {
  LOG_API_TIME();
  // Destroy the current quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
  const std::vector<int64_t> qubitDims(m_numQubits, 2);
  // Re-create the state
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubits,
      qubitDims.data(), cudaDataType, &m_quantumState));
}

template <typename ScalarType>
std::unique_ptr<TensorNetState<ScalarType>>
TensorNetState<ScalarType>::createFromStateVector(
    std::span<std::complex<ScalarType>> stateVec,
    ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle,
    std::mt19937 &randomEngine) {
  LOG_API_TIME();
  const std::size_t numQubits = std::log2(stateVec.size());
  auto state = std::make_unique<TensorNetState>(numQubits, inScratchPad, handle,
                                                randomEngine);

  // Support initializing the tensor network in a specific state vector state.
  // Note: this is not intended for large state vector but for relatively small
  // number of qubits. The purpose is to support sub-state (e.g., a portion of
  // the qubit register) initialization. For full state re-initialization, the
  // previous state should be in the tensor network form. Construct the state
  // projector matrix
  // FIXME: use CUDA toolkit, e.g., cuBlas, to construct this projector matrix.
  // Reverse the qubit order to match cutensornet convention
  auto newStateVec = reverseQubitOrder(stateVec);
  auto ket =
      Eigen::Map<Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>>(
          newStateVec.data(), newStateVec.size());
  Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic> initState =
      Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>::Zero(
          stateVec.size());
  initState(0) = std::complex<ScalarType>{1.0, 0.0};
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
      stateVecProj = ket * initState.transpose();
  assert(static_cast<std::size_t>(stateVecProj.size()) ==
         stateVec.size() * stateVec.size());
  stateVecProj.transposeInPlace();
  void *d_proj{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_proj, stateVecProj.size() *
                                            sizeof(std::complex<ScalarType>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_proj, stateVecProj.data(),
                 stateVecProj.size() * sizeof(std::complex<ScalarType>),
                 cudaMemcpyHostToDevice));

  std::vector<int32_t> qubitIdx(numQubits);
  std::iota(qubitIdx.begin(), qubitIdx.end(), 0);
  // Project the state to the input state.
  state->applyQubitProjector(d_proj, qubitIdx);
  state->m_tempDevicePtrs.emplace_back(d_proj, TempDevicePtrDeleter{});
  return state;
}

template <typename ScalarType>
TensorNetState<ScalarType>::~TensorNetState() {
  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
  m_tempDevicePtrs.clear();
}

} // namespace nvqir
