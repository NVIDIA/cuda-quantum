/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "tensornet_state.h"
#include "common/EigenDense.h"
#include <bitset>
#include <cassert>

namespace nvqir {

TensorNetState::TensorNetState(std::size_t numQubits,
                               ScratchDeviceMem &inScratchPad,
                               cutensornetHandle_t handle)
    : m_numQubits(numQubits), m_cutnHandle(handle), scratchPad(inScratchPad) {
  const std::vector<int64_t> qubitDims(m_numQubits, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubits,
      qubitDims.data(), CUDA_C_64F, &m_quantumState));
}

TensorNetState::TensorNetState(const std::vector<int> &basisState,
                               ScratchDeviceMem &inScratchPad,
                               cutensornetHandle_t handle)
    : TensorNetState(basisState.size(), inScratchPad, handle) {
  constexpr std::complex<double> h_xGate[4] = {0.0, 1.0, 1.0, 0.0};
  constexpr auto sizeBytes = 4 * sizeof(std::complex<double>);
  void *d_gate{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&d_gate, sizeBytes));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_gate, h_xGate, sizeBytes, cudaMemcpyHostToDevice));
  m_tempDevicePtrs.emplace_back(d_gate);
  for (int32_t qId = 0; const auto &bit : basisState) {
    if (bit == 1) {
      applyGate({}, {qId}, d_gate);
    }
    ++qId;
  }
}

std::unique_ptr<TensorNetState> TensorNetState::clone() const {
  return createFromOpTensors(m_numQubits, m_tensorOps, scratchPad,
                             m_cutnHandle);
}

void TensorNetState::applyGate(const std::vector<int32_t> &controlQubits,
                               const std::vector<int32_t> &targetQubits,
                               void *gateDeviceMem, bool adjoint) {
  LOG_API_TIME();
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

void TensorNetState::applyQubitProjector(void *proj_d,
                                         const std::vector<int32_t> &qubitIdx) {
  LOG_API_TIME();
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
      m_cutnHandle, m_quantumState, qubitIdx.size(), qubitIdx.data(), proj_d,
      nullptr,
      /*immutable*/ 1,
      /*adjoint*/ 0, /*unitary*/ 0, &m_tensorId));
  m_tensorOps.emplace_back(AppliedTensorOp{proj_d, qubitIdx, {}, false, false});
}

void TensorNetState::addQubits(std::size_t numQubits) {
  LOG_API_TIME();
  // Destroy the current quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
  m_numQubits += numQubits;
  const std::vector<int64_t> qubitDims(m_numQubits, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubits,
      qubitDims.data(), CUDA_C_64F, &m_quantumState));
  // Append any previously-applied gate tensors.
  // These tensors will only be appending to those existing qubit wires, i.e.,
  // the new wires are all empty (zero state).
  int64_t tensorId = 0;
  for (auto &op : m_tensorOps)
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
          /*stateTargetModes*/ op.targetQubitIds.data(), op.deviceData, nullptr,
          /*immutable*/ 1,
          /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
          /*unitary*/ static_cast<int32_t>(op.isUnitary), &m_tensorId));
    }
}

void TensorNetState::addQubits(std::span<std::complex<double>> stateVec) {
  LOG_API_TIME();
  const std::size_t numQubits = std::log2(stateVec.size());
  auto ket =
      Eigen::Map<const Eigen::VectorXcd>(stateVec.data(), stateVec.size());
  Eigen::VectorXcd initState = Eigen::VectorXcd::Zero(stateVec.size());
  initState(0) = std::complex<double>{1.0, 0.0};
  Eigen::MatrixXcd stateVecProj = ket * initState.transpose();
  assert(static_cast<std::size_t>(stateVecProj.size()) ==
         stateVec.size() * stateVec.size());
  stateVecProj.transposeInPlace();
  void *d_proj{nullptr};
  HANDLE_CUDA_ERROR(
      cudaMalloc(&d_proj, stateVecProj.size() * sizeof(std::complex<double>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_proj, stateVecProj.data(),
                 stateVecProj.size() * sizeof(std::complex<double>),
                 cudaMemcpyHostToDevice));

  std::vector<int32_t> qubitIdx(numQubits);
  std::iota(qubitIdx.begin(), qubitIdx.end(), m_numQubits);
  // Add qubits in zero state
  addQubits(numQubits);

  // Project the state of those new qubits to the input state.
  applyQubitProjector(d_proj, qubitIdx);
  m_tempDevicePtrs.emplace_back(d_proj);
}

std::unordered_map<std::string, size_t>
TensorNetState::sample(const std::vector<int32_t> &measuredBitIds,
                       int32_t shots) {
  LOG_API_TIME();
  // Create the quantum circuit sampler
  cutensornetStateSampler_t sampler;
  {
    ScopedTraceWithContext("cutensornetCreateSampler");
    HANDLE_CUTN_ERROR(cutensornetCreateSampler(
        m_cutnHandle, m_quantumState, measuredBitIds.size(),
        measuredBitIds.data(), &sampler));
  }

  // Configure the quantum circuit sampler
  constexpr int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  {
    ScopedTraceWithContext("cutensornetSamplerConfigure");
    HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
        m_cutnHandle, sampler, CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));
  }

  // Prepare the quantum circuit sampler
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext("cutensornetSamplerPrepare");
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

  // Sample the quantum circuit state
  std::unordered_map<std::string, size_t> counts;
  constexpr int MAX_SHOTS_PER_RUNS = 10000;
  int shotsToRun = shots;
  while (shotsToRun > 0) {
    const int numShots = std::min(shotsToRun, MAX_SHOTS_PER_RUNS);
    std::vector<int64_t> samples(measuredBitIds.size() * numShots);
    {
      ScopedTraceWithContext("cutensornetSamplerSample");
      HANDLE_CUTN_ERROR(cutensornetSamplerSample(
          m_cutnHandle, sampler, numShots, workDesc, samples.data(),
          /*cudaStream*/ 0));
    }

    const auto numMeasuredQubits = measuredBitIds.size();
    std::string bitstring(numMeasuredQubits, '0');
    for (int i = 0; i < numShots; ++i) {
      constexpr char digits[2] = {'0', '1'};
      for (std::size_t j = 0; j < numMeasuredQubits; ++j)
        bitstring[j] = digits[samples[i * numMeasuredQubits + j]];
      counts[bitstring] += 1;
    }
    shotsToRun -= numShots;
  }

  // Destroy the workspace descriptor
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  // Destroy the quantum circuit sampler
  HANDLE_CUTN_ERROR(cutensornetDestroySampler(sampler));
  return counts;
}

std::pair<void *, std::size_t> TensorNetState::contractStateVectorInternal(
    const std::vector<int32_t> &projectedModes,
    const std::vector<int64_t> &in_projectedModeValues) {
  // Make sure that we don't overflow the memory size calculation.
  // Note: the actual limitation will depend on the system memory.
  if ((m_numQubits - projectedModes.size()) > 64 ||
      (1ull << (m_numQubits - projectedModes.size())) >
          std::numeric_limits<uint64_t>::max() / sizeof(std::complex<double>))
    throw std::runtime_error(
        "Too many qubits are requested for full state vector contraction.");
  LOG_API_TIME();
  void *d_sv{nullptr};
  const uint64_t svDim = 1ull << (m_numQubits - projectedModes.size());
  {
    ScopedTraceWithContext("TensorNetState::contractStateVectorInternal "
                           "State vector allocation");
    HANDLE_CUDA_ERROR(cudaMalloc(&d_sv, svDim * sizeof(std::complex<double>)));
  }
  // Create the quantum state amplitudes accessor
  cutensornetStateAccessor_t accessor;
  {
    ScopedTraceWithContext("cutensornetCreateAccessor");
    HANDLE_CUTN_ERROR(cutensornetCreateAccessor(
        m_cutnHandle, m_quantumState, projectedModes.size(),
        projectedModes.data(), nullptr, &accessor));
  }

  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  {
    ScopedTraceWithContext("cutensornetAccessorConfigure");
    HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(
        m_cutnHandle, accessor, CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));
  }
  // Prepare the quantum state amplitudes accessor
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext("cutensornetAccessorPrepare");
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
  std::complex<double> stateNorm{0.0, 0.0};
  if (!in_projectedModeValues.empty() &&
      in_projectedModeValues.size() != projectedModes.size())
    throw std::invalid_argument(fmt::format(
        "The number of projected modes ({}) must equal the number of "
        "projected values ({}).",
        projectedModes.size(), in_projectedModeValues.size()));
  // All projected modes are assumed to be projected to 0 if none provided.
  std::vector<int64_t> projectedModeValues =
      in_projectedModeValues.empty()
          ? std::vector<int64_t>(projectedModes.size(), 0)
          : in_projectedModeValues;
  {
    ScopedTraceWithContext("cutensornetAccessorCompute");
    HANDLE_CUTN_ERROR(cutensornetAccessorCompute(
        m_cutnHandle, accessor, projectedModeValues.data(), workDesc, d_sv,
        static_cast<void *>(&stateNorm), 0));
  }
  // Free resources
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));

  return std::make_pair(d_sv, svDim);
}

std::vector<std::complex<double>> TensorNetState::getStateVector(
    const std::vector<int32_t> &projectedModes,
    const std::vector<int64_t> &projectedModeValues) {
  auto [d_sv, svDim] =
      contractStateVectorInternal(projectedModes, projectedModeValues);
  std::vector<std::complex<double>> h_sv(svDim);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_sv.data(), d_sv,
                               svDim * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));
  // Free resources
  HANDLE_CUDA_ERROR(cudaFree(d_sv));

  return h_sv;
}

std::vector<std::complex<double>>
TensorNetState::computeRDM(const std::vector<int32_t> &qubits) {
  // Make sure that we don't overflow the memory size calculation.
  // Note: the actual limitation will depend on the system memory.
  if (qubits.size() >= 32 ||
      (1ull << (2 * qubits.size())) >
          std::numeric_limits<uint64_t>::max() / sizeof(std::complex<double>))
    throw std::runtime_error("Too many qubits are requested for reduced "
                             "density matrix contraction.");
  LOG_API_TIME();
  void *d_rdm{nullptr};
  const uint64_t rdmSize = 1ull << (2 * qubits.size());
  const uint64_t rdmSizeBytes = rdmSize * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(cudaMalloc(&d_rdm, rdmSizeBytes));

  cutensornetStateMarginal_t marginal;
  {
    ScopedTraceWithContext("cutensornetCreateMarginal");
    HANDLE_CUTN_ERROR(cutensornetCreateMarginal(
        m_cutnHandle, m_quantumState, qubits.size(), qubits.data(),
        /*numProjectedModes*/ 0, /*projectedModes*/ nullptr,
        /*marginalTensorStrides*/ nullptr, &marginal));
  }

  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  {
    ScopedTraceWithContext("cutensornetMarginalConfigure");
    HANDLE_CUTN_ERROR(cutensornetMarginalConfigure(
        m_cutnHandle, marginal, CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES,
        &numHyperSamples, sizeof(numHyperSamples)));
  }

  // Prepare the specified quantum circuit reduced density matrix (marginal)
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext("cutensornetMarginalPrepare");
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
    ScopedTraceWithContext("cutensornetMarginalCompute");
    // Compute the specified quantum circuit reduced density matrix (marginal)
    HANDLE_CUTN_ERROR(cutensornetMarginalCompute(m_cutnHandle, marginal,
                                                 nullptr, workDesc, d_rdm, 0));
  }
  std::vector<std::complex<double>> h_rdm(rdmSize);
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
std::vector<MPSTensor>
TensorNetState::factorizeMPS(int64_t maxExtent, double absCutoff,
                             double relCutoff,
                             cutensornetTensorSVDAlgo_t algo) {
  LOG_API_TIME();
  if (m_numQubits == 0)
    return {};
  if (m_numQubits == 1) {
    // Single tensor
    MPSTensor tensor;
    tensor.extents = {2};
    HANDLE_CUDA_ERROR(
        cudaMalloc(&tensor.deviceData, 2 * sizeof(std::complex<double>)));
    // Just contract all the gates to the tensor.
    // Note: if none gates, don't call `getStateVector`, which performs a
    // contraction (`Flop count is zero` error).
    const std::vector<std::complex<double>> stateVec =
        isDirty() ? getStateVector()
                  : std::vector<std::complex<double>>{1.0, 0.0};
    assert(stateVec.size() == 2);
    HANDLE_CUDA_ERROR(cudaMemcpy(tensor.deviceData, stateVec.data(),
                                 2 * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    return {tensor};
  }
  std::vector<MPSTensor> mpsTensors(m_numQubits);
  std::vector<int64_t *> extentsPtr(m_numQubits);
  for (std::size_t i = 0; i < m_numQubits; ++i) {
    if (i == 0) {
      mpsTensors[i].extents = {2, maxExtent};
      HANDLE_CUDA_ERROR(
          cudaMalloc(&mpsTensors[i].deviceData,
                     2 * maxExtent * sizeof(std::complex<double>)));
    } else if (i == m_numQubits - 1) {
      mpsTensors[i].extents = {maxExtent, 2};
      HANDLE_CUDA_ERROR(
          cudaMalloc(&mpsTensors[i].deviceData,
                     2 * maxExtent * sizeof(std::complex<double>)));
    } else {
      mpsTensors[i].extents = {maxExtent, 2, maxExtent};
      HANDLE_CUDA_ERROR(
          cudaMalloc(&mpsTensors[i].deviceData,
                     2 * maxExtent * maxExtent * sizeof(std::complex<double>)));
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

  // Prepare the MPS computation and attach workspace
  cutensornetWorkspaceDescriptor_t workDesc;

  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext("cutensornetStatePrepare");
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

  std::vector<void *> allData(m_numQubits);
  for (std::size_t i = 0; auto &tensor : mpsTensors)
    allData[i++] = tensor.deviceData;
  {
    ScopedTraceWithContext("cutensornetStateCompute");
    // Execute MPS computation
    HANDLE_CUTN_ERROR(cutensornetStateCompute(
        m_cutnHandle, m_quantumState, workDesc, extentsPtr.data(),
        /*strides=*/nullptr, allData.data(), 0));
  }

  if (hostWork) {
    free(hostWork);
  }
  return mpsTensors;
}

std::complex<double> TensorNetState::computeExpVal(
    cutensornetNetworkOperator_t tensorNetworkOperator) {
  cutensornetStateExpectation_t tensorNetworkExpectation;
  LOG_API_TIME();

  // Step 1: create
  {
    ScopedTraceWithContext("cutensornetCreateExpectation");
    HANDLE_CUTN_ERROR(cutensornetCreateExpectation(m_cutnHandle, m_quantumState,
                                                   tensorNetworkOperator,
                                                   &tensorNetworkExpectation));
  }
  // Step 2: configure
  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  {
    ScopedTraceWithContext("cutensornetExpectationConfigure");
    HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(
        m_cutnHandle, tensorNetworkExpectation,
        CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES, &numHyperSamples,
        sizeof(numHyperSamples)));
  }

  // Step 3: Prepare
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext("cutensornetExpectationPrepare");
    HANDLE_CUTN_ERROR(cutensornetExpectationPrepare(
        m_cutnHandle, tensorNetworkExpectation, scratchPad.scratchSize,
        workDesc, /*cudaStream*/ 0));
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
  std::complex<double> expVal;
  std::complex<double> stateNorm{0.0, 0.0};
  {
    ScopedTraceWithContext("cutensornetExpectationCompute");
    HANDLE_CUTN_ERROR(cutensornetExpectationCompute(
        m_cutnHandle, tensorNetworkExpectation, workDesc, &expVal,
        static_cast<void *>(&stateNorm),
        /*cudaStream*/ 0));
  }
  // Step 5: clean up
  HANDLE_CUTN_ERROR(cutensornetDestroyExpectation(tensorNetworkExpectation));
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  return expVal / std::abs(stateNorm);
}

std::unique_ptr<TensorNetState> TensorNetState::createFromMpsTensors(
    const std::vector<MPSTensor> &in_mpsTensors, ScratchDeviceMem &inScratchPad,
    cutensornetHandle_t handle) {
  LOG_API_TIME();
  if (in_mpsTensors.empty())
    throw std::invalid_argument("Empty MPS tensor list");
  auto state = std::make_unique<TensorNetState>(in_mpsTensors.size(),
                                                inScratchPad, handle);
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
std::unique_ptr<TensorNetState> TensorNetState::createFromOpTensors(
    std::size_t numQubits, const std::vector<AppliedTensorOp> &opTensors,
    ScratchDeviceMem &inScratchPad, cutensornetHandle_t handle) {
  LOG_API_TIME();
  auto state =
      std::make_unique<TensorNetState>(numQubits, inScratchPad, handle);
  for (const auto &op : opTensors)
    if (op.isUnitary)
      state->applyGate(op.controlQubitIds, op.targetQubitIds, op.deviceData,
                       op.isAdjoint);
    else
      state->applyQubitProjector(op.deviceData, op.targetQubitIds);

  return state;
}

std::vector<std::complex<double>>
TensorNetState::reverseQubitOrder(std::span<std::complex<double>> stateVec) {
  std::vector<std::complex<double>> ket(stateVec.size());
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

std::unique_ptr<TensorNetState>
TensorNetState::createFromStateVector(std::span<std::complex<double>> stateVec,
                                      ScratchDeviceMem &inScratchPad,
                                      cutensornetHandle_t handle) {
  LOG_API_TIME();
  const std::size_t numQubits = std::log2(stateVec.size());
  auto state =
      std::make_unique<TensorNetState>(numQubits, inScratchPad, handle);

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
      Eigen::Map<Eigen::VectorXcd>(newStateVec.data(), newStateVec.size());
  Eigen::VectorXcd initState = Eigen::VectorXcd::Zero(stateVec.size());
  initState(0) = std::complex<double>{1.0, 0.0};
  Eigen::MatrixXcd stateVecProj = ket * initState.transpose();
  assert(static_cast<std::size_t>(stateVecProj.size()) ==
         stateVec.size() * stateVec.size());
  stateVecProj.transposeInPlace();
  void *d_proj{nullptr};
  HANDLE_CUDA_ERROR(
      cudaMalloc(&d_proj, stateVecProj.size() * sizeof(std::complex<double>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_proj, stateVecProj.data(),
                 stateVecProj.size() * sizeof(std::complex<double>),
                 cudaMemcpyHostToDevice));

  std::vector<int32_t> qubitIdx(numQubits);
  std::iota(qubitIdx.begin(), qubitIdx.end(), 0);
  // Project the state to the input state.
  state->applyQubitProjector(d_proj, qubitIdx);
  state->m_tempDevicePtrs.emplace_back(d_proj);
  return state;
}

TensorNetState::~TensorNetState() {
  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
  for (auto *ptr : m_tempDevicePtrs)
    HANDLE_CUDA_ERROR(cudaFree(ptr));
}

} // namespace nvqir
