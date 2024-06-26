/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "tensornet_state.h"
#include <cassert>

namespace nvqir {

TensorNetState::TensorNetState(std::size_t numQubits,
                               cutensornetHandle_t handle)
    : m_numQubits(numQubits), m_cutnHandle(handle) {
  const std::vector<int64_t> qubitDims(m_numQubits, 2);
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      m_cutnHandle, CUTENSORNET_STATE_PURITY_PURE, m_numQubits,
      qubitDims.data(), CUDA_C_64F, &m_quantumState));
}

void TensorNetState::applyGate(const std::vector<int32_t> &controlQubits,
                               const std::vector<int32_t> &targetQubits,
                               void *gateDeviceMem, bool adjoint) {
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
}

void TensorNetState::applyQubitProjector(void *proj_d, int32_t qubitIdx) {
  HANDLE_CUTN_ERROR(cutensornetStateApplyTensorOperator(
      m_cutnHandle, m_quantumState, 1, &qubitIdx, proj_d, nullptr,
      /*immutable*/ 1,
      /*adjoint*/ 0, /*unitary*/ 0, &m_tensorId));
}

std::unordered_map<std::string, size_t>
TensorNetState::sample(const std::vector<int32_t> &measuredBitIds,
                       int32_t shots) {
  LOG_API_TIME();
  // Create the quantum circuit sampler
  cutensornetStateSampler_t sampler;
  HANDLE_CUTN_ERROR(cutensornetCreateSampler(m_cutnHandle, m_quantumState,
                                             measuredBitIds.size(),
                                             measuredBitIds.data(), &sampler));

  ScratchDeviceMem scratchPad;
  // Configure the quantum circuit sampler
  constexpr int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  HANDLE_CUTN_ERROR(cutensornetSamplerConfigure(
      m_cutnHandle, sampler, CUTENSORNET_SAMPLER_OPT_NUM_HYPER_SAMPLES,
      &numHyperSamples, sizeof(numHyperSamples)));

  // Prepare the quantum circuit sampler
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext("TensorNetState::sample::cutensornetSamplerPrepare");
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
      ScopedTraceWithContext(
          "TensorNetState::sample::cutensornetSamplerSample");
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

std::vector<std::complex<double>>
TensorNetState::getStateVector(const std::vector<int32_t> &projectedModes) {
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
  HANDLE_CUDA_ERROR(cudaMalloc(&d_sv, svDim * sizeof(std::complex<double>)));
  ScratchDeviceMem scratchPad;

  // Create the quantum state amplitudes accessor
  cutensornetStateAccessor_t accessor;
  HANDLE_CUTN_ERROR(cutensornetCreateAccessor(
      m_cutnHandle, m_quantumState, projectedModes.size(),
      projectedModes.data(), nullptr, &accessor));

  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(
      m_cutnHandle, accessor, CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES,
      &numHyperSamples, sizeof(numHyperSamples)));
  // Prepare the quantum state amplitudes accessor
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  HANDLE_CUTN_ERROR(cutensornetAccessorPrepare(
      m_cutnHandle, accessor, scratchPad.scratchSize, workDesc, 0));

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
  // All projected modes are assumed to be projected to 0.
  std::vector<int64_t> projectedModeValues(projectedModes.size(), 0);
  HANDLE_CUTN_ERROR(cutensornetAccessorCompute(
      m_cutnHandle, accessor, projectedModeValues.data(), workDesc, d_sv,
      static_cast<void *>(&stateNorm), 0));
  std::vector<std::complex<double>> h_sv(svDim);
  HANDLE_CUDA_ERROR(cudaMemcpy(h_sv.data(), d_sv,
                               svDim * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));

  // Free resources
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));
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
  ScratchDeviceMem scratchPad;

  cutensornetStateMarginal_t marginal;
  HANDLE_CUTN_ERROR(cutensornetCreateMarginal(
      m_cutnHandle, m_quantumState, qubits.size(), qubits.data(),
      /*numProjectedModes*/ 0, /*projectedModes*/ nullptr,
      /*marginalTensorStrides*/ nullptr, &marginal));

  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  HANDLE_CUTN_ERROR(cutensornetMarginalConfigure(
      m_cutnHandle, marginal, CUTENSORNET_MARGINAL_OPT_NUM_HYPER_SAMPLES,
      &numHyperSamples, sizeof(numHyperSamples)));

  // Prepare the specified quantum circuit reduced density matrix (marginal)
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  HANDLE_CUTN_ERROR(cutensornetMarginalPrepare(
      m_cutnHandle, marginal, scratchPad.scratchSize, workDesc, 0));
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

  // Compute the specified quantum circuit reduced density matrix (marginal)
  HANDLE_CUTN_ERROR(cutensornetMarginalCompute(m_cutnHandle, marginal, nullptr,
                                               workDesc, d_rdm, 0));
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
std::vector<void *>
TensorNetState::factorizeMPS(int64_t maxExtent, double absCutoff,
                             double relCutoff,
                             cutensornetTensorSVDAlgo_t algo) {
  LOG_API_TIME();
  std::vector<std::vector<int64_t>> extents;
  std::vector<int64_t *> extentsPtr(m_numQubits);
  std::vector<void *> d_mpsTensors(m_numQubits, nullptr);
  for (std::size_t i = 0; i < m_numQubits; ++i) {
    if (i == 0) {
      extents.push_back({2, maxExtent});
      HANDLE_CUDA_ERROR(cudaMalloc(
          &d_mpsTensors[i], 2 * maxExtent * sizeof(std::complex<double>)));
    } else if (i == m_numQubits - 1) {
      extents.push_back({maxExtent, 2});
      HANDLE_CUDA_ERROR(cudaMalloc(
          &d_mpsTensors[i], 2 * maxExtent * sizeof(std::complex<double>)));
    } else {
      extents.push_back({maxExtent, 2, maxExtent});
      HANDLE_CUDA_ERROR(
          cudaMalloc(&d_mpsTensors[i],
                     2 * maxExtent * maxExtent * sizeof(std::complex<double>)));
    }
    extentsPtr[i] = extents[i].data();
  }

  // Specify the final target MPS representation (use default fortran strides)
  HANDLE_CUTN_ERROR(cutensornetStateFinalizeMPS(
      m_cutnHandle, m_quantumState, CUTENSORNET_BOUNDARY_CONDITION_OPEN,
      extentsPtr.data(), /*strides=*/nullptr));
  // Set up the SVD method for truncation.
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      m_cutnHandle, m_quantumState, CUTENSORNET_STATE_MPS_SVD_CONFIG_ALGO,
      &algo, sizeof(algo)));
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      m_cutnHandle, m_quantumState, CUTENSORNET_STATE_MPS_SVD_CONFIG_ABS_CUTOFF,
      &absCutoff, sizeof(absCutoff)));
  HANDLE_CUTN_ERROR(cutensornetStateConfigure(
      m_cutnHandle, m_quantumState, CUTENSORNET_STATE_MPS_SVD_CONFIG_REL_CUTOFF,
      &relCutoff, sizeof(relCutoff)));

  // Prepare the MPS computation and attach workspace
  cutensornetWorkspaceDescriptor_t workDesc;
  ScratchDeviceMem scratchPad;

  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  HANDLE_CUTN_ERROR(cutensornetStatePrepare(
      m_cutnHandle, m_quantumState, scratchPad.scratchSize, workDesc, 0));
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

  // Check whether we need host memory workspace
  std::int64_t hostWorkspaceSize{0};
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

  // Execute MPS computation
  HANDLE_CUTN_ERROR(cutensornetStateCompute(
      m_cutnHandle, m_quantumState, workDesc, extentsPtr.data(),
      /*strides=*/nullptr, d_mpsTensors.data(), 0));
  if (hostWork) {
    free(hostWork);
  }
  return d_mpsTensors;
}

std::complex<double> TensorNetState::computeExpVal(
    cutensornetNetworkOperator_t tensorNetworkOperator) {
  cutensornetStateExpectation_t tensorNetworkExpectation;
  LOG_API_TIME();

  // Step 1: create
  HANDLE_CUTN_ERROR(cutensornetCreateExpectation(m_cutnHandle, m_quantumState,
                                                 tensorNetworkOperator,
                                                 &tensorNetworkExpectation));
  // Step 2: configure
  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  HANDLE_CUTN_ERROR(cutensornetExpectationConfigure(
      m_cutnHandle, tensorNetworkExpectation,
      CUTENSORNET_EXPECTATION_OPT_NUM_HYPER_SAMPLES, &numHyperSamples,
      sizeof(numHyperSamples)));

  // Step 3: Prepare
  cutensornetWorkspaceDescriptor_t workDesc;
  ScratchDeviceMem scratchPad;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  {
    ScopedTraceWithContext(
        "TensorNetState::computeExpVal::cutensornetExpectationPrepare");
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
    ScopedTraceWithContext(
        "TensorNetState::computeExpVal::cutensornetExpectationCompute");
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

TensorNetState::~TensorNetState() {
  // Destroy the quantum circuit state
  HANDLE_CUTN_ERROR(cutensornetDestroyState(m_quantumState));
}

} // namespace nvqir
