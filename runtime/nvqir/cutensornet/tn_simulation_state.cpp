/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "tn_simulation_state.h"
#include <cuComplex.h>

namespace nvqir {
int deviceFromPointer(void *ptr) {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}
std::size_t TensorNetSimulationState::getNumQubits() const {
  return m_state->getNumQubits();
}

TensorNetSimulationState::TensorNetSimulationState(
    std::unique_ptr<TensorNetState> inState, cutensornetHandle_t cutnHandle)
    : m_state(std::move(inState)), m_cutnHandle(cutnHandle) {}

TensorNetSimulationState::~TensorNetSimulationState() {}

std::complex<double>
TensorNetSimulationState::overlap(const cudaq::SimulationState &other) {
  const cudaq::SimulationState *const otherStatePtr = &other;
  const TensorNetSimulationState *const tnOther =
      dynamic_cast<const TensorNetSimulationState *>(otherStatePtr);

  if (!tnOther)
    throw std::runtime_error("[tensornet state] Computing overlap with other "
                             "types of state is not supported.");
  auto tensorOps = tnOther->m_state->m_tensorOps;
  // Compute <bra|ket> by conjugating the entire |bra> tensor network.
  // Reverse them
  std::reverse(tensorOps.begin(), tensorOps.end());
  for (auto &op : tensorOps) {
    op.isAdjoint = !op.isAdjoint;
    if (!op.isUnitary) {
      // For non-unitary ops, i.e., projectors, we need to do a transpose to
      // reverse the leg connection.
      const auto dim = (1 << op.qubitIds.size());
      // FIXME: perform this in device memory.
      Eigen::MatrixXcd mat(dim, dim);
      HANDLE_CUDA_ERROR(cudaMemcpy(mat.data(), op.deviceData,
                                   mat.size() * sizeof(std::complex<double>),
                                   cudaMemcpyDeviceToHost));
      mat.transposeInPlace();
      HANDLE_CUDA_ERROR(cudaMemcpy(op.deviceData, mat.data(),
                                   mat.size() * sizeof(std::complex<double>),
                                   cudaMemcpyHostToDevice));
    }
  }
  // Append them to ket
  // Note: we clone a new ket tensor network to keep this ket as-is.
  const auto nbQubits = std::max(getNumQubits(), other.getNumQubits());
  const std::vector<int64_t> qubitDims(nbQubits, 2);
  cutensornetState_t tempQuantumState;
  auto &cutnHandle = m_state->m_cutnHandle;
  HANDLE_CUTN_ERROR(cutensornetCreateState(
      cutnHandle, CUTENSORNET_STATE_PURITY_PURE, nbQubits, qubitDims.data(),
      CUDA_C_64F, &tempQuantumState));

  int64_t tensorId = 0;
  // Append ket-side gate tensors + conjugated (reverse + adjoint) bra-side
  // tensors
  auto allTensorOps = m_state->m_tensorOps;
  allTensorOps.insert(allTensorOps.end(), tensorOps.begin(), tensorOps.end());

  for (auto &op : allTensorOps)
    HANDLE_CUTN_ERROR(cutensornetStateApplyTensor(
        cutnHandle, tempQuantumState, op.qubitIds.size(), op.qubitIds.data(),
        op.deviceData, nullptr, /*immutable*/ 1,
        /*adjoint*/ static_cast<int32_t>(op.isAdjoint),
        /*unitary*/ static_cast<int32_t>(op.isUnitary), &tensorId));

  // Cap off with all zero projection (initial state of bra)
  std::vector<int32_t> projectedModes(nbQubits);
  std::iota(projectedModes.begin(), projectedModes.end(), 0);
  std::vector<int64_t> projectedModeValues(nbQubits, 0);
  void *d_overlap;
  HANDLE_CUDA_ERROR(cudaMalloc(&d_overlap, sizeof(std::complex<double>)));
  // Create the quantum state amplitudes accessor
  cutensornetStateAccessor_t accessor;
  HANDLE_CUTN_ERROR(cutensornetCreateAccessor(
      cutnHandle, tempQuantumState, projectedModes.size(),
      projectedModes.data(), nullptr, &accessor));

  const int32_t numHyperSamples =
      8; // desired number of hyper samples used in the tensor network
         // contraction path finder
  HANDLE_CUTN_ERROR(cutensornetAccessorConfigure(
      cutnHandle, accessor, CUTENSORNET_ACCESSOR_OPT_NUM_HYPER_SAMPLES,
      &numHyperSamples, sizeof(numHyperSamples)));
  // Prepare the quantum state amplitudes accessor
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  HANDLE_CUTN_ERROR(cutensornetAccessorPrepare(
      cutnHandle, accessor, m_scratchPad.scratchSize, workDesc, 0));

  // Attach the workspace buffer
  int64_t worksize = 0;
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH, &worksize));
  if (worksize <= static_cast<int64_t>(m_scratchPad.scratchSize)) {
    HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
        cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
        CUTENSORNET_WORKSPACE_SCRATCH, m_scratchPad.d_scratch, worksize));
  } else {
    throw std::runtime_error("ERROR: Insufficient workspace size on Device!");
  }

  // Compute the quantum state amplitudes
  std::complex<double> stateNorm{0.0, 0.0};
  // Result overlap (host data)
  std::complex<double> h_overlap{0.0, 0.0};
  HANDLE_CUTN_ERROR(cutensornetAccessorCompute(
      cutnHandle, accessor, projectedModeValues.data(), workDesc, d_overlap,
      static_cast<void *>(&stateNorm), 0));
  HANDLE_CUDA_ERROR(cudaMemcpy(&h_overlap, d_overlap,
                               sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));
  // Free resources
  HANDLE_CUDA_ERROR(cudaFree(d_overlap));
  HANDLE_CUTN_ERROR(cutensornetDestroyWorkspaceDescriptor(workDesc));
  HANDLE_CUTN_ERROR(cutensornetDestroyAccessor(accessor));
  HANDLE_CUTN_ERROR(cutensornetDestroyState(tempQuantumState));

  return h_overlap;
}

std::complex<double>
TensorNetSimulationState::getAmplitude(const std::vector<int> &basisState) {
  std::vector<int32_t> projectedModes(m_state->getNumQubits());
  std::iota(projectedModes.begin(), projectedModes.end(), 0);
  std::vector<int64_t> projectedModeValues;
  projectedModeValues.assign(basisState.begin(), basisState.end());
  auto subStateVec =
      m_state->getStateVector(projectedModes, projectedModeValues);
  assert(subStateVec.size() == 1);
  return subStateVec[0];
}

cudaq::SimulationState::Tensor
TensorNetSimulationState::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx >= getNumTensors())
    throw std::out_of_range("Invalid tensor index");
  cudaq::SimulationState::Tensor tensor;
  auto &opTensor = m_state->m_tensorOps[tensorIdx];
  tensor.data = opTensor.deviceData;
  std::vector<std::size_t> extents(2 * opTensor.qubitIds.size(), 2);
  tensor.extents = extents;
  tensor.fp_precision = getPrecision();
  return tensor;
}

std::vector<cudaq::SimulationState::Tensor>
TensorNetSimulationState::getTensors() const {
  std::vector<cudaq::SimulationState::Tensor> tensors;
  tensors.reserve(m_state->m_tensorOps.size());

  for (auto &op : m_state->m_tensorOps) {
    cudaq::SimulationState::Tensor tensor;
    tensor.data = op.deviceData;
    std::vector<std::size_t> extents(2 * op.qubitIds.size(), 2);
    tensor.extents = extents;
    tensor.fp_precision = getPrecision();
    tensors.emplace_back(std::move(tensor));
  }
  return tensors;
}

std::size_t TensorNetSimulationState::getNumTensors() const {
  return m_state->m_tensorOps.size();
}

void TensorNetSimulationState::destroyState() {
  cudaq::info("mps-state destroying state vector handle.");
  m_state.reset();
}

void TensorNetSimulationState::dump(std::ostream &os) const {
  const auto tmp = m_state->getStateVector();
  for (auto &t : tmp)
    os << t << "\n";
}

std::unique_ptr<nvqir::TensorNetState>
TensorNetSimulationState::reconstructBackendState() {
  return m_state->clone();
}

std::unique_ptr<cudaq::SimulationState>
TensorNetSimulationState::toSimulationState() {
  return std::make_unique<TensorNetSimulationState>(
      m_state->clone(), m_state->getInternalContext());
}
} // namespace nvqir