/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mps_simulation_state.h"
#include "common/EigenDense.h"
#include <cuComplex.h>
#include <iostream>

namespace nvqir {
int deviceFromPointer(void *ptr) {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}
std::size_t MPSSimulationState::getNumQubits() const {
  return state->getNumQubits() - m_auxTensorIds.size();
}

MPSSimulationState::MPSSimulationState(
    std::unique_ptr<TensorNetState> inState,
    const std::vector<MPSTensor> &mpsTensors,
    const std::vector<std::size_t> &auxTensorIds,
    cutensornetHandle_t cutnHandle)
    : m_cutnHandle(cutnHandle), state(std::move(inState)),
      m_mpsTensors(mpsTensors), m_auxTensorIds(auxTensorIds) {}

MPSSimulationState::~MPSSimulationState() { deallocate(); }

std::complex<double> MPSSimulationState::computeOverlap(
    const std::vector<MPSTensor> &m_mpsTensors,
    const std::vector<MPSTensor> &mpsOtherTensors) {

  auto dataDevice = deviceFromPointer(m_mpsTensors[0].deviceData);
  auto otherDevice = deviceFromPointer(mpsOtherTensors[0].deviceData);

  if (otherDevice != dataDevice)
    throw std::runtime_error("MPS overlap requested but the two states are on "
                             "different GPU devices.");

  int currentDevice;
  cudaGetDevice(&currentDevice);
  if (currentDevice != dataDevice)
    cudaSetDevice(dataDevice);

  const int32_t mpsNumTensors = m_mpsTensors.size();
  assert(mpsNumTensors > 0);
  auto cutnHandle = state->getInternalContext();

  // Create a tensor network descriptor for the overlap
  const int32_t numTensors =
      mpsNumTensors *
      2; // the overlap tensor network contains two MPS tensor networks
  std::vector<int32_t> numModes(numTensors);
  std::vector<std::vector<int64_t>> tensExtents(numTensors);
  std::vector<cutensornetTensorQualifiers_t> tensAttr(numTensors);
  for (int i = 0; i < mpsNumTensors; ++i) {
    numModes[i] = m_mpsTensors[i].extents.size();
    numModes[mpsNumTensors + i] = mpsOtherTensors[i].extents.size();
    tensExtents[i] = m_mpsTensors[i].extents;
    tensExtents[mpsNumTensors + i] = mpsOtherTensors[i].extents;
    tensAttr[i] = cutensornetTensorQualifiers_t{0, 0, 0};
    tensAttr[mpsNumTensors + i] = cutensornetTensorQualifiers_t{1, 0, 0};
  }
  std::vector<std::vector<int32_t>> tensModes(numTensors);
  int32_t umode = 0;
  for (int i = 0; i < mpsNumTensors; ++i) {
    if (i == 0) {
      if (mpsNumTensors > 1) {
        tensModes[i] = std::initializer_list<int32_t>{umode, umode + 1};
        umode += 2;
      } else {
        tensModes[i] = std::initializer_list<int32_t>{umode};
        umode += 1;
      }
    } else if (i == (mpsNumTensors - 1)) {
      tensModes[i] = std::initializer_list<int32_t>{umode - 1, umode};
      umode += 1;
    } else {
      tensModes[i] =
          std::initializer_list<int32_t>{umode - 1, umode, umode + 1};
      umode += 2;
    }
  }
  int32_t lmode = umode;
  umode = 0;
  for (int i = 0; i < mpsNumTensors; ++i) {
    if (i == 0) {
      if (mpsNumTensors > 1) {
        tensModes[mpsNumTensors + i] =
            std::initializer_list<int32_t>{umode, lmode};
        umode += 2;
        lmode += 1;
      } else {
        tensModes[mpsNumTensors + i] = std::initializer_list<int32_t>{umode};
        umode += 1;
      }
    } else if (i == (mpsNumTensors - 1)) {
      tensModes[mpsNumTensors + i] =
          std::initializer_list<int32_t>{lmode - 1, umode};
      umode += 1;
    } else {
      tensModes[mpsNumTensors + i] =
          std::initializer_list<int32_t>{lmode - 1, umode, lmode};
      umode += 2;
      lmode += 1;
    }
  }
  std::size_t overlapSize = 0;
  cutensornetComputeType_t computeType;
  cudaDataType_t dataType;
  const auto prec = getPrecision();
  if (prec == precision::fp32) {
    overlapSize = sizeof(cuFloatComplex);
    dataType = CUDA_C_32F;
    computeType = CUTENSORNET_COMPUTE_32F;
  } else if (prec == precision::fp64) {
    overlapSize = sizeof(cuDoubleComplex);
    dataType = CUDA_C_64F;
    computeType = CUTENSORNET_COMPUTE_64F;
  } else {
    throw std::runtime_error("[tensornet-state] unknown precision.");
  }
  std::vector<const int64_t *> extentsIn(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    extentsIn[i] = tensExtents[i].data();
  }
  std::vector<const int32_t *> modesIn(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    modesIn[i] = tensModes[i].data();
  }

  cutensornetNetworkDescriptor_t m_tnDescr;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkDescriptor(
      cutnHandle, numTensors, numModes.data(), extentsIn.data(), NULL,
      modesIn.data(), tensAttr.data(), 0, NULL, NULL, NULL, dataType,
      computeType, &m_tnDescr));

  cutensornetContractionOptimizerConfig_t m_tnConfig;
  // Determine the tensor network contraction path and create the contraction
  // plan
  HANDLE_CUTN_ERROR(
      cutensornetCreateContractionOptimizerConfig(cutnHandle, &m_tnConfig));

  cutensornetContractionOptimizerInfo_t m_tnPath;
  HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(
      cutnHandle, m_tnDescr, &m_tnPath));
  assert(m_scratchPad.scratchSize > 0);
  HANDLE_CUTN_ERROR(cutensornetContractionOptimize(
      cutnHandle, m_tnDescr, m_tnConfig, m_scratchPad.scratchSize, m_tnPath));
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(cutnHandle, &workDesc));
  int64_t requiredWorkspaceSize = 0;
  HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeContractionSizes(
      cutnHandle, m_tnDescr, m_tnPath, workDesc));
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
      &requiredWorkspaceSize));
  assert(requiredWorkspaceSize > 0);
  assert(static_cast<std::size_t>(requiredWorkspaceSize) <=
         m_scratchPad.scratchSize);
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
      cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, m_scratchPad.d_scratch,
      requiredWorkspaceSize));
  cutensornetContractionPlan_t m_tnPlan;
  HANDLE_CUTN_ERROR(cutensornetCreateContractionPlan(
      cutnHandle, m_tnDescr, m_tnPath, workDesc, &m_tnPlan));

  // Compute the unnormalized overlap
  std::vector<const void *> rawDataIn(numTensors);
  for (int i = 0; i < mpsNumTensors; ++i) {
    rawDataIn[i] = m_mpsTensors[i].deviceData;
    rawDataIn[mpsNumTensors + i] = mpsOtherTensors[i].deviceData;
  }
  void *m_dOverlap{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dOverlap, overlapSize));
  HANDLE_CUTN_ERROR(cutensornetContractSlices(cutnHandle, m_tnPlan,
                                              rawDataIn.data(), m_dOverlap, 0,
                                              workDesc, NULL, 0x0));
  // Get the overlap value back to Host
  std::complex<double> overlap = 0.0;
  if (prec == precision::fp32) {
    cuFloatComplex overlapValue;
    HANDLE_CUDA_ERROR(cudaMemcpy(&overlapValue, m_dOverlap, overlapSize,
                                 cudaMemcpyDeviceToHost));
    overlap = static_cast<double>(cuCrealf(overlapValue));
  } else if (prec == precision::fp64) {
    cuDoubleComplex overlapValue;
    HANDLE_CUDA_ERROR(cudaMemcpy(&overlapValue, m_dOverlap, overlapSize,
                                 cudaMemcpyDeviceToHost));
    overlap = {cuCreal(overlapValue), cuCimag(overlapValue)};
  }

  // Clean up
  HANDLE_CUDA_ERROR(cudaFree(m_dOverlap));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionPlan(m_tnPlan));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(m_tnPath));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(m_tnConfig));
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkDescriptor(m_tnDescr));

  return overlap;
}

std::complex<double>
MPSSimulationState::overlap(const cudaq::SimulationState &other) {

  if (other.getNumTensors() != getNumTensors())
    throw std::runtime_error("[tensornet-state] overlap error - other state "
                             "dimension is not equal to this state dimension.");

  const auto &mpsOther = dynamic_cast<const MPSSimulationState &>(other);
  const auto &mpsOtherTensors = mpsOther.m_mpsTensors;

  return computeOverlap(m_mpsTensors, mpsOtherTensors);
}

std::complex<double>
MPSSimulationState::getAmplitude(const std::vector<int> &basisState) {
  if (getNumQubits() != basisState.size())
    throw std::runtime_error(
        fmt::format("[tensornet-state] getAmplitude with an invalid number "
                    "of bits in the "
                    "basis state: expected {}, provided {}.",
                    getNumQubits(), basisState.size()));
  if (std::any_of(basisState.begin(), basisState.end(),
                  [](int x) { return x != 0 && x != 1; }))
    throw std::runtime_error(
        "[tensornet-state] getAmplitude with an invalid basis state: only "
        "qubit state (0 or 1) is supported.");
  if (getNumQubits() > 1) {
    auto extendedBasisState = basisState;
    for (std::size_t i = 0; i < m_auxTensorIds.size(); ++i)
      extendedBasisState.emplace_back(0);

    TensorNetState basisTensorNetState(extendedBasisState,
                                       state->getInternalContext());
    // Note: this is a basis state, hence bond dim == 1
    std::vector<MPSTensor> basisStateTensors =
        basisTensorNetState.factorizeMPS(1, std::numeric_limits<double>::min(),
                                         std::numeric_limits<double>::min());
    const auto overlap = computeOverlap(m_mpsTensors, basisStateTensors);
    for (auto &mpsTensor : basisStateTensors) {
      HANDLE_CUDA_ERROR(cudaFree(mpsTensor.deviceData));
    }
    return overlap;
  }
  // Single-qubit
  assert(basisState.size() == 1);
  const auto idx = basisState[0];
  // It is just 2 complex numbers, so load them both rather than compute the
  // exact pointer.
  std::complex<double> amplitudes[2];
  HANDLE_CUDA_ERROR(cudaMemcpy(amplitudes, m_mpsTensors[0].deviceData,
                               2 * sizeof(std::complex<double>),
                               cudaMemcpyDeviceToHost));
  return amplitudes[idx];
}

cudaq::SimulationState::Tensor
MPSSimulationState::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx >= getNumTensors())
    throw std::runtime_error(
        "[tensornet-mps-state] invalid tensor idx requested.");

  std::vector<std::size_t> extents;
  for (auto &e : m_mpsTensors[tensorIdx].extents)
    extents.push_back(e);

  return cudaq::SimulationState::Tensor{m_mpsTensors[tensorIdx].deviceData,
                                        extents, getPrecision()};
}

std::vector<cudaq::SimulationState::Tensor>
MPSSimulationState::getTensors() const {
  std::vector<cudaq::SimulationState::Tensor> tensors;
  for (auto &tensor : m_mpsTensors) {
    std::vector<std::size_t> extents;
    for (auto &e : tensor.extents)
      extents.push_back(e);
    tensors.emplace_back(tensor.deviceData, extents, getPrecision());
  }
  return tensors;
}

std::size_t MPSSimulationState::getNumTensors() const {
  return m_mpsTensors.size();
}

void MPSSimulationState::deallocate() {
  for (auto &tensor : m_mpsTensors)
    HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
  m_mpsTensors.clear();
  state.reset();
}

void MPSSimulationState::destroyState() {
  cudaq::info("mps-state destroying state vector handle.");
  deallocate();
}

void MPSSimulationState::dump(std::ostream &os) const {
  const int32_t numTensors = m_mpsTensors.size();
  std::vector<int32_t> numModes(numTensors);
  std::vector<std::vector<int64_t>> tensExtents(numTensors);
  std::vector<int64_t> outExtents(numTensors, 2);
  std::vector<cutensornetTensorQualifiers_t> tensAttr(numTensors);
  std::vector<int32_t> outModes(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    numModes[i] = m_mpsTensors[i].extents.size();
    tensExtents[i] = m_mpsTensors[i].extents;
    tensAttr[i] = cutensornetTensorQualifiers_t{0, 0, 0};
  }
  std::vector<std::vector<int32_t>> tensModes(numTensors);
  int32_t umode = 0;
  for (int i = 0; i < numTensors; ++i) {
    if (i == 0) {
      if (numTensors > 1) {
        tensModes[i] = std::initializer_list<int32_t>{umode, umode + 1};
        outModes[i] = umode;
        umode += 2;
      } else {
        tensModes[i] = std::initializer_list<int32_t>{umode};
        outModes[i] = umode;
        umode += 1;
      }
    } else if (i == (numTensors - 1)) {
      tensModes[i] = std::initializer_list<int32_t>{umode - 1, umode};
      outModes[i] = umode;
      umode += 1;
    } else {
      tensModes[i] =
          std::initializer_list<int32_t>{umode - 1, umode, umode + 1};
      outModes[i] = umode;
      umode += 2;
    }
  }

  cutensornetComputeType_t computeType = CUTENSORNET_COMPUTE_64F;
  cudaDataType_t dataType = CUDA_C_64F;

  std::vector<const int64_t *> extentsIn(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    extentsIn[i] = tensExtents[i].data();
  }
  std::vector<const int32_t *> modesIn(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    modesIn[i] = tensModes[i].data();
  }

  cutensornetNetworkDescriptor_t tnDescr;
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkDescriptor(
      m_cutnHandle, numTensors, numModes.data(), extentsIn.data(), NULL,
      modesIn.data(), tensAttr.data(), outExtents.size(), outExtents.data(),
      NULL, outModes.data(), dataType, computeType, &tnDescr));

  cutensornetContractionOptimizerConfig_t tnConfig;
  // Determine the tensor network contraction path and create the contraction
  // plan
  HANDLE_CUTN_ERROR(
      cutensornetCreateContractionOptimizerConfig(m_cutnHandle, &tnConfig));

  cutensornetContractionOptimizerInfo_t tnPath;
  HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(
      m_cutnHandle, tnDescr, &tnPath));
  assert(m_scratchPad.scratchSize > 0);
  HANDLE_CUTN_ERROR(cutensornetContractionOptimize(
      m_cutnHandle, tnDescr, tnConfig, m_scratchPad.scratchSize, tnPath));
  cutensornetWorkspaceDescriptor_t workDesc;
  HANDLE_CUTN_ERROR(
      cutensornetCreateWorkspaceDescriptor(m_cutnHandle, &workDesc));
  int64_t requiredWorkspaceSize = 0;
  HANDLE_CUTN_ERROR(cutensornetWorkspaceComputeContractionSizes(
      m_cutnHandle, tnDescr, tnPath, workDesc));
  HANDLE_CUTN_ERROR(cutensornetWorkspaceGetMemorySize(
      m_cutnHandle, workDesc, CUTENSORNET_WORKSIZE_PREF_RECOMMENDED,
      CUTENSORNET_MEMSPACE_DEVICE, CUTENSORNET_WORKSPACE_SCRATCH,
      &requiredWorkspaceSize));
  assert(requiredWorkspaceSize > 0);
  assert(static_cast<std::size_t>(requiredWorkspaceSize) <=
         m_scratchPad.scratchSize);
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
      m_cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, m_scratchPad.d_scratch,
      requiredWorkspaceSize));
  cutensornetContractionPlan_t tnPlan;
  HANDLE_CUTN_ERROR(cutensornetCreateContractionPlan(
      m_cutnHandle, tnDescr, tnPath, workDesc, &tnPlan));

  // Contract the MPS network
  std::vector<const void *> rawDataIn(numTensors);
  for (int i = 0; i < numTensors; ++i) {
    rawDataIn[i] = m_mpsTensors[i].deviceData;
  }
  void *dState{nullptr};
  const auto stateVecSize = (1ULL << numTensors) * sizeof(std::complex<double>);
  HANDLE_CUDA_ERROR(cudaMalloc(&dState, stateVecSize));
  HANDLE_CUTN_ERROR(cutensornetContractSlices(
      m_cutnHandle, tnPlan, rawDataIn.data(), dState, 0, workDesc, NULL, 0x0));
  std::vector<std::complex<double>> tmp(1ULL << numTensors);
  HANDLE_CUDA_ERROR(
      cudaMemcpy(tmp.data(), dState, stateVecSize, cudaMemcpyDeviceToHost));

  // Clean up
  HANDLE_CUDA_ERROR(cudaFree(dState));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionPlan(tnPlan));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(tnPath));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(tnConfig));
  HANDLE_CUTN_ERROR(cutensornetDestroyNetworkDescriptor(tnDescr));

  // Print
  for (auto &t : tmp)
    os << t << "\n";
}

std::unique_ptr<nvqir::TensorNetState>
MPSSimulationState::reconstructBackendState() {
  [[maybe_unused]] std::vector<MPSTensor> tensors;
  return nvqir::TensorNetState::createFromMpsTensors(
      m_mpsTensors, state->getInternalContext(), tensors);
}

std::unique_ptr<cudaq::SimulationState>
MPSSimulationState::toSimulationState() {
  std::vector<MPSTensor> tensors;
  auto cloneState = nvqir::TensorNetState::createFromMpsTensors(
      m_mpsTensors, state->getInternalContext(), tensors);

  return std::make_unique<MPSSimulationState>(std::move(cloneState), tensors,
                                              m_auxTensorIds, m_cutnHandle);
}

static Eigen::MatrixXcd reshapeStateVec(const Eigen::VectorXcd &stateVec) {
  Eigen::MatrixXcd A = stateVec;
  A.transposeInPlace();
  Eigen::MatrixXcd B, C;
  const std::size_t rows = A.rows();
  const std::size_t cols = A.cols();
  B.resize(rows, cols / 2);
  C.resize(rows, cols / 2);
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols / 2; ++j) {
      B(i, j) = A(i, j);
      C(i, j) = A(i, j + cols / 2);
    }
  }
  Eigen::MatrixXcd stacked(B.rows() + C.rows(), C.cols());
  stacked << B, C;
  return stacked;
}

std::unique_ptr<cudaq::SimulationState>
MPSSimulationState::createFromSizeAndPtr(std::size_t size, void *ptr,
                                         std::size_t dataType) {
  Eigen::VectorXcd stateVec = Eigen::Map<Eigen::VectorXcd>(
      reinterpret_cast<std::complex<double> *>(ptr), size);
  const std::size_t numQubits = std::log2(size);
  auto state = std::make_unique<TensorNetState>(numQubits, m_cutnHandle);
  if (numQubits == 1) {
    // Easy case: construct the the tensor
    void *d_tensor = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_tensor, 2 * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_tensor, ptr,
                                 2 * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = std::vector<int64_t>{2};

    return std::make_unique<MPSSimulationState>(
        std::move(state), std::vector<MPSTensor>{stateTensor},
        std::vector<std::size_t>{}, m_cutnHandle);
  }

  // Recursively factor the state vector from left to right.
  //  - reshape the vector to a (2 * M) matrix (M = dim / 2)
  //  - perform SVD on this matrix yields: (MPS tensor) * Singular Values *
  //  Remaining Matrix.
  //  - Continue to do SVD of (Singular Values * Remaining
  // Matrix) till we reach the last qubit.
  // Note: currently, no truncation is implemented (exact).
  Eigen::MatrixXcd reshapedMat = reshapeStateVec(stateVec);
  std::vector<MPSTensor> mpsTensors;
  std::vector<int64_t> numSingularValues;
  for (std::size_t i = 0; i < numQubits - 1; ++i) {
    Eigen::BDCSVD<Eigen::MatrixXcd, Eigen::ComputeThinU | Eigen::ComputeThinV>
        svd(reshapedMat);
    const Eigen::MatrixXcd U = svd.matrixU();
    const Eigen::MatrixXcd V = svd.matrixV();
    const Eigen::VectorXd S = svd.singularValues();
    numSingularValues.emplace_back(S.size());
    reshapedMat = S.asDiagonal() * V;
    void *d_tensor = nullptr;
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_tensor, U.size() * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_tensor, U.data(),
                                 U.size() * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    // Note: this loop doesn't cover the last MPS tensor
    stateTensor.extents = (i == 0)
                              ? std::vector<int64_t>{2, numSingularValues[i]}
                              : std::vector<int64_t>{numSingularValues[i - 1],
                                                     2, numSingularValues[i]};
    mpsTensors.emplace_back(stateTensor);
  }

  // Last tensor (right most)
  void *d_tensor = nullptr;
  HANDLE_CUDA_ERROR(
      cudaMalloc(&d_tensor, reshapedMat.size() * sizeof(std::complex<double>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_tensor, reshapedMat.data(),
                 reshapedMat.size() * sizeof(std::complex<double>),
                 cudaMemcpyHostToDevice));
  MPSTensor stateTensor;
  stateTensor.deviceData = d_tensor;
  stateTensor.extents = std::vector<int64_t>{numSingularValues.back(), 2};
  mpsTensors.emplace_back(stateTensor);
  assert(mpsTensors.size() == numQubits);
  return std::make_unique<MPSSimulationState>(
      std::move(state), mpsTensors, std::vector<std::size_t>{}, m_cutnHandle);
}
} // namespace nvqir