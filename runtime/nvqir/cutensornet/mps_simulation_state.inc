/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EigenDense.h"
#include "cudaq/utils/cudaq_utils.h"
#include <bitset>
#include <charconv>
#include <cuComplex.h>

namespace nvqir {
int deviceFromPointer(void *ptr) {
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
  return attributes.device;
}

template <typename ScalarType>
std::size_t MPSSimulationState<ScalarType>::getNumQubits() const {
  return state->getNumQubits();
}

template <typename ScalarType>
MPSSimulationState<ScalarType>::MPSSimulationState(
    std::unique_ptr<TensorNetState<ScalarType>> inState,
    const std::vector<MPSTensor> &mpsTensors, ScratchDeviceMem &inScratchPad,
    cutensornetHandle_t cutnHandle, std::mt19937 &randomEngine)
    : m_cutnHandle(cutnHandle), state(std::move(inState)),
      m_mpsTensors(mpsTensors), scratchPad(inScratchPad),
      m_randomEngine(randomEngine) {}

template <typename ScalarType>
MPSSimulationState<ScalarType>::~MPSSimulationState() {
  deallocate();
}

template <typename ScalarType>
std::complex<double> MPSSimulationState<ScalarType>::computeOverlap(
    const std::vector<MPSTensor> &m_mpsTensors,
    const std::vector<MPSTensor> &mpsOtherTensors) {
  LOG_API_TIME();
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
  // Set up tensor network
  HANDLE_CUTN_ERROR(cutensornetCreateNetwork(cutnHandle, &m_tnDescr));

  int64_t tensorIDs[numTensors]; // for input tensors

  // attach the input tensors to the network
  for (int32_t t = 0; t < numTensors; ++t) {
    HANDLE_CUTN_ERROR(cutensornetNetworkAppendTensor(
        cutnHandle, m_tnDescr, tensModes[t].size(), tensExtents[t].data(),
        tensModes[t].data(), &tensAttr[t], dataType, &tensorIDs[t]));
  }

  // set the output tensor
  HANDLE_CUTN_ERROR(cutensornetNetworkSetOutputTensor(cutnHandle, m_tnDescr, 0,
                                                      NULL, dataType));

  // set the network compute type
  HANDLE_CUTN_ERROR(cutensornetNetworkSetAttribute(
      cutnHandle, m_tnDescr, CUTENSORNET_NETWORK_COMPUTE_TYPE, &computeType,
      sizeof(computeType)));

  cutensornetContractionOptimizerConfig_t m_tnConfig;

  // Determine the tensor network contraction path and create the contraction
  // plan
  {
    ScopedTraceWithContext("cutensornetCreateContractionOptimizerConfig");
    HANDLE_CUTN_ERROR(
        cutensornetCreateContractionOptimizerConfig(cutnHandle, &m_tnConfig));
  }
  cutensornetContractionOptimizerInfo_t m_tnPath;
  {
    ScopedTraceWithContext("cutensornetCreateContractionOptimizerInfo");
    HANDLE_CUTN_ERROR(cutensornetCreateContractionOptimizerInfo(
        cutnHandle, m_tnDescr, &m_tnPath));
  }
  assert(scratchPad.scratchSize > 0);
  {
    ScopedTraceWithContext("cutensornetContractionOptimize");
    HANDLE_CUTN_ERROR(cutensornetContractionOptimize(
        cutnHandle, m_tnDescr, m_tnConfig, scratchPad.scratchSize, m_tnPath));
  }
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
         scratchPad.scratchSize);
  HANDLE_CUTN_ERROR(cutensornetWorkspaceSetMemory(
      cutnHandle, workDesc, CUTENSORNET_MEMSPACE_DEVICE,
      CUTENSORNET_WORKSPACE_SCRATCH, scratchPad.d_scratch,
      requiredWorkspaceSize));
  {
    ScopedTraceWithContext("cutensornetNetworkPrepareContraction");
    HANDLE_CUTN_ERROR(
        cutensornetNetworkPrepareContraction(cutnHandle, m_tnDescr, workDesc));
  }
  // Compute the unnormalized overlap
  std::vector<const void *> rawDataIn(numTensors);
  for (int i = 0; i < mpsNumTensors; ++i) {
    rawDataIn[i] = m_mpsTensors[i].deviceData;
    rawDataIn[mpsNumTensors + i] = mpsOtherTensors[i].deviceData;
  }
  void *m_dOverlap{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dOverlap, overlapSize));

  // Set tensor's data buffers and strides
  for (int32_t t = 0; t < numTensors; ++t) {
    HANDLE_CUTN_ERROR(cutensornetNetworkSetInputTensorMemory(
        cutnHandle, m_tnDescr, tensorIDs[t], rawDataIn[t], NULL));
  }
  HANDLE_CUTN_ERROR(cutensornetNetworkSetOutputTensorMemory(
      cutnHandle, m_tnDescr, m_dOverlap, NULL));

  {
    ScopedTraceWithContext("cutensornetNetworkContract");
    HANDLE_CUTN_ERROR(cutensornetNetworkContract(cutnHandle, m_tnDescr, 0,
                                                 workDesc, NULL, 0x0));
  }
  // Get the overlap value back to Host
  std::complex<double> overlap = 0.0;
  if (prec == precision::fp32) {
    cuFloatComplex overlapValue;
    HANDLE_CUDA_ERROR(cudaMemcpy(&overlapValue, m_dOverlap, overlapSize,
                                 cudaMemcpyDeviceToHost));
    overlap = {static_cast<double>(cuCrealf(overlapValue)),
               static_cast<double>(cuCimagf(overlapValue))};
  } else if (prec == precision::fp64) {
    cuDoubleComplex overlapValue;
    HANDLE_CUDA_ERROR(cudaMemcpy(&overlapValue, m_dOverlap, overlapSize,
                                 cudaMemcpyDeviceToHost));
    overlap = {cuCreal(overlapValue), cuCimag(overlapValue)};
  }

  // Clean up
  HANDLE_CUDA_ERROR(cudaFree(m_dOverlap));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerInfo(m_tnPath));
  HANDLE_CUTN_ERROR(cutensornetDestroyContractionOptimizerConfig(m_tnConfig));
  HANDLE_CUTN_ERROR(cutensornetDestroyNetwork(m_tnDescr));

  return std::abs(overlap);
}

template <typename ScalarType>
std::complex<double>
MPSSimulationState<ScalarType>::overlap(const cudaq::SimulationState &other) {

  if (other.getNumTensors() != getNumTensors())
    throw std::runtime_error("[tensornet-state] overlap error - other state "
                             "dimension is not equal to this state dimension.");

  const auto &mpsOther = dynamic_cast<const MPSSimulationState &>(other);
  const auto &mpsOtherTensors = mpsOther.m_mpsTensors;

  return computeOverlap(m_mpsTensors, mpsOtherTensors);
}

template <typename ScalarType>
std::complex<double> MPSSimulationState<ScalarType>::getAmplitude(
    const std::vector<int> &basisState) {
  if (getNumQubits() != basisState.size())
    throw std::runtime_error(
        cudaq_fmt::format("[tensornet-state] getAmplitude with an invalid number "
                    "of bits in the "
                    "basis state: expected {}, provided {}.",
                    getNumQubits(), basisState.size()));
  if (std::any_of(basisState.begin(), basisState.end(),
                  [](int x) { return x != 0 && x != 1; }))
    throw std::runtime_error(
        "[tensornet-state] getAmplitude with an invalid basis state: only "
        "qubit state (0 or 1) is supported.");

  if (basisState.empty())
    throw std::runtime_error("[tensornet-state] Empty basis state.");

  if (m_mpsTensors.size() <= g_maxQubitsForStateContraction) {
    // If this is the first time, cache the state.
    if (m_contractedStateVec.empty())
      m_contractedStateVec = state->getStateVector();
    assert(m_contractedStateVec.size() == (1ULL << m_mpsTensors.size()));
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
    return m_contractedStateVec[idx];
  }

  if (getNumQubits() > 1) {
    TensorNetState basisTensorNetState(
        basisState, scratchPad, state->getInternalContext(), m_randomEngine);
    // Note: this is a basis state, hence bond dim == 1
    std::vector<MPSTensor> basisStateTensors = basisTensorNetState.factorizeMPS(
        1, std::numeric_limits<double>::min(),
        std::numeric_limits<double>::min(), MPSSettings().svdAlgo,
        MPSSettings().gaugeOption);
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
  std::complex<ScalarType> amplitudes[2];
  HANDLE_CUDA_ERROR(cudaMemcpy(amplitudes, m_mpsTensors[0].deviceData,
                               2 * sizeof(std::complex<ScalarType>),
                               cudaMemcpyDeviceToHost));
  return amplitudes[idx];
}

template <typename ScalarType>
cudaq::SimulationState::Tensor
MPSSimulationState<ScalarType>::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx >= getNumTensors())
    throw std::runtime_error(
        "[tensornet-mps-state] invalid tensor idx requested.");

  std::vector<std::size_t> extents;
  for (auto &e : m_mpsTensors[tensorIdx].extents)
    extents.push_back(e);

  return cudaq::SimulationState::Tensor{m_mpsTensors[tensorIdx].deviceData,
                                        extents, getPrecision()};
}

template <typename ScalarType>
std::vector<cudaq::SimulationState::Tensor>
MPSSimulationState<ScalarType>::getTensors() const {
  std::vector<cudaq::SimulationState::Tensor> tensors;
  for (auto &tensor : m_mpsTensors) {
    std::vector<std::size_t> extents;
    for (auto &e : tensor.extents)
      extents.push_back(e);
    tensors.emplace_back(tensor.deviceData, extents, getPrecision());
  }
  return tensors;
}

template <typename ScalarType>
std::size_t MPSSimulationState<ScalarType>::getNumTensors() const {
  return m_mpsTensors.size();
}

template <typename ScalarType>
void MPSSimulationState<ScalarType>::deallocate() {
  for (auto &tensor : m_mpsTensors)
    HANDLE_CUDA_ERROR(cudaFree(tensor.deviceData));
  m_mpsTensors.clear();
  state.reset();
}

template <typename ScalarType>
void MPSSimulationState<ScalarType>::destroyState() {
  CUDAQ_INFO("mps-state destroying state vector handle.");
  deallocate();
}

template <typename ScalarType>
void MPSSimulationState<ScalarType>::dump(std::ostream &os) const {
  const auto printState = [&os](const auto &stateVec) {
    for (auto &t : stateVec)
      os << t << "\n";
  };

  if (!m_contractedStateVec.empty())
    printState(m_contractedStateVec);
  else
    printState(state->getStateVector());
}

template <typename ScalarType>
Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
reshapeMatrix(const Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                                  Eigen::Dynamic> &A) {
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic> B, C;
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
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
      stacked(B.rows() + C.rows(), C.cols());
  stacked << B, C;
  return stacked;
}

template <typename ScalarType>
Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
reshapeStateVec(
    const Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic> &stateVec) {
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic> A =
      stateVec;
  A.transposeInPlace();
  return reshapeMatrix(A);
}

template <typename ScalarType>
MPSSimulationState<ScalarType>::MpsStateData
MPSSimulationState<ScalarType>::createFromStateVec(
    cutensornetHandle_t cutnHandle, ScratchDeviceMem &inScratchPad,
    std::size_t size, std::complex<ScalarType> *ptr, int bondDim,
    std::mt19937 &randomEngine) {
  const std::size_t numQubits = std::log2(size);
  // Reverse the qubit order to match cutensornet convention
  auto newStateVec = TensorNetState<ScalarType>::reverseQubitOrder(
      std::span<std::complex<ScalarType>>{ptr, size});
  Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic> stateVec =
      Eigen::Map<Eigen::Vector<std::complex<ScalarType>, Eigen::Dynamic>>(
          newStateVec.data(), newStateVec.size());

  if (numQubits == 1) {
    void *d_tensor = nullptr;
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_tensor, 2 * sizeof(std::complex<ScalarType>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_tensor, ptr,
                                 2 * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyHostToDevice));
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = std::vector<int64_t>{2};
    auto state = TensorNetState<ScalarType>::createFromMpsTensors(
        {stateTensor}, inScratchPad, cutnHandle, randomEngine);
    return {std::move(state), std::vector<MPSTensor>{stateTensor}};
  }

  // Recursively factor the state vector from left to right.
  //  - reshape the vector to a (2 * M) matrix (M = dim / 2)
  //  - perform SVD on this matrix yields: (MPS tensor) * Singular Values *
  //  Remaining Matrix.
  //  - Continue to do SVD of (Singular Values * Remaining
  // Matrix) till we reach the last qubit.
  // Note: currently, no truncation is implemented (exact).
  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
      reshapedMat = reshapeStateVec(stateVec);
  std::vector<MPSTensor> mpsTensors;
  std::vector<int64_t> numSingularValues;
  const auto enforceBondDim =
      [bondDim](const Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                                    Eigen::Dynamic> &U,
                const Eigen::Vector<ScalarType, Eigen::Dynamic> &S,
                const Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                                    Eigen::Dynamic> &V)
      -> std::tuple<Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                                  Eigen::Dynamic>,
                    Eigen::Vector<ScalarType, Eigen::Dynamic>,
                    Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                                  Eigen::Dynamic>> {
    assert(U.cols() == S.size());
    assert(V.cols() == S.size());
    if (S.size() <= bondDim)
      return {U, S, V};

    // Truncation
    Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
        newU(U.rows(), bondDim);
    newU = U.leftCols(bondDim);
    Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>
        newV(V.rows(), bondDim);
    newV = V.leftCols(bondDim);
    Eigen::Vector<ScalarType, Eigen::Dynamic> newS(bondDim);
    newS = S.head(bondDim);
    return std::make_tuple(newU, newS, newV);
  };
  for (std::size_t i = 0; i < numQubits - 1; ++i) {
    Eigen::BDCSVD<
        Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic, Eigen::Dynamic>,
        Eigen::ComputeThinU | Eigen::ComputeThinV>
        svd(reshapedMat);
    const Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                        Eigen::Dynamic>
        U_orig = svd.matrixU();
    const Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                        Eigen::Dynamic>
        V_orig = svd.matrixV();
    const Eigen::Vector<ScalarType, Eigen::Dynamic> S_orig =
        svd.singularValues();
    const auto [U, S, V] = enforceBondDim(U_orig, S_orig, V_orig);
    numSingularValues.emplace_back(S.size());
    reshapedMat =
        (i != (numQubits - 2))
            ? reshapeMatrix(
                  Eigen::Matrix<std::complex<ScalarType>, Eigen::Dynamic,
                                Eigen::Dynamic>(S.asDiagonal() * V.adjoint()))
            : (S.asDiagonal() * V.adjoint());
    void *d_tensor = nullptr;
    HANDLE_CUDA_ERROR(
        cudaMalloc(&d_tensor, U.size() * sizeof(std::complex<ScalarType>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_tensor, U.data(),
                                 U.size() * sizeof(std::complex<ScalarType>),
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
  HANDLE_CUDA_ERROR(cudaMalloc(
      &d_tensor, reshapedMat.size() * sizeof(std::complex<ScalarType>)));
  HANDLE_CUDA_ERROR(
      cudaMemcpy(d_tensor, reshapedMat.data(),
                 reshapedMat.size() * sizeof(std::complex<ScalarType>),
                 cudaMemcpyHostToDevice));
  MPSTensor stateTensor;
  stateTensor.deviceData = d_tensor;
  stateTensor.extents = std::vector<int64_t>{numSingularValues.back(), 2};
  mpsTensors.emplace_back(stateTensor);
  assert(mpsTensors.size() == numQubits);
  auto state = TensorNetState<ScalarType>::createFromMpsTensors(
      mpsTensors, inScratchPad, cutnHandle, randomEngine);
  return {std::move(state), mpsTensors};
}

template <typename ScalarType>
std::unique_ptr<cudaq::SimulationState>
MPSSimulationState<ScalarType>::createFromSizeAndPtr(std::size_t size,
                                                     void *ptr,
                                                     std::size_t dataType) {
  if (dataType == cudaq::detail::variant_index<cudaq::state_data,
                                               cudaq::TensorStateData>()) {
    std::vector<MPSTensor> mpsTensors;
    auto *casted = reinterpret_cast<cudaq::TensorStateData::value_type *>(ptr);
    for (std::size_t i = 0; i < size; ++i) {
      auto [dataPtr, extents] = casted[i];
      const auto numElements =
          std::reduce(extents.begin(), extents.end(), 1, std::multiplies());
      void *d_tensor = nullptr;
      HANDLE_CUDA_ERROR(cudaMalloc(
          &d_tensor, numElements * sizeof(std::complex<ScalarType>)));
      // Note: cudaMemcpyDefault to handle both device/host data
      HANDLE_CUDA_ERROR(cudaMemcpy(
          d_tensor, dataPtr, numElements * sizeof(std::complex<ScalarType>),
          cudaMemcpyDefault));
      std::vector<int64_t> mpsExtents(extents.begin(), extents.end());
      MPSTensor stateTensor{d_tensor, mpsExtents};
      mpsTensors.emplace_back(stateTensor);
    }
    auto state = TensorNetState<ScalarType>::createFromMpsTensors(
        mpsTensors, scratchPad, m_cutnHandle, m_randomEngine);
    return std::make_unique<MPSSimulationState>(
        std::move(state), mpsTensors, scratchPad, m_cutnHandle, m_randomEngine);
  }
  auto [state, mpsTensors] =
      createFromStateVec(m_cutnHandle, scratchPad, size,
                         reinterpret_cast<std::complex<ScalarType> *>(ptr),
                         MPSSettings().maxBond, m_randomEngine);
  return std::make_unique<MPSSimulationState>(
      std::move(state), mpsTensors, scratchPad, m_cutnHandle, m_randomEngine);
}

MPSSettings::MPSSettings() {
  if (auto *maxBondEnvVar = std::getenv("CUDAQ_MPS_MAX_BOND")) {
    const std::string maxBondStr(maxBondEnvVar);
    const char *nptr = maxBondStr.data();
    char *endptr = nullptr;
    errno = 0; // reset errno to 0 before call
    maxBond = strtol(nptr, &endptr, 10);

    if (nptr == endptr || errno != 0 || maxBond < 1)
      throw std::runtime_error("Invalid CUDAQ_MPS_MAX_BOND setting. Expected "
                               "a positive number. Got: " +
                               maxBondStr);

    CUDAQ_INFO("Setting MPS max bond dimension to {}.", maxBond);
  }
  // Cutoff values
  if (auto *absCutoffEnvVar = std::getenv("CUDAQ_MPS_ABS_CUTOFF")) {
    const std::string absCutoffStr(absCutoffEnvVar);
    const char *nptr = absCutoffStr.data();
    char *endptr = nullptr;
    errno = 0; // reset errno to 0 before call
    absCutoff = strtod(nptr, &endptr);

    if (nptr == endptr || errno != 0 || absCutoff <= 0.0 || absCutoff >= 1.0)
      throw std::runtime_error("Invalid CUDAQ_MPS_ABS_CUTOFF setting. Expected "
                               "a number in range (0.0, 1.0). Got: " +
                               absCutoffStr);

    CUDAQ_INFO("Setting MPS absolute cutoff to {}.", absCutoff);
  }
  if (auto *relCutoffEnvVar = std::getenv("CUDAQ_MPS_RELATIVE_CUTOFF")) {
    const std::string relCutoffStr(relCutoffEnvVar);
    const char *nptr = relCutoffStr.data();
    char *endptr = nullptr;
    errno = 0; // reset errno to 0 before call
    relCutoff = strtod(nptr, &endptr);

    if (nptr == endptr || errno != 0 || relCutoff <= 0.0 || relCutoff >= 1.0)
      throw std::runtime_error(
          "Invalid CUDAQ_MPS_RELATIVE_CUTOFF setting. Expected "
          "a number in range (0.0, 1.0). Got: " +
          relCutoffStr);

    CUDAQ_INFO("Setting MPS relative cutoff to {}.", relCutoff);
  }
  using namespace std::literals::string_view_literals;
  using SvdPair = std::pair<std::string_view, cutensornetTensorSVDAlgo_t>;
  constexpr std::array<SvdPair, 4> g_stringToAlgoEnum = {
      SvdPair{"GESVD"sv, CUTENSORNET_TENSOR_SVD_ALGO_GESVD},
      SvdPair{"GESVDJ"sv, CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ},
      SvdPair{"GESVDP"sv, CUTENSORNET_TENSOR_SVD_ALGO_GESVDP},
      SvdPair{"GESVDR"sv, CUTENSORNET_TENSOR_SVD_ALGO_GESVDR}};
  if (auto *svdAlgoEnvVar = std::getenv("CUDAQ_MPS_SVD_ALGO")) {
    std::string svdAlgoStr(svdAlgoEnvVar);
    std::transform(svdAlgoStr.begin(), svdAlgoStr.end(), svdAlgoStr.begin(),
                   ::toupper);
    const auto iter = std::lower_bound(
        g_stringToAlgoEnum.begin(), g_stringToAlgoEnum.end(), svdAlgoStr,
        [](const SvdPair &pair, const std::string &key) {
          return pair.first < key;
        });
    if (iter == g_stringToAlgoEnum.end() || iter->first != svdAlgoStr) {
      std::stringstream errorMsg;
      errorMsg << "Unknown CUDAQ_MPS_SVD_ALGO value ('" << svdAlgoEnvVar
               << "').\nValid values are:\n";
      for (const auto &[configStr, _] : g_stringToAlgoEnum)
        errorMsg << "  - " << configStr << "\n";
      throw std::runtime_error(errorMsg.str());
    }
    svdAlgo = iter->second;
    CUDAQ_INFO("Setting MPS SVD algorithm to {} ({}).",
               cudaq_fmt::underlying(svdAlgo), iter->first);
  }

  using GaugePair =
      std::pair<std::string_view, cutensornetStateMPSGaugeOption_t>;
  constexpr std::array<GaugePair, 2> g_stringToGaugeEnum = {
      GaugePair{"FREE"sv, CUTENSORNET_STATE_MPS_GAUGE_FREE},
      GaugePair{"SIMPLE"sv, CUTENSORNET_STATE_MPS_GAUGE_SIMPLE}};
  if (auto *gaugeEnvVar = std::getenv("CUDAQ_MPS_GAUGE")) {
    std::string gaugeOptionStr(gaugeEnvVar);
    std::transform(gaugeOptionStr.begin(), gaugeOptionStr.end(),
                   gaugeOptionStr.begin(), ::toupper);
    const auto iter = std::lower_bound(
        g_stringToGaugeEnum.begin(), g_stringToGaugeEnum.end(), gaugeOptionStr,
        [](const GaugePair &pair, const std::string &key) {
          return pair.first < key;
        });
    if (iter == g_stringToGaugeEnum.end() || iter->first != gaugeOptionStr) {
      std::stringstream errorMsg;
      errorMsg << "Unknown CUDAQ_MPS_GAUGE value ('" << gaugeEnvVar
               << "').\nValid values are:\n";
      for (const auto &[configStr, _] : g_stringToGaugeEnum)
        errorMsg << "  - " << configStr << "\n";
      throw std::runtime_error(errorMsg.str());
    }
    gaugeOption = iter->second;
    CUDAQ_INFO("Setting MPS GAUGE option to {} ({}).", iter->first,
               cudaq_fmt::underlying(iter->second));
  }
}

template <typename ScalarType>
template <typename T>
void MPSSimulationState<ScalarType>::toHostImpl(
    std::complex<T> *clientAllocatedData, std::size_t numElements) const {
  auto stateVec = state->getStateVector();
  if (stateVec.size() != numElements)
    throw std::runtime_error(
        cudaq_fmt::format("[MPSSimulationState] Dimension mismatch: expecting {} "
                    "elements but providing an array of size {}.",
                    stateVec.size(), numElements));
  if (std::is_same_v<ScalarType, T>) {
    std::memcpy(clientAllocatedData, stateVec.data(),
                numElements * sizeof(std::complex<ScalarType>));
  } else {
    for (std::size_t i = 0; i < numElements; ++i)
      clientAllocatedData[i] = stateVec[i];
  }
}

template <typename ScalarType>
void MPSSimulationState<ScalarType>::toHost(
    std::complex<double> *clientAllocatedData, std::size_t numElements) const {
  toHostImpl(clientAllocatedData, numElements);
}
template <typename ScalarType>
void MPSSimulationState<ScalarType>::toHost(
    std::complex<float> *clientAllocatedData, std::size_t numElements) const {
  toHostImpl(clientAllocatedData, numElements);
}
} // namespace nvqir
