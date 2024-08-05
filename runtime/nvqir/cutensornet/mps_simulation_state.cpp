/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "mps_simulation_state.h"
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
std::size_t MPSSimulationState::getNumQubits() const {
  return state->getNumQubits();
}

MPSSimulationState::MPSSimulationState(std::unique_ptr<TensorNetState> inState,
                                       const std::vector<MPSTensor> &mpsTensors,
                                       ScratchDeviceMem &inScratchPad,
                                       cutensornetHandle_t cutnHandle)
    : m_cutnHandle(cutnHandle), state(std::move(inState)),
      m_mpsTensors(mpsTensors), scratchPad(inScratchPad) {}

MPSSimulationState::~MPSSimulationState() { deallocate(); }

std::complex<double> MPSSimulationState::computeOverlap(
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
  HANDLE_CUTN_ERROR(cutensornetCreateNetworkDescriptor(
      cutnHandle, numTensors, numModes.data(), extentsIn.data(), NULL,
      modesIn.data(), tensAttr.data(), 0, NULL, NULL, NULL, dataType,
      computeType, &m_tnDescr));

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
  cutensornetContractionPlan_t m_tnPlan;
  {
    ScopedTraceWithContext("cutensornetCreateContractionPlan");
    HANDLE_CUTN_ERROR(cutensornetCreateContractionPlan(
        cutnHandle, m_tnDescr, m_tnPath, workDesc, &m_tnPlan));
  }
  // Compute the unnormalized overlap
  std::vector<const void *> rawDataIn(numTensors);
  for (int i = 0; i < mpsNumTensors; ++i) {
    rawDataIn[i] = m_mpsTensors[i].deviceData;
    rawDataIn[mpsNumTensors + i] = mpsOtherTensors[i].deviceData;
  }
  void *m_dOverlap{nullptr};
  HANDLE_CUDA_ERROR(cudaMalloc(&m_dOverlap, overlapSize));
  {
    ScopedTraceWithContext("cutensornetContractSlices");
    HANDLE_CUTN_ERROR(cutensornetContractSlices(cutnHandle, m_tnPlan,
                                                rawDataIn.data(), m_dOverlap, 0,
                                                workDesc, NULL, 0x0));
  }
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

  return std::abs(overlap);
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
    TensorNetState basisTensorNetState(basisState, scratchPad,
                                       state->getInternalContext());
    // Note: this is a basis state, hence bond dim == 1
    std::vector<MPSTensor> basisStateTensors = basisTensorNetState.factorizeMPS(
        1, std::numeric_limits<double>::min(),
        std::numeric_limits<double>::min(), MPSSettings().svdAlgo);
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
  const auto printState = [&os](const auto &stateVec) {
    for (auto &t : stateVec)
      os << t << "\n";
  };

  if (!m_contractedStateVec.empty())
    printState(m_contractedStateVec);
  else
    printState(state->getStateVector());
}

static Eigen::MatrixXcd reshapeMatrix(const Eigen::MatrixXcd &A) {
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

static Eigen::MatrixXcd reshapeStateVec(const Eigen::VectorXcd &stateVec) {
  Eigen::MatrixXcd A = stateVec;
  A.transposeInPlace();
  return reshapeMatrix(A);
}

MPSSimulationState::MpsStateData MPSSimulationState::createFromStateVec(
    cutensornetHandle_t cutnHandle, ScratchDeviceMem &inScratchPad,
    std::size_t size, std::complex<double> *ptr, int bondDim) {
  const std::size_t numQubits = std::log2(size);
  // Reverse the qubit order to match cutensornet convention
  auto newStateVec = TensorNetState::reverseQubitOrder(
      std::span<std::complex<double>>{ptr, size});
  auto stateVec =
      Eigen::Map<Eigen::VectorXcd>(newStateVec.data(), newStateVec.size());

  if (numQubits == 1) {
    void *d_tensor = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc(&d_tensor, 2 * sizeof(std::complex<double>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(d_tensor, ptr,
                                 2 * sizeof(std::complex<double>),
                                 cudaMemcpyHostToDevice));
    MPSTensor stateTensor;
    stateTensor.deviceData = d_tensor;
    stateTensor.extents = std::vector<int64_t>{2};
    auto state = TensorNetState::createFromMpsTensors({stateTensor},
                                                      inScratchPad, cutnHandle);
    return {std::move(state), std::vector<MPSTensor>{stateTensor}};
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
  const auto enforceBondDim = [bondDim](const Eigen::MatrixXcd &U,
                                        const Eigen::VectorXd &S,
                                        const Eigen::MatrixXcd &V)
      -> std::tuple<Eigen::MatrixXcd, Eigen::VectorXd, Eigen::MatrixXcd> {
    assert(U.cols() == S.size());
    assert(V.cols() == S.size());
    if (S.size() <= bondDim)
      return {U, S, V};

    // Truncation
    Eigen::MatrixXcd newU(U.rows(), bondDim);
    newU = U.leftCols(bondDim);
    Eigen::MatrixXcd newV(V.rows(), bondDim);
    newV = V.leftCols(bondDim);
    Eigen::VectorXd newS(bondDim);
    newS = S.head(bondDim);
    return std::make_tuple(newU, newS, newV);
  };
  for (std::size_t i = 0; i < numQubits - 1; ++i) {
    Eigen::BDCSVD<Eigen::MatrixXcd, Eigen::ComputeThinU | Eigen::ComputeThinV>
        svd(reshapedMat);
    const Eigen::MatrixXcd U_orig = svd.matrixU();
    const Eigen::MatrixXcd V_orig = svd.matrixV();
    const Eigen::VectorXd S_orig = svd.singularValues();
    const auto [U, S, V] = enforceBondDim(U_orig, S_orig, V_orig);
    numSingularValues.emplace_back(S.size());
    reshapedMat = (i != (numQubits - 2))
                      ? reshapeMatrix(S.asDiagonal() * V.adjoint())
                      : (S.asDiagonal() * V.adjoint());
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
  auto state = TensorNetState::createFromMpsTensors(mpsTensors, inScratchPad,
                                                    cutnHandle);
  return {std::move(state), mpsTensors};
}

std::unique_ptr<cudaq::SimulationState>
MPSSimulationState::createFromSizeAndPtr(std::size_t size, void *ptr,
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
      HANDLE_CUDA_ERROR(
          cudaMalloc(&d_tensor, numElements * sizeof(std::complex<double>)));
      // Note: cudaMemcpyDefault to handle both device/host data
      HANDLE_CUDA_ERROR(cudaMemcpy(d_tensor, dataPtr,
                                   numElements * sizeof(std::complex<double>),
                                   cudaMemcpyDefault));
      std::vector<int64_t> mpsExtents(extents.begin(), extents.end());
      MPSTensor stateTensor{d_tensor, mpsExtents};
      mpsTensors.emplace_back(stateTensor);
    }
    auto state = TensorNetState::createFromMpsTensors(mpsTensors, scratchPad,
                                                      m_cutnHandle);
    return std::make_unique<MPSSimulationState>(std::move(state), mpsTensors,
                                                scratchPad, m_cutnHandle);
  }
  auto [state, mpsTensors] = createFromStateVec(
      m_cutnHandle, scratchPad, size,
      reinterpret_cast<std::complex<double> *>(ptr), MPSSettings().maxBond);
  return std::make_unique<MPSSimulationState>(std::move(state), mpsTensors,
                                              scratchPad, m_cutnHandle);
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

    cudaq::info("Setting MPS max bond dimension to {}.", maxBond);
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

    cudaq::info("Setting MPS absolute cutoff to {}.", absCutoff);
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

    cudaq::info("Setting MPS relative cutoff to {}.", relCutoff);
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
    cudaq::info("Setting MPS SVD algorithm to {} ({}).", svdAlgo, iter->first);
  }
}

void MPSSimulationState::toHost(std::complex<double> *clientAllocatedData,
                                std::size_t numElements) const {
  auto stateVec = state->getStateVector();
  if (stateVec.size() != numElements)
    throw std::runtime_error(
        fmt::format("[MPSSimulationState] Dimension mismatch: expecting {} "
                    "elements but providing an array of size {}.",
                    stateVec.size(), numElements));
  for (std::size_t i = 0; i < numElements; ++i)
    clientAllocatedData[i] = stateVec[i];
}
} // namespace nvqir