/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecBatch.h"

#include "CuStateVecError.h"
#include "CuStateVecState.h"

#include <algorithm>
#include <bitset>
#include <cmath>
#include <complex>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace {

template <typename Scalar>
custatevecComputeType_t computeType() {
  // Pin FP32 explicitly so DEFAULT can never select a reduced-precision mode
  // (e.g. TF32) for a float (Complex64) state vector.
  if constexpr (std::is_same_v<Scalar, float>)
    return CUSTATEVEC_COMPUTE_32F;
  // For a double (Complex128) state vector, CUSTATEVEC_COMPUTE_DEFAULT resolves
  // to FP64, i.e. it is equivalent to CUSTATEVEC_COMPUTE_64F.
  return CUSTATEVEC_COMPUTE_DEFAULT;
}

template <typename Scalar>
cudaq::cusv::MatrixTask<Scalar> pauliGate(custatevecPauli_t pauli,
                                          int32_t target) {
  cudaq::cusv::MatrixTask<Scalar> task;
  task.targets = {target};
  const std::complex<Scalar> imaginary{0.0, 1.0};
  switch (pauli) {
  case CUSTATEVEC_PAULI_X:
    task.matrix = {0.0, 1.0, 1.0, 0.0};
    break;
  case CUSTATEVEC_PAULI_Y:
    task.matrix = {0.0, -imaginary, imaginary, 0.0};
    break;
  case CUSTATEVEC_PAULI_Z:
    task.matrix = {1.0, 0.0, 0.0, -1.0};
    break;
  default:
    throw std::invalid_argument("Invalid non-identity Pauli operator.");
  }
  return task;
}

} // namespace

namespace cudaq::cusv {

template <typename Scalar>
CuStateVecBatch<Scalar>::CuStateVecBatch(int32_t numWires, std::size_t capacity,
                                         bool allowFp32Emulation,
                                         bool initializeZeroState)
    : m_stateSize(std::size_t{1} << numWires), m_size(capacity),
      m_capacity(capacity), m_numWires(numWires) {
  if (numWires < 0 || capacity == 0 ||
      capacity > std::numeric_limits<uint32_t>::max())
    throw std::invalid_argument("Invalid batched state-vector dimensions.");
  HANDLE_CUSTATEVEC_ERROR(custatevecCreate(/*handle=*/&m_handle));
  try {
    HANDLE_CUSTATEVEC_ERROR(custatevecSetMathMode(
        /*handle=*/m_handle,
        /*mode=*/allowFp32Emulation
            ? CUSTATEVEC_MATH_MODE_ALLOW_FP32_EMULATED_BF16X9
            : CUSTATEVEC_MATH_MODE_DISALLOW_FP32_EMULATED_BF16X9));
    // cuBLAS handle backs the batched GEMM broadcast and inner-product paths.
    HANDLE_CUBLAS_ERROR(cublasCreate(/*handle=*/&m_blasHandle));
    HANDLE_CUDA_ERROR(cudaMalloc(&m_states, m_stateSize * m_capacity *
                                                sizeof(std::complex<Scalar>)));
    const std::vector<std::complex<Scalar>> factors(
        m_capacity, std::complex<Scalar>{1.0, 0.0});
    HANDLE_CUDA_ERROR(cudaMalloc(&m_broadcastFactors,
                                 m_capacity * sizeof(std::complex<Scalar>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(m_broadcastFactors, factors.data(),
                                 m_capacity * sizeof(std::complex<Scalar>),
                                 cudaMemcpyHostToDevice));
    if (initializeZeroState)
      setZeroState();
  } catch (...) {
    reset();
    throw;
  }
}

template <typename Scalar>
CuStateVecBatch<Scalar>::~CuStateVecBatch() {
  reset();
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::reset() noexcept {
  if (m_workspace)
    cudaFree(m_workspace);
  if (m_expectationStates)
    cudaFree(m_expectationStates);
  if (m_broadcastFactors)
    cudaFree(m_broadcastFactors);
  if (m_states)
    cudaFree(m_states);
  // Release the cuBLAS handle created for the batched GEMM operations.
  if (m_blasHandle)
    cublasDestroy(/*handle=*/m_blasHandle);
  if (m_handle)
    custatevecDestroy(/*handle=*/m_handle);
  m_workspace = nullptr;
  m_expectationStates = nullptr;
  m_broadcastFactors = nullptr;
  m_states = nullptr;
  m_blasHandle = nullptr;
  m_handle = nullptr;
  m_workspaceBytes = 0;
  m_stateSize = 0;
  m_size = 0;
  m_capacity = 0;
  m_numWires = 0;
}

template <typename Scalar>
void *CuStateVecBatch<Scalar>::statePointer(std::size_t stateIndex) const {
  if (stateIndex >= m_size)
    throw std::out_of_range("Batched state-vector index is out of range.");
  return static_cast<std::complex<Scalar> *>(m_states) +
         stateIndex * m_stateSize;
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::resize(std::size_t size) {
  if (size == 0 || size > m_capacity)
    throw std::invalid_argument("Invalid active batched state-vector count.");
  m_size = size;
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::setZeroState() {
  for (std::size_t index = 0; index < m_size; ++index)
    HANDLE_CUSTATEVEC_ERROR(custatevecInitializeStateVector(
        /*handle=*/m_handle, /*sv=*/statePointer(index),
        /*svDataType=*/complexDataType<Scalar>(),
        /*nIndexBits=*/static_cast<uint32_t>(m_numWires),
        /*svType=*/CUSTATEVEC_STATE_VECTOR_TYPE_ZERO));
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::setState(const void *deviceState) {
  if (!deviceState)
    throw std::invalid_argument("Initial device state cannot be null.");
  const std::size_t bytes = m_stateSize * sizeof(std::complex<Scalar>);
  if (deviceState != m_states)
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(m_states, deviceState, bytes,
                                      cudaMemcpyDeviceToDevice));
  broadcastState();
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::broadcastState() {
  if (m_size <= 1)
    return;
  if (m_stateSize >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("Batched state vector exceeds cuBLAS range.");
  const int64_t rows = static_cast<int64_t>(m_stateSize);
  const int64_t columns = static_cast<int64_t>(m_size - 1);
  // GEMM replicates the first state vector across the remaining batch slots via
  // a rank-1 outer product with the unit broadcast factors.
  if constexpr (std::is_same_v<Scalar, float>) {
    const cuFloatComplex alpha = make_cuFloatComplex(1.0F, 0.0F);
    const cuFloatComplex beta = make_cuFloatComplex(0.0F, 0.0F);
    HANDLE_CUBLAS_ERROR(cublasCgemm_64(
        /*handle=*/m_blasHandle, /*transa=*/CUBLAS_OP_N, /*transb=*/CUBLAS_OP_N,
        /*m=*/rows, /*n=*/columns, /*k=*/1, /*alpha=*/&alpha,
        /*A=*/static_cast<const cuFloatComplex *>(m_states), /*lda=*/rows,
        /*B=*/static_cast<const cuFloatComplex *>(m_broadcastFactors),
        /*ldb=*/1, /*beta=*/&beta,
        /*C=*/static_cast<cuFloatComplex *>(statePointer(1)), /*ldc=*/rows));
  } else {
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    HANDLE_CUBLAS_ERROR(cublasZgemm_64(
        /*handle=*/m_blasHandle, /*transa=*/CUBLAS_OP_N, /*transb=*/CUBLAS_OP_N,
        /*m=*/rows, /*n=*/columns, /*k=*/1, /*alpha=*/&alpha,
        /*A=*/static_cast<const cuDoubleComplex *>(m_states), /*lda=*/rows,
        /*B=*/static_cast<const cuDoubleComplex *>(m_broadcastFactors),
        /*ldb=*/1, /*beta=*/&beta,
        /*C=*/static_cast<cuDoubleComplex *>(statePointer(1)), /*ldc=*/rows));
  }
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::ensureExpectationState() {
  if (m_expectationStates)
    return;
  HANDLE_CUDA_ERROR(
      cudaMalloc(&m_expectationStates,
                 m_stateSize * m_capacity * sizeof(std::complex<Scalar>)));
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::ensureWorkspace(std::size_t requiredBytes) {
  if (requiredBytes <= m_workspaceBytes)
    return;
  if (m_workspace)
    HANDLE_CUDA_ERROR(cudaFree(m_workspace));
  m_workspace = nullptr;
  m_workspaceBytes = 0;
  HANDLE_CUDA_ERROR(cudaMalloc(&m_workspace, requiredBytes));
  m_workspaceBytes = requiredBytes;
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::applyMatrices(
    void *states, const MatrixTask<Scalar> &task,
    const std::vector<std::complex<Scalar>> &matrices, std::size_t matrixCount,
    custatevecMatrixMapType_t mapType, const int32_t *matrixIndices) {
  const std::vector<std::complex<Scalar>> *matrixData = &matrices;
  std::vector<std::complex<Scalar>> denseMatrices;
  if (task.matrixType != CUSTATEVEC_EX_MATRIX_DENSE) {
    if (task.matrixType != CUSTATEVEC_EX_MATRIX_DIAGONAL &&
        task.matrixType != CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL)
      throw std::invalid_argument("Invalid compact batched matrix type.");
    if (task.targets.size() >= std::numeric_limits<std::size_t>::digits)
      throw std::invalid_argument("Batched matrix is too wide.");
    const std::size_t dimension = std::size_t{1} << task.targets.size();
    if (matrixCount > std::numeric_limits<std::size_t>::max() / dimension ||
        matrices.size() != matrixCount * dimension ||
        dimension > std::numeric_limits<std::size_t>::max() / dimension ||
        matrixCount >
            std::numeric_limits<std::size_t>::max() / (dimension * dimension))
      throw std::invalid_argument("Invalid compact batched matrices.");
    denseMatrices.resize(matrixCount * dimension * dimension);
    for (std::size_t matrix = 0; matrix < matrixCount; ++matrix)
      for (std::size_t index = 0; index < dimension; ++index) {
        std::size_t row = index;
        std::size_t column = index;
        if (task.matrixType == CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL) {
          if (task.layout == CUSTATEVEC_MATRIX_LAYOUT_ROW)
            column = dimension - index - 1;
          else
            row = dimension - index - 1;
        }
        const std::size_t element = task.layout == CUSTATEVEC_MATRIX_LAYOUT_ROW
                                        ? row * dimension + column
                                        : column * dimension + row;
        denseMatrices[matrix * dimension * dimension + element] =
            matrices[matrix * dimension + index];
      }
    matrixData = &denseMatrices;
  }
  std::size_t requiredWorkspace = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecApplyMatrixBatchedGetWorkspaceSize(
      /*handle=*/m_handle, /*svDataType=*/complexDataType<Scalar>(),
      /*nIndexBits=*/m_numWires, /*nSVs=*/m_size, /*svStride=*/m_stateSize,
      /*mapType=*/mapType, /*matrixIndices=*/matrixIndices,
      /*matrices=*/matrixData->data(),
      /*matrixDataType=*/complexDataType<Scalar>(), /*layout=*/task.layout,
      /*adjoint=*/task.adjoint, /*nMatrices=*/matrixCount,
      /*nTargets=*/task.targets.size(), /*nControls=*/task.controls.size(),
      /*computeType=*/computeType<Scalar>(),
      /*extraWorkspaceSizeInBytes=*/&requiredWorkspace));
  ensureWorkspace(requiredWorkspace);
  HANDLE_CUSTATEVEC_ERROR(custatevecApplyMatrixBatched(
      /*handle=*/m_handle, /*batchedSv=*/states,
      /*svDataType=*/complexDataType<Scalar>(), /*nIndexBits=*/m_numWires,
      /*nSVs=*/m_size, /*svStride=*/m_stateSize, /*mapType=*/mapType,
      /*matrixIndices=*/matrixIndices, /*matrices=*/matrixData->data(),
      /*matrixDataType=*/complexDataType<Scalar>(), /*layout=*/task.layout,
      /*adjoint=*/task.adjoint, /*nMatrices=*/matrixCount,
      /*targets=*/task.targets.data(), /*nTargets=*/task.targets.size(),
      /*controls=*/task.controls.empty() ? nullptr : task.controls.data(),
      /*controlBitValues=*/
      task.controlValues.empty() ? nullptr : task.controlValues.data(),
      /*nControls=*/task.controls.size(), /*computeType=*/computeType<Scalar>(),
      /*extraWorkspace=*/m_workspace,
      /*extraWorkspaceSizeInBytes=*/m_workspaceBytes));
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::apply(const MatrixTask<Scalar> &task) {
  applyMatrices(m_states, task, task.matrix, 1,
                CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr);
}

template <typename Scalar>
void CuStateVecBatch<Scalar>::apply(
    const std::vector<MatrixTask<Scalar>> &tasks) {
  if (tasks.size() != m_size)
    throw std::invalid_argument("Invalid number of batched matrix tasks.");
  const auto &first = tasks.front();
  std::vector<std::vector<std::complex<Scalar>>> uniqueMatrices;
  std::vector<int32_t> matrixIndices;
  matrixIndices.reserve(tasks.size());
  for (const auto &task : tasks) {
    if (task.targets != first.targets || task.controls != first.controls ||
        task.controlValues != first.controlValues ||
        task.layout != first.layout || task.adjoint != first.adjoint ||
        task.matrixType != first.matrixType ||
        task.matrix.size() != first.matrix.size())
      throw std::invalid_argument(
          "Batched trajectory gates must have identical operands and layout.");
    const auto found =
        std::find(uniqueMatrices.begin(), uniqueMatrices.end(), task.matrix);
    if (found == uniqueMatrices.end()) {
      if (uniqueMatrices.size() >=
          static_cast<std::size_t>(std::numeric_limits<int32_t>::max()))
        throw std::overflow_error("Batched matrix count exceeds int32 range.");
      matrixIndices.push_back(static_cast<int32_t>(uniqueMatrices.size()));
      uniqueMatrices.push_back(task.matrix);
    } else {
      matrixIndices.push_back(
          static_cast<int32_t>(std::distance(uniqueMatrices.begin(), found)));
    }
  }
  std::vector<std::complex<Scalar>> matrices;
  matrices.reserve(uniqueMatrices.size() * first.matrix.size());
  for (const auto &matrix : uniqueMatrices)
    matrices.insert(matrices.end(), matrix.begin(), matrix.end());
  applyMatrices(m_states, first, matrices, uniqueMatrices.size(),
                CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED,
                matrixIndices.data());
}

template <typename Scalar>
std::vector<std::complex<Scalar>>
CuStateVecBatch<Scalar>::innerProducts(const void *otherStates) {
  if (!otherStates)
    throw std::invalid_argument("Inner-product state batch cannot be null.");
  if (m_stateSize >
      static_cast<std::size_t>(std::numeric_limits<int64_t>::max()))
    throw std::overflow_error("Batched state vector exceeds cuBLAS range.");

  ensureWorkspace(m_size * sizeof(std::complex<Scalar>));
  const int64_t elements = static_cast<int64_t>(m_stateSize);
  const int64_t count = static_cast<int64_t>(m_size);
  // Batched GEMM computes the per-state inner product <state|otherState> as a
  // conjugate-transpose dot product over each pair in the batch.
  if constexpr (std::is_same_v<Scalar, float>) {
    const cuFloatComplex alpha = make_cuFloatComplex(1.0F, 0.0F);
    const cuFloatComplex beta = make_cuFloatComplex(0.0F, 0.0F);
    HANDLE_CUBLAS_ERROR(cublasCgemmStridedBatched_64(
        /*handle=*/m_blasHandle, /*transa=*/CUBLAS_OP_C, /*transb=*/CUBLAS_OP_N,
        /*m=*/1, /*n=*/1, /*k=*/elements, /*alpha=*/&alpha,
        /*A=*/static_cast<const cuFloatComplex *>(m_states), /*lda=*/elements,
        /*strideA=*/elements,
        /*B=*/static_cast<const cuFloatComplex *>(otherStates),
        /*ldb=*/elements, /*strideB=*/elements, /*beta=*/&beta,
        /*C=*/static_cast<cuFloatComplex *>(m_workspace), /*ldc=*/1,
        /*strideC=*/1, /*batchCount=*/count));
  } else {
    const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
    const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);
    HANDLE_CUBLAS_ERROR(cublasZgemmStridedBatched_64(
        /*handle=*/m_blasHandle, /*transa=*/CUBLAS_OP_C, /*transb=*/CUBLAS_OP_N,
        /*m=*/1, /*n=*/1, /*k=*/elements, /*alpha=*/&alpha,
        /*A=*/static_cast<const cuDoubleComplex *>(m_states), /*lda=*/elements,
        /*strideA=*/elements,
        /*B=*/static_cast<const cuDoubleComplex *>(otherStates),
        /*ldb=*/elements, /*strideB=*/elements, /*beta=*/&beta,
        /*C=*/static_cast<cuDoubleComplex *>(m_workspace), /*ldc=*/1,
        /*strideC=*/1, /*batchCount=*/count));
  }

  std::vector<std::complex<Scalar>> result(m_size);
  HANDLE_CUDA_ERROR(cudaMemcpy(result.data(), m_workspace,
                               m_size * sizeof(std::complex<Scalar>),
                               cudaMemcpyDeviceToHost));
  return result;
}

template <typename Scalar>
std::vector<double> CuStateVecBatch<Scalar>::expectationPauli(
    const std::vector<std::vector<custatevecPauli_t>> &paulis,
    const std::vector<std::vector<int32_t>> &targets) {
  if (paulis.size() != targets.size())
    throw std::invalid_argument(
        "Pauli operators and target arrays must have the same size.");
  if (paulis.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error("Pauli-term count exceeds cuStateVec range.");
  for (std::size_t term = 0; term < paulis.size(); ++term) {
    if (paulis[term].size() != targets[term].size())
      throw std::invalid_argument(
          "Each Pauli term requires one target per operator.");
    if (targets[term].size() > std::numeric_limits<uint32_t>::max())
      throw std::overflow_error("Pauli-term width exceeds cuStateVec range.");
    for (std::size_t operatorIndex = 0; operatorIndex < paulis[term].size();
         ++operatorIndex) {
      const auto pauli = paulis[term][operatorIndex];
      if (pauli != CUSTATEVEC_PAULI_I && pauli != CUSTATEVEC_PAULI_X &&
          pauli != CUSTATEVEC_PAULI_Y && pauli != CUSTATEVEC_PAULI_Z)
        throw std::invalid_argument("Invalid Pauli operator.");
      const int32_t target = targets[term][operatorIndex];
      if (target < 0 || target >= m_numWires)
        throw std::invalid_argument("Pauli target is out of range.");
      const auto priorTargetsEnd = targets[term].begin() + operatorIndex;
      if (std::find(targets[term].begin(), priorTargetsEnd, target) !=
          priorTargetsEnd)
        throw std::invalid_argument("Pauli targets must be unique.");
    }
  }

  // <P> = Re<psi|P|psi> per term and batch member: apply the term's Pauli
  // factors to a scratch copy of every state, then take the batched inner
  // product with the originals. An all-identity term trivially has <P> = 1.
  std::vector<double> result(m_size * paulis.size());
  const std::size_t stateBytes =
      m_size * m_stateSize * sizeof(std::complex<Scalar>);
  for (std::size_t term = 0; term < paulis.size(); ++term) {
    const bool identity =
        std::all_of(paulis[term].begin(), paulis[term].end(),
                    [](auto pauli) { return pauli == CUSTATEVEC_PAULI_I; });
    if (identity) {
      for (std::size_t state = 0; state < m_size; ++state)
        result[state * paulis.size() + term] = 1.0;
      continue;
    }

    ensureExpectationState();
    HANDLE_CUDA_ERROR(cudaMemcpyAsync(m_expectationStates, m_states, stateBytes,
                                      cudaMemcpyDeviceToDevice));
    for (std::size_t operatorIndex = 0; operatorIndex < paulis[term].size();
         ++operatorIndex) {
      if (paulis[term][operatorIndex] == CUSTATEVEC_PAULI_I)
        continue;
      const auto task = pauliGate<Scalar>(paulis[term][operatorIndex],
                                          targets[term][operatorIndex]);
      applyMatrices(m_expectationStates, task, task.matrix, 1,
                    CUSTATEVEC_MATRIX_MAP_TYPE_BROADCAST, nullptr);
    }

    const auto values = innerProducts(m_expectationStates);
    for (std::size_t state = 0; state < m_size; ++state)
      result[state * paulis.size() + term] =
          static_cast<double>(values[state].real());
  }
  return result;
}

template <typename Scalar>
std::vector<std::size_t>
CuStateVecBatch<Scalar>::applyNoise(const NoiseTask<Scalar> &task,
                                    const std::vector<double> &randomNumbers) {
  if (randomNumbers.size() != m_size)
    throw std::invalid_argument(
        "Noise random-number count must match batch size.");
  if (task.matrices.empty() || task.matrixTypes.size() != task.matrices.size())
    throw std::invalid_argument("Invalid batched noise channel.");

  if (task.wires.size() >= std::numeric_limits<std::size_t>::digits)
    throw std::invalid_argument("Batched noise channel is too wide.");
  const std::size_t dimension = std::size_t{1} << task.wires.size();
  if (dimension > std::numeric_limits<std::size_t>::max() / dimension)
    throw std::invalid_argument("Batched noise matrix is too large.");
  const std::size_t matrixElements = dimension * dimension;
  // Expand any compact (diagonal / anti-diagonal) Kraus matrices to dense form
  // so every branch can be applied through one uniform indexed matrix map.
  const auto denseMatrices = [&] {
    std::vector<std::vector<std::complex<double>>> result;
    result.reserve(task.matrices.size());
    for (std::size_t matrixIndex = 0; matrixIndex < task.matrices.size();
         ++matrixIndex) {
      const auto &matrix = task.matrices[matrixIndex];
      const auto type = task.matrixTypes[matrixIndex];
      if (type == CUSTATEVEC_EX_MATRIX_DENSE) {
        if (matrix.size() != matrixElements)
          throw std::invalid_argument("Invalid dense Kraus matrix size.");
        result.push_back(matrix);
        continue;
      }
      if (type != CUSTATEVEC_EX_MATRIX_DIAGONAL &&
          type != CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL)
        throw std::invalid_argument("Invalid Kraus matrix type.");
      if (matrix.size() != dimension)
        throw std::invalid_argument("Invalid compact Kraus matrix size.");
      auto &dense = result.emplace_back(matrixElements);
      for (std::size_t row = 0; row < dimension; ++row) {
        const std::size_t column =
            type == CUSTATEVEC_EX_MATRIX_DIAGONAL ? row : dimension - row - 1;
        dense[row * dimension + column] = matrix[row];
      }
    }
    return result;
  }();

  // Select one Kraus branch independently for each batch member from its own
  // random number. selected[i] is the chosen branch; selectedProbabilities[i]
  // is used below to renormalize that member's collapsed state.
  std::vector<std::size_t> selected(m_size);
  std::vector<double> selectedProbabilities(m_size);
  bool selectsIdentity = false;
  // Sample of one Kraus branch for `stateIndex`, given a per-branch
  // probability accessor. Returns the first branch whose cumulative probability
  // exceeds this state's random number (with that branch's probability), or
  // {denseMatrices.size(), 0.0} if the random number falls past the last
  // branch.
  const auto sampleBranch =
      [&](std::size_t stateIndex,
          auto &&branchProbability) -> std::pair<std::size_t, double> {
    double cumulative = 0.0;
    for (std::size_t branch = 0; branch < denseMatrices.size(); ++branch) {
      const double probability = branchProbability(branch);
      cumulative += probability;
      if (randomNumbers[stateIndex] < cumulative)
        return {branch, probability};
    }
    return {denseMatrices.size(), 0.0};
  };

  // Mixed-unitary branches have fixed probabilities
  if (task.kind == NoiseChannelKind::MixedUnitary) {
    if (task.probabilities.size() != denseMatrices.size())
      throw std::invalid_argument("Invalid mixed-unitary probabilities.");
    for (std::size_t stateIndex = 0; stateIndex < m_size; ++stateIndex) {
      const std::size_t branch = sampleBranch(stateIndex, [&](std::size_t b) {
                                   return task.probabilities[b];
                                 }).first;
      selected[stateIndex] = branch;
      if (branch == denseMatrices.size()) // fell through -> implicit identity
        selectsIdentity = true;
      selectedProbabilities[stateIndex] = 1.0; // already normalized
    }
  } else {
    // General channel: branch i has probability <psi|K_i^dag K_i|psi>.
    std::vector<std::complex<Scalar>> products;
    products.reserve(denseMatrices.size() * matrixElements);
    for (const auto &matrix : denseMatrices) {
      for (std::size_t row = 0; row < dimension; ++row)
        for (std::size_t column = 0; column < dimension; ++column) {
          std::complex<Scalar> value{};
          for (std::size_t inner = 0; inner < dimension; ++inner)
            value += std::conj(std::complex<Scalar>(
                         matrix[inner * dimension + row])) *
                     std::complex<Scalar>(matrix[inner * dimension + column]);
          products.push_back(value);
        }
    }

    std::size_t requiredWorkspace = 0;
    HANDLE_CUSTATEVEC_ERROR(custatevecComputeExpectationBatchedGetWorkspaceSize(
        /*handle=*/m_handle, /*svDataType=*/complexDataType<Scalar>(),
        /*nIndexBits=*/m_numWires, /*nSVs=*/m_size, /*svStride=*/m_stateSize,
        /*matrices=*/products.data(),
        /*matrixDataType=*/complexDataType<Scalar>(),
        /*layout=*/CUSTATEVEC_MATRIX_LAYOUT_ROW,
        /*nMatrices=*/denseMatrices.size(), /*nBasisBits=*/task.wires.size(),
        /*computeType=*/computeType<Scalar>(),
        /*extraWorkspaceSizeInBytes=*/&requiredWorkspace));
    ensureWorkspace(requiredWorkspace);
    std::vector<double2> probabilities(m_size * denseMatrices.size());
    HANDLE_CUSTATEVEC_ERROR(custatevecComputeExpectationBatched(
        /*handle=*/m_handle, /*batchedSv=*/m_states,
        /*svDataType=*/complexDataType<Scalar>(), /*nIndexBits=*/m_numWires,
        /*nSVs=*/m_size, /*svStride=*/m_stateSize,
        /*expectationValues=*/probabilities.data(),
        /*matrices=*/products.data(),
        /*matrixDataType=*/complexDataType<Scalar>(),
        /*layout=*/CUSTATEVEC_MATRIX_LAYOUT_ROW,
        /*nMatrices=*/denseMatrices.size(), /*basisBits=*/task.wires.data(),
        /*nBasisBits=*/task.wires.size(),
        /*computeType=*/computeType<Scalar>(), /*extraWorkspace=*/m_workspace,
        /*extraWorkspaceSizeInBytes=*/m_workspaceBytes));

    for (std::size_t stateIndex = 0; stateIndex < m_size; ++stateIndex) {
      auto [branch, probability] = sampleBranch(stateIndex, [&](std::size_t b) {
        return probabilities[stateIndex * denseMatrices.size() + b].x;
      });
      if (branch == denseMatrices.size()) { // fell through -> keep last branch
        branch = denseMatrices.size() - 1;
        probability =
            probabilities[stateIndex * denseMatrices.size() + branch].x;
      }
      selected[stateIndex] = branch;
      selectedProbabilities[stateIndex] = probability;
      if (selectedProbabilities[stateIndex] <= 0.0)
        throw std::runtime_error("Selected Kraus branch has zero probability.");
    }
  }

  // Apply the selected branch to each batch member through an indexed matrix
  // map (matrixIndices[i] picks the matrix applied to state i).
  MatrixTask<Scalar> operation;
  operation.targets = task.wires;
  std::vector<std::complex<Scalar>> matrices;
  std::vector<int32_t> matrixIndices(m_size);
  // Mixed unitaries are already normalized: upload every branch matrix (plus an
  // appended identity when any state fell through) and index by the selection.
  if (task.kind == NoiseChannelKind::MixedUnitary) {
    matrices.reserve((denseMatrices.size() + selectsIdentity) * matrixElements);
    for (const auto &matrix : denseMatrices)
      for (const auto value : matrix)
        matrices.emplace_back(static_cast<Scalar>(value.real()),
                              static_cast<Scalar>(value.imag()));
    if (selectsIdentity)
      for (std::size_t row = 0; row < dimension; ++row)
        for (std::size_t column = 0; column < dimension; ++column)
          matrices.emplace_back(row == column ? Scalar{1} : Scalar{0},
                                Scalar{0});
    std::transform(
        selected.begin(), selected.end(), matrixIndices.begin(),
        [](std::size_t branch) { return static_cast<int32_t>(branch); });
    applyMatrices(
        m_states, operation, matrices, denseMatrices.size() + selectsIdentity,
        CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED, matrixIndices.data());
    return selected;
  }

  // A general Kraus operator is not norm-preserving, so scale each selected
  // matrix by 1/sqrt(branch probability) to renormalize its collapsed state.
  matrices.reserve(m_size * matrixElements);
  std::iota(matrixIndices.begin(), matrixIndices.end(), 0);
  for (std::size_t stateIndex = 0; stateIndex < m_size; ++stateIndex) {
    const auto &matrix = denseMatrices[selected[stateIndex]];
    const Scalar scale =
        static_cast<Scalar>(1.0 / std::sqrt(selectedProbabilities[stateIndex]));
    for (const auto value : matrix)
      matrices.emplace_back(static_cast<Scalar>(value.real()) * scale,
                            static_cast<Scalar>(value.imag()) * scale);
  }
  applyMatrices(m_states, operation, matrices, m_size,
                CUSTATEVEC_MATRIX_MAP_TYPE_MATRIX_INDEXED,
                matrixIndices.data());
  return selected;
}

template <typename Scalar>
std::vector<custatevecIndex_t>
CuStateVecBatch<Scalar>::measure(const std::vector<int32_t> &wires,
                                 const std::vector<double> &randomNumbers) {
  if (randomNumbers.size() != m_size)
    throw std::invalid_argument(
        "Measurement random-number count must match batch size.");
  std::vector<custatevecIndex_t> bitStrings(m_size);
  HANDLE_CUSTATEVEC_ERROR(custatevecMeasureBatched(
      /*handle=*/m_handle, /*batchedSv=*/m_states,
      /*svDataType=*/complexDataType<Scalar>(), /*nIndexBits=*/m_numWires,
      /*nSVs=*/m_size, /*svStride=*/m_stateSize,
      /*bitStrings=*/bitStrings.data(), /*bitOrdering=*/wires.data(),
      /*bitStringLen=*/wires.size(), /*randnums=*/randomNumbers.data(),
      /*collapse=*/CUSTATEVEC_COLLAPSE_NONE));
  return bitStrings;
}

template <typename Scalar>
cudaq::ExecutionResult CuStateVecBatch<Scalar>::sample(
    std::size_t stateIndex, const std::vector<int32_t> &wires,
    std::vector<double> randomNumbers, bool includeSequentialData) {
  if (randomNumbers.empty())
    return {};
  if (randomNumbers.size() > std::numeric_limits<uint32_t>::max())
    throw std::overflow_error(
        "Trajectory shot count exceeds cuStateVec range.");
  custatevecSamplerDescriptor_t sampler = nullptr;
  std::size_t requiredWorkspace = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecSamplerCreate(
      /*handle=*/m_handle, /*sv=*/statePointer(stateIndex),
      /*svDataType=*/complexDataType<Scalar>(), /*nIndexBits=*/m_numWires,
      /*sampler=*/&sampler, /*nMaxShots=*/randomNumbers.size(),
      /*extraWorkspaceSizeInBytes=*/&requiredWorkspace));
  try {
    ensureWorkspace(requiredWorkspace);
    HANDLE_CUSTATEVEC_ERROR(custatevecSamplerPreprocess(
        /*handle=*/m_handle, /*sampler=*/sampler,
        /*extraWorkspace=*/m_workspace,
        /*extraWorkspaceSizeInBytes=*/m_workspaceBytes));
    std::vector<custatevecIndex_t> bitStrings(randomNumbers.size());
    HANDLE_CUSTATEVEC_ERROR(custatevecSamplerSample(
        /*handle=*/m_handle, /*sampler=*/sampler,
        /*bitStrings=*/bitStrings.data(), /*bitOrdering=*/wires.data(),
        /*bitStringLen=*/wires.size(), /*randnums=*/randomNumbers.data(),
        /*nShots=*/randomNumbers.size(),
        /*output=*/CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

    cudaq::ExecutionResult result;
    for (std::size_t index = 0; index < bitStrings.size();) {
      const custatevecIndex_t value = bitStrings[index];
      std::size_t runLength = 1;
      while (index + runLength < bitStrings.size() &&
             bitStrings[index + runLength] == value)
        ++runLength;
      std::string bitString = formatBitString(value, wires.size());
      if (includeSequentialData)
        result.appendResult(bitString, runLength);
      else
        result.counts[bitString] += runLength;
      index += runLength;
    }
    HANDLE_CUSTATEVEC_ERROR(custatevecSamplerDestroy(/*sampler=*/sampler));
    return result;
  } catch (...) {
    custatevecSamplerDestroy(/*sampler=*/sampler);
    throw;
  }
}

template class CuStateVecBatch<float>;
template class CuStateVecBatch<double>;

} // namespace cudaq::cusv
