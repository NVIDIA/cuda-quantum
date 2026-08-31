/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecTasks.h"
#include "common/SampleResult.h"

#include <cublas_v2.h>
#include <custatevec.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace cudaq::cusv {
/// @brief Owns a batch of independent single-device state vectors.
///
/// This helper applies one matrix operation per trajectory and performs batched
/// measurement or sampling. It is used by the trajectory simulator when a
/// mixed-unitary or general Kraus noise workload is eligible for batched
/// execution.
// Note: this class encapsulates custatevec/custatevecEx details. Currently, use
// custatevec batching API as there is no custatevecEx batching API. This class
// will be updated to use custatevecEx batching API when it becomes available.
template <typename Scalar>
class CuStateVecBatch {
public:
  CuStateVecBatch(int32_t numWires, std::size_t capacity,
                  bool allowFp32Emulation, bool initializeZeroState = true);
  ~CuStateVecBatch();
  CuStateVecBatch(const CuStateVecBatch &) = delete;
  CuStateVecBatch &operator=(const CuStateVecBatch &) = delete;
  CuStateVecBatch(CuStateVecBatch &&) = delete;
  CuStateVecBatch &operator=(CuStateVecBatch &&) = delete;

  /// Set the number of active state vectors without reallocating device
  /// storage.
  void resize(std::size_t size);

  /// Initialize every batch member to the zero state.
  void setZeroState();

  /// Replicate one device-resident state across the batch.
  void setState(const void *deviceState);

  /// Apply one matrix task uniformly to the batch.
  void apply(const MatrixTask<Scalar> &task);

  /// Apply one structurally compatible matrix task to each active batch member.
  /// Matrices may differ and are passed through an indexed matrix map.
  void apply(const std::vector<MatrixTask<Scalar>> &tasks);

  /// Select and normalize one Kraus branch independently for each batch member.
  std::vector<std::size_t> applyNoise(const NoiseTask<Scalar> &task,
                                      const std::vector<double> &randomNumbers);

  /// Compute Pauli-basis expectations by applying individual Pauli gates to a
  /// scratch batch and taking state-wise inner products. Results are
  /// state-major, then term-major.
  std::vector<double>
  expectationPauli(const std::vector<std::vector<custatevecPauli_t>> &paulis,
                   const std::vector<std::vector<int32_t>> &targets);

  /// Measure each batch member once without collapse.
  std::vector<custatevecIndex_t>
  measure(const std::vector<int32_t> &wires,
          const std::vector<double> &randomNumbers);

  /// Sample one completed batch member with its exact trajectory shot count.
  cudaq::ExecutionResult sample(std::size_t stateIndex,
                                const std::vector<int32_t> &wires,
                                std::vector<double> randomNumbers,
                                bool includeSequentialData);

private:
  void applyMatrices(void *states, const MatrixTask<Scalar> &task,
                     const std::vector<std::complex<Scalar>> &matrices,
                     std::size_t matrixCount, custatevecMatrixMapType_t mapType,
                     const int32_t *matrixIndices);
  std::vector<std::complex<Scalar>> innerProducts(const void *otherStates);
  void broadcastState();
  void ensureExpectationState();
  void ensureWorkspace(std::size_t requiredBytes);
  void *statePointer(std::size_t stateIndex) const;
  void reset() noexcept;

  custatevecHandle_t m_handle = nullptr;
  cublasHandle_t m_blasHandle = nullptr;
  void *m_states = nullptr;
  /// Device-side scratch copy of the batched states used to evaluate
  /// expectation values.
  void *m_expectationStates = nullptr;
  /// Device buffer of `m_capacity` complex scale factors (initialized to 1).
  /// Serves as the `B` operand of the cuBLAS GEMM in `broadcastState()` that
  /// replicates a single state across all batch members.
  void *m_broadcastFactors = nullptr;
  /// General-purpose device scratch buffer (for cuStateVec/cuBLAS operations).
  void *m_workspace = nullptr;
  std::size_t m_workspaceBytes = 0;
  std::size_t m_stateSize = 0;
  std::size_t m_size = 0;
  std::size_t m_capacity = 0;
  int32_t m_numWires = 0;
};

extern template class CuStateVecBatch<float>;
extern template class CuStateVecBatch<double>;

} // namespace cudaq::cusv
