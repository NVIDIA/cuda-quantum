/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include <custatevec.h>
#include <custatevecEx.h>

#include <algorithm>
#include <bitset>
#include <cassert>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace cudaq::cusv {

/// Return the data pointer of `values`, or `nullptr` when it is empty, as
/// required by the cuStateVecEx arrays that encode an empty operand list.
template <typename T>
const T *dataOrNull(const std::vector<T> &values) {
  return values.empty() ? nullptr : values.data();
}

/// Convert a cuStateVec sample (ascending, most-significant-wire-first) into a
/// CUDA-Q little-endian bit-string spanning `numWires` characters.
inline std::string formatBitString(custatevecIndex_t value,
                                   std::size_t numWires) {
  // The sample is read from a 64-bit word, so it cannot span more wires.
  assert(numWires <= 64 && "A sample cannot span more than 64 wires.");
  std::string bits = std::bitset<64>(value).to_string();
  bits.erase(0, 64 - numWires);
  std::reverse(bits.begin(), bits.end());
  return bits;
}

/// Describes a dense or specialized matrix operation and its operands.
template <typename Scalar>
struct MatrixTask {
  std::vector<std::complex<Scalar>> matrix;
  std::vector<int32_t> targets;
  std::vector<int32_t> controls;
  std::vector<int32_t> controlValues;
  custatevecExMatrixType_t matrixType = CUSTATEVEC_EX_MATRIX_DENSE;
  custatevecMatrixLayout_t layout = CUSTATEVEC_MATRIX_LAYOUT_ROW;
  bool adjoint = false;
};

/// Replace a dense matrix with the compact diagonal or anti-diagonal storage
/// required by cuStateVecEx when its zero pattern permits that classification.
template <typename Scalar>
void compactMatrixTask(MatrixTask<Scalar> &task) {
  if (task.matrixType != CUSTATEVEC_EX_MATRIX_DENSE)
    return;
  if (task.targets.size() >= std::numeric_limits<std::size_t>::digits)
    throw std::invalid_argument("Matrix task target count is too large.");
  const std::size_t dimension = std::size_t{1} << task.targets.size();
  if (dimension > std::numeric_limits<std::size_t>::max() / dimension ||
      task.matrix.size() != dimension * dimension)
    throw std::invalid_argument("Invalid dense matrix task size.");

  const auto element = [&](std::size_t row, std::size_t column) {
    const std::size_t index = task.layout == CUSTATEVEC_MATRIX_LAYOUT_ROW
                                  ? row * dimension + column
                                  : column * dimension + row;
    return task.matrix[index];
  };
  // Every non-zero must lie on the main diagonal (row == column) to be
  // diagonal, or on the anti-diagonal (column == dimension - row - 1) to be
  // anti-diagonal; an all-zero matrix satisfies both.
  bool diagonal = true;
  bool antiDiagonal = true;
  for (std::size_t row = 0; row < dimension; ++row)
    for (std::size_t column = 0; column < dimension; ++column) {
      if (element(row, column) == std::complex<Scalar>{})
        continue;
      diagonal &= row == column;
      antiDiagonal &= column == dimension - row - 1;
    }

  if (!diagonal && !antiDiagonal)
    return;
  std::vector<std::complex<Scalar>> compact(dimension);
  if (diagonal) {
    for (std::size_t index = 0; index < dimension; ++index)
      compact[index] = element(index, index);
    task.matrixType = CUSTATEVEC_EX_MATRIX_DIAGONAL;
  } else {
    for (std::size_t index = 0; index < dimension; ++index)
      compact[index] = task.layout == CUSTATEVEC_MATRIX_LAYOUT_ROW
                           ? element(index, dimension - index - 1)
                           : element(dimension - index - 1, index);
    task.matrixType = CUSTATEVEC_EX_MATRIX_ANTI_DIAGONAL;
  }
  task.matrix = std::move(compact);
}

/// Describes a controlled Pauli rotation and its operands.
struct PauliRotationTask {
  double angle = 0.0;
  std::vector<custatevecPauli_t> paulis;
  std::vector<int32_t> targets;
  std::vector<int32_t> controls;
  std::vector<int32_t> controlValues;
};

/// Identifies whether noise probabilities are fixed or state dependent.
enum class NoiseChannelKind { MixedUnitary, General };

/// Describes a mixed-unitary or general Kraus channel on a set of wires.
template <typename Scalar>
struct NoiseTask {
  NoiseChannelKind kind = NoiseChannelKind::General;
  std::vector<std::vector<std::complex<double>>> matrices;
  std::vector<custatevecExMatrixType_t> matrixTypes;
  std::vector<double> probabilities;
  std::vector<int32_t> wires;
};

/// Compact each Kraus matrix of `task` to cuStateVecEx's diagonal or
/// anti-diagonal storage where its zero pattern permits, and record the
/// resulting matrix type in `task.matrixTypes`. `task.wires` must already hold
/// the channel's target wires. Kraus matrices are stored in double precision
/// regardless of the simulator scalar type.
template <typename Scalar>
void compactNoiseMatrices(NoiseTask<Scalar> &task) {
  task.matrixTypes.clear();
  task.matrixTypes.reserve(task.matrices.size());
  for (auto &matrix : task.matrices) {
    MatrixTask<double> compacted;
    compacted.matrix = std::move(matrix);
    compacted.targets = task.wires;
    compactMatrixTask(compacted);
    matrix = std::move(compacted.matrix);
    task.matrixTypes.push_back(compacted.matrixType);
  }
}

/// A gate, Pauli rotation, or noise-channel task accepted by a gate engine.
template <typename Scalar>
using SimulationTask =
    std::variant<MatrixTask<Scalar>, PauliRotationTask, NoiseTask<Scalar>>;

} // namespace cudaq::cusv
