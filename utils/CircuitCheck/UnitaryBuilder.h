/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EigenDense.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include <complex>
#include <vector>

namespace cudaq {

class UnitaryBuilder {
  using Complex = std::complex<double>;
  using UMatrix2 = Eigen::Matrix<Complex, 2, 2>;

public:
  using Qubit = unsigned;
  using UMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;

  UnitaryBuilder(UMatrix &matrix) : matrix(matrix) {}

  mlir::LogicalResult build(mlir::func::FuncOp func);

private:
  //===--------------------------------------------------------------------===//
  // Visitors
  //===--------------------------------------------------------------------===//

  mlir::WalkResult visitExtractOp(quake::ExtractRefOp op);

  mlir::WalkResult allocateQubits(mlir::Value value);

  //===--------------------------------------------------------------------===//
  // Helpers
  //===--------------------------------------------------------------------===//

  mlir::LogicalResult getValueAsInt(mlir::Value value, std::size_t &result);

  std::size_t getNumQubits() { return std::log2(matrix.rows()); }

  mlir::LogicalResult getQubits(mlir::ValueRange values,
                                mlir::SmallVectorImpl<Qubit> &qubits);

  void negatedControls(mlir::ArrayRef<bool> negatedControls,
                       mlir::ArrayRef<Qubit> qubits);

  mlir::LogicalResult deallocateAncillas(std::size_t numQubits);

  //===--------------------------------------------------------------------===//
  // Unitary
  //===--------------------------------------------------------------------===//

  void growMatrix(unsigned numQubits = 1u);

  void applyOperator(mlir::ArrayRef<Complex> m, unsigned numTargets,
                     mlir::ArrayRef<Qubit> qubits);

  /// Applies a general single-qubit unitary matrix
  void applyMatrix(mlir::ArrayRef<Complex> m, mlir::ArrayRef<Qubit> qubits);

  /// Applies a general multiple-control, multiple-target unitary matrix
  void applyMatrix(mlir::ArrayRef<Complex> m, unsigned numTargets,
                   mlir::ArrayRef<Qubit> qubits);

  /// Applies a general multiple-control, single-target unitary matrix
  void applyControlledMatrix(mlir::ArrayRef<Complex> m,
                             mlir::ArrayRef<Qubit> qubits);

  //===--------------------------------------------------------------------===//

  /// The unitary we are building
  UMatrix &matrix;

  /// Map values to qubits identifiers
  ///
  /// NOTE: To simplify the API and avoid the need to keep different maps for
  /// single qubits and registers, we add single qubits to this map as a vector
  /// of size one.
  mlir::DenseMap<mlir::Value, mlir::SmallVector<Qubit, 4>> qubitMap;
};

// rtol : Relative tolerance
// atol : Absolute tolerance

inline bool isApproxEqual(double lhs, double rhs, double rtol = 1e-05,
                          double atol = 1e-08) {
  return std::abs(rhs - lhs) <= (atol + rtol * std::abs(lhs));
}

/// @brief Returns the global phase conjugate.
///
/// Clients can use it to remove global phase from an unitary matrix.
inline std::complex<double>
getGlobalPhaseConjugate(const UnitaryBuilder::UMatrix &matrix, double atol) {
  // Since the matrix uses a column-major storage scheme, it is faster to search
  // for the first nonzero element in the first column.
  for (auto &elt : matrix.col(0)) {
    if (std::abs(elt) < atol)
      continue;
    // Speed up the case for 1 + 0i
    return elt == 1. ? 1. : std::exp(std::complex<double>(0., -std::arg(elt)));
  }
  llvm_unreachable("non unitary matrix");
}

inline bool isApproxEqual(const UnitaryBuilder::UMatrix &lhs,
                          const UnitaryBuilder::UMatrix &rhs,
                          bool up_to_global_phase = false, double rtol = 1e-05,
                          double atol = 1e-08) {
  using Complex = std::complex<double>;
  assert(rhs.size() == lhs.size());
  bool is_close = true;
  uint32_t const end = rhs.size();
  auto const *r_data = rhs.data();
  auto const *l_data = lhs.data();
  if (!up_to_global_phase) {
    for (uint32_t i = 0u; i < end && is_close; ++i) {
      is_close &= isApproxEqual(l_data[i].real(), r_data[i].real(), rtol, atol);
      is_close &= isApproxEqual(l_data[i].imag(), r_data[i].imag(), rtol, atol);
    }
    return is_close;
  }

  // Get a constant multiplier that will remove the global phase
  // for both the LHS and RHS matrices. Will return 1.0 if no global phase
  Complex lhsMultiplier = getGlobalPhaseConjugate(lhs, atol);
  Complex rhsMultiplier = getGlobalPhaseConjugate(rhs, atol);
  for (uint32_t i = 0u; i < end && is_close; ++i) {
    auto lhsEle = lhsMultiplier * l_data[i];
    auto rhsEle = rhsMultiplier * r_data[i];
    is_close &= isApproxEqual(lhsEle.real(), rhsEle.real(), rtol, atol);
    is_close &= isApproxEqual(lhsEle.imag(), rhsEle.imag(), rtol, atol);
  }
  return is_close;
}

} // namespace cudaq
