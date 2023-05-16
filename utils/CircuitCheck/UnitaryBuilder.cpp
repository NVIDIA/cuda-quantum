/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "UnitaryBuilder.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include <numeric>

using namespace cudaq;
using namespace mlir;

LogicalResult UnitaryBuilder::build(func::FuncOp func) {
  for (auto arg : func.getArguments()) {
    auto type = arg.getType();
    if (type.isa<quake::RefType>() || type.isa<quake::VeqType>())
      if (allocateQubits(arg) == WalkResult::interrupt())
        return failure();
  }
  SmallVector<Complex, 16> matrix;
  SmallVector<Qubit, 16> qubits;
  auto result = func.walk([&](Operation *op) {
    if (auto allocOp = dyn_cast<quake::AllocaOp>(op)) {
      return allocateQubits(allocOp.getResult());
    }
    if (auto extractOp = dyn_cast<quake::ExtractRefOp>(op)) {
      return visitExtractOp(extractOp);
    }
    if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
      optor.getOperatorMatrix(matrix);
      // If the operator couldn't produce a matrix, stop the walk.
      if (matrix.empty())
        return WalkResult::interrupt();
      // If we can't get the qubits involved in this operation, stop the walk
      if (failed(getQubits(optor.getControls(), qubits)) ||
          failed(getQubits(optor.getTargets(), qubits)))
        return WalkResult::interrupt();

      if (optor.getNegatedControls())
        negatedControls(*optor.getNegatedControls(), qubits);

      applyOperator(matrix, optor.getTargets().size(), qubits);

      if (optor.getNegatedControls())
        negatedControls(*optor.getNegatedControls(), qubits);

      matrix.clear();
      qubits.clear();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Visitors
//===----------------------------------------------------------------------===//

WalkResult UnitaryBuilder::visitExtractOp(quake::ExtractRefOp op) {
  auto veq = op.getVeq();
  auto qubits = qubitMap[veq];
  auto index = getValueAsInt(op.getIndex());
  if (!index && *index < 0)
    return WalkResult::interrupt();
  auto [entry, _] = qubitMap.try_emplace(op.getResult());
  entry->second.push_back(qubits[*index]);
  return WalkResult::advance();
}

WalkResult UnitaryBuilder::allocateQubits(Value value) {
  auto [entry, success] = qubitMap.try_emplace(value);
  if (!success)
    return WalkResult::interrupt();
  auto &qubits = entry->second;
  if (auto veq = value.getType().dyn_cast<quake::VeqType>()) {
    if (!veq.hasSpecifiedSize())
      return WalkResult::interrupt();
    qubits.resize(veq.getSize());
    std::iota(entry->second.begin(), entry->second.end(), getNextQubit());
  } else {
    qubits.push_back(getNextQubit());
  }
  growMatrix(qubits.size());
  return WalkResult::advance();
}

std::optional<int64_t> UnitaryBuilder::getValueAsInt(Value value) {
  if (auto constOp =
          dyn_cast_if_present<arith::ConstantOp>(value.getDefiningOp()))
    if (auto index = dyn_cast<IntegerAttr>(constOp.getValue()))
      return index.getInt();
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

LogicalResult UnitaryBuilder::getQubits(ValueRange values,
                                        SmallVectorImpl<Qubit> &qubits) {
  for (Value value : values) {
    if (dyn_cast<quake::WireType>(value.getType()))
      return failure();

    if (auto veq = dyn_cast<quake::VeqType>(value.getType())) {
      if (!veq.hasSpecifiedSize())
        return failure();
      llvm::copy(qubitMap[value], std::back_inserter(qubits));
    } else {
      qubits.push_back(qubitMap[value][0]);
    }
  }
  return success();
}

void UnitaryBuilder::negatedControls(ArrayRef<bool> negatedControls,
                                     ArrayRef<Qubit> qubits) {
  for (auto [isNegated, qubit] : llvm::zip(negatedControls, qubits))
    if (isNegated)
      applyMatrix({0, 1, 1, 0}, qubit); // Apply pauli-x to the qubit
}

//===----------------------------------------------------------------------===//
// Matrices
//===----------------------------------------------------------------------===//

void UnitaryBuilder::applyOperator(ArrayRef<Complex> m, unsigned numTargets,
                                   ArrayRef<Qubit> qubits) {
  if (qubits.size() == 1u) {
    applyMatrix(m, qubits);
    return;
  }
  if (numTargets == 1) {
    applyControlledMatrix(m, qubits);
    return;
  }
  applyMatrix(m, numTargets, qubits);
}

void UnitaryBuilder::growMatrix(unsigned numQubits) {
  if (matrix.size() == 0) {
    matrix = UMatrix::Identity((1 << numQubits), (1 << numQubits));
    return;
  }
  UMatrix m((matrix.rows() << numQubits), (matrix.cols() << numQubits));
  m.setZero();
  if (numQubits == 1) {
    m.block(0, 0, matrix.rows(), matrix.cols()) = matrix;
    m.block(matrix.rows(), matrix.cols(), matrix.rows(), matrix.cols()) =
        matrix;
  } else {
    for (unsigned i = 0; i < (1u << numQubits); ++i)
      m.block(i * matrix.rows(), i * matrix.cols(), matrix.rows(),
              matrix.cols()) = matrix;
  }
  matrix.swap(m);
}

// The code below update the unitary matrix when a applying a new operation.
// It is hard to understand it, but I will give a simple overview.
//
// Let `C` be an unitary matrix representing a mapping that describes the
// evolution of a `n`-qubit system.  Now, suppose we want to update this
// description by saying applying an operator `U` to a set of `m` qubits, i.e.
// `U` is a `m`-qubit operator.
//
// If `n = m` we can 'straightforwardly' update `C` by doing `C_1 = U * C_0`,
// where `C_0` and `C_1` indicate the original and the updated versions of `C`,
// respectively.
//
// However,  if `n > m`, we can't compute `U * C_0` because they have different
// dimensions.  So first, we combine it with identity matrices.
//
// Example: Suppose that `m = 1`, If we apply `U` to the first qubit.
//
//          ┌────┐┌───┐     ┌────┐┌───┐     ┌────┐┌─────┐     ┌────┐
//     q : ─┤    ├┤ U ├─   ─┤    ├┤ U ├─   ─┤    ├┤     ├─   ─┤    ├─
//      0   │    │└───┘     │    │└───┘     │    ││     │     │    │
//          │    │          │    │┌───┐     │    ││     │     │    │
//     q : ─┤ C  ├────── = ─┤ C  ├┤   ├─ = ─┤ C  ├┤ I⊗U ├─ = ─┤ C  ├─
//      1   │  0 │          │  0 ││   │     │  0 ││     │     │  1 │
//          │    │          │    ││ I │     │    ││     │     │    │
//     q : ─┤    ├──────   ─┤    ├┤   ├─   ─┤    ├┤     ├─   ─┤    ├─
//      n   └────┘          └────┘└───┘     └────┘└─────┘     └────┘
//
//   So we have the following:
//
//   C  = (I ⊗ U) * C
//    1              0
//
//   Where `I` is the identity matrix with appropriate dimensions to guarantee
//   that `dim(I ⊗ U) = dim(C)`.
//
// The code below is implementing a generalized version of what is happening
// in the example.  Here are are a couple things to keep in mind:
//
//   * We want use as little memory as possible, so we don't explicitly compute
//   the `I1 ⊗ U ⊗ I0`, and we modify `C` in-place.
//
//   * We represent `C` and `U` as contiguous one-dimensional vector using
//   column-major ordering:
//
//     M = | a b c d |  Column-major array = [ a, e, i, m,
//         | e f g h |                         b, f, j, n,
//         | i j k l |                         c, g, k, o,
//         | m n o p |                         d, h, l, p ]

static unsigned first_idx(ArrayRef<UnitaryBuilder::Qubit> qubits, unsigned k) {
  unsigned lowBits;
  unsigned result = k;
  for (auto qubit : qubits) {
    lowBits = result & ((1 << qubit) - 1);
    result >>= qubit;
    result <<= qubit + 1;
    result |= lowBits;
  }
  return result;
}

static std::vector<unsigned>
indicies(ArrayRef<UnitaryBuilder::Qubit> qubits,
         ArrayRef<UnitaryBuilder::Qubit> qubitsSorted, unsigned k) {
  std::vector<unsigned> result((1 << qubits.size()), 0u);
  result.at(0) = first_idx(qubitsSorted, k);
  for (unsigned i = 0u, end = qubits.size(); i < end; ++i) {
    unsigned n = (1u << i);
    unsigned bit = (1u << qubits[i]);
    for (size_t j = 0; j < n; j++)
      result.at(n + j) = result.at(j) | bit;
  }
  return result;
}

// TODO:  Optimize!  There are ways to specialize for diagonal and anti-diagonal
// matrices.
void UnitaryBuilder::applyMatrix(ArrayRef<Complex> u, ArrayRef<Qubit> qubits) {
  auto *m = matrix.data();
  for (unsigned k = 0u, end = (matrix.size() >> 1u); k < end; ++k) {
    auto idx = indicies(qubits, qubits, k);
    auto cache = m[idx.at(0)];
    m[idx.at(0)] = u[0] * cache + u[2] * m[idx.at(1)];
    m[idx.at(1)] = u[1] * cache + u[3] * m[idx.at(1)];
  }
}

void UnitaryBuilder::applyMatrix(ArrayRef<Complex> u, unsigned numTargets,
                                 ArrayRef<Qubit> qubits) {
  SmallVector<Qubit, 16> qubitsSorted(qubits);
  llvm::sort(qubitsSorted);

  auto *m = matrix.data();
  const size_t dim = (1u << numTargets);
  for (size_t k = 0u, end = (matrix.size() >> qubits.size()); k < end; ++k) {
    auto idx = indicies(qubits, qubitsSorted, k);
    SmallVector<Complex, 8> cache(dim, 0);
    for (size_t i = 0; i < dim; i++) {
      cache[i] = m[idx.at(i)];
      m[idx.at(i)] = 0.;
    }
    for (size_t i = 0; i < dim; i++)
      for (size_t j = 0; j < dim; j++)
        m[idx.at(i)] += u[i + dim * j] * cache[j];
  }
}

void UnitaryBuilder::applyControlledMatrix(ArrayRef<Complex> u,
                                           ArrayRef<Qubit> qubits) {
  SmallVector<Qubit, 16> qubitsSorted(qubits);
  llvm::sort(qubitsSorted);
  unsigned p0 = (1 << (qubits.size() - 1)) - 1;
  unsigned p1 = (1 << qubits.size()) - 1;

  auto *m = matrix.data();
  for (unsigned k = 0u, end = (matrix.size() >> qubits.size()); k < end; ++k) {
    auto idx = indicies(qubits, qubitsSorted, k);
    auto cache = m[idx.at(p0)];
    m[idx.at(p0)] = u[0] * cache + u[2] * m[idx.at(p1)];
    m[idx.at(p1)] = u[1] * cache + u[3] * m[idx.at(p1)];
  }
}
