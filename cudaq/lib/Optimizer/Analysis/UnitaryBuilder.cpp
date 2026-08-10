/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Analysis/UnitaryBuilder.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#include "llvm/ADT/bit.h"
#include <algorithm>
#include <numeric>

using namespace cudaq;
using namespace mlir;

LogicalResult UnitaryBuilder::build(func::FuncOp func) {
  for (auto arg : func.getArguments()) {
    auto type = arg.getType();
    if (isa<cudaq::quake::RefType, cudaq::quake::VeqType>(type))
      if (allocateQubits(arg) == WalkResult::interrupt())
        return failure();
  }
  // We need to keep track of which qubits are ancillas. Hence, we save the
  // current number of qubits and consider any local allocations as ancillas.
  const std::size_t numQubits = getNumQubits();

  SmallVector<Complex, 16> matrix;
  SmallVector<Qubit, 16> qubits;
  auto result = func.walk([&](Operation *op) {
    if (auto nullWireOp = dyn_cast<cudaq::quake::NullWireOp>(op))
      return allocateQubits(nullWireOp.getResult());
    if (auto borrowOp = dyn_cast<cudaq::quake::BorrowWireOp>(op))
      return allocateQubits(borrowOp.getResult());
    if (auto allocOp = dyn_cast<cudaq::quake::AllocaOp>(op))
      return allocateQubits(allocOp.getResult());
    if (auto extractOp = dyn_cast<cudaq::quake::ExtractRefOp>(op))
      return visitExtractOp(extractOp);
    if (auto unwrapOp = dyn_cast<cudaq::quake::UnwrapOp>(op))
      return visitUnwrapOp(unwrapOp);
    if (auto optor = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
      optor.getOperatorMatrix(matrix);
      // If the operator couldn't produce a matrix, stop the walk.
      if (matrix.empty()) {
        optor.emitOpError("Couldn't produce matrix.");
        return WalkResult::interrupt();
      }
      auto quantumOperands = cudaq::quake::getQuantumOperands(op);
      if (quantumOperands.empty()) {
        optor.emitOpError("Couldn't get quantum operands");
        return WalkResult::interrupt();
      }
      // If we can't get the qubits involved in this operation, stop the walk
      if (failed(getQubits(quantumOperands, qubits))) {
        optor.emitOpError("Couldn't get the qubits.");
        return WalkResult::interrupt();
      }

      for (auto &&[newQuantumOp, quantumOp] :
           llvm::zip(cudaq::quake::getQuantumResults(op),
                     cudaq::quake::getQuantumOperands(op))) {
        auto entry = qubitMap.find(quantumOp);
        if (entry == qubitMap.end()) {
          optor.emitOpError("Operand has no qubit mapping.");
          return WalkResult::interrupt();
        }
        SmallVector<Qubit, 4> mapped(entry->second);
        qubitMap.insert({newQuantumOp, std::move(mapped)});
      }

      // When checking mapped circuits, we do a software swap, i.e., just change
      // the qubit mapping instead of applying the swap operation.
      if (upToMapping && isa<cudaq::quake::SwapOp>(op)) {
        auto lhs = qubitMap.find(op->getResult(0));
        auto rhs = qubitMap.find(op->getResult(1));
        if (lhs == qubitMap.end() || rhs == qubitMap.end()) {
          optor.emitOpError("Swap result has no qubit mapping.");
          return WalkResult::interrupt();
        }
        std::swap(lhs->second, rhs->second);
      } else {
        if (optor.getNegatedControls())
          negatedControls(*optor.getNegatedControls(), qubits);

        applyOperator(matrix, optor.getTargets().size(), qubits);

        if (optor.getNegatedControls())
          negatedControls(*optor.getNegatedControls(), qubits);
      }

      matrix.clear();
      qubits.clear();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();
  return deallocateAncillas(numQubits);
}

//===----------------------------------------------------------------------===//
// Visitors
//===----------------------------------------------------------------------===//

WalkResult UnitaryBuilder::visitExtractOp(cudaq::quake::ExtractRefOp op) {
  std::size_t index = 0;
  // We need to check whether the index is a "raw" index or not.
  if (op.hasConstantIndex())
    index = op.getRawIndex();
  else if (failed(getValueAsInt(op.getIndex(), index))) {
    op.emitError("Failed to get index as a integer.");
    return WalkResult::interrupt();
  }
  Qubit qubit = 0;
  if (failed(lookupQubit(op, op.getVeq(), index, qubit)))
    return WalkResult::interrupt();
  qubitMap.try_emplace(op.getResult()).first->second.push_back(qubit);
  return WalkResult::advance();
}

WalkResult UnitaryBuilder::visitUnwrapOp(cudaq::quake::UnwrapOp op) {
  Qubit qubit = 0;
  if (failed(lookupQubit(op, op.getOperand(), 0, qubit)))
    return WalkResult::interrupt();
  qubitMap.try_emplace(op.getResult()).first->second.push_back(qubit);
  return WalkResult::advance();
}

WalkResult UnitaryBuilder::allocateQubits(Value value) {
  auto [entry, success] = qubitMap.try_emplace(value);
  if (!success) {
    value.getDefiningOp()->emitError("Qubit already allocated.");
    return WalkResult::interrupt();
  }
  auto &qubits = entry->second;
  if (auto veq = dyn_cast<cudaq::quake::VeqType>(value.getType())) {
    if (!veq.hasSpecifiedSize()) {
      value.getDefiningOp()->emitError("Veq doesn't have a specified size.");
      return WalkResult::interrupt();
    }
    qubits.resize(veq.getSize());
    std::iota(entry->second.begin(), entry->second.end(), getNumQubits());
  } else {
    qubits.push_back(getNumQubits());
  }
  if (cudaq::quake::isAncilla(value.getDefiningOp()))
    llvm::append_range(ancillaQubits, qubits);
  growMatrix(qubits.size());
  return WalkResult::advance();
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

LogicalResult UnitaryBuilder::getValueAsInt(Value value, std::size_t &result) {
  if (auto op =
          dyn_cast_if_present<arith::ConstantIntOp>(value.getDefiningOp()))
    if (auto index = dyn_cast<IntegerAttr>(op.getValue())) {
      result = index.getInt();
      return success();
    }
  return failure();
}

LogicalResult UnitaryBuilder::lookupQubit(Operation *op, Value value,
                                          std::size_t index, Qubit &qubit) {
  auto entry = qubitMap.find(value);
  if (entry == qubitMap.end()) {
    op->emitError("Value has no qubit mapping.");
    return failure();
  }
  if (index >= entry->second.size()) {
    op->emitError("Qubit index out of range.");
    return failure();
  }
  qubit = entry->second[index];
  return success();
}

LogicalResult UnitaryBuilder::getQubits(ValueRange values,
                                        SmallVectorImpl<Qubit> &qubits) {
  for (Value value : values) {
    auto entry = qubitMap.find(value);
    if (entry == qubitMap.end() || entry->second.empty())
      return failure();
    if (auto veq = dyn_cast<cudaq::quake::VeqType>(value.getType())) {
      if (!veq.hasSpecifiedSize())
        return failure();
      llvm::copy(entry->second, std::back_inserter(qubits));
    } else {
      qubits.push_back(entry->second[0]);
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

LogicalResult UnitaryBuilder::deallocateAncillas(std::size_t numQubits) {
  // Qubit `k` occupies bit `1 << k`, and marked ancillas can sit at any index,
  // hence the bitmask. Without markers, fall back to the older rule: anything
  // allocated after the function arguments is scratch.
  std::size_t ancillaMask = 0;
  if (!ancillaQubits.empty()) {
    for (Qubit qubit : ancillaQubits)
      ancillaMask |= (1ULL << qubit);
  } else {
    if (numQubits == 0)
      return success();
    for (std::size_t qubit = numQubits, end = getNumQubits(); qubit < end;
         ++qubit)
      ancillaMask |= (1ULL << qubit);
  }
  if (ancillaMask == 0)
    return success();

  const std::size_t dim = matrix.rows();
  // The ancillas must be returned to whichever computational basis state they
  // came in as, i.e. the operator has to be block diagonal in the ancilla
  // index. Anything that mixes two different ancilla states is a dirty
  // ancilla.
  constexpr double tolerance = 1e-12;
  for (std::size_t col = 0; col < dim; ++col)
    for (std::size_t row = 0; row < dim; ++row)
      if ((row & ancillaMask) != (col & ancillaMask) &&
          std::abs(matrix(row, col)) > tolerance) {
        llvm::errs() << "Failed to clean up ancilla qubits.\n";
        return failure();
      }

  // Gather the |ancillas = 0> block. Dropping the ancilla bits preserves the
  // order of the remaining indices, so collecting them in increasing order
  // already puts the system qubits back in their own index order.
  SmallVector<std::size_t> systemIndices;
  systemIndices.reserve(dim >> llvm::popcount(ancillaMask));
  for (std::size_t index = 0; index < dim; ++index)
    if ((index & ancillaMask) == 0)
      systemIndices.push_back(index);

  const std::size_t size = systemIndices.size();
  UMatrix newMatrix(size, size);
  for (std::size_t col = 0; col < size; ++col)
    for (std::size_t row = 0; row < size; ++row)
      newMatrix(row, col) = matrix(systemIndices[row], systemIndices[col]);

  matrix.swap(newMatrix);
  return success();
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
  // Multi-target gate. `OperatorInterface` matrices index the first target as
  // the most-significant qubit, whereas the index math in `applyMatrix` treats
  // its first qubit as least-significant. Reverse the target qubits (the
  // trailing `numTargets` operands; any controls come first and keep their
  // order) to reconcile the two conventions.
  SmallVector<Qubit, 16> reordered(qubits.begin(), qubits.end());
  std::reverse(reordered.end() - numTargets, reordered.end());
  applyMatrix(m, numTargets, reordered);
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
    for (std::size_t j = 0; j < n; j++)
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
  const std::size_t dim = (1u << numTargets);
  for (std::size_t k = 0u, end = (matrix.size() >> qubits.size()); k < end;
       ++k) {
    auto idx = indicies(qubits, qubitsSorted, k);
    SmallVector<Complex, 8> cache(dim, 0);
    for (std::size_t i = 0; i < dim; i++) {
      cache[i] = m[idx.at(i)];
      m[idx.at(i)] = 0.;
    }
    for (std::size_t i = 0; i < dim; i++)
      for (std::size_t j = 0; j < dim; j++)
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
