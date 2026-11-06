/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Analysis/CodePresentation.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/SHA256.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/SymbolTable.h"

#include <limits>
#include <utility>

using namespace mlir;

namespace {

using Rows = llvm::SmallVector<llvm::SmallBitVector>;

FailureOr<Rows> readBinaryRows(Operation *owner, StringRef name,
                               unsigned expectedRows,
                               unsigned expectedColumns) {
  auto matrix = owner->getAttrOfType<DenseIntElementsAttr>(name);
  if (!matrix || matrix.getType().getRank() != 2 ||
      !matrix.getType().getElementType().isInteger(1)) {
    owner->emitOpError()
        << "EDEM code presentation requires canonical rank-2 i1 " << name;
    return failure();
  }
  auto shape = matrix.getType().getShape();
  if (shape[0] != expectedRows || shape[1] != expectedColumns) {
    owner->emitOpError() << "EDEM code presentation " << name
                         << " must have shape " << expectedRows << "x"
                         << expectedColumns;
    return failure();
  }
  Rows rows;
  rows.reserve(expectedRows);
  auto values = matrix.getValues<APInt>();
  auto value = values.begin();
  for (unsigned row = 0; row < expectedRows; ++row) {
    llvm::SmallBitVector bits(expectedColumns);
    for (unsigned column = 0; column < expectedColumns; ++column, ++value)
      if (!(*value).isZero())
        bits.set(column);
    rows.push_back(std::move(bits));
  }
  return rows;
}

llvm::SmallBitVector symplecticFunctional(llvm::SmallBitVector const &pauli,
                                          unsigned physicalQubits) {
  llvm::SmallBitVector functional(2 * physicalQubits);
  for (int bit : pauli.set_bits()) {
    auto index = static_cast<unsigned>(bit);
    functional.set(index < physicalQubits ? index + physicalQubits
                                          : index - physicalQubits);
  }
  return functional;
}

} // namespace

namespace cudaq::logical {

FailureOr<CodePresentation>
CodePresentation::fromProfile(qlx::fabric::CodeProfileOp profile) {
  auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
      SymbolTable::lookupNearestSymbolFrom(profile, profile.getCodeAttr()));
  if (!code) {
    profile.emitOpError(
        "EDEM code presentation requires a linked fabric.code declaration");
    return failure();
  }

  const int64_t n = code.getN().value_or(0);
  const int64_t k = code.getK().value_or(1);
  const int64_t r = code.getR().value_or(0);
  if (n <= 0 || k < 0 || r < 0 || k > n || r > n - k ||
      n > std::numeric_limits<int>::max() / 2) {
    profile.emitOpError(
        "EDEM code presentation requires finite canonical code widths");
    return failure();
  }
  if (r != 0) {
    profile.emitOpError(
        "EDEM production currently requires a stabilizer code (r = 0)");
    return failure();
  }
  const auto physicalQubits = static_cast<unsigned>(n);
  const auto logicalQubits = static_cast<unsigned>(k);
  const auto keptSyndromes = static_cast<unsigned>(n - k - r);
  const auto physicalWidth = 2 * physicalQubits;

  // The canonical code/profile verifiers enforce the algebra. Require all
  // fields here as well: a CSS-only declaration does not provide a complete
  // EDEM code presentation.
  auto stabilizers =
      readBinaryRows(code, "stabilizer_basis", keptSyndromes, physicalWidth);
  auto antiStabilizers =
      readBinaryRows(code, "anti_stabilizers", keptSyndromes, physicalWidth);
  auto logicalX =
      readBinaryRows(code, "logical_x_basis", logicalQubits, physicalWidth);
  auto logicalZ =
      readBinaryRows(code, "logical_z_basis", logicalQubits, physicalWidth);
  if (failed(stabilizers) || failed(antiStabilizers) || failed(logicalX) ||
      failed(logicalZ))
    return failure();

  auto effectiveAttr =
      profile->getAttrOfType<DenseIntElementsAttr>("effective_stabilizers");
  if (!effectiveAttr || effectiveAttr.getType().getRank() != 2 ||
      !effectiveAttr.getType().getElementType().isInteger(1) ||
      effectiveAttr.getType().getShape()[1] != physicalWidth) {
    profile.emitOpError(
        "EDEM code presentation requires canonical effective_stabilizers");
    return failure();
  }
  const int64_t effectiveCount = effectiveAttr.getType().getShape()[0];
  if (effectiveCount < keptSyndromes ||
      effectiveCount > std::numeric_limits<int>::max()) {
    profile.emitOpError(
        "EDEM code presentation effective-syndrome width is invalid");
    return failure();
  }
  auto effectiveSyndromes = static_cast<unsigned>(effectiveCount);
  auto effective = readBinaryRows(profile, "effective_stabilizers",
                                  effectiveSyndromes, physicalWidth);
  auto decomposition = readBinaryRows(profile, "decomposition",
                                      effectiveSyndromes, keptSyndromes);
  auto keptFromEffective = readBinaryRows(profile, "kept_from_effective",
                                          keptSyndromes, effectiveSyndromes);
  if (failed(effective) || failed(decomposition) || failed(keptFromEffective))
    return failure();
  auto metachecksAttr =
      profile->getAttrOfType<DenseIntElementsAttr>("effective_metachecks");
  if (!metachecksAttr || metachecksAttr.getType().getRank() != 2 ||
      !metachecksAttr.getType().getElementType().isInteger(1) ||
      metachecksAttr.getType().getShape()[1] != effectiveSyndromes ||
      metachecksAttr.getType().getShape()[0] !=
          effectiveSyndromes - keptSyndromes) {
    profile.emitOpError(
        "EDEM code presentation requires complete effective_metachecks");
    return failure();
  }

  // Recheck the two identities that make a profile's effective syndrome
  // coordinates meaningful to this codec. The full Fabric verifier also
  // checks rank and metacheck completeness.
  for (unsigned effectiveIndex = 0; effectiveIndex < effectiveSyndromes;
       ++effectiveIndex) {
    llvm::SmallBitVector expanded(physicalWidth);
    for (int keptIndex : (*decomposition)[effectiveIndex].set_bits())
      expanded ^= (*stabilizers)[keptIndex];
    if (expanded != (*effective)[effectiveIndex]) {
      profile.emitOpError("EDEM code presentation decomposition does not "
                          "reproduce an effective check");
      return failure();
    }
  }
  for (unsigned keptIndex = 0; keptIndex < keptSyndromes; ++keptIndex) {
    llvm::SmallBitVector recovered(keptSyndromes);
    for (int effectiveIndex : (*keptFromEffective)[keptIndex].set_bits())
      recovered ^= (*decomposition)[effectiveIndex];
    if (recovered.count() != 1 || !recovered.test(keptIndex)) {
      profile.emitOpError(
          "EDEM code presentation kept_from_effective is not a left inverse");
      return failure();
    }
  }

  CodePresentation result(physicalQubits, logicalQubits);
  result.keptStabilizers_ = std::move(*stabilizers);
  // The binary decomposition omits the sign of a product of Hermitian
  // Paulis (e.g. XX * ZZ = -YY). Keep that affine shift for ideal checks;
  // error-coordinate commutation functionals themselves remain linear.
  for (unsigned index = 0; index < effectiveSyndromes; ++index) {
    llvm::SmallBitVector product(physicalWidth);
    unsigned phase = 0;
    for (int kept : (*decomposition)[index].set_bits()) {
      const auto &factor = result.keptStabilizers_[kept];
      for (unsigned q = 0; q < physicalQubits; ++q) {
        phase += factor[q] && factor[physicalQubits + q];
        phase += 2u * (product[physicalQubits + q] && factor[q]);
      }
      product ^= factor;
    }
    for (unsigned q = 0; q < physicalQubits; ++q)
      phase += 4u - unsigned(product[q] && product[physicalQubits + q]);
    result.effectiveSigns_.push_back((phase & 3u) == 2u);
  }
  result.effectiveToKept_ = std::move(*decomposition);
  result.syndromeInputs_.reserve(effectiveSyndromes);
  result.syndromeFunctionals_.reserve(effectiveSyndromes);
  for (unsigned effectiveIndex = 0; effectiveIndex < effectiveSyndromes;
       ++effectiveIndex) {
    llvm::SmallBitVector representative(physicalWidth);
    for (unsigned keptIndex = 0; keptIndex < keptSyndromes; ++keptIndex)
      if ((*keptFromEffective)[keptIndex].test(effectiveIndex))
        representative ^= (*antiStabilizers)[keptIndex];
    result.syndromeInputs_.push_back(std::move(representative));
    result.syndromeFunctionals_.push_back(
        symplecticFunctional((*effective)[effectiveIndex], physicalQubits));
  }
  result.logicalXInputs_ = std::move(*logicalX);
  result.logicalZInputs_ = std::move(*logicalZ);
  result.logicalXFunctionals_.reserve(logicalQubits);
  result.logicalZFunctionals_.reserve(logicalQubits);
  for (unsigned logical = 0; logical < logicalQubits; ++logical) {
    result.logicalXFunctionals_.push_back(
        symplecticFunctional(result.logicalZInputs_[logical], physicalQubits));
    result.logicalZFunctionals_.push_back(
        symplecticFunctional(result.logicalXInputs_[logical], physicalQubits));
  }
  llvm::SHA256 hash;
  hash.update("cudaq.logical.code-presentation/v1");
  auto number = [&](uint64_t value) {
    uint8_t bytes[8];
    for (unsigned i = 0; i < 8; ++i)
      bytes[i] = value >> (8 * i);
    hash.update(llvm::ArrayRef<uint8_t>(bytes));
  };
  auto matrix = [&](const auto &rows, unsigned columns) {
    number(rows.size());
    number(columns);
    for (const auto &row : rows)
      for (unsigned column = 0; column < columns; ++column) {
        uint8_t value = row[column];
        hash.update(llvm::ArrayRef<uint8_t>(&value, 1));
      }
  };
  number(physicalQubits);
  number(logicalQubits);
  matrix(result.keptStabilizers_, physicalWidth);
  matrix(*antiStabilizers, physicalWidth);
  matrix(result.logicalXInputs_, physicalWidth);
  matrix(result.logicalZInputs_, physicalWidth);
  matrix(*effective, physicalWidth);
  matrix(result.effectiveToKept_, keptSyndromes);
  matrix(*keptFromEffective, effectiveSyndromes);
  result.name_ = "logical-v1/";
  constexpr char hex[] = "0123456789abcdef";
  for (uint8_t byte : hash.final()) {
    result.name_ += hex[byte >> 4];
    result.name_ += hex[byte & 15];
  }
  return result;
}

} // namespace cudaq::logical
