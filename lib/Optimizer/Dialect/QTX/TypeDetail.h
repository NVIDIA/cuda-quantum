/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "cudaq/Optimizer/Dialect/QTX/QTXDialect.h"
#include "cudaq/Optimizer/Dialect/QTX/QTXTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeRange.h"

//===----------------------------------------------------------------------===//
// OperatorType Storage
//===----------------------------------------------------------------------===//

namespace qtx {
namespace detail {

/// OperatorType storage and uniquing.
struct OperatorTypeStorage : public mlir::TypeStorage {

  OperatorTypeStorage(unsigned numParameters, unsigned numTargets,
                      unsigned numClassicalResults, unsigned numTargetResults,
                      mlir::Type const *typeList)
      : numParameters(numParameters), numTargets(numTargets),
        numClassicalResults(numClassicalResults),
        numTargetResults(numTargetResults), typeList(typeList) {}

  // The hash key used for uniquing.
  using KeyTy =
      std::tuple<unsigned, unsigned, unsigned, unsigned, mlir::TypeRange>;

  bool operator==(const KeyTy &key) const {
    return key == KeyTy(numParameters, numTargets, numClassicalResults,
                        numTargetResults, getAllTypes());
  }

  static OperatorTypeStorage *construct(mlir::TypeStorageAllocator &allocator,
                                        const KeyTy &key) {
    auto [nParameters, nTargets, nClassicalResults, nTargetResults, typeRange] =
        key;
    assert(nParameters + nTargets + nClassicalResults + nTargetResults ==
           typeRange.size());

    // TODO: Find a well-motivated choice for the number of inlined elements for
    // this SmallVector
    mlir::SmallVector<mlir::Type> types;
    types.reserve(typeRange.size());
    types.append(typeRange.begin(), typeRange.end());
    auto typesArray = allocator.copyInto(mlir::ArrayRef<mlir::Type>(types));
    return new (allocator.allocate<OperatorTypeStorage>())
        OperatorTypeStorage(nParameters, nTargets, nClassicalResults,
                            nTargetResults, typesArray.data());
  }

  mlir::ArrayRef<mlir::Type> getParameters() const {
    return mlir::ArrayRef<mlir::Type>(typeList, numParameters);
  }

  mlir::ArrayRef<mlir::Type> getTargets() const {
    const unsigned begin = numParameters;
    return mlir::ArrayRef<mlir::Type>(typeList + begin, numTargets);
  }

  mlir::ArrayRef<mlir::Type> getClassicalResults() const {
    const unsigned begin = numParameters + numTargets;
    return mlir::ArrayRef<mlir::Type>(typeList + begin, numClassicalResults);
  }

  mlir::ArrayRef<mlir::Type> getTargetResults() const {
    const unsigned begin = numParameters + numTargets + numClassicalResults;
    return mlir::ArrayRef<mlir::Type>(typeList + begin, numTargetResults);
  }

  mlir::ArrayRef<mlir::Type> getResults() const {
    const unsigned begin = numParameters + numTargets;
    return mlir::ArrayRef<mlir::Type>(typeList + begin,
                                      numClassicalResults + numTargetResults);
  }

  mlir::ArrayRef<mlir::Type> getAllTypes() const {
    const unsigned length = numTypes();
    return mlir::ArrayRef<mlir::Type>(typeList, length);
  }

  unsigned numTypes() const {
    return numParameters + numTargets + numClassicalResults + numTargetResults;
  }

  unsigned numParameters;
  unsigned numTargets;
  unsigned numClassicalResults;
  unsigned numTargetResults;
  mlir::Type const *typeList;
};

} // namespace detail
} // namespace qtx
