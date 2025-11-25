/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/Registry.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
class RewritePatternSet;
}

namespace cudaq {

//===----------------------------------------------------------------------===//
// Base classes for decomposition patterns
//===----------------------------------------------------------------------===//

/// Base class for pattern types to enable registration via the llvm::Registry
/// system. Stores the pattern metadata and provides a factory method to create
/// new instances of the pattern.
///
/// Register decomposition patterns using
/// CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, MyPatternType,
/// pattern_name)
/// where pattern_name is the same as MyPatternType().getPatternName().
class DecompositionPatternType
    : public registry::RegisteredType<DecompositionPatternType> {
public:
  virtual ~DecompositionPatternType() = default;

  /// Get the source operation this pattern matches and decomposes.
  virtual llvm::StringRef getSourceOp() const = 0;

  /// Get the target operations this pattern may produce
  virtual llvm::ArrayRef<llvm::StringRef> getTargetOps() const = 0;

  /// Get the name of the pattern.
  virtual llvm::StringRef getPatternName() const = 0;

  /// Create a new instance of the pattern.
  virtual std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context,
         mlir::PatternBenefit benefit = 1) const = 0;
};

/// Base class for all decomposition patterns. All decomposition patterns must
/// inherit from this class. Templated on
///  - the pattern type (which inherits from DecompositionPatternType), and
///  - the operation type that the pattern matches.
/// Used as follows class MyPattern : public DecompositionPattern<MyType, Op>
/// {...};
template <typename PatternType, typename Op>
class DecompositionPattern : public mlir::OpRewritePattern<Op> {
public:
  using mlir::OpRewritePattern<Op>::OpRewritePattern;

  /// Set the debug name to the registered name
  void initialize() { this->setDebugName(PatternType().getPatternName()); }
};

void populateWithAllDecompositionPatterns(mlir::RewritePatternSet &patterns);

} // namespace cudaq
