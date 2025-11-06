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
#include "mlir/Transforms/DialectConversion.h"
#include <string>

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

/// Select subset of patterns relevant to decomposing to the given target basis.
///
/// The result of the pattern selection are cached, so that successive calls
/// with the same arguments will be O(1).
///
/// @param patterns The pattern set to add the selected patterns to
/// @param basisGates The basis gates to decompose to
/// @param disabledPatterns The patterns to disable
///
/// A subset of the decomposition patterns is selected such that:
/// - for every gate that can be decomposed to the target basis, the sequence of
///   decomposition to the target basis is unique.
/// - when more than one decomposition would exist, the one that requires the
///   fewest applications of patterns is chosen.
/// - `disabledPatterns` are never selected
void selectDecompositionPatterns(mlir::RewritePatternSet &patterns,
                                 llvm::ArrayRef<std::string> targetBasis,
                                 llvm::ArrayRef<std::string> disabledPatterns);

void populateWithAllDecompositionPatterns(mlir::RewritePatternSet &patterns);

/// Create a conversion target parsed from a target basis string.
///
/// The `targetBasis` should be made of strings of the form:
///
/// ```
/// <op-name>(`(` [<number-of-controls> | `n`] `)` )?
/// ```
///
/// The returned conversion target will accept operations in the MLIR dialects
/// arith::ArithDialect, cf::ControlFlowDialect, cudaq::cc::CCDialect,
/// func::FuncDialect, and math::MathDialect, as well as operations in the
/// quake::QuakeDialect that appear in `targetBasis`.
std::unique_ptr<mlir::ConversionTarget>
createBasisTarget(mlir::MLIRContext &context,
                  mlir::ArrayRef<std::string> targetBasis);

} // namespace cudaq
