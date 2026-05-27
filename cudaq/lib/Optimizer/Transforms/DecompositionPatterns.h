/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeInterfaces.h"
#define LLVM_DISABLE_ABI_BREAKING_CHECKS_ENFORCING 1
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Registry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <iterator>
#include <optional>
#include <string>
#include <vector>

namespace mlir {
class RewritePatternSet;
}

namespace cudaq {

//===----------------------------------------------------------------------===//
// Base classes for decomposition patterns
//===----------------------------------------------------------------------===//

struct DecompositionPatternVariant {
  DecompositionPatternVariant(llvm::StringRef sourceOp,
                              llvm::ArrayRef<llvm::StringRef> targetOps)
      : sourceOp(sourceOp.str()) {
    llvm::transform(targetOps, std::back_inserter(this->targetOps),
                    [](llvm::StringRef op) { return op.str(); });
  }

  DecompositionPatternVariant(std::initializer_list<llvm::StringRef> ops)
      : DecompositionPatternVariant(
            *ops.begin(),
            llvm::ArrayRef<llvm::StringRef>(ops.begin() + 1, ops.end())) {
    assert(ops.size() > 0 && "source op is required");
  }

  std::string sourceOp;
  std::vector<std::string> targetOps;
};

/// Base class for pattern types to enable registration via the llvm::Registry
/// system. Stores the pattern metadata and provides a factory method to create
/// new instances of the pattern.
///
/// Register decomposition patterns using
/// CUDAQ_REGISTER_TYPE(cudaq::DecompositionPatternType, MyPatternType,
/// pattern_name)
/// where pattern_name is the same as MyPatternType().getPatternName().
class DecompositionPatternType {
public:
  using RegistryType = llvm::Registry<DecompositionPatternType>;
  virtual ~DecompositionPatternType() = default;

  /// Get the source/target variants this pattern can implement. Source variants
  /// use `g(n)` for controlled forms only; add a separate `g` source variant
  /// when a bare operation should also rewrite.
  virtual llvm::ArrayRef<DecompositionPatternVariant> getVariants() const = 0;

  /// Get the name of the pattern.
  virtual llvm::StringRef getPatternName() const = 0;

  /// Create a new instance of the pattern.
  virtual std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1,
         llvm::ArrayRef<std::string> enabledSourceOps = {}) const = 0;
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
  DecompositionPattern(mlir::MLIRContext *context,
                       mlir::PatternBenefit benefit = 1)
      : mlir::OpRewritePattern<Op>(context, benefit) {}

  DecompositionPattern(mlir::MLIRContext *context, mlir::PatternBenefit benefit,
                       llvm::ArrayRef<std::string> enabledSourceOps)
      : mlir::OpRewritePattern<Op>(context, benefit),
        enabledSourceOps(enabledSourceOps.begin(), enabledSourceOps.end()) {}

  /// Set the debug name to the registered name
  void initialize() { this->setDebugName(PatternType().getPatternName()); }

protected:
  bool sourceOpEnabled(llvm::StringRef sourceOp) const {
    return enabledSourceOps.empty() ||
           llvm::is_contained(enabledSourceOps, sourceOp);
  }

  /// Check selected source variants for a possibly controlled gate.
  /// Patterns with source variants such as `g`, `g(1)`, and `g(n)` should use
  /// this helper so `matchAndRewrite` follows the decomposition graph result.
  /// Source `g(n)` means controlled forms of any arity. It intentionally does
  /// not match bare `g`.
  bool sourceOpEnabled(llvm::StringRef bareSourceOp,
                       std::optional<std::size_t> numControls) const {
    if (numControls.has_value() && *numControls == 0)
      return sourceOpEnabled(bareSourceOp);

    std::string sourcePrefix = bareSourceOp.str();
    std::string unboundedSource = sourcePrefix + "(n)";
    if (!numControls)
      return sourceOpEnabled(unboundedSource);

    if (*numControls == 0)
      return sourceOpEnabled(bareSourceOp);

    std::string concreteSource =
        sourcePrefix + "(" + std::to_string(*numControls) + ")";
    return sourceOpEnabled(unboundedSource) || sourceOpEnabled(concreteSource);
  }

private:
  std::vector<std::string> enabledSourceOps;
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
/// cudaq::quake::QuakeDialect that appear in `targetBasis`.
std::unique_ptr<mlir::ConversionTarget>
createBasisTarget(mlir::MLIRContext &context,
                  mlir::ArrayRef<std::string> targetBasis);

/// Get the number of controls to the operation, if known.
///
/// The size of the control vector may be unspecified, in which case this
/// function returns std::nullopt.
std::optional<std::size_t>
getKnownNumControls(cudaq::quake::OperatorInterface op);

using DecompositionPatternTypeRegistry =
    llvm::Registry<DecompositionPatternType>;
} // namespace cudaq

/// Register a decomposition pattern type with the LLVM registry.
/// This is compiler-internal only (no cross-DSO / Python concerns).
#define REGISTER_DECOMPOSITION_PATTERN(SUBTYPE, NAME)                          \
  static cudaq::DecompositionPatternType::RegistryType::Add<SUBTYPE>           \
      decomp_reg_##NAME(#NAME, "");
