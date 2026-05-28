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

namespace detail {
/// A structured representation of a `"g[<adj>][(<number-of-controls> | `n`)]"`
/// string.
struct OperatorInfo {
  std::string name;
  std::size_t numControls;
  bool isAdj;

  OperatorInfo(llvm::StringRef infoStr);
  OperatorInfo() = default;

  bool operator==(const OperatorInfo &other) const = default;

  bool isUnbounded() const;

  /// Return the join of two `OperatorInfo`s, if it exists.
  ///
  /// Two `OperatorInfo`s can be joined if
  ///  - one of them is empty, in which case the other is returned, or
  ///  - they have the same name and same `isAdj`, and either:
  ///     - they have the same number of controls, in which case `this ==
  ///     either`, or
  //      - at least one is unbounded, in which case the join is whichever has
  //      unbounded controls.
  std::optional<OperatorInfo> join(const OperatorInfo &other) const;

  std::string str() const;

  /// Check if this gate covers another gate.
  bool covers(const OperatorInfo &other) const;

  /// Check if this basis entry makes another gate legal, matching
  /// ConversionTarget semantics. A concrete basis entry does not make an
  /// unbounded source pattern legal.
  bool makesLegal(const OperatorInfo &other) const { return covers(other); }
};
} // namespace detail

//===----------------------------------------------------------------------===//
// Base classes for decomposition patterns
//===----------------------------------------------------------------------===//

struct DecompositionPatternVariant {
  /// Construct from an explicit source op and a list of target ops.
  DecompositionPatternVariant(llvm::StringRef sourceOp,
                              llvm::ArrayRef<llvm::StringRef> targetOps)
      : sourceOp(sourceOp) {
    llvm::transform(
        targetOps, std::back_inserter(this->targetOps),
        [](llvm::StringRef op) { return detail::OperatorInfo(op); });
  }

  DecompositionPatternVariant(std::initializer_list<llvm::StringRef> ops)
      : DecompositionPatternVariant(
            *ops.begin(),
            llvm::ArrayRef<llvm::StringRef>(ops.begin() + 1, ops.end())) {
    assert(ops.size() > 0 && "source op is required");
  }

  detail::OperatorInfo sourceOp;
  std::vector<detail::OperatorInfo> targetOps;
};

/// Base class for pattern types to enable registration via the llvm::Registry
/// system. Stores the pattern metadata and provides a factory method to create
/// new instances of the pattern.
///
/// Use the REGISTER_DECOMPOSITION_PATTERN macro to register a pattern type.
class DecompositionPatternType {
public:
  using RegistryType = llvm::Registry<DecompositionPatternType>;
  DecompositionPatternType(std::vector<DecompositionPatternVariant> variants_)
      : variants(std::move(variants_)) {
    for (const auto &variant : variants) {
      auto join = sourceOp.join(variant.sourceOp);
      assert(join.has_value() &&
             "all source ops of pattern variants must be joinable");
      sourceOp = *join;

      // Join all target ops of variants together. This is quadratic in the
      // number of target ops, but realistically there won't be >10 of those.
      for (const auto &targetOp : variant.targetOps) {
        bool inserted = false;
        for (auto &existingTargetOp : targetOps) {
          auto join = existingTargetOp.join(targetOp);
          if (join.has_value()) {
            existingTargetOp = *join;
            inserted = true;
            break;
          }
        }
        if (!inserted) {
          targetOps.push_back(targetOp);
        }
      }
    }
  }
  virtual ~DecompositionPatternType() = default;

  /// Get the name of the pattern.
  virtual llvm::StringRef getPatternName() const = 0;

  /// Create a new instance of the pattern.
  virtual std::unique_ptr<mlir::RewritePattern>
  create(mlir::MLIRContext *context, mlir::PatternBenefit benefit = 1,
         llvm::ArrayRef<std::string> enabledSourceOps = {}) const = 0;

private:
  std::vector<DecompositionPatternVariant> variants;
  detail::OperatorInfo sourceOp;
  std::vector<detail::OperatorInfo> targetOps;
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
                       llvm::ArrayRef<std::size_t> disabledControlCounts)
      : mlir::OpRewritePattern<Op>(context, benefit),
        disabledControlCounts(disabledControlCounts.begin(),
                              disabledControlCounts.end()) {}

  /// Set the debug name to the registered name
  void initialize() { this->setDebugName(PatternType().getPatternName()); }

protected:
  /// Check if a pattern is enabled for a given number of controls.
  bool isEnabled(std::optional<std::size_t> numControls) const {
    return std::find(
               disabledControlCounts.begin(), disabledControlCounts.end(),
               numControls.value_or(std::numeric_limits<std::size_t>::max())) ==
           disabledControlCounts.end();
  }

private:
  std::vector<std::size_t> disabledControlCounts;
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
