/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
///
/// This file declares the classes essential for converting Quake to QTX,
/// namely:
///    * A highly specialized pattern rewriter, `ConvertToQTXRewriter`.
///
///    * A pattern base class for operation conversion, `ConvertToQTXPattern`.
///
///    * A wrapper around the base class that allows matching and rewriting
///      against an instance of a derived operation class rather than a raw
///      operation, `ConvertOpToQTXPattern`.
///
/// The conversion from Quake to QTX involves building the SSA form for quantum
/// operations; that is, Quake operations, which do not consume and return new
/// values, will be converted into QTX operations that do.
///
/// Example: Take the following Quake operation.
/// ```
///   quake.x [%q0] %q1 : [!quake.qref] !quake.qref
/// ```
///
/// In QTX, this operation has the form:
/// ```
///   %q1_w1 = qtx.x [%q0_w0] %q1_w0 : [!qtx.wire] !qtx.wire
/// ```
///
/// For more information on how Quake quantum values relate to QTX ones, see
/// QTX's dialect documentation.  Here it suffices to understand that Quake
/// quantum values "become" a chain of QTX quantum values.  For `%q0` in the
/// above example, we have:
/// ```
///   Quake            QTX
///    %q1      [%q1_w0 -> %q1_w1]
/// ```
///
/// Note that each QTX value is only live until its consumption, i.e., until
/// when it is used as a target of a quantum operation. After that, a new QTX
/// value is added to the chain, becoming the live QTX value corresponding to
/// the Quake value.  For example, after the above `quake.x` operation, any
/// subsequent operation using `%q1` converts to a QTX operation using `%q0_w1`.
///
#pragma once
#include "mlir/Rewrite/FrozenRewritePatternSet.h"

namespace cudaq {

namespace detail {
struct ConvertToQTXRewriterImpl;
} // namespace detail

/// This class implements a highly specialized pattern rewriter for use with
/// `ConvertToQTXPattern`s.
///
/// This rewriter visits each operation in a kernel in post order and only once,
/// which means that an operation that has regions will be visit after its
/// regions. Example:
/// ```
/// quake.foo          visiting order: (1) quake.foo
/// op {                               (2) quake.bar
///   quake.bar                        (3) quake.baz
///   quake.baz                        (4) op {}
/// }
/// ```
///
/// The rewriter keeps track of which is the live QTX value that a Quake value
/// corresponds to. A user can query this information using the `getRemapped`
/// method or update the mapping using `mapOrRemap.`
///
/// **Important note:  The user of this class is responsible for updating the
/// mapping between Quake and QTX values.** In practice, the rewriting patterns
/// must use the `mapOrRemap` method whenever they create a new QTX operation
/// that consumes and returns new values.
///
/// Also note that the mapping of a Quake value to a QTX value depends on the
/// current block (hence the user's need to pass the operation that requires
/// this information).
class ConvertToQTXRewriter final : public mlir::PatternRewriter {
public:
  explicit ConvertToQTXRewriter(mlir::MLIRContext *context);

  ~ConvertToQTXRewriter() override;

  //===--------------------------------------------------------------------===//
  // PatternRewriter Hooks
  //===--------------------------------------------------------------------===//

  /// Hook for inserting operations, and make sure that newly inserted ops are
  /// tracked.
  void notifyOperationInserted(mlir::Operation *op) override;

  /// PatternRewriter hook creating a new block.
  void notifyBlockCreated(mlir::Block *block) override;

  /// PatternRewriter hook for moving blocks out of a region.  Moves the blocks
  /// belonging to "region" before the given position in another region.  The
  /// two regions must be different.  The caller is in charge to update create
  /// the operation transferring the control flow to the region and pass it the
  /// correct block arguments.
  void inlineRegionBefore(mlir::Region &region, mlir::Region &parent,
                          mlir::Region::iterator before) override;
  using PatternRewriter::inlineRegionBefore;

  /// This method erases an operation that is known to have no uses.
  void eraseOp(mlir::Operation *op) override;

  // The in-place update functions must be used with extreme care, and, most
  // likely, only for updating operations that use the classical results of
  // measurements.

  void startRootUpdate(mlir::Operation *op) override;

  void cancelRootUpdate(mlir::Operation *op) override;

  /// Adds an argument to a block.
  mlir::BlockArgument addArgument(mlir::Block *block, mlir::Type type,
                                  mlir::Location loc);

  // Given the requirements of this rewriter, the following operations are not
  // allowed.
  void eraseBlock(mlir::Block *block) override {
    llvm_unreachable("cannot use this method");
  };

  mlir::Block *splitBlock(mlir::Block *block,
                          mlir::Block::iterator before) override {
    llvm_unreachable("cannot use this method");
  }

  template <typename OpTy, typename... Args>
  OpTy replaceOpWithNewOp(mlir::Operation *op, Args &&...args) = delete;

  void replaceOpWithinBlock(mlir::Operation *op, mlir::ValueRange newValues,
                            mlir::Block *block,
                            bool *allUsesReplaced = nullptr) = delete;

  //===--------------------------------------------------------------------===//
  // Value (re)mapping and type conversion
  //===--------------------------------------------------------------------===//

  /// 1-1 type conversions. This function returns the type to convert to on
  /// success, and a null type on failure.
  mlir::Type convertType(mlir::Type type);

  /// Map or remap an `oldValue` to a `newValue`.
  void mapOrRemap(mlir::Value oldValue, mlir::Value newValue);

  /// Return the current mapping for `oldValue`.
  mlir::Value getRemapped(mlir::Operation *op, mlir::Value oldValue);

  /// Remap the given set of `oldValues`, filling 'remapped' as necessary. This
  /// returns failure if the remapping of any of the value fails, success
  /// otherwise.
  void getRemapped(mlir::Operation *op, mlir::ValueRange oldValues,
                   mlir::SmallVectorImpl<mlir::Value> &remapped);

  //===--------------------------------------------------------------------===//

  /// Return a reference to the internal implementation.
  detail::ConvertToQTXRewriterImpl &getImpl();

private:
  std::unique_ptr<detail::ConvertToQTXRewriterImpl> impl;
};

/// Base class for operation conversions targeting the QTX dialect. This pattern
/// class enables type conversions and other uses specific to the conversion
/// between Quake to QTX.
class ConvertToQTXPattern : public mlir::RewritePattern {
public:
  /// Hook for derived classes to implement rewriting.
  ///   * `op` is the (first) operation matched by the pattern,
  ///
  ///   * `implicitUsedQuantumValues` is a list of quantum values that cannot
  ///      be inferred from the operation's operands, e.g., this is case for
  ///      operations that have regions and terminators.
  ///
  ///   * `operands` is a list of the remapped operand values,
  ///
  /// The `rewriter` can be used to emit the new operations. This function
  /// should not fail.  If some specific cases of the operation are not
  /// supported, these cases should not be matched.
  virtual void rewrite(mlir::Operation *op,
                       mlir::ArrayRef<mlir::Value> implicitUsedQuantumValues,
                       mlir::ArrayRef<mlir::Value> operands,
                       ConvertToQTXRewriter &rewriter) const {
    llvm_unreachable("unimplemented rewrite");
  }

  /// Hook for derived classes to implement matchAndRewrite.
  virtual mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::ArrayRef<mlir::Value> implicitUsedQuantumValues,
                  mlir::ArrayRef<mlir::Value> operands,
                  ConvertToQTXRewriter &rewriter) const {
    if (failed(match(op)))
      return mlir::failure();
    rewrite(op, implicitUsedQuantumValues, operands, rewriter);
    return mlir::success();
  }

  // Attempt to match and rewrite the IR root at the specified operation.
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const final;

protected:
  /// See `RewritePattern::RewritePattern` for information on the other
  /// available constructors.
  using RewritePattern::RewritePattern;

private:
  using RewritePattern::rewrite;
};

/// This a wrapper around `ConvertToQTXPattern` that allows for matching and
/// rewriting against an instance of a derived operation class as opposed to a
/// raw `mlir::Operation`.
template <typename SourceOp>
class ConvertOpToQTXPattern : public ConvertToQTXPattern {
public:
  using OpAdaptor = typename SourceOp::Adaptor;

  ConvertOpToQTXPattern(mlir::MLIRContext *context,
                        mlir::PatternBenefit benefit = 1)
      : ConvertToQTXPattern(SourceOp::getOperationName(), benefit, context) {}

  // Wrappers around the `ConvertToQTXPattern` methods that pass the derived op
  // type.

  mlir::LogicalResult match(mlir::Operation *op) const final {
    return match(cast<SourceOp>(op));
  }

  void rewrite(mlir::Operation *op,
               mlir::ArrayRef<mlir::Value> implicitUsedQuantumValues,
               mlir::ArrayRef<mlir::Value> operands,
               ConvertToQTXRewriter &rewriter) const final {
    rewrite(cast<SourceOp>(op), implicitUsedQuantumValues,
            OpAdaptor(operands, op->getAttrDictionary()), rewriter);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::ArrayRef<mlir::Value> implicitUsedQuantumValues,
                  mlir::ArrayRef<mlir::Value> operands,
                  ConvertToQTXRewriter &rewriter) const final {
    return matchAndRewrite(cast<SourceOp>(op), implicitUsedQuantumValues,
                           OpAdaptor(operands, op->getAttrDictionary()),
                           rewriter);
  }

  // Rewrite and Match methods that operate on the SourceOp type. **Either
  // these or the `matchAndRewrite` must be overridden by the derived pattern
  // class.**

  virtual mlir::LogicalResult match(SourceOp op) const {
    llvm_unreachable("must override match or matchAndRewrite");
  }

  virtual void rewrite(SourceOp op, mlir::ArrayRef<mlir::Value> results,
                       OpAdaptor adaptor,
                       ConvertToQTXRewriter &rewriter) const {
    llvm_unreachable("must override matchAndRewrite or a rewrite method");
  }

  virtual mlir::LogicalResult
  matchAndRewrite(SourceOp op, mlir::ArrayRef<mlir::Value> results,
                  OpAdaptor adaptor, ConvertToQTXRewriter &rewriter) const {
    if (failed(match(op)))
      return mlir::failure();
    rewrite(op, results, adaptor, rewriter);
    return mlir::success();
  }

private:
  using RewritePattern::matchAndRewrite;
};

//===----------------------------------------------------------------------===//
// Conversion entry points
//===----------------------------------------------------------------------===//

mlir::LogicalResult applyPartialQuakeToQTXConversion(
    mlir::Operation *op, const mlir::FrozenRewritePatternSet &patterns,
    mlir::function_ref<void(mlir::Diagnostic &)> notifyCallback = nullptr);

} // namespace cudaq
