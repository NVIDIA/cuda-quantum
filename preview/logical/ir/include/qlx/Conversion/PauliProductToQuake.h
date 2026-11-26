/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

//===----------------------------------------------------------------------===//
// Pauli-product glue for lowering Logical ops to Quake.
//
// Most Logical-to-Quake conversion is one op to one op. Pauli-product
// measurements and rotations are not: Quake has no multi-qubit Pauli
// measurement, and `quake.exp_pauli` has no wire-form decomposition in CUDA-Q
// core (DecompositionPatterns.cpp declines it). This header captures the glue.
//
// There are two layers, both header-only and both free of Logical *types*:
//
//  1. Builders (`basisChange`, `measureProduct`, `rotateProduct`). They take
//     `!quake.wire` values and return new `!quake.wire` values, so every qubit
//     stays a linear value: nothing is rewritten in place, no uses are
//     replaced. Only Quake and arith ops are created.
//
//  2. Conversion-pattern templates (`MeasureProductToQuake`,
//     `RotateProductToQuake`). The Logical op, and which Logical type plays the
//     part of "a qubit as a wire", are template parameters supplied through a
//     `Traits` struct, so each stage (P2 patches, P3 physical states, ...)
//     picks its own mapping without this file naming any Logical dialect.
//
// Conventions, all taken from CUDA-Q core's own `ExpPauliDecomposition`:
//  * Basis change into Z: X -> H, Y -> Rx(pi/2); undone by H and Rx(-pi/2).
//  * Parity is gathered onto the last non-identity qubit with a CX chain.
//  * `rotateProduct` implements R_P(angle) = exp(-i * angle * P / 2); in core's
//    `exp(i*theta*P)` terms that is theta = -angle/2, i.e. `Rz(angle)` between
//    the basis change and its inverse.
//
// This is a sketch of a path from Logical to Quake, not a change to the Logical
// dialects. It is only usable when CUDA-Q's Quake dialect is available.
//===----------------------------------------------------------------------===//

#ifndef QLX_CONVERSION_PAULIPRODUCTTOQUAKE_H
#define QLX_CONVERSION_PAULIPRODUCTTOQUAKE_H

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cmath>

namespace qlx::pauli_quake {

enum class Pauli { I, X, Y, Z };

/// A Pauli word with an overall sign. `negated` means the product is -P.
struct PauliProduct {
  bool negated = false;
  llvm::SmallVector<Pauli> paulis;
};

inline mlir::FailureOr<Pauli> parsePauli(llvm::StringRef text) {
  if (text == "I")
    return Pauli::I;
  if (text == "X")
    return Pauli::X;
  if (text == "Y")
    return Pauli::Y;
  if (text == "Z")
    return Pauli::Z;
  return mlir::failure();
}

/// Parse `["X", "Z", ...]`, the P3 `paulis` attribute form.
inline mlir::FailureOr<llvm::SmallVector<Pauli>>
parsePaulis(mlir::ArrayAttr attr) {
  llvm::SmallVector<Pauli> result;
  for (mlir::Attribute element : attr) {
    auto text = mlir::dyn_cast<mlir::StringAttr>(element);
    if (!text)
      return mlir::failure();
    auto pauli = parsePauli(text.getValue());
    if (mlir::failed(pauli))
      return mlir::failure();
    result.push_back(*pauli);
  }
  return result;
}

/// Parse `"ZZ"` or `"-XZ"`, the P2 `pauli_product` string form.
inline mlir::FailureOr<PauliProduct> parsePauliProduct(llvm::StringRef text) {
  PauliProduct product;
  if (text.consume_front("-"))
    product.negated = true;
  else
    text.consume_front("+");
  for (char c : text) {
    auto pauli = parsePauli(llvm::StringRef(&c, 1));
    if (mlir::failed(pauli))
      return mlir::failure();
    product.paulis.push_back(*pauli);
  }
  return product;
}

namespace detail {

inline mlir::Type wireType(mlir::OpBuilder &builder) {
  return cudaq::quake::WireType::get(builder.getContext());
}

inline mlir::Value f64(mlir::OpBuilder &builder, mlir::Location loc,
                       double value) {
  return mlir::arith::ConstantFloatOp::create(
      builder, loc, builder.getF64Type(), llvm::APFloat(value));
}

/// One-wire gate `OpTy` with optional parameters; returns the new wire.
template <typename OpTy>
inline mlir::Value gate(mlir::OpBuilder &builder, mlir::Location loc,
                        mlir::Value wire, mlir::ValueRange params = {}) {
  return OpTy::create(builder, loc, mlir::TypeRange{wireType(builder)},
                      /*is_adj=*/false, params, mlir::ValueRange{},
                      mlir::ValueRange{wire}, mlir::DenseBoolArrayAttr{})
      .getResult(0);
}

/// Indices of the non-identity entries.
inline llvm::SmallVector<unsigned> support(llvm::ArrayRef<Pauli> paulis) {
  llvm::SmallVector<unsigned> indices;
  for (auto [i, pauli] : llvm::enumerate(paulis))
    if (pauli != Pauli::I)
      indices.push_back(i);
  return indices;
}

/// CX chain over the support, `s[0] -> s[1] -> ... -> s[k-1]`, so the parity of
/// the support ends up on the last wire. `undo` runs it in reverse.
inline void parityChain(mlir::OpBuilder &builder, mlir::Location loc,
                        llvm::SmallVectorImpl<mlir::Value> &wires,
                        llvm::ArrayRef<unsigned> sup, bool undo) {
  auto cx = [&](unsigned i) {
    auto op = cudaq::quake::XOp::create(
        builder, loc, mlir::TypeRange{wireType(builder), wireType(builder)},
        /*is_adj=*/false, mlir::ValueRange{}, mlir::ValueRange{wires[sup[i]]},
        mlir::ValueRange{wires[sup[i + 1]]}, mlir::DenseBoolArrayAttr{});
    wires[sup[i]] = op.getResult(0);
    wires[sup[i + 1]] = op.getResult(1);
  };
  if (sup.size() < 2)
    return;
  if (!undo) {
    for (unsigned i = 0; i + 1 < sup.size(); ++i)
      cx(i);
  } else {
    for (unsigned i = sup.size() - 1; i-- > 0;)
      cx(i);
  }
}

} // namespace detail

/// Rotate `wire` so that Pauli `p` becomes Z (`undo == false`), or the inverse
/// (`undo == true`). Z and I need nothing. Returns the new wire.
inline mlir::Value basisChange(mlir::OpBuilder &builder, mlir::Location loc,
                               mlir::Value wire, Pauli p, bool undo) {
  switch (p) {
  case Pauli::X:
    return detail::gate<cudaq::quake::HOp>(builder, loc, wire);
  case Pauli::Y:
    return detail::gate<cudaq::quake::RxOp>(
        builder, loc, wire, detail::f64(builder, loc, undo ? -M_PI_2 : M_PI_2));
  case Pauli::I:
  case Pauli::Z:
    return wire;
  }
  return wire;
}

/// Apply the basis changes for every non-identity qubit, in place in `wires`.
inline void basisChangeAll(mlir::OpBuilder &builder, mlir::Location loc,
                           llvm::SmallVectorImpl<mlir::Value> &wires,
                           llvm::ArrayRef<Pauli> paulis, bool undo) {
  for (auto [i, pauli] : llvm::enumerate(paulis))
    wires[i] = basisChange(builder, loc, wires[i], pauli, undo);
}

struct MeasureResult {
  /// The input qubits, in order, after the measurement. Non-destructive.
  llvm::SmallVector<mlir::Value> wires;
  /// The `i1` outcome.
  mlir::Value bit;
};

/// Measure the Pauli product non-destructively: the other qubits keep their
/// post-measurement state. Measuring each qubit and combining the bits would
/// give the same bit but destroy that state, so the parity is instead gathered
/// onto one qubit, measured in Z, and the circuit uncomputed.
///
/// Fails if every Pauli is identity (there is nothing to measure) or the sizes
/// disagree. `invert` complements the bit (a negated product).
inline mlir::FailureOr<MeasureResult>
measureProduct(mlir::OpBuilder &builder, mlir::Location loc,
               mlir::ValueRange wires, llvm::ArrayRef<Pauli> paulis,
               bool invert, mlir::StringAttr registerName = {}) {
  if (wires.size() != paulis.size())
    return mlir::failure();
  llvm::SmallVector<unsigned> sup = detail::support(paulis);
  if (sup.empty())
    return mlir::failure();

  MeasureResult result;
  result.wires.assign(wires.begin(), wires.end());
  basisChangeAll(builder, loc, result.wires, paulis, /*undo=*/false);
  detail::parityChain(builder, loc, result.wires, sup, /*undo=*/false);

  auto measure = cudaq::quake::MzOp::create(
      builder, loc,
      mlir::TypeRange{cudaq::quake::MeasureType::get(builder.getContext()),
                      detail::wireType(builder)},
      mlir::ValueRange{result.wires[sup.back()]}, registerName);
  result.wires[sup.back()] = measure.getWires()[0];

  detail::parityChain(builder, loc, result.wires, sup, /*undo=*/true);
  basisChangeAll(builder, loc, result.wires, paulis, /*undo=*/true);

  result.bit = cudaq::quake::DiscriminateOp::create(
      builder, loc, builder.getI1Type(), measure.getMeasOut());
  if (invert)
    result.bit = mlir::arith::XOrIOp::create(
        builder, loc, result.bit,
        mlir::arith::ConstantIntOp::create(builder, loc, 1, 1));
  return result;
}

enum class RotationStyle {
  /// basis change, CX chain, `Rz(angle)`, undo. Quake only, wire-clean.
  Decomposed,
  /// A single `quake.exp_pauli`. Core cannot decompose its wire form yet.
  ExpPauli,
};

/// Apply R_P(angle) = exp(-i * angle * P / 2) (`angle` is an `f64` value) to
/// `wires`. All-identity products are the identity and return `wires` as is.
/// Fails only if the sizes disagree.
inline mlir::FailureOr<llvm::SmallVector<mlir::Value>>
rotateProduct(mlir::OpBuilder &builder, mlir::Location loc,
              mlir::ValueRange wires, llvm::ArrayRef<Pauli> paulis,
              mlir::Value angle,
              RotationStyle style = RotationStyle::Decomposed) {
  if (wires.size() != paulis.size())
    return mlir::failure();
  llvm::SmallVector<mlir::Value> out(wires.begin(), wires.end());
  llvm::SmallVector<unsigned> sup = detail::support(paulis);
  if (sup.empty())
    return out;

  if (style == RotationStyle::ExpPauli) {
    // exp_pauli computes exp(i*theta*P); R_P(angle) needs theta = -angle/2.
    mlir::Value theta = mlir::arith::MulFOp::create(
        builder, loc, angle, detail::f64(builder, loc, -0.5));
    std::string word;
    for (Pauli pauli : paulis)
      word += (pauli == Pauli::X   ? 'X'
               : pauli == Pauli::Y ? 'Y'
               : pauli == Pauli::Z ? 'Z'
                                   : 'I');
    llvm::SmallVector<mlir::Type> types(out.size(), detail::wireType(builder));
    auto op = cudaq::quake::ExpPauliOp::create(
        builder, loc, mlir::TypeRange{types}, /*is_adj=*/mlir::UnitAttr{},
        mlir::ValueRange{theta}, mlir::ValueRange{}, mlir::ValueRange{out},
        mlir::DenseBoolArrayAttr{}, /*pauli=*/mlir::Value{},
        builder.getStringAttr(word));
    for (auto [i, result] : llvm::enumerate(op.getResults()))
      out[i] = result;
    return out;
  }

  basisChangeAll(builder, loc, out, paulis, /*undo=*/false);
  detail::parityChain(builder, loc, out, sup, /*undo=*/false);
  out[sup.back()] = detail::gate<cudaq::quake::RzOp>(
      builder, loc, out[sup.back()], mlir::ValueRange{angle});
  detail::parityChain(builder, loc, out, sup, /*undo=*/true);
  basisChangeAll(builder, loc, out, paulis, /*undo=*/true);
  return out;
}

//===----------------------------------------------------------------------===//
// Conversion-pattern templates
//===----------------------------------------------------------------------===//
//
// `Traits` is the only place a Logical type is named. It decides what "a qubit
// as a wire" means for `SourceOp`, typically via the pass's TypeConverter.
//
// Required for MeasureProductToQuake<SourceOp, Traits>:
//   static FailureOr<PauliProduct> product(SourceOp op);
//   static FailureOr<SmallVector<Value>> qubits(
//       SourceOp op, typename SourceOp::Adaptor adaptor,
//       ConversionPatternRewriter &rewriter);       // already `!quake.wire`
//   static StringAttr registerName(SourceOp op);    // may be null
//   static void replace(SourceOp op, ConversionPatternRewriter &rewriter,
//                       ValueRange wires, Value bit);
//
// Required for RotateProductToQuake<SourceOp, Traits>:
//   product(), qubits() as above;
//   static Value angle(SourceOp op, ConversionPatternRewriter &rewriter); //
//   f64 static void replace(SourceOp op, ConversionPatternRewriter &rewriter,
//                       ValueRange wires);
//
// The pattern owns the sign rule: a negated product P' = -P gives
// R_{P'}(angle) = R_P(-angle) and complements a measured bit.

template <typename SourceOp, typename Traits>
struct MeasureProductToQuake : mlir::OpConversionPattern<SourceOp> {
  using mlir::OpConversionPattern<SourceOp>::OpConversionPattern;

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto product = Traits::product(op);
    if (mlir::failed(product))
      return rewriter.notifyMatchFailure(op, "unrecognized Pauli product");
    auto wires = Traits::qubits(op, adaptor, rewriter);
    if (mlir::failed(wires))
      return rewriter.notifyMatchFailure(op, "operands are not wires");
    auto result = measureProduct(rewriter, op.getLoc(), *wires, product->paulis,
                                 product->negated, Traits::registerName(op));
    if (mlir::failed(result))
      return rewriter.notifyMatchFailure(
          op, "product has no non-identity Pauli or mismatched arity");
    Traits::replace(op, rewriter, result->wires, result->bit);
    return mlir::success();
  }
};

template <typename SourceOp, typename Traits>
struct RotateProductToQuake : mlir::OpConversionPattern<SourceOp> {
  RotateProductToQuake(mlir::MLIRContext *context,
                       RotationStyle style = RotationStyle::Decomposed)
      : mlir::OpConversionPattern<SourceOp>(context), style(style) {}
  RotateProductToQuake(const mlir::TypeConverter &converter,
                       mlir::MLIRContext *context,
                       RotationStyle style = RotationStyle::Decomposed)
      : mlir::OpConversionPattern<SourceOp>(converter, context), style(style) {}

  mlir::LogicalResult
  matchAndRewrite(SourceOp op, typename SourceOp::Adaptor adaptor,
                  mlir::ConversionPatternRewriter &rewriter) const override {
    auto product = Traits::product(op);
    if (mlir::failed(product))
      return rewriter.notifyMatchFailure(op, "unrecognized Pauli product");
    auto wires = Traits::qubits(op, adaptor, rewriter);
    if (mlir::failed(wires))
      return rewriter.notifyMatchFailure(op, "operands are not wires");
    mlir::Value angle = Traits::angle(op, rewriter);
    if (product->negated)
      angle = mlir::arith::NegFOp::create(rewriter, op.getLoc(), angle);
    auto result = rotateProduct(rewriter, op.getLoc(), *wires, product->paulis,
                                angle, style);
    if (mlir::failed(result))
      return rewriter.notifyMatchFailure(op, "mismatched Pauli arity");
    Traits::replace(op, rewriter, *result);
    return mlir::success();
  }

private:
  RotationStyle style;
};

} // namespace qlx::pauli_quake

#endif // QLX_CONVERSION_PAULIPRODUCTTOQUAKE_H
