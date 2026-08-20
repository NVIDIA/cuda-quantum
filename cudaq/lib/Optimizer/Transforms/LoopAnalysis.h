/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"

namespace cudaq::opt {

// Loops that are transformed into normal form have this attribute.
static constexpr char NormalizedLoopAttr[] = "normalized";
static constexpr char DeadLoopAttr[] = "dead";

struct LoopComponents {
  LoopComponents() = default;

  // Get the induction expression of the comparison.
  mlir::Value getCompareInduction() const;

  bool stepIsAnAddOp() const;
  bool shouldCommuteStepOp() const;
  bool isClosedIntervalForm() const;
  bool isLinearExpr() const;
  std::optional<std::size_t> getIterationsConstant() const;

  // Determine if the condition is always true. e.g., `x uge 0`.
  bool hasAlwaysTrueCondition() const;
  // Determine if the condition is always false. e.g., `x ult 0`.
  bool hasAlwaysFalseCondition() const;
  bool hasInvariantCondition() const {
    return hasAlwaysTrueCondition() || hasAlwaysFalseCondition();
  }

  std::int64_t extendValue(unsigned width, std::size_t val) const;

  std::optional<unsigned> induction = std::nullopt;
  mlir::Value initialValue;
  mlir::Operation *compareOp = nullptr;
  mlir::Value compareValue;
  mlir::Region *stepRegion = nullptr;
  mlir::Operation *stepOp = nullptr;
  mlir::Value stepValue;

  // For algebraic manipulation of linear expressions.
  // The comparison may involve a linear expression on the induction side of the
  // expression. So a comparison of the form `m * i + b < u` can be normalized
  // if all terms are loop invariant except for the induction `i`. If
  // `isLinearExpr` is false, there is no linear expression and the other fields
  // are not germane. (`u` is `compareValue` above.)
  bool negatedAddend = false;   // b is -b; b is subtracted.
  bool reciprocalScale = false; // m is 1/m; m is a divisor.
  bool minusOneMult = false;    // -1 * m; from b - m * i
  mlir::Value addendValue;      // b value
  mlir::Value scaleValue;       // m value
};

/// Does boundary test defines a semi-open interval?
bool isSemiOpenPredicate(mlir::arith::CmpIPredicate p);
bool isUnsignedPredicate(mlir::arith::CmpIPredicate p);
bool isSignedPredicate(mlir::arith::CmpIPredicate p);

/// A counted loop is defined to be a loop that will execute some compile-time
/// constant number of iterations. We recognize a normalized, semi-open interval
/// loop such as
/// ```
///   for(i = 0; i < constant_number_of_iterations; ++i)
/// ```
/// as a canonical counted loop.
bool isaCountedLoop(cc::LoopOp op, bool allowClosedInterval = true);

bool loopContainsBreak(cc::LoopOp op);

/// An indefinite counted loop is a counted loop which may have early exits.
bool isaIndefiniteCountedLoop(cc::LoopOp op, bool allowClosedInterval = true);

/// An invariant loop is defined to be a loop that will execute some run-time
/// invariant number of iterations. We recognize a normalized, semi-open
/// interval loop such as
/// ```
///   for(i = 0; i < invariant_expression; ++i)
/// ```
/// as a canonical invariant loop. If \p c is not null and the loop is
/// invariant, then the loop components will be returned via \p c.
bool isaInvariantLoop(cc::LoopOp op, bool allowClosedInterval = true,
                      bool allowEarlyExit = false, LoopComponents *c = nullptr);
bool isaInvariantLoop(const LoopComponents &c, bool allowClosedInterval);

/// An indefinite invariant loop is an invariant loop which may have early
/// exits. The number of iterations will be at most the upper bound expression.
/// We recognize the normalized, semi-open interval loop such as
/// ```
///   for(i = 0; i < invariant_expression; ++i) {
///     ...
///       break;
///     ...
///   }
/// ```
/// is a canonical indefinite loop.
inline bool isaIndefiniteInvariantLoop(cc::LoopOp op,
                                       bool allowClosedInterval = true,
                                       LoopComponents *c = nullptr) {
  return isaInvariantLoop(op, allowClosedInterval, /*allowEarlyExit=*/true, c);
}

// We expect the loop control value to have the following form.
//
//   %final = cc.loop while ((%iter = %initial) -> (iN)) {
//     ...
//     %cond = arith.cmpi {<.<=,!=,>=,>}, %iter, %bound : iN
//     cc.condition %cond (%iter : iN)
//   } do {
//    ^bb1(%iter : iN):
//     ...
//     cc.continue %iter : iN
//   } step {
//    ^bb2(%iter : iN):
//     ...
//     %next = arith.{addi,subi} %iter, %step : iN
//     cc.continue %next : iN
//   }
//
// with the additional requirement that none of the `...` sections can modify
// the value of `%bound` or `%step`. Those values are invariant if there are
// no side-effects in the loop Op (no store or call operations) and these values
// do not depend on a block argument. If \p c is not null and the loop is
/// invariant, then the loop components will be returned via \p c.
bool hasMonotonicControlInduction(cc::LoopOp loop, LoopComponents *c = nullptr);

/// A monotonic loop is defined to be a loop that will execute some bounded
/// number of iterations that can be predetermined before the loop, in fact,
/// executes. A semi-open interval loop such as
/// ```
///   for(i = start; i < stop; i += step)
/// ```
/// is a (definite) monotonic loop that must execute a number of iterations as
/// given by the following equation. Early exits (break statements) are
/// permitted in \e indefinite monotonic loops.
/// ```
///   let iterations = (stop - 1 - start + step) / step
///      iterations : if iterations > 0
///      0 : otherwise
/// ```
/// If \p c is not null and the loop is invariant, then the loop components will
/// be returned via \p c.
bool isaMonotonicLoop(mlir::Operation *op, bool allowEarlyExit = false,
                      LoopComponents *c = nullptr);

/// An indefinite monotonic loop is a monotonic loop that may have early exits.
inline bool isaIndefiniteMonotonicLoop(mlir::Operation *op,
                                       LoopComponents *c = nullptr) {
  return isaMonotonicLoop(op, /*allowEarlyExit=*/true, c);
}

/// A secondary induction variable is a loop argument other than the primary
/// control induction that steps by a loop-invariant amount on every iteration.
/// Because the primary has been normalized to start at 0 and step by 1, the
/// secondary can be expressed as a closed-form linear function of the primary:
///
///   j(i) = initialValue + i * stepValue   (stepIsAdd == true)
///   j(i) = initialValue - i * stepValue   (stepIsAdd == false)
///
/// This allows the secondary to be computed on-the-fly from the primary
/// instead of being threaded around the loop as an extra carried value.
struct SecondaryInduction {
  unsigned argIndex;        // Position of this arg in the loop's arg list.
  mlir::Value initialValue; // Value of j before the first iteration (j_0).
  mlir::Value stepValue;    // Loop-invariant amount added/subtracted per iter.
  bool stepIsAdd;           // true → j += step each iter; false → j -= step.
  // true when j is not an independent accumulator but is instead reassigned
  // each iteration to the primary induction's own per-iteration value (i.e.
  // j_{k+1} = i_k). Its closed form is then a one-iteration-shifted copy of
  // the primary's, with j_0 kept as given (see getSecondaryInductions).
  bool aliasesPrimary = false;
};

/// Find all secondary inductions in \p loop given that \p primary describes
/// the primary control induction.  The loop must be in invariant (normalized)
/// form — i.e., the primary starts at 0 and steps by 1 — otherwise the
/// closed-form j(i) derivation is invalid.
mlir::SmallVector<SecondaryInduction>
getSecondaryInductions(cc::LoopOp loop, const LoopComponents &primary);

/// Recover the different subexpressions from the loop if it conforms to the
/// pattern. Given a LoopOp where induction is in a register:
/// ```
///   for (int induction = initialValue;
///        induction compareOp compareValue;
///        induction = induction stepOp stepValue) ...
/// ```
///
/// Get references to each of: induction, initialValue, compareOp, compareValue,
/// stepOp, and stepValue regardless of the loop structure. Otherwise return
/// `std::nullopt`.
std::optional<LoopComponents> getLoopComponents(cc::LoopOp loop);

/// If \p v is loop-invariant with respect to \p loop (see isLoopInvariant)
/// but does not itself dominate \p loop -- i.e. it is a pure recomputation
/// of an invariant expression nested inside the loop rather than a value
/// hoisted above it -- clone the defining op (and, recursively, its
/// operands) at the current insertion point of \p rewriter so the result
/// can be used at a point that must dominate \p loop. If \p v already
/// dominates \p loop, it is returned unchanged.
mlir::Value materializeLoopInvariant(mlir::RewriterBase &rewriter,
                                     mlir::Value v, cc::LoopOp loop);

} // namespace cudaq::opt
