/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir {
class Operation;
namespace func {
class FuncOp;
}
} // namespace mlir

namespace cudaq::opt {

/// Analysis of `quake.borrow_wire` / `quake.return_wire` discipline in a
/// function that is in wire-set form.
///
/// A physical wire (a wire set name plus an identity) may be borrowed at most
/// once until it is returned. This analysis finds the places where that rule is
/// (or may be) violated. It works on structured (`cc.if`, `cc.scope`,
/// `cc.loop` with `cc.break` and `cc.continue`), CFG (`cf.br`, `cf.cond_br`)
/// and mixed forms of the IR, and it never modifies the IR.
///
/// The non-local `cc.unwind_*` operations are not modeled. They are purely
/// source-level operations, and wires do not exist at the source level, so
/// lowering to wire form necessarily eliminates them.
///
/// Wires are in SSI form: every wire value has exactly one use. There is
/// therefore no aliasing of wire values, and forwarding a wire from a def to
/// its use is a function rather than a relation.
///
/// The analysis has two parts.
///
/// 1. Each `!quake.wire` value is placed in a *thread*: the set of SSA values
///    that carry the same wire through gates, calls, block arguments, region
///    arguments and results. The set of physical wires borrowed into a thread
///    identifies which wire a `quake.return_wire` hands back. A thread with no
///    borrow (an argument or call result, say) is *unresolved*.
///
/// 2. A forward dataflow over the function's control flow computes, at every
///    program point, the set of threads that are borrowed on *some* path
///    ("may-held") and on *every* path ("must-held"). A `return_wire` releases
///    its thread. Borrowing a wire that a must-held thread can only be is a
///    definite conflict. Borrowing a wire that a may-held thread can carry is
///    a possible conflict.
///
/// A call is not a place where a wire can come from or go to. Wires come from
/// `quake.borrow_wire` and go to `quake.return_wire`. A borrowed wire that is
/// handed to a call that does not produce a wire vanishes into the callee, and
/// a call that produces a wire, when it is not handed one, produces a wire out
/// of nowhere. Either is reported, at the call, and means that the IR is
/// broken. The analysis then stops without looking at the rest of the function.
/// A call is any operation with the call interface, whatever its dialect.
struct WireBorrowAnalysis {
  enum class Kind {
    /// A wire was borrowed while it was already borrowed.
    DoubleBorrow,
    /// A wire was borrowed and is neither returned nor escapes the function.
    Unreturned,
    /// A borrowed wire was handed to a call that produces no wire, so the wire
    /// vanished into the callee.
    Vanished,
    /// A call produced a wire, but it was not handed a wire. The wire is of
    /// unknown origin.
    Conjured
  };

  enum class Certainty {
    /// The violation occurs on every path reaching the operation.
    Definite,
    /// The violation occurs on some path, or the analysis could not tell the
    /// identity of a wire precisely enough to rule it out.
    Possible
  };

  struct Conflict {
    Kind kind;
    Certainty certainty;
    /// The offending `quake.borrow_wire` for `DoubleBorrow`, the function exit
    /// operation for `Unreturned`, or the call for `Vanished` and `Conjured`.
    mlir::Operation *op;
    /// The wire. Empty for `Conjured`, as the wire is not known.
    llvm::StringRef setName;
    std::uint32_t identity;
    /// Other borrows of the same wire in the function. These are candidates
    /// for where the wire is (possibly) already held.
    llvm::SmallVector<mlir::Operation *, 2> otherBorrows;
  };

  /// Analyze \p func. A declaration yields an empty analysis.
  explicit WireBorrowAnalysis(mlir::func::FuncOp func);

  /// All conflicts found, definite and possible, in program order.
  const llvm::SmallVectorImpl<Conflict> &getConflicts() const {
    return conflicts;
  }

  /// `quake.return_wire` operations whose wire thread contains no borrow, so
  /// the analysis cannot tell which wire is being returned. The analysis
  /// conservatively assumes such a return may release any wire.
  const llvm::SmallVectorImpl<mlir::Operation *> &getUnresolvedReturns() const {
    return unresolvedReturns;
  }

private:
  llvm::SmallVector<Conflict> conflicts;
  llvm::SmallVector<mlir::Operation *> unresolvedReturns;
};

} // namespace cudaq::opt
