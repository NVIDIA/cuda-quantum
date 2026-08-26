/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstddef>
#include <memory>

namespace cudaq::opt {
namespace detail {
class CommutationAwareRewriteListener;
}

/// Observable analysis-maintenance work performed by one rewrite driver.
struct CommutationAwareRewriteStatistics {
  /// Number of block-local analysis instances constructed, including rebuilds.
  std::size_t analysisBuilds = 0;
  /// Number of analysis builds after an observed unsupported mutation discarded
  /// live state for the same block.
  std::size_t fallbackRebuilds = 0;
};

/// Backward block-local search owned by a rewrite driver.
///
/// Starting at the later `anchor`, the search follows defining operations from
/// its scalar-wire inputs and returns the nearest earlier endpoint accepted by
/// the consumer. This direction aligns consumers with MLIR's bottom-up greedy
/// schedule. No operation is moved.
///
/// Both endpoints require `OperatorInterface`; every control and target is a
/// scalar `!quake.wire`, and the operation threads those wires to its results.
/// A multi-wire endpoint matches only when every anchor wire reaches it at the
/// same frontier. Each result traversed backward must have exactly one use, and
/// that use must be the downstream operation expected on its lane. This guards
/// against treating a unique defining operation as linear when its result has
/// branched.
///
/// The search expects block-local linear-wire Quake. Candidate endpoints are
/// use-def frontier heads on the anchor's own wires. The search begins with a
/// block-order scan, including the latest head when the frontier is split. Once
/// analysis is required, it uses ordered per-qubit interaction streams when
/// every anchor wire has a known identity. Otherwise it continues the
/// block-order scan. A frontier head that the consumer declines must have a
/// pairwise commutation proof with the anchor before the frontier advances.
/// Every other enumerated scalar-wire operation requires either that pairwise
/// proof or a disjoint-support proof. Fresh local identity sources may be
/// crossed structurally because they cannot alias an existing logical qubit.
/// Other identity boundaries require a disjoint-support proof.
///
/// An ordinary single-block scope may be crossed when its captured scalar wires
/// have disjoint logical support. Marked atomic scopes, calls, unsupported
/// region owners, unsupported reference or aggregate quantum flow, unresolved
/// identity, multiple use, and incomplete frontiers end the search. Pass
/// pipelines can establish the supported operator form by running
/// `convert-to-linear-values`.
///
/// The consumer owns the accepted endpoint's algebraic identity, so the
/// endpoint needs no commutation proof. The search instead proves complete
/// frontier alignment, linear use-def threading, and distinct logical operand
/// roles. Single-wire endpoints establish distinctness structurally;
/// multi-wire endpoints require a known identity for every role. Traversable
/// measurement and reset frontier heads are never endpoints and require the
/// same pairwise commutation proof as any other declined frontier head.
class CommutationAwareRewriteMatcher {
public:
  ~CommutationAwareRewriteMatcher();

  CommutationAwareRewriteMatcher(const CommutationAwareRewriteMatcher &) =
      delete;
  CommutationAwareRewriteMatcher &
  operator=(const CommutationAwareRewriteMatcher &) = delete;

  /// Find the nearest compatible earlier quantum endpoint. The later anchor
  /// aligns the query with bottom-up greedy rewriting. The predicate is called
  /// only for an `OperatorInterface` candidate with
  /// supported scalar-wire flow, and before checking complete frontier
  /// alignment or deciding whether it may be crossed.
  mlir::Operation *
  find_nearest(mlir::Operation *anchor,
               llvm::function_ref<bool(mlir::Operation *)> isEndpoint);

  /// Return whether a supported scalar-wire operator uses a distinct logical
  /// qubit in every control and target role. Unary operators establish this
  /// structurally. Multi-wire operators require a known identity for every
  /// role and reject duplicate identities.
  bool has_distinct_quantum_operands(mlir::Operation *operation);

  /// Return whether controls and targets carry the same ordered qubit
  /// identities and occupy the same roles. Direct scalar-wire consumers are
  /// decided from exact ordered def-use threading; non-direct queries fall back
  /// to the block-local commutation analysis.
  bool have_same_ordered_quantum_operands(mlir::Operation *lhs,
                                          mlir::Operation *rhs);

private:
  class Impl;
  explicit CommutationAwareRewriteMatcher(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl;

  friend class CommutationAwareRewriteDriver;
  friend class detail::CommutationAwareRewriteListener;
};

/// Runs one commutation-aware greedy rewrite session.
///
/// The driver is intended for use inside a pass; it does not register a pass
/// itself. Consumers add patterns through `get_patterns()` and pass
/// `get_matcher()` to patterns that need block-local endpoint searches.
/// Consumer patterns own endpoint compatibility, replacement placement and
/// semantics, `PatternBenefit`, convergence, and SSA rewiring.
///
/// A driver accepts exactly one `run()` call on a `Region` isolated from above.
/// It owns the search state, pattern set, listener, and analysis state for that
/// session and chains any listener supplied in the greedy configuration.
///
/// All participating changes must use the observed `PatternRewriter`; direct IR
/// mutation is unsupported. Verified identity-preserving rewrites are
/// maintained incrementally. Moves, identity-changing or region-changing
/// rewrites, and unsupported mutation shapes conservatively rebuild affected
/// block state.
class CommutationAwareRewriteDriver {
public:
  /// Construct a single-session driver. Region simplification is always
  /// disabled because block-local analysis maintenance cannot support its
  /// implicit block and operation changes.
  explicit CommutationAwareRewriteDriver(
      mlir::MLIRContext &context,
      mlir::GreedyRewriteConfig config = mlir::GreedyRewriteConfig());
  ~CommutationAwareRewriteDriver();

  CommutationAwareRewriteDriver(const CommutationAwareRewriteDriver &) = delete;
  CommutationAwareRewriteDriver &
  operator=(const CommutationAwareRewriteDriver &) = delete;

  /// Return the mutable pattern set to populate before `run()`.
  mlir::RewritePatternSet &get_patterns();

  /// Return the driver-owned endpoint search for use by consumer patterns.
  CommutationAwareRewriteMatcher &get_matcher();

  /// Return stable event counts for incremental-maintenance verification.
  /// Counts cover this driver's single rewrite invocation and any endpoint
  /// search queries made before it.
  CommutationAwareRewriteStatistics get_statistics() const;

  /// Apply the owned pattern set to one region whose parent is isolated from
  /// above. Return failure on reuse or if the greedy driver does not converge.
  mlir::LogicalResult run(mlir::Region &region);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace cudaq::opt
