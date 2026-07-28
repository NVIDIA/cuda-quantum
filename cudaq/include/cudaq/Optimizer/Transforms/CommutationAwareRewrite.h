/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include <cstddef>
#include <memory>
#include <optional>

namespace cudaq::opt {
namespace detail {
class CommutationAwareRewriteListener;
}

/// Direction in which an anchor operation searches its block.
enum class CommutationSearchDirection { Forward, Backward };

/// The nearest compatible endpoint and the operations crossed to reach it.
struct CommutationAwareRewriteMatch {
  mlir::Operation *endpoint;
  /// Operations between the anchor and the endpoint that share at least one
  /// virtual qubit with the anchor, in block order. Operations acting only on
  /// other qubits are not listed: they commute by disjoint support and are
  /// never examined.
  llvm::SmallVector<mlir::Operation *> crossed;
};

/// Observable maintenance work performed during one rewrite invocation.
struct CommutationAwareRewriteStatistics {
  /// Number of block-local analysis instances constructed, including rebuilds.
  std::size_t analysisBuilds = 0;
  /// Number of analysis builds after an observed unsupported mutation discarded
  /// live state for the same block.
  std::size_t fallbackRebuilds = 0;
  /// Number of reached candidates queried for commutation after the endpoint
  /// predicate rejected them.
  std::size_t commutationProbes = 0;
  /// Number of block-local operations selected from the merged value-flow
  /// frontier, including accepted endpoints and stopping barriers.
  std::size_t frontierCandidates = 0;
};

/// Directional block-local search owned by a rewrite driver.
///
/// Starting at `anchor`, the search walks in the selected direction and returns
/// the first endpoint accepted by the consumer. No operation is moved.
///
/// The anchor is the operator supplied by the consumer pattern. The endpoint is
/// the first operator accepted by that pattern's endpoint predicate after every
/// anchor wire path reaches it at the same frontier. Both roles require
/// `OperatorInterface`; every control and target is a scalar `!quake.wire`, and
/// the operation threads those wires to its results.
///
/// The search expects block-local linear-wire Quake. Measurement instruments
/// and reset channels may be crossed, but cannot be anchors or endpoints. The
/// analysis must prove that a crossed operation commutes with the anchor, and
/// its scalar target wires must thread one-to-one. Other non-unitary quantum
/// operations, reusable `!quake.control`, reference and aggregate quantum
/// values, dataflow that leaves the block, and unresolved virtual qubits end
/// the search. Pass pipelines can establish the supported operator form by
/// running `linear-ctrl-form` after `memtoreg`.
///
/// Endpoints are found by following the anchor's own wire dataflow, not by
/// scanning the block. An operation that uses none of the anchor's quantum
/// values cannot act on its qubits, so the search neither examines it nor
/// treats it as a barrier. That is why an unrelated call or region-owning
/// operation passes unnoticed. When such an operation does touch one of the
/// anchor's qubits the walk still finds it, because a use nested in a region is
/// a use like any other, and the search then ends there for leaving the block.
///
/// A direct endpoint whose ordered scalar-wire operands are exactly the
/// anchor's ordered scalar-wire results has an empty crossing slice. For
/// identities such as A A^-1 = I or R(a) R(b) = R(a+b), that exact def-use
/// threading plus the consumer's endpoint algebra is sufficient. When any
/// operation lies in the anchor's crossed slice, the block-local
/// `CommutationAnalysis` must prove that the anchor commutes with that
/// operation before the search advances past it.
///
/// Two things are worth knowing before writing a consumer. Wire-set reuse
/// relies on Quake's borrow and return discipline, so the search will not
/// follow a qubit across a return, and a wire held concurrently elsewhere is
/// beyond what it can reason about. And while the anchor is proven to commute
/// with everything it crosses, the endpoint carries no such proof. It is
/// simply where the anchor stops, so it may even be known not to commute; the
/// consumer owns the endpoint pair's algebraic identity. A `DoesNotCommute` or
/// `Indeterminate` result therefore ends the search only for a candidate the
/// consumer declined.
class CommutationAwareRewriteMatcher {
public:
  ~CommutationAwareRewriteMatcher();

  CommutationAwareRewriteMatcher(const CommutationAwareRewriteMatcher &) =
      delete;
  CommutationAwareRewriteMatcher &
  operator=(const CommutationAwareRewriteMatcher &) = delete;

  /// Find the nearest consumer-compatible supported quantum endpoint in
  /// `direction`. The predicate is called only for an `OperatorInterface`
  /// candidate with supported scalar-wire flow, and before checking complete
  /// frontier alignment or deciding whether it may be crossed.
  std::optional<CommutationAwareRewriteMatch>
  findNearest(mlir::Operation *anchor, CommutationSearchDirection direction,
              llvm::function_ref<bool(mlir::Operation *)> isEndpoint);

  /// Return whether controls and targets carry the same ordered qubit
  /// identities and occupy the same roles. Direct scalar-wire consumers are
  /// decided from exact ordered def-use threading; non-direct queries fall back
  /// to the block-local commutation analysis.
  bool haveSameOrderedQuantumOperands(mlir::Operation *lhs,
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
/// itself. Consumers add patterns through `getPatterns()` and pass
/// `getMatcher()` to patterns that need block-local endpoint searches. Consumer
/// patterns own endpoint compatibility, replacement placement and semantics,
/// `PatternBenefit`, convergence, and SSA rewiring.
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
  explicit CommutationAwareRewriteDriver(
      mlir::MLIRContext &context,
      mlir::GreedyRewriteConfig config = mlir::GreedyRewriteConfig());
  ~CommutationAwareRewriteDriver();

  CommutationAwareRewriteDriver(const CommutationAwareRewriteDriver &) = delete;
  CommutationAwareRewriteDriver &
  operator=(const CommutationAwareRewriteDriver &) = delete;

  /// Return the mutable pattern set to populate before `run()`.
  mlir::RewritePatternSet &getPatterns();

  /// Return the driver-owned endpoint search for use by consumer patterns.
  CommutationAwareRewriteMatcher &getMatcher();

  /// Return stable event counts for incremental-maintenance verification and
  /// focused traversal-cost measurements. Counts cover this driver's single
  /// rewrite invocation and any `matcher` queries made before it.
  CommutationAwareRewriteStatistics getStatistics() const;

  /// Apply the owned pattern set to one region whose parent is isolated from
  /// above. Return failure on reuse or if the greedy driver does not converge.
  mlir::LogicalResult run(mlir::Region &region);

private:
  class Impl;
  std::unique_ptr<Impl> impl;
};

} // namespace cudaq::opt
