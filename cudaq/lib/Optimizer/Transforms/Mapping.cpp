/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Transforms/AddMetadata.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Device.h"
#include "cudaq/Support/Handle.h"
#include "cudaq/Support/Placement.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace cudaq::opt {
#define GEN_PASS_DEF_MAPPINGFUNC
#define GEN_PASS_DEF_MAPPINGPREP
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

#define DEBUG_TYPE "quantum-mapper"

using namespace mlir;

namespace {

constexpr StringRef mappedWireSetName("mapped_wireset");

//===----------------------------------------------------------------------===//
// Placement
//===----------------------------------------------------------------------===//

/// Initial-layout strategy selected by the `placement` pass option.
enum class PlacementStrategy { Auto, Identity, Greedy };

/// Parse the `placement` option string, or nullopt for an unknown value.
std::optional<PlacementStrategy> parsePlacementStrategy(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<PlacementStrategy>>(name)
      .Case("auto", PlacementStrategy::Auto)
      .Case("identity", PlacementStrategy::Identity)
      .Case("greedy", PlacementStrategy::Greedy)
      .Default(std::nullopt);
}

/// A symmetric weighted interaction graph over virtual qubits. Each edge counts
/// the two-qubit gates acting on a pair of virtual qubits. It is stored as
/// per-virtual sparse adjacency, since a circuit touches far fewer than the
/// `n^2` possible pairs, and the weighted degree of each virtual is cached as
/// interactions are recorded.
class VirtualInteractionGraph {
public:
  explicit VirtualInteractionGraph(unsigned numQubits)
      : adjacency(numQubits), weightedDegrees(numQubits, 0) {}

  /// Record one two-qubit interaction between virtual qubits `v0` and `v1`.
  /// Self-interactions are ignored.
  void addInteraction(unsigned v0, unsigned v1) {
    if (v0 == v1)
      return;
    ++adjacency[v0][v1];
    ++adjacency[v1][v0];
    ++weightedDegrees[v0];
    ++weightedDegrees[v1];
    anyInteraction = true;
  }

  /// The interaction neighbors of `u`, each mapped to the number of recorded
  /// interactions on that edge.
  const DenseMap<unsigned, unsigned> &neighbors(unsigned u) const {
    return adjacency[u];
  }

  /// The weighted degree of `u`: the total interaction count incident to it.
  unsigned weightedDegree(unsigned u) const { return weightedDegrees[u]; }

  /// Whether any interaction was recorded.
  bool hasInteractions() const { return anyInteraction; }

private:
  SmallVector<DenseMap<unsigned, unsigned>> adjacency;
  SmallVector<unsigned> weightedDegrees;
  bool anyInteraction = false;
};

/// Builds a deterministic topology-aware initial layout by assigning highly
/// interacting virtual qubits to central physical qubits first. The greedy
/// growth and its tie-breaks make the layout a deterministic function of the
/// interaction counts and the device, so it is reproducible across runs.
class GreedyInitialPlacer {
public:
  GreedyInitialPlacer(const cudaq::Device &device,
                      const VirtualInteractionGraph &interactions,
                      ArrayRef<bool> userVirtualQubits)
      : device(device), interactions(interactions),
        userVirtualQubits(userVirtualQubits), n(device.getNumQubits()),
        placedVirtual(n, false), vrToPhy(n, 0) {}

  /// Produce the `vrToPhy` seed layout.
  SmallVector<unsigned> run() {
    // No two-qubit interactions, so every layout routes identically; return the
    // identity seed for a deterministic result.
    if (!interactions.hasInteractions()) {
      for (unsigned v = 0; v < n; ++v)
        vrToPhy[v] = v;
      return vrToPhy;
    }

    computeCentrality();
    initWorklists();

    // Seed the highest-degree virtual qubit onto the most central physical
    // qubit, then grow the layout around it.
    place(chooseSeedVirtual(), bestFreePhysical());

    while (!unplacedUserVirtuals.empty()) {
      unsigned v = chooseNextVirtual();
      place(v, bestPhysicalFor(v));
    }

    assignRemainingVirtuals();
    return vrToPhy;
  }

private:
  /// Physical centrality used to break ties: total distance to every other
  /// qubit, and connectivity degree.
  void computeCentrality() {
    using Qubit = cudaq::Device::Qubit;
    distanceSum.assign(n, 0);
    physDegree.assign(n, 0);
    for (unsigned p = 0; p < n; ++p) {
      for (unsigned q = 0; q < n; ++q)
        distanceSum[p] += device.getDistance(Qubit(p), Qubit(q));
      physDegree[p] =
          static_cast<unsigned>(device.getNeighbours(Qubit(p)).size());
    }
  }

  /// Seed the worklists for the placement walk: every physical is free and
  /// every user virtual is unplaced, both in ascending order.
  void initWorklists() {
    freePhysicals.reserve(n);
    for (unsigned p = 0; p < n; ++p)
      freePhysicals.push_back(p);
    for (unsigned u = 0; u < n; ++u)
      if (userVirtualQubits[u])
        unplacedUserVirtuals.push_back(u);
  }

  /// True when physical qubit `a` is more central than `b` and should be
  /// preferred. The primary key is the total distance to every other qubit
  /// (smaller is more central). Ties break toward the higher connectivity
  /// degree, then toward the lower index so the layout stays deterministic.
  bool isMoreCentralPhysical(unsigned a, unsigned b) const {
    if (distanceSum[a] != distanceSum[b])
      return distanceSum[a] < distanceSum[b];
    if (physDegree[a] != physDegree[b])
      return physDegree[a] > physDegree[b];
    return a < b;
  }

  /// The most central physical qubit not yet used.
  unsigned bestFreePhysical() const {
    unsigned best = n;
    for (unsigned p : freePhysicals)
      if (best == n || isMoreCentralPhysical(p, best))
        best = p;
    return best;
  }

  /// The highest weighted-degree virtual qubit, breaking ties toward the lower
  /// index.
  unsigned chooseSeedVirtual() const {
    unsigned seed = n;
    for (unsigned u : unplacedUserVirtuals)
      if (seed == n ||
          interactions.weightedDegree(u) > interactions.weightedDegree(seed))
        seed = u;
    return seed;
  }

  struct CandidateScore {
    unsigned placedWeight = 0;
    unsigned degree = 0;
    unsigned index = 0;

    bool isBetterThan(const CandidateScore &other) const {
      if (placedWeight != other.placedWeight)
        return placedWeight > other.placedWeight;
      if (degree != other.degree)
        return degree > other.degree;
      return index < other.index;
    }
  };

  /// The unplaced virtual qubit most connected to the placed set. Ties break by
  /// total weighted degree, then by lower virtual index for determinism. The
  /// disconnected case (no interaction with the placed set) reduces to highest
  /// weighted degree, then lower index, since weighted degree is always at
  /// least the placed weight.
  unsigned chooseNextVirtual() const {
    unsigned pick = n;
    CandidateScore best;
    for (unsigned u : unplacedUserVirtuals) {
      CandidateScore score{0, interactions.weightedDegree(u), u};
      for (const auto &edge : interactions.neighbors(u))
        if (placedVirtual[edge.first])
          score.placedWeight += edge.second;
      if (pick == n || score.isBetterThan(best)) {
        pick = u;
        best = score;
      }
    }
    assert(pick != n &&
           "chooseNextVirtual called with no unplaced user qubits");
    return pick;
  }

  /// The free physical qubit minimizing weighted distance from `v` to its
  /// placed partners, breaking ties by centrality. When `v` has no interaction
  /// with any placed qubit every cost is zero, so this returns the most central
  /// free physical, exactly as `bestFreePhysical` would.
  unsigned bestPhysicalFor(unsigned v) const {
    using Qubit = cudaq::Device::Qubit;
    unsigned bestPhy = n;
    unsigned bestCost = 0;
    for (unsigned p : freePhysicals) {
      unsigned cost = 0;
      for (const auto &edge : interactions.neighbors(v))
        if (placedVirtual[edge.first])
          cost += edge.second *
                  device.getDistance(Qubit(p), Qubit(vrToPhy[edge.first]));
      bool better = bestPhy == n || cost < bestCost ||
                    (cost == bestCost && isMoreCentralPhysical(p, bestPhy));
      if (better) {
        bestPhy = p;
        bestCost = cost;
      }
    }
    return bestPhy;
  }

  /// Map virtual `v` onto physical `p`, marking `v` placed and `p` taken. The
  /// sorted worklists keep their order across the erases.
  void place(unsigned v, unsigned p) {
    vrToPhy[v] = p;
    placedVirtual[v] = true;
    freePhysicals.erase(llvm::lower_bound(freePhysicals, p));
    if (auto it = llvm::lower_bound(unplacedUserVirtuals, v);
        it != unplacedUserVirtuals.end() && *it == v)
      unplacedUserVirtuals.erase(it);
  }

  /// Assign any still-unplaced virtuals (non-user qubits) to the remaining free
  /// physicals, pairing them in ascending order. `freePhysicals` stays sorted,
  /// so this reproduces the ascending virtual to ascending physical pairing.
  void assignRemainingVirtuals() {
    unsigned next = 0;
    for (unsigned v = 0; v < n; ++v)
      if (!placedVirtual[v])
        vrToPhy[v] = freePhysicals[next++];
  }

  const cudaq::Device &device;
  const VirtualInteractionGraph &interactions;
  ArrayRef<bool> userVirtualQubits;
  const unsigned n;

  SmallVector<unsigned> distanceSum;
  SmallVector<unsigned> physDegree;

  SmallVector<bool> placedVirtual;
  SmallVector<unsigned> vrToPhy;

  // Worklists maintained by `place`, so the selection helpers iterate only the
  // relevant qubits instead of rescanning the full device. Both stay sorted
  // ascending.
  SmallVector<unsigned> freePhysicals;
  SmallVector<unsigned> unplacedUserVirtuals;
};

/// Generate the seed layouts to try, in deterministic order. Each seed only
/// proposes a starting vrToPhy. The router decides the rest. `interactions` is
/// required for the greedy strategies and ignored for identity.
SmallVector<SmallVector<unsigned>>
buildPlacementSeeds(PlacementStrategy strategy, unsigned numV,
                    const cudaq::Device &device,
                    const std::optional<VirtualInteractionGraph> &interactions,
                    ArrayRef<bool> userVirtualQubits) {
  SmallVector<SmallVector<unsigned>> seeds;

  if (strategy == PlacementStrategy::Auto ||
      strategy == PlacementStrategy::Identity) {
    SmallVector<unsigned> identity(numV);
    for (unsigned v = 0; v < numV; ++v)
      identity[v] = v;
    seeds.push_back(std::move(identity));
  }

  if (strategy == PlacementStrategy::Auto ||
      strategy == PlacementStrategy::Greedy) {
    assert(interactions.has_value() &&
           "greedy placement requires collected interactions");
    SmallVector<unsigned> greedy =
        GreedyInitialPlacer(device, *interactions, userVirtualQubits).run();
    // For `auto`, greedy degenerates to identity when there are no interactions
    // to place, so skip the duplicate rather than route the identity layout
    // twice.
    if (strategy == PlacementStrategy::Greedy || greedy != seeds.front())
      seeds.push_back(std::move(greedy));
  }

  return seeds;
}

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

/// The dependency DAG the router walks. It is built once from the IR and never
/// modified, so a layout can be routed without touching the circuit. Each node
/// is a routable operation. Its successors are the operations that consume its
/// result wires.
struct RoutingProblem {
  /// A handle to a node in the DAG, i.e. an index into `nodes`.
  struct NodeRef : cudaq::Handle {
    using Handle::Handle;
  };

  struct Node {
    mlir::Operation *op;
    /// Virtual qubits used by `op`, in quantum-operand order.
    SmallVector<cudaq::Placement::VirtualQ, 2> qubits;
    /// Routable users of `op`'s result wires, in use-list order. A user appears
    /// once per result wire it consumes, so a node becomes ready once it has
    /// been visited as many times as it has wire operands.
    SmallVector<NodeRef, 4> successors;
    bool isMeasure = false;
    /// A gate (not a measurement, sink, or return). Only these participate in
    /// the reverse-traversal pass.
    bool isUnitary = false;
    /// Two-qubit unitaries are the only nodes that join the extended layer.
    bool isTwoQ = false;
  };

  /// Routable operations, in program order.
  SmallVector<Node> nodes;
  /// Routable users of the source wires, in source order then use-list order.
  /// These seed the first front layer.
  SmallVector<NodeRef> sourceUsers;

  const Node &operator[](NodeRef n) const {
    assert(n.isValid() && "invalid node handle");
    return nodes[n.index];
  }
};

/// A single routing decision: a gate mapped onto physical qubits, or a swap
/// inserted between them. The router records these as it walks the circuit and
/// the emitter replays them to rewrite the IR.
struct RoutingEvent {
  enum class Kind { Gate, Swap };

  /// A gate mapped onto the physical qubits `phys`, in operand order.
  static RoutingEvent gate(mlir::Operation *op,
                           ArrayRef<cudaq::Placement::DeviceQ> phys) {
    return RoutingEvent{
        Kind::Gate, op,
        SmallVector<cudaq::Placement::DeviceQ, 2>(phys.begin(), phys.end())};
  }
  /// A swap inserted between physical qubits `q0` and `q1`.
  static RoutingEvent swap(cudaq::Placement::DeviceQ q0,
                           cudaq::Placement::DeviceQ q1) {
    return RoutingEvent{Kind::Swap, nullptr, {q0, q1}};
  }

  Kind kind;
  mlir::Operation *op;
  SmallVector<cudaq::Placement::DeviceQ, 2> phys;
};

/// The outcome of routing one layout. The emitter replays `trace` onto the IR.
/// `swapCount` is the metric used to compare layouts.
struct RoutingResult {
  /// Virtual-to-physical layout at the start of the walk, before any swap.
  SmallVector<unsigned> initialLayout;
  SmallVector<RoutingEvent> trace;
  unsigned swapCount = 0;
};

/// Look up a wire without letting DenseMap default a missing entry to virtual
/// qubit 0.
std::optional<cudaq::Placement::VirtualQ> lookupVirtualQ(
    const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ,
    Value wire) {
  auto it = wireToVirtualQ.find(wire);
  if (it == wireToVirtualQ.end())
    return std::nullopt;
  return it->second;
}

cudaq::Placement::VirtualQ requireVirtualQ(
    const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ,
    Value wire) {
  if (auto virtualQ = lookupVirtualQ(wireToVirtualQ, wire))
    return *virtualQ;
  llvm::report_fatal_error(
      "mapper invariant violated: quantum wire has no virtual qubit");
}

/// Build the routing problem from `block`. The nodes are the routable
/// operations that `isSupportedMappingOperation` accepts, other than the source
/// borrows. Edges and source successors are captured in MLIR use-list order so
/// the walk visits successors in the same order as the SSA use-def chains.
RoutingProblem buildRoutingProblem(
    Block &block, ArrayRef<cudaq::quake::BorrowWireOp> sources,
    const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ) {
  RoutingProblem problem;
  DenseMap<Operation *, RoutingProblem::NodeRef> nodeIndex;

  for (Operation &op : block) {
    if (isa<cudaq::quake::BorrowWireOp>(op) ||
        !cudaq::quake::isSupportedMappingOperation(&op))
      continue;
    RoutingProblem::Node node;
    node.op = &op;
    for (auto wire : cudaq::quake::getQuantumOperands(&op))
      node.qubits.push_back(requireVirtualQ(wireToVirtualQ, wire));
    node.isMeasure = op.hasTrait<cudaq::QuantumMeasure>();
    node.isUnitary = isa<cudaq::quake::OperatorInterface>(op);
    // A two-qubit gate the router has to make adjacent: a unitary on two wires,
    // not a measurement or a sink.
    node.isTwoQ = node.isUnitary && node.qubits.size() == 2;
    nodeIndex[&op] = RoutingProblem::NodeRef(problem.nodes.size());
    problem.nodes.push_back(std::move(node));
  }

  // Record successor edges by walking the uses of each quantum result wire. A
  // consumer is listed once per result wire it consumes, so a node's visit
  // count reaches its wire-operand count exactly when all of its inputs are
  // ready. Walking wire uses directly, rather than `Operation::getUsers`, makes
  // that multiplicity explicit and ignores classical results such as
  // measurement bits.
  // Splice cc::IfOp out of the dependency graph: a wire flowing into an IfOp
  // is forwarded to the IfOp's corresponding wire result so SABRE sees direct
  // edges from pre-if gates to post-if gates.
  std::function<void(Value, SmallVectorImpl<RoutingProblem::NodeRef> &)>
      recordWireUsers;
  recordWireUsers = [&](Value wire,
                        SmallVectorImpl<RoutingProblem::NodeRef> &out) {
    for (OpOperand &use : wire.getUses()) {
      auto *owner = use.getOwner();
      if (auto it = nodeIndex.find(owner); it != nodeIndex.end()) {
        out.push_back(it->second);
      } else if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(owner)) {
        unsigned linearIdx = 0;
        unsigned wireResultIdx = 0;
        for (Value linArg : ifOp.getLinearArgs()) {
          if (linArg == wire) {
            for (Value res : ifOp->getResults()) {
              if (isa<cudaq::quake::WireType>(res.getType())) {
                if (wireResultIdx == linearIdx) {
                  recordWireUsers(res, out);
                  break;
                }
                ++wireResultIdx;
              }
            }
            break;
          }
          ++linearIdx;
        }
      }
    }
  };
  for (auto &node : problem.nodes)
    for (Value wire : cudaq::quake::getQuantumResults(node.op))
      recordWireUsers(wire, node.successors);
  for (auto borrow : sources)
    recordWireUsers(borrow.getResult(), problem.sourceUsers);

  return problem;
}

/// Only unitary gates take part in the reverse-traversal pass. See
/// `buildReverseProblem` for why measurements, sinks, and returns drop out.
bool shouldIncludeInReverse(const RoutingProblem::Node &node) {
  return node.isUnitary;
}

/// Copy the routing-relevant fields of a forward unitary node into its reverse
/// counterpart. Successor and source-user edges are filled in afterwards, once
/// every included node has been assigned a reverse handle.
RoutingProblem::Node makeReverseNode(const RoutingProblem::Node &node) {
  RoutingProblem::Node rev;
  rev.op = node.op;
  rev.qubits = node.qubits;
  rev.isUnitary = true;
  rev.isTwoQ = node.isTwoQ;
  return rev;
}

/// Build the transposed problem over the unitary gates only, for the SABRE
/// reverse-traversal pass. Routing this forward is equivalent to routing the
/// original circuit in reverse: a node's successors here are its forward
/// predecessors, and result wires that do not feed another unitary (the
/// circuit's outputs) seed the walk. Measurements, sinks, and returns are not
/// unitary nodes, so they drop out, which is the paper's "skip measurements in
/// the reverse pass". Readiness is unchanged: a unitary has equal operand and
/// result arity, so its threshold is `qubits.size()` in both directions.
RoutingProblem buildReverseProblem(const RoutingProblem &forward) {
  RoutingProblem reverse;
  SmallVector<RoutingProblem::NodeRef> fwdToRev(forward.nodes.size());
  for (unsigned i = 0, end = forward.nodes.size(); i < end; ++i) {
    const RoutingProblem::Node &node = forward.nodes[i];
    if (!shouldIncludeInReverse(node))
      continue;
    fwdToRev[i] = RoutingProblem::NodeRef(reverse.nodes.size());
    reverse.nodes.push_back(makeReverseNode(node));
  }

  for (unsigned i = 0, end = forward.nodes.size(); i < end; ++i) {
    const RoutingProblem::Node &node = forward.nodes[i];
    if (!shouldIncludeInReverse(node))
      continue;
    unsigned unitarySuccessors = 0;
    for (RoutingProblem::NodeRef s : node.successors) {
      if (!shouldIncludeInReverse(forward[s]))
        continue;
      ++unitarySuccessors;
      // Processing the consumer in reverse makes this producer ready.
      reverse.nodes[fwdToRev[s.index].index].successors.push_back(fwdToRev[i]);
    }
    // Each result wire that does not feed a unitary is a reverse-circuit input.
    for (unsigned k = unitarySuccessors; k < node.qubits.size(); ++k)
      reverse.sourceUsers.push_back(fwdToRev[i]);
  }
  return reverse;
}

/// The `SabreRouter` class is modified implementation of the following paper:
/// Li, Gushu, Yufei Ding, and Yuan Xie. "Tackling the qubit mapping problem for
/// NISQ-era quantum devices." In Proceedings of the Twenty-Fourth International
/// Conference on Architectural Support for Programming Languages and Operating
/// Systems, pp. 1001-1014. 2019.
/// https://dl.acm.org/doi/pdf/10.1145/3297858.3304023
///
/// Routing starts with source operations collected during the analysis. These
/// operations form a layer, called the `frontLayer`, which is a set of
/// operations that have no unmapped predecessors. In the case of these source
/// operations, the router only needs to iterate over the front layer while
/// visiting all users of each operation. The processing of this front layer
/// will create a new front layer containing all operations that have being
/// visited the same number of times as their number of wire operands.
///
/// After processing the very first front layer, the algorithm proceeds to
/// process the newly created front layer. Once again, it processes the front
/// layer and map all operations that are compatible with the current placement,
/// i.e., one-wire operations and two-wire operations using wires that
/// correspond to qubits that are adjacently placed in the device. When an
/// operation is successfully mapped, it is removed from the front layer and all
/// its users are visited. Those users that have no unmapped predecessors are
/// added to the front layer. If the mapper cannot successfully map any
/// operation in the front layer, then it adds a swap to the circuit and tries
/// to map the front layer again. The routing process ends when the front layer
/// is empty.
///
/// Modifications from the published paper include the ability to defer
/// measurement mapping until the end, which is required for QIR Base Profile
/// programs (see the `allowMeasurementMapping` member variable).
class SabreRouter {
  using Swap = std::pair<cudaq::Placement::DeviceQ, cudaq::Placement::DeviceQ>;
  using NodeRef = RoutingProblem::NodeRef;

public:
  SabreRouter(const cudaq::Device &device, const RoutingProblem &problem,
              cudaq::Placement &placement, unsigned extendedLayerSize,
              float extendedLayerWeight, float decayDelta,
              unsigned roundsDecayReset, unsigned minStallSwapBudget,
              unsigned stallSwapBudgetPerQubit)
      : device(device), problem(problem), placement(placement),
        extendedLayerSize(extendedLayerSize),
        extendedLayerWeight(extendedLayerWeight), decayDelta(decayDelta),
        roundsDecayReset(roundsDecayReset),
        minStallSwapBudget(minStallSwapBudget),
        stallSwapBudgetPerQubit(stallSwapBudgetPerQubit),
        phyDecay(device.getNumQubits(), 1.0), allowMeasurementMapping(false) {}

  /// Main entry point into SabreRouter routing algorithm. Walks the DAG without
  /// modifying the IR and returns the decisions for the emitter to apply.
  RoutingResult route();

private:
  /// Visit each node in `successors` and bump its count. A node that has been
  /// visited once per wire operand joins `layer`, or `measureLayer` if it is a
  /// deferred measurement. `incremented` records the bumps so a lookahead can
  /// undo them.
  void visitSuccessors(ArrayRef<NodeRef> successors,
                       SmallVectorImpl<NodeRef> &layer,
                       SmallVectorImpl<NodeRef> *incremented = nullptr);

  LogicalResult mapOperation(NodeRef node);

  LogicalResult mapFrontLayer();

  void selectExtendedLayer();

  double computeLayerCost(ArrayRef<NodeRef> layer);

  Swap chooseSwap();

  /// Record a swap between two physical qubits: apply it to the placement and
  /// append it to the trace.
  void addSwap(cudaq::Placement::DeviceQ q0, cudaq::Placement::DeviceQ q1);

  /// Bring the closest front-layer two-qubit gate together along a shortest
  /// path, ignoring the heuristic. This is the action the release valve takes
  /// to guarantee forward progress.
  void forceClosestGate();

  /// Undo the swaps inserted since the last routed gate: revert the placement,
  /// drop the recorded events, and restore the swap count.
  void rewindEpisode(SmallVectorImpl<Swap> &episodeSwaps);

  /// Release valve for a stalled front layer. SABRE's decay only softly
  /// discourages the local minima the heuristic can fall into, so a stuck
  /// front layer would otherwise loop forever. Discard the current episode's
  /// swaps and force the closest gate together so the walk always makes
  /// progress. The decay state is left as is. It is a soft heuristic and resets
  /// on its own cycle. This follows the release-valve idea from LightSABRE
  /// (arXiv:2409.08368).
  void applyReleaseValve(SmallVectorImpl<Swap> &episodeSwaps);

private:
  const cudaq::Device &device;
  const RoutingProblem &problem;
  cudaq::Placement &placement;

  // Parameters
  const unsigned extendedLayerSize;
  const float extendedLayerWeight;
  const float decayDelta;
  const unsigned roundsDecayReset;
  // Release-valve stall budget: force a gate once this many consecutive swaps
  // route nothing. See `route` for how the floor and per-qubit terms combine.
  const unsigned minStallSwapBudget;
  const unsigned stallSwapBudgetPerQubit;

  // Internal data. The layers hold handles into `problem.nodes`.
  SmallVector<NodeRef> frontLayer;
  SmallVector<NodeRef> extendedLayer;
  SmallVector<NodeRef> measureLayer;
  SmallVector<bool> measureLayerSeen;
  llvm::SmallSet<cudaq::Placement::DeviceQ, 32> involvedPhy;
  SmallVector<float> phyDecay;

  /// The routing decisions accumulated during the current walk.
  RoutingResult result;

  /// How many times each node has been visited. A node is ready once this
  /// reaches its wire-operand count.
  SmallVector<unsigned> visitCount;

  /// Keep track of whether or not we're in the phase that allows measurements
  /// to be mapped
  bool allowMeasurementMapping;

#ifndef NDEBUG
  /// A logger used to emit diagnostics during the maping process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};

void SabreRouter::visitSuccessors(ArrayRef<NodeRef> successors,
                                  SmallVectorImpl<NodeRef> &layer,
                                  SmallVectorImpl<NodeRef> *incremented) {
  for (NodeRef s : successors) {
    unsigned count = ++visitCount[s.index];
    if (incremented)
      incremented->push_back(s);

    const RoutingProblem::Node &node = problem[s];
    if (count != node.qubits.size())
      continue;
    // Don't process measurements until we're ready.
    if (allowMeasurementMapping || !node.isMeasure) {
      layer.push_back(s);
    } else if (!measureLayerSeen[s.index]) {
      // Add to measureLayer. Don't add duplicates.
      measureLayerSeen[s.index] = true;
      measureLayer.push_back(s);
    }
  }
}

LogicalResult SabreRouter::mapOperation(NodeRef nodeRef) {
  const RoutingProblem::Node &node = problem[nodeRef];

  // Take the device qubits from this operation.
  SmallVector<cudaq::Placement::DeviceQ, 2> deviceQubits;
  for (auto vr : node.qubits)
    deviceQubits.push_back(placement.getPhy(vr));

  // An operation cannot be mapped if it is not a measurement and uses two
  // virtual qubits that are not adjacently placed.
  if (!node.isMeasure && deviceQubits.size() == 2 &&
      !device.areConnected(deviceQubits[0], deviceQubits[1]))
    return failure();

  // Record the placement. The emitter rewires the operation when it applies
  // the result.
  result.trace.push_back(RoutingEvent::gate(node.op, deviceQubits));
  return success();
}

LogicalResult SabreRouter::mapFrontLayer() {
  bool mappedAtLeastOne = false;
  SmallVector<NodeRef> newFrontLayer;

  LLVM_DEBUG({
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
  });
  for (NodeRef n : frontLayer) {
    const RoutingProblem::Node &node = problem[n];
    LLVM_DEBUG({
      logger.startLine() << "* ";
      node.op->print(logger.getOStream(),
                     OpPrintingFlags().printGenericOpForm());
    });
    if (failed(mapOperation(n))) {
      LLVM_DEBUG(logger.getOStream() << " --> FAILURE\n");
      newFrontLayer.push_back(n);
      for (auto vr : node.qubits)
        involvedPhy.insert(placement.getPhy(vr));
      LLVM_DEBUG({
        auto phy0 = placement.getPhy(node.qubits[0]);
        auto phy1 = placement.getPhy(node.qubits[1]);
        logger.indent();
        logger.startLine() << "+ virtual qubits: " << node.qubits[0] << ", "
                           << node.qubits[1] << '\n';
        logger.startLine() << "+ device qubits: " << phy0 << ", " << phy1
                           << '\n';
        logger.unindent();
      });
      continue;
    }
    LLVM_DEBUG(logger.getOStream() << " --> SUCCESS\n");
    mappedAtLeastOne = true;
    visitSuccessors(node.successors, newFrontLayer);
  }
  LLVM_DEBUG(logger.unindent());
  frontLayer = std::move(newFrontLayer);
  return mappedAtLeastOne ? success() : failure();
}

void SabreRouter::selectExtendedLayer() {
  extendedLayer.clear();
  SmallVector<NodeRef, 20> incremented;
  SmallVector<NodeRef> tmpLayer = frontLayer;
  while (!tmpLayer.empty() && extendedLayer.size() < extendedLayerSize) {
    SmallVector<NodeRef> newTmpLayer;
    for (NodeRef n : tmpLayer)
      visitSuccessors(problem[n].successors, newTmpLayer, &incremented);
    for (NodeRef n : newTmpLayer)
      // We only add operations that can influence placement to the extended
      // frontlayer, i.e., quantum operators that use two qubits.
      if (problem[n].isTwoQ)
        extendedLayer.push_back(n);
    tmpLayer = std::move(newTmpLayer);
  }

  for (NodeRef n : incremented)
    --visitCount[n.index];
}

double SabreRouter::computeLayerCost(ArrayRef<NodeRef> layer) {
  double cost = 0.0;
  for (NodeRef n : layer) {
    const RoutingProblem::Node &node = problem[n];
    auto phy0 = placement.getPhy(node.qubits[0]);
    auto phy1 = placement.getPhy(node.qubits[1]);
    cost += device.getDistance(phy0, phy1) - 1;
  }
  return cost / layer.size();
}

SabreRouter::Swap SabreRouter::chooseSwap() {
  // Obtain SWAP candidates
  SmallVector<Swap> candidates;
  for (auto phy0 : involvedPhy)
    for (auto phy1 : device.getNeighbours(phy0))
      candidates.emplace_back(phy0, phy1);

  if (extendedLayerSize)
    selectExtendedLayer();

  // Compute cost
  SmallVector<double> cost;
  for (auto [phy0, phy1] : candidates) {
    placement.swap(phy0, phy1);
    double swapCost = computeLayerCost(frontLayer);
    double maxDecay = std::max(phyDecay[phy0.index], phyDecay[phy1.index]);

    if (!extendedLayer.empty()) {
      double extendedLayerCost =
          computeLayerCost(extendedLayer) / extendedLayer.size();
      swapCost /= frontLayer.size();
      swapCost += extendedLayerWeight * extendedLayerCost;
    }

    cost.emplace_back(maxDecay * swapCost);
    placement.swap(phy0, phy1);
  }

  // Find and return the swap with minimal cost
  std::size_t minIdx = 0u;
  for (std::size_t i = 1u, end = cost.size(); i < end; ++i)
    if (cost[i] < cost[minIdx])
      minIdx = i;

  LLVM_DEBUG({
    logger.startLine() << "Choosing a swap:\n";
    logger.indent();
    logger.startLine() << "Involved device qubits:";
    for (auto phy : involvedPhy)
      logger.getOStream() << " " << phy;
    logger.getOStream() << "\n";
    logger.startLine() << "Swap candidates:\n";
    logger.indent();
    for (auto &&[qubits, c] : llvm::zip_equal(candidates, cost))
      logger.startLine() << "* " << qubits.first << ", " << qubits.second
                         << " (cost = " << c << ")\n";
    logger.getOStream() << "\n";
    logger.unindent();
    logger.startLine() << "Selected swap: " << candidates[minIdx].first << ", "
                       << candidates[minIdx].second << '\n';
    logger.unindent();
  });
  return candidates[minIdx];
}

void SabreRouter::addSwap(cudaq::Placement::DeviceQ q0,
                          cudaq::Placement::DeviceQ q1) {
  placement.swap(q0, q1);
  result.trace.push_back(RoutingEvent::swap(q0, q1));
  ++result.swapCount;
}

void SabreRouter::forceClosestGate() {
  NodeRef closest;
  unsigned bestDist = ~0u;
  for (NodeRef n : frontLayer) {
    const RoutingProblem::Node &node = problem[n];
    if (!node.isTwoQ)
      continue;
    unsigned d = device.getDistance(placement.getPhy(node.qubits[0]),
                                    placement.getPhy(node.qubits[1]));
    if (d < bestDist) {
      bestDist = d;
      closest = n;
    }
  }
  assert(closest.isValid() && "a stalled front layer must hold a 2-qubit gate");
  const RoutingProblem::Node &node = problem[closest];
  cudaq::Device::Path path = device.getShortestPath(
      placement.getPhy(node.qubits[0]), placement.getPhy(node.qubits[1]));
  // Move one qubit along the path until it is adjacent to the other.
  for (unsigned i = 0; i + 2 < path.size(); ++i)
    addSwap(path[i], path[i + 1]);
}

void SabreRouter::rewindEpisode(SmallVectorImpl<Swap> &episodeSwaps) {
  for (unsigned i = episodeSwaps.size(); i-- > 0;)
    placement.swap(episodeSwaps[i].first, episodeSwaps[i].second);
  result.trace.pop_back_n(episodeSwaps.size());
  result.swapCount -= episodeSwaps.size();
  episodeSwaps.clear();
}

void SabreRouter::applyReleaseValve(SmallVectorImpl<Swap> &episodeSwaps) {
  rewindEpisode(episodeSwaps);
  forceClosestGate();
  involvedPhy.clear();
}

RoutingResult SabreRouter::route() {
#ifndef NDEBUG
  constexpr char logLineComment[] =
      "//===-------------------------------------------===//\n";
#endif

  // Record the initial layout before routing starts moving qubits around.
  result = RoutingResult{};
  result.initialLayout.resize(placement.getNumVirtualQubits());
  for (unsigned v = 0, end = placement.getNumVirtualQubits(); v < end; ++v)
    result.initialLayout[v] =
        placement.getPhy(cudaq::Placement::VirtualQ(v)).index;

  visitCount.assign(problem.nodes.size(), 0);
  measureLayerSeen.assign(problem.nodes.size(), false);

  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logLineComment;
    logger.startLine() << "Mapping front layer:\n";
    logger.startLine() << logLineComment;
  });

  // The source ops can always be mapped.
  visitSuccessors(problem.sourceUsers, frontLayer);

  // SABRE's cost function is a heuristic. If it emits a long run of swaps
  // without making any front-layer gate executable, discard that local episode
  // and force one gate along a shortest path. This is the release valve from
  // LightSABRE (arXiv:2409.08368), triggered by the swap budget computed here.
  // The budget is deliberately loose: bringing one front-layer gate adjacent
  // costs at most the device diameter, and the qubit count upper-bounds the
  // diameter of a connected device. The per-qubit multiplier gives the
  // heuristic several times that worst-case direct-routing cost to explore and
  // recover before the valve fires. The floor keeps a usable budget on small
  // devices, where the scaled term would otherwise be too tight. Both terms are
  // pass options (`min-stall-swap-budget`, `stall-swap-budget-per-qubit`)
  // defaulting to 64 and 4.
  const unsigned stallSwapLimit = std::max(
      minStallSwapBudget, stallSwapBudgetPerQubit * device.getNumQubits());
  std::size_t numSwapSearches = 0;
  unsigned swapsSinceRouted = 0;
  SmallVector<Swap> episodeSwaps;
  bool done = false;
  while (!done) {
    // Once frontLayer is empty, grab everything from measureLayer and go again.
    if (frontLayer.empty()) {
      if (allowMeasurementMapping) {
        done = true;
      } else {
        allowMeasurementMapping = true;
        frontLayer = std::move(measureLayer);
      }
      continue;
    }

    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << logLineComment;
    });

    if (succeeded(mapFrontLayer())) {
      swapsSinceRouted = 0;
      episodeSwaps.clear();
      continue;
    }

    LLVM_DEBUG(logger.getOStream() << "\n";);

    if (swapsSinceRouted >= stallSwapLimit) {
      applyReleaseValve(episodeSwaps);
      swapsSinceRouted = 0;
      continue;
    }

    // Add a swap
    numSwapSearches++;
    auto [phy0, phy1] = chooseSwap();
    addSwap(phy0, phy1);
    episodeSwaps.push_back({phy0, phy1});
    ++swapsSinceRouted;
    involvedPhy.clear();

    // Update decay
    if ((numSwapSearches % roundsDecayReset) == 0) {
      std::fill(phyDecay.begin(), phyDecay.end(), 1.0);
    } else {
      phyDecay[phy0.index] += decayDelta;
      phyDecay[phy1.index] += decayDelta;
    }
  }
  LLVM_DEBUG(logger.startLine() << '\n' << logLineComment << '\n';);
  return std::move(result);
}

//===----------------------------------------------------------------------===//
// Search
//===----------------------------------------------------------------------===//

/// Layout search strategy selected by the `search` pass option.
enum class SearchStrategy { Sabre, None };

/// Parse the `search` option string, or nullopt for an unknown value.
std::optional<SearchStrategy> parseSearchStrategy(llvm::StringRef name) {
  return llvm::StringSwitch<std::optional<SearchStrategy>>(name)
      .Case("sabre", SearchStrategy::Sabre)
      .Case("none", SearchStrategy::None)
      .Default(std::nullopt);
}

/// Owns the layout search. For each seed it routes forward and, under `sabre`,
/// applies the paper's forward-backward-forward reverse-traversal refinement,
/// keeping both the unrefined and refined results as candidates so refinement
/// can never select a worse layout than the seed alone. The candidate with the
/// fewest routed swaps wins, compared through `isBetter`. No IR is touched
/// here.
class RoutingSearchStrategy {
public:
  RoutingSearchStrategy(const cudaq::Device &device,
                        const RoutingProblem &problem, bool refine,
                        unsigned extendedLayerSize, float extendedLayerWeight,
                        float decayDelta, unsigned roundsDecayReset,
                        unsigned minStallSwapBudget,
                        unsigned stallSwapBudgetPerQubit)
      : device(device), problem(problem), refine(refine),
        extendedLayerSize(extendedLayerSize),
        extendedLayerWeight(extendedLayerWeight), decayDelta(decayDelta),
        roundsDecayReset(roundsDecayReset),
        minStallSwapBudget(minStallSwapBudget),
        stallSwapBudgetPerQubit(stallSwapBudgetPerQubit),
        reverseProblem(refine ? buildReverseProblem(problem)
                              : RoutingProblem{}) {}

  /// The selected routing result and the final placement it produced (used for
  /// the mapping_v2p attributes).
  struct Selection {
    RoutingResult result;
    cudaq::Placement finalLayout;
  };

  /// Route every seed and return the candidate with the fewest swaps.
  Selection run(ArrayRef<SmallVector<unsigned>> seeds, unsigned numV,
                unsigned numPhy) {
    Selection best{RoutingResult{}, cudaq::Placement(numV, numPhy)};
    bool haveBest = false;
    auto consider = [&](RoutingResult result,
                        const cudaq::Placement &finalPlace) {
      if (!haveBest || isBetter(result, best.result)) {
        best.result = std::move(result);
        best.finalLayout = finalPlace;
        haveBest = true;
      }
    };

    for (ArrayRef<unsigned> seed : seeds)
      routeAndRefineSeed(seed, numV, numPhy, consider);
    return best;
  }

private:
  /// Fewer routed swaps wins. Ties keep the earlier candidate.
  static bool isBetter(const RoutingResult &a, const RoutingResult &b) {
    return a.swapCount < b.swapCount;
  }

  /// Route one seed and, when refining, run the SABRE reverse-traversal passes
  /// over it. Every forward route is offered to `consider` as a candidate.
  /// Reverse routes only refine the layout into the next seed and are never
  /// candidates themselves.
  void routeAndRefineSeed(
      ArrayRef<unsigned> seed, unsigned numV, unsigned numPhy,
      llvm::function_ref<void(RoutingResult, const cudaq::Placement &)>
          consider) {
    // SABRE's forward-reverse-forward reverse-traversal refinement (Li et al.
    // 2019, Sec. IV-C2).

    // Traversal 1 (forward): route the seed as given. A candidate.
    cudaq::Placement finalPlace(numV, numPhy);
    consider(routeSeed(seed, numV, numPhy, finalPlace), finalPlace);
    if (!refine)
      return;

    // Traversal 2 (reverse): route the reverse circuit from that layout to
    // refine it into the next seed. Not a candidate.
    SmallVector<unsigned> refined = reverseRefine(finalPlace, numV);

    // Traversal 3 (forward): route the refined seed. A candidate.
    cudaq::Placement refinedFinal(numV, numPhy);
    consider(routeSeed(refined, numV, numPhy, refinedFinal), refinedFinal);
  }

  /// Forward-route a seed layout. Returns its result and final placement.
  RoutingResult routeSeed(ArrayRef<unsigned> seed, unsigned numV,
                          unsigned numPhy, cudaq::Placement &finalOut) {
    cudaq::Placement layout(numV, numPhy);
    for (unsigned v = 0; v < numV; ++v)
      layout.map(cudaq::Placement::VirtualQ(v),
                 cudaq::Placement::DeviceQ(seed[v]));
    SabreRouter router(device, problem, layout, extendedLayerSize,
                       extendedLayerWeight, decayDelta, roundsDecayReset,
                       minStallSwapBudget, stallSwapBudgetPerQubit);
    RoutingResult result = router.route();
    finalOut = layout;
    return result;
  }

  /// Route the reverse circuit from a forward pass's final mapping. The place
  /// each virtual qubit lands becomes the refined seed.
  SmallVector<unsigned> reverseRefine(const cudaq::Placement &startFinal,
                                      unsigned numV) {
    cudaq::Placement layout = startFinal;
    SabreRouter router(device, reverseProblem, layout, extendedLayerSize,
                       extendedLayerWeight, decayDelta, roundsDecayReset,
                       minStallSwapBudget, stallSwapBudgetPerQubit);
    router.route();
    SmallVector<unsigned> refined(numV);
    for (unsigned v = 0; v < numV; ++v)
      refined[v] = layout.getPhy(cudaq::Placement::VirtualQ(v)).index;
    return refined;
  }

  const cudaq::Device &device;
  const RoutingProblem &problem;
  bool refine;
  unsigned extendedLayerSize;
  float extendedLayerWeight;
  float decayDelta;
  unsigned roundsDecayReset;
  unsigned minStallSwapBudget;
  unsigned stallSwapBudgetPerQubit;
  RoutingProblem reverseProblem;
};

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

/// Rewire all 1Q ops (and nested IfOps) in `region` so they use the correct
/// physical wires. `entryPhysicals[i]` is the physical qubit for
/// `entryBlockArgs[i]`; `vqToPhy[vq]` gives the physical qubit for each
/// virtual qubit; `localPhyToWire` (by value) tracks the live wire per
/// physical qubit inside the region.
static void rewireBranchRegion(
    Region &region, ArrayRef<cudaq::Placement::DeviceQ> entryPhysicals,
    ArrayRef<Value> entryBlockArgs, SmallVector<Value> localPhyToWire,
    const SmallVector<unsigned> &vqToPhy,
    const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ) {
  for (auto [phy, arg] : llvm::zip_equal(entryPhysicals, entryBlockArgs))
    localPhyToWire[phy.index] = arg;

  for (Block &block : region) {
    for (Operation &op : block) {
      if (auto nestedIf = dyn_cast<cudaq::cc::IfOp>(op)) {
        SmallVector<cudaq::Placement::DeviceQ> nestedPhys;
        SmallVector<Value> newLinearArgs;
        for (auto arg : nestedIf.getLinearArgs()) {
          auto it = wireToVirtualQ.find(arg);
          if (it == wireToVirtualQ.end())
            continue;
          unsigned phy = vqToPhy[it->second.index];
          nestedPhys.push_back(cudaq::Placement::DeviceQ(phy));
          newLinearArgs.push_back(localPhyToWire[phy]);
        }
        (void)cudaq::quake::setQuantumOperands(nestedIf.getOperation(), newLinearArgs);
        auto thenArgs = nestedIf.getThenEntryArguments();
        rewireBranchRegion(
            nestedIf.getThenRegion(), nestedPhys,
            SmallVector<Value>(thenArgs.begin(), thenArgs.end()),
            localPhyToWire, vqToPhy, wireToVirtualQ);
        if (nestedIf.hasElse()) {
          auto elseArgs = nestedIf.getElseEntryArguments();
          rewireBranchRegion(
              nestedIf.getElseRegion(), nestedPhys,
              SmallVector<Value>(elseArgs.begin(), elseArgs.end()),
              localPhyToWire, vqToPhy, wireToVirtualQ);
        }
        unsigned wireIdx = 0;
        for (Value res : nestedIf->getResults())
          if (isa<cudaq::quake::WireType>(res.getType()))
            localPhyToWire[nestedPhys[wireIdx++].index] = res;
      } else if (cudaq::quake::isSupportedMappingOperation(&op) &&
                 !isa<cudaq::quake::SinkOp, cudaq::quake::ReturnWireOp>(op)) {
        auto wireOps = cudaq::quake::getQuantumOperands(&op);
        SmallVector<Value> newWires;
        for (auto wire : wireOps) {
          auto it = wireToVirtualQ.find(wire);
          if (it == wireToVirtualQ.end()) {
            newWires.push_back(wire);
            continue;
          }
          newWires.push_back(localPhyToWire[vqToPhy[it->second.index]]);
        }
        (void)cudaq::quake::setQuantumOperands(&op, newWires);
        for (auto [result, origWire] :
             llvm::zip_equal(cudaq::quake::getQuantumResults(&op), wireOps)) {
          auto it = wireToVirtualQ.find(origWire);
          if (it == wireToVirtualQ.end())
            continue;
          localPhyToWire[vqToPhy[it->second.index]] = result;
        }
      }
    }
  }
}


/// Applies a RoutingResult to the IR. This is the only place routing rewrites
/// the circuit. It rewires each mapped operation and inserts the swaps,
/// threading the current wire on each physical qubit.
class RoutingEmitter {
public:
  RoutingEmitter(DenseMap<Value, cudaq::Placement::VirtualQ> &wireMap,
                 unsigned numPhysical)
      : wireToVirtualQ(wireMap), phyToWire(numPhysical) {}

  /// Apply `result` to `block`. Returns the final wire on each physical qubit,
  /// which the caller uses to create the return_wire ops.
  ArrayRef<Value> emit(Block &block,
                       ArrayRef<cudaq::quake::BorrowWireOp> sources,
                       const RoutingResult &result) {
    const unsigned numPhy = phyToWire.size();
    SmallVector<unsigned> vqToPhy(sources.size());
    SmallVector<unsigned> phyToVQ(numPhy, UINT_MAX);

    for (auto borrowWire : sources) {
      Value wire = borrowWire.getResult();
      auto vq = requireVirtualQ(wireToVirtualQ, wire);
      unsigned phy = result.initialLayout[vq.index];
      borrowWire.setIdentity(phy);
      phyToWire[phy] = wire;
      vqToPhy[vq.index] = phy;
      phyToVQ[phy] = vq.index;
    }

    OpBuilder builder(&block, block.begin());
    auto wireType = builder.getType<cudaq::quake::WireType>();

    // Build program-order index for each op in the block so we can decide
    // which trace events precede a given IfOp.
    DenseMap<Operation *, unsigned> programOrder;
    unsigned pos = 0;
    for (Operation &op : block)
      programOrder[&op] = pos++;

    // Apply one trace event, updating phyToWire and vqToPhy/phyToVQ.
    auto applyEvent = [&](const RoutingEvent &ev) {
      if (ev.kind == RoutingEvent::Kind::Swap) {
        auto q0 = ev.phys[0], q1 = ev.phys[1];
        auto swap = cudaq::quake::SwapOp::create(
            builder, builder.getUnknownLoc(), TypeRange{wireType, wireType},
            false, ValueRange{}, ValueRange{},
            ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
            DenseBoolArrayAttr{});
        unsigned vq0 = phyToVQ[q0.index], vq1 = phyToVQ[q1.index];
        if (vq0 != UINT_MAX)
          vqToPhy[vq0] = q1.index;
        if (vq1 != UINT_MAX)
          vqToPhy[vq1] = q0.index;
        std::swap(phyToVQ[q0.index], phyToVQ[q1.index]);
        phyToWire[q0.index] = swap.getResult(0);
        phyToWire[q1.index] = swap.getResult(1);
      } else {
        SmallVector<Value, 2> newOpWires;
        for (auto phy : ev.phys)
          newOpWires.push_back(phyToWire[phy.index]);
        [[maybe_unused]] LogicalResult rewired =
            cudaq::quake::setQuantumOperands(ev.op, newOpWires);
        assert(succeeded(rewired) &&
               "rewiring with a fixed operand count cannot fail");
        if (isa<cudaq::quake::SinkOp, cudaq::quake::ReturnWireOp>(ev.op))
          return;
        for (auto &&[w, q] :
             llvm::zip_equal(cudaq::quake::getQuantumResults(ev.op), ev.phys))
          phyToWire[q.index] = w;
      }
    };

    // Walk block ops in program order, flushing trace events as we go.
    // At each IfOp, flush all events for gates that appear before it in the
    // block, then rewire the IfOp's linear args and branch internals.
    unsigned nextTraceIdx = 0;
    for (Operation &op : block) {
      if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
        unsigned ifPos = programOrder[&op];
        while (nextTraceIdx < result.trace.size()) {
          const RoutingEvent &ev = result.trace[nextTraceIdx];
          if (ev.kind == RoutingEvent::Kind::Gate) {
            auto it = programOrder.find(ev.op);
            assert(it != programOrder.end() && "traced gate not in block");
            if (it->second > ifPos)
              break;
          }
          applyEvent(ev);
          ++nextTraceIdx;
        }
        SmallVector<cudaq::Placement::DeviceQ> linPhys;
        SmallVector<Value> newLinearArgs;
        for (auto linArg : ifOp.getLinearArgs()) {
          auto it = wireToVirtualQ.find(linArg);
          if (it == wireToVirtualQ.end())
            continue;
          unsigned phy = vqToPhy[it->second.index];
          linPhys.push_back(cudaq::Placement::DeviceQ(phy));
          newLinearArgs.push_back(phyToWire[phy]);
        }
        (void)cudaq::quake::setQuantumOperands(ifOp, newLinearArgs);
        auto thenArgs = ifOp.getThenEntryArguments();
        rewireBranchRegion(
            ifOp.getThenRegion(), linPhys,
            SmallVector<Value>(thenArgs.begin(), thenArgs.end()),
            SmallVector<Value>(phyToWire.begin(), phyToWire.end()), vqToPhy,
            wireToVirtualQ);
        if (ifOp.hasElse()) {
          auto elseArgs = ifOp.getElseEntryArguments();
          rewireBranchRegion(
              ifOp.getElseRegion(), linPhys,
              SmallVector<Value>(elseArgs.begin(), elseArgs.end()),
              SmallVector<Value>(phyToWire.begin(), phyToWire.end()), vqToPhy,
              wireToVirtualQ);
        }
        unsigned wireIdx = 0;
        for (Value res : ifOp->getResults())
          if (isa<cudaq::quake::WireType>(res.getType()))
            phyToWire[linPhys[wireIdx++].index] = res;
      }
    }
    // Flush any remaining trace events (including the case with no IfOps).
    while (nextTraceIdx < result.trace.size())
      applyEvent(result.trace[nextTraceIdx++]);
    return phyToWire;
  }

private:
  DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ;
  SmallVector<Value> phyToWire;
};

std::pair<bool, std::optional<cudaq::Device>>
deviceFromString(llvm::StringRef deviceString) {
  std::size_t deviceDim[2];
  deviceDim[0] = deviceDim[1] = 0;

  // Get device
  StringRef deviceTopoStr =
      deviceString.take_front(deviceString.find_first_of('('));

  // Trim the dimensions off of `deviceDef` if dimensions were provided in the
  // string
  if (deviceTopoStr.size() < deviceString.size())
    deviceString = deviceString.drop_front(deviceTopoStr.size());

  if (deviceTopoStr.equals_insensitive("file")) {
    StringRef deviceFilename;
    if (deviceString.consume_front("(")) {
      deviceString = deviceString.ltrim();
      if (deviceString.consume_back(")")) {
        deviceFilename = deviceString;
        // Remove any leading and trailing single quotes that may have been
        // added in order to pass files with spaces into the pass (required
        // for parsePassPipeline).
        if (deviceFilename.size() >= 2 && deviceFilename.front() == '\'' &&
            deviceFilename.back() == '\'')
          deviceFilename = deviceFilename.drop_front(1).drop_back(1);
        // Make sure the file exists before continuing
        if (!llvm::sys::fs::exists(deviceFilename)) {
          llvm::errs() << "Path " << deviceFilename << " does not exist\n";
          return std::make_pair(false, std::nullopt);
        }
      } else {
        llvm::errs() << "Missing closing ')' in device option\n";
        return std::make_pair(false, std::nullopt);
      }
    } else {
      llvm::errs() << "Filename must be provided in device option like "
                      "file(/full/path/to/device_file.txt): "
                   << deviceString << '\n';
      return std::make_pair(false, std::nullopt);
    }

    return std::make_pair(false, cudaq::Device::file(deviceFilename));
  } else {
    if (deviceString.consume_front("(")) {
      deviceString = deviceString.ltrim();

      // Parse first dimension
      deviceString.consumeInteger(/*Radix=*/10, deviceDim[0]);
      deviceString = deviceString.ltrim();

      // Parse second dimension if present
      unsigned argCount = 1;
      while (deviceString.consume_front(",")) {
        if (argCount == 2) {
          llvm::errs() << "Too many arguments provided for device\n";
          return std::make_pair(false, std::nullopt);
        }
        deviceString = deviceString.ltrim();
        deviceString.consumeInteger(/*Radix=*/10, deviceDim[1]);
        deviceString = deviceString.ltrim();
        ++argCount;
      }

      if (!deviceString.consume_front(")")) {
        llvm::errs() << "Missing closing ')' in device option\n";
        return std::make_pair(false, std::nullopt);
      }
    }

    if (deviceTopoStr == "path") {
      return std::make_pair(false, cudaq::Device::path(deviceDim[0]));
    } else if (deviceTopoStr == "ring") {
      return std::make_pair(false, cudaq::Device::ring(deviceDim[0]));
    } else if (deviceTopoStr == "star") {
      return std::make_pair(false,
                            cudaq::Device::star(deviceDim[0], deviceDim[1]));
    } else if (deviceTopoStr == "grid") {
      return std::make_pair(false,
                            cudaq::Device::grid(deviceDim[0], deviceDim[1]));
    } else if (deviceTopoStr == "bypass") {
      return std::make_pair(true, std::nullopt);
    } else {
      llvm::errs() << "Unknown device option: " << deviceTopoStr << '\n';
      return std::make_pair(false, std::nullopt);
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct MappingPrep : public cudaq::opt::impl::MappingPrepBase<MappingPrep> {
  using MappingPrepBase::MappingPrepBase;

  std::optional<cudaq::Device> deviceInstance;
  bool deviceBypass = false;

  virtual LogicalResult initialize(MLIRContext *context) override {
    std::tie(deviceBypass, deviceInstance) = deviceFromString(device);
    if (deviceInstance || deviceBypass || !nonComposable) {
      return success();
    }

    signalPassFailure();
    return failure();
  }

  /// Create an adjacency matrix attribute for a WireSetOp.
  SparseElementsAttr getAdjacencyFromDevice(cudaq::Device &d,
                                            MLIRContext *ctx) {
    int numEdges = 0;
    unsigned int qubitCardinality = static_cast<unsigned int>(d.getNumQubits());

    SmallVector<APInt, 32> edgeVector;
    for (unsigned int i = 0; i < qubitCardinality; i++) {
      auto neighbors = d.getNeighbours(cudaq::Device::Qubit(i));
      numEdges += neighbors.size();
      for (auto neighbor : neighbors) {
        edgeVector.emplace_back(64, i);
        edgeVector.emplace_back(64, neighbor.index);
      }
    }

    IntegerType boolTy = IntegerType::get(ctx, /*width=*/1);
    ShapedType tensorI1 =
        RankedTensorType::get({qubitCardinality, qubitCardinality}, boolTy);
    auto indicesType =
        RankedTensorType::get({numEdges, 2}, IntegerType::get(ctx, 64));
    auto indices = DenseIntElementsAttr::get(indicesType, edgeVector);
    auto intValue = mlir::DenseIntElementsAttr::get(
        mlir::RankedTensorType::get({static_cast<int64_t>(numEdges)}, boolTy),
        true);
    auto sparseInt = SparseElementsAttr::get(tensorI1, indices, intValue);

    return sparseInt;
  }

  cudaq::quake::WireSetOp insertWireSetOpForDevice(cudaq::Device &d,
                                                   ModuleOp mod) {
    if (auto wires =
            mod.lookupSymbol<cudaq::quake::WireSetOp>(mappedWireSetName))
      return wires;

    auto adjacency = getAdjacencyFromDevice(d, mod.getContext());
    OpBuilder builder(mod.getBodyRegion());
    auto wireSetOp = cudaq::quake::WireSetOp::create(
        builder, builder.getUnknownLoc(), mappedWireSetName, d.getNumQubits(),
        adjacency);
    wireSetOp.setPrivate();
    return wireSetOp;
  }

  void runOnOperation() override {
    auto mod = getOperation();

    if (deviceBypass)
      return;

    insertWireSetOpForDevice(*deviceInstance, mod);
  }
};

//===----------------------------------------------------------------------===//
// Measurement preconditions
//===----------------------------------------------------------------------===//

/// The first measurement in `func` whose measured wire flows to a non-terminal
/// user, or null. Because the mapper defers measurements to the end of the
/// circuit, a measured wire may only flow to a terminal consumer (`return_wire`
/// or `sink`). Any other use is a mid-circuit measurement the deferral cannot
/// preserve. Classical measurement feedback is detected separately, via
/// `QuakeFunctionAnalysis`.
Operation *findNonTerminalMeasuredWireUse(func::FuncOp func) {
  Operation *found = nullptr;
  func.walk([&](cudaq::quake::MeasurementInterface meas) {
    Operation *measOp = meas.getOperation();
    for (Value result : measOp->getResults()) {
      if (!isa<cudaq::quake::WireType>(result.getType()))
        continue;
      for (Operation *user : result.getUsers())
        if (!isa<cudaq::quake::ReturnWireOp, cudaq::quake::SinkOp>(user)) {
          found = measOp;
          return WalkResult::interrupt();
        }
    }
    return WalkResult::advance();
  });
  return found;
}

struct MappingFunc : public cudaq::opt::impl::MappingFuncBase<MappingFunc> {
  using MappingFuncBase::MappingFuncBase;

  bool deviceBypass = false;
  std::optional<cudaq::Device> deviceInstance;

  virtual LogicalResult initialize(MLIRContext *context) override {
    std::tie(deviceBypass, deviceInstance) = deviceFromString(device);
    if (deviceInstance || deviceBypass || !nonComposable) {
      return success();
    }

    signalPassFailure();
    return failure();
  }

  /// Add `op` and all of its users into `opsToMoveToEnd`. `op` may not be
  /// nullptr.
  void addOpAndUsersToList(Operation *op,
                           SmallVectorImpl<Operation *> &opsToMoveToEnd) {
    opsToMoveToEnd.push_back(op);
    for (auto user : op->getUsers())
      addOpAndUsersToList(user, opsToMoveToEnd);
  }

  /// Resolve `op`'s quantum operands to their virtual qubits. Returns nullopt
  /// after diagnosing (under `nonComposable`) when a supported op consumes an
  /// untracked wire, which means some earlier unsupported op produced it. Do
  /// not let DenseMap default a missing entry to virtual qubit 0.
  std::optional<SmallVector<cudaq::Placement::VirtualQ, 2>>
  lookupVirtualOperands(
      Operation &op, ValueRange wireOperands,
      const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ) {
    SmallVector<cudaq::Placement::VirtualQ, 2> virtualOperands;
    virtualOperands.reserve(wireOperands.size());
    for (Value wire : wireOperands) {
      auto virtualQ = lookupVirtualQ(wireToVirtualQ, wire);
      if (!virtualQ) {
        if (nonComposable) {
          op.emitOpError("has a quantum operand that is not tracked by "
                         "the mapper");
          signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << "untracked quantum operand in mapper\n");
        return std::nullopt;
      }
      virtualOperands.push_back(*virtualQ);
    }
    return virtualOperands;
  }

  /// Map `op`'s result wires onto the virtual qubits carried by its operands,
  /// updating `wireToVirtualQ` and `finalQubitWire`. Fails (diagnosing under
  /// `nonComposable`) when the operand and result wire counts disagree.
  LogicalResult recordQuantumResults(
      Operation &op, ValueRange wireOperands,
      ArrayRef<cudaq::Placement::VirtualQ> virtualOperands,
      DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ,
      DenseMap<std::size_t, Value> &finalQubitWire) {
    auto wireResults = cudaq::quake::getQuantumResults(&op);
    if (!wireResults.empty() && wireResults.size() != wireOperands.size()) {
      if (nonComposable) {
        op.emitOpError("has a different number of quantum operands and "
                       "quantum results");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "quantum operand/result arity mismatch\n");
      return failure();
    }
    for (auto &&[index, newWire] : llvm::enumerate(wireResults)) {
      cudaq::Placement::VirtualQ virtualQ = virtualOperands[index];
      wireToVirtualQ.insert({newWire, virtualQ});
      finalQubitWire[virtualQ.index] = newWire;
    }
    return success();
  }

  void runOnOperation() override {
    if (deviceBypass)
      return;

    auto func = getOperation();
    if (func.empty())
      return;
    auto &blocks = func.getBlocks();

    // Current limitations:
    //  * Can only map a entry-point kernel
    //  * The kernel can only have one block

    auto mod = func->getParentOfType<ModuleOp>();
    auto wireSetOp =
        mod.lookupSymbol<cudaq::quake::WireSetOp>(mappedWireSetName);
    if (!wireSetOp) {
      // Silently return without error if no mapped wire set is found in the
      // module.
      return;
    }

    // Verify that the function contains wiresets and return if it does not.
    // Also populate the highest identity borrow up as long as we're traversing
    // them.
    StringRef inputWireSet;
    std::optional<std::uint32_t> highestIdentity;
    auto walkResult = func.walk([&](cudaq::quake::BorrowWireOp borrowOp) {
      if (inputWireSet.empty()) {
        inputWireSet = borrowOp.getSetName();
      } else if (borrowOp.getSetName() != inputWireSet) {
        // Why is this here? It's entirely possible to have disjoint wire sets,
        // where the sets are for fundamentally distinct purposes in the target
        // model.
        if (nonComposable)
          func.emitOpError("function cannot use multiple WireSets");
        return WalkResult::interrupt();
      }
      highestIdentity = highestIdentity
                            ? std::max(*highestIdentity, borrowOp.getIdentity())
                            : borrowOp.getIdentity();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      if (nonComposable)
        signalPassFailure();
      LLVM_DEBUG(llvm::dbgs()
                 << "NYI: multiple wire sets for a target machine");
      return;
    }
    if (!highestIdentity) {
      if (nonComposable) {
        func.emitOpError("no borrow_wire ops found in " + func.getName());
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "no borrow_wire ops found in " << func.getName() << '\n');
      return;
    }

    // Measurement deferral is only safe for terminal readout. Reject
    // unsupported mid-circuit/adaptive uses before mutating IR, even in
    // composable mode. This must precede the multi-block limitation below, so
    // CFG-shaped adaptive measurements cannot pass through unmapped.
    if (Operation *measOp = findNonTerminalMeasuredWireUse(func)) {
      measOp->emitOpError(
          "unsupported mid-circuit measurement: a measured wire "
          "is used by a later operation");
      signalPassFailure();
      return;
    }
    // Measurement-dependent behavior is the adaptive shape the mapper cannot
    // preserve, so use AddMetadata's conservative measurement-dependence
    // analysis.
    const auto &measAnalysis =
        getAnalysis<cudaq::quake::detail::QuakeFunctionAnalysis>();
    const auto &measInfo = measAnalysis.getAnalysisInfo();
    auto measIt = measInfo.find(func);
    assert(measIt != measInfo.end() && "missing measurement analysis for func");
    if (measIt->second.hasConditionalsOnMeasure) {
      func.emitOpError(
          "unsupported measurement-dependent behavior: "
          "measurement-dependent control flow, quantum operations, "
          "calls, or resets cannot be preserved by qubit mapping");
      signalPassFailure();
      return;
    }

    // FIXME: Add the ability to handle multiple blocks.
    if (blocks.size() > 1) {
      if (nonComposable) {
        func.emitError("The mapper cannot handle multiple blocks");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "NYI: mapping with multiple blocks");
      return;
    }

    // Sanity checks and create a wire to virtual qubit mapping.
    Block &block = *blocks.begin();

    if (deviceInstance->getNumQubits() == 0) {
      if (nonComposable) {
        func.emitError("Trying to target an empty device.");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "device cannot be empty");
      return;
    }

    LLVM_DEBUG({ deviceInstance->dump(); });

    const std::size_t deviceNumQubits = deviceInstance->getNumQubits();

    SmallVector<cudaq::quake::BorrowWireOp> sources(deviceNumQubits);
    SmallVector<cudaq::quake::ReturnWireOp> returnsToRemove;
    DenseMap<Value, cudaq::Placement::VirtualQ> wireToVirtualQ;
    SmallVector<std::size_t> userQubitsMeasured;
    DenseMap<std::size_t, Value> finalQubitWire;
    Operation *lastSource = nullptr;

    // Resolve the placement and search strategies before deciding whether to
    // collect interaction data.
    std::optional<PlacementStrategy> parsedPlacement =
        parsePlacementStrategy(this->placement);
    if (!parsedPlacement) {
      if (nonComposable) {
        func.emitError("unknown qubit-mapping placement strategy '" +
                       this->placement + "'");
        signalPassFailure();
        return;
      }
      func.emitWarning("unknown qubit-mapping placement strategy '" +
                       this->placement + "'; using 'identity'");
    }
    PlacementStrategy placementStrategy =
        parsedPlacement.value_or(PlacementStrategy::Identity);

    std::optional<SearchStrategy> parsedSearch =
        parseSearchStrategy(this->search);
    if (!parsedSearch) {
      if (nonComposable) {
        func.emitError("unknown qubit-mapping search strategy '" +
                       this->search + "'");
        signalPassFailure();
        return;
      }
      func.emitWarning("unknown qubit-mapping search strategy '" +
                       this->search + "'; using 'none'");
    }
    SearchStrategy searchStrategy = parsedSearch.value_or(SearchStrategy::None);

    // Reject any cc::IfOp branch that contains a 2Q gate — Stage 1 only
    // handles 1Q gates inside branches (no placement reconciliation needed).
    auto branchCheckResult = func.walk([&](cudaq::cc::IfOp ifOp) {
      for (Region *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
        for (Block &b : *region) {
          for (Operation &op : b) {
            if (isa<cudaq::quake::OperatorInterface>(op) &&
                !isa<cudaq::cc::IfOp>(op) &&
                cudaq::quake::getQuantumOperands(&op).size() > 1) {
              if (nonComposable) {
                op.emitOpError(
                    "mapper cannot handle 2-qubit gates inside branches");
                signalPassFailure();
              }
              return WalkResult::interrupt();
            }
          }
        }
      }
      return WalkResult::advance();
    });
    if (branchCheckResult.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << "NYI: 2-qubit gates inside branches\n");
      return;
    }

    bool collectInteractions = placementStrategy != PlacementStrategy::Identity;

    // Two-qubit interaction data for placement, collected during the scan.
    std::optional<VirtualInteractionGraph> interactions;
    SmallVector<bool> userVirtualQubits;
    if (collectInteractions) {
      interactions.emplace(deviceNumQubits);
      userVirtualQubits.assign(deviceNumQubits, false);
    }

    // Recursive analysis: walk `b` and all IfOp branch regions, populating
    // wireToVirtualQ, finalQubitWire, sources, and interaction data.
    bool analysisOk = true;
    std::function<void(Block &, bool)> analyzeBlock =
        [&](Block &b, bool doCollectInteractions) {
          if (!analysisOk)
            return;
          for (Operation &op : b.getOperations()) {
            if (auto qop = dyn_cast<cudaq::quake::BorrowWireOp>(op)) {
              auto id = qop.getIdentity();
              wireToVirtualQ[qop.getResult()] = cudaq::Placement::VirtualQ(id);
              finalQubitWire[id] = qop.getResult();
              sources[id] = qop;
              if (doCollectInteractions)
                userVirtualQubits[id] = true;
              lastSource = &op;
            } else if (dyn_cast<cudaq::quake::NullWireOp>(op)) {
              if (nonComposable) {
                op.emitOpError("the mapper requires borrow operations and "
                               "prohibits null wires");
                signalPassFailure();
              }
              LLVM_DEBUG(llvm::dbgs() << "null_wire ops are not expected");
              analysisOk = false;
              return;
            } else if (dyn_cast<cudaq::quake::AllocaOp>(op)) {
              if (nonComposable) {
                op.emitOpError(
                    "the mapper requires borrow operations and prohibits "
                    "reference semantics");
                signalPassFailure();
              }
              LLVM_DEBUG(llvm::dbgs()
                         << "quantum reference semantics not expected");
              analysisOk = false;
              return;
            } else if (isa<cudaq::cc::ContinueOp>(op)) {
              // Branch terminator: wires pass to IfOp results, tracked above.
            } else if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
              auto linearArgs = ifOp.getLinearArgs();
              auto thenArgs = ifOp.getThenEntryArguments();
              for (auto [linArg, thenArg] :
                   llvm::zip_equal(linearArgs, thenArgs))
                wireToVirtualQ.insert({thenArg, wireToVirtualQ[linArg]});
              for (Block &innerBlock : ifOp.getThenRegion())
                analyzeBlock(innerBlock, /*doCollectInteractions=*/false);
              if (ifOp.hasElse()) {
                for (auto [linArg, elseArg] :
                     llvm::zip_equal(linearArgs, ifOp.getElseEntryArguments()))
                  wireToVirtualQ.insert({elseArg, wireToVirtualQ[linArg]});
                for (Block &innerBlock : ifOp.getElseRegion())
                  analyzeBlock(innerBlock, /*doCollectInteractions=*/false);
              }
              unsigned linearIdx = 0;
              for (Value res : ifOp->getResults()) {
                if (isa<cudaq::quake::WireType>(res.getType())) {
                  auto virt = wireToVirtualQ[linearArgs[linearIdx++]];
                  wireToVirtualQ.insert({res, virt});
                  finalQubitWire[virt.index] = res;
                }
              }
            } else if (cudaq::quake::isSupportedMappingOperation(&op)) {
              if (!cudaq::quake::isLinearValueForm(&op)) {
                if (nonComposable) {
                  llvm::errs() << "This is not SSA form: " << op << '\n';
                  llvm::errs() << "isa<cudaq::quake::NullWireOp>() = "
                               << isa<cudaq::quake::NullWireOp>(&op) << '\n';
                  llvm::errs() << "isAllReferences() = "
                               << cudaq::quake::isAllReferences(&op) << '\n';
                  llvm::errs() << "isWrapped() = "
                               << cudaq::quake::isWrapped(&op) << '\n';
                  func.emitError("The mapper requires value semantics.");
                  signalPassFailure();
                }
                LLVM_DEBUG(llvm::dbgs()
                           << "operation is not in proper value form");
                analysisOk = false;
                return;
              }
              auto wireOperands = cudaq::quake::getQuantumOperands(&op);
              auto maybeVirtualOperands =
                  lookupVirtualOperands(op, wireOperands, wireToVirtualQ);
              if (!maybeVirtualOperands) {
                analysisOk = false;
                return;
              }
              SmallVector<cudaq::Placement::VirtualQ, 2> virtualOperands =
                  std::move(*maybeVirtualOperands);
              if (auto rop = dyn_cast<cudaq::quake::ReturnWireOp>(op)) {
                returnsToRemove.push_back(rop);
                continue;
              }
              if (!op.hasTrait<cudaq::QuantumMeasure>() &&
                  wireOperands.size() > 2) {
                if (nonComposable) {
                  func.emitError(
                      "Cannot map a kernel with operators that use more "
                      "than two qubits.");
                  signalPassFailure();
                }
                LLVM_DEBUG(llvm::dbgs()
                           << "operator with >2 qubits not expected");
                analysisOk = false;
                return;
              }
              if (isa<cudaq::quake::MeasurementInterface>(op))
                for (auto virtualQ : virtualOperands)
                  userQubitsMeasured.push_back(virtualQ.index);
              if (doCollectInteractions &&
                  !isa<cudaq::quake::MeasurementInterface>(op) &&
                  wireOperands.size() == 2) {
                interactions->addInteraction(virtualOperands[0].index,
                                             virtualOperands[1].index);
              }
              if (failed(recordQuantumResults(op, wireOperands, virtualOperands,
                                              wireToVirtualQ, finalQubitWire))) {
                analysisOk = false;
                return;
              }
            } else if (!cudaq::quake::getQuantumOperands(&op).empty() ||
                       !cudaq::quake::getQuantumResults(&op).empty()) {
              if (nonComposable) {
                op.emitOpError("is not supported by the mapper");
                signalPassFailure();
              }
              LLVM_DEBUG(llvm::dbgs()
                         << "unsupported quantum operation in mapper\n");
              analysisOk = false;
              return;
            }
          }
        };
    analyzeBlock(block, collectInteractions);
    if (!analysisOk)
      return;

    if (sources.size() > deviceNumQubits) {
      if (nonComposable) {
        func.emitOpError("Too many qubits [" + std::to_string(sources.size()) +
                         "] for device [" + std::to_string(deviceNumQubits) +
                         "]");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "exceeded available qubits for target");
      return;
    }

    // Make all existing borrow_wire ops use the mapped wire set.
    func.walk([&](cudaq::quake::BorrowWireOp borrowOp) {
      borrowOp.setSetName(mappedWireSetName);
    });

    // We've made it past all the initial checks. Remove the returns now. They
    // will be added back in when the mapping is complete.
    for (auto ret : returnsToRemove)
      ret.erase();
    returnsToRemove.clear();

    OpBuilder builder(&block, block.begin());
    auto wireTy = builder.getType<cudaq::quake::WireType>();
    auto unknownLoc = builder.getUnknownLoc();

    // Add implicit measurements if necessary
    if (userQubitsMeasured.empty()) {
      builder.setInsertionPoint(block.getTerminator());
      auto measTy = cudaq::quake::MeasureType::get(builder.getContext());
      Type resTy = builder.getI1Type();
      for (unsigned i = 0; i < sources.size(); i++) {
        if (sources[i] != nullptr) {
          auto measureOp = cudaq::quake::MzOp::create(
              builder, finalQubitWire[i].getLoc(), TypeRange{measTy, wireTy},
              finalQubitWire[i]);
          cudaq::quake::DiscriminateOp::create(builder,
                                               finalQubitWire[i].getLoc(),
                                               resTy, measureOp.getMeasOut());

          wireToVirtualQ.insert(
              {measureOp.getWires()[0],
               requireVirtualQ(wireToVirtualQ, finalQubitWire[i])});

          userQubitsMeasured.push_back(i);
        }
      }
    }

    // Save the order of the measurements. They are not allowed to change.
    SmallVector<mlir::Operation *> measureOrder;
    func.walk([&](cudaq::quake::MeasurementInterface measure) {
      measureOrder.push_back(measure);
      for (auto user : measure->getUsers())
        measureOrder.push_back(user);
      return WalkResult::advance();
    });

    // Create or borrow auxillary qubits if needed. Place them after the last
    // allocated qubit.
    builder.setInsertionPointAfter(lastSource);
    for (unsigned i = 0; i < deviceInstance->getNumQubits(); i++) {
      if (!sources[i]) {
        auto borrowOp = cudaq::quake::BorrowWireOp::create(
            builder, unknownLoc, wireTy, mappedWireSetName, i);
        wireToVirtualQ[borrowOp.getResult()] = cudaq::Placement::VirtualQ(i);
        sources[i] = borrowOp;
      }
    }

    const unsigned numV = sources.size();
    const unsigned numPhy = deviceInstance->getNumQubits();

    SmallVector<SmallVector<unsigned>> seeds =
        buildPlacementSeeds(placementStrategy, numV, *deviceInstance,
                            interactions, userVirtualQubits);

    // Build the routing problem once (it does not depend on the layout), then
    // search over the seeds for the result with the fewest swaps.
    RoutingProblem problem =
        buildRoutingProblem(block, sources, wireToVirtualQ);
    RoutingSearchStrategy search(
        *deviceInstance, problem, searchStrategy == SearchStrategy::Sabre,
        extendedLayerSize, extendedLayerWeight, decayDelta, roundsDecayReset,
        minStallSwapBudget, stallSwapBudgetPerQubit);
    RoutingSearchStrategy::Selection selection =
        search.run(seeds, numV, numPhy);
    RoutingResult &best = selection.result;
    cudaq::Placement &bestLayout = selection.finalLayout;

    // Emit the selected result onto the IR exactly once.
    RoutingEmitter emitter(wireToVirtualQ, numPhy);
    auto phyToWire = emitter.emit(block, sources, best);
    sortTopologically(&block);

    // Ensure that the original measurement ordering is still honored by moving
    // the measurements to the end (in their original order). Note that we must
    // move the users of those measurements to the end as well.
    for (Operation *measure : measureOrder) {
      SmallVector<Operation *> opsToMoveToEnd;
      addOpAndUsersToList(measure, opsToMoveToEnd);
      for (Operation *op : opsToMoveToEnd)
        block.getOperations().splice(std::prev(block.end()),
                                     block.getOperations(), op->getIterator());
    }

    // Remove any unused BorrowWireOps and add ReturnWireOp's where needed. Each
    // source starts on physical qubit `initialLayout[i]`, so its final wire is
    // the one threaded onto that track.
    builder.setInsertionPoint(block.getTerminator());
    for (const auto &[i, s] : llvm::enumerate(sources)) {
      if (s->getUsers().empty()) {
        s->erase();
      } else {
        Value finalWire = phyToWire[best.initialLayout[i]];
        cudaq::quake::ReturnWireOp::create(builder, finalWire.getLoc(),
                                           finalWire);
      }
    }

    // Populate mapping_v2p attribute on this function such that:
    // - mapping_v2p[v] contains the final physical qubit placement for virtual
    //   qubit `v`.
    // To map the backend qubits back to the original user program (i.e. before
    // this pass), run something like this:
    //   for (int v = 0; v < numQubits; v++)
    //     dataForOriginalQubit[v] = dataFromBackendQubit[mapping_v2p[v]];
    llvm::SmallVector<Attribute> attrs(*highestIdentity + 1);
    for (unsigned int v = 0; v < *highestIdentity + 1; v++)
      attrs[v] = IntegerAttr::get(
          builder.getIntegerType(64),
          bestLayout.getPhy(cudaq::Placement::VirtualQ(v)).index);

    func->setAttr("mapping_v2p", builder.getArrayAttr(attrs));

    // Now populate mapping_reorder_idx attribute. This attribute will be used
    // by downstream processing to reconstruct a global register as if mapping
    // had not occurred. This is important because the global register is
    // required to be sorted by qubit allocation order, and mapping can change
    // that apparent order AND introduce ancilla qubits that we don't want to
    // appear in the final global register.

    // pair is <first=virtual, second=physical>
    using VirtPhyPairType = std::pair<std::size_t, std::size_t>;
    llvm::SmallVector<VirtPhyPairType> measuredQubits;
    measuredQubits.reserve(userQubitsMeasured.size());
    for (auto mq : userQubitsMeasured) {
      measuredQubits.emplace_back(
          mq, bestLayout.getPhy(cudaq::Placement::VirtualQ(mq)).index);
    }
    // First sort the pairs according to the physical qubits.
    llvm::sort(measuredQubits,
               [&](const VirtPhyPairType &a, const VirtPhyPairType &b) {
                 return a.second < b.second;
               });
    // Now find out how to reorder `measuredQubits` such that the elements are
    // ordered based on the *virtual* qubits (i.e. measuredQubits[].first).
    llvm::SmallVector<std::size_t> reorder_idx(measuredQubits.size());
    for (std::size_t ix = 0; auto &element : reorder_idx)
      element = ix++;
    llvm::sort(reorder_idx, [&](const std::size_t &i1, const std::size_t &i2) {
      return measuredQubits[i1].first < measuredQubits[i2].first;
    });
    // After kernel execution is complete, you can pass reorder_idx[] into
    // sample_result::reorder() in order to undo the ordering change to the
    // global register that the mapping pass induced.
    llvm::SmallVector<Attribute> mapping_reorder_idx(reorder_idx.size());
    for (std::size_t ix = 0; auto &element : mapping_reorder_idx)
      element = IntegerAttr::get(builder.getIntegerType(64), reorder_idx[ix++]);

    func->setAttr("mapping_reorder_idx",
                  builder.getArrayAttr(mapping_reorder_idx));
  }
};

} // namespace

namespace cudaq::opt {
/// This options structure is mostly a mirror copy of the options in
/// MappingFunc, but we've also added the `device` option from MappingPrep.
struct MappingPipelineOptions
    : public PassPipelineOptions<MappingPipelineOptions> {

#define DECLARE_SUB_OPTION(_PARENT_STRUCT, _FIELD, _NAME)                      \
  PassOptions::Option<decltype(_PARENT_STRUCT::_FIELD)> _FIELD{*this, _NAME}
  DECLARE_SUB_OPTION(MappingPrepOptions, device, "device");
  DECLARE_SUB_OPTION(MappingFuncOptions, extendedLayerSize,
                     "extended-layer-size");
  DECLARE_SUB_OPTION(MappingFuncOptions, extendedLayerWeight,
                     "extended-layer-weight");
  DECLARE_SUB_OPTION(MappingFuncOptions, decayDelta, "decay-delta");
  DECLARE_SUB_OPTION(MappingFuncOptions, roundsDecayReset,
                     "rounds-decay-reset");
  DECLARE_SUB_OPTION(MappingFuncOptions, minStallSwapBudget,
                     "min-stall-swap-budget");
  DECLARE_SUB_OPTION(MappingFuncOptions, stallSwapBudgetPerQubit,
                     "stall-swap-budget-per-qubit");
  DECLARE_SUB_OPTION(MappingFuncOptions, placement, "placement");
  DECLARE_SUB_OPTION(MappingFuncOptions, search, "search");
  PassOptions::Option<bool> nonComposable{*this, "raise-fatal-errors"};
};

/// Register the mapping pipeline. Route the appropriate options to the
/// appropriate pass in the pass pipeline.
void registerMappingPipeline() {
  PassPipelineRegistration<cudaq::opt::MappingPipelineOptions>(
      "qubit-mapping", "Perform qubit mapping pass pipeline.",
      [](OpPassManager &pm, const MappingPipelineOptions &opt) {
        auto setIt = [](auto &to, const auto &from) {
          if (from.hasValue())
            to = from;
        };

        // Add the prep pass
        MappingPrepOptions prepOpts;
        setIt(prepOpts.device, opt.device);
        setIt(prepOpts.nonComposable, opt.nonComposable);
        pm.addPass(cudaq::opt::createMappingPrep(prepOpts));

        // Add the per-function pass
        MappingFuncOptions funcOpts;
        setIt(funcOpts.device, opt.device);
        setIt(funcOpts.extendedLayerSize, opt.extendedLayerSize);
        setIt(funcOpts.extendedLayerWeight, opt.extendedLayerWeight);
        setIt(funcOpts.decayDelta, opt.decayDelta);
        setIt(funcOpts.roundsDecayReset, opt.roundsDecayReset);
        setIt(funcOpts.minStallSwapBudget, opt.minStallSwapBudget);
        setIt(funcOpts.stallSwapBudgetPerQubit, opt.stallSwapBudgetPerQubit);
        setIt(funcOpts.placement, opt.placement);
        setIt(funcOpts.search, opt.search);
        setIt(funcOpts.nonComposable, opt.nonComposable);
        pm.addNestedPass<func::FuncOp>(cudaq::opt::createMappingFunc(funcOpts));
      });
}
} // namespace cudaq::opt
