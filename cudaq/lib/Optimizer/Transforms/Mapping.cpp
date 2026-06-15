/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PassDetails.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Device.h"
#include "cudaq/Support/Handle.h"
#include "cudaq/Support/Placement.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
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

/// Builds a deterministic topology-aware initial layout by assigning highly
/// interacting virtual qubits to central physical qubits first. The greedy
/// growth and its tie-breaks make the layout a deterministic function of the
/// interaction counts and the device, so it is reproducible across runs.
class GreedyInitialPlacer {
public:
  GreedyInitialPlacer(const cudaq::Device &device,
                      ArrayRef<SmallVector<unsigned>> interactions,
                      ArrayRef<bool> userVirtualQubits)
      : device(device), interactions(interactions),
        userVirtualQubits(userVirtualQubits), n(device.getNumQubits()),
        vrToPhy(n, 0), placedVirtual(n, false), usedPhysical(n, false) {}

  /// Produce the `vrToPhy` seed layout.
  SmallVector<unsigned> run() {
    computeDegrees();

    // No two-qubit interactions, so every layout routes identically; return the
    // identity seed for a deterministic result.
    if (!hasInteraction) {
      for (unsigned v = 0; v < n; ++v)
        vrToPhy[v] = v;
      return vrToPhy;
    }

    computeCentrality();

    // Seed the highest-degree virtual qubit onto the most central physical
    // qubit, then grow the layout around it.
    place(chooseSeedVirtual(), bestFreePhysical());

    unsigned remaining = 0;
    for (unsigned u = 0; u < n; ++u)
      if (userVirtualQubits[u] && !placedVirtual[u])
        ++remaining;
    while (remaining > 0) {
      unsigned v = chooseNextVirtual();
      place(v, bestPhysicalFor(v));
      --remaining;
    }

    assignRemainingVirtuals();
    return vrToPhy;
  }

private:
  /// Weighted degree of each virtual qubit: the sum of its two-qubit
  /// interaction counts. Also records whether any interaction exists at all.
  void computeDegrees() {
    weightedDegree.assign(n, 0);
    for (unsigned u = 0; u < n; ++u) {
      for (unsigned v = 0; v < n; ++v)
        weightedDegree[u] += interactions[u][v];
      if (weightedDegree[u] > 0)
        hasInteraction = true;
    }
  }

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

  /// More central first: lower distance-sum, then higher degree, then lower
  /// index.
  bool physBetter(unsigned a, unsigned b) const {
    if (distanceSum[a] != distanceSum[b])
      return distanceSum[a] < distanceSum[b];
    if (physDegree[a] != physDegree[b])
      return physDegree[a] > physDegree[b];
    return a < b;
  }

  /// The most central physical qubit not yet used.
  unsigned bestFreePhysical() const {
    unsigned best = n;
    for (unsigned p = 0; p < n; ++p)
      if (!usedPhysical[p] && (best == n || physBetter(p, best)))
        best = p;
    return best;
  }

  /// The highest weighted-degree virtual qubit, breaking ties toward the lower
  /// index.
  unsigned chooseSeedVirtual() const {
    unsigned seed = n;
    for (unsigned u = 0; u < n; ++u)
      if (userVirtualQubits[u] &&
          (seed == n || weightedDegree[u] > weightedDegree[seed]))
        seed = u;
    return seed;
  }

  /// The unplaced virtual qubit most connected to the placed ones, falling back
  /// to weighted degree for disconnected components.
  unsigned chooseNextVirtual() const {
    unsigned pick = n;
    unsigned pickWeight = 0;
    for (unsigned u = 0; u < n; ++u) {
      if (!userVirtualQubits[u] || placedVirtual[u])
        continue;
      unsigned placedWeight = 0;
      for (unsigned v = 0; v < n; ++v)
        if (placedVirtual[v])
          placedWeight += interactions[u][v];
      if (pick == n || placedWeight > pickWeight) {
        pick = u;
        pickWeight = placedWeight;
      }
    }
    if (pickWeight == 0) {
      pick = n;
      unsigned pickDegree = 0;
      for (unsigned u = 0; u < n; ++u) {
        if (!userVirtualQubits[u] || placedVirtual[u])
          continue;
        if (pick == n || weightedDegree[u] > pickDegree) {
          pick = u;
          pickDegree = weightedDegree[u];
        }
      }
    }
    return pick;
  }

  /// The free physical qubit minimizing weighted distance from `v` to its
  /// placed partners, breaking ties by centrality.
  unsigned bestPhysicalFor(unsigned v) const {
    using Qubit = cudaq::Device::Qubit;
    unsigned bestPhy = n;
    unsigned bestCost = 0;
    for (unsigned p = 0; p < n; ++p) {
      if (usedPhysical[p])
        continue;
      unsigned cost = 0;
      for (unsigned w = 0; w < n; ++w)
        if (placedVirtual[w] && interactions[v][w] > 0)
          cost += interactions[v][w] *
                  device.getDistance(Qubit(p), Qubit(vrToPhy[w]));
      bool better = bestPhy == n || cost < bestCost ||
                    (cost == bestCost && physBetter(p, bestPhy));
      if (better) {
        bestPhy = p;
        bestCost = cost;
      }
    }
    return bestPhy;
  }

  /// Map virtual `v` onto physical `p` and mark both as taken.
  void place(unsigned v, unsigned p) {
    vrToPhy[v] = p;
    placedVirtual[v] = true;
    usedPhysical[p] = true;
  }

  /// Assign any still-unplaced virtuals (non-user qubits) to the remaining free
  /// physicals in order.
  void assignRemainingVirtuals() {
    unsigned nextPhy = 0;
    for (unsigned v = 0; v < n; ++v) {
      if (placedVirtual[v])
        continue;
      while (nextPhy < n && usedPhysical[nextPhy])
        ++nextPhy;
      place(v, nextPhy);
    }
  }

  const cudaq::Device &device;
  ArrayRef<SmallVector<unsigned>> interactions;
  ArrayRef<bool> userVirtualQubits;
  const unsigned n;

  SmallVector<unsigned> weightedDegree;
  bool hasInteraction = false;
  SmallVector<unsigned> distanceSum;
  SmallVector<unsigned> physDegree;

  SmallVector<bool> placedVirtual;
  SmallVector<bool> usedPhysical;
  SmallVector<unsigned> vrToPhy;
};

/// Greedy initial placement over the circuit interaction graph. Returns a
/// `vrToPhy` array proposing a starting layout for the router (the greedy
/// seed).
SmallVector<unsigned>
interactionPlacement(const cudaq::Device &device,
                     ArrayRef<SmallVector<unsigned>> interactions,
                     ArrayRef<bool> userVirtualQubits) {
  return GreedyInitialPlacer(device, interactions, userVirtualQubits).run();
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
      node.qubits.push_back(wireToVirtualQ.lookup(wire));
    node.isMeasure = op.hasTrait<cudaq::QuantumMeasure>();
    node.isUnitary = isa<cudaq::quake::OperatorInterface>(op);
    // A two-qubit gate the router has to make adjacent: a unitary on two wires,
    // not a measurement or a sink.
    node.isTwoQ = node.isUnitary && node.qubits.size() == 2;
    nodeIndex[&op] = RoutingProblem::NodeRef(problem.nodes.size());
    problem.nodes.push_back(std::move(node));
  }

  // Record successor edges in use-list order, keeping the routable users only.
  // A user is listed once per result wire it consumes, so a node's visit count
  // reaches its wire-operand count exactly when all of its inputs are ready.
  auto recordUsers = [&](Operation *producer,
                         SmallVectorImpl<RoutingProblem::NodeRef> &out) {
    for (Operation *user : producer->getUsers())
      if (auto it = nodeIndex.find(user); it != nodeIndex.end())
        out.push_back(it->second);
  };
  for (auto &node : problem.nodes)
    recordUsers(node.op, node.successors);
  for (auto borrow : sources)
    recordUsers(borrow.getOperation(), problem.sourceUsers);

  return problem;
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
    if (!node.isUnitary)
      continue;
    fwdToRev[i] = RoutingProblem::NodeRef(reverse.nodes.size());
    RoutingProblem::Node rev;
    rev.op = node.op;
    rev.qubits = node.qubits;
    rev.isUnitary = true;
    rev.isTwoQ = node.isTwoQ;
    reverse.nodes.push_back(std::move(rev));
  }

  for (unsigned i = 0, end = forward.nodes.size(); i < end; ++i) {
    const RoutingProblem::Node &node = forward.nodes[i];
    if (!node.isUnitary)
      continue;
    unsigned unitarySuccessors = 0;
    for (RoutingProblem::NodeRef s : node.successors) {
      if (!forward[s].isUnitary)
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
              unsigned roundsDecayReset)
      : device(device), problem(problem), placement(placement),
        extendedLayerSize(extendedLayerSize),
        extendedLayerWeight(extendedLayerWeight), decayDelta(decayDelta),
        roundsDecayReset(roundsDecayReset),
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

private:
  const cudaq::Device &device;
  const RoutingProblem &problem;
  cudaq::Placement &placement;

  // Parameters
  const unsigned extendedLayerSize;
  const float extendedLayerWeight;
  const float decayDelta;
  const unsigned roundsDecayReset;

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

  auto addSwap = [&](cudaq::Placement::DeviceQ q0,
                     cudaq::Placement::DeviceQ q1) {
    placement.swap(q0, q1);
    result.trace.push_back(RoutingEvent::swap(q0, q1));
    ++result.swapCount;
  };

  // Release valve: bring the closest front-layer gate together along a shortest
  // path, ignoring the heuristic. SABRE's decay only softly discourages the
  // local minima the heuristic can fall into, so a stuck front layer would
  // otherwise loop forever; forcing the gate guarantees progress. This follows
  // the release-valve idea from Qiskit/LightSABRE (arXiv:2409.08368).
  auto forceClosestGate = [&]() {
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
    assert(closest.isValid() &&
           "a stalled front layer must hold a 2-qubit gate");
    const RoutingProblem::Node &node = problem[closest];
    cudaq::Device::Path path = device.getShortestPath(
        placement.getPhy(node.qubits[0]), placement.getPhy(node.qubits[1]));
    // Move one qubit along the path until it is adjacent to the other.
    for (unsigned i = 0; i + 2 < path.size(); ++i)
      addSwap(path[i], path[i + 1]);
  };

  // SABRE's cost function is a heuristic. If it emits a long run of swaps
  // without making any front-layer gate executable, discard that local episode
  // and force one gate along a shortest path. This budget is deliberately
  // loose: bringing one front-layer gate adjacent costs at most the device
  // diameter, and the qubit count upper-bounds the diameter of a connected
  // device. The multiplier gives the heuristic several times that worst-case
  // direct-routing cost to explore and recover before the release valve fires.
  // The floor keeps a usable budget on small devices, where the scaled term
  // would otherwise be too tight.
  constexpr unsigned minStallSwapBudget = 64;
  constexpr unsigned stallSwapBudgetPerQubit = 4;
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
      // Unwind the heuristic swaps back to the last routed gate, then force the
      // closest gate together so the walk always makes progress. The decay
      // state is left as is; it is a soft heuristic and resets on its own
      // cycle.
      for (unsigned i = episodeSwaps.size(); i-- > 0;)
        placement.swap(episodeSwaps[i].first, episodeSwaps[i].second);
      result.trace.pop_back_n(episodeSwaps.size());
      result.swapCount -= episodeSwaps.size();
      episodeSwaps.clear();
      forceClosestGate();
      swapsSinceRouted = 0;
      involvedPhy.clear();
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
  /// Reverse-traversal initial-mapping refinement from the SABRE paper (Li et
  /// al. 2019, Sec. IV-C2): route forward, route backward from the resulting
  /// layout, then forward again.
  static constexpr unsigned numTraversals = 3;

public:
  RoutingSearchStrategy(const cudaq::Device &device,
                        const RoutingProblem &problem, bool refine,
                        unsigned extendedLayerSize, float extendedLayerWeight,
                        float decayDelta, unsigned roundsDecayReset)
      : device(device), problem(problem), refine(refine),
        extendedLayerSize(extendedLayerSize),
        extendedLayerWeight(extendedLayerWeight), decayDelta(decayDelta),
        roundsDecayReset(roundsDecayReset),
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

    for (ArrayRef<unsigned> seed : seeds) {
      cudaq::Placement finalPlace(numV, numPhy);
      consider(routeSeed(seed, numV, numPhy, finalPlace), finalPlace);
      if (!refine)
        continue;
      // Forward is done. Alternate backward/forward for the remaining
      // traversals, keeping every forward result as a candidate.
      SmallVector<unsigned> current(seed.begin(), seed.end());
      cudaq::Placement currentFinal = finalPlace;
      for (unsigned t = 2; t <= numTraversals; ++t) {
        if (t % 2 == 0) {
          current = reverseRefine(currentFinal, numV);
        } else {
          cudaq::Placement nextFinal(numV, numPhy);
          consider(routeSeed(current, numV, numPhy, nextFinal), nextFinal);
          currentFinal = nextFinal;
        }
      }
    }
    return best;
  }

private:
  /// Fewer routed swaps wins. Ties keep the earlier candidate.
  static bool isBetter(const RoutingResult &a, const RoutingResult &b) {
    return a.swapCount < b.swapCount;
  }

  /// Forward-route a seed layout. Returns its result and final placement.
  RoutingResult routeSeed(ArrayRef<unsigned> seed, unsigned numV,
                          unsigned numPhy, cudaq::Placement &finalOut) {
    cudaq::Placement layout(numV, numPhy);
    for (unsigned v = 0; v < numV; ++v)
      layout.map(cudaq::Placement::VirtualQ(v),
                 cudaq::Placement::DeviceQ(seed[v]));
    SabreRouter router(device, problem, layout, extendedLayerSize,
                       extendedLayerWeight, decayDelta, roundsDecayReset);
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
                       extendedLayerWeight, decayDelta, roundsDecayReset);
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
  RoutingProblem reverseProblem;
};

//===----------------------------------------------------------------------===//
// Emission
//===----------------------------------------------------------------------===//

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
    // Place each source wire on its physical qubit under the initial layout.
    // Backends read a wire's physical qubit from its borrow identity, so the
    // layout has to be written into that identity here: a non-identity layout
    // is not materialized by phyToWire tracking alone. The layout is a
    // permutation, so the rewritten identities stay distinct and within the
    // device range.
    for (auto borrowWire : sources) {
      Value wire = borrowWire.getResult();
      unsigned phy = result.initialLayout[wireToVirtualQ[wire].index];
      borrowWire.setIdentity(phy);
      phyToWire[phy] = wire;
    }

    OpBuilder builder(&block, block.begin());
    auto wireType = builder.getType<cudaq::quake::WireType>();
    for (const RoutingEvent &ev : result.trace) {
      if (ev.kind == RoutingEvent::Kind::Gate) {
        // Rewire the operation onto its physical qubits.
        SmallVector<Value, 2> newOpWires;
        for (auto phy : ev.phys)
          newOpWires.push_back(phyToWire[phy.index]);
        // The operand count is unchanged, so this cannot fail.
        [[maybe_unused]] LogicalResult rewired =
            cudaq::quake::setQuantumOperands(ev.op, newOpWires);
        assert(succeeded(rewired) &&
               "rewiring with a fixed operand count cannot fail");
        if (isa<cudaq::quake::SinkOp, cudaq::quake::ReturnWireOp>(ev.op))
          continue;
        for (auto &&[w, q] :
             llvm::zip_equal(cudaq::quake::getQuantumResults(ev.op), ev.phys))
          phyToWire[q.index] = w;
      } else {
        // Insert the swap and advance both wires past it.
        auto q0 = ev.phys[0];
        auto q1 = ev.phys[1];
        auto swap = cudaq::quake::SwapOp::create(
            builder, builder.getUnknownLoc(), TypeRange{wireType, wireType},
            false, ValueRange{}, ValueRange{},
            ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
            DenseBoolArrayAttr{});
        phyToWire[q0.index] = swap.getResult(0);
        phyToWire[q1.index] = swap.getResult(1);
      }
    }
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

    // FIXME: Add the ability to handle multiple blocks.
    if (blocks.size() > 1) {
      if (nonComposable) {
        func.emitError("The mapper cannot handle multiple blocks");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "NYI: mapping with multiple blocks");
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
                       this->placement + "'; using 'auto'");
    }
    PlacementStrategy placementStrategy =
        parsedPlacement.value_or(PlacementStrategy::Auto);

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
                       this->search + "'; using 'sabre'");
    }
    SearchStrategy searchStrategy =
        parsedSearch.value_or(SearchStrategy::Sabre);

    bool collectInteractions = placementStrategy != PlacementStrategy::Identity;

    // Two-qubit interaction data for placement, collected during the scan.
    SmallVector<SmallVector<unsigned>> interactions;
    SmallVector<bool> userVirtualQubits;
    if (collectInteractions) {
      interactions.assign(deviceNumQubits,
                          SmallVector<unsigned>(deviceNumQubits, 0));
      userVirtualQubits.assign(deviceNumQubits, false);
    }

    for (Operation &op : block.getOperations()) {
      if (auto qop = dyn_cast<cudaq::quake::BorrowWireOp>(op)) {
        // Assign a new virtual qubit to the resulting wire.
        auto id = qop.getIdentity();
        wireToVirtualQ[qop.getResult()] = cudaq::Placement::VirtualQ(id);
        finalQubitWire[id] = qop.getResult();
        sources[id] = qop;
        if (collectInteractions)
          userVirtualQubits[id] = true;
        lastSource = &op;
      } else if (dyn_cast<cudaq::quake::NullWireOp>(op)) {
        if (nonComposable) {
          op.emitOpError(
              "the mapper requires borrow operations and prohibits null wires");
          signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << "null_wire ops are not expected");
        return;
      } else if (dyn_cast<cudaq::quake::AllocaOp>(op)) {
        if (nonComposable) {
          op.emitOpError("the mapper requires borrow operations and prohibits "
                         "reference semantics");
          signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << "quantum reference semantics not expected");
        return;
      } else if (cudaq::quake::isSupportedMappingOperation(&op)) {
        // Make sure the operation is using value semantics.
        if (!cudaq::quake::isLinearValueForm(&op)) {
          if (nonComposable) {
            llvm::errs() << "This is not SSA form: " << op << '\n';
            llvm::errs() << "isa<cudaq::quake::NullWireOp>() = "
                         << isa<cudaq::quake::NullWireOp>(&op) << '\n';
            llvm::errs() << "isAllReferences() = "
                         << cudaq::quake::isAllReferences(&op) << '\n';
            llvm::errs() << "isWrapped() = " << cudaq::quake::isWrapped(&op)
                         << '\n';
            func.emitError("The mapper requires value semantics.");
            signalPassFailure();
          }
          LLVM_DEBUG(llvm::dbgs() << "operation is not in proper value form");
          return;
        }

        // Since `quake.return_wire` operations do not generate new wires, we
        // don't need to further analyze.
        if (auto rop = dyn_cast<cudaq::quake::ReturnWireOp>(op)) {
          returnsToRemove.push_back(rop);
          continue;
        }

        // Get the wire operands and check if the operators uses at most two
        // qubits. N.B: Measurements do not have this restriction.
        auto wireOperands = cudaq::quake::getQuantumOperands(&op);
        if (!op.hasTrait<cudaq::QuantumMeasure>() && wireOperands.size() > 2) {
          if (nonComposable) {
            func.emitError("Cannot map a kernel with operators that use more "
                           "than two qubits.");
            signalPassFailure();
          }
          LLVM_DEBUG(llvm::dbgs() << "operator with >2 qubits not expected");
          return;
        }

        // Save which qubits are measured
        if (isa<cudaq::quake::MeasurementInterface>(op))
          for (const auto &wire : wireOperands)
            userQubitsMeasured.push_back(wireToVirtualQ[wire].index);

        // Record two-qubit interactions for placement.
        if (collectInteractions &&
            !isa<cudaq::quake::MeasurementInterface>(op) &&
            wireOperands.size() == 2) {
          unsigned v0 = wireToVirtualQ[wireOperands[0]].index;
          unsigned v1 = wireToVirtualQ[wireOperands[1]].index;
          if (v0 != v1) {
            interactions[v0][v1] += 1;
            interactions[v1][v0] += 1;
          }
        }

        // Map the result wires to the appropriate virtual qubits.
        for (auto &&[wire, newWire] : llvm::zip_equal(
                 wireOperands, cudaq::quake::getQuantumResults(&op))) {
          // Don't use wireToVirtualQ[a] = wireToVirtualQ[b]. It will work
          // *most* of the time but cause memory corruption other times because
          // DenseMap references can be invalidated upon insertion of new pairs.
          wireToVirtualQ.insert({newWire, wireToVirtualQ[wire]});
          finalQubitWire[wireToVirtualQ[wire].index] = newWire;
        }
      }
    }

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
              {measureOp.getWires()[0], wireToVirtualQ[finalQubitWire[i]]});

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

    // Generate the seed layouts to try, in deterministic order. Each seed only
    // proposes a starting vrToPhy. The router decides the rest.
    SmallVector<SmallVector<unsigned>> seeds;
    auto identitySeed = [&]() {
      SmallVector<unsigned> seed(numV);
      for (unsigned v = 0; v < numV; ++v)
        seed[v] = v;
      return seed;
    };
    auto greedySeed = [&]() {
      return interactionPlacement(*deviceInstance, interactions,
                                  userVirtualQubits);
    };
    switch (placementStrategy) {
    case PlacementStrategy::Auto: {
      seeds.push_back(identitySeed());
      // Greedy degenerates to identity when there are no interactions to place;
      // routing it again would just repeat the identity pass.
      SmallVector<unsigned> greedy = greedySeed();
      if (greedy != seeds.front())
        seeds.push_back(std::move(greedy));
      break;
    }
    case PlacementStrategy::Identity:
      seeds.push_back(identitySeed());
      break;
    case PlacementStrategy::Greedy:
      seeds.push_back(greedySeed());
      break;
    }

    // Build the routing problem once (it does not depend on the layout), then
    // search over the seeds for the result with the fewest swaps.
    RoutingProblem problem =
        buildRoutingProblem(block, sources, wireToVirtualQ);
    RoutingSearchStrategy search(
        *deviceInstance, problem, searchStrategy == SearchStrategy::Sabre,
        extendedLayerSize, extendedLayerWeight, decayDelta, roundsDecayReset);
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

#define DECLARE_SUB_OPTION(_PARENT_STRUCT, _FIELD)                             \
  PassOptions::Option<decltype(_PARENT_STRUCT::_FIELD)> _FIELD{*this, #_FIELD}
  DECLARE_SUB_OPTION(MappingPrepOptions, device);
  DECLARE_SUB_OPTION(MappingFuncOptions, extendedLayerSize);
  DECLARE_SUB_OPTION(MappingFuncOptions, extendedLayerWeight);
  DECLARE_SUB_OPTION(MappingFuncOptions, decayDelta);
  DECLARE_SUB_OPTION(MappingFuncOptions, roundsDecayReset);
  DECLARE_SUB_OPTION(MappingFuncOptions, placement);
  DECLARE_SUB_OPTION(MappingFuncOptions, search);
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
        setIt(funcOpts.placement, opt.placement);
        setIt(funcOpts.search, opt.search);
        setIt(funcOpts.nonComposable, opt.nonComposable);
        pm.addNestedPass<func::FuncOp>(cudaq::opt::createMappingFunc(funcOpts));
      });
}
} // namespace cudaq::opt
