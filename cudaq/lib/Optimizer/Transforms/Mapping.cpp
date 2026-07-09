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
#include "mlir/IR/Diagnostics.h"

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
/// interacting virtual qubits to central candidate physical qubits first. The
/// greedy growth and its tie-breaks make the layout a deterministic function of
/// the interaction counts and the device, so it is reproducible across runs.
class GreedyInitialPlacer {
public:
  GreedyInitialPlacer(const cudaq::Device &device,
                      const VirtualInteractionGraph &interactions,
                      ArrayRef<bool> userVirtualQubits)
      : device(device), interactions(interactions),
        userVirtualQubits(userVirtualQubits), n(device.getNumQubits()),
        placedVirtual(n, false), vrToPhy(n, n) {}

  /// Produce the `vrToPhy` seed layout.
  SmallVector<unsigned> run() {
    // No two-qubit interactions, so every layout routes identically; return the
    // identity seed for a deterministic result.
    if (!interactions.hasInteractions()) {
      for (unsigned v = 0; v < n; ++v)
        vrToPhy[v] = v;
      return vrToPhy;
    }

    SmallVector<unsigned> physicals;
    physicals.reserve(n);
    for (unsigned p = 0; p < n; ++p)
      physicals.push_back(p);
    SmallVector<unsigned> virtuals;
    for (unsigned u = 0; u < n; ++u)
      if (userVirtualQubits[u])
        virtuals.push_back(u);

    // Stage 1: Greedily place all user virtual qubits.
    placeGreedily(virtuals, physicals);

    // Stage 2: Assign any still-unplaced virtuals (non-user qubits) to the
    // remaining free physicals, pairing them in ascending order.
    assignRemainingVirtuals();

    return vrToPhy;
  }

  /// Place the supplied virtual qubits using only the supplied physical
  /// qubits. Entries for virtual qubits outside this placement remain `n`.
  SmallVector<unsigned> runRestricted(ArrayRef<unsigned> virtuals,
                                      ArrayRef<unsigned> physicals) {
    // Deliberately skip stage 2 (`assignRemainingVirtuals`): `virtuals` and
    // `physicals` here are one island's qubits, not every qubit in the
    // device, so stage 2 would try to fill in every virtual qubit outside
    // this island from the handful of physicals this island has left over.
    placeGreedily(virtuals, physicals);
    return vrToPhy;
  }

private:
  /// The core greedy walk shared by `run` and `runRestricted`: rank the given
  /// `physicals` by centrality, seed the highest-degree virtual on the most
  /// central physical, then attach each remaining virtual to its cheapest
  /// reachable physical. Only `virtuals` are placed, only onto `physicals`.
  void placeGreedily(ArrayRef<unsigned> virtuals,
                     ArrayRef<unsigned> physicals) {
    initWorklists(virtuals, physicals);
    computeCentrality(physicals);

    // Seed the highest-degree virtual qubit onto the most central physical
    // qubit, then grow the layout around it.
    place(chooseSeedVirtual(), bestFreePhysical());

    while (!unplacedUserVirtuals.empty()) {
      unsigned v = chooseNextVirtual();
      place(v, bestPhysicalFor(v));
    }
  }

  /// Physical centrality used to break ties: total distance to every other
  /// candidate physical qubit, and connectivity degree. Callers pass one
  /// device island's physicals, so every pair is reachable; an unreachable
  /// pair here would be a caller bug (asserted below).
  void computeCentrality(ArrayRef<unsigned> physicals) {
    using Qubit = cudaq::Device::Qubit;
    distanceSum.assign(n, 0);
    physDegree.assign(n, 0);
    for (unsigned p : physicals) {
      for (unsigned q : physicals) {
        unsigned distance = device.getDistance(Qubit(p), Qubit(q));
        assert(distance != cudaq::Device::unreachableDistance &&
               "greedy physical candidates must belong to one device island");
        distanceSum[p] += distance;
      }
      physDegree[p] =
          static_cast<unsigned>(device.getNeighbours(Qubit(p)).size());
    }
  }

  /// Seed the placement worklists from the caller's candidate sets. Both are
  /// kept sorted because `place` erases consumed entries with `lower_bound`,
  /// and the incoming `virtuals`/`physicals` may be unsorted.
  void initWorklists(ArrayRef<unsigned> virtuals,
                     ArrayRef<unsigned> physicals) {
    freePhysicals.assign(physicals.begin(), physicals.end());
    unplacedUserVirtuals.assign(virtuals.begin(), virtuals.end());
    llvm::sort(freePhysicals);
    llvm::sort(unplacedUserVirtuals);
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
      bool reachable = true;
      for (const auto &edge : interactions.neighbors(v))
        if (placedVirtual[edge.first]) {
          unsigned distance =
              device.getDistance(Qubit(p), Qubit(vrToPhy[edge.first]));
          if (distance == cudaq::Device::unreachableDistance) {
            reachable = false;
            break;
          }
          cost += edge.second * distance;
        }
      if (!reachable)
        continue;
      bool better = bestPhy == n || cost < bestCost ||
                    (cost == bestCost && isMoreCentralPhysical(p, bestPhy));
      if (better) {
        bestPhy = p;
        bestCost = cost;
      }
    }
    // Within one island every free physical is reachable from `v`'s placed
    // partners, and the island plan reserves enough capacity, so a candidate
    // is always found.
    assert(bestPhy != n && "island plan must provide enough physical qubits");
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

/// One generic flood-fill so the device graph and the virtual interaction
/// graph are decomposed by the same code path instead of two near-copies.
/// BFS connected-component decomposition over a graph of `n` nodes.
/// `isActive(u)` returns false to exclude node `u` from the decomposition.
/// `forNeighbors(u, visit)` calls `visit(v)` for each neighbor `v` of `u`.
/// Returns sorted components in ascending node-index order.
template <typename IsActive, typename ForNeighbors>
static SmallVector<SmallVector<unsigned>>
computeComponents(unsigned n, IsActive isActive, ForNeighbors forNeighbors) {
  SmallVector<SmallVector<unsigned>> components;
  SmallVector<bool> visited(n, false);
  for (unsigned start = 0; start < n; ++start) {
    if (visited[start] || !isActive(start))
      continue;
    SmallVector<unsigned> component;
    SmallVector<unsigned> worklist{start};
    visited[start] = true;
    for (std::size_t cursor = 0; cursor < worklist.size(); ++cursor) {
      unsigned current = worklist[cursor];
      component.push_back(current);
      forNeighbors(current, [&](unsigned next) {
        if (visited[next] || !isActive(next))
          return;
        visited[next] = true;
        worklist.push_back(next);
      });
    }
    llvm::sort(component);
    components.push_back(std::move(component));
  }
  return components;
}

/// Physical islands of mutually reachable qubits. A two-qubit gate can never be
/// routed (even with swaps) across different islands, so placement and
/// rejection both reason in terms of these.
SmallVector<SmallVector<unsigned>>
computeDeviceIslands(const cudaq::Device &device) {
  using Qubit = cudaq::Device::Qubit;
  const unsigned n = device.getNumQubits();
  return computeComponents(
      n, [](unsigned) { return true; },
      [&](unsigned current, auto visit) {
        for (auto neighbor : device.getNeighbours(Qubit(current)))
          visit(neighbor.index);
      });
}

/// Groups of qubits that must be co-located because they interact, directly or
/// transitively. Idle and single-qubit-only qubits are excluded: they add no
/// placement constraint and must not consume island capacity.
SmallVector<SmallVector<unsigned>>
computeVirtualComponents(const VirtualInteractionGraph &interactions,
                         ArrayRef<bool> userVirtualQubits) {
  const unsigned n = userVirtualQubits.size();
  return computeComponents(
      n,
      [&](unsigned u) {
        return userVirtualQubits[u] && interactions.weightedDegree(u) != 0;
      },
      [&](unsigned current, auto visit) {
        for (const auto &edge : interactions.neighbors(current))
          visit(edge.first);
      });
}

/// One device island (its physical qubits) and the interacting virtual qubits
/// packed onto it. Idle and single-qubit-only virtuals impose no island
/// constraint and are assigned separately, so they never appear here.
struct IslandPlan {
  SmallVector<unsigned> physicalQubits;
  SmallVector<unsigned> virtualQubits;
};

/// Assign each interacting virtual component wholly to one device island.
/// Components are packed largest-first into the smallest remaining capacity
/// that fits. This best-fit-decreasing heuristic is deterministic but not
/// complete: it can fail even when a different packing exists.
std::optional<SmallVector<IslandPlan>>
buildBestFitIslandPlan(SmallVector<SmallVector<unsigned>> deviceIslands,
                       SmallVector<SmallVector<unsigned>> virtualComponents) {
  llvm::sort(virtualComponents, [](const auto &lhs, const auto &rhs) {
    if (lhs.size() != rhs.size())
      return lhs.size() > rhs.size();
    return lhs.front() < rhs.front();
  });

  SmallVector<IslandPlan> plan;
  plan.reserve(deviceIslands.size());
  for (auto &island : deviceIslands)
    plan.push_back({std::move(island), {}});

  for (const auto &virtualComponent : virtualComponents) {
    // `plan.size()` is the "no island chosen yet" sentinel.
    unsigned bestIsland = plan.size();
    for (unsigned islandId = 0; islandId < plan.size(); ++islandId) {
      const auto &island = plan[islandId];
      const std::size_t remainingCapacity =
          island.physicalQubits.size() - island.virtualQubits.size();
      if (remainingCapacity < virtualComponent.size())
        continue;
      if (bestIsland == plan.size()) {
        bestIsland = islandId;
        continue;
      }
      const auto &best = plan[bestIsland];
      const std::size_t bestRemainingCapacity =
          best.physicalQubits.size() - best.virtualQubits.size();
      // Best fit: keep the smallest remaining capacity that still fits so
      // larger islands stay free for larger components. Ties break on the
      // island's next free physical (its lowest unassigned index) for
      // determinism.
      if (remainingCapacity < bestRemainingCapacity ||
          (remainingCapacity == bestRemainingCapacity &&
           island.physicalQubits[island.virtualQubits.size()] <
               best.physicalQubits[best.virtualQubits.size()]))
        bestIsland = islandId;
    }
    if (bestIsland == plan.size())
      return std::nullopt;
    plan[bestIsland].virtualQubits.append(virtualComponent.begin(),
                                          virtualComponent.end());
  }
  return plan;
}

/// Build a greedy seed that never splits an interacting group of virtual
/// qubits across device islands. On a single-island (connected) device, one
/// greedy walk places every virtual qubit over the whole device. On a
/// multi-island (disconnected) device, each interacting virtual component is
/// packed onto one island first, then the same greedy heuristic runs again
/// restricted to that island's qubits.
std::optional<SmallVector<unsigned>>
buildGreedySeed(unsigned numV, const cudaq::Device &device,
                const VirtualInteractionGraph &interactions,
                ArrayRef<bool> userVirtualQubits) {
  SmallVector<SmallVector<unsigned>> deviceIslands =
      computeDeviceIslands(device);
  SmallVector<SmallVector<unsigned>> virtualComponents =
      computeVirtualComponents(interactions, userVirtualQubits);
  if (deviceIslands.size() <= 1 || virtualComponents.empty())
    return GreedyInitialPlacer(device, interactions, userVirtualQubits).run();

  auto plan = buildBestFitIslandPlan(std::move(deviceIslands),
                                     std::move(virtualComponents));
  if (!plan)
    return std::nullopt;

  const unsigned unassignedDeviceQubit = device.getNumQubits();
  SmallVector<unsigned> seed(numV, unassignedDeviceQubit);
  SmallVector<bool> usedDeviceQubits(device.getNumQubits(), false);
  for (const auto &island : *plan) {
    if (island.virtualQubits.empty())
      continue;
    SmallVector<unsigned> localSeed =
        GreedyInitialPlacer(device, interactions, userVirtualQubits)
            .runRestricted(island.virtualQubits, island.physicalQubits);
    for (unsigned virtualQubit : island.virtualQubits) {
      unsigned deviceQubit = localSeed[virtualQubit];
      seed[virtualQubit] = deviceQubit;
      usedDeviceQubits[deviceQubit] = true;
    }
  }

  // Idle and single-qubit-only virtuals impose no island constraint. Assign
  // them to the remaining physical qubits in ascending order.
  SmallVector<unsigned> remainingDeviceQubits;
  for (unsigned p = 0; p < device.getNumQubits(); ++p)
    if (!usedDeviceQubits[p])
      remainingDeviceQubits.push_back(p);
  unsigned nextRemaining = 0;
  for (unsigned v = 0; v < numV; ++v)
    if (seed[v] == unassignedDeviceQubit)
      seed[v] = remainingDeviceQubits[nextRemaining++];
  return seed;
}

/// The identity and greedy seeds can coincide; de-dupe so the router never pays
/// to route the same layout twice.
void pushSeedIfNew(SmallVector<SmallVector<unsigned>> &seeds,
                   SmallVector<unsigned> seed) {
  if (llvm::find(seeds, seed) == seeds.end())
    seeds.push_back(std::move(seed));
}

/// Generate the seed layouts to try, in deterministic order. Each seed only
/// proposes a starting virtual-to-physical qubit mapping. The router decides
/// the rest.
SmallVector<SmallVector<unsigned>>
buildPlacementSeeds(PlacementStrategy strategy, unsigned numV,
                    const cudaq::Device &device,
                    const VirtualInteractionGraph &interactions,
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
    if (auto greedy =
            buildGreedySeed(numV, device, interactions, userVirtualQubits)) {
      // For `auto`, greedy degenerates to identity when there are no
      // interactions to place, so skip the duplicate rather than route the
      // identity layout twice.
      pushSeedIfNew(seeds, std::move(*greedy));
    }
  }

  return seeds;
}

/// A seed is usable only if every active two-qubit interaction has both
/// endpoints in the same device island. Returns the first offending virtual
/// pair so a doomed seed is rejected with a precise diagnostic.
std::optional<std::pair<unsigned, unsigned>>
findUnroutableInteraction(ArrayRef<unsigned> seed, const cudaq::Device &device,
                          const VirtualInteractionGraph &interactions) {
  using Qubit = cudaq::Device::Qubit;
  for (unsigned v = 0, end = seed.size(); v < end; ++v) {
    std::optional<unsigned> firstUnroutableNeighbor;
    for (const auto &edge : interactions.neighbors(v)) {
      unsigned u = edge.first;
      if (u <= v)
        continue;
      if (!device.hasPath(Qubit(seed[v]), Qubit(seed[u])) &&
          (!firstUnroutableNeighbor || u < *firstUnroutableNeighbor))
        firstUnroutableNeighbor = u;
    }
    if (firstUnroutableNeighbor)
      return std::make_pair(v, *firstUnroutableNeighbor);
  }
  return std::nullopt;
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

/// A single routing decision: a gate mapped onto physical qubits, a swap
/// inserted between them, an if-op, or a loop-op. The router records Gate and
/// Swap events; the enrichment step adds If and Loop events. The emitter
/// replays the full trace to rewrite the IR. Body results for If/Loop events
/// live in the block map, not here.
struct RoutingEvent {
  enum class Kind { Gate, Swap, If, Loop };

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
  /// An if-op. `phys` holds the physical qubit for each wire linear arg at the
  /// point the if is reached; branch results are in the caller's block map.
  static RoutingEvent makeIf(mlir::Operation *op,
                             ArrayRef<cudaq::Placement::DeviceQ> phys) {
    return RoutingEvent{
        Kind::If, op,
        SmallVector<cudaq::Placement::DeviceQ, 2>(phys.begin(), phys.end())};
  }
  /// A loop-op. `phys` holds the physical qubit for each wire initialArg at
  /// the point the loop is reached; body results are in the caller's block map.
  static RoutingEvent makeLoop(mlir::Operation *op,
                               ArrayRef<cudaq::Placement::DeviceQ> phys) {
    return RoutingEvent{
        Kind::Loop, op,
        SmallVector<cudaq::Placement::DeviceQ, 2>(phys.begin(), phys.end())};
  }

  Kind kind;
  mlir::Operation *op;
  SmallVector<cudaq::Placement::DeviceQ, 2> phys;
};

/// The outcome of routing one block. Gate and Swap events come from SABRE; If
/// events are added by the enrichment step and carry per-branch results.
/// `swapCount` is the metric used to compare top-level layouts.
struct RoutingResult {
  /// Virtual-to-physical layout at the start of the walk, before any swap.
  SmallVector<unsigned> initialLayout;
  /// Virtual-to-physical layout at the end of the walk, after all swaps.
  /// Computed by enrichTrace; not set on raw SABRE output.
  SmallVector<unsigned> exitLayout;
  SmallVector<RoutingEvent> trace;
  /// Restoration SWAPs to emit before the block terminator (if branches and
  /// loop bodies). Filled by the join-point strategy; see JoinPointStrategy.
  SmallVector<RoutingEvent> cleanUpTrace;
  unsigned swapCount = 0;
};

/// Policy for reconciling the placements that meet at a control-flow join. Each
/// predecessor region has already been routed by the shared router, so its
/// RoutingResult carries the entry (`initialLayout`) and routed exit
/// (`exitLayout`) layouts. A strategy chooses a common exit layout for all
/// predecessors and fills each one's cleanUpTrace with the SWAPs that reach it,
/// updating its exitLayout to match. Swapping in a new strategy is the sole
/// extension point for smarter routing (greedy, cost-minimizing); the shared
/// router and the emitter are strategy-independent.
struct JoinPointStrategy {
  virtual ~JoinPointStrategy() = default;
  virtual void reconcile(MutableArrayRef<RoutingResult *> predecessors) = 0;
};

/// Stage 1.5: restore every predecessor to its own entry layout. If branches
/// share an entry, so they agree trivially; a loop body returns to its entry,
/// making that the loop invariant. Reified by replaying each region's swaps in
/// reverse before its terminator — no layout search, no pre-region SWAPs.
struct RestoreToEntryStrategy : JoinPointStrategy {
  void reconcile(MutableArrayRef<RoutingResult *> predecessors) override {
    for (RoutingResult *r : predecessors) {
      r->cleanUpTrace.clear();
      for (const RoutingEvent &swapEv : llvm::reverse(r->trace))
        if (swapEv.kind == RoutingEvent::Kind::Swap)
          r->cleanUpTrace.push_back(swapEv);
      r->exitLayout = r->initialLayout;
    }
  }
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
    Block &block, ArrayRef<Value> sources,
    const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ) {
  RoutingProblem problem;
  DenseMap<Operation *, RoutingProblem::NodeRef> nodeIndex;

  for (Operation &op : block) {
    if (isa<cudaq::quake::BorrowWireOp>(op))
      continue;
    RoutingProblem::Node node;
    node.op = &op;
    if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
      for (auto linArg : ifOp.getLinearArgs())
        if (isa<cudaq::quake::WireType>(linArg.getType()))
          node.qubits.push_back(requireVirtualQ(wireToVirtualQ, linArg));
    } else if (auto loopOp = dyn_cast<cudaq::cc::LoopOp>(op)) {
      for (auto initArg : loopOp.getInitialArgs())
        if (isa<cudaq::quake::WireType>(initArg.getType()))
          node.qubits.push_back(requireVirtualQ(wireToVirtualQ, initArg));
    } else {
      if (!cudaq::quake::isSupportedMappingOperation(&op))
        continue;
      for (auto wire : cudaq::quake::getQuantumOperands(&op))
        node.qubits.push_back(requireVirtualQ(wireToVirtualQ, wire));
      node.isMeasure = op.hasTrait<cudaq::QuantumMeasure>();
      node.isUnitary = isa<cudaq::quake::OperatorInterface>(op);
      node.isTwoQ = node.isUnitary && node.qubits.size() == 2;
    }
    nodeIndex[&op] = RoutingProblem::NodeRef(problem.nodes.size());
    problem.nodes.push_back(std::move(node));
  }

  // Record successor edges by walking the uses of each quantum result wire. A
  // consumer is listed once per result wire it consumes, so a node's visit
  // count reaches its wire-operand count exactly when all of its inputs are
  // ready. Walking wire uses directly, rather than `Operation::getUsers`, makes
  // that multiplicity explicit and ignores classical results such as
  // measurement bits.
  auto recordWireUsers = [&](Value wire,
                             SmallVectorImpl<RoutingProblem::NodeRef> &out) {
    for (OpOperand &use : wire.getUses())
      if (auto it = nodeIndex.find(use.getOwner()); it != nodeIndex.end())
        out.push_back(it->second);
  };
  for (auto &node : problem.nodes) {
    auto wireResults = isa<cudaq::cc::IfOp, cudaq::cc::LoopOp>(node.op)
                           ? node.op->getResults()
                           : cudaq::quake::getQuantumResults(node.op);
    for (Value wire : wireResults)
      if (isa<cudaq::quake::WireType>(wire.getType()))
        recordWireUsers(wire, node.successors);
  }
  for (auto source : sources)
    recordWireUsers(source, problem.sourceUsers);

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

  // Splice out IfOps transitively so unitary gates on either side of an IfOp
  // still form a dependency edge in the reverse problem.
  std::function<void(RoutingProblem::NodeRef, RoutingProblem::NodeRef,
                     unsigned &)>
      addReverseEdges;
  addReverseEdges = [&](RoutingProblem::NodeRef revSrc,
                        RoutingProblem::NodeRef fwdSucc, unsigned &count) {
    if (shouldIncludeInReverse(forward[fwdSucc])) {
      ++count;
      reverse.nodes[fwdToRev[fwdSucc.index].index].successors.push_back(revSrc);
    } else {
      for (RoutingProblem::NodeRef s : forward[fwdSucc].successors)
        addReverseEdges(revSrc, s, count);
    }
  };

  for (unsigned i = 0, end = forward.nodes.size(); i < end; ++i) {
    const RoutingProblem::Node &node = forward.nodes[i];
    if (!shouldIncludeInReverse(node))
      continue;
    unsigned unitarySuccessors = 0;
    for (RoutingProblem::NodeRef s : node.successors)
      addReverseEdges(fwdToRev[i], s, unitarySuccessors);
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

  // IfOps and LoopOps are opaque: pass through with their current qubit layout.
  if (isa<cudaq::cc::IfOp>(node.op)) {
    result.trace.push_back(RoutingEvent::makeIf(node.op, deviceQubits));
    return success();
  }
  if (isa<cudaq::cc::LoopOp>(node.op)) {
    result.trace.push_back(RoutingEvent::makeLoop(node.op, deviceQubits));
    return success();
  }

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
    unsigned distance = device.getDistance(phy0, phy1);
    // Invariant: findUnroutableInteraction admits only layouts whose two-qubit
    // interactions are intra-island, and swaps never cross islands, so a
    // routed gate always has a finite distance here.
    assert(distance != cudaq::Device::unreachableDistance &&
           "front-layer gate spans disconnected device islands");
    cost += distance - 1;
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
    // Co-island by construction (see `computeLayerCost` /
    // `findUnroutableInteraction`): a stalled front-layer gate is always
    // reachable, never unreachableDistance.
    assert(d != cudaq::Device::unreachableDistance &&
           "stalled front-layer gate spans disconnected device islands");
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

/// Applies a RoutingResult to the IR. This is the only place routing rewrites
/// the circuit. It rewires each mapped operation and inserts the swaps,
/// threading the current wire on each physical qubit.
class RoutingEmitter {
public:
  RoutingEmitter(const DenseMap<Value, cudaq::Placement::VirtualQ> &wireMap,
                 unsigned numPhysical,
                 const DenseMap<Block *, RoutingResult> &blockMap)
      : wireToVirtualQ(wireMap), phyToWire(numPhysical),
        blockResults(blockMap) {}

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
      unsigned phy =
          result.initialLayout[requireVirtualQ(wireToVirtualQ, wire).index];
      borrowWire.setIdentity(phy);
      phyToWire[phy] = wire;
    }

    emitBlock(block);
    return phyToWire;
  }

private:
  /// Emit the swaps and rewire the operations for a single block, recursing
  /// into the regions of any `if` operations encountered along the way.
  void emitBlock(Block &blk) {
    const RoutingResult &blkResult = blockResults.at(&blk);
    OpBuilder blkBuilder(&blk, blk.begin());
    auto wireType = blkBuilder.getType<cudaq::quake::WireType>();

    for (const RoutingEvent &ev : blkResult.trace) {
      if (ev.kind == RoutingEvent::Kind::Swap) {
        auto q0 = ev.phys[0], q1 = ev.phys[1];
        auto swap = cudaq::quake::SwapOp::create(
            blkBuilder, blkBuilder.getUnknownLoc(),
            TypeRange{wireType, wireType}, false, ValueRange{}, ValueRange{},
            ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
            DenseBoolArrayAttr{});
        phyToWire[q0.index] = swap.getResult(0);
        phyToWire[q1.index] = swap.getResult(1);
      } else if (ev.kind == RoutingEvent::Kind::Gate) {
        SmallVector<Value, 2> newOpWires;
        for (auto phy : ev.phys)
          newOpWires.push_back(phyToWire[phy.index]);
        [[maybe_unused]] LogicalResult rewired =
            cudaq::quake::setQuantumOperands(ev.op, newOpWires);
        assert(succeeded(rewired) &&
               "rewiring with a fixed operand count cannot fail");
        if (isa<cudaq::quake::SinkOp, cudaq::quake::ReturnWireOp>(ev.op))
          continue;
        for (auto &&[w, q] :
             llvm::zip_equal(cudaq::quake::getQuantumResults(ev.op), ev.phys))
          phyToWire[q.index] = w;
      } else if (ev.kind == RoutingEvent::Kind::If) {
        auto ifOp = cast<cudaq::cc::IfOp>(ev.op);
        SmallVector<Value> entryWires;
        for (auto phy : ev.phys)
          entryWires.push_back(phyToWire[phy.index]);
        (void)cudaq::quake::setQuantumOperands(ifOp, entryWires);

        auto processBranch = [&](Region &region) {
          const RoutingResult &branchResult = blockResults.at(&region.front());
          unsigned phyIdx = 0;
          for (auto [i, linArg] : llvm::enumerate(ifOp.getLinearArgs())) {
            if (!wireToVirtualQ.count(linArg))
              continue;
            phyToWire[ev.phys[phyIdx++].index] = region.front().getArgument(i);
          }
          emitBlock(region.front());
          // Rewire the branch's cc.continue to the wires left on each physical
          // qubit after in-branch routing and cleanup restoration.
          auto *contOp = region.front().getTerminator();
          for (auto [k, operand] : llvm::enumerate(contOp->getOperands())) {
            if (!isa<cudaq::quake::WireType>(operand.getType()))
              continue;
            auto vq = wireToVirtualQ.find(ifOp->getResult(k))->second;
            contOp->setOperand(k, phyToWire[branchResult.exitLayout[vq.index]]);
          }
          for (auto [i, phy] : llvm::enumerate(ev.phys))
            phyToWire[phy.index] = entryWires[i];
          return branchResult;
        };

        const RoutingResult &thenResult = processBranch(ifOp.getThenRegion());
        if (ifOp.hasElse())
          processBranch(ifOp.getElseRegion());

        for (Value res : ifOp->getResults())
          if (isa<cudaq::quake::WireType>(res.getType())) {
            auto vq = wireToVirtualQ.find(res)->second;
            phyToWire[thenResult.exitLayout[vq.index]] = res;
          }
      } else if (ev.kind == RoutingEvent::Kind::Loop) {
        auto loopOp = cast<cudaq::cc::LoopOp>(ev.op);
        auto *bodyBlock = loopOp.getDoEntryBlock();
        // Rewire each wire initialArg to the current physical wire.
        unsigned phyIdx = 0;
        for (auto [i, initArg] : llvm::enumerate(loopOp.getInitialArgs())) {
          if (!isa<cudaq::quake::WireType>(initArg.getType()))
            continue;
          loopOp->setOperand(i, phyToWire[ev.phys[phyIdx++].index]);
        }
        // Thread body block args into phyToWire and emit the body.
        phyIdx = 0;
        for (auto bodyArg : bodyBlock->getArguments()) {
          if (!isa<cudaq::quake::WireType>(bodyArg.getType()))
            continue;
          phyToWire[ev.phys[phyIdx++].index] = bodyArg;
        }
        emitBlock(*bodyBlock);
        // Update body cc.continue operands to use the restored wires.
        auto *contOp = bodyBlock->getTerminator();
        phyIdx = 0;
        for (unsigned j = 0; j < loopOp.getInitialArgs().size(); ++j) {
          if (!isa<cudaq::quake::WireType>(
                  loopOp.getInitialArgs()[j].getType()))
            continue;
          contOp->setOperand(j, phyToWire[ev.phys[phyIdx++].index]);
        }
        // For for-loops: emit the step block and update its cc.continue.
        if (loopOp.hasStep()) {
          auto *stepBlock = loopOp.getStepBlock();
          phyIdx = 0;
          for (auto stepArg : stepBlock->getArguments()) {
            if (!isa<cudaq::quake::WireType>(stepArg.getType()))
              continue;
            phyToWire[ev.phys[phyIdx++].index] = stepArg;
          }
          emitBlock(*stepBlock);
          auto *stepContOp = stepBlock->getTerminator();
          phyIdx = 0;
          for (unsigned j = 0; j < loopOp.getInitialArgs().size(); ++j) {
            if (!isa<cudaq::quake::WireType>(
                    loopOp.getInitialArgs()[j].getType()))
              continue;
            stepContOp->setOperand(j, phyToWire[ev.phys[phyIdx++].index]);
          }
        }
        // Update phyToWire with the loop results.
        phyIdx = 0;
        for (Value res : loopOp->getResults()) {
          if (!isa<cudaq::quake::WireType>(res.getType()))
            continue;
          phyToWire[ev.phys[phyIdx++].index] = res;
        }
      } else {
        llvm_unreachable("unhandled RoutingEvent::Kind");
      }
    }
    // Emit restoration SWAPs (cleanUpTrace) before the block terminator.
    if (!blkResult.cleanUpTrace.empty()) {
      OpBuilder cleanBuilder(blk.getTerminator());
      for (const RoutingEvent &cleanEv : blkResult.cleanUpTrace) {
        auto q0 = cleanEv.phys[0], q1 = cleanEv.phys[1];
        auto swap = cudaq::quake::SwapOp::create(
            cleanBuilder, cleanBuilder.getUnknownLoc(),
            TypeRange{wireType, wireType}, false, ValueRange{}, ValueRange{},
            ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
            DenseBoolArrayAttr{});
        phyToWire[q0.index] = swap.getResult(0);
        phyToWire[q1.index] = swap.getResult(1);
      }
    }
    sortTopologically(&blk);
  }

  const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ;
  SmallVector<Value> phyToWire;
  const DenseMap<Block *, RoutingResult> &blockResults;
};

llvm::Error deviceFromString(llvm::StringRef deviceString, bool &deviceBypass,
                             std::optional<cudaq::Device> &deviceInstance) {
  deviceBypass = false;
  deviceInstance.reset();
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
          return llvm::Error::success();
        }
      } else {
        llvm::errs() << "Missing closing ')' in device option\n";
        return llvm::Error::success();
      }
    } else {
      llvm::errs() << "Filename must be provided in device option like "
                      "file(/full/path/to/device_file.txt): "
                   << deviceString << '\n';
      return llvm::Error::success();
    }

    // Shortest paths retain views into Device-owned storage, so parse directly
    // into the pass member instead of moving a temporary through the error
    // path.
    deviceInstance.emplace();
    if (llvm::Error error =
            cudaq::Device::tryFile(deviceFilename, *deviceInstance)) {
      deviceInstance.reset();
      return error;
    }
    return llvm::Error::success();
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
          return llvm::Error::success();
        }
        deviceString = deviceString.ltrim();
        deviceString.consumeInteger(/*Radix=*/10, deviceDim[1]);
        deviceString = deviceString.ltrim();
        ++argCount;
      }

      if (!deviceString.consume_front(")")) {
        llvm::errs() << "Missing closing ')' in device option\n";
        return llvm::Error::success();
      }
    }

    if (deviceTopoStr == "path") {
      deviceInstance = cudaq::Device::path(deviceDim[0]);
    } else if (deviceTopoStr == "ring") {
      deviceInstance = cudaq::Device::ring(deviceDim[0]);
    } else if (deviceTopoStr == "star") {
      deviceInstance = cudaq::Device::star(deviceDim[0], deviceDim[1]);
    } else if (deviceTopoStr == "grid") {
      deviceInstance = cudaq::Device::grid(deviceDim[0], deviceDim[1]);
    } else if (deviceTopoStr == "bypass") {
      deviceBypass = true;
    } else {
      llvm::errs() << "Unknown device option: " << deviceTopoStr << '\n';
    }
    return llvm::Error::success();
  }
}

LogicalResult initializeDevice(llvm::StringRef deviceString, bool nonComposable,
                               MLIRContext *context, bool &deviceBypass,
                               std::optional<cudaq::Device> &deviceInstance) {
  if (llvm::Error error =
          deviceFromString(deviceString, deviceBypass, deviceInstance)) {
    emitError(UnknownLoc::get(context), llvm::toString(std::move(error)));
    return failure();
  }
  return success(deviceInstance || deviceBypass || !nonComposable);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct MappingPrep : public cudaq::opt::impl::MappingPrepBase<MappingPrep> {
  using MappingPrepBase::MappingPrepBase;

  std::optional<cudaq::Device> deviceInstance;
  bool deviceBypass = false;

  virtual LogicalResult initialize(MLIRContext *context) override {
    return initializeDevice(device, nonComposable, context, deviceBypass,
                            deviceInstance);
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

    // A composable run tolerates an unset device (the default "-" or an
    // unparsable device option) as a no-op; `initialize` already fails a
    // non-composable run before it reaches here. Guard the deref regardless so
    // a bad device string cannot crash the pass.
    if (!deviceInstance)
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
    return initializeDevice(device, nonComposable, context, deviceBypass,
                            deviceInstance);
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

  /// Recursively analyze `b` and all IfOp branch regions, populating
  /// wireToVirtualQ, finalQubitWire, sources, and interaction data. `parentOp`
  /// is the enclosing cc::IfOp when recursing into a branch; null at the top
  /// level. cc.continue uses it to set (first branch) or verify (else branch)
  /// the parent's wire result VQs directly in wireToVirtualQ. Clears
  /// `analysisOk` on the first unsupported construct.
  void
  analyzeBlock(Block &b, bool doCollectInteractions, Operation *parentOp,
               bool &analysisOk,
               SmallVectorImpl<cudaq::quake::BorrowWireOp> &sources,
               SmallVectorImpl<cudaq::quake::ReturnWireOp> &returnsToRemove,
               DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ,
               SmallVectorImpl<std::size_t> &userQubitsMeasured,
               DenseMap<std::size_t, Value> &finalQubitWire,
               Operation *&lastSource, VirtualInteractionGraph &interactions,
               SmallVectorImpl<bool> &userVirtualQubits) {
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
          op.emitOpError("the mapper requires borrow operations and prohibits "
                         "reference semantics");
          signalPassFailure();
        }
        LLVM_DEBUG(llvm::dbgs() << "quantum reference semantics not expected");
        analysisOk = false;
        return;
      } else if (auto cont = dyn_cast<cudaq::cc::ContinueOp>(op)) {
        if (!parentOp)
          continue;
        // cc.continue operands correspond positionally to parentOp
        // results. On the first branch, insert; on the else branch,
        // verify the ordering matches.
        for (auto [operand, res] :
             llvm::zip_equal(cont->getOperands(), parentOp->getResults())) {
          if (!isa<cudaq::quake::WireType>(operand.getType()))
            continue;
          auto contVQ = requireVirtualQ(wireToVirtualQ, operand);
          auto [it, inserted] = wireToVirtualQ.insert({res, contVQ});
          if (!inserted && it->second.index != contVQ.index) {
            parentOp->emitOpError("then and else branches return qubits "
                                  "in different orders");
            if (nonComposable)
              signalPassFailure();
            analysisOk = false;
            return;
          }
        }
      } else if (auto ifOp = dyn_cast<cudaq::cc::IfOp>(op)) {
        auto linearArgs = ifOp.getLinearArgs();
        for (Region *region : ifOp.getRegions()) {
          if (region->empty())
            continue;
          assert(region->hasOneBlock());
          for (auto [linArg, regionArg] :
               llvm::zip_equal(linearArgs, region->front().getArguments()))
            wireToVirtualQ.insert({regionArg, wireToVirtualQ[linArg]});
          analyzeBlock(region->front(), /*doCollectInteractions=*/false, &op,
                       analysisOk, sources, returnsToRemove, wireToVirtualQ,
                       userQubitsMeasured, finalQubitWire, lastSource,
                       interactions, userVirtualQubits);
          if (!analysisOk)
            return;
        }
        for (Value res : ifOp->getResults())
          if (isa<cudaq::quake::WireType>(res.getType()))
            finalQubitWire[wireToVirtualQ[res].index] = res;
      } else if (auto loopOp = dyn_cast<cudaq::cc::LoopOp>(op)) {
        auto *whileBlock = loopOp.getWhileBlock();
        auto *bodyBlock = loopOp.getDoEntryBlock();
        // Map loop initialArgs → while block args.
        for (auto [initArg, whileArg] : llvm::zip_equal(
                 loopOp.getInitialArgs(), whileBlock->getArguments())) {
          if (!isa<cudaq::quake::WireType>(initArg.getType()))
            continue;
          wireToVirtualQ.insert(
              {whileArg, requireVirtualQ(wireToVirtualQ, initArg)});
        }
        // cc.condition in the while block forwards iter args to the body
        // and to the loop exit. Map those → body block args and results.
        auto condOp = cast<cudaq::cc::ConditionOp>(whileBlock->getTerminator());
        for (auto [forwarded, bodyArg, loopResult] :
             llvm::zip_equal(condOp.getResults(), bodyBlock->getArguments(),
                             loopOp->getResults())) {
          if (!isa<cudaq::quake::WireType>(forwarded.getType()))
            continue;
          auto vq = requireVirtualQ(wireToVirtualQ, forwarded);
          wireToVirtualQ.insert({bodyArg, vq});
          wireToVirtualQ.insert({loopResult, vq});
        }
        // parentOp=nullptr: cc.continue in the body is a back-edge,
        // not a loop exit, so we don't map it to the loop results.
        analyzeBlock(*bodyBlock, /*doCollectInteractions=*/false, nullptr,
                     analysisOk, sources, returnsToRemove, wireToVirtualQ,
                     userQubitsMeasured, finalQubitWire, lastSource,
                     interactions, userVirtualQubits);
        if (!analysisOk)
          return;
        // Overwrite finalQubitWire with the loop results; the body
        // analysis may have updated them to body-internal wire values.
        for (Value res : loopOp->getResults())
          if (isa<cudaq::quake::WireType>(res.getType()))
            finalQubitWire[wireToVirtualQ[res].index] = res;
      } else if (cudaq::quake::isSupportedMappingOperation(&op)) {
        if (!cudaq::quake::isLinearValueForm(&op)) {
          if (nonComposable) {
            llvm::errs() << "This is not SSA form: " << op << '\n';
            llvm::errs() << "isa<cudaq::quake::NullWireOp>() = "
                         << isa<cudaq::quake::NullWireOp>(&op) << '\n';
            llvm::errs() << "isAllReferences() = "
                         << cudaq::quake::isAllReferences(&op) << '\n';
            llvm::errs() << "isWrapped() = " << cudaq::quake::isWrapped(&op)
                         << '\n';
            getOperation().emitError("The mapper requires value semantics.");
            signalPassFailure();
          }
          LLVM_DEBUG(llvm::dbgs() << "operation is not in proper value form");
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
        if (auto gate = dyn_cast<cudaq::quake::OperatorInterface>(op);
            gate && gate.getControls().size() + gate.getTargets().size() > 2) {
          op.emitOpError(
              "qubit mapping is defined over a two-qubit coupling graph; "
              "decompose multi-qubit gates before mapping");
          signalPassFailure();
          LLVM_DEBUG(llvm::dbgs() << "operator with >2 qubits not expected");
          analysisOk = false;
          return;
        }
        if (isa<cudaq::quake::MeasurementInterface>(op))
          for (auto virtualQ : virtualOperands)
            userQubitsMeasured.push_back(virtualQ.index);
        if (doCollectInteractions && isa<cudaq::quake::OperatorInterface>(op) &&
            wireOperands.size() == 2) {
          interactions.addInteraction(virtualOperands[0].index,
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
        LLVM_DEBUG(llvm::dbgs() << "unsupported quantum operation in mapper\n");
        analysisOk = false;
        return;
      }
    }
  }

  /// Search `block` over `seeds` for the layout with the fewest swaps, then
  /// record its RoutingResult (recursing through nested control flow) into
  /// `blockResults`. `sources` are the block's entry wires (borrow results for
  /// the outer block, wire block arguments for a nested one). Returns the
  /// winning final layout, whose placement feeds the mapping attributes.
  cudaq::Placement routeBlock(
      Block &block, ArrayRef<Value> sources,
      ArrayRef<SmallVector<unsigned>> seeds, unsigned numV, unsigned numPhy,
      SearchStrategy searchStrategy,
      const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ,
      JoinPointStrategy &joinStrategy,
      DenseMap<Block *, RoutingResult> &blockResults) {
    RoutingProblem problem = buildRoutingProblem(block, sources, wireToVirtualQ);
    RoutingSearchStrategy search(
        *deviceInstance, problem, searchStrategy == SearchStrategy::Sabre,
        extendedLayerSize, extendedLayerWeight, decayDelta, roundsDecayReset,
        minStallSwapBudget, stallSwapBudgetPerQubit);
    RoutingSearchStrategy::Selection selection = search.run(seeds, numV, numPhy);
    buildBlockResults(block, std::move(selection.result), numV, numPhy,
                      searchStrategy, wireToVirtualQ, joinStrategy,
                      blockResults);
    return std::move(selection.finalLayout);
  }

  /// For `blk` (outer or a branch block), store its RoutingResult in
  /// `blockResults`. Routes branch blocks using the placement at the point each
  /// cc::IfOp is reached as the seed. Recurses for nested IfOps.
  void buildBlockResults(
      Block &blk, RoutingResult flat, unsigned numV, unsigned numPhy,
      SearchStrategy searchStrategy,
      const DenseMap<Value, cudaq::Placement::VirtualQ> &wireToVirtualQ,
      JoinPointStrategy &joinStrategy,
      DenseMap<Block *, RoutingResult> &blockResults) {
    RoutingResult result;
    result.initialLayout = flat.initialLayout;
    result.swapCount = flat.swapCount;

    SmallVector<unsigned> replayVqToPhy(numV);
    SmallVector<unsigned> replayPhyToVQ(numPhy, UINT_MAX);
    for (unsigned v = 0; v < numV; ++v) {
      replayVqToPhy[v] = flat.initialLayout[v];
      replayPhyToVQ[flat.initialLayout[v]] = v;
    }
    unsigned srcIdx = 0;

    // Route one nested block (an if branch, loop body, or loop step) using the
    // current replay layout as its entry seed, then recurse to fill its
    // blockResults entry. Reconciliation is left to the caller, since if-joins
    // and loop back-edges reconcile different predecessor sets.
    auto routeNested = [&](Block &nested) {
      SmallVector<Value> sources;
      for (auto arg : nested.getArguments())
        if (isa<cudaq::quake::WireType>(arg.getType()))
          sources.push_back(arg);
      routeBlock(nested, sources, {SmallVector<unsigned>(replayVqToPhy)}, numV,
                 numPhy, searchStrategy, wireToVirtualQ, joinStrategy,
                 blockResults);
    };

    // flushTo processes all trace events up to limit, routing branch blocks
    // inline when a Kind::If event is encountered.
    std::function<void(unsigned)> flushTo;
    flushTo = [&](unsigned limit) {
      while (srcIdx < limit) {
        RoutingEvent &ev = flat.trace[srcIdx++];
        if (ev.kind == RoutingEvent::Kind::Swap) {
          unsigned p0 = ev.phys[0].index, p1 = ev.phys[1].index;
          unsigned v0 = replayPhyToVQ[p0], v1 = replayPhyToVQ[p1];
          if (v0 != UINT_MAX)
            replayVqToPhy[v0] = p1;
          if (v1 != UINT_MAX)
            replayVqToPhy[v1] = p0;
          std::swap(replayPhyToVQ[p0], replayPhyToVQ[p1]);
          result.trace.push_back(std::move(ev));
        } else if (ev.kind == RoutingEvent::Kind::Gate) {
          result.trace.push_back(std::move(ev));
        } else if (ev.kind == RoutingEvent::Kind::If) {
          auto ifOp = cast<cudaq::cc::IfOp>(ev.op);
          SmallVector<Block *> branchBlocks;
          for (Region *region : ifOp.getRegions()) {
            if (region->empty())
              continue;
            routeNested(region->front());
            branchBlocks.push_back(&region->front());
          }
          // Reconcile the branches at the if-join so they exit at a common
          // layout. Collect the predecessors after all branches are routed, as
          // building each result may rehash blockResults.
          SmallVector<RoutingResult *> branchResults;
          for (Block *b : branchBlocks)
            branchResults.push_back(&blockResults[b]);
          joinStrategy.reconcile(branchResults);
          assert(!ifOp.hasElse() ||
                 blockResults[&ifOp.getThenRegion().front()].exitLayout ==
                     blockResults[&ifOp.getElseRegion().front()].exitLayout);
          result.trace.push_back(std::move(ev));
        } else if (ev.kind == RoutingEvent::Kind::Loop) {
          auto loopOp = cast<cudaq::cc::LoopOp>(ev.op);
          auto *bodyBlock = loopOp.getDoEntryBlock();
          routeNested(*bodyBlock);
          // Reconcile the loop back-edge: the loop invariant is the entry
          // layout, restored before the body terminator each iteration.
          RoutingResult *bodyResult = &blockResults[bodyBlock];
          joinStrategy.reconcile(bodyResult);
          // Route the step block if present (for-loop style). It receives wires
          // in the restored (entry) layout, the same seed used for the body.
          if (loopOp.hasStep())
            routeNested(*loopOp.getStepBlock());
          result.trace.push_back(std::move(ev));
        } else {
          llvm_unreachable("unhandled RoutingEvent::Kind");
        }
      }
    };

    flushTo(flat.trace.size());
    result.exitLayout =
        SmallVector<unsigned>(replayVqToPhy.begin(), replayVqToPhy.end());
    blockResults[&blk] = std::move(result);
  }

  void runOnOperation() override {
    if (deviceBypass)
      return;

    // See MappingPrep::runOnOperation: a composable run with no device is a
    // no-op, so never dereference an unset device.
    if (!deviceInstance)
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

    // Borrow identities index device-sized tables (`sources`,
    // `userVirtualQubits`, `interactions`) during the scan below, so a circuit
    // that needs more qubits than the device provides must be rejected before
    // those writes go out of bounds. `highestIdentity` is the largest identity
    // in the function.
    if (*highestIdentity >= deviceNumQubits) {
      if (nonComposable) {
        func.emitOpError(
            "Too many qubits [" + std::to_string(*highestIdentity + 1) +
            "] for device [" + std::to_string(deviceNumQubits) + "]");
        signalPassFailure();
      }
      LLVM_DEBUG(llvm::dbgs() << "exceeded available qubits for target");
      return;
    }

    SmallVector<cudaq::quake::BorrowWireOp> sources(deviceNumQubits);
    SmallVector<cudaq::quake::ReturnWireOp> returnsToRemove;
    DenseMap<Value, cudaq::Placement::VirtualQ> wireToVirtualQ;
    SmallVector<std::size_t> userQubitsMeasured;
    DenseMap<std::size_t, Value> finalQubitWire;
    Operation *lastSource = nullptr;

    // Resolve the placement and search strategies before scanning the circuit.
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

    // Reject loop bodies that are not yet supported: multi-block, else
    // regions, or break statements.
    auto loopCheckResult = func.walk([&](cudaq::cc::LoopOp loopOp) {
      if (!loopOp.getBodyRegion().hasOneBlock() || loopOp.hasPythonElse()) {
        if (nonComposable) {
          loopOp.emitOpError(
              "mapper cannot handle loops with multi-block or else");
          signalPassFailure();
        }
        return WalkResult::interrupt();
      }
      if (loopOp.hasBreakInBody()) {
        if (nonComposable) {
          loopOp.emitOpError(
              "mapper cannot handle loops with break statements");
          signalPassFailure();
        }
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (loopCheckResult.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << "NYI: complex loop body in mapper\n");
      return;
    }

    // Reject measurements not directly inside the function — measure order must
    // be preserved and cannot yet be reconciled across branches or loops.
    auto measureCheckResult =
        func.walk([&](cudaq::quake::MeasurementInterface meas) {
          if (isa<func::FuncOp>(meas->getParentOp()))
            return WalkResult::advance();
          if (nonComposable) {
            meas->emitOpError(
                "mapper cannot handle measurements inside branches or loops");
            signalPassFailure();
          }
          return WalkResult::interrupt();
        });
    if (measureCheckResult.wasInterrupted()) {
      LLVM_DEBUG(llvm::dbgs() << "NYI: measurements inside branches\n");
      return;
    }

    // Interaction data is required by every placement strategy: greedy and
    // auto use it to build seeds, while identity uses it to reject interactions
    // that cross disconnected device islands.
    VirtualInteractionGraph interactions(deviceNumQubits);
    SmallVector<bool> userVirtualQubits(deviceNumQubits, false);

    bool analysisOk = true;
    analyzeBlock(block, /*doCollectInteractions=*/true, nullptr, analysisOk,
                 sources, returnsToRemove, wireToVirtualQ, userQubitsMeasured,
                 finalQubitWire, lastSource, interactions, userVirtualQubits);
    if (!analysisOk)
      return;

    const unsigned numV = sources.size();
    const unsigned numPhy = deviceInstance->getNumQubits();

    SmallVector<SmallVector<unsigned>> seeds =
        buildPlacementSeeds(placementStrategy, numV, *deviceInstance,
                            interactions, userVirtualQubits);
    std::optional<std::pair<unsigned, unsigned>> identityBlockedInteraction;
    auto shouldDiscardSeed = [&](ArrayRef<unsigned> seed) {
      auto blocked =
          findUnroutableInteraction(seed, *deviceInstance, interactions);
      if (!blocked)
        return false;
      if (placementStrategy == PlacementStrategy::Identity &&
          !identityBlockedInteraction)
        identityBlockedInteraction = blocked;
      return true;
    };
    llvm::erase_if(seeds, shouldDiscardSeed);
    if (seeds.empty()) {
      if (identityBlockedInteraction) {
        func.emitError("cannot place two-qubit interaction between virtual "
                       "qubits " +
                       std::to_string(identityBlockedInteraction->first) +
                       " and " +
                       std::to_string(identityBlockedInteraction->second) +
                       " on disconnected device topology");
      } else {
        func.emitError("could not find a routable initial layout for "
                       "disconnected device topology");
      }
      signalPassFailure();
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

    // Search the seeds for the fewest-swap layout and record every block's
    // RoutingResult, reconciling control-flow joins with the Stage 1.5 policy.
    SmallVector<Value> sourceValues;
    for (auto borrow : sources)
      sourceValues.push_back(borrow.getResult());
    DenseMap<Block *, RoutingResult> blockResults;
    RestoreToEntryStrategy joinStrategy;
    cudaq::Placement bestLayout =
        routeBlock(block, sourceValues, seeds, numV, numPhy, searchStrategy,
                   wireToVirtualQ, joinStrategy, blockResults);

    // Emit the selected result onto the IR exactly once.
    RoutingEmitter emitter(wireToVirtualQ, numPhy, blockResults);
    auto phyToWire = emitter.emit(block, sources, blockResults[&block]);
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
        Value finalWire = phyToWire[blockResults[&block].initialLayout[i]];
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
