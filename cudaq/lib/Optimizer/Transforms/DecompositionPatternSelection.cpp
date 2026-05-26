/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "PassDetails.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <unordered_map>
#include <unordered_set>

using namespace mlir;

namespace {

// ConversionTarget and OperatorInfo, parsed from target basis strings such as
// ["x", "x(1)", "z"]
struct OperatorInfo {
  std::string name;
  std::size_t numControls;
  bool isAdj;

  OperatorInfo(StringRef infoStr) : name(), numControls(0), isAdj(false) {
    auto nameEnd = infoStr.find_first_of("(<");
    name = infoStr.take_front(nameEnd).str();
    if (nameEnd < infoStr.size())
      infoStr = infoStr.drop_front(nameEnd);

    if (infoStr.consume_front("<adj>"))
      isAdj = true;

    if (infoStr.consume_front("(")) {
      infoStr = infoStr.ltrim();
      if (infoStr.consume_front("n"))
        numControls = std::numeric_limits<std::size_t>::max();
      else
        infoStr.consumeInteger(10, numControls);
      assert(infoStr.trim().consume_front(")"));
    }
  }

  bool operator==(const OperatorInfo &other) const {
    return name == other.name && numControls == other.numControls &&
           isAdj == other.isAdj;
  }

  bool isUnbounded() const {
    return numControls == std::numeric_limits<std::size_t>::max();
  }

  std::string str() const {
    std::string result = name;
    if (isAdj)
      result += "<adj>";
    if (isUnbounded())
      result += "(n)";
    else if (numControls != 0)
      result += "(" + std::to_string(numControls) + ")";
    return result;
  }

  /// Check if this gate covers another gate.
  bool covers(const OperatorInfo &other) const {
    if (name != other.name || isAdj != other.isAdj)
      return false;
    // Pattern metadata may use (n) as a wildcard, but target legality is
    // directional: x(n) covers x(1), while x(1) does not cover x(n).
    if (isUnbounded())
      return true;
    return numControls == other.numControls;
  }

  /// Check if this basis entry makes another gate legal, matching
  /// ConversionTarget semantics. A concrete basis entry does not make an
  /// unbounded source pattern legal.
  bool makesLegal(const OperatorInfo &other) const { return covers(other); }
};

static std::optional<std::size_t>
getKnownNumControls(cudaq::quake::OperatorInterface op) {
  std::size_t numControls = 0;
  for (auto control : op.getControls()) {
    if (auto veq = dyn_cast<cudaq::quake::VeqType>(control.getType())) {
      if (!veq.hasSpecifiedSize())
        return std::nullopt;
      numControls += veq.getSize();
      continue;
    }
    numControls += 1;
  }
  return numControls;
}

struct BasisTarget : public ConversionTarget {

  BasisTarget(MLIRContext &context, ArrayRef<std::string> targetBasis)
      : ConversionTarget(context) {
    constexpr std::size_t unbounded = std::numeric_limits<std::size_t>::max();

    // Parse the list of target operations and build a set of legal operations
    for (const std::string &targetInfo : targetBasis)
      legalOperatorSet.emplace_back(targetInfo);

    addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                    cudaq::cc::CCDialect, func::FuncDialect,
                    math::MathDialect>();
    addDynamicallyLegalDialect<cudaq::quake::QuakeDialect>([&](Operation *op) {
      if (auto optor = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
        auto name = optor->getName().stripDialect();
        auto numControls = getKnownNumControls(optor);
        for (auto info : legalOperatorSet) {
          if (info.name != name)
            continue;
          if (info.numControls == unbounded)
            return true;
          if (numControls && *numControls == info.numControls)
            return true;
        }
        return false;
      }

      // Handle quake.exp_pauli.
      if (isa<cudaq::quake::ExpPauliOp>(op)) {
        // If the target defines it as a legal op, return true, else false.
        return std::find_if(legalOperatorSet.begin(), legalOperatorSet.end(),
                            [](auto &&el) { return el.name == "exp_pauli"; }) !=
               legalOperatorSet.end();
      }

      return true;
    });
  }

  SmallVector<OperatorInfo, 8> legalOperatorSet;
};

} // namespace

//===----------------------------------------------------------------------===//
// std::hash specialization for OperatorInfo
//===----------------------------------------------------------------------===//

namespace std {
template <>
struct hash<OperatorInfo> {
  std::size_t operator()(const OperatorInfo &info) const {
    return llvm::hash_combine(info.name, info.numControls, info.isAdj);
  }
};
} // namespace std

// Computes a hash of the given unordered set using the hashes of the elements
// in the set.
template <typename T>
std::size_t computeSetHash(const std::unordered_set<T> &set) {
  std::vector<std::size_t> hashes;
  for (const auto &elem : set) {
    hashes.push_back(std::hash<T>()(elem));
  }
  std::sort(hashes.begin(), hashes.end());
  return llvm::hash_combine_range(hashes.begin(), hashes.end());
}

namespace {
//===----------------------------------------------------------------------===//
// Decomposition Graph for Pattern Selection
//===----------------------------------------------------------------------===//

/// DecompositionGraph constructs a hypergraph of decomposition patterns based
/// on pattern metadata and performs backward traversal to select patterns that
/// decompose to a basis.
///
/// Specifically, the decomposition graph is defined as a hypergraph in which
/// nodes are gate types and hyperedges are rewrite patterns connecting the
/// matched gate type to all newly inserted gate types.
class DecompositionGraph {
public:
  DecompositionGraph() = default;

  using PatternSelection = std::map<std::string, std::vector<std::string>>;

  struct VariantEdge {
    std::string patternName;
    cudaq::DecompositionPatternVariant variant;
  };

  struct ResolvedVariant {
    std::string patternName;
    std::string sourceOp;
    llvm::SmallVector<std::string> targetOps;
  };

  /// Construct a decomposition pattern graph from a collection of pattern
  /// types.
  DecompositionGraph(
      llvm::StringMap<std::unique_ptr<cudaq::DecompositionPatternType>>
          patterns)
      : patternTypes(std::move(patterns)) {
    // Build the graph from pattern metadata
    for (const auto &pattern : patternTypes) {
      const auto variants = pattern.getValue()->getVariants();
      for (const auto &variant : variants) {
        for (const auto &targetGate : variant.targetOps)
          targetToVariants[OperatorInfo(llvm::StringRef(targetGate))].push_back(
              VariantEdge{pattern.getKey().str(), variant});
      }
    }
  }

  /// Create a DecompositionGraph from the registry entries.
  static DecompositionGraph fromRegistry() {
    llvm::StringMap<std::unique_ptr<cudaq::DecompositionPatternType>> patterns;
    for (const auto &patternType :
         cudaq::DecompositionPatternTypeRegistry::entries()) {
      patterns.insert({patternType.getName(), patternType.instantiate()});
    }
    return DecompositionGraph(std::move(patterns));
  }

  /// Return all variants that have the given gate as one of their targets.
  llvm::SmallVector<ResolvedVariant>
  incomingVariants(const OperatorInfo &gate) const {
    llvm::SmallVector<ResolvedVariant> result;
    for (const auto &[targetInfo, variants] : targetToVariants) {
      if (gate.covers(targetInfo)) {
        for (const auto &variant : variants)
          result.push_back(resolveVariant(variant, std::nullopt));
        continue;
      }

      // A concrete controlled gate can prove the corresponding (n) target
      // reachable for that same arity. Bare gates intentionally do not promote
      // to (n).
      if (!targetInfo.isUnbounded() || gate.isUnbounded() ||
          gate.numControls == 0 || targetInfo.name != gate.name ||
          targetInfo.isAdj != gate.isAdj)
        continue;

      for (const auto &variant : variants)
        result.push_back(resolveVariant(variant, gate.numControls));
    }
    return result;
  }

  /// Select subset of patterns relevant to decomposing to the given basis
  /// gates.
  ///
  /// The result of the pattern selection are cached, so that successive calls
  /// with the same arguments will be O(1).
  ///
  /// @param patterns The pattern set to add the selected patterns to
  /// @param basisGates The basis gates to decompose to
  /// @param disabledPatterns The patterns to disable
  void selectPatterns(RewritePatternSet &patterns,
                      const std::unordered_set<OperatorInfo> &basisGates,
                      const std::unordered_set<std::string> &disabledPatterns) {
    auto hashVal = llvm::hash_combine(computeSetHash(basisGates),
                                      computeSetHash(disabledPatterns));

    if (!patternSelectionCache.contains(hashVal)) {
      patternSelectionCache[hashVal] =
          computePatternSelection(basisGates, disabledPatterns);
    }

    for (const auto &[patternName, enabledSources] :
         patternSelectionCache[hashVal]) {
      const auto &pattern = getPatternType(patternName);
      patterns.add(pattern->create(patterns.getContext(), 1, enabledSources));
    }
  }

  PatternSelection selectPatternSourceOps(
      const std::unordered_set<OperatorInfo> &basisGates,
      const std::unordered_set<std::string> &disabledPatterns = {}) const {
    return computePatternSelection(basisGates, disabledPatterns);
  }

private:
  const std::unique_ptr<cudaq::DecompositionPatternType> &
  getPatternType(const std::string &patternName) const {
    auto patternType = patternTypes.find(patternName);
    assert(patternType != patternTypes.end() && "pattern not found");
    return patternType->getValue();
  }

  /// Use Dijkstra's algorithm to compute the shortest decomposition path from
  /// every reachable gate type to the basis gates.
  ///
  /// This selects a unique decomposition path for each gate in the past of the
  /// basis gates in the decomposition graph, such that the number of patterns
  /// applied is minimized. `disabledPatterns` are ignored during the traversal
  /// and hence never selected.
  ///
  /// @param basisGates The set of basis gates to decompose to
  /// @param disabledPatterns The patterns to disable
  /// @return Selected source variants grouped by pattern name.
  PatternSelection computePatternSelection(
      const std::unordered_set<OperatorInfo> &basisGates,
      const std::unordered_set<std::string> &disabledPatterns) const {

    // An element in the priority queue of the Dijkstra algorithm (ordered by
    // smallest distance)
    struct GateDistancePair {
      OperatorInfo gate;
      std::size_t distance;
      std::optional<ResolvedVariant> outgoingVariant;

      bool operator<(const GateDistancePair &other) const {
        // We want to order by smallest distance, so we invert the comparison
        return distance > other.distance;
      }
    };

    // Map: visited gate -> distance from the basis gates
    std::unordered_map<OperatorInfo, std::size_t> visitedGates;
    // The set of selected pattern source variants to return.
    std::map<std::string, std::set<std::string>> selectedPatterns;
    // Priority queue of gates to visit, sorted by smallest distance from the
    // basis gates
    std::priority_queue<GateDistancePair> gatesToVisit;

    auto isBasisGate = [&](const OperatorInfo &gate) {
      for (const auto &basisGate : basisGates)
        if (basisGate.makesLegal(gate))
          return true;
      return false;
    };

    // Initialize the priority queue with the basis gates
    for (const auto &gate : basisGates) {
      gatesToVisit.push({gate, 0, std::nullopt});
    }

    /// Find the distance for a gate, handling unbounded (n) control counts.
    /// Exact hash lookup first for the common case, then a scan when the
    /// query or any visited entry uses unbounded controls.
    auto findGateDist = [&](const OperatorInfo &gate) -> std::size_t {
      auto it = visitedGates.find(gate);
      if (it != visitedGates.end())
        return it->second;
      // Scan for wildcard matches.
      std::size_t best = std::numeric_limits<std::size_t>::max();
      for (const auto &[visited, dist] : visitedGates) {
        if (visited.covers(gate))
          best = std::min(best, dist);
      }
      return best;
    };

    /// Compute the maximum distance from a pattern's targets to the basis
    /// gates.
    auto getPatternDist = [&](const ResolvedVariant &variant) {
      std::vector<std::size_t> targetDistances;
      for (const auto &targetGate : variant.targetOps)
        targetDistances.push_back(
            findGateDist(OperatorInfo(llvm::StringRef(targetGate))));
      return *std::max_element(targetDistances.begin(), targetDistances.end());
    };

    while (!gatesToVisit.empty()) {
      auto [gate, dist, outgoingVariant] = gatesToVisit.top();
      gatesToVisit.pop();

      auto [_, success] = visitedGates.insert({gate, dist});
      if (!success) {
        // Gate already visited
        continue;
      }

      if (outgoingVariant.has_value()) {
        if (isBasisGate(gate))
          continue;
        selectedPatterns[outgoingVariant->patternName].insert(
            outgoingVariant->sourceOp);
      }

      for (const auto &variant : incomingVariants(gate)) {
        if (disabledPatterns.contains(variant.patternName)) {
          // Ignore disabled patterns
          continue;
        }
        std::size_t dist = getPatternDist(variant);
        if (dist < std::numeric_limits<std::size_t>::max()) {
          gatesToVisit.push({OperatorInfo(llvm::StringRef(variant.sourceOp)),
                             dist + 1, variant});
        }
      }
    }

    PatternSelection result;
    for (const auto &[patternName, sourceOps] : selectedPatterns)
      result[patternName] =
          std::vector<std::string>(sourceOps.begin(), sourceOps.end());
    return result;
  }

  static std::string resolveOp(llvm::StringRef op,
                               std::optional<std::size_t> numControls) {
    if (!numControls)
      return op.str();
    OperatorInfo info(op);
    if (!info.isUnbounded())
      return op.str();
    info.numControls = *numControls;
    return info.str();
  }

  static ResolvedVariant
  resolveVariant(const VariantEdge &edge,
                 std::optional<std::size_t> numControls) {
    ResolvedVariant result{
        edge.patternName, resolveOp(edge.variant.sourceOp, numControls), {}};
    for (const auto &targetOp : edge.variant.targetOps)
      result.targetOps.push_back(resolveOp(targetOp, numControls));
    return result;
  }

  //===--------------------------------------------------------------------===//
  // Data structures for the graph definition
  //===--------------------------------------------------------------------===//

  /// All pattern types in the graph, keyed by pattern name.
  llvm::StringMap<std::unique_ptr<cudaq::DecompositionPatternType>>
      patternTypes;

  /// Map: target gate -> variants that produce it
  std::unordered_map<OperatorInfo, SmallVector<VariantEdge>> targetToVariants;

  //===--------------------------------------------------------------------===//
  // Other data (cache)
  //===--------------------------------------------------------------------===//

  /// Cache for `selectPatterns`: hash of basis gates, disabled patterns,
  /// enabled patterns -> selected patterns
  std::unordered_map<std::size_t, PatternSelection> patternSelectionCache;
};
} // namespace

std::unique_ptr<ConversionTarget>
cudaq::createBasisTarget(MLIRContext &context,
                         ArrayRef<std::string> targetBasis) {
  return std::make_unique<BasisTarget>(context, targetBasis);
}

void cudaq::selectDecompositionPatterns(
    RewritePatternSet &patterns, ArrayRef<std::string> targetBasis,
    ArrayRef<std::string> disabledPatterns) {
  // Static local graph - constructed once and reused
  static DecompositionGraph graph = DecompositionGraph::fromRegistry();

  BasisTarget target(*patterns.getContext(), targetBasis);

  // Convert targetBasis, disabledPatterns and enabledPatterns to sets for O(1)
  // lookup
  std::unordered_set<OperatorInfo> basisGatesSet(
      target.legalOperatorSet.begin(), target.legalOperatorSet.end());
  std::unordered_set<std::string> disabledPatternsSet(disabledPatterns.begin(),
                                                      disabledPatterns.end());

  return graph.selectPatterns(patterns, basisGatesSet, disabledPatternsSet);
}
