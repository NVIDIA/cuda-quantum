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

constexpr std::size_t UNBOUNDED = std::numeric_limits<std::size_t>::max();

cudaq::detail::OperatorInfo::OperatorInfo(llvm::StringRef infoStr)
    : name(), numControls(0), isAdj(false) {
  auto nameEnd = infoStr.find_first_of("(<");
  name = infoStr.take_front(nameEnd).str();
  if (nameEnd < infoStr.size())
    infoStr = infoStr.drop_front(nameEnd);

  if (infoStr.consume_front("<adj>"))
    isAdj = true;

  if (infoStr.consume_front("(")) {
    infoStr = infoStr.ltrim();
    if (infoStr.consume_front("n"))
      numControls = UNBOUNDED;
    else
      infoStr.consumeInteger(10, numControls);
    assert(infoStr.trim().consume_front(")"));
  }
}

bool cudaq::detail::OperatorInfo::isUnbounded() const {
  return numControls == UNBOUNDED;
}

std::optional<cudaq::detail::OperatorInfo>
cudaq::detail::OperatorInfo::join(const OperatorInfo &other) const {
  if (name.empty())
    return other;
  if (other.name.empty())
    return *this;
  if (isAdj != other.isAdj || name != other.name)
    return std::nullopt;
  if (numControls == other.numControls)
    return *this;
  // Mismatching arities: promote to unbounded
  OperatorInfo result = *this;
  result.numControls = UNBOUNDED;
  return result;
}

std::string cudaq::detail::OperatorInfo::str() const {
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
bool cudaq::detail::OperatorInfo::covers(const OperatorInfo &other) const {
  return join(other) == *this;
}

namespace {

struct BasisTarget : public ConversionTarget {

  BasisTarget(MLIRContext &context, ArrayRef<std::string> targetBasis)
      : ConversionTarget(context) {

    // Parse the list of target operations and build a set of legal operations
    for (const std::string &targetInfo : targetBasis)
      legalOperatorSet.emplace_back(targetInfo);

    addLegalDialect<arith::ArithDialect, cf::ControlFlowDialect,
                    cudaq::cc::CCDialect, func::FuncDialect,
                    math::MathDialect>();
    addDynamicallyLegalDialect<cudaq::quake::QuakeDialect>([&](Operation *op) {
      if (auto optor = dyn_cast<cudaq::quake::OperatorInterface>(op)) {
        auto name = optor->getName().stripDialect();
        auto numControls = cudaq::getKnownNumControls(optor);
        for (auto info : legalOperatorSet) {
          if (info.name != name)
            continue;
          if (info.numControls == UNBOUNDED)
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

  SmallVector<cudaq::detail::OperatorInfo, 8> legalOperatorSet;
};

} // namespace

//===----------------------------------------------------------------------===//
// std::hash specialization for OperatorInfo
//===----------------------------------------------------------------------===//

namespace std {
template <>
struct hash<cudaq::detail::OperatorInfo> {
  std::size_t operator()(const cudaq::detail::OperatorInfo &info) const {
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

/// Get all control arities in the basis gates that match the given gate type.
static std::vector<std::size_t> getExistingControlCounts(
    const cudaq::detail::OperatorInfo &gate,
    const std::unordered_set<cudaq::detail::OperatorInfo> &basisGates) {
  std::vector<std::size_t> result;
  for (const auto &basisGate : basisGates) {
    if (basisGate.name == gate.name && basisGate.isAdj == gate.isAdj) {
      result.push_back(basisGate.numControls);
    }
  }
  return result;
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
  using OperatorInfo = cudaq::detail::OperatorInfo;

public:
  DecompositionGraph() = default;

  /// Construct a decomposition pattern graph from a collection of pattern
  /// types.
  DecompositionGraph(
      llvm::StringMap<std::unique_ptr<cudaq::DecompositionPatternType>>
          patterns)
      : patternTypes(std::move(patterns)) {
    // Build the graph from pattern metadata
    for (const auto &pattern : patternTypes) {
      auto targetGates = pattern.getValue()->getTargetOps();
      for (const auto &targetGate : targetGates)
        targetToPatterns[targetGate].push_back(pattern.getKey().str());
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

  /// Return all patterns that have the given gate (or its unbounded version) as
  /// one of their targets.
  llvm::SmallVector<std::string>
  incomingPatterns(const OperatorInfo &gate) const {
    llvm::SmallVector<std::string> result;
    auto it = targetToPatterns.find(gate);
    if (it != targetToPatterns.end())
      result.append(it->second.begin(), it->second.end());

    if (gate.isUnbounded())
      return result;

    // Add patterns for the unbounded version of the gate.
    auto unboundedGate = gate;
    unboundedGate.numControls = UNBOUNDED;
    std::set<std::string> knownPatterns(result.begin(), result.end());
    it = targetToPatterns.find(unboundedGate);
    if (it != targetToPatterns.end()) {
      for (const auto &pattern : it->second) {
        if (!knownPatterns.contains(pattern))
          result.push_back(pattern);
      }
    }

    return result;
  }

  /// Select subset of patterns relevant to decomposing to the given basis
  /// gates.
  ///
  /// The result of the pattern selection are cached, so that successive calls
  /// with the same arguments will be fast.
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

    for (const auto &patternName : patternSelectionCache[hashVal]) {
      auto ctx = patterns.getContext();
      patterns.add(constructPattern(patternName, ctx, basisGates));
    }
  }

private:
  const std::unique_ptr<cudaq::DecompositionPatternType> &
  getPatternType(llvm::StringRef patternName) const {
    auto patternType = patternTypes.find(patternName);
    assert(patternType != patternTypes.end() && "pattern not found");
    return patternType->getValue();
  }

  std::unique_ptr<mlir::RewritePattern>
  constructPattern(llvm::StringRef patternName, mlir::MLIRContext *context,
                   const std::unordered_set<OperatorInfo> &basisGates) const {
    const auto &pattern = getPatternType(patternName);
    // Patterns with unbounded (n) control counts get lower benefit so
    // that specific patterns (e.g., CR1ToCX for r1(1)) are preferred
    // when both match the same op.
    auto sourceInfo = pattern->getSourceOp();
    PatternBenefit benefit;
    std::vector<std::size_t> disabledControlCounts;
    if (sourceInfo.isUnbounded()) {
      benefit = 1;
      disabledControlCounts = getExistingControlCounts(sourceInfo, basisGates);
    } else {
      benefit = 2;
    }
    return pattern->create(context, benefit, disabledControlCounts);
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
  /// @return A vector of selected pattern names
  std::vector<std::string> computePatternSelection(
      const std::unordered_set<OperatorInfo> &basisGates,
      const std::unordered_set<std::string> &disabledPatterns) const {

    // An element in the priority queue of the Dijkstra algorithm (ordered by
    // smallest distance)
    struct GateDistancePair {
      OperatorInfo gate;
      std::size_t distance;
      std::optional<std::string> outgoingPattern;

      bool operator<(const GateDistancePair &other) const {
        // We want to order by smallest distance, so we invert the comparison
        return distance > other.distance;
      }
    };

    // Map: visited gate -> distance from the basis gates
    std::unordered_map<OperatorInfo, std::size_t> visitedGates;
    // The set of selected patterns to return
    std::vector<std::string> selectedPatterns;
    // Priority queue of gates to visit, sorted by smallest distance from the
    // basis gates
    std::priority_queue<GateDistancePair> gatesToVisit;

    auto isBasisGate = [&](const cudaq::detail::OperatorInfo &gate) {
      if (basisGates.contains(gate))
        return true;
      if (gate.isUnbounded())
        return false;
      // Check for unbounded version of the gate.
      auto unboundedGate = gate;
      unboundedGate.numControls = UNBOUNDED;
      if (basisGates.contains(unboundedGate))
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

      if (gate.isUnbounded())
        return UNBOUNDED;

      // Check for distance of the unbounded version of the gate.
      auto unboundedGate = gate;
      unboundedGate.numControls = UNBOUNDED;
      it = visitedGates.find(unboundedGate);
      if (it != visitedGates.end())
        return it->second;

      return UNBOUNDED;
    };

    /// Compute the maximum distance from a pattern's targets to the basis
    /// gates.
    auto getPatternDist = [&](const auto &variant) {
      std::size_t maxDistance = 0;
      for (const auto &targetGate : variant.targetOps)
        maxDistance = std::max(maxDistance, findGateDist(targetGate));
      return maxDistance;
    };

    while (!gatesToVisit.empty()) {
      auto [gate, dist, outgoingPattern] = gatesToVisit.top();
      gatesToVisit.pop();

      auto [_, success] = visitedGates.insert({gate, dist});
      if (!success) {
        // Gate already visited
        continue;
      }

      if (outgoingPattern.has_value())
        selectedPatterns.push_back(*outgoingPattern);

      for (const auto &patternName : incomingPatterns(gate)) {
        if (disabledPatterns.contains(patternName)) {
          // Ignore disabled patterns
          continue;
        }
        const auto &pattern = getPatternType(patternName);
        for (const auto &variant : pattern->findCoveringVariants(gate)) {
          std::size_t dist = getPatternDist(variant);
          auto sourceOp = variant.sourceOp;
          if (dist < UNBOUNDED && !isBasisGate(sourceOp) &&
              !visitedGates.contains(sourceOp))
            gatesToVisit.push({sourceOp, dist + 1, patternName});
        }
      }
    }

    return selectedPatterns;
  }

  //===--------------------------------------------------------------------===//
  // Data structures for the graph definition
  //===--------------------------------------------------------------------===//

  /// All pattern types in the graph, keyed by pattern name.
  llvm::StringMap<std::unique_ptr<cudaq::DecompositionPatternType>>
      patternTypes;

  /// Map: target gate -> patterns that produce it
  std::unordered_map<cudaq::detail::OperatorInfo, SmallVector<std::string>>
      targetToPatterns;

  //===--------------------------------------------------------------------===//
  // Other data (cache)
  //===--------------------------------------------------------------------===//

  /// Cache for `selectPatterns`: hash of basis gates, disabled patterns,
  /// enabled patterns -> selected patterns
  std::unordered_map<std::size_t, std::vector<std::string>>
      patternSelectionCache;

  friend class BaseDecompositionPatternSelectionTest;
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
  std::unordered_set<detail::OperatorInfo> basisGatesSet(
      target.legalOperatorSet.begin(), target.legalOperatorSet.end());
  std::unordered_set<std::string> disabledPatternsSet(disabledPatterns.begin(),
                                                      disabledPatterns.end());

  return graph.selectPatterns(patterns, basisGatesSet, disabledPatternsSet);
}

cudaq::DecompositionPatternType::DecompositionPatternType(
    std::vector<DecompositionPatternVariant> variants_)
    : variants(std::move(variants_)) {
  for (const auto &variant : variants) {
    auto join = sourceOp.join(variant.sourceOp);
    assert(join.has_value() &&
           "all source ops of pattern variants must be joinable");
    sourceOp = *join;

    // Join all target ops of variants together. This is quadratic in the
    // number of target ops, but realistically there won't be >10 of those.
    for (const auto &targetOp : variant.targetOps) {
      bool inserted = false;
      for (auto &existingTargetOp : targetOps) {
        auto join = existingTargetOp.join(targetOp);
        if (join.has_value()) {
          existingTargetOp = *join;
          inserted = true;
          break;
        }
      }
      if (!inserted) {
        targetOps.push_back(targetOp);
      }
    }
  }
}

llvm::SmallVector<cudaq::DecompositionPatternVariant>
cudaq::DecompositionPatternType::findCoveringVariants(
    const cudaq::detail::OperatorInfo &targetGate) const {
  llvm::SmallVector<DecompositionPatternVariant> result;
  for (const auto &variant : variants) {
    for (const auto &targetOp : variant.targetOps) {
      if (targetOp.covers(targetGate)) {
        result.push_back(variant);
        break;
      }
    }
  }
  return result;
}
