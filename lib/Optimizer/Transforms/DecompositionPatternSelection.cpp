/*******************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <algorithm>
#include <llvm/ADT/ArrayRef.h>
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/StringMap.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;

namespace {

//===----------------------------------------------------------------------===//
// ConversionTarget and OperatorInfo, parsed from target basis strings such as
// ["x", "x(1)", "z"]
//===----------------------------------------------------------------------===//

struct OperatorInfo {
  StringRef name;
  std::size_t numControls;

  OperatorInfo(StringRef infoStr) : name(), numControls(0) {
    auto nameEnd = infoStr.find_first_of('(');
    name = infoStr.take_front(nameEnd);
    if (nameEnd < infoStr.size())
      infoStr = infoStr.drop_front(nameEnd);

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
    return name == other.name && numControls == other.numControls;
  }
};

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
    addDynamicallyLegalDialect<quake::QuakeDialect>([&](Operation *op) {
      if (auto optor = dyn_cast<quake::OperatorInterface>(op)) {
        auto name = optor->getName().stripDialect();
        for (auto info : legalOperatorSet) {
          if (info.name != name)
            continue;
          if (info.numControls == unbounded ||
              optor.getControls().size() == info.numControls)
            return true;
        }
        return false;
      }

      // Handle quake.exp_pauli.
      if (isa<quake::ExpPauliOp>(op)) {
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
    return llvm::hash_combine(info.name, info.numControls);
  }
};
} // namespace std

namespace {

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
         cudaq::DecompositionPatternType::RegistryType::entries()) {
      patterns.insert({patternType.getName(), patternType.instantiate()});
    }
    return DecompositionGraph(std::move(patterns));
  }

  /// Return all patterns that have the given gate as one of their targets.
  ///
  /// @param gate The gate to find incoming patterns for
  /// @return A vector of pattern names (StringRef) whose targets include the
  /// given gate
  llvm::ArrayRef<std::string> incomingPatterns(const OperatorInfo &gate) const {
    static const llvm::SmallVector<std::string> empty;
    auto it = targetToPatterns.find(gate);
    return it == targetToPatterns.end() ? empty : it->second;
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

    for (const auto &patternName : patternSelectionCache[hashVal]) {
      const auto &pattern = getPatternType(patternName);
      patterns.add(pattern->create(patterns.getContext()));
    }
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

    // Initialize the priority queue with the basis gates
    for (const auto &gate : basisGates) {
      gatesToVisit.push({gate, 0, std::nullopt});
    }

    /// Compute the maximum distance from a pattern's targets to the basis
    /// gates.
    auto getPatternDist = [&](const auto &pattern) {
      auto targetGates = pattern->getTargetOps();
      std::vector<std::size_t> targetDistances;
      for (const auto &targetGate : targetGates) {
        if (visitedGates.count(targetGate)) {
          targetDistances.push_back(visitedGates.at(targetGate));
        } else {
          targetDistances.push_back(std::numeric_limits<std::size_t>::max());
        }
      }
      return *std::max_element(targetDistances.begin(), targetDistances.end());
    };

    while (!gatesToVisit.empty()) {
      auto [gate, dist, outgoingPattern] = gatesToVisit.top();
      gatesToVisit.pop();

      auto [_, success] = visitedGates.insert({gate, dist});
      if (!success) {
        // Gate already visited
        continue;
      }

      if (outgoingPattern.has_value()) {
        selectedPatterns.push_back(*outgoingPattern);
      }

      for (const auto &patternName : incomingPatterns(gate)) {
        if (disabledPatterns.contains(patternName)) {
          // Ignore disabled patterns
          continue;
        }
        const auto &pattern = getPatternType(patternName);
        std::size_t dist = getPatternDist(pattern);
        if (dist < std::numeric_limits<std::size_t>::max()) {
          gatesToVisit.push({pattern->getSourceOp(), dist + 1, patternName});
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
  std::unordered_map<OperatorInfo, SmallVector<std::string>> targetToPatterns;

  //===--------------------------------------------------------------------===//
  // Other data (cache)
  //===--------------------------------------------------------------------===//

  /// Cache for `selectPatterns`: hash of basis gates, disabled patterns,
  /// enabled patterns -> selected patterns
  std::unordered_map<std::size_t, std::vector<std::string>>
      patternSelectionCache;
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
