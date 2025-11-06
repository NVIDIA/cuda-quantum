/*******************************************************************************
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "DecompositionPatterns.h"
#include "common/Logger.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include <llvm/ADT/Hashing.h>
#include <llvm/ADT/SetVector.h>
#include <llvm/ADT/StringRef.h>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace mlir;

#define DEBUG_TYPE "decomposition-pattern-selection"

namespace {

//===----------------------------------------------------------------------===//
// ConversionTarget and OperatorInfo, parsed from target basis strings such as
// ["x", "x(1)", "z"]
//===----------------------------------------------------------------------===//

struct OperatorInfo {
  StringRef name;
  size_t numControls;

  OperatorInfo(StringRef infoStr) : name(), numControls(0) {
    auto nameEnd = infoStr.find_first_of('(');
    name = infoStr.take_front(nameEnd);
    if (nameEnd < infoStr.size())
      infoStr = infoStr.drop_front(nameEnd);

    if (infoStr.consume_front("(")) {
      infoStr = infoStr.ltrim();
      if (infoStr.consume_front("n"))
        numControls = std::numeric_limits<size_t>::max();
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
    constexpr size_t unbounded = std::numeric_limits<size_t>::max();

    // Parse the list of target operations and build a set of legal operations
    for (const std::string &targetInfo : targetBasis) {
      legalOperatorSet.emplace_back(targetInfo);
    }

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
            return info.numControls == optor.getControls().size();
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
  size_t operator()(const OperatorInfo &info) const {
    return llvm::hash_combine(info.name, info.numControls);
  }
};
} // namespace std

namespace {

// Computes a hash of the given unordered set using the hashes of the elements
// in the set.
template <typename T>
size_t computeSetHash(const std::unordered_set<T> &set) {
  std::vector<size_t> hashes;
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
  DecompositionGraph() {
    // Build the graph from pattern metadata
    for (const auto &patternType :
         cudaq::DecompositionPatternType::RegistryType::entries()) {
      auto patternName = patternType.getName();
      auto pattern = patternType.instantiate();
      auto sourceGate = pattern->getSourceOp();
      auto targetGates = pattern->getTargetOps();

      // Map pattern -> source gate
      patternToSource.insert({patternName, sourceGate});
      sourceToPatterns[sourceGate].emplace_back(std::move(pattern));

      // Map pattern -> all its target gates
      SmallVector<OperatorInfo> targets;
      for (const auto &tgt : targetGates) {
        targets.push_back(tgt);
      }
      patternToTargets[patternName] = std::move(targets);
    }

    // Compute topological order of all gates
    computeTopologicalOrder();
  }

  /// Return all patterns that have the given gate as their source.
  ///
  /// @param gate The source gate to find outgoing patterns for
  /// @return A range view of pattern names (StringRef) whose source is the
  /// given gate
  const llvm::SmallVector<std::unique_ptr<cudaq::DecompositionPatternType>> &
  outgoingPatterns(const OperatorInfo &gate) const {
    return sourceToPatterns.find(gate)->second;
  }

  /// Select subset of patterns relevant to decomposing to the given basis
  /// gates.
  ///
  /// The result of the pattern selection are cached for a given basis gates,
  /// disabled patterns, and enabled patterns, so that successive calls with
  /// the same arguments will be O(1).
  ///
  /// @param patterns The pattern set to add the selected patterns to
  /// @param basisGates The basis gates to decompose to
  /// @param disabledPatterns The patterns to disable
  /// @param enabledPatterns The patterns to enable
  ///
  /// Using a backward topological traversal of the decomposition graph, we
  /// recursively select a subset of the decomposition patterns such that:
  /// - for every gate that can be decomposed to the target basis, a unique
  ///   decomposition is chosen.
  /// - when more than one decomposition would exist, pick the one that requires
  ///   the fewest applications of patterns
  /// - `disabledPatterns` are never selected
  /// - `enabledPatterns` are preferred over other patterns when multiple
  ///    decompositions are possible
  void selectPatterns(RewritePatternSet &patterns,
                      const std::unordered_set<OperatorInfo> &basisGates,
                      const std::unordered_set<std::string> &disabledPatterns,
                      const std::unordered_set<std::string> &enabledPatterns) {
    auto hashVal = llvm::hash_combine(computeSetHash(basisGates),
                                      computeSetHash(disabledPatterns),
                                      computeSetHash(enabledPatterns));

    if (!patternSelectionCache.contains(hashVal)) {
      patternSelectionCache[hashVal] = computePatternSelection(
          basisGates, disabledPatterns, enabledPatterns);
    }

    for (const auto &patternName : patternSelectionCache[hashVal]) {
      auto patternType =
          cudaq::registry::get<cudaq::DecompositionPatternType>(patternName);
      assert(patternType != nullptr && "pattern not found");
      patterns.add(patternType->create(patterns.getContext()));
    }
  }

private:
  std::vector<std::string> computePatternSelection(
      const std::unordered_set<OperatorInfo> &basisGates,
      const std::unordered_set<std::string> &disabledPatterns,
      const std::unordered_set<std::string> &enabledPatterns) const {
    // Map: pattern name -> number of hops from a basis gate
    llvm::StringMap<size_t> dist;
    // The set of selected patterns to return
    std::vector<std::string> selectedPatterns;

    // Backward traversal
    for (const auto &currentGate :
         traverseInReverseTopologicalOrder(basisGates)) {
      // For gates in basisGates, there is no decomposition needed
      if (basisGates.contains(currentGate)) {
        dist[currentGate.name] = 0;
        continue;
      }

      // For each gate not in basisGates, pick one pattern that decomposes it

      // Pick among the valid patterns, i.e. those that lead to reachable gates.
      std::vector<StringRef> validPatterns;
      for (const auto &pattern : outgoingPatterns(currentGate)) {
        const auto &targetGates = pattern->getTargetOps();
        if (std::all_of(targetGates.begin(), targetGates.end(),
                        [&dist](const auto &targetGate) {
                          return dist.find(targetGate) != dist.end();
                        })) {
          validPatterns.push_back(pattern->getPatternName());
        }
      }

      // Remove all disabled patterns
      size_t pos = 0;
      for (const auto &patternName : validPatterns) {
        if (!disabledPatterns.contains(patternName.str())) {
          validPatterns[pos++] = patternName;
        }
      }
      validPatterns.resize(pos);

      if (validPatterns.empty()) {
        LLVM_DEBUG(llvm::dbgs() << "No decomposition pattern found for gate "
                                << currentGate.name << '\n');
        continue;
      }

      // If there are enabled patterns within the valid patterns, then restrict
      // the valid patterns to only those
      pos = 0;
      for (const auto &patternName : validPatterns) {
        if (enabledPatterns.contains(patternName.str())) {
          validPatterns[pos++] = patternName;
        }
      }
      if (pos > 0) {
        validPatterns.resize(pos);
      }

      // Finally, break ties by picking the valid pattern with the least number
      // of hops (and lexicographically if the number of hops is the same).
      auto it = std::min_element(validPatterns.begin(), validPatterns.end(),
                                 [&dist](const auto &a, const auto &b) {
                                   if (dist[a] == dist[b])
                                     return a < b;
                                   return dist[a] < dist[b];
                                 });

      selectedPatterns.push_back(it->str());
    }

    return selectedPatterns;
  }

  /// Computes the topological order of all gates in the hypergraph using Kahn's
  /// algorithm (BFS-based topological sort).
  ///
  /// This is run once during construction of the DecompositionGraph.
  void computeTopologicalOrder() {
    // Build in-degree map
    std::unordered_map<OperatorInfo, size_t> inDegree;
    for (const auto &[patternName, sourceGate] : patternToSource) {
      // make sure the source gate is in the map
      if (!inDegree.contains(sourceGate))
        inDegree[sourceGate] = 0;
      auto targetsIt = patternToTargets.find(patternName);
      assert(targetsIt != patternToTargets.end() && "pattern not found");
      for (const auto &targetGate : targetsIt->second)
        inDegree[targetGate]++;
    }

    // Kahn's algorithm: process gates with in-degree 0
    std::queue<OperatorInfo> queue;
    for (const auto &[gate, degree] : inDegree) {
      if (degree == 0) {
        queue.push(gate);
      }
    }

    while (!queue.empty()) {
      auto gate = queue.front();
      queue.pop();

      topologicalOrder.push_back(gate);

      // Reduce in-degree of neighbors
      for (const auto &pattern : outgoingPatterns(gate)) {
        for (const auto &targetGate : pattern->getTargetOps()) {
          inDegree[targetGate]--;
          if (inDegree[targetGate] == 0) {
            queue.push(targetGate);
          }
        }
      }
    }
  }

  /// Get a vector of the given gates and all gates in its past, in reverse
  /// topological order.
  ///
  /// We define the "past" of `basisGates` in the DPG recursively:
  ///  1. a gate in `basisGates` is in its past;
  ///  2. a pattern is in the past if all its targets are in the past;
  ///  3. a gate is in the past if it has an outgoing pattern that is in the
  ///  past.
  ///
  /// @param basisGates The set of gates to start traversal from
  /// @return A vector of gates in reverse topological order
  ///
  /// Complexity: runs in O(V + E) time, i.e. will travere all gates in the DPG.
  /// An implementation that only traverses the gates in the past of
  /// `basisGates` should be possible using a heap, but benchmarking would be
  /// required to determine whether it would be faster. It would definitely be
  /// more complex.)
  std::vector<OperatorInfo> traverseInReverseTopologicalOrder(
      const std::unordered_set<OperatorInfo> &basisGates) const {
    std::unordered_set<OperatorInfo> visitedGates;
    std::vector<OperatorInfo> result;

    // Lambda to check if a gate not in basisGate should be added to the result,
    // i.e. it has an outgoing pattern such that all its target gates have been
    // visited already.
    auto existVisitedTargetGates = [&](const OperatorInfo &gate) -> bool {
      for (const auto &pattern : outgoingPatterns(gate)) {
        auto targetGates = pattern->getTargetOps();
        bool allTargetsVisited =
            std::all_of(targetGates.begin(), targetGates.end(),
                        [&visitedGates](const auto &targetGate) {
                          return visitedGates.contains(targetGate);
                        });
        if (allTargetsVisited) {
          return true;
        }
      }

      return false;
    };

    // Traverse all gates in reverse topological order, selecting those that
    // are in the past of a basis gate
    for (auto it = topologicalOrder.rbegin(); it != topologicalOrder.rend();
         ++it) {
      const auto &gate = *it;

      if (basisGates.contains(gate) || existVisitedTargetGates(gate)) {
        visitedGates.insert(gate);
        result.push_back(gate);
      }
    }

    return result;
  }

  // Map: target gate -> patterns that produce it
  std::unordered_map<
      OperatorInfo,
      SmallVector<std::unique_ptr<cudaq::DecompositionPatternType>>>
      sourceToPatterns;

  // Map: pattern name -> source gate it decomposes
  llvm::StringMap<OperatorInfo> patternToSource;

  // Map: pattern name -> all target gates it creates
  llvm::StringMap<SmallVector<OperatorInfo>> patternToTargets;

  // Topological order: vector of gates in topological order
  std::vector<OperatorInfo> topologicalOrder;

  // Cache for `selectPatterns`: hash of basis gates, disabled patterns, enabled
  // patterns -> selected patterns
  std::unordered_map<size_t, std::vector<std::string>> patternSelectionCache;
};

} // namespace

std::unique_ptr<ConversionTarget>
cudaq::createBasisTarget(MLIRContext &context,
                         ArrayRef<std::string> targetBasis) {
  return std::make_unique<BasisTarget>(context, targetBasis);
}

void cudaq::selectDecompositionPatterns(RewritePatternSet &patterns,
                                        ArrayRef<std::string> targetBasis,
                                        ArrayRef<std::string> disabledPatterns,
                                        ArrayRef<std::string> enabledPatterns) {
  // Static local graph - constructed once and reused
  static DecompositionGraph graph;
  BasisTarget target(*patterns.getContext(), targetBasis);

  // Convert targetBasis, disabledPatterns and enabledPatterns to sets for O(1)
  // lookup
  std::unordered_set<OperatorInfo> basisGatesSet(
      target.legalOperatorSet.begin(), target.legalOperatorSet.end());
  std::unordered_set<std::string> disabledPatternsSet(disabledPatterns.begin(),
                                                      disabledPatterns.end());
  std::unordered_set<std::string> enabledPatternsSet(enabledPatterns.begin(),
                                                     enabledPatterns.end());

  return graph.selectPatterns(patterns, basisGatesSet, disabledPatternsSet,
                              enabledPatternsSet);
}
