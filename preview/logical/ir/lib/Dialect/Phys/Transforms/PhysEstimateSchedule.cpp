/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Phys/Transforms/Passes.h"

#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

using namespace mlir;
using namespace qlx;
using namespace qlx::phys;

namespace qlx::phys {
#define GEN_PASS_DEF_PHYSESTIMATESCHEDULE
#include "qlx/Dialect/Phys/Transforms/Passes.h.inc"
} // namespace qlx::phys

namespace {

struct Entry {
  StringRef id;
  StringRef kind;
  double start = 0.0;
  double duration = 0.0;
  SmallVector<StringRef, 4> resources;
  SmallVector<NativeScheduleResourceId, 4> nativeResourceIds;
  SmallVector<StringRef, 4> dependencies;
  StringRef parent;
  StringRef branch;
  StringRef callee;
  StringRef profile;
  StringRef templateEvent;
  StringRef attempt;
  StringRef attemptEvent;
  StringRef decisionEvent;
  StringRef exhaustion;
  std::optional<int64_t> maxAttempts;
  std::optional<int64_t> repeatCount;
  std::optional<int64_t> maxIterations;
  std::optional<double> successProbability;

  double finish() const { return start + duration; }
};

using Constraint = std::pair<size_t, StringRef>;
using QubitSetId = unsigned;
using ResourceId = unsigned;
using ResourceSetId = unsigned;
using SummaryPathId = unsigned;

struct SummaryConstraint {
  unsigned condition = 0;
  StringRef branch;
};

struct MetricContribution {
  StringRef kind;
  double weightedDuration = 0.0;
  ResourceSetId resourceSet = 0;
  SummaryPathId path = 0;
};

struct SummaryPeakAtom {
  double start = 0.0;
  double duration = 0.0;
  int64_t leaves = 0;
  ResourceSetId resourceSet = 0;
  SummaryPathId path = 0;
};

struct NestedCallSummary {
  size_t invocation = 0;
  size_t root = 0;
  double start = 0.0;
  int64_t metricFactor = 1;
  SummaryPathId prefix = 0;
  unsigned conditionBase = 0;
};

struct CallSummary {
  size_t root = 0;
  SmallVector<MetricContribution, 0> metrics;
  SmallVector<SummaryPeakAtom, 0> peaks;
  SmallVector<ResourceSetId, 0> peakResourceSets;
  SmallVector<size_t, 0> peakFinishOrder;
  // Segment tree over peaks in start order.  Each node stores the latest
  // finish in its range, allowing a time shard to recover exactly the small
  // set of intervals active at its left boundary without replaying all prior
  // summary edges.
  SmallVector<double, 0> peakMaxFinishTree;
  size_t peakTreeBase = 0;
  SmallVector<NestedCallSummary, 0> nested;
  unsigned conditionCount = 0;
};

struct SummaryOccurrence {
  size_t source = 0;
  const CallSummary *summary = nullptr;
  double topStart = 0.0;
  int64_t metricFactor = 1;
  size_t conditionBase = 0;
  SmallVector<double, 4> offsets;
  SmallVector<size_t, 4> aliases;
  SmallVector<Constraint, 4> prefix;
};

struct OccurrenceResourceMapping {
  ResourceSetId mapped = 0;
  QubitSetId qubits = 0;
};

struct PeakAtom {
  double start = 0.0;
  double duration = 0.0;
  int64_t leaves = 0;
  QubitSetId qubitSet = 0;
  SmallVector<Constraint, 4> constraints;
};

struct ReplaySlice {
  size_t retry = 0;
  size_t attempt = 0;
  int64_t occurrences = 1;
  SmallVector<std::pair<size_t, int64_t>, 16> weighted;
  double replayDuration = 0.0;
  double attemptStart = 0.0;
  double retryStart = 0.0;
  size_t timeAnchor = 0;
  std::optional<size_t> summaryOccurrence;
};

/// Identity of one dynamically instantiated folded envelope.  `event` names
/// the canonical repeat/while row, while `expansion` names the call-template
/// invocation path that instantiated it.  The same canonical loop reused by
/// two different call templates therefore remains two different folded
/// contexts, while sibling calls inside one enclosing loop share its identity.
struct DynamicFoldIdentity {
  size_t event = 0;
  SmallVector<size_t, 4> expansion;
  StringRef branch;
  int64_t count = 1;
  double iterationFinish = 0.0;
  double envelopeFinish = 0.0;
};

struct RetryExecutionContext {
  SmallVector<DynamicFoldIdentity, 4> folds;
  SmallVector<Constraint, 4> conditions;
};

struct RetryStatistics {
  double exhaustionProbability = 0.0;
  double completionProbability = 1.0;
  double expectedExtraAttempts = 0.0;
  double maximumExtraAttempts = 0.0;
};

struct StoppedRetryTimeLayer {
  SmallVector<size_t, 4> slices;
  double expectedPerVisit = 0.0;
  double maximumPerVisit = 0.0;
};

/// A flat, reusable visited set whose reset cost is independent of the
/// schedule size. Retry-slice discovery performs several bounded graph walks
/// per retry; clearing an N-bit set for each walk would recreate the O(NQ)
/// behavior that the compact schedule representation is meant to avoid.
class GenerationMarks {
public:
  explicit GenerationMarks(size_t size) : marks(size, 0) {}

  void reset() {
    if (++generation != 0)
      return;
    std::fill(marks.begin(), marks.end(), 0);
    generation = 1;
  }

  bool insert(size_t index) {
    assert(index < marks.size() && "retry traversal index must be in range");
    if (marks[index] == generation)
      return false;
    marks[index] = generation;
    return true;
  }

  bool contains(size_t index) const {
    assert(index < marks.size() && "retry traversal index must be in range");
    return marks[index] == generation;
  }

private:
  std::vector<uint64_t> marks;
  uint64_t generation = 1;
};

/// Derive truncated-geometric statistics without subtracting nearly equal
/// probabilities.  In particular, `1 - pow(1 - p, attempts)` loses every bit
/// of the completion probability when p is below half an ulp at one.  The
/// log1p/expm1 spelling stays stable at both probability boundaries, and the
/// direct extra-attempt formula avoids a second cancellation at p ~= 1.
static RetryStatistics retryStatistics(double probability, int64_t attempts) {
  RetryStatistics result;
  if (probability == 1.0)
    return result;

  const double logFailure = std::log1p(-probability);
  result.exhaustionProbability =
      std::exp(static_cast<double>(attempts) * logFailure);
  result.completionProbability =
      -std::expm1(static_cast<double>(attempts) * logFailure);
  if (attempts > 1) {
    const double failureProbability = 1.0 - probability;
    const double laterCompletion =
        -std::expm1(static_cast<double>(attempts - 1) * logFailure);
    result.expectedExtraAttempts =
        failureProbability * laterCompletion / probability;
    result.maximumExtraAttempts = static_cast<double>(attempts - 1);
  }
  result.exhaustionProbability =
      std::clamp(result.exhaustionProbability, 0.0, 1.0);
  result.completionProbability =
      std::clamp(result.completionProbability, 0.0, 1.0);
  result.expectedExtraAttempts = std::clamp(result.expectedExtraAttempts, 0.0,
                                            result.maximumExtraAttempts);
  return result;
}

/// General parallel retry layers are exact order statistics over a union of
/// per-retry support points.  There is no safe constant-memory closed form for
/// arbitrary starts, durations, bounds, and probabilities.  Keep the exact
/// algorithm structurally bounded and fail closed before allocating when the
/// authenticated support exceeds this limit.  Serial retry estimation remains
/// O(1) for every signed-64-bit attempt bound.
static constexpr int64_t kMaximumParallelRetrySupportPoints = 262144;

/// Exact physical-footprint accounting currently interns one identity per
/// selected binding member.  Fail closed before allocating when a valid
/// machine description exceeds the estimator's bounded representation.  This
/// keeps unsupported machine scale a typed evidence limitation rather than an
/// allocator failure; ordinary schedules and retry bounds remain unaffected.
static constexpr int64_t kMaximumExpandedPhysicalBindingMembers = 262144;
static constexpr bool isBoundedPhysicalExpansion(int64_t members) {
  return members >= 0 && members <= kMaximumExpandedPhysicalBindingMembers;
}
static_assert(isBoundedPhysicalExpansion(262144));
static_assert(!isBoundedPhysicalExpansion(262145));

static bool isEnvelope(StringRef kind) {
  return kind == "call" || kind == "call_template" || kind == "repeat" ||
         kind == "if" || kind == "while" || kind == "try_take" ||
         kind == "spacetime_call";
}

static void parseList(StringRef value, SmallVectorImpl<StringRef> &result) {
  value.split(result, ',', /*MaxSplit=*/-1, /*KeepEmpty=*/false);
}

static FailureOr<Entry> parseEntry(StringAttr raw, Operation *source) {
  SmallVector<StringRef, 24> parts;
  raw.getValue().split(parts, '|', /*MaxSplit=*/-1, /*KeepEmpty=*/true);
  if (parts.size() < 5) {
    source->emitError("schedule estimate found a malformed entry");
    return failure();
  }
  Entry entry;
  entry.id = parts[0];
  entry.kind = parts[1];
  if (entry.id.empty() || entry.kind.empty() ||
      parts[2].getAsDouble(entry.start) ||
      parts[3].getAsDouble(entry.duration) || !std::isfinite(entry.start) ||
      !std::isfinite(entry.duration) || entry.start < 0.0 ||
      entry.duration < 0.0) {
    source->emitError("schedule estimate found invalid identity or timing");
    return failure();
  }
  parseList(parts[4], entry.resources);
  llvm::StringMap<StringRef> details;
  for (StringRef detail : ArrayRef<StringRef>(parts).drop_front(5)) {
    auto [key, value] = detail.split('=');
    if (key.empty() || !detail.contains('=') ||
        !details.try_emplace(key, value).second) {
      source->emitError("schedule estimate found malformed row details");
      return failure();
    }
  }
  auto text = [&](StringRef key) -> StringRef {
    auto found = details.find(key);
    return found == details.end() ? StringRef{} : found->second;
  };
  auto integer = [&](StringRef key,
                     std::optional<int64_t> &result) -> LogicalResult {
    auto found = details.find(key);
    if (found == details.end() || found->second.empty())
      return success();
    int64_t value = 0;
    if (found->second.getAsInteger(10, value) || value < 0)
      return source->emitError("schedule estimate found invalid ") << key;
    result = value;
    return success();
  };
  auto probability = details.find("success_probability");
  if (probability != details.end() && !probability->second.empty()) {
    double value = 0.0;
    if (probability->second.getAsDouble(value) || !std::isfinite(value) ||
        value <= 0.0 || value > 1.0) {
      source->emitError("schedule estimate found invalid success_probability");
      return failure();
    }
    entry.successProbability = value;
  }
  auto deps = details.find("deps");
  if (deps != details.end())
    parseList(deps->second, entry.dependencies);
  entry.parent = text("parent");
  entry.branch = text("branch");
  entry.callee = text("callee");
  entry.profile = text("profile");
  entry.templateEvent = text("template_event");
  entry.attempt = text("attempt");
  entry.attemptEvent = text("attempt_event");
  entry.decisionEvent = text("decision_event");
  entry.exhaustion = text("exhaustion");
  if (failed(integer("max_attempts", entry.maxAttempts)) ||
      failed(integer("repeat_count", entry.repeatCount)) ||
      failed(integer("max_iterations", entry.maxIterations)))
    return failure();
  return entry;
}

static std::string symbolKey(SymbolRefAttr symbol) {
  if (!symbol)
    return {};
  std::string result = symbol.getRootReference().getValue().str();
  for (FlatSymbolRefAttr nested : symbol.getNestedReferences())
    result += "::" + nested.getValue().str();
  return result;
}

class PhysicalEvidence {
public:
  PhysicalEvidence(ModuleOp module, GraphOp graph, Operation *diagnostic)
      : module(module), graph(graph), diagnostic(diagnostic), symbols(module) {}

  LogicalResult initialize() {
    architecture = symbols.lookup<ArchitectureOp>(graph.getArchitecture());
    if (!architecture)
      return diagnostic->emitError(
          "schedule estimate graph architecture does not resolve");

    for (DeviceOp candidate : module.getOps<DeviceOp>()) {
      if (!candidate.getPhysicalAttr() ||
          candidate.getPhysicalAttr() != graph.getArchitectureAttr())
        continue;
      if (device)
        return diagnostic->emitError(
            "schedule estimate graph matches several qlx.device closures");
      device = candidate;
    }
    architecture.walk([&](ResourceClassOp resourceClass) {
      classCounts[resourceClass.getSymName()] = resourceClass.getCount();
      classKinds[resourceClass.getSymName()] = resourceClass.getKind().str();
      int64_t physicalWeight = resourceClass.getKind() == "qubit" ? 1 : 0;
      if (auto granularity =
              resourceClass->getAttrOfType<StringAttr>("granularity");
          granularity && granularity.getValue() == "patch") {
        auto unitKind =
            resourceClass->getAttrOfType<StringAttr>("physical_unit_kind");
        if (!unitKind || unitKind.getValue() != "qubit") {
          resourceClass.emitOpError(
              "schedule estimate physical_qubits requires patch footprints "
              "to use physical_unit_kind = 'qubit'");
          physicalEvidenceInvalid = true;
          return;
        }
        auto footprint =
            resourceClass->getAttrOfType<IntegerAttr>("physical_units");
        if (!footprint || footprint.getInt() <= 0) {
          resourceClass.emitOpError(
              "patch resource class lacks a positive physical footprint");
          physicalEvidenceInvalid = true;
          return;
        }
        physicalWeight = footprint.getInt();
      }
      classPhysicalWeights[resourceClass.getSymName()] = physicalWeight;
    });
    if (physicalEvidenceInvalid)
      return failure();
    architecture.walk([&](QECBindingOp binding) {
      std::string name = binding.getSymName().str();
      bindingRegions[symbolKey(binding.getQecRegionAttr())] = name;
      SmallVector<std::string, 2> classes;
      for (Attribute raw : binding.getResources()) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
        if (reference) {
          classes.push_back(reference.getValue().str());
          classBindings[reference.getValue()].push_back(name);
        }
      }
      bindingClasses[name] = std::move(classes);
    });
    for (FactoryModelOp model : module.getOps<FactoryModelOp>())
      factoryModelBindings[("factory:" + model.getSymName()).str()] =
          model.getQecBinding().str();
    for (ResourceOp resource : module.getOps<ResourceOp>()) {
      std::string key = (resource.getResourceClassAttr().getValue() + "[" +
                         std::to_string(resource.getIndex()) + "]")
                            .str();
      if (auto region = resource.getQecRegionAttr()) {
        auto binding = bindingRegions.find(symbolKey(region));
        if (binding != bindingRegions.end()) {
          auto existing = concreteBinding.find(key);
          if (existing != concreteBinding.end() &&
              existing->second != binding->second) {
            ambiguousConcrete.insert(key);
          } else {
            concreteBinding[key] = binding->second;
          }
        }
      }
    }
    return success();
  }

  DeviceOp getDevice() const { return device; }

  FailureOr<const std::set<std::string> *> resourcesFor(StringRef resource) {
    auto cached = resourceMembers.find(resource);
    if (cached != resourceMembers.end())
      return &cached->second;
    std::set<std::string> result;
    if (resource.starts_with("control:")) {
      resourceMembers[resource] = result;
      return &resourceMembers.find(resource)->second;
    }
    if (auto model = factoryModelBindings.find(resource);
        model != factoryModelBindings.end()) {
      auto values = resourcesForBinding(model->second);
      if (failed(values))
        return failure();
      result = *values;
      resourceMembers[resource] = result;
      return &resourceMembers.find(resource)->second;
    }
    if (resource.starts_with("binding:")) {
      StringRef binding = resource.drop_front(StringRef("binding:").size());
      auto values = resourcesForBinding(binding);
      if (failed(values))
        return failure();
      result = *values;
      resourceMembers[resource] = result;
      return &resourceMembers.find(resource)->second;
    }
    if (resource.starts_with("class:")) {
      StringRef className = resource.drop_front(StringRef("class:").size());
      auto count = classCounts.find(className);
      if (count == classCounts.end()) {
        diagnostic->emitError(
            "scheduled physical resource class has no retained capacity: ")
            << className;
        return failure();
      }
      if (failed(checkClassExpansion(className)))
        return failure();
      for (int64_t index = 0; index < count->second; ++index)
        result.insert((className + "[" + Twine(index) + "]").str());
      resourceMembers[resource] = result;
      return &resourceMembers.find(resource)->second;
    }
    auto concrete = parseConcreteResource(resource);
    if (failed(concrete))
      return failure();
    result.insert(resource.str());
    resourceMembers[resource] = result;
    return &resourceMembers.find(resource)->second;
  }

  FailureOr<const std::set<std::string> *> qubitsFor(StringRef resource) {
    auto cached = resourceQubits.find(resource);
    if (cached != resourceQubits.end())
      return &cached->second;
    std::set<std::string> result;
    if (resource.starts_with("control:")) {
      resourceQubits[resource] = result;
      return &resourceQubits.find(resource)->second;
    }
    if (auto model = factoryModelBindings.find(resource);
        model != factoryModelBindings.end()) {
      auto values = qubitsForBinding(model->second);
      if (failed(values))
        return failure();
      result = *values;
      usedBindings.insert(model->second);
      resourceQubits[resource] = result;
      return &resourceQubits.find(resource)->second;
    }
    if (resource.starts_with("binding:")) {
      StringRef binding = resource.drop_front(StringRef("binding:").size());
      auto values = qubitsForBinding(binding);
      if (failed(values))
        return failure();
      result = *values;
      usedBindings.insert(binding);
      resourceQubits[resource] = result;
      return &resourceQubits.find(resource)->second;
    }
    if (resource.starts_with("class:")) {
      StringRef className = resource.drop_front(StringRef("class:").size());
      auto count = classCounts.find(className);
      auto weight = classPhysicalWeights.find(className);
      if (count == classCounts.end() || weight == classPhysicalWeights.end()) {
        diagnostic->emitError(
            "scheduled physical resource class has incomplete footprint: ")
            << className;
        return failure();
      }
      if (failed(checkClassExpansion(className)))
        return failure();
      if (weight->second > 0)
        for (int64_t index = 0; index < count->second; ++index) {
          std::string member = (className + "[" + Twine(index) + "]").str();
          result.insert(member);
          directProvisioned.insert(member);
        }
      resourceQubits[resource] = result;
      return &resourceQubits.find(resource)->second;
    }
    auto concrete = parseConcreteResource(resource);
    if (failed(concrete))
      return failure();
    StringRef className = concrete->first;
    auto kind = classKinds.find(className);
    if (kind == classKinds.end()) {
      diagnostic->emitError(
          "scheduled physical resource class has no retained kind: ")
          << className;
      return failure();
    }
    auto weight = classPhysicalWeights.find(className);
    if (weight == classPhysicalWeights.end()) {
      diagnostic->emitError(
          "scheduled physical resource class has no retained footprint: ")
          << className;
      return failure();
    }
    if (weight->second > 0)
      result.insert(resource.str());
    if (ambiguousConcrete.contains(resource)) {
      // A concrete member of a deliberately shared patch class can be
      // materialized under several QEC-region views at different points in
      // the schedule.  Its physical identity is still unambiguous. Retain all
      // contributing bindings; provisionedQubits() expands and deduplicates
      // the common class below.
      auto candidates = classBindings.find(className);
      if (candidates == classBindings.end() || candidates->second.empty()) {
        diagnostic->emitError(
            "shared scheduled physical resource has no QEC binding: ")
            << resource;
        return failure();
      }
      for (const std::string &binding : candidates->second)
        usedBindings.insert(binding);
    } else if (auto exact = concreteBinding.find(resource);
               exact != concreteBinding.end()) {
      usedBindings.insert(exact->second);
    } else {
      auto candidates = classBindings.find(className);
      if (candidates != classBindings.end()) {
        if (candidates->second.size() != 1) {
          diagnostic->emitError(
              "scheduled physical resource class has ambiguous QEC bindings: ")
              << className;
          return failure();
        }
        usedBindings.insert(candidates->second.front());
      } else if (weight->second > 0) {
        directProvisioned.insert(resource.str());
      }
    }
    resourceQubits[resource] = result;
    return &resourceQubits.find(resource)->second;
  }

  FailureOr<std::set<std::string>> provisionedQubits() {
    std::set<std::string> result = directProvisioned;
    for (const auto &binding : usedBindings) {
      auto values = qubitsForBinding(binding.getKey());
      if (failed(values))
        return failure();
      result.insert(values->begin(), values->end());
    }
    return result;
  }

  FailureOr<int64_t> physicalWeightFor(StringRef resource) {
    auto concrete = parseConcreteResource(resource);
    if (failed(concrete))
      return failure();
    auto weight = classPhysicalWeights.find(concrete->first);
    if (weight == classPhysicalWeights.end() || weight->second <= 0) {
      diagnostic->emitError(
          "scheduled physical resource has no positive qubit footprint: ")
          << resource;
      return failure();
    }
    return weight->second;
  }

private:
  LogicalResult checkClassExpansion(StringRef className) {
    auto count = classCounts.find(className);
    if (count == classCounts.end())
      return diagnostic->emitError(
                 "scheduled physical resource class has no retained capacity: ")
             << className;
    if (!isBoundedPhysicalExpansion(count->second))
      return diagnostic->emitError(
                 "missing evidence: exact scheduled physical resource-class "
                 "footprint exceeds the bounded 262144-member estimator "
                 "limit: ")
             << className;
    return success();
  }

  LogicalResult checkBindingExpansion(StringRef binding, bool qubitsOnly) {
    auto classes = bindingClasses.find(binding);
    if (classes == bindingClasses.end()) {
      return diagnostic->emitError(
                 "scheduled factory-model binding has no retained "
                 "phys.qec_binding: ")
             << binding;
    }

    int64_t members = 0;
    for (const std::string &className : classes->second) {
      auto count = classCounts.find(className);
      if (count == classCounts.end()) {
        return diagnostic->emitError("scheduled physical resource class has no "
                                     "retained capacity: ")
               << className;
      }
      if (qubitsOnly) {
        auto weight = classPhysicalWeights.find(className);
        if (weight == classPhysicalWeights.end()) {
          return diagnostic->emitError(
                     "scheduled physical resource class has no retained "
                     "footprint: ")
                 << className;
        }
        if (weight->second <= 0)
          continue;
      }
      if (count->second > kMaximumExpandedPhysicalBindingMembers - members) {
        return diagnostic->emitError(
                   "missing evidence: exact scheduled factory-model binding "
                   "footprint exceeds the bounded 262144-member estimator "
                   "limit: ")
               << binding;
      }
      members += count->second;
    }
    return success();
  }

  FailureOr<std::pair<StringRef, int64_t>>
  parseConcreteResource(StringRef resource) {
    size_t open = resource.rfind('[');
    if (open == StringRef::npos || !resource.ends_with("]")) {
      diagnostic->emitError("scheduled physical resource key is not a concrete "
                            "class/index binding: ")
          << resource;
      return failure();
    }
    StringRef className = resource.take_front(open);
    int64_t index = 0;
    if (resource.slice(open + 1, resource.size() - 1).getAsInteger(10, index) ||
        index < 0) {
      diagnostic->emitError("scheduled physical resource index is invalid: ")
          << resource;
      return failure();
    }
    auto count = classCounts.find(className);
    if (count == classCounts.end() || index >= count->second) {
      diagnostic->emitError(
          "scheduled physical resource has no retained class capacity: ")
          << resource;
      return failure();
    }
    return std::make_pair(className, index);
  }

  FailureOr<std::set<std::string>> resourcesForBinding(StringRef binding) {
    auto cached = bindingResources.find(binding);
    if (cached != bindingResources.end())
      return cached->second;
    if (failed(checkBindingExpansion(binding, /*qubitsOnly=*/false)))
      return failure();
    auto classes = bindingClasses.find(binding);
    if (classes == bindingClasses.end()) {
      diagnostic->emitError(
          "scheduled factory-model binding has no retained phys.qec_binding: ")
          << binding;
      return failure();
    }
    std::set<std::string> result;
    for (const std::string &className : classes->second) {
      auto count = classCounts.find(className);
      if (count == classCounts.end()) {
        diagnostic->emitError(
            "scheduled physical resource class has no retained capacity: ")
            << className;
        return failure();
      }
      for (int64_t index = 0; index < count->second; ++index)
        result.insert(className + "[" + std::to_string(index) + "]");
    }
    bindingResources[binding] = result;
    return result;
  }

  FailureOr<std::set<std::string>> qubitsForBinding(StringRef binding) {
    auto cached = bindingQubits.find(binding);
    if (cached != bindingQubits.end())
      return cached->second;
    if (failed(checkBindingExpansion(binding, /*qubitsOnly=*/true)))
      return failure();
    auto classes = bindingClasses.find(binding);
    if (classes == bindingClasses.end()) {
      diagnostic->emitError(
          "scheduled factory-model binding has no retained phys.qec_binding: ")
          << binding;
      return failure();
    }
    std::set<std::string> result;
    for (const std::string &className : classes->second) {
      auto count = classCounts.find(className);
      if (count == classCounts.end()) {
        diagnostic->emitError(
            "scheduled physical resource class has no retained capacity: ")
            << className;
        return failure();
      }
      auto weight = classPhysicalWeights.find(className);
      if (weight == classPhysicalWeights.end()) {
        diagnostic->emitError(
            "scheduled physical resource class has no retained footprint: ")
            << className;
        return failure();
      }
      if (weight->second > 0)
        for (int64_t index = 0; index < count->second; ++index)
          result.insert(className + "[" + std::to_string(index) + "]");
    }
    bindingQubits[binding] = result;
    return result;
  }

  ModuleOp module;
  GraphOp graph;
  Operation *diagnostic;
  SymbolTable symbols;
  ArchitectureOp architecture;
  DeviceOp device;
  llvm::StringMap<int64_t> classCounts;
  llvm::StringMap<std::string> classKinds;
  llvm::StringMap<int64_t> classPhysicalWeights;
  llvm::StringMap<SmallVector<std::string, 2>> bindingClasses;
  llvm::StringMap<SmallVector<std::string, 2>> classBindings;
  llvm::StringMap<std::string> factoryModelBindings;
  llvm::StringMap<std::string> bindingRegions;
  llvm::StringMap<std::string> concreteBinding;
  llvm::StringSet<> ambiguousConcrete;
  llvm::StringMap<std::set<std::string>> resourceMembers;
  llvm::StringMap<std::set<std::string>> resourceQubits;
  llvm::StringMap<std::set<std::string>> bindingResources;
  llvm::StringMap<std::set<std::string>> bindingQubits;
  llvm::StringSet<> usedBindings;
  std::set<std::string> directProvisioned;
  bool physicalEvidenceInvalid = false;
};

struct MetricTotals {
  double resourceTime = 0.0;
  double qubitTime = 0.0;
  llvm::StringMap<double> kindTime;
};

struct MetricNode {
  MetricTotals base;
  std::map<size_t, std::map<StringRef, std::unique_ptr<MetricNode>>> groups;
};

static void addMetrics(MetricTotals &target, const MetricTotals &source,
                       double factor = 1.0) {
  target.resourceTime += factor * source.resourceTime;
  target.qubitTime += factor * source.qubitTime;
  for (const auto &entry : source.kindTime)
    target.kindTime[entry.getKey()] += factor * entry.getValue();
}

static MetricTotals scaledMetrics(const MetricTotals &source, double factor) {
  MetricTotals result;
  addMetrics(result, source, factor);
  return result;
}

static void maximizeMetrics(MetricTotals &target, const MetricTotals &source) {
  target.resourceTime = std::max(target.resourceTime, source.resourceTime);
  target.qubitTime = std::max(target.qubitTime, source.qubitTime);
  for (const auto &entry : source.kindTime) {
    double &retained = target.kindTime[entry.getKey()];
    retained = std::max(retained, entry.getValue());
  }
}

static void accumulateMetric(MetricNode &root, StringRef kind,
                             double weightedDuration, size_t resourceCount,
                             int64_t qubitCount,
                             ArrayRef<Constraint> constraints) {
  if (weightedDuration == 0.0)
    return;
  MetricNode *node = &root;
  for (const auto &[condition, branch] : constraints) {
    auto &child = node->groups[condition][branch];
    if (!child)
      child = std::make_unique<MetricNode>();
    node = child.get();
  }
  node->base.resourceTime +=
      weightedDuration * static_cast<double>(resourceCount);
  node->base.qubitTime += weightedDuration * static_cast<double>(qubitCount);
  node->base.kindTime[kind] += weightedDuration;
}

static MetricTotals reduceMetrics(const MetricNode &node) {
  MetricTotals result;
  result.resourceTime = node.base.resourceTime;
  result.qubitTime = node.base.qubitTime;
  for (const auto &entry : node.base.kindTime)
    result.kindTime[entry.getKey()] = entry.getValue();
  for (const auto &[condition, branches] : node.groups) {
    (void)condition;
    MetricTotals maximum;
    for (const auto &[branch, child] : branches) {
      (void)branch;
      MetricTotals value = reduceMetrics(*child);
      maximum.resourceTime = std::max(maximum.resourceTime, value.resourceTime);
      maximum.qubitTime = std::max(maximum.qubitTime, value.qubitTime);
      for (const auto &entry : value.kindTime) {
        double &retained = maximum.kindTime[entry.getKey()];
        retained = std::max(retained, entry.getValue());
      }
    }
    result.resourceTime += maximum.resourceTime;
    result.qubitTime += maximum.qubitTime;
    for (const auto &entry : maximum.kindTime)
      result.kindTime[entry.getKey()] += entry.getValue();
  }
  return result;
}

static bool isSubset(const llvm::BitVector &candidate,
                     const llvm::BitVector &container) {
  llvm::BitVector combined = container;
  combined |= candidate;
  return combined == container;
}

static constexpr size_t kMaximumConditionalOccupancyWork = 262144;

static bool consumeConditionalOccupancyWork(size_t &work, size_t amount) {
  if (amount > kMaximumConditionalOccupancyWork - work)
    return false;
  work += amount;
  return true;
}

static FailureOr<std::vector<llvm::BitVector>>
pruneUnions(std::vector<llvm::BitVector> candidates, size_t &work) {
  llvm::stable_sort(candidates, [](const auto &left, const auto &right) {
    return left.count() > right.count();
  });
  std::vector<llvm::BitVector> result;
  for (auto &candidate : candidates) {
    bool contained = false;
    for (const auto &retained : result) {
      if (!consumeConditionalOccupancyWork(work, 1))
        return failure();
      if (isSubset(candidate, retained)) {
        contained = true;
        break;
      }
    }
    if (!contained)
      result.push_back(std::move(candidate));
  }
  return result;
}

/// Incremental exact branch-aware occupancy.  The previous implementation
/// rebuilt the complete active-leaf set and conditional union tree at every
/// schedule edge, making peak extraction quadratic in the number of rows.
/// This tree retains only the current branch-local state and recomputes the
/// nodes on the changed ancestry path.
class ActiveOccupancy {
public:
  explicit ActiveOccupancy(ArrayRef<int64_t> qubitWeights)
      : qubitWeights(qubitWeights), qubitCount(qubitWeights.size()),
        baseQubits(qubitCount), qubitUniverse(qubitCount),
        maximumQubits(qubitCount) {}

  bool update(const PeakAtom &atom, ArrayRef<unsigned> qubits, int direction) {
    return update(atom.leaves, atom.constraints, qubits, direction);
  }

  bool update(int64_t leaves, ArrayRef<Constraint> constraints,
              ArrayRef<unsigned> qubits, int direction) {
    // The overwhelmingly common case is an unconditional schedule.  Keep its
    // occupancy directly on the root instead of rebuilding and copying the
    // full qubit-universe bit vector after every edge.  If a conditional edge
    // is encountered later, the ordinary tree path recomputes the root from
    // the same retained base state.
    if (constraints.empty() && groups.empty()) {
      baseLeaves += direction * leaves;
      if (baseLeaves < 0)
        return false;
      if (direction != 0)
        for (unsigned qubit : qubits) {
          int64_t &count = baseQubitCounts[qubit];
          count += direction;
          if (count < 0)
            return false;
          if (count == 0) {
            baseQubitCounts.erase(qubit);
            baseQubits.reset(qubit);
            --baseQubitTotal;
            baseQubitWeight -= qubitWeights[qubit];
          } else if (count == 1 && direction > 0) {
            baseQubits.set(qubit);
            ++baseQubitTotal;
            baseQubitWeight += qubitWeights[qubit];
          }
        }
      activeLeaves = baseLeaves;
      maximumValid = false;
      return true;
    }
    return update(constraints, qubits, direction * leaves, direction);
  }

private:
  bool update(ArrayRef<Constraint> constraints, ArrayRef<unsigned> qubits,
              int64_t leafDelta, int qubitDelta) {
    SmallVector<ActiveOccupancy *, 8> path{this};
    ActiveOccupancy *node = this;
    for (const auto &[condition, branch] : constraints) {
      auto &child = node->groups[condition][branch];
      if (!child)
        child = std::make_unique<ActiveOccupancy>(qubitWeights);
      node = child.get();
      path.push_back(node);
    }
    node->baseLeaves += leafDelta;
    if (node->baseLeaves < 0)
      return false;
    if (qubitDelta != 0)
      for (unsigned qubit : qubits) {
        int64_t &count = node->baseQubitCounts[qubit];
        count += qubitDelta;
        if (count < 0)
          return false;
        if (count == 0) {
          node->baseQubitCounts.erase(qubit);
          node->baseQubits.reset(qubit);
          --node->baseQubitTotal;
          node->baseQubitWeight -= qubitWeights[qubit];
        } else if (count == 1 && qubitDelta > 0) {
          node->baseQubits.set(qubit);
          ++node->baseQubitTotal;
          node->baseQubitWeight += qubitWeights[qubit];
        }
      }
    node->recompute();
    for (size_t depth = path.size(); depth > 1; --depth) {
      ActiveOccupancy *child = path[depth - 1];
      ActiveOccupancy *parent = path[depth - 2];
      const auto &[condition, branch] = constraints[depth - 2];
      if (child->empty()) {
        auto group = parent->groups.find(condition);
        if (group == parent->groups.end())
          return false;
        group->second.erase(branch);
        if (group->second.empty())
          parent->groups.erase(group);
      }
      parent->recompute();
    }
    return true;
  }

public:
  int64_t leaves() const { return activeLeaves; }

  FailureOr<int64_t> qubits() {
    if (groups.empty()) {
      maximumWork = 0;
      return baseQubitWeight;
    }
    if (failed(ensureMaximumQubits()))
      return failure();
    return weightedCount(maximumQubits);
  }

  FailureOr<llvm::BitVector> largestQubitSet() {
    if (groups.empty()) {
      maximumWork = 0;
      return baseQubits;
    }
    if (failed(ensureMaximumQubits()))
      return failure();
    return maximumQubits;
  }

  size_t lastMaximumWork() const { return maximumWork; }

private:
  int64_t weightedCount(const llvm::BitVector &qubits) const {
    int64_t result = 0;
    for (int64_t index = qubits.find_first(); index >= 0;
         index = qubits.find_next(index))
      result += qubitWeights[index];
    return result;
  }

  bool empty() const {
    return baseLeaves == 0 && baseQubitCounts.empty() && groups.empty();
  }

  void recompute() {
    activeLeaves = baseLeaves;
    qubitUniverse = baseQubits;
    maximumValid = false;
    for (const auto &[condition, branches] : groups) {
      (void)condition;
      int64_t branchLeaves = 0;
      for (const auto &[branch, child] : branches) {
        (void)branch;
        branchLeaves = std::max(branchLeaves, child->activeLeaves);
        qubitUniverse |= child->qubitUniverse;
      }
      activeLeaves += branchLeaves;
    }
  }

  using Branches = std::map<StringRef, std::unique_ptr<ActiveOccupancy>>;

  FailureOr<std::vector<llvm::BitVector>>
  buildFrontier(const llvm::BitVector &covered, size_t &work) const {
    llvm::BitVector fixed = covered;
    fixed |= baseQubits;
    std::vector<llvm::BitVector> candidates{fixed};
    for (const auto &[condition, branches] : groups) {
      (void)condition;
      std::vector<llvm::BitVector> choices;
      for (const auto &[branch, child] : branches) {
        (void)branch;
        auto childChoices = child->buildFrontier(fixed, work);
        if (failed(childChoices))
          return failure();
        if (childChoices->size() >
            kMaximumConditionalOccupancyWork - choices.size())
          return failure();
        choices.insert(choices.end(),
                       std::make_move_iterator(childChoices->begin()),
                       std::make_move_iterator(childChoices->end()));
      }
      auto prunedChoices = pruneUnions(std::move(choices), work);
      if (failed(prunedChoices))
        return failure();
      if (candidates.size() != 0 &&
          prunedChoices->size() >
              (kMaximumConditionalOccupancyWork - work) / candidates.size())
        return failure();
      const size_t combinations = candidates.size() * prunedChoices->size();
      if (!consumeConditionalOccupancyWork(work, combinations))
        return failure();
      std::vector<llvm::BitVector> composed;
      composed.reserve(combinations);
      for (const llvm::BitVector &candidate : candidates)
        for (const llvm::BitVector &choice : *prunedChoices) {
          llvm::BitVector combined = candidate;
          combined |= choice;
          composed.push_back(std::move(combined));
        }
      auto pruned = pruneUnions(std::move(composed), work);
      if (failed(pruned))
        return failure();
      candidates = std::move(*pruned);
    }
    return candidates;
  }

  FailureOr<llvm::BitVector> solveMaximum(const llvm::BitVector &covered,
                                          size_t &work) const {
    llvm::BitVector result = covered;
    result |= baseQubits;
    if (groups.empty())
      return result;

    struct GroupView {
      const Branches *branches;
      llvm::BitVector universe;
    };
    SmallVector<GroupView, 8> views;
    views.reserve(groups.size());
    for (const auto &[condition, branches] : groups) {
      (void)condition;
      llvm::BitVector universe(qubitCount);
      for (const auto &[branch, child] : branches) {
        (void)branch;
        universe |= child->qubitUniverse;
      }
      universe.reset(result);
      views.push_back({&branches, std::move(universe)});
    }

    SmallVector<unsigned, 8> component(views.size());
    std::iota(component.begin(), component.end(), 0);
    auto root = [&](unsigned value) {
      while (component[value] != value) {
        component[value] = component[component[value]];
        value = component[value];
      }
      return value;
    };
    for (unsigned left = 0; left < views.size(); ++left)
      for (unsigned right = left + 1; right < views.size(); ++right)
        if (views[left].universe.anyCommon(views[right].universe)) {
          unsigned leftRoot = root(left);
          unsigned rightRoot = root(right);
          if (leftRoot != rightRoot)
            component[rightRoot] = leftRoot;
        }

    std::map<unsigned, SmallVector<unsigned, 4>> members;
    for (unsigned index = 0; index < views.size(); ++index)
      members[root(index)].push_back(index);
    for (const auto &[componentId, groupIndices] : members) {
      (void)componentId;
      if (groupIndices.size() == 1) {
        llvm::BitVector best = result;
        const Branches &branches = *views[groupIndices.front()].branches;
        for (const auto &[branch, child] : branches) {
          (void)branch;
          auto candidate = child->solveMaximum(result, work);
          if (failed(candidate))
            return failure();
          if (weightedCount(*candidate) > weightedCount(best))
            best = std::move(*candidate);
        }
        result = std::move(best);
        continue;
      }

      std::vector<llvm::BitVector> candidates{result};
      for (unsigned groupIndex : groupIndices) {
        std::vector<llvm::BitVector> choices;
        for (const auto &[branch, child] : *views[groupIndex].branches) {
          (void)branch;
          auto childChoices = child->buildFrontier(result, work);
          if (failed(childChoices))
            return failure();
          if (childChoices->size() >
              kMaximumConditionalOccupancyWork - choices.size())
            return failure();
          choices.insert(choices.end(),
                         std::make_move_iterator(childChoices->begin()),
                         std::make_move_iterator(childChoices->end()));
        }
        auto prunedChoices = pruneUnions(std::move(choices), work);
        if (failed(prunedChoices))
          return failure();
        if (candidates.size() != 0 &&
            prunedChoices->size() >
                (kMaximumConditionalOccupancyWork - work) / candidates.size())
          return failure();
        const size_t combinations = candidates.size() * prunedChoices->size();
        if (!consumeConditionalOccupancyWork(work, combinations))
          return failure();
        std::vector<llvm::BitVector> composed;
        composed.reserve(combinations);
        for (const llvm::BitVector &candidate : candidates)
          for (const llvm::BitVector &choice : *prunedChoices) {
            llvm::BitVector combined = candidate;
            combined |= choice;
            composed.push_back(std::move(combined));
          }
        auto pruned = pruneUnions(std::move(composed), work);
        if (failed(pruned))
          return failure();
        candidates = std::move(*pruned);
      }
      auto best = std::max_element(
          candidates.begin(), candidates.end(),
          [&](const llvm::BitVector &left, const llvm::BitVector &right) {
            return weightedCount(left) < weightedCount(right);
          });
      if (best == candidates.end())
        return failure();
      result = std::move(*best);
    }
    return result;
  }

  LogicalResult ensureMaximumQubits() {
    if (maximumValid) {
      maximumWork = 0;
      return success();
    }
    maximumWork = 0;
    auto result = solveMaximum(llvm::BitVector(qubitCount), maximumWork);
    if (failed(result))
      return failure();
    maximumQubits = std::move(*result);
    maximumValid = true;
    return success();
  }

  ArrayRef<int64_t> qubitWeights;
  unsigned qubitCount;
  int64_t baseLeaves = 0;
  llvm::DenseMap<unsigned, int64_t> baseQubitCounts;
  llvm::BitVector baseQubits;
  int64_t baseQubitTotal = 0;
  int64_t baseQubitWeight = 0;
  std::map<size_t, Branches> groups;
  int64_t activeLeaves = 0;
  llvm::BitVector qubitUniverse;
  llvm::BitVector maximumQubits;
  size_t maximumWork = 0;
  bool maximumValid = false;
};

class NativeScheduleEstimator {
public:
  NativeScheduleEstimator(
      ModuleOp module, StringRef requestedSchedule, StringRef requestedResult,
      StringRef requestedLowerTier, bool requireResult = true,
      std::optional<NativeScheduleView> nativeSchedule = std::nullopt,
      bool fullWorkload = false)
      : module(module), context(module.getContext()), symbols(module),
        requestedSchedule(requestedSchedule.str()),
        requestedResult(requestedResult.str()),
        requestedLowerTier(requestedLowerTier.str()),
        requireResult(requireResult), nativeSchedule(nativeSchedule),
        fullWorkload(fullWorkload) {}

  LogicalResult run() {
    const bool profile =
        std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") != nullptr;
    auto phase = [&](StringRef name, auto &&action) -> LogicalResult {
      const auto started = std::chrono::steady_clock::now();
      LogicalResult result = action();
      if (profile) {
        const double seconds = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - started)
                                   .count();
        llvm::errs() << "phys-estimate-schedule " << name << ' ' << seconds
                     << "s\n";
      }
      return result;
    };
    if (failed(phase("resolve", [&] { return resolve(); })) ||
        failed(phase("parse", [&] { return parse(); })) ||
        failed(phase("supply", [&] { return checkSupply(); })) ||
        failed(
            phase("multiplicities", [&] { return deriveMultiplicities(); })) ||
        failed(phase("physical-evidence",
                     [&] { return buildPhysicalEvidence(); })) ||
        failed(phase("dynamic-leaves", [&] { return buildDynamicLeaves(); })) ||
        failed(phase("peak-atoms", [&] { return buildPeakAtoms(); })) ||
        failed(phase("retry-slices", [&] { return buildRetrySlices(); })) ||
        failed(phase("aggregate", [&] { return aggregate(); })) ||
        (!requestedResult.empty() &&
         failed(phase("emit", [&] { return emitResult(); }))))
      return failure();
    return success();
  }

  std::string json() const {
    llvm::json::Object counts;
    for (const auto &[kind, count] : eventCounts)
      counts[kind] = count;
    llvm::json::Array assumptions;
    for (StringRef text : {
             "durations come from the selected verified P3 timing profile",
             "unresolved external resource supply is rejected",
             "folded repeat and bounded-while multiplicities are exact",
             "bounded retries replay the authenticated causal slice",
             "conditional occupancy uses the worst-case executable branch",
             "scheduled factory-model bindings contribute their full qubit "
             "footprint",
             "shared call templates retain exact canonical leaf occupancy",
             "scheduled_macro durations are deterministic mean-output slots; "
             "maximum_makespan_ns is conditional on those slots, not a "
             "physical factory worst case",
             "utilization is active resource-time divided by scheduled "
             "resource capacity",
         })
      assumptions.push_back(text.str());
    assumptions.push_back(
        fullWorkload
            ? "expected metrics price the full scheduled workload despite "
              "runtime abort semantics"
            : "expected metrics follow runtime abort semantics");
    llvm::json::Object result;
    result["event_count"] = static_cast<int64_t>(entries.size());
    result["event_counts"] = std::move(counts);
    result["makespan_ns"] = makespan;
    result["expected_makespan_ns"] = expectedMakespan;
    result["maximum_makespan_ns"] = maximumMakespan;
    result["active_resource_time_ns"] = activeResourceTime;
    result["expected_active_resource_time_ns"] = expectedResourceTime;
    result["maximum_active_resource_time_ns"] = maximumResourceTime;
    result["active_physical_qubit_time_ns"] = activeQubitTime;
    result["expected_active_physical_qubit_time_ns"] = expectedQubitTime;
    result["maximum_active_physical_qubit_time_ns"] = maximumQubitTime;
    result["physical_resources"] = static_cast<int64_t>(allResources.size());
    result["physical_qubits"] = provisionedPhysicalQubits;
    result["peak_concurrency"] = peakConcurrency;
    result["peak_active_physical_qubits"] = peakActiveQubits;
    result["utilization"] = utilization;
    result["expected_utilization"] = expectedUtilization;
    result["maximum_utilization"] = maximumUtilization;
    result["exhaustion_probability"] = exhaustionProbability;
    result["bottleneck"] = bottleneck;
    result["assumptions"] = std::move(assumptions);
    result["physical_model_identity"] = physicalModelIdentity;
    result["lower_tier_identity"] = requestedLowerTier;
    result["termination_semantics"] =
        fullWorkload ? "full_workload" : "program";
    if (operatingPointIdentity)
      result["operating_point_identity"] = *operatingPointIdentity;
    if (DeviceOp device = physical->getDevice())
      result["device_identity"] = device.getSymName().str();
    return llvm::formatv("{0}", llvm::json::Value(std::move(result))).str();
  }

private:
  LogicalResult resolve() {
    if (!requestedSchedule.empty()) {
      schedule = symbols.lookup<ScheduleOp>(requestedSchedule);
      if (!schedule)
        return module.emitError("phys-estimate-schedule schedule @")
               << requestedSchedule << " must resolve to phys.schedule";
    } else {
      for (ScheduleOp candidate : module.getOps<ScheduleOp>()) {
        if (schedule)
          return module.emitError(
              "phys-estimate-schedule requires schedule= when several "
              "schedules exist");
        schedule = candidate;
      }
      if (!schedule)
        return module.emitError(
            "phys-estimate-schedule requires one verified phys.schedule");
    }
    if ((requireResult && requestedResult.empty()) ||
        (!requestedResult.empty() && symbols.lookup(requestedResult)))
      return module.emitError(
          "phys-estimate-schedule result symbol must be nonempty and unique");
    graph = dyn_cast_or_null<GraphOp>(SymbolTable::lookupNearestSymbolFrom(
        schedule, schedule.getGraphAttr()));
    if (!graph)
      return schedule.emitOpError("graph must resolve to phys.graph");
    physicalModelIdentity = graph.getArchitecture().str();
    if (auto operatingPoint = graph.getOperatingPoint())
      operatingPointIdentity = operatingPoint->str();
    else
      operatingPointIdentity.reset();
    if (requestedLowerTier.empty())
      return module.emitError(
          "missing evidence: phys-estimate-schedule requires an exact "
          "analytical lower-tier result");
    lower = symbols.lookup<EstimateResultOp>(requestedLowerTier);
    if (!lower || lower.getTier() != "analytical")
      return module.emitError("phys-estimate-schedule lower-tier @")
             << requestedLowerTier
             << " must resolve to an analytical qlx.estimate_result";
    physical = std::make_unique<PhysicalEvidence>(module, graph, schedule);
    if (failed(physical->initialize()))
      return failure();
    if (!physical->getDevice() || !lower.getDeviceAttr() ||
        lower.getDeviceAttr().getValue() !=
            physical->getDevice().getSymName() ||
        !graph.getSourceProtocolAttr() ||
        graph.getSourceProtocolAttr() != lower.getRootAttr())
      return lower.emitOpError(
          "does not match the scheduled graph/device closure");
    return success();
  }

  LogicalResult parse() {
    if (nativeSchedule) {
      entries.reserve(nativeSchedule->rows.size());
      syntheticResourceLabels.reserve(llvm::count_if(
          nativeSchedule->rows, [](const NativeScheduleRow &row) {
            return row.resourceIds.empty();
          }));
      for (const NativeScheduleRow &row : nativeSchedule->rows) {
        Entry entry;
        entry.id = row.id;
        entry.kind = row.kind;
        entry.start = row.start;
        entry.duration = row.duration;
        if (row.resourceIds.empty()) {
          syntheticResourceLabels.push_back(("control:" + entry.id).str());
          entry.resources.push_back(syntheticResourceLabels.back());
        } else {
          entry.nativeResourceIds = row.resourceIds;
          for (NativeScheduleResourceId resource : row.resourceIds) {
            if (resource >= nativeSchedule->resourceLabels.size())
              return schedule.emitOpError(
                  "native schedule row names an invalid resource identity");
          }
        }
        for (StringRef value : row.dependencies)
          entry.dependencies.push_back(value);
        entry.parent = row.parent;
        entry.branch = row.branch;
        entry.callee = row.callee;
        entry.profile = row.profile;
        entry.templateEvent = row.templateEvent;
        entry.attempt = row.attempt;
        entry.attemptEvent = row.attemptEvent;
        entry.decisionEvent = row.decisionEvent;
        entry.exhaustion = row.exhaustion;
        entry.maxAttempts = row.maxAttempts;
        entry.repeatCount = row.repeatCount;
        entry.maxIterations = row.maxIterations;
        entry.successProbability = row.successProbability;
        entries.push_back(std::move(entry));
      }
    } else {
      for (Attribute raw : schedule.getEntries()) {
        auto text = dyn_cast<StringAttr>(raw);
        if (!text)
          return schedule.emitOpError("entries must be strings");
        auto entry = parseEntry(text, schedule);
        if (failed(entry))
          return failure();
        entries.push_back(std::move(*entry));
      }
    }
    for (size_t index = 0; index < entries.size(); ++index) {
      if (!byId.try_emplace(entries[index].id, index).second)
        return schedule.emitOpError("duplicate event identity in estimate");
    }
    children.resize(entries.size());
    dependents.resize(entries.size());
    for (size_t index = 0; index < entries.size(); ++index) {
      const Entry &entry = entries[index];
      if (!entry.parent.empty()) {
        auto parent = byId.find(entry.parent);
        if (parent == byId.end())
          return schedule.emitOpError("entry names missing parent '")
                 << entry.parent << "'";
        children[parent->second].push_back(index);
      }
      for (StringRef dependency : entry.dependencies) {
        auto found = byId.find(dependency);
        if (found == byId.end())
          return schedule.emitOpError("entry names missing dependency '")
                 << dependency << "'";
        dependents[found->second].push_back(index);
      }
    }
    return success();
  }

  LogicalResult checkSupply() {
    WalkResult result = graph.walk([&](qlx::phys::ResourceRequestOp request) {
      if (request->hasAttr("external") || !request->hasAttr("provider") ||
          !request.getPhysicalBindingAttr()) {
        request.emitOpError(
            "missing evidence: schedule estimation cannot assign "
            "authoritative duration to an unresolved external resource "
            "request");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return result.wasInterrupted() ? failure() : success();
  }

  FailureOr<int64_t> multiplicity(size_t index) const {
    int64_t result = 1;
    size_t child = index;
    llvm::SmallDenseSet<size_t, 8> visited;
    while (!entries[child].parent.empty()) {
      auto found = byId.find(entries[child].parent);
      if (found == byId.end() || !visited.insert(found->second).second)
        return failure();
      const Entry &parent = entries[found->second];
      int64_t factor = 1;
      if (parent.kind == "repeat") {
        if (entries[child].branch != "body" || !parent.repeatCount)
          return failure();
        factor = *parent.repeatCount;
      } else if (parent.kind == "while") {
        if (!parent.maxIterations)
          return failure();
        if (entries[child].branch == "condition") {
          if (*parent.maxIterations == std::numeric_limits<int64_t>::max())
            return failure();
          factor = *parent.maxIterations + 1;
        } else if (entries[child].branch == "body") {
          factor = *parent.maxIterations;
        } else {
          return failure();
        }
      }
      if (factor != 0 && result > std::numeric_limits<int64_t>::max() / factor)
        return failure();
      result *= factor;
      child = found->second;
    }
    return result;
  }

  LogicalResult deriveMultiplicities() {
    multiplicities.resize(entries.size());
    for (size_t index = 0; index < entries.size(); ++index) {
      auto value = multiplicity(index);
      if (failed(value))
        return schedule.getOperation()->emitOpError(
            "folded schedule multiplicity is malformed or overflows i64");
      multiplicities[index] = *value;
    }
    return success();
  }

  LogicalResult buildPhysicalEvidence() {
    entryQubitSets.resize(entries.size());
    entryResourceSets.resize(entries.size());
    internSummaryPath({});
    internResourceSet({});
    resourceSetQubits.assign(internedResourceSets.size(),
                             std::numeric_limits<QubitSetId>::max());
    SmallVector<ResourceId, 0> nativeResourceMap;
    if (nativeSchedule)
      nativeResourceMap.assign(nativeSchedule->resourceLabels.size(),
                               std::numeric_limits<ResourceId>::max());
    // Native rows already carry a stable, ordered resource-set identity.  A
    // shared call-heavy graph repeats the same small boundary thousands of
    // times, so resolve and intern each distinct native vector once while
    // retaining the ordinary string-key path for portable schedule replay.
    llvm::DenseMap<size_t, SmallVector<std::pair<size_t, ResourceSetId>, 1>>
        nativeResourceSetBuckets;
    auto resolveResource = [&](StringRef key) -> FailureOr<ResourceId> {
      auto found = resourceIds.find(key);
      if (found != resourceIds.end())
        return found->second;
      auto physicalResources = physical->resourcesFor(key);
      if (failed(physicalResources))
        return failure();
      auto qubits = physical->qubitsFor(key);
      if (failed(qubits))
        return failure();
      ResourceId identity = resourceQubitsById.size();
      SmallVector<unsigned, 4> resourcePhysicals;
      for (const std::string &physicalResource : **physicalResources) {
        auto [position, inserted] = physicalResourceIds.try_emplace(
            physicalResource, physicalResourceIds.size());
        (void)inserted;
        resourcePhysicals.push_back(position->second);
        allResources.insert(physicalResource);
      }
      SmallVector<unsigned, 4> resourceQubits;
      for (const std::string &qubit : **qubits) {
        auto [position, inserted] =
            qubitIds.try_emplace(qubit, qubitIds.size());
        if (inserted) {
          auto weight = physical->physicalWeightFor(qubit);
          if (failed(weight))
            return failure();
          qubitWeights.push_back(*weight);
        }
        resourceQubits.push_back(position->second);
      }
      llvm::sort(resourceQubits);
      resourceQubits.erase(
          std::unique(resourceQubits.begin(), resourceQubits.end()),
          resourceQubits.end());
      resourcePhysicalsById.push_back(std::move(resourcePhysicals));
      resourceQubitsById.push_back(std::move(resourceQubits));
      resourceIds.try_emplace(key, identity);
      return identity;
    };
    for (size_t index = 0; index < entries.size(); ++index) {
      std::optional<ResourceSetId> cachedNativeSet;
      size_t nativeSetHash = 0;
      if (!entries[index].nativeResourceIds.empty()) {
        ArrayRef<NativeScheduleResourceId> nativeIds =
            entries[index].nativeResourceIds;
        nativeSetHash = static_cast<size_t>(
            llvm::hash_combine_range(nativeIds.begin(), nativeIds.end()));
        auto cached = nativeResourceSetBuckets.find(nativeSetHash);
        if (cached != nativeResourceSetBuckets.end())
          for (const auto &[representative, resourceSet] : cached->second)
            if (ArrayRef<NativeScheduleResourceId>(
                    entries[representative].nativeResourceIds) == nativeIds) {
              cachedNativeSet = resourceSet;
              break;
            }
      }
      if (cachedNativeSet) {
        entryResourceSets[index] = *cachedNativeSet;
        auto qubits = qubitSetForResourceSet(*cachedNativeSet);
        if (failed(qubits))
          return schedule.emitOpError(
              "scheduled resource set lacks physical qubit evidence");
        entryQubitSets[index] = *qubits;
        continue;
      }
      SmallVector<ResourceId, 4> resources;
      size_t resourceCount = entries[index].nativeResourceIds.empty()
                                 ? entries[index].resources.size()
                                 : entries[index].nativeResourceIds.size();
      for (size_t position = 0; position < resourceCount; ++position) {
        StringRef key = entries[index].nativeResourceIds.empty()
                            ? entries[index].resources[position]
                            : nativeSchedule->resourceLabels
                                  [entries[index].nativeResourceIds[position]];
        ResourceId resource = std::numeric_limits<ResourceId>::max();
        if (!entries[index].nativeResourceIds.empty()) {
          NativeScheduleResourceId native =
              entries[index].nativeResourceIds[position];
          if (native >= nativeResourceMap.size())
            return schedule.emitOpError(
                "native schedule resource identity exceeds its label table");
          resource = nativeResourceMap[native];
          if (resource == std::numeric_limits<ResourceId>::max()) {
            auto resolved = resolveResource(key);
            if (failed(resolved))
              return failure();
            resource = *resolved;
            nativeResourceMap[native] = resource;
          }
        } else {
          auto resolved = resolveResource(key);
          if (failed(resolved))
            return failure();
          resource = *resolved;
        }
        if (!llvm::is_contained(resources, resource))
          resources.push_back(resource);
      }
      const ResourceSetId resourceSet = internResourceSet(resources);
      entryResourceSets[index] = resourceSet;
      if (!entries[index].nativeResourceIds.empty())
        nativeResourceSetBuckets[nativeSetHash].push_back({index, resourceSet});
      auto qubits = qubitSetForResourceSet(resourceSet);
      if (failed(qubits))
        return schedule.emitOpError(
            "scheduled resource set lacks physical qubit evidence");
      entryQubitSets[index] = *qubits;
    }
    templateResourceAliases.assign(entries.size(), 0);
    templateResourceAliasMaps.emplace_back();
    llvm::DenseMap<Attribute, size_t> internedTemplateResourceAliases;
    llvm::StringMap<CallTemplateOp> graphTemplates;
    graph.walk([&](CallTemplateOp invocation) {
      graphTemplates[invocation.getEventId()] = invocation;
    });
    llvm::StringMap<std::string> concreteResourceKeys;
    for (ResourceOp resource : module.getOps<ResourceOp>())
      concreteResourceKeys[resource.getSymName()] =
          (resource.getResourceClass() + "[" +
           std::to_string(resource.getIndex()) + "]")
              .str();
    auto resourceKey =
        [&](FlatSymbolRefAttr reference) -> FailureOr<std::string> {
      auto resource = concreteResourceKeys.find(reference.getValue());
      if (resource == concreteResourceKeys.end()) {
        schedule.emitOpError("call-template state alias does not resolve to "
                             "phys.resource ")
            << reference;
        return failure();
      }
      return resource->second;
    };
    for (size_t index = 0; index < entries.size(); ++index) {
      if (entries[index].kind != "call_template")
        continue;
      auto operation = graphTemplates.find(entries[index].id);
      if (operation == graphTemplates.end())
        return schedule.emitOpError("call-template schedule entry '")
               << entries[index].id
               << "' does not resolve to its typed graph invocation";
      auto aliases =
          operation->second->getAttrOfType<ArrayAttr>("state_aliases");
      if (!aliases)
        continue;
      auto cached = internedTemplateResourceAliases.find(aliases);
      if (cached != internedTemplateResourceAliases.end()) {
        templateResourceAliases[index] = cached->second;
        continue;
      }
      llvm::DenseMap<ResourceId, ResourceId> resolvedAliases;
      for (Attribute raw : aliases.getValue()) {
        auto mapping = dyn_cast<DictionaryAttr>(raw);
        auto canonical = mapping ? mapping.getAs<FlatSymbolRefAttr>("template")
                                 : FlatSymbolRefAttr{};
        auto invocation = mapping ? mapping.getAs<FlatSymbolRefAttr>("alias")
                                  : FlatSymbolRefAttr{};
        if (!canonical || !invocation)
          return schedule.emitOpError(
              "call-template state alias is malformed during estimation");
        auto canonicalKey = resourceKey(canonical);
        auto invocationKey = resourceKey(invocation);
        if (failed(canonicalKey) || failed(invocationKey))
          return failure();
        auto canonicalId = resourceIds.find(*canonicalKey);
        auto invocationId = resourceIds.find(*invocationKey);
        if (canonicalId == resourceIds.end() ||
            invocationId == resourceIds.end())
          return schedule.emitOpError(
              "call-template state alias lacks scheduled resource evidence");
        resolvedAliases[canonicalId->second] = invocationId->second;
      }
      size_t identity = templateResourceAliasMaps.size();
      templateResourceAliasMaps.push_back(std::move(resolvedAliases));
      internedTemplateResourceAliases.try_emplace(aliases, identity);
      templateResourceAliases[index] = identity;
    }
    auto provisioned = physical->provisionedQubits();
    if (failed(provisioned))
      return failure();
    provisionedQubits = std::move(*provisioned);
    provisionedPhysicalQubits = 0;
    for (const std::string &qubit : provisionedQubits) {
      auto weight = physical->physicalWeightFor(qubit);
      if (failed(weight) || *weight > std::numeric_limits<int64_t>::max() -
                                          provisionedPhysicalQubits)
        return schedule.emitOpError(
            "provisioned physical-qubit footprint overflows i64");
      provisionedPhysicalQubits += *weight;
    }
    int64_t indexedPhysicalQubits = 0;
    for (int64_t weight : qubitWeights) {
      if (weight > std::numeric_limits<int64_t>::max() - indexedPhysicalQubits)
        return schedule.emitOpError(
            "active physical-qubit footprint overflows i64");
      indexedPhysicalQubits += weight;
    }
    return success();
  }

  QubitSetId internQubitSet(ArrayRef<unsigned> qubits) {
    size_t hash = static_cast<size_t>(
        llvm::hash_combine_range(qubits.begin(), qubits.end()));
    auto &candidates = qubitSetBuckets[hash];
    for (QubitSetId candidate : candidates)
      if (ArrayRef<unsigned>(internedQubitSets[candidate]) == qubits)
        return candidate;
    QubitSetId result = internedQubitSets.size();
    internedQubitSets.emplace_back(qubits.begin(), qubits.end());
    candidates.push_back(result);
    return result;
  }

  ResourceSetId internResourceSet(ArrayRef<ResourceId> resources) {
    size_t hash = static_cast<size_t>(
        llvm::hash_combine_range(resources.begin(), resources.end()));
    auto &candidates = resourceSetBuckets[hash];
    for (ResourceSetId candidate : candidates)
      if (ArrayRef<ResourceId>(internedResourceSets[candidate]) == resources)
        return candidate;
    ResourceSetId result = internedResourceSets.size();
    internedResourceSets.emplace_back(resources.begin(), resources.end());
    candidates.push_back(result);
    return result;
  }

  FailureOr<QubitSetId> qubitSetForResourceSet(ResourceSetId resourceSet) {
    if (resourceSet >= internedResourceSets.size())
      return failure();
    if (resourceSetQubits.size() <= resourceSet)
      resourceSetQubits.resize(resourceSet + 1,
                               std::numeric_limits<QubitSetId>::max());
    QubitSetId &cached = resourceSetQubits[resourceSet];
    if (cached != std::numeric_limits<QubitSetId>::max())
      return cached;
    SmallVector<unsigned, 8> qubits;
    llvm::SmallDenseSet<unsigned, 8> seen;
    for (ResourceId resource : internedResourceSets[resourceSet]) {
      if (resource >= resourceQubitsById.size())
        return failure();
      for (unsigned qubit : resourceQubitsById[resource])
        if (seen.insert(qubit).second)
          qubits.push_back(qubit);
    }
    llvm::sort(qubits);
    cached = internQubitSet(qubits);
    return cached;
  }

  FailureOr<SmallVector<ResourceId, 4>>
  mappedResources(ResourceSetId source, ArrayRef<size_t> aliases) const {
    if (source >= internedResourceSets.size())
      return failure();
    SmallVector<ResourceId, 4> resources = internedResourceSets[source];
    for (size_t invocation : aliases) {
      if (invocation >= templateResourceAliases.size())
        return failure();
      size_t aliasIdentity = templateResourceAliases[invocation];
      if (aliasIdentity >= templateResourceAliasMaps.size())
        return failure();
      const auto &resourceAliases = templateResourceAliasMaps[aliasIdentity];
      SmallVector<ResourceId, 4> mapped;
      for (ResourceId resource : resources) {
        auto alias = resourceAliases.find(resource);
        ResourceId value =
            alias == resourceAliases.end() ? resource : alias->second;
        if (!llvm::is_contained(mapped, value))
          mapped.push_back(value);
      }
      resources = std::move(mapped);
    }
    return resources;
  }

  FailureOr<std::pair<size_t, int64_t>>
  mappedResourceCounts(ResourceSetId source, ArrayRef<size_t> aliases) const {
    auto resources = mappedResources(source, aliases);
    if (failed(resources))
      return failure();
    llvm::SmallDenseSet<unsigned, 8> physicalResources;
    llvm::SmallDenseSet<unsigned, 8> qubits;
    for (ResourceId resource : *resources) {
      if (resource >= resourceQubitsById.size() ||
          resource >= resourcePhysicalsById.size())
        return failure();
      for (unsigned physicalResource : resourcePhysicalsById[resource])
        physicalResources.insert(physicalResource);
      for (unsigned qubit : resourceQubitsById[resource])
        qubits.insert(qubit);
    }
    int64_t physicalQubits = 0;
    for (unsigned qubit : qubits)
      physicalQubits += qubitWeights[qubit];
    return std::make_pair(static_cast<size_t>(physicalResources.size()),
                          physicalQubits);
  }

  LogicalResult buildOccurrenceResourceEvidence() {
    struct PendingMapping {
      ResourceSetId source;
      SmallVector<ResourceId, 4> resources;
    };
    std::vector<SmallVector<PendingMapping, 4>> pending(
        summaryOccurrences.size());
    size_t shards = 1;
    if (context->isMultithreadingEnabled())
      shards = std::min(
          {size_t{64},
           static_cast<size_t>(context->getThreadPool().getMaxConcurrency()),
           std::max<size_t>(1, summaryOccurrences.size())});
    auto resolveShard = [&](size_t shard) -> LogicalResult {
      const size_t begin = summaryOccurrences.size() * shard / shards;
      const size_t end = summaryOccurrences.size() * (shard + 1) / shards;
      for (size_t index = begin; index < end; ++index) {
        const SummaryOccurrence &occurrence = summaryOccurrences[index];
        for (ResourceSetId source : occurrence.summary->peakResourceSets) {
          auto resources = mappedResources(source, occurrence.aliases);
          if (failed(resources))
            return failure();
          pending[index].push_back({source, std::move(*resources)});
        }
      }
      return success();
    };
    if (failed(failableParallelForEachN(context, 0, shards, resolveShard)))
      return failure();

    occurrenceResourceEvidence.resize(summaryOccurrences.size());
    for (size_t index = 0; index < pending.size(); ++index)
      for (PendingMapping &mapping : pending[index]) {
        ResourceSetId mapped = internResourceSet(mapping.resources);
        auto qubits = qubitSetForResourceSet(mapped);
        if (failed(qubits))
          return failure();
        occurrenceResourceEvidence[index].try_emplace(
            mapping.source, OccurrenceResourceMapping{mapped, *qubits});
      }
    return success();
  }

  const OccurrenceResourceMapping *
  occurrenceResourceMapping(size_t occurrence, ResourceSetId source) const {
    if (occurrence >= occurrenceResourceEvidence.size())
      return nullptr;
    auto found = occurrenceResourceEvidence[occurrence].find(source);
    return found == occurrenceResourceEvidence[occurrence].end()
               ? nullptr
               : &found->second;
  }

  SummaryPathId internSummaryPath(ArrayRef<SummaryConstraint> path) {
    llvm::hash_code hash = llvm::hash_code{};
    for (const SummaryConstraint &constraint : path)
      hash = llvm::hash_combine(hash, constraint.condition,
                                StringRef(constraint.branch));
    auto &candidates = summaryPathBuckets[static_cast<size_t>(hash)];
    for (SummaryPathId candidate : candidates) {
      ArrayRef<SummaryConstraint> retained = internedSummaryPaths[candidate];
      if (retained.size() == path.size() &&
          std::equal(retained.begin(), retained.end(), path.begin(),
                     [](const SummaryConstraint &left,
                        const SummaryConstraint &right) {
                       return left.condition == right.condition &&
                              left.branch == right.branch;
                     }))
        return candidate;
    }
    SummaryPathId result = internedSummaryPaths.size();
    internedSummaryPaths.emplace_back(path.begin(), path.end());
    candidates.push_back(result);
    return result;
  }

  FailureOr<SmallVector<Constraint, 4>>
  constraints(size_t index, std::optional<size_t> stop = std::nullopt,
              std::optional<size_t> invocation = std::nullopt) const {
    SmallVector<Constraint, 4> result;
    llvm::SmallDenseSet<size_t, 8> visited;
    size_t child = index;
    while (!entries[child].parent.empty()) {
      auto found = byId.find(entries[child].parent);
      if (found == byId.end() || !visited.insert(found->second).second)
        return failure();
      size_t parentIndex = found->second;
      if (stop && parentIndex == *stop)
        break;
      const Entry &parent = entries[parentIndex];
      if (parent.kind == "if" || parent.kind == "try_take") {
        if (entries[child].branch.empty())
          return failure();
        size_t identity = parentIndex;
        if (invocation) {
          const size_t size = entries.size();
          const size_t maximum = std::numeric_limits<size_t>::max();
          if (size == 0 || *invocation > (maximum - size - parentIndex) / size)
            return failure();
          identity = size + *invocation * size + parentIndex;
        }
        result.emplace_back(identity, entries[child].branch);
      }
      child = parentIndex;
    }
    std::reverse(result.begin(), result.end());
    return result;
  }

  FailureOr<SummaryPathId>
  summaryConstraints(size_t index, size_t root,
                     std::map<size_t, unsigned> &conditions,
                     CallSummary &summary) {
    auto path = constraints(index, root);
    if (failed(path))
      return failure();
    SmallVector<SummaryConstraint, 4> result;
    for (const auto &[identity, branch] : *path) {
      auto [found, inserted] =
          conditions.try_emplace(identity, summary.conditionCount);
      if (inserted)
        ++summary.conditionCount;
      result.push_back({found->second, branch});
    }
    return internSummaryPath(result);
  }

  FailureOr<const CallSummary *> summaryFor(size_t root) {
    auto cached = callSummaries.find(root);
    if (cached != callSummaries.end())
      return &cached->second;
    if (!buildingSummaries.insert(root).second) {
      schedule.emitOpError("call-template canonical call graph is recursive");
      return failure();
    }
    if (root >= entries.size() || entries[root].kind != "call" ||
        multiplicities[root] <= 0) {
      schedule.emitOpError(
          "call-template canonical call has invalid folded multiplicity");
      buildingSummaries.erase(root);
      return failure();
    }

    CallSummary summary;
    summary.root = root;
    std::map<size_t, unsigned> directConditions;
    std::function<LogicalResult(size_t)> append =
        [&](size_t index) -> LogicalResult {
      const Entry &entry = entries[index];
      if (entry.kind == "call" || entry.kind == "call_template") {
        size_t nestedRoot = index;
        if (entry.kind == "call_template") {
          auto canonical = byId.find(entry.templateEvent);
          if (entry.templateEvent.empty() || canonical == byId.end() ||
              entries[canonical->second].kind != "call")
            return schedule.emitOpError(
                "call_template lacks an earlier canonical call event");
          nestedRoot = canonical->second;
        }
        auto nested = summaryFor(nestedRoot);
        if (failed(nested))
          return failure();
        if (multiplicities[index] % multiplicities[root])
          return schedule.emitOpError(
              "call_template nested folded multiplicity is incommensurate");
        int64_t factor = multiplicities[index] / multiplicities[root];
        auto prefix =
            summaryConstraints(index, root, directConditions, summary);
        if (failed(prefix))
          return schedule.emitOpError(
              "call_template nested conditional ancestry is malformed");
        if ((*nested)->conditionCount >
            std::numeric_limits<unsigned>::max() - summary.conditionCount)
          return schedule.emitOpError(
              "call_template nested condition space overflows unsigned");
        summary.nested.push_back({index, nestedRoot,
                                  entry.start - entries[root].start, factor,
                                  *prefix, summary.conditionCount});
        summary.conditionCount += (*nested)->conditionCount;
        return success();
      }
      if (isEnvelope(entry.kind)) {
        for (size_t child : children[index])
          if (failed(append(child)))
            return failure();
        return success();
      }
      if (entry.duration <= 0.0 || multiplicities[index] == 0)
        return success();
      if (multiplicities[index] % multiplicities[root])
        return schedule.emitOpError(
            "call_template canonical leaf multiplicity is incommensurate");
      auto path = summaryConstraints(index, root, directConditions, summary);
      if (failed(path))
        return schedule.emitOpError(
            "call_template canonical conditional ancestry is malformed");
      int64_t factor = multiplicities[index] / multiplicities[root];
      summary.metrics.push_back({entry.kind,
                                 entry.duration * static_cast<double>(factor),
                                 entryResourceSets[index], *path});
      summary.peaks.push_back({entry.start - entries[root].start,
                               entry.duration, 1, entryResourceSets[index],
                               *path});
      return success();
    };
    for (size_t child : children[root])
      if (failed(append(child))) {
        buildingSummaries.erase(root);
        return failure();
      }

    llvm::sort(summary.metrics, [&](const MetricContribution &left,
                                    const MetricContribution &right) {
      if (left.path != right.path)
        return left.path < right.path;
      if (left.kind != right.kind)
        return left.kind < right.kind;
      return left.resourceSet < right.resourceSet;
    });
    SmallVector<MetricContribution, 0> metrics;
    for (MetricContribution &metric : summary.metrics) {
      if (!metrics.empty() && metrics.back().kind == metric.kind &&
          metrics.back().resourceSet == metric.resourceSet &&
          metrics.back().path == metric.path) {
        metrics.back().weightedDuration += metric.weightedDuration;
      } else {
        metrics.push_back(std::move(metric));
      }
    }
    summary.metrics = std::move(metrics);

    llvm::sort(summary.peaks,
               [&](const SummaryPeakAtom &left, const SummaryPeakAtom &right) {
                 if (left.path != right.path)
                   return left.path < right.path;
                 if (left.resourceSet != right.resourceSet)
                   return left.resourceSet < right.resourceSet;
                 return left.start < right.start;
               });
    SmallVector<SummaryPeakAtom, 0> peaks;
    size_t cursor = 0;
    while (cursor < summary.peaks.size()) {
      size_t end = cursor + 1;
      while (end < summary.peaks.size() &&
             summary.peaks[end].resourceSet ==
                 summary.peaks[cursor].resourceSet &&
             summary.peaks[end].path == summary.peaks[cursor].path)
        ++end;
      struct Edge {
        double time;
        int64_t delta;
      };
      SmallVector<Edge, 16> edges;
      for (size_t i = cursor; i < end; ++i) {
        const SummaryPeakAtom &atom = summary.peaks[i];
        double finish = atom.start + atom.duration;
        if (finish <= atom.start) {
          peaks.push_back(atom);
          continue;
        }
        edges.push_back({atom.start, atom.leaves});
        edges.push_back({finish, -atom.leaves});
      }
      llvm::sort(edges, [](const Edge &left, const Edge &right) {
        if (left.time != right.time)
          return left.time < right.time;
        return left.delta < right.delta;
      });
      int64_t active = 0;
      double segmentStart = 0.0;
      size_t edge = 0;
      while (edge < edges.size()) {
        double time = edges[edge].time;
        int64_t delta = 0;
        while (edge < edges.size() && edges[edge].time == time)
          delta += edges[edge++].delta;
        int64_t next = active + delta;
        if (next < 0) {
          buildingSummaries.erase(root);
          schedule.emitOpError(
              "call-template peak summary occupancy became negative");
          return failure();
        }
        if (next != active) {
          if (active > 0 && time > segmentStart)
            peaks.push_back({segmentStart, time - segmentStart, active,
                             summary.peaks[cursor].resourceSet,
                             summary.peaks[cursor].path});
          if (next > 0)
            segmentStart = time;
          active = next;
        }
      }
      if (active != 0) {
        buildingSummaries.erase(root);
        schedule.emitOpError(
            "call-template peak summary remains active after sweep");
        return failure();
      }
      cursor = end;
    }
    summary.peaks = std::move(peaks);
    llvm::SmallDenseSet<ResourceSetId, 16> seenPeakResourceSets;
    for (const SummaryPeakAtom &atom : summary.peaks)
      if (seenPeakResourceSets.insert(atom.resourceSet).second)
        summary.peakResourceSets.push_back(atom.resourceSet);
    llvm::sort(summary.peakResourceSets);
    llvm::sort(summary.peaks,
               [](const SummaryPeakAtom &left, const SummaryPeakAtom &right) {
                 if (left.start != right.start)
                   return left.start < right.start;
                 if (left.duration != right.duration)
                   return left.duration < right.duration;
                 if (left.path != right.path)
                   return left.path < right.path;
                 return left.resourceSet < right.resourceSet;
               });
    summary.peakFinishOrder.resize(summary.peaks.size());
    std::iota(summary.peakFinishOrder.begin(), summary.peakFinishOrder.end(),
              0);
    llvm::sort(summary.peakFinishOrder, [&](size_t left, size_t right) {
      const double leftFinish =
          summary.peaks[left].start + summary.peaks[left].duration;
      const double rightFinish =
          summary.peaks[right].start + summary.peaks[right].duration;
      if (leftFinish != rightFinish)
        return leftFinish < rightFinish;
      const bool leftCollapsed = summary.peaks[left].duration <= 0.0;
      const bool rightCollapsed = summary.peaks[right].duration <= 0.0;
      if (leftCollapsed != rightCollapsed)
        return leftCollapsed < rightCollapsed;
      return left < right;
    });
    if (!summary.peaks.empty()) {
      summary.peakTreeBase = 1;
      while (summary.peakTreeBase < summary.peaks.size())
        summary.peakTreeBase *= 2;
      summary.peakMaxFinishTree.assign(
          2 * summary.peakTreeBase, -std::numeric_limits<double>::infinity());
      for (size_t index = 0; index < summary.peaks.size(); ++index)
        summary.peakMaxFinishTree[summary.peakTreeBase + index] =
            summary.peaks[index].start + summary.peaks[index].duration;
      for (size_t index = summary.peakTreeBase; index-- > 1;)
        summary.peakMaxFinishTree[index] =
            std::max(summary.peakMaxFinishTree[2 * index],
                     summary.peakMaxFinishTree[2 * index + 1]);
    }
    buildingSummaries.erase(root);
    auto [stored, inserted] =
        callSummaries.try_emplace(root, std::move(summary));
    (void)inserted;
    return &stored->second;
  }

  SmallVector<Constraint, 8>
  occurrenceConstraints(const SummaryOccurrence &occurrence,
                        SummaryPathId path) const {
    SmallVector<Constraint, 8> result(occurrence.prefix.begin(),
                                      occurrence.prefix.end());
    for (const SummaryConstraint &constraint : internedSummaryPaths[path])
      result.emplace_back(occurrence.conditionBase + constraint.condition,
                          constraint.branch);
    return result;
  }

  LogicalResult appendSummaryOccurrences(
      size_t source, const CallSummary &summary, double topStart,
      ArrayRef<double> offsets, int64_t metricFactor, size_t conditionBase,
      ArrayRef<size_t> aliases, ArrayRef<Constraint> prefix) {
    SummaryOccurrence occurrence;
    occurrence.source = source;
    occurrence.summary = &summary;
    occurrence.topStart = topStart;
    occurrence.metricFactor = metricFactor;
    occurrence.conditionBase = conditionBase;
    occurrence.offsets.assign(offsets.begin(), offsets.end());
    occurrence.aliases.assign(aliases.begin(), aliases.end());
    occurrence.prefix.assign(prefix.begin(), prefix.end());
    const size_t occurrenceIndex = summaryOccurrences.size();
    summaryOccurrences.push_back(std::move(occurrence));
    summaryOccurrenceChildren.emplace_back();

    for (const NestedCallSummary &nested : summary.nested) {
      auto found = callSummaries.find(nested.root);
      if (found == callSummaries.end())
        return schedule.emitOpError(
            "call_template nested summary was not retained");
      if (nested.metricFactor != 0 &&
          metricFactor >
              std::numeric_limits<int64_t>::max() / nested.metricFactor)
        return schedule.emitOpError(
            "call_template nested metric multiplicity overflows i64");
      SmallVector<size_t, 4> nestedAliases;
      nestedAliases.reserve(aliases.size() + 1);
      nestedAliases.push_back(nested.invocation);
      nestedAliases.append(aliases.begin(), aliases.end());
      SmallVector<Constraint, 8> nestedPrefix(prefix.begin(), prefix.end());
      for (const SummaryConstraint &constraint :
           internedSummaryPaths[nested.prefix])
        nestedPrefix.emplace_back(conditionBase + constraint.condition,
                                  constraint.branch);
      if (nested.conditionBase >
          std::numeric_limits<size_t>::max() - conditionBase)
        return schedule.emitOpError(
            "call_template nested conditional identity overflows size_t");
      SmallVector<double, 4> nestedOffsets(offsets.begin(), offsets.end());
      nestedOffsets.push_back(nested.start);
      const size_t nestedOccurrenceIndex = summaryOccurrences.size();
      summaryOccurrenceChildren[occurrenceIndex].emplace_back(
          nested.invocation, nestedOccurrenceIndex);
      if (failed(appendSummaryOccurrences(source, found->second, topStart,
                                          nestedOffsets,
                                          metricFactor * nested.metricFactor,
                                          conditionBase + nested.conditionBase,
                                          nestedAliases, nestedPrefix)))
        return failure();
    }
    return success();
  }

  LogicalResult buildDynamicLeaves() {
    const bool profile =
        std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") != nullptr;
    auto checkpoint = [profile, last = std::chrono::steady_clock::now()](
                          StringRef name) mutable {
      const auto now = std::chrono::steady_clock::now();
      if (profile)
        llvm::errs() << "phys-estimate-schedule leaves-" << name << ' '
                     << std::chrono::duration<double>(now - last).count()
                     << "s\n";
      last = now;
    };
    for (size_t index = 0; index < entries.size(); ++index) {
      const Entry &entry = entries[index];
      if (isEnvelope(entry.kind) || entry.duration <= 0.0 ||
          multiplicities[index] == 0)
        continue;
      auto path = constraints(index);
      if (failed(path))
        return schedule.emitOpError(
            "conditional schedule ancestry is malformed");
      auto counts = mappedResourceCounts(entryResourceSets[index], {});
      if (failed(counts))
        return schedule.emitOpError(
            "schedule resource lacks physical capacity evidence");
      accumulateMetric(baseMetricRoot, entry.kind,
                       entry.duration *
                           static_cast<double>(multiplicities[index]),
                       counts->first, counts->second, *path);
      peakCandidates.push_back(
          {entry.start, entry.duration, 1, entryQubitSets[index], *path});
    }
    checkpoint("direct");

    summaryOccurrenceRanges.resize(entries.size());
    size_t nextCondition = entries.size();
    for (size_t index = 0; index < entries.size(); ++index) {
      const Entry &invocation = entries[index];
      if (invocation.kind != "call_template")
        continue;
      auto canonical = byId.find(invocation.templateEvent);
      if (invocation.templateEvent.empty() || canonical == byId.end() ||
          entries[canonical->second].kind != "call")
        return schedule.emitOpError(
            "call_template lacks an earlier canonical call event");
      auto summary = summaryFor(canonical->second);
      if (failed(summary))
        return failure();
      if ((*summary)->conditionCount >
          std::numeric_limits<size_t>::max() - nextCondition)
        return schedule.emitOpError(
            "call_template conditional identity space overflows size_t");
      auto outer = constraints(index);
      if (failed(outer))
        return schedule.emitOpError(
            "call_template conditional ancestry is malformed");
      const size_t begin = summaryOccurrences.size();
      const size_t alias = index;
      const ArrayRef<double> noOffsets;
      if (failed(appendSummaryOccurrences(index, **summary, invocation.start,
                                          noOffsets, multiplicities[index],
                                          nextCondition,
                                          ArrayRef<size_t>(&alias, 1), *outer)))
        return failure();
      summaryOccurrenceRanges[index] =
          std::make_pair(begin, summaryOccurrences.size());
      nextCondition += (*summary)->conditionCount;
    }
    retryConditionBase = nextCondition;
    checkpoint("roots");

    if (failed(buildOccurrenceResourceEvidence()))
      return schedule.emitOpError(
          "call_template resource alias lacks physical qubit evidence");
    checkpoint("resource-evidence");

    occurrenceMetricOffsets.assign(summaryOccurrences.size() + 1, 0);
    for (size_t index = 0; index < summaryOccurrences.size(); ++index)
      occurrenceMetricOffsets[index + 1] =
          occurrenceMetricOffsets[index] +
          summaryOccurrences[index].summary->metrics.size();
    occurrenceMetricCounts.resize(occurrenceMetricOffsets.back());
    size_t metricShards = 1;
    if (context->isMultithreadingEnabled())
      metricShards = std::min(
          {size_t{64},
           static_cast<size_t>(context->getThreadPool().getMaxConcurrency()),
           std::max<size_t>(1, summaryOccurrences.size())});
    auto resolveMetricShard = [&](size_t shard) -> LogicalResult {
      const size_t begin = summaryOccurrences.size() * shard / metricShards;
      const size_t end = summaryOccurrences.size() * (shard + 1) / metricShards;
      for (size_t index = begin; index < end; ++index) {
        const SummaryOccurrence &occurrence = summaryOccurrences[index];
        for (auto [position, metric] :
             llvm::enumerate(occurrence.summary->metrics)) {
          auto counts =
              mappedResourceCounts(metric.resourceSet, occurrence.aliases);
          if (failed(counts))
            return failure();
          occurrenceMetricCounts[occurrenceMetricOffsets[index] + position] =
              *counts;
        }
      }
      return success();
    };
    if (failed(failableParallelForEachN(context, 0, metricShards,
                                        resolveMetricShard)))
      return schedule.emitOpError(
          "call_template metric resource alias lacks physical qubit evidence");
    for (auto [index, occurrence] : llvm::enumerate(summaryOccurrences)) {
      for (auto [position, metric] :
           llvm::enumerate(occurrence.summary->metrics)) {
        const auto counts =
            occurrenceMetricCounts[occurrenceMetricOffsets[index] + position];
        accumulateMetric(baseMetricRoot, metric.kind,
                         metric.weightedDuration *
                             static_cast<double>(occurrence.metricFactor),
                         counts.first, counts.second,
                         occurrenceConstraints(occurrence, metric.path));
      }
    }
    checkpoint("metrics");
    if (profile) {
      size_t summaryMetrics = 0;
      size_t summaryPeaks = 0;
      size_t nested = 0;
      for (const auto &[root, summary] : callSummaries) {
        (void)root;
        summaryMetrics += summary.metrics.size();
        summaryPeaks += summary.peaks.size();
        nested += summary.nested.size();
      }
      llvm::errs() << "phys-estimate-schedule summary-counts calls="
                   << callSummaries.size() << " metrics=" << summaryMetrics
                   << " peaks=" << summaryPeaks << " nested=" << nested
                   << " occurrences=" << summaryOccurrences.size()
                   << " direct-peaks=" << peakCandidates.size()
                   << " resource-sets=" << internedResourceSets.size()
                   << " qubit-sets=" << internedQubitSets.size()
                   << " paths=" << internedSummaryPaths.size() << "\n";
    }
    return success();
  }

  LogicalResult buildPeakAtoms() {
    struct LocalEdge {
      double time;
      int64_t delta;
    };
    struct SignatureGroup {
      size_t exemplar;
      std::vector<LocalEdge> edges;
    };
    llvm::DenseMap<size_t, SmallVector<size_t, 1>> buckets;
    std::vector<SignatureGroup> groups;
    groups.reserve(peakCandidates.size() / 8);
    auto equalSignature = [&](size_t left, size_t right) {
      const PeakAtom &a = peakCandidates[left];
      const PeakAtom &b = peakCandidates[right];
      return a.qubitSet == b.qubitSet && a.constraints == b.constraints;
    };
    for (size_t index = 0; index < peakCandidates.size(); ++index) {
      const PeakAtom &candidate = peakCandidates[index];
      if (candidate.leaves == 0 || candidate.duration < 0.0)
        continue;
      double finish = candidate.start + candidate.duration;
      if (finish <= candidate.start) {
        peakAtoms.push_back(candidate);
        continue;
      }
      llvm::hash_code hash = llvm::hash_value(candidate.qubitSet);
      for (const auto &[condition, branch] : candidate.constraints)
        hash = llvm::hash_combine(hash, condition, StringRef(branch));
      auto &candidates = buckets[static_cast<size_t>(hash)];
      size_t group = std::numeric_limits<size_t>::max();
      for (size_t candidate : candidates)
        if (equalSignature(index, groups[candidate].exemplar)) {
          group = candidate;
          break;
        }
      if (group == std::numeric_limits<size_t>::max()) {
        group = groups.size();
        groups.push_back({index, {}});
        candidates.push_back(group);
      }
      groups[group].edges.push_back({candidate.start, candidate.leaves});
      groups[group].edges.push_back({finish, -candidate.leaves});
    }

    for (SignatureGroup &group : groups) {
      llvm::sort(group.edges,
                 [](const LocalEdge &left, const LocalEdge &right) {
                   if (left.time != right.time)
                     return left.time < right.time;
                   return left.delta < right.delta;
                 });
      int64_t active = 0;
      double segmentStart = 0.0;
      size_t cursor = 0;
      while (cursor < group.edges.size()) {
        const double time = group.edges[cursor].time;
        int64_t delta = 0;
        while (cursor < group.edges.size() && group.edges[cursor].time == time)
          delta += group.edges[cursor++].delta;
        int64_t next = active + delta;
        if (next < 0)
          return schedule.emitOpError(
              "schedule peak signature occupancy became negative");
        if (next != active) {
          if (active > 0 && time > segmentStart) {
            const PeakAtom &exemplar = peakCandidates[group.exemplar];
            peakAtoms.push_back({segmentStart, time - segmentStart, active,
                                 exemplar.qubitSet, exemplar.constraints});
          }
          if (next > 0)
            segmentStart = time;
          active = next;
        }
      }
      if (active != 0)
        return schedule.emitOpError(
            "schedule peak signature occupancy remains active after sweep");
    }
    if (std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE"))
      llvm::errs() << "phys-estimate-schedule peak-compression leaves="
                   << peakCandidates.size() << " signatures=" << groups.size()
                   << " atoms=" << peakAtoms.size() << "\n";
    return success();
  }

  LogicalResult buildRetrySlices() {
    std::vector<char> controlled(entries.size(), false);
    GenerationMarks traversal(entries.size());
    GenerationMarks coveredMarks(entries.size());
    uint64_t decisionNodes = 0;
    uint64_t forwardNodes = 0;
    uint64_t requiredNodes = 0;
    uint64_t coveredNodes = 0;
    uint64_t causalEdges = 0;
    uint64_t replayEvents = 0;
    for (size_t retryIndex = 0; retryIndex < entries.size(); ++retryIndex) {
      const Entry &retry = entries[retryIndex];
      if (retry.kind != "retry")
        continue;
      if (retry.exhaustion != "report_failure" && retry.exhaustion != "abort" &&
          retry.exhaustion != "return_last")
        return schedule.emitOpError("missing evidence: retry event '")
               << retry.id << "' has unsupported exhaustion policy '"
               << retry.exhaustion << "'";
      if (retry.attemptEvent.empty() || retry.decisionEvent.empty() ||
          !retry.maxAttempts || !retry.successProbability ||
          *retry.maxAttempts <= 0)
        return schedule.emitOpError("missing evidence: retry event '")
               << retry.id
               << "' lacks exact attempt, decision, bound, or probability";
      auto attemptIt = byId.find(retry.attemptEvent);
      auto decisionIt = byId.find(retry.decisionEvent);
      if (attemptIt == byId.end() || decisionIt == byId.end())
        return schedule.emitOpError("missing evidence: retry event '")
               << retry.id << "' references a missing schedule event";
      size_t attemptIndex = attemptIt->second;
      size_t decisionIndex = decisionIt->second;
      const Entry &attempt = entries[attemptIndex];
      if ((attempt.kind != "call" && attempt.kind != "call_template") ||
          attempt.callee != retry.attempt || attempt.profile != retry.profile)
        return schedule.emitOpError("missing evidence: retry event '")
               << retry.id << "' has inconsistent attempt/profile provenance";

      traversal.reset();
      SmallVector<size_t, 16> pending{decisionIndex};
      while (!pending.empty()) {
        size_t current = pending.pop_back_val();
        if ((current != attemptIndex && current < attemptIndex) ||
            !traversal.insert(current))
          continue;
        ++decisionNodes;
        if (current == attemptIndex)
          continue;
        for (StringRef dependency : entries[current].dependencies) {
          ++causalEdges;
          pending.push_back(byId.lookup(dependency));
        }
      }
      if (!traversal.contains(attemptIndex))
        return schedule.emitOpError("missing evidence: retry decision '")
               << retry.decisionEvent << "' is not derived from attempt '"
               << retry.attemptEvent << "'";

      traversal.reset();
      SmallVector<size_t, 16> forward;
      pending.push_back(attemptIndex);
      while (!pending.empty()) {
        size_t current = pending.pop_back_val();
        if (current == retryIndex ||
            (current != attemptIndex && current >= retryIndex) ||
            !traversal.insert(current))
          continue;
        ++forwardNodes;
        forward.push_back(current);
        causalEdges += dependents[current].size() + children[current].size();
        pending.append(dependents[current]);
        pending.append(children[current]);
      }

      traversal.reset();
      SmallVector<size_t, 16> required;
      pending.push_back(decisionIndex);
      for (StringRef dependency : retry.dependencies) {
        ++causalEdges;
        pending.push_back(byId.lookup(dependency));
      }
      while (!pending.empty()) {
        size_t current = pending.pop_back_val();
        if (current == retryIndex ||
            (current != attemptIndex && current < attemptIndex) ||
            !traversal.insert(current))
          continue;
        ++requiredNodes;
        required.push_back(current);
        if (current == attemptIndex)
          continue;
        for (StringRef dependency : entries[current].dependencies) {
          ++causalEdges;
          pending.push_back(byId.lookup(dependency));
        }
        if (!entries[current].parent.empty()) {
          ++causalEdges;
          pending.push_back(byId.lookup(entries[current].parent));
        }
      }
      coveredMarks.reset();
      pending.clear();
      for (size_t current : required)
        if (coveredMarks.insert(current)) {
          ++coveredNodes;
          pending.push_back(current);
        }
      while (!pending.empty()) {
        size_t current = pending.pop_back_val();
        for (size_t child : children[current]) {
          ++causalEdges;
          if (child < retryIndex && coveredMarks.insert(child)) {
            ++coveredNodes;
            pending.push_back(child);
          }
        }
      }
      SmallVector<size_t, 16> replay;
      for (size_t current : forward)
        if (current < retryIndex)
          replay.push_back(current);
      replay.push_back(attemptIndex);
      llvm::sort(replay);
      replay.erase(std::unique(replay.begin(), replay.end()), replay.end());
      for (size_t current : replay)
        if (!coveredMarks.contains(current))
          return schedule.emitOpError("missing evidence: retry event '")
                 << retry.id
                 << "' leaves attempt-descended events outside its replay "
                    "closure";
      for (size_t current : replay)
        if (controlled[current])
          return schedule.emitOpError("missing evidence: retry event '")
                 << retry.id << "' overlaps another retry causal slice";
      for (size_t current : replay)
        controlled[current] = true;
      replayEvents += replay.size();

      int64_t occurrences = multiplicities[retryIndex];
      ReplaySlice slice;
      slice.retry = retryIndex;
      slice.attempt = attemptIndex;
      slice.occurrences = occurrences;
      slice.replayDuration = retry.start - attempt.start;
      slice.attemptStart = attempt.start;
      slice.retryStart = retry.start;
      slice.timeAnchor = retryIndex;
      if (slice.replayDuration <= 0.0)
        return schedule.emitOpError("missing evidence: retry event '")
               << retry.id << "' has no positive replay interval";
      for (size_t current : replay) {
        int64_t total = multiplicities[current];
        if (occurrences == 0 || total % occurrences)
          return schedule.emitOpError("missing evidence: retry event '")
                 << retry.id
                 << "' has incommensurate folded replay multiplicity";
        slice.weighted.emplace_back(current, total / occurrences);
      }
      retrySlices.push_back(std::move(slice));
    }

    // A canonical phys.call executes once at its authored location, while each
    // call_template creates another real occurrence of the same compact body.
    // Reuse the authenticated canonical retry slice for those occurrences;
    // retain only the absolute time shift, folded execution factor, and the
    // existing summary occurrence that owns resource aliases and conditional
    // context.  This is exact in the size of the compact call graph and never
    // expands either the call body or the retry attempt bound.
    const size_t canonicalRetryCount = retrySlices.size();
    uint64_t templateRetryOccurrences = 0;
    llvm::DenseMap<size_t, SmallVector<size_t, 4>> occurrencesByRoot;
    for (auto [occurrenceIndex, occurrence] :
         llvm::enumerate(summaryOccurrences))
      occurrencesByRoot[occurrence.summary->root].push_back(occurrenceIndex);
    for (size_t sliceIndex = 0; sliceIndex < canonicalRetryCount;
         ++sliceIndex) {
      const ReplaySlice canonical = retrySlices[sliceIndex];
      std::optional<size_t> callRoot;
      size_t child = canonical.retry;
      llvm::SmallDenseSet<size_t, 8> visited;
      while (!entries[child].parent.empty()) {
        auto parent = byId.find(entries[child].parent);
        if (parent == byId.end() || !visited.insert(parent->second).second)
          return schedule.emitOpError(
              "retry call-template ancestry is malformed");
        child = parent->second;
        if (entries[child].kind == "call") {
          callRoot = child;
          break;
        }
      }
      if (!callRoot)
        continue;
      if (multiplicities[*callRoot] <= 0 ||
          canonical.occurrences % multiplicities[*callRoot])
        return schedule.emitOpError(
            "retry call-template folded multiplicity is incommensurate");
      const int64_t localOccurrences =
          canonical.occurrences / multiplicities[*callRoot];
      auto retainedOccurrences = occurrencesByRoot.find(*callRoot);
      if (retainedOccurrences == occurrencesByRoot.end())
        continue;
      for (size_t occurrenceIndex : retainedOccurrences->second) {
        const SummaryOccurrence &occurrence =
            summaryOccurrences[occurrenceIndex];
        if (localOccurrences != 0 &&
            occurrence.metricFactor >
                std::numeric_limits<int64_t>::max() / localOccurrences)
          return schedule.emitOpError(
              "retry call-template occurrence multiplicity overflows i64");
        const int64_t occurrences = localOccurrences * occurrence.metricFactor;
        if (occurrences == 0)
          continue;
        double occurrenceStart = occurrence.topStart;
        for (double offset : occurrence.offsets)
          occurrenceStart += offset;
        const double attemptOffset =
            entries[canonical.attempt].start - entries[*callRoot].start;
        const double retryOffset =
            entries[canonical.retry].start - entries[*callRoot].start;
        ReplaySlice clone = canonical;
        clone.occurrences = occurrences;
        clone.attemptStart = occurrenceStart + attemptOffset;
        clone.retryStart = occurrenceStart + retryOffset;
        clone.timeAnchor = occurrence.source;
        clone.summaryOccurrence = occurrenceIndex;
        retrySlices.push_back(std::move(clone));
        ++templateRetryOccurrences;
      }
    }
    if (std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE"))
      llvm::errs() << "phys-estimate-schedule retry-slice-work retries="
                   << retrySlices.size() << " decision-nodes=" << decisionNodes
                   << " forward-nodes=" << forwardNodes
                   << " required-nodes=" << requiredNodes
                   << " covered-nodes=" << coveredNodes
                   << " causal-edges=" << causalEdges
                   << " replay-events=" << replayEvents
                   << " template-occurrences=" << templateRetryOccurrences
                   << "\n";
    return success();
  }

  FailureOr<std::pair<int64_t, int64_t>> peaks() {
    const bool profile =
        std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") != nullptr;
    auto checkpoint = [profile, last = std::chrono::steady_clock::now()](
                          StringRef name) mutable {
      const auto now = std::chrono::steady_clock::now();
      if (profile)
        llvm::errs() << "phys-estimate-schedule peaks-" << name << ' '
                     << std::chrono::duration<double>(now - last).count()
                     << "s\n";
      last = now;
    };
    struct SweepEvent {
      double time;
      unsigned phase;
      size_t owner;
      size_t position;
      bool direct;
      bool finish;
    };
    struct SweepResult {
      int64_t peak = 0;
      int64_t qubitPeak = 0;
      double qubitPeakTime = std::numeric_limits<double>::infinity();
      double qubitPeakNextTime = 0.0;
      int64_t qubitPeakAfterCollapsed = 0;
      llvm::BitVector peakQubits;
      size_t appliedEvents = 0;
      size_t conditionalWork = 0;
      bool conditionalWorkExceeded = false;
      double seconds = 0.0;
    };
    auto later = [](const SweepEvent &left, const SweepEvent &right) {
      if (left.time != right.time)
        return left.time > right.time;
      if (left.phase != right.phase)
        return left.phase > right.phase;
      if (left.direct != right.direct)
        return left.direct > right.direct;
      if (left.owner != right.owner)
        return left.owner > right.owner;
      return left.position > right.position;
    };
    SmallVector<double, 0> occurrenceShifts;
    occurrenceShifts.reserve(summaryOccurrences.size());
    for (const SummaryOccurrence &occurrence : summaryOccurrences) {
      double shift = occurrence.topStart;
      for (double offset : occurrence.offsets)
        shift += offset;
      occurrenceShifts.push_back(shift);
    }
    checkpoint("edges");
    struct WeightedTime {
      double time;
      size_t weight;
    };
    SmallVector<WeightedTime, 0> weightedTimes;
    weightedTimes.reserve(peakAtoms.size() + summaryOccurrences.size());
    double minimumTime = std::numeric_limits<double>::infinity();
    double maximumTime = 0.0;
    for (const PeakAtom &atom : peakAtoms) {
      const double finish = atom.start + atom.duration;
      minimumTime = std::min(minimumTime, atom.start);
      maximumTime = std::max(maximumTime, finish);
      weightedTimes.push_back({(atom.start + finish) / 2.0, 2});
    }
    for (size_t owner = 0; owner < summaryOccurrences.size(); ++owner) {
      const CallSummary &summary = *summaryOccurrences[owner].summary;
      if (summary.peaks.empty())
        continue;
      const double first =
          occurrenceShifts[owner] + summary.peaks.front().start;
      const double last =
          occurrenceShifts[owner] + summary.peakMaxFinishTree[1];
      minimumTime = std::min(minimumTime, first);
      maximumTime = std::max(maximumTime, last);
      weightedTimes.push_back({(first + last) / 2.0, 2 * summary.peaks.size()});
    }
    if (weightedTimes.empty())
      return std::make_pair<int64_t, int64_t>(0, 0);
    llvm::sort(weightedTimes,
               [](const WeightedTime &left, const WeightedTime &right) {
                 return left.time < right.time;
               });
    size_t shardCount = 1;
    if (context->isMultithreadingEnabled() && maximumTime > minimumTime &&
        summaryOccurrences.size() > 1)
      shardCount = std::min(
          {size_t{64},
           static_cast<size_t>(context->getThreadPool().getMaxConcurrency()),
           weightedTimes.size()});
    SmallVector<double, 9> boundaries{minimumTime};
    if (shardCount > 1) {
      size_t totalWeight = 0;
      for (const WeightedTime &value : weightedTimes)
        totalWeight += value.weight;
      size_t cursor = 0;
      size_t cumulative = 0;
      for (size_t shard = 1; shard < shardCount; ++shard) {
        const size_t target = totalWeight * shard / shardCount;
        while (cursor < weightedTimes.size() && cumulative < target)
          cumulative += weightedTimes[cursor++].weight;
        boundaries.push_back(weightedTimes[cursor == 0 ? 0 : cursor - 1].time);
      }
    }
    boundaries.push_back(maximumTime);
    std::vector<SweepResult> results(shardCount);
    auto sweepShard = [&](size_t shard) -> LogicalResult {
      const auto shardStarted = std::chrono::steady_clock::now();
      const double begin = boundaries[shard];
      const double end = boundaries[shard + 1];
      const bool includeEnd = shard + 1 == shardCount;
      auto inRange = [&](double time) {
        return time >= begin && (includeEnd ? time <= end : time < end);
      };
      std::priority_queue<SweepEvent, std::vector<SweepEvent>, decltype(later)>
          events(later);
      ActiveOccupancy active(qubitWeights);
      std::map<std::pair<size_t, SummaryPathId>, SmallVector<Constraint, 8>>
          occurrencePaths;
      auto applySummaryAtom = [&](size_t owner, size_t atomIndex,
                                  int direction) -> LogicalResult {
        const SummaryOccurrence &occurrence = summaryOccurrences[owner];
        const SummaryPeakAtom &atom = occurrence.summary->peaks[atomIndex];
        const OccurrenceResourceMapping *qubits =
            occurrenceResourceMapping(owner, atom.resourceSet);
        if (!qubits)
          return failure();
        ArrayRef<Constraint> constraints;
        auto path = occurrencePaths.end();
        if (!occurrence.prefix.empty() ||
            !internedSummaryPaths[atom.path].empty()) {
          path = occurrencePaths.find({owner, atom.path});
          if (path == occurrencePaths.end())
            path =
                occurrencePaths
                    .try_emplace(std::make_pair(owner, atom.path),
                                 occurrenceConstraints(occurrence, atom.path))
                    .first;
          constraints = path->second;
        }
        return success(active.update(atom.leaves, constraints,
                                     internedQubitSets[qubits->qubits],
                                     direction));
      };
      auto apply = [&](const SweepEvent &event,
                       int direction) -> LogicalResult {
        ++results[shard].appliedEvents;
        if (event.direct) {
          const PeakAtom &atom = peakAtoms[event.owner];
          return success(
              active.update(atom, internedQubitSets[atom.qubitSet], direction));
        }
        const CallSummary &summary = *summaryOccurrences[event.owner].summary;
        size_t atomIndex = event.finish
                               ? summary.peakFinishOrder[event.position]
                               : event.position;
        return applySummaryAtom(event.owner, atomIndex, direction);
      };
      auto pushSummaryEvent = [&](size_t owner, size_t position, bool finish) {
        const CallSummary &summary = *summaryOccurrences[owner].summary;
        if (position >= summary.peaks.size())
          return;
        size_t atomIndex =
            finish ? summary.peakFinishOrder[position] : position;
        const SummaryPeakAtom &atom = summary.peaks[atomIndex];
        const double startTime = occurrenceShifts[owner] + atom.start;
        const double finishTime = startTime + atom.duration;
        const double time = finish ? finishTime : startTime;
        if (!inRange(time))
          return;
        unsigned phase = finish && finishTime > startTime ? 0u
                         : finish                         ? 2u
                                                          : 1u;
        events.push({time, phase, owner, position, false, finish});
      };

      for (size_t index = 0; index < peakAtoms.size(); ++index) {
        const PeakAtom &atom = peakAtoms[index];
        const double finish = atom.start + atom.duration;
        if (atom.start < begin && finish >= begin &&
            !active.update(atom, internedQubitSets[atom.qubitSet], 1))
          return failure();
        if (inRange(atom.start))
          events.push({atom.start, 1, index, 0, true, false});
        if (inRange(finish))
          events.push(
              {finish, finish <= atom.start ? 2u : 0u, index, 0, true, true});
      }
      for (size_t owner = 0; owner < summaryOccurrences.size(); ++owner) {
        const CallSummary &summary = *summaryOccurrences[owner].summary;
        if (summary.peaks.empty())
          continue;
        const double localBegin = begin - occurrenceShifts[owner];
        size_t startPosition =
            std::lower_bound(summary.peaks.begin(), summary.peaks.end(),
                             localBegin,
                             [](const SummaryPeakAtom &atom, double time) {
                               return atom.start < time;
                             }) -
            summary.peaks.begin();
        while (startPosition < summary.peaks.size() &&
               occurrenceShifts[owner] + summary.peaks[startPosition].start <
                   begin)
          ++startPosition;
        if (begin > minimumTime && startPosition != 0) {
          struct IntervalNode {
            size_t node;
            size_t left;
            size_t right;
          };
          SmallVector<IntervalNode, 32> pending{{1, 0, summary.peakTreeBase}};
          while (!pending.empty()) {
            IntervalNode current = pending.pop_back_val();
            if (current.left >= startPosition ||
                summary.peakMaxFinishTree[current.node] < localBegin)
              continue;
            if (current.right - current.left == 1) {
              if (current.left < summary.peaks.size() &&
                  failed(applySummaryAtom(owner, current.left, 1)))
                return failure();
              continue;
            }
            size_t middle = (current.left + current.right) / 2;
            pending.push_back({2 * current.node + 1, middle, current.right});
            pending.push_back({2 * current.node, current.left, middle});
          }
        }
        size_t finishPosition =
            std::lower_bound(summary.peakFinishOrder.begin(),
                             summary.peakFinishOrder.end(), localBegin,
                             [&](size_t index, double time) {
                               const SummaryPeakAtom &atom =
                                   summary.peaks[index];
                               return atom.start + atom.duration < time;
                             }) -
            summary.peakFinishOrder.begin();
        while (finishPosition < summary.peaks.size()) {
          const SummaryPeakAtom &atom =
              summary.peaks[summary.peakFinishOrder[finishPosition]];
          if (occurrenceShifts[owner] + atom.start + atom.duration >= begin)
            break;
          ++finishPosition;
        }
        pushSummaryEvent(owner, startPosition, false);
        pushSummaryEvent(owner, finishPosition, true);
      }

      while (!events.empty()) {
        const double time = events.top().time;
        while (!events.empty() && events.top().time == time &&
               events.top().phase == 0) {
          SweepEvent event = events.top();
          events.pop();
          if (failed(apply(event, -1)))
            return failure();
          if (!event.direct)
            pushSummaryEvent(event.owner, event.position + 1, true);
        }
        while (!events.empty() && events.top().time == time &&
               events.top().phase == 1) {
          SweepEvent event = events.top();
          events.pop();
          if (failed(apply(event, 1)))
            return failure();
          if (!event.direct)
            pushSummaryEvent(event.owner, event.position + 1, false);
        }
        SweepResult &result = results[shard];
        result.peak = std::max(result.peak, active.leaves());
        auto activeQubits = active.qubits();
        result.conditionalWork += active.lastMaximumWork();
        if (failed(activeQubits)) {
          result.conditionalWorkExceeded = true;
          return failure();
        }
        if (*activeQubits > result.qubitPeak) {
          result.qubitPeak = *activeQubits;
          result.qubitPeakTime = time;
          auto peakQubits = active.largestQubitSet();
          if (failed(peakQubits)) {
            result.conditionalWorkExceeded = true;
            return failure();
          }
          result.peakQubits = std::move(*peakQubits);
        }
        while (!events.empty() && events.top().time == time &&
               events.top().phase == 2) {
          SweepEvent event = events.top();
          events.pop();
          if (failed(apply(event, -1)))
            return failure();
          if (!event.direct)
            pushSummaryEvent(event.owner, event.position + 1, true);
        }
        if (time == result.qubitPeakTime) {
          auto afterCollapsed = active.qubits();
          result.conditionalWork += active.lastMaximumWork();
          if (failed(afterCollapsed)) {
            result.conditionalWorkExceeded = true;
            return failure();
          }
          result.qubitPeakAfterCollapsed = *afterCollapsed;
          result.qubitPeakNextTime = events.empty() ? time : events.top().time;
        }
      }
      results[shard].seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        shardStarted)
              .count();
      return success();
    };
    if (failed(failableParallelForEachN(context, 0, shardCount, sweepShard))) {
      if (llvm::any_of(results, [](const SweepResult &result) {
            return result.conditionalWorkExceeded;
          }))
        return schedule.emitOpError(
            "missing evidence: exact conditional peak-occupancy overlap "
            "exceeds the bounded work limit");
      return failure();
    }
    if (profile)
      for (size_t shard = 0; shard < shardCount; ++shard)
        llvm::errs() << "phys-estimate-schedule peak-shard " << shard
                     << " range=[" << boundaries[shard] << ','
                     << boundaries[shard + 1]
                     << "] events=" << results[shard].appliedEvents
                     << " conditional-work=" << results[shard].conditionalWork
                     << " seconds=" << results[shard].seconds << "\n";
    checkpoint("sweep");
    SweepResult combined;
    combined.peakQubits = llvm::BitVector(qubitIds.size());
    for (SweepResult &result : results) {
      combined.appliedEvents += result.appliedEvents;
      combined.peak = std::max(combined.peak, result.peak);
      if (result.qubitPeak > combined.qubitPeak ||
          (result.qubitPeak == combined.qubitPeak &&
           result.qubitPeakTime < combined.qubitPeakTime)) {
        combined.qubitPeak = result.qubitPeak;
        combined.qubitPeakTime = result.qubitPeakTime;
        combined.qubitPeakNextTime = result.qubitPeakNextTime;
        combined.qubitPeakAfterCollapsed = result.qubitPeakAfterCollapsed;
        combined.peakQubits = std::move(result.peakQubits);
      }
    }
    if (profile)
      llvm::errs() << "phys-estimate-schedule peak-result events="
                   << combined.appliedEvents << " shards=" << shardCount
                   << " concurrency=" << combined.peak
                   << " qubits=" << combined.qubitPeak
                   << " time=" << combined.qubitPeakTime
                   << " after-collapsed=" << combined.qubitPeakAfterCollapsed
                   << " next-time=" << combined.qubitPeakNextTime << "\n";
    if (profile) {
      std::map<std::string, int64_t> classes;
      std::map<std::string, std::pair<int64_t, int64_t>> indexRanges;
      for (const auto &entry : qubitIds) {
        if (!combined.peakQubits.test(entry.getValue()))
          continue;
        StringRef name = entry.getKey();
        auto [resourceClass, suffix] = name.split('[');
        classes[resourceClass.str()]++;
        if (!suffix.consume_back("]"))
          continue;
        int64_t index = 0;
        if (suffix.getAsInteger(10, index))
          continue;
        auto [range, inserted] = indexRanges.try_emplace(
            resourceClass.str(), std::make_pair(index, index));
        if (!inserted) {
          range->second.first = std::min(range->second.first, index);
          range->second.second = std::max(range->second.second, index);
        }
      }
      llvm::errs() << "phys-estimate-schedule peak-qubit-classes";
      for (const auto &[name, count] : classes) {
        llvm::errs() << ' ' << name << '=' << count;
        auto range = indexRanges.find(name);
        if (range != indexRanges.end())
          llvm::errs() << '[' << range->second.first << ".."
                       << range->second.second << ']';
      }
      llvm::errs() << "\n";
    }
    return std::make_pair(combined.peak, combined.qubitPeak);
  }

  LogicalResult retryTimeExtras(double &expected, double &maximum) {
    stoppedRetryTimeLayers.clear();
    stoppedRetryTimeLayerForSlice.assign(retrySlices.size(),
                                         std::numeric_limits<size_t>::max());
    if (retrySlices.empty())
      return success();
    const bool profile =
        std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") != nullptr;
    SmallVector<const ReplaySlice *, 8> ordered;
    for (const ReplaySlice &slice : retrySlices)
      ordered.push_back(&slice);
    llvm::sort(ordered, [](const ReplaySlice *left, const ReplaySlice *right) {
      if (left->attemptStart != right->attemptStart)
        return left->attemptStart < right->attemptStart;
      if (left->retryStart != right->retryStart)
        return left->retryStart < right->retryStart;
      if (left->attempt != right->attempt)
        return left->attempt < right->attempt;
      return left->summaryOccurrence < right->summaryOccurrence;
    });
    SmallVector<SmallVector<const ReplaySlice *, 4>, 4> groups;
    double groupEnd = -1.0;
    for (const ReplaySlice *slice : ordered) {
      double start = slice->attemptStart;
      double end = slice->retryStart;
      if (!groups.empty() && start >= groupEnd) {
        groups.emplace_back();
        groupEnd = -1.0;
      }
      if (groups.empty())
        groups.emplace_back();
      groups.back().push_back(slice);
      groupEnd = std::max(groupEnd, end);
    }
    uint64_t singletonGroups = 0;
    uint64_t parallelGroups = 0;
    uint64_t ownerPropagationRows = 0;
    uint64_t ownerDependencyEdges = 0;
    uint64_t controlledEvents = 0;
    uint64_t profiledAbortGroups = 0;
    auto sameContext = [](const RetryExecutionContext &left,
                          const RetryExecutionContext &right) {
      if (left.folds.size() != right.folds.size() ||
          left.conditions != right.conditions)
        return false;
      for (auto [leftFold, rightFold] : llvm::zip(left.folds, right.folds))
        if (leftFold.event != rightFold.event ||
            leftFold.expansion != rightFold.expansion ||
            leftFold.branch != rightFold.branch ||
            leftFold.count != rightFold.count ||
            leftFold.iterationFinish != rightFold.iterationFinish ||
            leftFold.envelopeFinish != rightFold.envelopeFinish)
          return false;
      return true;
    };
    auto executionContext =
        [&](const ReplaySlice &slice) -> FailureOr<RetryExecutionContext> {
      RetryExecutionContext result;
      llvm::SmallDenseSet<size_t, 8> visited;
      auto appendFolds = [&](size_t child, std::optional<size_t> stop,
                             ArrayRef<size_t> expansion,
                             double shift) -> LogicalResult {
        visited.clear();
        while (!entries[child].parent.empty()) {
          auto found = byId.find(entries[child].parent);
          if (found == byId.end() || !visited.insert(found->second).second)
            return failure();
          const size_t parentIndex = found->second;
          if (stop && parentIndex == *stop)
            break;
          const Entry &parent = entries[parentIndex];
          if (parent.kind == "repeat" || parent.kind == "while") {
            std::optional<int64_t> count = parent.kind == "repeat"
                                               ? parent.repeatCount
                                               : parent.maxIterations;
            if (!count)
              return failure();
            DynamicFoldIdentity fold;
            fold.event = parentIndex;
            fold.expansion.assign(expansion.begin(), expansion.end());
            fold.branch = entries[child].branch;
            fold.count = *count;
            double branchFinish = entries[child].finish();
            for (size_t sibling : children[parentIndex])
              if (entries[sibling].branch == entries[child].branch)
                branchFinish =
                    std::max(branchFinish, entries[sibling].finish());
            fold.iterationFinish = branchFinish + shift;
            fold.envelopeFinish = parent.finish() + shift;
            result.folds.push_back(std::move(fold));
          }
          child = parentIndex;
        }
        return success();
      };

      if (!slice.summaryOccurrence) {
        if (failed(appendFolds(slice.retry, std::nullopt, {}, 0.0)))
          return failure();
        auto path = constraints(slice.retry);
        if (failed(path))
          return failure();
        result.conditions.append(path->begin(), path->end());
        return result;
      }

      const size_t occurrenceIndex = *slice.summaryOccurrence;
      const SummaryOccurrence &occurrence = summaryOccurrences[occurrenceIndex];
      double occurrenceStart = occurrence.topStart;
      for (double offset : occurrence.offsets)
        occurrenceStart += offset;
      if (failed(appendFolds(
              slice.retry, occurrence.summary->root, occurrence.aliases,
              occurrenceStart - entries[occurrence.summary->root].start)))
        return failure();
      for (size_t aliasIndex = 0; aliasIndex < occurrence.aliases.size();
           ++aliasIndex) {
        const size_t alias = occurrence.aliases[aliasIndex];
        std::optional<size_t> stop;
        if (aliasIndex + 1 < occurrence.aliases.size()) {
          size_t child = alias;
          llvm::SmallDenseSet<size_t, 8> callVisited;
          while (!entries[child].parent.empty()) {
            auto parent = byId.find(entries[child].parent);
            if (parent == byId.end() ||
                !callVisited.insert(parent->second).second)
              return failure();
            child = parent->second;
            if (entries[child].kind == "call") {
              stop = child;
              break;
            }
          }
          if (!stop)
            return failure();
        }
        ArrayRef<size_t> expansion(occurrence.aliases);
        expansion = expansion.drop_front(aliasIndex + 1);
        double aliasStart = occurrence.topStart;
        const size_t retainedOffsets =
            aliasIndex < occurrence.offsets.size()
                ? occurrence.offsets.size() - aliasIndex
                : 0;
        for (size_t offset = 0; offset < retainedOffsets; ++offset)
          aliasStart += occurrence.offsets[offset];
        if (failed(appendFolds(alias, stop, expansion,
                               aliasStart - entries[alias].start)))
          return failure();
      }

      result.conditions.append(occurrence.prefix.begin(),
                               occurrence.prefix.end());
      auto local = constraints(slice.retry, occurrence.summary->root);
      if (failed(local))
        return failure();
      for (const auto &[condition, branch] : *local) {
        const size_t maximum = std::numeric_limits<size_t>::max();
        if (condition > maximum - retryConditionBase || entries.empty() ||
            occurrenceIndex >
                (maximum - retryConditionBase - condition) / entries.size())
          return failure();
        result.conditions.emplace_back(retryConditionBase + condition +
                                           occurrenceIndex * entries.size(),
                                       branch);
      }
      return result;
    };
    auto cdf = [](double probability, int64_t attempts, int64_t extraAttempts) {
      if (extraAttempts < 0)
        return 0.0;
      if (extraAttempts >= attempts - 1 || probability == 1.0)
        return 1.0;
      return -std::expm1(static_cast<double>(extraAttempts + 1) *
                         std::log1p(-probability));
    };
    using ProbabilityMass = std::map<double, double>;
    struct RetryTimeLayer {
      SmallVector<const ReplaySlice *, 4> slices;
      RetryExecutionContext context;
      int64_t occurrences = 1;
      double expected = 0.0;
      double maximum = 0.0;
      std::optional<ProbabilityMass> support;
      double baseExpected = 0.0;
      double baseMaximum = 0.0;
      uint64_t partialClips = 0;
      bool contributesToMakespan = false;
    };
    SmallVector<RetryTimeLayer, 4> layers;
    for (const auto &group : groups) {
      RetryTimeLayer layer;
      layer.slices.append(group.begin(), group.end());
      double latestStart = 0.0;
      double earliestEnd = std::numeric_limits<double>::infinity();
      std::optional<int64_t> occurrences;
      std::optional<RetryExecutionContext> context;
      for (const ReplaySlice *slice : group) {
        latestStart = std::max(latestStart, slice->attemptStart);
        earliestEnd = std::min(earliestEnd, slice->retryStart);
        if (occurrences && *occurrences != slice->occurrences)
          return schedule.emitOpError(
              "missing evidence: parallel retry layers must share one folded "
              "execution multiplicity");
        occurrences = slice->occurrences;
        auto currentContext = executionContext(*slice);
        if (failed(currentContext))
          return schedule.emitOpError(
              "missing evidence: parallel retry folded context is malformed");
        if (context && !sameContext(*context, *currentContext))
          return schedule.emitOpError(
              "missing evidence: parallel retry layers must share one folded "
              "execution context");
        context = *currentContext;
      }
      layer.occurrences = *occurrences;
      layer.context = std::move(*context);
      if (group.size() == 1) {
        ++singletonGroups;
        layers.push_back(std::move(layer));
        continue;
      }
      if (latestStart >= earliestEnd)
        return schedule.emitOpError(
            "missing evidence: overlapping retries must form one pairwise-"
            "overlapping parallel layer");
      ++parallelGroups;

      // For a parallel retry layer, retain only the ancestry fact required by
      // the independence check: no owner, exactly one local owner, or more
      // than one local owner.  Events before `windowBegin` cannot carry an
      // owner from this group, because every owned event belongs to a retry
      // slice starting at or after its attempt.  The verified schedule order
      // therefore permits one exact forward propagation over this window.
      constexpr int64_t noOwner = -1;
      constexpr int64_t multipleOwners = -2;
      auto mergeOwner = [](int64_t &target, int64_t source) {
        if (source == noOwner || target == multipleOwners)
          return;
        if (target == noOwner) {
          target = source;
          return;
        }
        if (source == multipleOwners || target != source)
          target = multipleOwners;
      };

      // Canonical row indices are reusable summary coordinates, not dynamic
      // identities.  Propagate owners independently inside every instantiated
      // call-summary occurrence so two parallel clones of the same retry do
      // not alias.  Slices in one occurrence still share an owner space and
      // retain the exact foreign-influence check.  Distinct occurrence roots
      // are independently authenticated call invocations; the common folded
      // and conditional context proven above establishes that their
      // pairwise-overlapping retry intervals are sibling variables in the
      // same dynamic layer.
      std::map<std::optional<size_t>, SmallVector<size_t, 4>> ownerSpaces;
      for (size_t owner = 0; owner < group.size(); ++owner)
        ownerSpaces[group[owner]->summaryOccurrence].push_back(owner);
      for (const auto &[occurrence, owners] : ownerSpaces) {
        (void)occurrence;
        size_t windowBegin = entries.size();
        size_t windowEnd = 0;
        for (size_t owner : owners) {
          const ReplaySlice &slice = *group[owner];
          windowBegin = std::min(windowBegin, slice.attempt);
          windowEnd = std::max(windowEnd, slice.retry);
        }
        const size_t windowSize = windowEnd - windowBegin + 1;
        std::vector<int64_t> eventOwner(windowSize, noOwner);
        auto assignOwner = [&](size_t event, int64_t owner) -> LogicalResult {
          int64_t &assigned = eventOwner[event - windowBegin];
          if (assigned != noOwner && assigned != owner)
            return schedule.emitOpError(
                "missing evidence: parallel retry replay slices overlap");
          assigned = owner;
          return success();
        };
        for (size_t owner : owners) {
          const ReplaySlice &slice = *group[owner];
          const int64_t localOwner = static_cast<int64_t>(owner);
          if (failed(assignOwner(slice.retry, localOwner)))
            return failure();
          for (const auto &[event, weight] : slice.weighted) {
            (void)weight;
            if (failed(assignOwner(event, localOwner)))
              return failure();
          }
        }

        std::vector<int64_t> ancestryOwner(windowSize, noOwner);
        for (size_t event = windowBegin; event <= windowEnd; ++event) {
          ++ownerPropagationRows;
          int64_t summary = noOwner;
          for (StringRef dependency : entries[event].dependencies) {
            ++ownerDependencyEdges;
            const size_t source = byId.lookup(dependency);
            if (source >= windowBegin && source < event)
              mergeOwner(summary, ancestryOwner[source - windowBegin]);
          }
          const int64_t own = eventOwner[event - windowBegin];
          mergeOwner(summary, own);
          ancestryOwner[event - windowBegin] = summary;
          if (own == noOwner)
            continue;
          ++controlledEvents;
          if (summary == multipleOwners ||
              (summary != noOwner && summary != own))
            return schedule.emitOpError(
                "missing evidence: parallel retry replay slices must be "
                "causally independent");
        }
      }

      if (profile && profiledAbortGroups < 4 &&
          llvm::any_of(
              group,
              [&](const ReplaySlice *slice) {
                return entries[slice->retry].exhaustion == "abort";
              })) {
        llvm::errs() << "phys-estimate-schedule abort-overlap-group "
                     << profiledAbortGroups << " slices=" << group.size()
                     << " occurrences=" << layer.occurrences
                     << " folds=" << layer.context.folds.size()
                     << " conditions=" << layer.context.conditions.size()
                     << " window=[" << latestStart << ',' << earliestEnd
                     << "]\n";
        for (const DynamicFoldIdentity &fold : layer.context.folds)
          llvm::errs() << "phys-estimate-schedule abort-overlap-fold event="
                       << entries[fold.event].id << " branch=" << fold.branch
                       << " count=" << fold.count
                       << " iteration-finish=" << fold.iterationFinish
                       << " envelope-finish=" << fold.envelopeFinish
                       << " expansion-depth=" << fold.expansion.size() << "\n";
        for (const ReplaySlice *slice : group) {
          ArrayRef<size_t> aliases;
          StringRef occurrenceSource = "none";
          StringRef occurrenceRoot = "none";
          if (slice->summaryOccurrence) {
            const SummaryOccurrence &occurrence =
                summaryOccurrences[*slice->summaryOccurrence];
            aliases = occurrence.aliases;
            occurrenceSource = entries[occurrence.source].id;
            occurrenceRoot = entries[occurrence.summary->root].id;
          }
          auto resources =
              mappedResources(entryResourceSets[slice->attempt], aliases);
          llvm::errs() << "phys-estimate-schedule abort-overlap-slice retry="
                       << entries[slice->retry].id
                       << " attempt=" << entries[slice->attempt].id
                       << " occurrence=";
          if (slice->summaryOccurrence)
            llvm::errs() << *slice->summaryOccurrence;
          else
            llvm::errs() << "none";
          llvm::errs() << " source=" << occurrenceSource
                       << " root=" << occurrenceRoot << " interval=["
                       << slice->attemptStart << ',' << slice->retryStart
                       << "] resources="
                       << (failed(resources) ? 0 : resources->size())
                       << " aliases=" << aliases.size()
                       << " time-anchor=" << entries[slice->timeAnchor].id
                       << " retry-deps="
                       << entries[slice->retry].dependencies.size() << "\n";
        }
        ++profiledAbortGroups;
      }

      layers.push_back(std::move(layer));
    }

    // Prove how each random layer reaches the baseline makespan.  For an edge
    // u -> v with baseline slack s, a delay x leaving u arrives at v as
    // max(0, x - s).  Along a path those slacks add; at a join the
    // minimum-slack path wins.  Therefore a layer is compactly exact when its
    // maximum delay is entirely hidden by the shortest downstream slack, or
    // when every possible winning retry has a zero-slack path to the makespan.
    // Distinct contributing layers are additive only when they form one
    // zero-slack causal chain.  Partial slack and independent critical paths
    // require a larger joint distribution and fail closed instead of
    // over-counting.
    const size_t sink = entries.size();
    struct SlackEdge {
      size_t event;
      double slack;
    };
    std::vector<SmallVector<SlackEdge, 4>> reverseEdges(entries.size() + 1);
    std::vector<SmallVector<size_t, 4>> zeroSlackEdges(entries.size());
    const double makespan = schedule.getMakespanNs().convertToDouble();
    const double timeScale = std::max(1.0, makespan);
    const double tolerance =
        64.0 * std::numeric_limits<double>::epsilon() * timeScale;
    auto addSlackEdge = [&](size_t source, size_t target, double slack) {
      if (source >= entries.size() || target > sink || slack < -tolerance)
        return;
      slack = std::max(0.0, slack);
      reverseEdges[target].push_back({source, slack});
      if (target < entries.size() && slack <= tolerance)
        zeroSlackEdges[source].push_back(target);
    };
    for (size_t target = 0; target < entries.size(); ++target) {
      for (StringRef dependency : entries[target].dependencies) {
        auto source = byId.find(dependency);
        if (source == byId.end())
          return schedule.emitOpError(
              "retry max-plus dependency evidence is malformed");
        addSlackEdge(source->second, target,
                     entries[target].start - entries[source->second].finish());
      }
      if (entries[target].parent.empty()) {
        addSlackEdge(target, sink, makespan - entries[target].finish());
        continue;
      }
      auto parent = byId.find(entries[target].parent);
      if (parent == byId.end())
        return schedule.emitOpError(
            "retry max-plus hierarchy evidence is malformed");
      // A folded repeat/while contains several dynamic copies of one compact
      // child row.  Its retry multiplicity is already folded into ReplaySlice;
      // treating the unexpanded parent-child timestamp gap as ordinary slack
      // would absorb real per-iteration delay.  Exact critical folded anchors
      // are promoted below instead.
      if (entries[parent->second].kind != "repeat" &&
          entries[parent->second].kind != "while")
        addSlackEdge(target, parent->second,
                     entries[parent->second].finish() -
                         entries[target].finish());
    }

    auto approximatelyEqual = [&](double left, double right) {
      return std::abs(left - right) <= tolerance;
    };
    auto climbActualAnchor = [&](size_t start) -> FailureOr<size_t> {
      size_t current = start;
      llvm::SmallDenseSet<size_t, 8> visited;
      while (!entries[current].parent.empty()) {
        auto parent = byId.find(entries[current].parent);
        if (parent == byId.end() || !visited.insert(parent->second).second)
          return failure();
        const Entry &envelope = entries[parent->second];
        if (envelope.kind == "repeat" || envelope.kind == "while")
          return failure();
        if (!approximatelyEqual(entries[current].finish(), envelope.finish()))
          break;
        current = parent->second;
      }
      return current;
    };
    auto validateVirtualCritical = [&](const ReplaySlice &slice) {
      if (!slice.summaryOccurrence)
        return success();
      const SummaryOccurrence &occurrence =
          summaryOccurrences[*slice.summaryOccurrence];
      size_t current = slice.retry;
      llvm::SmallDenseSet<size_t, 8> visited;
      while (current != occurrence.summary->root) {
        if (entries[current].parent.empty())
          return failure();
        auto parent = byId.find(entries[current].parent);
        if (parent == byId.end() || !visited.insert(parent->second).second)
          return failure();
        const Entry &envelope = entries[parent->second];
        if (envelope.kind == "repeat" || envelope.kind == "while" ||
            !approximatelyEqual(entries[current].finish(), envelope.finish()))
          return failure();
        current = parent->second;
      }
      return success();
    };

    auto updateStatistics = [](RetryTimeLayer &layer) {
      layer.expected = 0.0;
      layer.maximum = 0.0;
      if (!layer.support)
        return;
      for (const auto &[delay, probability] : *layer.support) {
        layer.expected += delay * probability;
        if (probability > 0.0)
          layer.maximum = std::max(layer.maximum, delay);
      }
    };
    auto singletonStatistics = [&](RetryTimeLayer &layer, double target) {
      const ReplaySlice &slice = *layer.slices.front();
      const Entry &retry = entries[slice.retry];
      double slack = target - slice.retryStart;
      if (slack < -tolerance)
        return failure();
      slack = std::max(0.0, slack);
      const int64_t maximumExtra = *retry.maxAttempts - 1;
      layer.maximum = std::max(
          0.0,
          static_cast<double>(maximumExtra) * slice.replayDuration - slack);
      layer.expected = 0.0;
      if (layer.maximum <= tolerance || *retry.successProbability == 1.0)
        return success();
      const int64_t firstContributing =
          static_cast<int64_t>(std::floor(slack / slice.replayDuration)) + 1;
      if (firstContributing > maximumExtra)
        return success();
      const double probability = *retry.successProbability;
      const double failureProbability = 1.0 - probability;
      const double logFailure = std::log1p(-probability);
      auto failurePower = [&](int64_t exponent) {
        if (exponent == 0)
          return 1.0;
        if (exponent == 1)
          return failureProbability;
        return std::exp(static_cast<double>(exponent) * logFailure);
      };
      const double firstTail = failurePower(firstContributing);
      double expected =
          (static_cast<double>(firstContributing) * slice.replayDuration -
           slack) *
          firstTail;
      const int64_t remainingTerms = maximumExtra - firstContributing;
      if (remainingTerms > 0) {
        const double laterTail = failurePower(firstContributing + 1);
        const double geometric =
            -std::expm1(static_cast<double>(remainingTerms) * logFailure) /
            probability;
        expected += slice.replayDuration * laterTail * geometric;
      }
      layer.expected = std::max(0.0, expected);
      return success();
    };
    auto materializeSingleton =
        [&](const ReplaySlice &slice,
            double target) -> FailureOr<ProbabilityMass> {
      const Entry &retry = entries[slice.retry];
      const int64_t supportAttempts =
          *retry.successProbability == 1.0 ? 1 : *retry.maxAttempts;
      if (supportAttempts > kMaximumParallelRetrySupportPoints)
        return failure();
      const double slack = std::max(0.0, target - slice.retryStart);
      const double probability = *retry.successProbability;
      const double failure = 1.0 - probability;
      double surviving = 1.0;
      ProbabilityMass result;
      for (int64_t extra = 0; extra < supportAttempts; ++extra) {
        const double mass =
            extra + 1 == supportAttempts ? surviving : probability * surviving;
        const double delay = std::max(
            0.0, static_cast<double>(extra) * slice.replayDuration - slack);
        result[delay] += mass;
        surviving *= failure;
      }
      return result;
    };
    auto buildParallelSupport = [&](RetryTimeLayer &layer, double target) {
      std::map<double, SmallVector<std::pair<size_t, double>, 2>> points;
      int64_t supportPointCount = 0;
      for (const ReplaySlice *slice : layer.slices) {
        const Entry &retry = entries[slice->retry];
        const int64_t supportAttempts =
            *retry.successProbability == 1.0 ? 1 : *retry.maxAttempts;
        if (supportAttempts >
            kMaximumParallelRetrySupportPoints - supportPointCount)
          return failure();
        supportPointCount += supportAttempts;
      }
      for (size_t variable = 0; variable < layer.slices.size(); ++variable) {
        const ReplaySlice &slice = *layer.slices[variable];
        const Entry &retry = entries[slice.retry];
        const int64_t supportAttempts =
            *retry.successProbability == 1.0 ? 1 : *retry.maxAttempts;
        for (int64_t extra = 0; extra < supportAttempts; ++extra) {
          const double delay = std::max(0.0, slice.retryStart +
                                                 static_cast<double>(extra) *
                                                     slice.replayDuration -
                                                 target);
          points[delay].emplace_back(variable, cdf(*retry.successProbability,
                                                   *retry.maxAttempts, extra));
        }
      }
      std::vector<double> cumulative(layer.slices.size(), 0.0);
      int64_t zeroCount = layer.slices.size();
      double logProduct = 0.0;
      double previousCDF = 0.0;
      ProbabilityMass result;
      for (const auto &[delay, updates] : points) {
        for (const auto &[variable, value] : updates) {
          const double previous = cumulative[variable];
          if (previous == 0.0)
            --zeroCount;
          else
            logProduct -= std::log(previous);
          cumulative[variable] = value;
          logProduct += std::log(value);
        }
        const double maximumCDF = zeroCount ? 0.0 : std::exp(logProduct);
        result[delay] += std::max(0.0, maximumCDF - previousCDF);
        previousCDF = maximumCDF;
      }
      layer.support = std::move(result);
      updateStatistics(layer);
      return success();
    };
    auto convolve =
        [&](const ProbabilityMass &left,
            const ProbabilityMass &right) -> FailureOr<ProbabilityMass> {
      if (!left.empty() &&
          right.size() > kMaximumParallelRetrySupportPoints / left.size())
        return failure();
      ProbabilityMass result;
      for (const auto &[leftDelay, leftProbability] : left)
        for (const auto &[rightDelay, rightProbability] : right) {
          result[leftDelay + rightDelay] += leftProbability * rightProbability;
          if (result.size() > kMaximumParallelRetrySupportPoints)
            return failure();
        }
      return result;
    };
    auto powerSupport = [&](ProbabilityMass support,
                            int64_t copies) -> FailureOr<ProbabilityMass> {
      ProbabilityMass result{{0.0, 1.0}};
      while (copies > 0) {
        if (copies & 1) {
          auto product = convolve(result, support);
          if (failed(product))
            return failure();
          result = std::move(*product);
        }
        copies >>= 1;
        if (copies == 0)
          break;
        auto square = convolve(support, support);
        if (failed(square))
          return failure();
        support = std::move(*square);
      }
      return result;
    };
    auto clipSupport = [](ProbabilityMass support, double slack) {
      ProbabilityMass result;
      for (const auto &[delay, probability] : support)
        result[std::max(0.0, delay - slack)] += probability;
      return result;
    };
    auto foldedCausalSlack =
        [&](size_t source,
            const DynamicFoldIdentity &fold) -> FailureOr<double> {
      if (source == fold.event)
        return 0.0;
      if (fold.event >= entries.size() || source >= entries.size())
        return failure();

      GenerationMarks inside(entries.size());
      SmallVector<size_t, 16> pending(children[fold.event].begin(),
                                      children[fold.event].end());
      while (!pending.empty()) {
        const size_t current = pending.pop_back_val();
        if (!inside.insert(current))
          continue;
        pending.append(children[current]);
      }
      if (!inside.contains(source))
        return failure();

      using Candidate = std::pair<double, size_t>;
      std::priority_queue<Candidate, std::vector<Candidate>,
                          std::greater<Candidate>>
          frontier;
      std::vector<double> distance(entries.size(),
                                   std::numeric_limits<double>::infinity());
      distance[source] = 0.0;
      frontier.emplace(0.0, source);
      double boundary = std::numeric_limits<double>::infinity();
      auto relax = [&](size_t from, size_t to, double slack) {
        if (!inside.contains(to) || slack < -tolerance)
          return;
        const double candidate = distance[from] + std::max(0.0, slack);
        if (candidate >= distance[to])
          return;
        distance[to] = candidate;
        frontier.emplace(candidate, to);
      };
      while (!frontier.empty()) {
        auto [currentDistance, current] = frontier.top();
        frontier.pop();
        if (currentDistance != distance[current] || currentDistance >= boundary)
          continue;
        for (size_t dependent : dependents[current])
          relax(current, dependent,
                entries[dependent].start - entries[current].finish());

        if (entries[current].parent.empty())
          continue;
        auto parent = byId.find(entries[current].parent);
        if (parent == byId.end())
          return failure();
        if (parent->second == fold.event) {
          const double slack = fold.iterationFinish - entries[current].finish();
          if (slack < -tolerance)
            return failure();
          boundary = std::min(boundary, currentDistance + std::max(0.0, slack));
          continue;
        }
        relax(current, parent->second,
              entries[parent->second].finish() - entries[current].finish());
      }
      if (!std::isfinite(boundary))
        return failure();
      return boundary;
    };

    for (RetryTimeLayer &layer : layers) {
      const bool singleton = layer.slices.size() == 1;
      std::optional<double> singletonTarget;
      auto buildBase = [&](double target) -> LogicalResult {
        if (singleton) {
          singletonTarget = target;
          if (failed(singletonStatistics(layer, target)))
            return failure();
          return success();
        }
        return buildParallelSupport(layer, target);
      };
      auto ensureSupport = [&]() -> LogicalResult {
        if (layer.support)
          return success();
        if (!singletonTarget)
          return failure();
        auto support =
            materializeSingleton(*layer.slices.front(), *singletonTarget);
        if (failed(support))
          return failure();
        layer.support = std::move(*support);
        return success();
      };
      auto applyPartialClip = [&](double slack,
                                  int64_t copies) -> LogicalResult {
        if (failed(ensureSupport()))
          return failure();
        if (copies != 1) {
          auto aggregate = powerSupport(std::move(*layer.support), copies);
          if (failed(aggregate))
            return failure();
          layer.support = std::move(*aggregate);
        }
        layer.support = clipSupport(std::move(*layer.support), slack);
        updateStatistics(layer);
        singletonTarget.reset();
        return success();
      };

      if (layer.context.folds.empty()) {
        if (layer.occurrences != 1)
          return schedule.emitOpError(
              "missing evidence: retry multiplicity lacks a folded execution "
              "context");
        double target = 0.0;
        for (const ReplaySlice *slice : layer.slices) {
          if (failed(validateVirtualCritical(*slice)))
            return schedule.emitOpError(
                "missing evidence: templated retry has unresolved internal "
                "critical-path slack");
          auto anchor = climbActualAnchor(slice->timeAnchor);
          if (failed(anchor))
            return schedule.emitOpError(
                "missing evidence: retry folded ancestry is malformed");
          const_cast<ReplaySlice *>(slice)->timeAnchor = *anchor;
          target = std::max(target, entries[*anchor].finish());
        }
        if (failed(buildBase(target)))
          return schedule.emitOpError(
              "missing evidence: exact parallel retry order-statistic support "
              "exceeds the bounded 262144-point estimator limit");
        layer.baseExpected = layer.expected;
        layer.baseMaximum = layer.maximum;
        continue;
      }

      int64_t foldedOccurrences = 1;
      for (const DynamicFoldIdentity &fold : layer.context.folds) {
        if (fold.count <= 0 ||
            foldedOccurrences >
                std::numeric_limits<int64_t>::max() / fold.count)
          return schedule.emitOpError(
              "missing evidence: retry folded multiplicity overflows i64");
        foldedOccurrences *= fold.count;
      }
      if (foldedOccurrences != layer.occurrences)
        return schedule.emitOpError(
            "missing evidence: retry folded execution context does not "
            "explain its multiplicity");

      const DynamicFoldIdentity &inner = layer.context.folds.front();
      double innerTarget = inner.iterationFinish;
      if (singleton) {
        auto causalSlack =
            foldedCausalSlack(layer.slices.front()->timeAnchor, inner);
        if (failed(causalSlack))
          return schedule.emitOpError(
              "missing evidence: folded retry has no exact causal path to "
              "its iteration boundary");
        innerTarget = layer.slices.front()->retryStart + *causalSlack;
      }
      if (failed(buildBase(innerTarget)))
        return schedule.emitOpError(
            "missing evidence: exact folded retry support exceeds the bounded "
            "262144-point estimator limit");
      layer.baseExpected = layer.expected;
      layer.baseMaximum = layer.maximum;
      int64_t pendingCopies = inner.count;
      double previousFinish = inner.envelopeFinish;
      for (const DynamicFoldIdentity &fold :
           ArrayRef<DynamicFoldIdentity>(layer.context.folds).drop_front()) {
        double slack = fold.iterationFinish - previousFinish;
        if (slack < -tolerance)
          return schedule.emitOpError(
              "missing evidence: nested retry fold timing is malformed");
        slack = std::max(0.0, slack);
        const double aggregateMaximum =
            layer.maximum * static_cast<double>(pendingCopies);
        if (aggregateMaximum <= slack + tolerance) {
          layer.expected = 0.0;
          layer.maximum = 0.0;
          layer.support.reset();
          singletonTarget.reset();
          pendingCopies = 1;
        } else if (slack <= tolerance) {
          if (pendingCopies > std::numeric_limits<int64_t>::max() / fold.count)
            return schedule.emitOpError(
                "missing evidence: nested retry fold multiplicity overflows "
                "i64");
          pendingCopies *= fold.count;
          previousFinish = fold.envelopeFinish;
          continue;
        } else {
          ++layer.partialClips;
          if (failed(applyPartialClip(slack, pendingCopies)))
            return schedule.emitOpError(
                "missing evidence: exact nested folded retry support exceeds "
                "the bounded 262144-point estimator limit");
          pendingCopies = fold.count;
        }
        previousFinish = fold.envelopeFinish;
      }

      const DynamicFoldIdentity &outer = layer.context.folds.back();
      size_t anchor = outer.event;
      if (!outer.expansion.empty()) {
        const ReplaySlice *slice = layer.slices.front();
        if (!slice->summaryOccurrence)
          return schedule.emitOpError(
              "missing evidence: virtual folded retry lacks an invocation "
              "anchor");
        anchor = summaryOccurrences[*slice->summaryOccurrence].source;
      }
      auto actualAnchor = climbActualAnchor(anchor);
      if (failed(actualAnchor))
        return schedule.emitOpError(
            "missing evidence: retry folded ancestry is malformed");
      const double postFoldSlack =
          entries[*actualAnchor].finish() - outer.envelopeFinish;
      if (postFoldSlack < -tolerance)
        return schedule.emitOpError(
            "missing evidence: retry folded invocation timing is malformed");
      const double aggregateMaximum =
          layer.maximum * static_cast<double>(pendingCopies);
      if (aggregateMaximum <= postFoldSlack + tolerance) {
        layer.expected = 0.0;
        layer.maximum = 0.0;
      } else if (postFoldSlack > tolerance) {
        ++layer.partialClips;
        if (failed(applyPartialClip(postFoldSlack, pendingCopies)))
          return schedule.emitOpError(
              "missing evidence: exact folded retry support exceeds the "
              "bounded 262144-point estimator limit");
        pendingCopies = 1;
      }
      layer.expected *= static_cast<double>(pendingCopies);
      layer.maximum *= static_cast<double>(pendingCopies);
      for (const ReplaySlice *slice : layer.slices)
        const_cast<ReplaySlice *>(slice)->timeAnchor = *actualAnchor;
    }

    std::vector<double> slackToMakespan(
        entries.size() + 1, std::numeric_limits<double>::infinity());
    using PendingSlack = std::pair<double, size_t>;
    std::priority_queue<PendingSlack, std::vector<PendingSlack>,
                        std::greater<PendingSlack>>
        pending;
    slackToMakespan[sink] = 0.0;
    pending.emplace(0.0, sink);
    while (!pending.empty()) {
      auto [distance, target] = pending.top();
      pending.pop();
      if (distance != slackToMakespan[target])
        continue;
      for (const SlackEdge &edge : reverseEdges[target]) {
        const double candidate = distance + edge.slack;
        if (candidate >= slackToMakespan[edge.event])
          continue;
        slackToMakespan[edge.event] = candidate;
        pending.emplace(candidate, edge.event);
      }
    }

    SmallVector<const RetryTimeLayer *, 4> contributing;
    uint64_t hiddenLayers = 0;
    for (RetryTimeLayer &layer : layers) {
      if (layer.maximum <= tolerance) {
        ++hiddenLayers;
        continue;
      }
      bool hidden = true;
      bool critical = true;
      for (const ReplaySlice *slice : layer.slices) {
        const double slack = slackToMakespan[slice->timeAnchor];
        hidden &= layer.maximum <= slack + tolerance;
        critical &= slack <= tolerance;
      }
      if (hidden) {
        ++hiddenLayers;
        continue;
      }
      if (!critical)
        return schedule.emitOpError(
            "missing evidence: retry delay intersects independent "
            "critical-path slack; exact max-plus support is unavailable");
      layer.contributesToMakespan = true;
      contributing.push_back(&layer);
    }

    GenerationMarks reachable(entries.size());
    for (size_t layerIndex = 1; layerIndex < contributing.size();
         ++layerIndex) {
      const RetryTimeLayer &previous = *contributing[layerIndex - 1];
      const RetryTimeLayer &current = *contributing[layerIndex];
      for (const ReplaySlice *source : previous.slices) {
        reachable.reset();
        SmallVector<size_t, 16> frontier{source->timeAnchor};
        while (!frontier.empty()) {
          size_t event = frontier.pop_back_val();
          if (!reachable.insert(event))
            continue;
          frontier.append(zeroSlackEdges[event]);
        }
        for (const ReplaySlice *target : current.slices)
          if (!reachable.contains(target->timeAnchor))
            return schedule.emitOpError(
                "missing evidence: nonoverlapping retry layers occupy "
                "independent critical paths; exact max-plus support is "
                "unavailable");
      }
    }
    const bool abortAware = llvm::any_of(entries, [](const Entry &entry) {
      return entry.kind == "retry" && entry.exhaustion == "abort";
    });
    for (const RetryTimeLayer &layer : layers) {
      const double perVisitExpected =
          layer.contributesToMakespan ? layer.baseExpected : 0.0;
      const double perVisitMaximum =
          layer.contributesToMakespan ? layer.baseMaximum : 0.0;
      const double effectiveExpected =
          layer.contributesToMakespan ? layer.expected : 0.0;
      const double effectiveMaximum =
          layer.contributesToMakespan ? layer.maximum : 0.0;
      const double expectedLinear =
          perVisitExpected * static_cast<double>(layer.occurrences);
      const double maximumLinear =
          perVisitMaximum * static_cast<double>(layer.occurrences);
      const double scale =
          std::max({1.0, std::abs(effectiveExpected), std::abs(expectedLinear),
                    std::abs(effectiveMaximum), std::abs(maximumLinear)});
      if (abortAware &&
          (layer.partialClips != 0 ||
           std::abs(effectiveExpected - expectedLinear) >
               64.0 * std::numeric_limits<double>::epsilon() * scale ||
           std::abs(effectiveMaximum - maximumLinear) >
               64.0 * std::numeric_limits<double>::epsilon() * scale))
        return schedule.emitOpError(
            "missing evidence: abort-aware retry timing crosses a nonlinear "
            "folded max-plus boundary");
      const size_t layerIndex = stoppedRetryTimeLayers.size();
      StoppedRetryTimeLayer evidence;
      evidence.expectedPerVisit = perVisitExpected;
      evidence.maximumPerVisit = perVisitMaximum;
      for (const ReplaySlice *slice : layer.slices) {
        const size_t sliceIndex =
            static_cast<size_t>(slice - retrySlices.data());
        if (sliceIndex >= retrySlices.size() ||
            stoppedRetryTimeLayerForSlice[sliceIndex] !=
                std::numeric_limits<size_t>::max())
          return schedule.emitOpError(
              "retry stopped-time layer mapping is inconsistent");
        evidence.slices.push_back(sliceIndex);
        stoppedRetryTimeLayerForSlice[sliceIndex] = layerIndex;
      }
      stoppedRetryTimeLayers.push_back(std::move(evidence));
    }
    for (const RetryTimeLayer *layer : contributing) {
      expected += layer->expected;
      maximum += layer->maximum;
    }
    if (std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE"))
      llvm::errs() << "phys-estimate-schedule retry-causality singleton-groups="
                   << singletonGroups << " parallel-groups=" << parallelGroups
                   << " hidden-layers=" << hiddenLayers
                   << " contributing-layers=" << contributing.size()
                   << " owner-propagation-rows=" << ownerPropagationRows
                   << " dependency-edges=" << ownerDependencyEdges
                   << " controlled-events=" << controlledEvents << "\n";
    return success();
  }

  LogicalResult aggregate() {
    const bool profile =
        std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") != nullptr;
    auto checkpoint = [profile, last = std::chrono::steady_clock::now()](
                          StringRef name) mutable {
      const auto now = std::chrono::steady_clock::now();
      if (profile)
        llvm::errs() << "phys-estimate-schedule aggregate-" << name << ' '
                     << std::chrono::duration<double>(now - last).count()
                     << "s\n";
      last = now;
    };
    std::map<std::string, int64_t> counts;
    for (const Entry &entry : entries)
      ++counts[entry.kind.str()];
    eventCounts = std::move(counts);
    checkpoint("counts");
    MetricTotals baseMetrics = reduceMetrics(baseMetricRoot);
    activeResourceTime = baseMetrics.resourceTime;
    activeQubitTime = baseMetrics.qubitTime;
    checkpoint("base-metrics");
    double expectedExtraDuration = 0.0;
    double maximumExtraDuration = 0.0;
    if (failed(retryTimeExtras(expectedExtraDuration, maximumExtraDuration)))
      return failure();
    checkpoint("retry-time");
    double expectedExtraResource = 0.0;
    double maximumExtraResource = 0.0;
    double expectedExtraQubit = 0.0;
    double maximumExtraQubit = 0.0;
    double successLog = 0.0;
    struct RetryMetricResult {
      MetricTotals metrics;
      double expectedReplays = 0.0;
      double maximumReplays = 0.0;
      double successLog = 0.0;
      const char *error = nullptr;
    };
    std::vector<RetryMetricResult> retryMetricResults(retrySlices.size());
    auto nestedReplayOccurrences =
        [&](size_t parentOccurrence,
            size_t event) -> FailureOr<SmallVector<size_t, 8>> {
      if (parentOccurrence >= summaryOccurrences.size() ||
          event >= entries.size() || entries[event].kind != "call_template")
        return failure();
      const SummaryOccurrence &parent = summaryOccurrences[parentOccurrence];

      // Recover the canonical invocation path from the replayed template back
      // to the call summarized by the parent occurrence.  SummaryOccurrence
      // aliases are stored inner-to-outer, while child links are traversed
      // outer-to-inner.
      SmallVector<size_t, 4> innerToOuter{event};
      size_t current = event;
      llvm::SmallDenseSet<size_t, 8> visited;
      while (current != parent.summary->root) {
        if (entries[current].parent.empty())
          return failure();
        auto ancestor = byId.find(entries[current].parent);
        if (ancestor == byId.end() || !visited.insert(ancestor->second).second)
          return failure();
        current = ancestor->second;
        if (current == parent.summary->root)
          break;
        if (entries[current].kind == "call" ||
            entries[current].kind == "call_template")
          innerToOuter.push_back(current);
      }

      size_t selected = parentOccurrence;
      for (size_t invocation : llvm::reverse(innerToOuter)) {
        std::optional<size_t> child;
        for (const auto &[candidateInvocation, candidateOccurrence] :
             summaryOccurrenceChildren[selected]) {
          if (candidateInvocation != invocation)
            continue;
          if (child)
            return failure();
          child = candidateOccurrence;
        }
        if (!child)
          return failure();
        selected = *child;
      }

      const SummaryOccurrence &root = summaryOccurrences[selected];
      auto canonical = byId.find(entries[event].templateEvent);
      if (canonical == byId.end() || root.summary->root != canonical->second ||
          root.source != parent.source || root.topStart != parent.topStart ||
          root.metricFactor <= 0)
        return failure();
      SmallVector<size_t, 8> expectedAliases(innerToOuter.begin(),
                                             innerToOuter.end());
      expectedAliases.append(parent.aliases.begin(), parent.aliases.end());
      if (expectedAliases != root.aliases ||
          root.prefix.size() < parent.prefix.size() ||
          !std::equal(parent.prefix.begin(), parent.prefix.end(),
                      root.prefix.begin()))
        return failure();

      double parentStart = parent.topStart;
      for (double offset : parent.offsets)
        parentStart += offset;
      double rootStart = root.topStart;
      for (double offset : root.offsets)
        rootStart += offset;
      const double expectedStart = parentStart + entries[event].start -
                                   entries[parent.summary->root].start;
      const double timeScale =
          std::max({1.0, std::abs(rootStart), std::abs(expectedStart)});
      if (std::abs(rootStart - expectedStart) >
          64.0 * std::numeric_limits<double>::epsilon() * timeScale)
        return failure();

      SmallVector<size_t, 8> result;
      SmallVector<size_t, 8> pending{selected};
      while (!pending.empty()) {
        const size_t occurrence = pending.pop_back_val();
        result.push_back(occurrence);
        const auto &nested = summaryOccurrenceChildren[occurrence];
        for (auto iterator = nested.rbegin(); iterator != nested.rend();
             ++iterator)
          pending.push_back(iterator->second);
      }
      return result;
    };
    auto reduceRetryMetrics = [&](size_t sliceIndex) -> LogicalResult {
      const ReplaySlice &slice = retrySlices[sliceIndex];
      RetryMetricResult &result = retryMetricResults[sliceIndex];
      const Entry &retry = entries[slice.retry];
      RetryStatistics statistics =
          retryStatistics(*retry.successProbability, *retry.maxAttempts);
      result.expectedReplays = static_cast<double>(slice.occurrences) *
                               statistics.expectedExtraAttempts;
      result.maximumReplays = static_cast<double>(slice.occurrences) *
                              statistics.maximumExtraAttempts;
      MetricNode replayRoot;
      const SummaryOccurrence *virtualOccurrence =
          slice.summaryOccurrence
              ? &summaryOccurrences[*slice.summaryOccurrence]
              : nullptr;
      auto virtualConstraints =
          [&](size_t event) -> FailureOr<SmallVector<Constraint, 8>> {
        assert(slice.summaryOccurrence && virtualOccurrence &&
               "virtual retry constraints require a summary occurrence");
        auto local = constraints(event, virtualOccurrence->summary->root);
        if (failed(local))
          return failure();
        SmallVector<Constraint, 8> result(virtualOccurrence->prefix.begin(),
                                          virtualOccurrence->prefix.end());
        for (const auto &[condition, branch] : *local) {
          const size_t occurrence = *slice.summaryOccurrence;
          const size_t maximum = std::numeric_limits<size_t>::max();
          if (condition > maximum - retryConditionBase)
            return failure();
          const size_t base = retryConditionBase + condition;
          if (entries.size() != 0 &&
              occurrence > (maximum - base) / entries.size())
            return failure();
          result.emplace_back(base + occurrence * entries.size(), branch);
        }
        return result;
      };
      for (const auto &[event, relative] : slice.weighted) {
        if (entries[event].kind == "call_template") {
          SmallVector<size_t, 8> replayOccurrences;
          int64_t replayMetricFactor = multiplicities[event];
          if (virtualOccurrence) {
            auto nested =
                nestedReplayOccurrences(*slice.summaryOccurrence, event);
            if (failed(nested) || nested->empty()) {
              result.error =
                  "templated retry replay nested occurrence mapping is "
                  "missing or ambiguous";
              return success();
            }
            replayOccurrences = std::move(*nested);
            replayMetricFactor =
                summaryOccurrences[replayOccurrences.front()].metricFactor;
          } else {
            const auto [begin, end] = summaryOccurrenceRanges[event];
            replayOccurrences.reserve(end - begin);
            for (size_t occurrenceIndex = begin; occurrenceIndex < end;
                 ++occurrenceIndex)
              replayOccurrences.push_back(occurrenceIndex);
          }
          if (replayMetricFactor <= 0) {
            result.error =
                "retry replay call-template multiplicity is malformed";
            return success();
          }
          for (size_t occurrenceIndex : replayOccurrences) {
            const SummaryOccurrence &occurrence =
                summaryOccurrences[occurrenceIndex];
            if (occurrence.metricFactor % replayMetricFactor) {
              result.error =
                  "retry replay call-template multiplicity is malformed";
              return success();
            }
            const int64_t nestedFactor =
                occurrence.metricFactor / replayMetricFactor;
            if (nestedFactor != 0 &&
                relative > std::numeric_limits<int64_t>::max() / nestedFactor) {
              result.error =
                  "retry replay call-template multiplicity overflows i64";
              return success();
            }
            const int64_t factor = relative * nestedFactor;
            for (auto [position, metric] :
                 llvm::enumerate(occurrence.summary->metrics)) {
              const auto counts = occurrenceMetricCounts
                  [occurrenceMetricOffsets[occurrenceIndex] + position];
              accumulateMetric(replayRoot, metric.kind,
                               metric.weightedDuration *
                                   static_cast<double>(factor),
                               counts.first, counts.second,
                               occurrenceConstraints(occurrence, metric.path));
            }
          }
          continue;
        }
        if (isEnvelope(entries[event].kind) || entries[event].duration <= 0.0)
          continue;
        SmallVector<Constraint, 8> resolvedPath;
        if (virtualOccurrence) {
          auto path = virtualConstraints(event);
          if (failed(path)) {
            result.error = "retry replay conditional ancestry is malformed";
            return success();
          }
          resolvedPath = std::move(*path);
        } else {
          auto path = constraints(event);
          if (failed(path)) {
            result.error = "retry replay conditional ancestry is malformed";
            return success();
          }
          resolvedPath.append(path->begin(), path->end());
        }
        auto baseCounts = mappedResourceCounts(entryResourceSets[event], {});
        if (failed(baseCounts)) {
          result.error = "retry replay resource lacks physical evidence";
          return success();
        }
        size_t resources = baseCounts->first;
        size_t qubits = baseCounts->second;
        if (virtualOccurrence) {
          auto counts = mappedResourceCounts(entryResourceSets[event],
                                             virtualOccurrence->aliases);
          if (failed(counts)) {
            result.error =
                "retry replay resource alias lacks physical evidence";
            return success();
          }
          resources = counts->first;
          qubits = counts->second;
        }
        accumulateMetric(replayRoot, entries[event].kind,
                         entries[event].duration *
                             static_cast<double>(relative),
                         resources, qubits, resolvedPath);
      }
      result.metrics = reduceMetrics(replayRoot);
      if (statistics.completionProbability < 1.0 ||
          statistics.exhaustionProbability > 0.0) {
        const double logCompletion =
            statistics.completionProbability <= 0.5
                ? std::log(statistics.completionProbability)
                : std::log1p(-statistics.exhaustionProbability);
        result.successLog =
            static_cast<double>(slice.occurrences) * logCompletion;
      }
      return success();
    };
    size_t retryMetricShards = 1;
    if (context->isMultithreadingEnabled())
      retryMetricShards = std::min(
          {size_t{64},
           static_cast<size_t>(context->getThreadPool().getMaxConcurrency()),
           std::max<size_t>(1, retrySlices.size())});
    auto reduceRetryShard = [&](size_t shard) -> LogicalResult {
      const size_t begin = retrySlices.size() * shard / retryMetricShards;
      const size_t end = retrySlices.size() * (shard + 1) / retryMetricShards;
      for (size_t index = begin; index < end; ++index)
        if (failed(reduceRetryMetrics(index)))
          return failure();
      return success();
    };
    if (failed(failableParallelForEachN(context, 0, retryMetricShards,
                                        reduceRetryShard)))
      return failure();
    for (const RetryMetricResult &result : retryMetricResults) {
      if (result.error)
        return schedule.emitOpError(result.error);
      expectedExtraResource +=
          result.expectedReplays * result.metrics.resourceTime;
      maximumExtraResource +=
          result.maximumReplays * result.metrics.resourceTime;
      expectedExtraQubit += result.expectedReplays * result.metrics.qubitTime;
      maximumExtraQubit += result.maximumReplays * result.metrics.qubitTime;
      successLog += result.successLog;
    }
    checkpoint("retry-metrics");

    struct StoppedExecutionSummary {
      double logContinuation = 0.0;
      double expectedElapsed = 0.0;
      MetricTotals expectedMetrics;
      SmallVector<size_t, 2> retrySlices;
      std::optional<double> firstAbortCut;
      bool hasRetry = false;
      bool hasAbort = false;
    };
    struct ScheduledExecutionSummary {
      StoppedExecutionSummary execution;
      double start = 0.0;
      double finish = 0.0;
      size_t source = std::numeric_limits<size_t>::max();
    };
    std::optional<StoppedExecutionSummary> abortExpected;
    const bool containsAbort =
        !fullWorkload && llvm::any_of(entries, [](const Entry &entry) {
          return entry.kind == "retry" && entry.exhaustion == "abort";
        });
    if (containsAbort) {
      const double scheduledMakespan =
          schedule.getMakespanNs().convertToDouble();
      const double tolerance = 64.0 * std::numeric_limits<double>::epsilon() *
                               std::max(1.0, scheduledMakespan);
      const size_t noOccurrence = std::numeric_limits<size_t>::max();

      // The retry-time analysis above has already authenticated every
      // overlapping group as one pairwise-overlapping, causally-independent
      // physical layer with a shared folded/conditional context.  Reuse that
      // exact identity for abort reachability instead of regrouping the same
      // slices.  Every attempt in a parallel layer launches before any
      // member's first decision, and therefore before bounded exhaustion is
      // possible.  Without cancellation evidence all members are committed
      // work; later composition gates only launches after the combined stopped
      // atom.
      for (const StoppedRetryTimeLayer &timeLayer : stoppedRetryTimeLayers) {
        ArrayRef<size_t> group = timeLayer.slices;
        if (group.size() == 1)
          continue;
        double latestLaunch = 0.0;
        double earliestDecision = std::numeric_limits<double>::infinity();
        int64_t occurrences = retrySlices[group.front()].occurrences;
        for (size_t sliceIndex : group) {
          const ReplaySlice &slice = retrySlices[sliceIndex];
          latestLaunch = std::max(latestLaunch, slice.attemptStart);
          earliestDecision = std::min(earliestDecision, slice.retryStart);
          if (slice.occurrences != occurrences)
            return schedule.emitOpError(
                "missing evidence: abort parallel launch layer has "
                "inconsistent folded multiplicity");
        }
        if (latestLaunch >= earliestDecision - tolerance)
          return schedule.emitOpError(
              "missing evidence: abort parallel launch layer crosses a "
              "partial-launch boundary without cancellation timing");
      }

      auto continuation = [](const StoppedExecutionSummary &summary) {
        return summary.logContinuation ==
                       -std::numeric_limits<double>::infinity()
                   ? 0.0
                   : std::exp(summary.logContinuation);
      };
      auto hasPositiveMetrics = [](const MetricTotals &metrics) {
        return metrics.resourceTime > 0.0 || metrics.qubitTime > 0.0 ||
               llvm::any_of(metrics.kindTime, [](const auto &entry) {
                 return entry.getValue() > 0.0;
               });
      };
      auto compose = [&](StoppedExecutionSummary left,
                         const StoppedExecutionSummary &right) {
        const double reach = continuation(left);
        left.expectedElapsed += reach * right.expectedElapsed;
        addMetrics(left.expectedMetrics, right.expectedMetrics, reach);
        left.logContinuation += right.logContinuation;
        left.retrySlices.append(right.retrySlices.begin(),
                                right.retrySlices.end());
        if (!left.firstAbortCut)
          left.firstAbortCut = right.firstAbortCut;
        left.hasRetry |= right.hasRetry;
        left.hasAbort |= right.hasAbort;
        return left;
      };
      auto power = [&](const StoppedExecutionSummary &value,
                       int64_t count) -> StoppedExecutionSummary {
        StoppedExecutionSummary result;
        if (count <= 0)
          return result;
        double visits = 0.0;
        if (value.logContinuation == 0.0) {
          visits = static_cast<double>(count);
        } else if (value.logContinuation ==
                   -std::numeric_limits<double>::infinity()) {
          visits = 1.0;
        } else {
          const double denominator = -std::expm1(value.logContinuation);
          visits =
              -std::expm1(static_cast<double>(count) * value.logContinuation) /
              denominator;
        }
        result.logContinuation =
            static_cast<double>(count) * value.logContinuation;
        result.expectedElapsed = visits * value.expectedElapsed;
        result.expectedMetrics = scaledMetrics(value.expectedMetrics, visits);
        result.retrySlices = value.retrySlices;
        result.firstAbortCut = value.firstAbortCut;
        result.hasRetry = value.hasRetry;
        result.hasAbort = value.hasAbort;
        return result;
      };

      std::map<std::pair<size_t, size_t>, size_t> retryResultByContext;
      for (size_t index = 0; index < retrySlices.size(); ++index) {
        const ReplaySlice &slice = retrySlices[index];
        const size_t occurrence =
            slice.summaryOccurrence.value_or(noOccurrence);
        if (!retryResultByContext
                 .try_emplace(std::make_pair(slice.retry, occurrence), index)
                 .second)
          return schedule.emitOpError(
              "missing evidence: abort retry occurrence mapping is "
              "ambiguous");
      }
      std::vector<size_t> occurrenceParents(summaryOccurrences.size(),
                                            noOccurrence);
      for (size_t parent = 0; parent < summaryOccurrenceChildren.size();
           ++parent)
        for (const auto &[invocation, child] :
             summaryOccurrenceChildren[parent]) {
          (void)invocation;
          if (child >= summaryOccurrences.size() || child <= parent ||
              occurrenceParents[child] != noOccurrence)
            return schedule.emitOpError(
                "missing evidence: abort call occurrence hierarchy is "
                "malformed");
          occurrenceParents[child] = parent;
        }
      std::vector<char> occurrenceHasRetry(summaryOccurrences.size(), false);
      for (const ReplaySlice &slice : retrySlices)
        if (slice.summaryOccurrence) {
          if (*slice.summaryOccurrence >= occurrenceHasRetry.size())
            return schedule.emitOpError(
                "missing evidence: abort retry occurrence is out of range");
          occurrenceHasRetry[*slice.summaryOccurrence] = true;
        }
      for (size_t occurrence = occurrenceHasRetry.size(); occurrence-- > 0;)
        if (occurrenceHasRetry[occurrence] &&
            occurrenceParents[occurrence] != noOccurrence)
          occurrenceHasRetry[occurrenceParents[occurrence]] = true;
      checkpoint("abort-index");
      auto occurrenceStart = [&](size_t occurrence) -> FailureOr<double> {
        if (occurrence >= summaryOccurrences.size())
          return failure();
        double result = summaryOccurrences[occurrence].topStart;
        for (double offset : summaryOccurrences[occurrence].offsets)
          result += offset;
        return result;
      };
      auto nestedOccurrence = [&](size_t parent,
                                  size_t invocation) -> FailureOr<size_t> {
        if (parent >= summaryOccurrenceChildren.size())
          return failure();
        std::optional<size_t> result;
        for (const auto &[candidate, occurrence] :
             summaryOccurrenceChildren[parent]) {
          if (candidate != invocation)
            continue;
          if (result)
            return failure();
          result = occurrence;
        }
        if (!result)
          return failure();
        return *result;
      };
      auto retryFreeOccurrenceMetrics =
          [&](size_t rootOccurrence) -> FailureOr<MetricTotals> {
        if (rootOccurrence >= summaryOccurrences.size())
          return failure();
        const int64_t rootFactor =
            summaryOccurrences[rootOccurrence].metricFactor;
        if (rootFactor <= 0)
          return failure();
        MetricNode metricRoot;
        SmallVector<size_t, 8> pending{rootOccurrence};
        while (!pending.empty()) {
          const size_t occurrenceIndex = pending.pop_back_val();
          const SummaryOccurrence &occurrence =
              summaryOccurrences[occurrenceIndex];
          if (occurrence.metricFactor <= 0 ||
              occurrence.metricFactor % rootFactor)
            return failure();
          const double factor =
              static_cast<double>(occurrence.metricFactor / rootFactor);
          for (auto [position, metric] :
               llvm::enumerate(occurrence.summary->metrics)) {
            const auto counts = occurrenceMetricCounts
                [occurrenceMetricOffsets[occurrenceIndex] + position];
            accumulateMetric(metricRoot, metric.kind,
                             metric.weightedDuration * factor, counts.first,
                             counts.second,
                             occurrenceConstraints(occurrence, metric.path));
          }
          for (const auto &[invocation, child] :
               summaryOccurrenceChildren[occurrenceIndex]) {
            (void)invocation;
            pending.push_back(child);
          }
        }
        return reduceMetrics(metricRoot);
      };
      std::function<FailureOr<ScheduledExecutionSummary>(
          size_t, std::optional<size_t>, double)>
          summarizeNode;
      std::function<FailureOr<StoppedExecutionSummary>(
          ArrayRef<size_t>, double, double, std::optional<size_t>, double)>
          summarizeSequence;

      summarizeSequence =
          [&](ArrayRef<size_t> indices, double regionStart, double regionFinish,
              std::optional<size_t> occurrence,
              double shift) -> FailureOr<StoppedExecutionSummary> {
        SmallVector<ScheduledExecutionSummary, 8> units;
        units.reserve(indices.size());
        for (size_t index : indices) {
          const Entry &entry = entries[index];
          std::optional<size_t> childOccurrence = occurrence;
          size_t source = index;
          if (entry.kind == "call") {
            if (occurrence) {
              auto nested = nestedOccurrence(*occurrence, index);
              if (failed(nested))
                return schedule.emitOpError(
                    "missing evidence: abort call occurrence mapping is "
                    "missing or ambiguous");
              childOccurrence = *nested;
            } else {
              childOccurrence.reset();
            }
          } else if (entry.kind == "call_template") {
            auto canonical = byId.find(entry.templateEvent);
            if (canonical == byId.end() ||
                entries[canonical->second].kind != "call")
              return schedule.emitOpError(
                  "missing evidence: abort call-template canonical mapping "
                  "is malformed");
            source = canonical->second;
            if (occurrence) {
              auto nested = nestedOccurrence(*occurrence, index);
              if (failed(nested))
                return schedule.emitOpError(
                    "missing evidence: abort nested call-template occurrence "
                    "mapping is missing or ambiguous");
              childOccurrence = *nested;
            } else {
              const auto [begin, end] = summaryOccurrenceRanges[index];
              if (begin >= end || begin >= summaryOccurrences.size())
                return schedule.emitOpError(
                    "missing evidence: abort call-template occurrence "
                    "mapping is missing");
              childOccurrence = begin;
            }
          }
          auto unit = summarizeNode(source, childOccurrence, shift);
          if (failed(unit))
            return failure();
          units.push_back(std::move(*unit));
        }

        // Merge authenticated sibling retry invocations that were all
        // launched before any member could decide exhaustion.  They are
        // committed non-cancelling work, so metrics add, continuation is the
        // product, and elapsed time is the exact maximum of their bounded
        // completion supports.  A group not wholly visible at this nesting
        // level is retained for its enclosing call/repeat sequence.
        std::vector<char> consumed(units.size(), false);
        llvm::SmallDenseSet<size_t, 8> seenTimeLayers;
        SmallVector<size_t, 4> activeTimeLayers;
        for (const ScheduledExecutionSummary &unit : units)
          for (size_t sliceIndex : unit.execution.retrySlices) {
            if (sliceIndex >= stoppedRetryTimeLayerForSlice.size())
              return schedule.emitOpError(
                  "missing evidence: abort retry lacks stopped-time layer "
                  "mapping");
            const size_t timeLayer = stoppedRetryTimeLayerForSlice[sliceIndex];
            if (timeLayer >= stoppedRetryTimeLayers.size())
              return schedule.emitOpError(
                  "missing evidence: abort retry lacks stopped-time layer "
                  "evidence");
            if (stoppedRetryTimeLayers[timeLayer].slices.size() > 1 &&
                seenTimeLayers.insert(timeLayer).second)
              activeTimeLayers.push_back(timeLayer);
          }
        for (size_t timeLayer : activeTimeLayers) {
          const StoppedRetryTimeLayer &timing =
              stoppedRetryTimeLayers[timeLayer];
          ArrayRef<size_t> group = timing.slices;
          llvm::SmallDenseSet<size_t, 8> members(group.begin(), group.end());
          llvm::SmallDenseSet<size_t, 8> found;
          SmallVector<size_t, 4> groupUnits;
          for (size_t unitIndex = 0; unitIndex < units.size(); ++unitIndex) {
            bool containsMember = false;
            for (size_t sliceIndex : units[unitIndex].execution.retrySlices) {
              if (!members.contains(sliceIndex))
                continue;
              if (!found.insert(sliceIndex).second)
                return schedule.emitOpError(
                    "missing evidence: abort parallel retry occurrence was "
                    "summarized more than once");
              containsMember = true;
            }
            if (containsMember)
              groupUnits.push_back(unitIndex);
          }
          if (found.empty() || found.size() != members.size() ||
              groupUnits.size() <= 1)
            continue;

          // The bounded maximum below advances each member from its retry
          // decision by whole replay durations.  That is exact only when the
          // retry is terminal in the summarized sibling unit.  A later suffix
          // may already have launched before another member aborts, or may be
          // gated by that abort; the schedule carries no cancellation timing
          // that distinguishes those cases.
          for (size_t sliceIndex : group) {
            std::optional<size_t> owner;
            for (size_t unitIndex : groupUnits) {
              if (!llvm::is_contained(units[unitIndex].execution.retrySlices,
                                      sliceIndex))
                continue;
              if (owner)
                return schedule.emitOpError(
                    "missing evidence: abort parallel retry occurrence has "
                    "multiple summarized owners");
              owner = unitIndex;
            }
            if (!owner)
              return schedule.emitOpError(
                  "missing evidence: abort parallel retry occurrence lacks "
                  "a summarized owner");
            if (std::abs(units[*owner].finish -
                         retrySlices[sliceIndex].retryStart) > tolerance)
              return schedule.emitOpError(
                  "missing evidence: abort parallel launch layer has "
                  "post-retry work without authenticated cancellation "
                  "timing");
          }

          double launchStart = std::numeric_limits<double>::infinity();
          double launchBoundary = std::numeric_limits<double>::infinity();
          for (size_t sliceIndex : group) {
            launchStart =
                std::min(launchStart, retrySlices[sliceIndex].attemptStart);
            launchBoundary =
                std::min(launchBoundary, retrySlices[sliceIndex].retryStart);
          }
          std::vector<char> selected(units.size(), false);
          for (size_t unitIndex : groupUnits)
            selected[unitIndex] = true;
          for (size_t unitIndex = 0; unitIndex < units.size(); ++unitIndex) {
            const ScheduledExecutionSummary &unit = units[unitIndex];
            if (unit.finish <= launchStart + tolerance ||
                unit.start >= launchBoundary - tolerance)
              continue;
            for (size_t sliceIndex : unit.execution.retrySlices)
              if (!members.contains(sliceIndex))
                return schedule.emitOpError(
                    "missing evidence: abort parallel launch layer crosses "
                    "a partial-launch retry boundary without cancellation "
                    "timing");
            if (!selected[unitIndex]) {
              selected[unitIndex] = true;
              groupUnits.push_back(unitIndex);
            }
          }
          llvm::sort(groupUnits);

          StoppedExecutionSummary combined;
          double combinedStart = launchStart;
          double combinedFinish = 0.0;
          for (size_t unitIndex : groupUnits) {
            const ScheduledExecutionSummary &unit = units[unitIndex];
            for (size_t sliceIndex : unit.execution.retrySlices)
              if (!members.contains(sliceIndex))
                return schedule.emitOpError(
                    "missing evidence: abort parallel launch layer crosses "
                    "an unresolved nested retry boundary");
            combinedStart = std::min(combinedStart, unit.start);
            combinedFinish = std::max(combinedFinish, unit.finish);
            combined.logContinuation += unit.execution.logContinuation;
            addMetrics(combined.expectedMetrics,
                       unit.execution.expectedMetrics);
            combined.retrySlices.append(unit.execution.retrySlices.begin(),
                                        unit.execution.retrySlices.end());
            if (unit.execution.firstAbortCut &&
                (!combined.firstAbortCut ||
                 *unit.execution.firstAbortCut < *combined.firstAbortCut))
              combined.firstAbortCut = unit.execution.firstAbortCut;
            combined.hasRetry |= unit.execution.hasRetry;
            combined.hasAbort |= unit.execution.hasAbort;
          }
          for (size_t unitIndex = 0; unitIndex < units.size(); ++unitIndex) {
            const ScheduledExecutionSummary &unit = units[unitIndex];
            if (selected[unitIndex] ||
                unit.start < launchBoundary - tolerance ||
                unit.start >= combinedFinish - tolerance)
              continue;
            if (unit.execution.hasRetry ||
                hasPositiveMetrics(unit.execution.expectedMetrics))
              return schedule.emitOpError(
                  "missing evidence: abort parallel launch layer crosses "
                  "an intermediate work-launch boundary without "
                  "cancellation timing");
          }
          combined.expectedElapsed =
              combinedFinish - combinedStart + timing.expectedPerVisit;
          const size_t retained = groupUnits.front();
          units[retained] = ScheduledExecutionSummary{
              std::move(combined), combinedStart, combinedFinish};
          for (size_t unitIndex : ArrayRef<size_t>(groupUnits).drop_front())
            consumed[unitIndex] = true;
        }
        if (llvm::is_contained(consumed, true)) {
          SmallVector<ScheduledExecutionSummary, 8> retained;
          retained.reserve(units.size());
          for (size_t index = 0; index < units.size(); ++index)
            if (!consumed[index])
              retained.push_back(std::move(units[index]));
          llvm::stable_sort(retained,
                            [](const ScheduledExecutionSummary &left,
                               const ScheduledExecutionSummary &right) {
                              if (left.start != right.start)
                                return left.start < right.start;
                              return left.finish < right.finish;
                            });
          units = std::move(retained);
        }

        MetricTotals allMetrics;
        bool hasRetry = false;
        for (const ScheduledExecutionSummary &unit : units) {
          addMetrics(allMetrics, unit.execution.expectedMetrics);
          hasRetry |= unit.execution.hasRetry;
        }
        if (!hasRetry) {
          StoppedExecutionSummary result;
          result.expectedElapsed = std::max(0.0, regionFinish - regionStart);
          result.expectedMetrics = std::move(allMetrics);
          return result;
        }

        StoppedExecutionSummary result;
        MetricTotals pendingMetrics;
        double cursor = regionStart;
        double pendingFinish = regionStart;
        size_t pendingSource = std::numeric_limits<size_t>::max();
        bool pending = false;
        bool afterRandom = false;
        auto reportCrossing = [&](StringRef site, size_t unitIndex,
                                  double boundary) {
          if (!profile)
            return;
          llvm::errs() << "phys-estimate-schedule abort-crossing site=" << site
                       << " region=[" << regionStart << ',' << regionFinish
                       << "] cursor=" << cursor
                       << " pending-finish=" << pendingFinish
                       << " boundary=" << boundary << " unit=" << unitIndex
                       << '/' << units.size();
          if (unitIndex < units.size())
            llvm::errs() << " unit-interval=[" << units[unitIndex].start << ','
                         << units[unitIndex].finish << "] unit-retry="
                         << units[unitIndex].execution.hasRetry
                         << " unit-abort="
                         << units[unitIndex].execution.hasAbort << " source="
                         << (units[unitIndex].source < entries.size()
                                 ? entries[units[unitIndex].source].id
                                 : StringRef("combined"))
                         << " pending-delta="
                         << llvm::formatv("{0:f17}", pendingFinish - boundary)
                         << " pending-source="
                         << (pendingSource < entries.size()
                                 ? entries[pendingSource].id
                                 : StringRef("none"));
          llvm::errs() << '\n';
        };
        auto flush = [&](double boundary) -> LogicalResult {
          if (pendingFinish > boundary + tolerance) {
            reportCrossing("flush", units.size(), boundary);
            return schedule.emitOpError(
                "missing evidence: independent scheduled work crosses an "
                "abort continuation cut");
          }
          if (afterRandom && pending && pendingFinish < boundary - tolerance)
            return schedule.emitOpError(
                "missing evidence: abort continuation has unresolved "
                "max-plus slack");
          if (afterRandom && !pending && boundary > cursor + tolerance)
            return schedule.emitOpError(
                "missing evidence: abort continuation has unresolved "
                "max-plus slack");
          StoppedExecutionSummary deterministic;
          deterministic.expectedElapsed = std::max(0.0, boundary - cursor);
          deterministic.expectedMetrics = std::move(pendingMetrics);
          result = compose(std::move(result), deterministic);
          pendingMetrics = MetricTotals{};
          pendingFinish = boundary;
          pendingSource = std::numeric_limits<size_t>::max();
          pending = false;
          cursor = boundary;
          return success();
        };

        for (auto [unitIndex, unit] : llvm::enumerate(units)) {
          if (!unit.execution.hasRetry) {
            if (afterRandom && unit.start < cursor - tolerance &&
                unit.finish <= cursor + tolerance) {
              StoppedExecutionSummary hidden;
              hidden.expectedMetrics = unit.execution.expectedMetrics;
              result = compose(std::move(result), hidden);
              continue;
            }
            if (unit.start < cursor - tolerance) {
              reportCrossing("deterministic", unitIndex, unit.start);
              return schedule.emitOpError(
                  "missing evidence: independent scheduled work crosses an "
                  "abort continuation cut");
            }
            if (afterRandom && !pending && unit.start > cursor + tolerance)
              return schedule.emitOpError(
                  "missing evidence: abort continuation has unresolved "
                  "max-plus slack");
            if (afterRandom && pending &&
                unit.start > pendingFinish + tolerance)
              return schedule.emitOpError(
                  "missing evidence: abort continuation has unresolved "
                  "max-plus slack");
            addMetrics(pendingMetrics, unit.execution.expectedMetrics);
            if (unit.finish >= pendingFinish) {
              pendingFinish = unit.finish;
              pendingSource = unit.source;
            }
            pending = true;
            continue;
          }

          if (unit.start < cursor - tolerance ||
              pendingFinish > unit.start + tolerance) {
            const bool committedBeforeAbort =
                unit.start >= cursor - tolerance && pending &&
                unit.execution.firstAbortCut &&
                pendingFinish <= *unit.execution.firstAbortCut + tolerance;
            if (!committedBeforeAbort) {
              reportCrossing("retry", unitIndex, unit.start);
              return schedule.emitOpError(
                  "missing evidence: independent scheduled work crosses an "
                  "abort continuation cut");
            }

            // This work was launched before the retry-containing unit and the
            // unit's authenticated inner abort boundary waits until it has
            // finished.  Its metrics are therefore unconditional committed
            // work, while its tail is hidden beneath the unit's pre-abort
            // prefix.  Close only the elapsed prefix at the unit launch; the
            // complete committed metrics remain before the retry continuation
            // factor.
            pendingFinish = unit.start;
          }
          if (failed(flush(unit.start)))
            return failure();
          result = compose(std::move(result), unit.execution);
          cursor = unit.finish;
          pendingFinish = cursor;
          afterRandom = true;
        }
        if (failed(flush(regionFinish)))
          return failure();
        return result;
      };

      summarizeNode =
          [&](size_t index, std::optional<size_t> occurrence,
              double inheritedShift) -> FailureOr<ScheduledExecutionSummary> {
        if (index >= entries.size())
          return failure();
        const Entry &entry = entries[index];
        double start = entry.start + inheritedShift;
        double shift = inheritedShift;
        if (entry.kind == "call" && occurrence) {
          auto mappedStart = occurrenceStart(*occurrence);
          if (failed(mappedStart) ||
              summaryOccurrences[*occurrence].summary->root != index)
            return schedule.emitOpError(
                "missing evidence: abort call occurrence has inconsistent "
                "canonical timing");
          start = *mappedStart;
          shift = start - entry.start;
        }
        const double finish = start + entry.duration;

        if (entry.kind == "retry") {
          const size_t occurrenceKey = occurrence.value_or(noOccurrence);
          auto retained =
              retryResultByContext.find(std::make_pair(index, occurrenceKey));
          if (retained == retryResultByContext.end())
            return schedule.emitOpError(
                "missing evidence: abort retry occurrence lacks an "
                "authenticated replay slice");
          const ReplaySlice &slice = retrySlices[retained->second];
          const RetryMetricResult &metrics =
              retryMetricResults[retained->second];
          RetryStatistics statistics =
              retryStatistics(*entry.successProbability, *entry.maxAttempts);
          StoppedExecutionSummary result;
          if (retained->second >= stoppedRetryTimeLayerForSlice.size())
            return schedule.emitOpError(
                "missing evidence: abort retry lacks stopped-time layer "
                "mapping");
          const size_t timeLayer =
              stoppedRetryTimeLayerForSlice[retained->second];
          if (timeLayer >= stoppedRetryTimeLayers.size())
            return schedule.emitOpError(
                "missing evidence: abort retry lacks stopped-time layer "
                "evidence");
          const StoppedRetryTimeLayer &timing =
              stoppedRetryTimeLayers[timeLayer];
          result.expectedElapsed =
              timing.slices.size() == 1 ? timing.expectedPerVisit : 0.0;
          result.expectedMetrics =
              scaledMetrics(metrics.metrics, statistics.expectedExtraAttempts);
          result.retrySlices.push_back(retained->second);
          result.hasRetry = true;
          result.hasAbort = entry.exhaustion == "abort";
          if (result.hasAbort) {
            result.firstAbortCut = start;
            if (statistics.completionProbability <= 0.0) {
              result.logContinuation = -std::numeric_limits<double>::infinity();
            } else {
              result.logContinuation =
                  statistics.completionProbability <= 0.5
                      ? std::log(statistics.completionProbability)
                      : std::log1p(-statistics.exhaustionProbability);
            }
          }
          return ScheduledExecutionSummary{std::move(result), start, finish,
                                           index};
        }

        if (entry.kind == "call") {
          if (occurrence && !occurrenceHasRetry[*occurrence]) {
            auto metrics = retryFreeOccurrenceMetrics(*occurrence);
            if (failed(metrics))
              return schedule.emitOpError(
                  "missing evidence: abort retry-free call summary metrics "
                  "are malformed");
            StoppedExecutionSummary result;
            result.expectedElapsed = entry.duration;
            result.expectedMetrics = std::move(*metrics);
            return ScheduledExecutionSummary{std::move(result), start, finish,
                                             index};
          }
          auto body = summarizeSequence(children[index], start, finish,
                                        occurrence, shift);
          if (failed(body))
            return failure();
          return ScheduledExecutionSummary{std::move(*body), start, finish,
                                           index};
        }

        if (entry.kind == "repeat") {
          if (!entry.repeatCount)
            return schedule.emitOpError(
                "missing evidence: abort repeat lacks a static count");
          SmallVector<size_t, 8> bodyChildren;
          double bodyFinish = start;
          for (size_t child : children[index]) {
            if (entries[child].branch != "body")
              return schedule.emitOpError(
                  "missing evidence: abort repeat body mapping is malformed");
            bodyChildren.push_back(child);
            bodyFinish = std::max(bodyFinish, entries[child].finish() + shift);
          }
          auto body = summarizeSequence(bodyChildren, start, bodyFinish,
                                        occurrence, shift);
          if (failed(body))
            return failure();
          const double expectedBaseline =
              static_cast<double>(*entry.repeatCount) * (bodyFinish - start);
          if (std::abs(entry.duration - expectedBaseline) > tolerance)
            return schedule.emitOpError(
                "missing evidence: abort repeat timing is not a compact "
                "serial fold");
          auto result = power(*body, *entry.repeatCount);
          return ScheduledExecutionSummary{std::move(result), start, finish,
                                           index};
        }

        if (entry.kind == "while") {
          if (!entry.maxIterations)
            return schedule.emitOpError(
                "missing evidence: abort while lacks a static bound");
          SmallVector<size_t, 8> conditionChildren;
          SmallVector<size_t, 8> bodyChildren;
          double conditionFinish = start;
          double bodyFinish = start;
          for (size_t child : children[index]) {
            if (entries[child].branch == "condition") {
              conditionChildren.push_back(child);
              conditionFinish =
                  std::max(conditionFinish, entries[child].finish() + shift);
            } else if (entries[child].branch == "body") {
              bodyChildren.push_back(child);
              bodyFinish =
                  std::max(bodyFinish, entries[child].finish() + shift);
            } else {
              return schedule.emitOpError(
                  "missing evidence: abort while branch mapping is malformed");
            }
          }
          bodyFinish = std::max(bodyFinish, conditionFinish);
          auto condition = summarizeSequence(
              conditionChildren, start, conditionFinish, occurrence, shift);
          auto body = summarizeSequence(bodyChildren, conditionFinish,
                                        bodyFinish, occurrence, shift);
          if (failed(condition) || failed(body))
            return failure();
          StoppedExecutionSummary iteration = compose(*condition, *body);
          StoppedExecutionSummary result =
              power(iteration, *entry.maxIterations);
          result = compose(std::move(result), *condition);
          const double expectedBaseline =
              (static_cast<double>(*entry.maxIterations) + 1.0) *
                  (conditionFinish - start) +
              static_cast<double>(*entry.maxIterations) *
                  (bodyFinish - conditionFinish);
          if (std::abs(entry.duration - expectedBaseline) > tolerance)
            return schedule.emitOpError(
                "missing evidence: abort while timing is not a compact "
                "serial fold");
          return ScheduledExecutionSummary{std::move(result), start, finish,
                                           index};
        }

        if (entry.kind == "if" || entry.kind == "try_take") {
          std::map<StringRef, SmallVector<size_t, 8>> branches;
          for (size_t child : children[index])
            branches[entries[child].branch].push_back(child);
          MetricTotals maximum;
          for (const auto &[branch, branchChildren] : branches) {
            (void)branch;
            double branchFinish = start;
            for (size_t child : branchChildren)
              branchFinish =
                  std::max(branchFinish, entries[child].finish() + shift);
            auto value = summarizeSequence(branchChildren, start, branchFinish,
                                           occurrence, shift);
            if (failed(value))
              return failure();
            if (value->hasRetry)
              return schedule.emitOpError(
                  "missing evidence: abort-aware expectation cannot compose "
                  "retry beneath conditional control without authenticated "
                  "branch probabilities");
            maximizeMetrics(maximum, value->expectedMetrics);
          }
          StoppedExecutionSummary result;
          result.expectedElapsed = entry.duration;
          result.expectedMetrics = std::move(maximum);
          return ScheduledExecutionSummary{std::move(result), start, finish,
                                           index};
        }

        StoppedExecutionSummary result;
        result.expectedElapsed = entry.duration;
        if (!isEnvelope(entry.kind) && entry.duration > 0.0) {
          ArrayRef<size_t> aliases;
          if (occurrence)
            aliases = summaryOccurrences[*occurrence].aliases;
          auto counts = mappedResourceCounts(entryResourceSets[index], aliases);
          if (failed(counts))
            return schedule.emitOpError(
                "missing evidence: abort expected metric resource alias "
                "lacks physical evidence");
          result.expectedMetrics.resourceTime =
              entry.duration * static_cast<double>(counts->first);
          result.expectedMetrics.qubitTime =
              entry.duration * static_cast<double>(counts->second);
          result.expectedMetrics.kindTime[entry.kind] = entry.duration;
        }
        return ScheduledExecutionSummary{std::move(result), start, finish,
                                         index};
      };

      SmallVector<size_t, 16> roots;
      for (size_t index = 0; index < entries.size(); ++index)
        if (entries[index].parent.empty())
          roots.push_back(index);
      auto summary =
          summarizeSequence(roots, 0.0, scheduledMakespan, std::nullopt, 0.0);
      if (failed(summary))
        return failure();
      abortExpected = std::move(*summary);
      checkpoint("abort-summary");
    }
    checkpoint("retry-resources");
    auto peakValues = peaks();
    if (failed(peakValues))
      return schedule.emitOpError(
          "schedule peak occupancy state became inconsistent");
    auto [concurrency, qubitPeak] = *peakValues;
    peakConcurrency = concurrency;
    peakActiveQubits = qubitPeak;
    checkpoint("peak-join");

    makespan = schedule.getMakespanNs().convertToDouble();
    expectedMakespan = abortExpected ? abortExpected->expectedElapsed
                                     : makespan + expectedExtraDuration;
    maximumMakespan = makespan + maximumExtraDuration;
    expectedResourceTime = abortExpected
                               ? abortExpected->expectedMetrics.resourceTime
                               : activeResourceTime + expectedExtraResource;
    maximumResourceTime = activeResourceTime + maximumExtraResource;
    expectedQubitTime = abortExpected ? abortExpected->expectedMetrics.qubitTime
                                      : activeQubitTime + expectedExtraQubit;
    maximumQubitTime = activeQubitTime + maximumExtraQubit;
    exhaustionProbability = successLog == 0.0 ? 0.0 : -std::expm1(successLog);

    auto enforceMetricFamily = [&](StringRef name, double first,
                                   double &expected,
                                   double &maximum) -> LogicalResult {
      if (!std::isfinite(first) || !std::isfinite(expected) ||
          !std::isfinite(maximum) || first < 0.0 || expected < 0.0 ||
          maximum < 0.0)
        return schedule.emitOpError("schedule estimate produced invalid ")
               << name << " evidence";
      const double scale = std::max({1.0, first, expected, maximum});
      const double tolerance =
          64.0 * std::numeric_limits<double>::epsilon() * scale;
      if (maximum + tolerance < first || expected > maximum + tolerance)
        return schedule.emitOpError(
                   "schedule estimate violates first <= maximum or expected "
                   "<= maximum for ")
               << name;
      if (maximum < first)
        maximum = first;
      expected = std::clamp(expected, 0.0, maximum);
      return success();
    };
    if (failed(enforceMetricFamily("makespan", makespan, expectedMakespan,
                                   maximumMakespan)) ||
        failed(enforceMetricFamily("active resource-time", activeResourceTime,
                                   expectedResourceTime,
                                   maximumResourceTime)) ||
        failed(enforceMetricFamily("active physical-qubit-time",
                                   activeQubitTime, expectedQubitTime,
                                   maximumQubitTime)))
      return failure();

    double capacity = makespan * static_cast<double>(allResources.size());
    utilization = capacity == 0.0 ? 0.0 : activeResourceTime / capacity;
    double expectedCapacity =
        expectedMakespan * static_cast<double>(allResources.size());
    expectedUtilization =
        expectedCapacity == 0.0 ? 0.0 : expectedResourceTime / expectedCapacity;
    double maximumCapacity =
        maximumMakespan * static_cast<double>(allResources.size());
    maximumUtilization =
        maximumCapacity == 0.0 ? 0.0 : maximumResourceTime / maximumCapacity;

    auto enforceUtilization = [&](StringRef name,
                                  double &value) -> LogicalResult {
      constexpr double tolerance =
          64.0 * std::numeric_limits<double>::epsilon();
      if (!std::isfinite(value) || value < 0.0 || value > 1.0 + tolerance)
        return schedule.emitOpError("schedule estimate produced invalid ")
               << name << " utilization " << value
               << " outside [0, 1] (maximum_resource_time="
               << maximumResourceTime
               << ", maximum_capacity=" << maximumCapacity
               << ", maximum_makespan=" << maximumMakespan
               << ", resource_count=" << allResources.size() << ")";
      value = std::clamp(value, 0.0, 1.0);
      return success();
    };
    if (failed(enforceUtilization("first-attempt", utilization)) ||
        failed(enforceUtilization("expected", expectedUtilization)) ||
        failed(enforceUtilization("maximum", maximumUtilization)))
      return failure();

    bottleneck = entries.empty() ? "empty" : "structural";
    double maximumDuration = -1.0;
    for (const auto &[kind, count] : eventCounts) {
      (void)count;
      if (isEnvelope(kind))
        continue;
      auto retained = baseMetrics.kindTime.find(kind);
      double duration =
          retained == baseMetrics.kindTime.end() ? 0.0 : retained->second;
      if (duration > maximumDuration) {
        maximumDuration = duration;
        bottleneck = kind;
      }
    }
    // Preserve the typed resource identity when every dynamic event in the
    // dominant family is serialized by the same resource. Factory starts are
    // the important case: the model identity is more useful than the generic
    // operation family, while a family spread over several patches remains an
    // operation-kind bottleneck.
    std::optional<StringRef> dominantResource;
    bool severalDominantResources = false;
    for (auto [index, entry] : llvm::enumerate(entries)) {
      if (entry.kind != bottleneck || entry.duration <= 0.0 ||
          multiplicities[index] == 0)
        continue;
      auto consider = [&](StringRef resource) {
        if (!dominantResource)
          dominantResource = resource;
        else if (*dominantResource != resource)
          severalDominantResources = true;
      };
      if (!entry.nativeResourceIds.empty()) {
        for (NativeScheduleResourceId resource : entry.nativeResourceIds)
          consider(nativeSchedule->resourceLabels[resource]);
      } else {
        for (StringRef resource : entry.resources)
          consider(resource);
      }
    }
    if (dominantResource && !severalDominantResources)
      bottleneck = dominantResource->str();
    checkpoint("bottleneck");
    auto finite = [](double value) {
      return std::isfinite(value) && value >= 0;
    };
    for (double value :
         {makespan, expectedMakespan, maximumMakespan, activeResourceTime,
          expectedResourceTime, maximumResourceTime, activeQubitTime,
          expectedQubitTime, maximumQubitTime, utilization, expectedUtilization,
          maximumUtilization, exhaustionProbability})
      if (!finite(value))
        return schedule.emitOpError(
            "schedule estimate produced non-finite or negative evidence");
    return success();
  }

  LogicalResult emitResult() {
    OpBuilder builder(context);
    auto i64 = builder.getI64Type();
    SmallVector<NamedAttribute> countFields;
    for (const auto &[kind, count] : eventCounts)
      countFields.emplace_back(builder.getStringAttr(kind),
                               IntegerAttr::get(i64, count));
    auto field = [&](StringRef name, Attribute value) {
      return builder.getNamedAttr(name, value);
    };
    SmallVector<NamedAttribute> fields = {
        field("schedule",
              FlatSymbolRefAttr::get(context, schedule.getSymName())),
        field("event_count", IntegerAttr::get(i64, entries.size())),
        field("event_counts", DictionaryAttr::get(context, countFields)),
        field("makespan_ns", builder.getF64FloatAttr(makespan)),
        field("expected_makespan_ns",
              builder.getF64FloatAttr(expectedMakespan)),
        field("maximum_makespan_ns", builder.getF64FloatAttr(maximumMakespan)),
        field("active_resource_time_ns",
              builder.getF64FloatAttr(activeResourceTime)),
        field("expected_active_resource_time_ns",
              builder.getF64FloatAttr(expectedResourceTime)),
        field("maximum_active_resource_time_ns",
              builder.getF64FloatAttr(maximumResourceTime)),
        field("active_physical_qubit_time_ns",
              builder.getF64FloatAttr(activeQubitTime)),
        field("expected_active_physical_qubit_time_ns",
              builder.getF64FloatAttr(expectedQubitTime)),
        field("maximum_active_physical_qubit_time_ns",
              builder.getF64FloatAttr(maximumQubitTime)),
        field("physical_resources", IntegerAttr::get(i64, allResources.size())),
        field("physical_qubits",
              IntegerAttr::get(i64, provisionedPhysicalQubits)),
        field("peak_concurrency", IntegerAttr::get(i64, peakConcurrency)),
        field("peak_active_physical_qubits",
              IntegerAttr::get(i64, peakActiveQubits)),
        field("utilization", builder.getF64FloatAttr(utilization)),
        field("expected_utilization",
              builder.getF64FloatAttr(expectedUtilization)),
        field("maximum_utilization",
              builder.getF64FloatAttr(maximumUtilization)),
        field("exhaustion_probability",
              builder.getF64FloatAttr(exhaustionProbability)),
        field(
            "termination_semantics",
            builder.getStringAttr(fullWorkload ? "full_workload" : "program")),
        field("bottleneck", builder.getStringAttr(bottleneck)),
    };
    SmallVector<Attribute> assumptions = {
        builder.getStringAttr(
            "durations come from the selected verified P3 timing profile"),
        builder.getStringAttr(
            "unresolved external resource supply is rejected"),
        builder.getStringAttr(
            "folded repeat and bounded-while multiplicities are exact"),
        builder.getStringAttr(
            "bounded retries replay the authenticated causal slice"),
        builder.getStringAttr(
            "conditional occupancy uses the worst-case executable branch"),
        builder.getStringAttr(
            "scheduled factory-model bindings contribute their full qubit "
            "footprint"),
        builder.getStringAttr(
            "shared call templates retain exact canonical leaf occupancy"),
        builder.getStringAttr(
            "scheduled_macro durations are deterministic mean-output slots; "
            "maximum_makespan_ns is conditional on those slots, not a physical "
            "factory worst case"),
        builder.getStringAttr(
            "utilization is active resource-time divided by scheduled resource "
            "capacity"),
        builder.getStringAttr(
            fullWorkload
                ? "expected metrics price the full scheduled workload despite "
                  "runtime abort semantics"
                : "expected metrics follow runtime abort semantics"),
    };
    FlatSymbolRefAttr lowerReference =
        lower ? FlatSymbolRefAttr::get(context, lower.getSymName())
              : FlatSymbolRefAttr{};
    builder.setInsertionPointToEnd(module.getBody());
    FlatSymbolRefAttr deviceReference =
        physical->getDevice() ? FlatSymbolRefAttr::get(
                                    context, physical->getDevice().getSymName())
                              : FlatSymbolRefAttr{};
    EstimateResultOp::create(
        builder, schedule.getLoc(), builder.getStringAttr(requestedResult),
        builder.getStringAttr("schedule"), schedule.getGraphAttr(),
        builder.getStringAttr("qlx.schedule-estimate/v2"),
        DictionaryAttr::get(context, fields), builder.getArrayAttr(assumptions),
        builder.getArrayAttr(
            {FlatSymbolRefAttr::get(context, schedule.getSymName()),
             graph.getArchitectureAttr()}),
        lowerReference, deviceReference,
        builder.getDictionaryAttr({
            builder.getNamedAttr(
                "producer", builder.getStringAttr("phys-estimate-schedule")),
            builder.getNamedAttr("producer_version",
                                 builder.getStringAttr("2")),
        }));
    if (failed(mlir::verify(module))) {
      if (auto result = symbols.lookup<EstimateResultOp>(requestedResult))
        result.erase();
      return schedule.emitOpError(
          "native schedule estimate failed independent module verification");
    }
    return success();
  }

  ModuleOp module;
  MLIRContext *context;
  SymbolTable symbols;
  std::string requestedSchedule;
  std::string requestedResult;
  std::string requestedLowerTier;
  bool requireResult;
  std::optional<NativeScheduleView> nativeSchedule;
  ScheduleOp schedule;
  GraphOp graph;
  EstimateResultOp lower;
  std::string physicalModelIdentity;
  std::optional<std::string> operatingPointIdentity;
  std::unique_ptr<PhysicalEvidence> physical;
  std::vector<Entry> entries;
  std::vector<std::string> syntheticResourceLabels;
  llvm::StringMap<size_t> byId;
  std::vector<SmallVector<size_t, 4>> children;
  std::vector<SmallVector<size_t, 4>> dependents;
  std::vector<int64_t> multiplicities;
  std::vector<QubitSetId> entryQubitSets;
  std::vector<ResourceSetId> entryResourceSets;
  llvm::StringMap<ResourceId> resourceIds;
  std::vector<SmallVector<unsigned, 4>> resourcePhysicalsById;
  std::vector<SmallVector<unsigned, 4>> resourceQubitsById;
  llvm::StringMap<unsigned> physicalResourceIds;
  llvm::StringMap<unsigned> qubitIds;
  std::vector<int64_t> qubitWeights;
  llvm::DenseMap<size_t, SmallVector<QubitSetId, 1>> qubitSetBuckets;
  std::vector<SmallVector<unsigned, 4>> internedQubitSets;
  std::vector<QubitSetId> resourceSetQubits;
  llvm::DenseMap<size_t, SmallVector<ResourceSetId, 1>> resourceSetBuckets;
  std::vector<SmallVector<ResourceId, 4>> internedResourceSets;
  llvm::DenseMap<size_t, SmallVector<SummaryPathId, 1>> summaryPathBuckets;
  std::vector<SmallVector<SummaryConstraint, 4>> internedSummaryPaths;
  llvm::StringSet<> allResources;
  std::set<std::string> provisionedQubits;
  int64_t provisionedPhysicalQubits = 0;
  std::vector<size_t> templateResourceAliases;
  std::vector<llvm::DenseMap<ResourceId, ResourceId>> templateResourceAliasMaps;
  std::map<size_t, CallSummary> callSummaries;
  llvm::SmallDenseSet<size_t, 8> buildingSummaries;
  MetricNode baseMetricRoot;
  SmallVector<SummaryOccurrence, 0> summaryOccurrences;
  std::vector<size_t> occurrenceMetricOffsets;
  std::vector<std::pair<size_t, size_t>> occurrenceMetricCounts;
  std::vector<SmallVector<std::pair<size_t, size_t>, 4>>
      summaryOccurrenceChildren;
  std::vector<llvm::SmallDenseMap<ResourceSetId, OccurrenceResourceMapping, 4>>
      occurrenceResourceEvidence;
  std::vector<std::pair<size_t, size_t>> summaryOccurrenceRanges;
  size_t retryConditionBase = 0;
  SmallVector<PeakAtom, 0> peakCandidates;
  SmallVector<PeakAtom, 0> peakAtoms;
  SmallVector<ReplaySlice, 4> retrySlices;
  SmallVector<StoppedRetryTimeLayer, 4> stoppedRetryTimeLayers;
  std::vector<size_t> stoppedRetryTimeLayerForSlice;
  std::map<std::string, int64_t> eventCounts;
  double makespan = 0.0;
  double expectedMakespan = 0.0;
  double maximumMakespan = 0.0;
  double activeResourceTime = 0.0;
  double expectedResourceTime = 0.0;
  double maximumResourceTime = 0.0;
  double activeQubitTime = 0.0;
  double expectedQubitTime = 0.0;
  double maximumQubitTime = 0.0;
  double utilization = 0.0;
  double expectedUtilization = 0.0;
  double maximumUtilization = 0.0;
  double exhaustionProbability = 0.0;
  int64_t peakConcurrency = 0;
  int64_t peakActiveQubits = 0;
  std::string bottleneck;
  bool fullWorkload = false;
};

struct PhysEstimateSchedulePass
    : public qlx::phys::impl::PhysEstimateScheduleBase<
          PhysEstimateSchedulePass> {
  using PhysEstimateScheduleBase::PhysEstimateScheduleBase;

  void runOnOperation() override {
    if (terminationMode != "program" && terminationMode != "full_workload") {
      getOperation().emitError(
          "phys-estimate-schedule termination must be program or "
          "full_workload");
      return signalPassFailure();
    }
    NativeScheduleEstimator estimator(getOperation(), scheduleSymbol,
                                      resultSymbol, lowerTierSymbol,
                                      /*requireResult=*/true, std::nullopt,
                                      terminationMode == "full_workload");
    if (failed(estimator.run()))
      signalPassFailure();
  }
};

} // namespace

static FailureOr<std::string>
estimateScheduleJSONImpl(ModuleOp module, StringRef schedule,
                         StringRef lowerTier, bool fullWorkload,
                         bool authenticateModule) {
  // Standalone estimation consumes portable schedule evidence. Authenticate
  // the complete immutable module before resolving or aggregating any of that
  // evidence; ScheduleOp::verify() parses the rows and invokes the same common
  // semantic proof used by the fused typed-claim path. A compiler-owned
  // in-process PhysicalSchedule already crossed that exact boundary and may
  // reuse its sealed ModuleOp without repeating the proof.
  if (authenticateModule && failed(mlir::verify(module)))
    return failure();
  NativeScheduleEstimator estimator(module, schedule, /*requestedResult=*/{},
                                    lowerTier, /*requireResult=*/false,
                                    std::nullopt, fullWorkload);
  if (failed(estimator.run()))
    return failure();
  return estimator.json();
}

FailureOr<std::string> qlx::phys::estimateScheduleJSON(ModuleOp module,
                                                       StringRef schedule,
                                                       StringRef lowerTier,
                                                       bool fullWorkload) {
  return estimateScheduleJSONImpl(module, schedule, lowerTier, fullWorkload,
                                  /*authenticateModule=*/true);
}

FailureOr<std::string>
qlx::phys::estimateVerifiedScheduleJSON(ModuleOp module, StringRef schedule,
                                        StringRef lowerTier,
                                        bool fullWorkload) {
  return estimateScheduleJSONImpl(module, schedule, lowerTier, fullWorkload,
                                  /*authenticateModule=*/false);
}

FailureOr<std::string> qlx::phys::estimateScheduleJSON(ModuleOp module,
                                                       StringRef schedule,
                                                       StringRef lowerTier,
                                                       NativeScheduleView view,
                                                       bool fullWorkload) {
  NativeScheduleEstimator estimator(module, schedule, /*requestedResult=*/{},
                                    lowerTier, /*requireResult=*/false, view,
                                    fullWorkload);
  if (failed(estimator.run()))
    return failure();
  return estimator.json();
}
