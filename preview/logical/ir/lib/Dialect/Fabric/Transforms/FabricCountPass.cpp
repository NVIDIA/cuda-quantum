/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Tier 1 of the resource-estimation stack
// (design/resource-estimation-proposal.md §4.1).
//
// Walks a Fabric module and emits a typed `qlx.estimate_result` summarizing
// logical-qubit / gate / round / protocol counts. The legacy top-level
// `fabric.counts` DictionaryAttr is mirrored during migration. Sub-second; no
// noise model, no schedule, no wall-clock.
//
// Walk semantics:
//   - fabric.gadget is IsolatedFromAbove. Top-level walk hits only the
//     entry gadget; non-entry gadgets are reached lazily through
//     fabric.call (inlining-by-counts at each call site).
//   - cflow.repeat: gate counts inside the body multiply by `count`.
//     Nested repeats compose multiplicatively.
//   - fabric.idle: `rounds` accumulates into rounds_by_kind.idle.
//   - cflow.if: conservatively counts both branches and propagates yielded
//     patch attribution so downstream analyses can continue. The resulting
//     counts are a static upper bound for dynamic feedback programs.
//   - fabric.transport: counted in both src (transport_out) and dst
//     (transport_in) regions, plus once under per_protocol.
//   - logical_qubits_peak: maximum live logical qubits, weighting each patch
//     by the resolved code's k while retaining patch concurrency separately.
//   - Protocol attrs resolve from FlatSymbolRefAttr or
//     #fabric.spec_only<"name">; kind derives from the owning op.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Fabric/Transforms/Passes.h"

#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Fabric/IR/FabricTypes.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <utility>

using namespace mlir;
using namespace qlx::fabric;

namespace qlx {
namespace fabric {
#define GEN_PASS_DEF_FABRICCOUNT
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"
} // namespace fabric
} // namespace qlx

namespace {

static bool isPatchLike(Type type) {
  return isa<PatchType, PatchFrameType>(type);
}

//===----------------------------------------------------------------------===//
// Accumulators
//===----------------------------------------------------------------------===//

struct PerRegion {
  StringRef role;
  StringRef code;
  int64_t distance = 0;
  int64_t patches = 0;
  llvm::MapVector<StringRef, int64_t> gateCounts;
  llvm::MapVector<StringRef, int64_t> roundsByKind;
  int64_t injectCount = 0;
  int64_t transportIn = 0;
  int64_t transportOut = 0;
  Attribute protocol;
};

struct ProtocolEntry {
  StringRef kind;
  int64_t opCount = 0;
};

struct ResourceDemand {
  StringAttr kind;
  SymbolRefAttr stream;
  int64_t count = 0;
};

struct RetryDemandSummary {
  FlatSymbolRefAttr attempt;
  FlatSymbolRefAttr profile;
  FlatSymbolRefAttr probabilitySource;
  StringAttr probabilityEvidence;
  StringAttr exhaustion;
  std::optional<double> successProbability;
  int64_t maxAttempts = 0;
  int64_t occurrences = 0;
  int64_t attemptOperationSites = 0;
  SmallVector<ResourceDemand> resourceRequests;
};

struct RetryAttemptFacts {
  int64_t operationSites = 0;
  SmallVector<ResourceDemand> resourceRequests;
};

struct CallableSummaryKey {
  Operation *callee = nullptr;
  bool suppressInlineAnalysis = false;
  SmallVector<StringRef, 4> inputRegions;
  SmallVector<std::pair<StringRef, int64_t>, 4> liveRegions;
  SmallVector<StringRef, 4> tickRegions;
  int64_t totalLive = 0;
  int64_t logicalLive = 0;

  bool operator<(const CallableSummaryKey &other) const {
    if (callee != other.callee)
      return std::less<Operation *>{}(callee, other.callee);
    if (suppressInlineAnalysis != other.suppressInlineAnalysis)
      return suppressInlineAnalysis < other.suppressInlineAnalysis;
    if (inputRegions != other.inputRegions)
      return std::lexicographical_compare(
          inputRegions.begin(), inputRegions.end(), other.inputRegions.begin(),
          other.inputRegions.end());
    if (liveRegions != other.liveRegions)
      return std::lexicographical_compare(
          liveRegions.begin(), liveRegions.end(), other.liveRegions.begin(),
          other.liveRegions.end());
    if (tickRegions != other.tickRegions)
      return std::lexicographical_compare(
          tickRegions.begin(), tickRegions.end(), other.tickRegions.begin(),
          other.tickRegions.end());
    if (totalLive != other.totalLive)
      return totalLive < other.totalLive;
    return logicalLive < other.logicalLive;
  }
};

/// Exact additive facts for one executable callable at unit multiplicity.
///
/// The key includes the exact abstract liveness state consumed by FabricCount,
/// so balanced allocation/tick-bearing protocol bodies can be reused without
/// changing peak or attribution semantics.  A closure whose exit liveness
/// differs is retained on the original exact-walk path.
struct CallableSummary {
  llvm::MapVector<StringRef, PerRegion> perRegion;
  llvm::MapVector<StringRef, ProtocolEntry> perProtocol;
  std::map<std::string, int64_t> operationCounts;
  std::map<std::string, int64_t> gadgetCalls;
  std::map<std::string, int64_t> protocolCalls;
  SmallVector<ResourceDemand> resourceRequests;
  SmallVector<RetryDemandSummary> retryDemands;
  SmallVector<std::pair<StringRef, StringRef>> transversalEdges;
  SmallVector<StringRef, 4> outputRegions;
  int64_t successCount = 0;
  int64_t syndromeRounds = 0;
  int64_t patchPeak = 0;
  int64_t logicalPeak = 0;
  bool requiresExtendedAnalyticalModel = false;
  bool sawQecSpec = false;
  bool sawSelectedDeviceGadget = false;
  bool sawSelectedDeviceRegionUse = false;
  uint64_t operationSites = 0;
};

struct CachedCallableSummary {
  bool reusable = false;
  CallableSummary summary;
};

//===----------------------------------------------------------------------===//
// Walker
//===----------------------------------------------------------------------===//

class Walker {
public:
  Walker(MLIRContext *ctx, SymbolTable &st, FlatSymbolRefAttr selectedQec = {})
      : ctx(ctx), symTab(st), selectedQec(selectedQec) {}

  llvm::MapVector<StringRef, PerRegion> perRegion;
  llvm::MapVector<StringRef, ProtocolEntry> perProtocol;
  std::map<std::string, int64_t> operationCounts;
  std::map<std::string, int64_t> gadgetCalls;
  std::map<std::string, int64_t> protocolCalls;
  std::map<std::string, int64_t> hierarchyDepths;
  llvm::SmallVector<ResourceDemand> resourceRequests;
  llvm::SmallVector<RetryDemandSummary> retryDemands;
  llvm::SmallVector<std::pair<StringRef, StringRef>> transversalEdges;
  int64_t successCount = 0;
  int64_t syndromeRounds = 0;
  int64_t patchPeak = 0;
  int64_t logicalPeak = 0;
  // Max-concurrent live patches (alloc - dealloc), globally and per region.
  // This is loop-invariant: a patch allocated and freed inside a loop body
  // is live one-iteration-at-a-time, so it must NOT be multiplied by the
  // trip count (unlike gate counts). PerRegion.patches holds the per-region
  // peak concurrency.
  int64_t totalLive = 0;
  int64_t logicalLive = 0;
  DenseMap<StringRef, int64_t> liveByRegion;
  bool hadError = false;
  bool requiresExtendedAnalyticalModel = false;
  bool sawQecSpec = false;
  bool sawSelectedDeviceGadget = false;
  bool sawSelectedDeviceRegionUse = false;

  void printProfile(llvm::raw_ostream &stream) const {
    stream << "fabric-count: operations=" << visitedOperations
           << " delegations=" << delegations << " retry-probes=" << retryProbes
           << " retry-probe-cache-hits=" << retryProbeCacheHits
           << " retry-probe-operations=" << retryProbeOperations
           << " retry-probe-seconds=" << retryProbeSeconds
           << "s callable-summary-probes=" << callableSummaryProbes
           << " callable-summary-hits=" << callableSummaryHits
           << " callable-summary-negative-hits=" << callableSummaryNegativeHits
           << " callable-summary-saved-operations="
           << callableSummarySavedOperations << "\n";
  }

  void recordRegionDecl(RegionOp r);
  void recordCodeDecl(CodeOp c);
  void walkEntry(Operation *entry);

private:
  MLIRContext *ctx;
  SymbolTable &symTab;
  DenseMap<Value, StringRef> valueToRegion;
  DenseMap<StringRef, int64_t> mappedValuesByRegion;
  DenseMap<StringRef, int64_t> distanceByCode;
  DenseMap<StringRef, int64_t> logicalsByCode;
  llvm::StringSet<> selectedRegions;
  llvm::SmallSet<StringRef, 8> activeCalls;
  bool suppressInlineAnalysis = false;
  FlatSymbolRefAttr selectedQec;
  uint64_t visitedOperations = 0;
  uint64_t delegations = 0;
  uint64_t retryProbes = 0;
  uint64_t retryProbeCacheHits = 0;
  uint64_t retryProbeOperations = 0;
  double retryProbeSeconds = 0.0;
  DenseMap<Operation *, RetryAttemptFacts> retryAttemptCache;
  llvm::SmallSet<StringRef, 8> tickContextRegions;
  uint64_t callableSummaryProbes = 0;
  uint64_t callableSummaryHits = 0;
  uint64_t callableSummaryNegativeHits = 0;
  uint64_t callableSummarySavedOperations = 0;
  bool enableCallableSummaries = true;
  std::map<CallableSummaryKey, CachedCallableSummary> callableSummaryCache;

  PerRegion &regionAccum(StringRef name);
  StringRef regionForValue(Value v) const;
  void mapValueToRegion(Value value, StringRef region);
  void eraseValueRegion(Value value);
  void propagatePatchValues(Operation *op);
  void bumpGate(Operation *op, StringRef gate, int64_t mult);
  void bumpRoundFor(Value patch, StringRef kind, int64_t n);
  void recordProtocol(Attribute protoAttr, StringRef kind, int64_t mult);
  void recordResourceRequest(ResourceRequestOp request, int64_t mult);
  void recordRetry(RetryOp retry, int64_t mult);
  static StringRef protocolNameOf(Attribute a);
  bool checkedAdd(int64_t &target, int64_t value, Operation *source,
                  StringRef what);
  FailureOr<int64_t> checkedMultiply(int64_t left, int64_t right,
                                     Operation *source, StringRef what);
  FailureOr<int64_t> logicalWeight(Value patch, Operation *source);
  void bump(std::map<std::string, int64_t> &counts, StringRef name,
            int64_t value, Operation *source, StringRef what);

  void walkBlock(Block &block, int64_t mult);
  void walkOp(Operation *op, int64_t mult);
  void walkCall(CallOp call, int64_t mult);
  void walkDelegation(Operation *wrapper, FlatSymbolRefAttr callee,
                      int64_t wrapperMultiplicity, int64_t calleeMultiplicity);
  CallableSummaryKey callableSummaryKey(Operation *callee,
                                        Operation *wrapper) const;
  FailureOr<CachedCallableSummary>
  buildCallableSummary(Operation *callee, Operation *wrapper,
                       const CallableSummaryKey &key);
  LogicalResult applyCallableSummary(Operation *wrapper,
                                     const CallableSummary &summary,
                                     int64_t multiplicity);
  void eraseRegionValueMappings(Region &region);
  FailureOr<int64_t> mapChildrenMultiplicity(MapChildrenOp map,
                                             int64_t multiplier);
  Block *resolveExecutableBody(Operation *callable, Operation *diagnosticOwner);
  void walkRepeat(qlx::cflow::RepeatOp rep, int64_t mult);
  void walkIf(qlx::cflow::IfOp ifop, int64_t mult);
};

PerRegion &Walker::regionAccum(StringRef name) {
  auto it = perRegion.find(name);
  if (it != perRegion.end())
    return it->second;
  perRegion.insert({name, PerRegion{}});
  return perRegion.find(name)->second;
}

void Walker::recordRegionDecl(RegionOp r) {
  StringRef name = r.getSymName();
  if (selectedQec)
    selectedRegions.insert(name);
  PerRegion &p = regionAccum(name);
  p.role = stringifyRole(r.getRole());
  p.code = r.getCode();
  p.protocol = r.getProtocolAttr();
  auto dIt = distanceByCode.find(p.code);
  if (dIt != distanceByCode.end())
    p.distance = dIt->second;
}

void Walker::recordCodeDecl(CodeOp c) {
  distanceByCode[c.getSymName()] = c.getDistance();
  logicalsByCode[c.getSymName()] = c.getK().value_or(1);
  // Backfill any regions already recorded.
  for (auto &kv : perRegion) {
    if (kv.second.code == c.getSymName())
      kv.second.distance = c.getDistance();
  }
}

StringRef Walker::protocolNameOf(Attribute a) {
  if (!a)
    return {};
  if (auto sym = dyn_cast<FlatSymbolRefAttr>(a))
    return sym.getValue();
  if (auto so = dyn_cast<SpecOnlyAttr>(a))
    return so.getName();
  return {};
}

bool Walker::checkedAdd(int64_t &target, int64_t value, Operation *source,
                        StringRef what) {
  int64_t result;
  if (value < 0 || llvm::AddOverflow(target, value, result)) {
    if (source)
      source->emitOpError()
          << "fabric-count " << what << " overflows signed i64";
    else
      emitError(UnknownLoc::get(ctx))
          << "fabric-count " << what << " overflows signed i64";
    hadError = true;
    return false;
  }
  target = result;
  return true;
}

FailureOr<int64_t> Walker::checkedMultiply(int64_t left, int64_t right,
                                           Operation *source, StringRef what) {
  int64_t result;
  if (left < 0 || right < 0 || llvm::MulOverflow(left, right, result)) {
    source->emitOpError() << "fabric-count " << what << " overflows signed i64";
    hadError = true;
    return failure();
  }
  return result;
}

FailureOr<int64_t> Walker::logicalWeight(Value patch, Operation *source) {
  auto patchType = dyn_cast<PatchType>(patch.getType());
  if (!patchType) {
    source->emitOpError("fabric-count expected a typed patch value");
    hadError = true;
    return failure();
  }
  StringRef code = patchType.getCodeType().getValue();
  auto found = logicalsByCode.find(code);
  if (found == logicalsByCode.end()) {
    source->emitOpError("fabric-count cannot resolve code @") << code;
    hadError = true;
    return failure();
  }
  sawQecSpec = true;
  return found->second;
}

void Walker::bump(std::map<std::string, int64_t> &counts, StringRef name,
                  int64_t value, Operation *source, StringRef what) {
  auto [iterator, inserted] = counts.try_emplace(name.str(), value);
  if (!inserted)
    checkedAdd(iterator->second, value, source, what);
}

void Walker::recordProtocol(Attribute protoAttr, StringRef kind, int64_t mult) {
  StringRef name = protocolNameOf(protoAttr);
  if (name.empty())
    return;
  auto it = perProtocol.find(name);
  if (it == perProtocol.end()) {
    perProtocol.insert({name, ProtocolEntry{kind, mult}});
    return;
  }
  checkedAdd(it->second.opCount, mult, nullptr, "protocol count");
}

void Walker::recordResourceRequest(ResourceRequestOp request, int64_t mult) {
  for (ResourceDemand &demand : resourceRequests) {
    if (demand.kind == request.getKindAttr() &&
        demand.stream == request.getStreamAttr()) {
      checkedAdd(demand.count, mult, request, "resource-request count");
      return;
    }
  }
  resourceRequests.push_back(
      ResourceDemand{request.getKindAttr(), request.getStreamAttr(), mult});
}

void Walker::recordRetry(RetryOp retry, int64_t mult) {
  auto attemptRef = retry.getAttemptAttr();
  Operation *attempt =
      attemptRef ? symTab.lookup(attemptRef.getValue()) : nullptr;
  if (!attempt || !isa<GadgetOp, ProtocolOp>(attempt)) {
    retry.emitOpError(
        "fabric-count retry requires one resolvable executable attempt");
    hadError = true;
    return;
  }

  // Count one attempt in an isolated walker. The selected attempt is already
  // explicit in the caller's static counts; this second walk derives only the
  // per-attempt facts needed by Tier 2 to account for bounded failed attempts.
  // A retained attempt callable may be reached by several retry occurrences,
  // so cache this exact, selected-QEC-specific callable summary by operation.
  auto cached = retryAttemptCache.find(attempt);
  if (cached == retryAttemptCache.end()) {
    const auto probeStarted = std::chrono::steady_clock::now();
    Walker probe(ctx, symTab, selectedQec);
    ModuleOp module = retry->getParentOfType<ModuleOp>();
    for (auto code : module.getOps<CodeOp>())
      probe.recordCodeDecl(code);
    if (selectedQec) {
      auto machine = symTab.lookup<DeviceOp>(selectedQec.getValue());
      if (!machine) {
        retry.emitOpError("fabric-count retry selected QEC machine is missing");
        hadError = true;
        return;
      }
      for (auto region : machine.getOps<RegionOp>())
        probe.recordRegionDecl(region);
    } else {
      for (auto machine : module.getOps<DeviceOp>())
        for (auto region : machine.getOps<RegionOp>())
          probe.recordRegionDecl(region);
    }
    probe.walkEntry(attempt);
    ++retryProbes;
    retryProbeOperations += probe.visitedOperations;
    retryProbeSeconds += std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - probeStarted)
                             .count();
    if (probe.hadError) {
      hadError = true;
      return;
    }
    if (auto nested = probe.operationCounts.find("retry");
        nested != probe.operationCounts.end() && nested->second != 0) {
      retry.emitOpError("fabric-count does not support nested retry demand");
      hadError = true;
      return;
    }

    static constexpr StringLiteral structural[] = {
        "call",         "establish_support", "establish_topological_record",
        "map_children", "relocate",          "repeat"};
    int64_t attemptSites = 0;
    for (const auto &[name, count] : probe.operationCounts) {
      if (name == "idle" || llvm::is_contained(structural, StringRef(name)))
        continue;
      if (!checkedAdd(attemptSites, count, retry, "retry attempt-site count"))
        return;
    }
    for (const auto &[name, region] : probe.perRegion) {
      (void)name;
      if (auto idle = region.roundsByKind.find("idle");
          idle != region.roundsByKind.end())
        if (!checkedAdd(attemptSites, idle->second, retry,
                        "retry attempt-site count"))
          return;
    }
    RetryAttemptFacts facts;
    facts.operationSites = attemptSites;
    facts.resourceRequests = std::move(probe.resourceRequests);
    cached = retryAttemptCache.try_emplace(attempt, std::move(facts)).first;
  } else {
    ++retryProbeCacheHits;
  }

  RetryDemandSummary summary;
  summary.attempt = attemptRef;
  summary.profile = retry.getProfileAttr();
  summary.probabilitySource = retry.getSuccessProbabilitySourceAttr();
  summary.probabilityEvidence = retry.getSuccessProbabilityEvidenceAttr();
  summary.exhaustion = retry.getExhaustionAttr();
  if (auto probability = retry.getSuccessProbabilityAttr())
    summary.successProbability = probability.getValueAsDouble();
  summary.maxAttempts = retry.getMaxAttempts();
  summary.occurrences = mult;
  summary.attemptOperationSites = cached->second.operationSites;
  summary.resourceRequests = cached->second.resourceRequests;
  retryDemands.push_back(std::move(summary));
}

StringRef Walker::regionForValue(Value v) const {
  auto it = valueToRegion.find(v);
  if (it == valueToRegion.end())
    return {};
  return it->second;
}

void Walker::mapValueToRegion(Value value, StringRef region) {
  auto existing = valueToRegion.find(value);
  if (existing != valueToRegion.end()) {
    if (existing->second == region)
      return;
    if (!existing->second.empty()) {
      auto count = mappedValuesByRegion.find(existing->second);
      if (count != mappedValuesByRegion.end() && --count->second == 0)
        mappedValuesByRegion.erase(count);
    }
    valueToRegion.erase(existing);
  }
  if (region.empty())
    return;
  valueToRegion[value] = region;
  ++mappedValuesByRegion[region];
}

void Walker::eraseValueRegion(Value value) {
  auto existing = valueToRegion.find(value);
  if (existing == valueToRegion.end())
    return;
  if (!existing->second.empty()) {
    auto count = mappedValuesByRegion.find(existing->second);
    if (count != mappedValuesByRegion.end() && --count->second == 0)
      mappedValuesByRegion.erase(count);
  }
  valueToRegion.erase(existing);
}

void Walker::bumpGate(Operation *op, StringRef gate, int64_t mult) {
  // Find a patch-like owner to attribute the gate to. A patch frame carries
  // the same persistent reservation as its source patch while a transform is
  // active; it does not introduce another live allocation.
  StringRef region;
  for (Value v : op->getOperands()) {
    if (isPatchLike(v.getType())) {
      region = regionForValue(v);
      if (!region.empty())
        break;
    }
  }
  if (region.empty())
    return;
  PerRegion &p = regionAccum(region);
  auto it = p.gateCounts.find(gate);
  if (it == p.gateCounts.end())
    p.gateCounts.insert({gate, mult});
  else
    checkedAdd(it->second, mult, op, "gate count");
}

void Walker::bumpRoundFor(Value patch, StringRef kind, int64_t n) {
  StringRef region = regionForValue(patch);
  if (region.empty())
    return;
  PerRegion &p = regionAccum(region);
  auto it = p.roundsByKind.find(kind);
  if (it == p.roundsByKind.end())
    p.roundsByKind.insert({kind, n});
  else
    checkedAdd(it->second, n, nullptr, "round count");
}

/// Propagate region attribution for ops that consume a patch-like owner and
/// produce one with the "same logical identity." This is true for all single-
/// patch gate ops (AllTypesMatch<patch, result>), measurement (which also
/// keeps patch_out), merge/split (compose from operand a), multi_measure /
/// product ops (each output_i ← input_i), and inject (result ← patch operand).
void Walker::propagatePatchValues(Operation *op) {
  // Identify the first patch-like operand. Patch frames preserve the source
  // patch's region throughout a typed transform_begin/transform_end lifetime.
  Value firstPatchOperand;
  for (Value v : op->getOperands()) {
    if (isPatchLike(v.getType())) {
      firstPatchOperand = v;
      break;
    }
  }

  // Merge/split need their own handling.
  if (auto m = dyn_cast<MergeOp>(op)) {
    StringRef r = regionForValue(m.getPatchA());
    if (r.empty())
      r = regionForValue(m.getPatchB());
    if (!r.empty())
      mapValueToRegion(m.getMerged(), r);
    return;
  }
  if (auto s = dyn_cast<SplitOp>(op)) {
    StringRef r = regionForValue(s.getMerged());
    if (r.empty())
      return;
    mapValueToRegion(s.getPatchA(), r);
    mapValueToRegion(s.getPatchB(), r);
    return;
  }
  if (auto mm = dyn_cast<MultiMeasureOp>(op)) {
    auto ins = mm.getPatches();
    auto outs = mm.getPatchesOut();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        mapValueToRegion(out, r);
    }
    return;
  }
  if (auto mp = dyn_cast<MeasureProductOp>(op)) {
    auto ins = mp.getPatches();
    auto outs = mp.getPatchResults();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        mapValueToRegion(out, r);
    }
    return;
  }
  if (auto rp = dyn_cast<RotateProductOp>(op)) {
    auto ins = rp.getPatches();
    auto outs = rp.getPatchResults();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        mapValueToRegion(out, r);
    }
    return;
  }
  if (auto rp = dyn_cast<ResourceRotateProductOp>(op)) {
    auto ins = rp.getPatches();
    auto outs = rp.getPatchResults();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        mapValueToRegion(out, r);
    }
    return;
  }
  if (auto tx = dyn_cast<TransversalCXOp>(op)) {
    StringRef rc = regionForValue(tx.getCtrl());
    StringRef rt = regionForValue(tx.getTarg());
    if (!rc.empty())
      mapValueToRegion(tx.getCtrlOut(), rc);
    if (!rt.empty())
      mapValueToRegion(tx.getTargOut(), rt);
    return;
  }
  if (auto barrier = dyn_cast<BarrierOp>(op)) {
    for (auto [in, out] :
         llvm::zip(barrier.getPatches(), barrier.getResults())) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        mapValueToRegion(out, r);
    }
    return;
  }

  // Default: a single patch-like operand projects to all patch-like results.
  if (!firstPatchOperand)
    return;
  StringRef r = regionForValue(firstPatchOperand);
  if (r.empty())
    return;
  for (Value v : op->getResults()) {
    if (isPatchLike(v.getType()))
      mapValueToRegion(v, r);
  }
}

void Walker::walkEntry(Operation *entry) {
  if (!entry)
    return;
  if (auto profile = dyn_cast<GadgetProfileOp>(entry)) {
    auto gadget = symTab.lookup<GadgetOp>(profile.getGadgetAttr().getValue());
    if (!gadget || gadget.getBody().empty()) {
      profile.emitOpError("fabric-count profile references missing gadget");
      hadError = true;
      return;
    }
    suppressInlineAnalysis = true;
    walkEntry(gadget);
    suppressInlineAnalysis = false;
    if (!profile.getBody().empty())
      walkBlock(profile.getBody().front(), 1);
    return;
  }
  if (auto gadget = dyn_cast<GadgetOp>(entry)) {
    if (gadget.getDeviceAttr()) {
      if (selectedQec && gadget.getDeviceAttr() != selectedQec) {
        gadget.emitOpError(
            "fabric-count executable root belongs to another P2 machine");
        hadError = true;
        return;
      }
      sawSelectedDeviceGadget = true;
    }
  }
  Block *body = resolveExecutableBody(entry, entry);
  if (!body)
    return;
  auto name = cast<StringAttr>(SymbolTable::getSymbolName(entry)).getValue();
  for (BlockArgument argument : body->getArguments()) {
    if (!isa<PatchType>(argument.getType()))
      continue;
    if (!checkedAdd(totalLive, 1, entry, "root live-patch count"))
      return;
    patchPeak = std::max(patchPeak, totalLive);
    auto weight = logicalWeight(argument, entry);
    if (failed(weight) ||
        !checkedAdd(logicalLive, *weight, entry, "root logical-qubit count"))
      return;
    logicalPeak = std::max(logicalPeak, logicalLive);
  }
  activeCalls.insert(name);
  walkBlock(*body, /*mult=*/1);
  activeCalls.erase(name);
}

Block *Walker::resolveExecutableBody(Operation *callable,
                                     Operation *diagnosticOwner) {
  if (auto gadget = dyn_cast<GadgetOp>(callable)) {
    if (auto realization = gadget.getRealizationAttr()) {
      auto circuit = symTab.lookup<CircuitOp>(realization.getValue());
      if (!circuit || circuit.getBody().empty()) {
        diagnosticOwner->emitOpError(
            "fabric-count gadget realization does not resolve to a "
            "fabric.circuit body");
        hadError = true;
        return nullptr;
      }
      return &circuit.getBody().front();
    }
  }
  if (callable->getNumRegions() != 1 || callable->getRegion(0).empty()) {
    diagnosticOwner->emitOpError(
        "fabric-count cannot resolve executable callable body");
    hadError = true;
    return nullptr;
  }
  return &callable->getRegion(0).front();
}

void Walker::walkBlock(Block &block, int64_t mult) {
  for (Operation &op : block)
    walkOp(&op, mult);
}

void Walker::eraseRegionValueMappings(Region &region) {
  // Region-local SSA values cannot escape except through the enclosing
  // operation's explicitly mapped results.  Keeping their attribution in the
  // global table after a call/control body has finished is both unnecessary
  // and disastrous for repeated callable closures: every later branch used to
  // copy all stale entries.  Erase the complete nested scope after its yielded
  // or returned regions have been captured.
  for (Block &block : region) {
    for (BlockArgument argument : block.getArguments())
      eraseValueRegion(argument);
    for (Operation &op : block) {
      for (Region &nested : op.getRegions())
        eraseRegionValueMappings(nested);
      for (Value result : op.getResults())
        eraseValueRegion(result);
    }
  }
}

void Walker::walkOp(Operation *op, int64_t mult) {
  ++visitedOperations;
  if (isa<RetryOp, ResourceRequestOp, ResourceRotateProductOp,
          ProduceResourceOp, UnpackResourceOp, PackResourceOp,
          DiscardResourceOp, InjectOp, TransportOp, qlx::event::SelectionOp,
          SuccessOp, qlx::event::TryTakeOp>(op))
    requiresExtendedAnalyticalModel = true;

  StringRef operationName = op->getName().getStringRef();
  if (operationName.starts_with("fabric.") &&
      !isa<ReturnOp, ProtocolReturnOp, ProfileEndOp>(op)) {
    bump(operationCounts, operationName.drop_front(7), mult, op,
         "operation count");
  } else if (StringRef legacyName =
                 // `event.*` ops embedded in a Fabric body used to be
                 // `fabric.event_*`/`fabric.fence`/`fabric.selection`;
                 // preserve their pre-migration operation_counts key
                 // spelling (asserted on directly by name in some
                 // callers) rather than switching to the bare
                 // `operationName.drop_front(6)` form a naive
                 // generalization would produce.
             llvm::StringSwitch<StringRef>(operationName)
                 .Case("event.test", "event_test")
                 .Case("event.poll", "event_poll")
                 .Case("event.is", "event_is")
                 .Case("event.select_ready", "event_select_ready")
                 .Case("event.try_take", "event_try_take")
                 .Case("event.cancel", "event_cancel")
                 .Case("event.await", "event_await")
                 .Case("event.fence", "fence")
                 .Case("event.selection", "selection")
                 .Default("");
             !legacyName.empty()) {
    bump(operationCounts, legacyName, mult, op, "operation count");
  }
  if (isa<ReadSyndromeAncillasOp, AssembleSyndromeOp>(op))
    checkedAdd(syndromeRounds, mult, op, "syndrome-round count");

  // Structural / scoping ops handled specially.
  if (auto a = dyn_cast<AllocOp>(op)) {
    StringRef r = a.getRegion();
    if (selectedQec) {
      if (!selectedRegions.contains(r)) {
        a.emitOpError("fabric-count allocation region @")
            << r << " is not part of the selected P2 machine";
        hadError = true;
        return;
      }
      sawSelectedDeviceRegionUse = true;
    }
    mapValueToRegion(a.getResult(), r);
    PerRegion &p = regionAccum(r);
    // Max-concurrent (not total*mult): a patch becomes live here. Loop
    // bodies are walked once, so an alloc/dealloc pair inside a loop counts
    // as one concurrent patch, correctly reused across iterations.
    int64_t &live = liveByRegion[r];
    if (!checkedAdd(live, 1, op, "live-patch count"))
      return;
    if (live > p.patches)
      p.patches = live;
    if (!checkedAdd(totalLive, 1, op, "global live-patch count"))
      return;
    patchPeak = std::max(patchPeak, totalLive);
    auto weight = logicalWeight(a.getResult(), op);
    if (failed(weight) ||
        !checkedAdd(logicalLive, *weight, op, "global logical-qubit count"))
      return;
    logicalPeak = std::max(logicalPeak, logicalLive);
    return;
  }
  if (auto c = dyn_cast<CallOp>(op)) {
    walkCall(c, mult);
    return;
  }
  if (auto map = dyn_cast<MapChildrenOp>(op)) {
    auto nested = mapChildrenMultiplicity(map, mult);
    if (succeeded(nested))
      walkDelegation(op, map.getCalleeAttr(), mult, *nested);
    return;
  }
  if (auto relocate = dyn_cast<RelocateOp>(op)) {
    walkDelegation(op, relocate.getCalleeAttr(), mult, mult);
    return;
  }
  if (auto support = dyn_cast<EstablishSupportOp>(op)) {
    walkDelegation(op, support.getCalleeAttr(), mult, mult);
    return;
  }
  if (auto topological = dyn_cast<EstablishTopologicalRecordOp>(op)) {
    walkDelegation(op, topological.getCalleeAttr(), mult, mult);
    return;
  }
  if (auto r = dyn_cast<qlx::cflow::RepeatOp>(op)) {
    bump(operationCounts, "repeat", mult, op, "operation count");
    walkRepeat(r, mult);
    return;
  }
  if (auto i = dyn_cast<qlx::cflow::IfOp>(op)) {
    walkIf(i, mult);
    return;
  }
  if (auto begin = dyn_cast<TransformBeginOp>(op)) {
    StringRef region = regionForValue(begin.getSource());
    if (!region.empty())
      mapValueToRegion(begin.getFrame(), region);
    return;
  }
  if (auto end = dyn_cast<TransformEndOp>(op)) {
    StringRef region = regionForValue(end.getFrame());
    if (!region.empty())
      mapValueToRegion(end.getDestination(), region);
    return;
  }
  if (auto retry = dyn_cast<RetryOp>(op)) {
    recordRetry(retry, mult);
    for (auto [carry, result] :
         llvm::zip(retry.getCarries(), retry.getResults())) {
      StringRef region = regionForValue(carry);
      if (!region.empty())
        mapValueToRegion(result, region);
    }
    return;
  }
  if (auto d = dyn_cast<DeallocOp>(op)) {
    // A patch becomes free here; decrement live concurrency (symmetric with
    // alloc). Still counted as a gate below.
    StringRef r = regionForValue(d.getPatch());
    if (!r.empty()) {
      int64_t &live = liveByRegion[r];
      if (live > 0)
        live -= 1;
    }
    if (totalLive > 0)
      totalLive -= 1;
    auto weight = logicalWeight(d.getPatch(), op);
    if (failed(weight))
      return;
    logicalLive = std::max<int64_t>(0, logicalLive - *weight);
    // fall through to gate-count handling (gate = "dealloc").
  }
  if (auto unpack = dyn_cast<EncodingUnpackOp>(op)) {
    mapValueToRegion(unpack.getChildren(), regionForValue(unpack.getParent()));
    return;
  }
  if (auto pack = dyn_cast<EncodingPackOp>(op)) {
    mapValueToRegion(pack.getParent(), regionForValue(pack.getChildren()));
    return;
  }
  if (op->getNumRegions() != 0) {
    op->emitOpError("fabric-count does not support this dynamic or "
                    "unrecognized region-bearing executable operation");
    hadError = true;
    return;
  }

  // Gate ops (BulkGate family and a few siblings). Order: most specific
  // first. Bumps gate count first, then propagates patch SSA values.
  StringRef gate;
  if (isa<HOp>(op))
    gate = "h";
  else if (isa<SOp>(op))
    gate = "s";
  else if (isa<SdgOp>(op))
    gate = "sdg";
  else if (isa<XOp>(op))
    gate = "x";
  else if (isa<ZOp>(op))
    gate = "z";
  else if (isa<TOp>(op))
    gate = "t";
  else if (isa<TdgOp>(op))
    gate = "tdg";
  else if (isa<ResetOp>(op))
    gate = "reset";
  else if (isa<InitBasisOp>(op))
    gate = "init_basis";
  else if (isa<CXOp>(op))
    gate = "cx";
  else if (isa<CZOp>(op))
    gate = "cz";
  else if (isa<TransversalCXOp>(op))
    gate = "transversal_cx";
  else if (isa<MzOp>(op))
    gate = "mz";
  else if (isa<MeasureBasisOp>(op))
    gate = "measure_basis";
  else if (isa<ReadSyndromeAncillasOp>(op))
    gate = "read_syndrome_ancillas";
  else if (isa<AssembleSyndromeOp>(op))
    gate = "assemble_syndrome";
  else if (isa<MergeOp>(op))
    gate = "merge";
  else if (isa<SplitOp>(op))
    gate = "split";
  else if (isa<MultiMeasureOp>(op))
    gate = "multi_measure";
  else if (isa<MeasureProductOp>(op))
    gate = "measure_product";
  else if (isa<RotateProductOp>(op))
    gate = "rotate_product";
  else if (isa<ResourceRotateProductOp>(op))
    gate = "resource_rotate_product";
  else if (isa<MoveOp>(op))
    gate = "move";
  else if (isa<PrepZOp>(op))
    gate = "prep_z";
  else if (isa<PrepXOp>(op))
    gate = "prep_x";
  else if (isa<DeallocOp>(op))
    gate = "dealloc";

  if (!gate.empty()) {
    if (auto tx = dyn_cast<TransversalCXOp>(op)) {
      // Bump on both endpoints (the op physically executes on both
      // patches) and record the directed cross-region edge.
      StringRef rc = regionForValue(tx.getCtrl());
      StringRef rt = regionForValue(tx.getTarg());
      auto bumpOne = [&](StringRef r) {
        if (r.empty())
          return;
        PerRegion &p = regionAccum(r);
        auto it = p.gateCounts.find(gate);
        if (it == p.gateCounts.end())
          p.gateCounts.insert({gate, mult});
        else
          checkedAdd(it->second, mult, op, "transversal gate count");
      };
      bumpOne(rc);
      if (!rt.empty() && rt != rc)
        bumpOne(rt);
      if (!rc.empty() && !rt.empty() && rc != rt)
        transversalEdges.push_back({rc, rt});
    } else {
      bumpGate(op, gate, mult);
    }

    propagatePatchValues(op);
    return;
  }

  if (isa<SuccessOp>(op)) {
    if (!suppressInlineAnalysis)
      checkedAdd(successCount, mult, op, "success count");
    return;
  }
  if (isa<TickOp>(op)) {
    // Tick is global within the gadget. Attribute to every region the
    // current valueToRegion map knows about — they all advance one
    // logical round on a tick.
    llvm::SmallSet<StringRef, 4> seen;
    seen.insert(tickContextRegions.begin(), tickContextRegions.end());
    for (const auto &[region, references] : mappedValuesByRegion)
      if (references > 0)
        if (auto live = liveByRegion.find(region);
            live != liveByRegion.end() && live->second > 0)
          seen.insert(region);
    for (StringRef r : seen) {
      // Skip region-less patches (e.g. scratch allocs with no region symbol
      // map to ""). Attributing ticks to the empty region would synthesise a
      // phantom "" entry in per_region — mirror the empty guard bumpGate()
      // already applies to gate counts.
      if (r.empty())
        continue;
      PerRegion &p = regionAccum(r);
      auto it = p.roundsByKind.find("tick");
      if (it == p.roundsByKind.end())
        p.roundsByKind.insert({"tick", mult});
      else
        checkedAdd(it->second, mult, op, "tick count");
    }
    return;
  }
  if (auto idle = dyn_cast<IdleOp>(op)) {
    auto rounds = checkedMultiply(mult, idle.getRounds(), op, "idle rounds");
    if (failed(rounds))
      return;
    bumpRoundFor(idle.getPatch(), "idle", *rounds);
    propagatePatchValues(op);
    return;
  }

  // Resource ops.
  if (auto pr = dyn_cast<ProduceResourceOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    PerRegion &p = regionAccum(pr.getRegion());
    // The production protocol is usually bound on the region (via
    // the selected factory definition, not on the produce_resource op itself.
    // Fall back to the region's protocol so factory demand is attributed
    // to a per_protocol production entry (otherwise p_factory would be 0).
    Attribute proto = pr.getProtocolAttr() ? pr.getProtocolAttr() : p.protocol;
    if (!p.protocol)
      p.protocol = proto;
    if (proto)
      recordProtocol(proto, "production", mult);
    return;
  }
  if (auto inj = dyn_cast<InjectOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    StringRef region = regionForValue(inj.getPatch());
    if (!region.empty())
      checkedAdd(regionAccum(region).injectCount, mult, op, "injection count");
    recordProtocol(inj.getProtocolAttr(), "injection", mult);
    mapValueToRegion(inj.getResult(), region);
    return;
  }
  if (auto tp = dyn_cast<TransportOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    checkedAdd(regionAccum(tp.getSrcRegion()).transportOut, mult, op,
               "transport count");
    checkedAdd(regionAccum(tp.getDstRegion()).transportIn, mult, op,
               "transport count");
    recordProtocol(tp.getProtocolAttr(), "transport", mult);
    return;
  }
  if (isa<DiscardResourceOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    // No region/protocol attribution; discards are summary-side only.
    return;
  }
  if (auto request = dyn_cast<ResourceRequestOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    recordResourceRequest(request, mult);
    propagatePatchValues(op);
    return;
  }
  if (auto unpack = dyn_cast<UnpackResourceOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    unsigned count = unpack.getAnchors().size();
    if (unpack.getOutputs().size() != count * 2) {
      unpack.emitOpError(
          "fabric-count requires one successor and payload per anchor");
      hadError = true;
      return;
    }
    for (unsigned index = 0; index < count; ++index) {
      StringRef region = regionForValue(unpack.getAnchors()[index]);
      if (!region.empty()) {
        mapValueToRegion(unpack.getOutputs()[index], region);
        mapValueToRegion(unpack.getOutputs()[count + index], region);
        int64_t &live = liveByRegion[region];
        if (!checkedAdd(live, 1, op, "unpacked live-patch count"))
          return;
        PerRegion &perRegion = regionAccum(region);
        perRegion.patches = std::max(perRegion.patches, live);
      }
      if (!checkedAdd(totalLive, 1, op, "unpacked live-patch count"))
        return;
      patchPeak = std::max(patchPeak, totalLive);
      Value payload = unpack.getOutputs()[count + index];
      auto weight = logicalWeight(payload, op);
      if (failed(weight) ||
          !checkedAdd(logicalLive, *weight, op, "unpacked logical-qubit count"))
        return;
      logicalPeak = std::max(logicalPeak, logicalLive);
    }
    return;
  }
  if (auto pack = dyn_cast<PackResourceOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    for (Value payload : pack.getPayloads()) {
      StringRef region = regionForValue(payload);
      if (!region.empty()) {
        int64_t &live = liveByRegion[region];
        if (live > 0)
          live -= 1;
      }
      if (totalLive > 0)
        totalLive -= 1;
      auto weight = logicalWeight(payload, op);
      if (failed(weight))
        return;
      logicalLive = std::max<int64_t>(0, logicalLive - *weight);
    }
    return;
  }
  if (isa<qlx::event::SelectionOp>(op)) {
    requiresExtendedAnalyticalModel = true;
    propagatePatchValues(op);
    return;
  }

  // Classical frame bookkeeping has no Tier-1 quantum cost. Keeping
  // this exemption typed makes the closed native-v1 support boundary explicit.
  if (isa<FrameCreateOp, FrameUpdateOp, FrameTransformOp, FrameInitOp,
          FramePropagateOp, FrameResolveOp, qlx::event::TestOp,
          qlx::event::PollOp, qlx::event::IsOp, qlx::event::SelectReadyOp,
          qlx::event::CancelOp, qlx::event::AwaitOp, qlx::event::FenceOp,
          SendOp, RecvOp, BarrierOp, XorOp, AllZeroOp, ParityOp, AllFalseOp>(
          op)) {
    propagatePatchValues(op);
    return;
  }

  bool touchesFabricSemantics =
      llvm::any_of(op->getOperandTypes(), [](Type type) {
        return type.getDialect().getNamespace() == "fabric";
      });
  touchesFabricSemantics |= llvm::any_of(op->getResultTypes(), [](Type type) {
    return type.getDialect().getNamespace() == "fabric";
  });
  if ((operationName.starts_with("fabric.") || touchesFabricSemantics) &&
      !isa<ReturnOp, ProtocolReturnOp, qlx::cflow::YieldOp, qlx::event::YieldOp,
           ProfileEndOp>(op)) {
    op->emitOpError(
        "fabric-count native v1 has no typed cost semantics for this "
        "executable operation");
    hadError = true;
    return;
  }

  // Non-Fabric classical bookkeeping and supported terminators do not carry
  // Tier-1 quantum-resource cost.
  propagatePatchValues(op);
}

void Walker::walkCall(CallOp call, int64_t mult) {
  bool priorSuppression = suppressInlineAnalysis;
  if (call.getProfileAttr())
    suppressInlineAnalysis = true;
  walkDelegation(call, call.getCalleeAttr(), mult, mult);
  suppressInlineAnalysis = priorSuppression;
  if (hadError || !call.getProfileAttr())
    return;
  auto profile =
      symTab.lookup<GadgetProfileOp>(call.getProfileAttr().getValue());
  if (!profile || profile.getBody().empty()) {
    call.emitOpError("fabric-count cannot resolve selected profile ")
        << call.getProfileAttr();
    hadError = true;
    return;
  }
  suppressInlineAnalysis = false;
  walkBlock(profile.getBody().front(), mult);
  suppressInlineAnalysis = priorSuppression;
}

CallableSummaryKey Walker::callableSummaryKey(Operation *callee,
                                              Operation *wrapper) const {
  CallableSummaryKey key;
  key.callee = callee;
  key.suppressInlineAnalysis = suppressInlineAnalysis;
  key.inputRegions.reserve(wrapper->getNumOperands());
  for (Value operand : wrapper->getOperands())
    key.inputRegions.push_back(regionForValue(operand));
  key.liveRegions.reserve(liveByRegion.size());
  for (const auto &[region, live] : liveByRegion)
    if (live != 0)
      key.liveRegions.emplace_back(region, live);
  llvm::sort(key.liveRegions, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });
  llvm::SmallSet<StringRef, 8> tickRegions = tickContextRegions;
  for (const auto &[region, references] : mappedValuesByRegion) {
    auto live = liveByRegion.find(region);
    if (references > 0 && !region.empty() && live != liveByRegion.end() &&
        live->second > 0)
      tickRegions.insert(region);
  }
  llvm::append_range(key.tickRegions, tickRegions);
  llvm::sort(key.tickRegions);
  key.totalLive = totalLive;
  key.logicalLive = logicalLive;
  return key;
}

FailureOr<CachedCallableSummary>
Walker::buildCallableSummary(Operation *callee, Operation *wrapper,
                             const CallableSummaryKey &key) {
  ++callableSummaryProbes;
  Walker probe(ctx, symTab, selectedQec);
  probe.enableCallableSummaries = false;
  probe.suppressInlineAnalysis = key.suppressInlineAnalysis;
  probe.distanceByCode = distanceByCode;
  probe.logicalsByCode = logicalsByCode;
  probe.selectedRegions = selectedRegions;
  probe.retryAttemptCache = retryAttemptCache;
  probe.tickContextRegions.insert(key.tickRegions.begin(),
                                  key.tickRegions.end());
  probe.totalLive = key.totalLive;
  probe.patchPeak = key.totalLive;
  probe.logicalLive = key.logicalLive;
  probe.logicalPeak = key.logicalLive;
  for (const auto &[region, live] : key.liveRegions)
    probe.liveByRegion[region] = live;
  for (const auto &[name, source] : perRegion) {
    PerRegion declaration = source;
    declaration.patches = probe.liveByRegion.lookup(name);
    declaration.gateCounts.clear();
    declaration.roundsByKind.clear();
    declaration.injectCount = 0;
    declaration.transportIn = 0;
    declaration.transportOut = 0;
    probe.perRegion.insert({name, std::move(declaration)});
  }

  auto calleeName = cast<StringAttr>(SymbolTable::getSymbolName(callee));
  probe.activeCalls.insert(calleeName.getValue());
  Block *body = probe.resolveExecutableBody(callee, wrapper);
  if (!body)
    return failure();
  for (auto [argument, region] :
       llvm::zip(body->getArguments(), key.inputRegions))
    if (!region.empty())
      probe.mapValueToRegion(argument, region);
  probe.walkBlock(*body, /*mult=*/1);

  visitedOperations += probe.visitedOperations;
  retryProbes += probe.retryProbes;
  retryProbeCacheHits += probe.retryProbeCacheHits;
  retryProbeOperations += probe.retryProbeOperations;
  retryProbeSeconds += probe.retryProbeSeconds;
  retryAttemptCache = std::move(probe.retryAttemptCache);
  if (probe.hadError) {
    hadError = true;
    return failure();
  }

  CachedCallableSummary result;
  result.reusable =
      probe.totalLive == key.totalLive && probe.logicalLive == key.logicalLive;
  if (result.reusable) {
    for (const auto &[region, live] : key.liveRegions)
      if (probe.liveByRegion.lookup(region) != live) {
        result.reusable = false;
        break;
      }
    if (result.reusable)
      for (const auto &[region, live] : probe.liveByRegion)
        if (live != 0 &&
            !llvm::is_contained(key.liveRegions,
                                std::pair<StringRef, int64_t>{region, live})) {
          result.reusable = false;
          break;
        }
  }
  if (!result.reusable)
    return result;

  CallableSummary &summary = result.summary;
  summary.perRegion = std::move(probe.perRegion);
  summary.perProtocol = std::move(probe.perProtocol);
  summary.operationCounts = std::move(probe.operationCounts);
  summary.gadgetCalls = std::move(probe.gadgetCalls);
  summary.protocolCalls = std::move(probe.protocolCalls);
  summary.resourceRequests = std::move(probe.resourceRequests);
  summary.retryDemands = std::move(probe.retryDemands);
  summary.transversalEdges = std::move(probe.transversalEdges);
  summary.successCount = probe.successCount;
  summary.syndromeRounds = probe.syndromeRounds;
  summary.patchPeak = probe.patchPeak;
  summary.logicalPeak = probe.logicalPeak;
  summary.requiresExtendedAnalyticalModel =
      probe.requiresExtendedAnalyticalModel;
  summary.sawQecSpec = probe.sawQecSpec;
  summary.sawSelectedDeviceGadget = probe.sawSelectedDeviceGadget;
  summary.sawSelectedDeviceRegionUse = probe.sawSelectedDeviceRegionUse;
  summary.operationSites = probe.visitedOperations;
  if (isa<ReturnOp, ProtocolReturnOp>(body->getTerminator()))
    for (Value returned : body->getTerminator()->getOperands())
      summary.outputRegions.push_back(probe.regionForValue(returned));
  if (summary.outputRegions.size() != wrapper->getNumResults()) {
    wrapper->emitOpError(
        "fabric-count callable summary result boundary is malformed");
    hadError = true;
    return failure();
  }
  return result;
}

LogicalResult Walker::applyCallableSummary(Operation *wrapper,
                                           const CallableSummary &summary,
                                           int64_t multiplicity) {
  auto scale = [&](int64_t value, StringRef what) -> FailureOr<int64_t> {
    return checkedMultiply(value, multiplicity, wrapper, what);
  };
  auto mergeMap = [&](std::map<std::string, int64_t> &target,
                      const std::map<std::string, int64_t> &source,
                      StringRef what) -> LogicalResult {
    for (const auto &[name, value] : source) {
      auto scaled = scale(value, what);
      if (failed(scaled))
        return failure();
      bump(target, name, *scaled, wrapper, what);
    }
    return success();
  };
  auto mergeNamedCounts = [&](llvm::MapVector<StringRef, int64_t> &target,
                              const llvm::MapVector<StringRef, int64_t> &source,
                              StringRef what) -> LogicalResult {
    for (const auto &[name, value] : source) {
      auto scaled = scale(value, what);
      if (failed(scaled))
        return failure();
      auto found = target.find(name);
      if (found == target.end())
        target.insert({name, *scaled});
      else if (!checkedAdd(found->second, *scaled, wrapper, what))
        return failure();
    }
    return success();
  };
  auto mergeScalar = [&](int64_t &target, int64_t value,
                         StringRef what) -> LogicalResult {
    auto scaled = scale(value, what);
    if (failed(scaled) || !checkedAdd(target, *scaled, wrapper, what))
      return failure();
    return success();
  };

  if (failed(mergeMap(operationCounts, summary.operationCounts,
                      "operation count")) ||
      failed(mergeMap(gadgetCalls, summary.gadgetCalls, "gadget-call count")) ||
      failed(mergeMap(protocolCalls, summary.protocolCalls,
                      "protocol-call count")))
    return failure();
  for (const auto &[name, source] : summary.perProtocol) {
    auto scaled = scale(source.opCount, "protocol count");
    if (failed(scaled))
      return failure();
    auto found = perProtocol.find(name);
    if (found == perProtocol.end()) {
      perProtocol.insert({name, ProtocolEntry{source.kind, *scaled}});
    } else if (!checkedAdd(found->second.opCount, *scaled, wrapper,
                           "protocol count")) {
      return failure();
    }
  }
  for (const auto &[name, source] : summary.perRegion) {
    PerRegion &target = regionAccum(name);
    target.patches = std::max(target.patches, source.patches);
    if (failed(mergeNamedCounts(target.gateCounts, source.gateCounts,
                                "gate count")) ||
        failed(mergeNamedCounts(target.roundsByKind, source.roundsByKind,
                                "round count")) ||
        failed(mergeScalar(target.injectCount, source.injectCount,
                           "injection count")) ||
        failed(mergeScalar(target.transportIn, source.transportIn,
                           "transport count")) ||
        failed(mergeScalar(target.transportOut, source.transportOut,
                           "transport count")))
      return failure();
  }
  patchPeak = std::max(patchPeak, summary.patchPeak);
  logicalPeak = std::max(logicalPeak, summary.logicalPeak);
  if (failed(
          mergeScalar(successCount, summary.successCount, "success count")) ||
      failed(mergeScalar(syndromeRounds, summary.syndromeRounds,
                         "syndrome-round count")))
    return failure();

  for (const ResourceDemand &source : summary.resourceRequests) {
    auto scaled = scale(source.count, "resource-request count");
    if (failed(scaled))
      return failure();
    auto found =
        llvm::find_if(resourceRequests, [&](const ResourceDemand &row) {
          return row.kind == source.kind && row.stream == source.stream;
        });
    if (found == resourceRequests.end())
      resourceRequests.push_back(
          ResourceDemand{source.kind, source.stream, *scaled});
    else if (!checkedAdd(found->count, *scaled, wrapper,
                         "resource-request count"))
      return failure();
  }
  for (const RetryDemandSummary &source : summary.retryDemands) {
    auto occurrences = scale(source.occurrences, "retry occurrence count");
    if (failed(occurrences))
      return failure();
    RetryDemandSummary row = source;
    row.occurrences = *occurrences;
    retryDemands.push_back(std::move(row));
  }
  llvm::append_range(transversalEdges, summary.transversalEdges);
  requiresExtendedAnalyticalModel |= summary.requiresExtendedAnalyticalModel;
  sawQecSpec |= summary.sawQecSpec;
  sawSelectedDeviceGadget |= summary.sawSelectedDeviceGadget;
  sawSelectedDeviceRegionUse |= summary.sawSelectedDeviceRegionUse;
  for (auto [result, region] :
       llvm::zip(wrapper->getResults(), summary.outputRegions))
    if (!region.empty())
      mapValueToRegion(result, region);
  return success();
}

void Walker::walkDelegation(Operation *wrapper, FlatSymbolRefAttr calleeAttr,
                            int64_t wrapperMultiplicity,
                            int64_t calleeMultiplicity) {
  ++delegations;
  Operation *callee = symTab.lookup(calleeAttr.getValue());
  if (!callee || !isa<GadgetOp, ProtocolOp>(callee)) {
    wrapper->emitOpError("fabric-count cannot resolve executable callee ")
        << calleeAttr;
    hadError = true;
    return;
  }
  Block *body = resolveExecutableBody(callee, wrapper);
  if (!body)
    return;
  auto calleeName = cast<StringAttr>(SymbolTable::getSymbolName(callee));
  if (auto gadget = dyn_cast<GadgetOp>(callee);
      gadget && gadget.getDeviceAttr()) {
    if (selectedQec && gadget.getDeviceAttr() != selectedQec) {
      wrapper->emitOpError(
          "fabric-count executable callee belongs to another P2 machine");
      hadError = true;
      return;
    }
    sawSelectedDeviceGadget = true;
  }
  if (!activeCalls.insert(calleeName.getValue()).second) {
    wrapper->emitOpError(
        "fabric-count rejects recursive executable call through @")
        << calleeName.getValue();
    hadError = true;
    return;
  }
  if (isa<GadgetOp>(callee))
    bump(gadgetCalls, calleeName.getValue(), calleeMultiplicity, wrapper,
         "gadget-call count");
  else
    bump(protocolCalls, calleeName.getValue(), calleeMultiplicity, wrapper,
         "protocol-call count");

  if (enableCallableSummaries) {
    CallableSummaryKey key = callableSummaryKey(callee, wrapper);
    auto cached = callableSummaryCache.find(key);
    bool cacheHit = cached != callableSummaryCache.end();
    if (!cacheHit) {
      auto built = buildCallableSummary(callee, wrapper, key);
      if (failed(built)) {
        activeCalls.erase(calleeName.getValue());
        return;
      }
      cached =
          callableSummaryCache.emplace(std::move(key), std::move(*built)).first;
    }
    if (cached->second.reusable) {
      ++callableSummaryHits;
      if (cacheHit)
        callableSummarySavedOperations += cached->second.summary.operationSites;
      if (failed(applyCallableSummary(wrapper, cached->second.summary,
                                      calleeMultiplicity))) {
        activeCalls.erase(calleeName.getValue());
        return;
      }
      activeCalls.erase(calleeName.getValue());
      return;
    }
    if (cacheHit)
      ++callableSummaryNegativeHits;
  }

  for (auto [arg, operand] :
       llvm::zip(body->getArguments(), wrapper->getOperands())) {
    StringRef r = regionForValue(operand);
    if (!r.empty())
      mapValueToRegion(arg, r);
  }

  walkBlock(*body, calleeMultiplicity);

  // After recursion, the callee's return op's operands carry region
  // attribution. Map them onto the call's results.
  if (!body->empty()) {
    if (isa<ReturnOp, ProtocolReturnOp>(body->getTerminator())) {
      for (auto [resVal, retVal] : llvm::zip(
               wrapper->getResults(), body->getTerminator()->getOperands())) {
        StringRef r = regionForValue(retVal);
        if (!r.empty())
          mapValueToRegion(resVal, r);
      }
    }
  }

  eraseRegionValueMappings(*body->getParent());
  activeCalls.erase(calleeName.getValue());
  (void)wrapperMultiplicity;
}

FailureOr<int64_t> Walker::mapChildrenMultiplicity(MapChildrenOp map,
                                                   int64_t multiplier) {
  Value bundle = map.getChildren();
  llvm::SmallPtrSet<Operation *, 8> seen;
  FlatSymbolRefAttr hierarchyRef;
  while (Operation *owner = bundle.getDefiningOp()) {
    if (!seen.insert(owner).second)
      break;
    if (auto unpack = dyn_cast<EncodingUnpackOp>(owner)) {
      hierarchyRef = unpack.getHierarchyAttr();
      break;
    }
    auto prior = dyn_cast<MapChildrenOp>(owner);
    if (!prior)
      break;
    bundle = prior.getChildren();
  }
  auto hierarchy =
      hierarchyRef ? symTab.lookup<EncodingHierarchyOp>(hierarchyRef.getValue())
                   : EncodingHierarchyOp{};
  if (!hierarchy || hierarchy.getMultiplicity() <= 0) {
    map.emitOpError(
        "fabric-count mapped children require a positive typed hierarchy");
    hadError = true;
    return failure();
  }
  return checkedMultiply(multiplier, hierarchy.getMultiplicity(), map,
                         "child hierarchy multiplicity");
}

void Walker::walkRepeat(qlx::cflow::RepeatOp rep, int64_t mult) {
  if (rep.getBody().empty())
    return;
  Block &body = rep.getBody().front();
  int64_t count = rep.getCount();

  if (count == 0) {
    for (auto [result, init] : llvm::zip(rep.getResults(), rep.getInits())) {
      StringRef region = regionForValue(init);
      if (!region.empty())
        mapValueToRegion(result, region);
    }
    return;
  }

  // Map iter-args to their init values' regions for the first iteration.
  for (auto [arg, init] : llvm::zip(body.getArguments(), rep.getInits())) {
    if (isPatchLike(arg.getType()) && isPatchLike(init.getType())) {
      StringRef r = regionForValue(init);
      if (!r.empty())
        mapValueToRegion(arg, r);
    }
  }

  int64_t foldedMultiplicity = 0;
  if (llvm::MulOverflow(mult, count, foldedMultiplicity)) {
    rep.emitOpError("fabric-count folded repeat multiplicity overflows i64");
    hadError = true;
    return;
  }
  // Walk the body once with mult * count: gate counts multiply by count.
  // Rounds inside the body (tick, idle, syndrome) multiply too.
  walkBlock(body, foldedMultiplicity);

  // Map yielded values back to the repeat's results so downstream ops
  // continue with proper attribution.
  if (auto yield = dyn_cast<qlx::cflow::YieldOp>(body.getTerminator())) {
    for (auto [resVal, yldVal] :
         llvm::zip(rep.getResults(), yield.getOperands())) {
      if (isPatchLike(resVal.getType())) {
        StringRef r = regionForValue(yldVal);
        if (!r.empty())
          mapValueToRegion(resVal, r);
      }
    }
  }

  eraseRegionValueMappings(rep.getBody());
}

void Walker::walkIf(qlx::cflow::IfOp ifop, int64_t mult) {
  auto yieldedOperands = [](Region &region) -> SmallVector<Value> {
    if (region.empty())
      return {};
    Operation *term = region.front().getTerminator();
    if (auto yield = dyn_cast_or_null<qlx::cflow::YieldOp>(term))
      return SmallVector<Value>(yield.getOperands().begin(),
                                yield.getOperands().end());
    return {};
  };

  auto beforeValues = valueToRegion;
  auto beforeMappedValues = mappedValuesByRegion;
  auto beforeLive = liveByRegion;
  int64_t beforeTotal = totalLive;
  int64_t beforeLogical = logicalLive;

  if (!ifop.getThenRegion().empty())
    walkBlock(ifop.getThenRegion().front(), mult);
  auto thenValues = valueToRegion;
  auto thenLive = liveByRegion;
  int64_t thenTotal = totalLive;
  int64_t thenLogical = logicalLive;
  int64_t branchPeak = logicalPeak;

  valueToRegion = beforeValues;
  mappedValuesByRegion = beforeMappedValues;
  liveByRegion = beforeLive;
  totalLive = beforeTotal;
  logicalLive = beforeLogical;
  if (!ifop.getElseRegion().empty())
    walkBlock(ifop.getElseRegion().front(), mult);
  auto elseValues = valueToRegion;
  auto elseLive = liveByRegion;
  int64_t elseTotal = totalLive;
  int64_t elseLogical = logicalLive;
  logicalPeak = std::max(logicalPeak, branchPeak);

  if (thenTotal != elseTotal || thenLogical != elseLogical ||
      thenLive != elseLive) {
    ifop.emitOpError(
        "fabric-count requires branches to agree on live patch ownership");
    hadError = true;
    return;
  }
  liveByRegion = thenLive;
  totalLive = thenTotal;
  logicalLive = thenLogical;
  valueToRegion = beforeValues;
  mappedValuesByRegion = beforeMappedValues;

  SmallVector<Value> thenYield = yieldedOperands(ifop.getThenRegion());
  SmallVector<Value> elseYield = yieldedOperands(ifop.getElseRegion());
  for (auto it : llvm::enumerate(ifop.getResults())) {
    Value result = it.value();
    if (!isPatchLike(result.getType()))
      continue;

    StringRef region;
    if (it.index() < thenYield.size())
      if (auto found = thenValues.find(thenYield[it.index()]);
          found != thenValues.end())
        region = found->second;
    StringRef other;
    if (it.index() < elseYield.size())
      if (auto found = elseValues.find(elseYield[it.index()]);
          found != elseValues.end())
        other = found->second;
    if (region != other) {
      ifop.emitOpError("fabric-count branch join changes patch region");
      hadError = true;
      return;
    }
    if (!region.empty())
      mapValueToRegion(result, region);
  }
}

//===----------------------------------------------------------------------===//
// DictionaryAttr builder
//===----------------------------------------------------------------------===//

static StringAttr s(MLIRContext *ctx, StringRef v) {
  return StringAttr::get(ctx, v);
}
static IntegerAttr i(MLIRContext *ctx, int64_t v) {
  return IntegerAttr::get(IntegerType::get(ctx, 64), v);
}

static DictionaryAttr buildCounts(MLIRContext *ctx, const Walker &w,
                                  ArrayAttr sourceFacets) {
  // Sort gate counts alphabetically and rounds by kind alphabetically for
  // stable FileCheck output.
  auto sortMap = [](const llvm::MapVector<StringRef, int64_t> &m) {
    llvm::SmallVector<std::pair<StringRef, int64_t>> out(m.begin(), m.end());
    llvm::sort(out,
               [](const auto &a, const auto &b) { return a.first < b.first; });
    return out;
  };
  auto countMap = [&](const std::map<std::string, int64_t> &counts) {
    SmallVector<NamedAttribute> fields;
    for (const auto &[name, count] : counts)
      fields.push_back({s(ctx, name), i(ctx, count)});
    return DictionaryAttr::get(ctx, fields);
  };

  // per_region
  llvm::SmallVector<NamedAttribute> perRegionFields;
  for (auto &kv : w.perRegion) {
    // Never serialise an empty-named region: a DictionaryAttr key of "" is
    // both meaningless (no such region exists) and unreadable by the MLIR
    // Python bindings (NamedAttribute.name throws on a null name).
    if (kv.first.empty())
      continue;
    const PerRegion &r = kv.second;
    llvm::SmallVector<NamedAttribute> gateFields;
    for (auto &g : sortMap(r.gateCounts))
      gateFields.push_back({s(ctx, g.first), i(ctx, g.second)});
    llvm::SmallVector<NamedAttribute> roundFields;
    for (auto &g : sortMap(r.roundsByKind))
      roundFields.push_back({s(ctx, g.first), i(ctx, g.second)});

    llvm::SmallVector<NamedAttribute> regionFields = {
        {s(ctx, "role"), s(ctx, r.role)},
        {s(ctx, "code"), s(ctx, r.code)},
        {s(ctx, "distance"), i(ctx, r.distance)},
        {s(ctx, "patches"), i(ctx, r.patches)},
        {s(ctx, "gate_counts"), DictionaryAttr::get(ctx, gateFields)},
        {s(ctx, "rounds_by_kind"), DictionaryAttr::get(ctx, roundFields)},
        {s(ctx, "inject_count"), i(ctx, r.injectCount)},
        {s(ctx, "transport_in"), i(ctx, r.transportIn)},
        {s(ctx, "transport_out"), i(ctx, r.transportOut)},
    };
    perRegionFields.push_back(
        {s(ctx, kv.first), DictionaryAttr::get(ctx, regionFields)});
  }

  // per_protocol
  llvm::SmallVector<NamedAttribute> perProtoFields;
  for (auto &kv : w.perProtocol) {
    llvm::SmallVector<NamedAttribute> entry = {
        {s(ctx, "kind"), s(ctx, kv.second.kind)},
        {s(ctx, "op_count"), i(ctx, kv.second.opCount)},
    };
    perProtoFields.push_back(
        {s(ctx, kv.first), DictionaryAttr::get(ctx, entry)});
  }

  // transversal_edges
  llvm::SmallVector<Attribute> edgeAttrs;
  for (auto &e : w.transversalEdges) {
    edgeAttrs.push_back(
        ArrayAttr::get(ctx, {s(ctx, e.first), s(ctx, e.second)}));
  }

  llvm::SmallVector<Attribute> resourceRequestAttrs;
  for (const ResourceDemand &demand : w.resourceRequests) {
    resourceRequestAttrs.push_back(DictionaryAttr::get(
        ctx, {
                 NamedAttribute(s(ctx, "kind"), demand.kind),
                 NamedAttribute(s(ctx, "stream"), demand.stream),
                 NamedAttribute(s(ctx, "count"), i(ctx, demand.count)),
             }));
  }

  llvm::SmallVector<Attribute> retryDemandAttrs;
  for (const RetryDemandSummary &retry : w.retryDemands) {
    llvm::SmallVector<Attribute> perAttemptRequests;
    for (const ResourceDemand &demand : retry.resourceRequests) {
      perAttemptRequests.push_back(DictionaryAttr::get(
          ctx, {
                   NamedAttribute(s(ctx, "kind"), demand.kind),
                   NamedAttribute(s(ctx, "stream"), demand.stream),
                   NamedAttribute(s(ctx, "count"), i(ctx, demand.count)),
               }));
    }
    llvm::SmallVector<NamedAttribute> fields = {
        {s(ctx, "attempt"), retry.attempt},
        {s(ctx, "exhaustion"), retry.exhaustion},
        {s(ctx, "max_attempts"), i(ctx, retry.maxAttempts)},
        {s(ctx, "occurrences"), i(ctx, retry.occurrences)},
        {s(ctx, "attempt_operation_sites"),
         i(ctx, retry.attemptOperationSites)},
        {s(ctx, "resource_requests_per_attempt"),
         ArrayAttr::get(ctx, perAttemptRequests)},
    };
    if (retry.profile)
      fields.emplace_back(s(ctx, "profile"), retry.profile);
    if (retry.probabilitySource)
      fields.emplace_back(s(ctx, "success_probability_source"),
                          retry.probabilitySource);
    if (retry.probabilityEvidence)
      fields.emplace_back(s(ctx, "success_probability_evidence"),
                          retry.probabilityEvidence);
    if (retry.successProbability)
      fields.emplace_back(
          s(ctx, "success_probability"),
          FloatAttr::get(Float64Type::get(ctx), *retry.successProbability));
    retryDemandAttrs.push_back(DictionaryAttr::get(ctx, fields));
  }

  llvm::SmallVector<NamedAttribute> root = {
      {s(ctx, "operation_counts"), countMap(w.operationCounts)},
      {s(ctx, "gadget_calls"), countMap(w.gadgetCalls)},
      {s(ctx, "protocol_calls"), countMap(w.protocolCalls)},
      {s(ctx, "hierarchy_depths"), countMap(w.hierarchyDepths)},
      {s(ctx, "success_count"), i(ctx, w.successCount)},
      {s(ctx, "syndrome_rounds"), i(ctx, w.syndromeRounds)},
      {s(ctx, "patches_peak"), i(ctx, w.patchPeak)},
      {s(ctx, "logical_qubits_peak"), i(ctx, w.logicalPeak)},
      {s(ctx, "source_stage"), s(ctx, "p2")},
      {s(ctx, "source_facets"), sourceFacets},
      {s(ctx, "requires_extended_analytical_model"),
       BoolAttr::get(ctx, w.requiresExtendedAnalyticalModel)},
      {s(ctx, "resource_requests"), ArrayAttr::get(ctx, resourceRequestAttrs)},
      {s(ctx, "per_region"), DictionaryAttr::get(ctx, perRegionFields)},
      {s(ctx, "per_protocol"), DictionaryAttr::get(ctx, perProtoFields)},
      {s(ctx, "transversal_edges"), ArrayAttr::get(ctx, edgeAttrs)},
  };
  if (!retryDemandAttrs.empty())
    root.emplace_back(s(ctx, "retry_demands"),
                      ArrayAttr::get(ctx, retryDemandAttrs));
  return DictionaryAttr::get(ctx, root);
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct FabricCountPass
    : public qlx::fabric::impl::FabricCountBase<FabricCountPass> {
  using FabricCountBase::FabricCountBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTable symTab(module);

    // Phase 1: resolve exactly one selected executable root.  A missing root
    // is an incompatible-stage error, never a valid all-zero estimate.
    Operation *entry = nullptr;
    if (!rootSymbol.empty()) {
      entry = symTab.lookup(rootSymbol);
      if (!entry) {
        module.emitError("fabric-count root @")
            << rootSymbol << " does not resolve to a top-level symbol";
        return signalPassFailure();
      }
      if (!isa<GadgetOp, ProtocolOp, GadgetProfileOp>(entry)) {
        module.emitError("fabric-count root @")
            << rootSymbol << " resolves to " << entry->getName()
            << ", not an executable Fabric root";
        return signalPassFailure();
      }
    } else {
      for (auto g : module.getOps<GadgetOp>()) {
        if (!g.getEntryAttr())
          continue;
        if (entry) {
          module.emitError(
              "fabric-count requires root= when several entry gadgets exist");
          return signalPassFailure();
        }
        entry = g.getOperation();
      }
      if (!entry) {
        module.emitError("fabric-count requires one selected entry "
                         "fabric.gadget; input is not selected P2 Fabric");
        return signalPassFailure();
      }
    }
    if (!deviceSymbol.empty() && resultSymbol.empty()) {
      module.emitError("fabric-count result symbol must be nonempty");
      return signalPassFailure();
    }

    qlx::DeviceOp device;
    if (!deviceSymbol.empty())
      device = dyn_cast_or_null<qlx::DeviceOp>(symTab.lookup(deviceSymbol));
    if (!deviceSymbol.empty() && !device) {
      module.emitError("fabric-count device @")
          << deviceSymbol << " must resolve to qlx.device";
      return signalPassFailure();
    }
    auto gadgetEntry = dyn_cast<GadgetOp>(entry);
    if (auto profile = dyn_cast<GadgetProfileOp>(entry))
      gadgetEntry = symTab.lookup<GadgetOp>(profile.getGadgetAttr().getValue());
    if (device && (!device.getQecAttr() ||
                   (gadgetEntry &&
                    (!gadgetEntry.getDeviceAttr() ||
                     device.getQecAttr() != gadgetEntry.getDeviceAttr())))) {
      module.emitError("fabric-count selected device and root name different "
                       "P2 machines");
      return signalPassFailure();
    }
    if (device && symTab.lookup(resultSymbol)) {
      module.emitError("fabric-count result symbol already exists: @")
          << resultSymbol;
      return signalPassFailure();
    }

    // Phase 2: count only the explicitly selected P2 implementation closure.
    Walker w(ctx, symTab, device ? device.getQecAttr() : FlatSymbolRefAttr{});
    for (auto code : module.getOps<CodeOp>())
      w.recordCodeDecl(code);
    for (auto hierarchy : module.getOps<EncodingHierarchyOp>())
      w.hierarchyDepths[hierarchy.getSymName().str()] = hierarchy.getDepth();
    if (device) {
      auto selectedMachine =
          symTab.lookup<DeviceOp>(device.getQecAttr().getValue());
      if (!selectedMachine) {
        device.emitOpError("selected QEC machine is unresolved");
        return signalPassFailure();
      }
      for (auto region : selectedMachine.getOps<RegionOp>())
        w.recordRegionDecl(region);
    } else {
      for (auto machine : module.getOps<DeviceOp>())
        for (auto region : machine.getOps<RegionOp>())
          w.recordRegionDecl(region);
    }

    w.walkEntry(entry);
    if (w.hadError) {
      signalPassFailure();
      return;
    }
    if (std::getenv("QLX_PROFILE_SCHEDULE_ESTIMATE"))
      w.printProfile(llvm::errs());
    auto protocolEntry = dyn_cast<ProtocolOp>(entry);
    bool selectedP2Protocol = false;
    if (device && protocolEntry) {
      DictionaryAttr metadata = protocolEntry.getMetadataAttr();
      auto selectedDevice =
          metadata ? metadata.getAs<StringAttr>("device") : StringAttr{};
      auto inputP1 =
          metadata ? metadata.getAs<StringAttr>("input_p1") : StringAttr{};
      auto selection = metadata
                           ? metadata.getAs<StringAttr>("qec_selection_sha256")
                           : StringAttr{};
      selectedP2Protocol = selectedDevice && inputP1 && selection &&
                           selectedDevice.getValue() == device.getSymName();
    }
    if (device && protocolEntry && !w.sawSelectedDeviceGadget &&
        !w.sawSelectedDeviceRegionUse && !selectedP2Protocol) {
      protocolEntry.emitOpError(
          "protocol root has no executable gadget or allocation region bound "
          "to the selected P2 machine");
      return signalPassFailure();
    }

    // Phase 3: retain the legacy annotation during migration and also emit the
    // symbol-addressable, versioned result consumed by higher tiers.
    OpBuilder builder(ctx);
    SmallVector<Attribute> sourceFacets;
    if (w.sawQecSpec)
      sourceFacets.push_back(builder.getStringAttr("qec_spec"));
    if (isa<GadgetOp>(entry) || w.sawSelectedDeviceGadget ||
        !w.gadgetCalls.empty())
      sourceFacets.push_back(builder.getStringAttr("qec_realization"));
    if (isa<ProtocolOp>(entry) || !w.protocolCalls.empty())
      sourceFacets.push_back(builder.getStringAttr("protocol_network"));
    auto entryName =
        cast<StringAttr>(SymbolTable::getSymbolName(entry)).getValue();
    bool hasPatchGraph =
        llvm::any_of(module.getOps<PatchGraphOp>(), [&](PatchGraphOp graph) {
          return graph.getRootAttr().getValue() == entryName;
        });
    if (hasPatchGraph)
      sourceFacets.push_back(builder.getStringAttr("patch_graph"));
    DictionaryAttr counts =
        buildCounts(ctx, w, builder.getArrayAttr(sourceFacets));
    module->setAttr("fabric.counts", counts);
    if (!device)
      return;
    builder.setInsertionPointToEnd(module.getBody());
    auto metadata = builder.getDictionaryAttr({
        builder.getNamedAttr("producer", builder.getStringAttr("fabric-count")),
        builder.getNamedAttr("producer_version", builder.getStringAttr("1")),
    });
    qlx::EstimateResultOp::create(
        builder, entry->getLoc(), builder.getStringAttr(resultSymbol),
        builder.getStringAttr("static"),
        FlatSymbolRefAttr::get(
            ctx,
            cast<StringAttr>(SymbolTable::getSymbolName(entry)).getValue()),
        builder.getStringAttr("qlx.fabric-counts/v1"), counts,
        builder.getArrayAttr({builder.getStringAttr(
            "dynamic branches are counted as a static upper bound")}),
        builder.getArrayAttr({FlatSymbolRefAttr::get(
            ctx,
            cast<StringAttr>(SymbolTable::getSymbolName(entry)).getValue())}),
        /*lowerTier=*/FlatSymbolRefAttr{},
        FlatSymbolRefAttr::get(ctx, device.getSymName()), metadata);
  }
};

} // namespace
