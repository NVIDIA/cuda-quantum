//===- FabricCountPass.cpp - Tier-1 static counter ------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// Tier 1 of the resource-estimation stack
// (design/resource-estimation-proposal.md §4.1).
//
// Walks a Fabric module and emits a typed `qlx.estimate_result` summarizing
// logical-qubit / gate / syndrome-round / protocol counts. The legacy top-level
// `fabric.counts` DictionaryAttr is mirrored during migration. Sub-second; no
// physical-event model, no schedule, no wall-clock.
//
// Walk semantics:
//   - fabric.gadget is IsolatedFromAbove. Top-level walk hits only the
//     entry gadget; non-entry gadgets are reached lazily through
//     fabric.call (inlining-by-counts at each call site).
//   - fabric.repeat: gate counts inside the body multiply by `count`.
//     Nested repeats compose multiplicatively.
//   - fabric.idle: `rounds` accumulates into rounds_by_kind.idle.
//   - fabric.if: conservatively counts both branches and propagates yielded
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
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <algorithm>
#include <map>
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

//===----------------------------------------------------------------------===//
// Accumulators
//===----------------------------------------------------------------------===//

struct PerRegion {
  StringRef role;
  StringRef code;
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
  std::map<std::string, int64_t> resourceRequests;
  std::map<std::string, int64_t> resourceStreamRequests;
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
  bool sawQecSpec = false;
  bool sawSelectedDeviceGadget = false;

  void recordRegionDecl(RegionOp r);
  void recordCodeDecl(CodeOp c);
  void walkEntry(Operation *entry);

private:
  MLIRContext *ctx;
  SymbolTable &symTab;
  DenseMap<Value, StringRef> valueToRegion;
  DenseMap<StringRef, int64_t> logicalsByCode;
  std::map<std::string, std::string> selectedRegionCodes;
  llvm::SmallSet<StringRef, 8> activeCalls;
  bool suppressInlineAnalysis = false;
  FlatSymbolRefAttr selectedQec;

  PerRegion &regionAccum(StringRef name);
  StringRef regionForValue(Value v) const;
  void propagatePatchValues(Operation *op);
  void bumpGate(Operation *op, StringRef gate, int64_t mult);
  void bumpRoundFor(Value patch, StringRef kind, int64_t n);
  void recordProtocol(Attribute protoAttr, StringRef kind, int64_t mult);
  static StringRef protocolNameOf(Attribute a);
  bool checkedAdd(int64_t &target, int64_t value, Operation *source,
                  StringRef what);
  FailureOr<int64_t> checkedMultiply(int64_t left, int64_t right,
                                     Operation *source, StringRef what);
  FailureOr<int64_t> logicalWeight(Value patch, Operation *source);
  void bump(std::map<std::string, int64_t> &counts, StringRef name,
            int64_t value, Operation *source, StringRef what);

  Block *resolveExecutableBody(Operation *callable, Operation *diagnosticOwner);
  void walkBlock(Block &block, int64_t mult);
  void walkOp(Operation *op, int64_t mult);
  void walkCall(CallOp call, int64_t mult);
  void walkDelegation(Operation *wrapper, FlatSymbolRefAttr callee,
                      int64_t wrapperMultiplicity, int64_t calleeMultiplicity);
  FailureOr<int64_t> mapChildrenMultiplicity(MapChildrenOp map,
                                             int64_t multiplier);
  void walkRepeat(RepeatOp rep, int64_t mult);
  void walkIf(IfOp ifop, int64_t mult);
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
  PerRegion &p = regionAccum(name);
  p.role = stringifyRole(r.getRole());
  p.code = r.getCode();
  p.protocol = r.getProtocolAttr();
  selectedRegionCodes[name.str()] = r.getCode().str();
}

void Walker::recordCodeDecl(CodeOp c) {
  logicalsByCode[c.getSymName()] = c.getK().value_or(1);
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
  if (selectedQec &&
      llvm::none_of(selectedRegionCodes, [&](const auto &region) {
        return StringRef(region.second) == code;
      })) {
    source->emitOpError("fabric-count patch code @")
        << code << " is not owned by the selected P2 machine " << selectedQec;
    hadError = true;
    return failure();
  }
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

StringRef Walker::regionForValue(Value v) const {
  auto it = valueToRegion.find(v);
  if (it == valueToRegion.end())
    return {};
  return it->second;
}

void Walker::bumpGate(Operation *op, StringRef gate, int64_t mult) {
  // Find a patch operand to attribute the gate to.
  StringRef region;
  for (Value v : op->getOperands()) {
    if (isa<PatchType>(v.getType())) {
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

/// Propagate region attribution for ops that consume a patch and produce
/// a patch with the "same logical identity." This is true for all single-
/// patch gate ops (AllTypesMatch<patch, result>), measurement (which also
/// keeps patch_out), merge/split (compose from operand a), multi_measure /
/// product ops (each output_i ← input_i), and inject (result ← patch operand).
void Walker::propagatePatchValues(Operation *op) {
  // Identify the first patch operand and the first patch result.
  Value firstPatchOperand;
  for (Value v : op->getOperands()) {
    if (isa<PatchType>(v.getType())) {
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
      valueToRegion[m.getMerged()] = r;
    return;
  }
  if (auto s = dyn_cast<SplitOp>(op)) {
    StringRef r = regionForValue(s.getMerged());
    if (r.empty())
      return;
    valueToRegion[s.getPatchA()] = r;
    valueToRegion[s.getPatchB()] = r;
    return;
  }
  if (auto mm = dyn_cast<MultiMeasureOp>(op)) {
    auto ins = mm.getPatches();
    auto outs = mm.getPatchesOut();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        valueToRegion[out] = r;
    }
    return;
  }
  if (auto mp = dyn_cast<MeasureProductOp>(op)) {
    auto ins = mp.getPatches();
    auto outs = mp.getPatchResults();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        valueToRegion[out] = r;
    }
    return;
  }
  if (auto rp = dyn_cast<RotateProductOp>(op)) {
    auto ins = rp.getPatches();
    auto outs = rp.getPatchResults();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        valueToRegion[out] = r;
    }
    return;
  }
  if (auto rp = dyn_cast<ResourceRotateProductOp>(op)) {
    auto ins = rp.getPatches();
    auto outs = rp.getPatchResults();
    for (auto [in, out] : llvm::zip(ins, outs)) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        valueToRegion[out] = r;
    }
    return;
  }
  if (auto tx = dyn_cast<TransversalCXOp>(op)) {
    StringRef rc = regionForValue(tx.getCtrl());
    StringRef rt = regionForValue(tx.getTarg());
    if (!rc.empty())
      valueToRegion[tx.getCtrlOut()] = rc;
    if (!rt.empty())
      valueToRegion[tx.getTargOut()] = rt;
    return;
  }
  if (auto barrier = dyn_cast<BarrierOp>(op)) {
    for (auto [in, out] :
         llvm::zip(barrier.getPatches(), barrier.getResults())) {
      StringRef r = regionForValue(in);
      if (!r.empty())
        valueToRegion[out] = r;
    }
    return;
  }

  // Default: a single patch operand projects to all patch results.
  if (!firstPatchOperand)
    return;
  StringRef r = regionForValue(firstPatchOperand);
  if (r.empty())
    return;
  for (Value v : op->getResults()) {
    if (isa<PatchType>(v.getType()))
      valueToRegion[v] = r;
  }
}

void Walker::walkEntry(Operation *entry) {
  if (!entry)
    return;
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

void Walker::walkOp(Operation *op, int64_t mult) {
  StringRef operationName = op->getName().getStringRef();
  if (operationName.starts_with("fabric.") &&
      !isa<ReturnOp, ProtocolReturnOp, YieldOp>(op))
    bump(operationCounts, operationName.drop_front(7), mult, op,
         "operation count");
  if (isa<ReadSyndromeAncillasOp, AssembleSyndromeOp>(op))
    checkedAdd(syndromeRounds, mult, op, "syndrome-round count");
  if (isa<SelectionOp>(op))
    checkedAdd(successCount, mult, op, "success-predicate count");

  // Structural / scoping ops handled specially.
  if (auto a = dyn_cast<AllocOp>(op)) {
    StringRef r = a.getRegion();
    if (selectedQec) {
      auto selected = selectedRegionCodes.find(r.str());
      if (selected == selectedRegionCodes.end()) {
        a.emitOpError("fabric-count allocation region @")
            << r << " is not owned by the selected P2 machine " << selectedQec;
        hadError = true;
        return;
      }
      if (selected->second != a.getCode()) {
        a.emitOpError("fabric-count allocation code @")
            << a.getCode() << " does not match selected region @" << r
            << " code @" << selected->second;
        hadError = true;
        return;
      }
    }
    valueToRegion[a.getResult()] = r;
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
  if (auto r = dyn_cast<RepeatOp>(op)) {
    walkRepeat(r, mult);
    return;
  }
  if (auto i = dyn_cast<IfOp>(op)) {
    walkIf(i, mult);
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
  if (auto unpack = dyn_cast<UnpackResourceOp>(op)) {
    auto anchors = unpack.getAnchors();
    auto outputs = unpack.getOutputs();
    if (outputs.size() != anchors.size() * 2) {
      unpack.emitOpError(
          "fabric-count requires one anchor successor and one payload patch "
          "per unpack anchor");
      hadError = true;
      return;
    }
    for (auto [index, anchor] : llvm::enumerate(anchors)) {
      StringRef region = regionForValue(anchor);
      valueToRegion[outputs[index]] = region;
      valueToRegion[outputs[index + anchors.size()]] = region;
      if (!region.empty()) {
        int64_t &live = liveByRegion[region];
        if (!checkedAdd(live, 1, op, "unpacked live-patch count"))
          return;
        PerRegion &perRegionEntry = regionAccum(region);
        perRegionEntry.patches = std::max(perRegionEntry.patches, live);
      }
      if (!checkedAdd(totalLive, 1, op, "unpacked global live-patch count"))
        return;
      patchPeak = std::max(patchPeak, totalLive);
      Value payload = outputs[index + anchors.size()];
      auto weight = logicalWeight(payload, op);
      if (failed(weight) ||
          !checkedAdd(logicalLive, *weight, op, "unpacked logical-qubit count"))
        return;
      logicalPeak = std::max(logicalPeak, logicalLive);
    }
    return;
  }
  if (auto pack = dyn_cast<PackResourceOp>(op)) {
    StringRef region = regionForValue(pack.getPayload());
    if (!region.empty()) {
      int64_t &live = liveByRegion[region];
      if (live > 0)
        --live;
    }
    if (totalLive > 0)
      --totalLive;
    auto weight = logicalWeight(pack.getPayload(), op);
    if (failed(weight))
      return;
    logicalLive = std::max<int64_t>(0, logicalLive - *weight);
    return;
  }
  if (auto unpack = dyn_cast<EncodingUnpackOp>(op)) {
    valueToRegion[unpack.getChildren()] = regionForValue(unpack.getParent());
    return;
  }
  if (auto pack = dyn_cast<EncodingPackOp>(op)) {
    valueToRegion[pack.getParent()] = regionForValue(pack.getChildren());
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
    StringRef region = regionForValue(inj.getPatch());
    if (!region.empty())
      checkedAdd(regionAccum(region).injectCount, mult, op, "injection count");
    recordProtocol(inj.getProtocolAttr(), "injection", mult);
    valueToRegion[inj.getResult()] = region;
    return;
  }
  if (auto tp = dyn_cast<TransportOp>(op)) {
    checkedAdd(regionAccum(tp.getSrcRegion()).transportOut, mult, op,
               "transport count");
    checkedAdd(regionAccum(tp.getDstRegion()).transportIn, mult, op,
               "transport count");
    recordProtocol(tp.getProtocolAttr(), "transport", mult);
    return;
  }
  if (isa<DiscardResourceOp>(op)) {
    // No region/protocol attribution; discards are summary-side only.
    return;
  }
  if (auto request = dyn_cast<ResourceRequestOp>(op)) {
    bump(resourceRequests, request.getKind(), mult, op,
         "resource-request count");
    SymbolRefAttr stream = request.getStreamAttr();
    std::string path = stream.getRootReference().getValue().str();
    for (FlatSymbolRefAttr nested : stream.getNestedReferences()) {
      path += "::";
      path += nested.getValue();
    }
    bump(resourceStreamRequests, path, mult, op,
         "resource-stream-request count");
    return;
  }
  if (isa<UnpackResourceOp, PackResourceOp, SelectionOp>(op)) {
    propagatePatchValues(op);
    return;
  }

  // Classical predicate/frame bookkeeping has no Tier-1 quantum cost. Keeping
  // this exemption typed makes the closed native-v1 support boundary explicit.
  if (isa<FrameCreateOp, FrameUpdateOp, FrameTransformOp, FrameInitOp,
          FramePropagateOp, FrameResolveOp, EventTestOp, EventPollOp, EventIsOp,
          EventSelectReadyOp, EventCancelOp, EventAwaitOp, FenceOp, SendOp,
          RecvOp, BarrierOp, XorOp, AllZeroOp, ParityOp, AllFalseOp>(op)) {
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
      !isa<ReturnOp, ProtocolReturnOp, YieldOp>(op)) {
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
  walkDelegation(call, call.getCalleeAttr(), mult, mult);
}

void Walker::walkDelegation(Operation *wrapper, FlatSymbolRefAttr calleeAttr,
                            int64_t wrapperMultiplicity,
                            int64_t calleeMultiplicity) {
  Operation *callee = symTab.lookup(calleeAttr.getValue());
  if (!callee || !isa<GadgetOp, ProtocolOp>(callee) ||
      callee->getNumRegions() != 1 || callee->getRegion(0).empty()) {
    wrapper->emitOpError("fabric-count cannot resolve executable callee ")
        << calleeAttr;
    hadError = true;
    return;
  }
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

  Block *body = resolveExecutableBody(callee, wrapper);
  if (!body) {
    activeCalls.erase(calleeName.getValue());
    return;
  }
  // Snapshot caller-side mapping so block-args can be temporarily mapped
  // and then cleared.
  llvm::SmallVector<BlockArgument> argsAdded;
  for (auto [arg, operand] :
       llvm::zip(body->getArguments(), wrapper->getOperands())) {
    StringRef r = regionForValue(operand);
    if (!r.empty()) {
      valueToRegion[arg] = r;
      argsAdded.push_back(arg);
    }
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
          valueToRegion[resVal] = r;
      }
    }
  }

  // Clear temporary block-arg mappings so a second call to the same
  // callee from a different region doesn't see stale state.
  for (BlockArgument a : argsAdded)
    valueToRegion.erase(a);
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

void Walker::walkRepeat(RepeatOp rep, int64_t mult) {
  if (rep.getBody().empty())
    return;
  Block &body = rep.getBody().front();
  int64_t count = rep.getCount();

  if (count == 0) {
    for (auto [result, init] : llvm::zip(rep.getResults(), rep.getInits())) {
      StringRef region = regionForValue(init);
      if (!region.empty())
        valueToRegion[result] = region;
    }
    return;
  }

  // Map iter-args to their init values' regions for the first iteration.
  llvm::SmallVector<BlockArgument> argsAdded;
  for (auto [arg, init] : llvm::zip(body.getArguments(), rep.getInits())) {
    if (isa<PatchType>(arg.getType()) && isa<PatchType>(init.getType())) {
      StringRef r = regionForValue(init);
      if (!r.empty()) {
        valueToRegion[arg] = r;
        argsAdded.push_back(arg);
      }
    }
  }

  int64_t foldedMultiplicity = 0;
  if (llvm::MulOverflow(mult, count, foldedMultiplicity)) {
    rep.emitOpError("fabric-count folded repeat multiplicity overflows i64");
    hadError = true;
    return;
  }
  // Walk the body once with mult * count: gate counts multiply by count.
  // Rounds inside the body (idle and syndrome) multiply too.
  walkBlock(body, foldedMultiplicity);

  // Map yielded values back to the repeat's results so downstream ops
  // continue with proper attribution.
  if (auto yield = dyn_cast<YieldOp>(body.getTerminator())) {
    for (auto [resVal, yldVal] :
         llvm::zip(rep.getResults(), yield.getOperands())) {
      if (isa<PatchType>(resVal.getType())) {
        StringRef r = regionForValue(yldVal);
        if (!r.empty())
          valueToRegion[resVal] = r;
      }
    }
  }

  for (BlockArgument a : argsAdded)
    valueToRegion.erase(a);
}

void Walker::walkIf(IfOp ifop, int64_t mult) {
  auto yieldedOperands = [](Region &region) -> SmallVector<Value> {
    if (region.empty())
      return {};
    Operation *term = region.front().getTerminator();
    if (auto yield = dyn_cast_or_null<YieldOp>(term))
      return SmallVector<Value>(yield.getOperands().begin(),
                                yield.getOperands().end());
    return {};
  };

  auto beforeValues = valueToRegion;
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

  SmallVector<Value> thenYield = yieldedOperands(ifop.getThenRegion());
  SmallVector<Value> elseYield = yieldedOperands(ifop.getElseRegion());
  for (auto it : llvm::enumerate(ifop.getResults())) {
    Value result = it.value();
    if (!isa<PatchType>(result.getType()))
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
      valueToRegion[result] = region;
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

  llvm::SmallVector<NamedAttribute> root = {
      {s(ctx, "operation_counts"), countMap(w.operationCounts)},
      {s(ctx, "gadget_calls"), countMap(w.gadgetCalls)},
      {s(ctx, "protocol_calls"), countMap(w.protocolCalls)},
      {s(ctx, "hierarchy_depths"), countMap(w.hierarchyDepths)},
      {s(ctx, "resource_requests"), countMap(w.resourceRequests)},
      {s(ctx, "resource_stream_requests"), countMap(w.resourceStreamRequests)},
      {s(ctx, "success_count"), i(ctx, w.successCount)},
      {s(ctx, "syndrome_rounds"), i(ctx, w.syndromeRounds)},
      {s(ctx, "patches_peak"), i(ctx, w.patchPeak)},
      {s(ctx, "logical_qubits_peak"), i(ctx, w.logicalPeak)},
      {s(ctx, "source_stage"), s(ctx, "p2")},
      {s(ctx, "source_facets"), sourceFacets},
      {s(ctx, "per_region"), DictionaryAttr::get(ctx, perRegionFields)},
      {s(ctx, "per_protocol"), DictionaryAttr::get(ctx, perProtoFields)},
      {s(ctx, "transversal_edges"), ArrayAttr::get(ctx, edgeAttrs)},
  };
  return DictionaryAttr::get(ctx, root);
}

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//

struct FabricCountPass
    : public qlx::fabric::impl::FabricCountBase<FabricCountPass> {
  using FabricCountBase::FabricCountBase;

  static bool hasAuthenticatedSelection(ProtocolOp protocol) {
    DictionaryAttr metadata = protocol.getMetadataAttr();
    if (!metadata)
      return false;
    auto input = metadata.getAs<StringAttr>("input_p1");
    auto commitment = metadata.getAs<StringAttr>("qec_selection_sha256");
    if (!input || input.getValue().empty() || !commitment)
      return false;
    StringRef value = commitment.getValue();
    if (!value.consume_front("sha256:") || value.size() != 64)
      return false;
    return llvm::all_of(value, [](char character) {
      return (character >= '0' && character <= '9') ||
             (character >= 'a' && character <= 'f');
    });
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTable symTab(module);

    // Phase 1: resolve exactly one selected executable root.  A missing root
    // is an incompatible-stage error, never a valid all-zero estimate.
    Operation *entry = nullptr;
    if (!rootSymbol.empty()) {
      entry = symTab.lookup(rootSymbol);
      if (!entry || !isa<GadgetOp, ProtocolOp>(entry)) {
        module.emitError("fabric-count root @")
            << rootSymbol << " must resolve to an executable Fabric root";
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
    if (device && isa<ProtocolOp>(entry) && !w.sawSelectedDeviceGadget &&
        !hasAuthenticatedSelection(cast<ProtocolOp>(entry))) {
      cast<ProtocolOp>(entry).emitOpError(
          "typed Tier-1 protocol root requires authenticated input_p1 and "
          "qec_selection_sha256 provenance for the selected P2 machine");
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
