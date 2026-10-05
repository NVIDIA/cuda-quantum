/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// Translates fabric.* IR to Stim circuit text format. Mirrors the fabric
// emitter (ir/lib/Translate/QLX/EmitStim.cpp) but walks the richer Fabric
// surface: fabric.code / fabric.gadget / stabilizer flows, structural
// operations, and raw measurement-record emission.
//
//===----------------------------------------------------------------------===//

#include "qlx/Target/Fabric/EmitStim.h"

#include "qlx/Dialect/Cflow/IR/CflowDialect.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"
#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Fabric/IR/FabricTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Tools/mlir-translate/Translation.h"

#include <algorithm>
#include <cmath>
#include <optional>
#include <string>

using namespace mlir;
using namespace qlx::fabric;

namespace {

//===----------------------------------------------------------------------===//
// Per-code registry: partition sizes and stabilizer/logical supports.
//===----------------------------------------------------------------------===//

struct CodeInfo {
  unsigned dataQubits = 0;
  unsigned sxQubits = 0;
  unsigned szQubits = 0;
  unsigned otherQubits = 0;
  /// hx[k] = list of data qubit indices coupled to ancilla sx[k].
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 4> hx;
  /// hz[k] = list of data qubit indices coupled to ancilla sz[k].
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 4> hz;
  /// Number of logical qubits (1 unless declared otherwise).
  unsigned k = 1;
  /// lx[j] / lz[j] = data qubit support for the j-th canonical logical
  /// X / Z operator. Empty when the code omitted the registry; emitters
  /// then fall back to "all data qubits" (only correct for k=1 codes
  /// whose logical is the uniform string, e.g. Steane).
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 4> lx;
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 4> lz;
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 4> gx;
  llvm::SmallVector<llvm::SmallVector<unsigned, 8>, 4> gz;
  bool isTrivialSingleCarrier = false;
  unsigned totalQubits() const {
    return dataQubits + sxQubits + szQubits + otherQubits;
  }
};

static unsigned partitionSize(DictionaryAttr dict, StringRef key) {
  if (auto a = dict.get(key))
    if (auto i = dyn_cast<IntegerAttr>(a))
      return static_cast<unsigned>(i.getInt());
  return 0;
}

static void
parseParityMatrix(ArrayAttr matrix,
                  llvm::SmallVectorImpl<llvm::SmallVector<unsigned, 8>> &out) {
  if (!matrix)
    return;
  for (auto rowAttr : matrix) {
    if (auto row = dyn_cast<DenseI64ArrayAttr>(rowAttr)) {
      llvm::SmallVector<unsigned, 8> v;
      for (int64_t j : row.asArrayRef())
        v.push_back(static_cast<unsigned>(j));
      out.push_back(std::move(v));
    }
  }
}

static bool hasRepresentedStructure(Operation *operation, StringRef name) {
  Attribute value = operation->getAttr(name);
  if (!value)
    return false;
  if (auto array = dyn_cast<ArrayAttr>(value))
    return !array.empty();
  if (auto elements = dyn_cast<ElementsAttr>(value))
    return elements.getNumElements() != 0;
  return true;
}

static bool hasUnexpectedCanonicalBasis(Operation *operation, StringRef name,
                                        ArrayRef<int64_t> expectedShape,
                                        ArrayRef<bool> expectedValues) {
  Attribute value = operation->getAttr(name);
  if (!value)
    return false;
  auto elements = dyn_cast<DenseIntElementsAttr>(value);
  if (!elements || elements.getType().getShape() != expectedShape ||
      elements.getNumElements() != expectedValues.size())
    return true;
  return !std::equal(elements.value_begin<bool>(), elements.value_end<bool>(),
                     expectedValues.begin(), expectedValues.end());
}

static llvm::StringMap<CodeInfo> buildCodeRegistry(ModuleOp module) {
  llvm::StringMap<CodeInfo> reg;
  module.walk([&](CodeOp codeOp) {
    CodeInfo info;
    auto parts = codeOp.getPartitions();
    info.dataQubits = partitionSize(parts, "data");
    info.sxQubits = partitionSize(parts, "sx");
    info.szQubits = partitionSize(parts, "sz");
    for (NamedAttribute entry : parts) {
      StringRef name = entry.getName().strref();
      if (name == "data" || name == "sx" || name == "sz")
        continue;
      if (auto size = dyn_cast<IntegerAttr>(entry.getValue()))
        info.otherQubits += static_cast<unsigned>(size.getInt());
    }

    if (auto hxAttr = codeOp.getHx())
      parseParityMatrix(*hxAttr, info.hx);
    if (auto hzAttr = codeOp.getHz())
      parseParityMatrix(*hzAttr, info.hz);
    if (auto kAttr = codeOp.getK())
      info.k = static_cast<unsigned>(*kAttr);
    if (auto lxAttr = codeOp.getLx())
      parseParityMatrix(*lxAttr, info.lx);
    if (auto lzAttr = codeOp.getLz())
      parseParityMatrix(*lzAttr, info.lz);
    if (auto gxAttr = codeOp.getGx())
      parseParityMatrix(*gxAttr, info.gx);
    if (auto gzAttr = codeOp.getGz())
      parseParityMatrix(*gzAttr, info.gz);

    int64_t n = codeOp.getN().value_or(info.dataQubits);
    int64_t r = codeOp.getR().value_or(0);
    static constexpr StringLiteral nontrivialStructureAttrs[] = {
        "hx",
        "hz",
        "gx",
        "gz",
        "stabilizers",
        "gauges",
        "stabilizer_basis",
        "gauge_x_basis",
        "gauge_z_basis",
        "anti_stabilizers"};
    bool hasStabilizerOrGauge =
        llvm::any_of(nontrivialStructureAttrs, [&](StringRef name) {
          return hasRepresentedStructure(codeOp.getOperation(), name);
        });
    static constexpr int64_t logicalShape[] = {1, 2};
    static constexpr bool logicalX[] = {true, false};
    static constexpr bool logicalZ[] = {false, true};
    static constexpr int64_t cliffordShape[] = {2, 2};
    static constexpr bool identityClifford[] = {true, false, false, true};
    bool hasNonidentityCanonicalBasis =
        hasUnexpectedCanonicalBasis(codeOp.getOperation(), "logical_x_basis",
                                    logicalShape, logicalX) ||
        hasUnexpectedCanonicalBasis(codeOp.getOperation(), "logical_z_basis",
                                    logicalShape, logicalZ) ||
        hasUnexpectedCanonicalBasis(codeOp.getOperation(), "encoding_clifford",
                                    cliffordShape, identityClifford);
    auto isIdentityLogicalSupport = [](const auto &rows) {
      return rows.empty() || (rows.size() == 1 && rows.front().size() == 1 &&
                              rows.front().front() == 0);
    };
    info.isTrivialSingleCarrier =
        n == 1 && info.k == 1 && r == 0 && codeOp.getDistance() == 1 &&
        info.dataQubits == 1 && info.totalQubits() == 1 &&
        !hasStabilizerOrGauge && !hasNonidentityCanonicalBasis &&
        isIdentityLogicalSupport(info.lx) && isIdentityLogicalSupport(info.lz);

    reg[codeOp.getSymName()] = std::move(info);
  });
  return reg;
}

//===----------------------------------------------------------------------===//
// Clifford-basis frame carried by patch SSA values.
//===----------------------------------------------------------------------===//

struct BasisFrame {
  std::string xTo;
  std::string zTo;
};

static BasisFrame identityFrame() { return BasisFrame{"x", "z"}; }

static BasisFrame composeFrame(const BasisFrame &outer,
                               const BasisFrame &inner) {
  auto lookup = [&](const BasisFrame &frame,
                    const std::string &basis) -> std::string {
    if (basis == "x")
      return frame.xTo;
    if (basis == "z")
      return frame.zTo;
    return basis;
  };
  return {lookup(outer, inner.xTo), lookup(outer, inner.zTo)};
}

static BasisFrame deltaFrame(const BasisFrame &prev, const BasisFrame &cur) {
  BasisFrame prevInv;
  auto inverseImage = [&](StringRef out) -> std::string {
    if (prev.xTo == out)
      return "x";
    if (prev.zTo == out)
      return "z";
    return out.str();
  };
  prevInv.xTo = inverseImage("x");
  prevInv.zTo = inverseImage("z");
  return composeFrame(prevInv, cur);
}

static StringRef framePreimage(const BasisFrame &frame, StringRef out) {
  if (frame.xTo == out)
    return "x";
  if (frame.zTo == out)
    return "z";
  return {};
}

//===----------------------------------------------------------------------===//
// Per-patch qubit info + syndrome / data-measurement records.
//===----------------------------------------------------------------------===//

struct PatchInfo {
  unsigned baseIndex = 0;
  CodeInfo codeInfo;
  std::string codeName;
  BasisFrame frame = identityFrame();
  /// Carrier initialization is a semantic property, not an allocation side
  /// effect. Entry arguments are caller-initialized; fabric.alloc starts with
  /// every carrier clear in this mask until an explicit preparation/reset.
  llvm::SmallBitVector initialized;

  std::pair<unsigned, unsigned> partitionRange(Partition p) const {
    unsigned d = codeInfo.dataQubits;
    unsigned sx = codeInfo.sxQubits;
    unsigned sz = codeInfo.szQubits;
    switch (p) {
    case Partition::data:
      return {baseIndex, baseIndex + d};
    case Partition::sx:
      return {baseIndex + d, baseIndex + d + sx};
    case Partition::sz:
      return {baseIndex + d + sx, baseIndex + d + sx + sz};
    case Partition::all:
      return {baseIndex, baseIndex + codeInfo.totalQubits()};
    }
    return {0, 0};
  }
};

struct SyndromeRecords {
  std::string codeName;
  BasisFrame frame = identityFrame();
  /// Absolute Stim measurement index of the first sx-ancilla measurement
  /// in this syndrome round; nSx of them are contiguous from that point.
  unsigned sxFirstAbs = 0;
  unsigned nSx = 0;
  unsigned szFirstAbs = 0;
  unsigned nSz = 0;
};

struct DataMeasRecords {
  std::string codeName;
  unsigned firstAbs = 0;
  unsigned count = 0;
};

//===----------------------------------------------------------------------===//
// Stim emitter.
//===----------------------------------------------------------------------===//

class StimEmitter {
public:
  StimEmitter(llvm::raw_ostream &os, ModuleOp module)
      : os(os), module(module), codeReg(buildCodeRegistry(module)) {}

  LogicalResult emit() {
    // 1. Find the entry gadget (must have `entry` attribute + device).
    GadgetOp entry;
    bool ambiguousEntry = false;
    module.walk([&](GadgetOp g) {
      if (!g.getEntry())
        return;
      if (entry) {
        ambiguousEntry = true;
        return;
      }
      entry = g;
    });
    if (!entry) {
      module.emitError("fabric-to-stim: no entry gadget found "
                       "(expected a fabric.gadget with {entry})");
      return failure();
    }
    if (ambiguousEntry) {
      module.emitError("fabric-to-stim: multiple entry gadgets are ambiguous");
      return failure();
    }

    // 2. Walk the selected realization body, inlining fabric.call targets.
    Block *body = resolveGadgetBody(entry, entry.getOperation());
    if (!body)
      return failure();
    if (failed(seedEntryArguments(entry, *body)))
      return failure();
    activeCallables.insert(entry.getOperation());
    walkBlock(*body);
    activeCallables.erase(entry.getOperation());
    return hadError ? failure() : success();
  }

private:
  //===--------------------------------------------------------------------===//
  // Core dispatch + walk.
  //===--------------------------------------------------------------------===//

  Block *resolveGadgetBody(GadgetOp gadget, Operation *diagnosticOwner) {
    if (auto realization = gadget.getRealizationAttr()) {
      auto circuit = module.lookupSymbol<CircuitOp>(realization.getValue());
      if (!circuit || circuit.getBody().empty()) {
        diagnosticOwner->emitError(
            "fabric-to-stim: gadget realization does not resolve to a "
            "fabric.circuit body");
        hadError = true;
        return nullptr;
      }
      return &circuit.getBody().front();
    }
    if (gadget.getBody().empty()) {
      diagnosticOwner->emitError(
          "fabric-to-stim: gadget has no executable body");
      hadError = true;
      return nullptr;
    }
    return &gadget.getBody().front();
  }

  LogicalResult seedEntryArguments(GadgetOp entry, Block &body) {
    for (BlockArgument argument : body.getArguments()) {
      auto patch = dyn_cast<PatchType>(argument.getType());
      if (!patch) {
        if (argument.getType().getDialect().getNamespace() == "fabric") {
          entry.emitError(
              "fabric-to-stim: unsupported non-patch Fabric entry boundary ")
              << argument.getType();
          hadError = true;
          return failure();
        }
        continue;
      }
      StringRef codeName = patch.getCodeType().getValue();
      auto code = codeReg.find(codeName);
      if (code == codeReg.end()) {
        entry.emitError("fabric-to-stim: entry patch references unknown code @")
            << codeName;
        hadError = true;
        return failure();
      }
      PatchInfo info;
      info.baseIndex = nextQubitIndex;
      info.codeInfo = code->second;
      info.codeName = codeName.str();
      info.initialized.resize(info.codeInfo.totalQubits(), true);
      nextQubitIndex += info.codeInfo.totalQubits();
      patchMap[argument] = std::move(info);
    }
    return success();
  }

  void walkBlock(Block &block) {
    for (Operation &op : block)
      emitOp(&op);
  }

  void emitOp(Operation *op) {
    llvm::TypeSwitch<Operation *>(op)
        // Lifecycle.
        .Case<AllocOp>([&](auto o) { emitAlloc(o); })
        .Case<DeallocOp>([&](auto o) {
          (void)requirePatch(o.getOperation(), o.getPatch(), "dealloc");
        })
        .Case<PrepZOp>([&](auto o) { emitPrep(o, /*xBasis=*/false); })
        .Case<PrepXOp>([&](auto o) { emitPrep(o, /*xBasis=*/true); })
        // Bulk single-patch Clifford gates.
        .Case<HOp>([&](auto o) {
          emitBulkGate("H", "h", o);
          updateBasisFrame(o, BasisFrame{"z", "x"});
        })
        .Case<SOp>([&](auto o) {
          emitBulkGate("S", "s", o);
          updateBasisFrame(o, BasisFrame{"y", "z"});
        })
        .Case<SdgOp>([&](auto o) {
          emitBulkGate("S_DAG", "sdg", o);
          updateBasisFrame(o, BasisFrame{"y", "z"});
        })
        .Case<XOp>([&](auto o) { emitBulkGate("X", "x", o); })
        .Case<ZOp>([&](auto o) { emitBulkGate("Z", "z", o); })
        .Case<TOp, TdgOp>([&](auto nonClifford) {
          nonClifford.emitError(
              "fabric-to-stim: non-Clifford operations are unsupported");
          hadError = true;
        })
        // Reset is the primitive carrier-initialization operation.
        .Case<ResetOp>([&](auto o) { emitReset(o); })
        .Case<InitBasisOp>([&](auto o) { emitInitBasis(o); })

        // Two-qubit gates.
        .Case<CXOp>([&](auto o) { emitCX(o); })
        .Case<CZOp>([&](auto o) { emitCZ(o); })
        .Case<TransversalCXOp>([&](auto o) { emitTransversalCX(o); })
        .Case<PermuteOp>([&](auto o) { emitPermute(o); })
        // Measurement.
        .Case<MzOp>([&](auto o) { emitMz(o); })
        .Case<MeasureBasisOp>([&](auto o) { emitMeasureBasis(o); })
        .Case<MppOp>([&](auto o) { emitMpp(o); })
        .Case<MeasureProductOp>([&](auto o) { emitMeasureProduct(o); })
        .Case<ReadSyndromeAncillasOp>(
            [&](auto o) { emitReadSyndromeAncillas(o); })
        .Case<AssembleSyndromeOp>([&](auto o) { emitAssembleSyndrome(o); })
        // Resource preparation/injection has no faithful Stim text form.
        .Case<ProduceResourceOp, InjectOp, DiscardResourceOp>([&](auto op) {
          op.emitError("fabric-to-stim: unsupported operation '")
              << op->getName().getStringRef() << "'";
          hadError = true;
        })
        // Control flow.
        .Case<CallOp>([&](auto o) { emitCall(o); })
        .Case<qlx::cflow::RepeatOp>([&](auto o) { emitRepeat(o); })
        .Case<qlx::cflow::IfOp>([&](auto o) { emitIf(o); })
        .Case<qlx::cflow::YieldOp>(
            [&](auto) { /* consumed by emitRepeat / emitIf */ })
        .Case<ReturnOp>([&](auto) { /* terminates the walk naturally */ })
        .Case<BarrierOp>([&](auto o) {
          for (auto [in, out] : llvm::zip(o.getPatches(), o.getResults())) {
            if (!requirePatch(o.getOperation(), in, "barrier"))
              continue;
            propagatePatch(in, out);
          }
        })
        .Case<IdleOp>([&](auto o) {
          // Idle: just propagate, no physical emission. Idle rounds should
          // be realized by the programmer via fabric.call to a memory
          // gadget; bare idle has no stabilizer measurements to emit.
          if (requirePatch(o.getOperation(), o.getPatch(), "idle"))
            propagatePatch(o.getPatch(), o.getResult());
        })
        .Case<FrameInitOp, FramePropagateOp, FrameResolveOp>([&](auto op) {
          op->emitError("fabric-to-stim: unsupported operation '")
              << op->getName().getStringRef()
              << "' (Pauli frame operations have no Stim text form)";
          hadError = true;
        })
        // High-level lattice-surgery ops must be lowered to primitive Fabric
        // operations by the device-selected MPP/lattice-surgery compiler
        // before invoking a circuit backend.
        .Case<MergeOp, SplitOp, MultiMeasureOp>([&](auto op) {
          op->emitError("fabric-to-stim: ")
              << op->getName().getStringRef()
              << " must be lowered to primitive fabric ops by the "
                 "device-selected MPP/lattice-surgery compiler before "
                 "invoking the Stim backend";
          hadError = true;
        })
        // Move / Send / Recv — transport ops still out of scope for the
        // Stim emitter.
        .Case<MoveOp, SendOp, RecvOp>([&](auto op) {
          op->emitError("fabric-to-stim: ")
              << op->getName().getStringRef()
              << " (transport op) has no Stim representation";
          hadError = true;
        })
        .Default([&](Operation *op) {
          op->emitError("fabric-to-stim: unsupported operation '")
              << op->getName().getStringRef() << "'";
          hadError = true;
        });
  }

  //===--------------------------------------------------------------------===//
  // Patch bookkeeping.
  //===--------------------------------------------------------------------===//

  // NB: copy the found value into a local *before* the inserting `map[to]`.
  // `map[to]` may grow/rehash the DenseMap, which invalidates `it` — reading
  // `it->second` afterwards is use-after-free and silently propagates garbage
  // (a nondeterministic, allocation-order-dependent bug).
  void propagatePatch(Value from, Value to) {
    auto it = patchMap.find(from);
    if (it != patchMap.end()) {
      PatchInfo info = it->second;
      patchMap[to] = std::move(info);
    }
  }

  void propagateSyndrome(Value from, Value to) {
    auto it = syndromeMap.find(from);
    if (it != syndromeMap.end()) {
      SyndromeRecords rec = it->second;
      syndromeMap[to] = std::move(rec);
    }
  }

  void propagateData(Value from, Value to) {
    auto it = dataMeas.find(from);
    if (it != dataMeas.end()) {
      DataMeasRecords rec = it->second;
      dataMeas[to] = std::move(rec);
    }
  }

  const PatchInfo *lookupPatch(Value patch) {
    auto it = patchMap.find(patch);
    if (it == patchMap.end())
      return nullptr;
    return &it->second;
  }

  const PatchInfo *requirePatch(Operation *op, Value patch, StringRef action) {
    const PatchInfo *info = lookupPatch(patch);
    if (!info) {
      op->emitError("fabric-to-stim: ")
          << action << " has an unresolved patch owner";
      hadError = true;
    }
    return info;
  }

  bool requireInitialized(Operation *op, Value patch, ArrayRef<unsigned> qubits,
                          StringRef action) {
    const PatchInfo *info = requirePatch(op, patch, action);
    if (!info)
      return false;
    for (unsigned qubit : qubits) {
      if (qubit < info->baseIndex ||
          qubit >= info->baseIndex + info->initialized.size() ||
          !info->initialized.test(qubit - info->baseIndex)) {
        op->emitError("fabric-to-stim: ")
            << action << " uses carrier " << qubit
            << " before explicit preparation or reset";
        hadError = true;
        return false;
      }
    }
    return true;
  }

  void markInitialized(Value patch, ArrayRef<unsigned> qubits) {
    auto found = patchMap.find(patch);
    if (found == patchMap.end())
      return;
    PatchInfo &info = found->second;
    for (unsigned qubit : qubits)
      if (qubit >= info.baseIndex &&
          qubit < info.baseIndex + info.initialized.size())
        info.initialized.set(qubit - info.baseIndex);
  }

  /// Resolve a (patch, partition, optional indices) tuple to a list of
  /// absolute qubit indices.
  llvm::SmallVector<unsigned, 16>
  resolvePartition(Value patch, Partition part,
                   std::optional<llvm::ArrayRef<int64_t>> indices) {
    llvm::SmallVector<unsigned, 16> out;
    const PatchInfo *pi = lookupPatch(patch);
    if (!pi)
      return out;
    auto [lo, hi] = pi->partitionRange(part);
    if (indices) {
      for (int64_t i : *indices) {
        unsigned abs = static_cast<unsigned>(lo + i);
        if (abs < hi)
          out.push_back(abs);
      }
    } else {
      for (unsigned q = lo; q < hi; ++q)
        out.push_back(q);
    }
    return out;
  }

  //===--------------------------------------------------------------------===//
  // Emission of individual ops.
  //===--------------------------------------------------------------------===//

  void emitAlloc(AllocOp op) {
    StringRef codeName = op.getCode();
    auto it = codeReg.find(codeName);
    if (it == codeReg.end()) {
      op.emitError("fabric-to-stim: unknown code '@") << codeName << "'";
      hadError = true;
      return;
    }
    PatchInfo pi;
    pi.baseIndex = nextQubitIndex;
    pi.codeInfo = it->second;
    pi.codeName = codeName.str();
    pi.initialized.resize(pi.codeInfo.totalQubits(), false);
    nextQubitIndex += pi.codeInfo.totalQubits();
    patchMap[op.getResult()] = std::move(pi);
  }

  template <typename PrepTy>
  void emitPrep(PrepTy op, bool xBasis) {
    propagatePatch(op.getPatch(), op.getResult());
    const PatchInfo *pi =
        requirePatch(op.getOperation(), op.getPatch(), "preparation");
    if (!pi)
      return;
    if (!pi->codeInfo.isTrivialSingleCarrier) {
      op.emitError("fabric-to-stim: prep_x/prep_z require a proven trivial "
                   "one-carrier code (n=1, k=1, distance=1, no stabilizer "
                   "or gauge structure)");
      hadError = true;
      return;
    }
    auto [lo, hi] = pi->partitionRange(Partition::data);
    if (lo == hi)
      return;
    os << (xBasis ? "RX" : "R");
    for (unsigned q = lo; q < hi; ++q)
      os << " " << q;
    os << "\n";
    llvm::SmallVector<unsigned, 16> qubits;
    for (unsigned q = lo; q < hi; ++q)
      qubits.push_back(q);
    markInitialized(op.getResult(), qubits);
  }

  template <typename GateOp>
  void emitBulkGate(StringRef stimName, StringRef operationKey, GateOp op) {
    (void)operationKey;
    propagatePatch(op.getPatch(), op.getResult());
    if (!requirePatch(op.getOperation(), op.getPatch(), stimName))
      return;
    std::optional<llvm::ArrayRef<int64_t>> indices;
    if (auto idx = op.getIndices())
      indices = *idx;
    auto qubits = resolvePartition(op.getPatch(), op.getPartition(), indices);
    if (qubits.empty())
      return;
    if (!requireInitialized(op.getOperation(), op.getPatch(), qubits,
                            operationKey))
      return;

    os << stimName;
    for (unsigned q : qubits)
      os << " " << q;
    os << "\n";
  }

  void emitReset(ResetOp op) {
    propagatePatch(op.getPatch(), op.getResult());
    if (!requirePatch(op.getOperation(), op.getPatch(), "reset"))
      return;
    std::optional<llvm::ArrayRef<int64_t>> indices;
    if (auto selected = op.getIndices())
      indices = *selected;
    auto qubits = resolvePartition(op.getPatch(), op.getPartition(), indices);
    if (qubits.empty())
      return;
    os << "R";
    for (unsigned qubit : qubits)
      os << " " << qubit;
    os << "\n";
    markInitialized(op.getResult(), qubits);
  }

  template <typename GateOp>
  void updateBasisFrame(GateOp op, const BasisFrame &gateFrame) {
    // Only a uniform Clifford over the complete data partition defines one
    // code-level frame. Ancilla gates used by syndrome extraction and indexed
    // physical gates do not conjugate the logical stabilizer basis wholesale.
    if (op.getPartition() != Partition::data || op.getIndices())
      return;
    auto it = patchMap.find(op.getResult());
    if (it == patchMap.end())
      return;
    it->second.frame = composeFrame(gateFrame, it->second.frame);
  }

  void emitCX(CXOp op) {
    if (op.getPatches().size() == 2)
      return emitCrossTwoQubit("CX", "cx", op);
    emitIntraTwoQubit(op, "CX", "cx", op.getPatches()[0], op.getCtrl(),
                      op.getTarg(), op.getSchedule(), op.getPairs(),
                      op.getResults()[0]);
  }

  void emitCZ(CZOp op) {
    if (op.getPatches().size() == 2)
      return emitCrossTwoQubit("CZ", "cz", op);
    emitIntraTwoQubit(op, "CZ", "cz", op.getPatches()[0], op.getCtrl(),
                      op.getTarg(), op.getSchedule(), op.getPairs(),
                      op.getResults()[0]);
  }

  // Cross-patch CX/CZ: each `pairs="c:t,..."` entry has the control
  // index in patches[0]'s ctrl partition and the target index in
  // patches[1]'s targ partition. Emits absolute-qubit Stim gates.
  template <typename OpT>
  void emitCrossTwoQubit(StringRef stimName, StringRef operationKey, OpT op) {
    (void)operationKey;
    Value patchA = op.getPatches()[0];
    Value patchB = op.getPatches()[1];
    propagatePatch(patchA, op.getResults()[0]);
    propagatePatch(patchB, op.getResults()[1]);
    const PatchInfo *piA = lookupPatch(patchA);
    const PatchInfo *piB = lookupPatch(patchB);
    if (!piA || !piB) {
      op.emitError("fabric-to-stim: ")
          << stimName << " has unresolved cross-patch owners";
      hadError = true;
      return;
    }
    auto pairsStr = op.getPairs();
    if (!pairsStr) {
      op.emitError("fabric-to-stim: ")
          << stimName << " cross-patch form requires explicit pairs";
      hadError = true;
      return;
    }
    auto [aLo, aHi] = piA->partitionRange(op.getCtrl());
    auto [bLo, bHi] = piB->partitionRange(op.getTarg());
    llvm::SmallVector<std::pair<unsigned, unsigned>, 16> pairs;
    StringRef s = *pairsStr;
    if (s.trim() == "index") {
      unsigned width = std::min(aHi - aLo, bHi - bLo);
      for (unsigned i = 0; i < width; ++i)
        pairs.push_back({aLo + i, bLo + i});
      s = {};
    }
    while (!s.empty()) {
      auto comma = s.find(',');
      StringRef tok = comma == StringRef::npos ? s : s.substr(0, comma);
      s = comma == StringRef::npos ? StringRef() : s.substr(comma + 1);
      auto colon = tok.find(':');
      if (colon == StringRef::npos)
        continue;
      unsigned ci, ti;
      if (tok.substr(0, colon).trim().getAsInteger(10, ci))
        continue;
      if (tok.substr(colon + 1).trim().getAsInteger(10, ti))
        continue;
      if (aLo + ci >= aHi || bLo + ti >= bHi)
        continue;
      pairs.push_back({aLo + ci, bLo + ti});
    }
    if (pairs.empty())
      return;
    llvm::SmallVector<unsigned, 16> controls;
    llvm::SmallVector<unsigned, 16> targets;
    for (auto [control, target] : pairs) {
      controls.push_back(control);
      targets.push_back(target);
    }
    if (!requireInitialized(op.getOperation(), patchA, controls,
                            operationKey) ||
        !requireInitialized(op.getOperation(), patchB, targets, operationKey))
      return;
    os << stimName;
    for (auto [c, t] : pairs)
      os << " " << c << " " << t;
    os << "\n";
  }

  template <typename OpT>
  void emitIntraTwoQubit(OpT op, StringRef stimName, StringRef operationKey,
                         Value patch, Partition ctrl, Partition targ,
                         std::optional<StringRef> schedule,
                         std::optional<StringRef> pairsStr, Value result) {
    propagatePatch(patch, result);
    const PatchInfo *pi = lookupPatch(patch);
    if (!pi) {
      op.emitError("fabric-to-stim: cannot resolve patch for ") << stimName;
      hadError = true;
      return;
    }

    llvm::SmallVector<std::pair<unsigned, unsigned>, 16> pairs;
    if (pairsStr) {
      pairs = parsePartitionLocalPairs(*pairsStr, *pi, ctrl, targ);
    } else if (schedule) {
      pairs = resolveCxSchedule(*schedule, *pi, patch);
    } else {
      op.emitError("fabric-to-stim: ")
          << stimName << " requires explicit pairs or a CSS schedule";
      hadError = true;
      return;
    }
    if (pairs.empty()) {
      if (schedule && (*schedule == "hx" || *schedule == "hz")) {
        op.emitOpError("schedule '")
            << *schedule << "' requires nonempty " << *schedule
            << " checks on code @" << pi->codeName;
      } else {
        op.emitError("fabric-to-stim: ")
            << stimName << " resolved to no carrier interactions";
      }
      hadError = true;
      return;
    }

    llvm::SmallVector<unsigned, 16> used;
    for (auto [control, target] : pairs) {
      used.push_back(control);
      used.push_back(target);
    }
    if (!requireInitialized(op.getOperation(), patch, used, operationKey))
      return;

    os << stimName;
    for (auto [c, t] : pairs)
      os << " " << c << " " << t;
    os << "\n";

    (void)operationKey;
  }

  /// Parse `pairs = "c:t,c:t,..."` attribute format (partition-local
  /// indices). Returns absolute qubit pairs.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 16>
  parsePartitionLocalPairs(StringRef s, const PatchInfo &pi, Partition ctrl,
                           Partition targ) {
    llvm::SmallVector<std::pair<unsigned, unsigned>, 16> out;
    auto [cLo, cHi] = pi.partitionRange(ctrl);
    auto [tLo, tHi] = pi.partitionRange(targ);
    if (s.trim() == "index") {
      unsigned width = std::min(cHi - cLo, tHi - tLo);
      for (unsigned i = 0; i < width; ++i)
        out.push_back({cLo + i, tLo + i});
      return out;
    }
    while (!s.empty()) {
      auto comma = s.find(',');
      StringRef tok = comma == StringRef::npos ? s : s.substr(0, comma);
      s = comma == StringRef::npos ? StringRef() : s.substr(comma + 1);
      auto colon = tok.find(':');
      if (colon == StringRef::npos)
        continue;
      unsigned ci, ti;
      if (tok.substr(0, colon).trim().getAsInteger(10, ci))
        continue;
      if (tok.substr(colon + 1).trim().getAsInteger(10, ti))
        continue;
      if (cLo + ci >= cHi || tLo + ti >= tHi)
        continue;
      out.push_back({cLo + ci, tLo + ti});
    }
    return out;
  }

  /// Resolve `schedule = "hx"` / `"hz"` to absolute qubit pairs using
  /// the code's parity check matrices.
  llvm::SmallVector<std::pair<unsigned, unsigned>, 16>
  resolveCxSchedule(StringRef schedule, const PatchInfo &pi, Value patch) {
    llvm::SmallVector<std::pair<unsigned, unsigned>, 16> out;
    if (schedule != "hx" && schedule != "hz")
      return out;
    auto codeIt = codeReg.find(pi.codeName);
    if (codeIt == codeReg.end())
      return out;
    const auto &matrix =
        (schedule == "hx") ? codeIt->second.hx : codeIt->second.hz;
    auto [dLo, dHi] = pi.partitionRange(Partition::data);
    (void)dHi;
    if (schedule == "hx") {
      auto [sxLo, sxHi] = pi.partitionRange(Partition::sx);
      (void)sxHi;
      for (unsigned k = 0; k < matrix.size(); ++k)
        for (unsigned j : matrix[k])
          out.push_back({sxLo + k, dLo + j});
    } else {
      auto [szLo, szHi] = pi.partitionRange(Partition::sz);
      (void)szHi;
      for (unsigned k = 0; k < matrix.size(); ++k)
        for (unsigned j : matrix[k])
          out.push_back({dLo + j, szLo + k});
    }
    return out;
  }

  void emitTransversalCX(TransversalCXOp op) {
    propagatePatch(op.getCtrl(), op.getCtrlOut());
    propagatePatch(op.getTarg(), op.getTargOut());
    const PatchInfo *cpi = lookupPatch(op.getCtrl());
    const PatchInfo *tpi = lookupPatch(op.getTarg());
    if (!cpi || !tpi) {
      op.emitError("fabric-to-stim: transversal_cx has unresolved patches");
      hadError = true;
      return;
    }
    if (cpi->codeInfo.dataQubits != tpi->codeInfo.dataQubits) {
      op.emitError("fabric-to-stim: transversal_cx requires equal data widths");
      hadError = true;
      return;
    }
    unsigned n = cpi->codeInfo.dataQubits;
    if (n == 0)
      return;
    auto [cLo, cHi] = cpi->partitionRange(Partition::data);
    auto [tLo, tHi] = tpi->partitionRange(Partition::data);
    (void)cHi;
    (void)tHi;

    llvm::SmallVector<unsigned, 16> controlQubits;
    llvm::SmallVector<unsigned, 16> targetQubits;
    for (unsigned i = 0; i < n; ++i) {
      controlQubits.push_back(cLo + i);
      targetQubits.push_back(tLo + i);
    }
    if (!requireInitialized(op.getOperation(), op.getCtrl(), controlQubits,
                            "transversal_cx") ||
        !requireInitialized(op.getOperation(), op.getTarg(), targetQubits,
                            "transversal_cx"))
      return;

    // Build target-index mapping: tLo + perm[i] if perm given, else tLo + i.
    llvm::SmallVector<unsigned, 16> tIdx(n);
    if (auto permAttr = op.getPerm()) {
      // getPerm() returns ArrayRef<int64_t> directly for DenseI64ArrayAttr.
      auto permArr = *permAttr;
      if (permArr.size() != n) {
        op.emitError("fabric-to-stim: transversal_cx permutation width ")
            << permArr.size() << " disagrees with data width " << n;
        hadError = true;
        return;
      }
      llvm::SmallDenseSet<int64_t, 16> seen;
      for (unsigned i = 0; i < n; ++i) {
        int64_t raw = permArr[i];
        if (raw < 0 || raw >= static_cast<int64_t>(n) ||
            !seen.insert(raw).second) {
          op.emitError("fabric-to-stim: transversal_cx perm must be a "
                       "permutation of target data indices");
          hadError = true;
          return;
        }
        unsigned p = static_cast<unsigned>(raw);
        tIdx[i] = tLo + p;
      }
    } else {
      for (unsigned i = 0; i < n; ++i)
        tIdx[i] = tLo + i;
    }

    os << "CX";
    for (unsigned i = 0; i < n; ++i)
      os << " " << (cLo + i) << " " << tIdx[i];
    os << "\n";
  }

  // A code-automorphism relabeling of the data qubits: realize it as a
  // *noiseless* SWAP network (a permutation is a relabeling, not a gate). The
  // content at data position i moves to position perm[i]; we emit the cycle
  // decomposition as transpositions.
  void emitPermute(PermuteOp op) {
    propagatePatch(op.getPatch(), op.getResult());
    const PatchInfo *pi = lookupPatch(op.getPatch());
    if (!pi) {
      op.emitError("fabric-to-stim: permute has an unresolved patch");
      hadError = true;
      return;
    }
    auto permArr = op.getPerm();
    unsigned n = pi->codeInfo.dataQubits;
    if (n == 0 || permArr.empty())
      return;
    auto [lo, hi] = pi->partitionRange(Partition::data);
    (void)hi;
    llvm::SmallVector<unsigned, 16> dataQubits;
    for (unsigned i = 0; i < n; ++i)
      dataQubits.push_back(lo + i);
    if (!requireInitialized(op.getOperation(), op.getPatch(), dataQubits,
                            "permute"))
      return;
    std::vector<bool> seen(n, false);
    std::string swaps;
    for (unsigned start = 0; start < n; ++start) {
      if (seen[start])
        continue;
      seen[start] = true;
      if (static_cast<unsigned>(permArr[start]) == start)
        continue;
      llvm::SmallVector<unsigned, 8> cyc{start};
      unsigned cur = static_cast<unsigned>(permArr[start]);
      while (cur < n && !seen[cur]) {
        seen[cur] = true;
        cyc.push_back(cur);
        cur = static_cast<unsigned>(permArr[cur]);
      }
      // SWAP(cyc[j], cyc[j+1]) for j = size-2 .. 0 moves content i -> perm[i].
      for (int j = static_cast<int>(cyc.size()) - 2; j >= 0; --j)
        swaps += " " + std::to_string(lo + cyc[j]) + " " +
                 std::to_string(lo + cyc[j + 1]);
    }
    if (!swaps.empty())
      os << "SWAP" << swaps << "\n";
  }

  //===--------------------------------------------------------------------===//
  // Measurement.
  //===--------------------------------------------------------------------===//

  void emitInitBasis(InitBasisOp op) {
    propagatePatch(op.getPatch(), op.getResult());
    if (!requirePatch(op.getOperation(), op.getPatch(), "init_basis"))
      return;
    std::optional<llvm::ArrayRef<int64_t>> indices;
    if (auto selected = op.getIndices())
      indices = *selected;
    auto qubits = resolvePartition(op.getPatch(), op.getPartition(), indices);
    if (qubits.empty())
      return;
    os << "R";
    for (unsigned qubit : qubits)
      os << " " << qubit;
    os << "\n";
    if (op.getBasis() == Prep::x) {
      os << "H";
      for (unsigned qubit : qubits)
        os << " " << qubit;
      os << "\n";
    }
    markInitialized(op.getResult(), qubits);
  }

  void emitMz(MzOp op) {
    propagatePatch(op.getPatch(), op.getPatchOut());
    const PatchInfo *pi =
        requirePatch(op.getOperation(), op.getPatch(), "measurement");
    if (!pi)
      return;
    std::optional<llvm::ArrayRef<int64_t>> indices;
    if (auto idx = op.getIndices())
      indices = *idx;
    auto qubits = resolvePartition(op.getPatch(), op.getPartition(), indices);
    if (qubits.empty())
      return;
    if (!requireInitialized(op.getOperation(), op.getPatch(), qubits,
                            "measurement"))
      return;

    os << "M";
    for (unsigned q : qubits)
      os << " " << q;
    os << "\n";

    unsigned firstAbs = nextMeasIndex;
    nextMeasIndex += qubits.size();
    DataMeasRecords dm;
    dm.firstAbs = firstAbs;
    dm.count = qubits.size();
    dm.codeName = pi->codeName;
    dataMeas[op.getBits()] = std::move(dm);
  }

  void emitMeasureBasis(MeasureBasisOp op) {
    propagatePatch(op.getPatch(), op.getPatchOut());
    const PatchInfo *pi =
        requirePatch(op.getOperation(), op.getPatch(), "measurement");
    if (!pi)
      return;
    std::optional<llvm::ArrayRef<int64_t>> indices;
    if (auto idx = op.getIndices())
      indices = *idx;
    auto qubits = resolvePartition(op.getPatch(), op.getPartition(), indices);
    if (qubits.empty())
      return;
    if (!requireInitialized(op.getOperation(), op.getPatch(), qubits,
                            "measurement"))
      return;

    os << (op.getBasis() == Prep::x ? "MX" : "M");
    for (unsigned q : qubits)
      os << " " << q;
    os << "\n";

    DataMeasRecords dm;
    dm.firstAbs = nextMeasIndex;
    dm.count = qubits.size();
    nextMeasIndex += qubits.size();
    dm.codeName = pi->codeName;
    dataMeas[op.getBits()] = std::move(dm);
  }

  void emitMpp(MppOp op) {
    propagatePatch(op.getPatch(), op.getPatchOut());
    const PatchInfo *pi = requirePatch(op.getOperation(), op.getPatch(), "mpp");
    if (!pi)
      return;
    auto indices = op.getIndices();
    std::optional<llvm::ArrayRef<int64_t>> selected = indices;
    auto qubits = resolvePartition(op.getPatch(), op.getPartition(), selected);
    StringRef paulis = op.getPaulis();
    bool negated = paulis.consume_front("-");
    if (qubits.size() != paulis.size()) {
      op.emitError("fabric-to-stim: mpp carrier/Pauli width mismatch");
      hadError = true;
      return;
    }
    if (!requireInitialized(op.getOperation(), op.getPatch(), qubits, "mpp"))
      return;
    os << "MPP ";
    for (unsigned index = 0; index < qubits.size(); ++index) {
      if (index)
        os << "*";
      if (negated && index == 0)
        os << "!";
      os << paulis[index] << qubits[index];
    }
    os << "\n";

    DataMeasRecords dm;
    dm.firstAbs = nextMeasIndex++;
    dm.count = 1;
    dm.codeName = pi->codeName;
    dataMeas[op.getBits()] = std::move(dm);
  }

  static std::pair<unsigned, unsigned> multiplyPauli(unsigned left,
                                                     unsigned right) {
    if (left == 0)
      return {0, right};
    if (right == 0)
      return {0, left};
    if (left == right)
      return {0, 0};
    // Return (phase exponent mod 4, Pauli mask), where i^exponent is the
    // multiplication phase and masks are X=1, Z=2, Y=3.
    if ((left == 1 && right == 3) || (left == 2 && right == 1) ||
        (left == 3 && right == 2))
      return {1, left ^ right};
    return {3, left ^ right};
  }

  void emitMeasureProduct(MeasureProductOp op) {
    // Propagate each input patch value to its corresponding result.
    auto patches = op.getPatches();
    auto patchResults = op.getPatchResults();
    for (auto [in, out] : llvm::zip(patches, patchResults))
      propagatePatch(in, out);

    auto patchIndices = op.getPatchIndices();
    auto logicalIndices = op.getLogicalIndices();
    StringRef pauli = op.getPauliProduct();
    // Optional leading '-' marks a negated product: same projectors, but the
    // recorded bit is complemented (Stim's inverted-target MPP form).
    bool negated = pauli.consume_front("-");
    unsigned phase = negated ? 2u : 0u;
    unsigned nTerms = pauli.size();

    // Resolve each logical term to its physical-qubit support, accumulating
    // the Pauli per physical qubit (X·X = Z·Z = I; X·Z = Y) so that repeated
    // qubits across logical-operator supports cancel correctly.  The result
    // is a single Stim multi-Pauli product (one measurement record).
    // Per qubit: bit 0 = X parity, bit 1 = Z parity → 1:X 2:Z 3:Y.
    llvm::MapVector<unsigned, unsigned> pauliByQubit;
    for (unsigned i = 0; i < nTerms; ++i) {
      Value patch = patches[patchIndices[i]];
      const PatchInfo *pi = lookupPatch(patch);
      if (!pi) {
        op.emitError("fabric-to-stim: measure_product on unresolved patch");
        hadError = true;
        return;
      }
      char p = pauli[i];
      if (p != 'X' && p != 'Y' && p != 'Z') {
        op.emitError("fabric-to-stim: invalid logical Pauli '") << p << "'";
        hadError = true;
        return;
      }
      unsigned lidx = static_cast<unsigned>(logicalIndices[i]);
      llvm::ArrayRef<unsigned> xSupport;
      llvm::ArrayRef<unsigned> zSupport;
      if (lidx < pi->codeInfo.k) {
        if (lidx < pi->codeInfo.lx.size() && lidx < pi->codeInfo.lz.size()) {
          xSupport = pi->codeInfo.lx[lidx];
          zSupport = pi->codeInfo.lz[lidx];
        }
      } else {
        unsigned gaugeIndex = lidx - pi->codeInfo.k;
        if (gaugeIndex < pi->codeInfo.gx.size() &&
            gaugeIndex < pi->codeInfo.gz.size()) {
          xSupport = pi->codeInfo.gx[gaugeIndex];
          zSupport = pi->codeInfo.gz[gaugeIndex];
        }
      }
      if (xSupport.empty() || zSupport.empty()) {
        op.emitError("fabric-to-stim: measure_product logical index out of "
                     "range for code logical-operator registry");
        hadError = true;
        return;
      }
      llvm::SmallVector<unsigned, 16> usedCarriers;
      for (unsigned carrier : xSupport)
        usedCarriers.push_back(pi->baseIndex + carrier);
      for (unsigned carrier : zSupport)
        usedCarriers.push_back(pi->baseIndex + carrier);
      if (!requireInitialized(op.getOperation(), patch, usedCarriers,
                              "measure_product"))
        return;
      auto multiplySupport = [&](llvm::ArrayRef<unsigned> support,
                                 unsigned mask) {
        for (unsigned j : support) {
          unsigned qubit = pi->baseIndex + j;
          auto [factor, result] =
              multiplyPauli(pauliByQubit.lookup(qubit), mask);
          phase = (phase + factor) & 3u;
          if (result)
            pauliByQubit[qubit] = result;
          else
            pauliByQubit.erase(qubit);
        }
      };
      if (p == 'X' || p == 'Y')
        multiplySupport(xSupport, 1u);
      if (p == 'Y')
        phase = (phase + 1u) & 3u;
      if (p == 'Z' || p == 'Y')
        multiplySupport(zSupport, 2u);
    }

    if (phase & 1u) {
      op.emitError("fabric-to-stim: logical Pauli product is not Hermitian");
      hadError = true;
      return;
    }

    llvm::SmallVector<std::string, 16> targets;
    for (auto &kv : pauliByQubit) {
      const char *pc = (kv.second == 1)   ? "X"
                       : (kv.second == 2) ? "Z"
                       : (kv.second == 3) ? "Y"
                                          : nullptr; // 0 ⇒ identity, drop
      if (!pc)
        continue;
      targets.push_back(std::string(pc) + std::to_string(kv.first));
    }
    if (targets.empty()) {
      op.emitError("fabric-to-stim: measure_product reduced to identity "
                   "(empty Pauli support)");
      hadError = true;
      return;
    }
    // Stim complements the record when the first target is inverted, which
    // is exactly the negated-product measurement convention.
    if (phase == 2u)
      targets.front().insert(0, "!");

    os << "MPP";
    os << " ";
    for (unsigned t = 0; t < targets.size(); ++t)
      os << (t ? "*" : "") << targets[t];
    os << "\n";

    // MPP yields exactly one measurement record.
    productMeas[op.getOutcome()] = nextMeasIndex;
    nextMeasIndex += 1;
  }

  void emitReadSyndromeAncillas(ReadSyndromeAncillasOp op) {
    propagatePatch(op.getPatch(), op.getPatchOut());
    const PatchInfo *pi = lookupPatch(op.getPatch());
    if (!pi) {
      op.emitError(
          "fabric-to-stim: read_syndrome_ancillas on unresolved patch");
      hadError = true;
      return;
    }
    auto [sxLo, sxHi] = pi->partitionRange(Partition::sx);
    auto [szLo, szHi] = pi->partitionRange(Partition::sz);
    unsigned nSx = sxHi - sxLo;
    unsigned nSz = szHi - szLo;

    llvm::SmallVector<unsigned, 16> ancillas;
    for (unsigned qubit = sxLo; qubit < sxHi; ++qubit)
      ancillas.push_back(qubit);
    for (unsigned qubit = szLo; qubit < szHi; ++qubit)
      ancillas.push_back(qubit);
    if (!requireInitialized(op.getOperation(), op.getPatch(), ancillas,
                            "read_syndrome_ancillas"))
      return;

    SyndromeRecords rec;
    rec.codeName = pi->codeName;
    rec.frame = pi->frame;

    // Read only: preparation and entangling gates are explicit Fabric ops.
    // The next extraction round is responsible for resetting its ancillas.
    if (nSx > 0) {
      llvm::SmallVector<unsigned, 8> sxQ;
      for (unsigned q = sxLo; q < sxHi; ++q)
        sxQ.push_back(q);
      os << "M";
      for (unsigned q : sxQ)
        os << " " << q;
      os << "\n";
      rec.sxFirstAbs = nextMeasIndex;
      rec.nSx = nSx;
      nextMeasIndex += nSx;
    }
    if (nSz > 0) {
      llvm::SmallVector<unsigned, 8> szQ;
      for (unsigned q = szLo; q < szHi; ++q)
        szQ.push_back(q);
      os << "M";
      for (unsigned q : szQ)
        os << " " << q;
      os << "\n";
      rec.szFirstAbs = nextMeasIndex;
      rec.nSz = nSz;
      nextMeasIndex += nSz;
    }

    syndromeMap[op.getSyndrome()] = std::move(rec);
  }

  void emitAssembleSyndrome(AssembleSyndromeOp op) {
    propagatePatch(op.getPatch(), op.getPatchOut());
    const PatchInfo *pi = lookupPatch(op.getPatch());
    auto sx = dataMeas.find(op.getSxBits());
    auto sz = dataMeas.find(op.getSzBits());
    if (!pi || sx == dataMeas.end() || sz == dataMeas.end()) {
      op.emitError(
          "fabric-to-stim: assemble_syndrome has unresolved patch or records");
      hadError = true;
      return;
    }
    if (sx->second.count != pi->codeInfo.sxQubits ||
        sz->second.count != pi->codeInfo.szQubits) {
      op.emitError(
          "fabric-to-stim: assemble_syndrome record widths do not match code");
      hadError = true;
      return;
    }
    SyndromeRecords rec;
    rec.codeName = pi->codeName;
    rec.frame = pi->frame;
    rec.sxFirstAbs = sx->second.firstAbs;
    rec.nSx = sx->second.count;
    rec.szFirstAbs = sz->second.firstAbs;
    rec.nSz = sz->second.count;
    syndromeMap[op.getSyndrome()] = std::move(rec);
  }

  //===--------------------------------------------------------------------===//
  // Measurement-record addressing.
  //===--------------------------------------------------------------------===//

  int recOffset(unsigned absIndex) const {
    return static_cast<int>(absIndex) - static_cast<int>(nextMeasIndex);
  }

  //===--------------------------------------------------------------------===//
  // Control flow: call, repeat, if.
  //===--------------------------------------------------------------------===//

  void emitCall(CallOp op) {
    Operation *callee = module.lookupSymbol(op.getCallee());
    if (!callee || !isa<GadgetOp, ProtocolOp>(callee)) {
      op.emitError("fabric-to-stim: unresolved executable call @")
          << op.getCallee();
      hadError = true;
      return;
    }

    if (!activeCallables.insert(callee).second) {
      op.emitError("fabric-to-stim: recursive executable call through @")
          << op.getCallee();
      hadError = true;
      return;
    }

    Block *entry = nullptr;
    if (auto gadget = dyn_cast<GadgetOp>(callee)) {
      entry = resolveGadgetBody(gadget, op.getOperation());
    } else if (callee->getNumRegions() == 1 && !callee->getRegion(0).empty()) {
      entry = &callee->getRegion(0).front();
    } else {
      op.emitError("fabric-to-stim: executable protocol has no body @")
          << op.getCallee();
      hadError = true;
    }
    if (!entry) {
      activeCallables.erase(callee);
      return;
    }

    // Propagate patch metadata from actuals into formals.
    for (auto [formal, actual] :
         llvm::zip(entry->getArguments(), op.getOperands())) {
      propagatePatch(actual, formal);
      propagateSyndrome(actual, formal);
      propagateData(actual, formal);
    }

    // Walk, capturing the ReturnOp to wire up results.
    SmallVector<Value> returnOperands;
    for (Operation &bodyOp : *entry) {
      if (isa<ReturnOp, ProtocolReturnOp>(&bodyOp)) {
        for (Value v : bodyOp.getOperands())
          returnOperands.push_back(v);
        break;
      }
      emitOp(&bodyOp);
    }

    // Wire call results to what the callee returned.
    for (auto [callRes, yielded] : llvm::zip(op.getResults(), returnOperands)) {
      propagatePatch(yielded, callRes);
      propagateSyndrome(yielded, callRes);
      propagateData(yielded, callRes);
    }
    activeCallables.erase(callee);
  }

  void emitRepeat(qlx::cflow::RepeatOp op) {
    int64_t count = op.getCount();
    auto &body = op.getBody().front();
    auto inits = op.getInits();

    // Initialize block args from the inits (patch/syndrome/data-bits
    // metadata propagation).
    auto seed = [&](ValueRange sources) {
      for (auto [src, arg] : llvm::zip(sources, body.getArguments())) {
        propagatePatch(src, arg);
        propagateSyndrome(src, arg);
        propagateData(src, arg);
      }
    };

    // Unroll.
    SmallVector<Value> lastYielded(inits.begin(), inits.end());
    for (int64_t iter = 0; iter < count; ++iter) {
      seed(lastYielded);
      SmallVector<Value> thisYielded;
      for (Operation &bodyOp : body) {
        if (auto y = dyn_cast<qlx::cflow::YieldOp>(&bodyOp)) {
          for (Value v : y.getOperands())
            thisYielded.push_back(v);
          break;
        }
        emitOp(&bodyOp);
      }
      // For the next iteration, we re-read metadata off the yielded
      // block-arg values, which reflect the state after this iteration.
      lastYielded.assign(thisYielded.begin(), thisYielded.end());
    }

    // Wire repeat results from the last iteration.
    for (auto [res, yielded] : llvm::zip(op.getResults(), lastYielded)) {
      propagatePatch(yielded, res);
      propagateSyndrome(yielded, res);
      propagateData(yielded, res);
    }
  }

  void emitIf(qlx::cflow::IfOp op) {
    op.emitError("fabric-to-stim: fabric.if conditional control flow is "
                 "not supported by this emitter");
    hadError = true;
  }

  //===--------------------------------------------------------------------===//
  // State.
  //===--------------------------------------------------------------------===//

  llvm::raw_ostream &os;
  ModuleOp module;
  llvm::StringMap<CodeInfo> codeReg;
  llvm::DenseMap<Value, PatchInfo> patchMap;
  llvm::DenseMap<Value, SyndromeRecords> syndromeMap;
  llvm::DenseMap<Value, DataMeasRecords> dataMeas;
  // Absolute Stim measurement index of each measure_product outcome record.
  llvm::DenseMap<Value, unsigned> productMeas;
  unsigned nextQubitIndex = 0;
  unsigned nextMeasIndex = 0;
  bool hadError = false;
  llvm::DenseSet<Operation *> activeCallables;
};

} // anonymous namespace

//===----------------------------------------------------------------------===//
// Public API.
//===----------------------------------------------------------------------===//

LogicalResult qlx::fabric::emitStim(ModuleOp module, llvm::raw_ostream &os) {
  if (failed(verify(module)))
    return failure();
  StimEmitter emitter(os, module);
  return emitter.emit();
}

//===----------------------------------------------------------------------===//
// Translation registration.
//===----------------------------------------------------------------------===//

void qlx::fabric::registerFabricToStimTranslation() {
  static TranslateFromMLIRRegistration reg(
      "fabric-to-stim", "Translate Fabric MLIR to Stim circuit text",
      [](ModuleOp module, llvm::raw_ostream &os) -> LogicalResult {
        return emitStim(module, os);
      },
      [](DialectRegistry &registry) {
        registry.insert<qlx::fabric::FabricDialect, qlx::cflow::CflowDialect>();
      });
}
