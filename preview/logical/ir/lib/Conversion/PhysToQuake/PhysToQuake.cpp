/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// Lowers the P3 `phys.graph` named by the `entry` option to one Quake entry
// point in value semantics.

#include "qlx/Conversion/PhysToQuakePasses.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/Builder/CompilerNames.h"
#include "cudaq/Optimizer/Builder/Intrinsics.h"
#include "cudaq/Optimizer/Builder/Marshal.h"
#include "cudaq/Optimizer/Builder/RuntimeNames.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/CC/CCTypes.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/JSON.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/IRMapping.h"
#include <cstdint>
#include <optional>

namespace qlx {
#define GEN_PASS_DEF_PHYSTOQUAKE
#include "qlx/Conversion/PhysToQuakePasses.h.inc"
} // namespace qlx

using namespace mlir;

/// Returns the builtin process name that an action or instrument declares, or
/// failure, with a diagnostic on `declaration`, for any other process.
/// The process, not the symbol name, defines what an action does: a device may
/// call its X action `@x`, `@flip`, or anything else.
static FailureOr<std::string> builtinProcess(Operation *declaration,
                                             StringRef process) {
  auto parsed = llvm::json::parse(process);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return declaration->emitOpError("has a process that is not valid JSON");
  }
  auto *object = parsed->getAsObject();
  auto kind = object ? object->getString("kind") : std::nullopt;
  auto name = object ? object->getString("name") : std::nullopt;
  if (!kind || *kind != "builtin" || !name)
    return declaration->emitOpError(
        "has a process that is not a builtin process, which has no exact "
        "lowering to Quake");
  // A process with parameter values, such as `"angle": 0.5`, is more than
  // the gate its name spells, so lowering it by name would drop the values.
  // Only an empty object states that there are none.
  if (const llvm::json::Value *parameters = object->get("parameters"))
    if (const llvm::json::Object *map = parameters->getAsObject();
        !map || !map->empty())
      return declaration->emitOpError(
          "has a parameterized process, which has no lowering to Quake");
  return name->str();
}

/// Resolves a symbol in the graph's machine first, then from `user` outward,
/// the way the Phys graph verifier resolves an action. A machine may define an
/// action with the same name as a module-level one, and the pass must lower
/// the action the verifier checked.
/// Note: Instruments resolve the same way here, although `MeasureOp::verify`
/// and `MeasureProductOp::verify` check the instrument they find outward from
/// the op, which skips the machine.
static Operation *resolveInMachine(SymbolTableCollection &symbols,
                                   qlx::phys::ArchitectureOp machine,
                                   Operation *user, FlatSymbolRefAttr symbol) {
  if (Operation *declared = symbols.lookupSymbolIn(machine, symbol.getAttr()))
    return declared;
  return symbols.lookupNearestSymbolFrom(user, symbol);
}

/// Builds one Quake gate on wires and returns its output wires, controls
/// first.
template <typename GateOp>
static SmallVector<Value> emitGate(OpBuilder &builder, Location loc, bool adj,
                                   ValueRange controls, ValueRange targets,
                                   ValueRange parameters = {}) {
  SmallVector<Type> wireTypes(
      controls.size() + targets.size(),
      cudaq::quake::WireType::get(builder.getContext()));
  auto gate = GateOp::create(builder, loc, wireTypes, adj, parameters, controls,
                             targets, DenseBoolArrayAttr{});
  return SmallVector<Value>(gate.getWires());
}

static Value emitH(OpBuilder &builder, Location loc, Value wire) {
  return emitGate<cudaq::quake::HOp>(builder, loc, /*adj=*/false, {}, wire)
      .front();
}

static Value emitS(OpBuilder &builder, Location loc, bool adj, Value wire) {
  return emitGate<cudaq::quake::SOp>(builder, loc, adj, {}, wire).front();
}

/// Applies `V` with `V P V^dagger = Z` for the Pauli label `pauli`, or its
/// inverse when `undo` is set, so that a Pauli product becomes a product of
/// Z operators.
static Value emitBasisChange(OpBuilder &builder, Location loc, StringRef pauli,
                             bool undo, Value wire) {
  // `H X H = Z`, and `H` is its own inverse.
  if (pauli == "X")
    return emitH(builder, loc, wire);
  // The verifier allows only X, Y, and Z, and Z needs no change.
  if (pauli != "Y")
    return wire;
  // `H Sdg Y S H = Z`, so `V` is `Sdg` then `H`, and its inverse `H` then `S`.
  if (undo)
    return emitS(builder, loc, /*adj=*/false, emitH(builder, loc, wire));
  return emitH(builder, loc, emitS(builder, loc, /*adj=*/true, wire));
}

static Value emitReset(OpBuilder &builder, Location loc, Value wire) {
  return cudaq::quake::ResetOp::create(builder, loc, TypeRange{wire.getType()},
                                       wire)
      ->getResult(0);
}

/// Applies the builtin gate `name` to the wires of one lane, in the action's
/// operand order. Returns the lane's output wires, or failure when the name
/// has no exact Quake gate of that arity. Quake has no `sdg`, `tdg`, `cx`, or
/// `cz` op: they are the adjoint `s` and `t`, and `x` and `z` controlled by
/// the lane's first wire.
static FailureOr<SmallVector<Value>> emitBuiltinGate(OpBuilder &builder,
                                                     Location loc,
                                                     StringRef name,
                                                     ValueRange lane) {
  if (lane.size() == 1) {
    if (name == "h")
      return emitGate<cudaq::quake::HOp>(builder, loc, /*adj=*/false, {}, lane);
    if (name == "x")
      return emitGate<cudaq::quake::XOp>(builder, loc, /*adj=*/false, {}, lane);
    if (name == "y")
      return emitGate<cudaq::quake::YOp>(builder, loc, /*adj=*/false, {}, lane);
    if (name == "z")
      return emitGate<cudaq::quake::ZOp>(builder, loc, /*adj=*/false, {}, lane);
    if (name == "s" || name == "sdg")
      return emitGate<cudaq::quake::SOp>(builder, loc, name == "sdg", {}, lane);
    if (name == "t" || name == "tdg")
      return emitGate<cudaq::quake::TOp>(builder, loc, name == "tdg", {}, lane);
    if (name == "reset")
      return SmallVector<Value>{emitReset(builder, loc, lane.front())};
  }
  if (lane.size() == 2) {
    if (name == "cx")
      return emitGate<cudaq::quake::XOp>(builder, loc, /*adj=*/false,
                                         lane.take_front(), lane.take_back());
    if (name == "cz")
      return emitGate<cudaq::quake::ZOp>(builder, loc, /*adj=*/false,
                                         lane.take_front(), lane.take_back());
    if (name == "swap")
      return emitGate<cudaq::quake::SwapOp>(builder, loc, /*adj=*/false, {},
                                            lane);
  }
  return failure();
}

namespace {

/// Translates one `phys.graph` into the body of a `func.func`. It walks the
/// graph's ops in program order and emits Quake at `builder`'s insertion
/// point. Call bodies and control-flow regions recurse through `lowerOp`, so
/// every op kind is handled in one place. A call template lowers its
/// canonical call's body again, once per replay.
class GraphLowering {
public:
  GraphLowering(ModuleOp module, qlx::phys::GraphOp graph)
      : module(module), graph(graph), builder(module.getContext()) {}

  /// Builds the entry point, or returns failure after a diagnostic.
  FailureOr<func::FuncOp> lower();

private:
  LogicalResult checkSidecars();
  FailureOr<FunctionType> entryType();
  LogicalResult lowerOp(Operation *op);
  LogicalResult checkCarrier(Operation *op, Value state);
  LogicalResult lowerAcquire(qlx::phys::AcquireOp op);
  LogicalResult lowerRelease(qlx::phys::ReleaseOp op);
  LogicalResult lowerPreparation(Operation *op, StringRef state,
                                 ValueRange inputs);
  LogicalResult lowerApply(qlx::phys::ApplyOp op);
  LogicalResult lowerMeasure(qlx::phys::MeasureOp op);
  LogicalResult lowerMeasureProduct(qlx::phys::MeasureProductOp op);
  LogicalResult lowerRotateProduct(qlx::phys::RotateProductOp op);
  LogicalResult lowerCall(qlx::phys::CallOp op);
  LogicalResult lowerCallTemplate(qlx::phys::CallTemplateOp op);
  LogicalResult lowerBody(qlx::phys::CallOp call, ValueRange inputs,
                          ValueRange outputs);
  LogicalResult lowerIf(qlx::cflow::IfOp op);
  LogicalResult lowerRepeat(qlx::cflow::RepeatOp op);
  LogicalResult lowerWhile(qlx::cflow::WhileOp op);
  LogicalResult lowerOps(Block &block);
  SmallVector<Attribute> liveCarriers() const;
  void enterRegion(ValueRange arguments, ArrayRef<Attribute> carriers,
                   ValueRange physArguments);
  SmallVector<Value> carriedValues(ArrayRef<Attribute> carriers,
                                   ValueRange physValues);
  void mapResults(ArrayRef<Attribute> carriers, ValueRange physResults,
                  ValueRange results);
  LogicalResult lowerCondition(qlx::phys::ConditionOp op);
  LogicalResult lowerAllFalse(qlx::phys::AllFalseOp op);
  LogicalResult lowerReturn(qlx::phys::ReturnOp op);
  Value toBoolean(Value physValue);
  Attribute carrierOf(Value state) const;
  Value wireOf(Value state) const;
  void setWire(Value state, Value wire);
  void mapValue(Value from, Value to);
  void rollbackValues(std::size_t mark);

  ModuleOp module;
  qlx::phys::GraphOp graph;
  qlx::phys::ArchitectureOp machine;
  OpBuilder builder;
  SymbolTableCollection symbols;
  /// Classical values and records. A measured record maps to its
  /// `!cc.measure_handle` until a Boolean is read from it, then to that `i1`.
  /// A record from `arith.select` or from a region boundary maps to an `i1`.
  /// States are tracked in `wires`.
  IRMapping values;
  /// The mapping that each write to `values` replaced, or null for a new key.
  /// A control-flow region rolls back its writes when it ends, because they
  /// do not dominate the code after it. Copying `values` per region instead
  /// makes a graph with many control-flow ops lower in quadratic time.
  SmallVector<std::pair<Value, Value>> undoLog;
  /// The current wire of each live carrier, keyed by its physical resource.
  /// Note: An elided `phys.call_template` advances its carriers without new
  /// SSA states, and later ops keep using the states from before it. So a
  /// state's wire is found through its resource, never through its SSA value.
  DenseMap<Attribute, Value> wires;
  /// While a template replays a canonical call body, maps each resource that
  /// the body names to the carrier that the invocation acts on.
  DenseMap<Attribute, Attribute> aliases;
  /// The `phys.call` that each `event_id` names, for `phys.call_template`.
  llvm::StringMap<qlx::phys::CallOp> canonicalCalls;
  /// Handles of inverted product measurements, whose bit is complemented
  /// where a Boolean is consumed.
  DenseSet<Value> invertedHandles;
  /// The `phys.call` bodies being lowered, as calls or as template replays.
  /// A template that replays one of them would recurse without end.
  SmallPtrSet<Operation *, 4> activeCalls;
};

} // namespace

/// A record or an `i1` is Boolean. Every record that the pass sees comes from
/// a measurement it lowers to one `quake.mz`, whatever the record's schema.
static bool isBoolean(Type type) {
  return isa<qlx::phys::RecordType>(type) || type.isSignlessInteger(1);
}

LogicalResult GraphLowering::checkSidecars() {
  for (auto sidecar : module.getOps<qlx::phys::SelectionSidecarOp>())
    if (sidecar.getGraph() == graph.getSymName())
      return sidecar.emitOpError(
          "post-selects shots of the entry graph, which `phys-to-quake` does "
          "not support");
  return success();
}

FailureOr<FunctionType> GraphLowering::entryType() {
  FunctionType graphType = graph.getFunctionType();
  SmallVector<Type> inputs;
  // An input becomes a parameter of the entry point, which is a CUDA-Q
  // kernel, so it must be a scalar type that a CUDA-Q kernel takes as an
  // argument. Integers must also be signless, because the `arith` ops that
  // the pass clones accept only signless integers. A state or a record has
  // no value outside the graph, so it cannot be a parameter.
  for (auto [i, type] : llvm::enumerate(graphType.getInputs())) {
    bool supported =
        isa<IndexType>(type) || type.isF32() || type.isF64() ||
        llvm::any_of(ArrayRef<unsigned>{1, 8, 16, 32, 64}, [&](unsigned width) {
          return type.isSignlessInteger(width);
        });
    if (!supported)
      return graph.emitOpError()
             << "takes input " << i << " of type " << type
             << ", but `phys-to-quake` translates only inputs of type index, "
                "f32, f64, and signless i1, i8, i16, i32, and i64";
    // `index` has no fixed width, so its parameter is an `i64`. Step 5 of
    // `lower()` casts it back.
    inputs.push_back(isa<IndexType>(type) ? builder.getI64Type() : type);
  }
  // The entry point returns nothing for a graph with no results, an `i1` for
  // one Boolean result, and one `!cc.sequence<i1>` in return order for two or
  // more. A Boolean result is a measurement record of any schema or an `i1`.
  auto ret = cast<qlx::phys::ReturnOp>(graph.getBody().front().getTerminator());
  for (auto [i, type] : llvm::enumerate(graphType.getResults()))
    if (!isBoolean(type))
      return ret.emitOpError()
             << "returns result " << i << " of type " << type
             << ", but `phys-to-quake` translates only Boolean results: "
                "measurement records and i1 values";
  SmallVector<Type> results;
  if (graphType.getNumResults() == 1)
    results.push_back(builder.getI1Type());
  else if (graphType.getNumResults() > 1)
    results.push_back(cudaq::cc::SequenceType::get(builder.getI1Type()));
  return builder.getFunctionType(inputs, results);
}

FailureOr<func::FuncOp> GraphLowering::lower() {
  // Step 1: Reject a graph that a selection sidecar post-selects.
  if (failed(checkSidecars()))
    return failure();
  // Step 2: Compute the entry signature. This rejects the input and result
  // types that the pass does not translate, before any Quake is emitted.
  auto type = entryType();
  if (failed(type))
    return failure();
  // Step 3: Look up the graph's machine, and index each `phys.call` by its
  // `event_id` for the call templates that name it.
  // The Phys verifier guarantees that the architecture resolves.
  machine = symbols.lookupNearestSymbolFrom<qlx::phys::ArchitectureOp>(
      graph, graph.getArchitectureAttr());
  // The Phys verifier guarantees that each template names exactly one
  // earlier `phys.call` of this graph.
  graph.walk([&](qlx::phys::CallOp call) {
    if (auto event = call.getEventIdAttr())
      canonicalCalls[event.getValue()] = call;
  });

  // Step 4: Create the entry point.
  // The entry point has no parent until the pass erases the source ops.
  // `builder` has no insertion point yet. The front-ends' kernel prefix keeps
  // the name apart from the intrinsics the pass loads. `cudaq-entrypoint`
  // marks the module's one entry point, and core's QIR conversion marks the
  // QIR entry point only on a `cudaq-kernel` function.
  auto entry = func::FuncOp::create(
      builder, graph.getLoc(),
      (Twine(cudaq::runtime::cudaqGenPrefixName) + graph.getSymName()).str(),
      *type);
  entry->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
  entry->setAttr(cudaq::kernelAttrName, builder.getUnitAttr());
  Block *body = entry.addEntryBlock();
  builder.setInsertionPointToStart(body);
  // Step 5: Map the graph's arguments to the entry's. An `index` argument
  // arrives as `i64` and is cast back, so cloned `index` arithmetic still
  // type-checks.
  for (auto [graphArg, entryArg] :
       llvm::zip(graph.getBody().getArguments(), body->getArguments())) {
    Value value = entryArg;
    if (isa<IndexType>(graphArg.getType()))
      value = arith::IndexCastOp::create(builder, graphArg.getLoc(),
                                         builder.getIndexType(), entryArg);
    mapValue(graphArg, value);
  }
  // Step 6: Translate each op in program order. On failure, the diagnostic is
  // already on the P3 op, and the half-built entry point is discarded.
  for (Operation &op : graph.getBody().front())
    if (failed(lowerOp(&op))) {
      entry.erase();
      return failure();
    }
  return entry;
}

/// Translates one op at `builder`'s insertion point. A case emits Quake, maps
/// the op's results, both, or neither. A rejected op gets a diagnostic. An
/// `arith` op with no case is cloned, and any other op with no case gets
/// "has no lowering to Quake".
LogicalResult GraphLowering::lowerOp(Operation *op) {
  Location loc = op->getLoc();
  return TypeSwitch<Operation *, LogicalResult>(op)
      .Case([&](qlx::phys::AcquireOp acquire) { return lowerAcquire(acquire); })
      .Case([&](qlx::phys::ReleaseOp release) { return lowerRelease(release); })
      .Case([&](qlx::phys::PrepareOp prepare) {
        return lowerPreparation(prepare, prepare.getState(),
                                prepare.getInputs());
      })
      .Case([&](qlx::phys::ResetOp reset) {
        return lowerPreparation(reset, reset.getState(), reset.getInputs());
      })
      .Case([&](qlx::phys::ApplyOp apply) { return lowerApply(apply); })
      .Case([&](qlx::phys::MeasureOp measure) { return lowerMeasure(measure); })
      .Case([&](qlx::phys::MeasureProductOp measure) {
        return lowerMeasureProduct(measure);
      })
      .Case([&](qlx::phys::RotateProductOp rotate) {
        return lowerRotateProduct(rotate);
      })
      .Case([&](qlx::phys::CallOp call) { return lowerCall(call); })
      .Case([&](qlx::phys::CallTemplateOp invocation) {
        return lowerCallTemplate(invocation);
      })
      .Case([&](qlx::cflow::IfOp ifOp) { return lowerIf(ifOp); })
      .Case([&](qlx::cflow::RepeatOp repeat) { return lowerRepeat(repeat); })
      .Case([&](qlx::cflow::WhileOp loop) { return lowerWhile(loop); })
      .Case([&](qlx::phys::ConditionOp condition) {
        return lowerCondition(condition);
      })
      .Case([&](qlx::phys::XorOp xorOp) {
        mapValue(xorOp.getResult(),
                 arith::XOrIOp::create(builder, loc,
                                       values.lookup(xorOp.getLhs()),
                                       values.lookup(xorOp.getRhs())));
        return success();
      })
      .Case([&](qlx::phys::AllFalseOp allFalse) {
        return lowerAllFalse(allFalse);
      })
      // Scheduling and transport do not change the ideal state, and they keep
      // each state's resource, so its wire stays where it is.
      .Case<qlx::phys::DelayOp, qlx::phys::BarrierOp, qlx::phys::MoveOp,
            qlx::event::FenceOp>([&](Operation *) { return success(); })
      .Case([&](qlx::phys::ReturnOp ret) { return lowerReturn(ret); })
      .Case([&](qlx::event::SelectionOp selection) {
        return selection.emitOpError(
            "post-selects shots, which `phys-to-quake` does not support");
      })
      // A resource flow moves an opaque payload, such as a factory's magic
      // state. The graph does not state the gates that make or consume it.
      .Case<qlx::phys::ProduceResourceOp, qlx::phys::TransportResourceOp,
            qlx::phys::UnpackResourceOp, qlx::phys::PackResourceOp,
            qlx::phys::ResourceRequestOp, qlx::phys::ConsumeResourceOp,
            qlx::phys::DiscardResourcePayloadOp,
            qlx::phys::ResourceRotateProductOp, qlx::phys::FactoryStartOp>(
          [&](Operation *flow) {
            return flow->emitOpError("is part of a resource or factory flow, "
                                     "which `phys-to-quake` does not support");
          })
      .Case([&](qlx::phys::EpochTransitionOp transition) {
        return transition.emitOpError(
            "changes the code epoch, and `phys-to-quake` cannot check that the "
            "operations before it realize the declared logical action");
      })
      // `arith` computes classical values, such as index arithmetic and
      // comparisons, that mean the same in a kernel, so it is cloned with
      // remapped operands. Anything else has no rule.
      .Default([&](Operation *other) -> LogicalResult {
        if (!isa_and_present<arith::ArithDialect>(other->getDialect()))
          return other->emitOpError("has no lowering to Quake");
        // A record is Boolean, so `arith.select`, the only arith op that takes
        // records, selects their bits. Selecting the measurement handles
        // instead would drop an inverted product's complement, which
        // `toBoolean` finds through the product's own handle.
        for (Value operand : other->getOperands())
          if (isa<qlx::phys::RecordType>(operand.getType()))
            toBoolean(operand);
        // The clone maps its results in `values`, so log what they replace.
        for (Value result : other->getResults())
          undoLog.emplace_back(result, values.lookupOrNull(result));
        Operation *lowered = builder.clone(*other, values);
        for (Value result : lowered->getResults())
          if (isa<qlx::phys::RecordType>(result.getType()))
            result.setType(builder.getI1Type());
        return success();
      });
}

/// Rejects a state of a patch-granularity resource. The pass gives each state
/// one qubit, which matches a carrier-granularity resource: one physical
/// two-level system. A patch-granularity resource stands for a whole encoded
/// code patch.
LogicalResult GraphLowering::checkCarrier(Operation *op, Value state) {
  // The Phys verifier resolves every state's resource, and the resource's
  // class in the graph's machine.
  auto resourceName = cast<qlx::phys::StateType>(state.getType()).getResource();
  auto resource =
      symbols.lookupNearestSymbolFrom<qlx::phys::ResourceOp>(op, resourceName);
  auto resourceClass = cast<qlx::phys::ResourceClassOp>(symbols.lookupSymbolIn(
      machine, resource.getResourceClassAttr().getAttr()));
  if (resourceClass.getGranularity() == "patch")
    return op->emitOpError()
           << "acquires resource " << resourceName
           << " of patch granularity, which `phys-to-quake` does not support";
  return success();
}

/// The physical resource that holds `state`. A state's type names its
/// resource. While a call template replays a body written for other
/// resources, `aliases` renames them to the invocation's.
Attribute GraphLowering::carrierOf(Value state) const {
  Attribute resource =
      cast<qlx::phys::StateType>(state.getType()).getResource();
  return aliases.lookup_or(resource, resource);
}

/// The current wire of the qubit that holds `state`.
Value GraphLowering::wireOf(Value state) const {
  return wires.lookup(carrierOf(state));
}

/// Makes `wire` the current wire of the qubit that holds `state`, so the next
/// op on that qubit continues from it.
void GraphLowering::setWire(Value state, Value wire) {
  wires[carrierOf(state)] = wire;
}

/// Maps `from` to `to` in `values`, and logs the mapping that it replaces.
void GraphLowering::mapValue(Value from, Value to) {
  undoLog.emplace_back(from, values.lookupOrNull(from));
  values.map(from, to);
}

/// Restores `values` to its state when `undoLog` held `mark` entries. A
/// replaced mapping comes back, such as the handle of a record that
/// `toBoolean` remapped to a region-local `i1`. A new key is erased, so that
/// the scope check in `lowerCallTemplate` does not find a region-local value.
void GraphLowering::rollbackValues(std::size_t mark) {
  while (undoLog.size() > mark) {
    auto [from, replaced] = undoLog.pop_back_val();
    if (replaced)
      values.map(from, replaced);
    else
      values.erase(from);
  }
}

LogicalResult GraphLowering::lowerAcquire(qlx::phys::AcquireOp op) {
  // The qubit is allocated here, not at the top of the kernel, so its
  // lifetime in Quake is its lifetime in P3.
  for (Value state : op.getStates()) {
    if (failed(checkCarrier(op, state)))
      return failure();
    // The Phys verifier allows one acquire op per resource, but a call
    // template replays the acquire ops of its canonical body.
    if (wires.contains(carrierOf(state)))
      return op.emitOpError() << "acquires resource " << carrierOf(state)
                              << ", which already holds a live state";
    setWire(state, cudaq::quake::NullWireOp::create(
                       builder, op.getLoc(),
                       cudaq::quake::WireType::get(builder.getContext())));
  }
  return success();
}

LogicalResult GraphLowering::lowerRelease(qlx::phys::ReleaseOp op) {
  // The resource can be acquired again later, which starts a new wire.
  for (Value state : op.getStates()) {
    cudaq::quake::SinkOp::create(builder, op.getLoc(), wireOf(state));
    wires.erase(carrierOf(state));
  }
  return success();
}

/// Lowers `phys.prepare` and `phys.reset`, whose `state` names mean the same
/// thing.
/// A preparation can act on a reused qubit in any state, so it resets instead
/// of assuming |0>.
LogicalResult GraphLowering::lowerPreparation(Operation *op, StringRef state,
                                              ValueRange inputs) {
  // The verifier accepts any `phys.prepare` state name, and `zero` or `plus`
  // for `phys.reset`. These are the `phys.prepare` names that
  // `cudaq.logical.lower.lanes` also lowers, with the same meaning.
  bool zero = state == "zero" || state == "0" || state == "loaded";
  bool plus = state == "plus" || state == "+";
  if (!zero && !plus)
    return op->emitOpError() << "prepares state '" << state
                             << "', which has no lowering to Quake";
  for (Value input : inputs) {
    Value wire = emitReset(builder, op->getLoc(), wireOf(input));
    if (plus)
      wire = emitH(builder, op->getLoc(), wire);
    setWire(input, wire);
  }
  return success();
}

LogicalResult GraphLowering::lowerApply(qlx::phys::ApplyOp op) {
  auto action = dyn_cast_or_null<qlx::phys::ActionOp>(
      resolveInMachine(symbols, machine, op, op.getActionAttr()));
  if (!action)
    return op.emitOpError() << "names action " << op.getActionAttr()
                            << ", which resolves to no phys.action";
  auto name = builtinProcess(action, action.getProcess());
  if (failed(name))
    return failure();
  // Reservations pass values that the process does not state, such as an
  // angle or a permutation.
  if (auto reservations = op.getReservationsAttr())
    return op.emitOpError() << "carries reservations " << reservations
                            << ", which have no lowering to Quake";
  // A broadcast action acts on each state alone. Otherwise each consecutive
  // group of `arity` states is one lane.
  std::size_t arity = action.getBroadcast() ? 1 : action.getArity();
  ValueRange inputs = op.getInputs();
  SmallVector<Value> inputWires;
  for (Value input : inputs)
    inputWires.push_back(wireOf(input));
  for (std::size_t lane = 0; lane < inputs.size(); lane += arity) {
    auto laneWires = ValueRange(inputWires).slice(lane, arity);
    auto outputs = emitBuiltinGate(builder, op.getLoc(), *name, laneWires);
    if (failed(outputs))
      return op.emitOpError()
             << "applies builtin process '" << *name << "' to " << arity
             << " states, which has no lowering to Quake";
    for (auto [i, wire] : llvm::enumerate(*outputs))
      setWire(inputs[lane + i], wire);
  }
  return success();
}

LogicalResult GraphLowering::lowerMeasure(qlx::phys::MeasureOp op) {
  auto instrument = dyn_cast_or_null<qlx::phys::InstrumentOp>(
      resolveInMachine(symbols, machine, op, op.getMeasurementAttr()));
  if (!instrument)
    return op.emitOpError() << "names instrument " << op.getMeasurementAttr()
                            << ", which resolves to no phys.instrument";
  // The process selects the basis. The Phys verifier ties the instrument's
  // kind to its process only for an instrument outside the machine.
  auto name = builtinProcess(instrument, instrument.getProcess());
  if (failed(name))
    return failure();
  bool xBasis = *name == "measure_x";
  if (!xBasis && *name != "measure_z")
    return op.emitOpError() << "measures with builtin process '" << *name
                            << "', which has no lowering to Quake";
  Location loc = op.getLoc();
  Value wire = wireOf(op.getInput());
  if (xBasis)
    wire = emitH(builder, loc, wire);
  auto *ctx = builder.getContext();
  auto mz = cudaq::quake::MzOp::create(
      builder, loc, cudaq::cc::MeasureHandleType::get(ctx),
      TypeRange{cudaq::quake::WireType::get(ctx)}, ValueRange{wire},
      StringAttr{});
  // The record stays a measurement handle until a Boolean is read from it
  // (`toBoolean`), as in the CUDA-Q front-ends, so a target can defer
  // reading the result.
  mapValue(op.getRecord(), mz.getMeasOut());
  wire = mz.getWires().front();
  if (op.getDestructive()) {
    cudaq::quake::SinkOp::create(builder, loc, wire);
    wires.erase(carrierOf(op.getInput()));
    return success();
  }
  // Core's `quake.mx` does not rotate the state back, so an X measurement
  // that keeps its state is `h; mz; h`.
  if (xBasis)
    wire = emitH(builder, loc, wire);
  setWire(op.getInput(), wire);
  return success();
}

LogicalResult
GraphLowering::lowerMeasureProduct(qlx::phys::MeasureProductOp op) {
  auto instrument = dyn_cast_or_null<qlx::phys::InstrumentOp>(
      resolveInMachine(symbols, machine, op, op.getInstrumentAttr()));
  if (!instrument)
    return op.emitOpError() << "names instrument " << op.getInstrumentAttr()
                            << ", which resolves to no phys.instrument";
  auto name = builtinProcess(instrument, instrument.getProcess());
  if (failed(name))
    return failure();
  if (*name != "measure_product")
    return op.emitOpError() << "measures with builtin process '" << *name
                            << "', which has no lowering to Quake";
  // Quake has no Pauli-product measurement, so the parity of the product
  // collects on a fresh ancilla:
  //   ancilla = |0>; for each state: V_P; cx(state, ancilla); V_P^dagger
  //   mz(ancilla); sink(ancilla)
  // Only the parity is measured, so every measured state keeps its
  // post-measurement state. Measuring each state instead would collapse more
  // than the product.
  Location loc = op.getLoc();
  auto *ctx = builder.getContext();
  Value ancilla = cudaq::quake::NullWireOp::create(
      builder, loc, cudaq::quake::WireType::get(ctx));
  for (auto [input, pauli] : llvm::zip(op.getInputs(), op.getPaulis())) {
    StringRef label = cast<StringAttr>(pauli).getValue();
    Value wire =
        emitBasisChange(builder, loc, label, /*undo=*/false, wireOf(input));
    auto cx =
        emitGate<cudaq::quake::XOp>(builder, loc, /*adj=*/false, wire, ancilla);
    ancilla = cx[1];
    setWire(input, emitBasisChange(builder, loc, label, /*undo=*/true, cx[0]));
  }
  auto mz = cudaq::quake::MzOp::create(
      builder, loc, cudaq::cc::MeasureHandleType::get(ctx),
      TypeRange{cudaq::quake::WireType::get(ctx)}, ValueRange{ancilla},
      StringAttr{});
  cudaq::quake::SinkOp::create(builder, loc, mz.getWires().front());
  mapValue(op.getRecord(), mz.getMeasOut());
  // The complement is applied where the bit is read, not as an `x` on the
  // ancilla, which a noise model would also act on.
  if (op.getInvert())
    invertedHandles.insert(mz.getMeasOut());
  return success();
}

LogicalResult GraphLowering::lowerRotateProduct(qlx::phys::RotateProductOp op) {
  // `exp(-i angle P / 2)` is Logical's rotation convention. Change each Pauli
  // to Z, collect the parity on the last state with a CX ladder, rotate it
  // about Z, and undo. A Z rotation of the parity is a rotation of the whole
  // product, and the reverse ladder uncomputes the parity.
  Location loc = op.getLoc();
  SmallVector<Value> rotated;
  SmallVector<StringRef> labels;
  for (auto [input, pauli] : llvm::zip(op.getInputs(), op.getPaulis())) {
    labels.push_back(cast<StringAttr>(pauli).getValue());
    rotated.push_back(emitBasisChange(builder, loc, labels.back(),
                                      /*undo=*/false, wireOf(input)));
  }
  Value &target = rotated.back();
  for (Value &control : MutableArrayRef(rotated).drop_back()) {
    auto cx = emitGate<cudaq::quake::XOp>(builder, loc, /*adj=*/false, control,
                                          target);
    control = cx[0];
    target = cx[1];
  }
  Value theta = arith::ConstantFloatOp::create(
      builder, loc, builder.getF64Type(), op.getAngle());
  target = emitGate<cudaq::quake::RzOp>(builder, loc, /*adj=*/false, {}, target,
                                        theta)
               .front();
  for (Value &control : llvm::reverse(MutableArrayRef(rotated).drop_back())) {
    auto cx = emitGate<cudaq::quake::XOp>(builder, loc, /*adj=*/false, control,
                                          target);
    control = cx[0];
    target = cx[1];
  }
  for (auto [input, label, wire] : llvm::zip(op.getInputs(), labels, rotated))
    setWire(input, emitBasisChange(builder, loc, label, /*undo=*/true, wire));
  return success();
}

/// Inlines a call body: its block arguments take `inputs`, and its yielded
/// values become `outputs`. States need no mapping, because a call keeps each
/// state's resource.
/// Rationale: An inlined body needs no calling convention for wires.
LogicalResult GraphLowering::lowerBody(qlx::phys::CallOp call,
                                       ValueRange inputs, ValueRange outputs) {
  Block &body = call.getBody().front();
  for (auto [argument, input] : llvm::zip(body.getArguments(), inputs))
    if (!isa<qlx::phys::StateType>(argument.getType()))
      mapValue(argument, values.lookup(input));
  activeCalls.insert(call);
  LogicalResult lowered = lowerOps(body);
  activeCalls.erase(call);
  if (failed(lowered))
    return failure();
  auto yield = cast<qlx::phys::YieldOp>(body.getTerminator());
  for (auto [yielded, output] : llvm::zip(yield.getOperands(), outputs))
    if (!isa<qlx::phys::StateType>(yielded.getType()))
      mapValue(output, values.lookup(yielded));
  return success();
}

LogicalResult GraphLowering::lowerCall(qlx::phys::CallOp op) {
  return lowerBody(op, op.getInputs(), op.getOutputs());
}

/// Replays the body of the earlier `phys.call` that `template_event` names.
/// P3 states a repeated gadget call once, as that canonical body, and then as
/// templates that run it again, on the same qubits or, through
/// `state_aliases`, on others.
LogicalResult GraphLowering::lowerCallTemplate(qlx::phys::CallTemplateOp op) {
  qlx::phys::CallOp canonical = canonicalCalls.lookup(op.getTemplateEvent());
  // The replay is inlined, so a template inside its own canonical call would
  // inline itself without end.
  if (activeCalls.contains(canonical))
    return op.emitOpError()
           << "replays the phys.call '" << op.getTemplateEvent()
           << "' that encloses it, which an inlined replay cannot express";
  // The canonical call can sit in a control-flow region that has ended. Its
  // body may read a classical value from that region, which has no mapping
  // here.
  Region &body = canonical.getBody();
  WalkResult scope = body.walk([&](Operation *inner) {
    // The Phys verifier rejects a call body that uses an outer state, so only
    // classical values and records are read from outside the body.
    for (Value operand : inner->getOperands())
      if (!body.isAncestor(operand.getParentRegion()) &&
          !values.contains(operand))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (scope.wasInterrupted())
    return op.emitOpError()
           << "replays the phys.call '" << op.getTemplateEvent()
           << "', whose body reads a value that is out of scope here";
  // The body names the canonical call's resources. `state_aliases` renames
  // them to this invocation's carriers, and an enclosing replay renames those
  // in turn. The Phys verifier makes `resource_substitutions`, when present,
  // mirror `state_aliases`, so the pass reads only `state_aliases`.
  DenseMap<Attribute, Attribute> enclosing = aliases;
  if (auto stateAliases = op.getStateAliasesAttr())
    for (auto alias : stateAliases.getAsRange<DictionaryAttr>()) {
      Attribute invocationResource = alias.get("alias");
      aliases[alias.get("template")] =
          enclosing.lookup_or(invocationResource, invocationResource);
    }
  // The elided form names no states, so the Phys verifier cannot check that
  // the carriers it acts on are live.
  if (op.getStateBoundaryElided())
    for (Type type : canonical.getInputs().getTypes()) {
      Attribute resource = cast<qlx::phys::StateType>(type).getResource();
      Attribute carrier = aliases.lookup_or(resource, resource);
      if (!wires.contains(carrier)) {
        aliases = std::move(enclosing);
        return op.emitOpError()
               << "replays the phys.call '" << op.getTemplateEvent()
               << "' on resource " << carrier
               << ", which holds no live state here";
      }
    }
  // The elided form has an all-state boundary, so the body needs no mapping:
  // it acts on the carriers' current wires.
  LogicalResult lowered =
      op.getStateBoundaryElided()
          ? lowerBody(canonical, ValueRange{}, ValueRange{})
          : lowerBody(canonical, op.getInputs(), op.getOutputs());
  aliases = std::move(enclosing);
  return lowered;
}

/// Lowers the ops of a call body or a control-flow region. The caller handles
/// the terminator, because only it knows what the yielded values become.
LogicalResult GraphLowering::lowerOps(Block &block) {
  for (Operation &op : block.without_terminator())
    if (failed(lowerOp(&op)))
      return failure();
  return success();
}

/// The live carriers in resource-name order, so that the Quake is
/// deterministic.
SmallVector<Attribute> GraphLowering::liveCarriers() const {
  SmallVector<Attribute> carriers(llvm::make_first_range(wires));
  llvm::sort(carriers, [](Attribute lhs, Attribute rhs) {
    return cast<FlatSymbolRefAttr>(lhs).getValue() <
           cast<FlatSymbolRefAttr>(rhs).getValue();
  });
  return carriers;
}

// Control flow carries every live carrier's wire through the Quake region,
// followed by the cflow op's classical values. A record crosses a region
// boundary as its `i1`, which `toBoolean` complements for an inverted product.
// States need no mapping inside a region, because each keeps its resource.
// Every live carrier is carried, not only the ones a region names, because a
// region can act on a carrier without naming it, through an elided call
// template.
//
//   cflow.if %c -> (state<@a>, record)      %r:2 = cc.if(%c) ((%w = %a)) {
//     ...                              =>     ...
//     cflow.yield %a2, %m                     cc.continue %w2, %bit
//   } else { ... }                          } else { ... }

/// Maps a region's arguments: `arguments` holds the carriers' wires, then the
/// carried values of `physArguments`.
void GraphLowering::enterRegion(ValueRange arguments,
                                ArrayRef<Attribute> carriers,
                                ValueRange physArguments) {
  for (auto [carrier, wire] : llvm::zip(carriers, arguments))
    wires[carrier] = wire;
  ValueRange carried = arguments.drop_front(carriers.size());
  for (Value physArgument : physArguments)
    if (!isa<qlx::phys::StateType>(physArgument.getType())) {
      mapValue(physArgument, carried.front());
      carried = carried.drop_front();
    }
}

/// The carriers' wires, then the carried values of `physValues`.
SmallVector<Value> GraphLowering::carriedValues(ArrayRef<Attribute> carriers,
                                                ValueRange physValues) {
  SmallVector<Value> carried;
  for (Attribute carrier : carriers)
    carried.push_back(wires.lookup(carrier));
  for (Value physValue : physValues) {
    if (isa<qlx::phys::StateType>(physValue.getType()))
      continue;
    carried.push_back(isa<qlx::phys::RecordType>(physValue.getType())
                          ? toBoolean(physValue)
                          : values.lookup(physValue));
  }
  return carried;
}

/// Maps a control-flow op's results back: `results` holds the carriers'
/// wires, then the carried values of `physResults`. A carrier that is not in
/// `carriers` is no longer live after the op.
void GraphLowering::mapResults(ArrayRef<Attribute> carriers,
                               ValueRange physResults, ValueRange results) {
  wires.clear();
  for (auto [carrier, wire] : llvm::zip(carriers, results))
    wires[carrier] = wire;
  ValueRange carried = results.drop_front(carriers.size());
  for (Value physResult : physResults)
    if (!isa<qlx::phys::StateType>(physResult.getType())) {
      mapValue(physResult, carried.front());
      carried = carried.drop_front();
    }
}

LogicalResult GraphLowering::lowerIf(qlx::cflow::IfOp op) {
  Location loc = op.getLoc();
  Value condition = values.lookup(op.getCondition());
  SmallVector<Attribute> carriers = liveCarriers();
  SmallVector<Value> linearArgs = carriedValues(carriers, {});
  // Mappings made in a branch do not dominate the code after the `cc.if`.
  std::size_t mark = undoLog.size();
  DenseMap<Attribute, Value> outerWires = wires;
  // The result types follow from the branches, so they are lowered into a
  // scratch `cc.if` whose regions then move to the final one.
  auto noRegion = [](OpBuilder &, Location, Region &) {};
  auto scratch = cudaq::cc::IfOp::create(builder, loc, TypeRange{}, condition,
                                         linearArgs, noRegion, noRegion);
  // The Phys verifier makes both branches consume and yield the same states,
  // so both end with the same carriers.
  SmallVector<Attribute> exitCarriers;
  SmallVector<Type> resultTypes;
  for (auto [source, target] :
       llvm::zip(op->getRegions(), scratch->getRegions())) {
    rollbackValues(mark);
    wires = outerWires;
    Block *block = new Block();
    target.push_back(block);
    for (Value wire : linearArgs)
      block->addArgument(wire.getType(), loc);
    enterRegion(block->getArguments(), carriers, {});
    builder.setInsertionPointToStart(block);
    if (failed(lowerOps(source.front())))
      return failure();
    exitCarriers = liveCarriers();
    SmallVector<Value> continued = carriedValues(
        exitCarriers, source.front().getTerminator()->getOperands());
    resultTypes = llvm::to_vector(ValueRange(continued).getTypes());
    cudaq::cc::ContinueOp::create(builder, loc, continued);
  }
  builder.setInsertionPoint(scratch);
  auto ifOp = cudaq::cc::IfOp::create(
      builder, loc, resultTypes, condition, linearArgs,
      [&](OpBuilder &, Location, Region &region) {
        region.takeBody(scratch.getThenRegion());
      },
      [&](OpBuilder &, Location, Region &region) {
        region.takeBody(scratch.getElseRegion());
      });
  scratch.erase();
  builder.setInsertionPointAfter(ifOp);
  rollbackValues(mark);
  mapResults(exitCarriers, op.getResults(), ifOp.getResults());
  return success();
}

/// Lowers `cflow.repeat` to a counted `cc.loop` with an `i64` index. The loop
/// stays rolled, as P3 states it. A count of 0 runs no iteration.
LogicalResult GraphLowering::lowerRepeat(qlx::cflow::RepeatOp op) {
  Location loc = op.getLoc();
  SmallVector<Attribute> carriers = liveCarriers();
  Value count = arith::ConstantIntOp::create(builder, loc, op.getCount(), 64);
  SmallVector<Value> iterArgs{
      arith::ConstantIntOp::create(builder, loc, 0, 64)};
  llvm::append_range(iterArgs, carriedValues(carriers, op.getInits()));
  auto types = llvm::to_vector(ValueRange(iterArgs).getTypes());
  std::size_t mark = undoLog.size();
  Block &body = op.getBody().front();
  LogicalResult lowered = success();
  // The Phys verifier makes the body yield every state it receives, so each
  // iteration ends with the carriers it began with.
  auto loop = cudaq::cc::LoopOp::create(
      builder, loc, types, iterArgs, /*postCond=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        Value more =
            arith::CmpIOp::create(builder, loc, arith::CmpIPredicate::slt,
                                  block.getArgument(0), count);
        cudaq::cc::ConditionOp::create(builder, loc, more,
                                       block.getArguments());
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        enterRegion(block.getArguments().drop_front(), carriers,
                    body.getArguments());
        lowered = lowerOps(body);
        if (failed(lowered))
          return;
        SmallVector<Value> continued{block.getArgument(0)};
        llvm::append_range(
            continued,
            carriedValues(carriers, body.getTerminator()->getOperands()));
        cudaq::cc::ContinueOp::create(builder, loc, continued);
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        SmallVector<Value> next(block.getArguments());
        next.front() = arith::AddIOp::create(
            builder, loc, next.front(),
            arith::ConstantIntOp::create(builder, loc, 1, 64));
        cudaq::cc::ContinueOp::create(builder, loc, next);
      });
  if (failed(lowered))
    return failure();
  rollbackValues(mark);
  mapResults(carriers, op.getResults(), loop.getResults().drop_front());
  return success();
}

/// Lowers `cflow.while` to a `cc.loop` whose condition region is the `before`
/// region and whose body is the `after` region.
LogicalResult GraphLowering::lowerWhile(qlx::cflow::WhileOp op) {
  if (op.getMaxIterationsAttr())
    return op.emitOpError(
        "bounds the loop by max_iterations, and Logical does not define what "
        "happens at the bound");
  Location loc = op.getLoc();
  SmallVector<Attribute> carriers = liveCarriers();
  SmallVector<Value> iterArgs = carriedValues(carriers, op.getInits());
  auto types = llvm::to_vector(ValueRange(iterArgs).getTypes());
  std::size_t mark = undoLog.size();
  Block &before = op.getBeforeRegion().front();
  Block &after = op.getAfterRegion().front();
  LogicalResult lowered = success();
  // Each region starts from the outer mappings: a Boolean read in the
  // condition does not dominate the body.
  auto loop = cudaq::cc::LoopOp::create(
      builder, loc, types, iterArgs, /*postCond=*/false,
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        rollbackValues(mark);
        enterRegion(block.getArguments(), carriers, before.getArguments());
        lowered = lowerOps(before);
        if (failed(lowered))
          return;
        auto condition =
            cast<qlx::cflow::WhileConditionOp>(before.getTerminator());
        cudaq::cc::ConditionOp::create(
            builder, loc, values.lookup(condition.getCondition()),
            carriedValues(carriers, condition.getForwarded()));
      },
      [&](OpBuilder &builder, Location loc, Region &region) {
        cudaq::cc::RegionBuilderGuard guard(builder, loc, region, types);
        Block &block = *builder.getBlock();
        if (failed(lowered))
          return;
        rollbackValues(mark);
        enterRegion(block.getArguments(), carriers, after.getArguments());
        lowered = lowerOps(after);
        if (failed(lowered))
          return;
        cudaq::cc::ContinueOp::create(
            builder, loc,
            carriedValues(carriers, after.getTerminator()->getOperands()));
      });
  if (failed(lowered))
    return failure();
  rollbackValues(mark);
  mapResults(carriers, op.getResults(), loop.getResults());
  return success();
}

/// Returns the `i1` that a record or Boolean holds. A record is discriminated
/// at its first read, and complemented for an inverted product. Later reads
/// that the first read dominates reuse that `i1`.
Value GraphLowering::toBoolean(Value physValue) {
  Value value = values.lookup(physValue);
  if (!isa<cudaq::cc::MeasureHandleType>(value.getType()))
    return value;
  Location loc = physValue.getLoc();
  Value bit = cudaq::quake::DiscriminateOp::create(builder, loc,
                                                   builder.getI1Type(), value);
  if (invertedHandles.contains(value))
    bit = arith::XOrIOp::create(
        builder, loc, bit, arith::ConstantIntOp::create(builder, loc, 1, 1));
  mapValue(physValue, bit);
  return bit;
}

/// `phys.condition` is true when its source bit, from a record or an `i1`,
/// equals `expected`.
LogicalResult GraphLowering::lowerCondition(qlx::phys::ConditionOp op) {
  Value bit = toBoolean(op.getSource());
  if (!op.getExpected()) {
    Value one = arith::ConstantIntOp::create(builder, op.getLoc(), 1, 1);
    bit = arith::XOrIOp::create(builder, op.getLoc(), bit, one);
  }
  mapValue(op.getResult(), bit);
  return success();
}

/// `phys.all_false` is true when no event is true: the negated OR of its
/// events.
LogicalResult GraphLowering::lowerAllFalse(qlx::phys::AllFalseOp op) {
  Location loc = op.getLoc();
  Value any = arith::ConstantIntOp::create(builder, loc, 0, 1);
  for (Value event : op.getEvents())
    any = arith::OrIOp::create(builder, loc, any, values.lookup(event));
  Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
  mapValue(op.getResult(), arith::XOrIOp::create(builder, loc, any, one));
  return success();
}

LogicalResult GraphLowering::lowerReturn(qlx::phys::ReturnOp op) {
  Location loc = op.getLoc();
  SmallVector<Value> bits;
  for (Value operand : op.getOperands())
    bits.push_back(toBoolean(operand));
  if (bits.size() <= 1) {
    func::ReturnOp::create(builder, loc, bits);
    return success();
  }
  // Two or more Booleans return as one list. The list is copied to heap
  // storage that the caller owns, because the `cc.alloca` buffer does not
  // outlive the entry point.
  // Note: Keep the `!cc.array<i1 x N>` buffer and its constant length. Core
  // folds that length through to `ReturnToOutputLog`, which records the list
  // on QIR profiles. A buffer cast to `!cc.ptr<i1>` loses the length there,
  // and profile output then drops the list without a diagnostic.
  auto i1Type = builder.getI1Type();
  Value buffer = cudaq::cc::AllocaOp::create(
      builder, loc,
      cudaq::cc::ArrayType::get(builder.getContext(), i1Type, bits.size()));
  auto slotType = cudaq::cc::PointerType::get(i1Type);
  for (auto [i, bit] : llvm::enumerate(bits)) {
    Value slot = cudaq::cc::ComputePtrOp::create(
        builder, loc, slotType, buffer,
        ArrayRef<cudaq::cc::ComputePtrArg>{static_cast<std::int32_t>(i)});
    cudaq::cc::StoreOp::create(builder, loc, bit, slot);
  }
  Value length = arith::ConstantIntOp::create(builder, loc, bits.size(), 64);
  Value list = cudaq::cc::SequenceInitOp::create(
      builder, loc, cudaq::cc::SequenceType::get(i1Type), buffer, length);
  func::ReturnOp::create(
      builder, loc,
      cudaq::opt::marshal::copyDynamicValueToHeap(loc, builder, list));
  return success();
}

namespace {

class PhysToQuakePass : public qlx::impl::PhysToQuakeBase<PhysToQuakePass> {
public:
  using PhysToQuakeBase::PhysToQuakeBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    // Step 1: Find the graph that the `entry` option names.
    auto graph = dyn_cast_or_null<qlx::phys::GraphOp>(
        SymbolTable::lookupSymbolIn(module, entry));
    if (entry.empty() || !graph) {
      module.emitError() << "`phys-to-quake` needs the entry option to name a "
                            "phys.graph, but '"
                         << entry << "' names none";
      signalPassFailure();
      return;
    }
    // Step 2: Collect the P3 source ops before translating. Step 4 erases
    // them, so that the pass emits only CUDA-Q dialects and exactly one entry
    // point.
    SmallVector<Operation *> source;
    for (Operation &op : *module.getBody())
      source.push_back(&op);
    // Step 3: Translate the graph into a detached entry point.
    auto entryPoint = GraphLowering(module, graph).lower();
    if (failed(entryPoint)) {
      signalPassFailure();
      return;
    }
    // Step 4: Replace the P3 source with the entry point. Other graphs in the
    // module are erased untranslated, so they never cause a rejection.
    for (Operation *op : llvm::reverse(source))
      op->erase();
    module.push_back(*entryPoint);
    // Step 5: Load the intrinsic that copies a returned list to the heap.
    // Loading it after the erasure keeps a source symbol of the same name
    // from hiding it.
    if (llvm::any_of(entryPoint->getResultTypes(),
                     llvm::IsaPred<cudaq::cc::SequenceType>) &&
        failed(cudaq::IRBuilder::atBlockEnd(module.getBody())
                   .loadIntrinsic(module, "__nvqpp_vectorCopyCtor"))) {
      signalPassFailure();
      return;
    }
    // Step 6: Drop Logical's `qlx.*` module attributes. They describe the P3
    // source, which is no longer in the module.
    SmallVector<StringAttr> logicalAttrs;
    for (NamedAttribute attr : module->getDiscardableAttrs())
      if (attr.getName().getValue().starts_with("qlx."))
        logicalAttrs.push_back(attr.getName());
    for (StringAttr name : logicalAttrs)
      module->removeAttr(name);
    // Step 7: Keep core's quantum optimizations from rewriting the scheduled
    // gates.
    // Rationale: With `quake.noOptimization` on the module, the JIT and AOT
    // drivers build the QIR pipeline without value-semantics optimization, so
    // `quake-simplify` and `dqe` do not run (see
    // `createTargetCodegenPipeline`). Without it, core could cancel or merge
    // scheduled gates, erase the `quake.reset` that follows a
    // `quake.null_wire`, or remove a qubit that nothing acts on.
    module->setAttr(cudaq::runtime::disableQuantumOpts,
                    UnitAttr::get(module.getContext()));
  }
};

} // namespace
