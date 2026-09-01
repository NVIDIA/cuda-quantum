//===- QuakeToQLX.cpp - Import CUDA-Q Quake as QLX P0 --------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#include "qlx/Conversion/QuakeToQLXPasses.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"

#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <string>

namespace qlx {
#define GEN_PASS_DEF_CONVERTQUAKETOQLX
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"
} // namespace qlx

using namespace mlir;

namespace {

// Mutable per-region conversion state threaded through importOperation so that
// structured bodies (qlx.repeat / qlx.if) can be imported recursively.
struct ImportState {
  DenseMap<Value, Value> wireOwners;    // quake wire -> qlx logical owner
  DenseMap<Value, Value> classicalBits; // quake measure-handle/i1 -> qlx i1
  DenseSet<Value> measuredWires;        // post-measurement wires (no discard)
  int64_t nextAllocation = 0;
};

// Which op set a region accepts. Repeat bodies additionally permit closed
// iteration-local prepare/measure/discard lifecycles; adaptive branches remain
// gate-only so their ownership join stays explicit.
enum class RegionPolicy { Entry, RepeatBody, IfBranch };

// A cc.loop proven to be a normalized constant-trip counted loop.
struct NormalizedCountedLoop {
  int64_t tripCount;
  unsigned inductionIndex;
  SmallVector<unsigned> wireCarries; // indices into initialArgs that are wires
};

static std::optional<int64_t> constInt(Value v) {
  if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto ia = dyn_cast<IntegerAttr>(c.getValue()))
      return ia.getInt();
  return std::nullopt;
}

static std::optional<double> constFloat(Value v) {
  if (auto c = v.getDefiningOp<mlir::arith::ConstantOp>())
    if (auto fa = dyn_cast<FloatAttr>(c.getValue()))
      return fa.getValue().convertToDouble();
  return std::nullopt;
}

// Structurally prove the QLX-owned counted-loop contract without CUDA-Q's
// private loop analysis: pre-condition loop; no else/break; single induction
// carry init 0, either `cmpi ne %ind, %constTrip` or
// `cmpi slt %ind, %constTrip`, step `%ind + 1`; every other carry a
// !quake.wire forwarded unchanged through while/step; body does not touch the
// induction except as an inert carry of a nested counted loop. The slt form is
// accepted only with the induction on the left and a nonnegative literal bound,
// making its trip count identical to the ne form. Non-wire carries are checked
// structurally but omitted from qlx.repeat.
static FailureOr<NormalizedCountedLoop>
matchNormalizedCountedLoop(cudaq::cc::LoopOp loop) {
  auto reject = [&](StringRef why) -> LogicalResult {
    loop.emitOpError("unsupported loop: ") << why;
    return failure();
  };
  if (loop.getPostCondition())
    return reject("must be a pre-condition (while) loop");
  if (!loop.getElseRegion().empty())
    return reject("Python for-else is not supported");
  if (!loop.getWhileRegion().hasOneBlock() ||
      !loop.getBodyRegion().hasOneBlock() ||
      !loop.getStepRegion().hasOneBlock())
    return reject("while/do/step regions must each be a single block");

  Block &whileBlk = loop.getWhileRegion().front();
  Block &doBlk = loop.getBodyRegion().front();
  Block &stepBlk = loop.getStepRegion().front();
  unsigned n = loop.getInitialArgs().size();

  bool hasBreak = false;
  loop.getBodyRegion().walk([&](cudaq::cc::BreakOp) { hasBreak = true; });
  if (hasBreak)
    return reject("break is not supported");

  auto cond = dyn_cast<cudaq::cc::ConditionOp>(whileBlk.getTerminator());
  if (!cond)
    return reject("while region must end in cc.condition");
  auto cmp = cond.getCondition().getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp)
    return reject("condition must be an integer comparison");

  auto lhsArg = dyn_cast<BlockArgument>(cmp.getLhs());
  auto rhsArg = dyn_cast<BlockArgument>(cmp.getRhs());
  BlockArgument indArg;
  std::optional<int64_t> trip;
  if (cmp.getPredicate() == mlir::arith::CmpIPredicate::ne) {
    if (lhsArg && lhsArg.getOwner() == &whileBlk &&
        (trip = constInt(cmp.getRhs())))
      indArg = lhsArg;
    else if (rhsArg && rhsArg.getOwner() == &whileBlk &&
             (trip = constInt(cmp.getLhs())))
      indArg = rhsArg;
  } else if (cmp.getPredicate() == mlir::arith::CmpIPredicate::slt) {
    // Unlike ne, slt is not symmetric. Only accept `%induction < %bound`.
    if (lhsArg && lhsArg.getOwner() == &whileBlk &&
        (trip = constInt(cmp.getRhs())))
      indArg = lhsArg;
  } else {
    return reject("condition must use `arith.cmpi ne`, or `arith.cmpi slt` "
                  "with the induction on the left");
  }
  if (!indArg || !trip)
    return reject("condition must compare the induction directly to a constant "
                  "trip count (affine-index loops unsupported)");
  unsigned indIdx = indArg.getArgNumber();
  if (*trip < 0)
    return reject("trip count must be nonnegative");

  auto initInd = constInt(loop.getInitialArgs()[indIdx]);
  if (!initInd || *initInd != 0)
    return reject("induction must be initialized to 0");

  auto stepCont = dyn_cast<cudaq::cc::ContinueOp>(stepBlk.getTerminator());
  if (!stepCont || stepCont.getOperands().size() != n)
    return reject("step region must forward all carries via cc.continue");
  for (unsigned i = 0; i < n; ++i) {
    Value op = stepCont.getOperands()[i];
    if (i == indIdx) {
      auto add = op.getDefiningOp<mlir::arith::AddIOp>();
      auto a0 = add ? dyn_cast<BlockArgument>(add.getLhs()) : nullptr;
      auto one = add ? constInt(add.getRhs()) : std::nullopt;
      if (!a0 || a0.getOwner() != &stepBlk || a0.getArgNumber() != indIdx ||
          !one || *one != 1)
        return reject("step must be `induction + 1`");
    } else {
      auto a = dyn_cast<BlockArgument>(op);
      if (!a || a.getOwner() != &stepBlk || a.getArgNumber() != i)
        return reject("step must forward every non-induction carry unchanged");
    }
  }

  if (cond.getResults().size() != n)
    return reject("cc.condition must forward all carries");
  for (unsigned i = 0; i < n; ++i) {
    if (i == indIdx)
      continue;
    auto a = dyn_cast<BlockArgument>(cond.getResults()[i]);
    if (!a || a.getOwner() != &whileBlk || a.getArgNumber() != i)
      return reject("while region must forward every wire carry unchanged");
  }

  auto doCont = dyn_cast<cudaq::cc::ContinueOp>(doBlk.getTerminator());
  if (!doCont || doCont.getOperands().size() != n)
    return reject("do region must forward all carries via cc.continue");
  BlockArgument doInd = doBlk.getArgument(indIdx);
  auto contInd = dyn_cast<BlockArgument>(doCont.getOperands()[indIdx]);
  if (!contInd || contInd.getOwner() != &doBlk ||
      contInd.getArgNumber() != indIdx)
    return reject("loop body must not modify the induction variable");
  for (Operation *user : doInd.getUsers()) {
    if (user == doCont.getOperation())
      continue;
    auto nested = dyn_cast<cudaq::cc::LoopOp>(user);
    if (!nested)
      return reject("loop body must not use the induction variable except as "
                    "an inert nested-loop carry");
    auto nestedMatch = matchNormalizedCountedLoop(nested);
    if (failed(nestedMatch))
      return failure();
    auto nestedArgs = nested.getInitialArgs();
    auto found = llvm::find(nestedArgs, doInd);
    if (found == nestedArgs.end() ||
        static_cast<unsigned>(std::distance(nestedArgs.begin(), found)) ==
            nestedMatch->inductionIndex)
      return reject("outer induction may only be an inert non-induction carry "
                    "of a nested counted loop");
  }

  NormalizedCountedLoop m;
  m.tripCount = *trip;
  m.inductionIndex = indIdx;
  for (unsigned i = 0; i < n; ++i) {
    if (i == indIdx)
      continue;
    if (isa<cudaq::quake::WireType>(loop.getInitialArgs()[i].getType()))
      m.wireCarries.push_back(i);
  }
  return m;
}

template <typename GateOp>
static FailureOr<qlx::BuiltinAction> actionForGate(GateOp gate) {
  const unsigned controls = gate.getControls().size();
  const unsigned targets = gate.getTargets().size();
  const bool adjoint = gate.getIsAdj();

  if (controls == 0 && targets == 1) {
    if constexpr (std::is_same_v<GateOp, cudaq::quake::HOp>)
      return qlx::BuiltinAction::h;
    if constexpr (std::is_same_v<GateOp, cudaq::quake::XOp>)
      return qlx::BuiltinAction::x;
    if constexpr (std::is_same_v<GateOp, cudaq::quake::YOp>)
      return qlx::BuiltinAction::y;
    if constexpr (std::is_same_v<GateOp, cudaq::quake::ZOp>)
      return qlx::BuiltinAction::z;
    if constexpr (std::is_same_v<GateOp, cudaq::quake::SOp>)
      return adjoint ? qlx::BuiltinAction::sdg : qlx::BuiltinAction::s;
    if constexpr (std::is_same_v<GateOp, cudaq::quake::TOp>)
      return adjoint ? qlx::BuiltinAction::tdg : qlx::BuiltinAction::t;
  }
  if (controls == 1 && targets == 1) {
    if constexpr (std::is_same_v<GateOp, cudaq::quake::XOp>)
      return qlx::BuiltinAction::cx;
    if constexpr (std::is_same_v<GateOp, cudaq::quake::ZOp>)
      return qlx::BuiltinAction::cz;
  }
  return failure();
}

class ConvertQuakeToQLXPass
    : public qlx::impl::ConvertQuakeToQLXBase<ConvertQuakeToQLXPass> {
public:
  using ConvertQuakeToQLXBase::ConvertQuakeToQLXBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<cudaq::quake::QuakeDialect, qlx::QLXDialect,
                qlx::lvm::LVMDialect, arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> entries;
    std::size_t functionDefinitionCount = 0;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (!function.isDeclaration())
        ++functionDefinitionCount;
      if (function->hasAttr("cudaq-entrypoint"))
        entries.push_back(function);
    }

    if (entries.empty()) {
      module.emitError("convert-quake-to-qlx found no func.func with the "
                       "cudaq-entrypoint attribute");
      return signalPassFailure();
    }
    if (functionDefinitionCount != entries.size()) {
      module.emitError("convert-quake-to-qlx requires entry points to be fully "
                       "inlined; non-entry func.func definitions remain");
      return signalPassFailure();
    }

    selectedSourceEntry.clear();
    if (!entryPoint.empty()) {
      SmallVector<func::FuncOp> exactMatches;
      SmallVector<func::FuncOp> baseMatches;
      for (func::FuncOp function : entries) {
        StringRef raw = function.getSymName();
        StringRef prefixFree = raw;
        prefixFree.consume_front("__nvqpp__mlirgen__");
        if (raw == entryPoint || prefixFree == entryPoint) {
          exactMatches.push_back(function);
          continue;
        }
        StringRef base = prefixFree;
        if (std::size_t suffix = base.find(".."); suffix != StringRef::npos)
          base = base.take_front(suffix);
        if (base == entryPoint)
          baseMatches.push_back(function);
      }
      ArrayRef<func::FuncOp> matches =
          exactMatches.empty() ? ArrayRef<func::FuncOp>(baseMatches)
                               : ArrayRef<func::FuncOp>(exactMatches);
      if (matches.size() != 1) {
        module.emitError("convert-quake-to-qlx entry-point '")
            << entryPoint << "' identified " << matches.size()
            << " source entries; select one exact CUDA-Q symbol";
        return signalPassFailure();
      }
      func::FuncOp selected = matches.front();
      selectedSourceEntry = selected.getSymName().str();
    }

    // Reject top-level facts that this conversion neither preserves nor
    // deliberately removes. In particular, do not treat later cleanup as a
    // legalization mechanism for arbitrary input operations.
    for (Operation &op : module.getBody()->getOperations()) {
      if (isa<func::FuncOp>(op))
        continue;
      if (op.getName().getDialectNamespace() == "qlx" &&
          isa<SymbolOpInterface>(op))
        continue;
      // A combined command-line module may carry the selected P1 logical
      // machine beside Quake.  The importer does not interpret or mutate the
      // machine; it preserves the symbol so qlx-to-lvm can consume it in the
      // next pipeline step.
      if (isa<qlx::lvm::DomainOp>(op))
        continue;
      // CUDA-Q also emits declaration-only host ABI symbols next to kernels.
      // Avoid a hard LLVM dialect dependency while recognizing only that exact
      // declaration shape.
      if (op.getName().getStringRef() == "llvm.func" &&
          op.getNumRegions() == 1 && op.getRegion(0).empty())
        continue;
      op.emitError("unexpected top-level operation at the Quake-to-P0 "
                   "boundary; preserve or lower it explicitly");
      return signalPassFailure();
    }

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, qlx::QLXDialect,
                           qlx::lvm::LVMDialect>();
    target.addLegalOp<ModuleOp>();
    target.addDynamicallyLegalOp<func::FuncOp>(
        [](func::FuncOp function) { return function.isDeclaration(); });
    target.markUnknownOpDynamicallyLegal([](Operation *operation) {
      return operation->getParentOp() &&
             isa<ModuleOp>(operation->getParentOp()) &&
             operation->getName().getStringRef() == "llvm.func" &&
             operation->getNumRegions() == 1 && operation->getRegion(0).empty();
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<EntryPattern>(&getContext(), *this);
    if (failed(applyFullConversion(module, target, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // CUDA-Q places host-runtime declarations next to the specialized kernel.
    // They do not belong to the portable P0 module.
    SmallVector<Operation *> sourceDeclarations;
    for (Operation &op : module.getBody()->getOperations()) {
      if (auto function = dyn_cast<func::FuncOp>(op);
          function && function.isDeclaration()) {
        sourceDeclarations.push_back(&op);
        continue;
      }
      if (op.getName().getStringRef() == "llvm.func" &&
          op.getNumRegions() == 1 && op.getRegion(0).empty()) {
        sourceDeclarations.push_back(&op);
        continue;
      }
      // Symbol-bearing QLX definitions are explicitly legal and preserved.
      if (op.getName().getDialectNamespace() == "qlx" &&
          isa<SymbolOpInterface>(op))
        continue;
    }
    for (Operation *op : sourceDeclarations)
      op->erase();

    module->removeAttr("quake.mangled_name_map");
    module->removeAttr("cc.python_uniqued");
    module->removeAttr("quake.python_uniqued");
    module->removeAttr("llvm.data_layout");
    module->removeAttr("llvm.target_triple");
  }

private:
  class EntryPattern : public OpConversionPattern<func::FuncOp> {
  public:
    EntryPattern(MLIRContext *context, ConvertQuakeToQLXPass &owner)
        : OpConversionPattern(context), owner(owner) {}

    LogicalResult
    matchAndRewrite(func::FuncOp function, OpAdaptor,
                    ConversionPatternRewriter &rewriter) const override {
      if (function.isDeclaration() || !function->hasAttr("cudaq-entrypoint"))
        return failure();
      if (!owner.shouldConvertEntry(function)) {
        rewriter.eraseOp(function);
        return success();
      }
      if (failed(owner.convertEntry(function, rewriter)))
        return failure();
      rewriter.eraseOp(function);
      return success();
    }

  private:
    ConvertQuakeToQLXPass &owner;
  };

  bool shouldConvertEntry(func::FuncOp function) const {
    return selectedSourceEntry.empty() ||
           function.getSymName() == selectedSourceEntry;
  }

  std::string selectedSourceEntry;

  static LogicalResult requireWireSemantics(func::FuncOp function) {
    WalkResult result = function.walk([](Operation *op) {
      for (Type type :
           llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        if (cudaq::quake::isQuantumType(type) &&
            !isa<cudaq::quake::WireType>(type)) {
          op->emitOpError("Quake-to-P0 supports only value-semantics "
                          "!quake.wire quantum values");
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    return success(!result.wasInterrupted());
  }

  // Emit one built-in qlx.apply preserving N logical-qubit owners (N in / N
  // out) and return its results.
  static SmallVector<Value> emitApply(OpBuilder &builder, Location loc,
                                      qlx::BuiltinAction action,
                                      ValueRange inputs) {
    SmallVector<Type> outputTypes(
        inputs.size(), qlx::LogicalQubitType::get(builder.getContext()));
    auto applied = qlx::ApplyOp::create(
        builder, loc, outputTypes,
        qlx::BuiltinActionAttr::get(builder.getContext(), action), inputs,
        nullptr);
    SmallVector<Value> results;
    for (Value result : applied.getResults())
      results.push_back(result);
    return results;
  }

  template <typename GateOp>
  static LogicalResult convertGate(GateOp gate, OpBuilder &builder,
                                   DenseMap<Value, Value> &values) {
    if (!gate.getParameters().empty())
      return gate.emitOpError(
          "parameterized gates are outside the Quake-to-P0 contract");
    if (auto negated = gate.getNegatedQubitControls())
      for (bool value : *negated)
        if (value)
          return gate.emitOpError(
              "negated controls are outside the Quake-to-P0 contract");

    // Map every quantum operand (controls then targets) to its P0 owner.
    SmallVector<Value> inputs;
    auto appendMapped = [&](ValueRange wires) -> LogicalResult {
      for (Value wire : wires) {
        auto found = values.find(wire);
        if (found == values.end())
          return gate.emitOpError("uses a wire with no live P0 owner");
        inputs.push_back(found->second);
        values.erase(found);
      }
      return success();
    };
    if (failed(appendMapped(gate.getControls())) ||
        failed(appendMapped(gate.getTargets())))
      return failure();

    auto wires = gate.getWires();
    if (wires.size() != inputs.size())
      return gate.emitOpError("does not return one wire per quantum operand");

    // Two-control Z is the standard logical CCZ action. Whether a later QEC
    // lowering realizes it natively or with a CCZ-state stream is not P0
    // source-import semantics.
    if constexpr (std::is_same_v<GateOp, cudaq::quake::ZOp>) {
      if (gate.getControls().size() == 2 && gate.getTargets().size() == 1) {
        auto results =
            emitApply(builder, gate.getLoc(), qlx::BuiltinAction::ccz, inputs);
        for (auto [quakeResult, qlxResult] : llvm::zip(wires, results))
          values[quakeResult] = qlxResult;
        return success();
      }
    }

    // Two-control X is Toffoli. Decompose to the standard-action identity
    // H-CCZ-H on the target. Result order is controls-then-targets.
    if constexpr (std::is_same_v<GateOp, cudaq::quake::XOp>) {
      if (gate.getControls().size() == 2 && gate.getTargets().size() == 1) {
        Location loc = gate.getLoc();
        auto h1 = emitApply(builder, loc, qlx::BuiltinAction::h, {inputs[2]});
        auto ccz = emitApply(builder, loc, qlx::BuiltinAction::ccz,
                             {inputs[0], inputs[1], h1[0]});
        auto h2 = emitApply(builder, loc, qlx::BuiltinAction::h, {ccz[2]});
        values[wires[0]] = ccz[0]; // control 1 out
        values[wires[1]] = ccz[1]; // control 2 out
        values[wires[2]] = h2[0];  // target out
        return success();
      }
    }

    auto action = actionForGate(gate);
    if (failed(action))
      return gate.emitOpError(
          "unsupported gate shape (supported: 1q H/S/Sdg/T/Tdg/X/Y/Z, "
          "single-control X/Z, and two-control X/Z)");

    SmallVector<Type> outputTypes(
        wires.size(), qlx::LogicalQubitType::get(gate.getContext()));
    auto applied = qlx::ApplyOp::create(
        builder, gate.getLoc(), outputTypes,
        qlx::BuiltinActionAttr::get(gate.getContext(), *action), inputs,
        nullptr);
    for (auto [quakeResult, qlxResult] : llvm::zip(wires, applied.getResults()))
      values[quakeResult] = qlxResult;
    return success();
  }

  template <typename MeasureOp>
  static LogicalResult convertMeasurement(MeasureOp measurement,
                                          qlx::Pauli basis, OpBuilder &builder,
                                          DenseMap<Value, Value> &values,
                                          DenseMap<Value, Value> &measurements,
                                          DenseSet<Value> &measuredWires) {
    if (measurement.getTargets().size() != 1)
      return measurement.emitOpError(
          "Quake-to-P0 supports scalar wire measurements only");
    Value target = measurement.getTargets().front();
    auto input = values.find(target);
    if (input == values.end())
      return measurement.emitOpError("uses a wire with no live P0 owner");

    auto measured =
        qlx::MeasureOp::create(builder, measurement.getLoc(),
                               builder.getI1Type(), basis, input->second);
    measurements[measurement.getMeasOut()] = measured.getResult();
    values.erase(input);
    for (Value wire : measurement.getWires())
      measuredWires.insert(wire);
    return success();
  }

  // swap has no built-in action; decompose to three alternating CX.
  static LogicalResult convertSwap(cudaq::quake::SwapOp swap,
                                   OpBuilder &builder,
                                   DenseMap<Value, Value> &values) {
    if (!swap.getControls().empty())
      return swap.emitOpError(
          "controlled swap (Fredkin) is outside the Quake-to-P0 contract");
    if (!swap.getParameters().empty())
      return swap.emitOpError(
          "parameterized swap is outside the Quake-to-P0 contract");
    if (swap.getTargets().size() != 2)
      return swap.emitOpError("swap must have exactly two targets");

    SmallVector<Value> inputs;
    for (Value wire : swap.getTargets()) {
      auto found = values.find(wire);
      if (found == values.end())
        return swap.emitOpError("uses a wire with no live P0 owner");
      inputs.push_back(found->second);
      values.erase(found);
    }

    Location loc = swap.getLoc();
    auto cx1 =
        emitApply(builder, loc, qlx::BuiltinAction::cx, {inputs[0], inputs[1]});
    auto cx2 =
        emitApply(builder, loc, qlx::BuiltinAction::cx, {cx1[1], cx1[0]});
    auto cx3 =
        emitApply(builder, loc, qlx::BuiltinAction::cx, {cx2[1], cx2[0]});
    auto wires = swap.getWires();
    values[wires[0]] = cx3[0];
    values[wires[1]] = cx3[1];
    return success();
  }

  // Emit a `qlx.apply #qlx.action<pauli_rotation>` preserving the quantum
  // owners, with the axis in a {x_mask, z_mask, sign} parameters dict and the
  // angle as a trailing f64 operand. Returns the (quantum) result owners.
  static SmallVector<Value> emitPauliRotation(OpBuilder &builder, Location loc,
                                              ArrayRef<Value> quantumOwners,
                                              Value angle,
                                              DictionaryAttr params) {
    SmallVector<Value> inputs(quantumOwners.begin(), quantumOwners.end());
    inputs.push_back(angle);
    SmallVector<Type> outputTypes(
        quantumOwners.size(), qlx::LogicalQubitType::get(builder.getContext()));
    auto applied = qlx::ApplyOp::create(
        builder, loc, outputTypes,
        qlx::BuiltinActionAttr::get(builder.getContext(),
                                    qlx::BuiltinAction::pauli_rotation),
        inputs, params);
    SmallVector<Value> results;
    for (Value r : applied.getResults())
      results.push_back(r);
    return results;
  }

  // Lower a single-qubit rotation rx/ry/rz to pauli_rotation. `xMask`/`zMask`
  // select the axis (rx=1,0; ry=1,1; rz=0,1). Constant angle only; controlled
  // rotations are rejected (they are not a plain pauli_rotation).
  template <typename GateOp>
  static LogicalResult convertRotation(GateOp gate, OpBuilder &builder,
                                       DenseMap<Value, Value> &values,
                                       int64_t xMask, int64_t zMask) {
    if (!gate.getControls().empty())
      return gate.emitOpError(
          "controlled rotations are outside the Quake-to-P0 "
          "contract; decompose (e.g. crz = rz/cx) upstream");
    if (auto negated = gate.getNegatedQubitControls())
      for (bool value : *negated)
        if (value)
          return gate.emitOpError(
              "negated controls are outside the Quake-to-P0 contract");
    if (gate.getParameters().size() != 1 || gate.getTargets().size() != 1)
      return gate.emitOpError("rotation must have one angle and one target");

    auto angle = constFloat(gate.getParameter(0));
    if (!angle)
      return gate.emitOpError("dynamic/non-constant rotation angle is "
                              "unsupported; specialize or const-fold upstream");
    // QLX keeps a rotation's angle as a nonnegative magnitude and carries the
    // full operator sign in parameters.sign. Apply the Quake adjoint first,
    // then normalize both adjoint and negative-source angles into that single
    // canonical representation.
    double signedAngle = gate.getIsAdj() ? -*angle : *angle;
    int64_t sign = signedAngle < 0.0 ? -1 : 1;
    double magnitude = std::abs(signedAngle);

    auto found = values.find(gate.getTargets()[0]);
    if (found == values.end())
      return gate.emitOpError("uses a wire with no live P0 owner");

    Value input = found->second;
    values.erase(found);
    Location loc = gate.getLoc();
    Value angleConst = mlir::arith::ConstantOp::create(
        builder, loc, builder.getF64FloatAttr(magnitude));
    auto i64 = builder.getI64Type();
    auto params = builder.getDictionaryAttr(
        {builder.getNamedAttr("x_mask", builder.getIntegerAttr(i64, xMask)),
         builder.getNamedAttr("z_mask", builder.getIntegerAttr(i64, zMask)),
         builder.getNamedAttr("sign", builder.getIntegerAttr(i64, sign))});
    auto results = emitPauliRotation(builder, loc, {input}, angleConst, params);
    values[gate.getWires()[0]] = results[0];
    return success();
  }

  // Discard live (unmeasured) wires' P0 owners at one ownership boundary.
  // CUDA-Q scalarizes a qvector deallocation into adjacent quake.sink ops;
  // retaining that adjacency as one variadic qlx.discard preserves the source
  // allocation's simultaneous end of lifetime for packed logical blocks.
  static LogicalResult discardWires(Operation *owner, ValueRange wires,
                                    OpBuilder &builder, ImportState &state) {
    SmallVector<Value> inputs;
    SmallVector<Value> liveWires;
    for (Value wire : wires) {
      if (state.measuredWires.contains(wire))
        continue;
      auto input = state.wireOwners.find(wire);
      if (input == state.wireOwners.end())
        return owner->emitOpError("uses a wire with no live P0 owner");
      inputs.push_back(input->second);
      liveWires.push_back(wire);
    }
    if (!inputs.empty())
      qlx::DiscardOp::create(builder, owner->getLoc(), inputs, nullptr);
    for (Value wire : liveWires)
      state.wireOwners.erase(wire);
    return success();
  }

  static LogicalResult discardWire(Operation *owner, Value wire,
                                   OpBuilder &builder, ImportState &state) {
    return discardWires(owner, ValueRange{wire}, builder, state);
  }

  // Convert one source operation into the current qlx region. Repeat bodies
  // may own iteration-local preparation and destructive measurement sites;
  // adaptive branches remain gate-only so their ownership join is explicit.
  LogicalResult importOperation(Operation &source, OpBuilder &builder,
                                ImportState &state, RegionPolicy policy) {
    const bool structured = policy != RegionPolicy::Entry;
    const bool repeatBody = policy == RegionPolicy::RepeatBody;
    auto rejectStructured = [&](Operation *op,
                                StringRef what) -> LogicalResult {
      return op->emitOpError(what) << " is not allowed inside a folded loop or "
                                      "conditional body";
    };
    return llvm::TypeSwitch<Operation *, LogicalResult>(&source)
        .Case<cudaq::quake::NullWireOp>([&](auto nullWire) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(nullWire, "allocation");
          auto i64 = builder.getI64Type();
          auto prepared = qlx::PrepareOp::create(
              builder, nullWire.getLoc(),
              qlx::LogicalQubitType::get(builder.getContext()), "zero",
              IntegerAttr::get(i64, state.nextAllocation),
              IntegerAttr::get(i64, 0));
          state.wireOwners[nullWire.getResult()] = prepared.getResult();
          ++state.nextAllocation;
          return success();
        })
        .Case<cudaq::quake::HOp, cudaq::quake::SOp, cudaq::quake::TOp,
              cudaq::quake::XOp, cudaq::quake::YOp, cudaq::quake::ZOp>(
            [&](auto gate) -> LogicalResult {
              return convertGate(gate, builder, state.wireOwners);
            })
        .Case<cudaq::quake::SwapOp>([&](auto swap) -> LogicalResult {
          return convertSwap(swap, builder, state.wireOwners);
        })
        .Case<cudaq::quake::R1Op>([&](auto g) -> LogicalResult {
          // On one uncontrolled target, R1(theta) and Rz(theta) differ only
          // by an unobservable global phase. Controlled R1 is rejected by
          // convertRotation because that phase would become relative.
          return convertRotation(g, builder, state.wireOwners, 0, 1);
        })
        .Case<cudaq::quake::RxOp>([&](auto g) -> LogicalResult {
          return convertRotation(g, builder, state.wireOwners, 1, 0);
        })
        .Case<cudaq::quake::RyOp>([&](auto g) -> LogicalResult {
          return convertRotation(g, builder, state.wireOwners, 1, 1);
        })
        .Case<cudaq::quake::RzOp>([&](auto g) -> LogicalResult {
          return convertRotation(g, builder, state.wireOwners, 0, 1);
        })
        .Case<cudaq::cc::ScopeOp>([&](auto scope) -> LogicalResult {
          return importScope(scope, builder, state, policy);
        })
        .Case<cudaq::cc::LoopOp>([&](auto loop) -> LogicalResult {
          return importLoop(loop, builder, state);
        })
        .Case<cudaq::cc::IfOp>([&](auto ifop) -> LogicalResult {
          if (structured && !constInt(ifop.getCondition()))
            return ifop.emitOpError(
                "nested adaptive control is outside the Quake-to-P0 "
                "contract; flatten or outline the nested region");
          return importIf(ifop, builder, state, policy);
        })
        .Case<mlir::arith::ConstantOp>([&](auto) -> LogicalResult {
          // Loop/conditional control constants are read structurally by the
          // matchers, never emitted into P0.
          return success();
        })
        .Case<cudaq::cc::UndefOp>([&](auto undef) -> LogicalResult {
          if (isa<cudaq::quake::WireType, cudaq::quake::RefType,
                  cudaq::quake::VeqType>(undef.getInputType().getType()))
            return undef.emitOpError(
                "quantum cc.undef is outside the Quake-to-P0 contract");
          return success();
        })
        .Case<cudaq::quake::MzOp>([&](auto m) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(m, "measurement");
          return convertMeasurement(m, qlx::Pauli::Z, builder, state.wireOwners,
                                    state.classicalBits, state.measuredWires);
        })
        .Case<cudaq::quake::MxOp>([&](auto m) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(m, "measurement");
          return convertMeasurement(m, qlx::Pauli::X, builder, state.wireOwners,
                                    state.classicalBits, state.measuredWires);
        })
        .Case<cudaq::quake::MyOp>([&](auto m) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(m, "measurement");
          return convertMeasurement(m, qlx::Pauli::Y, builder, state.wireOwners,
                                    state.classicalBits, state.measuredWires);
        })
        .Case<cudaq::quake::DiscriminateOp>(
            [&](auto discriminate) -> LogicalResult {
              if (structured && !repeatBody)
                return rejectStructured(discriminate, "discrimination");
              auto found =
                  state.classicalBits.find(discriminate.getMeasurement());
              if (found == state.classicalBits.end())
                return discriminate.emitOpError(
                    "uses a measurement not imported by this entry point");
              state.classicalBits[discriminate.getResult()] = found->second;
              return success();
            })
        .Case<cudaq::quake::SinkOp>([&](auto sink) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(sink, "sink");
          return discardWire(sink, sink.getTarget(), builder, state);
        })
        .Case<cudaq::quake::ReturnWireOp>(
            [&](auto returnWire) -> LogicalResult {
              if (structured && !repeatBody)
                return rejectStructured(returnWire, "wire return");
              return discardWire(returnWire, returnWire.getTarget(), builder,
                                 state);
            })
        .Case<func::ReturnOp>([&](auto sourceReturn) -> LogicalResult {
          if (structured)
            return rejectStructured(sourceReturn, "function return");
          SmallVector<Value> results;
          for (Value value : sourceReturn.getOperands()) {
            auto found = state.classicalBits.find(value);
            if (found == state.classicalBits.end())
              return sourceReturn.emitOpError(
                  "returns a value that is not a discriminated measurement");
            results.push_back(found->second);
          }
          qlx::ReturnOp::create(builder, sourceReturn.getLoc(), results);
          return success();
        })
        .Default([&](Operation *unsupported) -> LogicalResult {
          return unsupported->emitOpError(
              "is outside the typed Quake-to-P0 conversion contract");
        });
  }

  // cc.scope is transparent: import its body into the current region and map
  // scope results to the owners its cc.continue forwards.
  LogicalResult importScope(cudaq::cc::ScopeOp scope, OpBuilder &builder,
                            ImportState &state, RegionPolicy policy) {
    if (!scope->getRegion(0).hasOneBlock())
      return scope.emitOpError("only single-block cc.scope is supported");
    Block &blk = scope->getRegion(0).front();
    if (blk.getNumArguments() != 0)
      return scope.emitOpError("cc.scope with block arguments is unsupported");
    for (Operation &inner : blk.without_terminator())
      if (failed(importOperation(inner, builder, state, policy)))
        return failure();
    auto cont = dyn_cast<cudaq::cc::ContinueOp>(blk.getTerminator());
    if (!cont || cont.getOperands().size() != scope.getNumResults())
      return scope.emitOpError("cc.scope must end in a matching cc.continue");
    for (auto [res, val] : llvm::zip(scope.getResults(), cont.getOperands())) {
      if (!isa<cudaq::quake::WireType>(val.getType()))
        continue; // classical scope results are not tracked as owners
      auto found = state.wireOwners.find(val);
      if (found == state.wireOwners.end())
        return scope.emitOpError(
            "cc.scope yields a wire with no live P0 owner");
      Value owner = found->second;
      state.wireOwners.erase(found);
      state.wireOwners[res] = owner;
    }
    return success();
  }

  // Fold a normalized constant-trip cc.loop into one qlx.repeat (never unroll).
  LogicalResult importLoop(cudaq::cc::LoopOp loop, OpBuilder &builder,
                           ImportState &state) {
    auto matched = matchNormalizedCountedLoop(loop);
    if (failed(matched))
      return failure();
    NormalizedCountedLoop m = *matched;

    SmallVector<Value> inits;
    for (unsigned i : m.wireCarries) {
      auto found = state.wireOwners.find(loop.getInitialArgs()[i]);
      if (found == state.wireOwners.end())
        return loop.emitOpError("loop carry has no live P0 owner");
      inits.push_back(found->second);
      state.wireOwners.erase(found);
    }

    Location loc = loop.getLoc();
    auto lqty = qlx::LogicalQubitType::get(builder.getContext());
    SmallVector<Type> iterTypes(m.wireCarries.size(), lqty);
    auto repeat = qlx::RepeatOp::create(
        builder, loc, iterTypes, builder.getI64IntegerAttr(m.tripCount), inits);

    Block &doBlk = loop.getBodyRegion().front();
    {
      OpBuilder::InsertionGuard guard(builder);
      SmallVector<Location> locs(m.wireCarries.size(), loc);
      Block *rb = builder.createBlock(&repeat.getBody(), repeat.getBody().end(),
                                      iterTypes, locs);
      builder.setInsertionPointToEnd(rb);

      ImportState bodyState;
      bodyState.nextAllocation = state.nextAllocation;
      for (unsigned j = 0; j < m.wireCarries.size(); ++j)
        bodyState.wireOwners[doBlk.getArgument(m.wireCarries[j])] =
            rb->getArgument(j);

      for (Operation &inner : doBlk.without_terminator())
        if (failed(importOperation(inner, builder, bodyState,
                                   RegionPolicy::RepeatBody)))
          return failure();

      auto doCont = cast<cudaq::cc::ContinueOp>(doBlk.getTerminator());
      SmallVector<Value> yields;
      for (unsigned i : m.wireCarries) {
        auto found = bodyState.wireOwners.find(doCont.getOperands()[i]);
        if (found == bodyState.wireOwners.end())
          return doCont.emitOpError("yields a wire with no live P0 owner");
        yields.push_back(found->second);
      }
      if (bodyState.wireOwners.size() != yields.size())
        return doCont.emitOpError(
            "folded loop leaks an iteration-local quantum owner; measure or "
            "discard every body-local allocation before yielding");
      state.nextAllocation = bodyState.nextAllocation;
      qlx::YieldOp::create(builder, loc, yields);
    }

    for (unsigned j = 0; j < m.wireCarries.size(); ++j)
      state.wireOwners[loop.getResult(m.wireCarries[j])] = repeat.getResult(j);
    return success();
  }

  // Constant-condition cc.if: inline the selected branch into the current
  // region (feedback-free), mapping its wire block-args to the incoming owners.
  LogicalResult inlineIfBranch(cudaq::cc::IfOp ifop, Region &sel,
                               OpBuilder &builder, ImportState &state,
                               RegionPolicy policy) {
    auto linearArgs = ifop.getLinearArgs();
    if (sel.empty()) {
      if (ifop.getNumResults() != 0)
        return ifop.emitOpError(
            "constant-false cc.if with results requires an else branch");
      return success();
    }
    Block &sb = sel.front();
    if (sb.getNumArguments() != linearArgs.size())
      return ifop.emitOpError("cc.if branch arity does not match its carries");
    for (unsigned i = 0; i < linearArgs.size(); ++i) {
      auto found = state.wireOwners.find(linearArgs[i]);
      if (found == state.wireOwners.end())
        return ifop.emitOpError("conditional carry has no live P0 owner");
      state.wireOwners[sb.getArgument(i)] = found->second;
      state.wireOwners.erase(found);
    }
    for (Operation &inner : sb.without_terminator())
      if (failed(importOperation(inner, builder, state, policy)))
        return failure();
    auto cont = cast<cudaq::cc::ContinueOp>(sb.getTerminator());
    for (auto [res, val] : llvm::zip(ifop.getResults(), cont.getOperands())) {
      auto found = state.wireOwners.find(val);
      if (found == state.wireOwners.end())
        return cont.emitOpError("yields a wire with no live P0 owner");
      Value owner = found->second;
      state.wireOwners.erase(found);
      state.wireOwners[res] = owner;
    }
    return success();
  }

  // cc.if lowering: constant condition folds (B2a); a discriminated
  // (measurement) condition becomes an adaptive qlx.if (B2b).
  LogicalResult importIf(cudaq::cc::IfOp ifop, OpBuilder &builder,
                         ImportState &state, RegionPolicy policy) {
    Value cond = ifop.getCondition();
    if (auto c = constInt(cond)) {
      Region &sel = (*c != 0) ? ifop.getThenRegion() : ifop.getElseRegion();
      return inlineIfBranch(ifop, sel, builder, state, policy);
    }

    auto condOwner = state.classicalBits.find(cond);
    if (condOwner == state.classicalBits.end())
      return ifop.emitOpError("conditional requires a constant or a "
                              "measurement-derived (discriminated) condition");

    auto linearArgs = ifop.getLinearArgs();
    unsigned k = ifop.getNumResults();
    if (linearArgs.size() != k)
      return ifop.emitOpError("conditional must thread one wire carry per "
                              "result (gate-only branches)");
    for (Type t : ifop.getResultTypes())
      if (!isa<cudaq::quake::WireType>(t))
        return ifop.emitOpError("conditional results must be !quake.wire");

    SmallVector<Value> incoming;
    for (Value w : linearArgs) {
      auto found = state.wireOwners.find(w);
      if (found == state.wireOwners.end())
        return ifop.emitOpError("conditional carry has no live P0 owner");
      incoming.push_back(found->second);
      state.wireOwners.erase(found);
    }

    Location loc = ifop.getLoc();
    auto lqty = qlx::LogicalQubitType::get(builder.getContext());
    SmallVector<Type> resTypes(k, lqty);
    auto qif = qlx::IfOp::create(builder, loc, resTypes, condOwner->second);

    // Lower one cc.if branch into a qlx.if branch block (carries captured, not
    // block-args). An absent else is a synthesized identity.
    auto lowerBranch = [&](Region &src, Region &dst) -> LogicalResult {
      OpBuilder::InsertionGuard guard(builder);
      Block *bb = builder.createBlock(&dst);
      builder.setInsertionPointToEnd(bb);
      SmallVector<Value> yields;
      if (src.empty()) {
        yields.assign(incoming.begin(), incoming.end());
      } else {
        Block &sb = src.front();
        if (sb.getNumArguments() != linearArgs.size())
          return ifop.emitOpError("cc.if branch arity mismatch");
        ImportState bs;
        for (unsigned i = 0; i < linearArgs.size(); ++i)
          bs.wireOwners[sb.getArgument(i)] = incoming[i];
        for (Operation &inner : sb.without_terminator())
          if (failed(
                  importOperation(inner, builder, bs, RegionPolicy::IfBranch)))
            return failure();
        auto cont = cast<cudaq::cc::ContinueOp>(sb.getTerminator());
        for (Value v : cont.getOperands()) {
          auto found = bs.wireOwners.find(v);
          if (found == bs.wireOwners.end())
            return cont.emitOpError("yields a wire with no live P0 owner");
          yields.push_back(found->second);
        }
      }
      if (yields.size() != k)
        return ifop.emitOpError("conditional branch yields the wrong arity");
      qlx::YieldOp::create(builder, loc, yields);
      return success();
    };

    if (failed(lowerBranch(ifop.getThenRegion(), qif.getThenRegion())) ||
        failed(lowerBranch(ifop.getElseRegion(), qif.getElseRegion())))
      return failure();

    for (unsigned i = 0; i < k; ++i)
      state.wireOwners[ifop.getResult(i)] = qif.getResult(i);
    return success();
  }

  LogicalResult convertEntry(func::FuncOp function,
                             ConversionPatternRewriter &builder) {
    bool entryBlock = true;
    for (Block &block : function.getBody()) {
      if (!entryBlock)
        for (BlockArgument argument : block.getArguments())
          if (isa<cudaq::quake::WireType>(argument.getType()))
            return function.emitOpError(
                "quantum CFG block arguments are unsupported; normalize "
                "control flow to cc.loop/cc.if with explicit wire carries");
      entryBlock = false;
    }
    if (!function.getBody().hasOneBlock())
      return function.emitOpError(
          "Quake-to-P0 currently requires a single-block entry point");
    for (BlockArgument argument : function.getArguments()) {
      if (isa<cudaq::quake::WireType, cudaq::quake::RefType,
              cudaq::quake::VeqType>(argument.getType()))
        return function.emitOpError(
            "Quake-to-P0 requires a specialized entry point with no quantum "
            "arguments");
      if (!argument.use_empty())
        return function.emitOpError(
            "Quake-to-P0 requires every specialized classical argument to be "
            "constant-folded or unused");
    }
    for (Type type : function.getResultTypes())
      if (!type.isInteger(1))
        return function.emitOpError(
            "Quake-to-P0 currently supports only i1 entry-point results");
    if (failed(requireWireSemantics(function)))
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(function);
    StringRef sourceName = function.getSymName();
    sourceName.consume_front("__nvqpp__mlirgen__");
    if (std::size_t uniqueSuffix = sourceName.find("..");
        uniqueSuffix != StringRef::npos)
      sourceName = sourceName.take_front(uniqueSuffix);
    auto functionType =
        FunctionType::get(function.getContext(), {}, function.getResultTypes());
    auto program = qlx::ProgramOp::create(builder, function.getLoc(),
                                          sourceName, functionType,
                                          /*estimateOnly=*/nullptr,
                                          /*specialization=*/nullptr);
    program->setAttr("qlx.stage", builder.getStringAttr("p0"));
    program->setAttr("qlx.profile", builder.getStringAttr("p0"));
    Block *body = &program.getBody().emplaceBlock();
    builder.setInsertionPointToStart(body);

    ImportState state;
    Block &sourceBody = function.getBody().front();
    for (auto current = sourceBody.begin(), end = sourceBody.end();
         current != end;) {
      Operation &source = *current;
      if (auto sink = dyn_cast<cudaq::quake::SinkOp>(source)) {
        SmallVector<Value> wires{sink.getTarget()};
        Operation *owner = &source;
        ++current;
        while (current != end) {
          auto adjacent = dyn_cast<cudaq::quake::SinkOp>(*current);
          if (!adjacent)
            break;
          wires.push_back(adjacent.getTarget());
          ++current;
        }
        if (failed(discardWires(owner, wires, builder, state)))
          return failure();
        continue;
      }
      ++current;
      if (failed(importOperation(source, builder, state, RegionPolicy::Entry)))
        return failure();
    }

    // Close the source ownership transaction before dialect conversion commits.
    // This is the native boundary used by qlx-opt as well as Python, so a wire
    // leak must not be accepted merely because a higher-level wrapper happens
    // to run an additional verifier.
    if (!state.wireOwners.empty())
      return function.emitOpError()
             << "leaves " << state.wireOwners.size()
             << " live quantum owner(s); measure, sink, or return every wire";

    if (body->empty() || !isa<qlx::ReturnOp>(body->back()))
      qlx::ReturnOp::create(builder, function.getLoc());
    return success();
  }
};

} // namespace
