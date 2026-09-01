//===- QLXEstimateLogical.cpp - Native Tier-0 estimator --------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/QLX/Transforms/Passes.h"

#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"

#include <limits>
#include <map>
#include <string>

using namespace mlir;
using namespace qlx;

namespace qlx {
#define GEN_PASS_DEF_QLXESTIMATELOGICAL
#include "qlx/Dialect/QLX/Transforms/Passes.h.inc"
} // namespace qlx

namespace {

static bool isLogicalQubit(Type type) { return isa<LogicalQubitType>(type); }

static std::string printAttribute(Attribute value) {
  std::string text;
  llvm::raw_string_ostream stream(text);
  value.print(stream);
  return stream.str();
}

static std::string normalizeInlineName(Attribute attribute, StringRef prefix,
                                       StringRef fallbackPrefix) {
  std::string text = printAttribute(attribute);
  if (StringRef(text).starts_with(prefix) && StringRef(text).ends_with(">")) {
    StringRef value(text);
    value = value.drop_front(prefix.size()).drop_back().trim('"');
    return (fallbackPrefix + value).str();
  }
  if (auto symbol = dyn_cast<FlatSymbolRefAttr>(attribute))
    return symbol.getValue().str();
  return text;
}

static DictionaryAttr
countDictionary(MLIRContext *context,
                const std::map<std::string, int64_t> &counts) {
  SmallVector<NamedAttribute> fields;
  fields.reserve(counts.size());
  for (const auto &[name, count] : counts)
    fields.emplace_back(StringAttr::get(context, name),
                        IntegerAttr::get(IntegerType::get(context, 64), count));
  return DictionaryAttr::get(context, fields);
}

class LogicalWalker {
public:
  LogicalWalker(ModuleOp module, SymbolTable &symbols)
      : module(module), symbols(symbols) {}

  LogicalResult run(ProgramOp root) {
    int64_t live = 0;
    if (!root.getBody().empty())
      for (BlockArgument argument : root.getBody().front().getArguments())
        live += isLogicalQubit(argument.getType());
    peak = live;
    callStack.insert(root.getOperation());
    if (failed(walkBlock(root.getBody().front(), /*multiplier=*/1, live)))
      return failure();
    callStack.erase(root.getOperation());
    return success();
  }

  std::map<std::string, int64_t> actions;
  std::map<std::string, int64_t> instruments;
  std::map<std::string, int64_t> synthesis;
  int64_t idleSites = 0;
  int64_t discards = 0;
  int64_t depth = 0;
  int64_t peak = 0;

private:
  ModuleOp module;
  SymbolTable &symbols;
  llvm::SmallPtrSet<Operation *, 8> callStack;

  LogicalResult add(int64_t &target, int64_t value, Operation *source,
                    StringRef what) {
    int64_t result;
    if (llvm::AddOverflow(target, value, result))
      return source->emitOpError() << what << " count overflows signed i64";
    target = result;
    return success();
  }

  LogicalResult bump(std::map<std::string, int64_t> &target, std::string name,
                     int64_t value, Operation *source, StringRef what) {
    auto iterator = target.find(name);
    if (iterator == target.end()) {
      target.emplace(std::move(name), value);
      return success();
    }
    return add(iterator->second, value, source, what);
  }

  FailureOr<int64_t> multiply(int64_t left, int64_t right, Operation *source) {
    int64_t result;
    if (llvm::MulOverflow(left, right, result)) {
      source->emitOpError("folded multiplicity overflows signed i64");
      return failure();
    }
    return result;
  }

  LogicalResult walkBlock(Block &block, int64_t multiplier, int64_t &live) {
    for (Operation &operation : block)
      if (failed(walkOperation(&operation, multiplier, live)))
        return failure();
    return success();
  }

  LogicalResult walkOperation(Operation *operation, int64_t multiplier,
                              int64_t &live) {
    if (auto repeat = dyn_cast<RepeatOp>(operation)) {
      auto nested = multiply(multiplier, repeat.getCount(), operation);
      if (failed(nested))
        return failure();
      if (repeat.getCount() == 0)
        return success();
      int64_t before = live;
      if (failed(walkBlock(repeat.getBody().front(), *nested, live)))
        return failure();
      if (live != before)
        return repeat.emitOpError("logical estimation requires each folded "
                                  "iteration to preserve live ownership");
      return success();
    }

    if (auto call = dyn_cast<CallOp>(operation)) {
      Operation *callee = symbols.lookup(call.getCalleeAttr().getValue());
      auto program = dyn_cast_or_null<ProgramOp>(callee);
      if (!program)
        return call.emitOpError("callee must resolve to qlx.program");
      if (!callStack.insert(program.getOperation()).second)
        return call.emitOpError(
            "recursive logical call graph is not estimable");
      LogicalResult result =
          walkBlock(program.getBody().front(), multiplier, live);
      callStack.erase(program.getOperation());
      return result;
    }

    if (auto conditional = dyn_cast<IfOp>(operation)) {
      int64_t thenLive = live;
      int64_t elseLive = live;
      if (failed(walkBlock(conditional.getThenRegion().front(), multiplier,
                           thenLive)) ||
          failed(walkBlock(conditional.getElseRegion().front(), multiplier,
                           elseLive)))
        return failure();
      if (thenLive != elseLive)
        return conditional.emitOpError(
            "logical estimation requires branches to agree on live ownership");
      live = thenLive;
      peak = std::max({peak, thenLive, elseLive});
      return success();
    }

    // Dynamic iteration has no exact folded multiplicity in P0.  Treating a
    // while region as a leaf would silently omit its body from every count.
    // Research policies may first specialize/normalize it to qlx.repeat.
    if (isa<WhileOp>(operation))
      return operation->emitOpError(
          "qlx-estimate-logical requires dynamic while control to be "
          "specialized to an exact folded form");
    if (operation->getNumRegions() != 0)
      return operation->emitOpError(
          "qlx-estimate-logical does not support this region-bearing "
          "operation");

    // Region/function terminators forward ownership across their enclosing
    // boundary; they do not consume logical owners for liveness purposes.
    if (isa<YieldOp, ReturnOp>(operation))
      return success();

    bool supported = true;
    if (auto apply = dyn_cast<ApplyOp>(operation)) {
      std::string action = normalizeInlineName(apply.getActionAttr(),
                                               "#qlx.action<", "qlx_standard_");
      if (failed(bump(actions, action, multiplier, operation, "action")) ||
          failed(add(depth, multiplier, operation, "depth")))
        return failure();
      auto builtin = dyn_cast<BuiltinActionAttr>(apply.getActionAttr());
      if (builtin && (builtin.getValue() == BuiltinAction::t ||
                      builtin.getValue() == BuiltinAction::tdg ||
                      builtin.getValue() == BuiltinAction::ccz))
        if (failed(bump(synthesis, action, multiplier, operation, "synthesis")))
          return failure();
    } else if (auto instrument = dyn_cast<InstrumentOp>(operation)) {
      std::string name = normalizeInlineName(
          instrument.getInstrumentAttr(), "#qlx.instrument<", "qlx_standard_");
      if (failed(bump(instruments, std::move(name), multiplier, operation,
                      "instrument")) ||
          failed(add(depth, multiplier, operation, "depth")))
        return failure();
    } else if (auto prepare = dyn_cast<PrepareOp>(operation)) {
      std::string state;
      if (auto text = dyn_cast<StringAttr>(prepare.getStateAttr()))
        state = text.getValue().str();
      else
        state = printAttribute(prepare.getStateAttr());
      if (StringRef(state).starts_with("#qlx.state<"))
        state = StringRef(state).drop_front(11).drop_back().trim('"').str();
      if (failed(bump(instruments, "qlx_standard_prepare_" + state, multiplier,
                      operation, "instrument")) ||
          failed(add(depth, multiplier, operation, "depth")))
        return failure();
    } else if (auto measure = dyn_cast<MeasureOp>(operation)) {
      std::string basis = printAttribute(measure.getBasisAttr());
      if (StringRef(basis).starts_with("#qlx.pauli<"))
        basis = StringRef(basis).drop_front(11).drop_back().trim('"').lower();
      if (failed(bump(instruments, "qlx_standard_measure_" + basis, multiplier,
                      operation, "instrument")) ||
          failed(add(depth, multiplier, operation, "depth")))
        return failure();
    } else if (isa<IdleOp>(operation)) {
      if (failed(add(idleSites, multiplier, operation, "idle")) ||
          failed(add(depth, multiplier, operation, "depth")))
        return failure();
    } else if (isa<DiscardOp>(operation)) {
      if (failed(add(discards, multiplier, operation, "discard")))
        return failure();
    } else {
      supported = false;
    }

    bool touchesLogicalSemantics =
        llvm::any_of(operation->getOperandTypes(), isLogicalQubit);
    touchesLogicalSemantics |=
        llvm::any_of(operation->getResultTypes(), isLogicalQubit);
    if (!supported && touchesLogicalSemantics)
      return operation->emitOpError(
          "qlx-estimate-logical native v1 has no typed cost semantics for "
          "this operation on logical quantum values");

    int64_t delta = 0;
    for (Value operand : operation->getOperands())
      delta -= isLogicalQubit(operand.getType());
    for (Value result : operation->getResults())
      delta += isLogicalQubit(result.getType());
    live += delta;
    if (live < 0)
      return operation->emitOpError(
          "logical ownership accounting became negative during estimation");
    peak = std::max(peak, live);
    return success();
  }
};

struct QLXEstimateLogicalPass
    : public qlx::impl::QLXEstimateLogicalBase<QLXEstimateLogicalPass> {
  using QLXEstimateLogicalBase::QLXEstimateLogicalBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    ProgramOp root;
    if (!rootSymbol.empty()) {
      root = dyn_cast_or_null<ProgramOp>(symbols.lookup(rootSymbol));
      if (!root) {
        module.emitError("qlx-estimate-logical root @")
            << rootSymbol << " must resolve to qlx.program";
        return signalPassFailure();
      }
    } else {
      for (ProgramOp candidate : module.getOps<ProgramOp>()) {
        if (root) {
          module.emitError("qlx-estimate-logical requires root= when several "
                           "qlx.program symbols exist");
          return signalPassFailure();
        }
        root = candidate;
      }
      if (!root) {
        module.emitError("qlx-estimate-logical requires one qlx.program root");
        return signalPassFailure();
      }
    }

    if (resultSymbol.empty()) {
      module.emitError("qlx-estimate-logical result symbol must be nonempty");
      return signalPassFailure();
    }
    if (symbols.lookup(resultSymbol)) {
      module.emitError("qlx-estimate-logical result symbol already exists: @")
          << resultSymbol;
      return signalPassFailure();
    }

    LogicalWalker walker(module, symbols);
    if (failed(walker.run(root)))
      return signalPassFailure();

    MLIRContext *context = &getContext();
    auto i64 = IntegerType::get(context, 64);
    SmallVector<NamedAttribute> fields = {
        {StringAttr::get(context, "actions"),
         countDictionary(context, walker.actions)},
        {StringAttr::get(context, "instruments"),
         countDictionary(context, walker.instruments)},
        {StringAttr::get(context, "idle_sites"),
         IntegerAttr::get(i64, walker.idleSites)},
        {StringAttr::get(context, "discards"),
         IntegerAttr::get(i64, walker.discards)},
        {StringAttr::get(context, "logical_qubits_peak"),
         IntegerAttr::get(i64, walker.peak)},
        {StringAttr::get(context, "action_depth_upper_bound"),
         IntegerAttr::get(i64, walker.depth)},
        {StringAttr::get(context, "synthesis_demand"),
         countDictionary(context, walker.synthesis)},
    };

    OpBuilder builder(context);
    builder.setInsertionPointToEnd(module.getBody());
    auto metadata = builder.getDictionaryAttr({
        builder.getNamedAttr("producer",
                             builder.getStringAttr("qlx-estimate-logical")),
        builder.getNamedAttr("producer_version", builder.getStringAttr("1")),
    });
    EstimateResultOp::create(
        builder, root.getLoc(), builder.getStringAttr(resultSymbol),
        builder.getStringAttr("logical"),
        FlatSymbolRefAttr::get(context, root.getSymName()),
        builder.getStringAttr("qlx.logical-profile/v1"),
        DictionaryAttr::get(context, fields),
        builder.getArrayAttr({builder.getStringAttr(
            "dynamic branches are counted as a static upper bound")}),
        builder.getArrayAttr(
            {FlatSymbolRefAttr::get(context, root.getSymName())}),
        /*lowerTier=*/FlatSymbolRefAttr{}, /*device=*/FlatSymbolRefAttr{},
        metadata);
  }
};

} // namespace
