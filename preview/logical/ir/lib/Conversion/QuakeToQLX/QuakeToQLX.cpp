/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Conversion/QuakeToQLXPasses.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/DialectConversion.h"

#include <cmath>
#include <functional>
#include <string>

#ifdef QLX_HAS_CUDAQ_QUAKE
#include "cudaq/Optimizer/Dialect/CC/CCOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#endif

namespace qlx {
#define GEN_PASS_DEF_CONVERTQUAKETOQLX
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"
} // namespace qlx

using namespace mlir;

namespace {

#ifdef QLX_HAS_CUDAQ_QUAKE

// Mutable per-region conversion state threaded through importOperation so that
// structured bodies (cflow.repeat / cflow.if) can be imported recursively.
struct ImportState {
  DenseMap<Value, Value> wireOwners;    // quake wire -> qlx logical owner
  DenseMap<Value, Value> classicalBits; // quake measure-handle/i1 -> qlx i1
  DenseSet<Value> measuredWires;        // post-measurement wires (no discard)
  DenseMap<Value, SmallVector<Value>> cableOwners;
  // CUDA-Q reference helpers may unwrap the same ref/veq slot repeatedly.
  // Every such wire is an alias handle for one mutable source slot, not an
  // independent quantum owner.  Resolve these handles through the slot's
  // current P0 owner at each operation so the emitted QLX remains linear.
  DenseMap<Value, unsigned> wireReferenceSlots;
  DenseMap<Value, SmallVector<int64_t>> cableReferenceSlots;
  // CUDA-Q's reference helper ABI unwraps each scalar !quake.ref argument to
  // one wire and wraps the successor wire back to the same reference before
  // returning.  These maps make that implicit mutation an explicit linear P0
  // input/result boundary.
  DenseMap<Value, unsigned> referenceIndices;
  DenseMap<Value, std::pair<unsigned, unsigned>> vectorArgumentSlices;
  SmallVector<Value> referenceResults;
  bool referenceABI = false;
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
// structurally but omitted from cflow.repeat.
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
    registry.insert<cudaq::quake::QuakeDialect, qlx::QLXDialect,
                    qlx::lvm::LVMDialect, qlx::cflow::CflowDialect,
                    arith::ArithDialect, func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<func::FuncOp> entries;
    for (auto function : module.getOps<func::FuncOp>()) {
      if (function->hasAttr("cudaq-entrypoint"))
        entries.push_back(function);
    }

    if (entries.empty()) {
      module.emitError("convert-quake-to-qlx found no func.func with the "
                       "cudaq-entrypoint attribute");
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

    if (failed(prepareReachableFunctions(module, entries))) {
      signalPassFailure();
      return;
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

    // Declare retained logical objectives before converting any caller.  The
    // dialect verifier resolves qlx.apply symbols as each converted entry is
    // created, while dialect-conversion pattern order is intentionally not a
    // declaration-before-use ordering guarantee.
    SmallVector<func::FuncOp> orderedObjectiveDeclarations;
    for (Operation *operation : objectiveDeclarations)
      orderedObjectiveDeclarations.push_back(cast<func::FuncOp>(operation));
    llvm::sort(orderedObjectiveDeclarations,
               [](func::FuncOp lhs, func::FuncOp rhs) {
                 return lhs.getSymName() < rhs.getSymName();
               });
    OpBuilder objectiveBuilder(&getContext());
    for (func::FuncOp declaration : orderedObjectiveDeclarations)
      if (failed(convertObjectiveDeclaration(declaration, objectiveBuilder))) {
        signalPassFailure();
        return;
      }

    ConversionTarget target(getContext());
    target.addLegalDialect<arith::ArithDialect, qlx::QLXDialect,
                           qlx::lvm::LVMDialect, qlx::cflow::CflowDialect>();
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
    patterns.add<FunctionPattern>(&getContext(), *this);
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
  class FunctionPattern : public OpConversionPattern<func::FuncOp> {
  public:
    FunctionPattern(MLIRContext *context, ConvertQuakeToQLXPass &owner)
        : OpConversionPattern(context), owner(owner) {}

    LogicalResult
    matchAndRewrite(func::FuncOp function, OpAdaptor,
                    ConversionPatternRewriter &rewriter) const override {
      if (!owner.shouldConvertFunction(function)) {
        rewriter.eraseOp(function);
        return success();
      }
      if (function.isDeclaration()) {
        rewriter.eraseOp(function);
        return success();
      }
      LogicalResult converted =
          function->hasAttr("cudaq-entrypoint")
              ? owner.convertEntry(function, rewriter)
              : owner.convertReferenceHelper(function, rewriter);
      if (failed(converted))
        return failure();
      rewriter.eraseOp(function);
      return success();
    }

  private:
    ConvertQuakeToQLXPass &owner;
  };

  bool shouldConvertFunction(func::FuncOp function) const {
    return reachableFunctions.contains(function.getOperation());
  }

  std::string selectedSourceEntry;
  DenseSet<Operation *> reachableFunctions;
  DenseSet<Operation *> objectiveDeclarations;
  llvm::StringMap<std::string> qlxSymbols;
  llvm::StringMap<bool> objectiveSymbols;
  llvm::StringMap<SmallVector<uint64_t>> helperArgumentWidths;

  static std::string portableSymbol(StringRef source) {
    source.consume_front("__nvqpp__mlirgen__");
    if (std::size_t uniqueSuffix = source.find("..");
        uniqueSuffix != StringRef::npos)
      source = source.take_front(uniqueSuffix);
    return source.str();
  }

  LogicalResult prepareReachableFunctions(ModuleOp module,
                                          ArrayRef<func::FuncOp> entries) {
    reachableFunctions.clear();
    objectiveDeclarations.clear();
    qlxSymbols.clear();
    objectiveSymbols.clear();
    helperArgumentWidths.clear();
    DenseSet<Operation *> active;

    std::function<LogicalResult(func::FuncOp)> visit =
        [&](func::FuncOp function) -> LogicalResult {
      if (active.contains(function.getOperation()))
        return function.emitOpError(
            "recursive CUDA-Q helper calls are outside the Quake-to-P0 "
            "contract");
      if (reachableFunctions.contains(function.getOperation()))
        return success();
      if (function.isDeclaration()) {
        if (function->hasAttr("cudaq-entrypoint") ||
            !function->hasAttr("cudaq-kernel") ||
            !function->hasAttr("qlx-objective"))
          return function.emitOpError(
              "reachable body-less helper must be an explicitly marked "
              "non-entry CUDA-Q Logical objective declaration");
        reachableFunctions.insert(function.getOperation());
        objectiveDeclarations.insert(function.getOperation());
        return success();
      }

      active.insert(function.getOperation());
      LogicalResult result = success();
      function.walk([&](cudaq::quake::CallByRefOp call) {
        if (failed(result))
          return;
        auto callee = dyn_cast_or_null<func::FuncOp>(
            SymbolTable::lookupSymbolIn(module, call.getCalleeAttr()));
        if (!callee) {
          call.emitOpError("references an unresolved CUDA-Q helper ")
              << call.getCalleeAttr();
          result = failure();
          return;
        }
        if (callee.getNumArguments() != call.getArgs().size()) {
          call.emitOpError(
              "helper call argument count does not match the callee ABI");
          result = failure();
          return;
        }
        SmallVector<uint64_t> widths;
        for (auto [argument, parameter] :
             llvm::zip(call.getArgs(), callee.getArgumentTypes())) {
          if (isa<cudaq::quake::RefType>(parameter) &&
              isa<cudaq::quake::WireType>(argument.getType())) {
            widths.push_back(1);
            continue;
          }
          auto vector = dyn_cast<cudaq::quake::VeqType>(parameter);
          auto cable = dyn_cast<cudaq::quake::CableType>(argument.getType());
          if (vector && cable) {
            widths.push_back(cable.getSize());
            continue;
          }
          call.emitOpError(
              "helper ABI requires !quake.ref/!quake.wire or specialized "
              "!quake.veq/!quake.cable argument pairs");
          result = failure();
          return;
        }
        auto specialization = helperArgumentWidths.find(callee.getSymName());
        if (specialization == helperArgumentWidths.end()) {
          helperArgumentWidths[callee.getSymName()] = widths;
        } else if (specialization->second != widths) {
          call.emitOpError(
              "one CUDA-Q helper is called with several cable widths; "
              "specialize it to one exact signature before QLX import");
          result = failure();
          return;
        }
        result = visit(callee);
      });
      active.erase(function.getOperation());
      if (failed(result))
        return failure();
      reachableFunctions.insert(function.getOperation());
      return success();
    };

    for (func::FuncOp entry : entries) {
      if (!selectedSourceEntry.empty() &&
          entry.getSymName() != selectedSourceEntry)
        continue;
      if (failed(visit(entry)))
        return failure();
    }

    llvm::StringMap<Operation *> normalizedOwners;
    SmallVector<func::FuncOp> ordered;
    for (Operation *operation : reachableFunctions)
      ordered.push_back(cast<func::FuncOp>(operation));
    llvm::sort(ordered, [](func::FuncOp lhs, func::FuncOp rhs) {
      return lhs.getSymName() < rhs.getSymName();
    });
    for (func::FuncOp function : ordered) {
      std::string normalized = portableSymbol(function.getSymName());
      if (normalized.empty())
        return function.emitOpError(
            "CUDA-Q symbol normalizes to an empty QLX program name");
      auto [owner, inserted] =
          normalizedOwners.try_emplace(normalized, function.getOperation());
      if (!inserted && owner->second != function.getOperation())
        return function.emitOpError()
               << "CUDA-Q helper name collision after QLX normalization: @"
               << normalized;
      qlxSymbols[function.getSymName()] = normalized;
      if (objectiveDeclarations.contains(function.getOperation()))
        objectiveSymbols[normalized] = true;
    }
    return success();
  }

  FailureOr<StringRef> qlxSymbolFor(StringRef source, Operation *diagnostic) {
    auto found = qlxSymbols.find(source);
    if (found == qlxSymbols.end()) {
      diagnostic->emitOpError("references CUDA-Q helper outside the selected "
                              "reachable call graph: @")
          << source;
      return failure();
    }
    return StringRef(found->second);
  }

  static LogicalResult requireWireSemantics(func::FuncOp function) {
    WalkResult result = function.walk([](Operation *op) {
      for (Type type :
           llvm::concat<Type>(op->getOperandTypes(), op->getResultTypes())) {
        if (!cudaq::quake::isQuantumType(type) ||
            isa<cudaq::quake::WireType>(type))
          continue;
        bool supportedCableBoundary =
            isa<cudaq::quake::CableType>(type) &&
            isa<cudaq::quake::BundleCableOp, cudaq::quake::SplitCableOp,
                cudaq::quake::CallByRefOp>(op);
        // CUDA-Q inserts quake.log_output solely to keep Python-owned quantum
        // values live until the end of an entry point.  It is transparent to
        // the logical program and may mention an aggregate that has otherwise
        // already been scalarized.
        supportedCableBoundary |=
            op->getName().getStringRef() == "quake.log_output";
        if (!supportedCableBoundary) {
          op->emitOpError("Quake-to-P0 supports only value-semantics "
                          "!quake.wire values plus statically sized helper "
                          "cable boundaries");
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

    // Two-control X is the standard logical Toffoli action. Preserve that
    // objective through P0/P1 so P2 can select an integrated code-specific
    // realization (for example, AutoCCZ consumption) instead of committing
    // early to an H-CCZ-H decomposition.
    if constexpr (std::is_same_v<GateOp, cudaq::quake::XOp>) {
      if (gate.getControls().size() == 2 && gate.getTargets().size() == 1) {
        auto results =
            emitApply(builder, gate.getLoc(), qlx::BuiltinAction::ccx, inputs);
        for (auto [quakeResult, qlxResult] : llvm::zip(wires, results))
          values[quakeResult] = qlxResult;
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
    if (state.wireReferenceSlots.contains(wire))
      return owner->emitOpError(
          "cannot discard a borrowed reference-helper owner");
    return discardWires(owner, ValueRange{wire}, builder, state);
  }

  static LogicalResult hydrateReferenceWires(Operation *owner, ValueRange wires,
                                             ImportState &state) {
    DenseSet<unsigned> seen;
    for (Value wire : wires) {
      auto slot = state.wireReferenceSlots.find(wire);
      if (slot == state.wireReferenceSlots.end())
        continue;
      if (slot->second >= state.referenceResults.size() ||
          !state.referenceResults[slot->second])
        return owner->emitOpError(
            "reference-helper wire resolves to no current P0 owner");
      if (!seen.insert(slot->second).second)
        return owner->emitOpError(
            "uses two aliases of the same reference slot in one operation");
      state.wireOwners[wire] = state.referenceResults[slot->second];
    }
    return success();
  }

  static LogicalResult hydrateReferenceCable(Operation *owner, Value cable,
                                             ImportState &state) {
    auto slots = state.cableReferenceSlots.find(cable);
    if (slots == state.cableReferenceSlots.end())
      return success();
    auto owners = state.cableOwners.find(cable);
    if (owners == state.cableOwners.end() ||
        owners->second.size() != slots->second.size())
      return owner->emitOpError(
          "reference-helper cable has inconsistent owner provenance");
    DenseSet<unsigned> seen;
    for (auto [index, slot] : llvm::enumerate(slots->second)) {
      if (slot < 0)
        continue;
      auto unsignedSlot = static_cast<unsigned>(slot);
      if (unsignedSlot >= state.referenceResults.size() ||
          !state.referenceResults[unsignedSlot])
        return owner->emitOpError(
            "reference-helper cable resolves to no current P0 owner");
      if (!seen.insert(unsignedSlot).second)
        return owner->emitOpError(
            "contains two aliases of the same reference slot");
      owners->second[index] = state.referenceResults[unsignedSlot];
    }
    return success();
  }

  static void refreshReferenceAliases(unsigned slot, Value owner,
                                      ImportState &state) {
    state.referenceResults[slot] = owner;
    for (auto [wire, aliasSlot] : state.wireReferenceSlots)
      if (aliasSlot == slot)
        state.wireOwners[wire] = owner;
    for (auto [cable, slots] : state.cableReferenceSlots) {
      auto owners = state.cableOwners.find(cable);
      if (owners == state.cableOwners.end())
        continue;
      for (auto [index, aliasSlot] : llvm::enumerate(slots))
        if (aliasSlot == static_cast<int64_t>(slot))
          owners->second[index] = owner;
    }
  }

  static LogicalResult commitReferenceWireResults(Operation *owner,
                                                  ValueRange inputs,
                                                  ValueRange results,
                                                  ImportState &state) {
    if (inputs.size() != results.size())
      return owner->emitOpError(
          "does not preserve reference-helper owner arity");
    for (auto [input, result] : llvm::zip(inputs, results)) {
      auto slot = state.wireReferenceSlots.find(input);
      if (slot == state.wireReferenceSlots.end())
        continue;
      auto mapped = state.wireOwners.find(result);
      if (mapped == state.wireOwners.end())
        return owner->emitOpError(
            "reference-helper result has no live P0 owner");
      state.wireReferenceSlots[result] = slot->second;
      refreshReferenceAliases(slot->second, mapped->second, state);
    }
    return success();
  }

  template <typename GateOp>
  static LogicalResult convertReferenceAwareGate(GateOp gate,
                                                 OpBuilder &builder,
                                                 ImportState &state) {
    SmallVector<Value> inputs;
    inputs.append(gate.getControls().begin(), gate.getControls().end());
    inputs.append(gate.getTargets().begin(), gate.getTargets().end());
    if (failed(hydrateReferenceWires(gate, inputs, state)))
      return failure();
    if (failed(convertGate(gate, builder, state.wireOwners)))
      return failure();
    return commitReferenceWireResults(gate, inputs, gate.getWires(), state);
  }

  static LogicalResult convertReferenceAwareSwap(cudaq::quake::SwapOp swap,
                                                 OpBuilder &builder,
                                                 ImportState &state) {
    SmallVector<Value> inputs(swap.getTargets().begin(),
                              swap.getTargets().end());
    if (failed(hydrateReferenceWires(swap, inputs, state)))
      return failure();
    if (failed(convertSwap(swap, builder, state.wireOwners)))
      return failure();
    return commitReferenceWireResults(swap, inputs, swap.getWires(), state);
  }

  template <typename GateOp>
  static LogicalResult
  convertReferenceAwareRotation(GateOp gate, OpBuilder &builder,
                                ImportState &state, int64_t xMask,
                                int64_t zMask) {
    SmallVector<Value> inputs(gate.getTargets().begin(),
                              gate.getTargets().end());
    if (failed(hydrateReferenceWires(gate, inputs, state)))
      return failure();
    if (failed(convertRotation(gate, builder, state.wireOwners, xMask, zMask)))
      return failure();
    return commitReferenceWireResults(gate, inputs, gate.getWires(), state);
  }

  LogicalResult convertReferenceCall(cudaq::quake::CallByRefOp call,
                                     OpBuilder &builder, ImportState &state) {
    SmallVector<Value> inputs;
    SmallVector<int64_t> inputSlots;
    for (Value argument : call.getArgs()) {
      if (isa<cudaq::quake::WireType>(argument.getType())) {
        if (failed(hydrateReferenceWires(call, ValueRange{argument}, state)))
          return failure();
        auto found = state.wireOwners.find(argument);
        if (found == state.wireOwners.end())
          return call.emitOpError("uses a wire with no live P0 owner");
        inputs.push_back(found->second);
        auto slot = state.wireReferenceSlots.find(argument);
        inputSlots.push_back(slot == state.wireReferenceSlots.end()
                                 ? -1
                                 : static_cast<int64_t>(slot->second));
        state.wireOwners.erase(found);
        continue;
      }
      if (isa<cudaq::quake::CableType>(argument.getType())) {
        if (failed(hydrateReferenceCable(call, argument, state)))
          return failure();
        auto found = state.cableOwners.find(argument);
        if (found == state.cableOwners.end())
          return call.emitOpError("uses a cable with no live P0 owners");
        inputs.append(found->second.begin(), found->second.end());
        auto slots = state.cableReferenceSlots.find(argument);
        if (slots == state.cableReferenceSlots.end())
          inputSlots.append(found->second.size(), -1);
        else
          inputSlots.append(slots->second.begin(), slots->second.end());
        state.cableOwners.erase(found);
        state.cableReferenceSlots.erase(argument);
        continue;
      }
      return call.emitOpError(
          "helper arguments must be scalar wires or statically sized cables");
    }

    uint64_t flattenedResults = 0;
    for (Type type : call.getResultTypes()) {
      if (isa<cudaq::quake::WireType>(type)) {
        ++flattenedResults;
        continue;
      }
      if (auto cable = dyn_cast<cudaq::quake::CableType>(type)) {
        flattenedResults += cable.getSize();
        continue;
      }
      return call.emitOpError(
          "helper results must be scalar wires or statically sized cables");
    }
    if (flattenedResults != inputs.size())
      return call.emitOpError(
          "helper must return one flattened successor per input owner");

    auto callee = qlxSymbolFor(call.getCallee(), call.getOperation());
    if (failed(callee))
      return failure();
    SmallVector<Type> resultTypes(
        flattenedResults, qlx::LogicalQubitType::get(builder.getContext()));
    SmallVector<Value> importedResults;
    if (objectiveSymbols.contains(*callee)) {
      auto imported = qlx::ApplyOp::create(
          builder, call.getLoc(), resultTypes,
          FlatSymbolRefAttr::get(builder.getContext(), *callee), inputs,
          nullptr);
      importedResults.append(imported.getResults().begin(),
                             imported.getResults().end());
    } else {
      auto imported = qlx::CallOp::create(builder, call.getLoc(), resultTypes,
                                          *callee, inputs);
      importedResults.append(imported.getResults().begin(),
                             imported.getResults().end());
    }
    unsigned offset = 0;
    for (Value source : call.getResults()) {
      if (isa<cudaq::quake::WireType>(source.getType())) {
        Value result = importedResults[offset];
        state.wireOwners[source] = result;
        if (inputSlots[offset] >= 0) {
          unsigned slot = static_cast<unsigned>(inputSlots[offset]);
          state.wireReferenceSlots[source] = slot;
          refreshReferenceAliases(slot, result, state);
        }
        ++offset;
        continue;
      }
      auto cable = cast<cudaq::quake::CableType>(source.getType());
      SmallVector<Value> owners;
      SmallVector<int64_t> slots;
      for (uint64_t index = 0; index < cable.getSize(); ++index) {
        Value result = importedResults[offset];
        owners.push_back(result);
        slots.push_back(inputSlots[offset]);
        if (inputSlots[offset] >= 0)
          refreshReferenceAliases(static_cast<unsigned>(inputSlots[offset]),
                                  result, state);
        ++offset;
      }
      state.cableOwners[source] = std::move(owners);
      state.cableReferenceSlots[source] = std::move(slots);
    }
    return success();
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
              return convertReferenceAwareGate(gate, builder, state);
            })
        .Case<cudaq::quake::SwapOp>([&](auto swap) -> LogicalResult {
          return convertReferenceAwareSwap(swap, builder, state);
        })
        .Case<cudaq::quake::CallByRefOp>([&](auto call) -> LogicalResult {
          return convertReferenceCall(call, builder, state);
        })
        .Case<cudaq::quake::BundleCableOp>([&](auto bundle) -> LogicalResult {
          SmallVector<Value> owners;
          SmallVector<int64_t> slots;
          if (failed(hydrateReferenceWires(bundle, bundle.getWires(), state)))
            return failure();
          for (Value wire : bundle.getWires()) {
            auto found = state.wireOwners.find(wire);
            if (found == state.wireOwners.end())
              return bundle.emitOpError("bundles a wire with no live P0 owner");
            owners.push_back(found->second);
            auto slot = state.wireReferenceSlots.find(wire);
            slots.push_back(slot == state.wireReferenceSlots.end()
                                ? -1
                                : static_cast<int64_t>(slot->second));
            state.wireOwners.erase(found);
          }
          auto type =
              cast<cudaq::quake::CableType>(bundle.getResult().getType());
          if (type.getSize() != owners.size())
            return bundle.emitOpError(
                "cable width does not match its flattened owners");
          state.cableOwners[bundle.getResult()] = std::move(owners);
          state.cableReferenceSlots[bundle.getResult()] = std::move(slots);
          return success();
        })
        .Case<cudaq::quake::SplitCableOp>([&](auto split) -> LogicalResult {
          auto found = state.cableOwners.find(split.getCable());
          if (found == state.cableOwners.end())
            return split.emitOpError("splits a cable with no live P0 owners");
          if (found->second.size() != split.getNumResults())
            return split.emitOpError(
                "split result count does not match the cable width");
          auto slots = state.cableReferenceSlots.find(split.getCable());
          for (auto [index, wire, owner] :
               llvm::enumerate(split.getResults(), found->second)) {
            state.wireOwners[wire] = owner;
            if (slots != state.cableReferenceSlots.end() &&
                slots->second[index] >= 0)
              state.wireReferenceSlots[wire] =
                  static_cast<unsigned>(slots->second[index]);
          }
          state.cableOwners.erase(found);
          state.cableReferenceSlots.erase(split.getCable());
          return success();
        })
        .Case<cudaq::quake::ExtractRefOp>([&](auto extract) -> LogicalResult {
          if (!state.referenceABI)
            return extract.emitOpError(
                "quake.extract_ref is legal only in a specialized "
                "reference helper");
          auto slice = state.vectorArgumentSlices.find(extract.getVeq());
          if (slice == state.vectorArgumentSlices.end())
            return extract.emitOpError(
                "vector source is not a specialized helper argument");
          uint64_t index = extract.getRawIndex();
          if (index >= slice->second.second)
            return extract.emitOpError(
                "static vector index exceeds the specialized cable width");
          unsigned flattened = slice->second.first + index;
          if (!state.referenceResults[flattened])
            return extract.emitOpError(
                "extracts a vector element with no current owner");
          state.referenceIndices[extract.getRef()] = flattened;
          return success();
        })
        .Case<cudaq::quake::UnwrapOp>([&](auto unwrap) -> LogicalResult {
          if (!state.referenceABI)
            return unwrap.emitOpError(
                "quake.unwrap is legal only in a scalar reference helper");
          Value reference = unwrap.getRefValue();
          auto index = state.referenceIndices.find(reference);
          if (index == state.referenceIndices.end())
            return unwrap.emitOpError(
                "reference is not a specialized helper input slot");
          if (!state.referenceResults[index->second])
            return unwrap.emitOpError(
                "reference-helper slot has no current P0 owner");
          state.wireReferenceSlots[unwrap.getResult()] = index->second;
          state.wireOwners[unwrap.getResult()] =
              state.referenceResults[index->second];
          return success();
        })
        .Case<cudaq::quake::WrapOp>([&](auto wrap) -> LogicalResult {
          if (!state.referenceABI)
            return wrap.emitOpError(
                "quake.wrap is legal only in a scalar reference helper");
          Value reference = wrap.getRefValue();
          auto index = state.referenceIndices.find(reference);
          if (index == state.referenceIndices.end())
            return wrap.emitOpError("target is not a helper input reference");
          auto owner = state.wireOwners.find(wrap.getWireValue());
          if (owner == state.wireOwners.end())
            return wrap.emitOpError("uses a wire with no live P0 owner");
          auto slot = state.wireReferenceSlots.find(wrap.getWireValue());
          if (slot == state.wireReferenceSlots.end() ||
              slot->second != index->second)
            return wrap.emitOpError(
                "wire provenance does not match the wrapped reference slot");
          if (state.referenceResults[index->second] != owner->second)
            return wrap.emitOpError(
                "wrapped wire is not the current reference-slot owner");
          return success();
        })
        .Case<cudaq::quake::R1Op>([&](auto g) -> LogicalResult {
          // On one uncontrolled target, R1(theta) and Rz(theta) differ only
          // by an unobservable global phase. Controlled R1 is rejected by
          // convertRotation because that phase would become relative.
          return convertReferenceAwareRotation(g, builder, state, 0, 1);
        })
        .Case<cudaq::quake::RxOp>([&](auto g) -> LogicalResult {
          return convertReferenceAwareRotation(g, builder, state, 1, 0);
        })
        .Case<cudaq::quake::RyOp>([&](auto g) -> LogicalResult {
          return convertReferenceAwareRotation(g, builder, state, 1, 1);
        })
        .Case<cudaq::quake::RzOp>([&](auto g) -> LogicalResult {
          return convertReferenceAwareRotation(g, builder, state, 0, 1);
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
          if (llvm::any_of(m.getTargets(), [&](Value wire) {
                return state.wireReferenceSlots.contains(wire);
              }))
            return m.emitOpError(
                "reference-helper inputs cannot be destructively measured");
          return convertMeasurement(m, qlx::Pauli::Z, builder, state.wireOwners,
                                    state.classicalBits, state.measuredWires);
        })
        .Case<cudaq::quake::MxOp>([&](auto m) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(m, "measurement");
          if (llvm::any_of(m.getTargets(), [&](Value wire) {
                return state.wireReferenceSlots.contains(wire);
              }))
            return m.emitOpError(
                "reference-helper inputs cannot be destructively measured");
          return convertMeasurement(m, qlx::Pauli::X, builder, state.wireOwners,
                                    state.classicalBits, state.measuredWires);
        })
        .Case<cudaq::quake::MyOp>([&](auto m) -> LogicalResult {
          if (structured && !repeatBody)
            return rejectStructured(m, "measurement");
          if (llvm::any_of(m.getTargets(), [&](Value wire) {
                return state.wireReferenceSlots.contains(wire);
              }))
            return m.emitOpError(
                "reference-helper inputs cannot be destructively measured");
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
          if (state.referenceABI) {
            if (!sourceReturn.getOperands().empty())
              return sourceReturn.emitOpError(
                  "scalar reference helper must return through quake.wrap, "
                  "not func.return operands");
            for (auto [wire, owner] : state.wireOwners)
              if (!state.wireReferenceSlots.contains(wire))
                return sourceReturn.emitOpError(
                    "helper leaves a live non-reference quantum owner");
            if (llvm::any_of(state.referenceResults,
                             [](Value value) { return !value; }))
              return sourceReturn.emitOpError(
                  "helper does not wrap one successor for every input "
                  "reference");
            qlx::ReturnOp::create(builder, sourceReturn.getLoc(),
                                  state.referenceResults);
            return success();
          }
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
          if (unsupported->getName().getStringRef() == "quake.log_output") {
            // Python frontend lifetime logging has no logical effect.  The
            // scalar form forwards its wire, so retain the current owner for
            // the result; aggregate logging has no results and can disappear.
            if (unsupported->getNumResults() == 0)
              return success();
            if (unsupported->getNumOperands() != 1 ||
                unsupported->getNumResults() != 1 ||
                !isa<cudaq::quake::WireType>(
                    unsupported->getOperand(0).getType()) ||
                !isa<cudaq::quake::WireType>(
                    unsupported->getResult(0).getType()))
              return unsupported->emitOpError(
                  "unsupported quantum lifetime logging shape");
            auto owner = state.wireOwners.find(unsupported->getOperand(0));
            if (owner == state.wireOwners.end())
              return unsupported->emitOpError(
                  "logs a wire with no live P0 owner");
            Value logicalOwner = owner->second;
            state.wireOwners.erase(owner);
            state.wireOwners[unsupported->getResult(0)] = logicalOwner;
            return success();
          }
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

  // Fold a normalized constant-trip cc.loop into one cflow.repeat (never
  // unroll).
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
    auto repeat = qlx::cflow::RepeatOp::create(
        builder, loc, iterTypes, builder.getI64IntegerAttr(m.tripCount), inits,
        StringAttr{});

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
      qlx::cflow::YieldOp::create(builder, loc, yields);
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
  // (measurement) condition becomes an adaptive cflow.if (B2b).
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
    auto qif = qlx::cflow::IfOp::create(builder, loc, resTypes,
                                        condOwner->second, StringAttr{});

    // Lower one cc.if branch into a cflow.if branch block (carries captured,
    // not block-args). An absent else is a synthesized identity.
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
      qlx::cflow::YieldOp::create(builder, loc, yields);
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
    auto sourceName =
        qlxSymbolFor(function.getSymName(), function.getOperation());
    if (failed(sourceName))
      return failure();
    auto functionType =
        FunctionType::get(function.getContext(), {}, function.getResultTypes());
    auto program = qlx::ProgramOp::create(builder, function.getLoc(),
                                          *sourceName, functionType,
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

  LogicalResult convertObjectiveDeclaration(func::FuncOp function,
                                            OpBuilder &builder) {
    if (!function.getResultTypes().empty())
      return function.emitOpError(
          "CUDA-Q objective declarations return updated qubits through their "
          "reference arguments");
    if (function.getNumArguments() == 0)
      return function.emitOpError(
          "CUDA-Q objective declarations require at least one quantum input");
    for (BlockArgument argument : function.getArguments())
      if (!isa<cudaq::quake::RefType, cudaq::quake::VeqType>(
              argument.getType()))
        return function.emitOpError(
            "CUDA-Q objective declarations accept only quantum reference "
            "arguments");

    auto specialization = helperArgumentWidths.find(function.getSymName());
    if (specialization == helperArgumentWidths.end() ||
        specialization->second.size() != function.getNumArguments())
      return function.emitOpError(
          "CUDA-Q objective declaration is missing its exact call-site width "
          "specialization");
    auto symbol = qlxSymbolFor(function.getSymName(), function.getOperation());
    if (failed(symbol))
      return failure();

    uint64_t flattenedWidth = 0;
    for (uint64_t width : specialization->second)
      flattenedWidth += width;
    auto logicalType = qlx::LogicalQubitType::get(function.getContext());
    SmallVector<Type> boundaryTypes(flattenedWidth, logicalType);
    auto functionType =
        FunctionType::get(function.getContext(), boundaryTypes, boundaryTypes);

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(function);
    OperationState objective(function.getLoc(),
                             qlx::ActionOp::getOperationName());
    objective.addAttribute(SymbolTable::getSymbolAttrName(),
                           builder.getStringAttr(*symbol));
    objective.addAttribute("function_type", TypeAttr::get(functionType));
    objective.addAttribute("kind", builder.getStringAttr("composite"));
    builder.create(objective);
    return success();
  }

  LogicalResult convertReferenceHelper(func::FuncOp function,
                                       ConversionPatternRewriter &builder) {
    if (!function.getBody().hasOneBlock())
      return function.emitOpError(
          "Quake-to-P0 scalar reference helpers require one block");
    if (!function.getResultTypes().empty())
      return function.emitOpError(
          "reference helper results must be returned through "
          "quake.wrap");
    if (function.getNumArguments() == 0)
      return function.emitOpError(
          "reachable helper without scalar quantum inputs is unsupported");
    for (BlockArgument argument : function.getArguments())
      if (!isa<cudaq::quake::RefType, cudaq::quake::VeqType>(
              argument.getType()))
        return function.emitOpError(
            "reachable helpers require only !quake.ref or specialized "
            "!quake.veq arguments");

    auto specialization = helperArgumentWidths.find(function.getSymName());
    if (specialization == helperArgumentWidths.end() ||
        specialization->second.size() != function.getNumArguments())
      return function.emitOpError(
          "reachable helper is missing its exact call-site width "
          "specialization");

    auto symbol = qlxSymbolFor(function.getSymName(), function.getOperation());
    if (failed(symbol))
      return failure();

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPoint(function);
    auto logicalType = qlx::LogicalQubitType::get(function.getContext());
    uint64_t flattenedWidth = 0;
    for (uint64_t width : specialization->second)
      flattenedWidth += width;
    SmallVector<Type> boundaryTypes(flattenedWidth, logicalType);
    auto functionType =
        FunctionType::get(function.getContext(), boundaryTypes, boundaryTypes);
    auto program = qlx::ProgramOp::create(builder, function.getLoc(), *symbol,
                                          functionType,
                                          /*estimateOnly=*/nullptr,
                                          /*specialization=*/nullptr);
    program->setAttr("qlx.stage", builder.getStringAttr("p0"));
    program->setAttr("qlx.profile", builder.getStringAttr("p0"));
    Block *body = &program.getBody().emplaceBlock();
    for (Type type : boundaryTypes)
      body->addArgument(type, function.getLoc());
    builder.setInsertionPointToStart(body);

    ImportState state;
    state.referenceABI = true;
    state.referenceResults.append(body->getArguments().begin(),
                                  body->getArguments().end());
    unsigned offset = 0;
    for (auto [argumentIndex, source] :
         llvm::enumerate(function.getArguments())) {
      unsigned width = specialization->second[argumentIndex];
      if (isa<cudaq::quake::RefType>(source.getType())) {
        if (width != 1)
          return function.emitOpError(
              "scalar reference argument has a non-unit specialization");
        state.referenceIndices[source] = offset;
      } else {
        state.vectorArgumentSlices[source] = {offset, width};
      }
      offset += width;
    }

    Block &sourceBody = function.getBody().front();
    for (Operation &source : sourceBody)
      if (failed(importOperation(source, builder, state, RegionPolicy::Entry)))
        return failure();

    if (body->empty() || !isa<qlx::ReturnOp>(body->back()))
      return function.emitOpError(
          "scalar reference helper is missing its func.return boundary");
    return success();
  }
};

#else

class ConvertQuakeToQLXPass
    : public qlx::impl::ConvertQuakeToQLXBase<ConvertQuakeToQLXPass> {
public:
  using ConvertQuakeToQLXBase::ConvertQuakeToQLXBase;

  void runOnOperation() override {
    getOperation().emitError(
        "Quake-to-P0 support was not enabled; enable QLX_USE_CUDAQ_SDK");
    signalPassFailure();
  }
};

#endif

} // namespace
