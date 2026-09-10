/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Compiler/SpacetimeProviders.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Fabric/IR/ResourceContract.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/Phys/IR/SpacetimeDerivation.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>

using namespace mlir;

namespace {

static std::optional<int64_t>
checkedProduct(std::initializer_list<int64_t> factors) {
  int64_t result = 1;
  for (int64_t factor : factors)
    if (llvm::MulOverflow(result, factor, result))
      return std::nullopt;
  return result;
}

static std::optional<int64_t> checkedSum(std::initializer_list<int64_t> terms) {
  int64_t result = 0;
  for (int64_t term : terms)
    if (llvm::AddOverflow(result, term, result))
      return std::nullopt;
  return result;
}

namespace resource_provider {
using namespace qlx::fabric;

static bool sameValues(ValueRange actual, ArrayRef<Value> expected) {
  return actual.size() == expected.size() && llvm::equal(actual, expected);
}

static LogicalResult verifyAutoCCZRoles(Operation *owner, ArrayAttr roles,
                                        unsigned payloadCount) {
  static constexpr std::array<llvm::StringLiteral, 9> expected = {
      "main_a",     "main_b",     "main_c",     "route_ab_a", "route_ab_b",
      "route_bc_b", "route_bc_c", "route_ca_c", "route_ca_a"};
  if (!roles)
    return owner->emitOpError(
        "AutoCCZ payload requires the exact nine-role schema");
  if (payloadCount != expected.size() || roles.size() != expected.size())
    return owner->emitOpError(
        "AutoCCZ payload requires exactly three main and six routing roles");
  for (auto [index, raw] : llvm::enumerate(roles)) {
    auto role = dyn_cast<StringAttr>(raw);
    if (!role || role.getValue() != expected[index])
      return owner->emitOpError(
          "AutoCCZ payload roles must use the canonical order");
  }
  return success();
}

static qlx::fabric::GadgetSpecOp
resolvedGadgetSpec(qlx::fabric::GadgetOp gadget) {
  if (!gadget || !gadget.getSpecAttr())
    return {};
  return dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
      SymbolTable::lookupNearestSymbolFrom(gadget, gadget.getSpecAttr()));
}

static std::optional<StringRef>
gadgetLogicalAction(qlx::fabric::GadgetOp gadget) {
  auto spec = resolvedGadgetSpec(gadget);
  if (!spec)
    return std::nullopt;
  auto objective = dyn_cast_or_null<qlx::fabric::ObjectiveOp>(
      SymbolTable::lookupNearestSymbolFrom(spec, spec.getObjectiveAttr()));
  if (!objective || !objective.getLogicalAttr())
    return std::nullopt;
  auto logical =
      dyn_cast_or_null<qlx::ActionOp>(SymbolTable::lookupNearestSymbolFrom(
          objective, objective.getLogicalAttr()));
  return logical ? std::optional<StringRef>(logical.getKind()) : std::nullopt;
}

static qlx::InstrumentDeclOp
gadgetLogicalInstrument(qlx::fabric::GadgetOp gadget) {
  auto spec = resolvedGadgetSpec(gadget);
  if (!spec)
    return {};
  auto objective = dyn_cast_or_null<qlx::fabric::ObjectiveOp>(
      SymbolTable::lookupNearestSymbolFrom(spec, spec.getObjectiveAttr()));
  if (!objective || !objective.getLogicalAttr())
    return {};
  return dyn_cast_or_null<qlx::InstrumentDeclOp>(
      SymbolTable::lookupNearestSymbolFrom(objective,
                                           objective.getLogicalAttr()));
}

static std::optional<qlx::Pauli>
logicalMeasurementBasis(qlx::InstrumentDeclOp instrument) {
  if (!instrument)
    return std::nullopt;
  auto semantics =
      dyn_cast_or_null<FlatSymbolRefAttr>(instrument.getSemanticsAttr());
  auto body =
      semantics
          ? dyn_cast_or_null<qlx::ObjectiveBodyOp>(
                SymbolTable::lookupNearestSymbolFrom(instrument, semantics))
          : qlx::ObjectiveBodyOp{};
  if (!body || body.getBody().empty())
    return std::nullopt;
  Block &block = body.getBody().front();
  if (block.getNumArguments() != 1 ||
      std::distance(block.begin(), block.end()) != 2)
    return std::nullopt;
  auto measure = dyn_cast<qlx::MeasureOp>(block.front());
  auto returned = dyn_cast<qlx::ReturnOp>(block.back());
  if (!measure || !returned || measure.getInput() != block.getArgument(0) ||
      returned.getResults().size() != 1 ||
      returned.getResults()[0] != measure.getResult())
    return std::nullopt;
  return measure.getBasis();
}

static bool exactFabricMeasureCall(qlx::fabric::CallOp call, qlx::Pauli basis,
                                   Value input) {
  if (!call || !sameValues(call.getOperands(), {input}) ||
      call.getNumResults() != 1 || !call.getResult(0).getType().isInteger(1))
    return false;
  auto gadget = dyn_cast_or_null<qlx::fabric::GadgetOp>(
      SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
  auto actual = logicalMeasurementBasis(gadgetLogicalInstrument(gadget));
  return actual && *actual == basis;
}

} // namespace resource_provider

constexpr StringLiteral kSpacetimeProvider = "qlx.compiler.spacetime";
constexpr StringLiteral kSpacetimeProviderVersion = "1";
constexpr StringLiteral kSurfaceFactoryRecurrence =
    "surface_autoccz_factory_recurrence";
constexpr int64_t kSurfaceFactoryRecurrenceVersion = 3;
constexpr StringLiteral kSurfaceFactoryRecurrenceEvidence =
    "fabric.factory-circuit/v2+gidney-fowler-autoccz-layout/v3+selected-"
    "device-closure/v1";
constexpr StringLiteral kSurfaceAutoCCZApplication =
    "surface_autoccz_application";
constexpr int64_t kSurfaceAutoCCZApplicationVersion = 1;
constexpr StringLiteral kSurfaceAutoCCZApplicationEvidence =
    "fabric.autoccz-consumer/v1+selected-device-reaction-pipeline/v1";
constexpr StringLiteral kSurfaceSpacelikeCallable =
    "surface_spacelike_callable";
constexpr int64_t kSurfaceSpacelikeCallableVersion = 1;
constexpr StringLiteral kSurfaceSpacelikeCallableEvidence =
    "fabric.autoccz-reaction-layer/v1+surface-alternating-access/v1+"
    "selected-device-closure/v1";
constexpr StringLiteral kRawTState = "raw_t_state";
constexpr StringLiteral kTState = "t_state";
constexpr StringLiteral kCCZState = "ccz_state";
constexpr StringLiteral kAutoCCZState = "auto_ccz_state";

namespace resource_provider {
using namespace qlx::fabric;
static bool hasExactFifteenToOneStructure(ProtocolOp protocol);
static bool hasExactGidneyFowlerCCZStructure(ProtocolOp protocol);
static bool hasExactGidneyFowlerAutoCCZStructure(ProtocolOp protocol);
static bool hasExactSurfaceAutoCCZApplicationStructure(ProtocolOp protocol);
static bool isLogicalCZCall(CallOp call);
} // namespace resource_provider

namespace plan_provider {
using namespace qlx::phys;

struct VerifiedSurfaceFactoryRecurrenceLayout {
  int64_t level1Distance;
  int64_t level2Distance;
  int64_t level1Lanes;
  int64_t tStatesPerCCZ;
  int64_t fixupBoxes;
  int64_t level1PatchesPerLane;
  int64_t cczPatches;
  int64_t autoCCZPatches;
  int64_t factoryPatches;
  int64_t level1ResourceOffset;
  int64_t level2ResourceOffset;
  std::string level1ResourceClass;
  std::string level2ResourceClass;

  SmallVector<int64_t, 9> geometry() const {
    return {level1Distance, level2Distance, level1Lanes,
            tStatesPerCCZ,  fixupBoxes,     level1PatchesPerLane,
            cczPatches,     autoCCZPatches, factoryPatches};
  }
};

struct VerifiedSurfaceFactoryStageShape {
  int64_t level1QuarterLayers;
  int64_t level2Layers;
  int64_t autoCCZLayers;
};

static std::optional<double> spacetimeNumericValue(Attribute attribute) {
  if (auto value = dyn_cast_or_null<FloatAttr>(attribute))
    return value.getValueAsDouble();
  if (auto value = dyn_cast_or_null<IntegerAttr>(attribute))
    return static_cast<double>(value.getInt());
  if (auto value = dyn_cast_or_null<StringAttr>(attribute)) {
    double parsed = 0.0;
    if (!value.getValue().getAsDouble(parsed))
      return parsed;
  }
  return std::nullopt;
}

static Region *spacetimeCallableBody(Operation *callable) {
  if (auto protocol = dyn_cast<qlx::fabric::ProtocolOp>(callable))
    return &protocol.getBody();
  if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callable))
    return &gadget.getBody();
  return nullptr;
}

static FailureOr<SmallVector<Operation *>>
spacetimeCallableClosure(SpacetimePlanOp plan, qlx::fabric::ProtocolOp source) {
  SmallVector<Operation *> closure;
  llvm::StringSet<> visited;
  llvm::StringSet<> active;
  SymbolTable moduleSymbols(plan->getParentOfType<ModuleOp>());
  std::function<LogicalResult(Operation *)> visit =
      [&](Operation *callable) -> LogicalResult {
    StringRef name = SymbolTable::getSymbolName(callable).getValue();
    if (name.empty() || active.contains(name))
      return plan.emitOpError(
          "registered spacetime derivation source is recursive or unnamed");
    if (!visited.insert(name).second)
      return success();
    active.insert(name);
    llvm::scope_exit eraseActive([&] { active.erase(name); });
    closure.push_back(callable);
    Region *body = spacetimeCallableBody(callable);
    if (!body)
      return plan.emitOpError(
          "registered spacetime derivation reached a non-callable symbol");
    SmallVector<qlx::fabric::CallOp> calls;
    body->walk([&](qlx::fabric::CallOp call) { calls.push_back(call); });
    for (qlx::fabric::CallOp call : calls) {
      Operation *callee = moduleSymbols.lookup(call.getCallee());
      if (!callee || !spacetimeCallableBody(callee))
        return plan.emitOpError(
            "registered spacetime derivation has an unresolved callable "
            "closure");
      if (failed(visit(callee)))
        return failure();
    }
    return success();
  };
  if (failed(visit(source)))
    return failure();
  return closure;
}

static FailureOr<FactoryModelOp>
expectedSpacetimeFactoryModel(SpacetimePlanOp plan,
                              ArrayRef<Operation *> closure) {
  auto module = plan->getParentOfType<ModuleOp>();
  FactoryModelOp selected;
  bool failedModel = false;
  for (Operation *callable : closure) {
    Region *body = spacetimeCallableBody(callable);
    body->walk([&](qlx::fabric::ResourceRequestOp request) {
      FactoryModelOp match;
      for (FactoryModelOp candidate : module.getOps<FactoryModelOp>()) {
        if (candidate.getStreamAttr() != request.getStreamAttr() ||
            candidate.getResourceKindAttr().getValue() != request.getKind())
          continue;
        if (match) {
          plan.emitOpError(
              "resource request matches several physical factory models");
          failedModel = true;
          return WalkResult::interrupt();
        }
        match = candidate;
      }
      if (!match) {
        plan.emitOpError(
            "resource request has no selected physical factory model");
        failedModel = true;
        return WalkResult::interrupt();
      }
      if (selected && selected != match) {
        plan.emitOpError(
            "spacetime derivation uses several physical factory models");
        failedModel = true;
        return WalkResult::interrupt();
      }
      selected = match;
      return WalkResult::advance();
    });
    if (failedModel)
      return failure();
  }
  if (!selected) {
    plan.emitOpError(
        "spacetime derivation has no authenticated factory demand");
    return failure();
  }
  return selected;
}

static FailureOr<ResourceClassOp>
expectedAutoCCZRoutingClass(SpacetimePlanOp plan,
                            qlx::fabric::ProtocolOp source) {
  SmallVector<qlx::fabric::AllocOp, 6> allocations;
  for (auto allocation :
       source.getBody().front().getOps<qlx::fabric::AllocOp>())
    allocations.push_back(allocation);
  if (allocations.size() != 6)
    return plan.emitOpError(
        "AutoCCZ application must retain exactly six routing allocations");
  StringRef region = allocations.front().getRegion();
  if (region.empty() ||
      llvm::any_of(allocations, [&](qlx::fabric::AllocOp allocation) {
        return allocation.getRegion() != region;
      }))
    return plan.emitOpError(
        "AutoCCZ routing allocations must share one selected QEC region");

  auto module = plan->getParentOfType<ModuleOp>();
  SymbolTable moduleSymbols(module);
  auto architecture =
      moduleSymbols.lookup<ArchitectureOp>(plan.getArchitecture());
  if (!architecture)
    return failure();
  SmallVector<QECBindingOp, 2> bindings;
  for (QECBindingOp binding :
       architecture.getBody().front().getOps<QECBindingOp>())
    if (binding.getQecRegionAttr().getLeafReference().getValue() == region)
      bindings.push_back(binding);
  if (bindings.size() != 1)
    return plan.emitOpError(
        "AutoCCZ routing region must have one exact physical binding");
  SymbolTable architectureSymbols(architecture);
  ResourceClassOp selected;
  for (Attribute raw : bindings.front().getResources()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
    auto resource =
        reference
            ? architectureSymbols.lookup<ResourceClassOp>(reference.getValue())
            : ResourceClassOp{};
    auto granularity = resource
                           ? resource->getAttrOfType<StringAttr>("granularity")
                           : StringAttr{};
    if (!resource || !granularity || granularity.getValue() != "patch")
      continue;
    if (selected)
      return plan.emitOpError(
          "AutoCCZ routing region has several patch resource classes");
    selected = resource;
  }
  if (!selected || selected.getCount() < 6)
    return plan.emitOpError(
        "AutoCCZ routing requires a patch resource class with capacity six");
  return selected;
}

static FailureOr<SmallVector<qlx::fabric::ProtocolOp, 4>>
expectedAutoCCZApplications(SpacetimePlanOp plan,
                            ArrayRef<Operation *> closure) {
  SmallVector<qlx::fabric::ProtocolOp, 4> selected;
  for (Operation *callable : closure) {
    auto protocol = dyn_cast<qlx::fabric::ProtocolOp>(callable);
    if (!protocol ||
        !resource_provider::hasExactSurfaceAutoCCZApplicationStructure(
            protocol))
      continue;
    selected.push_back(protocol);
  }
  if (selected.empty())
    return plan.emitOpError(
        "surface spacelike callable has no exact AutoCCZ application");
  return selected;
}

static FailureOr<ResourceClassOp>
expectedAutoCCZRoutingClass(SpacetimePlanOp plan,
                            ArrayRef<qlx::fabric::ProtocolOp> applications) {
  ResourceClassOp selected;
  for (qlx::fabric::ProtocolOp application : applications) {
    auto routing = expectedAutoCCZRoutingClass(plan, application);
    if (failed(routing))
      return failure();
    if (selected && selected != *routing)
      return plan.emitOpError(
          "surface spacelike callable uses several routing classes");
    selected = *routing;
  }
  return selected;
}

static FailureOr<int64_t>
expectedSpacetimeCodeDistance(SpacetimePlanOp plan,
                              ArrayRef<Operation *> closure) {
  SymbolTable symbols(plan->getParentOfType<ModuleOp>());
  std::optional<int64_t> selected;
  auto inspect = [&](Type type, Operation *owner) -> LogicalResult {
    auto patch = dyn_cast<qlx::fabric::PatchType>(type);
    if (!patch)
      return success();
    auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
        symbols.lookup(patch.getCodeType().getValue()));
    if (!code || code.getDistance() <= 0)
      return owner->emitOpError(
          "surface spacelike callable uses a non-positive-distance code");
    if (selected && *selected != code.getDistance())
      return owner->emitOpError(
          "surface spacelike callable mixes code distances");
    selected = code.getDistance();
    return success();
  };
  for (Operation *callable : closure) {
    for (Type type : callable->getOperandTypes())
      if (failed(inspect(type, callable)))
        return failure();
    for (Type type : callable->getResultTypes())
      if (failed(inspect(type, callable)))
        return failure();
    Region *body = spacetimeCallableBody(callable);
    WalkResult result = body->walk([&](Operation *operation) {
      for (Type type : operation->getOperandTypes())
        if (failed(inspect(type, operation)))
          return WalkResult::interrupt();
      for (Type type : operation->getResultTypes())
        if (failed(inspect(type, operation)))
          return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
  }
  if (!selected)
    return plan.emitOpError(
        "surface spacelike callable contains no encoded patch");
  return *selected;
}

static std::optional<double> expectedSurfaceCycle(SpacetimePlanOp plan) {
  auto point = SymbolTable(plan->getParentOfType<ModuleOp>())
                   .lookup<OperatingPointOp>(plan.getOperatingPoint());
  auto timing = point ? point.getTimingAttr() : DictionaryAttr{};
  if (!timing)
    return std::nullopt;
  auto cycle = spacetimeNumericValue(timing.get("surface_cycle_ns"));
  if (!cycle)
    cycle = spacetimeNumericValue(timing.get("cycle_ns"));
  return cycle;
}

static LogicalResult verifySurfaceAutoCCZApplicationPlan(SpacetimePlanOp plan) {
  if (plan.getProvider() != kSpacetimeProvider ||
      plan.getProviderVersion() != kSpacetimeProviderVersion ||
      plan.getDerivation() != kSurfaceAutoCCZApplication ||
      plan.getDerivationVersion() != kSurfaceAutoCCZApplicationVersion ||
      plan.getEvidence() != kSurfaceAutoCCZApplicationEvidence ||
      plan.getRecurrenceResourceKindAttr() ||
      plan.getRecurrenceOutputEventAttr() || plan.getGeometryAttr())
    return plan.emitOpError(
        "surface AutoCCZ application has noncanonical provider/evidence "
        "identity");
  auto source = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      SymbolTable(plan->getParentOfType<ModuleOp>())
          .lookup(plan.getSourceProtocol()));
  if (!source ||
      !resource_provider::hasExactSurfaceAutoCCZApplicationStructure(source)) {
    auto diagnostic = plan.emitOpError(
        "surface AutoCCZ application source is not the exact selected "
        "three-owner protocol");
    if (source) {
      diagnostic << " (ops="
                 << std::distance(source.getBody().front().begin(),
                                  source.getBody().front().end())
                 << ": ";
      bool first = true;
      for (Operation &operation : source.getBody().front()) {
        if (!first)
          diagnostic << ", ";
        first = false;
        diagnostic << operation.getName();
      }
      diagnostic << ")";
    }
    return failure();
  }
  auto closure = spacetimeCallableClosure(plan, source);
  if (failed(closure))
    return failure();
  auto factory = expectedSpacetimeFactoryModel(plan, *closure);
  auto routing = expectedAutoCCZRoutingClass(plan, source);
  if (failed(factory) || failed(routing))
    return failure();

  auto point = SymbolTable(plan->getParentOfType<ModuleOp>())
                   .lookup<OperatingPointOp>(plan.getOperatingPoint());
  auto timing = point ? point.getTimingAttr() : DictionaryAttr{};
  auto reaction = timing ? spacetimeNumericValue(timing.get("reaction_time_ns"))
                         : std::optional<double>{};
  if (!reaction || !std::isfinite(*reaction) || *reaction <= 0.0)
    return plan.emitOpError(
        "surface AutoCCZ application requires positive reaction_time_ns");
  int64_t lanes = routing->getCount() / 6;
  if (lanes <= 0)
    return plan.emitOpError(
        "surface AutoCCZ application has no complete routing workspace");
  double expectedInterval =
      std::max(*reaction / static_cast<double>(lanes),
               factory->getOutputIntervalNs().convertToDouble());
  auto forwarding = plan.getForwardingLatencyNsAttr();
  auto interval = plan.getInitiationIntervalNsAttr();
  if (!forwarding || forwarding.getValueAsDouble() != *reaction || !interval ||
      interval.getValueAsDouble() != expectedInterval)
    return plan.emitOpError(
        "surface AutoCCZ pipeline timing must be derived from the selected "
        "reaction time, routing capacity, and factory cadence");

  SmallVector<SpacetimePhaseOp, 2> phases;
  for (auto phase : plan.getBody().front().getOps<SpacetimePhaseOp>())
    phases.push_back(phase);
  if (phases.size() != 1 || phases.front().getSymName() != "reaction" ||
      phases.front().getSteps() != 1 ||
      phases.front().getStepDurationNs().convertToDouble() != *reaction ||
      !phases.front().getAfter().empty() ||
      phases.front().getResourceClasses().size() != 1 ||
      phases.front().getFactoryModels().size() != 1)
    return plan.emitOpError(
        "surface AutoCCZ application must contain one canonical reaction "
        "phase");
  auto resource =
      dyn_cast<SymbolRefAttr>(phases.front().getResourceClasses()[0]);
  auto model =
      dyn_cast<FlatSymbolRefAttr>(phases.front().getFactoryModels()[0]);
  if (!resource ||
      resource.getRootReference().getValue() != plan.getArchitecture() ||
      resource.getNestedReferences().size() != 1 ||
      resource.getLeafReference().getValue() != routing->getSymName() ||
      !model || model.getValue() != factory->getSymName())
    return plan.emitOpError(
        "surface AutoCCZ reaction phase must claim the derived routing class "
        "and factory model");
  return success();
}

static LogicalResult verifySurfaceSpacelikeCallablePlan(SpacetimePlanOp plan) {
  if (plan.getProvider() != kSpacetimeProvider ||
      plan.getProviderVersion() != kSpacetimeProviderVersion ||
      plan.getDerivation() != kSurfaceSpacelikeCallable ||
      plan.getDerivationVersion() != kSurfaceSpacelikeCallableVersion ||
      plan.getEvidence() != kSurfaceSpacelikeCallableEvidence ||
      plan.getRecurrenceResourceKindAttr() ||
      plan.getRecurrenceOutputEventAttr() || plan.getGeometryAttr())
    return plan.emitOpError(
        "surface spacelike callable has noncanonical provider/evidence "
        "identity");
  auto source = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      SymbolTable(plan->getParentOfType<ModuleOp>())
          .lookup(plan.getSourceProtocol()));
  auto shape =
      source ? qlx::spacetime::surfaceSpacelikeShape(source) : std::nullopt;
  if (!shape)
    return plan.emitOpError(
        "surface spacelike callable source does not have the required exact "
        "one-layer boundary-preserving AutoCCZ P2 graph");
  auto closure = spacetimeCallableClosure(plan, source);
  if (failed(closure))
    return failure();
  auto applications = expectedAutoCCZApplications(plan, *closure);
  auto factory = expectedSpacetimeFactoryModel(plan, *closure);
  auto distance = expectedSpacetimeCodeDistance(plan, *closure);
  if (failed(applications) || failed(factory) || failed(distance))
    return failure();
  if (applications->size() != shape->reactionApplications)
    return plan.emitOpError(
               "surface spacelike structural reaction count differs from "
               "its exact AutoCCZ closure; expected ")
           << shape->reactionApplications << ", found " << applications->size();
  auto routing = expectedAutoCCZRoutingClass(plan, *applications);
  if (failed(routing))
    return failure();

  auto point = SymbolTable(plan->getParentOfType<ModuleOp>())
                   .lookup<OperatingPointOp>(plan.getOperatingPoint());
  auto timing = point ? point.getTimingAttr() : DictionaryAttr{};
  auto reaction = timing ? spacetimeNumericValue(timing.get("reaction_time_ns"))
                         : std::optional<double>{};
  auto cycle = expectedSurfaceCycle(plan);
  if (!reaction || !cycle || !std::isfinite(*reaction) ||
      !std::isfinite(*cycle) || *reaction <= 0.0 || *cycle <= 0.0)
    return plan.emitOpError(
        "surface spacelike callable requires positive reaction and surface "
        "cycle timing");
  int64_t lanes = routing->getCount() / 6;
  if (lanes <= 0)
    return plan.emitOpError(
        "surface spacelike callable has no complete AutoCCZ routing lane");
  if (shape->peakReactionWidth > static_cast<unsigned>(lanes))
    return plan.emitOpError(
        "surface spacelike callable exceeds selected routing concurrency");
  double access = static_cast<double>(shape->accessLayers) *
                  static_cast<double>(*distance) * *cycle / 2.0;
  double reactionDuration =
      static_cast<double>(shape->reactionDepth) * *reaction;
  double forwarding = std::max(reactionDuration, access);
  double interval =
      std::max({access, reactionDuration,
                static_cast<double>(shape->reactionApplications) *
                    factory->getOutputIntervalNs().convertToDouble()});
  auto actualForwarding = plan.getForwardingLatencyNsAttr();
  auto actualInterval = plan.getInitiationIntervalNsAttr();
  if (!actualForwarding || actualForwarding.getValueAsDouble() != forwarding ||
      !actualInterval || actualInterval.getValueAsDouble() != interval)
    return plan.emitOpError(
        "surface spacelike timing must be derived from exact P2 access "
        "layers, code distance, reaction latency, routing capacity, and "
        "factory cadence");

  SmallVector<SpacetimePhaseOp, 2> phases;
  for (auto phase : plan.getBody().front().getOps<SpacetimePhaseOp>())
    phases.push_back(phase);
  if (phases.size() != 1 || phases.front().getSymName() != "pipeline" ||
      phases.front().getSteps() != 1 ||
      phases.front().getStepDurationNs().convertToDouble() != forwarding ||
      !phases.front().getAfter().empty() ||
      phases.front().getResourceClasses().size() != 1 ||
      phases.front().getFactoryModels().size() != 1)
    return plan.emitOpError(
        "surface spacelike callable must contain one canonical pipeline "
        "phase");
  auto resource =
      dyn_cast<SymbolRefAttr>(phases.front().getResourceClasses()[0]);
  auto model =
      dyn_cast<FlatSymbolRefAttr>(phases.front().getFactoryModels()[0]);
  if (!resource ||
      resource.getRootReference().getValue() != plan.getArchitecture() ||
      resource.getNestedReferences().size() != 1 ||
      resource.getLeafReference().getValue() != routing->getSymName() ||
      !model || model.getValue() != factory->getSymName())
    return plan.emitOpError(
        "surface spacelike pipeline must claim the derived routing class and "
        "factory model");
  return success();
}

static LogicalResult
verifySurfaceSpacelikeCallableInvocation(SpacetimeCallOp call,
                                         SpacetimePlanOp plan) {
  SymbolTable moduleSymbols(plan->getParentOfType<ModuleOp>());
  auto sourceAttr = call.getSourceProtocolAttr();
  auto source =
      sourceAttr
          ? moduleSymbols.lookup<qlx::fabric::ProtocolOp>(sourceAttr.getValue())
          : qlx::fabric::ProtocolOp{};
  auto canonical = moduleSymbols.lookup<qlx::fabric::ProtocolOp>(
      plan.getSourceProtocolAttr().getValue());
  auto shape =
      source ? qlx::spacetime::surfaceSpacelikeShape(source) : std::nullopt;
  auto canonicalShape = canonical
                            ? qlx::spacetime::surfaceSpacelikeShape(canonical)
                            : std::nullopt;
  if (!shape || !canonicalShape ||
      shape->boundaryOwners != canonicalShape->boundaryOwners ||
      shape->accessLayers != canonicalShape->accessLayers ||
      shape->reactionDepth != canonicalShape->reactionDepth ||
      shape->reactionApplications != canonicalShape->reactionApplications ||
      shape->peakReactionWidth != canonicalShape->peakReactionWidth) {
    auto diagnostic = call.emitOpError(
        "reused surface spacelike plan requires the same exact P2 owner-graph "
        "shape as its canonical source");
    if (shape)
      diagnostic << "; actual=(owners=" << shape->boundaryOwners
                 << ", access_layers=" << shape->accessLayers
                 << ", reaction_depth=" << shape->reactionDepth
                 << ", reaction_applications=" << shape->reactionApplications
                 << ", peak_width=" << shape->peakReactionWidth << ")";
    else
      diagnostic << "; actual=unsupported";
    if (canonicalShape)
      diagnostic << ", canonical=(owners=" << canonicalShape->boundaryOwners
                 << ", access_layers=" << canonicalShape->accessLayers
                 << ", reaction_depth=" << canonicalShape->reactionDepth
                 << ", reaction_applications="
                 << canonicalShape->reactionApplications
                 << ", peak_width=" << canonicalShape->peakReactionWidth << ")";
    else
      diagnostic << ", canonical=unsupported";
    return failure();
  }
  auto closure = spacetimeCallableClosure(plan, source);
  if (failed(closure))
    return failure();
  auto applications = expectedAutoCCZApplications(plan, *closure);
  auto factory = expectedSpacetimeFactoryModel(plan, *closure);
  auto distance = expectedSpacetimeCodeDistance(plan, *closure);
  if (failed(applications) || failed(factory) || failed(distance))
    return failure();
  if (applications->size() != shape->reactionApplications)
    return call.emitOpError(
               "reused surface spacelike structural reaction count differs "
               "from its exact AutoCCZ closure; expected ")
           << shape->reactionApplications << ", found " << applications->size();
  auto routing = expectedAutoCCZRoutingClass(plan, *applications);
  if (failed(routing))
    return failure();

  auto point = SymbolTable(plan->getParentOfType<ModuleOp>())
                   .lookup<OperatingPointOp>(plan.getOperatingPoint());
  auto timing = point ? point.getTimingAttr() : DictionaryAttr{};
  auto reaction = timing ? spacetimeNumericValue(timing.get("reaction_time_ns"))
                         : std::optional<double>{};
  auto cycle = expectedSurfaceCycle(plan);
  if (!reaction || !cycle || !std::isfinite(*reaction) ||
      !std::isfinite(*cycle) || *reaction <= 0.0 || *cycle <= 0.0)
    return call.emitOpError(
        "reused surface spacelike plan has incomplete selected timing");
  int64_t lanes = routing->getCount() / 6;
  if (lanes <= 0)
    return call.emitOpError(
        "reused surface spacelike plan has no routing capacity");
  if (shape->peakReactionWidth > static_cast<unsigned>(lanes))
    return call.emitOpError(
        "reused surface spacelike source exceeds routing concurrency");
  double access = static_cast<double>(shape->accessLayers) *
                  static_cast<double>(*distance) * *cycle / 2.0;
  double reactionDuration =
      static_cast<double>(shape->reactionDepth) * *reaction;
  double forwarding = std::max(reactionDuration, access);
  double interval =
      std::max({access, reactionDuration,
                static_cast<double>(shape->reactionApplications) *
                    factory->getOutputIntervalNs().convertToDouble()});
  if (!plan.getForwardingLatencyNsAttr() ||
      plan.getForwardingLatencyNsAttr().getValueAsDouble() != forwarding ||
      !plan.getInitiationIntervalNsAttr() ||
      plan.getInitiationIntervalNsAttr().getValueAsDouble() != interval)
    return call.emitOpError(
        "reused surface spacelike plan timing differs from the actual P2 "
        "source and selected device");
  auto phases = plan.getBody().front().getOps<SpacetimePhaseOp>();
  if (phases.empty())
    return call.emitOpError(
        "reused surface spacelike plan has no physical phase");
  auto phase = *phases.begin();
  if (phase.getResourceClasses().size() != 1 ||
      phase.getFactoryModels().size() != 1)
    return call.emitOpError(
        "reused surface spacelike plan has noncanonical physical claims");
  auto resource = dyn_cast<SymbolRefAttr>(phase.getResourceClasses()[0]);
  auto model = dyn_cast<FlatSymbolRefAttr>(phase.getFactoryModels()[0]);
  if (!resource ||
      resource.getLeafReference().getValue() != routing->getSymName() ||
      !model || model.getValue() != factory->getSymName())
    return call.emitOpError(
        "reused surface spacelike plan resources differ from the actual P2 "
        "source and selected device");
  return success();
}

static FlatSymbolRefAttr spacetimeProducedResourceKind(Operation *callable) {
  if (!callable)
    return {};
  FunctionType type;
  if (auto protocol = dyn_cast<qlx::fabric::ProtocolOp>(callable))
    type = protocol.getFunctionType();
  else if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callable))
    type = gadget.getFunctionType();
  if (!type || type.getNumResults() != 1)
    return {};
  auto resource = dyn_cast<qlx::fabric::ResourceStateType>(type.getResult(0));
  return resource ? dyn_cast<FlatSymbolRefAttr>(resource.getKind())
                  : FlatSymbolRefAttr{};
}

static bool spacetimeHasPackedResource(qlx::fabric::ProtocolOp protocol,
                                       StringRef kind, size_t payloads) {
  size_t matches = 0;
  for (auto pack :
       protocol.getBody().front().getOps<qlx::fabric::PackResourceOp>())
    if (pack.getResourceKindAttr().getValue() == kind &&
        pack.getPayloads().size() == payloads)
      ++matches;
  return matches == 1;
}

static LogicalResult
verifySurfaceFactorySource(SpacetimePlanOp plan, qlx::fabric::ProtocolOp source,
                           qlx::fabric::ProtocolOp &level1,
                           qlx::fabric::ProtocolOp &level2) {
  auto produced = spacetimeProducedResourceKind(source);
  if (!produced || produced.getValue() != kAutoCCZState ||
      !spacetimeHasPackedResource(source, kAutoCCZState, 9))
    return plan.emitOpError(
        "surface factory recurrence source must produce one typed "
        "nine-patch AutoCCZ resource");

  SymbolTable moduleSymbols(plan->getParentOfType<ModuleOp>());
  size_t cczCalls = 0;
  for (auto call : source.getBody().front().getOps<qlx::fabric::CallOp>()) {
    Operation *callee = moduleSymbols.lookup(call.getCallee());
    auto kind = spacetimeProducedResourceKind(callee);
    if (!kind || kind.getValue() != kCCZState)
      continue;
    level2 = dyn_cast_or_null<qlx::fabric::ProtocolOp>(callee);
    ++cczCalls;
  }
  if (cczCalls != 1 || !level2 ||
      !spacetimeHasPackedResource(level2, kCCZState, 3))
    return plan.emitOpError(
        "surface factory recurrence source must call one typed "
        "three-patch CCZ producer");

  size_t level1Calls = 0;
  for (auto call : level2.getBody().front().getOps<qlx::fabric::CallOp>()) {
    Operation *callee = moduleSymbols.lookup(call.getCallee());
    auto kind = spacetimeProducedResourceKind(callee);
    if (!kind || kind.getValue() != kTState)
      continue;
    auto candidate = dyn_cast_or_null<qlx::fabric::ProtocolOp>(callee);
    if (!candidate || !spacetimeHasPackedResource(candidate, kTState, 1) ||
        !resource_provider::hasExactFifteenToOneStructure(candidate))
      return plan.emitOpError(
          "surface factory recurrence level-1 callee must implement the "
          "canonical fifteen-input distillation structure");
    size_t rawRequests = 0;
    candidate.getBody().walk([&](qlx::fabric::ResourceRequestOp request) {
      if (request.getKind() == kRawTState)
        ++rawRequests;
    });
    if (rawRequests != 15)
      return plan.emitOpError(
          "surface factory recurrence level-1 callee must consume exactly "
          "fifteen raw T states");
    if (level1 && level1 != candidate)
      return plan.emitOpError(
          "surface factory recurrence requires one canonical 15-to-1 "
          "protocol definition");
    level1 = candidate;
    ++level1Calls;
  }
  if (level1Calls != 8 || !level1)
    return plan.emitOpError(
        "surface factory recurrence CCZ producer must call exactly eight "
        "15-to-1 lanes");
  if (!resource_provider::hasExactGidneyFowlerCCZStructure(level2))
    return plan.emitOpError(
        "surface factory recurrence CCZ producer must implement the canonical "
        "Figure-5 distillation structure");
  if (!resource_provider::hasExactGidneyFowlerAutoCCZStructure(source))
    return plan.emitOpError(
        "surface factory recurrence source must be the exact closed "
        "Gidney--Fowler nine-patch AutoCCZ producer");

  return success();
}

static FailureOr<VerifiedSurfaceFactoryStageShape>
expectedSurfaceFactoryStageShape(SpacetimePlanOp plan,
                                 qlx::fabric::ProtocolOp source,
                                 qlx::fabric::ProtocolOp level1,
                                 qlx::fabric::ProtocolOp level2) {
  int64_t rotations = 0;
  for (auto ignored : level1.getBody()
                          .front()
                          .getOps<qlx::fabric::ResourceRotateProductOp>()) {
    (void)ignored;
    ++rotations;
  }
  int64_t checks = 0;
  for (auto ignored :
       level2.getBody().front().getOps<qlx::fabric::MeasureProductOp>()) {
    (void)ignored;
    ++checks;
  }
  int64_t ringEdges = 0;
  for (auto call : source.getBody().front().getOps<qlx::fabric::CallOp>())
    if (resource_provider::isLogicalCZCall(call))
      ++ringEdges;
  if (rotations <= 0 || checks <= 0 || ringEdges < 3) {
    plan.emitOpError(
        "surface factory recurrence has no complete authenticated physical "
        "stage shape");
    return failure();
  }
  return VerifiedSurfaceFactoryStageShape{
      /*level1QuarterLayers=*/2 * rotations + 1,
      /*level2Layers=*/checks + 1,
      /*autoCCZLayers=*/ringEdges % 2 == 0 ? 2 : 3};
}

static FailureOr<double> expectedSurfaceFactoryTiming(SpacetimePlanOp plan,
                                                      DictionaryAttr timing,
                                                      StringRef action,
                                                      int64_t distance) {
  if (!timing || action.empty() || distance <= 0) {
    plan.emitOpError(
        "surface factory recurrence requires an explicit selected-device "
        "timing profile");
    return failure();
  }
  std::string qualified =
      (action + "_d" + std::to_string(distance) + "_ns").str();
  Attribute raw = timing.get(qualified);
  if (!raw)
    raw = timing.get((action + "_ns").str());
  auto value = spacetimeNumericValue(raw);
  if (!value || !std::isfinite(*value) || *value <= 0.0) {
    plan.emitOpError("surface factory recurrence requires finite positive "
                     "selected-device timing for '")
        << action << "' at code distance " << distance;
    return failure();
  }
  return *value;
}

static FailureOr<VerifiedSurfaceFactoryRecurrenceLayout>
expectedSurfaceFactoryRecurrenceLayout(SpacetimePlanOp plan,
                                       ArrayRef<Operation *> closure) {
  auto module = plan->getParentOfType<ModuleOp>();
  SymbolTable moduleSymbols(module);
  auto architecture =
      moduleSymbols.lookup<ArchitectureOp>(plan.getArchitecture());
  if (!architecture)
    return failure();
  SymbolTable architectureSymbols(architecture);
  std::map<int64_t, std::string> classByDistance;
  std::map<int64_t, int64_t> capacityByDistance;
  bool invalid = false;
  for (Operation *callable : closure) {
    Region *body = spacetimeCallableBody(callable);
    body->walk([&](qlx::fabric::AllocOp allocation) {
      auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
          moduleSymbols.lookup(allocation.getCodeAttr().getValue()));
      SmallVector<QECBindingOp, 2> bindings;
      for (QECBindingOp binding :
           architecture.getBody().front().getOps<QECBindingOp>())
        if (binding.getQecRegionAttr().getLeafReference().getValue() ==
            allocation.getRegion())
          bindings.push_back(binding);
      if (!code || code.getDistance() <= 0 || bindings.size() != 1) {
        plan.emitOpError(
            "surface factory recurrence allocation lacks one exact "
            "positive-distance physical binding");
        invalid = true;
        return WalkResult::interrupt();
      }
      ResourceClassOp selected;
      for (Attribute raw : bindings.front().getResources()) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
        auto resource = reference ? architectureSymbols.lookup<ResourceClassOp>(
                                        reference.getValue())
                                  : ResourceClassOp{};
        auto granularity =
            resource ? resource->getAttrOfType<StringAttr>("granularity")
                     : StringAttr{};
        if (!resource || !granularity || granularity.getValue() != "patch")
          continue;
        if (selected) {
          plan.emitOpError(
              "surface factory recurrence QEC region has several patch "
              "resource classes");
          invalid = true;
          return WalkResult::interrupt();
        }
        selected = resource;
      }
      if (!selected) {
        plan.emitOpError(
            "surface factory recurrence QEC region has no patch resource "
            "class");
        invalid = true;
        return WalkResult::interrupt();
      }
      auto [entry, inserted] = classByDistance.try_emplace(
          code.getDistance(), selected.getSymName().str());
      if (!inserted && entry->second != selected.getSymName()) {
        plan.emitOpError(
            "surface factory recurrence maps one distance to several patch "
            "resource classes");
        invalid = true;
        return WalkResult::interrupt();
      }
      auto qecReference = bindings.front().getQecRegionAttr();
      auto qecMachine = dyn_cast_or_null<qlx::fabric::DeviceOp>(
          moduleSymbols.lookup(qecReference.getRootReference().getValue()));
      auto region = qecMachine
                        ? SymbolTable(qecMachine)
                              .lookup<qlx::fabric::RegionOp>(
                                  qecReference.getLeafReference().getValue())
                        : qlx::fabric::RegionOp{};
      auto capacity = region ? region.getBlockCapacity() : std::nullopt;
      if (!region || !capacity || *capacity <= 0) {
        plan.emitOpError(
            "surface factory recurrence requires positive QEC-region block "
            "capacities for every selected code distance");
        invalid = true;
        return WalkResult::interrupt();
      }
      auto [capacityEntry, capacityInserted] =
          capacityByDistance.try_emplace(code.getDistance(), *capacity);
      if (!capacityInserted && capacityEntry->second != *capacity) {
        plan.emitOpError(
            "surface factory recurrence has inconsistent QEC-region capacity "
            "for one code distance");
        invalid = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (invalid)
      return failure();
  }

  if (classByDistance.size() != 2)
    return plan.emitOpError(
        "surface factory recurrence requires exactly two selected code "
        "distances and physical patch layout");
  auto first = classByDistance.begin();
  auto second = std::next(first);
  auto level1 = architectureSymbols.lookup<ResourceClassOp>(first->second);
  auto level2 = architectureSymbols.lookup<ResourceClassOp>(second->second);
  if (!level1 || !level2 || level1.getCount() <= 0 || level2.getCount() <= 0)
    return plan.emitOpError(
        "surface factory recurrence physical patch bindings must have positive "
        "capacity");
  constexpr int64_t level1PatchesPerLane = 5;
  constexpr int64_t cczPatches = 11;
  constexpr int64_t autoCCZPatches = 9;
  int64_t level1Capacity = capacityByDistance[first->first];
  int64_t level2Capacity = capacityByDistance[second->first];
  if (level1Capacity % level1PatchesPerLane != 0 ||
      level2Capacity <= cczPatches ||
      (level2Capacity - cczPatches) % autoCCZPatches != 0)
    return plan.emitOpError(
        "surface factory recurrence resource capacities do not decompose "
        "into five-patch 15-to-1 lanes, one eleven-patch CCZ stage, and "
        "nine-patch AutoCCZ workspaces");
  int64_t level1Lanes = level1Capacity / level1PatchesPerLane;
  int64_t fixupBoxes = (level2Capacity - cczPatches) / autoCCZPatches;
  if (level1Lanes != 6)
    return plan.emitOpError(
        "the registered Gidney--Fowler layout requires exactly six "
        "five-patch level-1 lanes");
  if (fixupBoxes != 2)
    return plan.emitOpError(
        "the registered AutoCCZ layout requires exactly two nine-patch "
        "fixup workspaces");
  int64_t factoryPatches = level1.getCount();
  auto distanceWidth = checkedSum({second->first, 1});
  auto patchPhysicalUnits =
      distanceWidth ? checkedProduct({2, *distanceWidth, *distanceWidth})
                    : std::nullopt;
  auto occupiedPatches = checkedSum({level1Capacity, level2Capacity});
  auto footprint = level1.getPhysicalUnitsAttr();
  if (first->second != second->second || !patchPhysicalUnits ||
      !occupiedPatches || !footprint ||
      footprint.getInt() != *patchPhysicalUnits ||
      *occupiedPatches > factoryPatches)
    return plan.emitOpError(
        "the registered Gidney--Fowler layout requires one shared "
        "distance-2 footprint patch class large enough for its exact "
        "level-1 and level-2 regions");
  return VerifiedSurfaceFactoryRecurrenceLayout{
      first->first,
      second->first,
      level1Lanes,
      /*tStatesPerCCZ=*/8,
      fixupBoxes,
      level1PatchesPerLane,
      cczPatches,
      autoCCZPatches,
      factoryPatches,
      /*level1ResourceOffset=*/0,
      /*level2ResourceOffset=*/level1Capacity,
      first->second,
      second->second,
  };
}

static LogicalResult verifySurfaceFactoryRecurrencePlan(SpacetimePlanOp plan) {
  if (plan.getProvider() != kSpacetimeProvider ||
      plan.getProviderVersion() != kSpacetimeProviderVersion ||
      plan.getDerivation() != kSurfaceFactoryRecurrence ||
      plan.getDerivationVersion() != kSurfaceFactoryRecurrenceVersion ||
      plan.getEvidence() != kSurfaceFactoryRecurrenceEvidence)
    return plan.emitOpError(
        "surface factory recurrence has a noncanonical provider/evidence "
        "tuple");
  auto module = plan->getParentOfType<ModuleOp>();
  SymbolTable moduleSymbols(module);
  auto source = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      moduleSymbols.lookup(plan.getSourceProtocol()));
  qlx::fabric::ProtocolOp level1;
  qlx::fabric::ProtocolOp level2;
  if (!source ||
      failed(verifySurfaceFactorySource(plan, source, level1, level2)))
    return failure();
  auto closure = spacetimeCallableClosure(plan, source);
  if (failed(closure))
    return failure();
  auto layout = expectedSurfaceFactoryRecurrenceLayout(plan, *closure);
  auto stageShape =
      expectedSurfaceFactoryStageShape(plan, source, level1, level2);
  auto rawModel = expectedSpacetimeFactoryModel(plan, *closure);
  if (failed(layout) || failed(stageShape) || failed(rawModel))
    return failure();
  if ((*rawModel).getResourceKindAttr().getValue() != kRawTState)
    return plan.emitOpError(
        "surface factory recurrence requires one authenticated raw-T "
        "physical supply");

  SmallVector<GraphOp, 1> graphs;
  for (GraphOp graph : module.getOps<GraphOp>())
    if (graph.getSourceProtocolAttr() &&
        graph.getSourceProtocolAttr() == plan.getSourceProtocolAttr())
      graphs.push_back(graph);
  if (graphs.size() != 1)
    return plan.emitOpError(
        "surface factory recurrence requires exactly one source physical "
        "graph");
  GraphOp graph = graphs.front();
  if (graph.getArchitectureAttr() != plan.getArchitectureAttr() ||
      graph.getOperatingPointAttr() != plan.getOperatingPointAttr())
    return plan.emitOpError(
        "surface factory recurrence must use the source graph architecture "
        "and operating point");
  SmallVector<PackResourceOp, 1> outputs;
  graph.walk([&](PackResourceOp pack) {
    if (pack.getResourceKindAttr().getValue() == kAutoCCZState)
      outputs.push_back(pack);
  });
  if (outputs.size() != 1 || !outputs.front().getEventIdAttr())
    return plan.emitOpError(
        "surface factory recurrence requires exactly one physical AutoCCZ "
        "output event");
  if (!plan.getRecurrenceResourceKindAttr() ||
      plan.getRecurrenceResourceKindAttr().getValue() != kAutoCCZState ||
      !plan.getRecurrenceOutputEventAttr() ||
      plan.getRecurrenceOutputEventAttr() != outputs.front().getEventIdAttr())
    return plan.emitOpError(
        "surface factory recurrence output identity does not match the "
        "materialized physical output");

  auto geometry = plan.getGeometryAttr();
  SmallVector<int64_t, 9> expectedGeometry = layout->geometry();
  if (!geometry || geometry.asArrayRef() != ArrayRef(expectedGeometry))
    return plan.emitOpError(
        "surface factory recurrence geometry does not equal the independently "
        "derived P2/device layout");
  auto point = moduleSymbols.lookup<OperatingPointOp>(plan.getOperatingPoint());
  auto timing = point ? point.getTimingAttr() : DictionaryAttr{};
  auto cycle = timing ? spacetimeNumericValue(timing.get("surface_cycle_ns"))
                      : std::nullopt;
  if (!cycle && timing)
    cycle = spacetimeNumericValue(timing.get("cycle_ns"));
  if (!cycle || !std::isfinite(*cycle) || *cycle <= 0.0)
    return plan.emitOpError(
        "surface factory recurrence requires finite positive surface-cycle "
        "timing");
  auto level1RotationTiming = expectedSurfaceFactoryTiming(
      plan, timing, "resource_rpp", layout->level1Distance);
  auto level2LayerTiming = expectedSurfaceFactoryTiming(
      plan, timing, "resource_rpp", layout->level2Distance);
  auto autoCCZLayerTiming =
      expectedSurfaceFactoryTiming(plan, timing, "cz", layout->level2Distance);
  if (failed(level1RotationTiming) || failed(level2LayerTiming) ||
      failed(autoCCZLayerTiming))
    return failure();
  double rawInterval = (*rawModel).getOutputIntervalNs().convertToDouble();
  double rawStartup = (*rawModel).getStartupNs().convertToDouble();
  if (!std::isfinite(rawInterval) || rawInterval <= 0.0 ||
      !std::isfinite(rawStartup) || rawStartup < 0.0)
    return plan.emitOpError(
        "surface factory recurrence derives no finite positive input timing");

  SmallVector<SpacetimeEventOp, 48> events;
  for (SpacetimeEventOp event :
       plan.getBody().front().getOps<SpacetimeEventOp>())
    events.push_back(event);
  if (events.size() != 48)
    return plan.emitOpError(
        "surface factory recurrence requires the canonical 48-event, "
        "three-output physical schedule");

  auto equalTime = [](double left, double right) {
    double scale = std::max({1.0, std::abs(left), std::abs(right)});
    return std::abs(left - right) <= scale * 1.0e-12;
  };
  size_t cursor = 0;
  auto check =
      [&](StringRef name, StringRef kind, int64_t iteration, double start,
          double duration, StringRef resourceClass, int64_t resourceOffset,
          int64_t resourceCount, bool usesRawFactory, int64_t factoryUnits,
          ArrayRef<std::string> after, StringRef outputKind) -> LogicalResult {
    if (cursor >= events.size())
      return plan.emitOpError("surface factory recurrence is missing event @")
             << name;
    SpacetimeEventOp event = events[cursor++];
    auto resource = event.getResourceClassAttr();
    auto model = event.getFactoryModelAttr();
    auto output = event.getOutputResourceKindAttr();
    if (event.getSymName() != name || event.getKind() != kind ||
        event.getIteration() != iteration ||
        !equalTime(event.getStartNs().convertToDouble(), start) ||
        !equalTime(event.getDurationNs().convertToDouble(), duration) ||
        !resource || resource.getLeafReference().getValue() != resourceClass ||
        !event.getResourceOffsetAttr() ||
        event.getResourceOffsetAttr().getInt() != resourceOffset ||
        !event.getResourceCountAttr() ||
        event.getResourceCountAttr().getInt() != resourceCount ||
        static_cast<bool>(model) != usesRawFactory ||
        (usesRawFactory &&
         (model.getValue() != (*rawModel).getSymName() ||
          !event.getFactoryUnitsAttr() ||
          event.getFactoryUnitsAttr().getInt() != factoryUnits)) ||
        static_cast<bool>(output) != !outputKind.empty() ||
        (output && output.getValue() != outputKind) ||
        event.getAfter().size() != after.size())
      return event.emitOpError(
          "does not match the independently reconstructed published "
          "surface-factory event");
    for (auto [actual, expected] : llvm::zip(event.getAfter(), after))
      if (cast<FlatSymbolRefAttr>(actual).getValue() != expected)
        return event.emitOpError(
            "has a noncanonical physical schedule dependency");
    return success();
  };

  constexpr int64_t outputsToProve = 3;
  constexpr int64_t tStatesPerOutput = 8;
  double level1Duration = static_cast<double>(stageShape->level1QuarterLayers) *
                          *level1RotationTiming / 4.0;
  double level2LayerDuration = *level2LayerTiming;
  double autoCCZLayerDuration = *autoCCZLayerTiming;
  SmallVector<double, 6> laneReady(layout->level1Lanes, rawStartup);

  SmallVector<std::string, 24> tEvents;
  SmallVector<double, 24> tReady;
  int64_t rawConsumed = 0;
  for (int64_t index = 0; index < outputsToProve * tStatesPerOutput; ++index) {
    int64_t lane = index % layout->level1Lanes;
    double start = laneReady[lane];
    double finish = start + level1Duration;
    std::string name = llvm::formatv("level1_t{0:00}", index).str();
    if (failed(check(
            name, "level1_15to1", index / tStatesPerOutput, start,
            level1Duration, layout->level1ResourceClass,
            layout->level1ResourceOffset + lane * layout->level1PatchesPerLane,
            layout->level1PatchesPerLane,
            /*usesRawFactory=*/true, /*factoryUnits=*/15, {}, kTState)))
      return failure();
    rawConsumed += 15;
    long double elapsed =
        std::max(0.0L, static_cast<long double>(start) - rawStartup);
    long double producedAfterStartup =
        std::floor(elapsed / static_cast<long double>(rawInterval) + 1.0e-9L);
    int64_t lanes = (*rawModel).getLaneCount();
    bool supplied = rawConsumed <= lanes;
    if (!supplied) {
      long double requiredAfterStartup =
          static_cast<long double>(rawConsumed - lanes);
      supplied = !std::isfinite(producedAfterStartup) ||
                 producedAfterStartup >= requiredAfterStartup;
    }
    if (!supplied)
      return events[index].emitOpError(
          "starts before the selected raw-T bank can supply its inputs");
    laneReady[lane] = finish;
    tEvents.push_back(name);
    tReady.push_back(finish);
  }

  double coreReady = 0.0;
  SmallVector<double, 2> fixupReady(layout->fixupBoxes, 0.0);
  SmallVector<std::string, 2> fixupTail(layout->fixupBoxes);
  SmallVector<double, 3> outputTimes;
  for (int64_t iteration = 0; iteration < outputsToProve; ++iteration) {
    SmallVector<std::string> dependencies;
    double inputsReady = 0.0;
    for (int64_t offset = 0; offset < tStatesPerOutput; ++offset) {
      int64_t index = iteration * tStatesPerOutput + offset;
      dependencies.push_back(tEvents[index]);
      inputsReady = std::max(inputsReady, tReady[index]);
    }
    double layerStart = std::max(coreReady, inputsReady);
    std::string previous;
    for (int64_t layer = 0; layer < stageShape->level2Layers; ++layer) {
      std::string name =
          llvm::formatv("ccz_{0}_layer_{1}", iteration, layer).str();
      SmallVector<std::string> after =
          layer == 0 ? dependencies : SmallVector<std::string>{previous};
      if (failed(check(name, "level2_ccz_layer", iteration, layerStart,
                       level2LayerDuration, layout->level2ResourceClass,
                       layout->level2ResourceOffset, layout->cczPatches,
                       /*usesRawFactory=*/false, 0, after,
                       layer + 1 == stageShape->level2Layers
                           ? StringRef(kCCZState)
                           : StringRef{})))
        return failure();
      previous = name;
      layerStart += level2LayerDuration;
    }
    coreReady = layerStart;

    int64_t workspace = iteration % layout->fixupBoxes;
    layerStart = std::max(coreReady, fixupReady[workspace]);
    previous.clear();
    for (int64_t layer = 0; layer < stageShape->autoCCZLayers; ++layer) {
      std::string name =
          llvm::formatv("autoccz_{0}_layer_{1}", iteration, layer).str();
      SmallVector<std::string> after;
      if (layer == 0) {
        after.push_back(llvm::formatv("ccz_{0}_layer_{1}", iteration,
                                      stageShape->level2Layers - 1)
                            .str());
        if (!fixupTail[workspace].empty())
          after.push_back(fixupTail[workspace]);
      } else {
        after.push_back(previous);
      }
      StringRef outputKind = layer + 1 == stageShape->autoCCZLayers
                                 ? StringRef(kAutoCCZState)
                                 : StringRef{};
      if (failed(check(name, "autoccz_ring_layer", iteration, layerStart,
                       autoCCZLayerDuration, layout->level2ResourceClass,
                       layout->level2ResourceOffset + layout->cczPatches +
                           workspace * layout->autoCCZPatches,
                       layout->autoCCZPatches,
                       /*usesRawFactory=*/false, 0, after, outputKind)))
        return failure();
      previous = name;
      layerStart += autoCCZLayerDuration;
    }
    fixupReady[workspace] = layerStart;
    fixupTail[workspace] = previous;
    outputTimes.push_back(layerStart);
  }
  if (cursor != events.size() || !equalTime(outputTimes[1] - outputTimes[0],
                                            outputTimes[2] - outputTimes[1]))
    return plan.emitOpError(
        "surface factory recurrence does not prove a stable output cadence");
  return success();
}

static bool canonicalDigest(StringRef value, bool prefixed) {
  if (prefixed && !value.consume_front("sha256:"))
    return false;
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

static FailureOr<std::string>
selectedProtocolDigest(qlx::fabric::ProtocolOp source) {
  auto module = source->getParentOfType<ModuleOp>();
  if (!module)
    return failure();
  SymbolTable symbols(module);
  SmallVector<Operation *, 8> pending{source.getOperation()};
  llvm::SmallPtrSet<Operation *, 8> seen;
  SmallVector<std::pair<StringRef, Operation *>, 8> closure;
  bool unresolved = false;
  while (!pending.empty()) {
    Operation *current = pending.pop_back_val();
    if (!seen.insert(current).second)
      continue;
    auto symbol = SymbolTable::getSymbolName(current);
    if (!symbol)
      return failure();
    closure.emplace_back(symbol.getValue(), current);
    current->walk([&](qlx::fabric::CallOp call) {
      Operation *callee = symbols.lookup(call.getCallee());
      if (!callee) {
        unresolved = true;
        return;
      }
      pending.push_back(callee);
    });
  }
  if (unresolved)
    return failure();
  llvm::sort(closure, [](const auto &left, const auto &right) {
    return left.first < right.first;
  });

  std::string payload = "qlx.selected-protocol-closure/v1\n";
  llvm::raw_string_ostream stream(payload);
  auto appendField = [&](StringRef value) {
    stream << value.size() << ':' << value << '\n';
  };
  for (const auto &[symbol, current] : closure) {
    Operation *clone = current->clone();
    if (isa<qlx::fabric::ProtocolOp>(clone)) {
      clone->removeAttr("component_source_sha256");
      clone->removeAttr("component_objective_sha256");
      clone->removeAttr("component_boundary_sha256");
    }
    std::string text;
    llvm::raw_string_ostream operationStream(text);
    clone->print(operationStream);
    operationStream.flush();
    clone->destroy();
    appendField(symbol);
    appendField(current->getName().getStringRef());
    appendField(text);
  }
  stream.flush();
  llvm::SHA256 digest;
  digest.update(payload);
  auto bytes = digest.final();
  std::string result = "sha256:";
  static constexpr char hex[] = "0123456789abcdef";
  for (uint8_t byte : bytes) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0xf]);
  }
  return result;
}

static std::string sha256Hex(StringRef text) {
  llvm::SHA256 digest;
  digest.update(text);
  auto bytes = digest.final();
  std::string result;
  static constexpr char hex[] = "0123456789abcdef";
  for (uint8_t byte : bytes) {
    result.push_back(hex[byte >> 4]);
    result.push_back(hex[byte & 0xf]);
  }
  return result;
}

static LogicalResult verifyComponentPlan(SpacetimePlanOp plan) {
  if (plan.getProvider() != "qlx.component-model" ||
      plan.getProviderVersion() != "1" || plan.getDerivationVersion() != 1 ||
      (plan.getDerivation() != "characterized" &&
       plan.getDerivation() != "asserted") ||
      plan.getRecurrenceResourceKindAttr() ||
      plan.getRecurrenceOutputEventAttr() || plan.getGeometryAttr())
    return plan.emitOpError(
        "component plan has noncanonical provider/derivation identity");
  auto source = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      SymbolTable(plan->getParentOfType<ModuleOp>())
          .lookup(plan.getSourceProtocol()));
  if (!source)
    return plan.emitOpError(
        "component plan source must resolve to fabric.protocol");
  auto sourceDigest = plan.getSourceProtocolSha256Attr();
  auto objectiveDigest = plan.getSourceObjectiveSha256Attr();
  auto boundaryDigest = plan.getSourceBoundarySha256Attr();
  auto architectureDigest = plan.getSourceArchitectureSha256Attr();
  auto modelDigest = plan.getModelSha256Attr();
  if (!sourceDigest || !objectiveDigest || !boundaryDigest ||
      !architectureDigest || !modelDigest ||
      !canonicalDigest(sourceDigest.getValue(), true) ||
      !canonicalDigest(objectiveDigest.getValue(), true) ||
      !canonicalDigest(boundaryDigest.getValue(), true) ||
      !canonicalDigest(architectureDigest.getValue(), true) ||
      !canonicalDigest(modelDigest.getValue(), false))
    return plan.emitOpError(
        "component plan requires complete canonical source/model commitments");
  if (source->getAttrOfType<StringAttr>("component_source_sha256") !=
          sourceDigest ||
      source->getAttrOfType<StringAttr>("component_objective_sha256") !=
          objectiveDigest ||
      source->getAttrOfType<StringAttr>("component_boundary_sha256") !=
          boundaryDigest)
    return plan.emitOpError(
        "component plan source/objective/boundary commitment differs from "
        "its exact retained protocol");
  bool characterized = plan.getDerivation() == "characterized";
  bool hasBuild = static_cast<bool>(plan.getSourceBuildSha256Attr());
  bool hasSchedule = static_cast<bool>(plan.getSourceScheduleSha256Attr());
  bool hasSelected =
      static_cast<bool>(plan.getSourceSelectedProtocolSha256Attr());
  if (hasBuild != hasSchedule || characterized != hasBuild || !hasSelected)
    return plan.emitOpError(
        "component plans require an exact selected-protocol commitment; only "
        "characterized plans carry Build and schedule commitments");
  if (!canonicalDigest(plan.getSourceSelectedProtocolSha256Attr().getValue(),
                       true) ||
      (characterized &&
       (!canonicalDigest(plan.getSourceBuildSha256Attr().getValue(), false) ||
        !canonicalDigest(plan.getSourceScheduleSha256Attr().getValue(),
                         false))))
    return plan.emitOpError(
        "component compiler source commitments must be canonical SHA-256");
  auto selectedDigest = selectedProtocolDigest(source);
  if (failed(selectedDigest))
    return plan.emitOpError(
        "component selected P2 protocol has an unresolved linked closure");
  if (*selectedDigest != plan.getSourceSelectedProtocolSha256Attr().getValue())
    return plan.emitOpError("component selected P2 protocol differs from its ")
           << (characterized ? "characterized compiler source; computed "
                             : "asserted source; computed ")
           << *selectedDigest;
  if (!plan.getPolicyAttr() || !plan.getIntervalSemanticsAttr() ||
      !plan.getSourceTimingProfileAttr() || !plan.getSourceCodeDistancesAttr())
    return plan.emitOpError(
        "component plan requires policy, interval, timing, and domain facts");
  auto point = SymbolTable(plan->getParentOfType<ModuleOp>())
                   .lookup<OperatingPointOp>(plan.getOperatingPoint());
  if (!point || point.getTimingSourceAttr() != plan.getSourceTimingSourceAttr())
    return plan.emitOpError(
        "component plan timing source differs from its operating point");
  DictionaryAttr selectedTiming = point.getTimingAttr();
  size_t consequentialTimingFacts = 0;
  if (selectedTiming)
    for (NamedAttribute item : selectedTiming)
      if (item.getName() == "cycle_ns" ||
          item.getName().getValue().ends_with("_ns"))
        ++consequentialTimingFacts;
  if (plan.getSourceTimingProfileAttr().size() != consequentialTimingFacts)
    return plan.emitOpError(
        "component timing profile must exactly cover the selected operating "
        "point timing facts");
  for (NamedAttribute item : plan.getSourceTimingProfileAttr()) {
    Attribute expected =
        selectedTiming ? selectedTiming.get(item.getName()) : Attribute{};
    auto expectedNumber = spacetimeNumericValue(expected);
    auto actualNumber = spacetimeNumericValue(item.getValue());
    if (!expectedNumber || !actualNumber || *expectedNumber != *actualNumber)
      return plan.emitOpError("component timing fact '")
             << item.getName() << "' differs from its operating point";
  }
  for (int64_t distance : plan.getSourceCodeDistancesAttr().asArrayRef())
    if (distance <= 0)
      return plan.emitOpError(
          "component source code distances must be positive");
  auto commitment = plan.getModelCommitmentAttr();
  if (!commitment)
    return plan.emitOpError(
        "component plan requires a replayable model commitment");
  if (sha256Hex(commitment.getValue()) != modelDigest.getValue())
    return plan.emitOpError(
        "model_sha256 does not authenticate model_commitment");
  auto parsed = llvm::json::parse(commitment.getValue());
  const llvm::json::Object *root = parsed ? parsed->getAsObject() : nullptr;
  if (!root || root->size() != 8 ||
      root->getString("schema") != "qlx.spacetime-plan-model/v2")
    return plan.emitOpError(
        "model_commitment must be canonical qlx.spacetime-plan-model/v2 JSON");
  auto protocol = root->getObject("protocol");
  auto committedLatency = root->getNumber("latency_cycles");
  auto committedInterval = root->getNumber("initiation_interval_cycles");
  auto committedSemantics = root->getString("interval_semantics");
  auto committedPolicy = root->getString("policy");
  auto committedDistances = root->getArray("code_distances");
  auto committedPhases = root->getArray("phases");
  Attribute cycleRaw =
      selectedTiming ? selectedTiming.get("surface_cycle_ns") : Attribute{};
  if (!cycleRaw && selectedTiming)
    cycleRaw = selectedTiming.get("cycle_ns");
  auto cycle = spacetimeNumericValue(cycleRaw);
  double latency = plan.getForwardingLatencyNsAttr().getValueAsDouble();
  double interval = plan.getInitiationIntervalNsAttr().getValueAsDouble();
  SmallVector<SpacetimePhaseOp, 4> phases;
  for (SpacetimePhaseOp phase :
       plan.getBody().front().getOps<SpacetimePhaseOp>())
    phases.push_back(phase);
  if (!protocol || protocol->size() != 6 ||
      protocol->getString("name") != plan.getSourceProtocol() ||
      protocol->getString("source_sha256") != sourceDigest.getValue() ||
      protocol->getString("objective_sha256") != objectiveDigest.getValue() ||
      protocol->getString("boundary_sha256") != boundaryDigest.getValue() ||
      !committedLatency || !committedInterval || !committedSemantics ||
      !committedPolicy || !committedDistances || !committedPhases || !cycle ||
      !std::isfinite(*cycle) || *cycle <= 0.0 ||
      latency != *committedLatency * *cycle ||
      interval != *committedInterval * *cycle ||
      plan.getIntervalSemantics() != *committedSemantics ||
      plan.getPolicy() != *committedPolicy ||
      committedDistances->size() != plan.getSourceCodeDistancesAttr().size() ||
      committedPhases->size() != phases.size())
    return plan.emitOpError(
        "component plan facts differ from model_commitment");
  for (auto [raw, distance] : llvm::zip(
           *committedDistances, plan.getSourceCodeDistancesAttr().asArrayRef()))
    if (raw.getAsInteger() != distance)
      return plan.emitOpError(
          "component code distances differ from model_commitment");
  auto architecture = SymbolTable(plan->getParentOfType<ModuleOp>())
                          .lookup<ArchitectureOp>(plan.getArchitecture());
  if (!architecture)
    return plan.emitOpError("component architecture does not resolve");
  for (auto [phase, committed] : llvm::zip(phases, *committedPhases)) {
    auto *record = committed.getAsObject();
    auto resources = record ? record->getArray("resources") : nullptr;
    auto factories = record ? record->getArray("factories") : nullptr;
    auto dependencies = record ? record->getArray("after") : nullptr;
    auto claims = phase.getResourceClaimsAttr();
    if (!record || record->size() != 6 || !resources || !factories ||
        !dependencies || record->getString("name") != phase.getSymName() ||
        record->getInteger("steps") != phase.getSteps() ||
        record->getNumber("step_duration_cycles") !=
            phase.getStepDurationNs().convertToDouble() / *cycle ||
        resources->size() != (claims ? claims.size() : 0) ||
        factories->size() != phase.getFactoryModels().size() ||
        dependencies->size() != phase.getAfter().size())
      return phase.emitOpError("facts differ from model_commitment");
    ArrayRef<Attribute> claimValues =
        claims ? claims.getValue() : ArrayRef<Attribute>{};
    for (auto [rawClaim, committedClaim] : llvm::zip(claimValues, *resources)) {
      auto claim = cast<DictionaryAttr>(rawClaim);
      auto reference = claim.getAs<SymbolRefAttr>("resource_class");
      auto resource =
          SymbolTable(architecture)
              .lookup<ResourceClassOp>(reference.getLeafReference().getValue());
      auto *resourceRecord = committedClaim.getAsObject();
      if (!resource || !resourceRecord || resourceRecord->size() != 7 ||
          resourceRecord->getString("resource") != resource.getSymName() ||
          resourceRecord->getString("kind") != resource.getKind() ||
          resourceRecord->getInteger("class_count") != resource.getCount() ||
          resourceRecord->getString("granularity") !=
              resource.getGranularity() ||
          resourceRecord->getInteger("offset") !=
              claim.getAs<IntegerAttr>("offset").getInt() ||
          resourceRecord->getInteger("count") !=
              claim.getAs<IntegerAttr>("count").getInt() ||
          resourceRecord->getInteger("units") !=
              claim.getAs<IntegerAttr>("units").getInt())
        return phase.emitOpError(
            "resource claims differ from model_commitment");
    }
    for (auto [rawFactory, committedFactory] :
         llvm::zip(phase.getFactoryModels(), *factories)) {
      auto reference = cast<FlatSymbolRefAttr>(rawFactory);
      auto factory = SymbolTable(plan->getParentOfType<ModuleOp>())
                         .lookup<FactoryModelOp>(reference.getValue());
      auto *factoryRecord = committedFactory.getAsObject();
      if (!factory || !factoryRecord || factoryRecord->size() != 4 ||
          factoryRecord->getNumber("startup_cycles") !=
              factory.getStartupNs().convertToDouble() / *cycle ||
          factoryRecord->getNumber("output_interval_cycles") !=
              factory.getOutputIntervalNs().convertToDouble() / *cycle ||
          factoryRecord->getString("policy") != factory.getPolicy() ||
          factoryRecord->getString("evidence") != factory.getEvidence())
        return phase.emitOpError(
            "factory dependencies differ from model_commitment");
    }
    for (auto [rawDependency, committedDependency] :
         llvm::zip(phase.getAfter(), *dependencies))
      if (committedDependency.getAsString() !=
          cast<FlatSymbolRefAttr>(rawDependency).getValue())
        return phase.emitOpError("dependencies differ from model_commitment");
  }
  if (characterized) {
    std::string expectedEvidence =
        ("computation:qlx.spacetime-plan-characterization/v1:" +
         plan.getSourceBuildSha256Attr().getValue() + ":" +
         plan.getSourceScheduleSha256Attr().getValue() + ":" +
         modelDigest.getValue())
            .str();
    if (plan.getEvidence() != expectedEvidence)
      return plan.emitOpError(
          "compiler component evidence differs from its exact source/model "
          "commitments");
  }
  return success();
}

static LogicalResult verifyComponentInvocation(SpacetimeCallOp call,
                                               SpacetimePlanOp plan) {
  auto source = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      SymbolTable(call->getParentOfType<ModuleOp>())
          .lookup(call.getSourceProtocolAttr().getValue()));
  if (!source)
    return call.emitOpError(
        "component invocation source must resolve to fabric.protocol");
  if (source->getAttrOfType<StringAttr>("component_source_sha256") !=
          plan.getSourceProtocolSha256Attr() ||
      source->getAttrOfType<StringAttr>("component_objective_sha256") !=
          plan.getSourceObjectiveSha256Attr() ||
      source->getAttrOfType<StringAttr>("component_boundary_sha256") !=
          plan.getSourceBoundarySha256Attr())
    return call.emitOpError(
        "component invocation source/objective/boundary differs from its plan");
  return success();
}

} // namespace plan_provider

namespace resource_provider {
using namespace qlx::fabric;

/// Follow one AutoCCZ patch owner backwards through the only ownership-
/// preserving operations used by the Figure-4 producer. The returned value is
/// the stable SSA origin against which a semantic payload role is checked.
static bool isLogicalCZCall(CallOp call);

static FailureOr<Value> traceAutoCCZPatchRoot(Value value) {
  llvm::SmallDenseSet<Value, 16> seen;
  for (unsigned depth = 0; depth < 64; ++depth) {
    if (!seen.insert(value).second)
      return failure();
    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return value;
    Operation *owner = result.getOwner();
    unsigned index = result.getResultNumber();
    if (auto call = dyn_cast<CallOp>(owner)) {
      if (!isLogicalCZCall(call) ||
          call.getNumResults() != call.getNumOperands() ||
          index >= call.getNumOperands())
        return failure();
      value = call.getOperand(index);
      continue;
    }
    if (auto prepare = dyn_cast<PrepXOp>(owner)) {
      if (index != 0)
        return failure();
      value = prepare.getPatch();
      continue;
    }
    return value;
  }
  return failure();
}

static bool isLogicalCZCall(CallOp call) {
  auto *callee =
      SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr());
  auto gadget = dyn_cast_or_null<GadgetOp>(callee);
  if (!gadget || !gadget.getSpecAttr())
    return false;
  auto *specTarget =
      SymbolTable::lookupNearestSymbolFrom(gadget, gadget.getSpecAttr());
  auto spec = dyn_cast_or_null<GadgetSpecOp>(specTarget);
  if (!spec)
    return false;
  auto *objectiveTarget =
      SymbolTable::lookupNearestSymbolFrom(spec, spec.getObjectiveAttr());
  auto objective = dyn_cast_or_null<ObjectiveOp>(objectiveTarget);
  if (!objective || !objective.getLogicalAttr())
    return false;
  auto *logicalTarget = SymbolTable::lookupNearestSymbolFrom(
      objective, objective.getLogicalAttr());
  auto logical = dyn_cast_or_null<::qlx::ActionOp>(logicalTarget);
  auto equivalence = spec.getActionEquivalence();
  return logical && logical.getKind() == "cz" && equivalence &&
         (*equivalence == "derived_exact_clifford_call_composition" ||
          *equivalence == "derived_exact_bare_physical_cz");
}

static bool exactAutoCCZGadgetActionCall(CallOp call, qlx::BuiltinAction action,
                                         ArrayRef<Value> operands) {
  if (!call || !sameValues(call.getOperands(), operands) ||
      call.getNumResults() != operands.size())
    return false;
  auto gadget = dyn_cast_or_null<GadgetOp>(
      SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
  auto logical = gadgetLogicalAction(gadget);
  return logical && *logical == qlx::stringifyBuiltinAction(action);
}

static bool exactAutoCCZYield(Block &block, ArrayRef<Value> values) {
  auto terminator = dyn_cast<qlx::cflow::YieldOp>(block.getTerminator());
  return terminator && sameValues(terminator.getOperands(), values);
}

static bool exactAutoCCZDelayedBasis(qlx::cflow::IfOp conditional, Value choice,
                                     Value left, Value right) {
  if (!conditional || conditional.getCondition() != choice ||
      conditional.getNumResults() != 2 || conditional.getThenRegion().empty() ||
      conditional.getElseRegion().empty())
    return false;
  Block &thenBlock = conditional.getThenRegion().front();
  Block &elseBlock = conditional.getElseRegion().front();
  if (thenBlock.getNumArguments() != 0 || elseBlock.getNumArguments() != 0 ||
      std::distance(thenBlock.begin(), thenBlock.end()) != 3 ||
      std::distance(elseBlock.begin(), elseBlock.end()) != 1)
    return false;
  auto first = dyn_cast<CallOp>(thenBlock.front());
  auto second = dyn_cast<CallOp>(*std::next(thenBlock.begin()));
  return exactAutoCCZGadgetActionCall(first, qlx::BuiltinAction::h, {left}) &&
         exactAutoCCZGadgetActionCall(second, qlx::BuiltinAction::h, {right}) &&
         exactAutoCCZYield(thenBlock,
                           {first.getResult(0), second.getResult(0)}) &&
         exactAutoCCZYield(elseBlock, {left, right});
}

static bool exactAutoCCZOutcomeSwap(qlx::cflow::IfOp conditional, Value choice,
                                    Value left, Value right) {
  if (!conditional || conditional.getCondition() != choice ||
      conditional.getNumResults() != 2 || conditional.getThenRegion().empty() ||

      conditional.getElseRegion().empty())
    return false;
  Block &thenBlock = conditional.getThenRegion().front();
  Block &elseBlock = conditional.getElseRegion().front();
  return thenBlock.getNumArguments() == 0 && elseBlock.getNumArguments() == 0 &&
         std::distance(thenBlock.begin(), thenBlock.end()) == 1 &&
         std::distance(elseBlock.begin(), elseBlock.end()) == 1 &&
         exactAutoCCZYield(thenBlock, {right, left}) &&
         exactAutoCCZYield(elseBlock, {left, right});
}

static bool exactAutoCCZApplyZIf(qlx::cflow::IfOp conditional, Value bit,
                                 Value input) {
  if (!conditional || conditional.getCondition() != bit ||
      conditional.getNumResults() != 1 || conditional.getThenRegion().empty() ||
      conditional.getElseRegion().empty())
    return false;
  Block &thenBlock = conditional.getThenRegion().front();
  Block &elseBlock = conditional.getElseRegion().front();
  if (thenBlock.getNumArguments() != 0 || elseBlock.getNumArguments() != 0 ||
      std::distance(thenBlock.begin(), thenBlock.end()) != 2 ||
      std::distance(elseBlock.begin(), elseBlock.end()) != 1)
    return false;
  auto z = dyn_cast<CallOp>(thenBlock.front());
  return exactAutoCCZGadgetActionCall(z, qlx::BuiltinAction::z, {input}) &&
         exactAutoCCZYield(thenBlock, {z.getResult(0)}) &&
         exactAutoCCZYield(elseBlock, {input});
}

static bool exactAutoCCZApplyZIfBoth(qlx::cflow::IfOp conditional, Value first,
                                     Value second, Value input) {
  if (!conditional || conditional.getCondition() != first ||
      conditional.getNumResults() != 1 || conditional.getThenRegion().empty() ||
      conditional.getElseRegion().empty())
    return false;
  Block &thenBlock = conditional.getThenRegion().front();
  Block &elseBlock = conditional.getElseRegion().front();
  if (thenBlock.getNumArguments() != 0 || elseBlock.getNumArguments() != 0 ||
      std::distance(thenBlock.begin(), thenBlock.end()) != 2 ||
      std::distance(elseBlock.begin(), elseBlock.end()) != 1)
    return false;
  auto nested = dyn_cast<qlx::cflow::IfOp>(thenBlock.front());
  return exactAutoCCZApplyZIf(nested, second, input) &&
         exactAutoCCZYield(thenBlock, {nested.getResult(0)}) &&
         exactAutoCCZYield(elseBlock, {input});
}

static bool exactFactoryProtocolProduces(ProtocolOp protocol, StringRef kind) {
  if (!protocol || protocol.getBody().empty())
    return false;
  FunctionType type = protocol.getFunctionType();
  if (type.getNumInputs() != 0 || type.getNumResults() != 1)
    return false;
  auto resource = dyn_cast<ResourceStateType>(type.getResult(0));
  auto reference = resource ? dyn_cast<FlatSymbolRefAttr>(resource.getKind())
                            : FlatSymbolRefAttr{};
  return reference && reference.getValue() == kind;
}

static bool exactFactoryInstrumentCall(CallOp call, StringRef kind,
                                       Value input) {
  if (!call || !sameValues(call.getOperands(), {input}) ||
      call.getNumResults() != 1)
    return false;
  auto gadget = dyn_cast_or_null<GadgetOp>(
      SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
  auto instrument = gadgetLogicalInstrument(gadget);
  return instrument && instrument.getKind() == kind;
}

static bool exactFactoryProduct(ValueRange actual, ValueRange results,
                                ArrayRef<int64_t> patchIndices,
                                ArrayRef<int64_t> logicalIndices,
                                StringRef pauliProduct,
                                ArrayRef<Value> expected, char pauli) {
  if (actual.size() != expected.size() || results.size() != expected.size() ||
      patchIndices.size() != expected.size() ||
      logicalIndices.size() != expected.size() ||
      pauliProduct.size() != expected.size())
    return false;
  llvm::SmallDenseSet<Value, 8> expectedValues(expected.begin(),
                                               expected.end());
  if (expectedValues.size() != expected.size())
    return false;
  for (Value value : actual)
    if (!expectedValues.erase(value))
      return false;
  if (!expectedValues.empty())
    return false;
  for (auto [index, terms] :
       llvm::enumerate(llvm::zip(patchIndices, logicalIndices, pauliProduct)))
    if (std::get<0>(terms) != static_cast<int64_t>(index) ||
        std::get<1>(terms) != 0 || std::get<2>(terms) != pauli)
      return false;
  return true;
}

static bool advanceFactoryProduct(ValueRange inputs, ValueRange results,
                                  MutableArrayRef<Value> state) {

  if (inputs.size() != results.size())
    return false;
  llvm::SmallDenseSet<unsigned, 8> updated;
  for (auto [input, result] : llvm::zip(inputs, results)) {
    auto found = llvm::find(state, input);
    if (found == state.end())
      return false;
    unsigned index = static_cast<unsigned>(std::distance(state.begin(), found));
    if (!updated.insert(index).second)
      return false;
    state[index] = result;
  }
  return true;
}

static bool exactFactoryResourceRotation(ResourceRotateProductOp rotation,
                                         Value resource,
                                         ArrayRef<Value> patches,
                                         StringRef kind) {
  if (!rotation)
    return false;
  bool sameResource = rotation.getResource() == resource;
  bool sameAngle = rotation.getAngle().convertToDouble() == 0.7853981633974483;
  auto state = dyn_cast<ResourceStateType>(resource.getType());
  auto reference = state ? dyn_cast<FlatSymbolRefAttr>(state.getKind())
                         : FlatSymbolRefAttr{};
  bool sameKind = reference && reference.getValue() == kind;
  bool sameProduct = exactFactoryProduct(
      rotation.getPatches(), rotation.getPatchResults(),
      rotation.getPatchIndices(), rotation.getLogicalIndices(),
      rotation.getPauliProduct(), patches, 'Z');
  return sameResource && sameAngle && sameKind && sameProduct;
}

static bool exactFactorySelection(qlx::event::SelectionOp selection,
                                  Value predicate) {
  return selection && selection.getPredicate() == predicate &&
         selection.getMode() == "abort_on" && !selection.getAcceptWhen();
}

static bool hasExactFifteenToOneStructure(ProtocolOp protocol) {
  if (!exactFactoryProtocolProduces(protocol, "t_state"))
    return false;
  Block &block = protocol.getBody().front();
  SmallVector<Operation *, 64> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  size_t cursor = 0;
  auto take = [&](StringRef name) -> Operation * {
    if (cursor >= operations.size() ||
        operations[cursor]->getName().getStringRef() != name)
      return nullptr;
    return operations[cursor++];
  };

  SmallVector<Value, 15> raw;
  for (unsigned index = 0; index < 15; ++index) {
    auto request =
        dyn_cast_or_null<ResourceRequestOp>(take("fabric.resource_request"));
    auto await = dyn_cast_or_null<qlx::event::AwaitOp>(take("event.await"));
    if (!request || request.getKind() != "raw_t_state" || !await ||
        await->getNumOperands() != 1 || await->getNumResults() != 1 ||
        await->getOperand(0) != request->getResult(0))
      return false;
    raw.push_back(await->getResult(0));
  }

  auto allocation = dyn_cast_or_null<AllocOp>(take("fabric.alloc"));
  auto prepare = dyn_cast_or_null<CallOp>(take("fabric.call"));
  if (!allocation || !exactFactoryInstrumentCall(prepare, "prepare_plus",
                                                 allocation->getResult(0)))
    return false;
  Value output = prepare.getResult(0);
  SmallVector<Value, 5> values;
  for (unsigned index = 0; index < 4; ++index) {
    auto unpack =
        dyn_cast_or_null<UnpackResourceOp>(take("fabric.unpack_resource"));
    if (!unpack || unpack->getNumOperands() != 2 ||
        unpack->getNumResults() != 2 || unpack->getOperand(0) != raw[index] ||
        unpack->getOperand(1) != output)
      return false;
    output = unpack->getResult(0);
    values.push_back(unpack->getResult(1));
  }
  values.push_back(output);

  const std::array<SmallVector<unsigned, 5>, 11> supports = {
      SmallVector<unsigned, 5>{2, 3, 4},      SmallVector<unsigned, 5>{1, 3, 4},
      SmallVector<unsigned, 5>{1, 2, 4},      SmallVector<unsigned, 5>{1, 2, 3},
      SmallVector<unsigned, 5>{0, 3, 4},      SmallVector<unsigned, 5>{0, 2, 4},
      SmallVector<unsigned, 5>{0, 2, 3},      SmallVector<unsigned, 5>{0, 1, 4},
      SmallVector<unsigned, 5>{0, 1, 3},      SmallVector<unsigned, 5>{0, 1, 2},
      SmallVector<unsigned, 5>{0, 1, 2, 3, 4}};
  for (auto [offset, support] : llvm::enumerate(supports)) {

    SmallVector<Value, 5> inputs;
    for (unsigned index : support)
      inputs.push_back(values[index]);
    auto rotation = dyn_cast_or_null<ResourceRotateProductOp>(
        take("fabric.resource_rotate_product"));
    if (!exactFactoryResourceRotation(rotation, raw[offset + 4], inputs,
                                      "raw_t_state") ||
        !advanceFactoryProduct(rotation.getPatches(),
                               rotation.getPatchResults(), values))
      return false;
  }

  auto fold = dyn_cast_or_null<CallOp>(take("fabric.call"));
  if (!exactAutoCCZGadgetActionCall(fold, qlx::BuiltinAction::s, {values[4]}))
    return false;
  values[4] = fold.getResult(0);
  for (unsigned index = 0; index < 4; ++index) {
    auto measurement = dyn_cast_or_null<CallOp>(take("fabric.call"));
    auto selection =
        dyn_cast_or_null<qlx::event::SelectionOp>(take("event.selection"));
    if (!exactFabricMeasureCall(measurement, qlx::Pauli::X, values[index]) ||
        !exactFactorySelection(selection, measurement.getResult(0)))
      return false;
  }

  auto pack = dyn_cast_or_null<PackResourceOp>(take("fabric.pack_resource"));
  auto returned =
      dyn_cast_or_null<ProtocolReturnOp>(take("fabric.protocol_return"));
  return pack && pack.getResourceKindAttr().getValue() == "t_state" &&
         sameValues(pack.getPayloads(), {values[4]}) && returned &&
         sameValues(returned.getOperands(), {pack.getResource()}) &&
         cursor == operations.size();
}

static bool hasExactGidneyFowlerCCZStructure(ProtocolOp protocol) {
  if (!exactFactoryProtocolProduces(protocol, "ccz_state"))
    return false;
  Block &block = protocol.getBody().front();
  SmallVector<Operation *, 96> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  size_t cursor = 0;
  auto take = [&](StringRef name) -> Operation * {
    if (cursor >= operations.size() ||
        operations[cursor]->getName().getStringRef() != name)
      return nullptr;
    return operations[cursor++];
  };

  SmallVector<Value, 8> distilled;
  ProtocolOp level1;
  for (unsigned index = 0; index < 8; ++index) {
    auto call = dyn_cast_or_null<CallOp>(take("fabric.call"));
    auto callee =
        call
            ? dyn_cast_or_null<ProtocolOp>(SymbolTable::lookupNearestSymbolFrom(
                  call, call.getCalleeAttr()))
            : ProtocolOp{};
    if (!call || call->getNumOperands() != 0 || call->getNumResults() != 1 ||
        !hasExactFifteenToOneStructure(callee) || (level1 && level1 != callee))
      return false;
    level1 = callee;
    distilled.push_back(call->getResult(0));
  }

  SmallVector<Value, 11> patches;
  for (unsigned index = 0; index < 11; ++index) {
    auto allocation = dyn_cast_or_null<AllocOp>(take("fabric.alloc"));
    auto prepare = dyn_cast_or_null<CallOp>(take("fabric.call"));
    if (!allocation || !exactFactoryInstrumentCall(prepare, "prepare_zero",
                                                   allocation->getResult(0)))
      return false;
    patches.push_back(prepare.getResult(0));
  }

  const std::array<SmallVector<unsigned, 8>, 4> checkSupports = {
      SmallVector<unsigned, 8>{0, 3, 4, 5, 6},
      SmallVector<unsigned, 8>{3, 4, 5, 6, 7, 8, 9, 10},
      SmallVector<unsigned, 8>{2, 3, 5, 7, 9},
      SmallVector<unsigned, 8>{1, 3, 4, 7, 8}};
  SmallVector<Value, 4> checks;
  for (const auto &support : checkSupports) {
    SmallVector<Value, 8> inputs;
    for (unsigned index : support)
      inputs.push_back(patches[index]);
    auto measurement =
        dyn_cast_or_null<MeasureProductOp>(take("fabric.measure_product"));
    if (!measurement ||
        !exactFactoryProduct(
            measurement.getPatches(), measurement.getPatchResults(),
            measurement.getPatchIndices(), measurement.getLogicalIndices(),
            measurement.getPauliProduct(), inputs, 'X'))
      return false;
    if (!advanceFactoryProduct(measurement.getPatches(),
                               measurement.getPatchResults(), patches))
      return false;
    checks.push_back(measurement.getOutcome());
  }

  SmallVector<Value, 8> injectionBits;
  for (unsigned index = 0; index < 8; ++index) {
    unsigned target = index + 3;
    auto rotation = dyn_cast_or_null<ResourceRotateProductOp>(
        take("fabric.resource_rotate_product"));
    if (!exactFactoryResourceRotation(rotation, distilled[index],
                                      {patches[target]}, "t_state"))
      return false;
    patches[target] = rotation.getPatchResults().front();
    auto hadamard = dyn_cast_or_null<CallOp>(take("fabric.call"));
    if (!exactAutoCCZGadgetActionCall(hadamard, qlx::BuiltinAction::h,
                                      {patches[target]}))
      return false;
    patches[target] = hadamard.getResult(0);
    auto measurement = dyn_cast_or_null<CallOp>(take("fabric.call"));
    if (!exactFabricMeasureCall(measurement, qlx::Pauli::Z, patches[target]))
      return false;
    injectionBits.push_back(measurement.getResult(0));
  }

  Value parity = checks[1];
  for (Value bit : injectionBits) {
    auto xorOp = dyn_cast_or_null<XorOp>(take("fabric.xor"));
    if (!xorOp || xorOp.getLhs() != parity || xorOp.getRhs() != bit)
      return false;
    parity = xorOp.getResult();
  }
  auto selection =
      dyn_cast_or_null<qlx::event::SelectionOp>(take("event.selection"));
  if (!exactFactorySelection(selection, parity))
    return false;

  constexpr std::array<unsigned, 8> correctionMasks = {7, 6, 5, 4, 3, 2, 1, 0};
  for (auto [bit, mask] : llvm::zip(injectionBits, correctionMasks)) {
    for (unsigned output = 0; output < 3; ++output) {
      if (!(mask & (1u << (2 - output))))
        continue;
      auto conditional = dyn_cast_or_null<qlx::cflow::IfOp>(take("cflow.if"));
      if (!exactAutoCCZApplyZIf(conditional, bit, patches[output]))
        return false;
      patches[output] = conditional.getResult(0);
    }
  }
  constexpr std::array<unsigned, 3> syndromeChecks = {0, 2, 3};
  constexpr std::array<unsigned, 3> syndromeOutputs = {0, 2, 1};
  for (auto [check, output] : llvm::zip(syndromeChecks, syndromeOutputs)) {
    auto conditional = dyn_cast_or_null<qlx::cflow::IfOp>(take("cflow.if"));
    if (!exactAutoCCZApplyZIf(conditional, checks[check], patches[output]))
      return false;
    patches[output] = conditional.getResult(0);
  }

  for (unsigned output = 0; output < 3; ++output) {
    auto logicalX = dyn_cast_or_null<CallOp>(take("fabric.call"));
    if (!exactAutoCCZGadgetActionCall(logicalX, qlx::BuiltinAction::x,
                                      {patches[output]}))
      return false;
    patches[output] = logicalX.getResult(0);
  }
  auto pack = dyn_cast_or_null<PackResourceOp>(take("fabric.pack_resource"));
  auto returned =
      dyn_cast_or_null<ProtocolReturnOp>(take("fabric.protocol_return"));
  return pack && pack.getResourceKindAttr().getValue() == "ccz_state" &&
         sameValues(pack.getPayloads(), {patches[0], patches[1], patches[2]}) &&
         returned && sameValues(returned.getOperands(), {pack.getResource()}) &&
         cursor == operations.size();
}

static bool hasExactGidneyFowlerAutoCCZStructure(ProtocolOp protocol) {
  if (!exactFactoryProtocolProduces(protocol, "auto_ccz_state") ||
      !protocol.getFunctionType().getInputs().empty())
    return false;
  Block &block = protocol.getBody().front();
  SmallVector<Operation *, 40> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  size_t cursor = 0;
  auto take = [&](StringRef name) -> Operation * {
    if (cursor >= operations.size() ||
        operations[cursor]->getName().getStringRef() != name)
      return nullptr;
    return operations[cursor++];
  };

  SmallVector<Value, 3> anchors;
  for (unsigned index = 0; index < 3; ++index) {
    auto allocation = dyn_cast_or_null<AllocOp>(take("fabric.alloc"));
    if (!allocation)
      return false;
    anchors.push_back(allocation.getResult());
  }

  SmallVector<Value, 6> routing;
  for (unsigned index = 0; index < 6; ++index) {
    auto allocation = dyn_cast_or_null<AllocOp>(take("fabric.alloc"));
    auto prepare = dyn_cast_or_null<PrepXOp>(take("fabric.prep_x"));

    if (!allocation || !prepare || prepare.getPatch() != allocation.getResult())
      return false;
    routing.push_back(prepare.getResult());
  }

  auto cczCall = dyn_cast_or_null<CallOp>(take("fabric.call"));
  auto ccz =
      cczCall
          ? dyn_cast_or_null<ProtocolOp>(SymbolTable::lookupNearestSymbolFrom(
                cczCall, cczCall.getCalleeAttr()))
          : ProtocolOp{};
  if (!cczCall || cczCall->getNumOperands() != 0 ||
      cczCall->getNumResults() != 1 || !hasExactGidneyFowlerCCZStructure(ccz))
    return false;

  auto unpack =
      dyn_cast_or_null<UnpackResourceOp>(take("fabric.unpack_resource"));
  if (!unpack || unpack.getResource() != cczCall.getResult(0) ||
      !sameValues(unpack.getAnchors(), anchors) ||
      unpack.getOutputs().size() != 6 || !unpack.getPayloadActionAttr() ||
      unpack.getPayloadActionAttr().getValue() != qlx::BuiltinAction::ccz)
    return false;
  for (unsigned index = 0; index < 3; ++index) {
    auto deallocation = dyn_cast_or_null<DeallocOp>(take("fabric.dealloc"));
    if (!deallocation || deallocation.getPatch() != unpack.getOutputs()[index])
      return false;
  }

  // Track the current SSA owner for each semantic payload role.  Authors may
  // serialize the ring edge-by-edge or in three disjoint matchings; both are
  // the same circuit.  The provider authenticates dataflow, not source order.
  SmallVector<Value, 9> current{unpack.getOutputs()[3],
                                unpack.getOutputs()[4],
                                unpack.getOutputs()[5],
                                routing[0],
                                routing[1],
                                routing[2],
                                routing[3],
                                routing[4],
                                routing[5]};
  static constexpr std::array<unsigned, 9> ring = {0, 3, 4, 1, 5, 6, 2, 7, 8};
  llvm::SmallDenseSet<std::pair<unsigned, unsigned>, 16> remainingEdges;
  for (unsigned index = 0; index < ring.size(); ++index) {
    unsigned left = ring[index];
    unsigned right = ring[(index + 1) % ring.size()];
    remainingEdges.insert({std::min(left, right), std::max(left, right)});
  }
  for (unsigned edge = 0; edge < ring.size(); ++edge) {
    auto call = dyn_cast_or_null<CallOp>(take("fabric.call"));
    if (!call || !isLogicalCZCall(call) || call.getNumOperands() != 2 ||
        call.getNumResults() != 2)
      return false;
    auto left = llvm::find(current, call.getOperand(0));
    auto right = llvm::find(current, call.getOperand(1));
    if (left == current.end() || right == current.end() || left == right)
      return false;
    unsigned leftIndex = std::distance(current.begin(), left);
    unsigned rightIndex = std::distance(current.begin(), right);
    if (!remainingEdges.erase(
            {std::min(leftIndex, rightIndex), std::max(leftIndex, rightIndex)}))
      return false;
    current[leftIndex] = call.getResult(0);
    current[rightIndex] = call.getResult(1);
  }

  auto pack = dyn_cast_or_null<PackResourceOp>(take("fabric.pack_resource"));
  auto returned =
      dyn_cast_or_null<ProtocolReturnOp>(take("fabric.protocol_return"));
  return pack && pack.getResourceKindAttr().getValue() == "auto_ccz_state" &&
         remainingEdges.empty() && sameValues(pack.getPayloads(), current) &&
         returned && sameValues(returned.getOperands(), {pack.getResource()}) &&
         cursor == operations.size();
}

/// Prove the exact adaptive Figure-4 AutoCCZ-to-Toffoli consumer rather than
/// trusting a callee name or a protocol boundary.  The proof is deliberately
/// structural: every data/payload owner and measurement result is followed
/// through the canonical cross-coupled basis choices and Pauli corrections.
static bool exactAutoCCZConsumer(ProtocolOp protocol) {
  if (!protocol || protocol.getBody().empty() ||
      protocol.getFunctionType().getNumInputs() != 12 ||
      protocol.getFunctionType().getNumResults() != 3)
    return false;
  Block &block = protocol.getBody().front();
  if (block.getNumArguments() != 12 ||
      std::distance(block.begin(), block.end()) != 30)
    return false;
  SmallVector<Operation *, 32> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  size_t cursor = 0;
  auto take = [&](StringRef name) -> Operation * {
    if (cursor >= operations.size() ||
        operations[cursor]->getName().getStringRef() != name)
      return nullptr;
    return operations[cursor++];
  };
  auto consumeAction = [&](qlx::BuiltinAction action,
                           ArrayRef<Value> inputs) -> CallOp {
    auto call = dyn_cast_or_null<CallOp>(take("fabric.call"));
    return exactAutoCCZGadgetActionCall(call, action, inputs) ? call : CallOp{};
  };
  auto consumeMeasurement = [&](Value input) -> CallOp {
    auto call = dyn_cast_or_null<CallOp>(take("fabric.call"));
    return exactFabricMeasureCall(call, qlx::Pauli::Z, input) ? call : CallOp{};
  };
  auto consumeDelayedPair =
      [&](Value choice, Value left,

          Value right) -> FailureOr<std::pair<Value, Value>> {
    auto basis = dyn_cast_or_null<qlx::cflow::IfOp>(take("cflow.if"));
    if (!exactAutoCCZDelayedBasis(basis, choice, left, right))
      return failure();
    auto leftMeasurement = consumeMeasurement(basis.getResult(0));
    auto rightMeasurement = consumeMeasurement(basis.getResult(1));
    if (!leftMeasurement || !rightMeasurement)
      return failure();
    auto swap = dyn_cast_or_null<qlx::cflow::IfOp>(take("cflow.if"));
    if (!exactAutoCCZOutcomeSwap(swap, choice, leftMeasurement.getResult(0),
                                 rightMeasurement.getResult(0)))
      return failure();
    return std::pair<Value, Value>{swap.getResult(0), swap.getResult(1)};
  };
  auto consumeZIf = [&](Value bit, Value input) -> Value {
    auto conditional = dyn_cast_or_null<qlx::cflow::IfOp>(take("cflow.if"));
    return exactAutoCCZApplyZIf(conditional, bit, input)
               ? conditional.getResult(0)
               : Value{};
  };
  auto consumeZIfBoth = [&](Value first, Value second, Value input) -> Value {
    auto conditional = dyn_cast_or_null<qlx::cflow::IfOp>(take("cflow.if"));
    return exactAutoCCZApplyZIfBoth(conditional, first, second, input)
               ? conditional.getResult(0)
               : Value{};
  };

  Value controlA = block.getArgument(0);
  Value controlB = block.getArgument(1);
  Value target = block.getArgument(2);
  auto initialH = consumeAction(qlx::BuiltinAction::h, {target});
  if (!initialH)
    return false;
  target = initialH.getResult(0);
  auto coupleA =
      consumeAction(qlx::BuiltinAction::cx, {controlA, block.getArgument(3)});
  auto coupleB =
      consumeAction(qlx::BuiltinAction::cx, {controlB, block.getArgument(4)});
  auto coupleC =
      consumeAction(qlx::BuiltinAction::cx, {target, block.getArgument(5)});
  if (!coupleA || !coupleB || !coupleC)
    return false;
  controlA = coupleA.getResult(0);
  controlB = coupleB.getResult(0);
  target = coupleC.getResult(0);
  auto outcomeA = consumeMeasurement(coupleA.getResult(1));
  auto outcomeB = consumeMeasurement(coupleB.getResult(1));
  auto outcomeC = consumeMeasurement(coupleC.getResult(1));
  if (!outcomeA || !outcomeB || !outcomeC)
    return false;

  auto ab = consumeDelayedPair(outcomeC.getResult(0), block.getArgument(6),
                               block.getArgument(7));
  auto bc = consumeDelayedPair(outcomeA.getResult(0), block.getArgument(8),
                               block.getArgument(9));
  auto ca = consumeDelayedPair(outcomeB.getResult(0), block.getArgument(10),
                               block.getArgument(11));
  if (failed(ab) || failed(bc) || failed(ca))
    return false;

  controlA = consumeZIf(ab->first, controlA);
  controlB = consumeZIf(ab->second, controlB);
  controlB = consumeZIf(bc->first, controlB);
  target = consumeZIf(bc->second, target);
  target = consumeZIf(ca->first, target);
  controlA = consumeZIf(ca->second, controlA);
  if (!controlA || !controlB || !target)
    return false;
  target = consumeZIfBoth(outcomeA.getResult(0), outcomeB.getResult(0), target);
  controlA =
      consumeZIfBoth(outcomeB.getResult(0), outcomeC.getResult(0), controlA);
  controlB =
      consumeZIfBoth(outcomeA.getResult(0), outcomeC.getResult(0), controlB);
  if (!controlA || !controlB || !target)
    return false;
  auto finalH = consumeAction(qlx::BuiltinAction::h, {target});
  auto returned =
      dyn_cast_or_null<ProtocolReturnOp>(take("fabric.protocol_return"));
  return finalH && returned &&
         sameValues(returned.getOperands(),
                    {controlA, controlB, finalH.getResult(0)}) &&
         cursor == operations.size();
}

/// Match the complete selected AutoCCZ application boundary.  This is a
/// reusable QEC implementation contract, not an algorithm recognizer: any P1
/// program may select this three-owner CCX realization.
static bool hasExactSurfaceAutoCCZApplicationStructure(ProtocolOp protocol) {
  if (!protocol || protocol.getBody().empty() ||
      protocol.getFunctionType().getNumInputs() != 3 ||
      protocol.getFunctionType().getNumResults() != 3)
    return false;
  Block &block = protocol.getBody().front();
  if (block.getNumArguments() != 3 ||
      std::distance(block.begin(), block.end()) != 17)
    return false;
  SmallVector<Operation *, 20> operations;
  for (Operation &operation : block)
    operations.push_back(&operation);
  size_t cursor = 0;
  auto take = [&](StringRef name) -> Operation * {
    if (cursor >= operations.size() ||
        operations[cursor]->getName().getStringRef() != name)
      return nullptr;
    return operations[cursor++];
  };
  auto request =
      dyn_cast_or_null<ResourceRequestOp>(take("fabric.resource_request"));
  auto awaited = dyn_cast_or_null<qlx::event::AwaitOp>(take("event.await"));
  if (!request || request.getKind() != kAutoCCZState || !awaited ||
      awaited.getEvent() != request.getEvent())
    return false;
  SmallVector<Value, 6> routing;
  for (unsigned index = 0; index < 6; ++index) {
    auto allocation = dyn_cast_or_null<AllocOp>(take("fabric.alloc"));
    if (!allocation)
      return false;
    routing.push_back(allocation.getResult());
  }
  auto unpack =
      dyn_cast_or_null<UnpackResourceOp>(take("fabric.unpack_resource"));
  if (!unpack || unpack.getResource() != awaited.getPayload() ||
      unpack.getAnchors().size() != 9 || unpack.getOutputs().size() != 18 ||
      failed(verifyAutoCCZRoles(unpack, unpack.getPayloadRolesAttr(), 9)))
    return false;
  SmallVector<Value, 9> anchors(block.getArguments().begin(),
                                block.getArguments().end());
  llvm::append_range(anchors, routing);
  if (!llvm::equal(unpack.getAnchors(), anchors))
    return false;
  for (unsigned index = 0; index < 6; ++index) {
    auto deallocation = dyn_cast_or_null<DeallocOp>(take("fabric.dealloc"));
    if (!deallocation ||
        deallocation.getPatch() != unpack.getOutputs()[3 + index])
      return false;
  }
  auto consumer = dyn_cast_or_null<CallOp>(take("fabric.call"));
  SmallVector<Value, 12> expectedOperands(unpack.getOutputs().begin(),
                                          unpack.getOutputs().begin() + 3);
  expectedOperands.append(unpack.getOutputs().begin() + 9,
                          unpack.getOutputs().end());
  auto consumerProtocol =
      consumer
          ? dyn_cast_or_null<ProtocolOp>(SymbolTable::lookupNearestSymbolFrom(
                consumer, consumer.getCalleeAttr()))
          : ProtocolOp{};
  auto returned =
      dyn_cast_or_null<ProtocolReturnOp>(take("fabric.protocol_return"));
  return consumer && llvm::equal(consumer.getOperands(), expectedOperands) &&
         exactAutoCCZConsumer(consumerProtocol) && returned &&
         llvm::equal(returned.getOperands(), consumer.getResults()) &&
         cursor == operations.size();
}

static LogicalResult verifyAutoCCZUnpackProvenance(UnpackResourceOp unpack) {
  unsigned count = unpack.getAnchors().size();
  if (failed(verifyAutoCCZRoles(unpack, unpack.getPayloadRolesAttr(), count)))
    return failure();
  auto protocol = unpack->getParentOfType<ProtocolOp>();
  if (!protocol || protocol.getBody().empty() ||
      protocol.getFunctionType().getNumInputs() != 3)
    return unpack.emitOpError(
        "AutoCCZ unpack anchor provenance requires one three-owner protocol "
        "boundary");
  Block &entry = protocol.getBody().front();
  if (entry.getNumArguments() != 3)
    return unpack.emitOpError(
        "AutoCCZ unpack anchor provenance requires three entry owners");
  for (unsigned index = 0; index < 3; ++index)
    if (unpack.getAnchors()[index] != entry.getArgument(index))
      return unpack.emitOpError(
          "AutoCCZ unpack anchor provenance must bind the three main roles "
          "to the ordered protocol owners");

  SmallVector<Value, 6> routing(unpack.getAnchors().drop_front(3));
  SmallVector<Value, 6> orderedAllocations;
  for (Operation &operation : entry) {
    auto allocation = dyn_cast<AllocOp>(operation);
    if (allocation && llvm::is_contained(routing, allocation.getResult()))
      orderedAllocations.push_back(allocation.getResult());
  }
  if (routing.size() != 6 || !llvm::equal(routing, orderedAllocations))
    return unpack.emitOpError(
        "AutoCCZ unpack anchor provenance requires six ordered fresh routing "
        "allocations");
  for (Value anchor : routing) {
    auto allocation = anchor.getDefiningOp<AllocOp>();
    if (!allocation || allocation->getBlock() != &entry ||
        !anchor.hasOneUse() || *anchor.getUsers().begin() != unpack)
      return unpack.emitOpError(
          "AutoCCZ unpack anchor provenance requires six ordered fresh "
          "routing allocations");
  }
  SmallVector<Value, 12> expectedConsumerOperands;
  expectedConsumerOperands.append(unpack.getOutputs().begin(),
                                  unpack.getOutputs().begin() + 3);
  expectedConsumerOperands.append(unpack.getOutputs().begin() + count,
                                  unpack.getOutputs().end());
  CallOp consumer;
  for (Value value : expectedConsumerOperands) {
    if (!value.hasOneUse())
      return unpack.emitOpError(
          "AutoCCZ payload roles must each have one direct consumer use");
    auto candidate = dyn_cast<CallOp>(*value.getUsers().begin());
    if (!candidate)
      return unpack.emitOpError(
          "AutoCCZ payload roles must feed one direct protocol call");
    if (consumer && candidate != consumer)
      return unpack.emitOpError(
          "AutoCCZ payload roles must feed the same protocol call");
    consumer = candidate;
  }
  if (!consumer ||
      consumer.getOperands().size() != expectedConsumerOperands.size() ||
      !llvm::equal(consumer.getOperands(), expectedConsumerOperands))
    return unpack.emitOpError(
        "AutoCCZ consumer operands must follow the canonical payload roles");
  if (consumer->getParentOp() != protocol || consumer.getNumResults() != 3)
    return unpack.emitOpError(
        "AutoCCZ consumer must be one direct three-result protocol call");
  auto consumerProtocol = dyn_cast_or_null<ProtocolOp>(
      SymbolTable::lookupNearestSymbolFrom(consumer, consumer.getCalleeAttr()));
  if (!exactAutoCCZConsumer(consumerProtocol))
    return unpack.emitOpError(
        "AutoCCZ consumer must implement the canonical adaptive Figure-4 "
        "protocol");
  auto returned =
      dyn_cast<ProtocolReturnOp>(protocol.getBody().front().getTerminator());
  if (!returned || !llvm::equal(returned.getOperands(), consumer.getResults()))
    return unpack.emitOpError(
        "AutoCCZ consumer results must form the protocol boundary");
  return success();
}

static LogicalResult verifyAutoCCZPackProvenance(PackResourceOp pack) {
  if (failed(verifyAutoCCZRoles(pack, pack.getPayloadRolesAttr(),
                                pack.getPayloads().size())))
    return failure();
  auto protocol = pack->getParentOfType<ProtocolOp>();
  if (!protocol || protocol.getBody().empty() ||
      !protocol.getFunctionType().getInputs().empty())
    return pack.emitOpError(
        "AutoCCZ producer payload provenance requires a closed producer "
        "protocol");
  Block &entry = protocol.getBody().front();

  SmallVector<UnpackResourceOp, 2> sources;
  for (Operation &operation : entry) {
    auto unpack = dyn_cast<UnpackResourceOp>(operation);
    if (!unpack || unpack.getAnchors().size() != 3 ||
        unpack.getOutputs().size() != 6 || !unpack.getPayloadActionAttr() ||
        unpack.getPayloadActionAttr().getValue() != qlx::BuiltinAction::ccz)
      continue;
    auto resource = cast<ResourceStateType>(unpack.getResource().getType());
    auto kind = dyn_cast<SymbolRefAttr>(resource.getKind());
    if (kind && kind.getRootReference().getValue() == "ccz_state")
      sources.push_back(unpack);
  }
  if (sources.size() != 1)
    return pack.emitOpError(
        "AutoCCZ producer payload provenance requires one three-patch CCZ "
        "payload source");
  UnpackResourceOp source = sources.front();

  SmallVector<Value, 9> roots;
  for (Value payload : pack.getPayloads()) {
    FailureOr<Value> root = traceAutoCCZPatchRoot(payload);
    if (failed(root))
      return pack.emitOpError(
          "AutoCCZ producer payload provenance is not traceable");
    roots.push_back(*root);
  }
  for (unsigned index = 0; index < 3; ++index)
    if (roots[index] != source.getOutputs()[3 + index])
      return pack.emitOpError(
          "AutoCCZ producer payload provenance must bind the three main roles "
          "to the ordered CCZ payload");

  llvm::SmallDenseSet<Value, 16> uniqueRoots;
  for (Value root : roots)
    if (!uniqueRoots.insert(root).second)
      return pack.emitOpError(
          "AutoCCZ producer payload provenance must name nine distinct patch "
          "owners");

  SmallVector<Value, 6> routing(roots.begin() + 3, roots.end());
  SmallVector<Value, 6> orderedAllocations;
  for (Operation &operation : entry) {
    auto allocation = dyn_cast<AllocOp>(operation);
    if (allocation && llvm::is_contained(routing, allocation.getResult()))
      orderedAllocations.push_back(allocation.getResult());
  }
  if (!llvm::equal(routing, orderedAllocations))
    return pack.emitOpError(
        "AutoCCZ producer payload provenance requires six ordered fresh "
        "routing allocations");
  for (Value root : routing) {
    auto allocation = root.getDefiningOp<AllocOp>();
    if (!allocation || allocation->getBlock() != &entry || !root.hasOneUse() ||
        !isa<PrepXOp>(*root.getUsers().begin()))
      return pack.emitOpError(
          "AutoCCZ producer payload provenance requires six ordered fresh "
          "X-prepared routing patches");
  }

  auto rootIndex = [&](Value value) -> std::optional<unsigned> {
    FailureOr<Value> root = traceAutoCCZPatchRoot(value);
    if (failed(root))
      return std::nullopt;
    auto found = llvm::find(roots, *root);
    if (found == roots.end())
      return std::nullopt;
    return static_cast<unsigned>(std::distance(roots.begin(), found));
  };

  SmallVector<std::pair<unsigned, unsigned>, 9> observedEdges;
  for (Operation &operation : entry) {
    auto call = dyn_cast<CallOp>(operation);
    if (!call || !isLogicalCZCall(call))
      continue;
    if (call.getNumOperands() != 2 || call.getNumResults() != 2)
      return pack.emitOpError(
          "AutoCCZ ring CZ must preserve exactly two patch owners");
    auto left = rootIndex(call.getOperand(0));
    auto right = rootIndex(call.getOperand(1));
    if (!left || !right || *left == *right)
      return pack.emitOpError("AutoCCZ ring CZ has foreign patch provenance");
    observedEdges.emplace_back(std::min(*left, *right),
                               std::max(*left, *right));
  }
  static constexpr std::array<unsigned, 9> ring = {0, 3, 4, 1, 5, 6, 2, 7, 8};
  SmallVector<std::pair<unsigned, unsigned>, 9> expectedEdges;
  for (unsigned index = 0; index < ring.size(); ++index) {
    unsigned left = ring[index];
    unsigned right = ring[(index + 1) % ring.size()];
    expectedEdges.emplace_back(std::min(left, right), std::max(left, right));
  }
  llvm::sort(observedEdges);
  llvm::sort(expectedEdges);
  if (observedEdges != expectedEdges)
    return pack.emitOpError(
        "AutoCCZ producer payload provenance must form the canonical "
        "nine-patch CZ ring");
  return success();
}

static FlatSymbolRefAttr producedResourceKind(Operation *callable) {
  if (!callable)
    return {};
  FunctionType type;
  if (auto protocol = dyn_cast<ProtocolOp>(callable))
    type = protocol.getFunctionType();
  else if (auto gadget = dyn_cast<GadgetOp>(callable))
    type = gadget.getFunctionType();
  if (!type || type.getNumResults() != 1)
    return {};
  auto resource = dyn_cast<ResourceStateType>(type.getResult(0));
  return resource ? dyn_cast<FlatSymbolRefAttr>(resource.getKind())
                  : FlatSymbolRefAttr{};
}

static bool hasPackedResource(ProtocolOp protocol, StringRef kind,
                              size_t payloads) {
  size_t matches = 0;
  for (PackResourceOp pack :
       protocol.getBody().front().getOps<PackResourceOp>())
    if (pack.getResourceKindAttr().getValue() == kind &&
        pack.getPayloads().size() == payloads)
      ++matches;
  return matches == 1;
}

static bool matchesSurfaceAutoCCZFactory(ProtocolOp protocol) {
  auto produced = producedResourceKind(protocol);
  if (!produced || produced.getValue() != kAutoCCZState ||
      !hasPackedResource(protocol, kAutoCCZState, 9) ||
      !hasExactGidneyFowlerAutoCCZStructure(protocol))
    return false;

  SymbolTable moduleSymbols(protocol->getParentOfType<ModuleOp>());
  ProtocolOp ccz;
  size_t cczCalls = 0;
  for (CallOp call : protocol.getBody().front().getOps<CallOp>()) {
    Operation *callee = moduleSymbols.lookup(call.getCallee());
    auto kind = producedResourceKind(callee);
    if (!kind || kind.getValue() != kCCZState)
      continue;
    ccz = dyn_cast_or_null<ProtocolOp>(callee);
    ++cczCalls;
  }
  if (cczCalls != 1 || !ccz || !hasPackedResource(ccz, kCCZState, 3))
    return false;

  SmallVector<ProtocolOp, 8> level1;
  for (CallOp call : ccz.getBody().front().getOps<CallOp>()) {
    Operation *callee = moduleSymbols.lookup(call.getCallee());
    auto kind = producedResourceKind(callee);
    if (kind && kind.getValue() == kTState)
      if (auto producer = dyn_cast_or_null<ProtocolOp>(callee))
        level1.push_back(producer);
  }
  if (level1.size() != 8)
    return false;
  for (ProtocolOp producer : level1) {
    if (!hasPackedResource(producer, kTState, 1) ||
        !hasExactFifteenToOneStructure(producer))
      return false;
    size_t rawRequests = 0;
    producer.getBody().walk([&](ResourceRequestOp request) {
      if (request.getKind() == kRawTState)
        ++rawRequests;
    });
    if (rawRequests != 15)
      return false;
  }
  return hasExactGidneyFowlerCCZStructure(ccz);
}

static bool matchesSurfaceAutoCCZApplication(ProtocolOp protocol) {
  return hasExactSurfaceAutoCCZApplicationStructure(protocol);
}

static bool isLogicalCXCall(CallOp call) {
  if (!call || call.getNumOperands() != 2 || call.getNumResults() != 2 ||
      llvm::any_of(call.getOperandTypes(),
                   [](Type type) { return !isa<PatchType>(type); }) ||
      llvm::any_of(call.getResultTypes(),
                   [](Type type) { return !isa<PatchType>(type); }))
    return false;
  auto gadget = dyn_cast_or_null<GadgetOp>(
      SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
  auto logical = gadgetLogicalAction(gadget);
  auto spec = resolvedGadgetSpec(gadget);
  auto equivalence = spec ? spec.getActionEquivalence() : std::nullopt;
  return logical && *logical == "cx" && equivalence &&
         (*equivalence == "derived_exact_css_transversal_cx" ||
          *equivalence == "derived_exact_clifford_call_composition");
}

/// Recognize one exact, boundary-preserving surface-code call box.  The
/// analysis follows SSA ownership, never a helper spelling or an algorithm
/// role. A valid box contains one layer of one or more authenticated AutoCCZ
/// applications; exact CX calls may connect additional boundary owners to a
/// three-owner reaction core. Each such external connection is one
/// alternating-access layer in the physical layout selected by the P3
/// provider.
static std::optional<qlx::spacetime::SurfaceSpacelikeShape>
analyzeSurfaceSpacelike(ProtocolOp protocol,
                        llvm::SmallPtrSetImpl<Operation *> &active) {
  if (!protocol || protocol.getBody().empty() ||
      protocol.getBody().getBlocks().size() != 1 ||
      !active.insert(protocol.getOperation()).second)
    return std::nullopt;
  llvm::scope_exit eraseActive([&] { active.erase(protocol.getOperation()); });
  Block &block = protocol.getBody().front();
  unsigned boundary = block.getNumArguments();
  if (boundary < 3 || boundary > 262144 ||
      protocol.getFunctionType().getNumInputs() != boundary ||
      protocol.getFunctionType().getNumResults() != boundary ||
      llvm::any_of(block.getArgumentTypes(),
                   [](Type type) { return !isa<PatchType>(type); }) ||
      llvm::any_of(protocol.getFunctionType().getResults(),
                   [](Type type) { return !isa<PatchType>(type); }))
    return std::nullopt;

  SmallVector<unsigned> parent(boundary);
  for (unsigned index = 0; index < boundary; ++index)
    parent[index] = index;
  std::function<unsigned(unsigned)> find = [&](unsigned value) {
    if (parent[value] != value)
      parent[value] = find(parent[value]);
    return parent[value];
  };
  auto unite = [&](unsigned left, unsigned right) {
    left = find(left);
    right = find(right);
    if (left != right)
      parent[right] = left;
  };

  DenseMap<Value, unsigned> roots;
  for (auto [index, argument] : llvm::enumerate(block.getArguments()))
    roots[argument] = index;
  SmallVector<std::pair<unsigned, unsigned>, 8> cxEdges;
  SmallVector<llvm::SmallDenseSet<unsigned, 8>, 4> reactionGroups;
  SmallVector<unsigned, 8> reactionReady(boundary, 0);
  SmallVector<unsigned, 8> reactionWidthByLayer;
  unsigned reactionApplications = 0;
  unsigned nestedAccessLayers = 0;
  bool returned = false;

  for (Operation &operation : block) {
    if (auto terminator = dyn_cast<ProtocolReturnOp>(operation)) {
      if (&operation != block.getTerminator() || returned ||
          terminator.getNumOperands() != boundary)
        return std::nullopt;
      for (auto [index, value] : llvm::enumerate(terminator.getOperands())) {
        auto root = roots.find(value);
        if (root == roots.end() || root->second != index)
          return std::nullopt;
      }
      returned = true;
      continue;
    }
    auto call = dyn_cast<CallOp>(operation);
    if (!call || call.getNumOperands() != call.getNumResults() ||
        call.getNumOperands() == 0 ||
        llvm::any_of(call.getOperandTypes(),
                     [](Type type) { return !isa<PatchType>(type); }) ||
        llvm::any_of(call.getResultTypes(),
                     [](Type type) { return !isa<PatchType>(type); }))
      return std::nullopt;
    SmallVector<unsigned, 8> callRoots;
    for (Value operand : call.getOperands()) {
      auto root = roots.find(operand);
      if (root == roots.end())
        return std::nullopt;
      callRoots.push_back(root->second);
    }

    Operation *callee =
        SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr());
    std::optional<qlx::spacetime::SurfaceSpacelikeShape> nested;
    if (auto nestedProtocol = dyn_cast_or_null<ProtocolOp>(callee)) {
      if (hasExactSurfaceAutoCCZApplicationStructure(nestedProtocol))
        nested =
            qlx::spacetime::SurfaceSpacelikeShape{/*boundaryOwners=*/3,
                                                  /*accessLayers=*/0,
                                                  /*reactionDepth=*/1,
                                                  /*reactionApplications=*/1,
                                                  /*peakReactionWidth=*/1};
      else
        nested = analyzeSurfaceSpacelike(nestedProtocol, active);
    }
    if (nested) {
      if (nested->boundaryOwners != callRoots.size())
        return std::nullopt;
      nestedAccessLayers += nested->accessLayers;
      unsigned startLayer = 0;
      for (unsigned root : callRoots)
        startLayer = std::max(startLayer, reactionReady[root]);
      unsigned finishLayer = startLayer + nested->reactionDepth;
      if (reactionWidthByLayer.size() < finishLayer)
        reactionWidthByLayer.resize(finishLayer, 0);
      for (unsigned layer = startLayer; layer < finishLayer; ++layer)
        reactionWidthByLayer[layer] += nested->peakReactionWidth;
      for (unsigned root : callRoots)
        reactionReady[root] = finishLayer;
      reactionApplications += nested->reactionApplications;
      reactionGroups.emplace_back(callRoots.begin(), callRoots.end());
      for (unsigned index = 1; index < callRoots.size(); ++index)
        unite(callRoots.front(), callRoots[index]);
    } else if (isLogicalCXCall(call)) {
      cxEdges.emplace_back(callRoots[0], callRoots[1]);
      unite(callRoots[0], callRoots[1]);
    } else {
      return std::nullopt;
    }
    for (auto [result, root] : llvm::zip(call.getResults(), callRoots))
      roots[result] = root;
  }
  if (!returned || reactionApplications == 0 || reactionGroups.empty())
    return std::nullopt;
  for (unsigned index = 0; index < boundary; ++index) {
    bool reachesReaction = llvm::any_of(
        reactionGroups, [&](const llvm::SmallDenseSet<unsigned, 8> &group) {
          return llvm::any_of(
              group, [&](unsigned root) { return find(index) == find(root); });
        });
    if (!reachesReaction)
      return std::nullopt;
  }

  unsigned accessLayers = nestedAccessLayers;
  for (auto [left, right] : cxEdges)
    if (!llvm::any_of(reactionGroups,
                      [&](const llvm::SmallDenseSet<unsigned, 8> &group) {
                        return group.contains(left) && group.contains(right);
                      }))
      ++accessLayers;
  unsigned reactionDepth = *llvm::max_element(reactionReady);
  unsigned peakReactionWidth = reactionWidthByLayer.empty()
                                   ? 0
                                   : *llvm::max_element(reactionWidthByLayer);
  // A reusable spacelike macro is exactly one reaction layer.  Longer
  // callable DAGs remain ordinary structured calls so the scheduler retains
  // their phase ordering instead of collapsing an algorithm-scale body into
  // one opaque plan.
  if (reactionDepth != 1 || peakReactionWidth == 0)
    return std::nullopt;
  return qlx::spacetime::SurfaceSpacelikeShape{
      boundary, accessLayers, reactionDepth, reactionApplications,
      peakReactionWidth};
}

} // namespace resource_provider

} // namespace

std::optional<qlx::spacetime::BuiltinProviderKind>
qlx::spacetime::providerFor(fabric::ProtocolOp protocol) {
  if (resource_provider::matchesSurfaceAutoCCZFactory(protocol))
    return BuiltinProviderKind::SurfaceAutoCCZFactory;
  if (resource_provider::matchesSurfaceAutoCCZApplication(protocol))
    return BuiltinProviderKind::SurfaceAutoCCZApplication;
  if (surfaceSpacelikeShape(protocol))
    return BuiltinProviderKind::SurfaceSpacelikeCallable;
  return std::nullopt;
}

std::optional<qlx::spacetime::SurfaceSpacelikeShape>
qlx::spacetime::surfaceSpacelikeShape(fabric::ProtocolOp protocol) {
  llvm::SmallPtrSet<Operation *, 8> active;
  return resource_provider::analyzeSurfaceSpacelike(protocol, active);
}

void qlx::spacetime::registerBuiltinProviders() {
  static std::once_flag once;
  std::call_once(once, [] {
    qlx::fabric::registerResourceContractVerifier(
        kAutoCCZState, resource_provider::verifyAutoCCZPackProvenance,
        resource_provider::verifyAutoCCZUnpackProvenance);
    qlx::phys::registerSpacetimePlanVerifier(
        kSpacetimeProvider, kSpacetimeProviderVersion,
        kSurfaceFactoryRecurrence, kSurfaceFactoryRecurrenceVersion,
        plan_provider::verifySurfaceFactoryRecurrencePlan);
    qlx::phys::registerSpacetimePlanVerifier(
        kSpacetimeProvider, kSpacetimeProviderVersion,
        kSurfaceAutoCCZApplication, kSurfaceAutoCCZApplicationVersion,
        plan_provider::verifySurfaceAutoCCZApplicationPlan);
    qlx::phys::registerSpacetimePlanVerifier(
        kSpacetimeProvider, kSpacetimeProviderVersion,
        kSurfaceSpacelikeCallable, kSurfaceSpacelikeCallableVersion,
        plan_provider::verifySurfaceSpacelikeCallablePlan,
        plan_provider::verifySurfaceSpacelikeCallableInvocation);
    qlx::phys::registerSpacetimePlanVerifier(
        "qlx.component-model", "1", "characterized", 1,
        plan_provider::verifyComponentPlan,
        plan_provider::verifyComponentInvocation);
    qlx::phys::registerSpacetimePlanVerifier(
        "qlx.component-model", "1", "asserted", 1,
        plan_provider::verifyComponentPlan,
        plan_provider::verifyComponentInvocation);
  });
}
