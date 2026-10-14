/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/Phys/IR/ScheduleVerification.h"
#include "qlx/Dialect/Phys/Transforms/Passes.h"
#include "qlx/Dialect/Phys/Transforms/ScheduleModel.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <optional>
#include <string>

namespace qlx {
namespace phys {
#define GEN_PASS_DEF_PHYSSCHEDULE
#include "qlx/Dialect/Phys/Transforms/Passes.h.inc"
} // namespace phys
} // namespace qlx

using namespace mlir;

namespace {

using ResourceId = qlx::phys::NativeScheduleResourceId;
using ScheduleRow = qlx::phys::NativeScheduleRow;

struct ScheduledResourceState {
  double ready = 0.0;
  StringRef producer;
  StringRef pipelinePlan;
  StringRef pipelineInvocation;
};

struct ScheduleState {
  SmallVector<ScheduledResourceState, 0> resources;
};

struct MappedCallTemplateSummary;

struct OperationScheduleInputs {
  SmallVector<ResourceId, 4> resourceIds;
  SmallVector<std::pair<ResourceId, unsigned>, 4> inputResourcePositions;
  SmallVector<StringRef, 4> dataDependencies;
  SmallVector<StringRef, 4> resourceDependencies;
  const MappedCallTemplateSummary *mappedCallTemplate = nullptr;
  double operandReady = 0.0;
  bool stateOnlyBoundary = true;
};

struct OperationResultState {
  double ready = 0.0;
  StringRef producer;
};

struct CallResourceAvailability {
  ResourceId resource;
  double offset = 0.0;
  StringRef pipelinePlan;
};

struct CallPipelineTransition {
  StringRef plan;
  double firstUseOffset = 0.0;
  double availabilityOffset = 0.0;
};

struct CallResourceTransition {
  std::string resource;
  double firstUseOffset = 0.0;
  double availabilityOffset = 0.0;
};

struct CallTemplateSummary {
  double duration = 0.0;
  bool containsModeledTransport = false;
  SmallVector<ResourceId, 8> boundaryResources;
  SmallVector<ResourceId, 8> descendantResources;
  SmallVector<std::pair<ResourceId, unsigned>, 8> inputResourcePositions;
  bool stateOnlyBoundary = true;
  SmallVector<std::pair<ResourceId, double>, 8> firstUseOffsets;
  SmallVector<CallResourceAvailability, 8> availabilityOffsets;
  SmallVector<CallPipelineTransition, 2> pipelineTransitions;
  std::optional<double> globalBarrierFirstUseOffset;
  std::optional<double> globalBarrierAvailabilityOffset;
};

struct MappedCallTemplateSummary {
  const DenseMap<ResourceId, ResourceId> *resourceAliases = nullptr;
  SmallVector<ResourceId, 8> closureResources;
  SmallVector<ResourceId, 8> elidedClosureResources;
  SmallVector<ResourceId, 8> inputResources;
  SmallVector<std::pair<ResourceId, double>, 8> firstUseOffsets;
  SmallVector<CallResourceAvailability, 8> availabilityOffsets;
};

struct ResourceChangeScope {
  DenseMap<ResourceId, ScheduledResourceState> previous;
  DenseMap<ResourceId, double> firstUses;
  std::optional<double> globalBarrierFirstUse;
  std::optional<double> globalBarrierAvailability;
};

struct PipelineChangeScope {
  llvm::StringMap<std::optional<OperationResultState>> previous;
  llvm::StringMap<double> firstUses;
};

using ResourceStateDelta = DenseMap<ResourceId, ScheduledResourceState>;

static std::optional<double> numericValue(Attribute attribute) {
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

static std::string symbolText(Attribute attribute) {
  if (auto value = dyn_cast_or_null<FlatSymbolRefAttr>(attribute))
    return value.getValue().str();
  if (auto value = dyn_cast_or_null<SymbolRefAttr>(attribute))
    return value.getLeafReference().getValue().str();
  if (auto value = dyn_cast_or_null<StringAttr>(attribute))
    return value.getValue().str();
  return {};
}

static double readyAt(const ScheduleState &state, ResourceId resource) {
  return resource < state.resources.size() ? state.resources[resource].ready
                                           : 0.0;
}

static StringRef producedBy(const ScheduleState &state, ResourceId resource) {
  return resource < state.resources.size() ? state.resources[resource].producer
                                           : StringRef{};
}

static StringRef pipelinePlanAt(const ScheduleState &state,
                                ResourceId resource) {
  return resource < state.resources.size()
             ? state.resources[resource].pipelinePlan
             : StringRef{};
}

static StringRef pipelineInvocationAt(const ScheduleState &state,
                                      ResourceId resource) {
  return resource < state.resources.size()
             ? state.resources[resource].pipelineInvocation
             : StringRef{};
}

static std::string formatTime(double value) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::format("%.17g", value);
  return result;
}

static std::string join(ArrayRef<StringRef> values) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  llvm::interleave(values, stream, ",");
  return result;
}

static std::string serialize(const ScheduleRow &row,
                             ArrayRef<std::string> resourceLabels) {
  auto integer = [](std::optional<int64_t> value) {
    return value ? std::to_string(*value) : std::string{};
  };
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << row.id << "|" << row.kind << "|" << formatTime(row.start) << "|"
         << formatTime(row.duration) << "|";
  if (row.resourceIds.empty())
    stream << "control:" << row.id;
  else
    llvm::interleave(
        row.resourceIds, stream,
        [&](ResourceId resource) { stream << resourceLabels[resource]; }, ",");
  stream << "|deps=" << join(row.dependencies)
         << "|data_deps=" << join(row.dataDependencies)
         << "|resource_deps=" << join(row.resourceDependencies)
         << "|domain_deps=" << join(row.domainDependencies)
         << "|parent=" << row.parent << "|branch=" << row.branch
         << "|condition=" << row.condition
         << "|max_attempts=" << integer(row.maxAttempts)
         << "|commit_point=" << row.commitPoint
         << "|repeat_count=" << integer(row.repeatCount) << "|repeat_period_ns="
         << (row.repeatPeriod ? formatTime(*row.repeatPeriod) : std::string{})
         << "|repeat_epilogue_ns="
         << (row.repeatEpilogue ? formatTime(*row.repeatEpilogue)
                                : std::string{})
         << "|max_iterations=" << integer(row.maxIterations)
         << "|callee=" << row.callee << "|instance=" << row.instance
         << "|profile=" << row.profile
         << "|template_event=" << row.templateEvent
         << "|attempt=" << row.attempt << "|attempt_event=" << row.attemptEvent
         << "|decision_event=" << row.decisionEvent
         << "|exhaustion=" << row.exhaustion << "|success_probability="
         << (row.successProbability ? formatTime(*row.successProbability)
                                    : std::string{})
         << "|success_probability_source=" << row.successProbabilitySource
         << "|success_probability_evidence=" << row.successProbabilityEvidence;
  return result;
}

class NativeScheduler {
public:
  NativeScheduler(ModuleOp module, StringRef requestedGraph,
                  StringRef requestedResult, bool deferOutputVerification,
                  bool deferEntryMaterialization = false)
      : module(module), context(module.getContext()), symbols(module),
        requestedGraph(requestedGraph.str()),
        requestedResult(requestedResult.str()),
        deferOutputVerification(deferOutputVerification),
        deferEntryMaterialization(deferEntryMaterialization) {}

  LogicalResult run() {
    const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
    profileEnabled = profile;
    auto phase = [&](StringRef name, auto &&action) -> LogicalResult {
      const auto started = std::chrono::steady_clock::now();
      LogicalResult result = action();
      if (profile)
        llvm::errs() << "phys-schedule: phase " << name << ' '
                     << std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started)
                            .count()
                     << "s\n";
      return result;
    };
    if (profile)
      llvm::errs() << "phys-schedule: start\n";
    if (failed(phase("resolve", [&] { return resolve(); })) ||
        failed(phase("graph", [&] {
          return scheduleBlock(graph.getBody().front(), state, rows);
        })))
      return failure();
    if (profile)
      llvm::errs() << "phys-schedule: graph-done rows=" << rows.size()
                   << " prelude-seconds=" << scheduledPreludeSeconds
                   << " template-prelude-seconds="
                   << scheduledTemplatePreludeSeconds
                   << " call-prelude-seconds=" << scheduledCallPreludeSeconds
                   << " leaf-prelude-seconds=" << scheduledLeafPreludeSeconds
                   << " call-seconds=" << scheduledCallSeconds
                   << " call-finalize-seconds=" << scheduledCallFinalizeSeconds
                   << " call-change-entries=" << scheduledCallChangeEntries
                   << " call-max-depth=" << scheduledCallMaxDepth
                   << " template-total-seconds="
                   << scheduledTemplateTotalSeconds
                   << " template-row-seconds=" << scheduledTemplateRowSeconds
                   << " template-push-seconds=" << scheduledTemplatePushSeconds
                   << " template-result-seconds="
                   << scheduledTemplateResultSeconds
                   << " template-results=" << scheduledTemplateResults << "\n";
    double makespan = 0.0;
    llvm::StringMap<bool> activeRows;
    llvm::StringMap<const ScheduleRow *> rowsById;
    for (const ScheduleRow &row : rows) {
      bool active = row.parent.empty();
      if (!row.parent.empty()) {
        auto parent = rowsById.find(row.parent);
        active = parent != rowsById.end() && activeRows.lookup(row.parent) &&
                 !(parent->second->kind == "repeat" &&
                   parent->second->repeatCount.value_or(0) == 0);
      }
      activeRows[row.id] = active;
      rowsById[row.id] = &row;
      if (active)
        makespan = std::max(makespan, row.finish());
    }
    LogicalResult result =
        phase("emit", [&] { return emitSchedule(makespan); });
    if (succeeded(result) && profile)
      llvm::errs() << "phys-schedule: done\n";
    return result;
  }

  qlx::phys::NativeScheduleView view() const { return {rows, resourceLabels}; }

  StringRef getResultName() const { return resultName; }

  LogicalResult verifyTypedClaims(qlx::phys::ScheduleVerificationStats &stats) {
    if (!schedule)
      return module.emitError(
          "native scheduler has no retained schedule to verify");
    SmallVector<qlx::phys::ScheduleClaim, 0> claims;
    SmallVector<std::string, 0> syntheticResourceLabels;
    claims.reserve(rows.size());
    syntheticResourceLabels.reserve(rows.size());
    for (const ScheduleRow &row : rows) {
      qlx::phys::ScheduleClaim claim;
      claim.id = row.id;
      claim.kind = row.kind;
      claim.start = row.start;
      claim.duration = row.duration;
      if (row.resourceIds.empty()) {
        syntheticResourceLabels.push_back(("control:" + row.id).str());
        claim.resources.push_back(syntheticResourceLabels.back());
      } else {
        for (ResourceId resource : row.resourceIds) {
          if (resource >= resourceLabels.size())
            return schedule.emitOpError(
                "native schedule row names an invalid resource identity");
          claim.resources.push_back(resourceLabels[resource]);
        }
      }
      llvm::append_range(claim.dependencies, row.dependencies);
      llvm::append_range(claim.dataDependencies, row.dataDependencies);
      llvm::append_range(claim.resourceDependencies, row.resourceDependencies);
      llvm::append_range(claim.domainDependencies, row.domainDependencies);
      claim.parent = row.parent;
      claim.branch = row.branch;
      claim.condition = row.condition;
      claim.maxAttempts = row.maxAttempts;
      claim.commitPoint = row.commitPoint;
      claim.repeatCount = row.repeatCount;
      claim.repeatPeriod = row.repeatPeriod;
      claim.repeatEpilogue = row.repeatEpilogue;
      claim.maxIterations = row.maxIterations;
      claim.callee = row.callee;
      claim.instance = row.instance;
      claim.profile = row.profile;
      claim.templateEvent = row.templateEvent;
      claim.attempt = row.attempt;
      claim.attemptEvent = row.attemptEvent;
      claim.decisionEvent = row.decisionEvent;
      claim.exhaustion = row.exhaustion;
      claim.successProbability = row.successProbability;
      claim.successProbabilitySource = row.successProbabilitySource;
      claim.successProbabilityEvidence = row.successProbabilityEvidence;
      // Native serialization always carries all three exact dependency
      // partitions, including an explicitly empty partition.
      claim.hasDataDependencies = true;
      claim.hasResourceDependencies = true;
      claim.hasDomainDependencies = true;
      claims.push_back(std::move(claim));
    }
    if (failed(qlx::phys::verifyScheduleClaims(schedule, claims, &stats)))
      return graph.emitOpError(
          "native greedy-ASAP schedule failed independent verification");
    return success();
  }

  void eraseTransientSchedule() {
    if (!schedule)
      return;
    schedule.erase();
    schedule = {};
  }

private:
  ResourceId registerResourceKey(StringRef key) {
    auto [entry, inserted] =
        resourceIdsByKey.try_emplace(key, resourceLabels.size());
    if (inserted)
      resourceLabels.push_back(key.str());
    return entry->second;
  }

  bool resourceLabelLess(ResourceId left, ResourceId right) const {
    assert(left < resourceLabelRanks.size() &&
           right < resourceLabelRanks.size() &&
           "schedule resource registered after canonical rank construction");
    return resourceLabelRanks[left] < resourceLabelRanks[right];
  }

  LogicalResult resolve() {
    if (!requestedGraph.empty()) {
      graph = symbols.lookup<qlx::phys::GraphOp>(requestedGraph);
      if (!graph)
        return module.emitError("phys-schedule graph @")
               << requestedGraph << " must resolve to phys.graph";
    } else {
      for (qlx::phys::GraphOp candidate : module.getOps<qlx::phys::GraphOp>()) {
        if (graph)
          return module.emitError(
              "phys-schedule requires graph= when several graphs exist");
        graph = candidate;
      }
      if (!graph)
        return module.emitError("phys-schedule requires one phys.graph");
    }
    resultName = requestedResult.empty()
                     ? (graph.getSymName() + "_schedule").str()
                     : requestedResult;
    if (SymbolTable::lookupSymbolIn(module, resultName))
      return module.emitError("phys-schedule result @")
             << resultName << " already exists";

    module.walk([&](qlx::phys::AllocationMappingOp mapping) {
      Operation *mappedGraph =
          SymbolTable::lookupNearestSymbolFrom(mapping, mapping.getGraphAttr());
      if (mappedGraph != graph.getOperation())
        return;
      for (Attribute raw : mapping.getEntries()) {
        auto record = dyn_cast<DictionaryAttr>(raw);
        auto acquire = record ? record.getAs<StringAttr>("acquire") : nullptr;
        auto after = record ? record.getAs<ArrayAttr>("after") : nullptr;
        if (!acquire || !after)
          continue;
        auto &required = allocationAfterDependencies[acquire.getValue()];
        for (Attribute rawPredecessor : after) {
          auto predecessor = dyn_cast<StringAttr>(rawPredecessor);
          if (predecessor &&
              !llvm::is_contained(required, predecessor.getValue()))
            required.push_back(predecessor.getValue());
        }
      }
    });

    for (qlx::phys::ResourceOp resource :
         module.getOps<qlx::phys::ResourceOp>()) {
      std::string key = (resource.getResourceClass() + "[" +
                         std::to_string(resource.getIndex()) + "]")
                            .str();
      ResourceId identity = registerResourceKey(key);
      if (!llvm::is_contained(physicalResourceIds, identity))
        physicalResourceIds.push_back(identity);
      resourceKeys[resource.getSymName()] = key;
      resourceIdsBySymbol[resource.getSymName()] = identity;
      auto reference = FlatSymbolRefAttr::get(context, resource.getSymName());
      resourceIdsByStateType[qlx::phys::StateType::get(context, reference)] =
          identity;
      if (auto distance = resource.getCodeDistanceAttr())
        codeDistancesByStateType[qlx::phys::StateType::get(
            context, reference)] = distance.getInt();
    }
    graph.walk([&](qlx::phys::ResourceRequestOp request) {
      if (auto binding = request.getPhysicalBindingAttr())
        registerResourceKey(
            ("binding:" + binding.getLeafReference().getValue()).str());
    });
    LogicalResult registeredFactoryResources = success();
    WalkResult registeredFactories = graph.walk([&](Operation *operation) {
      auto model = factoryModel(operation);
      if (failed(model)) {
        registeredFactoryResources = failure();
        return WalkResult::interrupt();
      }
      if (!*model)
        return WalkResult::advance();
      ResourceId identity =
          registerResourceKey(("factory:" + (*model).getSymName()).str());
      if (!llvm::is_contained(factoryResourceIds, identity))
        factoryResourceIds.push_back(identity);
      return WalkResult::advance();
    });
    if (registeredFactories.wasInterrupted() ||
        failed(registeredFactoryResources))
      return failure();
    LogicalResult registeredModelResources = success();
    WalkResult registeredModels = graph.walk([&](Operation *operation) {
      if (auto transport =
              dyn_cast<qlx::phys::TransportResourceOp>(operation)) {
        if (failed(preRegisterTransportResources(transport))) {
          registeredModelResources = failure();
          return WalkResult::interrupt();
        }
      }
      auto invocation = dyn_cast<qlx::phys::SpacetimeCallOp>(operation);
      if (!invocation)
        return WalkResult::advance();
      auto plan = symbols.lookup<qlx::phys::SpacetimePlanOp>(
          invocation.getPlanAttr().getValue());
      if (!plan) {
        invocation.emitOpError("references unresolved spacetime plan @")
            << invocation.getPlan();
        registeredModelResources = failure();
        return WalkResult::interrupt();
      }
      for (qlx::phys::SpacetimePhaseOp phase :
           plan.getBody().front().getOps<qlx::phys::SpacetimePhaseOp>())
        if (failed(spacetimePhaseResources(phase))) {
          registeredModelResources = failure();
          return WalkResult::interrupt();
        }
      return WalkResult::advance();
    });
    if (registeredModels.wasInterrupted() || failed(registeredModelResources))
      return failure();
    clockResource = registerResourceKey(clockKey);
    SmallVector<ResourceId, 0> resourcesByLabel(resourceLabels.size());
    std::iota(resourcesByLabel.begin(), resourcesByLabel.end(), ResourceId{0});
    llvm::sort(resourcesByLabel, [&](ResourceId left, ResourceId right) {
      return resourceLabels[left] < resourceLabels[right];
    });
    resourceLabelRanks.resize(resourceLabels.size());
    for (auto [rank, resource] : llvm::enumerate(resourcesByLabel))
      resourceLabelRanks[resource] = rank;
    llvm::sort(physicalResourceIds, [&](ResourceId left, ResourceId right) {
      return resourceLabelLess(left, right);
    });
    llvm::sort(factoryResourceIds, [&](ResourceId left, ResourceId right) {
      return resourceLabelLess(left, right);
    });
    state.resources.resize(resourceLabels.size());

    if (auto pointRef =
            graph->getAttrOfType<FlatSymbolRefAttr>("operating_point")) {
      auto point = dyn_cast_or_null<qlx::phys::OperatingPointOp>(
          symbols.lookup(pointRef.getValue()));
      if (!point || point.getMachineAttr() != graph.getArchitectureAttr())
        return graph.emitOpError(
            "operating_point must resolve to this graph's architecture");
      if (auto timing = point.getTimingAttr())
        for (NamedAttribute item : timing) {
          StringRef name = item.getName().getValue();
          if (name != "cycle_ns" && !name.ends_with("_ns"))
            continue;
          auto value = numericValue(item.getValue());
          if (!value || !std::isfinite(*value) || *value < 0.0)
            return point.emitOpError("timing fact '")
                   << name << "' must be finite and nonnegative";
          timings[name] = *value;
        }
    }
    if (timings.contains("cycle_ns"))
      cycle = timings.lookup("cycle_ns");
    else if (timings.contains("surface_cycle_ns"))
      cycle = timings.lookup("surface_cycle_ns");
    else
      cycle = 1.0;
    return success();
  }

  FailureOr<std::optional<ResourceId>> resourceId(Type type, Operation *owner) {
    auto stateType = dyn_cast<qlx::phys::StateType>(type);
    if (!stateType)
      return std::optional<ResourceId>{};
    auto found = resourceIdsByStateType.find(type);
    if (found == resourceIdsByStateType.end()) {
      owner->emitOpError("uses unresolved physical resource ")
          << stateType.getResource();
      return failure();
    }
    return std::optional<ResourceId>{found->second};
  }

  FailureOr<OperationScheduleInputs>
  analyzeOperation(Operation *operation, bool trackInputPositions = false) {
    OperationScheduleInputs result;
    llvm::SmallDenseSet<ResourceId, 8> seen;
    llvm::SmallDenseSet<ResourceId, 8> seenInputPositions;
    auto appendResource = [&](Type type) -> LogicalResult {
      if (!isa<qlx::phys::StateType>(type))
        result.stateOnlyBoundary = false;
      auto resource = resourceId(type, operation);
      if (failed(resource))
        return failure();
      if (*resource && seen.insert(**resource).second)
        result.resourceIds.push_back(**resource);
      return success();
    };
    for (auto [position, operand] : llvm::enumerate(operation->getOperands())) {
      if (failed(appendResource(operand.getType())))
        return failure();
      if (trackInputPositions)
        if (isa<qlx::phys::StateType>(operand.getType())) {
          auto resource = resourceIdsByStateType.find(operand.getType());
          if (resource != resourceIdsByStateType.end() &&
              seenInputPositions.insert(resource->second).second)
            result.inputResourcePositions.emplace_back(resource->second,
                                                       position);
        }
      ScheduledResourceState valueState = scheduleState(operand);
      result.operandReady = std::max(result.operandReady, valueState.ready);
      if (!valueState.producer.empty() &&
          !llvm::is_contained(result.dataDependencies, valueState.producer))
        result.dataDependencies.push_back(valueState.producer);
    }
    for (Type type : operation->getResultTypes()) {
      if (failed(appendResource(type)))
        return failure();
    }
    auto appendFactory = [&](Operation *candidate) -> LogicalResult {
      auto model = factoryModel(candidate);
      if (failed(model))
        return failure();
      if (*model) {
        std::string key = ("factory:" + (*model).getSymName()).str();
        ResourceId resource = registerResourceKey(key);
        if (seen.insert(resource).second)
          result.resourceIds.push_back(resource);
      }
      return success();
    };
    if (failed(appendFactory(operation)))
      return failure();
    if (auto transport = dyn_cast<qlx::phys::TransportResourceOp>(operation)) {
      auto selected = transportResources(transport);
      if (failed(selected))
        return failure();
      for (ResourceId resource : *selected)
        if (seen.insert(resource).second)
          result.resourceIds.push_back(resource);
    }
    if (auto invocation = dyn_cast<qlx::phys::SpacetimeCallOp>(operation)) {
      auto plan = symbols.lookup<qlx::phys::SpacetimePlanOp>(
          invocation.getPlanAttr().getValue());
      if (!plan) {
        invocation.emitOpError("references unresolved spacetime plan @")
            << invocation.getPlan();
        return failure();
      }
      for (qlx::phys::SpacetimePhaseOp phase :
           plan.getBody().front().getOps<qlx::phys::SpacetimePhaseOp>()) {
        auto phaseResources = spacetimePhaseResources(phase);
        if (failed(phaseResources))
          return failure();
        for (ResourceId resource : *phaseResources)
          if (seen.insert(resource).second)
            result.resourceIds.push_back(resource);
      }
    }
    if (isa<qlx::cflow::RepeatOp>(operation)) {
      WalkResult walked = operation->walk([&](Operation *candidate) {
        if (failed(appendFactory(candidate)))
          return WalkResult::interrupt();
        auto invocation = dyn_cast<qlx::phys::SpacetimeCallOp>(candidate);
        if (!invocation)
          return WalkResult::advance();
        auto plan = symbols.lookup<qlx::phys::SpacetimePlanOp>(
            invocation.getPlanAttr().getValue());
        if (!plan) {
          invocation.emitOpError("references unresolved spacetime plan @")
              << invocation.getPlan();
          return WalkResult::interrupt();
        }
        for (qlx::phys::SpacetimePhaseOp phase :
             plan.getBody().front().getOps<qlx::phys::SpacetimePhaseOp>()) {
          auto claims = spacetimePhaseResources(phase);
          if (failed(claims))
            return WalkResult::interrupt();
          for (ResourceId claim : *claims)
            if (StringRef(resourceLabels[claim]).starts_with("factory:") &&
                seen.insert(claim).second)
              result.resourceIds.push_back(claim);
        }
        return WalkResult::advance();
      });
      if (walked.wasInterrupted())
        return failure();
    }
    if (auto request = dyn_cast<qlx::phys::ResourceRequestOp>(operation))
      if (auto binding = request.getPhysicalBindingAttr()) {
        std::string key =
            ("binding:" + binding.getLeafReference().getValue()).str();
        auto resource = resourceIdsByKey.find(key);
        if (resource == resourceIdsByKey.end()) {
          operation->emitOpError("uses unresolved schedule resource '")
              << key << "'";
          return failure();
        }
        if (seen.insert(resource->second).second)
          result.resourceIds.push_back(resource->second);
      }
    return result;
  }

  FailureOr<const DenseMap<ResourceId, ResourceId> *>
  templateResourceAliases(qlx::phys::CallTemplateOp invocation) {
    auto values = invocation->getAttrOfType<ArrayAttr>("state_aliases");
    if (!values)
      return &emptyTemplateResourceAliases;
    auto cached = templateResourceAliasCacheIds.find(values);
    if (cached != templateResourceAliasCacheIds.end())
      return &templateResourceAliasMaps[cached->second];
    DenseMap<ResourceId, ResourceId> aliases;
    for (Attribute raw : values) {
      auto entry = cast<DictionaryAttr>(raw);
      auto templateResource = entry.getAs<FlatSymbolRefAttr>("template");
      auto invocationResource = entry.getAs<FlatSymbolRefAttr>("alias");
      auto templateKey = resourceIdsBySymbol.find(templateResource.getValue());
      auto invocationKey =
          resourceIdsBySymbol.find(invocationResource.getValue());
      if (templateKey == resourceIdsBySymbol.end() ||
          invocationKey == resourceIdsBySymbol.end()) {
        invocation.emitOpError(
            "state alias references an unresolved scheduled resource");
        return failure();
      }
      aliases[templateKey->second] = invocationKey->second;
    }
    unsigned id = templateResourceAliasMaps.size();
    templateResourceAliasMaps.push_back(std::move(aliases));
    templateResourceAliasCacheIds.try_emplace(values, id);
    return &templateResourceAliasMaps.back();
  }

  FailureOr<const MappedCallTemplateSummary *>
  mappedCallTemplate(qlx::phys::CallTemplateOp invocation,
                     const CallTemplateSummary &canonical) {
    std::pair<Attribute, Attribute> key = {
        invocation->getAttr("template_event"),
        invocation->getAttr("state_aliases")};
    auto cached = mappedCallTemplateCacheIds.find(key);
    if (cached != mappedCallTemplateCacheIds.end())
      return &mappedCallTemplateSummaries[cached->second];
    auto aliases = templateResourceAliases(invocation);
    if (failed(aliases))
      return failure();
    auto remap = [&](ResourceId resource) {
      auto alias = (*aliases)->find(resource);
      return alias == (*aliases)->end() ? resource : alias->second;
    };

    MappedCallTemplateSummary mapped;
    mapped.resourceAliases = *aliases;
    llvm::SmallDenseSet<ResourceId, 32> closureSeen;
    auto appendClosure = [&](ResourceId resource) {
      if (closureSeen.insert(resource).second)
        mapped.closureResources.push_back(resource);
    };
    for (ResourceId resource : canonical.descendantResources)
      appendClosure(remap(resource));
    if (canonical.globalBarrierFirstUseOffset)
      for (ResourceId resource : physicalResourceIds)
        appendClosure(resource);
    for (const auto &[resource, offset] : canonical.firstUseOffsets) {
      ResourceId invocationResource = remap(resource);
      appendClosure(invocationResource);
      mapped.firstUseOffsets.emplace_back(invocationResource, offset);
    }
    llvm::sort(mapped.closureResources, [&](ResourceId left, ResourceId right) {
      return resourceLabelLess(left, right);
    });

    mapped.elidedClosureResources = mapped.closureResources;
    for (const auto &[resource, position] : canonical.inputResourcePositions) {
      (void)position;
      ResourceId invocationResource = remap(resource);
      mapped.inputResources.push_back(invocationResource);
      if (closureSeen.insert(invocationResource).second)
        mapped.elidedClosureResources.push_back(invocationResource);
    }
    llvm::sort(mapped.elidedClosureResources,
               [&](ResourceId left, ResourceId right) {
                 return resourceLabelLess(left, right);
               });
    for (const CallResourceAvailability &effect :
         canonical.availabilityOffsets) {
      CallResourceAvailability replayed = effect;
      replayed.resource = remap(effect.resource);
      mapped.availabilityOffsets.push_back(replayed);
    }

    unsigned id = mappedCallTemplateSummaries.size();
    mappedCallTemplateSummaries.push_back(std::move(mapped));
    mappedCallTemplateCacheIds.try_emplace(key, id);
    return &mappedCallTemplateSummaries.back();
  }

  FailureOr<OperationScheduleInputs>
  analyzeCallTemplate(qlx::phys::CallTemplateOp invocation,
                      const ScheduleState &current) {
    // The invocation's actual boundary is authoritative for ordinary SSA
    // readiness.  In particular, a retained classical result makes the
    // boundary mixed, but does not make the canonical body's physical effects
    // disappear.  Start from the complete boundary analysis and then add the
    // compact body's independently summarized resource readiness below.
    auto analyzed = analyzeOperation(invocation.getOperation());
    if (failed(analyzed))
      return failure();
    OperationScheduleInputs result = std::move(*analyzed);
    auto canonical = callTemplates.find(invocation.getTemplateEvent());
    if (canonical == callTemplates.end())
      return result;
    if (canonical->second.containsModeledTransport)
      return invocation.emitOpError(
          "call-template reuse does not support model-bound transport; keep "
          "the modeled transport call expanded");
    auto mapped = mappedCallTemplate(invocation, canonical->second);
    if (failed(mapped))
      return failure();
    // Compact templates may summarize hundreds of resources and occur
    // hundreds of thousands of times. Use indexed membership rather than
    // repeatedly scanning the growing vector for every boundary, descendant,
    // and first-use resource; the final canonical label sort remains below.
    llvm::SmallDenseSet<StringRef, 8> resourceDependenciesSeen;
    resourceDependenciesSeen.insert(result.resourceDependencies.begin(),
                                    result.resourceDependencies.end());
    auto appendResourceDependency = [&](StringRef producer) {
      if (!producer.empty() && resourceDependenciesSeen.insert(producer).second)
        result.resourceDependencies.push_back(producer);
    };
    // The compact form intentionally omits the ordinary SSA boundary, but it
    // does not omit that boundary's ownership/readiness effect. Reconstruct
    // the canonical inputs at offset zero through the independently verified
    // alias bijection.
    if (invocation->hasAttr("state_boundary_elided"))
      for (ResourceId resource : (*mapped)->inputResources) {
        result.operandReady =
            std::max(result.operandReady, readyAt(current, resource));
        appendResourceDependency(producedBy(current, resource));
      }
    // The compact envelope must retain the exact physical-resource closure of
    // the canonical body's emitted rows, including resources that do not
    // participate in the active first-use/availability frontier (for example,
    // an inspectable zero-repeat body). This is a proof claim only: readiness
    // continues to come exclusively from firstUseOffsets below.
    const auto &cachedClosure = invocation->hasAttr("state_boundary_elided")
                                    ? (*mapped)->elidedClosureResources
                                    : (*mapped)->closureResources;
    if (result.resourceIds.empty()) {
      result.resourceIds.assign(cachedClosure.begin(), cachedClosure.end());
    } else {
      llvm::SmallDenseSet<ResourceId, 32> resourceIdsSeen;
      resourceIdsSeen.insert(result.resourceIds.begin(),
                             result.resourceIds.end());
      for (ResourceId resource : cachedClosure)
        if (resourceIdsSeen.insert(resource).second)
          result.resourceIds.push_back(resource);
      llvm::sort(result.resourceIds, [&](ResourceId left, ResourceId right) {
        return resourceLabelLess(left, right);
      });
    }
    // A zero-operand clock barrier is a domain-global effect over concrete
    // physical resources, not an effect in the canonical call's state-symbol
    // namespace. Keep its closure and readiness separate so state_aliases
    // cannot project away physical identities or pull opaque binding/factory
    // keys into the envelope.
    if (canonical->second.globalBarrierFirstUseOffset) {
      double offset = *canonical->second.globalBarrierFirstUseOffset;
      for (ResourceId resource : physicalResourceIds) {
        result.operandReady =
            std::max(result.operandReady, readyAt(current, resource) - offset);
        if (offset == 0.0)
          appendResourceDependency(producedBy(current, resource));
      }
    }
    // A compact invocation has no rows for the canonical body's internal
    // resources.  Retain the canonical first-use offset for each such
    // resource and translate the caller frontier into the earliest legal
    // envelope start.  A zero-offset owner is also an ordinary start
    // dependency; positive offsets permit independent prefix work to overlap.
    for (const auto &[resource, offset] : (*mapped)->firstUseOffsets) {
      result.operandReady =
          std::max(result.operandReady, readyAt(current, resource) - offset);
      if (offset == 0.0)
        appendResourceDependency(producedBy(current, resource));
    }
    for (const CallPipelineTransition &transition :
         canonical->second.pipelineTransitions) {
      auto prior = pipelineInitiation.find(transition.plan);
      if (prior == pipelineInitiation.end())
        continue;
      result.operandReady = std::max(
          result.operandReady, prior->second.ready - transition.firstUseOffset);
      appendResourceDependency(prior->second.producer);
    }
    result.mappedCallTemplate = *mapped;
    return result;
  }

  FailureOr<double> durationOf(Operation *operation) {
    auto model = factoryModel(operation);
    if (failed(model))
      return failure();
    if (*model) {
      if (isa<qlx::phys::FactoryStartOp>(operation)) {
        std::string key =
            ("factory_model." + (*model).getSymName() + ".startup_ns").str();
        double value = (*model).getStartupNs().convertToDouble();
        usedTimings[key] = value;
        return value;
      }
      if (isa<qlx::phys::ResourceRequestOp>(operation)) {
        std::string key =
            ("factory_model." + (*model).getSymName() + ".output_interval_ns")
                .str();
        usedTimings[key] = (*model).getOutputIntervalNs().convertToDouble();
        return 0.0;
      }
    }
    if (auto transport = dyn_cast<qlx::phys::TransportResourceOp>(operation)) {
      auto selected = transportModel(transport);
      if (failed(selected))
        return failure();
      if (*selected) {
        std::string key =
            ("transport_model." + (*selected).getSymName() + ".latency_ns")
                .str();
        double value = (*selected).getLatencyNs().convertToDouble();
        usedTimings[key] = value;
        return value;
      }
      auto binding = transportBinding(transport);
      if (failed(binding))
        return failure();
      if (*binding) {
        auto timing = timings.find("transport_resource_ns");
        if (timing != timings.end()) {
          usedTimings["transport_resource_ns"] = timing->second;
          return timing->second;
        }
      }
    }
    if (Attribute duration = operation->getAttr("duration_ns")) {
      auto value = numericValue(duration);
      if (!value || !std::isfinite(*value) || *value < 0.0) {
        operation->emitOpError("has invalid duration_ns");
        return failure();
      }
      return *value;
    }
    if (isa<qlx::phys::AcquireOp, qlx::phys::ReleaseOp, qlx::phys::RetryOp,
            qlx::event::FenceOp, qlx::phys::BarrierOp>(operation))
      return 0.0;
    std::optional<int64_t> codeDistance;
    bool unresolvedPhysicalDistance = false;
    for (Type type : operation->getOperandTypes()) {
      if (!isa<qlx::phys::StateType>(type))
        continue;
      auto found = codeDistancesByStateType.find(type);
      if (found == codeDistancesByStateType.end()) {
        unresolvedPhysicalDistance = true;
        continue;
      }
      if (codeDistance && *codeDistance != found->second) {
        operation->emitOpError(
            "uses physical states with different code distances; add an "
            "explicit duration_ns for this mixed-distance event");
        return failure();
      }
      codeDistance = found->second;
    }
    auto hasQualifiedTiming = [&](StringRef action) {
      std::string prefix = (action + "_d").str();
      for (const auto &entry : timings)
        if (entry.getKey().starts_with(prefix) &&
            entry.getKey().ends_with("_ns"))
          return true;
      return false;
    };
    auto resolve = [&](StringRef action) -> FailureOr<double> {
      if (!action.empty() && hasQualifiedTiming(action)) {
        if (!codeDistance || unresolvedPhysicalDistance) {
          operation->emitOpError("requires an authenticated code distance to "
                                 "select timing for '")
              << action << "'";
          return failure();
        }
        std::string qualified =
            (action + "_d" + std::to_string(*codeDistance) + "_ns").str();
        auto timing = timings.find(qualified);
        if (timing == timings.end()) {
          operation->emitOpError("has no timing for '")
              << action << "' at code distance " << *codeDistance;
          return failure();
        }
        usedTimings[qualified] = timing->second;
        return timing->second;
      }
      std::string timingName = (action + "_ns").str();
      auto timing = timings.find(timingName);
      if (!action.empty() && timing != timings.end()) {
        usedTimings[timingName] = timing->second;
        return timing->second;
      }
      usedTimings["cycle_ns"] = cycle;
      return cycle;
    };
    Attribute actionAttribute;
    for (StringRef key : {"action", "instrument", "measurement", "route"})
      if (Attribute value = operation->getAttr(key)) {
        actionAttribute = value;
        break;
      }
    if (actionAttribute) {
      return resolve(symbolText(actionAttribute));
    }
    StringRef operationName = operation->getName().getStringRef();
    StringRef action = llvm::StringSwitch<StringRef>(operationName)
                           .Case("phys.measure_product", "mpp")
                           .Case("phys.rotate_product", "rpp")
                           .Case("phys.resource_rotate_product", "resource_rpp")
                           .Default("");
    if (action.empty())
      (void)operationName.consume_front("phys.");
    return resolve(action.empty() ? operationName : action);
  }

  ScheduledResourceState scheduleState(Value value) const {
    auto result = dyn_cast<OpResult>(value);
    if (result) {
      auto state = operationResultStates.find(result.getOwner());
      if (state != operationResultStates.end())
        return {state->second.ready, state->second.producer};
      return {};
    }
    ScheduledResourceState state;
    auto ready = valueReadyTimes.find(value);
    if (ready != valueReadyTimes.end())
      state.ready = ready->second;
    auto producer = valueProducers.find(value);
    if (producer != valueProducers.end())
      state.producer = producer->second;
    return state;
  }

  double valueReady(Value value) const { return scheduleState(value).ready; }

  StringRef valueProducer(Value value) const {
    return scheduleState(value).producer;
  }

  static bool resourceChanged(const ScheduleState &before,
                              const ScheduleState &after, ResourceId resource) {
    return readyAt(before, resource) != readyAt(after, resource) ||
           producedBy(before, resource) != producedBy(after, resource) ||
           pipelinePlanAt(before, resource) !=
               pipelinePlanAt(after, resource) ||
           pipelineInvocationAt(before, resource) !=
               pipelineInvocationAt(after, resource);
  }

  void setResourceState(ScheduleState &current, ResourceId resource,
                        double ready, StringRef producer,
                        StringRef pipelinePlan = {},
                        StringRef pipelineInvocation = {}) {
    if (!resourceChangeScopes.empty()) {
      ResourceChangeScope *scope = resourceChangeScopes.back();
      scope->previous.try_emplace(
          resource,
          ScheduledResourceState{readyAt(current, resource),
                                 producedBy(current, resource),
                                 pipelinePlanAt(current, resource),
                                 pipelineInvocationAt(current, resource)});
    }
    if (current.resources.size() <= resource)
      current.resources.resize(resource + 1);
    current.resources[resource] = {ready, producer, pipelinePlan,
                                   pipelineInvocation};
  }

  void setPipelineInitiation(StringRef plan, double ready, StringRef producer,
                             double firstUse) {
    if (!pipelineChangeScopes.empty()) {
      PipelineChangeScope *scope = pipelineChangeScopes.back();
      if (!scope->previous.contains(plan)) {
        auto found = pipelineInitiation.find(plan);
        scope->previous.try_emplace(
            plan, found == pipelineInitiation.end()
                      ? std::optional<OperationResultState>{}
                      : std::optional<OperationResultState>{found->second});
      }
      auto [entry, inserted] = scope->firstUses.try_emplace(plan, firstUse);
      if (!inserted)
        entry->second = std::min(entry->second, firstUse);
    }
    pipelineInitiation[plan] = {ready, producer};
  }

  void noteResourceUse(ResourceId resource, double at) {
    if (resourceChangeScopes.empty())
      return;
    ResourceChangeScope *scope = resourceChangeScopes.back();
    auto [entry, inserted] = scope->firstUses.try_emplace(resource, at);
    if (!inserted)
      entry->second = std::min(entry->second, at);
  }

  void mergeResourceUses(const ResourceChangeScope &changes) {
    if (resourceChangeScopes.empty())
      return;
    ResourceChangeScope *parent = resourceChangeScopes.back();
    for (const auto &[resource, firstUse] : changes.firstUses) {
      auto [entry, inserted] =
          parent->firstUses.try_emplace(resource, firstUse);
      if (!inserted)
        entry->second = std::min(entry->second, firstUse);
    }
  }

  void noteGlobalBarrier(double firstUse, double availability) {
    if (resourceChangeScopes.empty())
      return;
    ResourceChangeScope *scope = resourceChangeScopes.back();
    scope->globalBarrierFirstUse =
        scope->globalBarrierFirstUse
            ? std::min(*scope->globalBarrierFirstUse, firstUse)
            : firstUse;
    scope->globalBarrierAvailability =
        scope->globalBarrierAvailability
            ? std::max(*scope->globalBarrierAvailability, availability)
            : availability;
  }

  void mergeGlobalBarrierUses(const ResourceChangeScope &changes) {
    if (changes.globalBarrierFirstUse && changes.globalBarrierAvailability)
      noteGlobalBarrier(*changes.globalBarrierFirstUse,
                        *changes.globalBarrierAvailability);
  }

  ResourceStateDelta
  captureResourceChangesAndRestore(ScheduleState &current,
                                   const ResourceChangeScope &changes) {
    ResourceStateDelta final;
    for (const auto &[resource, previous] : changes.previous) {
      ScheduledResourceState next{readyAt(current, resource),
                                  producedBy(current, resource),
                                  pipelinePlanAt(current, resource),
                                  pipelineInvocationAt(current, resource)};
      if (next.ready != previous.ready || next.producer != previous.producer ||
          next.pipelinePlan != previous.pipelinePlan ||
          next.pipelineInvocation != previous.pipelineInvocation)
        final.try_emplace(resource, next);
      current.resources[resource] = previous;
    }
    return final;
  }

  static ScheduledResourceState
  branchResourceState(const ResourceStateDelta &changes,
                      const ScheduleState &before, ResourceId resource) {
    auto changed = changes.find(resource);
    if (changed != changes.end())
      return changed->second;
    return {readyAt(before, resource), producedBy(before, resource),
            pipelinePlanAt(before, resource),
            pipelineInvocationAt(before, resource)};
  }

  void bindArguments(Block &block, ValueRange operands, double start,
                     bool skipYieldPassThrough = false) {
    Operation *terminator = block.getTerminator();
    size_t position = 0;
    for (auto [argument, operand] : llvm::zip(block.getArguments(), operands)) {
      if (skipYieldPassThrough &&
          isa<qlx::phys::StateType>(argument.getType()) && terminator &&
          position < terminator->getNumOperands() &&
          terminator->getOperand(position) == argument &&
          argument.hasOneUse()) {
        OpOperand &use = *argument.use_begin();
        if (use.getOwner() == terminator &&
            use.getOperandNumber() == position) {
          ++position;
          continue;
        }
      }
      ScheduledResourceState operandState = scheduleState(operand);
      valueReadyTimes[argument] = operandState.ready;
      if (!operandState.producer.empty())
        valueProducers[argument] = operandState.producer;
      else
        valueReadyTimes[argument] = start;
      ++position;
    }
  }

  static SmallVector<Value, 4> yieldedValues(Block &block) {
    Operation *terminator = block.getTerminator();
    if (!terminator)
      return {};
    if (auto condition = dyn_cast<qlx::cflow::WhileConditionOp>(terminator))
      return SmallVector<Value, 4>(condition.getForwarded().begin(),
                                   condition.getForwarded().end());
    return SmallVector<Value, 4>(terminator->getOperands().begin(),
                                 terminator->getOperands().end());
  }

  FailureOr<StringRef> eventIdentity(Operation *operation) {
    auto event = operation->getAttrOfType<StringAttr>("event_id");
    if (!event || event.getValue().empty()) {
      operation->emitOpError(
          "schedulable event is missing its stable event_id");
      return failure();
    }
    return event.getValue();
  }

  LogicalResult scheduleBlock(Block &block, ScheduleState &current,
                              SmallVectorImpl<ScheduleRow> &sink,
                              StringRef parent = {}, StringRef branch = {},
                              StringRef inheritedCondition = {},
                              double earliest = 0.0,
                              int64_t repeatMultiplicity = 1,
                              double *directFinish = nullptr) {
    for (Operation &operation : block) {
      std::chrono::steady_clock::time_point operationStarted;
      if (profileEnabled)
        operationStarted = std::chrono::steady_clock::now();
      ++scheduledOperations;
      if (profileEnabled && scheduledOperations % 10000 == 0) {
        auto elapsed = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - profileStarted)
                           .count();
        llvm::errs() << "phys-schedule: operations=" << scheduledOperations
                     << " calls=" << scheduledCalls
                     << " templates=" << scheduledTemplateCalls
                     << " rows=" << rows.size() << " elapsed=" << elapsed
                     << " current=" << operation.getName() << "\n";
      }
      StringRef name = operation.getName().getStringRef();
      if (isa<qlx::phys::ReturnOp, qlx::phys::YieldOp, qlx::cflow::YieldOp,
              qlx::cflow::WhileConditionOp, qlx::event::YieldOp>(operation))
        continue;
      if (operation.getNumRegions() != 0 &&
          !isa<qlx::phys::CallOp, qlx::cflow::RepeatOp, qlx::cflow::WhileOp,
               qlx::cflow::IfOp, qlx::event::TryTakeOp>(operation))
        return operation.emitOpError(
                   "physical scheduling does not support region control ")
               << name;
      if (!name.starts_with("phys.") && !operation.hasAttr("event_id")) {
        double ready = 0.0;
        for (Value operand : operation.getOperands())
          ready = std::max(ready, valueReady(operand));
        if (operation.getNumResults() != 0)
          operationResultStates[&operation] = {ready, {}};
        continue;
      }

      auto event = eventIdentity(&operation);
      auto templateInvocation = dyn_cast<qlx::phys::CallTemplateOp>(operation);
      auto inputs =
          templateInvocation
              ? analyzeCallTemplate(templateInvocation, current)
              : analyzeOperation(&operation, isa<qlx::phys::CallOp>(operation));
      if (failed(event) || failed(inputs))
        return failure();
      size_t directRowIndex = sink.size();
      auto noteDirectFinish = [&] {
        if (directFinish)
          *directFinish =
              std::max(*directFinish, sink[directRowIndex].finish());
      };
      SmallVector<ResourceId, 4> resourceIds = std::move(inputs->resourceIds);
      bool envelope =
          isa<qlx::phys::CallOp, qlx::phys::CallTemplateOp,
              qlx::phys::SpacetimeCallOp, qlx::cflow::RepeatOp,
              qlx::cflow::WhileOp, qlx::cflow::IfOp, qlx::event::TryTakeOp>(
              operation);
      double operandReady = inputs->operandReady;
      double resourceReady = 0.0;
      if (!envelope)
        for (ResourceId resource : resourceIds)
          resourceReady = std::max(resourceReady, readyAt(current, resource));
      if (isa<qlx::cflow::RepeatOp>(operation))
        for (ResourceId resource : resourceIds)
          if (StringRef(resourceLabels[resource]).starts_with("factory:"))
            resourceReady = std::max(resourceReady, readyAt(current, resource));

      auto domains = operation.getAttrOfType<ArrayAttr>("domains");
      auto hasDomain = [&](StringRef name) {
        return domains && llvm::any_of(domains, [&](Attribute value) {
                 auto text = dyn_cast<StringAttr>(value);
                 return text && text.getValue() == name;
               });
      };
      bool clockBarrier = isa<qlx::phys::BarrierOp>(operation) &&
                          operation.getNumOperands() == 0 && hasDomain("clock");
      bool factoryBarrier = clockBarrier && hasDomain("factory");
      if (factoryBarrier)
        for (ResourceId resource : factoryResourceIds) {
          if (!llvm::is_contained(resourceIds, resource))
            resourceIds.push_back(resource);
          resourceReady = std::max(resourceReady, readyAt(current, resource));
        }
      double clockReady = readyAt(current, clockResource);
      StringRef clockProducer = producedBy(current, clockResource);
      SmallVector<ResourceId, 8> synchronized;
      if (clockBarrier)
        for (ResourceId resource : physicalResourceIds) {
          synchronized.push_back(resource);
          resourceReady = std::max(resourceReady, readyAt(current, resource));
        }
      if (factoryBarrier)
        for (ResourceId resource : factoryResourceIds)
          if (!llvm::is_contained(synchronized, resource))
            synchronized.push_back(resource);
      // analyzeCallTemplate already canonicalizes its potentially wide
      // resource closure. Avoid sorting that same closure again at every
      // compact invocation; ordinary operations still need normalization.
      if (!templateInvocation)
        llvm::sort(resourceIds, [&](ResourceId left, ResourceId right) {
          return resourceLabelLess(left, right);
        });

      // A structured envelope starts when its explicit SSA/control inputs are
      // ready.  Resources mentioned only by its result types are committed by
      // the nested operations that actually touch them; treating those result
      // types as reads delays the whole branch and defeats legal overlap.
      double start = std::max({earliest, operandReady, clockReady});
      if (!envelope) {
        start = std::max(start, resourceReady);
        if (!clockBarrier && !resourceIds.empty())
          start = std::max(start, readyAt(current, resourceIds.front()));
      }
      if (isa<qlx::cflow::RepeatOp>(operation))
        start = std::max(start, resourceReady);

      SmallVector<StringRef, 4> data = std::move(inputs->dataDependencies);
      SmallVector<StringRef, 4> resourceDeps =
          std::move(inputs->resourceDependencies);
      llvm::SmallDenseSet<StringRef, 8> resourceDepsSeen;
      resourceDepsSeen.insert(resourceDeps.begin(), resourceDeps.end());
      auto appendResourceDependency = [&](StringRef producer) {
        if (!producer.empty() && resourceDepsSeen.insert(producer).second)
          resourceDeps.push_back(producer);
      };
      SmallVector<StringRef, 4> domainDeps;
      if (clockBarrier) {
        llvm::StringSet<> seenDomainDependencies;
        for (ResourceId resource : synchronized) {
          StringRef producer = producedBy(current, resource);
          if (!producer.empty() &&
              seenDomainDependencies.insert(producer).second)
            domainDeps.push_back(producer);
        }
        if (!clockProducer.empty() &&
            seenDomainDependencies.insert(clockProducer).second)
          domainDeps.push_back(clockProducer);
      } else {
        if (!envelope)
          for (ResourceId resource : resourceIds) {
            if (StringRef(resourceLabels[resource])
                    .starts_with("control:transport-init:"))
              continue;
            appendResourceDependency(producedBy(current, resource));
          }
        if (isa<qlx::cflow::RepeatOp>(operation))
          for (ResourceId resource : resourceIds) {
            if (!StringRef(resourceLabels[resource]).starts_with("factory:"))
              continue;
            appendResourceDependency(producedBy(current, resource));
          }
        if (!clockProducer.empty())
          domainDeps.push_back(clockProducer);
      }
      auto allocationOrder = allocationAfterDependencies.find(*event);
      if (allocationOrder != allocationAfterDependencies.end())
        for (StringRef predecessor : allocationOrder->second)
          appendResourceDependency(predecessor);
      SmallVector<StringRef, 8> dependencies;
      llvm::SmallDenseSet<StringRef, 8> dependenciesSeen;
      auto appendDependencies = [&](const auto &values) {
        for (const auto &value : values)
          if (dependenciesSeen.insert(StringRef(value)).second)
            dependencies.push_back(StringRef(value));
      };
      appendDependencies(data);
      appendDependencies(resourceDeps);
      appendDependencies(domainDeps);

      ScheduleRow base;
      base.id = *event;
      base.kind = name;
      if (!base.kind.consume_front("phys.") &&
          !base.kind.consume_front("cflow."))
        base.kind.consume_front("event.");
      base.start = start;
      base.resourceIds = resourceIds;
      base.inputResourcePositions = std::move(inputs->inputResourcePositions);
      base.stateOnlyBoundary = inputs->stateOnlyBoundary;
      base.dependencies.assign(dependencies.begin(), dependencies.end());
      base.dataDependencies = data;
      base.resourceDependencies = resourceDeps;
      base.domainDependencies = domainDeps;
      base.parent = parent;
      base.branch = branch;
      base.condition = inheritedCondition;
      if (!envelope)
        for (ResourceId resource : resourceIds)
          noteResourceUse(resource, start);
      if (profileEnabled) {
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          operationStarted)
                .count();
        scheduledPreludeSeconds += elapsed;
        if (isa<qlx::phys::CallTemplateOp>(operation))
          scheduledTemplatePreludeSeconds += elapsed;
        else if (isa<qlx::phys::CallOp>(operation))
          scheduledCallPreludeSeconds += elapsed;
        else
          scheduledLeafPreludeSeconds += elapsed;
      }

      if (auto call = dyn_cast<qlx::phys::CallOp>(operation)) {
        if (failed(scheduleCall(call, current, sink, std::move(base),
                                inheritedCondition, repeatMultiplicity)))
          return failure();
        noteDirectFinish();
        continue;
      }
      if (auto invocation = dyn_cast<qlx::phys::CallTemplateOp>(operation)) {
        if (failed(scheduleCallTemplate(invocation, current, sink,
                                        std::move(base), inheritedCondition,
                                        inputs->mappedCallTemplate)))
          return failure();
        noteDirectFinish();
        continue;
      }
      if (auto invocation = dyn_cast<qlx::phys::SpacetimeCallOp>(operation)) {
        if (failed(scheduleSpacetimeCall(invocation, current, sink, base,
                                         inheritedCondition)))
          return failure();
        continue;
      }
      if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
        if (failed(scheduleRepeat(repeat, current, sink, std::move(base),
                                  inheritedCondition, repeatMultiplicity)))
          return failure();
        noteDirectFinish();
        continue;
      }
      if (auto branchOp = dyn_cast<qlx::cflow::IfOp>(operation)) {
        if (failed(scheduleIf(branchOp, current, sink, std::move(base),
                              inheritedCondition, repeatMultiplicity)))
          return failure();
        noteDirectFinish();
        continue;
      }
      if (auto dispatch = dyn_cast<qlx::event::TryTakeOp>(operation)) {
        if (failed(scheduleEventTryTake(dispatch, current, sink,
                                        std::move(base), inheritedCondition,
                                        repeatMultiplicity)))
          return failure();
        noteDirectFinish();
        continue;
      }
      if (auto loop = dyn_cast<qlx::cflow::WhileOp>(operation)) {
        if (failed(scheduleWhile(loop, current, sink, std::move(base),
                                 inheritedCondition, repeatMultiplicity)))
          return failure();
        noteDirectFinish();
        continue;
      }

      auto duration = durationOf(&operation);
      if (failed(duration))
        return failure();
      base.duration = *duration;
      if (auto retry = dyn_cast<qlx::phys::RetryOp>(operation)) {
        base.maxAttempts = retry.getMaxAttempts();
        base.exhaustion = retry.getExhaustion();
        base.commitPoint = retry.getCommitPoint().value_or("");
        base.profile = retry.getProfile();
        base.attempt = retry.getAttempt();
        base.attemptEvent = retry.getAttemptEvent();
        base.decisionEvent = retry.getDecisionEvent();
        if (auto probability = retry.getSuccessProbabilityAttr())
          base.successProbability = probability.getValueAsDouble();
        if (auto source = retry.getSuccessProbabilitySourceAttr())
          base.successProbabilitySource = source.getValue();
        if (auto evidence = retry.getSuccessProbabilityEvidenceAttr())
          base.successProbabilityEvidence = evidence.getValue();
      }
      double finish = base.finish();
      double eventStart = base.start;
      StringRef eventId = base.id;
      if (clockBarrier)
        noteGlobalBarrier(eventStart, finish);
      sink.push_back(std::move(base));
      noteDirectFinish();
      for (ResourceId resource : resourceIds)
        setResourceState(current, resource, finish, eventId);
      if (auto request = dyn_cast<qlx::phys::ResourceRequestOp>(operation)) {
        auto model = factoryModel(request);
        if (failed(model))
          return failure();
        if (*model) {
          std::string key = ("factory:" + (*model).getSymName()).str();
          auto resource = resourceIdsByKey.find(key);
          if (resource == resourceIdsByKey.end())
            return request.emitOpError(
                "has no registered factory scheduling resource");
          setResourceState(current, resource->second,
                           eventStart +
                               (*model).getOutputIntervalNs().convertToDouble(),
                           eventId);
        }
      }
      if (auto transport =
              dyn_cast<qlx::phys::TransportResourceOp>(operation)) {
        auto model = transportModel(transport);
        if (failed(model))
          return failure();
        if (*model) {
          std::string key =
              ("control:transport-init:" + (*model).getSymName()).str();
          auto resource = resourceIdsByKey.find(key);
          if (resource == resourceIdsByKey.end())
            return transport.emitOpError(
                "has no registered transport initiation resource");
          double ready =
              eventStart + (*model).getInitiationIntervalNs().convertToDouble();
          setResourceState(current, resource->second, ready, eventId);
          usedTimings[("transport_model." + (*model).getSymName() +
                       ".initiation_interval_ns")
                          .str()] =
              (*model).getInitiationIntervalNs().convertToDouble();
        } else {
          auto binding = transportBinding(transport);
          if (failed(binding))
            return failure();
          if (*binding && (*binding).getTransportClaimsAttr()) {
            std::string key =
                ("control:transport-init:" + (*binding).getSymName()).str();
            auto resource = resourceIdsByKey.find(key);
            if (resource == resourceIdsByKey.end())
              return transport.emitOpError(
                  "has no registered detailed transport initiation resource");
            double interval = finish - eventStart;
            auto selected =
                timings.find("transport_resource_initiation_interval_ns");
            if (selected != timings.end()) {
              interval = selected->second;
              usedTimings["transport_resource_initiation_interval_ns"] =
                  interval;
            }
            setResourceState(current, resource->second, eventStart + interval,
                             eventId);
          }
        }
      }
      if (clockBarrier)
        setResourceState(current, clockResource, finish, eventId);
      if (operation.getNumResults() != 0)
        operationResultStates[&operation] = {finish, eventId};
    }
    return success();
  }

  LogicalResult scheduleCall(qlx::phys::CallOp call, ScheduleState &current,
                             SmallVectorImpl<ScheduleRow> &sink,
                             ScheduleRow base, StringRef condition,
                             int64_t repeatMultiplicity) {
    std::chrono::steady_clock::time_point callStarted;
    if (profileEnabled)
      callStarted = std::chrono::steady_clock::now();
    ++scheduledCalls;
    base.callee = call.getCallee();
    base.instance = call.getInstance();
    if (auto profile = call.getProfileAttr())
      base.profile = profile.getValue();
    size_t rowIndex = sink.size();
    StringRef callId = base.id;
    double callStart = base.start;
    sink.push_back(std::move(base));

    // A call body is scheduled sequentially into the caller's frontier.  A
    // change scope journals the before-image of only resources actually
    // written by the body.  Nested calls write through every active scope, so
    // their effects remain visible to each enclosing reusable summary without
    // copying or pre-scanning a wide physical-state signature.
    ResourceChangeScope changes;
    resourceChangeScopes.push_back(&changes);
    PipelineChangeScope pipelineChanges;
    pipelineChangeScopes.push_back(&pipelineChanges);
    scheduledCallMaxDepth =
        std::max(scheduledCallMaxDepth, resourceChangeScopes.size());
    Block &bodyBlock = call.getBody().front();
    bindArguments(bodyBlock, call.getOperands(), callStart);
    double bodyDirectFinish = callStart;
    LogicalResult bodyResult =
        scheduleBlock(bodyBlock, current, sink, callId, "body", condition,
                      callStart, repeatMultiplicity, &bodyDirectFinish);
    pipelineChangeScopes.pop_back();
    resourceChangeScopes.pop_back();
    if (failed(bodyResult))
      return failure();
    std::chrono::steady_clock::time_point finalizeStarted;
    if (profileEnabled)
      finalizeStarted = std::chrono::steady_clock::now();
    scheduledCallChangeEntries += changes.previous.size();
    if (!resourceChangeScopes.empty()) {
      ResourceChangeScope *parent = resourceChangeScopes.back();
      for (const auto &[resource, previous] : changes.previous)
        parent->previous.try_emplace(resource, previous);
    }
    if (!pipelineChangeScopes.empty()) {
      PipelineChangeScope *parent = pipelineChangeScopes.back();
      for (const auto &entry : pipelineChanges.previous)
        parent->previous.try_emplace(entry.getKey(), entry.getValue());
      for (const auto &entry : pipelineChanges.firstUses) {
        auto [found, inserted] =
            parent->firstUses.try_emplace(entry.getKey(), entry.getValue());
        if (!inserted)
          found->second = std::min(found->second, entry.getValue());
      }
    }
    mergeResourceUses(changes);
    mergeGlobalBarrierUses(changes);
    SmallVector<Value, 4> yielded = yieldedValues(bodyBlock);
    double finish = bodyDirectFinish;
    for (Value value : yielded)
      finish = std::max(finish, valueReady(value));
    sink[rowIndex].duration = finish - callStart;
    CallTemplateSummary summary;
    summary.duration = finish - callStart;
    call.getBody().walk([&](qlx::phys::TransportResourceOp transport) {
      summary.containsModeledTransport |= bool(transport.getModelAttr());
    });
    summary.boundaryResources.assign(sink[rowIndex].resourceIds.begin(),
                                     sink[rowIndex].resourceIds.end());
    llvm::SmallDenseSet<ResourceId, 8> seenDescendantResources;
    for (const ScheduleRow &row : ArrayRef(sink).drop_front(rowIndex + 1))
      for (ResourceId resource : row.resourceIds)
        if (seenDescendantResources.insert(resource).second)
          summary.descendantResources.push_back(resource);
    llvm::sort(summary.descendantResources,
               [&](ResourceId left, ResourceId right) {
                 return resourceLabelLess(left, right);
               });
    summary.inputResourcePositions.assign(
        sink[rowIndex].inputResourcePositions.begin(),
        sink[rowIndex].inputResourcePositions.end());
    llvm::sort(summary.inputResourcePositions,
               [&](const auto &left, const auto &right) {
                 return resourceLabelLess(left.first, right.first);
               });
    summary.stateOnlyBoundary = sink[rowIndex].stateOnlyBoundary;
    for (const auto &[resource, firstUse] : changes.firstUses)
      summary.firstUseOffsets.emplace_back(resource, firstUse - callStart);
    llvm::sort(summary.firstUseOffsets,
               [&](const auto &left, const auto &right) {
                 return resourceLabelLess(left.first, right.first);
               });
    for (const auto &[resource, previous] : changes.previous) {
      double nextReady = readyAt(current, resource);
      StringRef nextProducer = producedBy(current, resource);
      if (nextReady == previous.ready && nextProducer == previous.producer)
        continue;
      summary.availabilityOffsets.push_back(
          {resource, nextReady - callStart, pipelinePlanAt(current, resource)});
    }
    llvm::sort(summary.availabilityOffsets,
               [&](const auto &left, const auto &right) {
                 return resourceLabelLess(left.resource, right.resource);
               });
    for (const auto &entry : pipelineChanges.previous) {
      auto firstUse = pipelineChanges.firstUses.find(entry.getKey());
      auto next = pipelineInitiation.find(entry.getKey());
      if (firstUse == pipelineChanges.firstUses.end() ||
          next == pipelineInitiation.end())
        continue;
      summary.pipelineTransitions.push_back({entry.getKey(),
                                             firstUse->second - callStart,
                                             next->second.ready - callStart});
      next->second.producer = callId;
    }
    llvm::sort(summary.pipelineTransitions,
               [](const auto &left, const auto &right) {
                 return left.plan < right.plan;
               });
    if (changes.globalBarrierFirstUse)
      summary.globalBarrierFirstUseOffset =
          *changes.globalBarrierFirstUse - callStart;
    if (changes.globalBarrierAvailability)
      summary.globalBarrierAvailabilityOffset =
          *changes.globalBarrierAvailability - callStart;
    // Outside the expanded canonical body, the call envelope is the stable
    // owner of every exported resource effect.  This gives later compact
    // invocations a dependency that survives body sharing and IR replay.
    for (const auto &effect : summary.availabilityOffsets)
      setResourceState(current, effect.resource, callStart + effect.offset,
                       callId, effect.pipelinePlan,
                       effect.pipelinePlan.empty() ? StringRef{} : callId);
    callTemplates[callId] = std::move(summary);
    if (call.getNumResults() != 0)
      operationResultStates[call.getOperation()] = {finish, callId};
    if (profileEnabled) {
      scheduledCallFinalizeSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        finalizeStarted)
              .count();
      scheduledCallSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        callStarted)
              .count();
    }
    return success();
  }

  LogicalResult scheduleCallTemplate(qlx::phys::CallTemplateOp invocation,
                                     ScheduleState &current,
                                     SmallVectorImpl<ScheduleRow> &sink,
                                     ScheduleRow base, StringRef condition,
                                     const MappedCallTemplateSummary *mapped) {
    std::chrono::steady_clock::time_point templateStarted;
    if (profileEnabled)
      templateStarted = std::chrono::steady_clock::now();
    ++scheduledTemplateCalls;
    std::chrono::steady_clock::time_point aliasStarted;
    if (profileEnabled)
      aliasStarted = std::chrono::steady_clock::now();
    auto canonical = callTemplates.find(invocation.getTemplateEvent());
    if (canonical == callTemplates.end())
      return invocation.emitOpError(
          "references no earlier canonical phys.call");
    if (!mapped)
      return invocation.emitOpError(
          "has no prepared canonical call-template resource summary");
    base.callee = invocation.getCallee();
    base.instance = invocation.getInstance();
    base.templateEvent = invocation.getTemplateEvent();
    if (auto profile = invocation.getProfileAttr())
      base.profile = profile.getValue();
    scheduledAliasEntries += mapped->resourceAliases->size();
    if (profileEnabled)
      scheduledAliasSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        aliasStarted)
              .count();
    std::chrono::steady_clock::time_point rowStarted;
    if (profileEnabled)
      rowStarted = std::chrono::steady_clock::now();
    base.duration = canonical->second.duration;
    StringRef invocationId = base.id;
    double invocationStart = base.start;
    double finish = base.finish();
    for (const auto &[resource, offset] : mapped->firstUseOffsets)
      noteResourceUse(resource, invocationStart + offset);
    if (canonical->second.globalBarrierFirstUseOffset &&
        canonical->second.globalBarrierAvailabilityOffset)
      noteGlobalBarrier(
          invocationStart + *canonical->second.globalBarrierFirstUseOffset,
          invocationStart + *canonical->second.globalBarrierAvailabilityOffset);
    std::chrono::steady_clock::time_point pushStarted;
    if (profileEnabled)
      pushStarted = std::chrono::steady_clock::now();
    sink.push_back(std::move(base));
    if (profileEnabled)
      scheduledTemplatePushSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        pushStarted)
              .count();
    std::chrono::steady_clock::time_point resultsStarted;
    if (profileEnabled)
      resultsStarted = std::chrono::steady_clock::now();
    scheduledTemplateResults += invocation.getNumResults();
    if (invocation.getNumResults() != 0)
      operationResultStates[invocation.getOperation()] = {finish, invocationId};
    if (profileEnabled)
      scheduledTemplateResultSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        resultsStarted)
              .count();
    if (profileEnabled)
      scheduledTemplateRowSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        rowStarted)
              .count();
    scheduledSummaryEntries += canonical->second.availabilityOffsets.size();
    std::chrono::steady_clock::time_point applyStarted;
    if (profileEnabled)
      applyStarted = std::chrono::steady_clock::now();
    for (const CallResourceAvailability &effect : mapped->availabilityOffsets) {
      double replayedReady = invocationStart + effect.offset;
      double previousReady = readyAt(current, effect.resource);
      if (replayedReady >= previousReady)
        setResourceState(current, effect.resource, replayedReady, invocationId,
                         effect.pipelinePlan,
                         effect.pipelinePlan.empty() ? StringRef{}
                                                     : invocationId);
    }
    for (const CallPipelineTransition &transition :
         canonical->second.pipelineTransitions)
      setPipelineInitiation(
          transition.plan, invocationStart + transition.availabilityOffset,
          invocationId, invocationStart + transition.firstUseOffset);
    if (canonical->second.globalBarrierAvailabilityOffset) {
      double replayedReady =
          invocationStart + *canonical->second.globalBarrierAvailabilityOffset;
      for (ResourceId resource : physicalResourceIds)
        if (replayedReady >= readyAt(current, resource))
          setResourceState(current, resource, replayedReady, invocationId);
      if (replayedReady >= readyAt(current, clockResource))
        setResourceState(current, clockResource, replayedReady, invocationId);
    }
    if (profileEnabled)
      scheduledApplySeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        applyStarted)
              .count();
    if (profileEnabled && scheduledTemplateCalls % 1000 == 0)
      llvm::errs() << "phys-schedule: template-profile calls="
                   << scheduledTemplateCalls
                   << " aliases=" << scheduledAliasEntries
                   << " summary=" << scheduledSummaryEntries
                   << " alias-seconds=" << scheduledAliasSeconds
                   << " apply-seconds=" << scheduledApplySeconds << "\n";
    if (profileEnabled)
      scheduledTemplateTotalSeconds +=
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        templateStarted)
              .count();
    return success();
  }

  FailureOr<SmallVector<ResourceId, 4>>
  spacetimePhaseResources(qlx::phys::SpacetimePhaseOp phase) {
    SmallVector<ResourceId, 4> result;
    for (Attribute raw : phase.getResourceClasses()) {
      auto reference = dyn_cast<SymbolRefAttr>(raw);
      auto plan = phase->getParentOfType<qlx::phys::SpacetimePlanOp>();
      auto architecture = plan ? symbols.lookup<qlx::phys::ArchitectureOp>(
                                     plan.getArchitecture())
                               : qlx::phys::ArchitectureOp{};
      auto resource = reference && architecture &&
                              reference.getRootReference().getValue() ==
                                  plan.getArchitecture() &&
                              reference.getNestedReferences().size() == 1
                          ? SymbolTable(architecture)
                                .lookup<qlx::phys::ResourceClassOp>(
                                    reference.getLeafReference().getValue())
                          : qlx::phys::ResourceClassOp{};
      if (!resource) {
        phase.emitOpError("has an unresolved physical resource-class claim");
        return failure();
      }
      result.push_back(
          registerResourceKey(("class:" + resource.getSymName()).str()));
    }
    if (auto claims = phase.getResourceClaimsAttr())
      for (Attribute raw : claims) {
        auto claim = cast<DictionaryAttr>(raw);
        auto reference = claim.getAs<SymbolRefAttr>("resource_class");
        int64_t offset = claim.getAs<IntegerAttr>("offset").getInt();
        int64_t count = claim.getAs<IntegerAttr>("count").getInt();
        for (int64_t index = offset; index < offset + count; ++index) {
          ResourceId identity =
              registerResourceKey((reference.getLeafReference().getValue() +
                                   "[" + std::to_string(index) + "]")
                                      .str());
          result.push_back(identity);
          if (!llvm::is_contained(physicalResourceIds, identity))
            physicalResourceIds.push_back(identity);
        }
      }
    for (Attribute raw : phase.getFactoryModels()) {
      auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
      auto model =
          reference
              ? symbols.lookup<qlx::phys::FactoryModelOp>(reference.getValue())
              : qlx::phys::FactoryModelOp{};
      if (!model) {
        phase.emitOpError("has an unresolved factory-model claim");
        return failure();
      }
      result.push_back(
          registerResourceKey(("factory:" + model.getSymName()).str()));
    }
    // During resolve() this helper pre-registers the complete plan-resource
    // closure before canonical label ranks exist.  Once those ranks have been
    // built, every emitted schedule row must use the same stable label order
    // as ordinary graph events and the independent verifier.
    if (resourceLabelRanks.empty())
      llvm::sort(result);
    else
      llvm::sort(result, [&](ResourceId left, ResourceId right) {
        return resourceLabelLess(left, right);
      });
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

  StringRef saveString(std::string value) {
    ownedStrings.push_back(std::move(value));
    return ownedStrings.back();
  }

  StringRef pipelineIdentity(qlx::phys::SpacetimePlanOp plan) {
    auto found = pipelineIdentities.find(plan.getOperation());
    if (found != pipelineIdentities.end())
      return found->second;
    std::string value;
    llvm::raw_string_ostream stream(value);
    stream << plan.getArchitecture() << '|' << plan.getOperatingPoint() << '|'
           << plan.getProvider() << '|' << plan.getProviderVersion() << '|'
           << plan.getDerivation() << '|' << plan.getDerivationVersion() << '|'
           << plan.getEvidence() << '|' << plan.getForwardingLatencyNsAttr()
           << '|' << plan.getInitiationIntervalNsAttr();
    for (qlx::phys::SpacetimePhaseOp phase :
         plan.getBody().front().getOps<qlx::phys::SpacetimePhaseOp>())
      stream << '|' << phase.getSymName() << ':' << phase.getSteps() << ':'
             << phase.getStepDurationNsAttr() << ':' << phase.getAfterAttr()
             << ':' << phase.getResourceClassesAttr() << ':'
             << phase.getResourceClaimsAttr() << ':'
             << phase.getFactoryModelsAttr();
    StringRef retained = saveString(std::move(value));
    pipelineIdentities[plan.getOperation()] = retained;
    return retained;
  }

  LogicalResult scheduleSpacetimeCall(qlx::phys::SpacetimeCallOp invocation,
                                      ScheduleState &current,
                                      SmallVectorImpl<ScheduleRow> &sink,
                                      ScheduleRow base, StringRef condition) {
    auto plan = symbols.lookup<qlx::phys::SpacetimePlanOp>(
        invocation.getPlanAttr().getValue());
    if (!plan)
      return invocation.emitOpError("references unresolved spacetime plan");
    base.callee = plan.getSourceProtocol();
    base.instance = invocation.getInstance();
    base.profile = plan.getDerivation();
    auto forwarding = plan.getForwardingLatencyNsAttr();
    auto interval = plan.getInitiationIntervalNsAttr();
    bool pipelined = forwarding && interval;
    StringRef planName = pipelineIdentity(plan);
    if (pipelined) {
      auto previous = pipelineInitiation.find(planName);
      if (previous != pipelineInitiation.end()) {
        base.start = std::max(base.start, previous->second.ready);
        if (!previous->second.producer.empty() &&
            !llvm::is_contained(base.resourceDependencies,
                                previous->second.producer)) {
          base.resourceDependencies.push_back(previous->second.producer);
          if (!llvm::is_contained(base.dependencies, previous->second.producer))
            base.dependencies.push_back(previous->second.producer);
        }
      }
    }
    // Root phases may wait for a factory startup or another non-pipeline
    // resource.  That wait is part of the invocation's forwarding boundary,
    // so establish it before emitting the envelope rather than allowing the
    // envelope to promise outputs before its physical phase has begun.
    for (qlx::phys::SpacetimePhaseOp phase :
         plan.getBody().front().getOps<qlx::phys::SpacetimePhaseOp>()) {
      if (!phase.getAfter().empty())
        continue;
      auto resources = spacetimePhaseResources(phase);
      if (failed(resources))
        return failure();
      for (ResourceId resource : *resources) {
        StringRef producer = producedBy(current, resource);
        bool samePipeline = pipelined && !producer.empty() &&
                            pipelinePlanAt(current, resource) == planName &&
                            pipelineInvocationAt(current, resource) != base.id;
        if (samePipeline)
          continue;
        base.start = std::max(base.start, readyAt(current, resource));
        if (!producer.empty() &&
            !llvm::is_contained(base.resourceDependencies, producer)) {
          base.resourceDependencies.push_back(producer);
          if (!llvm::is_contained(base.dependencies, producer))
            base.dependencies.push_back(producer);
        }
      }
    }
    size_t rowIndex = sink.size();
    StringRef callId = base.id;
    double callStart = base.start;
    sink.push_back(std::move(base));

    llvm::StringMap<double> phaseFinish;
    llvm::StringMap<StringRef> phaseEvent;
    double finish = callStart;
    for (qlx::phys::SpacetimePhaseOp phase :
         plan.getBody().front().getOps<qlx::phys::SpacetimePhaseOp>()) {
      auto phaseResources = spacetimePhaseResources(phase);
      if (failed(phaseResources))
        return failure();
      ScheduleRow row;
      row.id = saveString((callId + "." + phase.getSymName()).str());
      row.kind = "spacetime_phase";
      row.start = callStart;
      row.resourceIds = *phaseResources;
      row.parent = callId;
      row.branch = "phase";
      row.condition = condition;

      if (phase.getAfter().empty()) {
        llvm::append_range(row.dataDependencies,
                           sink[rowIndex].dataDependencies);
        llvm::append_range(row.domainDependencies,
                           sink[rowIndex].domainDependencies);
      }
      for (Attribute raw : phase.getAfter()) {
        auto reference = cast<FlatSymbolRefAttr>(raw);
        row.start =
            std::max(row.start, phaseFinish.lookup(reference.getValue()));
        StringRef predecessor = phaseEvent.lookup(reference.getValue());
        if (!predecessor.empty())
          row.dataDependencies.push_back(predecessor);
      }
      for (ResourceId resource : row.resourceIds) {
        StringRef producer = producedBy(current, resource);
        bool samePipeline = pipelined && !producer.empty() &&
                            pipelinePlanAt(current, resource) == planName &&
                            pipelineInvocationAt(current, resource) != callId;
        if (!samePipeline) {
          row.start = std::max(row.start, readyAt(current, resource));
          if (!producer.empty() &&
              !llvm::is_contained(row.resourceDependencies, producer))
            row.resourceDependencies.push_back(producer);
        }
      }
      auto appendDependencies = [&](ArrayRef<StringRef> dependencies) {
        for (StringRef dependency : dependencies)
          if (!llvm::is_contained(row.dependencies, dependency))
            row.dependencies.push_back(dependency);
      };
      appendDependencies(row.dataDependencies);
      appendDependencies(row.resourceDependencies);
      appendDependencies(row.domainDependencies);

      double stepDuration = phase.getStepDurationNs().convertToDouble();
      row.duration = stepDuration * static_cast<double>(phase.getSteps());
      if (!std::isfinite(row.duration) || row.duration <= 0.0)
        return phase.emitOpError(
            "derived spacetime phase duration is not finite and positive");
      double phaseEnd = row.finish();
      StringRef phaseId = row.id;
      sink.push_back(std::move(row));
      phaseFinish[phase.getSymName()] = phaseEnd;
      phaseEvent[phase.getSymName()] = phaseId;
      finish = std::max(finish, phaseEnd);
      for (ResourceId resource : *phaseResources) {
        noteResourceUse(resource, phaseEnd - stepDuration * phase.getSteps());
        setResourceState(current, resource, phaseEnd, phaseId,
                         pipelined ? planName : StringRef{},
                         pipelined ? callId : StringRef{});
      }
    }

    // A callable pipeline has two independently meaningful boundaries.  Its
    // envelope is the SSA forwarding boundary consumed by enclosing calls;
    // the nested phase rows retain the complete physical occupancy through
    // cleanup.  Non-pipelined plans keep the historical one-shot completion
    // boundary.
    sink[rowIndex].duration =
        pipelined ? forwarding.getValueAsDouble() : finish - callStart;
    if (pipelined)
      setPipelineInitiation(planName, callStart + interval.getValueAsDouble(),
                            callId, callStart);
    if (invocation.getNumResults() != 0) {
      double ready =
          pipelined ? callStart + forwarding.getValueAsDouble() : finish;
      operationResultStates[invocation.getOperation()] = {ready, callId};
    }
    return success();
  }

  LogicalResult scheduleRepeat(qlx::cflow::RepeatOp repeat,
                               ScheduleState &current,
                               SmallVectorImpl<ScheduleRow> &sink,
                               ScheduleRow base, StringRef condition,
                               int64_t repeatMultiplicity) {
    int64_t count = repeat.getCountAttr().getInt();
    if (count != 0 &&
        repeatMultiplicity > std::numeric_limits<int64_t>::max() / count)
      return repeat.emitOpError(
          "composed repeat multiplicity exceeds signed 64-bit range");
    int64_t nestedMultiplicity = repeatMultiplicity * count;
    base.repeatCount = count;
    size_t rowIndex = sink.size();
    sink.push_back(base);
    ResourceChangeScope changes;
    resourceChangeScopes.push_back(&changes);
    Block &bodyBlock = repeat.getBody().front();
    bindArguments(bodyBlock, repeat.getOperands(), base.start,
                  /*skipYieldPassThrough=*/true);
    double templateFinish = base.start;
    LogicalResult bodyResult =
        scheduleBlock(bodyBlock, current, sink, base.id, "body", condition,
                      base.start, nestedMultiplicity, &templateFinish);
    resourceChangeScopes.pop_back();
    if (failed(bodyResult))
      return failure();
    for (Value value : yieldedValues(bodyBlock))
      templateFinish = std::max(templateFinish, valueReady(value));
    // Folded repeats retain only one scheduled body template.  scheduleBlock
    // reports every direct child event, including control-only calls that have
    // no yielded value or physical-resource effect.  Nested envelopes report
    // their folded finish, so inactive children of a zero repeat stay purely
    // inspectable and do not extend this template.
    SmallVector<ResourceId, 8> touched;
    DenseMap<ResourceId, double> templateAvailability;
    for (const auto &[resource, previous] : changes.previous)
      if (readyAt(current, resource) != previous.ready ||
          producedBy(current, resource) != previous.producer) {
        touched.push_back(resource);
        templateAvailability[resource] = readyAt(current, resource);
      }
    llvm::sort(touched);
    // Duration is the canonical folded quantity.  Store that product directly
    // rather than reconstructing it through (start + duration) - start: at a
    // large absolute clock frontier the extra add/subtract pair can round to a
    // different double than the verifier's exact template-duration product.
    double recurrenceFinish = templateFinish;
    for (ResourceId resource : touched)
      recurrenceFinish = std::max(recurrenceFinish, readyAt(current, resource));
    double period = recurrenceFinish - base.start;
    double epilogue = templateFinish - base.start;
    double duration =
        count == 0 ? 0.0
                   : (static_cast<double>(count) - 1.0) * period + epilogue;
    sink[rowIndex].duration = duration;
    sink[rowIndex].repeatPeriod = period;
    sink[rowIndex].repeatEpilogue = epilogue;
    double finish = base.start + duration;
    if (repeat.getNumResults() != 0)
      operationResultStates[repeat.getOperation()] = {finish, base.id};
    for (const auto &[resource, previous] : changes.previous)
      current.resources[resource] = previous;
    if (count != 0) {
      mergeResourceUses(changes);
      if (changes.globalBarrierFirstUse && changes.globalBarrierAvailability)
        noteGlobalBarrier(
            *changes.globalBarrierFirstUse,
            base.start + (static_cast<double>(count) - 1.0) * period +
                (*changes.globalBarrierAvailability - base.start));
      for (ResourceId resource : touched) {
        double availability = finish;
        if (StringRef(resourceLabels[resource]).starts_with("factory:"))
          availability = base.start +
                         (static_cast<double>(count) - 1.0) * period +
                         (templateAvailability.lookup(resource) - base.start);
        setResourceState(current, resource, availability, base.id);
      }
    }
    return success();
  }

  LogicalResult scheduleIf(qlx::cflow::IfOp branchOp, ScheduleState &current,
                           SmallVectorImpl<ScheduleRow> &sink, ScheduleRow base,
                           StringRef inheritedCondition,
                           int64_t repeatMultiplicity) {
    StringRef condition = valueProducer(branchOp.getCondition());
    base.condition = condition.empty() ? inheritedCondition : condition;
    size_t rowIndex = sink.size();
    sink.push_back(base);
    SmallVector<ResourceStateDelta, 2> branches;
    SmallVector<double, 2> finishes;
    SmallVector<Region *, 2> regions{&branchOp.getThenRegion(),
                                     &branchOp.getElseRegion()};
    SmallVector<StringRef, 2> branchNames{"then", "else"};
    for (auto [name, region] : llvm::zip(branchNames, regions)) {
      // Branches begin from one frontier. Journal and restore only actual
      // writes rather than cloning the machine-wide resource vector.
      ResourceChangeScope changes;
      resourceChangeScopes.push_back(&changes);
      Block &branchBlock = region->front();
      size_t branchRowBegin = sink.size();
      LogicalResult branchResult =
          scheduleBlock(branchBlock, current, sink, base.id, name,
                        base.condition, base.start, repeatMultiplicity);
      resourceChangeScopes.pop_back();
      if (failed(branchResult))
        return failure();
      mergeResourceUses(changes);
      mergeGlobalBarrierUses(changes);
      double finish = base.start;
      for (Value value : yieldedValues(branchBlock))
        finish = std::max(finish, valueReady(value));
      // Control/event-only branch work may yield no schedulable value and
      // touch no physical resource, but it still belongs to the envelope.
      for (const ScheduleRow &row : ArrayRef(sink).drop_front(branchRowBegin))
        finish = std::max(finish, row.finish());
      finishes.push_back(finish);
      branches.push_back(captureResourceChangesAndRestore(current, changes));
    }
    double finish = std::max(finishes[0], finishes[1]);
    llvm::SmallDenseSet<ResourceId, 8> touchedSet;
    for (const ResourceStateDelta &branch : branches)
      for (const auto &change : branch)
        touchedSet.insert(change.first);
    SmallVector<ResourceId, 8> touched(touchedSet.begin(), touchedSet.end());
    llvm::sort(touched);
    for (ResourceId resource : touched) {
      double previous = readyAt(current, resource);
      double merged = previous;
      ScheduledResourceState first =
          branchResourceState(branches[0], current, resource);
      bool same = true;
      for (const ResourceStateDelta &branch : branches) {
        ScheduledResourceState state =
            branchResourceState(branch, current, resource);
        merged = std::max(merged, state.ready);
        same &= state.producer == first.producer;
      }
      setResourceState(current, resource, merged,
                       same ? first.producer : StringRef(base.id));
      finish = std::max(finish, merged);
    }
    sink[rowIndex].duration = finish - base.start;
    // The structured branch is the externally visible producer for every
    // carried physical resource, even when both alternatives happen to leave
    // the same nested producer in their local schedule state.  This matches
    // the SSA envelope: users of the branch results depend on the branch
    // decision, not on one alternative's internal event identity.
    for (ResourceId resource : base.resourceIds)
      setResourceState(current, resource, finish, base.id);
    if (branchOp.getNumResults() != 0)
      operationResultStates[branchOp.getOperation()] = {finish, base.id};
    return success();
  }

  LogicalResult scheduleWhile(qlx::cflow::WhileOp loop, ScheduleState &current,
                              SmallVectorImpl<ScheduleRow> &sink,
                              ScheduleRow base, StringRef inheritedCondition,
                              int64_t repeatMultiplicity) {
    auto bound = loop.getMaxIterationsAttr();
    if (!bound)
      return loop.emitOpError(
          "the generic physical scheduler requires a bounded cflow.while; use "
          "a symbolic controller scheduler for an unbounded runtime loop");
    int64_t maxIterations = bound.getInt();
    base.maxIterations = maxIterations;
    size_t rowIndex = sink.size();
    sink.push_back(base);

    ScheduleState before = current;
    Block &beforeBlock = loop.getBeforeRegion().front();
    bindArguments(beforeBlock, loop.getInits(), base.start);
    // The condition region executes as a whole before the body.  Its returned
    // predicate and forwarded values need not depend on every side event, so
    // retain the folded finish of each direct row as well.  Nested structured
    // work is represented by its direct envelope and requires no subtree scan.
    double conditionFinish = base.start;
    if (failed(scheduleBlock(beforeBlock, before, sink, base.id, "condition",
                             inheritedCondition, base.start, repeatMultiplicity,
                             &conditionFinish)))
      return failure();
    auto condition =
        dyn_cast<qlx::cflow::WhileConditionOp>(beforeBlock.getTerminator());
    if (!condition)
      return loop.emitOpError(
          "native phys-schedule requires cflow.while_condition");
    StringRef conditionEvent = valueProducer(condition.getCondition());
    conditionFinish =
        std::max(conditionFinish, valueReady(condition.getCondition()));
    for (Value value : condition.getForwarded())
      conditionFinish = std::max(conditionFinish, valueReady(value));
    for (ResourceId resource = 0; resource < before.resources.size();
         ++resource)
      if (resourceChanged(current, before, resource))
        conditionFinish = std::max(conditionFinish, readyAt(before, resource));

    ScheduleState body = before;
    Block &bodyBlock = loop.getAfterRegion().front();
    for (BlockArgument argument : bodyBlock.getArguments()) {
      valueReadyTimes[argument] = conditionFinish;
      valueProducers[argument] =
          conditionEvent.empty() ? base.id : conditionEvent;
    }
    if (failed(scheduleBlock(bodyBlock, body, sink, base.id, "body",
                             conditionEvent, conditionFinish,
                             repeatMultiplicity)))
      return failure();

    double bodyFinish = conditionFinish;
    for (Value value : yieldedValues(bodyBlock))
      bodyFinish = std::max(bodyFinish, valueReady(value));
    SmallVector<ResourceId, 8> touched;
    for (ResourceId resource = 0; resource < body.resources.size(); ++resource)
      if (resourceChanged(current, body, resource)) {
        touched.push_back(resource);
        bodyFinish = std::max(bodyFinish, readyAt(body, resource));
      }
    double conditionDuration = conditionFinish - base.start;
    double bodyDuration = bodyFinish - conditionFinish;
    double finish =
        base.start +
        conditionDuration * (static_cast<double>(maxIterations) + 1.0) +
        bodyDuration * static_cast<double>(maxIterations);
    if (!std::isfinite(finish))
      return loop.emitOpError("bounded schedule duration is not finite");
    sink[rowIndex].duration = finish - base.start;
    sink[rowIndex].condition =
        conditionEvent.empty() ? inheritedCondition : conditionEvent;
    if (loop.getNumResults() != 0)
      operationResultStates[loop.getOperation()] = {finish, base.id};
    for (ResourceId resource : touched)
      setResourceState(current, resource, finish, base.id);
    return success();
  }

  LogicalResult scheduleEventTryTake(qlx::event::TryTakeOp dispatch,
                                     ScheduleState &current,
                                     SmallVectorImpl<ScheduleRow> &sink,
                                     ScheduleRow base,
                                     StringRef inheritedCondition,
                                     int64_t repeatMultiplicity) {
    StringRef inputProducer = valueProducer(dispatch.getEvent());
    base.condition = inputProducer.empty() ? inheritedCondition : inputProducer;
    size_t rowIndex = sink.size();
    sink.push_back(base);

    SmallVector<ResourceStateDelta, 3> branches;
    SmallVector<double, 3> finishes;
    SmallVector<Region *, 3> regions{
        &dispatch.getReady(), &dispatch.getPending(), &dispatch.getFailed()};
    SmallVector<StringRef, 3> branchNames{"ready", "pending", "failed"};
    for (auto [name, region] : llvm::zip(branchNames, regions)) {
      ResourceChangeScope changes;
      resourceChangeScopes.push_back(&changes);
      Block &branchBlock = region->front();
      for (BlockArgument argument : branchBlock.getArguments()) {
        valueReadyTimes[argument] = base.start;
        valueProducers[argument] = base.id;
      }
      size_t branchRowBegin = sink.size();
      LogicalResult branchResult =
          scheduleBlock(branchBlock, current, sink, base.id, name, base.id,
                        base.start, repeatMultiplicity);
      resourceChangeScopes.pop_back();
      if (failed(branchResult))
        return failure();
      mergeResourceUses(changes);
      mergeGlobalBarrierUses(changes);
      double finish = base.start;
      for (Value value : yieldedValues(branchBlock))
        finish = std::max(finish, valueReady(value));
      for (const ScheduleRow &row : ArrayRef(sink).drop_front(branchRowBegin))
        finish = std::max(finish, row.finish());
      finishes.push_back(finish);
      branches.push_back(captureResourceChangesAndRestore(current, changes));
    }

    double finish = *llvm::max_element(finishes);
    llvm::SmallDenseSet<ResourceId, 8> touchedSet;
    for (const ResourceStateDelta &branch : branches)
      for (const auto &change : branch)
        touchedSet.insert(change.first);
    SmallVector<ResourceId, 8> touched(touchedSet.begin(), touchedSet.end());
    llvm::sort(touched);
    SmallVector<ResourceId, 8> envelopedResources;
    for (ResourceId resource : touched) {
      double previous = readyAt(current, resource);
      StringRef previousProducer = producedBy(current, resource);
      double merged = previous;
      ScheduledResourceState first =
          branchResourceState(branches[0], current, resource);
      bool same = true;
      bool branchChanged = false;
      for (const ResourceStateDelta &branch : branches) {
        ScheduledResourceState state =
            branchResourceState(branch, current, resource);
        merged = std::max(merged, state.ready);
        same &= state.producer == first.producer;
        branchChanged |=
            state.ready != previous || state.producer != previousProducer;
      }
      if (same) {
        setResourceState(current, resource, merged, first.producer);
      } else {
        setResourceState(current, resource, merged, base.id);
        envelopedResources.push_back(resource);
      }
      if (llvm::is_contained(base.resourceIds, resource) || branchChanged)
        finish = std::max(finish, merged);
    }

    sink[rowIndex].duration = finish - base.start;
    if (dispatch.getNumResults() != 0)
      operationResultStates[dispatch.getOperation()] = {finish, base.id};
    for (ResourceId resource : base.resourceIds)
      setResourceState(current, resource, finish, base.id);
    for (ResourceId resource : envelopedResources)
      setResourceState(current, resource, finish, base.id);
    return success();
  }

  ArrayAttr serializedEntries(OpBuilder &builder) const {
    SmallVector<Attribute> serialized;
    serialized.reserve(rows.size());
    for (const ScheduleRow &row : rows)
      serialized.push_back(
          builder.getStringAttr(serialize(row, resourceLabels)));
    return builder.getArrayAttr(serialized);
  }

  LogicalResult validateSerializedRows() {
    auto rejectScalar = [&](const ScheduleRow &row, StringRef field,
                            StringRef value) -> LogicalResult {
      if (!value.contains('|'))
        return success();
      return graph.emitOpError("schedule event '")
             << row.id << "' field " << field
             << " contains reserved row delimiter '|'";
    };
    auto rejectListToken = [&](const ScheduleRow &row, StringRef field,
                               StringRef value) -> LogicalResult {
      if (!value.contains('|') && !value.contains(','))
        return success();
      return graph.emitOpError("schedule event '")
             << row.id << "' field " << field
             << " contains reserved list delimiter '|' or ','";
    };
    auto rejectDuplicates = [&](const ScheduleRow &row, StringRef field,
                                ArrayRef<StringRef> values) -> LogicalResult {
      llvm::SmallDenseSet<StringRef, 8> unique;
      for (StringRef value : values)
        if (!unique.insert(value).second)
          return graph.emitOpError("schedule event '")
                 << row.id << "' field " << field
                 << " contains duplicate dependency '" << value << "'";
      return success();
    };
    for (const ScheduleRow &row : rows) {
      if (failed(rejectListToken(row, "event_id", row.id)) ||
          failed(rejectScalar(row, "kind", row.kind)) ||
          failed(rejectDuplicates(row, "dependencies", row.dependencies)) ||
          failed(rejectDuplicates(row, "data_deps", row.dataDependencies)) ||
          failed(rejectDuplicates(row, "resource_deps",
                                  row.resourceDependencies)) ||
          failed(rejectDuplicates(row, "domain_deps", row.domainDependencies)))
        return failure();
      for (StringRef value : row.dependencies)
        if (failed(rejectListToken(row, "dependencies", value)))
          return failure();
      for (StringRef value : row.dataDependencies)
        if (failed(rejectListToken(row, "data_deps", value)))
          return failure();
      for (StringRef value : row.resourceDependencies)
        if (failed(rejectListToken(row, "resource_deps", value)))
          return failure();
      for (StringRef value : row.domainDependencies)
        if (failed(rejectListToken(row, "domain_deps", value)))
          return failure();
      for (ResourceId resource : row.resourceIds)
        if (failed(rejectListToken(row, "resources", resourceLabels[resource])))
          return failure();
      const std::pair<StringRef, StringRef> scalarFields[] = {
          {"parent", row.parent},
          {"branch", row.branch},
          {"condition", row.condition},
          {"commit_point", row.commitPoint},
          {"callee", row.callee},
          {"instance", row.instance},
          {"profile", row.profile},
          {"template_event", row.templateEvent},
          {"attempt", row.attempt},
          {"attempt_event", row.attemptEvent},
          {"decision_event", row.decisionEvent},
          {"exhaustion", row.exhaustion},
          {"success_probability_source", row.successProbabilitySource},
          {"success_probability_evidence", row.successProbabilityEvidence},
      };
      for (auto [field, value] : scalarFields)
        if (failed(rejectScalar(row, field, value)))
          return failure();
    }
    return success();
  }

  LogicalResult emitSchedule(double makespan) {
    if (failed(validateSerializedRows()))
      return failure();
    OpBuilder builder(context);
    SmallVector<NamedAttribute> timing;
    for (const auto &item : usedTimings)
      timing.emplace_back(builder.getStringAttr(item.first),
                          builder.getF64FloatAttr(item.second));
    llvm::sort(timing, [](NamedAttribute left, NamedAttribute right) {
      return left.getName().getValue() < right.getName().getValue();
    });
    SmallVector<Attribute> constraints;
    for (StringRef value : {
             "graph_ssa_dependencies",
             "physical_resource_exclusion",
             "allocation_mapping_after",
             "structured_control_exclusivity",
             "folded_region_bounds",
             "resolved_event_durations",
         })
      constraints.push_back(builder.getStringAttr(value));

    OperationState scheduleState(graph.getLoc(), "phys.schedule");
    scheduleState.addAttributes({
        builder.getNamedAttr("sym_name", builder.getStringAttr(resultName)),
        builder.getNamedAttr(
            "graph", FlatSymbolRefAttr::get(context, graph.getSymName())),
        builder.getNamedAttr("strategy", builder.getStringAttr("greedy_asap")),
        builder.getNamedAttr("strategy_domain",
                             builder.getStringAttr("physical")),
        builder.getNamedAttr("provider",
                             builder.getStringAttr("qlx.compiler.greedy_asap")),
        builder.getNamedAttr("provider_version", builder.getStringAttr("1")),
        builder.getNamedAttr(
            "constraint_profile",
            builder.getStringAttr("qlx.physical_schedule.constraints/v1")),
        builder.getNamedAttr("constraints", builder.getArrayAttr(constraints)),
        builder.getNamedAttr("timing_profile",
                             builder.getDictionaryAttr(timing)),
        builder.getNamedAttr("tie_break",
                             builder.getStringAttr("stable_graph_order")),
        builder.getNamedAttr("optimization_status",
                             builder.getStringAttr("not_applicable")),
        builder.getNamedAttr("entries", deferEntryMaterialization
                                            ? builder.getArrayAttr({})
                                            : serializedEntries(builder)),
        builder.getNamedAttr("makespan_ns", builder.getF64FloatAttr(makespan)),
    });
    builder.setInsertionPointToEnd(module.getBody());
    Operation *created = builder.create(scheduleState);
    schedule = cast<qlx::phys::ScheduleOp>(created);
    // Validate the generated operation's ODS/schema contract before the
    // independent semantic verifier reconstructs the portable schedule below.
    if (failed(schedule.verifyInvariantsImpl())) {
      created->erase();
      return graph.emitOpError(
          "native greedy-ASAP schedule failed schedule schema verification");
    }
    // Calling ModuleOp::verify from inside a module pass can participate in
    // the pass manager's active recursive-verification frame.  Invoke the new
    // operation's semantic verifier directly so the proof cannot be skipped
    // by verifier re-entrancy before Python seals a verified increment.
    if (!deferOutputVerification && failed(schedule.verify())) {
      created->erase();
      return graph.emitOpError(
          "native greedy-ASAP schedule failed independent verification");
    }
    return success();
  }

  FailureOr<qlx::phys::FactoryModelOp>
  factoryModel(Operation *operation) const {
    FlatSymbolRefAttr reference;
    if (auto start = dyn_cast<qlx::phys::FactoryStartOp>(operation))
      reference = start.getFactoryModelAttr();
    else if (auto request = dyn_cast<qlx::phys::ResourceRequestOp>(operation))
      reference = request.getFactoryModelAttr();
    if (!reference)
      return qlx::phys::FactoryModelOp{};
    auto model =
        symbols.lookup<qlx::phys::FactoryModelOp>(reference.getValue());
    if (!model) {
      operation->emitOpError("references unresolved factory model @")
          << reference.getValue();
      return failure();
    }
    return model;
  }

  FailureOr<qlx::phys::TransportModelOp>
  transportModel(qlx::phys::TransportResourceOp operation) const {
    auto reference = operation.getModelAttr();
    if (!reference)
      return qlx::phys::TransportModelOp{};
    auto model =
        symbols.lookup<qlx::phys::TransportModelOp>(reference.getValue());
    if (!model) {
      operation.emitOpError("references unresolved transport model @")
          << reference.getValue();
      return failure();
    }
    return model;
  }

  FailureOr<qlx::phys::QECChannelBindingOp>
  transportBinding(qlx::phys::TransportResourceOp operation) const {
    auto model = transportModel(operation);
    if (failed(model))
      return failure();
    SymbolRefAttr reference =
        *model ? (*model).getQecBindingAttr() : operation.getRouteAttr();
    if (!reference)
      return qlx::phys::QECChannelBindingOp{};
    Operation *architecture =
        SymbolTable::lookupSymbolIn(module, reference.getRootReference());
    auto binding =
        architecture ? dyn_cast_or_null<qlx::phys::QECChannelBindingOp>(
                           SymbolTable(architecture)
                               .lookup(reference.getLeafReference().getValue()))
                     : qlx::phys::QECChannelBindingOp{};
    if (!binding) {
      operation.emitOpError("references unresolved physical channel binding ")
          << reference;
      return failure();
    }
    return binding;
  }

  LogicalResult
  preRegisterTransportResources(qlx::phys::TransportResourceOp operation) {
    auto model = transportModel(operation);
    if (failed(model))
      return failure();
    auto binding = transportBinding(operation);
    if (failed(binding))
      return failure();
    if (!*binding)
      return success();
    ArrayAttr claims = *model ? (*model).getResourceClaimsAttr()
                              : (*binding).getTransportClaimsAttr();
    if (!claims)
      return success();
    for (Attribute raw : claims) {
      auto claim = cast<DictionaryAttr>(raw);
      StringRef resourceName =
          *model ? claim.getAs<SymbolRefAttr>("resource_class")
                       .getLeafReference()
                       .getValue()
                 : claim.getAs<FlatSymbolRefAttr>("resource_class").getValue();
      int64_t offset = claim.getAs<IntegerAttr>("offset").getInt();
      int64_t count = claim.getAs<IntegerAttr>("count").getInt();
      for (int64_t index = offset; index < offset + count; ++index) {
        ResourceId identity = registerResourceKey(
            (resourceName + "[" + std::to_string(index) + "]").str());
        if (!llvm::is_contained(physicalResourceIds, identity))
          physicalResourceIds.push_back(identity);
      }
    }
    auto qecReference = (*binding).getQecChannelAttr();
    Operation *qecMachine =
        SymbolTable::lookupSymbolIn(module, qecReference.getRootReference());
    auto interconnect =
        qecMachine
            ? dyn_cast_or_null<qlx::fabric::InterconnectOp>(
                  SymbolTable(qecMachine)
                      .lookup(qecReference.getLeafReference().getValue()))
            : qlx::fabric::InterconnectOp{};
    if (!interconnect)
      return operation.emitOpError(
          "transport model has an unresolved selected interconnect");
    auto registerLanes = [&](StringRef prefix, int64_t capacity) {
      for (int64_t lane = 0; lane < capacity; ++lane)
        registerResourceKey((prefix + std::to_string(lane)).str());
    };
    registerLanes(
        ("control:transport-port:" + *interconnect.getPortAName() + ":").str(),
        interconnect.getPortAConcurrency().value_or(1));
    registerLanes(
        ("control:transport-port:" + *interconnect.getPortBName() + ":").str(),
        interconnect.getPortBConcurrency().value_or(1));
    registerLanes(
        ("control:transport-channel:" + interconnect.getSymName() + ":").str(),
        interconnect.getConcurrency().value_or(1));
    StringRef identity =
        *model ? (*model).getSymName() : (*binding).getSymName();
    registerResourceKey(("control:transport-init:" + identity).str());
    return success();
  }

  FailureOr<SmallVector<ResourceId, 8>>
  transportResources(qlx::phys::TransportResourceOp operation) {
    SmallVector<ResourceId, 8> result;
    auto model = transportModel(operation);
    if (failed(model))
      return failure();
    auto binding = transportBinding(operation);
    if (failed(binding))
      return failure();
    if (!*binding)
      return result;
    ArrayAttr claims = *model ? (*model).getResourceClaimsAttr()
                              : (*binding).getTransportClaimsAttr();
    if (!claims)
      return result;
    StringRef identity =
        *model ? (*model).getSymName() : (*binding).getSymName();
    uint64_t occurrence = transportOccurrences[identity]++;
    auto append = [&](std::string key) {
      result.push_back(registerResourceKey(key));
    };
    for (Attribute raw : claims) {
      auto claim = cast<DictionaryAttr>(raw);
      StringRef resourceName =
          *model ? claim.getAs<SymbolRefAttr>("resource_class")
                       .getLeafReference()
                       .getValue()
                 : claim.getAs<FlatSymbolRefAttr>("resource_class").getValue();
      int64_t offset = claim.getAs<IntegerAttr>("offset").getInt();
      int64_t count = claim.getAs<IntegerAttr>("count").getInt();
      int64_t units = claim.getAs<IntegerAttr>("units").getInt();
      int64_t lanes = count / units;
      if (lanes <= 0) {
        operation.emitOpError(
            "transport resource claim has no complete acquisition lane");
        return failure();
      }
      int64_t selected =
          offset + static_cast<int64_t>(occurrence % lanes) * units;
      for (int64_t index = selected; index < selected + units; ++index)
        append((resourceName + "[" + std::to_string(index) + "]").str());
    }
    auto qecReference = (*binding).getQecChannelAttr();
    Operation *qecMachine =
        SymbolTable::lookupSymbolIn(module, qecReference.getRootReference());
    auto interconnect =
        qecMachine
            ? dyn_cast_or_null<qlx::fabric::InterconnectOp>(
                  SymbolTable(qecMachine)
                      .lookup(qecReference.getLeafReference().getValue()))
            : qlx::fabric::InterconnectOp{};
    if (!interconnect) {
      operation.emitOpError(
          "transport model has an unresolved selected interconnect");
      return failure();
    }
    int64_t sourceUnits =
        *model ? (*model).getSourceEndpointOccupancy()
               : (*binding).getSourceEndpointOccupancyAttr().getInt();
    int64_t destinationUnits =
        *model ? (*model).getDestinationEndpointOccupancy()
               : (*binding).getDestinationEndpointOccupancyAttr().getInt();
    auto appendPort = [&](StringRef name, int64_t capacity, int64_t units) {
      int64_t lanes = capacity / units;
      if (lanes <= 0)
        return failure();
      int64_t first = static_cast<int64_t>(occurrence % lanes) * units;
      for (int64_t lane = first; lane < first + units; ++lane)
        append(("control:transport-port:" + name + ":" + std::to_string(lane))
                   .str());
      return success();
    };
    if (failed(appendPort(*interconnect.getPortAName(),
                          interconnect.getPortAConcurrency().value_or(1),
                          sourceUnits)) ||
        failed(appendPort(*interconnect.getPortBName(),
                          interconnect.getPortBConcurrency().value_or(1),
                          destinationUnits))) {
      operation.emitOpError(
          "transport endpoint occupancy has no complete scheduling lane");
      return failure();
    }
    append(
        ("control:transport-channel:" + interconnect.getSymName() + ":" +
         std::to_string(occurrence % interconnect.getConcurrency().value_or(1)))
            .str());
    append(("control:transport-init:" + identity).str());
    llvm::sort(result);
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  }

  static constexpr StringLiteral clockKey = "__qlx_clock_domain__";
  ModuleOp module;
  MLIRContext *context;
  SymbolTable symbols;
  std::string requestedGraph;
  std::string requestedResult;
  std::string resultName;
  qlx::phys::GraphOp graph;
  qlx::phys::ScheduleOp schedule;
  llvm::StringMap<std::string> resourceKeys;
  llvm::StringMap<ResourceId> resourceIdsByKey;
  llvm::StringMap<ResourceId> resourceIdsBySymbol;
  DenseMap<Type, ResourceId> resourceIdsByStateType;
  DenseMap<Type, int64_t> codeDistancesByStateType;
  llvm::StringMap<SmallVector<StringRef, 4>> allocationAfterDependencies;
  SmallVector<std::string, 0> resourceLabels;
  SmallVector<unsigned, 0> resourceLabelRanks;
  SmallVector<ResourceId, 0> physicalResourceIds;
  SmallVector<ResourceId, 0> factoryResourceIds;
  std::deque<std::string> ownedStrings;
  ResourceId clockResource = 0;
  llvm::StringMap<double> timings;
  std::map<std::string, double> usedTimings;
  llvm::StringMap<CallTemplateSummary> callTemplates;
  llvm::StringMap<OperationResultState> pipelineInitiation;
  llvm::StringMap<uint64_t> transportOccurrences;
  SmallVector<PipelineChangeScope *, 8> pipelineChangeScopes;
  DenseMap<Operation *, StringRef> pipelineIdentities;
  DenseMap<ResourceId, ResourceId> emptyTemplateResourceAliases;
  DenseMap<Attribute, unsigned> templateResourceAliasCacheIds;
  std::deque<DenseMap<ResourceId, ResourceId>> templateResourceAliasMaps;
  DenseMap<std::pair<Attribute, Attribute>, unsigned>
      mappedCallTemplateCacheIds;
  std::deque<MappedCallTemplateSummary> mappedCallTemplateSummaries;
  DenseMap<Value, double> valueReadyTimes;
  DenseMap<Value, StringRef> valueProducers;
  DenseMap<Operation *, OperationResultState> operationResultStates;
  SmallVector<ResourceChangeScope *, 4> resourceChangeScopes;
  ScheduleState state;
  SmallVector<ScheduleRow, 0> rows;
  std::chrono::steady_clock::time_point profileStarted =
      std::chrono::steady_clock::now();
  uint64_t scheduledOperations = 0;
  uint64_t scheduledCalls = 0;
  uint64_t scheduledTemplateCalls = 0;
  uint64_t scheduledAliasEntries = 0;
  uint64_t scheduledSummaryEntries = 0;
  double scheduledAliasSeconds = 0.0;
  double scheduledApplySeconds = 0.0;
  double scheduledPreludeSeconds = 0.0;
  double scheduledTemplatePreludeSeconds = 0.0;
  double scheduledCallPreludeSeconds = 0.0;
  double scheduledLeafPreludeSeconds = 0.0;
  double scheduledCallSeconds = 0.0;
  double scheduledCallFinalizeSeconds = 0.0;
  uint64_t scheduledCallChangeEntries = 0;
  size_t scheduledCallMaxDepth = 0;
  double scheduledTemplateTotalSeconds = 0.0;
  double scheduledTemplateRowSeconds = 0.0;
  double scheduledTemplatePushSeconds = 0.0;
  double scheduledTemplateResultSeconds = 0.0;
  uint64_t scheduledTemplateResults = 0;
  double cycle = 1.0;
  bool deferOutputVerification = false;
  bool deferEntryMaterialization = false;
  bool profileEnabled = false;
};

class PhysSchedulePass
    : public qlx::phys::impl::PhysScheduleBase<PhysSchedulePass> {
public:
  using PhysScheduleBase::PhysScheduleBase;

  void runOnOperation() override {
    NativeScheduler scheduler(getOperation(), graphSymbol, resultSymbol,
                              deferOutputVerification);
    if (failed(scheduler.run()))
      signalPassFailure();
  }
};

} // namespace

static FailureOr<std::string>
scheduleAndEstimateJSONImpl(ModuleOp module, StringRef graph,
                            StringRef schedule, StringRef lowerTier,
                            bool fullWorkload, bool authenticateModule) {
  const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
  auto reportPhase = [&](StringRef name,
                         std::chrono::steady_clock::time_point started) {
    if (profile)
      llvm::errs() << "phys-schedule: fused-phase " << name << ' '
                   << std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - started)
                          .count()
                   << "s\n";
  };
  // The direct fused C/C++ entry point must fail closed even when its caller
  // did not arrive through a verifying pass manager.  Verify the immutable
  // input once before mutation; the scheduler then adds only one fresh,
  // uniqueness-checked transient symbol. Its ODS contract is checked at
  // creation and its typed claims go through the same independent semantic
  // core used by the public portable ScheduleOp verifier.
  auto phaseStarted = std::chrono::steady_clock::now();
  if (authenticateModule && failed(mlir::verify(module)))
    return failure();
  reportPhase("input-verification", phaseStarted);
  NativeScheduler scheduler(module, graph, schedule,
                            /*deferOutputVerification=*/true,
                            /*deferEntryMaterialization=*/true);
  phaseStarted = std::chrono::steady_clock::now();
  if (failed(scheduler.run()))
    return failure();
  reportPhase("construction", phaseStarted);
  llvm::scope_exit eraseSchedule([&] { scheduler.eraseTransientSchedule(); });
  qlx::phys::ScheduleVerificationStats verificationStats;
  phaseStarted = std::chrono::steady_clock::now();
  if (failed(scheduler.verifyTypedClaims(verificationStats)))
    return failure();
  reportPhase("semantic-proof", phaseStarted);
  const size_t rowCount = scheduler.view().rows.size();
  if (verificationStats.semanticVerifierRuns != 1 ||
      verificationStats.claimsVisited != rowCount ||
      verificationStats.hierarchyPostorderVisits != rowCount)
    return module.emitError(
        "fused schedule proof did not make one complete semantic pass");
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-schedule: fused-proof rows=" << rowCount
                 << " serialized=0 parsed=0 semantic-verifier-runs="
                 << verificationStats.semanticVerifierRuns
                 << " hierarchy-postorder-visits="
                 << verificationStats.hierarchyPostorderVisits << "\n";
  phaseStarted = std::chrono::steady_clock::now();
  auto estimate =
      estimateScheduleJSON(module, scheduler.getResultName(), lowerTier,
                           scheduler.view(), fullWorkload);
  if (failed(estimate))
    return failure();
  reportPhase("estimate", phaseStarted);
  return estimate;
}

FailureOr<std::string> qlx::phys::scheduleAndEstimateJSON(ModuleOp module,
                                                          StringRef graph,
                                                          StringRef schedule,
                                                          StringRef lowerTier,
                                                          bool fullWorkload) {
  return scheduleAndEstimateJSONImpl(module, graph, schedule, lowerTier,
                                     fullWorkload,
                                     /*authenticateModule=*/true);
}

FailureOr<std::string> qlx::phys::scheduleVerifiedAndEstimateJSON(
    ModuleOp module, StringRef graph, StringRef schedule, StringRef lowerTier,
    bool fullWorkload) {
  return scheduleAndEstimateJSONImpl(module, graph, schedule, lowerTier,
                                     fullWorkload,
                                     /*authenticateModule=*/false);
}
