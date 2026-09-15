/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/Phys/IR/PhysTypes.h"
#include "qlx/Dialect/Phys/IR/ScheduleVerification.h"
#include "qlx/Dialect/Phys/IR/SpacetimeDerivation.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
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
#include "llvm/ADT/Twine.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Verifier.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <set>
#include <string>

using namespace mlir;
using namespace qlx::phys;

namespace {

struct SpacetimeVerifierRegistration {
  std::string provider;
  std::string providerVersion;
  std::string derivation;
  int64_t derivationVersion;
  qlx::phys::SpacetimePlanVerifier verifier;
  qlx::phys::SpacetimeCallVerifier callVerifier;
};

static std::mutex &spacetimeVerifierMutex() {
  static std::mutex mutex;
  return mutex;
}

static SmallVector<SpacetimeVerifierRegistration> &spacetimeVerifiers() {
  static SmallVector<SpacetimeVerifierRegistration> verifiers;
  return verifiers;
}

} // namespace

void qlx::phys::registerSpacetimePlanVerifier(
    StringRef provider, StringRef providerVersion, StringRef derivation,
    int64_t derivationVersion, SpacetimePlanVerifier verifier,
    SpacetimeCallVerifier callVerifier) {
  if (provider.empty() || providerVersion.empty() || derivation.empty() ||
      derivationVersion <= 0 || !verifier)
    llvm::report_fatal_error(
        "invalid QLX spacetime-plan verifier registration");
  std::lock_guard<std::mutex> lock(spacetimeVerifierMutex());
  for (const SpacetimeVerifierRegistration &existing : spacetimeVerifiers()) {
    if (existing.provider != provider ||
        existing.providerVersion != providerVersion ||
        existing.derivation != derivation ||
        existing.derivationVersion != derivationVersion)
      continue;
    if (existing.verifier != verifier || existing.callVerifier != callVerifier)
      llvm::report_fatal_error(
          "conflicting QLX spacetime-plan verifier registration");
    return;
  }
  spacetimeVerifiers().push_back({provider.str(), providerVersion.str(),
                                  derivation.str(), derivationVersion, verifier,
                                  callVerifier});
}

LogicalResult
qlx::phys::verifyRegisteredSpacetimeDerivation(SpacetimePlanOp plan) {
  SpacetimePlanVerifier verifier = nullptr;
  {
    std::lock_guard<std::mutex> lock(spacetimeVerifierMutex());
    for (const SpacetimeVerifierRegistration &candidate :
         spacetimeVerifiers()) {
      if (candidate.provider == plan.getProvider() &&
          candidate.providerVersion == plan.getProviderVersion() &&
          candidate.derivation == plan.getDerivation() &&
          candidate.derivationVersion == plan.getDerivationVersion()) {
        verifier = candidate.verifier;
        break;
      }
    }
  }
  if (!verifier)
    return plan.emitOpError("unregistered spacetime-plan provider/derivation '")
           << plan.getProvider() << "/" << plan.getDerivation() << "'";
  return verifier(plan);
}

LogicalResult
qlx::phys::verifyRegisteredSpacetimeInvocation(SpacetimeCallOp call,
                                               SpacetimePlanOp plan) {
  SpacetimeCallVerifier verifier = nullptr;
  {
    std::lock_guard<std::mutex> lock(spacetimeVerifierMutex());
    for (const SpacetimeVerifierRegistration &candidate :
         spacetimeVerifiers()) {
      if (candidate.provider == plan.getProvider() &&
          candidate.providerVersion == plan.getProviderVersion() &&
          candidate.derivation == plan.getDerivation() &&
          candidate.derivationVersion == plan.getDerivationVersion()) {
        verifier = candidate.callVerifier;
        break;
      }
    }
  }
  auto source = call.getSourceProtocolAttr();
  if (!source || source == plan.getSourceProtocolAttr())
    return success();
  if (!verifier)
    return call.emitOpError(
        "source_protocol differs from its plan source, but the registered "
        "provider does not authenticate structural plan reuse");
  return verifier(call, plan);
}

using PhysVerifyClock = std::chrono::steady_clock;
static thread_local std::optional<PhysVerifyClock::time_point>
    profiledGraphLocalDone;

static double physVerifySecondsSince(PhysVerifyClock::time_point started) {
  return std::chrono::duration<double>(PhysVerifyClock::now() - started)
      .count();
}

struct SidecarGraphVerificationIndex {
  explicit SidecarGraphVerificationIndex(GraphOp graph) : graph(graph) {
    graph.walk([&](MeasureOp measurement) {
      produced.insert(measurement.getRecordId());
    });
    graph.walk([&](MeasureProductOp measurement) {
      produced.insert(measurement.getRecordId());
    });
    graph.walk([&](CallOp call) {
      invocations[call.getInstance()].push_back(call.getOperation());
    });
    graph.walk([&](CallTemplateOp invocation) {
      invocations[invocation.getInstance()].push_back(
          invocation.getOperation());
      auto aliases = invocation->getAttrOfType<ArrayAttr>("record_aliases");
      if (!aliases)
        return;
      for (Attribute raw : aliases) {
        auto alias = dyn_cast<DictionaryAttr>(raw);
        auto name = alias ? alias.getAs<StringAttr>("alias") : StringAttr{};
        if (name)
          produced.insert(name.getValue());
      }
    });
  }

  GraphOp graph;
  llvm::StringSet<> produced;
  llvm::StringMap<SmallVector<Operation *, 1>> invocations;
};

static thread_local SidecarGraphVerificationIndex
    *activeSidecarGraphVerificationIndex = nullptr;

#include "qlx/Dialect/Phys/IR/PhysDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/Phys/IR/PhysTypes.cpp.inc"

// Two carries of `!phys.state<@resource>` denote the same physical resource
// regardless of SSA identity, so `event.try_take`'s duplicate-owner check
// (via `UniqueCarryOwnerInterface`) keys on the resource symbol itself.
Attribute StateType::getUniqueOwnerKey() const { return getResource(); }

static LogicalResult verifyUniqueStringArray(Operation *owner, StringRef name,
                                             ArrayAttr values) {
  if (!values)
    return success();
  llvm::SmallDenseSet<StringRef, 8> unique;
  for (Attribute value : values) {
    auto text = dyn_cast<StringAttr>(value);
    if (!text || text.getValue().empty())
      return owner->emitOpError()
             << name << " entries must be nonempty strings";
    if (!unique.insert(text.getValue()).second)
      return owner->emitOpError() << "contains duplicate " << name << " entry '"
                                  << text.getValue() << "'";
  }
  return success();
}

LogicalResult ActionOp::verify() {
  if ((*this)->hasAttr("modalities"))
    return emitOpError(
        "modalities was removed; compatibility comes from resource classes "
        "and their native_actions");
  if ((*this)->hasAttr("noise_sites"))
    return emitOpError("noise_sites was removed");
  if (getArity() <= 0)
    return emitOpError("arity must be positive");
  if (getBroadcast() && getArity() != 1)
    return emitOpError("broadcast actions must have arity one");
  if (getProcess().empty())
    return emitOpError("process must be nonempty");
  auto controllerBindings =
      (*this)->getAttrOfType<DictionaryAttr>("controller_bindings");
  for (NamedAttribute binding : controllerBindings
                                    ? controllerBindings
                                    : DictionaryAttr::get(getContext())) {
    if (binding.getName().empty())
      return emitOpError("controller binding names must be nonempty");
    auto semantic = dyn_cast<StringAttr>(binding.getValue());
    if (!semantic || semantic.getValue().empty())
      return emitOpError("controller binding values must be nonempty strings");
  }
  if (failed(verifyUniqueStringArray(
          *this, "parameter", (*this)->getAttrOfType<ArrayAttr>("parameters"))))
    return failure();
  return success();
}

LogicalResult InstrumentOp::verify() {
  if ((*this)->hasAttr("noise_sites"))
    return emitOpError("noise_sites was removed");
  bool hasArity = static_cast<bool>(getArityAttr());
  bool isVariadic = static_cast<bool>(getVariadic());
  if (hasArity == isVariadic)
    return emitOpError("requires exactly one of arity or variadic");
  if (hasArity && getArityAttr().getInt() <= 0)
    return emitOpError("arity must be positive");
  if (getKind().empty() || getRecordSchema().empty())
    return emitOpError("kind and record_schema must be nonempty");
  if (getProcess().empty())
    return emitOpError("process must be nonempty");
  auto controllerBindings =
      (*this)->getAttrOfType<DictionaryAttr>("controller_bindings");
  for (NamedAttribute binding : controllerBindings
                                    ? controllerBindings
                                    : DictionaryAttr::get(getContext())) {
    auto semantic = dyn_cast<StringAttr>(binding.getValue());
    if (binding.getName().empty() || !semantic || semantic.getValue().empty())
      return emitOpError(
          "controller binding names and values must be nonempty strings");
  }
  if (failed(verifyUniqueStringArray(
          *this, "parameter", (*this)->getAttrOfType<ArrayAttr>("parameters"))))
    return failure();
  return success();
}

LogicalResult MeasureOp::verify() {
  if (getRecordId().empty())
    return emitOpError("record_id must be nonempty");
  bool hasOutput = static_cast<bool>(getOutput());
  bool destructive = static_cast<bool>(getDestructive());
  if (hasOutput == destructive)
    return emitOpError("destructive measurement must omit its state result and "
                       "non-destructive measurement must return one state");
  if (hasOutput && getOutput().getType() != getInput().getType())
    return emitOpError("measurement input/output state types must match");

  auto *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getMeasurementAttr());
  if (!target)
    return success(); // Partial linked modules resolve at link time.
  auto instrument = dyn_cast<InstrumentOp>(target);
  if (!instrument)
    return emitOpError("measurement reference must resolve to phys.instrument");
  if (!instrument.getArityAttr() || instrument.getArityAttr().getInt() != 1)
    return emitOpError("phys.measure requires a fixed unary instrument");
  if (instrument.getRecordSchema() !=
      getRecord().getType().getSchema().getValue())
    return emitOpError("record type does not match instrument record_schema");

  if (hasOutput && !instrument.getPreservesInputs())
    return emitOpError(
        "state-preserving phys.measure requires a preserves_inputs instrument");

  StringRef expectedProcess;
  if (instrument.getKind() == "measure")
    expectedProcess = "measure_z";
  else if (instrument.getKind() == "measure_x")
    expectedProcess = "measure_x";
  else
    return emitOpError("instrument must implement measure or measure_x");

  auto process = llvm::json::parse(instrument.getProcess());
  if (!process) {
    llvm::consumeError(process.takeError());
    return emitOpError("instrument process must be valid JSON");
  }
  auto *object = process->getAsObject();
  auto processKind = object ? object->getString("kind") : std::nullopt;
  auto processName = object ? object->getString("name") : std::nullopt;
  if (!processKind || *processKind != "builtin" || !processName ||
      *processName != expectedProcess)
    return emitOpError("instrument process must be builtin ")
           << expectedProcess << " for kind " << instrument.getKind();
  return success();
}

static LogicalResult
verifyResourceAgainstClasses(ResourceOp resource,
                             ArrayRef<ResourceClassOp> relevantClasses);

LogicalResult ArchitectureOp::verify() {
  if ((*this)->hasAttr("modality"))
    return emitOpError(
        "modality was removed; declare resource kinds, native_actions, "
        "native_instruments, and topology instead");

  // Resource declarations are module-level symbols while their classes live
  // in a machine symbol table. Validate that cross-symbol relation once for
  // the whole module instead of rebuilding the same architecture symbol table
  // independently for every physical qubit. The first machine owns the check;
  // graph verification separately checks every resource used by a P3 graph.
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("must be nested in a module");
  auto architectures = module.getOps<ArchitectureOp>();
  if (architectures.empty() || *architectures.begin() != *this)
    return success();
  llvm::StringSet<> graphResources;
  auto rememberStateResource = [&](Type type) {
    if (auto state = dyn_cast<StateType>(type))
      graphResources.insert(state.getResource().getValue());
  };
  for (GraphOp graph : module.getOps<GraphOp>()) {
    for (Type type : graph.getFunctionType().getInputs())
      rememberStateResource(type);
    for (Type type : graph.getFunctionType().getResults())
      rememberStateResource(type);
    graph.walk([&](Operation *operation) {
      for (Type type : operation->getOperandTypes())
        rememberStateResource(type);
      for (Type type : operation->getResultTypes())
        rememberStateResource(type);
    });
  }

  llvm::StringMap<SmallVector<ResourceClassOp, 1>> classesByName;
  for (ArchitectureOp architecture : architectures)
    for (Block &block : architecture.getBody())
      for (ResourceClassOp resourceClass : block.getOps<ResourceClassOp>())
        classesByName[resourceClass.getSymName()].push_back(resourceClass);
  for (ResourceOp resource : module.getOps<ResourceOp>()) {
    // Graph-owned resources are checked against that graph's selected
    // architecture by GraphOp::verify(). A same-named class in an unrelated
    // machine must not make the resource ambiguous. Only unattached resources
    // use the module-wide unique-class rule below.
    if (graphResources.contains(resource.getSymName()))
      continue;
    auto found = classesByName.find(resource.getResourceClass());
    ArrayRef<ResourceClassOp> relevantClasses =
        found == classesByName.end() ? ArrayRef<ResourceClassOp>{}
                                     : ArrayRef<ResourceClassOp>(found->second);
    if (failed(verifyResourceAgainstClasses(resource, relevantClasses)))
      return failure();
  }
  return success();
}

LogicalResult OperatingPointOp::verify() {
  if ((*this)->hasAttr("noise"))
    return emitOpError("noise was removed from physical operating points");
  bool hasDistanceQualifiedTiming = false;
  if (auto timing = getTimingAttr()) {
    for (NamedAttribute entry : timing) {
      StringRef name = entry.getName().strref();
      if (!name.ends_with("_ns"))
        continue;
      StringRef stem = name.drop_back(3);
      size_t marker = stem.rfind("_d");
      if (marker == StringRef::npos)
        continue;
      StringRef distance = stem.drop_front(marker + 2);
      if (distance.empty() || !llvm::all_of(distance, [](char value) {
            return value >= '0' && value <= '9';
          }))
        continue;
      if (distance.front() == '0')
        return emitOpError("distance-qualified timing names require a "
                           "canonical positive code distance");
      hasDistanceQualifiedTiming = true;
    }
  }
  if (hasDistanceQualifiedTiming &&
      (!getTimingSourceAttr() || getTimingSourceAttr().getValue().empty()))
    return emitOpError(
        "distance-qualified timings require a nonempty timing_source");
  return success();
}

LogicalResult SpacetimePlanOp::verify() {
  if (getProvider().empty() || getProviderVersion().empty())
    return emitOpError("provider and provider_version must be nonempty");
  if (getDerivation().empty() || getDerivationVersion() <= 0)
    return emitOpError(
        "derivation must be nonempty with a positive derivation_version");
  if (getEvidence().empty())
    return emitOpError("evidence must be nonempty");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("body must contain exactly one block");
  Block &body = getBody().front();
  if (body.empty())
    return emitOpError("body must contain at least one spacetime operation");

  auto module = (*this)->getParentOfType<ModuleOp>();
  mlir::SymbolTable moduleSymbols(module);
  auto architecture = moduleSymbols.lookup<ArchitectureOp>(getArchitecture());
  if (!architecture)
    return emitOpError("architecture must resolve to phys.machine");
  auto source = moduleSymbols.lookup(getSourceProtocol());
  if (!source || !isa<qlx::fabric::ProtocolOp>(source))
    return emitOpError("source_protocol must resolve to fabric.protocol");
  auto operatingPoint =
      moduleSymbols.lookup<OperatingPointOp>(getOperatingPoint());
  if (!operatingPoint ||
      operatingPoint.getMachineAttr() != getArchitectureAttr())
    return emitOpError(
        "operating_point must resolve for the selected architecture");
  if (auto geometry = getGeometryAttr())
    for (int64_t extent : geometry.asArrayRef())
      if (extent <= 0)
        return emitOpError("geometry extents must be positive");
  bool hasRecurrenceKind = static_cast<bool>(getRecurrenceResourceKindAttr());
  bool hasRecurrenceOutput = static_cast<bool>(getRecurrenceOutputEventAttr());
  bool hasForwarding = static_cast<bool>(getForwardingLatencyNsAttr());
  bool hasInterval = static_cast<bool>(getInitiationIntervalNsAttr());
  if (hasRecurrenceKind != hasRecurrenceOutput)
    return emitOpError(
        "recurrence resource kind and output event must appear together");
  if (hasRecurrenceKind) {
    if (getRecurrenceOutputEventAttr().getValue().empty())
      return emitOpError("recurrence output event must be nonempty");
    if (hasForwarding || hasInterval)
      return emitOpError(
          "recurring factory plans cannot carry callable pipeline timing");
  }
  if (hasForwarding != hasInterval)
    return emitOpError(
        "callable pipeline forwarding latency and initiation interval must "
        "appear together");
  if (hasForwarding) {
    double forwarding = getForwardingLatencyNsAttr().getValueAsDouble();
    double interval = getInitiationIntervalNsAttr().getValueAsDouble();
    if (!std::isfinite(forwarding) || forwarding <= 0.0 ||
        !std::isfinite(interval) || interval <= 0.0)
      return emitOpError(
          "callable pipeline timings must be finite and positive");
    if (auto semantics = getIntervalSemanticsAttr()) {
      if (semantics.getValue() != "pipelined" &&
          semantics.getValue() != "backpressured")
        return emitOpError(
            "interval_semantics must be 'pipelined' or 'backpressured'");
      if (interval > forwarding && semantics.getValue() != "backpressured")
        return emitOpError(
            "initiation interval greater than one-shot latency requires "
            "backpressured semantics");
    }
  }
  if (auto policy = getPolicyAttr())
    if (policy.getValue() != "guaranteed" && policy.getValue() != "single_shot")
      return emitOpError("policy must be 'guaranteed' or 'single_shot'");
  for (Operation &operation : body) {
    if (hasRecurrenceKind && !isa<SpacetimeEventOp>(operation))
      return emitOpError(
          "recurrence plans may contain only phys.spacetime_event");
    if (!hasRecurrenceKind && !isa<SpacetimePhaseOp>(operation))
      return emitOpError(
          "non-recurrence plans may contain only phys.spacetime_phase");
  }
  return verifyRegisteredSpacetimeDerivation(*this);
}

LogicalResult SpacetimePhaseOp::verify() {
  auto plan = (*this)->getParentOfType<SpacetimePlanOp>();
  if (!plan)
    return emitOpError("must be nested in phys.spacetime_plan");
  if (getSteps() <= 0)
    return emitOpError("steps must be positive");
  double stepDuration = getStepDurationNs().convertToDouble();
  if (!std::isfinite(stepDuration) || stepDuration <= 0.0)
    return emitOpError("step_duration_ns must be finite and positive");
  double duration = stepDuration * static_cast<double>(getSteps());
  if (!std::isfinite(duration) || duration <= 0.0)
    return emitOpError(
        "steps times step_duration_ns must be finite and positive");
  if (getResourceClasses().empty() && !getResourceClaimsAttr() &&
      getFactoryModels().empty())
    return emitOpError(
        "phase must claim at least one physical resource or factory model");

  auto architecture = dyn_cast_or_null<ArchitectureOp>(
      mlir::SymbolTable(plan->getParentOfType<ModuleOp>())
          .lookup(plan.getArchitecture()));
  if (!architecture)
    return emitOpError("parent plan architecture must resolve to phys.machine");
  llvm::SmallDenseSet<Attribute, 8> resources;
  for (Attribute value : getResourceClasses()) {
    auto reference = dyn_cast<SymbolRefAttr>(value);
    if (!reference || !resources.insert(reference).second)
      return emitOpError(
          "resource_classes must contain unique symbol references");
    if (reference.getRootReference().getValue() != plan.getArchitecture() ||
        reference.getNestedReferences().size() != 1)
      return emitOpError("resource class ")
             << reference
             << " must be nested directly in the selected architecture";
    auto resource =
        SymbolTable(architecture)
            .lookup<ResourceClassOp>(reference.getLeafReference().getValue());
    if (!resource || resource->getParentOp() != architecture.getOperation())
      return emitOpError("resource class ")
             << reference << " must resolve in the selected architecture";
  }
  if (auto claims = getResourceClaimsAttr()) {
    llvm::StringMap<SmallVector<std::pair<int64_t, int64_t>, 2>> intervals;
    for (Attribute raw : claims) {
      auto claim = dyn_cast<DictionaryAttr>(raw);
      auto reference = claim ? claim.getAs<SymbolRefAttr>("resource_class")
                             : SymbolRefAttr{};
      auto offset = claim ? claim.getAs<IntegerAttr>("offset") : IntegerAttr{};
      auto count = claim ? claim.getAs<IntegerAttr>("count") : IntegerAttr{};
      auto units = claim ? claim.getAs<IntegerAttr>("units") : IntegerAttr{};
      if (!reference || !offset || !count || !units || claim.size() != 4)
        return emitOpError(
            "resource_claims must contain exact typed resource_class, offset, "
            "count, and units records");
      if (reference.getRootReference().getValue() != plan.getArchitecture() ||
          reference.getNestedReferences().size() != 1)
        return emitOpError(
            "resource claim must name a class in the selected architecture");
      auto resource =
          SymbolTable(architecture)
              .lookup<ResourceClassOp>(reference.getLeafReference().getValue());
      int64_t begin = offset.getInt();
      int64_t width = count.getInt();
      if (!resource || begin < 0 || width <= 0 || begin > resource.getCount() ||
          width > resource.getCount() - begin)
        return emitOpError(
            "resource claim slice must fit inside its physical class");
      if (units.getInt() != width)
        return emitOpError(
            "synchronous phase claims must acquire their complete slice");
      int64_t end = begin + width;
      auto &selected = intervals[reference.getLeafReference().getValue()];
      for (auto [otherBegin, otherEnd] : selected)
        if (begin < otherEnd && otherBegin < end)
          return emitOpError("resource_claims must not overlap");
      selected.push_back({begin, end});
      for (Attribute legacy : getResourceClasses())
        if (legacy == reference)
          return emitOpError(
              "one resource class cannot be claimed by both legacy and exact "
              "phase claims");
    }
  }

  llvm::SmallDenseSet<Attribute, 8> models;
  for (Attribute value : getFactoryModels()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference || !models.insert(reference).second)
      return emitOpError(
          "factory_models must contain unique flat symbol references");
    if (!isa_and_nonnull<FactoryModelOp>(
            mlir::SymbolTable(plan->getParentOfType<ModuleOp>())
                .lookup(reference.getValue())))
      return emitOpError("factory model ")
             << reference << " must resolve to phys.factory_model";
  }

  llvm::SmallDenseSet<Attribute, 8> dependencies;
  llvm::SmallDenseSet<StringRef, 8> earlier;
  for (Operation &candidate : plan.getBody().front()) {
    if (&candidate == getOperation())
      break;
    if (auto phase = dyn_cast<SpacetimePhaseOp>(candidate))
      earlier.insert(phase.getSymName());
  }
  for (Attribute value : getAfter()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference || !dependencies.insert(reference).second)
      return emitOpError("after must contain unique flat symbol references");
    if (!earlier.contains(reference.getValue()))
      return emitOpError("after dependency @")
             << reference.getValue()
             << " must name an earlier phase in this plan";
  }
  return success();
}

LogicalResult SpacetimeEventOp::verify() {
  auto plan = (*this)->getParentOfType<SpacetimePlanOp>();
  if (!plan || !plan.getRecurrenceResourceKindAttr())
    return emitOpError("must be nested in a recurrence phys.spacetime_plan");
  if (getKind().empty())
    return emitOpError("kind must be nonempty");
  if (getIteration() < 0)
    return emitOpError("iteration must be nonnegative");
  double start = getStartNs().convertToDouble();
  double duration = getDurationNs().convertToDouble();
  if (!std::isfinite(start) || start < 0.0)
    return emitOpError("start_ns must be finite and nonnegative");
  if (!std::isfinite(duration) || duration <= 0.0 ||
      !std::isfinite(start + duration))
    return emitOpError(
        "duration_ns and event finish must be finite and positive");

  bool hasClass = static_cast<bool>(getResourceClassAttr());
  bool hasOffset = static_cast<bool>(getResourceOffsetAttr());
  bool hasCount = static_cast<bool>(getResourceCountAttr());
  if (hasClass != hasOffset || hasClass != hasCount)
    return emitOpError(
        "resource_class, resource_offset, and resource_count must appear "
        "together");
  bool hasFactory = static_cast<bool>(getFactoryModelAttr());
  bool hasFactoryUnits = static_cast<bool>(getFactoryUnitsAttr());
  if (hasFactory != hasFactoryUnits)
    return emitOpError("factory_model and factory_units must appear together");
  if (!hasClass && !hasFactory)
    return emitOpError(
        "event must claim a physical resource slice or input factory");

  auto module = plan->getParentOfType<ModuleOp>();
  SymbolTable moduleSymbols(module);
  auto architecture =
      moduleSymbols.lookup<ArchitectureOp>(plan.getArchitecture());
  if (!architecture)
    return emitOpError("parent plan architecture must resolve to phys.machine");
  ResourceClassOp resource;
  if (hasClass) {
    SymbolRefAttr reference = getResourceClassAttr();
    if (reference.getRootReference().getValue() != plan.getArchitecture() ||
        reference.getNestedReferences().size() != 1)
      return emitOpError(
          "resource_class must be nested directly in the plan architecture");
    resource =
        SymbolTable(architecture)
            .lookup<ResourceClassOp>(reference.getLeafReference().getValue());
    int64_t offset = getResourceOffsetAttr().getInt();
    int64_t count = getResourceCountAttr().getInt();
    if (!resource || offset < 0 || count <= 0 || offset > resource.getCount() ||
        count > resource.getCount() - offset)
      return emitOpError(
          "resource slice must resolve and fit inside its physical class");
  }
  if (hasFactory) {
    auto model =
        moduleSymbols.lookup<FactoryModelOp>(getFactoryModelAttr().getValue());
    if (!model || model.getOperatingPointAttr() != plan.getOperatingPointAttr())
      return emitOpError(
          "factory_model must resolve at the recurrence operating point");
    if (getFactoryUnitsAttr().getInt() <= 0)
      return emitOpError("factory_units must be positive");
  }
  // Recurring layouts may expose typed intermediate products as well as the
  // final resource named by the parent plan.  The registered plan verifier
  // authenticates the exact intermediate/final sequence; the generic event
  // verifier must not reinterpret every typed output as the recurrence result.

  llvm::SmallDenseSet<Attribute, 8> dependencies;
  llvm::StringMap<SpacetimeEventOp> earlier;
  for (Operation &candidate : plan.getBody().front()) {
    if (&candidate == getOperation())
      break;
    if (auto event = dyn_cast<SpacetimeEventOp>(candidate))
      earlier[event.getSymName()] = event;
  }
  for (Attribute raw : getAfter()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
    if (!reference || !dependencies.insert(reference).second)
      return emitOpError("after must contain unique flat symbol references");
    auto found = earlier.find(reference.getValue());
    if (found == earlier.end())
      return emitOpError("after dependency @")
             << reference.getValue()
             << " must name an earlier event in this plan";
    double finish = found->second.getStartNs().convertToDouble() +
                    found->second.getDurationNs().convertToDouble();
    if (finish > start)
      return emitOpError("starts before dependency @")
             << reference.getValue() << " finishes";
  }

  if (resource) {
    int64_t offset = getResourceOffsetAttr().getInt();
    int64_t end = offset + getResourceCountAttr().getInt();
    double finish = start + duration;
    for (const auto &entry : earlier) {
      SpacetimeEventOp prior = entry.getValue();
      if (prior.getResourceClassAttr() != getResourceClassAttr())
        continue;
      int64_t priorOffset = prior.getResourceOffsetAttr().getInt();
      int64_t priorEnd = priorOffset + prior.getResourceCountAttr().getInt();
      double priorStart = prior.getStartNs().convertToDouble();
      double priorFinish = priorStart + prior.getDurationNs().convertToDouble();
      if (offset < priorEnd && priorOffset < end && start < priorFinish &&
          priorStart < finish)
        return emitOpError("overlaps physical resource slice with @")
               << prior.getSymName();
    }
  }
  return success();
}

LogicalResult ResourceClassOp::verify() {
  if (getCountAttr().getInt() < 0)
    return emitOpError("count must be nonnegative");
  auto granularity = (*this)->getAttrOfType<StringAttr>("granularity");
  StringRef granularityValue = granularity ? granularity.getValue() : "carrier";
  if (granularityValue != "carrier" && granularityValue != "patch")
    return emitOpError("granularity must be 'carrier' or 'patch'");
  auto unitKind = (*this)->getAttrOfType<StringAttr>("physical_unit_kind");
  auto units = (*this)->getAttrOfType<IntegerAttr>("physical_units");
  auto evidence = (*this)->getAttrOfType<StringAttr>("footprint_evidence");
  bool hasAnyFootprint = unitKind || units || evidence;
  if (granularityValue == "patch") {
    if (!unitKind || unitKind.getValue().empty() || !units ||
        units.getInt() <= 0 || !evidence || evidence.getValue().empty())
      return emitOpError(
          "patch granularity requires nonempty physical_unit_kind and "
          "footprint_evidence plus positive physical_units");
  } else if (hasAnyFootprint) {
    return emitOpError(
        "carrier granularity must not declare a patch physical footprint");
  }
  llvm::SmallDenseSet<StringRef, 8> unique;
  for (Attribute value : getNativeActions()) {
    auto ref = dyn_cast<FlatSymbolRefAttr>(value);
    auto compatibilityName = dyn_cast<StringAttr>(value);
    if (!ref && (!compatibilityName || compatibilityName.getValue().empty()))
      return emitOpError(
          "native_actions entries must be action references or nonempty "
          "compatibility strings");
    StringRef name = ref ? ref.getValue() : compatibilityName.getValue();
    if (!unique.insert(name).second)
      return emitOpError("contains duplicate native action '") << name << "'";
    if (!ref)
      continue;
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, ref);
    if (!target)
      if (auto module = (*this)->getParentOfType<ModuleOp>())
        target = SymbolTable(module).lookup(ref.getValue());
    if (target && !isa<ActionOp>(target))
      return emitOpError("native action @")
             << ref.getValue() << " must resolve to phys.action";
  }
  if (auto instruments =
          (*this)->getAttrOfType<ArrayAttr>("native_instruments")) {
    llvm::SmallDenseSet<StringRef, 8> instrumentNames;
    for (Attribute value : instruments) {
      auto ref = dyn_cast<FlatSymbolRefAttr>(value);
      if (!ref)
        return emitOpError(
            "native_instruments entries must be instrument references");
      if (!instrumentNames.insert(ref.getValue()).second)
        return emitOpError("contains duplicate native instrument '")
               << ref.getValue() << "'";
      Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, ref);
      if (!target)
        if (auto module = (*this)->getParentOfType<ModuleOp>())
          target = SymbolTable(module).lookup(ref.getValue());
      if (target && !isa<InstrumentOp>(target))
        return emitOpError("native instrument @")
               << ref.getValue() << " must resolve to phys.instrument";
    }
  }
  return success();
}

static FailureOr<std::pair<int64_t, int64_t>>
physicalEdgeEndpoints(Operation *owner, Attribute value) {
  auto edge = dyn_cast<DenseI64ArrayAttr>(value);
  if (!edge || edge.size() != 2) {
    owner->emitOpError("topology edges must be two-element i64 arrays");
    return failure();
  }
  return std::pair<int64_t, int64_t>{edge[0], edge[1]};
}

static bool physicalTopologyHasEdge(TopologyOp topology, int64_t source,
                                    int64_t target) {
  auto edges = topology.getEdges();
  if (!edges)
    return false;
  for (Attribute value : *edges) {
    auto endpoints = physicalEdgeEndpoints(topology, value);
    if (failed(endpoints))
      return false;
    bool forward = endpoints->first == source && endpoints->second == target;
    bool reverse = endpoints->first == target && endpoints->second == source;
    if (forward || reverse)
      return true;
  }
  return false;
}

LogicalResult TopologyOp::verify() {
  if (getKind().empty())
    return emitOpError("kind must be nonempty");
  std::optional<int64_t> numNodes;
  if (auto value = getNumNodes()) {
    if (*value < 0)
      return emitOpError("num_nodes must be nonnegative");
    numNodes = *value;
  }
  if (getStrict() && !numNodes)
    return emitOpError("strict topology requires num_nodes");
  llvm::DenseSet<std::pair<int64_t, int64_t>> seen;
  if (auto edges = getEdges()) {
    if (!numNodes)
      return emitOpError("canonical topology edges require num_nodes");
    for (Attribute value : *edges) {
      auto endpoints = physicalEdgeEndpoints(*this, value);
      if (failed(endpoints))
        return failure();
      auto [source, target] = *endpoints;
      if (source < 0 || target <= source || target >= *numNodes)
        return emitOpError("edge references invalid topology nodes");
      if (!seen.insert({source, target}).second)
        return emitOpError("contains duplicate edge");
    }
  }
  if (auto coordinates = getCoordinates()) {
    if (!numNodes)
      return emitOpError("topology coordinates require num_nodes");
    if (static_cast<int64_t>(coordinates->size()) != *numNodes)
      return emitOpError(
          "coordinates must contain one entry per topology node");
    llvm::DenseSet<std::pair<int64_t, int64_t>> occupied;
    for (Attribute value : *coordinates) {
      auto coordinate = dyn_cast<DenseI64ArrayAttr>(value);
      if (!coordinate || coordinate.size() != 2)
        return emitOpError("coordinates must be two-element i64 arrays");
      if (!occupied.insert({coordinate[0], coordinate[1]}).second)
        return emitOpError("coordinates must be unique");
    }
  }
  return success();
}

LogicalResult PatchTopologyOp::verify() {
  int64_t capacity = getCapacity();
  if (capacity <= 0)
    return emitOpError("capacity must be positive");
  if (static_cast<int64_t>(getCarrierGroups().size()) != capacity ||
      static_cast<int64_t>(getCategories().size()) != capacity)
    return emitOpError(
        "carrier_groups and categories must have one entry per implicit slot");

  auto architecture = (*this)->getParentOfType<ArchitectureOp>();
  if (!architecture)
    return emitOpError("must be nested in phys.machine");
  auto carrierTopology = dyn_cast_or_null<TopologyOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getCarrierTopologyAttr()));
  auto resourceClass = dyn_cast_or_null<ResourceClassOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getResourceClassAttr()));
  if (!carrierTopology)
    return emitOpError("carrier_topology must resolve to phys.topology");
  if (!resourceClass)
    return emitOpError("resource_class must resolve to phys.resource_class");
  auto carrierNodeCount = carrierTopology.getNumNodes();
  if (!carrierTopology.getStrict() || !carrierNodeCount)
    return emitOpError("carrier_topology must be strict with num_nodes");
  llvm::DenseMap<int64_t, int64_t> owner;
  llvm::SmallVector<llvm::SmallVector<int64_t>> groups;
  for (auto [slot, value] : llvm::enumerate(getCarrierGroups())) {
    auto group = dyn_cast<DenseI64ArrayAttr>(value);
    if (!group || group.empty())
      return emitOpError("carrier_groups entries must be nonempty i64 arrays");
    llvm::SmallVector<int64_t> members;
    for (int64_t carrier : group.asArrayRef()) {
      if (carrier < 0 || carrier >= resourceClass.getCount() ||
          carrier >= *carrierNodeCount)
        return emitOpError("carrier group references an unavailable carrier");
      if (!owner.try_emplace(carrier, slot).second)
        return emitOpError("carrier groups must be pairwise disjoint");
      members.push_back(carrier);
    }
    groups.push_back(std::move(members));
  }
  for (Attribute value : getCategories())
    if (!isa<StringAttr>(value))
      return emitOpError("categories entries must be strings");

  llvm::DenseSet<std::pair<int64_t, int64_t>> declaredEdges;
  for (Attribute value : getEdges()) {
    auto edge = dyn_cast<DenseI64ArrayAttr>(value);
    if (!edge || edge.size() != 2)
      return emitOpError("edges must be two-element i64 arrays");
    int64_t left = edge[0], right = edge[1];
    if (left < 0 || right <= left || right >= capacity ||
        !declaredEdges.insert({left, right}).second)
      return emitOpError("edges must be unique ordered pairs of valid slots");
  }

  llvm::DenseMap<int64_t, llvm::SmallVector<int64_t>> neighbors;
  if (auto edges = carrierTopology.getEdges()) {
    for (Attribute value : *edges) {
      auto endpoints = physicalEdgeEndpoints(carrierTopology, value);
      if (failed(endpoints))
        return failure();
      neighbors[endpoints->first].push_back(endpoints->second);
      neighbors[endpoints->second].push_back(endpoints->first);
    }
  }
  for (int64_t left = 0; left < capacity; ++left) {
    for (int64_t right = left + 1; right < capacity; ++right) {
      llvm::DenseSet<int64_t> targets(groups[right].begin(),
                                      groups[right].end());
      llvm::SmallVector<int64_t> queue(groups[left].begin(),
                                       groups[left].end());
      llvm::DenseSet<int64_t> visited(queue.begin(), queue.end());
      bool connected = false;
      for (size_t cursor = 0; cursor < queue.size() && !connected; ++cursor) {
        for (int64_t next : neighbors[queue[cursor]]) {
          if (targets.contains(next)) {
            connected = true;
            break;
          }
          if (visited.contains(next) || owner.contains(next))
            continue;
          visited.insert(next);
          queue.push_back(next);
        }
      }
      if (declaredEdges.contains({left, right}) != connected)
        return emitOpError("edges must exactly match structural paths through "
                           "unassigned carriers");
    }
  }
  return success();
}

LogicalResult QECBindingOp::verify() {
  auto architecture = (*this)->getParentOfType<ArchitectureOp>();
  if (!architecture)
    return emitOpError("must be nested in phys.machine");

  auto module = (*this)->getParentOfType<ModuleOp>();
  auto qecReference = getQecRegionAttr();
  Operation *qecMachine =
      module
          ? SymbolTable::lookupSymbolIn(module, qecReference.getRootReference())
          : nullptr;
  Operation *qecRegion =
      qecMachine ? SymbolTable(qecMachine)
                       .lookup(qecReference.getLeafReference().getValue())
                 : nullptr;
  if (qecMachine &&
      (!qecRegion || qecRegion->getName().getStringRef() != "fabric.region"))
    return emitOpError("qec_region must resolve to fabric.region");

  if (getResources().empty())
    return emitOpError("requires at least one bound resource class");
  SymbolTable architectureSymbols(architecture);
  llvm::SmallDenseSet<StringRef, 8> resources;
  for (Attribute value : getResources()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference || !resources.insert(reference.getValue()).second)
      return emitOpError(
          "resources must contain unique flat symbol references");
    if (!isa_and_nonnull<ResourceClassOp>(
            architectureSymbols.lookup(reference.getValue())))
      return emitOpError(
          "resources must resolve to phys.resource_class in this machine");
  }

  if (getTopology()) {
    auto topology = dyn_cast_or_null<TopologyOp>(
        architectureSymbols.lookup(getTopologyAttr().getValue()));
    if (!topology)
      return emitOpError("topology must resolve to phys.topology");
  }
  if (!getPatchTopology())
    return success();
  auto patch = dyn_cast_or_null<PatchTopologyOp>(
      architectureSymbols.lookup(getPatchTopologyAttr().getValue()));
  if (!patch)
    return emitOpError("patch_topology must resolve to phys.patch_topology");
  if (getResources().size() != 1)
    return emitOpError(
        "patch_topology currently requires exactly one bound resource class");
  if (!getTopology() || *getTopology() != patch.getCarrierTopology())
    return emitOpError(
        "patch_topology must derive from this binding's carrier topology");
  bool ownsResource = llvm::any_of(getResources(), [&](Attribute value) {
    auto ref = dyn_cast<FlatSymbolRefAttr>(value);
    return ref && ref.getValue() == patch.getResourceClass();
  });
  if (!ownsResource)
    return emitOpError(
        "patch_topology resource_class must belong to this space binding");
  if (Operation *region =
          SymbolTable::lookupNearestSymbolFrom(*this, getQecRegionAttr())) {
    if (region->getName().getStringRef() == "fabric.region")
      if (auto capacity = region->getAttrOfType<IntegerAttr>("block_capacity"))
        if (capacity.getInt() != patch.getCapacity())
          return emitOpError(
              "patch_topology capacity must equal QEC block capacity");
  }
  return success();
}

LogicalResult QECChannelBindingOp::verify() {
  auto architecture = (*this)->getParentOfType<ArchitectureOp>();
  if (!architecture)
    return emitOpError("must be nested in phys.machine");

  auto module = (*this)->getParentOfType<ModuleOp>();
  auto qecReference = getQecChannelAttr();
  Operation *qecMachine =
      module
          ? SymbolTable::lookupSymbolIn(module, qecReference.getRootReference())
          : nullptr;
  Operation *qecChannel =
      qecMachine ? SymbolTable(qecMachine)
                       .lookup(qecReference.getLeafReference().getValue())
                 : nullptr;
  if (qecMachine && (!qecChannel || qecChannel->getName().getStringRef() !=
                                        "fabric.interconnect"))
    return emitOpError("qec_channel must resolve to fabric.interconnect");

  if (getResources().empty())
    return emitOpError("requires at least one bound resource class");
  SymbolTable architectureSymbols(architecture);
  llvm::SmallDenseSet<StringRef, 8> resources;
  for (Attribute value : getResources()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference || !resources.insert(reference.getValue()).second)
      return emitOpError(
          "resources must contain unique flat symbol references");
    if (!isa_and_nonnull<ResourceClassOp>(
            architectureSymbols.lookup(reference.getValue())))
      return emitOpError(
          "resources must resolve to phys.resource_class in this machine");
  }
  bool hasClaims = static_cast<bool>(getTransportClaimsAttr());
  bool hasSource = static_cast<bool>(getSourceEndpointOccupancyAttr());
  bool hasDestination =
      static_cast<bool>(getDestinationEndpointOccupancyAttr());
  if (hasClaims != hasSource || hasClaims != hasDestination)
    return emitOpError(
        "transport_claims and both endpoint occupancy facts must appear "
        "together");
  if (!hasClaims)
    return success();
  if (getTransportClaimsAttr().empty())
    return emitOpError("transport_claims must be nonempty when present");
  auto interconnect = dyn_cast_or_null<qlx::fabric::InterconnectOp>(qecChannel);
  if (!interconnect)
    return emitOpError(
        "detailed transport facts require a resolved fabric.interconnect");
  int64_t sourceUnits = getSourceEndpointOccupancyAttr().getInt();
  int64_t destinationUnits = getDestinationEndpointOccupancyAttr().getInt();
  if (sourceUnits <= 0 || destinationUnits <= 0 ||
      sourceUnits > interconnect.getPortAConcurrency() ||
      destinationUnits > interconnect.getPortBConcurrency() ||
      std::max(sourceUnits, destinationUnits) > interconnect.getConcurrency())
    return emitOpError(
        "endpoint occupancy must fit the selected channel and ports");
  llvm::StringMap<SmallVector<std::pair<int64_t, int64_t>, 2>> intervals;
  for (Attribute raw : getTransportClaimsAttr()) {
    auto claim = dyn_cast<DictionaryAttr>(raw);
    auto reference = claim ? claim.getAs<FlatSymbolRefAttr>("resource_class")
                           : FlatSymbolRefAttr{};
    auto offset = claim ? claim.getAs<IntegerAttr>("offset") : IntegerAttr{};
    auto count = claim ? claim.getAs<IntegerAttr>("count") : IntegerAttr{};
    auto units = claim ? claim.getAs<IntegerAttr>("units") : IntegerAttr{};
    if (!reference || !offset || !count || !units || claim.size() != 4 ||
        !resources.contains(reference.getValue()))
      return emitOpError(
          "transport_claims must contain exact bound resource_class, offset, "
          "count, and units records");
    auto resource =
        architectureSymbols.lookup<ResourceClassOp>(reference.getValue());
    int64_t begin = offset.getInt();
    int64_t width = count.getInt();
    int64_t acquired = units.getInt();
    if (!resource || begin < 0 || width <= 0 || acquired <= 0 ||
        acquired > width || begin > resource.getCount() ||
        width > resource.getCount() - begin)
      return emitOpError(
          "transport claim slice/acquisition must fit its physical class");
    int64_t end = begin + width;
    for (auto [otherBegin, otherEnd] : intervals[reference.getValue()])
      if (begin < otherEnd && otherBegin < end)
        return emitOpError("transport_claims must not overlap");
    intervals[reference.getValue()].push_back({begin, end});
  }
  return success();
}

static LogicalResult
verifyResourceAgainstClasses(ResourceOp resource,
                             ArrayRef<ResourceClassOp> relevantClasses) {
  if (relevantClasses.empty())
    return resource.emitOpError("resource_class ")
           << resource.getResourceClassAttr()
           << " must resolve in a phys.machine";
  if (relevantClasses.size() != 1)
    return resource.emitOpError("resource_class ")
           << resource.getResourceClassAttr()
           << " is ambiguous across physical architectures";

  ResourceClassOp resourceClass = relevantClasses.front();
  if (resource.getKind() != resourceClass.getKind())
    return resource.emitOpError("kind must match resource class @")
           << resourceClass.getSymName() << " kind '" << resourceClass.getKind()
           << "'";
  if (resource.getIndex() >= resourceClass.getCount())
    return resource.emitOpError("index is outside resource class @")
           << resourceClass.getSymName() << " capacity "
           << resourceClass.getCount();
  return success();
}

LogicalResult ResourceOp::verify() {
  if (getKind().empty())
    return emitOpError("kind must be nonempty");
  if (getIndex() < 0)
    return emitOpError("index must be nonnegative");
  if (auto distance = getCodeDistanceAttr()) {
    if (distance.getInt() <= 0)
      return emitOpError("code_distance must be positive");
    auto qecReference = getQecRegionAttr();
    if (!qecReference)
      return emitOpError("code_distance requires qec_region");
    auto module = (*this)->getParentOfType<ModuleOp>();
    Operation *qecMachine =
        module ? SymbolTable::lookupSymbolIn(module,
                                             qecReference.getRootReference())
               : nullptr;
    auto qecRegion =
        qecMachine
            ? dyn_cast_or_null<qlx::fabric::RegionOp>(
                  SymbolTable(qecMachine)
                      .lookup(qecReference.getLeafReference().getValue()))
            : qlx::fabric::RegionOp{};
    auto code =
        qecRegion
            ? dyn_cast_or_null<qlx::fabric::CodeOp>(SymbolTable(module).lookup(
                  qecRegion.getCodeAttr().getValue()))
            : qlx::fabric::CodeOp{};
    if (!code)
      return emitOpError(
          "code_distance requires qec_region to resolve a fabric.code");
    if (distance.getInt() != code.getDistance())
      return emitOpError(
          "code_distance must equal the selected QEC region code distance");
  }

  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("must be nested in a module");
  // With no machine declaration there is no ArchitectureOp verifier to own
  // the module-level cross-symbol check. Fail closed here rather than allowing
  // a detached resource with an unresolvable class to verify.
  if (module.getOps<ArchitectureOp>().empty())
    return verifyResourceAgainstClasses(*this, {});
  return success();
}

LogicalResult ResourceOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto architectureRef = getArchitectureAttr();
  if (!architectureRef)
    return success();
  auto architecture = symbolTable.lookupNearestSymbolFrom<ArchitectureOp>(
      getOperation(), architectureRef);
  auto resourceClass = architecture
                           ? symbolTable.lookupSymbolIn<ResourceClassOp>(
                                 architecture, getResourceClassAttr())
                           : ResourceClassOp{};
  if (!architecture || !resourceClass)
    return emitOpError("architecture and resource_class must resolve to one "
                       "physical resource class");
  if (getKind() != resourceClass.getKind())
    return emitOpError("kind must match resource class @")
           << resourceClass.getSymName() << " kind '" << resourceClass.getKind()
           << "'";
  if (getIndex() >= resourceClass.getCount())
    return emitOpError("index is outside resource class @")
           << resourceClass.getSymName() << " capacity "
           << resourceClass.getCount();
  return success();
}

static LogicalResult verifyApplyActionShape(ApplyOp apply, ActionOp action) {
  int64_t batchLanes =
      apply.getBatchLanesAttr() ? apply.getBatchLanesAttr().getInt() : 1;
  if (action.getBroadcast() && apply.getBatchLanesAttr())
    return apply.emitOpError("broadcast actions cannot also carry batch_lanes");
  if (!action.getBroadcast() &&
      static_cast<int64_t>(apply.getInputs().size()) !=
          action.getArity() * batchLanes)
    return apply.emitOpError("operand count does not match action @")
           << action.getSymName() << " arity " << action.getArity()
           << " across " << batchLanes << " lane(s)";
  if (action.getBroadcast() && action.getArity() != 1)
    return apply.emitOpError("broadcast action must have arity one");
  return success();
}

LogicalResult ApplyOp::verify() {
  if ((*this)->hasAttr("noise_sites"))
    return emitOpError("noise_sites was removed");
  if (getInputs().empty())
    return emitOpError("requires at least one physical state operand");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every physical state type");
  auto batchLanes = getBatchLanesAttr();
  if (batchLanes && batchLanes.getInt() <= 1)
    return emitOpError("batch_lanes must be greater than one when present");
  if (batchLanes) {
    llvm::SmallDenseSet<Attribute, 8> resources;
    for (Value input : getInputs()) {
      auto state = cast<StateType>(input.getType());
      if (!resources.insert(state.getResource()).second)
        return emitOpError(
            "batched action lanes must own distinct physical resources");
    }
  }
  if (auto resources = getResources()) {
    if (resources->size() != getInputs().size())
      return emitOpError("resources must name every physical state operand");
    for (auto [value, input] : llvm::zip(*resources, getInputs())) {
      auto reference = dyn_cast<FlatSymbolRefAttr>(value);
      auto state = dyn_cast<StateType>(input.getType());
      if (!reference || !state || reference != state.getResource())
        return emitOpError(
            "resources must match the resource carried by each state type");
    }
  }
  // GraphOp owns architecture, capability, resource-class, and topology
  // validation. A resolvable action outside a graph must still prove its local
  // shape at the ordinary operation-verification boundary.
  if (!(*this)->getParentOfType<GraphOp>()) {
    Operation *target =
        SymbolTable::lookupNearestSymbolFrom(*this, getActionAttr());
    if (target) {
      auto action = dyn_cast<ActionOp>(target);
      if (!action)
        return emitOpError("action reference must resolve to phys.action");
      if (failed(verifyApplyActionShape(*this, action)))
        return failure();
    }
  }
  // Cross-symbol action, resource-class, and topology obligations are owned by
  // the enclosing graph. GraphOp indexes the linked architecture once and
  // validates all applies in one linear pass.
  return success();
}

static LogicalResult
verifyMappingRecords(Operation *owner, ArrayAttr values, StringRef label,
                     llvm::SmallDenseSet<StringRef, 16> &roles,
                     SymbolTable &symbols) {
  for (Attribute value : values) {
    auto record = dyn_cast<DictionaryAttr>(value);
    auto role = record ? record.getAs<StringAttr>("role") : nullptr;
    auto resource =
        record ? record.getAs<FlatSymbolRefAttr>("resource") : nullptr;
    if (!record || !role || role.getValue().empty() || !resource)
      return owner->emitOpError()
             << label << " entries require role string and resource reference";
    if (!roles.insert(role.getValue()).second)
      return owner->emitOpError()
             << label << " contains duplicate role '" << role.getValue() << "'";
    Operation *target = symbols.lookup(resource.getValue());
    if (!target || !isa<ResourceOp>(target))
      return owner->emitOpError() << label << " resource " << resource
                                  << " must resolve to phys.resource";
  }
  return success();
}

LogicalResult MappingOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module)
    return emitOpError("must be nested directly in a module symbol table");
  SymbolTable symbols(module);
  Operation *graphTarget = symbols.lookup(getGraphAttr().getValue());
  if (!graphTarget || !isa<GraphOp>(graphTarget))
    return emitOpError("graph must resolve to phys.graph");
  llvm::SmallDenseSet<StringRef, 16> initialRoles;
  if (failed(verifyMappingRecords(*this, getInitial(), "initial mapping",
                                  initialRoles, symbols)))
    return failure();
  if (getInitial() == getFinal())
    return success();
  llvm::SmallDenseSet<StringRef, 16> finalRoles;
  if (failed(verifyMappingRecords(*this, getFinal(), "final mapping",
                                  finalRoles, symbols)))
    return failure();
  if (initialRoles != finalRoles)
    return emitOpError(
        "initial and final mappings must contain the same carrier roles");
  return success();
}

static FailureOr<
    std::pair<StringRef, SmallVector<std::pair<StringRef, int64_t>, 4>>>
parseQualifiedRecord(StringRef value) {
  constexpr StringLiteral marker = ".__qlx_repeat[";
  size_t first = value.find(marker);
  StringRef base = first == StringRef::npos ? value : value.take_front(first);
  StringRef remaining =
      first == StringRef::npos ? StringRef{} : value.drop_front(first);
  SmallVector<std::pair<StringRef, int64_t>, 4> qualifiers;
  while (!remaining.empty()) {
    if (!remaining.consume_front(marker))
      return failure();
    size_t eventEnd = remaining.find("][");
    if (eventEnd == StringRef::npos)
      return failure();
    StringRef event = remaining.take_front(eventEnd);
    remaining = remaining.drop_front(eventEnd + 2);
    size_t iterationEnd = remaining.find(']');
    if (event.empty() || iterationEnd == StringRef::npos)
      return failure();
    int64_t iteration = -1;
    if (remaining.take_front(iterationEnd).getAsInteger(10, iteration) ||
        iteration < 0)
      return failure();
    qualifiers.emplace_back(event, iteration);
    remaining = remaining.drop_front(iterationEnd + 1);
  }
  return std::make_pair(base, std::move(qualifiers));
}

static bool qualifiedRecordMatchesProducer(
    Operation *producer, ArrayRef<std::pair<StringRef, int64_t>> qualified,
    Operation *exclusiveBoundary = nullptr) {
  SmallVector<std::pair<StringRef, int64_t>, 4> expected;
  for (Operation *ancestor = producer->getParentOp();
       ancestor && ancestor != exclusiveBoundary;
       ancestor = ancestor->getParentOp()) {
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(ancestor)) {
      auto event = repeat.getEventIdAttr();
      if (!event || event.empty() || repeat.getCount() <= 0)
        return false;
      expected.emplace_back(event.getValue(), repeat.getCount());
    }
  }
  if (qualified.size() != expected.size())
    return false;
  for (auto [actual, context] : llvm::zip(qualified, expected))
    if (actual.first != context.first || actual.second < 0 ||
        actual.second >= context.second)
      return false;
  return true;
}

LogicalResult RecordProjectionOp::verify() {
  const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
  const auto started = PhysVerifyClock::now();
  if (profile)
    llvm::errs() << "phys-verify: record-projection-start\n";
  struct Entry {
    StringAttr instance;
    StringAttr source;
    StringAttr physical;
    ArrayAttr repeatEvents;
    DenseI64ArrayAttr repeatCounts;
  };
  llvm::StringSet<> uniquePairs;
  SmallVector<Entry> entries;
  for (Attribute raw : getEntries()) {
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto instance = entry ? entry.getAs<StringAttr>("instance") : StringAttr{};
    auto source =
        entry ? entry.getAs<StringAttr>("source_record") : StringAttr{};
    auto physical =
        entry ? entry.getAs<StringAttr>("physical_record") : StringAttr{};
    auto repeatEvents =
        entry ? entry.getAs<ArrayAttr>("repeat_events") : ArrayAttr{};
    auto repeatCounts = entry ? entry.getAs<DenseI64ArrayAttr>("repeat_counts")
                              : DenseI64ArrayAttr{};
    unsigned expectedFields = repeatEvents || repeatCounts ? 5 : 3;
    if (!entry || entry.size() != expectedFields || !instance ||
        instance.empty() || !source || source.empty() || !physical ||
        physical.empty() ||
        static_cast<bool>(repeatEvents) != static_cast<bool>(repeatCounts) ||
        (repeatEvents && repeatEvents.size() != repeatCounts.size()))
      return emitOpError(
          "entries require nonempty instance, source_record, and "
          "physical_record strings plus an optional aligned repeat_events/"
          "repeat_counts pair");
    if (repeatEvents)
      for (auto [rawEvent, count] :
           llvm::zip(repeatEvents, repeatCounts.asArrayRef())) {
        auto event = dyn_cast<StringAttr>(rawEvent);
        if (!event || event.empty() || count <= 0)
          return emitOpError(
              "repeat_events/counts require nonempty event strings and "
              "positive counts");
      }
    std::string key = (Twine(instance.getValue()) + "\n" + source.getValue() +
                       "\n" + physical.getValue())
                          .str();
    if (!uniquePairs.insert(key).second)
      return emitOpError(
                 "contains duplicate instance/source-to-physical record pair '")
             << instance.getValue() << "' / '" << source.getValue() << "' -> '"
             << physical.getValue() << "'";
    entries.push_back({instance, source, physical, repeatEvents, repeatCounts});
  }
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: record-entries-done\n";

  SmallVector<bool> claimed(entries.size(), false);
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (module) {
    for (Operation &candidate : module.getBody()->getOperations()) {
      StringRef name = candidate.getName().getStringRef();
      if (name != "phys.selection_sidecar")
        continue;
      auto projection =
          candidate.getAttrOfType<FlatSymbolRefAttr>("record_projection");
      if (!projection || projection.getValue() != getSymName())
        continue;
      auto instance = candidate.getAttrOfType<StringAttr>("source_instance");
      auto sourceRecords = candidate.getAttrOfType<ArrayAttr>("source_records");
      auto physicalRecords = candidate.getAttrOfType<ArrayAttr>("records");
      auto indices =
          candidate.getAttrOfType<DenseI64ArrayAttr>("projection_indices");
      if (!instance || !sourceRecords || !physicalRecords || !indices ||
          sourceRecords.size() != physicalRecords.size() ||
          sourceRecords.size() != indices.size())
        continue;
      for (auto [lane, rawIndex] : llvm::enumerate(indices.asArrayRef())) {
        if (rawIndex < 0 || static_cast<uint64_t>(rawIndex) >= entries.size())
          continue;
        auto source = dyn_cast<StringAttr>(sourceRecords[lane]);
        auto physical = dyn_cast<StringAttr>(physicalRecords[lane]);
        const Entry &entry = entries[rawIndex];
        if (source && physical && entry.instance == instance &&
            entry.source == source && entry.physical == physical)
          claimed[rawIndex] = true;
      }
    }
  }
  for (auto [index, isClaimed] : llvm::enumerate(claimed))
    if (!isClaimed)
      return emitOpError("entry ")
             << index << " is not authenticated by any physical sidecar lane";
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: record-claims-done\n";

  Operation *source =
      SymbolTable::lookupNearestSymbolFrom(*this, getSourceProtocolAttr());
  if (source && source->getName().getStringRef() != "fabric.gadget" &&
      source->getName().getStringRef() != "fabric.protocol")
    return emitOpError(
        "source_protocol must resolve to fabric.gadget or fabric.protocol");

  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getGraphAttr());
  if (!target)
    return success(); // Partial linked modules close the graph at link time.
  auto graph = dyn_cast<GraphOp>(target);
  if (!graph)
    return emitOpError("graph must resolve to phys.graph");
  if (!graph.getSourceProtocolAttr() ||
      graph.getSourceProtocolAttr() != getSourceProtocolAttr())
    return emitOpError(
        "source_protocol must equal the referenced graph source_protocol");

  llvm::StringMap<SmallVector<Operation *, 1>> produced;
  graph.walk([&](MeasureOp measurement) {
    produced[measurement.getRecordId()].push_back(measurement.getOperation());
  });
  graph.walk([&](MeasureProductOp measurement) {
    produced[measurement.getRecordId()].push_back(measurement.getOperation());
  });
  graph.walk([&](CallTemplateOp invocation) {
    auto aliases = invocation->getAttrOfType<ArrayAttr>("record_aliases");
    if (!aliases)
      return;
    for (Attribute raw : aliases) {
      auto alias = dyn_cast<DictionaryAttr>(raw);
      auto name = alias ? alias.getAs<StringAttr>("alias") : StringAttr{};
      if (name)
        produced[name.getValue()].push_back(invocation.getOperation());
    }
  });
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: record-producers-done\n";

  llvm::StringMap<CallOp> calls;
  llvm::StringMap<CallTemplateOp> directTemplates;
  llvm::StringSet<> duplicateCallInstances;
  graph.walk([&](CallOp call) {
    if (!calls.try_emplace(call.getInstance(), call).second)
      duplicateCallInstances.insert(call.getInstance());
  });
  graph.walk([&](CallTemplateOp invocation) {
    if (calls.contains(invocation.getInstance()) ||
        !directTemplates.try_emplace(invocation.getInstance(), invocation)
             .second)
      duplicateCallInstances.insert(invocation.getInstance());
  });
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: record-calls-done\n";
  StringRef rootInstance = getSourceProtocolAttr().getValue();
  if (calls.contains(rootInstance) || directTemplates.contains(rootInstance))
    return emitOpError("phys.call instance '")
           << rootInstance << "' collides with the graph source_protocol";
  auto repeatContextMatches =
      [&](const Entry &entry, Operation *producer,
          ArrayRef<std::pair<StringRef, int64_t>> qualified) -> bool {
    SmallVector<std::pair<StringRef, int64_t>, 4> expected;
    for (auto repeat = producer->getParentOfType<qlx::cflow::RepeatOp>();
         repeat; repeat = repeat->getParentOfType<qlx::cflow::RepeatOp>()) {
      auto event = repeat.getEventIdAttr();
      if (!event || event.empty() || repeat.getCount() <= 0)
        return false;
      expected.emplace_back(event.getValue(), repeat.getCount());
    }
    if (qualified.size() > expected.size())
      return false;
    for (auto [index, context] : llvm::enumerate(qualified))
      if (context.first != expected[index].first || context.second < 0 ||
          context.second >= expected[index].second)
        return false;
    size_t actualSize = entry.repeatEvents ? entry.repeatEvents.size() : 0;
    if (actualSize != expected.size() - qualified.size())
      return false;
    for (size_t index = 0; index < actualSize; ++index) {
      auto context = expected[index + qualified.size()];
      auto event = dyn_cast<StringAttr>(entry.repeatEvents[index]);
      if (!event || event.getValue() != context.first ||
          entry.repeatCounts.asArrayRef()[index] != context.second)
        return false;
    }
    return true;
  };
  for (const Entry &entry : entries) {
    auto qualified = parseQualifiedRecord(entry.physical.getValue());
    if (failed(qualified))
      return emitOpError("physical_record '")
             << entry.physical.getValue()
             << "' has malformed folded-repeat qualification";
    auto producedIt = produced.find(qualified->first);
    size_t count = producedIt == produced.end() ? 0 : producedIt->second.size();
    if (count != 1)
      return emitOpError("physical_record '")
             << entry.physical.getValue() << "' for source_record '"
             << entry.source.getValue()
             << "' must resolve to exactly one measurement in graph @"
             << graph.getSymName() << "; found " << count;
    if (!repeatContextMatches(entry, producedIt->second.front(),
                              qualified->second))
      return emitOpError("physical_record '")
             << entry.physical.getValue()
             << "' repeat_events/counts must exactly match its enclosing "
                "cflow.repeat hierarchy";
    CallOp enclosingCall =
        producedIt->second.front()->getParentOfType<CallOp>();
    if (entry.instance.getValue() == rootInstance) {
      if (enclosingCall)
        return emitOpError("root-instance entry for physical_record '")
               << entry.physical.getValue()
               << "' is nested in phys.call instance '"
               << enclosingCall.getInstance() << "'";
      continue;
    }
    auto call = calls.find(entry.instance.getValue());
    auto directTemplate = directTemplates.find(entry.instance.getValue());
    if ((call == calls.end() && directTemplate == directTemplates.end()) ||
        duplicateCallInstances.contains(entry.instance.getValue()))
      return emitOpError("entry instance '")
             << entry.instance.getValue()
             << "' must resolve to exactly one phys.call in graph @"
             << graph.getSymName();
    if (directTemplate != directTemplates.end()) {
      if (producedIt->second.front() != directTemplate->second.getOperation())
        return emitOpError("physical_record '")
               << entry.physical.getValue()
               << "' is not produced by direct template instance '"
               << entry.instance.getValue() << "'";
      continue;
    }
    bool nestedInClaimedCall = false;
    for (Operation *ancestor = producedIt->second.front()->getParentOp();
         ancestor; ancestor = ancestor->getParentOp()) {
      if (ancestor == call->second.getOperation()) {
        nestedInClaimedCall = true;
        break;
      }
    }
    if (!nestedInClaimedCall)
      return emitOpError("physical_record '")
             << entry.physical.getValue() << "' is not produced inside entry "
             << "instance '" << entry.instance.getValue() << "'";
  }
  if (profile)
    llvm::errs() << "phys-verify: record-projection-done "
                 << physVerifySecondsSince(started) << "s\n";
  return success();
}

static Operation *directGraphChild(Operation *operation, GraphOp graph) {
  Operation *graphOperation = graph.getOperation();
  while (operation && operation->getParentOp() != graphOperation)
    operation = operation->getParentOp();
  return operation;
}

LogicalResult AcquireOp::verify() {
  if (getResources().empty())
    return emitOpError("requires at least one physical resource");
  if (getResources().size() != getStates().size())
    return emitOpError(
        "resource count must match the number of acquired state results");
  llvm::SmallDenseSet<Attribute, 16> unique;
  for (auto [rawResource, stateValue] :
       llvm::zip(getResources(), getStates())) {
    auto resource = dyn_cast<FlatSymbolRefAttr>(rawResource);
    auto state = dyn_cast<StateType>(stateValue.getType());
    if (!resource || !state || state.getResource() != resource)
      return emitOpError(
          "each acquired state result must be positionally qualified by its "
          "matching resource");
    if (!unique.insert(resource).second)
      return emitOpError("resources must be unique");
  }
  return success();
}

LogicalResult ReleaseOp::verify() {
  if (getStates().empty())
    return emitOpError("requires at least one physical state");
  llvm::SmallDenseSet<Attribute, 16> unique;
  for (Value stateValue : getStates()) {
    auto state = dyn_cast<StateType>(stateValue.getType());
    if (!state || !unique.insert(state.getResource()).second)
      return emitOpError("released physical-state resources must be unique");
  }
  return success();
}

static bool hasSSAPath(Operation *producer, Operation *consumer) {
  SmallVector<Operation *, 16> worklist;
  llvm::SmallPtrSet<Operation *, 32> visited;
  for (Value result : producer->getResults())
    llvm::append_range(worklist, result.getUsers());
  while (!worklist.empty()) {
    Operation *operation = worklist.pop_back_val();
    if (operation == consumer || operation->isProperAncestor(consumer))
      return true;
    if (!visited.insert(operation).second)
      continue;
    for (Value result : operation->getResults())
      llvm::append_range(worklist, result.getUsers());
  }
  return false;
}

static bool causallyPrecedes(Operation *producer, Operation *consumer,
                             GraphOp graph) {
  Operation *producerAnchor = directGraphChild(producer, graph);
  Operation *consumerAnchor = directGraphChild(consumer, graph);
  if (!producerAnchor || !consumerAnchor)
    return false;
  if (producerAnchor == consumerAnchor)
    return hasSSAPath(producer, consumer);
  return hasSSAPath(producerAnchor, consumerAnchor);
}

static Operation *unsupportedLifetimeControl(Operation *operation,
                                             GraphOp graph) {
  for (Operation *parent = operation->getParentOp();
       parent && parent != graph.getOperation(); parent = parent->getParentOp())
    if (parent->getNumRegions() != 0 &&
        !isa<CallOp, qlx::cflow::IfOp, qlx::event::TryTakeOp,
             qlx::cflow::RepeatOp, qlx::cflow::WhileOp>(parent))
      return parent;
  return nullptr;
}

/// Indexed physical-state provenance for one graph verification.
///
/// A release may be separated from its acquire by a large graph of ordinary
/// state transformations and structured region boundaries.  Allocation
/// mappings often qualify thousands of states at once, so answering every
/// release/acquire pair by repeatedly scanning wide call boundaries is
/// quadratic in their arity.  This helper indexes wide operations once while
/// keeping each short backward traversal local.  The chains are independent
/// across physical resources, so globally retaining every visited SSA value
/// would add memory without reuse.  The resulting predicate is the same
/// resource-qualified reachability proof as the former traversal.
class PhysicalStateLineage {
public:
  bool descendsFrom(Value value, Value ancestor) {
    ++queries;
    auto ancestorState = dyn_cast<StateType>(ancestor.getType());
    auto valueState = dyn_cast<StateType>(value.getType());
    if (!ancestorState || !valueState ||
        ancestorState.getResource() != valueState.getResource())
      return false;

    llvm::SmallDenseSet<Value, 64> visited;
    SmallVector<Value, 64> pending{value};
    while (!pending.empty()) {
      Value current = pending.pop_back_val();
      if (current == ancestor)
        return true;
      if (!visited.insert(current).second)
        continue;
      ++lineageValues;
      llvm::append_range(pending, predecessors(current));
    }
    return false;
  }

  void printProfile(raw_ostream &os) const {
    os << "phys-verify: allocation-lineage-queries=" << queries
       << " values=" << lineageValues
       << " operand-candidates=" << operandCandidates
       << " indexed-operations=" << operandIndices.size()
       << " operand-index-hits=" << operandIndexHits << "\n";
  }

private:
  using OperandIndex = llvm::DenseMap<Attribute, SmallVector<Value, 1>>;

  void appendMatchingOperands(Operation *operation, Attribute resource,
                              SmallVectorImpl<Value> &values) {
    constexpr unsigned indexThreshold = 8;
    if (operation->getNumOperands() <= indexThreshold) {
      for (Value candidate : operation->getOperands()) {
        ++operandCandidates;
        if (auto state = dyn_cast<StateType>(candidate.getType());
            state && state.getResource() == resource)
          values.push_back(candidate);
      }
      return;
    }

    auto [it, inserted] = operandIndices.try_emplace(operation);
    if (!inserted) {
      ++operandIndexHits;
    } else {
      for (Value candidate : operation->getOperands()) {
        ++operandCandidates;
        if (auto state = dyn_cast<StateType>(candidate.getType()))
          it->second[state.getResource()].push_back(candidate);
      }
    }
    auto found = it->second.find(resource);
    if (found != it->second.end())
      llvm::append_range(values, found->second);
  }

  SmallVector<Value, 4> predecessors(Value value) {
    SmallVector<Value, 4> result;
    auto state = dyn_cast<StateType>(value.getType());
    if (!state)
      return result;
    Attribute resource = state.getResource();

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      Region *region = argument.getOwner()->getParent();
      Operation *parent = region ? region->getParentOp() : nullptr;
      if (parent)
        appendMatchingOperands(parent, resource, result);
      return result;
    }

    auto operationResult = cast<OpResult>(value);
    Operation *owner = operationResult.getOwner();
    if (isa<AcquireOp>(owner))
      return result;
    appendMatchingOperands(owner, resource, result);
    unsigned resultNumber = operationResult.getResultNumber();
    for (Region &region : owner->getRegions())
      for (Block &block : region)
        if (Operation *terminator = block.getTerminator();
            resultNumber < terminator->getNumOperands()) {
          Value yielded = terminator->getOperand(resultNumber);
          if (auto yieldedState = dyn_cast<StateType>(yielded.getType());
              yieldedState && yieldedState.getResource() == resource)
            result.push_back(yielded);
        }
    return result;
  }

  llvm::DenseMap<Operation *, OperandIndex> operandIndices;
  uint64_t queries = 0;
  uint64_t lineageValues = 0;
  uint64_t operandCandidates = 0;
  uint64_t operandIndexHits = 0;
};

LogicalResult AllocationMappingOp::verify() {
  const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
  const auto started = PhysVerifyClock::now();
  if (profile)
    llvm::errs() << "phys-verify: allocation-mapping-start\n";
  Operation *graphTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getGraphAttr());
  auto graph = dyn_cast_or_null<GraphOp>(graphTarget);
  if (!graph)
    return emitOpError("graph must resolve to phys.graph");

  auto module = (*this)->getParentOfType<ModuleOp>();
  llvm::DenseMap<Attribute, ResourceOp> resourceSymbols;
  for (ResourceOp resource : module.getOps<ResourceOp>())
    resourceSymbols.try_emplace(
        FlatSymbolRefAttr::get(getContext(), resource.getSymName()), resource);
  unsigned graphAllocationMappings = 0;
  module.walk([&](AllocationMappingOp mapping) {
    Operation *candidateGraph =
        SymbolTable::lookupNearestSymbolFrom(mapping, mapping.getGraphAttr());
    if (candidateGraph == graph.getOperation())
      ++graphAllocationMappings;
  });
  if (graphAllocationMappings != 1)
    return emitOpError("graph must have exactly one phys.allocation_mapping");

  Operation *architectureTarget =
      SymbolTable::lookupNearestSymbolFrom(graph, graph.getArchitectureAttr());
  auto architecture = dyn_cast_or_null<ArchitectureOp>(architectureTarget);
  if (!architecture)
    return emitOpError("graph architecture must resolve to phys.machine");
  SymbolTable architectureSymbols(architecture);

  llvm::StringMap<Operation *> events;
  llvm::DenseMap<Operation *, unsigned> eventOrder;
  llvm::SmallDenseSet<Operation *, 16> acquiredResourceDeclarations;
  llvm::StringSet<> graphAcquireEvents;
  unsigned nextEvent = 0;
  bool duplicateEvent = false;
  bool malformedAcquiredResource = false;
  graph.walk([&](Operation *operation) {
    auto event = operation->getAttrOfType<StringAttr>("event_id");
    if (event && !event.getValue().empty()) {
      if (!events.try_emplace(event.getValue(), operation).second)
        duplicateEvent = true;
      eventOrder.try_emplace(operation, nextEvent++);
    }
    auto acquire = dyn_cast<AcquireOp>(operation);
    if (!acquire)
      return;
    if (!event || event.getValue().empty()) {
      malformedAcquiredResource = true;
      return;
    }
    graphAcquireEvents.insert(event.getValue());
    for (Attribute raw : acquire.getResources()) {
      auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
      auto found =
          reference ? resourceSymbols.find(reference) : resourceSymbols.end();
      ResourceOp resource =
          found == resourceSymbols.end() ? ResourceOp{} : found->second;
      if (!resource) {
        malformedAcquiredResource = true;
        continue;
      }
      acquiredResourceDeclarations.insert(resource.getOperation());
    }
  });
  if (duplicateEvent)
    return emitOpError("cannot verify allocation lifetimes when graph event "
                       "IDs are not unique");
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: allocation-events-done\n";
  if (malformedAcquiredResource)
    return emitOpError("graph acquire resources must resolve to phys.resource");
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: allocation-acquires-done\n";

  llvm::SmallDenseSet<StringRef, 16> allocations;
  llvm::SmallDenseSet<Operation *, 16> resourceDeclarations;
  llvm::SmallDenseSet<StringRef, 16> acquireEvents;
  struct AllocationLifetime {
    StringAttr allocation;
    Operation *acquireOp;
    Operation *releaseOp;
    unsigned acquire;
    std::optional<unsigned> release;
    StringAttr releaseEvent;
    ArrayAttr after;
  };
  llvm::DenseMap<std::pair<Operation *, int64_t>,
                 SmallVector<AllocationLifetime, 2>>
      physicalLifetimes;
  llvm::DenseMap<std::pair<Operation *, Operation *>, bool> causalOrder;
  auto causallyOrdered = [&](Operation *producer, Operation *consumer) {
    auto key = std::make_pair(producer, consumer);
    auto found = causalOrder.find(key);
    if (found != causalOrder.end())
      return found->second;
    bool result = causallyPrecedes(producer, consumer, graph);
    causalOrder.try_emplace(key, result);
    return result;
  };
  llvm::StringMap<SmallVector<std::pair<Operation *, int64_t>, 4>>
      releasedPhysicalIdentities;
  struct ExplicitResourceOrder {
    StringAttr allocation;
    StringAttr release;
    Operation *acquireOp;
    SmallVector<std::pair<Operation *, int64_t>, 4> physicalIdentities;
  };
  SmallVector<ExplicitResourceOrder, 8> explicitResourceOrders;
  PhysicalStateLineage physicalStateLineage;
  int64_t verifiedAllocationEntries = 0;
  for (Attribute value : getEntries()) {
    ++verifiedAllocationEntries;
    auto record = dyn_cast<DictionaryAttr>(value);
    auto allocation = record ? record.getAs<StringAttr>("allocation") : nullptr;
    auto resourceClass =
        record ? record.getAs<FlatSymbolRefAttr>("resource_class") : nullptr;
    auto resources = record ? record.getAs<ArrayAttr>("resources") : nullptr;
    auto indices =
        record ? record.getAs<DenseI64ArrayAttr>("indices") : nullptr;
    auto acquire = record ? record.getAs<StringAttr>("acquire") : nullptr;
    auto release = record ? record.getAs<StringAttr>("release") : nullptr;
    auto after = record ? record.getAs<ArrayAttr>("after") : nullptr;
    auto qecRegion = record ? record.getAs<StringAttr>("qec_region") : nullptr;
    auto physicalBinding =
        record ? record.getAs<StringAttr>("physical_binding") : nullptr;
    if (!record || !allocation || allocation.getValue().empty() ||
        !resourceClass || !resources || resources.empty() || !indices ||
        indices.empty() || resources.size() != indices.size() || !acquire ||
        acquire.getValue().empty())
      return emitOpError(
          "entries require a nonempty allocation, resource_class, equally "
          "sized resources/indices, and acquire event");
    if (static_cast<bool>(qecRegion) != static_cast<bool>(physicalBinding))
      return emitOpError(
          "qec_region and physical_binding must be present together");
    if (!allocations.insert(allocation.getValue()).second)
      return emitOpError("contains duplicate allocation '")
             << allocation.getValue() << "'";
    if (!acquireEvents.insert(acquire.getValue()).second)
      return emitOpError("acquire event '")
             << acquire.getValue() << "' belongs to multiple allocations";
    if (after) {
      if (after.empty())
        return emitOpError(
            "after must contain at least one release-event string");
      llvm::SmallDenseSet<StringRef, 4> uniquePredecessors;
      for (Attribute rawPredecessor : after) {
        auto predecessor = dyn_cast<StringAttr>(rawPredecessor);
        if (!predecessor || predecessor.getValue().empty())
          return emitOpError(
              "after entries must be nonempty release-event strings");
        if (!uniquePredecessors.insert(predecessor.getValue()).second)
          return emitOpError("after contains duplicate release event '")
                 << predecessor.getValue() << "'";
      }
    }

    Operation *classTarget =
        architectureSymbols.lookup(resourceClass.getValue());
    auto resourceClassOp = dyn_cast_or_null<ResourceClassOp>(classTarget);
    if (!resourceClassOp)
      return emitOpError("resource_class ")
             << resourceClass << " must resolve in the graph architecture";
    QECBindingOp binding;
    if (physicalBinding) {
      binding = dyn_cast_or_null<QECBindingOp>(
          architectureSymbols.lookup(physicalBinding.getValue()));
      if (!binding)
        return emitOpError("physical_binding @")
               << physicalBinding.getValue()
               << " must resolve in the graph architecture";
      if (binding.getQecRegionAttr().getLeafReference().getValue() !=
          qecRegion.getValue())
        return emitOpError("qec_region does not match physical_binding @")
               << physicalBinding.getValue();
      bool ownsResourceClass =
          llvm::any_of(binding.getResources(), [&](Attribute value) {
            auto reference = dyn_cast<FlatSymbolRefAttr>(value);
            return reference && reference == resourceClass;
          });
      if (!ownsResourceClass)
        return emitOpError("resource_class ")
               << resourceClass << " does not belong to physical_binding @"
               << physicalBinding.getValue();
    }

    SmallVector<std::pair<Operation *, int64_t>, 16> physicalIdentities;
    for (auto [ordinal, resourceValue] : llvm::enumerate(resources)) {
      auto reference = dyn_cast<FlatSymbolRefAttr>(resourceValue);
      auto found =
          reference ? resourceSymbols.find(reference) : resourceSymbols.end();
      ResourceOp resource =
          found == resourceSymbols.end() ? ResourceOp{} : found->second;
      if (!resource)
        return emitOpError(
            "allocation resources must resolve to phys.resource");
      resourceDeclarations.insert(resource);
      if (resource.getResourceClassAttr() != resourceClass)
        return emitOpError("resource ")
               << reference << " does not belong to " << resourceClass;
      if (resource.getIndex() != indices[ordinal])
        return emitOpError("resource ")
               << reference << " index does not match allocation binding";
      if (resource.getIndex() < 0 ||
          resource.getIndex() >= resourceClassOp.getCount())
        return emitOpError("resource ")
               << reference << " index is outside resource-class capacity";
      if (qecRegion) {
        auto resourceQEC = resource.getQecRegionAttr();
        if (!resourceQEC ||
            resourceQEC.getLeafReference().getValue() != qecRegion.getValue())
          return emitOpError("resource ")
                 << reference
                 << " QEC region does not match its allocation binding";
      }
      auto identity = std::make_pair(resourceClassOp.getOperation(),
                                     static_cast<int64_t>(resource.getIndex()));
      if (llvm::is_contained(physicalIdentities, identity))
        return emitOpError("physical resource identity @")
               << resourceClassOp.getSymName() << "[" << resource.getIndex()
               << "] appears more than once in allocation '"
               << allocation.getValue() << "'";
      physicalIdentities.push_back(identity);
    }

    auto acquireIt = events.find(acquire.getValue());
    if (acquireIt == events.end() || !isa<AcquireOp>(acquireIt->second))
      return emitOpError("acquire event '")
             << acquire.getValue() << "' must resolve to phys.acquire";
    auto acquireOp = cast<AcquireOp>(acquireIt->second);
    if (acquireOp.getResources() != resources)
      return emitOpError("acquire event '")
             << acquire.getValue()
             << "' resources do not match the allocation binding";
    if (acquireOp.getStates().size() != resources.size())
      return emitOpError("acquire event '")
             << acquire.getValue()
             << "' state count does not match the allocation binding";
    if (Operation *control =
            unsupportedLifetimeControl(acquireOp.getOperation(), graph))
      return emitOpError("acquire event '")
             << acquire.getValue()
             << "' is nested in unsupported structured control "
             << control->getName();
    unsigned acquireOrder = eventOrder.lookup(acquireOp.getOperation());
    std::optional<unsigned> releaseOrder;
    if (release) {
      if (release.getValue().empty() || release == acquire)
        return emitOpError(
            "release event must be nonempty and differ from acquire");
      auto releaseIt = events.find(release.getValue());
      if (releaseIt == events.end() ||
          !isa<ReleaseOp, PackResourceOp, MeasureOp, CallTemplateOp>(
              releaseIt->second))
        return emitOpError("release event '")
               << release.getValue()
               << "' must resolve to phys.release, phys.pack_resource, a "
                  "destructive phys.measure, or a consuming "
                  "phys.call_template";
      Operation *releaseOp = releaseIt->second;
      SmallVector<Value> releasedValues;
      if (auto explicitRelease = dyn_cast<ReleaseOp>(releaseOp))
        llvm::append_range(releasedValues, explicitRelease.getStates());
      else if (auto pack = dyn_cast<PackResourceOp>(releaseOp))
        llvm::append_range(releasedValues, pack.getInputs());
      else if (auto measurement = dyn_cast<MeasureOp>(releaseOp)) {
        if (!measurement.getDestructiveAttr() || measurement.getOutput())
          return emitOpError("release event '")
                 << release.getValue()
                 << "' phys.measure must be destructive and return no state";
        releasedValues.push_back(measurement.getInput());
      } else {
        auto invocation = cast<CallTemplateOp>(releaseOp);
        llvm::SmallDenseSet<Attribute, 4> returnedResources;
        for (Value output : invocation.getOutputs())
          if (auto state = dyn_cast<StateType>(output.getType()))
            returnedResources.insert(state.getResource());
        for (Value input : invocation.getInputs())
          if (auto state = dyn_cast<StateType>(input.getType());
              state && !returnedResources.contains(state.getResource()))
            releasedValues.push_back(input);
      }
      releaseOrder = eventOrder.lookup(releaseOp);
      if (Operation *control = unsupportedLifetimeControl(releaseOp, graph))
        return emitOpError("release event '")
               << release.getValue()
               << "' is nested in unsupported structured control "
               << control->getName()
               << "; allocation closure must hold on every execution path";
      if (!causallyOrdered(acquireOp.getOperation(), releaseOp))
        return emitOpError("release event '")
               << release.getValue()
               << "' must be causally after acquire event '"
               << acquire.getValue() << "'";
      if (isa<ReleaseOp>(releaseOp) &&
          releasedValues.size() != resources.size())
        return emitOpError("release event '")
               << release.getValue()
               << "' state count does not match the allocation binding";
      llvm::DenseMap<Attribute, Value> releasedStates;
      for (Value stateValue : releasedValues) {
        auto state = dyn_cast<StateType>(stateValue.getType());
        if (!state ||
            !releasedStates.try_emplace(state.getResource(), stateValue).second)
          return emitOpError("release event '")
                 << release.getValue()
                 << "' resources do not match the allocation binding";
      }
      for (Attribute resourceValue : resources) {
        auto released = releasedStates.find(resourceValue);
        if (released == releasedStates.end())
          return emitOpError("release event '")
                 << release.getValue()
                 << "' resources do not match the allocation binding";
        auto resourcePosition = llvm::find(resources, resourceValue);
        unsigned ordinal = static_cast<unsigned>(
            std::distance(resources.begin(), resourcePosition));
        if (!physicalStateLineage.descendsFrom(released->second,
                                               acquireOp.getStates()[ordinal]))
          return emitOpError("release event '")
                 << release.getValue() << "' state for " << resourceValue
                 << " is not derived from acquire event '" << acquire.getValue()
                 << "'";
      }
      releasedPhysicalIdentities[release.getValue()].append(
          physicalIdentities.begin(), physicalIdentities.end());
    }
    if (after)
      for (Attribute rawPredecessor : after) {
        ExplicitResourceOrder order{allocation,
                                    cast<StringAttr>(rawPredecessor),
                                    acquireOp.getOperation(),
                                    {}};
        order.physicalIdentities.append(physicalIdentities.begin(),
                                        physicalIdentities.end());
        explicitResourceOrders.push_back(std::move(order));
      }
    for (auto identity : physicalIdentities)
      physicalLifetimes[identity].push_back(
          {allocation, acquireOp.getOperation(),
           release ? events.find(release.getValue())->second : nullptr,
           acquireOrder, releaseOrder, release, after});
    if (std::getenv("QLX_PROFILE_P2_TO_P3") &&
        verifiedAllocationEntries % 25 == 0)
      llvm::errs() << "phys-verify: allocation-entry="
                   << verifiedAllocationEntries << "\n";
  }
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: allocation-entries-done\n";
  if (profile)
    physicalStateLineage.printProfile(llvm::errs());

  if (resourceDeclarations.size() != acquiredResourceDeclarations.size() ||
      !llvm::all_of(acquiredResourceDeclarations,
                    [&](Operation *resource) {
                      return resourceDeclarations.contains(resource);
                    }) ||
      acquireEvents.size() != graphAcquireEvents.size() ||
      !llvm::all_of(graphAcquireEvents, [&](const auto &event) {
        return acquireEvents.contains(event.getKey());
      }))
    return emitOpError("must cover every graph-acquired phys.resource");

  for (const ExplicitResourceOrder &order : explicitResourceOrders) {
    auto predecessor = events.find(order.release.getValue());
    if (predecessor == events.end() ||
        !isa<ReleaseOp, PackResourceOp, MeasureOp, CallTemplateOp>(
            predecessor->second))
      return emitOpError("after release event '")
             << order.release.getValue()
             << "' must resolve to a mapped physical lifetime terminator";
    auto released = releasedPhysicalIdentities.find(order.release.getValue());
    if (released == releasedPhysicalIdentities.end())
      return emitOpError("after release event '")
             << order.release.getValue() << "' must close a mapped allocation";
    if (!llvm::any_of(order.physicalIdentities, [&](auto identity) {
          return llvm::is_contained(released->second, identity);
        }))
      return emitOpError("allocation '")
             << order.allocation.getValue() << "' after release event '"
             << order.release.getValue()
             << "' does not order a shared physical resource identity";
    if (eventOrder.lookup(predecessor->second) >=
        eventOrder.lookup(order.acquireOp))
      return emitOpError("allocation '")
             << order.allocation.getValue() << "' after release event '"
             << order.release.getValue() << "' must precede its acquire event";
  }
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-verify: allocation-orders-done\n";

  for (auto &[identity, lifetimes] : physicalLifetimes) {
    auto explicitlyOrderedAfter = [](ArrayAttr after, StringAttr release) {
      return after && llvm::is_contained(after, release);
    };
    llvm::sort(lifetimes, [](const AllocationLifetime &left,
                             const AllocationLifetime &right) {
      return left.acquire < right.acquire;
    });
    for (auto adjacent : llvm::zip(ArrayRef(lifetimes).drop_back(),
                                   ArrayRef(lifetimes).drop_front())) {
      const AllocationLifetime &left = std::get<0>(adjacent);
      const AllocationLifetime &right = std::get<1>(adjacent);
      bool leftBeforeRight =
          left.releaseOp &&
          ((left.releaseEvent &&
            explicitlyOrderedAfter(right.after, left.releaseEvent)) ||
           causallyOrdered(left.releaseOp, right.acquireOp));
      if (leftBeforeRight)
        continue;

      auto resourceClass = cast<ResourceClassOp>(identity.first);
      bool textuallySeparated = left.release && *left.release < right.acquire;
      InFlightDiagnostic diagnostic =
          emitOpError("physical resource identity @");
      diagnostic << resourceClass.getSymName() << "[" << identity.second
                 << "] has "
                 << (textuallySeparated
                         ? "causally unordered adjacent allocation lifetimes ('"
                         : "overlapping allocation lifetimes ('")
                 << left.allocation.getValue() << "' and '"
                 << right.allocation.getValue() << "')";
      return diagnostic;
    }
  }
  if (profile)
    llvm::errs() << "phys-verify: allocation-mapping-done "
                 << physVerifySecondsSince(started) << "s\n";
  return success();
}

static bool equivalentLogicalChannels(Operation *canonical,
                                      Operation *selected) {
  if (canonical->getAttr("from") != selected->getAttr("from") ||
      canonical->getAttr("to") != selected->getAttr("to") ||
      canonical->getAttr("capacity") != selected->getAttr("capacity"))
    return false;

  auto channelDirection = [](Operation *channel) {
    auto direction = channel->getAttrOfType<StringAttr>("direction");
    return direction ? direction.getValue() : StringRef("forward");
  };
  if (channelDirection(canonical) != channelDirection(selected))
    return false;

  auto canonicalCapabilities =
      canonical->getAttrOfType<ArrayAttr>("capabilities");
  auto selectedCapabilities =
      selected->getAttrOfType<ArrayAttr>("capabilities");
  if (!canonicalCapabilities || !selectedCapabilities)
    return canonicalCapabilities == selectedCapabilities;
  if (canonicalCapabilities.size() != selectedCapabilities.size())
    return false;
  llvm::SmallDenseSet<Attribute, 8> canonicalSet(canonicalCapabilities.begin(),
                                                 canonicalCapabilities.end());
  llvm::SmallDenseSet<Attribute, 8> selectedSet(selectedCapabilities.begin(),
                                                selectedCapabilities.end());
  if (canonicalSet.size() != canonicalCapabilities.size() ||
      selectedSet.size() != selectedCapabilities.size() ||
      canonicalSet.size() != selectedSet.size())
    return false;
  return llvm::all_of(selectedSet, [&](Attribute channelCapability) {
    return canonicalSet.contains(channelCapability);
  });
}

LogicalResult RoutingOp::verify() {
  Operation *graphTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getGraphAttr());
  auto graph = dyn_cast_or_null<GraphOp>(graphTarget);
  if (!graph)
    return emitOpError("graph must resolve to phys.graph");
  Operation *architectureTarget =
      SymbolTable::lookupNearestSymbolFrom(graph, graph.getArchitectureAttr());
  auto architecture = dyn_cast_or_null<ArchitectureOp>(architectureTarget);
  if (!architecture)
    return emitOpError("graph architecture must resolve to phys.machine");
  SymbolTable architectureSymbols(architecture);
  auto topology = dyn_cast_or_null<TopologyOp>(
      architectureSymbols.lookup(getTopologyAttr().getValue()));
  if (!topology)
    return emitOpError(
        "topology must resolve inside the graph's physical architecture");
  auto module = (*this)->getParentOfType<ModuleOp>();

  llvm::StringMap<Operation *> nativeEvents;
  llvm::DenseMap<Operation *, unsigned> nativeEventOrder;
  unsigned nextNativeEvent = 0;
  graph.walk([&](Operation *operation) {
    if (auto event = operation->getAttrOfType<StringAttr>("event_id")) {
      nativeEvents.try_emplace(event.getValue(), operation);
      nativeEventOrder.try_emplace(operation, nextNativeEvent++);
    }
  });

  // Allocation mappings are the graph-local authority for which QEC/physical
  // binding owns every concrete resource used by a routed bridge.
  llvm::DenseMap<Attribute, StringAttr> resourceBindings;
  bool ambiguousResourceBinding = false;
  module.walk([&](AllocationMappingOp mapping) {
    if (mapping.getGraphAttr().getValue() != graph.getSymName())
      return;
    for (Attribute raw : mapping.getEntries()) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto resources = entry ? entry.getAs<ArrayAttr>("resources") : nullptr;
      auto binding =
          entry ? entry.getAs<StringAttr>("physical_binding") : nullptr;
      if (!resources || !binding)
        continue;
      for (Attribute resource : resources) {
        auto [found, inserted] =
            resourceBindings.try_emplace(resource, binding);
        if (!inserted && found->second != binding)
          ambiguousResourceBinding = true;
      }
    }
  });
  if (ambiguousResourceBinding)
    return emitOpError(
        "graph allocation mappings assign one resource to multiple physical "
        "bindings");

  llvm::SmallDenseSet<StringRef, 16> events;
  llvm::SmallDenseSet<StringRef, 32> usedNativeEvents;
  for (Attribute value : getSteps()) {
    auto step = dyn_cast<DictionaryAttr>(value);
    auto event = step ? step.getAs<StringAttr>("event") : nullptr;
    auto path = step ? step.getAs<DenseI64ArrayAttr>("path") : nullptr;
    auto action = step ? step.getAs<StringAttr>("action") : nullptr;
    auto routedEvents = step ? step.getAs<ArrayAttr>("native_events") : nullptr;
    if (!step || !event || event.getValue().empty() || !path ||
        path.size() < 2 || !action || action.getValue().empty())
      return emitOpError(
          "routing steps require event, action, and a path of at least two "
          "nodes");
    if (!events.insert(event.getValue()).second)
      return emitOpError("contains duplicate routing event '")
             << event.getValue() << "'";
    ArrayRef<int64_t> nodes = path.asArrayRef();
    for (size_t i = 0; i + 2 < nodes.size(); ++i)
      if (!physicalTopologyHasEdge(topology, nodes[i], nodes[i + 1]))
        return emitOpError("routing event '")
               << event.getValue() << "' requires a missing adjacency edge "
               << nodes[i] << " -> " << nodes[i + 1];
    if (!physicalTopologyHasEdge(topology, nodes[nodes.size() - 2],
                                 nodes.back()))
      return emitOpError("routing event '")
             << event.getValue() << "' ends on a missing adjacency edge";

    bool bridge =
        static_cast<bool>(step.getAs<UnitAttr>("communication_bridge"));
    if (bridge && (!routedEvents || routedEvents.empty()))
      return emitOpError("communication bridge '")
             << event.getValue() << "' requires concrete native_events";

    auto channel = step.getAs<SymbolRefAttr>("channel");
    Attribute channelCapability = step.get("channel_capability");
    auto endpoints = step.getAs<ArrayAttr>("endpoints");
    auto actionSite = step.getAs<SymbolRefAttr>("action_site");
    auto generatedBy = step.getAs<FlatSymbolRefAttr>("generated_by");
    auto sourceCall = step.getAs<StringAttr>("source_call");
    unsigned communicationFields =
        static_cast<unsigned>(static_cast<bool>(channel)) +
        static_cast<unsigned>(static_cast<bool>(channelCapability)) +
        static_cast<unsigned>(static_cast<bool>(endpoints)) +
        static_cast<unsigned>(static_cast<bool>(actionSite)) +
        static_cast<unsigned>(static_cast<bool>(generatedBy)) +
        static_cast<unsigned>(static_cast<bool>(sourceCall));
    if (!bridge && communicationFields)
      return emitOpError("routing event '")
             << event.getValue()
             << "' has communication provenance without a bridge marker";

    auto sourceRegion = step.getAs<StringAttr>("source_region");
    auto destinationRegion = step.getAs<StringAttr>("destination_region");
    auto sourceQEC = step.getAs<StringAttr>("source_qec_region");
    auto destinationQEC = step.getAs<StringAttr>("destination_qec_region");
    auto sourceBinding = step.getAs<StringAttr>("source_binding");
    auto destinationBinding = step.getAs<StringAttr>("destination_binding");
    if (bridge && (communicationFields != 6 || !sourceCall ||
                   sourceCall.getValue().empty()))
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' requires channel, channel capability, endpoints, "
                "action_site, "
                "generated_by, and source_call";
    if (bridge && (!sourceRegion || !destinationRegion || !sourceQEC ||
                   !destinationQEC || !sourceBinding || !destinationBinding))
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' requires complete region and physical-binding evidence";

    CallOp physicalCall;
    if (routedEvents) {
      size_t expectedEvents = 2 * (nodes.size() - 2) + 1;
      if (routedEvents.size() != expectedEvents)
        return emitOpError("routing event '")
               << event.getValue() << "' requires " << expectedEvents
               << " concrete native events";
      size_t routedAction = nodes.size() - 2;
      std::optional<unsigned> previousEventOrder;
      for (auto [ordinal, rawEvent] : llvm::enumerate(routedEvents)) {
        auto eventID = dyn_cast<StringAttr>(rawEvent);
        if (!eventID || eventID.getValue().empty())
          return emitOpError(
              "routing native_events must contain nonempty strings");
        if (!usedNativeEvents.insert(eventID.getValue()).second)
          return emitOpError("native event '")
                 << eventID.getValue() << "' belongs to multiple routes";
        auto found = nativeEvents.find(eventID.getValue());
        if (found == nativeEvents.end() || !isa<ApplyOp>(found->second))
          return emitOpError("routing native event '")
                 << eventID.getValue() << "' must resolve to phys.apply";
        auto apply = cast<ApplyOp>(found->second);
        if (bridge) {
          CallOp enclosingCall;
          for (Operation *parent = apply->getParentOp(); parent;
               parent = parent->getParentOp()) {
            auto candidate = dyn_cast<CallOp>(parent);
            if (candidate && candidate->hasAttr("channel")) {
              enclosingCall = candidate;
              break;
            }
          }
          if (!enclosingCall)
            return emitOpError("routing native event '")
                   << eventID.getValue()
                   << "' must be nested in the communication phys.call";
          if (!physicalCall)
            physicalCall = enclosingCall;
          else if (physicalCall != enclosingCall)
            return emitOpError("communication bridge '")
                   << event.getValue()
                   << "' native events span multiple phys.call instances";
        }
        StringRef expectedAction =
            ordinal == routedAction ? action.getValue() : "swap";
        if (apply.getActionAttr().getValue() != expectedAction)
          return emitOpError("routing native event '")
                 << eventID.getValue() << "' must apply @" << expectedAction;
        if (apply.getTopologyAttr() != getTopologyAttr())
          return emitOpError("routing native event '")
                 << eventID.getValue()
                 << "' must use the routing evidence topology";
        unsigned eventOrder = nativeEventOrder.lookup(apply);
        if (previousEventOrder && eventOrder <= *previousEventOrder)
          return emitOpError("routing native event '")
                 << eventID.getValue()
                 << "' is out of physical execution order";
        previousEventOrder = eventOrder;

        if (apply.getInputs().size() != 2)
          return emitOpError("routing native event '")
                 << eventID.getValue()
                 << "' must act on exactly two physical resources";
        int64_t actualNodes[2];
        for (auto [inputOrdinal, input] : llvm::enumerate(apply.getInputs())) {
          auto state = dyn_cast<StateType>(input.getType());
          auto resource = state ? dyn_cast_or_null<ResourceOp>(
                                      SymbolTable::lookupNearestSymbolFrom(
                                          apply, state.getResource()))
                                : nullptr;
          if (!resource)
            return emitOpError("routing native event '")
                   << eventID.getValue()
                   << "' resource must resolve to phys.resource";
          actualNodes[inputOrdinal] = resource.getIndex();
          if (bridge) {
            auto owner = resourceBindings.find(state.getResource());
            if (owner == resourceBindings.end())
              return emitOpError("routing native event '")
                     << eventID.getValue()
                     << "' resource lacks graph-local physical-binding "
                        "allocation evidence";
            StringRef expectedBinding =
                ordinal == routedAction && inputOrdinal == 1
                    ? destinationBinding.getValue()
                    : sourceBinding.getValue();
            if (owner->second.getValue() != expectedBinding)
              return emitOpError("routing native event '")
                     << eventID.getValue()
                     << "' resource binding does not match the ordered "
                        "bridge endpoints";
          }
        }
        size_t edge =
            ordinal <= routedAction ? ordinal : 2 * routedAction - ordinal;
        if (actualNodes[0] != nodes[edge] || actualNodes[1] != nodes[edge + 1])
          return emitOpError("routing native event '")
                 << eventID.getValue()
                 << "' resources do not match routing path edge " << nodes[edge]
                 << " -> " << nodes[edge + 1];
      }
    }

    if (!bridge)
      continue;
    auto sourceProtocolRef = graph.getSourceProtocolAttr();
    Operation *sourceProtocol =
        sourceProtocolRef
            ? SymbolTable::lookupNearestSymbolFrom(graph, sourceProtocolRef)
            : nullptr;
    if (!sourceProtocol ||
        (sourceProtocol->getName().getStringRef() != "fabric.protocol" &&
         sourceProtocol->getName().getStringRef() != "fabric.gadget"))
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' requires a retained graph source_protocol";
    if (sourceRegion == destinationRegion || sourceQEC == destinationQEC ||
        sourceBinding == destinationBinding)
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' must cross distinct logical, QEC, and physical bindings";
    if (!topology.getStrict())
      return emitOpError("communication bridge '")
             << event.getValue() << "' requires a strict physical topology";
    if (endpoints.size() != 2)
      return emitOpError("communication bridge '")
             << event.getValue() << "' requires two endpoints";
    auto firstEndpoint = dyn_cast<SymbolRefAttr>(endpoints[0]);
    auto secondEndpoint = dyn_cast<SymbolRefAttr>(endpoints[1]);
    if (!firstEndpoint || !secondEndpoint ||
        firstEndpoint.getLeafReference().getValue() !=
            sourceRegion.getValue() ||
        secondEndpoint.getLeafReference().getValue() !=
            destinationRegion.getValue())
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' endpoint order must match its source/destination regions";

    auto verifyBinding = [&](StringAttr bindingName, StringAttr qecRegion,
                             StringRef role) -> FailureOr<QECBindingOp> {
      auto binding = dyn_cast_or_null<QECBindingOp>(
          architectureSymbols.lookup(bindingName.getValue()));
      if (!binding) {
        emitOpError("communication bridge '")
            << event.getValue() << "' " << role << " binding @"
            << bindingName.getValue() << " must resolve to phys.qec_binding";
        return failure();
      }
      if (binding.getQecRegionAttr().getLeafReference().getValue() !=
          qecRegion.getValue()) {
        emitOpError("communication bridge '")
            << event.getValue() << "' " << role
            << " QEC region does not match binding @" << bindingName.getValue();
        return failure();
      }
      auto qecReference = binding.getQecRegionAttr();
      Operation *qecMachine =
          SymbolTable::lookupSymbolIn(module, qecReference.getRootReference());
      Operation *region =
          qecMachine ? SymbolTable(qecMachine)
                           .lookup(qecReference.getLeafReference().getValue())
                     : nullptr;
      if (!region || region->getName().getStringRef() != "fabric.region") {
        emitOpError("communication bridge '")
            << event.getValue() << "' " << role
            << " QEC region must resolve to fabric.region";
        return failure();
      }
      if (!binding.getTopologyAttr() ||
          binding.getTopologyAttr() != getTopologyAttr()) {
        emitOpError("communication bridge '")
            << event.getValue() << "' " << role
            << " binding must use the routing evidence topology";
        return failure();
      }
      return binding;
    };
    FailureOr<QECBindingOp> verifiedSource =
        verifyBinding(sourceBinding, sourceQEC, "source");
    FailureOr<QECBindingOp> verifiedDestination =
        verifyBinding(destinationBinding, destinationQEC, "destination");
    if (failed(verifiedSource) || failed(verifiedDestination))
      return failure();

    // Close the retained layered-device refinement for both ordered logical
    // endpoints. This prevents swapping otherwise-valid QEC/binding labels.
    StringAttr selectedDevice;
    if (auto metadata =
            sourceProtocol->getAttrOfType<DictionaryAttr>("metadata"))
      selectedDevice = metadata.getAs<StringAttr>("device");
    Operation *device = nullptr;
    bool ambiguousDevice = false;
    module.walk([&](Operation *candidate) {
      if (candidate->getName().getStringRef() != "qlx.device")
        return;
      auto qec = candidate->getAttrOfType<FlatSymbolRefAttr>("qec");
      auto physical = candidate->getAttrOfType<FlatSymbolRefAttr>("physical");
      auto symbol = candidate->getAttrOfType<StringAttr>("sym_name");
      if (!qec || !physical || !symbol ||
          (selectedDevice && selectedDevice.getValue() != symbol.getValue()) ||
          qec.getValue() != verifiedSource->getQecRegionAttr()
                                .getRootReference()
                                .getValue() ||
          physical.getValue() != graph.getArchitectureAttr().getValue())
        return;
      if (device)
        ambiguousDevice = true;
      device = candidate;
    });
    if (!device || ambiguousDevice)
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' requires one unambiguous retained qlx.device refinement";
    if (firstEndpoint.getRootReference() != secondEndpoint.getRootReference() ||
        verifiedSource->getQecRegionAttr().getRootReference() !=
            verifiedDestination->getQecRegionAttr().getRootReference())
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' endpoints must share the retained logical and QEC "
                "machines";

    auto logicalToQECRef =
        device->getAttrOfType<FlatSymbolRefAttr>("logical_to_qec");
    auto qecToPhysicalRef =
        device->getAttrOfType<FlatSymbolRefAttr>("qec_to_physical");
    Operation *logicalToQEC =
        logicalToQECRef
            ? SymbolTable::lookupNearestSymbolFrom(device, logicalToQECRef)
            : nullptr;
    Operation *qecToPhysical =
        qecToPhysicalRef
            ? SymbolTable::lookupNearestSymbolFrom(device, qecToPhysicalRef)
            : nullptr;
    if (!logicalToQEC ||
        logicalToQEC->getName().getStringRef() != "qlx.logical_to_qec" ||
        !qecToPhysical ||
        qecToPhysical->getName().getStringRef() != "qlx.qec_to_physical")
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' requires retained logical_to_qec and qec_to_physical "
                "refinements";

    auto verifyRefinement = [&](StringAttr logicalName, StringAttr qecName,
                                StringAttr bindingName,
                                StringRef role) -> LogicalResult {
      auto logicalEntries = logicalToQEC->getAttrOfType<ArrayAttr>("entries");
      auto physicalEntries = qecToPhysical->getAttrOfType<ArrayAttr>("entries");
      DictionaryAttr logicalEntry;
      DictionaryAttr physicalEntry;
      for (Attribute raw : logicalEntries) {
        auto entry = dyn_cast<DictionaryAttr>(raw);
        if (entry && entry.getAs<StringAttr>("logical") == logicalName)
          logicalEntry = entry;
      }
      for (Attribute raw : physicalEntries) {
        auto entry = dyn_cast<DictionaryAttr>(raw);
        if (entry && entry.getAs<StringAttr>("qec") == qecName)
          physicalEntry = entry;
      }
      if (!logicalEntry || logicalEntry.getAs<StringAttr>("qec") != qecName)
        return emitOpError("communication bridge '")
               << event.getValue() << "' " << role
               << " endpoint does not match retained logical_to_qec";
      if (!physicalEntry ||
          physicalEntry.getAs<StringAttr>("binding") != bindingName)
        return emitOpError("communication bridge '")
               << event.getValue() << "' " << role
               << " QEC region does not match retained qec_to_physical";
      return success();
    };
    if (failed(verifyRefinement(sourceRegion, sourceQEC, sourceBinding,
                                "source")) ||
        failed(verifyRefinement(destinationRegion, destinationQEC,
                                destinationBinding, "destination")))
      return failure();

    Operation *channelTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, channel);
    Operation *siteTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, actionSite);
    Operation *generatorTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, generatedBy);
    if (!channelTarget ||
        channelTarget->getName().getStringRef() != "lvm.channel")
      return emitOpError("communication bridge channel must resolve to "
                         "lvm.channel");
    auto deviceLogicalRef = device->getAttrOfType<FlatSymbolRefAttr>("logical");
    Operation *deviceLogical =
        deviceLogicalRef
            ? SymbolTable::lookupNearestSymbolFrom(device, deviceLogicalRef)
            : nullptr;
    Operation *deviceChannel =
        deviceLogical && deviceLogical->getName().getStringRef() == "lvm.domain"
            ? SymbolTable(deviceLogical)
                  .lookup(channel.getLeafReference().getValue())
            : nullptr;
    if (!deviceChannel ||
        deviceChannel->getName().getStringRef() != "lvm.channel" ||
        !equivalentLogicalChannels(channelTarget, deviceChannel))
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' selected device channel contradicts canonical P1 channel";
    if (!siteTarget ||
        siteTarget->getName().getStringRef() != "lvm.action_site")
      return emitOpError("communication bridge action_site must resolve to "
                         "lvm.action_site");
    if (!generatorTarget ||
        generatorTarget->getName().getStringRef() != "qlx.qec_lowering")
      return emitOpError("communication bridge generated_by must resolve to "
                         "qlx.qec_lowering");
    if (siteTarget->getAttr("channel") != channel ||
        siteTarget->getAttr("channel_capability") != channelCapability ||
        siteTarget->getAttr("endpoints") != endpoints)
      return emitOpError("communication bridge provenance does not match its "
                         "action_site");

    if (!physicalCall || physicalCall.getInstanceAttr() != sourceCall)
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' source_call must name its enclosing phys.call instance";
    if (physicalCall->getAttr("channel") != channel ||
        physicalCall->getAttr("channel_capability") != channelCapability ||
        physicalCall->getAttr("endpoints") != endpoints ||
        physicalCall->getAttr("action_site") != actionSite ||
        physicalCall->getAttr("generated_by") != generatedBy)
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' provenance does not match its enclosing phys.call";

    unsigned matchedCalls = 0;
    sourceProtocol->walk([&](Operation *candidate) {
      if (candidate->getName().getStringRef() != "fabric.call")
        return;
      matchedCalls +=
          candidate->getAttr("channel") == channel &&
          candidate->getAttr("channel_capability") == channelCapability &&
          candidate->getAttr("endpoints") == endpoints &&
          candidate->getAttr("action_site") == actionSite &&
          candidate->getAttr("generated_by") == generatedBy &&
          candidate->getAttr("callee") == physicalCall.getCalleeAttr();
    });
    if (matchedCalls != 1)
      return emitOpError("communication bridge '")
             << event.getValue()
             << "' must match exactly one retained communication-qualified "
                "fabric.call with the same callee";
  }
  return success();
}

LogicalResult EpochTransitionOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every physical state type");
  if (getSourceEpochAttr() == getDestinationEpochAttr())
    return emitOpError("source and destination epochs must differ");
  if (getEvidence().empty())
    return emitOpError("requires nonempty transition evidence");
  return success();
}

struct PhysicalStateUse {
  Operation *owner;
  SmallVector<std::pair<Operation *, unsigned>, 4> chain;
};

static bool isExclusivePhysicalControl(Operation *operation) {
  return isa<qlx::cflow::IfOp, qlx::event::TryTakeOp>(operation);
}

static bool isLoopPhysicalControl(Operation *operation) {
  return isa<qlx::cflow::RepeatOp, qlx::cflow::WhileOp>(operation);
}

static bool isSupportedPhysicalStateRegion(Operation *operation) {
  return isa<CallOp, qlx::cflow::RepeatOp, qlx::cflow::IfOp,
             qlx::cflow::WhileOp, qlx::event::TryTakeOp>(operation);
}

struct GraphResourceIndex {
  llvm::StringMap<ResourceOp> resources;
  std::optional<SymbolTable> architectureSymbols;
};

static FailureOr<GraphResourceIndex> buildGraphResourceIndex(GraphOp graph) {
  GraphResourceIndex index;
  auto module = graph->getParentOfType<ModuleOp>();
  for (ResourceOp resource : module.getOps<ResourceOp>())
    index.resources.try_emplace(resource.getSymName(), resource);
  auto architecture = dyn_cast_or_null<ArchitectureOp>(
      SymbolTable::lookupNearestSymbolFrom(graph, graph.getArchitectureAttr()));
  if (!architecture) {
    graph.emitOpError("architecture must resolve to phys.machine");
    return failure();
  }
  index.architectureSymbols.emplace(architecture);
  return index;
}

static LogicalResult verifyGraphStateResource(GraphOp graph, StateType state,
                                              GraphResourceIndex &index) {
  auto found = index.resources.find(state.getResource().getValue());
  ResourceOp resource =
      found == index.resources.end() ? ResourceOp{} : found->second;
  if (!resource)
    return graph.emitOpError("physical-state resource ")
           << state.getResource() << " must resolve to phys.resource";

  auto resourceClass =
      dyn_cast_or_null<ResourceClassOp>(index.architectureSymbols->lookup(
          resource.getResourceClassAttr().getAttr()));
  if (!resourceClass)
    return graph.emitOpError("physical-state resource ")
           << state.getResource()
           << " class must resolve in the graph architecture";
  if (resource.getIndex() < 0 ||
      resource.getIndex() >= resourceClass.getCount())
    return graph.emitOpError("physical-state resource ")
           << state.getResource() << " index is outside resource class @"
           << resourceClass.getSymName() << " capacity "
           << resourceClass.getCount();
  return success();
}

static Region *definingRegion(Value value) {
  if (auto argument = dyn_cast<BlockArgument>(value))
    return argument.getOwner()->getParent();
  return cast<OpResult>(value).getOwner()->getParentRegion();
}

static PhysicalStateUse physicalStateUse(Value value, OpOperand &use) {
  PhysicalStateUse result{use.getOwner(), {}};
  Region *scope = definingRegion(value);
  for (Region *region = use.getOwner()->getParentRegion();
       region && region != scope;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    result.chain.push_back({parent, region->getRegionNumber()});
    region = parent->getParentRegion();
  }
  std::reverse(result.chain.begin(), result.chain.end());
  return result;
}

static bool physicalStateUsesConflict(const PhysicalStateUse &left,
                                      const PhysicalStateUse &right) {
  unsigned depth = 0;
  while (true) {
    bool hasLeft = depth < left.chain.size();
    bool hasRight = depth < right.chain.size();
    if (!hasLeft && !hasRight)
      return true;
    if (!hasLeft || !hasRight)
      return true;
    auto [leftOperation, leftRegion] = left.chain[depth];
    auto [rightOperation, rightRegion] = right.chain[depth];
    if (leftOperation == rightOperation) {
      if (leftRegion != rightRegion)
        return !isExclusivePhysicalControl(leftOperation);
      ++depth;
      continue;
    }
    return leftOperation->getBlock() == rightOperation->getBlock();
  }
}

static bool isLinearPhysicalType(Type type) {
  if (isa<StateType, ResourcePayloadType>(type))
    return true;
  auto event = dyn_cast<qlx::event::HandleType>(type);
  return event && event.getOwnership() == "linear";
}

static bool isNonconsumingEventUse(Value value, OpOperand &use) {
  if (!isa<qlx::event::HandleType>(value.getType()))
    return false;
  return isa<qlx::event::TestOp, qlx::event::PollOp, qlx::event::SelectReadyOp>(
      use.getOwner());
}

static bool hasSingleLocalConsumingUse(Value value) {
  if (!value.hasOneUse())
    return false;
  OpOperand &use = *value.use_begin();
  if (isNonconsumingEventUse(value, use))
    return false;
  Block *owner = nullptr;
  if (auto argument = dyn_cast<BlockArgument>(value))
    owner = argument.getOwner();
  else if (Operation *producer = value.getDefiningOp())
    owner = producer->getBlock();
  return owner && use.getOwner()->getBlock() == owner;
}

static LogicalResult verifyLinearPhysicalValue(GraphOp graph, Value value,
                                               StringRef ownerName) {
  // A sole consuming use in the defining block cannot cross region control or
  // conflict with another owner. Keep the general path for observations,
  // multiple uses, and every cross-block/region value.
  if (hasSingleLocalConsumingUse(value))
    return success();
  SmallVector<PhysicalStateUse, 4> uses;
  for (OpOperand &use : value.getUses())
    if (!isNonconsumingEventUse(value, use))
      uses.push_back(physicalStateUse(value, use));
  if (uses.empty())
    return graph.emitOpError()
           << ownerName
           << " has no consuming owner; it must be transferred, returned, "
              "awaited, cancelled, taken, or explicitly discarded";

  llvm::DenseMap<Operation *, llvm::SmallDenseSet<unsigned, 4>> coveredBranches;
  for (const PhysicalStateUse &use : uses) {
    for (auto [operation, region] : use.chain) {
      if (!isSupportedPhysicalStateRegion(operation))
        return graph.emitOpError()
               << ownerName << " crosses unsupported region control "
               << operation->getName();
      if (isa<CallOp>(operation))
        return graph.emitOpError()
               << ownerName
               << " defined outside phys.call is consumed inside its body; "
                  "thread it through an explicit call input";
      if (isLoopPhysicalControl(operation))
        return graph.emitOpError()
               << ownerName
               << " defined outside a loop is consumed inside its body; "
                  "thread it through an explicit init/carry";
      if (isExclusivePhysicalControl(operation))
        coveredBranches[operation].insert(region);
    }
  }
  for (auto &[operation, regions] : coveredBranches)
    if (regions.size() != operation->getNumRegions())
      return graph.emitOpError()
             << ownerName
             << " captured by conditional control must be consumed on every "
                "branch";

  for (auto [index, left] : llvm::enumerate(uses))
    for (const PhysicalStateUse &right : ArrayRef(uses).drop_front(index + 1))
      if (physicalStateUsesConflict(left, right))
        return graph.emitOpError()
               << ownerName << " is consumed more than once by "
               << left.owner->getName() << " and " << right.owner->getName();
  return success();
}

static LogicalResult verifyPhysicalStateLinearity(GraphOp graph,
                                                  int64_t &linearValueCount) {
  struct StateCheck {
    Value value;
    bool verifyResource;
    bool verifyOwnership;
  };
  Operation *foreignLinearCarrier = nullptr;
  Operation *unsupportedInterface = nullptr;
  Operation *unsupportedEndpoint = nullptr;
  Operation *unsupportedControl = nullptr;
  SmallVector<StateCheck, 64> stateChecks;
  llvm::DenseSet<Attribute> discoveredStateResources;
  SmallVector<Value, 32> otherLinearValues;
  auto collectLinearValue = [&](Value value) {
    Type type = value.getType();
    if (isLinearPhysicalType(type))
      ++linearValueCount;
    if (auto state = dyn_cast<StateType>(type)) {
      bool verifyResource =
          discoveredStateResources.insert(state.getResource()).second;
      bool verifyOwnership = !hasSingleLocalConsumingUse(value);
      if (verifyResource || verifyOwnership)
        stateChecks.push_back({value, verifyResource, verifyOwnership});
    } else if (isa<ResourcePayloadType>(type) ||
               (isa<qlx::event::HandleType>(type) &&
                cast<qlx::event::HandleType>(type).getOwnership() ==
                    "linear")) {
      otherLinearValues.push_back(value);
    }
  };
  graph.walk([&](Operation *operation) {
    if (!foreignLinearCarrier && operation != graph.getOperation() &&
        operation->getNumRegions() == 0 &&
        operation->getName().getDialectNamespace() != "phys" &&
        // Shared cflow boundary operations carry linear physical state
        // through cflow.if/repeat/while by construction.
        operation->getName().getDialectNamespace() != "cflow" &&
        // The shared `event` dialect's ops (test/poll/is/select_ready/
        // cancel/await/fence/selection/yield) are embedded directly in
        // `phys.graph` bodies now, exactly like `phys`'s own ops -- not a
        // foreign dialect carrying physical linear state incidentally.
        operation->getName().getDialectNamespace() != "event") {
      bool carriesLinear =
          llvm::any_of(operation->getOperandTypes(), isLinearPhysicalType) ||
          llvm::any_of(operation->getResultTypes(), isLinearPhysicalType);
      if (carriesLinear)
        foreignLinearCarrier = operation;
    }

    if (!unsupportedInterface && operation != graph.getOperation() &&
        operation->getNumRegions() != 0 &&
        !isSupportedPhysicalStateRegion(operation)) {
      bool carriesState =
          llvm::any_of(
              operation->getOperands(),
              [](Value value) { return isa<StateType>(value.getType()); }) ||
          llvm::any_of(operation->getResults(), [](Value value) {
            return isa<StateType>(value.getType());
          });
      if (!carriesState)
        for (Region &region : operation->getRegions())
          for (Block &block : region)
            carriesState |=
                llvm::any_of(block.getArguments(), [](BlockArgument argument) {
                  return isa<StateType>(argument.getType());
                });
      if (carriesState)
        unsupportedInterface = operation;
    }

    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          collectLinearValue(argument);
    for (Value result : operation->getResults())
      collectLinearValue(result);

    if (!unsupportedEndpoint && isa<AcquireOp, ReleaseOp>(operation))
      if (Operation *control = unsupportedLifetimeControl(operation, graph)) {
        unsupportedEndpoint = operation;
        unsupportedControl = control;
      }
  });
  if (foreignLinearCarrier)
    return graph.emitOpError(
               "physical linear type crosses unsupported regionless operation ")
           << foreignLinearCarrier->getName();

  if (unsupportedInterface)
    return graph.emitOpError(
               "physical state crosses unsupported region control ")
           << unsupportedInterface->getName();

  auto resourceIndex = buildGraphResourceIndex(graph);
  if (failed(resourceIndex))
    return failure();
  for (const StateCheck &check : stateChecks) {
    Value state = check.value;
    if (check.verifyResource) {
      auto stateType = cast<StateType>(state.getType());
      if (failed(verifyGraphStateResource(graph, stateType, *resourceIndex)))
        return failure();
    }
    if (!check.verifyOwnership)
      continue;
    SmallVector<PhysicalStateUse, 4> uses;
    for (OpOperand &use : state.getUses())
      uses.push_back(physicalStateUse(state, use));
    if (uses.empty())
      return graph.emitOpError(
          "physical state is never consumed; every state owner must be "
          "returned, yielded, transformed, or released");

    llvm::DenseMap<Operation *, llvm::SmallDenseSet<unsigned, 4>>
        coveredBranches;
    for (const PhysicalStateUse &use : uses) {
      for (auto [operation, region] : use.chain) {
        if (!isSupportedPhysicalStateRegion(operation))
          return graph.emitOpError(
                     "physical state crosses unsupported region control ")
                 << operation->getName();
        if (isa<CallOp>(operation))
          return graph.emitOpError(
              "physical state defined outside phys.call is consumed inside "
              "its body; thread state through an explicit call input");
        if (isLoopPhysicalControl(operation))
          return graph.emitOpError(
              "physical state defined outside a loop is consumed inside its "
              "body; thread state through an explicit init/carry");
        if (isExclusivePhysicalControl(operation))
          coveredBranches[operation].insert(region);
      }
    }
    for (auto &[operation, regions] : coveredBranches)
      if (regions.size() != operation->getNumRegions())
        return graph.emitOpError(
            "physical state captured by conditional control must be consumed "
            "on every branch");

    for (auto [index, left] : llvm::enumerate(uses))
      for (const PhysicalStateUse &right : ArrayRef(uses).drop_front(index + 1))
        if (physicalStateUsesConflict(left, right))
          return graph.emitOpError()
                 << "physical state is consumed more than once by "
                 << left.owner->getName() << " and " << right.owner->getName();
  }

  for (Value value : otherLinearValues) {
    StringRef ownerName = isa<ResourcePayloadType>(value.getType())
                              ? "physical resource payload"
                              : "linear physical event";
    if (failed(verifyLinearPhysicalValue(graph, value, ownerName)))
      return failure();
  }

  if (unsupportedEndpoint)
    return graph.emitOpError("physical ownership endpoint ")
           << unsupportedEndpoint->getName()
           << " is nested in unsupported region control "
           << unsupportedControl->getName();
  return success();
}

static bool isSchedulableGraphEvent(Operation *operation, GraphOp graph) {
  if (operation == graph.getOperation())
    return false;
  StringRef name = operation->getName().getStringRef();
  if (!name.starts_with("phys."))
    return bool(operation->getAttrOfType<StringAttr>("event_id"));
  return name != "phys.return" && name != "phys.yield" &&
         name != "cflow.yield" && name != "cflow.while_condition" &&
         name != "event.yield";
}

static std::optional<double> physicalTimingValue(Attribute raw) {
  double value = -1.0;
  if (auto number = dyn_cast_or_null<FloatAttr>(raw))
    value = number.getValueAsDouble();
  else if (auto integer = dyn_cast_or_null<IntegerAttr>(raw))
    value = static_cast<double>(integer.getInt());
  else if (auto text = dyn_cast_or_null<StringAttr>(raw)) {
    if (text.getValue().getAsDouble(value))
      return std::nullopt;
  } else {
    return std::nullopt;
  }
  if (!std::isfinite(value) || value < 0.0)
    return std::nullopt;
  return value;
}

template <typename Types>
static LogicalResult verifyUniqueCarriedStateResources(Operation *owner,
                                                       Types types) {
  llvm::SmallDenseSet<Attribute, 8> resources;
  for (Type type : types)
    if (auto state = dyn_cast<StateType>(type);
        state && !resources.insert(state.getResource()).second)
      return owner->emitOpError("cannot carry more than one physical-state "
                                "owner for resource ")
             << state.getResource();
  return success();
}

static LogicalResult verifyScheduledMacroResourceExclusion(GraphOp graph) {
  auto architecture = dyn_cast_or_null<ArchitectureOp>(
      SymbolTable::lookupNearestSymbolFrom(graph, graph.getArchitectureAttr()));
  if (!architecture)
    return success();

  llvm::DenseMap<Attribute, ResourceRequestOp> requestByResourceClass;
  graph.walk([&](ResourceRequestOp request) {
    auto factoryMode = request->getAttrOfType<StringAttr>("factory_mode");
    auto bindingRef = request.getPhysicalBindingAttr();
    if (!factoryMode || factoryMode.getValue() != "scheduled_macro" ||
        !bindingRef)
      return;
    auto binding = dyn_cast_or_null<QECBindingOp>(
        SymbolTable::lookupNearestSymbolFrom(request, bindingRef));
    if (!binding || binding->getParentOp() != architecture.getOperation())
      return;
    for (Attribute resourceClass : binding.getResources())
      requestByResourceClass.try_emplace(resourceClass, request);
  });
  if (requestByResourceClass.empty())
    return success();

  auto module = graph->getParentOfType<ModuleOp>();
  llvm::StringMap<ResourceOp> resources;
  for (ResourceOp resource : module.getOps<ResourceOp>())
    resources.try_emplace(resource.getSymName(), resource);

  ResourceRequestOp conflictingRequest;
  ResourceOp conflictingResource;
  auto checkType = [&](Type type) {
    if (conflictingRequest)
      return;
    auto state = dyn_cast<StateType>(type);
    if (!state)
      return;
    auto resource = resources.find(state.getResource().getValue());
    if (resource == resources.end())
      return;
    auto request =
        requestByResourceClass.find(resource->second.getResourceClassAttr());
    if (request == requestByResourceClass.end())
      return;
    conflictingRequest = request->second;
    conflictingResource = resource->second;
  };
  for (Type type : graph.getFunctionType().getInputs())
    checkType(type);
  for (Type type : graph.getFunctionType().getResults())
    checkType(type);
  graph.walk([&](Operation *operation) {
    if (conflictingRequest)
      return;
    for (Type type : operation->getOperandTypes())
      checkType(type);
    for (Type type : operation->getResultTypes())
      checkType(type);
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          checkType(argument.getType());
  });
  if (!conflictingRequest)
    return success();
  return conflictingRequest.emitOpError(
             "scheduled-macro binding resource class ")
         << conflictingResource.getResourceClassAttr()
         << " is also used through concrete physical resource @"
         << conflictingResource.getSymName()
         << "; member-level exclusion is unsupported, so scheduled engines "
            "must use resource classes disjoint from graph-carried states";
}

LogicalResult GraphOp::verify() {
  const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
  const auto started = PhysVerifyClock::now();
  if (profile)
    llvm::errs() << "phys-verify: graph-local-start\n";
  if (getBody().empty())
    return emitOpError("requires one entry block");
  Block &entry = getBody().front();
  FunctionType type = getFunctionType();
  if (entry.getArgumentTypes() != type.getInputs())
    return emitOpError("entry arguments must match function_type inputs");
  auto ret = dyn_cast<ReturnOp>(entry.getTerminator());
  if (!ret || ret.getOperandTypes() != type.getResults())
    return emitOpError("phys.return operands must match function_type results");

  auto verifyUniqueBoundaryResources =
      [&](TypeRange types, StringRef boundary) -> LogicalResult {
    llvm::SmallDenseSet<Attribute, 8> resources;
    for (Type value : types)
      if (auto state = dyn_cast<StateType>(value);
          state && !resources.insert(state.getResource()).second)
        return emitOpError()
               << boundary
               << " cannot contain more than one physical-state owner for "
                  "resource "
               << state.getResource();
    return success();
  };
  if (failed(verifyUniqueBoundaryResources(type.getInputs(), "graph inputs")) ||
      failed(verifyUniqueBoundaryResources(type.getResults(), "graph results")))
    return failure();
  if (failed(verifyScheduledMacroResourceExclusion(*this)))
    return failure();

  llvm::StringMap<Operation *> eventIds;
  llvm::SmallVector<CallOp, 16> physicalCalls;
  bool graphAcquiresResources = false;
  Operation *malformedEvent = nullptr;
  Operation *duplicateEvent = nullptr;
  StringRef duplicateEventId;
  walk([&](Operation *operation) {
    if (auto call = dyn_cast<CallOp>(operation))
      physicalCalls.push_back(call);
    if (auto acquire = dyn_cast<AcquireOp>(operation))
      graphAcquiresResources |= !acquire.getResources().empty();
    auto event = operation->getAttrOfType<StringAttr>("event_id");
    if (!event)
      return;
    if (malformedEvent || duplicateEvent)
      return;
    if (event.getValue().empty()) {
      malformedEvent = operation;
      return;
    }
    if (!eventIds.try_emplace(event.getValue(), operation).second) {
      duplicateEvent = operation;
      duplicateEventId = event.getValue();
    }
  });
  if (malformedEvent)
    return emitOpError(
               "event_id values must be nonempty; found an empty ID on ")
           << malformedEvent->getName();
  if (duplicateEvent)
    return emitOpError("event_id values must be graph-global and unique; "
                       "duplicate '")
           << duplicateEventId << "'";

  // `cflow.repeat`/`cflow.if`/`cflow.while` are shared, dialect-agnostic ops:
  // they cannot themselves know about `!phys.state<@resource>`, so this
  // graph-level walk enforces the same "at most one owner per resource
  // among carried values" invariant that each op's own verifier used to.
  LogicalResult carriedResourceResult = success();
  walk([&](Operation *operation) {
    if (failed(carriedResourceResult))
      return;
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation))
      carriedResourceResult = verifyUniqueCarriedStateResources(
          repeat, repeat.getInits().getTypes());
    else if (auto branch = dyn_cast<qlx::cflow::IfOp>(operation))
      carriedResourceResult =
          verifyUniqueCarriedStateResources(branch, branch.getResultTypes());
    else if (auto loop = dyn_cast<qlx::cflow::WhileOp>(operation))
      carriedResourceResult =
          verifyUniqueCarriedStateResources(loop, loop.getInits().getTypes());
  });
  if (failed(carriedResourceResult))
    return failure();

  llvm::SmallDenseSet<Attribute, 16> acquiredResourceSymbols;
  bool duplicateAcquiredResource = false;
  walk([&](AcquireOp acquire) {
    for (Attribute resource : acquire.getResources())
      if (!acquiredResourceSymbols.insert(resource).second)
        duplicateAcquiredResource = true;
  });
  if (duplicateAcquiredResource)
    return emitOpError(
        "each phys.resource symbol may be acquired only once; concrete "
        "resource reuse requires distinct allocation resource symbols");
  auto module = (*this)->getParentOfType<ModuleOp>();
  llvm::DenseMap<Attribute, Operation *> expectedCommunicationCalls;
  llvm::StringMap<llvm::SmallDenseSet<Attribute, 4>>
      retainedCommunicationCalleesByProtocol;
  bool malformedExpectedCommunication = false;
  if (auto sourceProtocol = getSourceProtocolAttr()) {
    Operation *source =
        SymbolTable::lookupNearestSymbolFrom(*this, sourceProtocol);
    if (!source || (source->getName().getStringRef() != "fabric.protocol" &&
                    source->getName().getStringRef() != "fabric.gadget")) {
      malformedExpectedCommunication = true;
    } else {
      source->walk([&](Operation *candidate) {
        if (candidate->getName().getStringRef() != "fabric.call" ||
            !candidate->hasAttr("channel"))
          return;
        Attribute actionSite = candidate->getAttr("action_site");
        if (!actionSite ||
            !expectedCommunicationCalls.try_emplace(actionSite, candidate)
                 .second)
          malformedExpectedCommunication = true;
      });
    }
  } else {
    module.walk([&](Operation *candidate) {
      if (candidate->getName().getStringRef() != "fabric.call" ||
          !candidate->hasAttr("channel"))
        return;
      Operation *source = candidate->getParentOp();
      while (source && source->getName().getStringRef() != "fabric.protocol" &&
             source->getName().getStringRef() != "fabric.gadget")
        source = source->getParentOp();
      auto sourceName =
          source ? source->getAttrOfType<StringAttr>("sym_name") : nullptr;
      Attribute callee = candidate->getAttr("callee");
      if (sourceName && callee)
        retainedCommunicationCalleesByProtocol[sourceName.getValue()].insert(
            callee);
    });
    for (CallOp call : physicalCalls) {
      StringRef sourceName = call.getInstance().split('.').first;
      auto source = retainedCommunicationCalleesByProtocol.find(sourceName);
      if (source != retainedCommunicationCalleesByProtocol.end() &&
          source->second.contains(call.getCalleeAttr()))
        malformedExpectedCommunication = true;
    }
  }

  // The retained P2 source defines which physical calls must remain
  // communication-qualified. Physical call instances then own graph-local
  // route coverage, and a native event cannot be borrowed by another routing
  // container for the same graph.
  llvm::StringMap<CallOp> communicationCalls;
  llvm::DenseMap<Attribute, CallOp> projectedCommunicationCalls;
  bool duplicateCallInstance = false;
  bool mismatchedProjectedCommunication = false;
  for (CallOp call : physicalCalls) {
    if (!call->hasAttr("channel"))
      continue;
    auto [_, inserted] =
        communicationCalls.try_emplace(call.getInstance(), call);
    duplicateCallInstance |= !inserted;
    Attribute actionSite = call->getAttr("action_site");
    if (!actionSite) {
      mismatchedProjectedCommunication = true;
      continue;
    }
    auto expected = expectedCommunicationCalls.find(actionSite);
    if (expected == expectedCommunicationCalls.end() ||
        call->getAttr("channel") != expected->second->getAttr("channel") ||
        call->getAttr("channel_capability") !=
            expected->second->getAttr("channel_capability") ||
        call->getAttr("endpoints") != expected->second->getAttr("endpoints") ||
        call->getAttr("generated_by") !=
            expected->second->getAttr("generated_by") ||
        call->getAttr("callee") != expected->second->getAttr("callee") ||
        !projectedCommunicationCalls.try_emplace(actionSite, call).second)
      mismatchedProjectedCommunication = true;
  }
  if (duplicateCallInstance)
    return emitOpError(
        "communication phys.call instances must be unique within a graph");
  if (malformedExpectedCommunication || mismatchedProjectedCommunication ||
      projectedCommunicationCalls.size() != expectedCommunicationCalls.size())
    return emitOpError(
        "projected communication calls must exactly match retained "
        "communication calls in the source protocol");

  llvm::DenseMap<Attribute, StringAttr> resourceBindings;
  bool ambiguousResourceBinding = false;
  unsigned graphAllocationMappings = 0;
  module.walk([&](AllocationMappingOp mapping) {
    Operation *candidateGraph =
        SymbolTable::lookupNearestSymbolFrom(mapping, mapping.getGraphAttr());
    if (candidateGraph != getOperation())
      return;
    ++graphAllocationMappings;
    for (Attribute raw : mapping.getEntries()) {
      auto allocation = dyn_cast<DictionaryAttr>(raw);
      auto resources =
          allocation ? allocation.getAs<ArrayAttr>("resources") : nullptr;
      auto binding = allocation
                         ? allocation.getAs<StringAttr>("physical_binding")
                         : nullptr;
      if (!resources || !binding)
        continue;
      for (Attribute resource : resources) {
        auto [found, inserted] =
            resourceBindings.try_emplace(resource, binding);
        if (!inserted && found->second != binding)
          ambiguousResourceBinding = true;
      }
    }
  });
  if (ambiguousResourceBinding)
    return emitOpError(
        "graph allocation evidence assigns one resource to multiple physical "
        "bindings");
  if (!expectedCommunicationCalls.empty() && graphAcquiresResources &&
      graphAllocationMappings != 1)
    return emitOpError(
        "a communication graph with acquired resources requires exactly one "
        "complete phys.allocation_mapping");

  llvm::StringSet<> coveredCalls;
  llvm::StringMap<StringRef> routedNativeEventOwners;
  bool malformedRouting = false;
  module.walk([&](RoutingOp routing) {
    if (routing.getGraphAttr().getValue() != getSymName())
      return;
    for (Attribute raw : routing.getSteps()) {
      auto step = dyn_cast<DictionaryAttr>(raw);
      if (!step || !step.getAs<UnitAttr>("communication_bridge"))
        continue;
      auto sourceCall = step.getAs<StringAttr>("source_call");
      if (!sourceCall || sourceCall.getValue().empty()) {
        malformedRouting = true;
        continue;
      }
      auto call = communicationCalls.find(sourceCall.getValue());
      if (call == communicationCalls.end() ||
          step.get("channel") != call->second->getAttr("channel") ||
          step.get("channel_capability") !=
              call->second->getAttr("channel_capability") ||
          step.get("endpoints") != call->second->getAttr("endpoints") ||
          step.get("action_site") != call->second->getAttr("action_site") ||
          step.get("generated_by") != call->second->getAttr("generated_by")) {
        malformedRouting = true;
        continue;
      }
      coveredCalls.insert(sourceCall.getValue());
      auto nativeEvents = step.getAs<ArrayAttr>("native_events");
      if (!nativeEvents) {
        malformedRouting = true;
        continue;
      }
      for (Attribute value : nativeEvents) {
        auto event = dyn_cast<StringAttr>(value);
        if (!event || !routedNativeEventOwners
                           .try_emplace(event.getValue(), sourceCall.getValue())
                           .second)
          malformedRouting = true;
      }
    }
  });
  if (malformedRouting)
    return emitOpError(
        "communication routing must be graph-, call-, and native-event-local");
  for (auto &entry : communicationCalls)
    if (!coveredCalls.contains(entry.getKey()))
      return emitOpError(
          "projected communication calls require verifier-closed physical "
          "bridge routing");

  bool incompleteAllocationEvidence = false;
  bool missingCrossBindingRoute = false;
  for (auto &entry : communicationCalls) {
    CallOp communicationCall = entry.getValue();
    communicationCall.walk([&](ApplyOp apply) {
      CallOp enclosingCommunicationCall;
      for (Operation *parent = apply->getParentOp(); parent;
           parent = parent->getParentOp()) {
        auto candidate = dyn_cast<CallOp>(parent);
        if (candidate && candidate->hasAttr("channel")) {
          enclosingCommunicationCall = candidate;
          break;
        }
      }
      if (enclosingCommunicationCall != communicationCall)
        return;

      llvm::SmallDenseSet<StringRef, 2> bindings;
      for (Value input : apply.getInputs()) {
        auto state = dyn_cast<StateType>(input.getType());
        auto owner = state ? resourceBindings.find(state.getResource())
                           : resourceBindings.end();
        if (!state || owner == resourceBindings.end()) {
          incompleteAllocationEvidence = true;
          return;
        }
        bindings.insert(owner->second.getValue());
      }
      if (bindings.size() <= 1)
        return;

      auto event = apply->getAttrOfType<StringAttr>("event_id");
      auto routed = event ? routedNativeEventOwners.find(event.getValue())
                          : routedNativeEventOwners.end();
      if (!event || routed == routedNativeEventOwners.end() ||
          routed->second != entry.getKey())
        missingCrossBindingRoute = true;
    });
  }
  if (incompleteAllocationEvidence)
    return emitOpError(
        "communication physical actions require complete graph-local "
        "physical-binding allocation evidence");
  if (missingCrossBindingRoute)
    return emitOpError(
        "communication bridge routing must cover every cross-binding native "
        "event");
  if (profile) {
    llvm::errs() << "phys-verify: graph-local-done "
                 << physVerifySecondsSince(started) << "s\n";
    profiledGraphLocalDone = PhysVerifyClock::now();
  }
  return success();
}

static void
appendNestedOperationsPreOrder(Operation *owner,
                               llvm::SmallVectorImpl<Operation *> &result) {
  llvm::SmallVector<Operation *, 16> pending;
  auto pushChildren = [&](Operation *parent) {
    llvm::SmallVector<Operation *, 16> children;
    for (Region &region : parent->getRegions())
      for (Block &block : region)
        for (Operation &child : block)
          children.push_back(&child);
    for (Operation *child : llvm::reverse(children))
      pending.push_back(child);
  };
  pushChildren(owner);
  while (!pending.empty()) {
    Operation *operation = pending.pop_back_val();
    result.push_back(operation);
    pushChildren(operation);
  }
}

static LogicalResult verifyCallTemplateAgainstCanonical(
    CallTemplateOp invocation, CallOp canonical,
    llvm::function_ref<ResourceOp(FlatSymbolRefAttr)> resolveResource,
    llvm::DenseMap<Operation *, llvm::StringMap<SmallVector<Operation *, 1>>>
        &recordProducers,
    DenseSet<std::pair<Operation *, Attribute>> &verifiedElidedStateAliases);

LogicalResult GraphOp::verifyRegions() {
  const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
  const auto started = PhysVerifyClock::now();
  auto last = started;
  auto checkpoint = [&](StringRef name) {
    const auto now = PhysVerifyClock::now();
    if (profile)
      llvm::errs() << "phys-verify: graph-regions-" << name << ' '
                   << std::chrono::duration<double>(now - last).count()
                   << "s total="
                   << std::chrono::duration<double>(now - started).count()
                   << "s\n";
    last = now;
  };
  if (profile) {
    llvm::errs() << "phys-verify: graph-regions-start";
    if (profiledGraphLocalDone)
      llvm::errs() << " child-local="
                   << physVerifySecondsSince(*profiledGraphLocalDone) << "s";
    llvm::errs() << "\n";
  }
  int64_t linearValueCount = 0;
  if (failed(verifyPhysicalStateLinearity(*this, linearValueCount)))
    return failure();
  if (auto authored = (*this)->getAttrOfType<IntegerAttr>("linear_value_count"))
    if (authored.getInt() != linearValueCount)
      return emitOpError() << "linear_value_count is " << authored.getInt()
                           << " but the verified graph defines "
                           << linearValueCount << " linear physical values";
  checkpoint("linearity");
  using CallProjectionWitness = std::pair<FlatSymbolRefAttr, StringAttr>;
  using ResourceCallWitness =
      std::tuple<SymbolRefAttr, Attribute, FlatSymbolRefAttr, StringAttr>;
  llvm::SmallVector<CallProjectionWitness> selectedFabricCallTree;
  llvm::SmallVector<ResourceCallWitness> selectedFabricCalls;
  llvm::SmallVector<CallProjectionWitness> projectedPhysicalCallTree;
  llvm::SmallVector<ResourceCallWitness> projectedPhysicalCalls;
  bool requiresSelectedCallTree = false;
  walk([&](CallOp call) {
    requiresSelectedCallTree |= call->hasAttr("resource_action_site") ||
                                call->hasAttr("resource_objective");
  });
  auto module = (*this)->getParentOfType<ModuleOp>();
  llvm::DenseMap<Attribute, qlx::lvm::ActionSiteOp> resourceActionSites;
  if (requiresSelectedCallTree)
    for (auto domain : module.getOps<qlx::lvm::DomainOp>())
      for (auto site : domain.getBody().getOps<qlx::lvm::ActionSiteOp>()) {
        auto kind = site.getKindAttr();
        if (!kind || kind.getValue() != "resource_action")
          continue;
        auto reference = SymbolRefAttr::get(
            getContext(), domain.getSymName(),
            {FlatSymbolRefAttr::get(getContext(), site.getSymName())});
        resourceActionSites.try_emplace(reference, site);
      }

  // A projected graph owns one exact P2 source callable.  Reconstruct the
  // folded call-instance hierarchy using the same depth-first numbering as the
  // P2-to-P3 projector.  Walking every fabric.call in the module would let an
  // unrelated linked definition lend (or poison) a resource witness, while a
  // tuple-only comparison would let the witness move to a nested call.
  if (auto sourceRef = getSourceProtocolAttr()) {
    Operation *source = SymbolTable::lookupNearestSymbolFrom(*this, sourceRef);
    if (!source || (source->getName().getStringRef() != "fabric.protocol" &&
                    source->getName().getStringRef() != "fabric.gadget"))
      return emitOpError(
          "source_protocol must resolve to the projected Fabric callable");
    auto sourceName = source->getAttrOfType<StringAttr>("sym_name");
    if (!sourceName || sourceName.getValue().empty())
      return emitOpError(
          "projected Fabric source callable requires a nonempty symbol");

    llvm::DenseMap<Operation *, llvm::SmallVector<Operation *, 16>>
        callableOperations;
    llvm::DenseMap<Operation *, Operation *> resolvedInvocationTargets;
    auto resolveInvocationTarget = [&](Operation *invocation,
                                       FlatSymbolRefAttr callee) {
      auto [cached, inserted] =
          resolvedInvocationTargets.try_emplace(invocation, nullptr);
      if (inserted)
        cached->second =
            SymbolTable::lookupNearestSymbolFrom(invocation, callee);
      return cached->second;
    };
    auto collectCallableOperations =
        [&](Operation *callable) -> FailureOr<ArrayRef<Operation *>> {
      auto [cached, inserted] = callableOperations.try_emplace(callable);
      if (!inserted)
        return ArrayRef<Operation *>(cached->second);
      Operation *bodyOwner = callable;
      if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callable)) {
        if (auto realization = gadget.getRealizationAttr()) {
          bodyOwner =
              SymbolTable::lookupNearestSymbolFrom(callable, realization);
          if (!isa_and_nonnull<qlx::fabric::CircuitOp>(bodyOwner)) {
            emitOpError("projected Fabric gadget @")
                << gadget.getSymName()
                << " must resolve its retained fabric.circuit realization";
            return failure();
          }
        }
      }
      appendNestedOperationsPreOrder(bodyOwner, cached->second);
      return ArrayRef<Operation *>(cached->second);
    };

    // First validate the reachable source closure once, without expanding a
    // shared callable for every dynamic invocation or constructing instance
    // strings.  The exact call-tree reconstruction below is only observable
    // for resource-qualified calls, but omitting projected provenance must not
    // be able to suppress that comparison when it remains present in P2.
    struct ClosureFrame {
      Operation *callable;
      ArrayRef<Operation *> operations;
      size_t next = 0;
    };
    llvm::SmallDenseMap<Operation *, uint8_t, 16> closureState;
    llvm::SmallVector<ClosureFrame, 16> closureFrames;
    auto pushClosureCallable = [&](Operation *callable) -> LogicalResult {
      uint8_t &state = closureState[callable];
      if (state == 1)
        return emitOpError(
            "recursive Fabric call graph cannot define physical call "
            "instances");
      if (state == 2)
        return success();
      auto operations = collectCallableOperations(callable);
      if (failed(operations))
        return failure();
      state = 1;
      closureFrames.push_back(ClosureFrame{callable, *operations});
      return success();
    };
    if (failed(pushClosureCallable(source)))
      return failure();
    while (!closureFrames.empty()) {
      ClosureFrame &frame = closureFrames.back();
      if (frame.next == frame.operations.size()) {
        closureState[frame.callable] = 2;
        closureFrames.pop_back();
        continue;
      }
      Operation *operation = frame.operations[frame.next++];
      FlatSymbolRefAttr callee;
      if (auto call = dyn_cast<qlx::fabric::CallOp>(operation)) {
        callee = call.getCalleeAttr();
        bool hasSite = call->hasAttr("resource_action_site");
        bool hasObjective = call->hasAttr("resource_objective");
        if (hasSite != hasObjective)
          return emitOpError(
              "selected Fabric resource call has incomplete provenance");
        requiresSelectedCallTree |= hasSite;
      } else if (operation->getName().getStringRef() == "fabric.relocate" ||
                 operation->getName().getStringRef() ==
                     "fabric.establish_support" ||
                 operation->getName().getStringRef() ==
                     "fabric.establish_topological_record") {
        callee = operation->getAttrOfType<FlatSymbolRefAttr>("callee");
      } else {
        continue;
      }
      if (!callee)
        return emitOpError(
            "projected Fabric invocation requires a typed callee");
      Operation *target = resolveInvocationTarget(operation, callee);
      if (!target || (target->getName().getStringRef() != "fabric.protocol" &&
                      target->getName().getStringRef() != "fabric.gadget"))
        return emitOpError("projected Fabric invocation @")
               << callee.getValue()
               << " must resolve inside the retained source closure";
      if (failed(pushClosureCallable(target)))
        return failure();
    }

    if (requiresSelectedCallTree) {
      struct CallableFrame {
        Operation *callable;
        std::string instance;
        ArrayRef<Operation *> operations;
        size_t next = 0;
      };

      int64_t invocationOrdinal = 0;
      llvm::SmallPtrSet<Operation *, 16> activeCallables;
      llvm::SmallVector<CallableFrame, 16> frames;
      auto pushCallable = [&](Operation *callable,
                              std::string instance) -> LogicalResult {
        if (!activeCallables.insert(callable).second)
          return emitOpError(
              "recursive Fabric call graph cannot define physical call "
              "instances");
        auto operations = collectCallableOperations(callable);
        if (failed(operations)) {
          activeCallables.erase(callable);
          return failure();
        }
        frames.push_back(
            CallableFrame{callable, std::move(instance), *operations});
        return success();
      };
      if (failed(pushCallable(source, sourceName.getValue().str())))
        return failure();

      while (!frames.empty()) {
        CallableFrame &frame = frames.back();
        if (frame.next == frame.operations.size()) {
          activeCallables.erase(frame.callable);
          frames.pop_back();
          continue;
        }
        Operation *operation = frame.operations[frame.next++];
        StringRef suffix;
        FlatSymbolRefAttr callee;
        qlx::fabric::CallOp selectedCall;
        if (auto call = dyn_cast<qlx::fabric::CallOp>(operation)) {
          selectedCall = call;
          callee = call.getCalleeAttr();
          suffix = "call";
        } else if (operation->getName().getStringRef() == "fabric.relocate") {
          callee = operation->getAttrOfType<FlatSymbolRefAttr>("callee");
          suffix = "relocate";
        } else if (operation->getName().getStringRef() ==
                   "fabric.establish_support") {
          callee = operation->getAttrOfType<FlatSymbolRefAttr>("callee");
          suffix = "support";
        } else if (operation->getName().getStringRef() ==
                   "fabric.establish_topological_record") {
          callee = operation->getAttrOfType<FlatSymbolRefAttr>("callee");
          suffix = "topological";
        } else {
          continue;
        }
        if (!callee)
          return emitOpError(
              "projected Fabric invocation requires a typed callee");
        Operation *target = resolveInvocationTarget(operation, callee);
        if (!target || (target->getName().getStringRef() != "fabric.protocol" &&
                        target->getName().getStringRef() != "fabric.gadget"))
          return emitOpError("projected Fabric invocation @")
                 << callee.getValue()
                 << " must resolve inside the retained source closure";
        std::string instance =
            (Twine(frame.instance) + "." + callee.getValue() + "." + suffix +
             Twine(invocationOrdinal++))
                .str();
        if (selectedCall) {
          auto instanceAttr = StringAttr::get(getContext(), instance);
          selectedFabricCallTree.emplace_back(callee, instanceAttr);
          auto site = selectedCall->getAttrOfType<SymbolRefAttr>(
              "resource_action_site");
          Attribute objective = selectedCall->getAttr("resource_objective");
          if (static_cast<bool>(site) != static_cast<bool>(objective))
            return emitOpError(
                "selected Fabric resource call has incomplete provenance");
          if (site) {
            auto retained = resourceActionSites.find(site);
            if (retained == resourceActionSites.end() ||
                retained->second->getAttr("objective") != objective)
              return emitOpError(
                  "selected Fabric resource call must resolve one exact "
                  "resource action site and objective");
            selectedFabricCalls.emplace_back(site, objective, callee,
                                             instanceAttr);
          }
        }
        if (failed(pushCallable(target, std::move(instance))))
          return failure();
      }
    }
  }

  checkpoint("selected-call-tree");
  llvm::SmallVector<Operation *, 16> physicalOperations;
  appendNestedOperationsPreOrder(getOperation(), physicalOperations);

  Operation *architectureTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getArchitectureAttr());
  auto architecture = dyn_cast_or_null<ArchitectureOp>(architectureTarget);
  if (!architecture)
    return emitOpError("architecture must resolve to phys.machine");
  SymbolTable architectureSymbols(architecture);
  llvm::StringMap<ResourceOp> resourceSymbols;
  for (ResourceOp resource : module.getOps<ResourceOp>())
    resourceSymbols.try_emplace(resource.getSymName(), resource);
  auto resolveResource = [&](FlatSymbolRefAttr reference) -> ResourceOp {
    auto found = resourceSymbols.find(reference.getValue());
    return found == resourceSymbols.end() ? ResourceOp{} : found->second;
  };
  llvm::DenseMap<Attribute, Operation *> resolvedGraphSymbols;
  auto resolveGraphSymbol = [&](Operation *anchor,
                                Attribute reference) -> Operation * {
    auto found = resolvedGraphSymbols.find(reference);
    if (found != resolvedGraphSymbols.end())
      return found->second;
    auto symbol = dyn_cast<SymbolRefAttr>(reference);
    Operation *resolved =
        symbol ? SymbolTable::lookupNearestSymbolFrom(anchor, symbol) : nullptr;
    resolvedGraphSymbols.try_emplace(reference, resolved);
    return resolved;
  };
  for (Operation *operation : physicalOperations) {
    auto apply = dyn_cast<ApplyOp>(operation);
    if (!apply)
      continue;
    Operation *target = architectureSymbols.lookup(apply.getAction());
    if (!target)
      target = resolveGraphSymbol(apply, apply.getActionAttr());
    if (!target)
      continue; // Partial linked modules resolve at link time.
    auto action = dyn_cast<ActionOp>(target);
    if (!action)
      return apply.emitOpError("action reference must resolve to phys.action");
    int64_t batchLanes =
        apply.getBatchLanesAttr() ? apply.getBatchLanesAttr().getInt() : 1;
    if (failed(verifyApplyActionShape(apply, action)))
      return failure();

    for (auto [index, input] : llvm::enumerate(apply.getInputs())) {
      auto state = dyn_cast<StateType>(input.getType());
      if (!state)
        continue;
      ResourceOp resource = resolveResource(state.getResource());
      if (!resource)
        continue;
      Operation *classTarget =
          architectureSymbols.lookup(resource.getResourceClassAttr().getAttr());
      if (!classTarget)
        continue;
      auto resourceClass = dyn_cast<ResourceClassOp>(classTarget);
      if (!resourceClass)
        return apply.emitOpError("resource class @")
               << resource.getResourceClass() << " bound to state operand "
               << index << " must resolve to phys.resource_class";
      bool advertised =
          llvm::any_of(resourceClass.getNativeActions(), [&](Attribute value) {
            if (auto ref = dyn_cast<FlatSymbolRefAttr>(value))
              return ref.getValue() == apply.getAction();
            if (auto name = dyn_cast<StringAttr>(value))
              return name.getValue() == apply.getAction();
            return false;
          });
      if (!advertised)
        return apply.emitOpError("resource class @")
               << resourceClass.getSymName() << " of state operand " << index
               << " does not advertise native action @" << apply.getAction();
    }
    if (!apply.getTopology() || action.getBroadcast())
      continue;
    if (action.getArity() != 2)
      return apply.emitOpError("topology-qualified action must have arity two");
    Operation *topologyTarget =
        architectureSymbols.lookup(*apply.getTopology());
    auto topology = dyn_cast_or_null<TopologyOp>(topologyTarget);
    if (!topology)
      return apply.emitOpError(
          "topology reference must resolve in graph architecture");
    for (int64_t lane = 0; lane < batchLanes; ++lane) {
      int64_t indices[2];
      bool completeTopology = true;
      for (int64_t ordinal = 0; ordinal < 2; ++ordinal) {
        Value input = apply.getInputs()[2 * lane + ordinal];
        auto state = cast<StateType>(input.getType());
        ResourceOp resource = resolveResource(state.getResource());
        if (!resource) {
          completeTopology = false;
          break;
        }
        indices[ordinal] = resource.getIndex();
      }
      if (!completeTopology)
        continue;
      if (!physicalTopologyHasEdge(topology, indices[0], indices[1]))
        return apply.emitOpError("nonlocal action @")
               << action.getSymName() << " lane " << lane
               << " between topology nodes " << indices[0] << " and "
               << indices[1];
    }
  }

  checkpoint("applies");
  constexpr llvm::StringLiteral nativeProductRotationCapability =
      "qlx.physical/native_pauli_product_rotation";
  for (Operation *operation : physicalOperations) {
    auto rotation = dyn_cast<RotateProductOp>(operation);
    if (!rotation)
      continue;
    for (auto [index, input] : llvm::enumerate(rotation.getInputs())) {
      auto state = dyn_cast<StateType>(input.getType());
      if (!state)
        continue;
      ResourceOp resource = resolveResource(state.getResource());
      if (!resource)
        continue;
      bool equipped = false;
      if (auto capabilities =
              resource->getAttrOfType<ArrayAttr>("capabilities"))
        equipped = llvm::any_of(capabilities, [&](Attribute value) {
          auto key = dyn_cast<StringAttr>(value);
          return key && key.getValue() == nativeProductRotationCapability;
        });
      if (equipped)
        continue;

      Operation *classTarget =
          architectureSymbols.lookup(resource.getResourceClassAttr().getAttr());
      if (!classTarget)
        continue;
      auto resourceClass = dyn_cast<ResourceClassOp>(classTarget);
      if (!resourceClass)
        return rotation.emitOpError("resource class @")
               << resource.getResourceClass() << " bound to state operand "
               << index << " must resolve to phys.resource_class";

      // Preserve the documented alpha compatibility boundary. A bare `rpp`
      // entry is accepted only when it is not a typed action declaration; the
      // canonical path is the carrier-local capability checked above.
      bool compatibilityRpp =
          llvm::any_of(resourceClass.getNativeActions(), [&](Attribute value) {
            if (auto name = dyn_cast<StringAttr>(value))
              return name.getValue() == "rpp";
            auto reference = dyn_cast<FlatSymbolRefAttr>(value);
            return reference && reference.getValue() == "rpp" &&
                   !resolveGraphSymbol(rotation, reference);
          });
      if (!compatibilityRpp)
        return rotation.emitOpError("state operand ")
               << index << " carrier @" << resource.getSymName()
               << " lacks typed native Pauli-product-rotation capability '"
               << nativeProductRotationCapability << "'";
    }
  }

  checkpoint("product-rotations");
  llvm::StringMap<CallOp> canonicalTemplateEvents;
  llvm::DenseMap<Operation *, llvm::StringMap<SmallVector<Operation *, 1>>>
      canonicalRecordProducers;
  DenseSet<std::pair<Operation *, Attribute>> verifiedElidedStateAliases;
  for (Operation *operation : physicalOperations) {
    if (auto invocation = dyn_cast<CallTemplateOp>(operation)) {
      auto canonical =
          canonicalTemplateEvents.find(invocation.getTemplateEvent());
      if (canonical == canonicalTemplateEvents.end())
        return invocation.emitOpError(
            "template_event must resolve to exactly one earlier phys.call in "
            "the same graph");
      if (failed(verifyCallTemplateAgainstCanonical(
              invocation, canonical->second, resolveResource,
              canonicalRecordProducers, verifiedElidedStateAliases)))
        return failure();
      continue;
    }
    auto call = dyn_cast<CallOp>(operation);
    if (!call)
      continue;
    if (auto event = call.getEventIdAttr(); event && !event.empty()) {
      auto [_, inserted] =
          canonicalTemplateEvents.try_emplace(event.getValue(), call);
      if (!inserted)
        return call.emitOpError(
            "canonical phys.call event_id must be unique before use by "
            "phys.call_template");
    }
    auto instance = call->getAttrOfType<StringAttr>("instance");
    projectedPhysicalCallTree.emplace_back(call.getCalleeAttr(), instance);
    auto site = call->getAttrOfType<SymbolRefAttr>("resource_action_site");
    Attribute objective = call->getAttr("resource_objective");
    if (site && objective)
      projectedPhysicalCalls.emplace_back(site, objective, call.getCalleeAttr(),
                                          instance);
  }
  checkpoint("templates");
  if (!selectedFabricCalls.empty() || !projectedPhysicalCalls.empty()) {
    if (!getSourceProtocolAttr())
      return emitOpError("resource-qualified physical calls require a retained "
                         "graph source_protocol");
    if (selectedFabricCallTree != projectedPhysicalCallTree)
      return emitOpError("physical call callee and instance hierarchy must "
                         "exactly match the selected Fabric call tree from "
                         "the retained source protocol");
    if (selectedFabricCalls != projectedPhysicalCalls)
      return emitOpError("resource-qualified physical calls must preserve the "
                         "selected Fabric call order, exact instance, and "
                         "provenance");
  }
  Operation *missingEvent = nullptr;
  walk([&](Operation *operation) {
    if (!missingEvent && isSchedulableGraphEvent(operation, *this) &&
        !operation->getAttrOfType<StringAttr>("event_id"))
      missingEvent = operation;
  });
  if (missingEvent)
    return emitOpError("every schedulable graph operation requires a stable "
                       "event_id; missing on ")
           << missingEvent->getName();

  if (auto pointRef =
          (*this)->getAttrOfType<FlatSymbolRefAttr>("operating_point")) {
    Operation *resolved = SymbolTable::lookupNearestSymbolFrom(*this, pointRef);
    auto point = dyn_cast_or_null<OperatingPointOp>(resolved);
    if (!point)
      return emitOpError(
          "operating_point must resolve to phys.operating_point");
    if (point.getMachineAttr() != getArchitectureAttr())
      return emitOpError(
          "operating_point must belong to the graph architecture");
  }
  checkpoint("done");
  profiledGraphLocalDone.reset();
  return success();
}

LogicalResult DelayOp::verify() {
  if (getDurationNs().convertToDouble() < 0.0)
    return emitOpError("duration_ns must be nonnegative");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("delay must preserve state types");
  return success();
}

LogicalResult ProduceResourceOp::verify() {
  if (getResource().getType().getKind() != getResourceKindAttr())
    return emitOpError("resource_kind must match the result resource type");
  return success();
}

static ProduceResourceOp resolveConcreteProducer(Value value) {
  llvm::DenseSet<Value> visited;
  while (visited.insert(value).second) {
    if (auto producer = value.getDefiningOp<ProduceResourceOp>())
      return producer;
    if (auto argument = dyn_cast<BlockArgument>(value)) {
      auto call = dyn_cast_or_null<CallOp>(argument.getOwner()->getParentOp());
      if (!call || argument.getArgNumber() >= call.getInputs().size())
        return {};
      value = call.getInputs()[argument.getArgNumber()];
      continue;
    }
    auto result = dyn_cast<OpResult>(value);
    auto call = value.getDefiningOp<CallOp>();
    if (!result || !call || call.getBody().empty())
      return {};
    auto yield = dyn_cast<YieldOp>(call.getBody().front().getTerminator());
    if (!yield || result.getResultNumber() >= yield.getNumOperands())
      return {};
    value = yield.getOperand(result.getResultNumber());
  }
  return {};
}

static CallOp resolveEnclosingProducerCall(ProduceResourceOp producer) {
  auto protocol = dyn_cast<FlatSymbolRefAttr>(producer.getProtocolAttr());
  if (!protocol)
    return {};
  CallOp matched;
  unsigned matches = 0;
  for (Operation *parent = producer->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto call = dyn_cast<CallOp>(parent);
    if (!call || call.getCalleeAttr() != protocol)
      continue;
    ++matches;
    if (matches == 1)
      matched = call;
  }
  return matches == 1 ? matched : CallOp();
}

static bool flowsDirectlyToUnpack(Value value,
                                  CallOp *selectedConsumer = nullptr) {
  llvm::DenseSet<Value> visited;
  while (visited.insert(value).second) {
    if (!value.hasOneUse())
      return false;
    OpOperand &use = *value.use_begin();
    Operation *owner = use.getOwner();
    if (isa<UnpackResourceOp>(owner))
      return use.getOperandNumber() == 0;
    auto call = dyn_cast<CallOp>(owner);
    if (call) {
      bool hasResourceSite = call->hasAttr("resource_action_site");
      bool hasResourceObjective = call->hasAttr("resource_objective");
      if (hasResourceSite != hasResourceObjective)
        return false;
      if (selectedConsumer && hasResourceSite) {
        if (*selectedConsumer)
          return false;
        *selectedConsumer = call;
      }
      if (use.getOperandNumber() >= call.getBody().front().getNumArguments())
        return false;
      value = call.getBody().front().getArgument(use.getOperandNumber());
      continue;
    }
    auto yield = dyn_cast<YieldOp>(owner);
    auto parent =
        yield ? dyn_cast_or_null<CallOp>(yield->getParentOp()) : CallOp{};
    if (!parent || use.getOperandNumber() >= parent.getNumResults())
      return false;
    value = parent.getResult(use.getOperandNumber());
  }
  return false;
}

static std::string selectedProtocolDigest(qlx::fabric::ProtocolOp source) {
  Operation *clone = source->clone();
  llvm::scope_exit cleanup([&] { clone->destroy(); });
  clone->removeAttr("component_source_sha256");
  clone->removeAttr("component_objective_sha256");
  clone->removeAttr("component_boundary_sha256");
  std::string text;
  llvm::raw_string_ostream stream(text);
  clone->print(stream);
  stream.flush();
  llvm::SHA256 digest;
  digest.update(text);
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

LogicalResult TransportModelOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  SymbolTable symbols(module);
  auto architecture = symbols.lookup<ArchitectureOp>(getArchitecture());
  if (!architecture)
    return emitOpError("architecture must resolve to phys.machine");
  auto point = symbols.lookup<OperatingPointOp>(getOperatingPoint());
  if (!point || point.getMachineAttr() != getArchitectureAttr())
    return emitOpError(
        "operating_point must resolve for the selected architecture");
  double latency = getLatencyNs().convertToDouble();
  double interval = getInitiationIntervalNs().convertToDouble();
  if (!std::isfinite(latency) || latency <= 0.0 || !std::isfinite(interval) ||
      interval <= 0.0)
    return emitOpError(
        "latency and initiation interval must be finite and positive");
  if (getIntervalSemantics() != "pipelined" &&
      getIntervalSemantics() != "backpressured")
    return emitOpError(
        "interval_semantics must be 'pipelined' or 'backpressured'");
  if (interval > latency && getIntervalSemantics() != "backpressured")
    return emitOpError(
        "initiation interval greater than one-shot latency requires "
        "backpressured semantics");
  if (getPolicy() != "guaranteed" && getPolicy() != "single_shot")
    return emitOpError("policy must be 'guaranteed' or 'single_shot'");
  if (getProvider().empty() || getProviderVersion().empty() ||
      getEvidence().empty())
    return emitOpError(
        "provider, provider_version, and evidence must be nonempty");

  auto bindingReference = getQecBindingAttr();
  if (bindingReference.getRootReference().getValue() != getArchitecture() ||
      bindingReference.getNestedReferences().size() != 1)
    return emitOpError(
        "qec_binding must be nested directly in the selected architecture");
  auto binding = SymbolTable(architecture)
                     .lookup<QECChannelBindingOp>(
                         bindingReference.getLeafReference().getValue());
  if (!binding)
    return emitOpError("qec_binding must resolve to phys.qec_channel_binding");
  if (binding.getQecChannelAttr() != getQecChannelAttr())
    return emitOpError(
        "qec_channel must equal the selected physical binding's channel");
  auto qecReference = getQecChannelAttr();
  Operation *qecMachine =
      SymbolTable::lookupSymbolIn(module, qecReference.getRootReference());
  auto interconnect =
      qecMachine ? dyn_cast_or_null<qlx::fabric::InterconnectOp>(
                       SymbolTable(qecMachine)
                           .lookup(qecReference.getLeafReference().getValue()))
                 : qlx::fabric::InterconnectOp{};
  if (!interconnect || !interconnect.getProtocolAttr() ||
      interconnect.getProtocolAttr() != getProtocolAttr())
    return emitOpError(
        "protocol must equal the selected typed interconnect protocol");
  auto protocol = symbols.lookup<qlx::fabric::ProtocolOp>(getProtocol());
  if (!protocol || protocol->getAttrOfType<StringAttr>(
                       "component_source_sha256") != getProtocolSha256Attr())
    return emitOpError(
        "protocol_sha256 must equal the exact retained selected protocol");
  int64_t sourceOccupancy = getSourceEndpointOccupancy();
  int64_t destinationOccupancy = getDestinationEndpointOccupancy();
  if (sourceOccupancy <= 0 || destinationOccupancy <= 0 ||
      sourceOccupancy > interconnect.getPortAConcurrency() ||
      destinationOccupancy > interconnect.getPortBConcurrency() ||
      std::max(sourceOccupancy, destinationOccupancy) >
          interconnect.getConcurrency())
    return emitOpError(
        "endpoint occupancy must fit the selected channel and ports");

  llvm::SmallDenseSet<StringRef, 8> bound;
  for (Attribute raw : binding.getResources())
    if (auto reference = dyn_cast<FlatSymbolRefAttr>(raw))
      bound.insert(reference.getValue());
  if (getResourceClaims().empty())
    return emitOpError("resource_claims must be nonempty");
  llvm::StringMap<SmallVector<std::pair<int64_t, int64_t>, 2>> intervals;
  SmallVector<std::tuple<std::string, int64_t, int64_t, int64_t>, 4>
      modelClaims;
  for (Attribute raw : getResourceClaims()) {
    auto claim = dyn_cast<DictionaryAttr>(raw);
    auto reference =
        claim ? claim.getAs<SymbolRefAttr>("resource_class") : SymbolRefAttr{};
    auto offset = claim ? claim.getAs<IntegerAttr>("offset") : IntegerAttr{};
    auto count = claim ? claim.getAs<IntegerAttr>("count") : IntegerAttr{};
    auto units = claim ? claim.getAs<IntegerAttr>("units") : IntegerAttr{};
    if (!reference || !offset || !count || !units || claim.size() != 4)
      return emitOpError(
          "resource_claims must contain exact typed resource_class, offset, "
          "count, and units records");
    if (reference.getRootReference().getValue() != getArchitecture() ||
        reference.getNestedReferences().size() != 1 ||
        !bound.contains(reference.getLeafReference().getValue()))
      return emitOpError(
          "transport resources must belong to the selected P3 binding");
    auto resource =
        SymbolTable(architecture)
            .lookup<ResourceClassOp>(reference.getLeafReference().getValue());
    int64_t begin = offset.getInt();
    int64_t width = count.getInt();
    int64_t acquired = units.getInt();
    if (!resource || begin < 0 || width <= 0 || acquired <= 0 ||
        acquired > width || begin > resource.getCount() ||
        width > resource.getCount() - begin)
      return emitOpError(
          "transport resource slice/acquisition must fit its physical class");
    int64_t end = begin + width;
    auto &selected = intervals[reference.getLeafReference().getValue()];
    for (auto [otherBegin, otherEnd] : selected)
      if (begin < otherEnd && otherBegin < end)
        return emitOpError("resource_claims must not overlap");
    selected.push_back({begin, end});
    modelClaims.emplace_back(reference.getLeafReference().getValue().str(),
                             begin, width, acquired);
  }
  if (auto detailed = binding.getTransportClaimsAttr()) {
    if (binding.getSourceEndpointOccupancyAttr().getInt() != sourceOccupancy ||
        binding.getDestinationEndpointOccupancyAttr().getInt() !=
            destinationOccupancy)
      return emitOpError(
          "transport model endpoint occupancy differs from its detailed P3 "
          "binding");
    SmallVector<std::tuple<std::string, int64_t, int64_t, int64_t>, 4>
        detailedClaims;
    for (Attribute raw : detailed) {
      auto claim = cast<DictionaryAttr>(raw);
      detailedClaims.emplace_back(
          claim.getAs<FlatSymbolRefAttr>("resource_class").getValue().str(),
          claim.getAs<IntegerAttr>("offset").getInt(),
          claim.getAs<IntegerAttr>("count").getInt(),
          claim.getAs<IntegerAttr>("units").getInt());
    }
    if (modelClaims != detailedClaims)
      return emitOpError(
          "transport model claims differ from its detailed P3 binding");
  }

  auto digest = [&](StringRef value, bool prefixed,
                    StringRef name) -> LogicalResult {
    StringRef payload = value;
    if (prefixed && !payload.consume_front("sha256:"))
      payload = {};
    if (payload.size() != 64 || !llvm::all_of(payload, [](char character) {
          return (character >= '0' && character <= '9') ||
                 (character >= 'a' && character <= 'f');
        }))
      return emitOpError()
             << name << " must be a canonical lowercase SHA-256 commitment";
    return success();
  };
  if (failed(digest(getChannelSha256(), true, "channel_sha256")) ||
      failed(digest(getRealizationSha256(), true, "realization_sha256")) ||
      failed(digest(getProtocolSha256(), true, "protocol_sha256")) ||
      failed(digest(getBindingSha256(), true, "binding_sha256")) ||
      failed(digest(getArchitectureSha256(), true, "architecture_sha256")) ||
      failed(digest(getModelSha256(), false, "model_sha256")))
    return failure();
  bool hasBuild = static_cast<bool>(getSourceBuildSha256Attr());
  bool hasSchedule = static_cast<bool>(getSourceScheduleSha256Attr());
  bool hasSelected = static_cast<bool>(getSelectedProtocolSha256Attr());
  if (hasBuild != hasSchedule || hasBuild != hasSelected)
    return emitOpError(
        "compiler source protocol, Build, and schedule commitments must appear "
        "together");
  if (hasBuild && (failed(digest(getSourceBuildSha256Attr().getValue(), false,
                                 "source_build_sha256")) ||
                   failed(digest(getSourceScheduleSha256Attr().getValue(),
                                 false, "source_schedule_sha256")) ||
                   failed(digest(getSelectedProtocolSha256Attr().getValue(),
                                 true, "selected_protocol_sha256"))))
    return failure();
  if ((getProvider() == "qlx.compiler.transport") != hasBuild)
    return emitOpError(
        "compiler transport authority requires complete source commitments");
  if (hasBuild) {
    std::string selectedDigest = selectedProtocolDigest(protocol);
    if (selectedDigest != getSelectedProtocolSha256Attr().getValue())
      return emitOpError("transport selected P2 protocol differs from its "
                         "characterized compiler source; computed ")
             << selectedDigest;
  }
  if (getTimingSourceAttr() != point.getTimingSourceAttr())
    return emitOpError(
        "timing_source must equal the selected operating point source");
  auto number = [](Attribute attribute) -> std::optional<double> {
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
  };
  DictionaryAttr selectedTiming = point.getTimingAttr();
  size_t consequentialTimingFacts = 0;
  if (selectedTiming)
    for (NamedAttribute item : selectedTiming)
      if (item.getName() == "cycle_ns" ||
          item.getName().getValue().ends_with("_ns"))
        ++consequentialTimingFacts;
  if (getTimingProfile().size() != consequentialTimingFacts)
    return emitOpError(
        "timing_profile must exactly cover the selected operating-point "
        "timing facts");
  for (NamedAttribute item : getTimingProfile()) {
    Attribute expected =
        selectedTiming ? selectedTiming.get(item.getName()) : Attribute{};
    auto selected = number(expected);
    auto retained = number(item.getValue());
    if (!selected || !retained || *selected != *retained)
      return emitOpError("timing_profile fact '")
             << item.getName() << "' differs from the operating point";
  }
  if (getProvider() == "qlx.compiler.transport" && !getModelCommitmentAttr())
    return emitOpError(
        "compiler-characterized transport requires a replayable model "
        "commitment");
  if (auto commitment = getModelCommitmentAttr()) {
    if (sha256Hex(commitment.getValue()) != getModelSha256())
      return emitOpError("model_sha256 does not authenticate model_commitment");
    auto parsed = llvm::json::parse(commitment.getValue());
    const llvm::json::Object *root = parsed ? parsed->getAsObject() : nullptr;
    if (!root || root->size() != 7 ||
        root->getString("schema") != "qlx.transport-model/v2")
      return emitOpError(
          "model_commitment must be canonical qlx.transport-model/v2 JSON");
    auto committedLatency = root->getNumber("latency_cycles");
    auto committedInterval = root->getNumber("initiation_interval_cycles");
    auto committedSemantics = root->getString("interval_semantics");
    auto committedPolicy = root->getString("policy");
    auto committedResources = root->getArray("resources");
    auto committedOccupancy = root->getArray("endpoint_occupancy");
    Attribute cycleRaw =
        selectedTiming ? selectedTiming.get("surface_cycle_ns") : Attribute{};
    if (!cycleRaw && selectedTiming)
      cycleRaw = selectedTiming.get("cycle_ns");
    auto cycle = number(cycleRaw);
    if (!committedLatency || !committedInterval || !committedSemantics ||
        !committedPolicy || !committedResources || !committedOccupancy ||
        !cycle || !std::isfinite(*cycle) || *cycle <= 0.0 ||
        latency != *committedLatency * *cycle ||
        interval != *committedInterval * *cycle ||
        getIntervalSemantics() != *committedSemantics ||
        getPolicy() != *committedPolicy ||
        committedResources->size() != getResourceClaims().size() ||
        committedOccupancy->size() != 2 ||
        (*committedOccupancy)[0].getAsInteger() != sourceOccupancy ||
        (*committedOccupancy)[1].getAsInteger() != destinationOccupancy)
      return emitOpError("transport model facts differ from model_commitment");
    for (auto [raw, committed] :
         llvm::zip(getResourceClaims(), *committedResources)) {
      auto claim = cast<DictionaryAttr>(raw);
      auto reference = claim.getAs<SymbolRefAttr>("resource_class");
      auto resource =
          SymbolTable(architecture)
              .lookup<ResourceClassOp>(reference.getLeafReference().getValue());
      auto *record = committed.getAsObject();
      if (!resource || !record || record->size() != 7 ||
          record->getString("resource") != resource.getSymName() ||
          record->getString("kind") != resource.getKind() ||
          record->getInteger("class_count") != resource.getCount() ||
          record->getString("granularity") != resource.getGranularity() ||
          record->getInteger("offset") !=
              claim.getAs<IntegerAttr>("offset").getInt() ||
          record->getInteger("count") !=
              claim.getAs<IntegerAttr>("count").getInt() ||
          record->getInteger("units") !=
              claim.getAs<IntegerAttr>("units").getInt())
        return emitOpError(
            "transport resource claims differ from model_commitment");
    }
    if (hasBuild) {
      std::string expectedEvidence =
          ("computation:qlx.transport-characterization/v1:" +
           getSourceBuildSha256Attr().getValue() + ":" +
           getSourceScheduleSha256Attr().getValue() + ":" + getModelSha256())
              .str();
      if (getEvidence() != expectedEvidence)
        return emitOpError(
            "compiler transport evidence differs from its exact source/model "
            "commitments");
    }
  }
  return success();
}

LogicalResult TransportResourceOp::verify() {
  if (getResource().getType() != getResult().getType())
    return emitOpError("transport must preserve the resource payload type");
  if (getModelAttr() && !getRouteAttr()) {
    auto module = (*this)->getParentOfType<ModuleOp>();
    auto model = module ? SymbolTable(module).lookup<TransportModelOp>(
                              getModelAttr().getValue())
                        : TransportModelOp{};
    auto graph = (*this)->getParentOfType<GraphOp>();
    if (!model)
      return emitOpError("model must resolve to phys.transport_model");
    if (!graph || graph.getArchitectureAttr() != model.getArchitectureAttr() ||
        graph.getOperatingPointAttr() != model.getOperatingPointAttr())
      return emitOpError(
          "transport model must use the graph architecture and operating "
          "point");
    if (getProtocolAttr() != model.getProtocolAttr())
      return emitOpError(
          "transport protocol must equal its compact model protocol");
    auto qecReference = model.getQecChannelAttr();
    Operation *qecMachine =
        module ? SymbolTable::lookupSymbolIn(module,
                                             qecReference.getRootReference())
               : nullptr;
    auto interconnect =
        qecMachine
            ? dyn_cast_or_null<qlx::fabric::InterconnectOp>(
                  SymbolTable(qecMachine)
                      .lookup(qecReference.getLeafReference().getValue()))
            : qlx::fabric::InterconnectOp{};
    if (!interconnect || getSourceAttr() != interconnect.getRegionAAttr() ||
        getDestinationAttr() != interconnect.getRegionBAttr())
      return emitOpError(
          "transport endpoints must equal its compact model channel");
    if ((*this)->hasAttr("duration_ns"))
      return emitOpError(
          "raw duration metadata cannot override a typed transport model");
  }
  if (auto route = getRouteAttr()) {
    auto module = (*this)->getParentOfType<ModuleOp>();
    Operation *physicalMachine =
        module ? SymbolTable::lookupSymbolIn(module, route.getRootReference())
               : nullptr;
    Operation *routeBinding =
        physicalMachine ? SymbolTable(physicalMachine)
                              .lookup(route.getLeafReference().getValue())
                        : nullptr;
    auto binding = dyn_cast_or_null<QECChannelBindingOp>(routeBinding);
    if (!binding)
      return emitOpError(
          "route must resolve to the selected phys.qec_channel_binding");
    if (auto modelReference = getModelAttr()) {
      auto model = module ? SymbolTable(module).lookup<TransportModelOp>(
                                modelReference.getValue())
                          : TransportModelOp{};
      if (!model)
        return emitOpError("model must resolve to phys.transport_model");
      if (model.getQecBindingAttr() != route ||
          model.getProtocolAttr() != getProtocolAttr())
        return emitOpError(
            "transport model must refine this exact route and protocol");
      auto graph = (*this)->getParentOfType<GraphOp>();
      if (!graph ||
          graph.getArchitectureAttr() != model.getArchitectureAttr() ||
          graph.getOperatingPointAttr() != model.getOperatingPointAttr())
        return emitOpError("transport model must use the graph architecture "
                           "and operating point");
      if ((*this)->hasAttr("duration_ns"))
        return emitOpError(
            "raw duration metadata cannot override a typed transport model");
    }

    auto qecReference = binding.getQecChannelAttr();
    Operation *qecMachine =
        module ? SymbolTable::lookupSymbolIn(module,
                                             qecReference.getRootReference())
               : nullptr;
    Operation *qecTarget =
        qecMachine ? SymbolTable(qecMachine)
                         .lookup(qecReference.getLeafReference().getValue())
                   : nullptr;
    auto interconnect =
        dyn_cast_or_null<qlx::fabric::InterconnectOp>(qecTarget);
    if (!interconnect)
      return emitOpError(
          "physical route must bind one selected fabric.interconnect");
    if (getSourceAttr() != interconnect.getRegionAAttr() ||
        getDestinationAttr() != interconnect.getRegionBAttr())
      return emitOpError(
          "physical transport endpoints must match the selected QEC route");
    if (!interconnect.getProtocolAttr() ||
        getProtocolAttr() != interconnect.getProtocolAttr())
      return emitOpError(
          "physical transport protocol must match the selected QEC route");

    auto producer = resolveConcreteProducer(getResource());
    if (!producer)
      return emitOpError(
          "routed physical transport must consume a concrete producer result");
    if (producer.getRegionAttr() != getSourceAttr())
      return emitOpError(
          "physical producer region must equal the selected route source");
    if (producer.getResourceKindAttr() != getResource().getType().getKind())
      return emitOpError(
          "physical producer kind must match the transported resource kind");
    CallOp selectedConsumer;
    if (!flowsDirectlyToUnpack(getResult(), &selectedConsumer))
      return emitOpError(
          "routed physical resource must flow directly to one unpack consumer");
    if (!resolveEnclosingProducerCall(producer))
      return emitOpError(
          "resource producer call must exactly match the concrete producer "
          "protocol");
    if (!selectedConsumer)
      return emitOpError(
          "physical transport must flow through one selected resource "
          "consumer call");
  }
  return success();
}

static std::optional<unsigned> builtinActionArity(qlx::BuiltinAction action) {
  switch (action) {
  case qlx::BuiltinAction::cx:
  case qlx::BuiltinAction::cz:
    return 2;
  case qlx::BuiltinAction::ccz:
  case qlx::BuiltinAction::ccx:
    return 3;
  case qlx::BuiltinAction::pauli_rotation:
    return std::nullopt;
  default:
    return 1;
  }
}

static LogicalResult verifyPhysicalPayloadRoles(Operation *owner,
                                                ArrayAttr roles,
                                                int64_t payloadCount) {
  if (!roles)
    return success();
  if (static_cast<int64_t>(roles.size()) != payloadCount)
    return owner->emitOpError(
        "payload_roles must exactly cover the ordered payloads");
  llvm::StringSet<> unique;
  for (Attribute raw : roles) {
    auto role = dyn_cast<StringAttr>(raw);
    if (!role || role.getValue().empty())
      return owner->emitOpError("payload_roles must contain nonempty strings");
    if (!unique.insert(role.getValue()).second)
      return owner->emitOpError("payload_roles must be unique");
  }
  return success();
}

LogicalResult UnpackResourceOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one destination carrier");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve destination carrier state types");
  auto segments = getPayloadCarrierSegmentsAttr();
  qlx::fabric::EncodingOp encoding;
  if (segments) {
    if (segments.size() < 2 || segments.asArrayRef().front() != 0 ||
        segments.asArrayRef().back() !=
            static_cast<int64_t>(getInputs().size()))
      return emitOpError(
          "payload carrier segments must start at zero and cover all carriers");
    for (auto pair : llvm::zip(segments.asArrayRef().drop_back(),
                               segments.asArrayRef().drop_front()))
      if (std::get<0>(pair) >= std::get<1>(pair))
        return emitOpError(
            "payload carrier segments must be strictly increasing");

    auto *encodingTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, getEncodingAttr());
    encoding = dyn_cast_or_null<qlx::fabric::EncodingOp>(encodingTarget);
    if (!encoding)
      return emitOpError("encoding must resolve to fabric.encoding");
    auto *codeTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, encoding.getCodeAttr());
    auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(codeTarget);
    if (!code)
      return emitOpError(
          "encoding code must resolve unambiguously to fabric.code");
    int64_t carrierWidth = 0;
    for (NamedAttribute partition : code.getPartitions()) {
      auto size = dyn_cast<IntegerAttr>(partition.getValue());
      if (!size || size.getInt() < 0)
        return emitOpError(
            "encoding code partitions must define nonnegative carrier widths");
      if (size.getInt() > std::numeric_limits<int64_t>::max() - carrierWidth)
        return emitOpError(
            "encoding code carrier count exceeds the supported signed i64 "
            "range");
      carrierWidth += size.getInt();
    }
    if (carrierWidth <= 0)
      return emitOpError(
          "encoding code must declare a positive canonical carrier count");
    auto segmentWidth = [&](int64_t begin, int64_t end) {
      int64_t expected = carrierWidth;
      if (end - begin != 1)
        return expected;
      auto state = dyn_cast<StateType>(getInputs()[begin].getType());
      auto resource = state ? dyn_cast_or_null<ResourceOp>(
                                  SymbolTable::lookupNearestSymbolFrom(
                                      *this, state.getResource()))
                            : ResourceOp{};
      auto graph = (*this)->getParentOfType<GraphOp>();
      auto architecture = graph ? dyn_cast_or_null<ArchitectureOp>(
                                      SymbolTable::lookupNearestSymbolFrom(
                                          graph, graph.getArchitectureAttr()))
                                : ArchitectureOp{};
      auto resourceClass =
          resource && architecture
              ? dyn_cast_or_null<ResourceClassOp>(
                    SymbolTable(architecture)
                        .lookup(resource.getResourceClassAttr().getAttr()))
              : ResourceClassOp{};
      auto granularity =
          resourceClass
              ? resourceClass->getAttrOfType<StringAttr>("granularity")
              : StringAttr{};
      if (granularity && granularity.getValue() == "patch")
        expected = 1;
      return expected;
    };
    for (auto pair : llvm::zip(segments.asArrayRef().drop_back(),
                               segments.asArrayRef().drop_front()))
      if (std::get<1>(pair) - std::get<0>(pair) !=
          segmentWidth(std::get<0>(pair), std::get<1>(pair)))
        return emitOpError(
            "every payload segment width must match its resolved physical "
            "resource granularity");
  }
  auto blocks = getPayloadLogicalBlocksAttr();
  auto ports = getPayloadLogicalPortsAttr();
  auto blockIDs = getPayloadLogicalBlockIdsAttr();
  auto action = getPayloadActionAttr();
  if (static_cast<bool>(blocks) != static_cast<bool>(ports))
    return emitOpError(
        "payload logical block and port maps must appear together");
  if (blocks && !segments)
    return emitOpError("payload logical maps require payload carrier segments");
  if (blocks && (blocks.size() == 0 || blocks.size() != ports.size()))
    return emitOpError(
        "payload logical block and port maps must have equal nonzero length");
  int64_t payloadCount =
      segments ? static_cast<int64_t>(segments.size()) - 1 : 1;
  if (failed(verifyPhysicalPayloadRoles(*this, getPayloadRolesAttr(),
                                        payloadCount)))
    return failure();
  if (payloadCount > 1 && (!blocks || !action))
    return emitOpError(
        "multi-payload unpack requires an action and exact logical maps");
  if (blocks && !action)
    return emitOpError("payload logical maps require a typed payload_action");
  if (action && action.getValue() == qlx::BuiltinAction::ccz && !blockIDs) {
    // A generated action realization must authenticate the selected P1 owners.
    // Ordinary resource-producing protocols may also unpack a CCZ internally;
    // those payloads have no selected action site and therefore no P1 block
    // identities to retain.
    bool selectedActionBoundary = true;
    if (auto physicalCall = (*this)->getParentOfType<CallOp>()) {
      auto source = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
          SymbolTable::lookupNearestSymbolFrom(*this,
                                               physicalCall.getCalleeAttr()));
      selectedActionBoundary =
          source && source.getGeneratedByAttr() && source.getActionSiteAttr();
    } else if (auto graph = (*this)->getParentOfType<GraphOp>()) {
      auto sourceRef =
          graph->getAttrOfType<FlatSymbolRefAttr>("source_protocol");
      auto source =
          sourceRef
              ? dyn_cast_or_null<qlx::fabric::ProtocolOp>(
                    SymbolTable::lookupNearestSymbolFrom(graph, sourceRef))
              : qlx::fabric::ProtocolOp{};
      selectedActionBoundary =
          source && source.getGeneratedByAttr() && source.getActionSiteAttr();
    }
    if (selectedActionBoundary)
      return emitOpError(
          "selected CCZ payload handoff requires exact QEC block identities");
  }
  if (blockIDs && (!blocks || blockIDs.size() != blocks.size()))
    return emitOpError(
        "payload logical block identities must exactly cover the logical map");
  if (blocks && segments) {
    auto arity = builtinActionArity(action.getValue());
    if (!arity)
      return emitOpError("payload_action must have a fixed logical arity");
    if (blocks.size() != *arity)
      return emitOpError(
          "payload logical maps must exactly cover the action arity");
    llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 8> mappings;
    llvm::SmallDenseSet<int64_t, 8> coveredBlocks;
    llvm::StringMap<int64_t> indexByBlockID;
    SmallVector<StringRef, 8> blockIDByIndex(payloadCount);
    auto rawBlockIDs = blockIDs ? blockIDs.getValue() : ArrayRef<Attribute>{};
    int64_t logicalCapacity = encoding.getLogicalPorts().size();
    for (auto [ordinal, pair] :
         llvm::enumerate(llvm::zip(blocks.asArrayRef(), ports.asArrayRef()))) {
      auto [block, port] = pair;
      if (block < 0 || block >= payloadCount)
        return emitOpError("payload logical block index is out of range");
      if (port < 0)
        return emitOpError("payload logical port index must be nonnegative");
      if (port >= logicalCapacity)
        return emitOpError(
            "payload logical port index exceeds the encoding logical capacity");
      if (!mappings.insert({block, port}).second)
        return emitOpError(
            "payload logical mapping must not alias a payload port");
      coveredBlocks.insert(block);
      if (blockIDs) {
        auto blockID = dyn_cast<StringAttr>(rawBlockIDs[ordinal]);
        if (!blockID || blockID.getValue().empty())
          return emitOpError(
              "payload logical block identities must be nonempty strings");
        auto [entry, inserted] =
            indexByBlockID.try_emplace(blockID.getValue(), block);
        if (!inserted && entry->second != block)
          return emitOpError(
              "one selected QEC block identity maps to several payload blocks");
        if (blockIDByIndex[block].empty())
          blockIDByIndex[block] = blockID.getValue();
        else if (blockIDByIndex[block] != blockID.getValue())
          return emitOpError("one payload block maps to several selected QEC "
                             "block identities");
      }
    }
    if (!getPayloadRolesAttr() &&
        static_cast<int64_t>(coveredBlocks.size()) != payloadCount)
      return emitOpError(
          "payload logical maps must cover every required payload block");
    if (blockIDs && indexByBlockID.size() != coveredBlocks.size())
      return emitOpError("payload logical block identities must name every "
                         "mapped payload block exactly");
  }
  return success();
}

LogicalResult PackResourceOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one encoded payload carrier");
  if (getResource().getType().getKind() != getResourceKindAttr())
    return emitOpError("resource_kind must match the result resource type");
  int64_t payloadCount = 1;
  if (auto segments = getPayloadCarrierSegmentsAttr()) {
    if (segments.size() < 2 || segments.asArrayRef().front() != 0 ||
        segments.asArrayRef().back() !=
            static_cast<int64_t>(getInputs().size()))
      return emitOpError(
          "payload carrier segments must start at zero and cover all carriers");
    for (auto pair : llvm::zip(segments.asArrayRef().drop_back(),
                               segments.asArrayRef().drop_front()))
      if (std::get<0>(pair) >= std::get<1>(pair))
        return emitOpError(
            "payload carrier segments must be strictly increasing");
    payloadCount = static_cast<int64_t>(segments.size()) - 1;
  }
  if (failed(verifyPhysicalPayloadRoles(*this, getPayloadRolesAttr(),
                                        payloadCount)))
    return failure();
  return success();
}

static qlx::fabric::RegionOp lookupUniqueResourceRegion(Operation *from,
                                                        StringRef name) {
  auto module = from->getParentOfType<ModuleOp>();
  qlx::fabric::RegionOp matched;
  unsigned matches = 0;
  if (!module)
    return matched;
  // fabric.region is structurally required to be a direct child of one
  // fabric.machine.  Walking the complete linked P2+P3 module for every
  // physical resource request made verification quadratic in the number of
  // requests and unrelated graph operations.
  for (auto machine : module.getOps<qlx::fabric::DeviceOp>())
    for (auto candidate : machine.getBody().getOps<qlx::fabric::RegionOp>()) {
      if (candidate.getSymName() != name)
        continue;
      ++matches;
      if (matches == 1)
        matched = candidate;
    }
  return matches == 1 ? matched : qlx::fabric::RegionOp();
}

template <typename MemberOp>
static MemberOp lookupLogicalDomainMember(Operation *from,
                                          SymbolRefAttr reference) {
  auto module = from->getParentOfType<ModuleOp>();
  if (!module || reference.getNestedReferences().size() != 1)
    return dyn_cast_or_null<MemberOp>(
        SymbolTable::lookupNearestSymbolFrom(from, reference));

  bool foundDomain = false;
  for (auto domain : module.getOps<qlx::lvm::DomainOp>()) {
    if (domain.getSymName() != reference.getRootReference().getValue())
      continue;
    foundDomain = true;
    for (auto member : domain.getBody().template getOps<MemberOp>())
      if (member.getSymName() == reference.getLeafReference().getValue())
        return member;
  }
  // Retain generic nearest-symbol behavior for detached/noncanonical nesting;
  // a present top-level domain with no such direct member is an exact miss.
  return foundDomain
             ? MemberOp{}
             : dyn_cast_or_null<MemberOp>(
                   SymbolTable::lookupNearestSymbolFrom(from, reference));
}

LogicalResult FactoryModelOp::verify() {
  if (getLaneCount() <= 0 || getBufferCapacity() != 1 ||
      getPhysicalUnits() <= 0)
    return emitOpError("requires positive lanes/physical_units and "
                       "buffer_capacity exactly one");
  double startup = getStartupNs().convertToDouble();
  double interval = getOutputIntervalNs().convertToDouble();
  if (!std::isfinite(startup) || !std::isfinite(interval) || startup <= 0.0 ||
      interval <= 0.0 || interval > startup)
    return emitOpError(
        "startup/output interval must be finite positive values with "
        "output interval no greater than startup");
  if (getPolicy() != "guaranteed" && getPolicy() != "single_shot")
    return emitOpError(
        "factory model policy must be guaranteed or single_shot");
  if (getEvidence().empty())
    return emitOpError("evidence must be nonempty");
  StringRef digest = getProviderSha256();
  if (!digest.consume_front("sha256:") || digest.size() != 64 ||
      !llvm::all_of(digest, [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
      }))
    return emitOpError(
        "provider_sha256 must be a canonical lowercase sha256 commitment");

  auto stream =
      lookupLogicalDomainMember<qlx::lvm::StreamOp>(*this, getStreamAttr());
  if (!stream || !stream.getProducedByAttr() ||
      stream.getProducedByAttr().getValue() != getProviderAttr().getValue() ||
      !stream.getBackingRegionAttr() ||
      stream.getBackingRegionAttr().getValue() !=
          getRegionAttr().getLeafReference().getValue() ||
      !stream.getCapacityAttr() ||
      stream.getCapacityAttr().getInt() != getBufferCapacity())
    return emitOpError(
        "stream/provider/region/buffer must exactly match the retained backed "
        "P1 stream");
  auto retainedDigest = stream->getAttrOfType<StringAttr>("produced_by_sha256");
  if (!retainedDigest || retainedDigest.getValue() != getProviderSha256())
    return emitOpError(
        "provider_sha256 must equal the retained P1 producer commitment");
  auto provider = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getProviderAttr()));
  if (!provider)
    return emitOpError("provider must resolve to a retained fabric.protocol");
  auto region = dyn_cast_or_null<qlx::lvm::SpaceOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getRegionAttr()));
  if (!region || !region.getCapacityAttr() ||
      region.getCapacityAttr().getInt() != getLaneCount())
    return emitOpError(
        "lane_count must equal the retained factory-region capacity");

  auto point = dyn_cast_or_null<OperatingPointOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getOperatingPointAttr()));
  if (!point)
    return emitOpError("operating_point must resolve to phys.operating_point");
  auto machine = dyn_cast_or_null<ArchitectureOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, point.getMachineAttr()));
  if (!machine)
    return emitOpError("operating point must resolve its physical machine");
  auto qecBinding = dyn_cast_or_null<QECBindingOp>(
      SymbolTable(machine).lookup(getQecBindingAttr().getValue()));
  if (!qecBinding)
    return emitOpError(
        "qec_binding must resolve inside the selected physical machine");
  auto resourceClass = dyn_cast_or_null<ResourceClassOp>(
      SymbolTable(machine).lookup(getPhysicalResourceClassAttr().getValue()));
  if (!resourceClass)
    return emitOpError(
        "physical_resource_class must resolve in the selected machine");
  int64_t memberWeight = 1;
  if (auto granularity =
          resourceClass->getAttrOfType<StringAttr>("granularity");
      granularity && granularity.getValue() == "patch") {
    auto footprint =
        resourceClass->getAttrOfType<IntegerAttr>("physical_units");
    if (!footprint || footprint.getInt() <= 0)
      return emitOpError(
          "patch factory resource class lacks a positive physical footprint");
    memberWeight = footprint.getInt();
  }
  if (resourceClass.getCount() != 0 &&
      memberWeight >
          std::numeric_limits<int64_t>::max() / resourceClass.getCount())
    return emitOpError("factory physical footprint overflows i64");
  int64_t expectedPhysicalUnits = resourceClass.getCount() * memberWeight;
  if (expectedPhysicalUnits != getPhysicalUnits())
    return emitOpError(
        "physical_units must equal the dedicated factory resource-class "
        "base-unit footprint");
  if (!llvm::any_of(qecBinding.getResources(), [&](Attribute raw) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
        return reference && reference == getPhysicalResourceClassAttr();
      }))
    return emitOpError(
        "physical resource class must belong to the selected factory binding");

  auto timing = point.getTimingAttr();
  auto cycleText =
      timing ? timing.getAs<StringAttr>("surface_cycle_ns") : StringAttr{};
  double cycleNs = 0.0;
  if (!cycleText || cycleText.getValue().getAsDouble(cycleNs) ||
      !std::isfinite(cycleNs) || cycleNs <= 0.0)
    return emitOpError(
        "operating point requires finite positive surface_cycle_ns");

  auto module = (*this)->getParentOfType<ModuleOp>();
  DictionaryAttr retainedBinding;
  if (module)
    module.walk([&](Operation *candidate) {
      if (retainedBinding ||
          candidate->getName().getStringRef() != "qlx.qec_to_physical")
        return;
      auto entries = candidate->getAttrOfType<ArrayAttr>("entries");
      if (!entries)
        return;
      for (Attribute raw : entries) {
        auto entry = dyn_cast<DictionaryAttr>(raw);
        auto name = entry ? entry.getAs<StringAttr>("binding") : StringAttr{};
        if (name && name.getValue() == getQecBindingAttr().getValue()) {
          retainedBinding = entry;
          return;
        }
      }
    });
  auto startupCycles =
      retainedBinding
          ? retainedBinding.getAs<StringAttr>("factory_startup_cycles")
          : StringAttr{};
  auto intervalCycles =
      retainedBinding
          ? retainedBinding.getAs<StringAttr>("factory_output_interval_cycles")
          : StringAttr{};
  auto policy = retainedBinding
                    ? retainedBinding.getAs<StringAttr>("factory_model_policy")
                    : StringAttr{};
  auto evidence =
      retainedBinding
          ? retainedBinding.getAs<StringAttr>("factory_model_evidence")
          : StringAttr{};
  auto sourceProvider = getSourceProviderAttr();
  auto sourceProviderSha256 = getSourceProviderSha256Attr();
  auto sourceStartupCycles = getSourceStartupCyclesAttr();
  auto sourceIntervalCycles = getSourceOutputIntervalCyclesAttr();
  auto sourceBuild = getSourceBuildSha256Attr();
  auto sourceSchedule = getSourceScheduleSha256Attr();
  auto sourcePoint = getSourceOperatingPointAttr();
  auto sourceTimingSource = getSourceTimingSourceAttr();
  auto sourceTimingProfile = getSourceTimingProfileAttr();
  auto sourceOutputs = getSourceOutputEventsAttr();
  auto sourceSelections = getSourceSelectionEventsAttr();
  auto sourceUnits = getSourcePhysicalUnitsAttr();
  auto sourceUnitKind = getSourcePhysicalUnitKindAttr();
  auto sourceDistances = getSourceCodeDistancesAttr();
  auto retainedSourceBuild =
      retainedBinding.getAs<StringAttr>("factory_source_build_sha256");
  auto retainedSourceProvider =
      retainedBinding.getAs<StringAttr>("factory_source_provider");
  auto retainedSourceProviderSha256 =
      retainedBinding.getAs<StringAttr>("factory_source_provider_sha256");
  auto retainedSourceStartupCycles =
      retainedBinding.getAs<StringAttr>("factory_source_startup_cycles");
  auto retainedSourceIntervalCycles = retainedBinding.getAs<StringAttr>(
      "factory_source_output_interval_cycles");
  auto retainedSourceSchedule =
      retainedBinding.getAs<StringAttr>("factory_source_schedule_sha256");
  auto retainedSourcePoint =
      retainedBinding.getAs<StringAttr>("factory_source_operating_point");
  auto retainedSourceTimingSource =
      retainedBinding.getAs<StringAttr>("factory_source_timing_source");
  auto retainedSourceTimingProfile =
      retainedBinding.getAs<DictionaryAttr>("factory_source_timing_profile");
  auto retainedSourceOutputs =
      retainedBinding.getAs<ArrayAttr>("factory_source_output_events");
  auto retainedSourceSelections =
      retainedBinding.getAs<ArrayAttr>("factory_source_selection_events");
  auto retainedSourceUnits =
      retainedBinding.getAs<IntegerAttr>("factory_source_physical_units");
  auto retainedSourceUnitKind =
      retainedBinding.getAs<StringAttr>("factory_source_physical_unit_kind");
  auto retainedSourceDistances =
      retainedBinding.getAs<DenseI64ArrayAttr>("factory_source_code_distances");
  unsigned sourceFields = static_cast<unsigned>(bool(sourceProvider)) +
                          static_cast<unsigned>(bool(sourceProviderSha256)) +
                          static_cast<unsigned>(bool(sourceStartupCycles)) +
                          static_cast<unsigned>(bool(sourceIntervalCycles)) +
                          static_cast<unsigned>(bool(sourceBuild)) +
                          static_cast<unsigned>(bool(sourceSchedule)) +
                          static_cast<unsigned>(bool(sourcePoint)) +
                          static_cast<unsigned>(bool(sourceTimingProfile)) +
                          static_cast<unsigned>(bool(sourceOutputs)) +
                          static_cast<unsigned>(bool(sourceSelections)) +
                          static_cast<unsigned>(bool(sourceUnits)) +
                          static_cast<unsigned>(bool(sourceUnitKind)) +
                          static_cast<unsigned>(bool(sourceDistances));
  if (sourceFields != 0 && sourceFields != 13)
    return emitOpError(
        "compiled factory source evidence must appear as one complete tuple");
  unsigned retainedSourceFields =
      static_cast<unsigned>(bool(retainedSourceProvider)) +
      static_cast<unsigned>(bool(retainedSourceProviderSha256)) +
      static_cast<unsigned>(bool(retainedSourceStartupCycles)) +
      static_cast<unsigned>(bool(retainedSourceIntervalCycles)) +
      static_cast<unsigned>(bool(retainedSourceBuild)) +
      static_cast<unsigned>(bool(retainedSourceSchedule)) +
      static_cast<unsigned>(bool(retainedSourcePoint)) +
      static_cast<unsigned>(bool(retainedSourceTimingProfile)) +
      static_cast<unsigned>(bool(retainedSourceOutputs)) +
      static_cast<unsigned>(bool(retainedSourceSelections)) +
      static_cast<unsigned>(bool(retainedSourceUnits)) +
      static_cast<unsigned>(bool(retainedSourceUnitKind)) +
      static_cast<unsigned>(bool(retainedSourceDistances));
  if (retainedSourceFields != sourceFields)
    return emitOpError(
        "compiled factory source evidence must be present identically on the "
        "model and retained device binding");
  if (static_cast<bool>(retainedSourceTimingSource) !=
          static_cast<bool>(sourceTimingSource) ||
      retainedSourceTimingSource != sourceTimingSource)
    return emitOpError(
        "compiled factory timing source must match the retained device "
        "binding");
  if (retainedSourceTimingProfile != sourceTimingProfile)
    return emitOpError(
        "compiled factory timing profile must match the retained device "
        "binding");
  if (getPolicy() == "single_shot" &&
      (!sourceSelections || sourceSelections.empty()))
    return emitOpError(
        "single_shot factory model requires retained selection events");
  if (getPolicy() == "guaranteed" && sourceSelections &&
      !sourceSelections.empty())
    return emitOpError(
        "guaranteed factory model cannot retain selection events");
  if (sourceFields == 13) {
    auto canonicalDigest = [](StringRef value) {
      return value.size() == 64 && llvm::all_of(value, [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
             });
    };
    StringRef semanticsDigest = sourceProviderSha256.getValue();
    double sourceStartup = 0.0;
    double sourceInterval = 0.0;
    if (!semanticsDigest.consume_front("sha256:") ||
        !canonicalDigest(semanticsDigest) ||
        sourceProvider.getValue().empty() ||
        sourceStartupCycles.getValue().getAsDouble(sourceStartup) ||
        sourceIntervalCycles.getValue().getAsDouble(sourceInterval) ||
        !std::isfinite(sourceStartup) || !std::isfinite(sourceInterval) ||
        sourceStartup <= 0.0 || sourceInterval <= 0.0 ||
        sourceInterval > sourceStartup ||
        !canonicalDigest(sourceBuild.getValue()) ||
        !canonicalDigest(sourceSchedule.getValue()) ||
        sourcePoint.getValue().empty() || sourceOutputs.empty() ||
        sourceUnits.getInt() <= 0 || sourceUnitKind.getValue().empty() ||
        sourceDistances.empty() ||
        llvm::any_of(sourceDistances.asArrayRef(),
                     [](int64_t distance) { return distance <= 0; }) ||
        !llvm::is_sorted(sourceDistances.asArrayRef()) ||
        std::adjacent_find(sourceDistances.asArrayRef().begin(),
                           sourceDistances.asArrayRef().end()) !=
            sourceDistances.asArrayRef().end())
      return emitOpError("compiled factory source evidence is malformed");
    if (sourceTimingSource != point.getTimingSourceAttr())
      return emitOpError(
          "compiled factory timing source must match the selected operating "
          "point");
    for (NamedAttribute entry : sourceTimingProfile) {
      auto expected = dyn_cast<FloatAttr>(entry.getValue());
      if (!expected || !std::isfinite(expected.getValueAsDouble()) ||
          expected.getValueAsDouble() < 0.0)
        return emitOpError(
            "compiled factory timing profile must contain finite "
            "nonnegative f64 values");
      StringRef sourceName = entry.getName().strref();
      StringRef selectedName =
          sourceName == "cycle_ns" ? StringRef("surface_cycle_ns") : sourceName;
      auto selected = timing.getAs<StringAttr>(selectedName);
      double selectedValue = 0.0;
      if (!selected || selected.getValue().getAsDouble(selectedValue) ||
          !std::isfinite(selectedValue) ||
          selectedValue != expected.getValueAsDouble())
        return emitOpError("compiled factory timing fact '")
               << sourceName << "' does not match selected operating-point '"
               << selectedName << "'";
    }
    auto retainedIdentity =
        stream->getAttrOfType<StringAttr>("producer_identity");
    auto retainedSemantics =
        stream->getAttrOfType<StringAttr>("producer_semantics_sha256");
    if (!retainedIdentity || !retainedSemantics ||
        retainedIdentity != sourceProvider ||
        retainedSemantics != sourceProviderSha256)
      return emitOpError(
          "compiled factory source producer must match the retained P1 "
          "factory semantics commitment");
    if (sourceUnits.getInt() >
        std::numeric_limits<int64_t>::max() / getLaneCount())
      return emitOpError("compiled factory source footprint overflows i64");
    if (sourceUnits.getInt() * getLaneCount() > expectedPhysicalUnits)
      return emitOpError(
          "selected factory binding is smaller than the compiled per-lane "
          "source footprint");
    StringRef selectedUnitKind = resourceClass.getKind();
    if (auto physicalUnitKind =
            resourceClass->getAttrOfType<StringAttr>("physical_unit_kind"))
      selectedUnitKind = physicalUnitKind.getValue();
    if (sourceUnitKind.getValue() != selectedUnitKind)
      return emitOpError(
          "compiled factory source and selected binding use different "
          "physical base units");
    auto qecReference = qecBinding.getQecRegionAttr();
    Operation *selectedQEC =
        module ? SymbolTable::lookupSymbolIn(module,
                                             qecReference.getRootReference())
               : nullptr;
    auto selectedRegion =
        selectedQEC
            ? dyn_cast_or_null<qlx::fabric::RegionOp>(
                  SymbolTable(selectedQEC)
                      .lookup(qecReference.getLeafReference().getValue()))
            : qlx::fabric::RegionOp{};
    auto selectedCode =
        selectedRegion
            ? dyn_cast_or_null<qlx::fabric::CodeOp>(SymbolTable(module).lookup(
                  selectedRegion.getCodeAttr().getValue()))
            : qlx::fabric::CodeOp{};
    if (!selectedCode)
      return emitOpError(
          "selected factory binding code must resolve to fabric.code");
    if (!std::binary_search(sourceDistances.asArrayRef().begin(),
                            sourceDistances.asArrayRef().end(),
                            selectedCode.getDistance()))
      return emitOpError("compiled factory source code distances must include "
                         "the selected "
                         "factory binding code distance ")
             << selectedCode.getDistance();

    if (retainedSourceProvider != sourceProvider ||
        retainedSourceProviderSha256 != sourceProviderSha256 ||
        retainedSourceStartupCycles != sourceStartupCycles ||
        retainedSourceIntervalCycles != sourceIntervalCycles ||
        sourceStartupCycles != startupCycles ||
        sourceIntervalCycles != intervalCycles ||
        retainedSourceBuild != sourceBuild ||
        retainedSourceSchedule != sourceSchedule ||
        retainedSourcePoint != sourcePoint ||
        retainedSourceTimingProfile != sourceTimingProfile ||
        retainedSourceOutputs != sourceOutputs ||
        retainedSourceSelections != sourceSelections ||
        retainedSourceUnits != sourceUnits ||
        retainedSourceUnitKind != sourceUnitKind ||
        retainedSourceDistances != sourceDistances)
      return emitOpError(
          "compiled factory source evidence must exactly match the retained "
          "device binding");
  }
  double retainedStartup = 0.0;
  double retainedInterval = 0.0;
  if (!startupCycles || !intervalCycles || !policy || !evidence ||
      startupCycles.getValue().getAsDouble(retainedStartup) ||
      intervalCycles.getValue().getAsDouble(retainedInterval) ||
      policy.getValue() != getPolicy() ||
      evidence.getValue() != getEvidence() ||
      retainedStartup * cycleNs != startup ||
      retainedInterval * cycleNs / static_cast<double>(getLaneCount()) !=
          interval)
    return emitOpError("factory model timing/policy/evidence must derive "
                       "exactly from the retained "
                       "device binding and operating point");
  return success();
}

LogicalResult FactoryStartOp::verify() {
  auto model = dyn_cast_or_null<FactoryModelOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getFactoryModelAttr()));
  if (!model)
    return emitOpError("factory_model must resolve to phys.factory_model");
  if (getEventId().empty())
    return emitOpError("event_id must be nonempty");
  return success();
}

LogicalResult ResourceRequestOp::verify() {
  auto event = getEvent().getType();
  auto payload = dyn_cast<ResourcePayloadType>(event.getPayload());
  if (!payload)
    return emitOpError("event payload must be a physical resource payload");
  if (payload.getKind().getValue() != getKind())
    return emitOpError("resource kind must match the event payload kind");
  if (event.getOwnership() != "linear")
    return emitOpError("physical resource payloads require linear ownership");

  Attribute rawProvider = (*this)->getAttr("provider");
  Attribute rawRegion = (*this)->getAttr("region");
  Attribute rawTransfer = (*this)->getAttr("transfer");
  bool external = (*this)->hasAttr("external");
  Attribute rawConsumerSite = (*this)->getAttr("consumer_action_site");
  Attribute rawConsumerCallee = (*this)->getAttr("consumer_callee");
  Attribute rawConsumerObjective = (*this)->getAttr("consumer_objective");
  bool linked = rawProvider || rawRegion || rawTransfer ||
                getFactoryModelAttr() || getPhysicalBindingAttr() ||
                getDurationNsAttr();
  auto model = getFactoryModelAttr() ? dyn_cast_or_null<FactoryModelOp>(
                                           SymbolTable::lookupNearestSymbolFrom(
                                               *this, getFactoryModelAttr()))
                                     : FactoryModelOp();
  if (getFactoryModelAttr() && !model)
    return emitOpError("factory_model must resolve to phys.factory_model");
  auto stream = dyn_cast_or_null<qlx::lvm::StreamOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getStreamAttr()));
  if (!stream && linked)
    return emitOpError("linked request stream must resolve to an lvm.stream");
  if (stream) {
    if (stream.getProducesAttr().getValue() != getKind())
      return emitOpError("stream produces a different resource kind");
    if (stream->hasAttr("frame_domains"))
      return emitOpError("retained P1 stream must not carry a P2 frame schema");
    if (stream->hasAttr("external") != external)
      return emitOpError(
          "external boundary must exactly match the retained stream");
  }
  if (!linked)
    return success();

  auto provider = dyn_cast_or_null<SymbolRefAttr>(rawProvider);
  auto region = dyn_cast_or_null<SymbolRefAttr>(rawRegion);
  auto transfer = dyn_cast_or_null<SymbolRefAttr>(rawTransfer);
  if (!provider || (!external && !region) || (external && region) ||
      (rawTransfer && !transfer))
    return emitOpError(
        "linked backed requests require typed provider and region references; "
        "linked external requests require a typed provider and no region; "
        "either may retain an optional typed transfer reference");
  if (external && model)
    return emitOpError(
        "external resource requests cannot use a deterministic factory model");
  if (model && (model.getStreamAttr() != getStreamAttr() ||
                model.getProviderAttr().getValue() !=
                    provider.getLeafReference().getValue() ||
                model.getRegionAttr() != region ||
                model.getResourceKindAttr().getValue() != getKind()))
    return emitOpError("factory model must exactly match request "
                       "kind/stream/provider/region");
  auto retainedProducer = stream.getProducedByAttr();
  if (!retainedProducer ||
      retainedProducer.getValue() != provider.getLeafReference().getValue())
    return emitOpError(
        "provider must resolve to a typed fabric.protocol and exactly match "
        "the retained stream produced_by; provider signature is "
        "unauthenticated");
  auto providerOp = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, provider));
  if (!providerOp)
    return emitOpError("provider must resolve to a fabric.protocol");
  FunctionType providerType = providerOp.getFunctionType();
  if (providerType.getNumInputs() != 0 || providerType.getNumResults() != 1)
    return emitOpError("provider must produce exactly one resource payload");
  auto produced =
      dyn_cast<qlx::fabric::ResourceStateType>(providerType.getResult(0));
  auto producedKind =
      produced ? dyn_cast<SymbolRefAttr>(produced.getKind()) : SymbolRefAttr();
  if (!producedKind || producedKind.getLeafReference().getValue() !=
                           payload.getKind().getValue())
    return emitOpError(
        "provider signature must produce exactly this resource payload");

  if (!external) {
    auto regionOp =
        lookupUniqueResourceRegion(*this, region.getLeafReference().getValue());
    if (!regionOp)
      return emitOpError("region must resolve to a fabric.region");
    auto streamRegion = stream.getBackingRegionAttr();
    if (!streamRegion ||
        streamRegion.getValue() != region.getLeafReference().getValue())
      return emitOpError("region must match the retained stream backing");

    auto physicalBinding = getPhysicalBindingAttr();
    if (!physicalBinding)
      return emitOpError(
          "linked backed request requires a typed physical_binding");
    auto binding = dyn_cast_or_null<QECBindingOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, physicalBinding));
    if (!binding)
      return emitOpError("physical_binding must resolve to phys.qec_binding");
    if (binding.getQecRegionAttr().getLeafReference().getValue() !=
        region.getLeafReference().getValue())
      return emitOpError(
          "physical_binding QEC region must match the retained stream backing");

    auto graph = (*this)->getParentOfType<GraphOp>();
    if (!graph)
      return emitOpError(
          "linked backed request must be nested in a phys.graph");
    auto architecture =
        dyn_cast_or_null<ArchitectureOp>(SymbolTable::lookupNearestSymbolFrom(
            graph, graph.getArchitectureAttr()));
    if (!architecture || binding->getParentOp() != architecture.getOperation())
      return emitOpError(
          "physical_binding must belong to the graph architecture");
    if (model) {
      if (physicalBinding.getLeafReference().getValue() !=
          model.getQecBindingAttr().getValue())
        return emitOpError(
            "physical_binding must equal the deterministic factory model "
            "QEC binding");
      if (!graph.getOperatingPointAttr() ||
          graph.getOperatingPointAttr() != model.getOperatingPointAttr())
        return emitOpError(
            "factory model operating point must match the enclosing graph");
    } else {
      auto pointRef = graph.getOperatingPointAttr();
      auto point =
          pointRef ? dyn_cast_or_null<OperatingPointOp>(
                         SymbolTable::lookupNearestSymbolFrom(graph, pointRef))
                   : OperatingPointOp{};
      if (!point || point.getMachineAttr() != graph.getArchitectureAttr())
        return emitOpError(
            "scheduled resource model requires the graph operating_point");
      auto timing = point.getTimingAttr();
      auto cycleNs = timing ? physicalTimingValue(timing.get("cycle_ns"))
                            : std::optional<double>{};
      if (!cycleNs)
        cycleNs = timing ? physicalTimingValue(timing.get("surface_cycle_ns"))
                         : std::optional<double>{};
      if (!cycleNs || *cycleNs <= 0.0)
        return emitOpError(
            "scheduled resource model requires an explicit positive cycle_ns "
            "or surface_cycle_ns operating-point fact");

      auto duration = getDurationNsAttr();
      if (!duration || !std::isfinite(duration.getValueAsDouble()) ||
          duration.getValueAsDouble() <= 0.0)
        return emitOpError(
            "linked backed request requires a finite positive duration_ns");
      auto attemptDuration =
          (*this)->getAttrOfType<FloatAttr>("factory_attempt_duration_ns");
      auto acceptance =
          (*this)->getAttrOfType<FloatAttr>("factory_acceptance_probability");
      auto depth =
          (*this)->getAttrOfType<IntegerAttr>("factory_pipeline_depth");
      auto factoryMode = (*this)->getAttrOfType<StringAttr>("factory_mode");
      if (!attemptDuration || !acceptance || !depth || !factoryMode)
        return emitOpError(
            "linked backed request requires complete scheduled-macro factory "
            "evidence");
      double attemptNs = attemptDuration.getValueAsDouble();
      double probability = acceptance.getValueAsDouble();
      int64_t pipelineDepth = depth.getInt();
      if (!std::isfinite(attemptNs) || attemptNs <= 0.0 ||
          !std::isfinite(probability) || probability <= 0.0 ||
          probability > 1.0 || pipelineDepth <= 0)
        return emitOpError("scheduled factory evidence is invalid");

      DictionaryAttr providerMetadata =
          providerOp->getAttrOfType<DictionaryAttr>("metadata");
      auto providerMode =
          providerMetadata ? providerMetadata.getAs<StringAttr>("factory_mode")
                           : StringAttr{};
      auto cyclesText =
          providerMetadata
              ? providerMetadata.getAs<StringAttr>("cycles_per_attempt")
              : StringAttr{};
      auto acceptanceText =
          providerMetadata
              ? providerMetadata.getAs<StringAttr>("acceptance_probability")
              : StringAttr{};
      auto depthText =
          providerMetadata
              ? providerMetadata.getAs<StringAttr>("pipeline_depth")
              : StringAttr{};
      auto footprintText =
          providerMetadata
              ? providerMetadata.getAs<StringAttr>("physical_qubits")
              : StringAttr{};
      double providerCycles = 0.0;
      double providerAcceptance = 0.0;
      int64_t providerDepth = 0;
      int64_t providerFootprint = 0;
      if (!providerMode || providerMode.getValue() != "scheduled_macro" ||
          !cyclesText || cyclesText.getValue().getAsDouble(providerCycles) ||
          !acceptanceText ||
          acceptanceText.getValue().getAsDouble(providerAcceptance) ||
          !depthText || depthText.getValue().getAsInteger(10, providerDepth) ||
          !footprintText ||
          footprintText.getValue().getAsInteger(10, providerFootprint) ||
          !std::isfinite(providerCycles) || providerCycles <= 0.0 ||
          !std::isfinite(providerAcceptance) || providerAcceptance <= 0.0 ||
          providerAcceptance > 1.0 || providerDepth <= 0 ||
          providerFootprint <= 0)
        return emitOpError(
            "provider metadata has invalid or incomplete scheduled-macro "
            "factory evidence");
      if (factoryMode != providerMode)
        return emitOpError(
            "factory_mode must equal the retained provider metadata");

      SymbolTable architectureSymbols(architecture);
      std::optional<int64_t> bindingFootprint;
      for (Attribute rawResource : binding.getResources()) {
        auto resourceRef = dyn_cast<FlatSymbolRefAttr>(rawResource);
        auto resourceClass =
            resourceRef
                ? dyn_cast_or_null<ResourceClassOp>(
                      architectureSymbols.lookup(resourceRef.getValue()))
                : ResourceClassOp{};
        if (!resourceClass || resourceClass.getKind() != "qubit")
          continue;
        if (bindingFootprint)
          return emitOpError(
              "scheduled resource physical_binding must select exactly one "
              "qubit resource class");
        bindingFootprint = resourceClass.getCount();
      }
      if (!bindingFootprint)
        return emitOpError(
            "scheduled resource physical_binding must select exactly one qubit "
            "resource class");
      if (*bindingFootprint != providerFootprint)
        return emitOpError(
                   "provider physical_qubits must equal the exact selected "
                   "binding footprint; binding has ")
               << *bindingFootprint << " qubits but provider declares "
               << providerFootprint;

      auto equalFactoryFact = [](double actual, double expected) {
        double tolerance = std::max(1.0, std::abs(expected)) * 1.0e-12;
        return std::abs(actual - expected) <= tolerance;
      };
      double providerAttemptNs = providerCycles * *cycleNs;
      if (!std::isfinite(providerAttemptNs) || providerAttemptNs <= 0.0)
        return emitOpError(
            "provider cycles_per_attempt times the resolved graph "
            "cycle duration must be finite and positive");
      if (!equalFactoryFact(attemptNs, providerAttemptNs))
        return emitOpError("factory_attempt_duration_ns must equal provider "
                           "cycles_per_attempt times the resolved graph cycle "
                           "duration");
      if (!equalFactoryFact(probability, providerAcceptance))
        return emitOpError(
            "factory_acceptance_probability must equal the retained provider "
            "metadata");
      if (pipelineDepth != providerDepth)
        return emitOpError(
            "factory_pipeline_depth must equal the retained provider metadata");
      double expectedDuration =
          providerAttemptNs / (providerAcceptance * providerDepth);
      if (!std::isfinite(expectedDuration) || expectedDuration <= 0.0)
        return emitOpError(
            "provider-authenticated expected output slot must be finite and "
            "positive");
      if (!equalFactoryFact(duration.getValueAsDouble(), expectedDuration))
        return emitOpError("duration_ns must equal the provider-authenticated "
                           "expected factory output "
                           "slot");
    }
  }
  if (transfer) {
    auto transferOp = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, transfer));
    if (!transferOp)
      return emitOpError("transfer must resolve to a fabric.protocol");
    auto retainedTransfer = stream.getTransferAttr();
    if (!retainedTransfer ||
        retainedTransfer.getValue() != transfer.getLeafReference().getValue())
      return emitOpError(
          "transfer must exactly match the retained stream transfer");
    bool consumesPayload =
        llvm::any_of(transferOp.getFunctionType().getInputs(), [&](Type input) {
          auto resource = dyn_cast<qlx::fabric::ResourceStateType>(input);
          if (!resource)
            return false;
          auto kind = dyn_cast<SymbolRefAttr>(resource.getKind());
          return kind && kind.getLeafReference().getValue() ==
                             payload.getKind().getValue();
        });
    if (!consumesPayload)
      return emitOpError(
          "transfer signature must consume this resource payload");
  }
  unsigned consumerFields = static_cast<unsigned>(!!rawConsumerSite) +
                            static_cast<unsigned>(!!rawConsumerCallee) +
                            static_cast<unsigned>(!!rawConsumerObjective);
  if (consumerFields != 0 && consumerFields != 3)
    return emitOpError(
        "consumer action site, callee, and objective must appear together");
  if (!rawConsumerSite)
    return success();
  auto consumerSite = dyn_cast<SymbolRefAttr>(rawConsumerSite);
  auto consumerCallee = dyn_cast<FlatSymbolRefAttr>(rawConsumerCallee);
  if (!consumerSite || !consumerCallee)
    return emitOpError("consumer witness references must be typed symbols");
  Operation *site =
      lookupLogicalDomainMember<qlx::lvm::ActionSiteOp>(*this, consumerSite);
  auto siteKind = site ? site->getAttrOfType<StringAttr>("kind") : StringAttr();
  if (!site || site->getName().getStringRef() != "lvm.action_site" ||
      !siteKind || siteKind.getValue() != "resource_action")
    return emitOpError(
        "consumer_action_site must resolve to a resource action site");
  Operation *callee =
      SymbolTable::lookupNearestSymbolFrom(*this, consumerCallee);
  if (!callee || (callee->getName().getStringRef() != "fabric.gadget" &&
                  callee->getName().getStringRef() != "fabric.protocol"))
    return emitOpError(
        "consumer_callee must resolve to a Fabric gadget or protocol");
  if (!site->getAttr("objective") ||
      site->getAttr("objective") != rawConsumerObjective)
    return emitOpError(
        "consumer action objective must match the selected callee");
  unsigned matchingCalls = 0;
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (module)
    module.walk([&](CallOp call) {
      if (call.getCalleeAttr() != consumerCallee ||
          call->getAttrOfType<SymbolRefAttr>("resource_action_site") !=
              consumerSite ||
          call->getAttr("resource_objective") != rawConsumerObjective)
        return;
      for (Value input : call.getInputs()) {
        auto await = input.getDefiningOp<qlx::event::AwaitOp>();
        if (await && await.getEvent() == getEvent()) {
          ++matchingCalls;
          break;
        }
      }
    });
  if (matchingCalls != 1)
    return emitOpError("selected consumer witness must identify exactly one "
                       "call fed by this request");
  return success();
}

LogicalResult PrepareOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("prepare must preserve every physical state type");
  return success();
}

LogicalResult MoveOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("move must preserve every physical state type");
  return success();
}

LogicalResult ResetOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every physical state type");
  if (getState() != "zero" && getState() != "plus")
    return emitOpError("state must be zero or plus");
  return success();
}

LogicalResult BarrierOp::verify() {
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every physical state type");
  if (getInputs().empty() && (!getDomains() || getDomains()->empty()))
    return emitOpError("requires physical states or clock/resource domains");
  if (auto domains = getDomains()) {
    llvm::SmallDenseSet<StringRef, 8> seen;
    for (Attribute value : *domains) {
      auto domain = dyn_cast<StringAttr>(value);
      if (!domain || domain.getValue().empty())
        return emitOpError("domains must be nonempty strings");
      if (!seen.insert(domain.getValue()).second)
        return emitOpError("domains must be unique");
    }
  }
  return success();
}

LogicalResult ConsumeResourceOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto stages =
      module ? module->getAttrOfType<ArrayAttr>("qlx.stages") : ArrayAttr();
  if (stages) {
    for (Attribute value : stages) {
      auto stage = dyn_cast<StringAttr>(value);
      if (stage && stage.getValue() == "p3")
        return emitOpError(
            "is legacy logical intent and is not legal in canonical P3; "
            "project a concrete realization using phys.unpack_resource");
    }
  } else {
    auto profiles =
        module ? module->getAttrOfType<ArrayAttr>("qlx.profiles") : ArrayAttr();
    if (profiles) {
      for (Attribute value : profiles) {
        auto profile = dyn_cast<StringAttr>(value);
        if (profile && profile.getValue() == "p3")
          return emitOpError(
              "is legacy logical intent and is not legal in canonical P3; "
              "project a concrete realization using phys.unpack_resource");
      }
    }
  }
  if (getInputs().empty())
    return emitOpError("requires at least one carrier state");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every carrier state type");
  return success();
}

LogicalResult MeasureProductOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state operand");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every measured physical state type");
  if (getPaulis().size() != getInputs().size())
    return emitOpError("requires one Pauli label per state operand");
  for (Attribute value : getPaulis()) {
    auto pauli = dyn_cast<StringAttr>(value);
    if (!pauli || (pauli.getValue() != "X" && pauli.getValue() != "Y" &&
                   pauli.getValue() != "Z"))
      return emitOpError("Pauli labels must be X, Y, or Z strings");
  }
  if (getRecordId().empty())
    return emitOpError("record_id must be nonempty");
  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getInstrumentAttr());
  if (!target)
    return success(); // Partial linked modules resolve at link time.
  auto instrument = dyn_cast<InstrumentOp>(target);
  if (!instrument)
    return emitOpError("instrument reference must resolve to phys.instrument");
  if (instrument.getKind() != "measure_product")
    return emitOpError("instrument must implement measure_product");
  if (!instrument.getPreservesInputs())
    return emitOpError("measure_product instrument must preserve input states");
  if (instrument.getArityAttr() && instrument.getArityAttr().getInt() !=
                                       static_cast<int64_t>(getInputs().size()))
    return emitOpError("operand count does not match fixed instrument arity");
  if (instrument.getRecordSchema() !=
      getRecord().getType().getSchema().getValue())
    return emitOpError("record type does not match instrument record_schema");
  // Native-instrument gate (spec 05 "measure_product", conformance case 83):
  // a native product measurement is legal only when every participating
  // resource class advertises the typed instrument in native_instruments.
  // Resolution walks, per state operand:
  //   !phys.state<@q> -> phys.resource @q {resource_class = @cls}
  //   -> phys.resource_class @cls inside the phys.machine bound by the
  //      enclosing phys.graph ("@graph on @arch").
  // Partially-linked tolerance applies to every link: an unresolvable symbol
  // (no enclosing graph, missing architecture, missing resource declaration,
  // or a class name absent from the bound architecture's symbol table) skips
  // the check, while a fully resolved resource class that does not advertise
  // the instrument is an error. States whose resource symbol resolves to
  // something other than phys.resource (for example fixture IR hosted in a
  // func.func) are outside this resolvable boundary and are skipped too.
  auto graph = (*this)->getParentOfType<GraphOp>();
  if (!graph)
    return success();
  Operation *architectureTarget =
      SymbolTable::lookupNearestSymbolFrom(graph, graph.getArchitectureAttr());
  auto architecture = dyn_cast_or_null<ArchitectureOp>(architectureTarget);
  if (!architecture)
    return success();
  SymbolTable architectureSymbols(architecture);
  for (auto [index, input] : llvm::enumerate(getInputs())) {
    auto state = dyn_cast<StateType>(input.getType());
    if (!state)
      continue;
    Operation *resourceTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, state.getResource());
    auto resource = dyn_cast_or_null<ResourceOp>(resourceTarget);
    if (!resource)
      continue;
    Operation *classTarget =
        architectureSymbols.lookup(resource.getResourceClassAttr().getAttr());
    if (!classTarget)
      continue;
    auto resourceClass = dyn_cast<ResourceClassOp>(classTarget);
    if (!resourceClass)
      return emitOpError("resource class @")
             << resource.getResourceClass() << " bound to state operand "
             << index << " must resolve to phys.resource_class";
    bool advertised = false;
    if (auto instruments = resourceClass.getNativeInstruments())
      for (Attribute value : *instruments)
        if (auto ref = dyn_cast<FlatSymbolRefAttr>(value))
          advertised |= ref.getValue() == getInstrument();
    if (!advertised)
      return emitOpError("resource class @")
             << resourceClass.getSymName() << " of state operand " << index
             << " does not advertise native instrument @" << getInstrument();
  }
  return success();
}

LogicalResult RotateProductOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state operand");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every rotated physical state type");
  if (getPaulis().size() != getInputs().size())
    return emitOpError("requires one Pauli label per state operand");
  for (Attribute value : getPaulis()) {
    auto pauli = dyn_cast<StringAttr>(value);
    if (!pauli || (pauli.getValue() != "X" && pauli.getValue() != "Y" &&
                   pauli.getValue() != "Z"))
      return emitOpError("Pauli labels must be X, Y, or Z strings");
  }
  if (!std::isfinite(getAngle().convertToDouble()))
    return emitOpError("angle must be finite");
  return success();
}

LogicalResult ResourceRotateProductOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one physical state operand");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("must preserve every rotated physical state type");
  if (getPaulis().size() != getInputs().size())
    return emitOpError("requires one Pauli label per state operand");
  for (Attribute value : getPaulis()) {
    auto pauli = dyn_cast<StringAttr>(value);
    if (!pauli || (pauli.getValue() != "X" && pauli.getValue() != "Y" &&
                   pauli.getValue() != "Z"))
      return emitOpError("Pauli labels must be X, Y, or Z strings");
  }
  if (!std::isfinite(getAngle().convertToDouble()))
    return emitOpError("angle must be finite");
  return success();
}

LogicalResult ConditionOp::verify() {
  if (!getSource().getType().isInteger(1) &&
      !isa<RecordType>(getSource().getType()))
    return emitOpError("source must be i1 or a phys.record");
  return success();
}

enum class VerifiedPatchMacroKind { Action, Prepare, Measure };

struct VerifiedPatchMacro {
  VerifiedPatchMacroKind kind;
  std::string physicalOperation;
};

static FailureOr<ResourceClassOp>
callResourceClass(CallOp call, StateType state,
                  SymbolTableCollection &symbolTables) {
  auto graph = call->getParentOfType<GraphOp>();
  auto resource = graph ? symbolTables.lookupNearestSymbolFrom<ResourceOp>(
                              call, state.getResource())
                        : ResourceOp{};
  auto architecture =
      graph ? symbolTables.lookupNearestSymbolFrom<ArchitectureOp>(
                  graph, graph.getArchitectureAttr())
            : ArchitectureOp{};
  auto resourceClass =
      resource && architecture
          ? dyn_cast_or_null<ResourceClassOp>(symbolTables.lookupSymbolIn(
                architecture, resource.getResourceClassAttr().getAttr()))
          : ResourceClassOp{};
  if (!resourceClass)
    return failure();
  return resourceClass;
}

static FailureOr<VerifiedPatchMacro>
verifiedPatchMacro(qlx::fabric::GadgetOp gadget) {
  auto spec = gadget.getSpecAttr()
                  ? dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
                        SymbolTable::lookupNearestSymbolFrom(
                            gadget, gadget.getSpecAttr()))
                  : qlx::fabric::GadgetSpecOp{};
  auto objective = spec ? dyn_cast_or_null<qlx::fabric::ObjectiveOp>(
                              SymbolTable::lookupNearestSymbolFrom(
                                  spec, spec.getObjectiveAttr()))
                        : qlx::fabric::ObjectiveOp{};
  Operation *logical = objective && objective.getLogicalAttr()
                           ? SymbolTable::lookupNearestSymbolFrom(
                                 objective, objective.getLogicalAttr())
                           : nullptr;
  if (!spec || !objective || !logical)
    return failure();
  if (auto action = dyn_cast<qlx::ActionOp>(logical)) {
    StringRef kind = action.getKind();
    if (!llvm::is_contained(ArrayRef<StringRef>{"h", "s", "sdg", "x", "y", "z",
                                                "cx", "cz", "swap"},
                            kind))
      return failure();
    return VerifiedPatchMacro{VerifiedPatchMacroKind::Action, kind.str()};
  }
  auto instrument = dyn_cast<qlx::InstrumentDeclOp>(logical);
  if (!instrument)
    return failure();
  if (instrument.getKind() == "prepare_zero")
    return VerifiedPatchMacro{VerifiedPatchMacroKind::Prepare, "zero"};
  if (instrument.getKind() == "prepare_plus")
    return VerifiedPatchMacro{VerifiedPatchMacroKind::Prepare, "plus"};
  auto semanticsRef =
      dyn_cast_or_null<FlatSymbolRefAttr>(instrument.getSemanticsAttr());
  auto semantics =
      instrument.getKind() == "composite" && semanticsRef
          ? dyn_cast_or_null<qlx::ObjectiveBodyOp>(
                SymbolTable::lookupNearestSymbolFrom(instrument, semanticsRef))
          : qlx::ObjectiveBodyOp{};
  if (!semantics || semantics.getBody().empty())
    return failure();
  Block &body = semantics.getBody().front();
  qlx::MeasureOp measurement;
  for (Operation &operation : body.without_terminator()) {
    auto candidate = dyn_cast<qlx::MeasureOp>(operation);
    if (!candidate || measurement)
      return failure();
    measurement = candidate;
  }
  if (!measurement || measurement.getInput() != body.getArgument(0) ||
      body.getTerminator()->getNumOperands() != 1 ||
      body.getTerminator()->getOperand(0) != measurement.getResult())
    return failure();
  if (measurement.getBasis() == qlx::Pauli::X)
    return VerifiedPatchMacro{VerifiedPatchMacroKind::Measure, "measure_x"};
  if (measurement.getBasis() == qlx::Pauli::Z)
    return VerifiedPatchMacro{VerifiedPatchMacroKind::Measure, "measure"};
  return failure();
}

static LogicalResult verifyPatchMacroCall(CallOp call,
                                          qlx::fabric::GadgetOp gadget,
                                          Block &body, YieldOp yield) {
  auto macro = verifiedPatchMacro(gadget);
  if (failed(macro))
    return success();
  bool hasPatch = false;
  bool hasCarrier = false;
  SymbolTableCollection symbolTables;
  for (Type type : call.getInputs().getTypes()) {
    auto state = dyn_cast<StateType>(type);
    if (!state)
      continue;
    auto resourceClass = callResourceClass(call, state, symbolTables);
    if (failed(resourceClass))
      return call.emitOpError(
          "cannot resolve physical resource class for patch-macro input");
    auto raw = (*resourceClass)->getAttrOfType<StringAttr>("granularity");
    StringRef granularity = raw ? raw.getValue() : StringRef("carrier");
    hasPatch |= granularity == "patch";
    hasCarrier |= granularity == "carrier";
  }
  if (!hasPatch)
    return success();
  if (hasCarrier)
    return call.emitOpError(
        "selected patch macro mixes carrier- and patch-granularity inputs");
  if (std::distance(body.begin(), body.end()) != 2)
    return call.emitOpError(
        "patch-macro call body must contain exactly one physical event");
  Operation &event = body.front();
  if (macro->kind == VerifiedPatchMacroKind::Action) {
    auto apply = dyn_cast<ApplyOp>(event);
    auto action =
        apply ? dyn_cast_or_null<ActionOp>(SymbolTable::lookupNearestSymbolFrom(
                    apply, apply.getActionAttr()))
              : ActionOp{};
    if (!apply || !action || action.getSymName() != macro->physicalOperation ||
        yield.getOperands() != apply.getOutputs())
      return call.emitOpError(
          "patch-macro action body contradicts the retained gadget objective");
    return success();
  }
  if (macro->kind == VerifiedPatchMacroKind::Prepare) {
    auto prepare = dyn_cast<PrepareOp>(event);
    if (!prepare || prepare.getState() != macro->physicalOperation ||
        yield.getOperands() != prepare.getOutputs())
      return call.emitOpError("patch-macro preparation body contradicts the "
                              "retained gadget objective");
    return success();
  }
  auto measure = dyn_cast<MeasureOp>(event);
  auto instrument =
      measure
          ? dyn_cast_or_null<InstrumentOp>(SymbolTable::lookupNearestSymbolFrom(
                measure, measure.getMeasurementAttr()))
          : InstrumentOp{};
  if (!measure || !measure.getDestructive() || !instrument ||
      instrument.getKind() != macro->physicalOperation ||
      yield.getNumOperands() != 1 || yield.getOperand(0) != measure.getRecord())
    return call.emitOpError("patch-macro measurement body contradicts the "
                            "retained gadget objective");
  return success();
}

LogicalResult CallOp::verify() {
  if (getInstance().empty())
    return emitOpError("instance must be nonempty");
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("body must contain exactly one block");
  Block &body = getBody().front();
  if (body.getArgumentTypes() != getInputs().getTypes())
    return emitOpError("body arguments must match call input types");
  if (failed(verifyUniqueCarriedStateResources(*this, getInputs().getTypes())))
    return failure();
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield)
    return emitOpError("body must terminate with phys.yield");
  if (yield.getOperandTypes() != getResultTypes())
    return emitOpError("body yield types must match call result types");
  Operation *callee =
      SymbolTable::lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (callee && callee->getName().getStringRef() != "fabric.gadget" &&
      callee->getName().getStringRef() != "fabric.protocol")
    return emitOpError(
        "callee must resolve to fabric.gadget or fabric.protocol");
  if (auto gadget = dyn_cast_or_null<qlx::fabric::GadgetOp>(callee))
    if (failed(verifyPatchMacroCall(*this, gadget, body, yield)))
      return failure();
  if (auto profile = getProfileAttr()) {
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, profile);
    if (target && target->getName().getStringRef() != "fabric.gadget_profile")
      return emitOpError("profile must resolve to fabric.gadget_profile");
  }
  Attribute rawResourceSite = (*this)->getAttr("resource_action_site");
  Attribute rawResourceObjective = (*this)->getAttr("resource_objective");
  if (static_cast<bool>(rawResourceSite) !=
      static_cast<bool>(rawResourceObjective))
    return emitOpError(
        "resource action site and objective must appear together");
  if (rawResourceSite) {
    auto resourceSite = dyn_cast<SymbolRefAttr>(rawResourceSite);
    if (!resourceSite)
      return emitOpError("resource_action_site must be a typed symbol");
    if (!(*this)->getParentOfType<GraphOp>()) {
      Operation *site = lookupLogicalDomainMember<qlx::lvm::ActionSiteOp>(
          *this, resourceSite);
      auto siteKind =
          site ? site->getAttrOfType<StringAttr>("kind") : StringAttr();
      if (!site || site->getName().getStringRef() != "lvm.action_site" ||
          !siteKind || siteKind.getValue() != "resource_action")
        return emitOpError(
            "resource_action_site must resolve to a resource action site");
      if (!site->getAttr("objective") ||
          site->getAttr("objective") != rawResourceObjective)
        return emitOpError(
            "resource action objective must match the selected callee");
    }
    if (!llvm::any_of(getInputs().getTypes(),
                      [](Type type) { return isa<ResourcePayloadType>(type); }))
      return emitOpError(
          "resource-qualified call must consume one resource payload");
    // GraphOp owns the cross-stage authentication because only the enclosing
    // graph identifies the retained P2 source protocol.  A module-wide lookup
    // here would allow an unrelated linked protocol to lend or poison this
    // call's witness.  GraphOp::verifyRegions reconstructs the exact projected
    // call-instance hierarchy and compares it with the graph-local phys.call
    // tree.
  }
  unsigned communicationFields =
      static_cast<unsigned>((*this)->hasAttr("channel")) +
      static_cast<unsigned>((*this)->hasAttr("channel_capability")) +
      static_cast<unsigned>((*this)->hasAttr("endpoints")) +
      static_cast<unsigned>((*this)->hasAttr("action_site")) +
      static_cast<unsigned>((*this)->hasAttr("generated_by"));
  if (communicationFields != 0 && communicationFields != 5)
    return emitOpError("communication qualification requires channel, channel "
                       "capability, endpoints, "
                       "action_site, and generated_by together");
  return success();
}

LogicalResult CallTemplateOp::verify() {
  if (getTemplateEvent().empty())
    return emitOpError("template_event must be nonempty");
  if (getInstance().empty())
    return emitOpError("instance must be nonempty");
  if (getEventId().empty())
    return emitOpError("event_id must be nonempty");

  auto graph = (*this)->getParentOfType<GraphOp>();
  if (!graph)
    return emitOpError("must be nested in phys.graph");
  if ((*this)->hasAttr("state_boundary_elided") &&
      (!getInputs().empty() || !getOutputs().empty()))
    return emitOpError(
        "state_boundary_elided requires an empty invocation boundary");
  return success();
}

static LogicalResult verifyCallTemplateAgainstCanonical(
    CallTemplateOp invocation, CallOp canonical,
    llvm::function_ref<ResourceOp(FlatSymbolRefAttr)> resolveResource,
    llvm::DenseMap<Operation *, llvm::StringMap<SmallVector<Operation *, 1>>>
        &recordProducers,
    DenseSet<std::pair<Operation *, Attribute>> &verifiedElidedStateAliases) {
  ArrayAttr substitutions =
      invocation->getAttrOfType<ArrayAttr>("resource_substitutions");
  if (canonical.getCalleeAttr() != invocation.getCalleeAttr()) {
    Operation *canonicalCallee = SymbolTable::lookupNearestSymbolFrom(
        invocation, canonical.getCalleeAttr());
    Operation *invocationCallee = SymbolTable::lookupNearestSymbolFrom(
        invocation, invocation.getCalleeAttr());
    if (!canonicalCallee || !invocationCallee ||
        !qlx::fabric::arePhysicallyTemplateEquivalent(canonicalCallee,
                                                      invocationCallee))
      return invocation.emitOpError(
          "a different callee requires a physically equivalent "
          "compiler-generated realization");
  } else if (substitutions) {
    return invocation.emitOpError(
        "resource_substitutions are only valid for physically equivalent "
        "per-site callees");
  }
  if (canonical.getProfileAttr() != invocation.getProfileAttr()) {
    auto canonicalProfile = dyn_cast_or_null<qlx::fabric::GadgetProfileOp>(
        canonical.getProfileAttr() ? SymbolTable::lookupNearestSymbolFrom(
                                         invocation, canonical.getProfileAttr())
                                   : nullptr);
    auto invocationProfile = dyn_cast_or_null<qlx::fabric::GadgetProfileOp>(
        invocation.getProfileAttr()
            ? SymbolTable::lookupNearestSymbolFrom(invocation,
                                                   invocation.getProfileAttr())
            : nullptr);
    if (!canonicalProfile || !invocationProfile ||
        canonicalProfile.getGadgetAttr() != invocation.getCalleeAttr() ||
        invocationProfile.getGadgetAttr() != invocation.getCalleeAttr())
      return invocation.emitOpError(
          "profile must match the canonical phys.call or select another "
          "profile for the same Fabric gadget");
  }
  llvm::DenseMap<Attribute, Attribute> expectedStateAliases;
  llvm::DenseMap<Attribute, Attribute> expectedStateAliasesInverse;
  bool stateBoundaryElided = invocation->hasAttr("state_boundary_elided");
  auto rawStateAliases = invocation->getAttrOfType<ArrayAttr>("state_aliases");
  const std::pair<Operation *, Attribute> elidedAliasKey{
      canonical.getOperation(), rawStateAliases};
  bool elidedAliasProofCached =
      stateBoundaryElided &&
      verifiedElidedStateAliases.contains(elidedAliasKey);
  auto alignTypes = [&](TypeRange canonicalTypes,
                        TypeRange invocationTypes) -> LogicalResult {
    if (canonicalTypes.size() != invocationTypes.size())
      return invocation.emitOpError(
          "input and output arities must match the canonical phys.call");
    for (auto [canonicalType, invocationType] :
         llvm::zip(canonicalTypes, invocationTypes)) {
      if (canonicalType == invocationType)
        continue;
      auto canonicalState = dyn_cast<StateType>(canonicalType);
      auto invocationState = dyn_cast<StateType>(invocationType);
      if (!canonicalState || !invocationState)
        return invocation.emitOpError(
            "non-state input and output types must match the canonical "
            "phys.call exactly");
      ResourceOp canonicalResource =
          resolveResource(canonicalState.getResource());
      ResourceOp invocationResource =
          resolveResource(invocationState.getResource());
      if (!canonicalResource || !invocationResource)
        return invocation.emitOpError(
            "state aliases must resolve canonical and invocation resources");
      if (canonicalResource.getResourceClassAttr() !=
              invocationResource.getResourceClassAttr() ||
          canonicalResource.getKindAttr() != invocationResource.getKindAttr())
        return invocation.emitOpError(
            "state aliases must preserve physical resource class and kind");
      auto [forward, inserted] = expectedStateAliases.try_emplace(
          canonicalState.getResource(), invocationState.getResource());
      if (!inserted && forward->second != invocationState.getResource())
        return invocation.emitOpError(
            "one canonical state resource cannot map to several invocation "
            "resources");
      auto [reverse, reverseInserted] = expectedStateAliasesInverse.try_emplace(
          invocationState.getResource(), canonicalState.getResource());
      if (!reverseInserted && reverse->second != canonicalState.getResource())
        return invocation.emitOpError(
            "state resource aliases must be bijective");
    }
    return success();
  };
  if (stateBoundaryElided && !elidedAliasProofCached) {
    if (canonical.getInputs().empty() ||
        canonical.getInputs().getTypes() != canonical.getOutputs().getTypes() ||
        !llvm::all_of(canonical.getInputs().getTypes(),
                      [](Type type) { return isa<StateType>(type); }))
      return invocation.emitOpError(
          "state_boundary_elided requires an all-state, type-identical "
          "canonical call boundary");
    llvm::DenseSet<Attribute> canonicalResources;
    for (Type type : canonical.getInputs().getTypes())
      canonicalResources.insert(cast<StateType>(type).getResource());
    llvm::DenseSet<Attribute> aliasTargets;
    if (rawStateAliases)
      for (Attribute raw : rawStateAliases) {
        auto alias = dyn_cast<DictionaryAttr>(raw);
        auto source = alias ? alias.getAs<FlatSymbolRefAttr>("template")
                            : FlatSymbolRefAttr{};
        auto target = alias ? alias.getAs<FlatSymbolRefAttr>("alias")
                            : FlatSymbolRefAttr{};
        if (!source || !target || source == target ||
            !canonicalResources.contains(source) ||
            !aliasTargets.insert(target).second ||
            !expectedStateAliases.try_emplace(source, target).second)
          return invocation.emitOpError(
              "elided state aliases must be a nonidentity bijection over "
              "canonical boundary resources");
        ResourceOp sourceResource = resolveResource(source);
        ResourceOp targetResource = resolveResource(target);
        if (!sourceResource || !targetResource ||
            sourceResource.getResourceClassAttr() !=
                targetResource.getResourceClassAttr() ||
            sourceResource.getKindAttr() != targetResource.getKindAttr())
          return invocation.emitOpError(
              "elided state aliases must resolve and preserve physical "
              "resource class and kind");
      }
    llvm::DenseSet<Attribute> invocationResources;
    for (Attribute source : canonicalResources) {
      auto alias = expectedStateAliases.find(source);
      Attribute target =
          alias == expectedStateAliases.end() ? source : alias->second;
      if (!invocationResources.insert(target).second)
        return invocation.emitOpError(
            "elided state aliases must preserve unique physical owners");
    }
  } else if (!stateBoundaryElided &&
             (failed(alignTypes(canonical.getInputs().getTypes(),
                                invocation.getInputs().getTypes())) ||
              failed(alignTypes(canonical.getOutputs().getTypes(),
                                invocation.getOutputs().getTypes())))) {
    return failure();
  }

  if (!elidedAliasProofCached && rawStateAliases && rawStateAliases.empty())
    return invocation.emitOpError(
        "state_aliases must be omitted when no aliases exist");
  llvm::DenseMap<Attribute, Attribute> actualStateAliases;
  StringRef previousTemplate;
  if (!elidedAliasProofCached || substitutions)
    for (Attribute raw :
         rawStateAliases ? rawStateAliases.getValue() : ArrayRef<Attribute>{}) {
      auto alias = dyn_cast<DictionaryAttr>(raw);
      auto templateResource = alias ? alias.getAs<FlatSymbolRefAttr>("template")
                                    : FlatSymbolRefAttr{};
      auto invocationResource =
          alias ? alias.getAs<FlatSymbolRefAttr>("alias") : FlatSymbolRefAttr{};
      if (!alias || alias.size() != 2 || !templateResource ||
          templateResource.getValue().empty() || !invocationResource ||
          invocationResource.getValue().empty())
        return invocation.emitOpError(
            "state_aliases entries require exactly nonempty template and alias "
            "resource references");
      StringRef templateName = templateResource.getValue();
      if (!previousTemplate.empty() && previousTemplate >= templateName)
        return invocation.emitOpError(
            "state_aliases must be ordered by unique template resource");
      previousTemplate = templateName;
      if (!actualStateAliases.try_emplace(templateResource, invocationResource)
               .second)
        return invocation.emitOpError(
            "state_aliases must name unique template resources");
    }
  if (!elidedAliasProofCached) {
    bool exactStateAliases =
        actualStateAliases.size() == expectedStateAliases.size();
    for (const auto &[resource, expected] : expectedStateAliases) {
      auto actual = actualStateAliases.find(resource);
      exactStateAliases &=
          actual != actualStateAliases.end() && actual->second == expected;
    }
    if (!exactStateAliases)
      return invocation.emitOpError(
          "state_aliases must exactly cover every renamed physical state "
          "resource");
    if (stateBoundaryElided)
      verifiedElidedStateAliases.insert(elidedAliasKey);
  }
  if (substitutions) {
    llvm::DenseMap<Attribute, Attribute> actualSubstitutions;
    llvm::DenseSet<Attribute> substitutionTargets;
    for (Attribute raw : substitutions) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto source =
          entry ? entry.getAs<FlatSymbolRefAttr>("from") : FlatSymbolRefAttr{};
      auto target =
          entry ? entry.getAs<FlatSymbolRefAttr>("to") : FlatSymbolRefAttr{};
      if (!entry || entry.size() != 2 || !source || !target ||
          source == target ||
          !actualSubstitutions.try_emplace(source, target).second ||
          !substitutionTargets.insert(target).second)
        return invocation.emitOpError(
            "resource_substitutions require unique, nonidentity from/to "
            "physical resource references");
    }
    bool exactSubstitutions =
        actualSubstitutions.size() == actualStateAliases.size();
    for (const auto &[source, target] : actualStateAliases) {
      auto actual = actualSubstitutions.find(source);
      exactSubstitutions &=
          actual != actualSubstitutions.end() && actual->second == target;
    }
    if (!exactSubstitutions)
      return invocation.emitOpError(
          "resource_substitutions must exactly mirror state_aliases");
  }
  if (failed(verifyUniqueCarriedStateResources(
          invocation, invocation.getInputs().getTypes())))
    return failure();
  auto aliases = invocation->getAttrOfType<ArrayAttr>("record_aliases");
  if (aliases && !invocation.getProfileAttr())
    return invocation.emitOpError("record_aliases require a selected profile");
  llvm::StringSet<> uniqueAliases;
  llvm::StringSet<> uniqueTemplates;
  for (Attribute raw : aliases ? aliases.getValue() : ArrayRef<Attribute>{}) {
    auto alias = dyn_cast<DictionaryAttr>(raw);
    auto aliasName = alias ? alias.getAs<StringAttr>("alias") : StringAttr{};
    auto templateName =
        alias ? alias.getAs<StringAttr>("template") : StringAttr{};
    if (!alias || alias.size() != 2 || !aliasName || aliasName.empty() ||
        !templateName || templateName.empty())
      return invocation.emitOpError(
          "record_aliases entries require exactly nonempty alias and template "
          "strings");
    if (!uniqueAliases.insert(aliasName.getValue()).second ||
        !uniqueTemplates.insert(templateName.getValue()).second)
      return invocation.emitOpError(
          "record_aliases must name unique aliases and canonical templates");
    auto qualified = parseQualifiedRecord(templateName.getValue());
    if (failed(qualified))
      return invocation.emitOpError(
          "record alias template has malformed folded-repeat qualification");
    auto [producerIndex, inserted] =
        recordProducers.try_emplace(canonical.getOperation());
    if (inserted)
      canonical.getBody().walk([&](Operation *operation) {
        if (auto measurement = dyn_cast<MeasureOp>(operation))
          producerIndex->second[measurement.getRecordId()].push_back(operation);
        if (auto measurement = dyn_cast<MeasureProductOp>(operation))
          producerIndex->second[measurement.getRecordId()].push_back(operation);
      });
    auto producers = producerIndex->second.find(qualified->first);
    if (producers == producerIndex->second.end() ||
        producers->second.size() != 1 ||
        !qualifiedRecordMatchesProducer(producers->second.front(),
                                        qualified->second,
                                        canonical.getOperation()))
      return invocation.emitOpError("record alias template '")
             << templateName.getValue()
             << "' must resolve to one exact canonical measurement occurrence";
  }
  return success();
}

LogicalResult SpacetimeCallOp::verify() {
  if (getInstance().empty())
    return emitOpError("instance must be nonempty");
  if (getEventId().empty())
    return emitOpError("event_id must be nonempty");
  auto graph = (*this)->getParentOfType<GraphOp>();
  if (!graph)
    return emitOpError("must be nested in phys.graph");
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("input and output types must be identical");
  if (failed(verifyUniqueCarriedStateResources(*this, getInputs().getTypes())))
    return failure();
  auto plan = dyn_cast_or_null<SpacetimePlanOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getPlanAttr()));
  if (!plan)
    return emitOpError("plan must resolve to phys.spacetime_plan");
  if (plan.getRecurrenceResourceKindAttr())
    return emitOpError(
        "periodic producer plans characterize recurring factory capacity and "
        "cannot be invoked as one-shot spacetime calls");
  if (plan.getArchitectureAttr() != graph.getArchitectureAttr())
    return emitOpError("plan architecture must match the enclosing graph");
  if (!graph.getOperatingPointAttr() ||
      plan.getOperatingPointAttr() != graph.getOperatingPointAttr())
    return emitOpError("plan operating point must match the enclosing graph");
  if (auto source = getSourceProtocolAttr()) {
    auto protocol = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, source));
    if (!protocol)
      return emitOpError(
          "source_protocol must resolve to a retained fabric.protocol");
  }
  return verifyRegisteredSpacetimeInvocation(*this, plan);
}
LogicalResult RetryOp::verify() {
  // Retry replay is meaningful only with an exact projected attempt boundary.
  if (getMaxAttempts() <= 0)
    return emitOpError("max_attempts must be positive");
  auto attemptAttr = getAttemptAttr();
  if (!attemptAttr)
    return emitOpError("requires the named projected attempt");
  Operation *attempt = SymbolTable::lookupNearestSymbolFrom(*this, attemptAttr);
  auto attemptGadget = dyn_cast_or_null<qlx::fabric::GadgetOp>(attempt);
  auto attemptProtocol = dyn_cast_or_null<qlx::fabric::ProtocolOp>(attempt);
  if (!attemptGadget && !attemptProtocol)
    return emitOpError(
        "attempt must resolve to a retained fabric.gadget or contracted "
        "fabric.protocol");

  auto profileAttr = getProfileAttr();
  if (!profileAttr)
    return emitOpError("requires the selected projected-attempt profile");
  Operation *profile = SymbolTable::lookupNearestSymbolFrom(*this, profileAttr);
  if (!profile || profile->getName().getStringRef() != "fabric.gadget_profile")
    return emitOpError(
        "profile must resolve to a retained fabric.gadget_profile");
  auto profileGadget = profile->getAttrOfType<FlatSymbolRefAttr>("gadget");
  if (attemptProtocol) {
    if (!attemptProtocol.getPredicateGadgetAttr() ||
        !attemptProtocol.getPredicateProfileAttr() ||
        attemptProtocol.getPredicateProfileAttr() != profileAttr ||
        attemptProtocol.getPredicateGadgetAttr() != profileGadget)
      return emitOpError(
          "retry profile must analyze the contracted protocol predicate");
  } else if (profileGadget != attemptAttr) {
    return emitOpError("profile must analyze the retry attempt");
  }

  if (getSuccessProbabilityAttr()) {
    double probability = getSuccessProbabilityAttr().getValueAsDouble();
    if (!std::isfinite(probability) || probability <= 0.0 || probability > 1.0)
      return emitOpError(
          "success_probability must be finite and lie in (0, 1]");
    auto sourceAttr = getSuccessProbabilitySourceAttr();
    auto evidenceAttr = getSuccessProbabilityEvidenceAttr();
    if (!sourceAttr || !evidenceAttr || evidenceAttr.getValue().empty())
      return emitOpError(
          "success_probability requires a source and nonempty evidence");
    if (sourceAttr != attemptAttr && sourceAttr != getProfileAttr())
      return emitOpError(
          "success_probability source must be the attempt or selected profile");

    DictionaryAttr metadata;
    if (sourceAttr == profileAttr) {
      metadata = profile->getAttrOfType<DictionaryAttr>("metadata");
    } else if (sourceAttr == attemptAttr) {
      if (attemptProtocol) {
        metadata = attemptProtocol->getAttrOfType<DictionaryAttr>("metadata");
      } else {
        auto specAttr = attempt->getAttrOfType<FlatSymbolRefAttr>("spec");
        Operation *spec =
            specAttr ? SymbolTable::lookupNearestSymbolFrom(*this, specAttr)
                     : nullptr;
        if (spec && spec->getName().getStringRef() == "fabric.gadget_spec") {
          if (attempt->getAttrOfType<TypeAttr>("function_type") !=
              spec->getAttrOfType<TypeAttr>("function_type"))
            return emitOpError("gadget probability source disagrees with its "
                               "retained gadget_spec signature");
          auto boundary =
              attempt->getAttrOfType<DictionaryAttr>("realization_boundary");
          if (!boundary ||
              boundary.getAs<ArrayAttr>("ports") !=
                  spec->getAttrOfType<ArrayAttr>("ports") ||
              boundary.getAs<ArrayAttr>("flows") !=
                  spec->getAttrOfType<ArrayAttr>("flows"))
            return emitOpError("gadget probability source disagrees with its "
                               "retained gadget_spec realization boundary");
          if (failed(mlir::verify(spec, /*verifyRecursively=*/false)))
            return emitOpError("gadget probability source has an invalid "
                               "retained gadget_spec boundary");
          metadata = spec->getAttrOfType<DictionaryAttr>("metadata");
        } else {
          return emitOpError(
              "gadget probability source requires a retained gadget_spec");
        }
      }
    }
    auto probabilityText =
        metadata ? metadata.getAs<StringAttr>("success_probability")
                 : StringAttr{};
    auto evidenceText =
        metadata ? metadata.getAs<StringAttr>("success_probability_evidence")
                 : StringAttr{};
    double establishedProbability = 0.0;
    if (!probabilityText ||
        probabilityText.getValue().getAsDouble(establishedProbability) ||
        establishedProbability != probability)
      return emitOpError(
          "success_probability must equal the value established by its "
          "retained source");
    if (!evidenceText || evidenceText.getValue() != evidenceAttr.getValue())
      return emitOpError(
          "success_probability evidence must match its retained source");

    StringRef evidence = evidenceAttr.getValue();
    constexpr StringLiteral synthesisPrefix = "synthesis:sha256:";
    constexpr StringLiteral analysisPrefix = "analysis:";
    if (evidence.starts_with(synthesisPrefix)) {
      StringRef digest = evidence.drop_front(synthesisPrefix.size());
      auto sourceDigest = metadata
                              ? metadata.getAs<StringAttr>("synthesis_sha256")
                              : StringAttr{};
      if (sourceAttr != attemptAttr || digest.size() != 64 ||
          !llvm::all_of(digest,
                        [](char character) {
                          return (character >= '0' && character <= '9') ||
                                 (character >= 'a' && character <= 'f');
                        }) ||
          !sourceDigest || sourceDigest.getValue() != digest)
        return emitOpError(
            "synthesis probability evidence must bind the attempt digest");
    } else if (evidence.starts_with(analysisPrefix)) {
      if (sourceAttr != getProfileAttr() ||
          evidence.size() == analysisPrefix.size())
        return emitOpError(
            "analysis probability evidence must bind the selected profile");
    } else {
      return emitOpError(
          "success_probability evidence must be synthesis:sha256:... or "
          "analysis:...");
    }
  } else if (getSuccessProbabilitySourceAttr() ||
             getSuccessProbabilityEvidenceAttr()) {
    return emitOpError("probability provenance requires success_probability");
  }
  if (getInputs().getTypes() != getOutputs().getTypes())
    return emitOpError("retry must preserve every carried physical state type");
  if (getExhaustion() != "report_failure" && getExhaustion() != "abort" &&
      getExhaustion() != "return_last")
    return emitOpError(
        "exhaustion must be report_failure, abort, or return_last");
  if (auto commitPoint = getCommitPoint()) {
    StringRef value = *commitPoint;
    bool qualifiedOutput = value.consume_front("before_output:");
    if ((qualifiedOutput && value.empty()) ||
        (!qualifiedOutput && value != "before_output" &&
         value != "pack_resource"))
      return emitOpError(
          "commit_point must be before_output, before_output:<endpoint>, or "
          "pack_resource");
  }

  if (getAttemptEvent().empty() || getDecisionEvent().empty())
    return emitOpError("attempt_event and decision_event must be nonempty");
  Block *block = (*this)->getBlock();
  auto findEvent = [&](StringRef id) -> FailureOr<Operation *> {
    Operation *match = nullptr;
    for (Operation &candidate : *block) {
      auto candidateId = candidate.getAttrOfType<StringAttr>("event_id");
      if (!candidateId || candidateId.getValue() != id)
        continue;
      if (match)
        return failure();
      match = &candidate;
    }
    if (!match)
      return failure();
    return match;
  };

  FailureOr<Operation *> attemptEvent = findEvent(getAttemptEvent());
  if (failed(attemptEvent))
    return emitOpError(
        "attempt_event must identify exactly one event in the retry block");
  Operation *attemptOperation = *attemptEvent;
  FlatSymbolRefAttr attemptCallee;
  FlatSymbolRefAttr attemptProfile;
  if (auto attemptCall = dyn_cast<CallOp>(attemptOperation)) {
    attemptCallee = attemptCall.getCalleeAttr();
    attemptProfile = attemptCall.getProfileAttr();
  } else if (auto attemptTemplate =
                 dyn_cast<CallTemplateOp>(attemptOperation)) {
    attemptCallee = attemptTemplate.getCalleeAttr();
    attemptProfile = attemptTemplate.getProfileAttr();
  } else {
    return emitOpError(
        "attempt_event must identify a folded phys.call or authenticated "
        "phys.call_template");
  }
  if (attemptCallee != getAttemptAttr())
    return emitOpError("attempt_event callee must equal the selected attempt");
  if (!attemptProfile || attemptProfile != getProfileAttr())
    return emitOpError("attempt_event profile must equal the selected profile");
  llvm::SmallDenseSet<Value, 8> carriedStates;
  for (Value input : getInputs()) {
    if (input.getDefiningOp() != attemptOperation)
      return emitOpError(
          "every carried physical state must originate from attempt_event");
    if (!carriedStates.insert(input).second)
      return emitOpError(
          "every attempt physical-state result must be carried exactly once");
  }
  for (Value result : attemptOperation->getResults())
    if (isa<StateType>(result.getType()) && !carriedStates.contains(result))
      return emitOpError(
          "every attempt physical-state result must be carried exactly once");

  FailureOr<Operation *> decisionEvent = findEvent(getDecisionEvent());
  if (failed(decisionEvent))
    return emitOpError(
        "decision_event must identify exactly one event in the retry block");
  if (getSuccess().getDefiningOp() != *decisionEvent)
    return emitOpError(
        "decision_event must produce the retry success predicate");
  if (auto condition = dyn_cast<ConditionOp>(*decisionEvent);
      condition && condition.getSource().getDefiningOp() == attemptOperation &&
      condition.getSource().getType().isInteger(1) && !condition.getExpected())
    return emitOpError(
        "decision_event must not invert a direct i1 attempt result");
  if (!attemptOperation->isBeforeInBlock(*decisionEvent) ||
      !(*decisionEvent)->isBeforeInBlock(getOperation()))
    return emitOpError(
        "attempt_event and decision_event must precede retry in that order");

  SmallVector<Value, 8> pending{getSuccess()};
  llvm::SmallPtrSet<Operation *, 8> visited;
  bool reachesAttempt = false;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    Operation *producer = value.getDefiningOp();
    if (!producer || !visited.insert(producer).second)
      continue;
    if (producer == attemptOperation) {
      reachesAttempt = true;
      continue;
    }
    pending.append(producer->operand_begin(), producer->operand_end());
  }
  if (!reachesAttempt)
    return emitOpError(
        "decision_event must be causally derived from attempt_event results");
  return success();
}

static LogicalResult parseScheduleList(Operation *owner, StringRef value,
                                       StringRef field,
                                       SmallVectorImpl<StringRef> &result) {
  if (value.empty())
    return success();
  llvm::SmallDenseSet<StringRef, 8> unique;
  SmallVector<StringRef, 8> parts;
  value.split(parts, ',', -1, true);
  for (StringRef part : parts) {
    if (part.empty())
      return owner->emitOpError()
             << "schedule " << field << " must not contain empty entries";
    if (!unique.insert(part).second)
      return owner->emitOpError()
             << "schedule " << field << " must contain unique entries";
    result.push_back(part);
  }
  return success();
}

static FailureOr<ScheduleClaim> parseScheduleEntry(ScheduleOp schedule,
                                                   Attribute raw) {
  auto text = dyn_cast<StringAttr>(raw);
  if (!text) {
    schedule.emitOpError("entries must be serialized schedule strings");
    return failure();
  }
  SmallVector<StringRef, 20> fields;
  text.getValue().split(fields, '|', -1, true);
  if (fields.size() < 5) {
    schedule.emitOpError(
        "schedule entries require event_id, kind, start, duration, and "
        "resources fields");
    return failure();
  }

  ScheduleClaim result;
  result.id = fields[0];
  result.kind = fields[1];
  if (result.id.empty() || result.kind.empty()) {
    schedule.emitOpError("schedule event_id and kind must be nonempty");
    return failure();
  }
  if (fields[2].getAsDouble(result.start) ||
      fields[3].getAsDouble(result.duration) || !std::isfinite(result.start) ||
      !std::isfinite(result.duration) || result.start < 0.0 ||
      result.duration < 0.0) {
    schedule.emitOpError(
        "schedule start and duration must be finite nonnegative numbers");
    return failure();
  }
  if (failed(parseScheduleList(schedule, fields[4], "resources",
                               result.resources)))
    return failure();

  static const llvm::StringSet<> allowedFields = [] {
    llvm::StringSet<> fields;
    for (StringRef name : {"deps",
                           "data_deps",
                           "resource_deps",
                           "domain_deps",
                           "parent",
                           "branch",
                           "condition",
                           "max_attempts",
                           "commit_point",
                           "repeat_count",
                           "repeat_period_ns",
                           "repeat_epilogue_ns",
                           "max_iterations",
                           "callee",
                           "instance",
                           "profile",
                           "attempt",
                           "template_event",
                           "attempt_event",
                           "decision_event",
                           "exhaustion",
                           "success_probability",
                           "success_probability_source",
                           "success_probability_evidence"})
      fields.insert(name);
    return fields;
  }();
  llvm::StringMap<StringRef> details;
  for (StringRef field : ArrayRef(fields).drop_front(5)) {
    auto [name, value] = field.split('=');
    if (name.empty() || !field.contains('=') || !allowedFields.contains(name)) {
      schedule.emitOpError("schedule entry contains unknown detail field '")
          << name << "'";
      return failure();
    }
    if (!details.try_emplace(name, value).second) {
      schedule.emitOpError("schedule entry contains duplicate detail field '")
          << name << "'";
      return failure();
    }
  }

  auto detail = [&](StringRef name) -> StringRef {
    auto found = details.find(name);
    return found == details.end() ? StringRef() : found->second;
  };
  auto parseInteger =
      [&](StringRef name,
          std::optional<int64_t> &destination) -> LogicalResult {
    StringRef value = detail(name);
    if (value.empty())
      return success();
    int64_t parsed = 0;
    if (value.getAsInteger(10, parsed))
      return schedule.emitOpError()
             << "schedule " << name << " must be an integer";
    destination = parsed;
    return success();
  };
  auto parseFloat = [&](StringRef name,
                        std::optional<double> &destination) -> LogicalResult {
    StringRef value = detail(name);
    if (value.empty())
      return success();
    double parsed = 0.0;
    if (value.getAsDouble(parsed) || !std::isfinite(parsed) || parsed < 0.0)
      return schedule.emitOpError()
             << "schedule " << name << " must be a finite nonnegative number";
    destination = parsed;
    return success();
  };

  result.hasDataDependencies = details.contains("data_deps");
  result.hasResourceDependencies = details.contains("resource_deps");
  result.hasDomainDependencies = details.contains("domain_deps");
  if (failed(parseScheduleList(schedule, detail("deps"), "dependencies",
                               result.dependencies)) ||
      failed(parseScheduleList(schedule, detail("data_deps"),
                               "data dependencies", result.dataDependencies)) ||
      failed(parseScheduleList(schedule, detail("resource_deps"),
                               "resource dependencies",
                               result.resourceDependencies)) ||
      failed(parseScheduleList(schedule, detail("domain_deps"),
                               "domain dependencies",
                               result.domainDependencies)) ||
      failed(parseInteger("max_attempts", result.maxAttempts)) ||
      failed(parseInteger("repeat_count", result.repeatCount)) ||
      failed(parseFloat("repeat_period_ns", result.repeatPeriod)) ||
      failed(parseFloat("repeat_epilogue_ns", result.repeatEpilogue)) ||
      failed(parseInteger("max_iterations", result.maxIterations)))
    return failure();
  result.parent = detail("parent");
  result.branch = detail("branch");
  result.condition = detail("condition");
  result.callee = detail("callee");
  result.instance = detail("instance");
  result.profile = detail("profile");
  result.templateEvent = detail("template_event");
  result.attempt = detail("attempt");
  result.attemptEvent = detail("attempt_event");
  result.decisionEvent = detail("decision_event");
  result.commitPoint = detail("commit_point");
  result.exhaustion = detail("exhaustion");
  result.successProbabilitySource = detail("success_probability_source");
  result.successProbabilityEvidence = detail("success_probability_evidence");
  StringRef probability = detail("success_probability");
  if (!probability.empty()) {
    double parsed = 0.0;
    if (probability.getAsDouble(parsed) || !std::isfinite(parsed) ||
        parsed <= 0.0 || parsed > 1.0) {
      schedule.emitOpError("schedule success_probability must lie in (0, 1]");
      return failure();
    }
    result.successProbability = parsed;
  }
  return result;
}

struct ScheduleVerifierProfileStats {
  uint64_t expectedResourceCalls = 0;
  uint64_t expectedResourceTypes = 0;
  uint64_t expectedResourceDedupComparisons = 0;
  uint64_t expectedResourceHashProbes = 0;
  uint64_t expectedResourceMaxWidth = 0;
  uint64_t dataDependencyDerivations = 0;
  uint64_t aliasMapBuilds = 0;
  uint64_t aliasMapCacheHits = 0;
  uint64_t aliasMapEntries = 0;
  uint64_t expectedResourceCacheHits = 0;
  uint64_t expectedResourceCacheBuilds = 0;
  uint64_t collectFirstUseVisits = 0;
  uint64_t ancestryCalls = 0;
  uint64_t journalKeyBytes = 0;
  uint64_t overlapCandidates = 0;
};

struct ScheduleGraphFacts {
  ArrayRef<StringRef> resources;
  SmallVector<StringRef, 4> dataDependencies;
  bool derived = false;
};

static bool
scheduleEntriesAreExclusive(unsigned left, unsigned right,
                            ArrayRef<ScheduleClaim> entries,
                            const llvm::StringMap<unsigned> &entryById) {
  auto lineage = [&](unsigned index) {
    SmallVector<unsigned, 8> path;
    llvm::SmallDenseSet<unsigned, 8> visited;
    while (visited.insert(index).second) {
      path.push_back(index);
      StringRef parent = entries[index].parent;
      if (parent.empty())
        break;
      auto found = entryById.find(parent);
      if (found == entryById.end())
        break;
      index = found->second;
    }
    std::reverse(path.begin(), path.end());
    return path;
  };
  SmallVector<unsigned, 8> leftPath = lineage(left);
  SmallVector<unsigned, 8> rightPath = lineage(right);
  unsigned common = 0;
  while (common < leftPath.size() && common < rightPath.size() &&
         leftPath[common] == rightPath[common])
    ++common;
  if (common == 0 || common == leftPath.size() || common == rightPath.size())
    return false;
  unsigned parent = leftPath[common - 1];
  StringRef parentKind = entries[parent].kind;
  if (parentKind != "if" && parentKind != "try_take")
    return false;
  StringRef leftBranch = entries[leftPath[common]].branch;
  StringRef rightBranch = entries[rightPath[common]].branch;
  return !leftBranch.empty() && !rightBranch.empty() &&
         leftBranch != rightBranch;
}

static std::pair<double, double>
scheduleDynamicInterval(unsigned index, unsigned other,
                        ArrayRef<ScheduleClaim> entries,
                        const llvm::StringMap<unsigned> &entryById) {
  auto lineage = [&](unsigned current) {
    SmallVector<unsigned, 8> path;
    llvm::SmallDenseSet<unsigned, 8> visited;
    while (visited.insert(current).second) {
      path.push_back(current);
      StringRef parent = entries[current].parent;
      if (parent.empty())
        break;
      auto found = entryById.find(parent);
      if (found == entryById.end())
        break;
      current = found->second;
    }
    std::reverse(path.begin(), path.end());
    return path;
  };

  SmallVector<unsigned, 8> path = lineage(index);
  SmallVector<unsigned, 8> otherPath = lineage(other);
  unsigned common = 0;
  while (common < path.size() && common < otherPath.size() &&
         path[common] == otherPath[common])
    ++common;

  // Rows nested under a folded loop describe one template occurrence.  When
  // the other row is outside that same loop lineage, the loop's finite
  // envelope is the compact exact exclusion boundary for all dynamic
  // occurrences.  This catches later-iteration collisions without expanding
  // even very large repeat counts.
  for (unsigned ancestor : ArrayRef(path).drop_front(common)) {
    const ScheduleClaim &candidate = entries[ancestor];
    if ((candidate.kind == "repeat" && candidate.repeatCount &&
         *candidate.repeatCount > 1) ||
        candidate.kind == "while")
      return {candidate.start, candidate.finish()};
  }
  return {entries[index].start, entries[index].finish()};
}

static Operation *scheduleDataProducerImpl(Value value, GraphOp graph,
                                           llvm::DenseSet<Value> &active) {
  if (!active.insert(value).second)
    return nullptr;
  auto finish = [&](Operation *result) {
    active.erase(value);
    return result;
  };
  if (auto result = dyn_cast<OpResult>(value)) {
    Operation *owner = result.getOwner();
    return finish(isSchedulableGraphEvent(owner, graph) ? owner : nullptr);
  }

  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return finish(nullptr);
  Block *block = argument.getOwner();
  Operation *parent = block->getParentOp();
  if (!parent || parent == graph.getOperation())
    return finish(nullptr);
  unsigned ordinal = argument.getArgNumber();

  if (isa<qlx::event::TryTakeOp>(parent))
    return finish(parent);
  if (isa<CallOp, qlx::cflow::RepeatOp>(parent)) {
    Operation *producer = ordinal < parent->getNumOperands()
                              ? scheduleDataProducerImpl(
                                    parent->getOperand(ordinal), graph, active)
                              : nullptr;
    return finish(producer);
  }
  if (auto loop = dyn_cast<qlx::cflow::WhileOp>(parent)) {
    if (block->getParent() == &loop.getBeforeRegion()) {
      Operation *producer = ordinal < loop.getNumOperands()
                                ? scheduleDataProducerImpl(
                                      loop->getOperand(ordinal), graph, active)
                                : nullptr;
      return finish(producer);
    }
    auto condition = dyn_cast<qlx::cflow::WhileConditionOp>(
        loop.getBeforeRegion().front().getTerminator());
    Operation *producer =
        condition
            ? scheduleDataProducerImpl(condition.getCondition(), graph, active)
            : nullptr;
    return finish(producer ? producer : parent);
  }
  return finish(nullptr);
}

static Operation *scheduleDataProducer(Value value, GraphOp graph) {
  llvm::DenseSet<Value> active;
  return scheduleDataProducerImpl(value, graph, active);
}

static SmallVector<StringRef, 4> expectedScheduleDataDependencies(
    Operation *operation, GraphOp graph,
    ScheduleVerifierProfileStats *profileStats = nullptr) {
  if (profileStats)
    ++profileStats->dataDependencyDerivations;
  SmallVector<StringRef, 4> result;
  for (Value operand : operation->getOperands()) {
    Operation *producer = scheduleDataProducer(operand, graph);
    auto event =
        producer ? producer->getAttrOfType<StringAttr>("event_id") : nullptr;
    if (event && !llvm::is_contained(result, event.getValue()))
      result.push_back(event.getValue());
  }
  return result;
}

static FactoryModelOp scheduleFactoryModel(Operation *operation) {
  FlatSymbolRefAttr model;
  if (auto start = dyn_cast<FactoryStartOp>(operation))
    model = start.getFactoryModelAttr();
  else if (auto request = dyn_cast<ResourceRequestOp>(operation))
    model = request.getFactoryModelAttr();
  if (!model)
    return {};
  return dyn_cast_or_null<FactoryModelOp>(
      SymbolTable::lookupNearestSymbolFrom(operation, model));
}

static bool isOperandFreeDomainBarrier(Operation *operation,
                                       StringRef domainName) {
  auto barrier = dyn_cast<BarrierOp>(operation);
  if (!barrier || !barrier.getInputs().empty())
    return false;
  ArrayAttr domains = barrier.getDomainsAttr();
  return domains && llvm::any_of(domains, [&](Attribute raw) {
           auto domain = dyn_cast<StringAttr>(raw);
           return domain && domain.getValue() == domainName;
         });
}

static bool isOperandFreeClockBarrier(Operation *operation) {
  return isOperandFreeDomainBarrier(operation, "clock");
}

static bool isOperandFreeFactoryBarrier(Operation *operation) {
  return isOperandFreeDomainBarrier(operation, "factory");
}

static std::string scheduleFactoryResource(FactoryModelOp model) {
  return ("factory:" + model.getSymName()).str();
}

static FailureOr<SmallVector<std::string, 4>>
expectedSpacetimePhaseResources(ScheduleOp schedule, SpacetimePhaseOp phase) {
  SmallVector<std::string, 4> result;
  auto plan = phase->getParentOfType<SpacetimePlanOp>();
  auto module = phase->getParentOfType<ModuleOp>();
  SymbolTable moduleSymbols(module);
  auto architecture =
      plan ? moduleSymbols.lookup<ArchitectureOp>(plan.getArchitecture())
           : ArchitectureOp{};
  if (!plan || !architecture)
    return schedule.emitOpError(
        "cannot resolve a spacetime phase's physical architecture");
  SymbolTable architectureSymbols(architecture);
  for (Attribute raw : phase.getResourceClasses()) {
    auto reference = dyn_cast<SymbolRefAttr>(raw);
    auto resource = reference &&
                            reference.getRootReference().getValue() ==
                                plan.getArchitecture() &&
                            reference.getNestedReferences().size() == 1
                        ? architectureSymbols.lookup<ResourceClassOp>(
                              reference.getLeafReference().getValue())
                        : ResourceClassOp{};
    if (!resource)
      return schedule.emitOpError(
          "cannot resolve a spacetime phase resource-class claim");
    result.push_back(("class:" + resource.getSymName()).str());
  }
  if (auto claims = phase.getResourceClaimsAttr())
    for (Attribute raw : claims) {
      auto claim = cast<DictionaryAttr>(raw);
      auto reference = claim.getAs<SymbolRefAttr>("resource_class");
      int64_t offset = claim.getAs<IntegerAttr>("offset").getInt();
      int64_t count = claim.getAs<IntegerAttr>("count").getInt();
      for (int64_t index = offset; index < offset + count; ++index)
        result.push_back((reference.getLeafReference().getValue() + "[" +
                          std::to_string(index) + "]")
                             .str());
    }
  for (Attribute raw : phase.getFactoryModels()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
    auto model =
        reference ? moduleSymbols.lookup<FactoryModelOp>(reference.getValue())
                  : FactoryModelOp{};
    if (!model)
      return schedule.emitOpError(
          "cannot resolve a spacetime phase factory-model claim");
    result.push_back(scheduleFactoryResource(model));
  }
  llvm::sort(result);
  result.erase(std::unique(result.begin(), result.end()), result.end());
  return result;
}

struct CanonicalScheduleResourceSummary {
  SmallVector<FlatSymbolRefAttr, 8> stateResources;
  SmallVector<std::string, 4> opaqueResources;
  bool synchronizesClockDomain = false;
};

using CanonicalScheduleResourceCache =
    DenseMap<Operation *, CanonicalScheduleResourceSummary>;
using ScheduleCanonicalCallIndex = llvm::StringMap<CallOp>;
using ExpectedCallTemplateResourceKey = std::pair<Operation *, Attribute>;
using ExpectedCallTemplateResourceCache =
    DenseMap<ExpectedCallTemplateResourceKey, ArrayRef<StringRef>>;

static std::optional<std::string>
schedulePhysicalBindingResource(Operation *operation) {
  auto request = dyn_cast<ResourceRequestOp>(operation);
  auto binding = request ? request.getPhysicalBindingAttr() : SymbolRefAttr{};
  if (!binding)
    return std::nullopt;
  return ("binding:" + binding.getLeafReference().getValue()).str();
}

static FailureOr<CallOp>
scheduleCanonicalCall(ScheduleOp schedule,
                      const ScheduleCanonicalCallIndex &index,
                      CallTemplateOp invocation) {
  auto found = index.find(invocation.getTemplateEvent());
  if (found == index.end()) {
    schedule.emitOpError("call-template graph event '")
        << invocation.getEventId()
        << "' must resolve exactly one canonical phys.call event '"
        << invocation.getTemplateEvent() << "' during resource proof";
    return failure();
  }
  return found->second;
}

static FailureOr<DenseMap<Attribute, Attribute>>
scheduleStateAliasAttributes(ScheduleOp schedule, CallTemplateOp invocation) {
  DenseMap<Attribute, Attribute> result;
  auto aliases = invocation.getStateAliasesAttr();
  for (Attribute raw : aliases ? aliases.getValue() : ArrayRef<Attribute>{}) {
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto source = entry ? entry.getAs<FlatSymbolRefAttr>("template")
                        : FlatSymbolRefAttr{};
    auto target =
        entry ? entry.getAs<FlatSymbolRefAttr>("alias") : FlatSymbolRefAttr{};
    if (!entry || entry.size() != 2 || !source || !target ||
        !result.try_emplace(source, target).second) {
      schedule.emitOpError("call-template graph event '")
          << invocation.getEventId()
          << "' has malformed or duplicate state_aliases during resource "
             "proof";
      return failure();
    }
  }
  return result;
}

static FailureOr<const CanonicalScheduleResourceSummary *>
deriveCanonicalScheduleResources(
    ScheduleOp schedule, GraphOp graph, CallOp canonical,
    const ScheduleCanonicalCallIndex &canonicalCalls,
    CanonicalScheduleResourceCache &cache,
    DenseSet<Operation *> &activeCanonicalCalls) {
  auto cached = cache.find(canonical.getOperation());
  if (cached != cache.end())
    return &cached->second;
  if (!activeCanonicalCalls.insert(canonical.getOperation()).second) {
    schedule.emitOpError("recursive phys.call_template resource proof reaches "
                         "canonical call event '")
        << canonical.getEventIdAttr().getValue()
        << "' more than once on one active path";
    return failure();
  }
  llvm::scope_exit leaveCanonical(
      [&] { activeCanonicalCalls.erase(canonical.getOperation()); });

  CanonicalScheduleResourceSummary summary;
  DenseSet<Attribute> seenStates;
  llvm::StringSet<> seenOpaque;
  auto appendState = [&](FlatSymbolRefAttr resource) {
    if (seenStates.insert(resource).second)
      summary.stateResources.push_back(resource);
  };
  auto appendOpaque = [&](StringRef resource) {
    if (seenOpaque.insert(resource).second)
      summary.opaqueResources.push_back(resource.str());
  };

  LogicalResult derived = success();
  WalkResult walked = canonical.getBody().walk([&](Operation *nested) {
    for (Type type : nested->getOperandTypes())
      if (auto state = dyn_cast<StateType>(type))
        appendState(state.getResource());
    for (Type type : nested->getResultTypes())
      if (auto state = dyn_cast<StateType>(type))
        appendState(state.getResource());
    if (FactoryModelOp model = scheduleFactoryModel(nested))
      appendOpaque(scheduleFactoryResource(model));
    if (auto binding = schedulePhysicalBindingResource(nested))
      appendOpaque(*binding);
    if (auto transport = dyn_cast<TransportResourceOp>(nested);
        transport && transport.getModelAttr()) {
      schedule.emitOpError(
          "call-template resource proof does not support model-bound "
          "transport; keep the modeled transport call expanded");
      derived = failure();
      return WalkResult::interrupt();
    }
    if (auto barrier = dyn_cast<BarrierOp>(nested)) {
      auto domains = barrier.getDomainsAttr();
      summary.synchronizesClockDomain |=
          barrier.getNumOperands() == 0 && domains &&
          llvm::any_of(domains, [](Attribute raw) {
            auto domain = dyn_cast<StringAttr>(raw);
            return domain && domain.getValue() == "clock";
          });
    }

    auto invocation = dyn_cast<CallTemplateOp>(nested);
    if (!invocation)
      return WalkResult::advance();
    auto nestedCanonical =
        scheduleCanonicalCall(schedule, canonicalCalls, invocation);
    auto aliases = scheduleStateAliasAttributes(schedule, invocation);
    if (failed(nestedCanonical) || failed(aliases)) {
      derived = failure();
      return WalkResult::interrupt();
    }
    auto nestedSummary = deriveCanonicalScheduleResources(
        schedule, graph, *nestedCanonical, canonicalCalls, cache,
        activeCanonicalCalls);
    if (failed(nestedSummary)) {
      derived = failure();
      return WalkResult::interrupt();
    }
    // The nested summary is expressed in its canonical call's state-resource
    // namespace. Project it through this invocation before retaining it in
    // the enclosing canonical summary. Repeating this at every recursion
    // level composes state_aliases without trusting scheduler-owned effects.
    for (FlatSymbolRefAttr resource : (*nestedSummary)->stateResources) {
      auto replacement = aliases->find(resource);
      appendState(replacement == aliases->end()
                      ? resource
                      : cast<FlatSymbolRefAttr>(replacement->second));
    }
    for (const std::string &resource : (*nestedSummary)->opaqueResources)
      appendOpaque(resource);
    summary.synchronizesClockDomain |=
        (*nestedSummary)->synchronizesClockDomain;
    return WalkResult::advance();
  });
  if (walked.wasInterrupted() || failed(derived))
    return failure();

  llvm::sort(summary.stateResources,
             [](FlatSymbolRefAttr left, FlatSymbolRefAttr right) {
               return left.getValue() < right.getValue();
             });
  llvm::sort(summary.opaqueResources);
  auto [inserted, didInsert] =
      cache.try_emplace(canonical.getOperation(), std::move(summary));
  assert(didInsert && "canonical schedule-resource cache changed mid-proof");
  return &inserted->second;
}

static FailureOr<ArrayRef<StringRef>> expectedScheduleResources(
    ScheduleOp schedule, Operation *operation, StringRef eventId,
    llvm::StringMap<std::string> &resourceKeys,
    llvm::StringMap<uint64_t> &transportOccurrences,
    ArrayRef<std::string> concreteResourceKeys,
    ArrayRef<std::string> factoryResourceKeys,
    llvm::UniqueStringSaver &resourceStrings,
    const ScheduleCanonicalCallIndex &canonicalCalls,
    CanonicalScheduleResourceCache &canonicalResourceCache,
    ExpectedCallTemplateResourceCache &callTemplateResourceCache,
    std::deque<SmallVector<StringRef, 4>> &resourceStorage,
    ScheduleVerifierProfileStats *profileStats = nullptr) {
  if (profileStats)
    ++profileStats->expectedResourceCalls;
  SmallVector<StringRef, 4> result;
  llvm::SmallDenseSet<StringRef, 16> seenResources;
  auto invocation = dyn_cast<CallTemplateOp>(operation);
  CallOp canonical;
  const CanonicalScheduleResourceSummary *invocationSummary = nullptr;
  ExpectedCallTemplateResourceKey cacheKey;
  DenseMap<Attribute, Attribute> stateAliases;
  if (invocation) {
    auto resolved = scheduleCanonicalCall(schedule, canonicalCalls, invocation);
    if (failed(resolved))
      return failure();
    canonical = *resolved;
    auto aliases = scheduleStateAliasAttributes(schedule, invocation);
    if (failed(aliases))
      return failure();
    stateAliases = std::move(*aliases);
    GraphOp graph = invocation->getParentOfType<GraphOp>();
    if (!graph)
      return schedule.emitOpError("call-template graph event '")
             << eventId << "' is not nested in a phys.graph";
    DenseSet<Operation *> activeCanonicalCalls;
    auto summary = deriveCanonicalScheduleResources(
        schedule, graph, canonical, canonicalCalls, canonicalResourceCache,
        activeCanonicalCalls);
    if (failed(summary))
      return failure();
    invocationSummary = *summary;
    // State aliases cannot change a machine-global clock barrier's concrete
    // resource closure.  Key that exact derived fact by its canonical call so
    // every invocation shares one immutable proof vector.
    cacheKey = {canonical.getOperation(),
                invocationSummary->synchronizesClockDomain
                    ? Attribute()
                    : invocation.getStateAliasesAttr()};
    auto cached = callTemplateResourceCache.find(cacheKey);
    if (cached != callTemplateResourceCache.end()) {
      if (profileStats)
        ++profileStats->expectedResourceCacheHits;
      return cached->second;
    }
    if (profileStats)
      ++profileStats->expectedResourceCacheBuilds;
  }
  auto appendResource = [&](StringRef key) {
    if (profileStats)
      ++profileStats->expectedResourceHashProbes;
    StringRef retained = resourceStrings.save(key);
    if (!seenResources.insert(retained).second)
      return;
    result.push_back(retained);
  };
  auto appendStateResource =
      [&](FlatSymbolRefAttr resourceRef,
          Attribute diagnosticResource) -> LogicalResult {
    if (profileStats)
      ++profileStats->expectedResourceTypes;
    if (auto replacement = stateAliases.find(resourceRef);
        replacement != stateAliases.end())
      resourceRef = cast<FlatSymbolRefAttr>(replacement->second);
    StringRef resourceName = resourceRef.getValue();
    auto found = resourceKeys.find(resourceName);
    if (found == resourceKeys.end()) {
      auto resource = dyn_cast_or_null<ResourceOp>(
          SymbolTable::lookupNearestSymbolFrom(operation, resourceRef));
      if (!resource)
        return schedule.emitOpError("cannot resolve physical resource ")
               << diagnosticResource << " used by graph event '" << eventId
               << "'";
      std::string key = (resource.getResourceClass() + "[" +
                         std::to_string(resource.getIndex()) + "]")
                            .str();
      found = resourceKeys.try_emplace(resourceName, std::move(key)).first;
    }
    appendResource(found->second);
    return success();
  };
  auto appendState = [&](Type type) -> LogicalResult {
    auto state = dyn_cast<StateType>(type);
    return state ? appendStateResource(state.getResource(), state.getResource())
                 : success();
  };
  for (Type type : operation->getOperandTypes())
    if (failed(appendState(type)))
      return failure();
  for (Type type : operation->getResultTypes())
    if (failed(appendState(type)))
      return failure();
  if (FactoryModelOp model = scheduleFactoryModel(operation))
    appendResource(scheduleFactoryResource(model));
  if (isOperandFreeFactoryBarrier(operation))
    for (const std::string &resource : factoryResourceKeys)
      appendResource(resource);
  if (auto transport = dyn_cast<TransportResourceOp>(operation)) {
    auto reference = transport.getModelAttr();
    auto model =
        reference
            ? dyn_cast_or_null<TransportModelOp>(
                  SymbolTable::lookupNearestSymbolFrom(transport, reference))
            : TransportModelOp{};
    if (reference && !model)
      return schedule.emitOpError("cannot resolve transport model for event '")
             << eventId << "'";
    SymbolRefAttr bindingReference =
        model ? model.getQecBindingAttr() : transport.getRouteAttr();
    Operation *architecture = bindingReference
                                  ? SymbolTable::lookupSymbolIn(
                                        operation->getParentOfType<ModuleOp>(),
                                        bindingReference.getRootReference())
                                  : nullptr;
    auto binding =
        architecture
            ? dyn_cast_or_null<QECChannelBindingOp>(
                  SymbolTable(architecture)
                      .lookup(bindingReference.getLeafReference().getValue()))
            : QECChannelBindingOp{};
    ArrayAttr claims =
        model ? model.getResourceClaimsAttr()
              : (binding ? binding.getTransportClaimsAttr() : ArrayAttr{});
    if (claims) {
      StringRef identity = model ? model.getSymName() : binding.getSymName();
      uint64_t occurrence = transportOccurrences[identity]++;
      for (Attribute raw : claims) {
        auto claim = cast<DictionaryAttr>(raw);
        StringRef resource =
            model ? claim.getAs<SymbolRefAttr>("resource_class")
                        .getLeafReference()
                        .getValue()
                  : claim.getAs<FlatSymbolRefAttr>("resource_class").getValue();
        int64_t offset = claim.getAs<IntegerAttr>("offset").getInt();
        int64_t count = claim.getAs<IntegerAttr>("count").getInt();
        int64_t units = claim.getAs<IntegerAttr>("units").getInt();
        int64_t lanes = count / units;
        int64_t selected =
            offset + static_cast<int64_t>(occurrence % lanes) * units;
        for (int64_t index = selected; index < selected + units; ++index)
          appendResource((resource + "[" + std::to_string(index) + "]").str());
      }
      auto qecReference = binding.getQecChannelAttr();
      Operation *qecMachine =
          SymbolTable::lookupSymbolIn(operation->getParentOfType<ModuleOp>(),
                                      qecReference.getRootReference());
      auto interconnect =
          qecMachine
              ? dyn_cast_or_null<qlx::fabric::InterconnectOp>(
                    SymbolTable(qecMachine)
                        .lookup(qecReference.getLeafReference().getValue()))
              : qlx::fabric::InterconnectOp{};
      if (!interconnect)
        return schedule.emitOpError(
                   "cannot resolve selected interconnect for transport event '")
               << eventId << "'";
      int64_t sourceUnits =
          model ? model.getSourceEndpointOccupancy()
                : binding.getSourceEndpointOccupancyAttr().getInt();
      int64_t destinationUnits =
          model ? model.getDestinationEndpointOccupancy()
                : binding.getDestinationEndpointOccupancyAttr().getInt();
      auto appendPort = [&](StringRef name, int64_t capacity, int64_t units) {
        int64_t lanes = capacity / units;
        int64_t first = static_cast<int64_t>(occurrence % lanes) * units;
        for (int64_t lane = first; lane < first + units; ++lane)
          appendResource(
              ("control:transport-port:" + name + ":" + std::to_string(lane))
                  .str());
      };
      appendPort(*interconnect.getPortAName(),
                 interconnect.getPortAConcurrency().value_or(1), sourceUnits);
      appendPort(*interconnect.getPortBName(),
                 interconnect.getPortBConcurrency().value_or(1),
                 destinationUnits);
      appendResource(("control:transport-channel:" + interconnect.getSymName() +
                      ":" +
                      std::to_string(occurrence %
                                     interconnect.getConcurrency().value_or(1)))
                         .str());
      appendResource(("control:transport-init:" + identity).str());
    }
  }
  if (isa<qlx::cflow::RepeatOp>(operation)) {
    LogicalResult nestedResources = success();
    auto module = operation->getParentOfType<ModuleOp>();
    SymbolTable moduleSymbols(module);
    WalkResult walked = operation->walk([&](Operation *nested) {
      if (FactoryModelOp model = scheduleFactoryModel(nested))
        appendResource(scheduleFactoryResource(model));
      auto spacetimeCall = dyn_cast<SpacetimeCallOp>(nested);
      if (!spacetimeCall)
        return WalkResult::advance();
      auto plan = moduleSymbols.lookup<SpacetimePlanOp>(
          spacetimeCall.getPlanAttr().getValue());
      if (!plan) {
        schedule.emitOpError(
            "cannot resolve nested spacetime plan for repeat event '")
            << eventId << "'";
        nestedResources = failure();
        return WalkResult::interrupt();
      }
      for (SpacetimePhaseOp phase :
           plan.getBody().front().getOps<SpacetimePhaseOp>()) {
        auto phaseResources = expectedSpacetimePhaseResources(schedule, phase);
        if (failed(phaseResources)) {
          nestedResources = failure();
          return WalkResult::interrupt();
        }
        for (const std::string &resource : *phaseResources)
          if (StringRef(resource).starts_with("factory:"))
            appendResource(resource);
      }
      return WalkResult::advance();
    });
    if (walked.wasInterrupted() || failed(nestedResources))
      return failure();
  }
  if (auto invocation = dyn_cast<SpacetimeCallOp>(operation)) {
    auto plan =
        dyn_cast_or_null<SpacetimePlanOp>(SymbolTable::lookupNearestSymbolFrom(
            invocation, invocation.getPlanAttr()));
    if (!plan)
      return schedule.emitOpError("cannot resolve spacetime plan for event '")
             << eventId << "'";
    for (SpacetimePhaseOp phase :
         plan.getBody().front().getOps<SpacetimePhaseOp>()) {
      auto phaseResources = expectedSpacetimePhaseResources(schedule, phase);
      if (failed(phaseResources))
        return failure();
      for (const std::string &resource : *phaseResources)
        appendResource(resource);
    }
  }
  if (auto binding = schedulePhysicalBindingResource(operation))
    appendResource(*binding);
  if (invocation) {
    for (FlatSymbolRefAttr resource : invocationSummary->stateResources)
      if (failed(appendStateResource(resource, resource)))
        return failure();
    for (const std::string &resource : invocationSummary->opaqueResources) {
      appendResource(resource);
    }
    // A zero-operand clock barrier is a machine-wide effect. The native
    // scheduler summarizes it by advancing every concrete resource
    // frontier, so the independently derived template-resource proof must
    // retain those same identities rather than inspecting state types only.
    if (invocationSummary->synchronizesClockDomain)
      for (const std::string &resource : concreteResourceKeys)
        appendResource(resource);
  }
  if (result.empty())
    result.push_back(resourceStrings.save("control:" + eventId));
  llvm::sort(result);
  if (profileStats)
    profileStats->expectedResourceMaxWidth = std::max<uint64_t>(
        profileStats->expectedResourceMaxWidth, result.size());
  resourceStorage.emplace_back(std::move(result));
  ArrayRef<StringRef> stored = resourceStorage.back();
  if (invocation)
    callTemplateResourceCache.try_emplace(cacheKey, stored);
  return stored;
}

using ScheduleStateAliasMap = DenseMap<StringRef, StringRef>;
using ScheduleStateAliasCache = DenseMap<Attribute, ScheduleStateAliasMap>;

static FailureOr<const ScheduleStateAliasMap *>
scheduleStateAliases(ScheduleOp schedule, CallTemplateOp invocation,
                     llvm::StringMap<std::string> &resourceKeys,
                     ScheduleStateAliasCache &cache,
                     ScheduleVerifierProfileStats *profileStats = nullptr) {
  auto aliases = invocation->getAttrOfType<ArrayAttr>("state_aliases");
  if (!aliases) {
    static const ScheduleStateAliasMap empty;
    return &empty;
  }
  auto cached = cache.find(aliases);
  if (cached != cache.end()) {
    if (profileStats)
      ++profileStats->aliasMapCacheHits;
    return &cached->second;
  }
  if (profileStats)
    ++profileStats->aliasMapBuilds;
  ScheduleStateAliasMap result;
  auto keyFor = [&](FlatSymbolRefAttr reference) -> FailureOr<StringRef> {
    auto found = resourceKeys.find(reference.getValue());
    if (found != resourceKeys.end())
      return StringRef(found->second);
    auto resource = dyn_cast_or_null<ResourceOp>(
        SymbolTable::lookupNearestSymbolFrom(invocation, reference));
    if (!resource)
      return schedule.emitOpError("cannot resolve call-template state alias ")
             << reference;
    std::string key = (resource.getResourceClass() + "[" +
                       std::to_string(resource.getIndex()) + "]")
                          .str();
    auto inserted =
        resourceKeys.try_emplace(reference.getValue(), std::move(key)).first;
    return StringRef(inserted->second);
  };
  for (Attribute raw : aliases.getValue()) {
    if (profileStats)
      ++profileStats->aliasMapEntries;
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto templateResource = entry ? entry.getAs<FlatSymbolRefAttr>("template")
                                  : FlatSymbolRefAttr{};
    auto invocationResource =
        entry ? entry.getAs<FlatSymbolRefAttr>("alias") : FlatSymbolRefAttr{};
    if (!templateResource || !invocationResource)
      return schedule.emitOpError(
          "call-template state alias is malformed during schedule proof");
    auto templateKey = keyFor(templateResource);
    auto invocationKey = keyFor(invocationResource);
    if (failed(templateKey) || failed(invocationKey))
      return failure();
    result.try_emplace(*templateKey, *invocationKey);
  }
  auto inserted = cache.try_emplace(aliases, std::move(result)).first;
  return &inserted->second;
}

static std::pair<StringRef, StringRef>
expectedScheduleParent(Operation *operation, GraphOp graph) {
  Operation *child = operation;
  for (Operation *parent = operation->getParentOp();
       parent && parent != graph.getOperation();
       child = parent, parent = parent->getParentOp()) {
    if (!isSchedulableGraphEvent(parent, graph))
      continue;
    auto event = parent->getAttrOfType<StringAttr>("event_id");
    if (!event)
      return {};
    Region *region = child->getParentRegion();
    if (isa<CallOp, qlx::cflow::RepeatOp>(parent))
      return {event.getValue(), "body"};
    if (auto loop = dyn_cast<qlx::cflow::WhileOp>(parent))
      return {event.getValue(),
              region == &loop.getBeforeRegion() ? "condition" : "body"};
    if (auto branch = dyn_cast<qlx::cflow::IfOp>(parent))
      return {event.getValue(),
              region == &branch.getThenRegion() ? "then" : "else"};
    if (auto dispatch = dyn_cast<qlx::event::TryTakeOp>(parent)) {
      if (region == &dispatch.getReady())
        return {event.getValue(), "ready"};
      if (region == &dispatch.getPending())
        return {event.getValue(), "pending"};
      return {event.getValue(), "failed"};
    }
    return {};
  }
  return {};
}

static StringRef scheduleProducerEvent(Value value, GraphOp graph) {
  Operation *producer = scheduleDataProducer(value, graph);
  auto event =
      producer ? producer->getAttrOfType<StringAttr>("event_id") : nullptr;
  return event ? event.getValue() : StringRef();
}

static StringRef whileScheduleCondition(qlx::cflow::WhileOp loop,
                                        GraphOp graph) {
  auto condition = dyn_cast<qlx::cflow::WhileConditionOp>(
      loop.getBeforeRegion().front().getTerminator());
  return condition ? scheduleProducerEvent(condition.getCondition(), graph)
                   : StringRef();
}

static StringRef expectedScheduleCondition(Operation *operation,
                                           GraphOp graph) {
  if (auto branch = dyn_cast<qlx::cflow::IfOp>(operation))
    return scheduleProducerEvent(branch.getCondition(), graph);
  if (auto dispatch = dyn_cast<qlx::event::TryTakeOp>(operation))
    return scheduleProducerEvent(dispatch.getEvent(), graph);
  if (auto loop = dyn_cast<qlx::cflow::WhileOp>(operation))
    return whileScheduleCondition(loop, graph);

  Operation *child = operation;
  for (Operation *parent = operation->getParentOp();
       parent && parent != graph.getOperation();
       child = parent, parent = parent->getParentOp()) {
    Region *region = child->getParentRegion();
    if (auto branch = dyn_cast<qlx::cflow::IfOp>(parent))
      return scheduleProducerEvent(branch.getCondition(), graph);
    if (auto dispatch = dyn_cast<qlx::event::TryTakeOp>(parent)) {
      auto event = dispatch->getAttrOfType<StringAttr>("event_id");
      return event ? event.getValue() : StringRef();
    }
    if (auto loop = dyn_cast<qlx::cflow::WhileOp>(parent)) {
      if (region == &loop.getAfterRegion())
        return whileScheduleCondition(loop, graph);
      continue;
    }
  }
  return {};
}

/// A sparse, nested rollback journal for verifier frontiers.  Each scope
/// records the first prior value only for keys mutated directly in that scope.
/// Nested scopes restore before their selected delta is applied through the
/// parent, so work is proportional to touched keys rather than frontier size.
template <typename Value>
class JournaledStringMap {
public:
  struct Change {
    std::string key;
    std::optional<Value> before;
    std::optional<Value> after;
  };

  class Scope {
  public:
    explicit Scope(JournaledStringMap &owner) : owner(owner) {
      owner.scopes.push_back(this);
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { restore(); }

    SmallVector<Change, 8> takeChangesAndRestore() {
      SmallVector<Change, 8> changes;
      changes.reserve(priors.size());
      for (const auto &[key, before] : priors) {
        auto found = owner.values.find(key);
        std::optional<Value> after;
        if (found != owner.values.end())
          after = found->second;
        if (before.has_value() == after.has_value() &&
            (!before || *before == *after))
          continue;
        changes.push_back(Change{key, before, after});
      }
      llvm::sort(changes, [](const Change &left, const Change &right) {
        return left.key < right.key;
      });
      restore();
      return changes;
    }

  private:
    friend class JournaledStringMap;

    void record(StringRef key) {
      if (!seen.insert(key).second)
        return;
      if (owner.profileStats)
        owner.profileStats->journalKeyBytes += key.size();
      auto found = owner.values.find(key);
      priors.emplace_back(key.str(), found == owner.values.end()
                                         ? std::optional<Value>{}
                                         : std::optional<Value>{found->second});
      if (owner.stats)
        ++owner.stats->frontierJournalTouches;
    }

    void restore() {
      if (!active)
        return;
      assert(!owner.scopes.empty() && owner.scopes.back() == this &&
             "frontier rollback scopes must be nested");
      owner.scopes.pop_back();
      for (auto iterator = priors.rbegin(); iterator != priors.rend();
           ++iterator) {
        if (iterator->second)
          owner.values[iterator->first] = *iterator->second;
        else
          owner.values.erase(iterator->first);
      }
      active = false;
    }

    JournaledStringMap &owner;
    llvm::StringSet<> seen;
    SmallVector<std::pair<std::string, std::optional<Value>>, 8> priors;
    bool active = true;
  };

  explicit JournaledStringMap(
      ScheduleVerificationStats *stats = nullptr,
      ScheduleVerifierProfileStats *profileStats = nullptr)
      : stats(stats), profileStats(profileStats) {}
  JournaledStringMap(const JournaledStringMap &) = delete;
  JournaledStringMap &operator=(const JournaledStringMap &) = delete;
  JournaledStringMap(JournaledStringMap &&) = delete;
  JournaledStringMap &operator=(JournaledStringMap &&) = delete;

  const llvm::StringMap<Value> &items() const { return values; }
  auto find(StringRef key) const { return values.find(key); }
  auto end() const { return values.end(); }
  std::optional<Value> get(StringRef key) const {
    auto found = values.find(key);
    return found == values.end() ? std::optional<Value>{}
                                 : std::optional<Value>{found->second};
  }
  void set(StringRef key, const Value &value) {
    if (!scopes.empty())
      scopes.back()->record(key);
    values[key] = value;
  }
  void erase(StringRef key) {
    if (!scopes.empty())
      scopes.back()->record(key);
    values.erase(key);
  }

private:
  llvm::StringMap<Value> values;
  SmallVector<Scope *, 4> scopes;
  ScheduleVerificationStats *stats = nullptr;
  ScheduleVerifierProfileStats *profileStats = nullptr;
};

struct ScheduleClockState {
  explicit ScheduleClockState(
      ScheduleVerificationStats *stats = nullptr,
      ScheduleVerifierProfileStats *profileStats = nullptr)
      : resourceProducers(stats, profileStats), stats(stats) {}

  struct Delta {
    SmallVector<JournaledStringMap<StringRef>::Change, 8> resources;
    StringRef clockProducer;
    double clockAvailable = 0.0;
    bool clockChanged = false;
  };

  class Scope {
  public:
    explicit Scope(ScheduleClockState &state)
        : state(state), resources(state.resourceProducers),
          clockProducer(state.clockProducer),
          clockAvailable(state.clockAvailable) {}
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    ~Scope() { restore(); }

    Delta takeDeltaAndRestore() {
      Delta delta;
      delta.clockProducer = state.clockProducer;
      delta.clockAvailable = state.clockAvailable;
      delta.clockChanged = state.clockProducer != clockProducer ||
                           state.clockAvailable != clockAvailable;
      if (delta.clockChanged && state.stats)
        ++state.stats->frontierJournalTouches;
      delta.resources = resources.takeChangesAndRestore();
      restoreClock();
      active = false;
      return delta;
    }

  private:
    void restore() {
      if (!active)
        return;
      resources.takeChangesAndRestore();
      restoreClock();
      active = false;
    }
    void restoreClock() {
      state.clockProducer = clockProducer;
      state.clockAvailable = clockAvailable;
    }

    ScheduleClockState &state;
    JournaledStringMap<StringRef>::Scope resources;
    StringRef clockProducer;
    double clockAvailable;
    bool active = true;
  };

  void setResourceProducer(StringRef resource, StringRef producer) {
    if (producer.empty())
      resourceProducers.erase(resource);
    else
      resourceProducers.set(resource, producer);
  }

  JournaledStringMap<StringRef> resourceProducers;
  StringRef clockProducer;
  double clockAvailable = 0.0;
  ScheduleVerificationStats *stats = nullptr;
};

static StringRef scheduleResourceProducer(const ScheduleClockState &state,
                                          StringRef resource) {
  auto producer = state.resourceProducers.find(resource);
  return producer == state.resourceProducers.end() ? StringRef()
                                                   : producer->second;
}

static LogicalResult verifyScheduleDomainDependencies(
    ScheduleOp schedule, GraphOp graph, ArrayRef<ScheduleClaim> entries,
    const llvm::StringMap<unsigned> &entryById,
    llvm::StringMap<std::string> &resourceKeys,
    ScheduleStateAliasCache &stateAliasCache,
    llvm::UniqueStringSaver &resourceStrings,
    ExpectedCallTemplateResourceCache &callTemplateResourceCache,
    std::deque<SmallVector<StringRef, 4>> &resourceStorage,
    MutableArrayRef<ScheduleGraphFacts> graphFacts,
    SmallVectorImpl<double> &clockReadyByEntry,
    ScheduleVerificationStats *stats,
    ScheduleVerifierProfileStats *profileStats) {
  auto dependencySet = [](ArrayRef<StringRef> values) {
    return llvm::SmallDenseSet<StringRef, 8>(values.begin(), values.end());
  };
  SmallVector<std::string, 8> concreteResourceKeys;
  if (auto module = graph->getParentOfType<ModuleOp>())
    module.walk([&](ResourceOp resource) {
      std::string key = (resource.getResourceClass() + "[" +
                         std::to_string(resource.getIndex()) + "]")
                            .str();
      resourceKeys.try_emplace(resource.getSymName(), key);
      concreteResourceKeys.push_back(std::move(key));
    });
  llvm::sort(concreteResourceKeys);
  concreteResourceKeys.erase(
      std::unique(concreteResourceKeys.begin(), concreteResourceKeys.end()),
      concreteResourceKeys.end());
  SmallVector<std::string, 4> factoryResourceKeys;
  graph.walk([&](Operation *operation) {
    if (FactoryModelOp model = scheduleFactoryModel(operation))
      factoryResourceKeys.push_back(scheduleFactoryResource(model));
  });
  llvm::sort(factoryResourceKeys);
  factoryResourceKeys.erase(
      std::unique(factoryResourceKeys.begin(), factoryResourceKeys.end()),
      factoryResourceKeys.end());
  ScheduleCanonicalCallIndex canonicalCalls;
  bool duplicateCanonicalCall = false;
  graph.walk([&](CallOp call) {
    auto event = call.getEventIdAttr();
    if (!event || !canonicalCalls.try_emplace(event.getValue(), call).second)
      duplicateCanonicalCall = true;
  });
  if (duplicateCanonicalCall)
    return schedule.emitOpError(
        "physical graph has missing or duplicate canonical call event IDs");
  struct CallClockEffect {
    SmallVector<std::string, 8> resources;
    bool clock = false;
    double clockAvailabilityOffset = 0.0;
    std::optional<double> globalBarrierFirstUseOffset;
    std::optional<double> globalBarrierAvailabilityOffset;
  };
  struct ActiveGlobalBarrierEffect {
    std::optional<double> firstUse;
    std::optional<double> availability;
  };
  llvm::StringMap<CallClockEffect> callClockEffects;
  CanonicalScheduleResourceCache canonicalResourceCache;
  llvm::StringMap<uint64_t> transportOccurrences;
  SmallVector<ActiveGlobalBarrierEffect *, 4> activeCallEffects;
  auto noteGlobalBarrier = [&](double firstUse, double availability) {
    for (ActiveGlobalBarrierEffect *effect : activeCallEffects) {
      effect->firstUse =
          effect->firstUse ? std::min(*effect->firstUse, firstUse) : firstUse;
      effect->availability = effect->availability
                                 ? std::max(*effect->availability, availability)
                                 : availability;
    }
  };

  std::function<LogicalResult(Block &, ScheduleClockState &)> verifyBlock;
  verifyBlock = [&](Block &block, ScheduleClockState &state) -> LogicalResult {
    for (Operation &operation : block.getOperations()) {
      if (!isSchedulableGraphEvent(&operation, graph))
        continue;
      auto event = operation.getAttrOfType<StringAttr>("event_id");
      assert(event && entryById.contains(event.getValue()) &&
             "graph event coverage was checked before clock analysis");
      StringRef eventId = event.getValue();
      unsigned entryIndex = entryById.lookup(eventId);
      const ScheduleClaim &entry = entries[entryIndex];
      clockReadyByEntry[entryIndex] = state.clockAvailable;
      ScheduleGraphFacts &facts = graphFacts[entryIndex];
      assert(!facts.derived &&
             "stable graph traversal must derive every schedule claim once");
      auto resources = expectedScheduleResources(
          schedule, &operation, eventId, resourceKeys, transportOccurrences,
          concreteResourceKeys, factoryResourceKeys, resourceStrings,
          canonicalCalls, canonicalResourceCache, callTemplateResourceCache,
          resourceStorage, profileStats);
      if (failed(resources))
        return failure();
      facts.resources = std::move(*resources);
      facts.dataDependencies =
          expectedScheduleDataDependencies(&operation, graph, profileStats);
      facts.derived = true;

      SmallVector<StringRef, 4> expected;
      auto appendExpected = [&](StringRef producer) {
        if (!producer.empty() && !llvm::is_contained(expected, producer))
          expected.push_back(producer);
      };
      bool clockBarrier = isOperandFreeClockBarrier(&operation);
      if (clockBarrier) {
        for (const std::string &resource : concreteResourceKeys)
          appendExpected(scheduleResourceProducer(state, resource));
        if (isOperandFreeFactoryBarrier(&operation))
          for (const std::string &resource : factoryResourceKeys)
            appendExpected(scheduleResourceProducer(state, resource));
        appendExpected(state.clockProducer);
      } else {
        appendExpected(state.clockProducer);
      }
      if (dependencySet(entry.domainDependencies) != dependencySet(expected)) {
        auto diagnostic = schedule.emitOpError("schedule event '")
                          << eventId
                          << "' domain_deps must exactly match its graph clock "
                             "frontier (scheduled=";
        llvm::interleaveComma(entry.domainDependencies, diagnostic);
        diagnostic << "; expected=";
        llvm::interleaveComma(expected, diagnostic);
        diagnostic << ")";
        return failure();
      }
      if (state.clockAvailable > entry.start)
        return schedule.emitOpError("schedule event '")
               << eventId << "' starts at " << entry.start
               << " before its graph clock frontier is available at "
               << state.clockAvailable;

      auto forEachPhysicalResource = [&](auto &&action) {
        for (StringRef resource : facts.resources)
          if (!resource.starts_with("control:"))
            action(resource);
      };

      if (auto invocation = dyn_cast<SpacetimeCallOp>(operation)) {
        auto plan = dyn_cast_or_null<SpacetimePlanOp>(
            SymbolTable::lookupNearestSymbolFrom(invocation,
                                                 invocation.getPlanAttr()));
        if (!plan)
          return schedule.emitOpError("spacetime schedule event '")
                 << eventId << "' has no resolved plan";
        llvm::StringMap<StringRef> phaseEvents;
        for (SpacetimePhaseOp phase :
             plan.getBody().front().getOps<SpacetimePhaseOp>()) {
          std::string phaseIdStorage =
              (eventId + "." + phase.getSymName()).str();
          auto scheduledPhase = entryById.find(phaseIdStorage);
          if (scheduledPhase == entryById.end())
            return schedule.emitOpError("schedule omits spacetime phase '")
                   << phaseIdStorage << "'";
          unsigned phaseIndex = scheduledPhase->second;
          const ScheduleClaim &phaseEntry = entries[phaseIndex];
          ScheduleGraphFacts &phaseFacts = graphFacts[phaseIndex];
          if (phaseFacts.derived)
            return schedule.emitOpError(
                "spacetime phase schedule identity is ambiguous");
          auto phaseResources =
              expectedSpacetimePhaseResources(schedule, phase);
          if (failed(phaseResources))
            return failure();
          SmallVector<StringRef, 4> storedPhaseResources;
          for (const std::string &resource : *phaseResources)
            storedPhaseResources.push_back(resourceStrings.save(resource));
          resourceStorage.emplace_back(std::move(storedPhaseResources));
          phaseFacts.resources = resourceStorage.back();
          if (phase.getAfter().empty()) {
            phaseFacts.dataDependencies = facts.dataDependencies;
            clockReadyByEntry[phaseIndex] = state.clockAvailable;
            if (dependencySet(phaseEntry.domainDependencies) !=
                dependencySet(expected))
              return schedule.emitOpError("spacetime phase '")
                     << phaseIdStorage
                     << "' domain_deps must match its call frontier";
          } else {
            for (Attribute raw : phase.getAfter()) {
              auto predecessor = cast<FlatSymbolRefAttr>(raw);
              StringRef predecessorEvent =
                  phaseEvents.lookup(predecessor.getValue());
              if (predecessorEvent.empty())
                return schedule.emitOpError("spacetime phase '")
                       << phaseIdStorage
                       << "' references an unavailable predecessor";
              phaseFacts.dataDependencies.push_back(predecessorEvent);
            }
            if (!phaseEntry.domainDependencies.empty())
              return schedule.emitOpError("spacetime phase '")
                     << phaseIdStorage
                     << "' may inherit domain_deps only at plan entry";
          }
          phaseFacts.derived = true;
          phaseEvents[phase.getSymName()] = entries[phaseIndex].id;
          for (StringRef resource : phaseFacts.resources)
            state.setResourceProducer(resource, entries[phaseIndex].id);
        }
        continue;
      }

      if (auto call = dyn_cast<CallOp>(operation)) {
        ScheduleClockState::Scope bodyScope(state);
        ActiveGlobalBarrierEffect globalEffect;
        activeCallEffects.push_back(&globalEffect);
        LogicalResult verified = verifyBlock(call.getBody().front(), state);
        activeCallEffects.pop_back();
        if (failed(verified))
          return failure();
        ScheduleClockState::Delta delta = bodyScope.takeDeltaAndRestore();
        CallClockEffect effect;
        for (const auto &change : delta.resources)
          effect.resources.push_back(change.key);
        effect.clock = state.clockProducer != delta.clockProducer;
        for (const std::string &resource : effect.resources)
          state.setResourceProducer(resource, eventId);
        if (effect.clock) {
          effect.clockAvailabilityOffset = delta.clockAvailable - entry.start;
          state.clockAvailable = delta.clockAvailable;
          state.clockProducer = eventId;
        }
        if (globalEffect.firstUse)
          effect.globalBarrierFirstUseOffset =
              *globalEffect.firstUse - entry.start;
        if (globalEffect.availability)
          effect.globalBarrierAvailabilityOffset =
              *globalEffect.availability - entry.start;
        callClockEffects[eventId] = std::move(effect);
        continue;
      }
      if (auto invocation = dyn_cast<CallTemplateOp>(operation)) {
        auto effect = callClockEffects.find(invocation.getTemplateEvent());
        if (effect == callClockEffects.end())
          return schedule.emitOpError("call-template schedule event '")
                 << eventId << "' has no earlier canonical clock effect";
        auto aliases = scheduleStateAliases(schedule, invocation, resourceKeys,
                                            stateAliasCache, profileStats);
        if (failed(aliases))
          return failure();
        for (const std::string &resource : effect->second.resources) {
          auto alias = (*aliases)->find(resource);
          StringRef invocationResource = alias == (*aliases)->end()
                                             ? StringRef(resource)
                                             : StringRef(alias->second);
          state.setResourceProducer(invocationResource, eventId);
        }
        if (effect->second.globalBarrierFirstUseOffset &&
            effect->second.globalBarrierAvailabilityOffset) {
          noteGlobalBarrier(
              entry.start + *effect->second.globalBarrierFirstUseOffset,
              entry.start + *effect->second.globalBarrierAvailabilityOffset);
          state.clockAvailable =
              entry.start + *effect->second.globalBarrierAvailabilityOffset;
          state.clockProducer = eventId;
          for (const std::string &resource : concreteResourceKeys)
            state.setResourceProducer(resource, eventId);
        } else if (effect->second.clock) {
          state.clockAvailable =
              entry.start + effect->second.clockAvailabilityOffset;
          state.clockProducer = eventId;
        }
        continue;
      }
      if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
        ScheduleClockState::Scope bodyScope(state);
        if (failed(verifyBlock(repeat.getBody().front(), state)))
          return failure();
        ScheduleClockState::Delta delta = bodyScope.takeDeltaAndRestore();
        if (repeat.getCountAttr().getInt() > 0) {
          for (const auto &change : delta.resources)
            state.setResourceProducer(change.key, eventId);
          if (state.clockProducer != delta.clockProducer) {
            state.clockProducer = eventId;
            state.clockAvailable = entry.finish();
          }
        }
        continue;
      }
      if (auto loop = dyn_cast<qlx::cflow::WhileOp>(operation)) {
        ScheduleClockState::Scope loopScope(state);
        if (failed(verifyBlock(loop.getBeforeRegion().front(), state)) ||
            failed(verifyBlock(loop.getAfterRegion().front(), state)))
          return failure();
        ScheduleClockState::Delta delta = loopScope.takeDeltaAndRestore();
        for (const auto &change : delta.resources)
          state.setResourceProducer(change.key, eventId);
        if (state.clockProducer != delta.clockProducer) {
          state.clockProducer = eventId;
          state.clockAvailable = entry.finish();
        }
        continue;
      }
      if (auto branch = dyn_cast<qlx::cflow::IfOp>(operation)) {
        SmallVector<ScheduleClockState::Delta, 2> branchDeltas;
        for (Region *region :
             {&branch.getThenRegion(), &branch.getElseRegion()}) {
          ScheduleClockState::Scope branchScope(state);
          if (failed(verifyBlock(region->front(), state)))
            return failure();
          branchDeltas.push_back(branchScope.takeDeltaAndRestore());
        }
        llvm::StringSet<> keys;
        for (const ScheduleClockState::Delta &delta : branchDeltas)
          for (const auto &change : delta.resources)
            keys.insert(change.key);
        auto branchProducer = [&](const ScheduleClockState::Delta &delta,
                                  StringRef key) {
          auto found = llvm::lower_bound(
              delta.resources, key, [](const auto &change, StringRef value) {
                return StringRef(change.key) < value;
              });
          return found != delta.resources.end() && StringRef(found->key) == key
                     ? found->after.value_or(StringRef())
                     : scheduleResourceProducer(state, key);
        };
        for (const auto &key : keys) {
          StringRef producer = branchProducer(branchDeltas[0], key.getKey());
          bool same = llvm::all_of(branchDeltas, [&](const auto &candidate) {
            return branchProducer(candidate, key.getKey()) == producer;
          });
          state.setResourceProducer(key.getKey(), same ? producer : eventId);
        }
        StringRef mergedClock = branchDeltas[0].clockProducer;
        state.clockProducer =
            llvm::all_of(branchDeltas,
                         [&](const auto &candidate) {
                           return candidate.clockProducer == mergedClock;
                         })
                ? mergedClock
                : eventId;
        state.clockAvailable = std::max(branchDeltas[0].clockAvailable,
                                        branchDeltas[1].clockAvailable);
        forEachPhysicalResource([&](StringRef resource) {
          state.setResourceProducer(resource, eventId);
        });
        continue;
      }
      if (auto dispatch = dyn_cast<qlx::event::TryTakeOp>(operation)) {
        SmallVector<ScheduleClockState::Delta, 3> branchDeltas;
        for (Region *region : {&dispatch.getReady(), &dispatch.getPending(),
                               &dispatch.getFailed()}) {
          ScheduleClockState::Scope branchScope(state);
          if (failed(verifyBlock(region->front(), state)))
            return failure();
          branchDeltas.push_back(branchScope.takeDeltaAndRestore());
        }
        llvm::StringSet<> keys;
        for (const ScheduleClockState::Delta &delta : branchDeltas)
          for (const auto &change : delta.resources)
            keys.insert(change.key);
        auto branchProducer = [&](const ScheduleClockState::Delta &delta,
                                  StringRef key) {
          auto found = llvm::lower_bound(
              delta.resources, key, [](const auto &change, StringRef value) {
                return StringRef(change.key) < value;
              });
          return found != delta.resources.end() && StringRef(found->key) == key
                     ? found->after.value_or(StringRef())
                     : scheduleResourceProducer(state, key);
        };
        for (const auto &key : keys) {
          StringRef producer = branchProducer(branchDeltas[0], key.getKey());
          bool same = llvm::all_of(branchDeltas, [&](const auto &candidate) {
            return branchProducer(candidate, key.getKey()) == producer;
          });
          state.setResourceProducer(key.getKey(), same ? producer : eventId);
        }
        StringRef mergedClock = branchDeltas[0].clockProducer;
        bool sameClock = llvm::all_of(branchDeltas, [&](const auto &candidate) {
          return candidate.clockProducer == mergedClock;
        });
        state.clockProducer = sameClock ? mergedClock : eventId;
        // Event dispatch cannot expose a branch-specific clock transition
        // before the selected ready/pending/failed path has completed.  The
        // scheduler therefore promotes a differing branch clock producer to
        // the dispatch envelope finish.  Preserve an unchanged incoming
        // frontier when every branch agrees.
        state.clockAvailable = sameClock
                                   ? std::max({branchDeltas[0].clockAvailable,
                                               branchDeltas[1].clockAvailable,
                                               branchDeltas[2].clockAvailable})
                                   : entry.finish();
        forEachPhysicalResource([&](StringRef resource) {
          state.setResourceProducer(resource, eventId);
        });
        continue;
      }

      forEachPhysicalResource([&](StringRef resource) {
        state.setResourceProducer(resource, eventId);
      });
      if (clockBarrier) {
        noteGlobalBarrier(entry.start, entry.finish());
        state.clockProducer = eventId;
        state.clockAvailable = entry.finish();
      }
    }
    return success();
  };

  ScheduleClockState state(stats, profileStats);
  assert(clockReadyByEntry.size() == entries.size() &&
         "clock-readiness output must cover every schedule entry");
  return verifyBlock(graph.getBody().front(), state);
}

namespace {

/// One semantic implementation shared by portable and native schedule claims.
/// The forwarding accessors keep the proof body expressed in the same terms as
/// the ScheduleOp verifier while allowing its row source to remain neutral.
class ScheduleClaimVerifier {
public:
  explicit ScheduleClaimVerifier(ScheduleOp schedule) : schedule(schedule) {}

  LogicalResult verify(ArrayRef<ScheduleClaim> entries,
                       ScheduleVerificationStats *stats);

private:
  operator ScheduleOp() const { return schedule; }
  operator Operation *() { return schedule.getOperation(); }
  Operation *operator->() { return schedule.getOperation(); }
  Operation *getOperation() { return schedule.getOperation(); }
  InFlightDiagnostic emitOpError(const Twine &message = {}) {
    return schedule.emitOpError(message);
  }
  auto getGraphAttr() { return schedule.getGraphAttr(); }
  auto getStrategy() { return schedule.getStrategy(); }
  auto getStrategyDomain() { return schedule.getStrategyDomain(); }
  auto getProvider() { return schedule.getProvider(); }
  auto getProviderVersion() { return schedule.getProviderVersion(); }
  auto getConstraintProfile() { return schedule.getConstraintProfile(); }
  auto getConstraints() { return schedule.getConstraints(); }
  auto getTimingProfile() { return schedule.getTimingProfile(); }
  auto getTieBreak() { return schedule.getTieBreak(); }
  auto getOptimizationStatus() { return schedule.getOptimizationStatus(); }
  auto getObjectiveValueAttr() { return schedule.getObjectiveValueAttr(); }
  auto getMakespanNs() { return schedule.getMakespanNs(); }

  ScheduleOp schedule;
};

} // namespace

LogicalResult ScheduleClaimVerifier::verify(ArrayRef<ScheduleClaim> entries,
                                            ScheduleVerificationStats *stats) {
  // Schedule verification resolves the same module-level graph, plans, and
  // physical resources for many thousands of rows.  The static symbol lookup
  // helper constructs transient symbol tables on every call; retain one lazy
  // collection for the whole proof so repeated state-resource lookups remain
  // exact while becoming amortized constant-time.
  SymbolTableCollection symbolTables;
  const bool profileEnabled = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
  ScheduleVerifierProfileStats profileStorage;
  ScheduleVerifierProfileStats *profileStats =
      profileEnabled ? &profileStorage : nullptr;
  PhysVerifyClock::time_point phaseStarted = PhysVerifyClock::now();
  auto reportPhase = [&](StringRef name) {
    if (!profileEnabled)
      return;
    llvm::errs() << "phys-schedule: verifier-phase " << name << ' '
                 << physVerifySecondsSince(phaseStarted) << "s\n";
    phaseStarted = PhysVerifyClock::now();
  };
  if (stats) {
    ++stats->semanticVerifierRuns;
    stats->claimsVisited += entries.size();
  }
  auto verifyList = [&](ArrayRef<StringRef> values,
                        StringRef field) -> LogicalResult {
    bool sorted = true;
    bool hasPrevious = false;
    StringRef previous;
    for (StringRef value : values) {
      if (value.empty())
        return emitOpError()
               << "schedule " << field << " must not contain empty entries";
      if (hasPrevious) {
        if (value == previous)
          return emitOpError()
                 << "schedule " << field << " must contain unique entries";
        sorted &= previous < value;
      }
      previous = value;
      hasPrevious = true;
    }
    // Native schedule resource claims are emitted in canonical label order.
    // Prove uniqueness with adjacent comparisons on that common path and
    // retain the general order-insensitive verifier for authored schedules.
    if (sorted)
      return success();
    llvm::SmallDenseSet<StringRef, 8> unique;
    for (StringRef value : values) {
      if (!unique.insert(value).second)
        return emitOpError()
               << "schedule " << field << " must contain unique entries";
    }
    return success();
  };
  for (const ScheduleClaim &entry : entries) {
    if (entry.id.empty() || entry.kind.empty())
      return emitOpError("schedule event_id and kind must be nonempty");
    if (!std::isfinite(entry.start) || !std::isfinite(entry.duration) ||
        entry.start < 0.0 || entry.duration < 0.0)
      return emitOpError(
          "schedule start and duration must be finite nonnegative numbers");
    if (failed(verifyList(entry.resources, "resources")) ||
        failed(verifyList(entry.dependencies, "dependencies")) ||
        failed(verifyList(entry.dataDependencies, "data dependencies")) ||
        failed(
            verifyList(entry.resourceDependencies, "resource dependencies")) ||
        failed(verifyList(entry.domainDependencies, "domain dependencies")))
      return failure();
    if (entry.successProbability &&
        (!std::isfinite(*entry.successProbability) ||
         *entry.successProbability <= 0.0 || *entry.successProbability > 1.0))
      return emitOpError("schedule success_probability must lie in (0, 1]");
  }
  auto graph = dyn_cast_or_null<GraphOp>(
      symbolTables.lookupNearestSymbolFrom(*this, getGraphAttr()));
  if (!graph)
    return emitOpError("graph must resolve to phys.graph");
  Operation *unserializableEvent = nullptr;
  StringRef unserializableEventId;
  graph.walk([&](Operation *operation) {
    if (unserializableEvent || !isSchedulableGraphEvent(operation, graph))
      return;
    auto event = operation->getAttrOfType<StringAttr>("event_id");
    if (event &&
        (event.getValue().contains('|') || event.getValue().contains(','))) {
      unserializableEvent = operation;
      unserializableEventId = event.getValue();
    }
  });
  if (unserializableEvent)
    return emitOpError("graph event_id '")
           << unserializableEventId
           << "' contains reserved schedule list delimiter '|' or ','";
  DictionaryAttr committedTiming;
  if (auto pointRef =
          graph->getAttrOfType<FlatSymbolRefAttr>("operating_point")) {
    Operation *resolved = symbolTables.lookupNearestSymbolFrom(graph, pointRef);
    auto point = dyn_cast_or_null<OperatingPointOp>(resolved);
    if (!point || point.getMachineAttr() != graph.getArchitectureAttr())
      return emitOpError(
          "graph operating_point must resolve to the graph architecture's "
          "phys.operating_point");
    committedTiming = resolved->getAttrOfType<DictionaryAttr>("timing");
  }
  if (getStrategy() != "greedy_asap")
    return emitOpError(
        "strategy must be the executable 'greedy_asap' strategy");
  if (getStrategyDomain() != "physical")
    return emitOpError("strategy_domain must be 'physical'");
  if (getProvider() != "qlx.compiler.greedy_asap")
    return emitOpError("provider must be 'qlx.compiler.greedy_asap'");
  if (getProviderVersion() != "1")
    return emitOpError("provider_version must be '1'");
  if (getConstraintProfile() != "qlx.physical_schedule.constraints/v1")
    return emitOpError(
        "constraint_profile must be 'qlx.physical_schedule.constraints/v1'");
  static constexpr const char *expectedConstraints[] = {
      "graph_ssa_dependencies",   "physical_resource_exclusion",
      "allocation_mapping_after", "structured_control_exclusivity",
      "folded_region_bounds",     "resolved_event_durations",
  };
  if (getConstraints().size() != std::size(expectedConstraints))
    return emitOpError("constraints must name the canonical enforced set");
  for (auto [actual, expected] :
       llvm::zip(getConstraints(), expectedConstraints)) {
    auto text = dyn_cast<StringAttr>(actual);
    if (!text || text.getValue() != expected)
      return emitOpError("constraints must name the canonical enforced set");
  }
  for (NamedAttribute timing : getTimingProfile()) {
    auto value = dyn_cast<FloatAttr>(timing.getValue());
    double resolved = value ? value.getValueAsDouble() : -1.0;
    if (timing.getName().empty() || !value || !std::isfinite(resolved) ||
        resolved < 0.0)
      return emitOpError(
          "timing_profile values must be finite nonnegative f64 values");
  }
  if (getTieBreak() != "stable_graph_order")
    return emitOpError("tie_break must be 'stable_graph_order'");
  if (getOptimizationStatus() != "not_applicable")
    return emitOpError("optimization_status must be 'not_applicable'");
  if (getObjectiveValueAttr())
    return emitOpError(
        "greedy_asap schedules cannot carry an optimization objective_value");
  double makespan = getMakespanNs().convertToDouble();
  if (!std::isfinite(makespan) || makespan < 0.0)
    return emitOpError("makespan_ns must be finite and nonnegative");

  llvm::StringMap<unsigned> entryById;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    if (!entryById.try_emplace(entry.id, index).second)
      return emitOpError("schedule event IDs must be unique; duplicate '")
             << entry.id << "'";
  }

  llvm::StringMap<std::pair<Operation *, std::string>> graphEvents;
  SmallVector<std::string, 32> orderedGraphEventIds;
  llvm::StringMap<Operation *> spacetimePhaseParents;
  bool ambiguousExpectedEvent = false;
  Operation *missingExpectedEvent = nullptr;
  Operation *unsupportedScheduleControl = nullptr;
  graph->walk<WalkOrder::PreOrder>([&](Operation *operation) {
    if (!unsupportedScheduleControl && operation != graph.getOperation() &&
        operation->getNumRegions() != 0 &&
        !isSupportedPhysicalStateRegion(operation))
      unsupportedScheduleControl = operation;
    if (!isSchedulableGraphEvent(operation, graph))
      return;
    StringRef fullName = operation->getName().getStringRef();
    StringRef kind = fullName;
    // `cflow.if`/`cflow.repeat`/`cflow.while` and the shared `event.*`
    // family are embedded here from other dialects; schedule text names
    // them by their bare mnemonic ("if"/"repeat"/"while",
    // "test"/"try_take"/"await"/...), stripped of whichever dialect prefix
    // they carry.
    if (!kind.consume_front("phys.") && !kind.consume_front("cflow."))
      kind.consume_front("event.");
    auto event = operation->getAttrOfType<StringAttr>("event_id");
    if (!event || event.getValue().empty()) {
      missingExpectedEvent = operation;
      return;
    }
    orderedGraphEventIds.push_back(event.getValue().str());
    if (!graphEvents.try_emplace(event.getValue(), operation, kind.str())
             .second)
      ambiguousExpectedEvent = true;
    auto invocation = dyn_cast<SpacetimeCallOp>(operation);
    auto plan = invocation ? dyn_cast_or_null<SpacetimePlanOp>(
                                 SymbolTable::lookupNearestSymbolFrom(
                                     invocation, invocation.getPlanAttr()))
                           : SpacetimePlanOp{};
    if (!plan)
      return;
    for (SpacetimePhaseOp phase :
         plan.getBody().front().getOps<SpacetimePhaseOp>()) {
      std::string phaseId = (event.getValue() + "." + phase.getSymName()).str();
      spacetimePhaseParents[phaseId] = operation;
      orderedGraphEventIds.push_back(phaseId);
      if (!graphEvents
               .try_emplace(phaseId, phase.getOperation(), "spacetime_phase")
               .second)
        ambiguousExpectedEvent = true;
    }
  });
  if (unsupportedScheduleControl)
    return emitOpError(
               "graph event scheduling does not support region control ")
           << unsupportedScheduleControl->getName();
  if (missingExpectedEvent)
    return emitOpError("graph event ") << missingExpectedEvent->getName()
                                       << " is missing its stable event_id";
  if (ambiguousExpectedEvent)
    return emitOpError(
        "graph operations do not have unique schedulable event identities");
  for (const auto &event : graphEvents) {
    auto scheduled = entryById.find(event.getKey());
    if (scheduled == entryById.end())
      return emitOpError("schedule omits graph event '")
             << event.getKey() << "'";
    StringRef expectedKind = event.getValue().second;
    if (entries[scheduled->second].kind != expectedKind)
      return emitOpError("schedule event '")
             << event.getKey() << "' kind must be '" << expectedKind << "'";
  }
  for (const ScheduleClaim &entry : entries)
    if (!graphEvents.contains(entry.id))
      return emitOpError("schedule entry '")
             << entry.id << "' is not backed by an event in graph @"
             << graph.getSymName();
  if (entries.size() != orderedGraphEventIds.size())
    return emitOpError(
        "schedule rows must exactly cover the graph in stable graph order");
  for (auto [index, expected] : llvm::enumerate(orderedGraphEventIds))
    if (entries[index].id != expected)
      return emitOpError(
          "schedule rows must follow deterministic stable graph order");

  auto dependencySet = [](ArrayRef<StringRef> values) {
    return llvm::SmallDenseSet<StringRef, 8>(values.begin(), values.end());
  };
  llvm::StringSet<> usedTimingProfile;
  auto resolvedTiming = [&](StringRef name, StringRef eventId,
                            double committed) -> FailureOr<double> {
    auto value = getTimingProfile().getAs<FloatAttr>(name);
    if (!value) {
      emitOpError("timing_profile is missing enforced key '")
          << name << "' for event '" << eventId << "'";
      return failure();
    }
    if (value.getValueAsDouble() != committed) {
      emitOpError("timing_profile fact '")
          << name << "' for event '" << eventId
          << "' must equal the graph's committed operating-point timing";
      return failure();
    }
    usedTimingProfile.insert(name);
    return committed;
  };
  auto eventTimingName = [](Operation *operation) -> std::string {
    for (StringRef key : {"action", "instrument", "measurement", "route"}) {
      Attribute raw = operation->getAttr(key);
      if (auto symbol = dyn_cast_or_null<FlatSymbolRefAttr>(raw))
        return symbol.getValue().str();
      if (auto text = dyn_cast_or_null<StringAttr>(raw))
        return text.getValue().str();
    }
    StringRef operationName = operation->getName().getStringRef();
    StringRef selected =
        llvm::StringSwitch<StringRef>(operationName)
            .Case("phys.measure_product", "mpp")
            .Case("phys.rotate_product", "rpp")
            .Case("phys.resource_rotate_product", "resource_rpp")
            .Default("");
    if (!selected.empty())
      return selected.str();
    if (operationName.consume_front("phys."))
      return operationName.str();
    return {};
  };
  auto eventCodeDistance = [&](Operation *operation)
      -> FailureOr<std::pair<std::optional<int64_t>, bool>> {
    std::optional<int64_t> selected;
    bool unresolved = false;
    for (Type type : operation->getOperandTypes()) {
      auto state = dyn_cast<StateType>(type);
      if (!state)
        continue;
      auto resource = dyn_cast_or_null<ResourceOp>(
          symbolTables.lookupNearestSymbolFrom(operation, state.getResource()));
      auto distance = resource ? resource.getCodeDistanceAttr() : IntegerAttr{};
      if (!distance) {
        unresolved = true;
        continue;
      }
      if (selected && *selected != distance.getInt()) {
        operation->emitOpError(
            "uses physical states with different code distances; add an "
            "explicit duration_ns for this mixed-distance event");
        return failure();
      }
      selected = distance.getInt();
    }
    return std::pair(selected, unresolved);
  };
  auto hasQualifiedTiming = [&](StringRef action) {
    if (!committedTiming || action.empty())
      return false;
    std::string prefix = (action + "_d").str();
    return llvm::any_of(committedTiming, [&](NamedAttribute entry) {
      StringRef name = entry.getName().getValue();
      return name.starts_with(prefix) && name.ends_with("_ns");
    });
  };
  auto expectedDuration = [&](Operation *operation,
                              StringRef eventId) -> FailureOr<double> {
    if (FactoryModelOp model = scheduleFactoryModel(operation)) {
      if (isa<FactoryStartOp>(operation)) {
        std::string key =
            ("factory_model." + model.getSymName() + ".startup_ns").str();
        return resolvedTiming(key, eventId,
                              model.getStartupNs().convertToDouble());
      }
      if (isa<ResourceRequestOp>(operation)) {
        std::string key =
            ("factory_model." + model.getSymName() + ".output_interval_ns")
                .str();
        auto interval = resolvedTiming(
            key, eventId, model.getOutputIntervalNs().convertToDouble());
        if (failed(interval))
          return failure();
        return 0.0;
      }
    }
    if (auto transport = dyn_cast<TransportResourceOp>(operation))
      if (auto reference = transport.getModelAttr()) {
        auto model = dyn_cast_or_null<TransportModelOp>(
            SymbolTable::lookupNearestSymbolFrom(transport, reference));
        if (!model)
          return emitOpError("event '")
                 << eventId << "' references an unresolved transport model";
        std::string latencyKey =
            ("transport_model." + model.getSymName() + ".latency_ns").str();
        auto latency = resolvedTiming(latencyKey, eventId,
                                      model.getLatencyNs().convertToDouble());
        if (failed(latency))
          return failure();
        std::string intervalKey = ("transport_model." + model.getSymName() +
                                   ".initiation_interval_ns")
                                      .str();
        if (failed(resolvedTiming(
                intervalKey, eventId,
                model.getInitiationIntervalNs().convertToDouble())))
          return failure();
        return *latency;
      } else if (auto route = transport.getRouteAttr()) {
        Operation *architecture = SymbolTable::lookupSymbolIn(
            operation->getParentOfType<ModuleOp>(), route.getRootReference());
        auto binding =
            architecture ? dyn_cast_or_null<QECChannelBindingOp>(
                               SymbolTable(architecture)
                                   .lookup(route.getLeafReference().getValue()))
                         : QECChannelBindingOp{};
        if (binding && binding.getTransportClaimsAttr() && committedTiming) {
          constexpr StringLiteral intervalKey =
              "transport_resource_initiation_interval_ns";
          Attribute raw = committedTiming.get(intervalKey);
          if (raw) {
            auto value = physicalTimingValue(raw);
            if (!value || !std::isfinite(*value) || *value <= 0.0)
              return emitOpError("event '")
                     << eventId
                     << "' has invalid detailed transport initiation timing";
            if (failed(resolvedTiming(intervalKey, eventId, *value)))
              return failure();
          }
        }
      }
    if (Attribute raw = operation->getAttr("duration_ns")) {
      if (auto value = physicalTimingValue(raw))
        return *value;
      return emitOpError("event '")
             << eventId
             << "' duration_ns must be finite, nonnegative, and numeric";
    }
    if (auto delay = dyn_cast<DelayOp>(operation))
      return delay.getDurationNs().convertToDouble();
    if (isa<AcquireOp, ReleaseOp, RetryOp, qlx::event::FenceOp, BarrierOp>(
            operation))
      return 0.0;
    if (isa<CallOp, CallTemplateOp, SpacetimeCallOp, qlx::cflow::RepeatOp,
            qlx::cflow::WhileOp, qlx::cflow::IfOp, qlx::event::TryTakeOp>(
            operation))
      return failure();
    std::string action = eventTimingName(operation);
    auto distanceState = eventCodeDistance(operation);
    if (failed(distanceState))
      return failure();
    auto [distance, unresolvedDistance] = *distanceState;
    if (hasQualifiedTiming(action)) {
      if (!distance || unresolvedDistance)
        return emitOpError("event '")
               << eventId << "' requires an authenticated code distance to "
               << "select timing for '" << action << "'";
      std::string qualified =
          (action + "_d" + std::to_string(*distance) + "_ns");
      Attribute raw = committedTiming.get(qualified);
      if (!raw)
        return emitOpError("event '")
               << eventId << "' has no timing for '" << action
               << "' at code distance " << *distance;
      if (auto value = physicalTimingValue(raw))
        return resolvedTiming(qualified, eventId, *value);
      return emitOpError("graph operating-point timing fact '")
             << qualified << "' must be finite, nonnegative, and numeric";
    }
    std::string actionKey = action + "_ns";
    if (!action.empty() && committedTiming) {
      Attribute raw = committedTiming.get(actionKey);
      if (raw && !physicalTimingValue(raw))
        return emitOpError("graph operating-point timing fact '")
               << actionKey << "' must be finite, nonnegative, and numeric";
      if (auto value = physicalTimingValue(raw))
        return resolvedTiming(actionKey, eventId, *value);
    }
    double cycle = 1.0;
    if (committedTiming) {
      Attribute raw = committedTiming.get("cycle_ns");
      if (raw && !physicalTimingValue(raw))
        return emitOpError(
            "graph operating-point timing fact 'cycle_ns' must be finite, "
            "nonnegative, and numeric");
      if (auto value = physicalTimingValue(raw))
        cycle = *value;
      else {
        raw = committedTiming.get("surface_cycle_ns");
        if (raw && !physicalTimingValue(raw))
          return emitOpError(
              "graph operating-point timing fact 'surface_cycle_ns' must be "
              "finite, nonnegative, and numeric");
        if (auto value = physicalTimingValue(raw))
          cycle = *value;
      }
    }
    return resolvedTiming("cycle_ns", eventId, cycle);
  };
  llvm::StringMap<std::string> resourceKeys;
  ScheduleStateAliasCache stateAliasCache;
  ExpectedCallTemplateResourceCache callTemplateResourceCache;
  std::deque<SmallVector<StringRef, 4>> expectedResourceStorage;
  llvm::BumpPtrAllocator resourceStringAllocator;
  llvm::UniqueStringSaver resourceStrings(resourceStringAllocator);
  SmallVector<ScheduleGraphFacts, 32> graphFacts(entries.size());
  SmallVector<double, 32> clockReadyByEntry(entries.size(), 0.0);
  reportPhase("basic-graph");
  if (failed(verifyScheduleDomainDependencies(
          *this, graph, entries, entryById, resourceKeys, stateAliasCache,
          resourceStrings, callTemplateResourceCache, expectedResourceStorage,
          graphFacts, clockReadyByEntry, stats, profileStats)))
    return failure();
  reportPhase("domain-proof");

  SmallVector<SmallVector<unsigned, 4>, 32> children(entries.size());
  SmallVector<std::optional<unsigned>, 32> parents(entries.size());
  SmallVector<unsigned, 32> roots;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    if (entry.parent.empty()) {
      roots.push_back(index);
      continue;
    }
    auto parent = entryById.find(entry.parent);
    if (parent == entryById.end() || parent->second == index)
      return emitOpError("schedule parent must name another entry");
    parents[index] = parent->second;
    children[parent->second].push_back(index);
  }
  // Stable Euler intervals make the millions of ancestry queries in the
  // resource and dependency proofs constant-time. The intervals are derived
  // from the authenticated parent relation, not from schedule row order.
  SmallVector<unsigned, 32> hierarchyPreorder(
      entries.size(), std::numeric_limits<unsigned>::max());
  SmallVector<unsigned, 32> hierarchyEnd(entries.size(), 0);
  struct HierarchyFrame {
    unsigned index;
    unsigned nextChild = 0;
  };
  SmallVector<HierarchyFrame, 16> hierarchyStack;
  unsigned hierarchyOrdinal = 0;
  for (unsigned root : roots) {
    hierarchyPreorder[root] = hierarchyOrdinal++;
    hierarchyStack.push_back({root, 0});
    while (!hierarchyStack.empty()) {
      HierarchyFrame &frame = hierarchyStack.back();
      if (frame.nextChild == children[frame.index].size()) {
        hierarchyEnd[frame.index] = hierarchyOrdinal;
        hierarchyStack.pop_back();
        continue;
      }
      unsigned child = children[frame.index][frame.nextChild++];
      if (hierarchyPreorder[child] != std::numeric_limits<unsigned>::max())
        return emitOpError("schedule parent hierarchy contains a cycle");
      hierarchyPreorder[child] = hierarchyOrdinal++;
      hierarchyStack.push_back({child, 0});
    }
  }
  if (hierarchyOrdinal != entries.size())
    return emitOpError("schedule parent hierarchy contains a cycle");
  auto entryIsAncestor = [&](unsigned ancestor, unsigned descendant) {
    if (profileStats)
      ++profileStats->ancestryCalls;
    return hierarchyPreorder[ancestor] <= hierarchyPreorder[descendant] &&
           hierarchyPreorder[descendant] < hierarchyEnd[ancestor];
  };
  for (auto [index, entry] : llvm::enumerate(entries)) {
    auto graphEvent = graphEvents.find(entry.id);
    assert(graphEvent != graphEvents.end() &&
           "entry coverage was checked before semantic validation");
    Operation *eventOperation = graphEvent->second.first;
    bool isEnvelope =
        isa<CallOp, CallTemplateOp, SpacetimeCallOp, qlx::cflow::RepeatOp,
            qlx::cflow::WhileOp, qlx::cflow::IfOp, qlx::event::TryTakeOp>(
            eventOperation);
    if (!isEnvelope) {
      FailureOr<double> duration = failure();
      if (auto phase = dyn_cast<SpacetimePhaseOp>(eventOperation))
        duration = phase.getStepDurationNs().convertToDouble() *
                   static_cast<double>(phase.getSteps());
      else
        duration = expectedDuration(eventOperation, entry.id);
      if (failed(duration))
        return failure();
      if (*duration != entry.duration)
        return emitOpError("schedule event '")
               << entry.id
               << "' duration must match its enforced timing profile";
    }
    const ScheduleGraphFacts &facts = graphFacts[index];
    assert(facts.derived &&
           "domain proof must derive every authenticated graph event");
    bool resourcesMatch = llvm::equal(entry.resources, facts.resources);
    if (!resourcesMatch && entry.resources.size() == facts.resources.size()) {
      // Resource claims are set-valued. Native schedules hit the allocation-
      // free canonical-order comparison above; preserve support for authored
      // schedules with arbitrary order through this exact set fallback.
      llvm::SmallDenseSet<StringRef, 8> resolvedResources(
          facts.resources.begin(), facts.resources.end());
      resourcesMatch = llvm::all_of(entry.resources, [&](StringRef resource) {
        return resolvedResources.contains(resource);
      });
    }
    if (!resourcesMatch) {
      auto diagnostic = emitOpError("schedule event '")
                        << entry.id
                        << "' resources must exactly match its resolved "
                           "physical resource identities (scheduled=";
      llvm::interleaveComma(entry.resources, diagnostic);
      diagnostic << "; expected=";
      llvm::interleaveComma(facts.resources, diagnostic);
      diagnostic << ")";
      return failure();
    }

    if (!llvm::equal(entry.dataDependencies, facts.dataDependencies))
      return emitOpError("schedule event '")
             << entry.id
             << "' data_deps must exactly match its graph SSA dependencies "
                "in stable order";

    auto [expectedParent, expectedBranch] =
        expectedScheduleParent(eventOperation, graph);
    if (isa<SpacetimePhaseOp>(eventOperation)) {
      auto parent = spacetimePhaseParents.find(entry.id);
      auto parentEvent =
          parent == spacetimePhaseParents.end()
              ? StringAttr{}
              : parent->getValue()->getAttrOfType<StringAttr>("event_id");
      expectedParent = parentEvent ? parentEvent.getValue() : StringRef{};
      expectedBranch = "phase";
    }
    if (entry.parent != expectedParent || entry.branch != expectedBranch)
      return emitOpError("schedule event '")
             << entry.id
             << "' parent/branch must exactly match graph containment";

    StringRef expectedCondition =
        expectedScheduleCondition(eventOperation, graph);
    if (isa<SpacetimePhaseOp>(eventOperation)) {
      auto parent = spacetimePhaseParents.find(entry.id);
      if (parent != spacetimePhaseParents.end())
        expectedCondition =
            expectedScheduleCondition(parent->getValue(), graph);
    }
    if (entry.condition != expectedCondition)
      return emitOpError("schedule event '")
             << entry.id << "' condition must exactly match graph control";

    if (auto call = dyn_cast<CallOp>(eventOperation)) {
      if (entry.callee != call.getCallee() ||
          entry.instance != call.getInstance())
        return emitOpError("call schedule entry '")
               << entry.id << "' must match graph callee and instance";
    }
    if (auto invocation = dyn_cast<CallTemplateOp>(eventOperation)) {
      if (entry.callee != invocation.getCallee() ||
          entry.instance != invocation.getInstance() ||
          entry.templateEvent != invocation.getTemplateEvent())
        return emitOpError("call-template schedule entry '")
               << entry.id << "' must match graph callee and instance";
      auto canonicalEvent = graphEvents.find(invocation.getTemplateEvent());
      auto canonicalEntry = entryById.find(invocation.getTemplateEvent());
      if (canonicalEvent == graphEvents.end() ||
          !isa<CallOp>(canonicalEvent->second.first) ||
          canonicalEntry == entryById.end() ||
          entries[canonicalEntry->second].duration != entry.duration)
        return emitOpError("call-template schedule entry '")
               << entry.id << "' duration must match its canonical phys.call";
    }
    if (auto invocation = dyn_cast<SpacetimeCallOp>(eventOperation)) {
      auto plan = dyn_cast_or_null<SpacetimePlanOp>(
          SymbolTable::lookupNearestSymbolFrom(invocation,
                                               invocation.getPlanAttr()));
      if (!plan || entry.callee != plan.getSourceProtocol() ||
          entry.instance != invocation.getInstance() ||
          entry.profile != plan.getDerivation())
        return emitOpError("spacetime-call schedule entry '")
               << entry.id
               << "' must match its typed plan, instance, and derivation";
    }
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(eventOperation))
      if (!entry.repeatCount ||
          *entry.repeatCount != repeat.getCountAttr().getInt())
        return emitOpError("repeat schedule entry '")
               << entry.id << "' count must match the graph";
    if (auto loop = dyn_cast<qlx::cflow::WhileOp>(eventOperation)) {
      auto bound = loop.getMaxIterationsAttr();
      if (!bound || !entry.maxIterations ||
          *entry.maxIterations != bound.getInt())
        return emitOpError("while schedule entry '")
               << entry.id << "' bound must match the graph";
    }
    if (auto retry = dyn_cast<RetryOp>(eventOperation)) {
      auto probability = retry.getSuccessProbabilityAttr();
      auto source = retry.getSuccessProbabilitySourceAttr();
      auto evidence = retry.getSuccessProbabilityEvidenceAttr();
      if (!entry.maxAttempts || *entry.maxAttempts != retry.getMaxAttempts() ||
          entry.exhaustion != retry.getExhaustion() ||
          entry.profile != retry.getProfile() ||
          entry.attempt != retry.getAttempt() ||
          entry.attemptEvent != retry.getAttemptEvent() ||
          entry.decisionEvent != retry.getDecisionEvent() ||
          entry.commitPoint != retry.getCommitPoint().value_or("") ||
          static_cast<bool>(entry.successProbability) !=
              static_cast<bool>(probability) ||
          (probability &&
           *entry.successProbability != probability.getValueAsDouble()) ||
          entry.successProbabilitySource !=
              (source ? source.getValue() : StringRef{}) ||
          entry.successProbabilityEvidence !=
              (evidence ? evidence.getValue() : StringRef{}))
        return emitOpError("retry schedule entry '")
               << entry.id << "' policy must match the graph";
    }

    if (static_cast<bool>(entry.callee.empty()) !=
        static_cast<bool>(entry.instance.empty()))
      return emitOpError(
          "schedule callee and instance must be present together");
    if ((entry.kind == "call" || entry.kind == "call_template" ||
         entry.kind == "spacetime_call") &&
        (entry.callee.empty() || entry.instance.empty()))
      return emitOpError("call, call-template, and spacetime-call schedule "
                         "entries require callee and instance");
    if (entry.kind != "call" && entry.kind != "call_template" &&
        entry.kind != "spacetime_call" &&
        (!entry.callee.empty() || !entry.instance.empty()))
      return emitOpError(
          "callee and instance are valid only on call schedule entries");
    if (entry.repeatCount && (entry.kind != "repeat" || *entry.repeatCount < 0))
      return emitOpError(
          "repeat_count must be nonnegative and appear only on repeat entries");
    if (entry.kind == "repeat" && !entry.repeatCount)
      return emitOpError("repeat schedule entries require repeat_count");
    if (entry.repeatPeriod.has_value() != entry.repeatEpilogue.has_value() ||
        ((entry.repeatPeriod || entry.repeatEpilogue) &&
         entry.kind != "repeat"))
      return emitOpError(
          "repeat_period_ns and repeat_epilogue_ns must appear together only "
          "on repeat entries");
    if (entry.kind == "repeat" && !entry.repeatPeriod)
      return emitOpError(
          "repeat schedule entries require period and epilogue evidence");
    if (entry.repeatPeriod && *entry.repeatEpilogue > *entry.repeatPeriod)
      return emitOpError("repeat_epilogue_ns must not exceed repeat_period_ns");
    if (entry.maxIterations &&
        (entry.kind != "while" || *entry.maxIterations <= 0))
      return emitOpError(
          "max_iterations must be positive and appear only on while entries");
    if (entry.kind == "while" && !entry.maxIterations)
      return emitOpError("while schedule entries require max_iterations");
    if (entry.maxAttempts && (entry.kind != "retry" || *entry.maxAttempts <= 0))
      return emitOpError(
          "max_attempts must be positive and appear only on retry entries");
    if (!entry.exhaustion.empty() &&
        (entry.kind != "retry" ||
         (entry.exhaustion != "report_failure" && entry.exhaustion != "abort" &&
          entry.exhaustion != "return_last")))
      return emitOpError("exhaustion must be a valid policy on a retry entry");

    if (entry.parent.empty() != entry.branch.empty())
      return emitOpError(
          "nested schedule entries require parent and branch together");
    if (!entry.parent.empty()) {
      auto parent = entryById.find(entry.parent);
      if (parent == entryById.end() || parent->second == index)
        return emitOpError("schedule parent must name another entry");
      StringRef parentKind = entries[parent->second].kind;
      bool validBranch =
          (parentKind == "if" &&
           (entry.branch == "then" || entry.branch == "else")) ||
          (parentKind == "try_take" &&
           (entry.branch == "ready" || entry.branch == "pending" ||
            entry.branch == "failed")) ||
          ((parentKind == "call" || parentKind == "repeat") &&
           entry.branch == "body") ||
          (parentKind == "spacetime_call" && entry.branch == "phase") ||
          (parentKind == "while" &&
           (entry.branch == "condition" || entry.branch == "body"));
      if (!validBranch)
        return emitOpError(
            "schedule branch is invalid for its structured parent");
      const ScheduleClaim &envelope = entries[parent->second];
      bool zeroCountTemplate = envelope.kind == "repeat" &&
                               envelope.repeatCount &&
                               *envelope.repeatCount == 0;
      bool forwardingPipelinePhase = false;
      if (envelope.kind == "spacetime_call" && entry.branch == "phase") {
        auto parentEvent = graphEvents.find(envelope.id);
        auto invocation =
            parentEvent == graphEvents.end()
                ? SpacetimeCallOp{}
                : dyn_cast<SpacetimeCallOp>(parentEvent->second.first);
        auto plan = invocation ? dyn_cast_or_null<SpacetimePlanOp>(
                                     SymbolTable::lookupNearestSymbolFrom(
                                         invocation, invocation.getPlanAttr()))
                               : SpacetimePlanOp{};
        forwardingPipelinePhase = plan && plan.getForwardingLatencyNsAttr() &&
                                  plan.getInitiationIntervalNsAttr();
      }
      if (entry.start < envelope.start ||
          (!zeroCountTemplate && !forwardingPipelinePhase &&
           entry.finish() > envelope.finish()))
        return emitOpError(
            "nested schedule entry must lie within its parent interval");
    }
    if (!entry.condition.empty() &&
        entryById.find(entry.condition) == entryById.end())
      return emitOpError("schedule condition references unknown event '")
             << entry.condition << "'";

    auto dependencyIsPresent = [&](StringRef dependency) {
      return llvm::is_contained(entry.dependencies, dependency);
    };
    if (!llvm::all_of(entry.dataDependencies, dependencyIsPresent) ||
        !llvm::all_of(entry.resourceDependencies, dependencyIsPresent) ||
        !llvm::all_of(entry.domainDependencies, dependencyIsPresent))
      return emitOpError(
          "data/resource/domain dependencies must be included in dependencies");
    if (entry.hasDataDependencies && entry.hasResourceDependencies) {
      bool exactlyClassified =
          llvm::all_of(entry.dependencies, [&](StringRef dependency) {
            return llvm::is_contained(entry.dataDependencies, dependency) ||
                   llvm::is_contained(entry.resourceDependencies, dependency) ||
                   llvm::is_contained(entry.domainDependencies, dependency);
          });
      if (!exactlyClassified) {
        if (entry.hasDomainDependencies)
          return emitOpError("dependencies must equal the union of data_deps, "
                             "resource_deps, and domain_deps");
        return emitOpError("dependencies must equal the union of data_deps and "
                           "resource_deps");
      }
    }
    for (StringRef dependency : entry.dependencies) {
      auto predecessor = entryById.find(dependency);
      if (predecessor == entryById.end() || predecessor->second == index)
        return emitOpError("schedule dependency '")
               << dependency << "' must name another entry";
      StringRef predecessorKind = entries[predecessor->second].kind;
      bool effectLevelEnvelopeDependency =
          (predecessorKind == "call" || predecessorKind == "call_template" ||
           predecessorKind == "spacetime_call") &&
          !llvm::is_contained(entry.dataDependencies, dependency) &&
          (llvm::is_contained(entry.resourceDependencies, dependency) ||
           llvm::is_contained(entry.domainDependencies, dependency));
      // A compact, canonical, or pipelined spacetime call owns exported
      // physical-resource and clock effects that may become available before
      // the displayed envelope finishes. Those effect-specific times are
      // verified independently by resource replay and graph-clock
      // reconstruction.
      // Physical SSA/data dependencies still require the whole call result.
      if (effectLevelEnvelopeDependency)
        continue;
      bool enclosingDispatch = entryIsAncestor(predecessor->second, index);
      double requiredTime = enclosingDispatch
                                ? entries[predecessor->second].start
                                : entries[predecessor->second].finish();
      if (requiredTime > entry.start)
        return emitOpError("schedule dependency '")
               << dependency << "' does not finish before event '" << entry.id
               << "' starts";
    }
    for (StringRef dependency : entry.resourceDependencies) {
      unsigned predecessor = entryById.find(dependency)->second;
      // Compact/canonical call rows expose only their authored boundary.  A
      // dependency on an internal exported resource is authenticated by the
      // canonical resource-state replay below, not by the envelope row's
      // display resources.
      if (entries[predecessor].kind == "call" ||
          entries[predecessor].kind == "call_template")
        continue;
      if (!llvm::any_of(entry.resources, [&](StringRef resource) {
            return llvm::is_contained(entries[predecessor].resources, resource);
          }))
        return emitOpError("resource dependency '")
               << dependency << "' does not share a resource with event '"
               << entry.id << "'";
    }
  }

  if (usedTimingProfile.size() != getTimingProfile().size())
    return emitOpError("timing_profile must contain only timing keys enforced "
                       "by schedule entries");
  reportPhase("row-contract");

  bool missingAllocationOrder = false;
  llvm::StringMap<SmallVector<StringRef, 4>> allocationAfterDependencies;
  auto module = (*this)->getParentOfType<ModuleOp>();
  module.walk([&](AllocationMappingOp mapping) {
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
      auto scheduledAcquire = entryById.find(acquire.getValue());
      if (scheduledAcquire == entryById.end()) {
        missingAllocationOrder = true;
        continue;
      }
      auto resourceDependencies =
          dependencySet(entries[scheduledAcquire->second].resourceDependencies);
      auto &required = allocationAfterDependencies[acquire.getValue()];
      for (Attribute rawPredecessor : after) {
        auto predecessor = dyn_cast<StringAttr>(rawPredecessor);
        if (!predecessor) {
          missingAllocationOrder = true;
          continue;
        }
        if (!llvm::is_contained(required, predecessor.getValue()))
          required.push_back(predecessor.getValue());
        if (!resourceDependencies.contains(predecessor.getValue()))
          missingAllocationOrder = true;
      }
    }
  });
  if (missingAllocationOrder)
    return emitOpError(
        "schedule resource_deps must retain every allocation-mapping after "
        "edge");

  auto entryIsEnvelope = [](const ScheduleClaim &entry) {
    return entry.kind == "call" || entry.kind == "call_template" ||
           entry.kind == "spacetime_call" || entry.kind == "repeat" ||
           entry.kind == "while" || entry.kind == "if" ||
           entry.kind == "try_take";
  };
  // Authenticate the provider's canonical greedy-ASAP result rather than
  // merely proving that caller-authored rows happen to be legal.  Replay the
  // provider's structured resource-state transitions: calls export their body
  // state, repeats/while loops promote only changed resources to their folded
  // envelope, and exclusive branches merge through their envelope.
  struct ResourcePoint {
    double available = 0.0;
    StringRef producer;
    StringRef pipelinePlan;
    StringRef pipelineInvocation;

    bool operator==(const ResourcePoint &other) const {
      return available == other.available && producer == other.producer &&
             pipelinePlan == other.pipelinePlan &&
             pipelineInvocation == other.pipelineInvocation;
    }
  };
  using ResourceState = JournaledStringMap<ResourcePoint>;
  // Summarize the exact active hierarchy once in child-to-parent order.  A
  // zero-count repeat's body remains independently inspectable, but that
  // repeat and its subtree are inactive relative to its parent.  Direct-child
  // branch labels are sufficient for while condition/body summaries because
  // every deeper descendant inherits that first branch edge.
  SmallVector<double, 32> activeSubtreeFinish(entries.size());
  SmallVector<double, 32> activeDescendantFinish(entries.size());
  SmallVector<double, 32> conditionDescendantFinish(entries.size());
  SmallVector<double, 32> bodyDescendantFinish(entries.size());
  SmallVector<unsigned, 32> remainingChildren(entries.size());
  SmallVector<unsigned, 32> hierarchyReady;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    activeSubtreeFinish[index] = entry.finish();
    activeDescendantFinish[index] = entry.start;
    conditionDescendantFinish[index] = entry.start;
    bodyDescendantFinish[index] = entry.start;
    remainingChildren[index] = children[index].size();
    if (children[index].empty())
      hierarchyReady.push_back(index);
  }
  uint64_t hierarchyVisits = 0;
  while (!hierarchyReady.empty()) {
    unsigned child = hierarchyReady.pop_back_val();
    ++hierarchyVisits;
    if (!parents[child])
      continue;
    unsigned parent = *parents[child];
    const ScheduleClaim &childEntry = entries[child];
    bool inactiveChild =
        childEntry.kind == "repeat" && childEntry.repeatCount.value_or(0) == 0;
    if (!inactiveChild) {
      double finish = activeSubtreeFinish[child];
      activeDescendantFinish[parent] =
          std::max(activeDescendantFinish[parent], finish);
      activeSubtreeFinish[parent] =
          std::max(activeSubtreeFinish[parent], finish);
      if (childEntry.branch == "condition")
        conditionDescendantFinish[parent] =
            std::max(conditionDescendantFinish[parent], finish);
      else if (childEntry.branch == "body")
        bodyDescendantFinish[parent] =
            std::max(bodyDescendantFinish[parent], finish);
    }
    if (--remainingChildren[parent] == 0)
      hierarchyReady.push_back(parent);
  }
  if (hierarchyVisits != entries.size())
    return emitOpError("schedule parent hierarchy contains a cycle");
  if (stats)
    stats->hierarchyPostorderVisits += hierarchyVisits;
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-schedule: verifier-hierarchy entries="
                 << entries.size() << " postorder-visits=" << hierarchyVisits
                 << "\n";

  // Precompute dynamic inactivity in one root-to-leaf traversal for the
  // resource-interval sweep.  This replaces a parent-chain walk per row.
  SmallVector<char, 32> inactive(entries.size(), false);
  SmallVector<unsigned, 32> hierarchyPending(roots.begin(), roots.end());
  while (!hierarchyPending.empty()) {
    unsigned index = hierarchyPending.pop_back_val();
    bool zeroRepeat = entries[index].kind == "repeat" &&
                      entries[index].repeatCount.value_or(0) == 0;
    inactive[index] =
        zeroRepeat || (parents[index] && inactive[*parents[index]]);
    llvm::append_range(hierarchyPending, children[index]);
  }
  reportPhase("allocation-hierarchy");
  SmallVector<SmallVector<StringRef, 4>, 32> expectedResources(entries.size());
  SmallVector<double, 32> expectedStarts(entries.size(), 0.0);
  auto availability = [](const ResourceState &state, StringRef resource) {
    auto found = state.find(resource);
    return found == state.end() ? 0.0 : found->second.available;
  };
  auto dependenciesReady = [&](ArrayRef<StringRef> dependencies,
                               unsigned index) {
    double ready = 0.0;
    for (StringRef dependency : dependencies) {
      unsigned predecessor = entryById.find(dependency)->second;
      ready = std::max(ready, entryIsAncestor(predecessor, index)
                                  ? entries[predecessor].start
                                  : entries[predecessor].finish());
    }
    return ready;
  };
  auto branchChildren = [&](unsigned parent, StringRef branch) {
    SmallVector<unsigned, 8> result;
    for (unsigned child : children[parent])
      if (entries[child].branch == branch)
        result.push_back(child);
    return result;
  };
  auto conditionFinish = [&](unsigned loop) {
    return inactive[loop] ? entries[loop].start
                          : conditionDescendantFinish[loop];
  };

  struct CallResourceAvailabilityEffect {
    std::string resource;
    double offset = 0.0;
    std::string pipelinePlan;
  };
  struct CallPipelineEffect {
    std::string plan;
    double firstUseOffset = 0.0;
    double availabilityOffset = 0.0;
  };
  struct CallResourceEffect {
    SmallVector<std::string, 8> boundaryResources;
    SmallVector<std::pair<std::string, double>, 8> firstUseOffsets;
    SmallVector<CallResourceAvailabilityEffect, 8> availabilityOffsets;
    SmallVector<CallPipelineEffect, 2> pipelineTransitions;
    std::optional<double> globalBarrierFirstUseOffset;
    std::optional<double> globalBarrierAvailabilityOffset;
  };
  struct MappedCallResourceEffect {
    struct Availability {
      StringRef resource;
      double offset = 0.0;
      StringRef pipelinePlan;
    };
    SmallVector<StringRef, 8> boundaryResources;
    SmallVector<std::pair<StringRef, double>, 8> firstUseOffsets;
    SmallVector<Availability, 8> availabilityOffsets;
  };
  SmallVector<std::string, 8> concreteResourceKeys;
  for (const auto &resource : resourceKeys)
    concreteResourceKeys.push_back(resource.getValue());
  llvm::sort(concreteResourceKeys);
  concreteResourceKeys.erase(
      std::unique(concreteResourceKeys.begin(), concreteResourceKeys.end()),
      concreteResourceKeys.end());
  llvm::StringMap<CallResourceEffect> callResourceEffects;
  DenseMap<std::pair<Attribute, Attribute>, unsigned>
      mappedCallResourceEffectIds;
  std::deque<MappedCallResourceEffect> mappedCallResourceEffects;
  ResourceState pipelineInitiations(stats, profileStats);
  std::deque<std::string> pipelineIdentityStorage;
  DenseMap<Operation *, StringRef> pipelineIdentities;
  auto pipelineIdentity = [&](SpacetimePlanOp plan) -> StringRef {
    auto found = pipelineIdentities.find(plan.getOperation());
    if (found != pipelineIdentities.end())
      return found->second;
    std::string value;
    llvm::raw_string_ostream stream(value);
    stream << plan.getArchitecture() << '|' << plan.getOperatingPoint() << '|'
           << plan.getProvider() << '|' << plan.getProviderVersion() << '|'
           << plan.getDerivation() << '|' << plan.getDerivationVersion() << '|'
           << plan.getEvidence() << '|' << plan.getForwardingLatencyNsAttr()
           << '|' << plan.getInitiationIntervalNsAttr() << '|'
           << plan.getModelSha256Attr() << '|' << plan.getModelCommitmentAttr();
    for (SpacetimePhaseOp phase :
         plan.getBody().front().getOps<SpacetimePhaseOp>())
      stream << '|' << phase.getSymName() << ':' << phase.getSteps() << ':'
             << phase.getStepDurationNsAttr() << ':' << phase.getAfterAttr()
             << ':' << phase.getResourceClassesAttr() << ':'
             << phase.getResourceClaimsAttr() << ':'
             << phase.getFactoryModelsAttr();
    pipelineIdentityStorage.push_back(std::move(value));
    StringRef retained = pipelineIdentityStorage.back();
    pipelineIdentities[plan.getOperation()] = retained;
    return retained;
  };
  bool missingCallResourceEffect = false;
  bool invalidRepeatRecurrence = false;

  auto mappedCallResourceEffect = [&](CallTemplateOp invocation,
                                      const CallResourceEffect &effect)
      -> FailureOr<const MappedCallResourceEffect *> {
    std::pair<Attribute, Attribute> key = {
        invocation->getAttr("template_event"),
        invocation->getAttr("state_aliases")};
    auto cached = mappedCallResourceEffectIds.find(key);
    if (cached != mappedCallResourceEffectIds.end())
      return &mappedCallResourceEffects[cached->second];
    auto aliases = scheduleStateAliases(*this, invocation, resourceKeys,
                                        stateAliasCache, profileStats);
    if (failed(aliases))
      return failure();
    auto remap = [&](StringRef resource) -> StringRef {
      auto alias = (*aliases)->find(resource);
      return resourceStrings.save(alias == (*aliases)->end() ? resource
                                                             : alias->second);
    };
    MappedCallResourceEffect mapped;
    for (const std::string &resource : effect.boundaryResources)
      mapped.boundaryResources.push_back(remap(resource));
    for (const auto &[resource, offset] : effect.firstUseOffsets)
      mapped.firstUseOffsets.emplace_back(remap(resource), offset);
    for (const CallResourceAvailabilityEffect &availability :
         effect.availabilityOffsets) {
      mapped.availabilityOffsets.push_back(
          {remap(availability.resource), availability.offset,
           resourceStrings.save(availability.pipelinePlan)});
    }
    unsigned id = mappedCallResourceEffects.size();
    mappedCallResourceEffects.push_back(std::move(mapped));
    mappedCallResourceEffectIds.try_emplace(key, id);
    return &mappedCallResourceEffects.back();
  };

  auto phasePipeline = [&](unsigned index) -> std::pair<StringRef, StringRef> {
    auto operation = graphEvents.find(entries[index].id);
    auto phase = operation == graphEvents.end()
                     ? SpacetimePhaseOp{}
                     : dyn_cast<SpacetimePhaseOp>(operation->second.first);
    auto plan =
        phase ? phase->getParentOfType<SpacetimePlanOp>() : SpacetimePlanOp{};
    if (!plan || !plan.getForwardingLatencyNsAttr() ||
        !plan.getInitiationIntervalNsAttr() || !parents[index])
      return {};
    return {pipelineIdentity(plan), entries[*parents[index]].id};
  };

  // Reconstruct compact first-use offsets exclusively from the canonical
  // schedule rows.  This proof intentionally does not consult scheduler-owned
  // summary state.  A zero-count repeat contributes an inspectable template
  // but no dynamic resource use to its enclosing call.
  std::function<void(unsigned, unsigned, llvm::StringMap<double> &,
                     std::optional<double> &, std::optional<double> &)>
      collectFirstUses;
  collectFirstUses = [&](unsigned index, unsigned origin,
                         llvm::StringMap<double> &uses,
                         std::optional<double> &globalFirstUse,
                         std::optional<double> &globalAvailability) {
    if (profileStats)
      ++profileStats->collectFirstUseVisits;
    const ScheduleClaim &entry = entries[index];
    if (entry.kind == "repeat" && entry.repeatCount.value_or(0) == 0)
      return;
    if (entry.kind == "call_template") {
      auto operation = graphEvents.find(entry.id);
      auto invocation = operation == graphEvents.end()
                            ? CallTemplateOp{}
                            : dyn_cast<CallTemplateOp>(operation->second.first);
      auto effect =
          invocation ? callResourceEffects.find(invocation.getTemplateEvent())
                     : callResourceEffects.end();
      if (effect == callResourceEffects.end()) {
        missingCallResourceEffect = true;
        return;
      }
      auto mapped = mappedCallResourceEffect(invocation, effect->second);
      if (failed(mapped)) {
        missingCallResourceEffect = true;
        return;
      }
      if (invocation->hasAttr("state_boundary_elided"))
        for (StringRef resource : (*mapped)->boundaryResources) {
          double firstUse = entry.start - entries[origin].start;
          auto [found, inserted] = uses.try_emplace(resource, firstUse);
          if (!inserted)
            found->second = std::min(found->second, firstUse);
        }
      for (const auto &[resource, offset] : (*mapped)->firstUseOffsets) {
        double firstUse = entry.start + offset - entries[origin].start;
        auto [found, inserted] = uses.try_emplace(resource, firstUse);
        if (!inserted)
          found->second = std::min(found->second, firstUse);
      }
      if (effect->second.globalBarrierFirstUseOffset) {
        double firstUse = entry.start +
                          *effect->second.globalBarrierFirstUseOffset -
                          entries[origin].start;
        globalFirstUse =
            globalFirstUse ? std::min(*globalFirstUse, firstUse) : firstUse;
      }
      if (effect->second.globalBarrierAvailabilityOffset) {
        double availability = entry.start +
                              *effect->second.globalBarrierAvailabilityOffset -
                              entries[origin].start;
        globalAvailability = globalAvailability
                                 ? std::max(*globalAvailability, availability)
                                 : availability;
      }
      return;
    }
    if (!entryIsEnvelope(entry)) {
      auto operation = graphEvents.find(entry.id);
      if (operation != graphEvents.end() &&
          isOperandFreeClockBarrier(operation->second.first)) {
        double firstUse = entry.start - entries[origin].start;
        globalFirstUse =
            globalFirstUse ? std::min(*globalFirstUse, firstUse) : firstUse;
        double availability = entry.finish() - entries[origin].start;
        globalAvailability = globalAvailability
                                 ? std::max(*globalAvailability, availability)
                                 : availability;
      }
      for (StringRef resource : graphFacts[index].resources) {
        if (resource.starts_with("control:"))
          continue;
        double firstUse = entry.start - entries[origin].start;
        auto [found, inserted] = uses.try_emplace(resource, firstUse);
        if (!inserted)
          found->second = std::min(found->second, firstUse);
      }
      return;
    }
    if (entry.kind == "repeat") {
      std::optional<double> repeatedFirstUse;
      std::optional<double> repeatedAvailability;
      for (unsigned child : children[index])
        collectFirstUses(child, origin, uses, repeatedFirstUse,
                         repeatedAvailability);
      if (repeatedFirstUse)
        globalFirstUse = globalFirstUse
                             ? std::min(*globalFirstUse, *repeatedFirstUse)
                             : repeatedFirstUse;
      if (repeatedAvailability) {
        double availability = *repeatedAvailability;
        if (entry.repeatCount.value_or(0) > 1 && entry.repeatPeriod)
          availability +=
              static_cast<double>(*entry.repeatCount - 1) * *entry.repeatPeriod;
        globalAvailability = globalAvailability
                                 ? std::max(*globalAvailability, availability)
                                 : availability;
      }
      return;
    }
    for (unsigned child : children[index])
      collectFirstUses(child, origin, uses, globalFirstUse, globalAvailability);
  };

  std::function<void(unsigned, unsigned, llvm::StringMap<double> &)>
      collectPipelineUses;
  collectPipelineUses = [&](unsigned index, unsigned origin,
                            llvm::StringMap<double> &uses) {
    const ScheduleClaim &entry = entries[index];
    if (entry.kind == "repeat" && entry.repeatCount.value_or(0) == 0)
      return;
    if (entry.kind == "call_template") {
      auto operation = graphEvents.find(entry.id);
      auto invocation = operation == graphEvents.end()
                            ? CallTemplateOp{}
                            : dyn_cast<CallTemplateOp>(operation->second.first);
      auto effect =
          invocation ? callResourceEffects.find(invocation.getTemplateEvent())
                     : callResourceEffects.end();
      if (effect == callResourceEffects.end()) {
        missingCallResourceEffect = true;
        return;
      }
      for (const CallPipelineEffect &transition :
           effect->second.pipelineTransitions) {
        double firstUse =
            entry.start + transition.firstUseOffset - entries[origin].start;
        auto [found, inserted] = uses.try_emplace(transition.plan, firstUse);
        if (!inserted)
          found->second = std::min(found->second, firstUse);
      }
      return;
    }
    if (entry.kind == "spacetime_call") {
      auto operation = graphEvents.find(entry.id);
      auto invocation =
          operation == graphEvents.end()
              ? SpacetimeCallOp{}
              : dyn_cast<SpacetimeCallOp>(operation->second.first);
      auto plan = invocation ? dyn_cast_or_null<SpacetimePlanOp>(
                                   SymbolTable::lookupNearestSymbolFrom(
                                       invocation, invocation.getPlanAttr()))
                             : SpacetimePlanOp{};
      if (plan && plan.getForwardingLatencyNsAttr() &&
          plan.getInitiationIntervalNsAttr()) {
        double firstUse = entry.start - entries[origin].start;
        auto [found, inserted] =
            uses.try_emplace(pipelineIdentity(plan), firstUse);
        if (!inserted)
          found->second = std::min(found->second, firstUse);
      }
      return;
    }
    for (unsigned child : children[index])
      collectPipelineUses(child, origin, uses);
  };

  std::function<void(ArrayRef<unsigned>, ResourceState &, double)> replay;
  replay = [&](ArrayRef<unsigned> sequence, ResourceState &state,
               double regionEarliest) {
    for (unsigned index : sequence) {
      const ScheduleClaim &entry = entries[index];
      SmallVector<StringRef, 4> resourceDependencies;
      auto appendResourceDependency = [&](StringRef producer) {
        if (!producer.empty() &&
            !llvm::is_contained(resourceDependencies, producer))
          resourceDependencies.push_back(producer);
      };
      auto graphEvent = graphEvents.find(entry.id);
      ArrayRef<StringRef> expectedData = graphFacts[index].dataDependencies;
      double earliest =
          std::max({regionEarliest, dependenciesReady(expectedData, index),
                    clockReadyByEntry[index]});
      if (!entryIsEnvelope(entry)) {
        auto [pipelinePlan, pipelineInvocation] = phasePipeline(index);
        const bool factoryBarrier =
            graphEvent != graphEvents.end() &&
            isOperandFreeFactoryBarrier(graphEvent->second.first);
        if (graphEvent != graphEvents.end() &&
            isOperandFreeClockBarrier(graphEvent->second.first))
          for (const auto &resource : resourceKeys)
            earliest =
                std::max(earliest, availability(state, resource.getValue()));
        for (StringRef resource : graphFacts[index].resources) {
          const bool transportControl =
              resource.starts_with("control:transport-port:") ||
              resource.starts_with("control:transport-init:");
          if (resource.starts_with("control:") && !transportControl)
            continue;
          auto found = state.find(resource);
          bool offsetInitiation =
              resource.starts_with("control:transport-init:");
          bool samePipeline =
              found != state.end() && !pipelinePlan.empty() &&
              found->second.pipelinePlan == pipelinePlan &&
              found->second.pipelineInvocation != pipelineInvocation;
          if (!factoryBarrier && !samePipeline && !offsetInitiation &&
              found != state.end())
            appendResourceDependency(found->second.producer);
          if (!samePipeline)
            earliest = std::max(
                earliest, found == state.end() ? 0.0 : found->second.available);
          state.set(resource, ResourcePoint{entry.finish(), entry.id,
                                            pipelinePlan, pipelineInvocation});
        }
        if (graphEvent != graphEvents.end())
          if (auto request =
                  dyn_cast<ResourceRequestOp>(graphEvent->second.first))
            if (FactoryModelOp model = scheduleFactoryModel(request))
              state.set(scheduleFactoryResource(model),
                        ResourcePoint{
                            entry.start +
                                model.getOutputIntervalNs().convertToDouble(),
                            entry.id});
        if (graphEvent != graphEvents.end())
          if (auto transport =
                  dyn_cast<TransportResourceOp>(graphEvent->second.first)) {
            if (auto reference = transport.getModelAttr())
              if (auto model = dyn_cast_or_null<TransportModelOp>(
                      SymbolTable::lookupNearestSymbolFrom(transport,
                                                           reference)))
                state.set(
                    ("control:transport-init:" + model.getSymName()).str(),
                    ResourcePoint{
                        entry.start +
                            model.getInitiationIntervalNs().convertToDouble(),
                        entry.id});
            if (!transport.getModelAttr())
              if (auto route = transport.getRouteAttr()) {
                Operation *architecture = SymbolTable::lookupSymbolIn(
                    graph.getOperation()->getParentOfType<ModuleOp>(),
                    route.getRootReference());
                auto binding =
                    architecture
                        ? dyn_cast_or_null<QECChannelBindingOp>(
                              SymbolTable(architecture)
                                  .lookup(route.getLeafReference().getValue()))
                        : QECChannelBindingOp{};
                if (binding && binding.getTransportClaimsAttr()) {
                  double interval = entry.duration;
                  if (committedTiming)
                    if (auto value = physicalTimingValue(committedTiming.get(
                            "transport_resource_initiation_interval_ns")))
                      interval = *value;
                  state.set(
                      ("control:transport-init:" + binding.getSymName()).str(),
                      ResourcePoint{entry.start + interval, entry.id});
                }
              }
          }
      } else if (entry.kind == "call") {
        ResourceState::Scope bodyScope(state);
        ResourceState::Scope pipelineScope(pipelineInitiations);
        replay(children[index], state, entry.start);
        auto changes = bodyScope.takeChangesAndRestore();
        auto pipelineChanges = pipelineScope.takeChangesAndRestore();
        auto pointAfter = [&](StringRef resource) {
          auto found = llvm::lower_bound(
              changes, resource, [](const auto &change, StringRef value) {
                return StringRef(change.key) < value;
              });
          if (found != changes.end() && StringRef(found->key) == resource)
            return found->after;
          return state.get(resource);
        };
        llvm::StringSet<> touched;
        // A call envelope owns only resources whose canonical body changes the
        // replay frontier.  Boundary states yielded unchanged remain owned by
        // their prior producer.  Change equality includes both availability
        // and producer, so zero-duration producer transitions are retained.
        for (const auto &change : changes)
          if (change.after)
            touched.insert(change.key);
        SmallVector<std::string, 8> orderedTouched;
        for (const auto &resource : touched)
          orderedTouched.push_back(resource.getKey().str());
        llvm::sort(orderedTouched);
        CallResourceEffect effect;
        if (auto operation = graphEvents.find(entry.id);
            operation != graphEvents.end())
          if (auto call = dyn_cast<CallOp>(operation->second.first))
            for (Type type : call.getInputs().getTypes())
              if (auto stateType = dyn_cast<StateType>(type)) {
                auto key =
                    resourceKeys.find(stateType.getResource().getValue());
                if (key != resourceKeys.end())
                  effect.boundaryResources.push_back(key->second);
              }
        llvm::sort(effect.boundaryResources);
        effect.boundaryResources.erase(
            std::unique(effect.boundaryResources.begin(),
                        effect.boundaryResources.end()),
            effect.boundaryResources.end());
        llvm::StringMap<double> firstUses;
        std::optional<double> globalFirstUse;
        std::optional<double> globalAvailability;
        for (unsigned child : children[index])
          collectFirstUses(child, index, firstUses, globalFirstUse,
                           globalAvailability);
        for (const auto &use : firstUses)
          effect.firstUseOffsets.emplace_back(use.getKey().str(),
                                              use.getValue());
        llvm::sort(effect.firstUseOffsets,
                   [](const auto &left, const auto &right) {
                     return left.first < right.first;
                   });
        llvm::StringMap<double> pipelineFirstUses;
        for (unsigned child : children[index])
          collectPipelineUses(child, index, pipelineFirstUses);
        for (const std::string &resource : orderedTouched) {
          auto found = pointAfter(resource);
          if (!found)
            continue;
          auto previous = state.get(resource);
          double previousAvailable = previous ? previous->available : 0.0;
          StringRef previousProducer =
              previous ? previous->producer : StringRef();
          if (found->available == previousAvailable &&
              found->producer == previousProducer)
            continue;
          effect.availabilityOffsets.push_back({resource,
                                                found->available - entry.start,
                                                found->pipelinePlan.str()});
        }
        llvm::sort(effect.availabilityOffsets,
                   [](const auto &left, const auto &right) {
                     return left.resource < right.resource;
                   });
        for (const auto &change : pipelineChanges) {
          auto firstUse = pipelineFirstUses.find(change.key);
          if (firstUse == pipelineFirstUses.end() || !change.after)
            continue;
          effect.pipelineTransitions.push_back(
              {change.key, firstUse->second,
               change.after->available - entry.start});
        }
        llvm::sort(effect.pipelineTransitions,
                   [](const auto &left, const auto &right) {
                     return left.plan < right.plan;
                   });
        effect.globalBarrierFirstUseOffset = globalFirstUse;
        effect.globalBarrierAvailabilityOffset = globalAvailability;
        for (const std::string &resource : orderedTouched)
          if (auto point = pointAfter(resource);
              point && !point->producer.empty())
            state.set(resource,
                      ResourcePoint{point->available, entry.id,
                                    point->pipelinePlan,
                                    point->pipelinePlan.empty() ? StringRef{}
                                                                : entry.id});
        for (const auto &change : pipelineChanges)
          if (change.after)
            pipelineInitiations.set(
                change.key, ResourcePoint{change.after->available, entry.id});
        callResourceEffects[entry.id] = std::move(effect);
      } else if (entry.kind == "call_template") {
        auto operation = graphEvents.find(entry.id);
        auto invocation =
            operation == graphEvents.end()
                ? CallTemplateOp{}
                : dyn_cast<CallTemplateOp>(operation->second.first);
        auto effect =
            invocation ? callResourceEffects.find(invocation.getTemplateEvent())
                       : callResourceEffects.end();
        if (effect != callResourceEffects.end()) {
          auto mapped = mappedCallResourceEffect(invocation, effect->second);
          if (failed(mapped)) {
            missingCallResourceEffect = true;
            return;
          }
          if (invocation->hasAttr("state_boundary_elided"))
            for (StringRef resource : (*mapped)->boundaryResources) {
              earliest = std::max(earliest, availability(state, resource));
              auto prior = state.find(resource);
              if (prior != state.end())
                appendResourceDependency(prior->second.producer);
            }
          // Replay the scheduler's canonical effect order exactly: explicit
          // boundary ownership, then the unaliased machine-global barrier
          // frontier, followed by ordinary first-use and pipeline effects.
          // The order is observable in the authenticated resource_deps list
          // even when the set of predecessor events is identical.
          if (effect->second.globalBarrierFirstUseOffset) {
            double offset = *effect->second.globalBarrierFirstUseOffset;
            for (const std::string &resource : concreteResourceKeys) {
              earliest =
                  std::max(earliest, availability(state, resource) - offset);
              auto prior = state.find(resource);
              if (offset == 0.0 && prior != state.end())
                appendResourceDependency(prior->second.producer);
            }
          }
          for (const auto &[resource, offset] : (*mapped)->firstUseOffsets) {
            earliest =
                std::max(earliest, availability(state, resource) - offset);
            auto prior = state.find(resource);
            if (offset == 0.0 && prior != state.end())
              appendResourceDependency(prior->second.producer);
          }
          for (const CallPipelineEffect &transition :
               effect->second.pipelineTransitions) {
            auto prior = pipelineInitiations.find(transition.plan);
            if (prior == pipelineInitiations.end())
              continue;
            earliest = std::max(earliest, prior->second.available -
                                              transition.firstUseOffset);
            appendResourceDependency(prior->second.producer);
          }
          for (const MappedCallResourceEffect::Availability
                   &availabilityEffect : (*mapped)->availabilityOffsets) {
            double replayed = entry.start + availabilityEffect.offset;
            auto prior = state.find(availabilityEffect.resource);
            if (prior == state.end() || replayed >= prior->second.available)
              state.set(availabilityEffect.resource,
                        ResourcePoint{replayed, entry.id,
                                      availabilityEffect.pipelinePlan,
                                      availabilityEffect.pipelinePlan.empty()
                                          ? StringRef{}
                                          : entry.id});
          }
          for (const CallPipelineEffect &transition :
               effect->second.pipelineTransitions)
            pipelineInitiations.set(
                transition.plan,
                ResourcePoint{entry.start + transition.availabilityOffset,
                              entry.id});
          if (effect->second.globalBarrierAvailabilityOffset) {
            double replayed =
                entry.start + *effect->second.globalBarrierAvailabilityOffset;
            for (const std::string &resource : concreteResourceKeys) {
              auto prior = state.find(resource);
              if (prior == state.end() || replayed >= prior->second.available)
                state.set(resource, ResourcePoint{replayed, entry.id});
            }
          }
        } else
          missingCallResourceEffect = true;
      } else if (entry.kind == "spacetime_call") {
        auto operation = graphEvents.find(entry.id);
        auto invocation =
            operation == graphEvents.end()
                ? SpacetimeCallOp{}
                : dyn_cast<SpacetimeCallOp>(operation->second.first);
        auto plan = invocation ? dyn_cast_or_null<SpacetimePlanOp>(
                                     SymbolTable::lookupNearestSymbolFrom(
                                         invocation, invocation.getPlanAttr()))
                               : SpacetimePlanOp{};
        bool pipelined = plan && plan.getForwardingLatencyNsAttr() &&
                         plan.getInitiationIntervalNsAttr();
        if (pipelined) {
          StringRef planIdentity = pipelineIdentity(plan);
          auto prior = pipelineInitiations.find(planIdentity);
          if (prior != pipelineInitiations.end()) {
            earliest = std::max(earliest, prior->second.available);
            appendResourceDependency(prior->second.producer);
          }
          for (unsigned child : children[index]) {
            auto childOperation = graphEvents.find(entries[child].id);
            auto phase =
                childOperation == graphEvents.end()
                    ? SpacetimePhaseOp{}
                    : dyn_cast<SpacetimePhaseOp>(childOperation->second.first);
            if (!phase || !phase.getAfter().empty())
              continue;
            for (StringRef resource : graphFacts[child].resources) {
              auto found = state.find(resource);
              bool samePipeline = found != state.end() &&
                                  found->second.pipelinePlan == planIdentity &&
                                  found->second.pipelineInvocation != entry.id;
              if (samePipeline)
                continue;
              earliest = std::max(earliest, availability(state, resource));
              if (found != state.end())
                appendResourceDependency(found->second.producer);
            }
          }
        }
        replay(children[index], state, entry.start);
        if (pipelined)
          pipelineInitiations.set(
              pipelineIdentity(plan),
              ResourcePoint{
                  entry.start +
                      plan.getInitiationIntervalNsAttr().getValueAsDouble(),
                  entry.id});
      } else if (entry.kind == "repeat") {
        for (StringRef resource : graphFacts[index].resources) {
          if (!StringRef(resource).starts_with("factory:"))
            continue;
          auto found = state.find(resource);
          if (found != state.end())
            appendResourceDependency(found->second.producer);
          earliest = std::max(earliest, availability(state, resource));
        }
        ResourceState::Scope bodyScope(state);
        replay(children[index], state, entry.start);
        auto changes = bodyScope.takeChangesAndRestore();
        double expectedPeriod = entry.repeatEpilogue.value_or(0.0);
        for (const auto &change : changes)
          if (change.after)
            expectedPeriod =
                std::max(expectedPeriod, change.after->available - entry.start);
        if (!entry.repeatPeriod || *entry.repeatPeriod != expectedPeriod) {
          invalidRepeatRecurrence = true;
          return;
        }
        if (entry.repeatCount.value_or(0) > 0)
          for (const auto &change : changes)
            if (change.after) {
              auto previous = state.get(change.key);
              if (previous && *change.after == *previous)
                continue;
              double available = entry.finish();
              if (entry.repeatPeriod &&
                  StringRef(change.key).starts_with("factory:"))
                available =
                    entry.start +
                    (entry.repeatCount.value_or(1) - 1) * *entry.repeatPeriod +
                    (change.after->available - entry.start);
              state.set(change.key, ResourcePoint{available, entry.id});
            }
      } else if (entry.kind == "while") {
        ResourceState::Scope loopScope(state);
        SmallVector<unsigned, 8> condition = branchChildren(index, "condition");
        replay(condition, state, entry.start);
        SmallVector<unsigned, 8> bodyChildren = branchChildren(index, "body");
        replay(bodyChildren, state, conditionFinish(index));
        auto changes = loopScope.takeChangesAndRestore();
        for (const auto &change : changes)
          if (change.after) {
            auto previous = state.get(change.key);
            if (previous && *change.after == *previous)
              continue;
            state.set(change.key, ResourcePoint{entry.finish(), entry.id});
          }
      } else {
        SmallVector<StringRef, 3> branches =
            entry.kind == "if"
                ? SmallVector<StringRef, 3>{"then", "else"}
                : SmallVector<StringRef, 3>{"ready", "pending", "failed"};
        SmallVector<SmallVector<ResourceState::Change, 8>, 3> branchDeltas;
        llvm::StringSet<> resources;
        for (StringRef branch : branches) {
          ResourceState::Scope branchScope(state);
          SmallVector<unsigned, 8> branchSequence =
              branchChildren(index, branch);
          replay(branchSequence, state, entry.start);
          branchDeltas.push_back(branchScope.takeChangesAndRestore());
          for (const auto &change : branchDeltas.back())
            resources.insert(change.key);
        }
        auto branchAvailability = [&](ArrayRef<ResourceState::Change> delta,
                                      StringRef resource) {
          auto found = llvm::lower_bound(
              delta, resource, [](const auto &change, StringRef value) {
                return StringRef(change.key) < value;
              });
          return found != delta.end() && StringRef(found->key) == resource &&
                         found->after
                     ? found->after->available
                     : availability(state, resource);
        };
        for (auto &resource : resources) {
          double previous = availability(state, resource.getKey());
          double merged = previous;
          for (const auto &branch : branchDeltas)
            merged =
                std::max(merged, branchAvailability(branch, resource.getKey()));
          if (merged != previous)
            state.set(resource.getKey(), ResourcePoint{merged, entry.id});
        }
        for (StringRef resource : graphFacts[index].resources)
          if (!resource.starts_with("control:"))
            state.set(resource, ResourcePoint{entry.finish(), entry.id});
      }
      auto allocationOrder = allocationAfterDependencies.find(entry.id);
      if (allocationOrder != allocationAfterDependencies.end())
        for (StringRef predecessor : allocationOrder->second)
          appendResourceDependency(predecessor);
      expectedResources[index] = std::move(resourceDependencies);
      expectedStarts[index] = earliest;
    }
  };
  ResourceState rootState(stats, profileStats);
  replay(roots, rootState, 0.0);
  if (missingCallResourceEffect)
    return emitOpError(
        "call-template schedule requires an earlier canonical resource effect");
  if (invalidRepeatRecurrence)
    return emitOpError(
        "repeat schedule period does not match its canonical resource "
        "recurrence");
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-schedule: verifier-frontier state-copies="
                 << (stats ? stats->frontierStateCopies : 0)
                 << " journal-touches="
                 << (stats ? stats->frontierJournalTouches : 0) << "\n";
  reportPhase("resource-replay");

  for (auto [index, entry] : llvm::enumerate(entries)) {
    if (!llvm::equal(entry.dataDependencies,
                     graphFacts[index].dataDependencies))
      return emitOpError("schedule event '")
             << entry.id
             << "' data_deps must exactly match its graph SSA dependencies in "
                "stable order";
    if (!llvm::equal(entry.resourceDependencies, expectedResources[index])) {
      auto diagnostic = emitOpError("schedule event '")
                        << entry.id
                        << "' resource_deps must exactly match the stable "
                           "greedy-ASAP resource predecessors (scheduled=";
      llvm::interleaveComma(entry.resourceDependencies, diagnostic);
      diagnostic << "; expected=";
      llvm::interleaveComma(expectedResources[index], diagnostic);
      diagnostic << ")";
      return failure();
    }
    SmallVector<StringRef, 8> expectedDependencies(
        graphFacts[index].dataDependencies.begin(),
        graphFacts[index].dataDependencies.end());
    for (StringRef dependency : expectedResources[index])
      if (!llvm::is_contained(expectedDependencies, dependency))
        expectedDependencies.push_back(dependency);
    for (StringRef dependency : entry.domainDependencies)
      if (!llvm::is_contained(expectedDependencies, dependency))
        expectedDependencies.push_back(dependency);
    if (!llvm::equal(entry.dependencies, expectedDependencies))
      return emitOpError("schedule event '")
             << entry.id
             << "' dependencies must exactly equal canonical "
                "data/resource/domain dependencies in stable order";
    if (entry.start != expectedStarts[index])
      return emitOpError("schedule event '")
             << entry.id << "' must start at its canonical earliest legal time "
             << expectedStarts[index] << " ns";
  }

  // Authenticate folded structured envelopes relative to each envelope's own
  // start.  Zero-count repeats retain one inspectable template but contribute
  // an exact zero-duration envelope; their inactive children do not extend an
  // enclosing call or branch.
  for (auto [index, entry] : llvm::enumerate(entries)) {
    if (!entryIsEnvelope(entry))
      continue;
    double expectedDuration = 0.0;
    if (entry.kind == "repeat" && entry.repeatCount &&
        *entry.repeatCount == 0) {
      expectedDuration = 0.0;
    } else if (entry.kind == "call_template") {
      auto operation = graphEvents.find(entry.id);
      auto invocation = operation == graphEvents.end()
                            ? CallTemplateOp{}
                            : dyn_cast<CallTemplateOp>(operation->second.first);
      auto canonical = invocation
                           ? entryById.find(invocation.getTemplateEvent())
                           : entryById.end();
      if (canonical == entryById.end())
        return emitOpError("call-template schedule event '")
               << entry.id << "' has no canonical call schedule entry";
      expectedDuration = entries[canonical->second].duration;
    } else if (entry.kind == "spacetime_call") {
      auto operation = graphEvents.find(entry.id);
      auto invocation =
          operation == graphEvents.end()
              ? SpacetimeCallOp{}
              : dyn_cast<SpacetimeCallOp>(operation->second.first);
      auto plan = invocation ? dyn_cast_or_null<SpacetimePlanOp>(
                                   SymbolTable::lookupNearestSymbolFrom(
                                       invocation, invocation.getPlanAttr()))
                             : SpacetimePlanOp{};
      if (!plan)
        return emitOpError("spacetime-call schedule event '")
               << entry.id << "' has no resolved plan";
      if (auto forwarding = plan.getForwardingLatencyNsAttr())
        expectedDuration = forwarding.getValueAsDouble();
      else
        expectedDuration = activeDescendantFinish[index] - entry.start;
    } else if (entry.kind == "while") {
      double conditionFinish = conditionDescendantFinish[index];
      double bodyFinish = bodyDescendantFinish[index];
      bodyFinish = std::max(bodyFinish, conditionFinish);
      int64_t bound = entry.maxIterations.value_or(0);
      double conditionDuration = conditionFinish - entry.start;
      double bodyDuration = bodyFinish - conditionFinish;
      expectedDuration = conditionDuration * (bound + 1) + bodyDuration * bound;
    } else {
      double templateFinish = activeDescendantFinish[index];
      // A structured region may only select or reorder values produced
      // outside the region.  Such a branch has no schedulable descendants,
      // but its results are not ready before the selected value is ready.
      // Authenticate that epilogue wait from the actual yielded SSA values.
      auto graphEvent = graphEvents.find(entry.id);
      Operation *envelopeOperation =
          graphEvent == graphEvents.end() ? nullptr : graphEvent->second.first;
      if (envelopeOperation)
        for (Region &region : envelopeOperation->getRegions()) {
          if (region.empty())
            continue;
          Operation *terminator = region.front().getTerminator();
          for (Value yielded : terminator->getOperands()) {
            if (auto argument = dyn_cast<BlockArgument>(yielded);
                argument &&
                argument.getOwner()->getParentOp() == envelopeOperation)
              continue;
            Operation *producer = scheduleDataProducer(yielded, graph);
            if (!producer || producer == envelopeOperation)
              continue;
            auto producerEvent =
                producer->getAttrOfType<StringAttr>("event_id");
            auto scheduled = producerEvent
                                 ? entryById.find(producerEvent.getValue())
                                 : entryById.end();
            if (scheduled != entryById.end())
              templateFinish =
                  std::max(templateFinish, entries[scheduled->second].finish());
          }
        }
      double templateDuration = templateFinish - entry.start;
      if (entry.kind == "repeat") {
        if (!entry.repeatPeriod || !entry.repeatEpilogue ||
            *entry.repeatEpilogue != templateDuration)
          return emitOpError("repeat schedule event '")
                 << entry.id
                 << "' epilogue must equal the canonical final-iteration "
                    "span";
        int64_t count = entry.repeatCount.value_or(0);
        expectedDuration = count == 0 ? 0.0
                                      : (static_cast<double>(count) - 1.0) *
                                                *entry.repeatPeriod +
                                            templateDuration;
      } else {
        expectedDuration = templateDuration;
      }
    }
    if (entry.duration != expectedDuration)
      return emitOpError("structured schedule event '")
             << entry.id
             << "' duration must equal its canonical folded "
                "envelope duration "
             << expectedDuration << " ns";
  }

  SmallVector<unsigned, 32> dependencyCounts(entries.size(), 0);
  SmallVector<SmallVector<unsigned, 4>, 32> dependents(entries.size());
  for (auto [index, entry] : llvm::enumerate(entries))
    for (StringRef dependency : entry.dependencies) {
      unsigned predecessor = entryById.find(dependency)->second;
      ++dependencyCounts[index];
      dependents[predecessor].push_back(index);
    }
  SmallVector<unsigned, 32> ready;
  for (auto [index, count] : llvm::enumerate(dependencyCounts))
    if (count == 0)
      ready.push_back(index);
  unsigned scheduledCount = 0;
  while (!ready.empty()) {
    unsigned predecessor = ready.pop_back_val();
    ++scheduledCount;
    for (unsigned dependent : dependents[predecessor])
      if (--dependencyCounts[dependent] == 0)
        ready.push_back(dependent);
  }
  if (scheduledCount != entries.size())
    return emitOpError("schedule dependency graph contains a cycle");
  reportPhase("final-compare-envelope-dag");

  auto isEnvelope = [](const ScheduleClaim &entry) {
    return entry.kind == "call" || entry.kind == "call_template" ||
           entry.kind == "spacetime_call" || entry.kind == "repeat" ||
           entry.kind == "while" || entry.kind == "if" ||
           entry.kind == "try_take";
  };
  // Index positive-duration leaf intervals by resolved physical resource and
  // sweep only simultaneously active intervals.  Valid schedules are
  // O(N log N); malformed/exclusive schedules are output-sensitive in the
  // number of actual temporal intersections rather than unconditionally
  // comparing every pair of rows.
  llvm::StringMap<SmallVector<unsigned, 8>> intervalsByResource;
  for (auto [index, entry] : llvm::enumerate(entries)) {
    // Structured entries summarize their nested leaves; they are not an
    // additional reservation.  Inactive folded templates likewise consume no
    // dynamic resource.
    if (entry.duration == 0.0 || isEnvelope(entry) || inactive[index])
      continue;
    for (StringRef resource : entry.resources)
      if (!resource.starts_with("control:transport-init:"))
        intervalsByResource[resource].push_back(index);
  }
  struct ResourceConflict {
    unsigned left = 0;
    unsigned right = 0;
    std::string resource;
  };
  std::optional<ResourceConflict> firstConflict;
  for (auto &resourceIntervals : intervalsByResource) {
    auto &ordered = resourceIntervals.getValue();
    llvm::sort(ordered, [&](unsigned left, unsigned right) {
      if (entries[left].start != entries[right].start)
        return entries[left].start < entries[right].start;
      return left < right;
    });
    SmallVector<unsigned, 8> active;
    for (unsigned current : ordered) {
      llvm::erase_if(active, [&](unsigned candidate) {
        return entries[candidate].finish() <= entries[current].start;
      });
      for (unsigned candidate : active) {
        if (profileStats)
          ++profileStats->overlapCandidates;
        unsigned left = std::min(candidate, current);
        unsigned right = std::max(candidate, current);
        if (entryIsAncestor(left, right) || entryIsAncestor(right, left) ||
            scheduleEntriesAreExclusive(left, right, entries, entryById))
          continue;
        auto [leftPlan, leftInvocation] = phasePipeline(left);
        auto [rightPlan, rightInvocation] = phasePipeline(right);
        if (!leftPlan.empty() && leftPlan == rightPlan &&
            leftInvocation != rightInvocation)
          continue;
        ResourceConflict conflict{left, right,
                                  resourceIntervals.getKey().str()};
        if (!firstConflict || conflict.left < firstConflict->left ||
            (conflict.left == firstConflict->left &&
             (conflict.right < firstConflict->right ||
              (conflict.right == firstConflict->right &&
               conflict.resource < firstConflict->resource))))
          firstConflict = std::move(conflict);
      }
      active.push_back(current);
    }
  }
  if (firstConflict)
    return emitOpError("schedule events '")
           << entries[firstConflict->left].id << "' and '"
           << entries[firstConflict->right].id << "' overlap on resource '"
           << firstConflict->resource << "'";

  double graphFinish = 0.0;
  for (auto [index, entry] : llvm::enumerate(entries))
    if (!inactive[index])
      graphFinish = std::max(graphFinish, entry.finish());
  if (makespan != graphFinish)
    return emitOpError(
        "makespan_ns must equal the latest top-level schedule finish");
  reportPhase("overlap");
  if (profileStats)
    llvm::errs() << "phys-schedule: verifier-work expected-resource-calls="
                 << profileStats->expectedResourceCalls
                 << " expected-resource-types="
                 << profileStats->expectedResourceTypes
                 << " expected-resource-dedup-comparisons="
                 << profileStats->expectedResourceDedupComparisons
                 << " expected-resource-hash-probes="
                 << profileStats->expectedResourceHashProbes
                 << " expected-resource-max-width="
                 << profileStats->expectedResourceMaxWidth
                 << " data-dependency-derivations="
                 << profileStats->dataDependencyDerivations
                 << " alias-map-builds=" << profileStats->aliasMapBuilds
                 << " alias-map-cache-hits=" << profileStats->aliasMapCacheHits
                 << " alias-map-entries=" << profileStats->aliasMapEntries
                 << " expected-resource-cache-builds="
                 << profileStats->expectedResourceCacheBuilds
                 << " expected-resource-cache-hits="
                 << profileStats->expectedResourceCacheHits
                 << " collect-first-use-visits="
                 << profileStats->collectFirstUseVisits
                 << " ancestry-calls=" << profileStats->ancestryCalls
                 << " journal-key-bytes=" << profileStats->journalKeyBytes
                 << " overlap-candidates=" << profileStats->overlapCandidates
                 << "\n";
  return success();
}

LogicalResult
qlx::phys::verifyScheduleClaims(ScheduleOp schedule,
                                ArrayRef<ScheduleClaim> claims,
                                ScheduleVerificationStats *stats) {
  return ScheduleClaimVerifier(schedule).verify(claims, stats);
}

LogicalResult ScheduleOp::verify() {
  SmallVector<ScheduleClaim, 32> claims;
  claims.reserve(getEntries().size());
  for (Attribute raw : getEntries()) {
    auto claim = parseScheduleEntry(*this, raw);
    if (failed(claim))
      return failure();
    claims.push_back(std::move(*claim));
  }
  ScheduleVerificationStats stats;
  LogicalResult result = verifyScheduleClaims(*this, claims, &stats);
  if (std::getenv("QLX_PROFILE_P2_TO_P3"))
    llvm::errs() << "phys-schedule: portable-proof rows=" << claims.size()
                 << " serialized=" << getEntries().size()
                 << " parsed=" << claims.size()
                 << " semantic-verifier-runs=" << stats.semanticVerifierRuns
                 << "\n";
  return result;
}

namespace {

struct AffineSourceRow {
  llvm::SmallVector<std::string, 8> records;
  ArrayAttr inputSyndromes;
  bool constant = false;
};

static LogicalResult
verifyStringSupport(Operation *owner, StringRef name, ArrayAttr values,
                    llvm::SmallVectorImpl<std::string> *out = nullptr) {
  llvm::SmallDenseSet<StringRef, 16> unique;
  for (Attribute value : values) {
    auto text = dyn_cast<StringAttr>(value);
    if (!text || text.getValue().empty())
      return owner->emitOpError()
             << name << " entries must be nonempty strings";
    if (!unique.insert(text.getValue()).second)
      return owner->emitOpError() << name << " contains duplicate affine term '"
                                  << text.getValue() << "'";
    if (out)
      out->push_back(text.getValue().str());
  }
  return success();
}

static LogicalResult verifyInputSyndromes(Operation *owner, ArrayAttr terms) {
  if (!terms)
    return success();
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 8> unique;
  for (Attribute value : terms) {
    auto term = dyn_cast<DictionaryAttr>(value);
    if (!term)
      return owner->emitOpError(
          "input_syndromes entries must be dictionary attributes");
    auto index = term.getAs<IntegerAttr>("index");
    auto port = term.getAs<IntegerAttr>("port");
    auto portIndex = term.getAs<IntegerAttr>("port_index");
    if (!index || static_cast<bool>(port) == static_cast<bool>(portIndex) ||
        term.size() != 2)
      return owner->emitOpError(
          "input_syndromes entries require exactly index and one of port or "
          "port_index");
    int64_t portValue = port ? port.getInt() : portIndex.getInt();
    if (portValue < 0 || index.getInt() < 0)
      return owner->emitOpError(
          "input_syndromes port and index must be nonnegative");
    if (!unique.insert({portValue, index.getInt()}).second)
      return owner->emitOpError(
          "input_syndromes must not repeat an affine seam term");
  }
  return success();
}

static bool sameStringSupport(ArrayAttr actual,
                              ArrayRef<std::string> expected) {
  if (!actual || actual.size() != expected.size())
    return false;
  for (auto [value, name] : llvm::zip(actual, expected)) {
    auto text = dyn_cast<StringAttr>(value);
    if (!text || text.getValue() != name)
      return false;
  }
  return true;
}

static bool sameArrayOrEmpty(ArrayAttr actual, ArrayAttr expected) {
  if ((!actual || actual.empty()) && (!expected || expected.empty()))
    return true;
  return actual == expected;
}

static FailureOr<int64_t> profileDeclarationWidth(Operation *declaration) {
  (void)declaration;
  return 1;
}

static FailureOr<AffineSourceRow>
resolveProfileDeclarationRow(Operation *declaration, int64_t ordinal) {
  if (ordinal < 0)
    return failure();
  auto inputSyndromes =
      declaration->getAttrOfType<ArrayAttr>("input_syndromes");
  bool constant = false;
  if (auto value = declaration->getAttrOfType<BoolAttr>("constant"))
    constant = value.getValue();
  AffineSourceRow row;
  row.inputSyndromes = inputSyndromes;
  row.constant = constant;
  if (auto records = declaration->getAttrOfType<ArrayAttr>("records")) {
    if (ordinal != 0)
      return failure();
    for (Attribute value : records) {
      auto text = dyn_cast<StringAttr>(value);
      if (!text)
        return failure();
      row.records.push_back(text.getValue().str());
    }
    return row;
  }

  if (ordinal != 0)
    return failure();
  return row;
}

static FailureOr<AffineSourceRow>
findProfileRow(Operation *profile, StringRef operationName, int64_t sourceRow) {
  if (sourceRow < 0 || profile->getNumRegions() != 1 ||
      profile->getRegion(0).empty())
    return failure();

  SmallVector<Operation *> declarations;
  for (Operation &candidate : profile->getRegion(0).front())
    if (candidate.getName().getStringRef() == operationName)
      declarations.push_back(&candidate);

  int64_t remaining = sourceRow;
  for (Operation *candidate : declarations) {
    auto width = profileDeclarationWidth(candidate);
    if (failed(width))
      return failure();
    if (remaining < *width)
      return resolveProfileDeclarationRow(candidate, remaining);
    remaining -= *width;
  }
  return failure();
}

static FailureOr<AffineSourceRow> findOutcomeRow(Operation *sidecar,
                                                 Operation *profile,
                                                 int64_t sourceRow,
                                                 StringRef role) {
  auto gadgetRef = profile->getAttrOfType<FlatSymbolRefAttr>("gadget");
  Operation *gadget =
      gadgetRef ? SymbolTable::lookupNearestSymbolFrom(sidecar, gadgetRef)
                : nullptr;
  auto specRef = gadget ? gadget->getAttrOfType<FlatSymbolRefAttr>("spec")
                        : FlatSymbolRefAttr();
  Operation *spec = specRef
                        ? SymbolTable::lookupNearestSymbolFrom(sidecar, specRef)
                        : nullptr;
  auto outcome = spec ? spec->getAttrOfType<DictionaryAttr>("outcome_map")
                      : DictionaryAttr();
  if (!gadget || gadget->getName().getStringRef() != "fabric.gadget" || !spec ||
      spec->getName().getStringRef() != "fabric.gadget_spec" || !outcome)
    return failure();

  auto records = outcome.getAs<ArrayAttr>("records");
  auto matrix = outcome.getAs<DenseIntElementsAttr>("rows");
  auto constants = outcome.getAs<DenseI64ArrayAttr>("constants");
  if (!records || !matrix || !constants)
    return failure();
  auto type = dyn_cast<ShapedType>(matrix.getType());
  if (!type || type.getRank() != 2 || sourceRow < 0 ||
      sourceRow >= type.getShape()[0] ||
      constants.size() != static_cast<size_t>(type.getShape()[0]))
    return failure();

  auto roles = outcome.getAs<ArrayAttr>("roles");
  if ((roles && roles.size() != static_cast<size_t>(type.getShape()[0])) ||
      records.size() != static_cast<size_t>(type.getShape()[1]))
    return failure();
  auto hasRole = [&](int64_t row, StringRef wanted) {
    if (!roles)
      return wanted == "result";
    auto values = dyn_cast<ArrayAttr>(roles[row]);
    return values && llvm::any_of(values, [&](Attribute value) {
             auto text = dyn_cast<StringAttr>(value);
             return text && text.getValue() == wanted;
           });
  };
  if (!hasRole(sourceRow, role))
    return failure();

  AffineSourceRow row;
  int64_t columns = type.getShape()[1];
  int64_t flatIndex = 0;
  for (APInt selected : matrix.getValues<APInt>()) {
    int64_t recordRow = flatIndex / columns;
    int64_t recordColumn = flatIndex % columns;
    if (recordRow == sourceRow && selected.getBoolValue()) {
      auto name = dyn_cast<StringAttr>(records[recordColumn]);
      if (!name)
        return failure();
      row.records.push_back(
          (gadgetRef.getValue() + "." + name.getValue()).str());
    }
    ++flatIndex;
  }
  if (auto inputRows = outcome.getAs<ArrayAttr>("input_syndromes")) {
    if (inputRows.size() != static_cast<size_t>(type.getShape()[0]))
      return failure();
    row.inputSyndromes = dyn_cast<ArrayAttr>(inputRows[sourceRow]);
    if (!row.inputSyndromes)
      return failure();
  }
  row.constant = constants.asArrayRef()[sourceRow] != 0;
  return row;
}

static LogicalResult verifySourceRow(Operation *sidecar, StringRef profileOp,
                                     StringRef outcomeRole) {
  auto sourceProfile =
      sidecar->getAttrOfType<FlatSymbolRefAttr>("source_profile");
  auto sourceKind = sidecar->getAttrOfType<StringAttr>("source_kind");
  auto sourceRow = sidecar->getAttrOfType<IntegerAttr>("source_row");
  auto sourceInstance = sidecar->getAttrOfType<StringAttr>("source_instance");
  auto sourceRecords = sidecar->getAttrOfType<ArrayAttr>("source_records");
  auto inputSyndromes = sidecar->getAttrOfType<ArrayAttr>("input_syndromes");
  if (!sourceProfile) {
    if (sourceKind || sourceRow || sourceInstance || sourceRecords ||
        sidecar->getAttr("record_projection") ||
        sidecar->getAttr("projection_indices"))
      return sidecar->emitOpError(
          "source provenance attributes require source_profile");
    return success(); // Legacy or inline rows have no detached source symbol.
  }
  if (!sourceKind || !sourceRow || !sourceInstance || sourceInstance.empty() ||
      !sourceRecords || !inputSyndromes)
    return sidecar->emitOpError(
        "source_profile requires source_kind, source_row, nonempty "
        "source_instance, source_records, and input_syndromes");
  if (sourceKind.getValue() != "profile" &&
      sourceKind.getValue() != "outcome_map")
    return sidecar->emitOpError(
        "source_kind must be profile or outcome_map when source_profile is "
        "set");
  if (sourceRow.getInt() < 0)
    return sidecar->emitOpError("source_row must be nonnegative");
  if (failed(verifyStringSupport(sidecar, "source_records", sourceRecords)) ||
      failed(verifyInputSyndromes(sidecar, inputSyndromes)))
    return failure();

  Operation *profile =
      SymbolTable::lookupNearestSymbolFrom(sidecar, sourceProfile);
  if (!profile)
    return success(); // Partial linked modules resolve at link time.
  if (profile->getName().getStringRef() != "fabric.gadget_profile")
    return sidecar->emitOpError(
        "source_profile must resolve to fabric.gadget_profile");

  FailureOr<AffineSourceRow> expected = failure();
  if (sourceKind.getValue() == "profile")
    expected = findProfileRow(profile, profileOp, sourceRow.getInt());
  else
    expected =
        findOutcomeRow(sidecar, profile, sourceRow.getInt(), outcomeRole);
  if (failed(expected))
    return sidecar->emitOpError(
               "source provenance does not resolve to the claimed ")
           << outcomeRole << " row " << sourceRow.getInt();

  if (!sameStringSupport(sourceRecords, expected->records))
    return sidecar->emitOpError(
        "source_records do not match the claimed authoritative source row");
  if (!sameArrayOrEmpty(inputSyndromes, expected->inputSyndromes))
    return sidecar->emitOpError(
        "input_syndromes do not match the claimed authoritative source row");
  bool constant = false;
  if (auto value = sidecar->getAttrOfType<BoolAttr>("constant"))
    constant = value.getValue();
  if (constant != expected->constant)
    return sidecar->emitOpError(
        "constant does not match the claimed authoritative source row");
  return success();
}

static LogicalResult
verifySidecarInvocation(Operation *sidecar, FlatSymbolRefAttr sourceProfile,
                        StringAttr sourceInstance,
                        SidecarGraphVerificationIndex &index) {
  Operation *profileTarget =
      SymbolTable::lookupNearestSymbolFrom(sidecar, sourceProfile);
  FlatSymbolRefAttr profileGadget;
  if (profileTarget)
    profileGadget = profileTarget->getAttrOfType<FlatSymbolRefAttr>("gadget");
  GraphOp graph = index.graph;
  auto matching = index.invocations.find(sourceInstance.getValue());
  bool hasMatching = matching != index.invocations.end();
  size_t matchingCount = hasMatching ? matching->second.size() : 0;
  if (graph.getSourceProtocolAttr() &&
      sourceInstance.getValue() == graph.getSourceProtocolAttr().getValue()) {
    if (matchingCount != 0)
      return sidecar->emitOpError(
          "root source_instance collides with a physical call instance");
    if (profileGadget && profileGadget != graph.getSourceProtocolAttr())
      return sidecar->emitOpError(
          "root source_instance requires source_profile to analyze the "
          "physical graph source_protocol");
    return success();
  }
  if (matchingCount != 1)
    return sidecar->emitOpError("source_instance '")
           << sourceInstance.getValue()
           << "' must resolve to exactly one phys.call in graph @"
           << graph.getSymName();
  Operation *invocation = matching->second.front();
  auto invocationProfile =
      invocation->getAttrOfType<FlatSymbolRefAttr>("profile");
  auto invocationCallee =
      invocation->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!invocationProfile || invocationProfile != sourceProfile)
    return sidecar->emitOpError(
        "source_instance physical call must reference source_profile");
  if (profileGadget && invocationCallee != profileGadget)
    return sidecar->emitOpError(
        "source_instance physical call callee must match the "
        "source_profile gadget");
  return success();
}

static LogicalResult verifyRecordProjection(Operation *sidecar,
                                            FlatSymbolRefAttr graphRef,
                                            ArrayAttr records,
                                            ArrayAttr sourceRecords) {
  auto sourceProfile =
      sidecar->getAttrOfType<FlatSymbolRefAttr>("source_profile");
  if (!sourceProfile)
    return success();
  auto sourceInstance = sidecar->getAttrOfType<StringAttr>("source_instance");
  if (!sourceInstance || sourceInstance.empty())
    return sidecar->emitOpError(
        "source_profile requires a nonempty source_instance");

  auto projectionRef =
      sidecar->getAttrOfType<FlatSymbolRefAttr>("record_projection");
  auto projectionIndices =
      sidecar->getAttrOfType<DenseI64ArrayAttr>("projection_indices");
  if (!projectionRef || !projectionIndices)
    return sidecar->emitOpError(
        "source_profile requires record_projection and projection_indices");
  if (!sourceRecords || projectionIndices.size() != sourceRecords.size() ||
      projectionIndices.size() != records.size())
    return sidecar->emitOpError(
        "projection_indices must align one-to-one with source_records and "
        "records");
  for (int64_t index : projectionIndices.asArrayRef())
    if (index < 0)
      return sidecar->emitOpError("projection_indices must be nonnegative");

  Operation *projectionTarget =
      SymbolTable::lookupNearestSymbolFrom(sidecar, projectionRef);
  RecordProjectionOp projection;
  if (projectionTarget) {
    projection = dyn_cast<RecordProjectionOp>(projectionTarget);
    if (!projection)
      return sidecar->emitOpError(
          "record_projection must resolve to phys.record_projection");
    if (projection.getGraphAttr() != graphRef)
      return sidecar->emitOpError(
          "record_projection must reference the same physical graph");
  }

  Operation *graphTarget =
      SymbolTable::lookupNearestSymbolFrom(sidecar, graphRef);
  if (graphTarget) {
    auto graph = dyn_cast<GraphOp>(graphTarget);
    if (!graph)
      return sidecar->emitOpError("graph reference must resolve to phys.graph");
    std::optional<SidecarGraphVerificationIndex> localIndex;
    SidecarGraphVerificationIndex *index = activeSidecarGraphVerificationIndex;
    if (!index || index->graph != graph) {
      localIndex.emplace(graph);
      index = &*localIndex;
    }
    if (failed(verifySidecarInvocation(sidecar, sourceProfile, sourceInstance,
                                       *index)))
      return failure();
  }

  if (!projectionTarget)
    return success(); // Partial linked modules close the map at link time.

  ArrayAttr entries = projection.getEntries();
  for (auto [lane, rawIndex] :
       llvm::enumerate(projectionIndices.asArrayRef())) {
    if (static_cast<uint64_t>(rawIndex) >= entries.size())
      return sidecar->emitOpError("projection index ")
             << rawIndex << " is outside record_projection @"
             << projection.getSymName();
    auto entry = dyn_cast<DictionaryAttr>(entries[rawIndex]);
    auto expectedInstance =
        entry ? entry.getAs<StringAttr>("instance") : StringAttr{};
    auto expectedSource =
        entry ? entry.getAs<StringAttr>("source_record") : StringAttr{};
    auto expectedPhysical =
        entry ? entry.getAs<StringAttr>("physical_record") : StringAttr{};
    auto actualSource = dyn_cast<StringAttr>(sourceRecords[lane]);
    auto actualPhysical = dyn_cast<StringAttr>(records[lane]);
    if (!expectedInstance || expectedInstance != sourceInstance ||
        !expectedSource || !expectedPhysical || !actualSource ||
        !actualPhysical || expectedSource != actualSource ||
        expectedPhysical != actualPhysical)
      return sidecar->emitOpError(
          "source_instance, source_records, and records do not match the "
          "claimed record_projection rows");
  }
  return success();
}

} // namespace

static LogicalResult verifyRecordSidecar(Operation *sidecar,
                                         FlatSymbolRefAttr graphRef,
                                         ArrayAttr records, StringRef profileOp,
                                         StringRef outcomeRole) {
  llvm::SmallVector<std::string, 8> recordNames;
  if (failed(verifyStringSupport(sidecar, "records", records, &recordNames)))
    return failure();
  auto sourceRecords = sidecar->getAttrOfType<ArrayAttr>("source_records");
  if (sourceRecords &&
      failed(verifyStringSupport(sidecar, "source_records", sourceRecords)))
    return failure();
  if (failed(verifyInputSyndromes(
          sidecar, sidecar->getAttrOfType<ArrayAttr>("input_syndromes"))))
    return failure();
  if (sourceRecords && sourceRecords.size() != records.size())
    return sidecar->emitOpError(
        "records and source_records must have equal affine support width");
  auto *target = SymbolTable::lookupNearestSymbolFrom(sidecar, graphRef);
  std::optional<SidecarGraphVerificationIndex> localIndex;
  SidecarGraphVerificationIndex *previousIndex =
      activeSidecarGraphVerificationIndex;
  if (target) {
    GraphOp graph = dyn_cast<GraphOp>(target);
    if (!graph)
      return sidecar->emitOpError("graph reference must resolve to phys.graph");
    if (!previousIndex || previousIndex->graph != graph) {
      localIndex.emplace(graph);
      activeSidecarGraphVerificationIndex = &*localIndex;
    }
    for (const std::string &record : recordNames) {
      auto qualified = parseQualifiedRecord(record);
      if (failed(qualified) ||
          !activeSidecarGraphVerificationIndex->produced.contains(
              qualified->first))
        return sidecar->emitOpError("references unknown physical record '")
               << record << "' in graph @" << graph.getSymName();
    }
  }
  llvm::scope_exit restoreIndex(
      [&] { activeSidecarGraphVerificationIndex = previousIndex; });
  if (failed(verifySourceRow(sidecar, profileOp, outcomeRole)))
    return failure();
  if (failed(verifyRecordProjection(sidecar, graphRef, records, sourceRecords)))
    return failure();
  // Graph and profile symbols are independently linkable.  A missing graph
  // defers only physical-record existence; it must not suppress validation of
  // an already-available authoritative profile (and vice versa).
  return success();
}

LogicalResult SelectionSidecarOp::verify() {
  if (getExpected())
    return emitOpError(
        "expected must be false because success sidecars store mismatch bits");
  if (getRecords().empty())
    return emitOpError("requires at least one physical record");
  return verifyRecordSidecar(*this, getGraphAttr(), getRecords(),
                             "fabric.success", "success");
}

LogicalResult qlx::phys::verifySidecarBatch(ArrayRef<Operation *> sidecars) {
  if (sidecars.empty())
    return success();
  auto graphRef = sidecars.front()->getAttrOfType<FlatSymbolRefAttr>("graph");
  Operation *target =
      graphRef
          ? SymbolTable::lookupNearestSymbolFrom(sidecars.front(), graphRef)
          : nullptr;
  auto graph = dyn_cast_or_null<GraphOp>(target);
  if (!graph) {
    // Preserve partial-link behavior and ordinary diagnostics when no concrete
    // graph can provide a reusable index.
    for (Operation *sidecar : sidecars)
      if (failed(mlir::verify(sidecar, /*verifyRecursively=*/true)))
        return failure();
    return success();
  }
  for (Operation *sidecar : sidecars) {
    StringRef name = sidecar->getName().getStringRef();
    if (name != "phys.selection_sidecar")
      return sidecar->emitOpError(
          "batched sidecar verification accepts only selection sidecars");
    if (sidecar->getAttrOfType<FlatSymbolRefAttr>("graph") != graphRef)
      return sidecar->emitOpError(
          "batched sidecars must reference one physical graph");
  }
  SidecarGraphVerificationIndex index(graph);
  SidecarGraphVerificationIndex *previous = activeSidecarGraphVerificationIndex;
  activeSidecarGraphVerificationIndex = &index;
  llvm::scope_exit restore(
      [&] { activeSidecarGraphVerificationIndex = previous; });
  for (Operation *sidecar : sidecars)
    if (failed(mlir::verify(sidecar, /*verifyRecursively=*/true)))
      return failure();
  return success();
}

LogicalResult qlx::phys::verifyClosedSidecarLinks(ModuleOp module,
                                                  StringRef graphSymbol) {
  for (Operation &operation : module.getBody()->getOperations()) {
    StringRef kind = operation.getName().getStringRef();
    if (kind != "phys.selection_sidecar")
      continue;
    auto graph = operation.getAttrOfType<FlatSymbolRefAttr>("graph");
    if (!graph || graph.getValue() != graphSymbol)
      continue;
    Operation *graphTarget =
        SymbolTable::lookupNearestSymbolFrom(&operation, graph);
    if (!graphTarget || !isa<GraphOp>(graphTarget))
      return operation.emitOpError(
          "terminal sidecar graph reference must resolve to phys.graph");
    auto profile = operation.getAttrOfType<FlatSymbolRefAttr>("source_profile");
    if (!profile)
      continue;
    Operation *profileTarget =
        SymbolTable::lookupNearestSymbolFrom(&operation, profile);
    if (!profileTarget ||
        profileTarget->getName().getStringRef() != "fabric.gadget_profile")
      return operation.emitOpError(
          "terminal sidecar source_profile must resolve to "
          "fabric.gadget_profile");
    auto projection =
        operation.getAttrOfType<FlatSymbolRefAttr>("record_projection");
    Operation *projectionTarget =
        projection
            ? SymbolTable::lookupNearestSymbolFrom(&operation, projection)
            : nullptr;
    if (!projectionTarget || !isa<RecordProjectionOp>(projectionTarget))
      return operation.emitOpError(
          "terminal sidecar record_projection must resolve to "
          "phys.record_projection");
  }
  return success();
}

LogicalResult HierarchyProjectionOp::verify() {
  auto *target = SymbolTable::lookupNearestSymbolFrom(*this, getGraphAttr());
  if (target && !isa<GraphOp>(target))
    return emitOpError("graph reference must resolve to phys.graph");
  auto *hierarchy =
      SymbolTable::lookupNearestSymbolFrom(*this, getSourceHierarchyAttr());
  if (hierarchy &&
      hierarchy->getName().getStringRef() != "fabric.encoding_hierarchy")
    return emitOpError(
        "source_hierarchy must resolve to fabric.encoding_hierarchy");
  if (getEntries().empty())
    return emitOpError("requires at least one hierarchy-to-resource entry");
  llvm::SmallDenseSet<StringRef, 16> unique;
  for (Attribute value : getEntries()) {
    auto entry = dyn_cast<StringAttr>(value);
    if (!entry || entry.getValue().empty())
      return emitOpError("entries must be nonempty strings");
    if (!unique.insert(entry.getValue()).second)
      return emitOpError("contains duplicate entry '")
             << entry.getValue() << "'";
  }
  return success();
}

#define GET_OP_CLASSES
#include "qlx/Dialect/Phys/IR/PhysOps.cpp.inc"

void PhysDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "qlx/Dialect/Phys/IR/PhysTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "qlx/Dialect/Phys/IR/PhysOps.cpp.inc"
      >();
}
