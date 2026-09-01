//===- QLXDialect.cpp - QLX dialect registration ---------------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//
//
// This is the single translation unit for the QLX dialect. It includes all
// generated .cpp.inc files for types, attributes, enums, and ops (logical
// and block-graph sub-namespaces), ensuring that storage types are complete
// at the point of dialect initialization.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"
#include "qlx/Dialect/QLX/IR/QLXTypes.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include <functional>

using namespace mlir;
using namespace qlx;

MLIR_DEFINE_EXPLICIT_TYPE_ID(qlx::DeviceBindingDialectInterface)

LogicalResult EstimateResultOp::verify() {
  static constexpr StringLiteral tiers[] = {"logical", "static"};
  if (!llvm::is_contained(tiers, getTier()))
    return emitOpError("tier must be logical or static");
  if (getSchema().empty())
    return emitOpError("schema must be nonempty");
  Operation *root = SymbolTable::lookupNearestSymbolFrom(*this, getRootAttr());
  if (!root)
    return emitOpError("root must resolve to a linked symbol");
  auto verifyResultRef = [&](FlatSymbolRefAttr reference,
                             StringRef name) -> LogicalResult {
    if (!reference)
      return success();
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, reference);
    if (!target || !isa<EstimateResultOp>(target))
      return emitOpError() << name << " must resolve to qlx.estimate_result";
    return success();
  };
  if (failed(verifyResultRef(getLowerTierAttr(), "lower_tier")))
    return failure();
  if (auto device = getDeviceAttr()) {
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, device);
    if (!target || !isa<DeviceOp>(target))
      return emitOpError("device must resolve to qlx.device");
  }
  auto lower =
      getLowerTierAttr()
          ? dyn_cast_or_null<EstimateResultOp>(
                SymbolTable::lookupNearestSymbolFrom(*this, getLowerTierAttr()))
          : EstimateResultOp{};
  if (getTier() == "logical" && (lower || getDeviceAttr()))
    return emitOpError("logical tier must remain device independent");
  if (getTier() == "static") {
    if (!getDeviceAttr() || lower)
      return emitOpError(
          "static tier requires a device and no lower-tier result");
  }
  auto metadata = getMetadataAttr();
  if (!metadata || !metadata.getAs<StringAttr>("producer") ||
      !metadata.getAs<StringAttr>("producer_version"))
    return emitOpError(
        "metadata requires producer and producer_version strings");
  for (Attribute assumption : getAssumptions())
    if (!isa<StringAttr>(assumption))
      return emitOpError("assumptions must contain only strings");
  for (Attribute item : getEvidence())
    if (!isa<StringAttr, FlatSymbolRefAttr, SymbolRefAttr>(item))
      return emitOpError("evidence must contain strings or symbol references");
    else if (auto reference = dyn_cast<SymbolRefAttr>(item);
             reference &&
             !SymbolTable::lookupNearestSymbolFrom(*this, reference))
      return emitOpError("evidence contains an unresolved symbol reference");
  return success();
}

LogicalResult DeviceOp::verify() {
  auto has = [&](StringRef name) { return (*this)->hasAttr(name); };
  for (StringRef retired : {StringRef("physical"), StringRef("qec_to_physical"),
                            StringRef("operating_point")})
    if (has(retired))
      return emitOpError() << "retired P3 attribute '" << retired
                           << "' is outside the P0-P2 product slice";
  if (has("logical_to_qec") != has("qec"))
    return emitOpError(
        "logical_to_qec must be present exactly when qec is present");

  auto verifyReference = [&](StringRef attribute,
                             StringRef expected) -> LogicalResult {
    auto reference = (*this)->getAttrOfType<FlatSymbolRefAttr>(attribute);
    if (!reference)
      return success();
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, reference);
    if (!target)
      return emitOpError() << attribute << " must resolve to " << expected;
    if (target->getName().getStringRef() != expected)
      return emitOpError() << attribute << " must resolve to " << expected;
    return success();
  };
  if (failed(verifyReference("logical", "lvm.domain")) ||
      failed(verifyReference("qec", "fabric.machine")) ||
      failed(verifyReference("logical_to_qec", "qlx.logical_to_qec")))
    return failure();

  auto verifyRefinementEndpoints =
      [&](StringRef bindingAttribute,
          ArrayRef<StringRef> endpointAttributes) -> LogicalResult {
    auto bindingReference =
        (*this)->getAttrOfType<FlatSymbolRefAttr>(bindingAttribute);
    Operation *binding =
        bindingReference
            ? SymbolTable::lookupNearestSymbolFrom(*this, bindingReference)
            : nullptr;
    if (!binding)
      return success();
    for (StringRef endpoint : endpointAttributes)
      if (binding->getAttr(endpoint) != (*this)->getAttr(endpoint))
        return emitOpError() << bindingAttribute << " " << endpoint
                             << " endpoint must match the device layer";
    return success();
  };
  if (failed(verifyRefinementEndpoints("logical_to_qec", {"logical", "qec"})))
    return failure();

  if (auto bindings = (*this)->getAttrOfType<ArrayAttr>("resource_bindings")) {
    StringRef logical = getLogicalAttr().getValue();
    Operation *logicalDomain =
        SymbolTable::lookupNearestSymbolFrom(*this, getLogicalAttr());
    llvm::StringSet<> boundStreams;
    llvm::StringMap<Attribute> boundStreamRefs;
    llvm::DenseMap<Attribute, Operation *> streamProducers;
    llvm::DenseMap<Attribute, SmallVector<Operation *>> streamCallables;
    auto bindingInterface = [](Dialect &dialect) {
      return dialect.getRegisteredInterface<DeviceBindingDialectInterface>();
    };
    auto matchesResource = [&](Type type, FlatSymbolRefAttr resource) {
      auto *interface = bindingInterface(type.getDialect());
      return interface && interface->getResourceKind(type) == resource;
    };
    auto hasCapability = [&](Attribute attribute, StringRef key) {
      auto values = dyn_cast_or_null<ArrayAttr>(attribute);
      if (!values)
        return false;
      for (Attribute value : values) {
        auto *interface = bindingInterface(value.getDialect());
        if (interface && interface->getLogicalCapabilityKey(value) == key)
          return true;
      }
      return false;
    };
    for (Attribute raw : bindings) {
      auto binding = dyn_cast<DictionaryAttr>(raw);
      if (!binding)
        return emitOpError("resource_bindings entries must be dictionaries");
      auto stream = dyn_cast_or_null<SymbolRefAttr>(binding.get("stream"));
      auto producer =
          dyn_cast_or_null<FlatSymbolRefAttr>(binding.get("producer"));
      if (!stream || !producer)
        return emitOpError(
            "resource binding requires stream and producer symbol refs");
      if (stream.getRootReference().getValue() != logical)
        return emitOpError(
            "resource binding stream must belong to the device logical domain");
      if (!boundStreams.insert(stream.getLeafReference().getValue()).second)
        return emitOpError("resource_bindings contains duplicate stream ")
               << stream;
      boundStreamRefs.try_emplace(stream.getLeafReference().getValue(), stream);
      Operation *streamTarget =
          SymbolTable::lookupNearestSymbolFrom(*this, stream);
      if (!streamTarget ||
          streamTarget->getName().getStringRef() != "lvm.stream")
        return emitOpError(
            "resource binding stream must resolve to lvm.stream");
      if (streamTarget->hasAttr("external"))
        return emitOpError("resource binding stream cannot also be external");
      auto resource =
          streamTarget->getAttrOfType<FlatSymbolRefAttr>("produces");
      if (!resource)
        return emitOpError(
            "resource binding stream requires a symbolic produces kind");
      Operation *producerTarget =
          SymbolTable::lookupNearestSymbolFrom(*this, producer);
      if (!producerTarget ||
          producerTarget->getName().getStringRef() != "fabric.protocol")
        return emitOpError(
            "resource binding producer must resolve to fabric.protocol");
      auto producerTypeAttr =
          producerTarget->getAttrOfType<TypeAttr>("function_type");
      auto producerType =
          producerTypeAttr ? dyn_cast<FunctionType>(producerTypeAttr.getValue())
                           : FunctionType();
      if (!producerType || producerType.getNumInputs() != 0 ||
          producerType.getNumResults() != 1 ||
          !matchesResource(producerType.getResult(0), resource))
        return emitOpError("resource binding producer must return exactly the "
                           "stream resource");
      streamProducers.try_emplace(stream, producerTarget);
      streamCallables[stream].push_back(producerTarget);
      auto objective =
          producerTarget->getAttrOfType<SymbolRefAttr>("objective");
      Operation *objectiveTarget =
          objective ? SymbolTable::lookupNearestSymbolFrom(*this, objective)
                    : nullptr;
      auto objectiveTypeAttr =
          objectiveTarget
              ? objectiveTarget->getAttrOfType<TypeAttr>("function_type")
              : TypeAttr();
      auto objectiveType =
          objectiveTypeAttr
              ? dyn_cast<FunctionType>(objectiveTypeAttr.getValue())
              : FunctionType();
      if (!objectiveTarget ||
          objectiveTarget->getName().getStringRef() != "qlx.action" ||
          !objectiveType || objectiveType.getNumInputs() != 0 ||
          objectiveType.getNumResults() != 1 ||
          !matchesResource(objectiveType.getResult(0), resource))
        return emitOpError("resource binding producer must implement the "
                           "stream production objective");
      auto objectiveKind = objectiveTarget->getAttrOfType<StringAttr>("kind");
      if (!objectiveKind ||
          objectiveKind.getValue() != ("produce_" + resource.getValue()).str())
        return emitOpError("resource binding producer objective kind must name "
                           "stream production");
      if (auto factory =
              dyn_cast_or_null<SymbolRefAttr>(binding.get("factory"))) {
        if (factory.getRootReference().getValue() != logical)
          return emitOpError("resource binding factory must belong to the "
                             "device logical domain");
        Operation *factoryTarget =
            SymbolTable::lookupNearestSymbolFrom(*this, factory);
        if (!factoryTarget ||
            factoryTarget->getName().getStringRef() != "lvm.space")
          return emitOpError(
              "resource binding factory must resolve to lvm.space");
        if (!hasCapability(factoryTarget->getAttr("capabilities"),
                           "qlx.machine/logical_factory"))
          return emitOpError(
              "resource binding factory must have logical_factory capability");
        Operation *domainTarget = factoryTarget->getParentOp();
        bool hasSupply = false;
        if (domainTarget)
          domainTarget->walk([&](Operation *candidate) {
            if (candidate->getName().getStringRef() != "lvm.channel")
              return;
            auto from = candidate->getAttrOfType<SymbolRefAttr>("from");
            auto to = candidate->getAttrOfType<SymbolRefAttr>("to");
            if (from && to &&
                from.getLeafReference() == factory.getLeafReference() &&
                to.getLeafReference() == stream.getLeafReference() &&
                hasCapability(candidate->getAttr("capabilities"),
                              "qlx.machine/resource_transfer"))
              hasSupply = true;
          });
        if (!hasSupply)
          return emitOpError("resource binding factory must supply its stream "
                             "through a resource_transfer channel");
        bool sawReturnedProductionHome = false;
        bool mismatchedReturnedProductionHome = false;
        llvm::SmallPtrSet<Operation *, 16> visited;
        std::function<void(Value)> traceReturnedResource = [&](Value value) {
          Operation *definition = value.getDefiningOp();
          if (!definition || !visited.insert(definition).second)
            return;
          if (definition->getName().getStringRef() ==
              "fabric.produce_resource") {
            sawReturnedProductionHome = true;
            auto region = definition->getAttrOfType<SymbolRefAttr>("region");
            if (!region ||
                region.getLeafReference() != factory.getLeafReference())
              mismatchedReturnedProductionHome = true;
            return;
          }
          for (Value operand : definition->getOperands())
            traceReturnedResource(operand);
        };
        producerTarget->walk([&](Operation *candidate) {
          if (candidate->getName().getStringRef() != "fabric.protocol_return")
            return;
          for (Value operand : candidate->getOperands())
            traceReturnedResource(operand);
        });
        if (sawReturnedProductionHome && mismatchedReturnedProductionHome)
          return emitOpError(
              "resource binding returned producer home must match its factory");
      }
      if (auto transfer =
              dyn_cast_or_null<FlatSymbolRefAttr>(binding.get("transfer"))) {
        Operation *transferTarget =
            SymbolTable::lookupNearestSymbolFrom(*this, transfer);
        if (!transferTarget ||
            transferTarget->getName().getStringRef() != "fabric.protocol")
          return emitOpError(
              "resource binding transfer must resolve to fabric.protocol");
        auto transferTypeAttr =
            transferTarget->getAttrOfType<TypeAttr>("function_type");
        auto transferType =
            transferTypeAttr
                ? dyn_cast<FunctionType>(transferTypeAttr.getValue())
                : FunctionType();
        if (!transferType ||
            llvm::none_of(transferType.getInputs(), [&](Type input) {
              return matchesResource(input, resource);
            }))
          return emitOpError(
              "resource binding transfer must consume the stream resource");
        streamCallables[stream].push_back(transferTarget);
      }
    }

    // A produced stream may depend on raw/external input streams, but bound
    // producer streams form a directed dependency graph and must be acyclic.
    // Follow executable callee references so a request hidden behind a helper
    // protocol cannot evade the device-level supply invariant.
    llvm::DenseMap<Attribute, SmallVector<Attribute>> dependencies;
    for (const auto &entry : streamProducers) {
      Attribute stream = entry.first;
      llvm::SmallPtrSet<Operation *, 16> visitedCallables;
      std::function<void(Operation *)> collect = [&](Operation *callable) {
        if (!callable || !visitedCallables.insert(callable).second)
          return;
        callable->walk([&](Operation *candidate) {
          if (candidate->getName().getStringRef() ==
              "fabric.resource_request") {
            auto requested = candidate->getAttrOfType<SymbolRefAttr>("stream");
            if (!requested)
              return;
            auto bound =
                boundStreamRefs.find(requested.getLeafReference().getValue());
            if (bound != boundStreamRefs.end()) {
              dependencies[stream].push_back(bound->second);
              return;
            }
            Operation *requestedTarget =
                SymbolTable::lookupNearestSymbolFrom(candidate, requested);
            // Portable producer protocols use a flat stream identity. Resolve
            // that leaf inside the device's selected logical domain before
            // classifying it as an unbound dependency.
            if (!requestedTarget && logicalDomain &&
                requested.getNestedReferences().empty())
              requestedTarget =
                  SymbolTable(logicalDomain)
                      .lookup(requested.getRootReference().getValue());
            if (!requestedTarget ||
                requestedTarget->getName().getStringRef() != "lvm.stream" ||
                !requestedTarget->hasAttr("external"))
              dependencies[stream].push_back(Attribute{});
          }
          auto callee = candidate->getAttrOfType<SymbolRefAttr>("callee");
          if (!callee)
            return;
          Operation *target =
              SymbolTable::lookupNearestSymbolFrom(candidate, callee);
          if (target &&
              (target->getName().getStringRef() == "fabric.protocol" ||
               target->getName().getStringRef() == "fabric.gadget"))
            collect(target);
        });
      };
      for (Operation *callable : streamCallables[stream])
        collect(callable);
    }
    llvm::DenseMap<Attribute, unsigned> marks;
    std::function<LogicalResult(Attribute)> visit =
        [&](Attribute stream) -> LogicalResult {
      unsigned &mark = marks[stream];
      if (mark == 1)
        return emitOpError("resource producer dependency cycle reaches ")
               << cast<SymbolRefAttr>(stream);
      if (mark == 2)
        return success();
      mark = 1;
      for (Attribute dependency : dependencies[stream]) {
        if (!dependency)
          return emitOpError(
              "resource producer depends on an unbound non-external stream");
        if (failed(visit(dependency)))
          return failure();
      }
      mark = 2;
      return success();
    };
    for (const auto &entry : streamProducers)
      if (failed(visit(entry.first)))
        return failure();
  }
  return success();
}

LogicalResult LogicalToQECBindingOp::verify() {
  Operation *logicalMachine =
      SymbolTable::lookupNearestSymbolFrom(*this, getLogicalAttr());
  Operation *qecMachine =
      SymbolTable::lookupNearestSymbolFrom(*this, getQecAttr());
  if (logicalMachine &&
      logicalMachine->getName().getStringRef() != "lvm.domain")
    return emitOpError("logical must resolve to lvm.domain");
  if (qecMachine && qecMachine->getName().getStringRef() != "fabric.machine")
    return emitOpError("qec must resolve to fabric.machine");

  llvm::StringSet<> logicalRegions;
  llvm::StringSet<> qecRegions;
  for (Attribute raw : getEntries()) {
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto logical = entry ? entry.getAs<StringAttr>("logical") : nullptr;
    auto qec = entry ? entry.getAs<StringAttr>("qec") : nullptr;
    if (!entry || !logical || logical.getValue().empty() || !qec ||
        qec.getValue().empty())
      return emitOpError(
          "entries require nonempty logical and qec region names");
    if (!logicalRegions.insert(logical.getValue()).second)
      return emitOpError("contains duplicate logical region '")
             << logical.getValue() << "'";
    if (!qecRegions.insert(qec.getValue()).second)
      return emitOpError("contains duplicate QEC region '")
             << qec.getValue() << "'";
    if (logicalMachine) {
      Operation *target =
          SymbolTable(logicalMachine).lookup(logical.getValue());
      if (!target || target->getName().getStringRef() != "lvm.space")
        return emitOpError("logical region @")
               << logical.getValue()
               << " must resolve inside the referenced lvm.domain";
    }
    if (qecMachine) {
      Operation *target = SymbolTable(qecMachine).lookup(qec.getValue());
      if (!target || target->getName().getStringRef() != "fabric.region")
        return emitOpError("QEC region @")
               << qec.getValue()
               << " must resolve inside the referenced fabric.machine";
    }
  }
  return success();
}

LogicalResult QECLoweringOp::verify() {
  if (getManifestName().empty())
    return emitOpError("manifest_name must be nonempty");
  StringRef manifestDigest = getManifestSha256();
  if (!manifestDigest.consume_front("sha256:") || manifestDigest.size() != 64 ||
      !llvm::all_of(manifestDigest, [](char value) {
        return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
      }))
    return emitOpError("manifest_sha256 must be sha256: followed by 64 "
                       "lowercase hexadecimal digits");
  if (getObjectiveFamily().empty() || getCompilerPlugin().empty() ||
      getCompilerSymbol().empty() || getCompilerVersion().empty())
    return emitOpError(
        "objective_family and compiler provenance must be nonempty");
  if (getCodes().empty())
    return emitOpError("requires at least one accepted code");

  llvm::StringSet<> codes;
  for (Attribute value : getCodes()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference)
      return emitOpError("codes must contain flat symbol references");
    if (!codes.insert(reference.getValue()).second)
      return emitOpError("contains duplicate accepted code or encoding @")
             << reference.getValue();
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, reference);
    if (!target)
      return emitOpError("accepted code or encoding @")
             << reference.getValue() << " does not resolve";
    if (target->getName().getStringRef() != "fabric.code" &&
        target->getName().getStringRef() != "fabric.encoding")
      return emitOpError("accepted code or encoding must resolve to "
                         "fabric.code or fabric.encoding");
  }

  auto verifyStringDictionary = [&](StringRef field) -> LogicalResult {
    auto dictionary = getOperation()->getAttrOfType<DictionaryAttr>(field);
    if (!dictionary)
      return success();
    for (NamedAttribute entry : dictionary) {
      if (entry.getName().empty() || !isa<StringAttr>(entry.getValue()))
        return emitOpError()
               << field << " must contain nonempty keys and string values";
    }
    return success();
  };
  if (failed(verifyStringDictionary("policy_schema")) ||
      failed(verifyStringDictionary("metadata")))
    return failure();

  llvm::StringSet<> requirements;
  bool remoteObservable = false;
  for (Attribute value : getRequirements()) {
    auto requirement = dyn_cast<StringAttr>(value);
    if (!requirement || requirement.getValue().empty())
      return emitOpError("requirements must contain nonempty strings");
    if (!requirements.insert(requirement.getValue()).second)
      return emitOpError("contains duplicate requirement ") << requirement;
    remoteObservable |=
        requirement.getValue() == "qlx.machine/observable_remote";
  }

  llvm::StringSet<> dependencies;
  for (Attribute value : getDependencies()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference)
      return emitOpError("dependencies must contain flat symbol references");
    if (!dependencies.insert(reference.getValue()).second)
      return emitOpError("contains duplicate dependency @")
             << reference.getValue();
  }

  if (getInputStage() != "p1" || getOutputStage() != "p2")
    return emitOpError("must lower exactly from p1 to p2");
  llvm::StringSet<> facets;
  for (Attribute value : getProvidesFacets()) {
    auto facet = dyn_cast<StringAttr>(value);
    if (!facet || facet.getValue().empty())
      return emitOpError("provides_facets must contain nonempty strings");
    facets.insert(facet.getValue());
  }
  if (!facets.contains("qec_realization") ||
      !facets.contains("protocol_network"))
    return emitOpError(
        "must provide qec_realization and protocol_network facets");

  if (!remoteObservable)
    return success();
  if (getObjectiveFamily() != "pauli_product_measurement")
    return emitOpError("observable_remote requires pauli_product_measurement "
                       "objective_family");
  auto objective = dyn_cast_or_null<BuiltinInstrumentAttr>(getObjectiveAttr());
  if (!objective || objective.getValue() != BuiltinInstrument::mpp)
    return emitOpError(
        "observable_remote lowering objective must be the built-in MPP");
  return success();
}

LogicalResult LoweringRecipeOp::verify() {
  if (getCapability().empty())
    return emitOpError("capability must be nonempty");
  if (getCapability() == "sample")
    return emitOpError(
        "sampling capabilities are outside the P0-P2 product slice");
  if (getFinalizer().empty())
    return emitOpError("finalizer must be nonempty");
  if (getEffect() != "local" && getEffect() != "filesystem" &&
      getEffect() != "external")
    return emitOpError("effect must be local, filesystem, or external");
  for (auto [label, values] :
       {std::pair<StringRef, ArrayAttr>{"accepted_stages", getAcceptedStages()},
        {"required_facets", getRequiredFacets()},
        {"provides_facets", getProvidesFacets()},
        {"stages", getStages()}}) {
    for (Attribute value : values)
      if (!isa<StringAttr>(value))
        return emitOpError() << label << " entries must be strings";
  }
  auto isProductStage = [](StringRef stage) {
    return stage == "p0" || stage == "p1" || stage == "p2";
  };
  for (Attribute value : getAcceptedStages())
    if (!isProductStage(cast<StringAttr>(value).getValue()))
      return emitOpError("accepted_stages entries must be p0, p1, or p2");
  if (getProducedStage() && !isProductStage(*getProducedStage()))
    return emitOpError("produced_stage must be p0, p1, or p2");
  return success();
}

LogicalResult TargetManifestOp::verify() {
  llvm::StringSet<> declared;
  for (Attribute value : getCapabilities()) {
    auto capability = dyn_cast<StringAttr>(value);
    if (!capability || capability.getValue().empty())
      return emitOpError("capabilities must be nonempty strings");
    if (!declared.insert(capability.getValue()).second)
      return emitOpError("contains duplicate capability '")
             << capability.getValue() << "'";
  }
  llvm::StringSet<> derived;
  for (Attribute value : getRecipes()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference)
      return emitOpError("recipes must be flat symbol references");
    auto recipe = dyn_cast_or_null<LoweringRecipeOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, reference));
    if (!recipe)
      return emitOpError(
          "recipe reference must resolve to qlx.lowering_recipe");
    if (!derived.insert(recipe.getCapability()).second)
      return emitOpError("contains duplicate recipe capability '")
             << recipe.getCapability() << "'";
  }
  if (declared.size() != derived.size())
    return emitOpError(
        "capabilities must equal referenced recipe capabilities");
  for (auto &entry : declared)
    if (!derived.contains(entry.getKey()))
      return emitOpError(
          "capabilities must equal referenced recipe capabilities");
  return success();
}

LogicalResult ExperimentOp::verify() {
  static constexpr StringLiteral allowedStages[] = {"p0", "p1", "p2"};
  if (llvm::none_of(allowedStages,
                    [&](StringRef stage) { return getStage() == stage; }))
    return emitOpError("stage must be one of p0, p1, or p2");

  for (Attribute value : getFacets())
    if (!isa<StringAttr>(value))
      return emitOpError("facets entries must be strings");

  auto root = SymbolTable::lookupNearestSymbolFrom(*this, getRootAttr());
  if (!root || !root->hasAttr(SymbolTable::getSymbolAttrName()))
    return emitOpError("root must resolve to a module-level symbol");

  llvm::StringSet<> closure;
  bool containsRoot = false;
  for (Attribute value : getClosure()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(value);
    if (!reference)
      return emitOpError("closure entries must be flat symbol references");
    if (!closure.insert(reference.getValue()).second)
      return emitOpError("contains duplicate closure symbol '")
             << reference.getValue() << "'";
    if (!SymbolTable::lookupNearestSymbolFrom(*this, reference))
      return emitOpError("closure reference must resolve within the module");
    containsRoot |= reference == getRootAttr();
  }
  if (!containsRoot)
    return emitOpError("closure must contain the selected root");

  for (Attribute value : getPassRecipe()) {
    auto entry = dyn_cast<DictionaryAttr>(value);
    auto name = entry ? entry.getAs<StringAttr>("name") : StringAttr();
    auto options =
        entry ? entry.getAs<DictionaryAttr>("options") : DictionaryAttr();
    if (!entry || !name || name.empty() || !options)
      return emitOpError(
          "pass_recipe entries require nonempty name and dictionary options");
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Generated dialect definition
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/QLX/IR/QLXDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated type definitions (storage classes become complete here)
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/QLX/IR/QLXTypes.cpp.inc"

Type QLXDialect::parseType(DialectAsmParser &parser) const {
  SMLoc location = parser.getCurrentLocation();
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)))
    return {};
  if (mnemonic == LogicalQubitType::getMnemonic())
    return LogicalQubitType::get(parser.getContext());
  parser.emitError(location) << "unknown type in dialect 'qlx': " << mnemonic;
  return {};
}

void QLXDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (isa<LogicalQubitType>(type)) {
    printer << LogicalQubitType::getMnemonic();
    return;
  }
  llvm_unreachable("attempted to print an unregistered QLX type");
}

//===----------------------------------------------------------------------===//
// Generated enum definitions
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/QLX/IR/QLXEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated attribute definitions (storage classes become complete here)
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "qlx/Dialect/QLX/IR/QLXAttrs.cpp.inc"

LogicalResult
CliffordActionAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                           DenseI64ArrayAttr matrix, DenseI64ArrayAttr phases,
                           ArrayAttr ports) {
  const int64_t arity = static_cast<int64_t>(ports.size());
  const int64_t width = 2 * arity;
  if (static_cast<int64_t>(matrix.size()) != width * width)
    return emitError() << "matrix must contain " << width * width
                       << " row-major entries for " << arity << " ports";
  if (static_cast<int64_t>(phases.size()) != width)
    return emitError() << "phases must contain one sign for each of the "
                       << width << " generator images";

  llvm::StringSet<> seenPorts;
  for (Attribute port : ports) {
    auto name = dyn_cast<StringAttr>(port);
    if (!name || name.empty())
      return emitError() << "ports must be nonempty string attributes";
    if (!seenPorts.insert(name.getValue()).second)
      return emitError() << "ports must be unique";
  }
  for (int64_t bit : matrix.asArrayRef())
    if (bit != 0 && bit != 1)
      return emitError() << "matrix entries must be binary";
  for (int64_t bit : phases.asArrayRef())
    if (bit != 0 && bit != 1)
      return emitError() << "phase entries must be binary";

  ArrayRef<int64_t> values = matrix.asArrayRef();
  auto at = [&](int64_t row, int64_t column) {
    return values[row * width + column];
  };
  for (int64_t left = 0; left < width; ++left) {
    for (int64_t right = 0; right < width; ++right) {
      int64_t pairing = 0;
      for (int64_t qubit = 0; qubit < arity; ++qubit)
        pairing ^= (at(left, qubit) & at(right, arity + qubit)) ^
                   (at(left, arity + qubit) & at(right, qubit));
      const int64_t expected = (left < arity && right == arity + left) ||
                               (right < arity && left == arity + right);
      if (pairing != expected)
        return emitError()
               << "matrix does not preserve the binary symplectic form";
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// EntryOp: custom assembly format
//
// Syntax:
//   qlx.entry @name(%arg0: !qlx.region, %arg1: !qlx.region) {
//     ...
//   }
//===----------------------------------------------------------------------===//

ParseResult EntryOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseLParen())
    return failure();

  if (parser.parseOptionalRParen()) {
    do {
      OpAsmParser::Argument arg;
      if (parser.parseArgument(arg, /*allowType=*/true, /*allowAttrs=*/false))
        return failure();
      args.push_back(arg);
    } while (succeeded(parser.parseOptionalComma()));

    if (parser.parseRParen())
      return failure();
  }

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args))
    return failure();

  if (body->empty())
    body->emplaceBlock();

  EntryOp::ensureTerminator(*body, parser.getBuilder(), result.location);

  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  return success();
}

void EntryOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getSymName());
  p << '(';
  auto &entryBlock = getBody().front();
  llvm::interleaveComma(entryBlock.getArguments(), p,
                        [&](BlockArgument arg) { p.printRegionArgument(arg); });
  p << ") ";
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                /*printBlockTerminators=*/true);

  SmallVector<StringRef> elided = {SymbolTable::getSymbolAttrName()};
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), elided);
}

//===----------------------------------------------------------------------===//
// MppOp: custom assembly format
//===----------------------------------------------------------------------===//

ParseResult MppOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> allOperands;
  if (parser.parseOperandList(allOperands))
    return failure();

  std::string pauliStr;
  if (parser.parseKeyword("pauli") || parser.parseEqual() ||
      parser.parseString(&pauliStr))
    return failure();
  result.addAttribute("pauli", parser.getBuilder().getStringAttr(pauliStr));

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  if (parser.parseColon())
    return failure();

  SmallVector<Type> resultTypes;
  if (parser.parseTypeList(resultTypes))
    return failure();

  unsigned numQubits = 0;
  for (auto ty : resultTypes) {
    if (isa<qlx::LQBitType>(ty))
      numQubits++;
  }

  if (allOperands.size() != 2 * numQubits)
    return parser.emitError(parser.getNameLoc(),
                            "expected 2*N operands (N qubits + N regions)");

  auto lqbitType = qlx::LQBitType::get(parser.getContext());
  auto regionType = qlx::RegionType::get(parser.getContext());

  SmallVector<OpAsmParser::UnresolvedOperand> qubits(
      allOperands.begin(), allOperands.begin() + numQubits);
  SmallVector<OpAsmParser::UnresolvedOperand> regions(
      allOperands.begin() + numQubits, allOperands.end());

  SmallVector<Type> qubitTypes(numQubits, lqbitType);
  SmallVector<Type> regionTypes(numQubits, regionType);

  if (parser.resolveOperands(qubits, qubitTypes, parser.getNameLoc(),
                             result.operands) ||
      parser.resolveOperands(regions, regionTypes, parser.getNameLoc(),
                             result.operands))
    return failure();

  result.addAttribute(MppOp::getOperandSegmentSizeAttr(),
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {(int32_t)numQubits, (int32_t)numQubits}));
  result.addAttribute(
      MppOp::getResultSegmentSizeAttr(),
      parser.getBuilder().getDenseI32ArrayAttr({(int32_t)numQubits, 1}));

  result.addTypes(resultTypes);
  return success();
}

void MppOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printOperands(getQubits());
  if (!getQubits().empty() && !getQubitRegions().empty())
    p << ", ";
  p.printOperands(getQubitRegions());
  p << " pauli = \"" << getPauli() << "\"";

  // Pauli is rendered above; everything else (operand_segment_sizes /
  // result_segment_sizes) lives in Properties storage and isn't in the
  // discardable attr dict, so no elision is needed.
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      /*elidedAttrs=*/{"pauli", "operandSegmentSizes", "resultSegmentSizes"});
  p << " : ";
  llvm::interleaveComma((*this)->getResultTypes(), p);
}

LogicalResult MppOp::verify() {
  StringRef p = getPauli();
  size_t nq = getQubits().size();
  if (p.size() != nq)
    return emitOpError() << "pauli length (" << p.size()
                         << ") must equal qubit count (" << nq << ")";
  for (char c : p)
    if (c != 'X' && c != 'Y' && c != 'Z')
      return emitOpError() << "pauli must contain only X/Y/Z; got '" << c
                           << "'";
  if (getQubitRegions().size() != nq)
    return emitOpError() << "qubit_regions count (" << getQubitRegions().size()
                         << ") must equal qubits count (" << nq << ")";
  if (getQubitResults().size() != nq)
    return emitOpError() << "qubit_results count (" << getQubitResults().size()
                         << ") must equal qubits count (" << nq << ")";
  if (getBitResults().size() != 1)
    return emitOpError() << "bit_results must have exactly 1 entry; got "
                         << getBitResults().size();
  return success();
}

LogicalResult RppOp::verify() {
  StringRef p = getPauliProduct();
  size_t nq = getQubits().size();
  if (nq == 0)
    return emitOpError("requires at least one qubit operand");
  if (p.size() != nq)
    return emitOpError() << "pauli_product length (" << p.size()
                         << ") must equal qubit count (" << nq << ")";
  for (char c : p)
    if (c != 'X' && c != 'Y' && c != 'Z')
      return emitOpError() << "pauli_product must contain only X/Y/Z; got '"
                           << c << "'";
  if (getQubitRegions().size() != nq)
    return emitOpError() << "qubit_regions count (" << getQubitRegions().size()
                         << ") must equal qubits count (" << nq << ")";
  if (getQubitResults().size() != nq)
    return emitOpError() << "qubit_results count (" << getQubitResults().size()
                         << ") must equal qubits count (" << nq << ")";
  StringRef synthesis = getSynthesis();
  if (synthesis != "auto" && synthesis != "native" && synthesis != "decompose")
    return emitOpError("synthesis must be one of auto, native, decompose; got ")
           << synthesis;
  return success();
}

//===----------------------------------------------------------------------===//
// QLX P0 verification
//===----------------------------------------------------------------------===//

static LogicalResult verifyCallSignature(Operation *call, FunctionType type,
                                         ValueRange inputs, TypeRange results) {
  if (inputs.size() != type.getNumInputs() ||
      !llvm::equal(inputs.getTypes(), type.getInputs()))
    return call->emitOpError("operand types ")
           << inputs.getTypes() << " do not match objective inputs "
           << type.getInputs();
  if (results.size() != type.getNumResults() ||
      !llvm::equal(results, type.getResults()))
    return call->emitOpError("result types ")
           << results << " do not match objective results "
           << type.getResults();
  return success();
}

template <typename SymbolOp>
static SymbolOp lookupVisibleSymbol(Operation *from, FlatSymbolRefAttr ref) {
  for (Operation *scope = from; scope; scope = scope->getParentOp()) {
    if (!scope->hasTrait<OpTrait::SymbolTable>())
      continue;
    if (auto symbol = dyn_cast_or_null<SymbolOp>(
            SymbolTable::lookupSymbolIn(scope, ref.getValue())))
      return symbol;
  }
  return {};
}

LogicalResult ProgramOp::verify() {
  auto stage = (*this)->getAttrOfType<StringAttr>("qlx.stage");
  auto profile = (*this)->getAttrOfType<StringAttr>("qlx.profile");
  if ((!stage || stage.getValue() != "p0") &&
      (!profile || profile.getValue() != "p0"))
    return emitOpError("requires qlx.stage = \"p0\"");
  if (stage && profile && stage.getValue() != profile.getValue())
    return emitOpError(
        "requires matching qlx.stage and compatibility qlx.profile");

  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one entry block");
  Block &entry = getBody().front();
  FunctionType type = getFunctionType();
  if (entry.getNumArguments() != type.getNumInputs() ||
      !llvm::equal(entry.getArgumentTypes(), type.getInputs()))
    return emitOpError("entry argument types ")
           << entry.getArgumentTypes() << " do not match function inputs "
           << type.getInputs();

  auto ret = dyn_cast<ReturnOp>(entry.getTerminator());
  if (!ret)
    return emitOpError("must terminate with qlx.return");
  if (ret.getNumOperands() != type.getNumResults() ||
      !llvm::equal(ret.getOperandTypes(), type.getResults()))
    return emitOpError("return operand types ")
           << ret.getOperandTypes() << " do not match function results "
           << type.getResults();

  LogicalResult legality = success();
  getBody().walk([&](Operation *op) {
    if (failed(legality) || op == getOperation())
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if (ns == "lvm" || ns == "fabric" || ns == "phys" || ns == "rt_sched" ||
        ns == "rt_decode" || ns == "rt_abi") {
      op->emitError("is not legal inside a P0 qlx.program");
      legality = failure();
      return;
    }
    for (Type type : op->getOperandTypes()) {
      if (isa<RegionType>(type)) {
        op->emitError("P0 may not carry !qlx.region values");
        legality = failure();
        return;
      }
    }
    for (Type type : op->getResultTypes()) {
      if (isa<RegionType>(type)) {
        op->emitError("P0 may not carry !qlx.region values");
        legality = failure();
        return;
      }
    }
  });
  return legality;
}

LogicalResult ObjectiveBodyOp::verify() {
  auto stage = (*this)->getAttrOfType<StringAttr>("qlx.stage");
  auto profile = (*this)->getAttrOfType<StringAttr>("qlx.profile");
  if ((!stage || stage.getValue() != "p0") &&
      (!profile || profile.getValue() != "p0"))
    return emitOpError("requires qlx.stage = \"p0\"");
  if (stage && profile && stage.getValue() != profile.getValue())
    return emitOpError(
        "requires matching qlx.stage and compatibility qlx.profile");

  auto kind = (*this)->getAttrOfType<StringAttr>("objective_kind");
  if (!kind || (kind.getValue() != "action" && kind.getValue() != "instrument"))
    return emitOpError("objective_kind must be action or instrument");

  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one entry block");
  Block &entry = getBody().front();
  FunctionType type = getFunctionType();
  if (entry.getNumArguments() != type.getNumInputs() ||
      !llvm::equal(entry.getArgumentTypes(), type.getInputs()))
    return emitOpError("entry argument types ")
           << entry.getArgumentTypes() << " do not match function inputs "
           << type.getInputs();

  auto ret = dyn_cast<ReturnOp>(entry.getTerminator());
  if (!ret)
    return emitOpError("must terminate with qlx.return");
  if (ret.getNumOperands() != type.getNumResults() ||
      !llvm::equal(ret.getOperandTypes(), type.getResults()))
    return emitOpError("return operand types ")
           << ret.getOperandTypes() << " do not match function results "
           << type.getResults();

  static const llvm::StringSet<> forbidden = [] {
    llvm::StringSet<> names;
    for (StringRef name :
         {"qlx.resource_request", "qlx.event_test", "qlx.event_poll",
          "qlx.event_is", "qlx.event_select_ready", "qlx.event_try_take",
          "qlx.event_cancel", "qlx.event_await", "qlx.fence",
          "qlx.consume_resource"})
      names.insert(name);
    return names;
  }();

  LogicalResult legality = success();
  getBody().walk([&](Operation *op) {
    if (failed(legality) || op == getOperation())
      return;
    StringRef ns = op->getName().getDialectNamespace();
    if (ns == "lvm" || ns == "fabric" || ns == "phys" || ns == "rt_sched" ||
        ns == "rt_decode" || ns == "rt_abi") {
      op->emitError("is not legal inside a P0 qlx.objective_body");
      legality = failure();
      return;
    }
    if (forbidden.contains(op->getName().getStringRef())) {
      op->emitError("runtime and resource orchestration is not legal inside "
                    "a closed logical objective");
      legality = failure();
    }
  });
  return legality;
}

LogicalResult ApplyOp::verify() {
  if (auto reference = dyn_cast<FlatSymbolRefAttr>(getActionAttr())) {
    auto objective = lookupVisibleSymbol<ActionOp>(getOperation(), reference);
    if (!objective)
      return emitOpError("references unknown qlx.action ") << reference;
    return verifyCallSignature(getOperation(), objective.getFunctionType(),
                               getInputs(), getResultTypes());
  }
  auto builtin = dyn_cast<BuiltinActionAttr>(getActionAttr());
  if (!builtin)
    return emitOpError(
        "action must be #qlx.action<...> or a qlx.action symbol reference");
  unsigned expectedArity = 1;
  switch (builtin.getValue()) {
  case BuiltinAction::cx:
  case BuiltinAction::cz:
    expectedArity = 2;
    break;
  case BuiltinAction::ccz:
    expectedArity = 3;
    break;
  case BuiltinAction::pauli_rotation:
    expectedArity = 0;
    break;
  default:
    break;
  }
  unsigned quantumInputs =
      llvm::count_if(getInputs().getTypes(),
                     [](Type type) { return isa<LogicalQubitType>(type); });
  unsigned quantumResults = llvm::count_if(
      getResultTypes(), [](Type type) { return isa<LogicalQubitType>(type); });
  if (quantumInputs == 0 || quantumInputs != quantumResults)
    return emitOpError(
        "built-in actions must preserve at least one logical-qubit owner");
  if (expectedArity && quantumInputs != expectedArity)
    return emitOpError("built-in action has the wrong logical arity");
  if (builtin.getValue() == BuiltinAction::pauli_rotation &&
      (getInputs().size() != quantumInputs + 1 ||
       !getInputs().back().getType().isF64()))
    return emitOpError(
        "Pauli rotation requires one trailing f64 angle operand");
  return success();
}

LogicalResult InstrumentOp::verify() {
  if (auto reference = dyn_cast<FlatSymbolRefAttr>(getInstrumentAttr())) {
    auto objective =
        lookupVisibleSymbol<InstrumentDeclOp>(getOperation(), reference);
    if (!objective)
      return emitOpError("references unknown qlx.instrument_decl ")
             << reference;
    return verifyCallSignature(getOperation(), objective.getFunctionType(),
                               getInputs(), getResultTypes());
  }
  auto builtin = dyn_cast<BuiltinInstrumentAttr>(getInstrumentAttr());
  if (!builtin)
    return emitOpError(
        "instrument must be #qlx.instrument<...> or a declaration reference");
  unsigned quantumInputs =
      llvm::count_if(getInputs().getTypes(),
                     [](Type type) { return isa<LogicalQubitType>(type); });
  unsigned quantumResults = llvm::count_if(
      getResultTypes(), [](Type type) { return isa<LogicalQubitType>(type); });
  unsigned classicalResults = getNumResults() - quantumResults;
  if (builtin.getValue() == BuiltinInstrument::mpp &&
      (quantumInputs == 0 || quantumInputs != quantumResults ||
       classicalResults != 1 ||
       !getResult(getNumResults() - 1).getType().isInteger(1)))
    return emitOpError(
        "MPP must preserve its logical owners and return one i1 outcome");
  return success();
}

LogicalResult PrepareOp::verify() {
  if (getState().empty())
    return emitOpError("state must not be empty");
  if (static_cast<bool>(getAllocationAttr()) !=
      static_cast<bool>(getValueIndexAttr()))
    return emitOpError("allocation and value_index must either both be present "
                       "or both absent");
  if ((getAllocationAttr() && getAllocationAttr().getInt() < 0) ||
      (getValueIndexAttr() && getValueIndexAttr().getInt() < 0))
    return emitOpError("allocation coordinates must be nonnegative");
  return success();
}

LogicalResult CallOp::verify() {
  auto callee = lookupVisibleSymbol<ProgramOp>(getOperation(), getCalleeAttr());
  if (!callee)
    return emitOpError("references unknown qlx.program ") << getCalleeAttr();
  if (callee.getEstimateOnly()) {
    auto caller = (*this)->getParentOfType<ProgramOp>();
    if (caller && !caller.getEstimateOnly())
      return emitOpError("executable qlx.program cannot call estimate-only ")
             << getCalleeAttr();
  }
  return verifyCallSignature(getOperation(), callee.getFunctionType(),
                             getInputs(), getResultTypes());
}

LogicalResult IdleOp::verify() {
  if (getInputs().empty())
    return emitOpError("requires at least one logical qubit");
  if (getInputs().size() != getResults().size())
    return emitOpError("must return one successor for each input");
  return success();
}

LogicalResult ResourceRequestOp::verify() {
  auto eventType = getEvent().getType();
  if (eventType.getOwnership() != "linear")
    return emitOpError("resource request event payload must be linear");
  auto resourceType = dyn_cast<LogicalResourceType>(eventType.getPayload());
  if (!resourceType)
    return emitOpError("event payload must be !qlx.logical_resource");
  if (resourceType.getKind() != getKind())
    return emitOpError("resource kind must match event payload kind");
  return success();
}

static LogicalResult verifyEventState(Operation *op, StringRef state) {
  if (state != "pending" && state != "ready" && state != "failed" &&
      state != "cancelled" && state != "exhausted")
    return op->emitOpError(
        "event state must be pending, ready, failed, cancelled, or exhausted");
  return success();
}

static LogicalResult verifyReadySelection(Operation *op, ValueRange events,
                                          StringRef policy) {
  if (events.empty())
    return op->emitOpError("requires at least one event");
  Type eventType = events.front().getType();
  if (!llvm::all_of(events,
                    [&](Value event) { return event.getType() == eventType; }))
    return op->emitOpError("all selected events must have the same type");
  if (policy != "priority" && policy != "deterministic" && policy != "fair")
    return op->emitOpError("policy must be priority, deterministic, or fair");
  return success();
}

static LogicalResult verifyFenceEffects(Operation *op, ArrayAttr effects) {
  if (effects.empty())
    return op->emitOpError("requires at least one semantic effect");
  llvm::StringSet<> seen;
  for (Attribute effect : effects) {
    auto value = dyn_cast<StringAttr>(effect);
    if (!value)
      return op->emitOpError("effects must be strings");
    StringRef name = value.getValue();
    if (name != "all" && name != "quantum" && name != "classical" &&
        name != "resource" && name != "event" && name != "frame" &&
        name != "outcome" && name != "selection")
      return op->emitOpError("unknown semantic effect '") << name << "'";
    if (!seen.insert(name).second)
      return op->emitOpError("semantic effects must be unique");
  }
  if (seen.contains("all") && effects.size() != 1)
    return op->emitOpError(
        "effect 'all' cannot be combined with other effects");
  return success();
}

LogicalResult EventIsOp::verify() {
  return verifyEventState(getOperation(), getState());
}

LogicalResult EventSelectReadyOp::verify() {
  return verifyReadySelection(getOperation(), getEvents(), getPolicy());
}

LogicalResult EventTryTakeOp::verify() {
  if (getCarries().getTypes() != getResultTypes())
    return emitOpError("carry and result types must match exactly");
  auto verifyBranch = [&](Region &region, Type alternative,
                          StringRef label) -> LogicalResult {
    if (!llvm::hasSingleElement(region))
      return emitOpError() << label << " region must contain one block";
    Block &block = region.front();
    if (block.getNumArguments() != getCarries().size() + 1)
      return emitOpError()
             << label
             << " region requires one alternative argument plus carries";
    if (block.getArgument(0).getType() != alternative)
      return emitOpError() << label
                           << " alternative argument has the wrong type";
    for (auto [argument, carry] :
         llvm::zip(block.getArguments().drop_front(), getCarries()))
      if (argument.getType() != carry.getType())
        return emitOpError()
               << label << " carry arguments have the wrong types";
    auto yield = dyn_cast<YieldOp>(block.getTerminator());
    if (!yield || yield.getOperandTypes() != getResultTypes())
      return emitOpError() << label << " yield types must match results";
    return success();
  };
  auto eventType = getEvent().getType();
  if (failed(verifyBranch(getReady(), eventType.getPayload(), "ready")) ||
      failed(verifyBranch(getPending(), eventType, "pending")) ||
      failed(verifyBranch(getFailed(), IntegerType::get(getContext(), 8),
                          "failed")))
    return failure();
  return success();
}

LogicalResult FenceOp::verify() {
  return verifyFenceEffects(getOperation(), getEffects());
}

LogicalResult SelectionOp::verify() {
  StringRef mode = getMode();
  if (mode != "require" && mode != "condition_results" && mode != "abort_on")
    return emitOpError("mode must be require, condition_results, or abort_on");
  bool expected = mode != "abort_on";
  if (getAcceptWhen() != expected)
    return emitOpError("accept_when disagrees with the selection mode");
  return success();
}

LogicalResult EventAwaitOp::verify() {
  if (getPayload().getType() != getEvent().getType().getPayload())
    return emitOpError("result type must match the event payload type");
  return success();
}

LogicalResult ConsumeResourceOp::verify() {
  if (getInputs().size() != getResults().size())
    return emitOpError("must return one logical-qubit successor per input");
  if (auto reference = dyn_cast<FlatSymbolRefAttr>(getActionAttr())) {
    auto objective = lookupVisibleSymbol<ActionOp>(getOperation(), reference);
    if (!objective)
      return emitOpError("references unknown qlx.action ") << reference;
    if (objective.getFunctionType().getNumInputs() != getInputs().size() ||
        objective.getFunctionType().getNumResults() != getResults().size())
      return emitOpError(
          "resource action signature does not match quantum operands");
    return success();
  }
  auto builtin = dyn_cast<BuiltinActionAttr>(getActionAttr());
  if (!builtin)
    return emitOpError(
        "resource action must be #qlx.action<...> or a declaration reference");
  unsigned expectedArity = 0;
  switch (builtin.getValue()) {
  case BuiltinAction::h:
  case BuiltinAction::s:
  case BuiltinAction::sdg:
  case BuiltinAction::x:
  case BuiltinAction::y:
  case BuiltinAction::z:
  case BuiltinAction::t:
  case BuiltinAction::tdg:
  case BuiltinAction::idle:
    expectedArity = 1;
    break;
  case BuiltinAction::cx:
  case BuiltinAction::cz:
    expectedArity = 2;
    break;
  case BuiltinAction::ccz:
    expectedArity = 3;
    break;
  case BuiltinAction::pauli_rotation:
    return emitOpError(
        "resource consumption cannot encode a parameterized Pauli rotation");
  }
  if (getInputs().size() != expectedArity)
    return emitOpError("built-in resource action has the wrong logical arity");
  return success();
}

LogicalResult FrameInitOp::verify() {
  if (getFrame().getType().getDomain() != getDomain())
    return emitOpError("frame result domain must match the requested domain");
  return success();
}

LogicalResult FrameUpdateOp::verify() {
  if (getFrame().getType() != getResult().getType())
    return emitOpError("frame update must preserve its frame domain");
  return success();
}

LogicalResult FrameTransformOp::verify() {
  if (getFrame().getType() != getResult().getType())
    return emitOpError("frame transform must preserve its frame domain");
  return success();
}

LogicalResult XorOp::verify() {
  Type type = getLhs().getType();
  if (!type.isInteger(1))
    return emitOpError("operands must be builtin i1");
  if (getRhs().getType() != type || getResult().getType() != type)
    return emitOpError("operand and result types must be identical");
  return success();
}

LogicalResult IfOp::verify() {
  Type conditionType = getCondition().getType();
  if (!conditionType.isInteger(1))
    return emitOpError("condition must be builtin i1");
  for (Region *region : {&getThenRegion(), &getElseRegion()}) {
    if (!llvm::hasSingleElement(*region))
      return emitOpError("branches must each contain exactly one block");
    auto yield = dyn_cast<YieldOp>(region->front().getTerminator());
    if (!yield)
      return emitOpError("branches must terminate with qlx.yield");
    if (yield.getNumOperands() != getNumResults() ||
        !llvm::equal(yield.getOperandTypes(), getResultTypes()))
      return emitOpError("branch yield types must match qlx.if results");
  }
  return success();
}

LogicalResult WhileOp::verify() {
  if (getMaxIterationsAttr() && getMaxIterationsAttr().getInt() <= 0)
    return emitOpError("max_iterations must be positive when present");
  if (!llvm::hasSingleElement(getBeforeRegion()) ||
      !llvm::hasSingleElement(getAfterRegion()))
    return emitOpError("before and after regions must each contain one block");
  if (getInits().getTypes() != getResultTypes())
    return emitOpError("init and result types must be identical");

  Block &before = getBeforeRegion().front();
  Block &after = getAfterRegion().front();
  if (before.getArgumentTypes() != getResultTypes() ||
      after.getArgumentTypes() != getResultTypes())
    return emitOpError(
        "before/after block arguments must match the carried result types");

  auto condition = dyn_cast<WhileConditionOp>(before.getTerminator());
  if (!condition)
    return emitOpError("before region must terminate with qlx.while_condition");
  if (condition.getForwarded().getTypes() != getResultTypes())
    return emitOpError(
        "while_condition forwarded types must match loop result types");

  auto yield = dyn_cast<YieldOp>(after.getTerminator());
  if (!yield)
    return emitOpError("after region must terminate with qlx.yield");
  if (yield.getOperandTypes() != getResultTypes())
    return emitOpError("after-region yield types must match loop result types");
  return success();
}

//===----------------------------------------------------------------------===//
// RepeatOp — structured loop (assembly mirrors fabric.repeat)
//===----------------------------------------------------------------------===//

ParseResult RepeatOp::parse(OpAsmParser &parser, OperationState &result) {
  // Parse count.
  int64_t count;
  if (parser.parseInteger(count))
    return failure();
  result.addAttribute("count", parser.getBuilder().getI64IntegerAttr(count));

  // Parse iter args: iter(%name : type = %init, ...)
  SmallVector<OpAsmParser::Argument> iterArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<Type> initTypes;

  if (parser.parseKeyword("iter") || parser.parseLParen())
    return failure();

  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          OpAsmParser::Argument arg;
          OpAsmParser::UnresolvedOperand init;
          if (parser.parseArgument(arg, /*allowType=*/true,
                                   /*allowAttrs=*/false))
            return failure();
          if (parser.parseEqual() || parser.parseOperand(init))
            return failure();
          iterArgs.push_back(arg);
          initOperands.push_back(init);
          initTypes.push_back(arg.type);
          return success();
        }))
      return failure();
    if (parser.parseRParen())
      return failure();
  }

  // Resolve init operands.
  if (parser.resolveOperands(initOperands, initTypes,
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // Result types match iter-arg types.
  result.addTypes(initTypes);

  // Parse body region with iter-args as block arguments.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, iterArgs, /*enableNameShadowing=*/false))
    return failure();
  ensureTerminator(*body, parser.getBuilder(), result.location);

  return success();
}

void RepeatOp::print(OpAsmPrinter &printer) {
  printer << " " << getCount();

  auto &entryBlock = getBody().front();
  auto inits = getInits();
  printer << "\n    iter(";
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
    if (i > 0)
      printer << ",\n         ";
    printer.printRegionArgument(entryBlock.getArgument(i));
    printer << " = " << inits[i];
  }
  printer << ")";

  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
}

LogicalResult RepeatOp::verify() {
  if (getCountAttr().getInt() < 0)
    return emitOpError("count must be non-negative");
  Block &body = getBody().front();
  if (body.getNumArguments() != getInits().size())
    return emitOpError("body block argument count (")
           << body.getNumArguments() << ") must equal iter-init count ("
           << getInits().size() << ")";
  for (auto [arg, init] : llvm::zip(body.getArguments(), getInits()))
    if (arg.getType() != init.getType())
      return emitOpError("body block argument type ")
             << arg.getType() << " must match iter-init type "
             << init.getType();

  auto yield = cast<YieldOp>(body.getTerminator());
  if (yield.getOperands().size() != getResults().size())
    return emitOpError("yield operand count (")
           << yield.getOperands().size() << ") must equal result count ("
           << getResults().size() << ")";
  for (auto [res, yv] : llvm::zip(getResults(), yield.getOperands()))
    if (res.getType() != yv.getType())
      return emitOpError("result type ")
             << res.getType() << " must match yielded type " << yv.getType();

  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qlx/Dialect/QLX/IR/QLXOps.cpp.inc"

//===----------------------------------------------------------------------===//
// QLX Dialect initialization
//===----------------------------------------------------------------------===//

void QLXDialect::initialize() {
  // The source tree still carries alpha/migration definitions so historical
  // branches can be compared and rebased, but the product dialect registers
  // only the P0-P2 contract. Unregistered physical-region, asynchronous
  // resource/event, timing, route, and execution operations fail at parse.
  addTypes<LogicalQubitType>();
  addAttributes<BuiltinActionAttr, BuiltinInstrumentAttr, PauliAttr,
                CliffordActionAttr>();
  addOperations<EstimateResultOp, DeviceOp, LogicalToQECBindingOp,
                QECLoweringOp, LoweringRecipeOp, TargetManifestOp, ExperimentOp,
                ReturnOp, ProgramOp, ObjectiveBodyOp, ActionOp,
                InstrumentDeclOp, ApplyOp, InstrumentOp, PrepareOp, MeasureOp,
                CallOp, IdleOp, DiscardOp, SelectionOp, RepeatOp, YieldOp,
                WhileConditionOp, WhileOp, IfOp, XorOp>();
}
