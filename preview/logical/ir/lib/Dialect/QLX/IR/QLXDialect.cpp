/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
//
// This is the single translation unit for the QLX dialect. It includes all
// generated .cpp.inc files for types, attributes, enums, and ops (logical
// and block-graph sub-namespaces), ensuring that storage types are complete
// at the point of dialect initialization.
//
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/Event/IR/EventTypes.h"
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

namespace {

constexpr StringLiteral allowedStages[] = {"p0", "p1", "p2", "p3"};
constexpr StringLiteral allowedProfiles[] = {
    "common", "p0", "p1", "p2", "p2s", "p2a", "p2n", "p2d", "p3"};

bool isAllowedValue(StringRef value, ArrayRef<StringLiteral> allowed) {
  return llvm::is_contained(allowed, value);
}

LogicalResult verifyStringValue(Operation *operation, NamedAttribute attribute,
                                ArrayRef<StringLiteral> allowed,
                                StringRef allowedDescription) {
  auto value = dyn_cast<StringAttr>(attribute.getValue());
  if (!value)
    return operation->emitError() << attribute.getName() << " must be a string";
  if (!isAllowedValue(value.getValue(), allowed))
    return operation->emitError()
           << attribute.getName() << " must be one of " << allowedDescription
           << "; got '" << value.getValue() << "'";
  return success();
}

LogicalResult verifyStringArray(Operation *operation, NamedAttribute attribute,
                                ArrayRef<StringLiteral> allowed,
                                StringRef allowedDescription) {
  auto values = dyn_cast<ArrayAttr>(attribute.getValue());
  if (!values)
    return operation->emitError()
           << attribute.getName() << " must be an array of strings";
  for (auto [index, item] : llvm::enumerate(values)) {
    auto value = dyn_cast<StringAttr>(item);
    if (!value)
      return operation->emitError() << attribute.getName() << " entry " << index
                                    << " must be a string";
    if (!isAllowedValue(value.getValue(), allowed))
      return operation->emitError()
             << attribute.getName() << " entry " << index << " must be one of "
             << allowedDescription << "; got '" << value.getValue() << "'";
  }
  return success();
}

} // namespace

LogicalResult QLXDialect::verifyOperationAttribute(Operation *operation,
                                                   NamedAttribute attribute) {
  StringRef name = attribute.getName().getValue();
  if (name == "qlx.stage")
    return verifyStringValue(operation, attribute, allowedStages,
                             "p0, p1, p2, or p3");
  if (name == "qlx.stages")
    return verifyStringArray(operation, attribute, allowedStages,
                             "p0, p1, p2, or p3");
  if (name == "qlx.profile")
    return verifyStringValue(operation, attribute, allowedProfiles,
                             "common, p0, p1, p2, p2s, p2a, p2n, p2d, or p3");
  if (name == "qlx.profiles")
    return verifyStringArray(operation, attribute, allowedProfiles,
                             "common, p0, p1, p2, p2s, p2a, p2n, p2d, or p3");
  return success();
}

LogicalResult EstimateResultOp::verify() {
  static constexpr StringLiteral tiers[] = {"logical", "static", "analytical",
                                            "schedule", "twin"};
  if (!llvm::is_contained(tiers, getTier()))
    return emitOpError(
        "tier must be logical, static, analytical, schedule, or twin");
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
  } else if (getTier() == "analytical") {
    if (!getDeviceAttr() || !lower || lower.getTier() != "static" ||
        lower.getRootAttr() != getRootAttr() ||
        lower.getDeviceAttr() != getDeviceAttr())
      return emitOpError("analytical tier requires matching static root/device "
                         "closure");
  } else if (getTier() == "schedule") {
    Operation *graph =
        SymbolTable::lookupNearestSymbolFrom(*this, getRootAttr());
    if (!graph || graph->getName().getStringRef() != "phys.graph")
      return emitOpError("schedule tier root must resolve to phys.graph");
    if (!lower || !getDeviceAttr() || lower.getTier() != "analytical" ||
        lower.getDeviceAttr() != getDeviceAttr())
      return emitOpError(
          "schedule tier requires a matching analytical device closure");
    auto device = dyn_cast_or_null<DeviceOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, getDeviceAttr()));
    if (!device ||
        graph->getAttrOfType<FlatSymbolRefAttr>("source_protocol") !=
            lower.getRootAttr() ||
        graph->getAttrOfType<FlatSymbolRefAttr>("architecture") !=
            device.getPhysicalAttr())
      return emitOpError(
          "schedule root must refine the analytical protocol/device graph");
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
  if (has("physical") && !has("qec"))
    return emitOpError("cannot skip QEC between logical and physical");
  if (has("logical_to_qec") != has("qec"))
    return emitOpError(
        "logical_to_qec must be present exactly when qec is present");
  if (has("qec_to_physical") != has("physical"))
    return emitOpError(
        "qec_to_physical must be present exactly when physical is present");
  if (has("operating_point") && !has("physical"))
    return emitOpError("operating_point requires a physical machine");

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
      failed(verifyReference("physical", "phys.machine")) ||
      failed(verifyReference("logical_to_qec", "qlx.logical_to_qec")) ||
      failed(verifyReference("qec_to_physical", "qlx.qec_to_physical")) ||
      failed(verifyReference("operating_point", "phys.operating_point")))
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
  if (failed(verifyRefinementEndpoints("logical_to_qec", {"logical", "qec"})) ||
      failed(verifyRefinementEndpoints("qec_to_physical", {"qec", "physical"})))
    return failure();

  if (has("physical")) {
    auto qecRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("qec");
    auto physicalRef = (*this)->getAttrOfType<FlatSymbolRefAttr>("physical");
    Operation *qecMachine =
        qecRef ? SymbolTable::lookupNearestSymbolFrom(*this, qecRef) : nullptr;
    Operation *physicalMachine =
        physicalRef ? SymbolTable::lookupNearestSymbolFrom(*this, physicalRef)
                    : nullptr;
    if (qecMachine && physicalMachine) {
      llvm::StringSet<> selected;
      qecMachine->walk([&](Operation *operation) {
        if (operation->getName().getStringRef() != "fabric.interconnect" ||
            !operation->hasAttr("logical_channel"))
          return;
        if (auto name = operation->getAttrOfType<StringAttr>(
                SymbolTable::getSymbolAttrName()))
          selected.insert(name.getValue());
      });

      llvm::StringMap<unsigned> coverage;
      LogicalResult bindingStatus = success();
      physicalMachine->walk([&](Operation *operation) {
        if (failed(bindingStatus) ||
            operation->getName().getStringRef() != "phys.qec_channel_binding")
          return;
        auto reference = operation->getAttrOfType<SymbolRefAttr>("qec_channel");
        if (!reference ||
            reference.getRootReference().getValue() != qecRef.getValue() ||
            reference.getNestedReferences().size() != 1) {
          emitOpError(
              "physical QEC-channel binding must target this device's QEC "
              "machine");
          bindingStatus = failure();
          return;
        }
        StringRef name = reference.getLeafReference().getValue();
        if (!selected.contains(name)) {
          emitOpError() << "physical QEC-channel binding targets unselected "
                           "interconnect @"
                        << name;
          bindingStatus = failure();
          return;
        }
        ++coverage[name];
      });
      if (failed(bindingStatus))
        return failure();
      for (StringRef name : selected.keys()) {
        unsigned count = coverage.lookup(name);
        if (count != 1)
          return emitOpError()
                 << "selected P2 interconnect @" << name
                 << " must have exactly one P3 qec_channel_binding; found "
                 << count;
      }
    }
  }

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

LogicalResult QECToPhysicalBindingOp::verify() {
  Operation *qecMachine =
      SymbolTable::lookupNearestSymbolFrom(*this, getQecAttr());
  Operation *physicalMachine =
      SymbolTable::lookupNearestSymbolFrom(*this, getPhysicalAttr());
  if (qecMachine && qecMachine->getName().getStringRef() != "fabric.machine")
    return emitOpError("qec must resolve to fabric.machine");
  if (physicalMachine &&
      physicalMachine->getName().getStringRef() != "phys.machine")
    return emitOpError("physical must resolve to phys.machine");

  llvm::StringSet<> qecRegions;
  llvm::StringSet<> physicalBindings;
  for (Attribute raw : getEntries()) {
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto qec = entry ? entry.getAs<StringAttr>("qec") : nullptr;
    auto binding = entry ? entry.getAs<StringAttr>("binding") : nullptr;
    auto resources = entry ? entry.getAs<ArrayAttr>("resources") : nullptr;
    if (!entry || !qec || qec.getValue().empty() || !binding ||
        binding.getValue().empty() || !resources || resources.empty())
      return emitOpError(
          "entries require nonempty qec, binding, and resources");
    if (!qecRegions.insert(qec.getValue()).second)
      return emitOpError("contains duplicate QEC region '")
             << qec.getValue() << "'";
    if (!physicalBindings.insert(binding.getValue()).second)
      return emitOpError("contains duplicate physical binding '")
             << binding.getValue() << "'";

    if (qecMachine) {
      Operation *target = SymbolTable(qecMachine).lookup(qec.getValue());
      if (!target || target->getName().getStringRef() != "fabric.region")
        return emitOpError("QEC region @")
               << qec.getValue()
               << " must resolve inside the referenced fabric.machine";
    }
    if (!physicalMachine)
      continue;
    Operation *physicalBinding =
        SymbolTable(physicalMachine).lookup(binding.getValue());
    if (!physicalBinding ||
        physicalBinding->getName().getStringRef() != "phys.qec_binding")
      return emitOpError("binding @")
             << binding.getValue()
             << " must resolve inside the referenced phys.machine";

    auto boundQEC = physicalBinding->getAttrOfType<SymbolRefAttr>("qec_region");
    if (!boundQEC ||
        boundQEC.getRootReference().getValue() != getQecAttr().getValue() ||
        boundQEC.getLeafReference().getValue() != qec.getValue())
      return emitOpError("entry for QEC region @")
             << qec.getValue() << " contradicts phys.qec_binding @"
             << binding.getValue();

    auto boundResources =
        physicalBinding->getAttrOfType<ArrayAttr>("resources");
    if (!boundResources || boundResources.size() != resources.size())
      return emitOpError("resources for physical binding @")
             << binding.getValue() << " must exactly match phys.qec_binding";
    llvm::StringSet<> expectedResources;
    for (Attribute value : resources) {
      auto name = dyn_cast<StringAttr>(value);
      if (!name || name.getValue().empty() ||
          !expectedResources.insert(name.getValue()).second)
        return emitOpError(
            "entry resources must contain unique nonempty strings");
    }
    for (Attribute value : boundResources) {
      auto reference = dyn_cast<FlatSymbolRefAttr>(value);
      if (!reference || !expectedResources.contains(reference.getValue()))
        return emitOpError("resources for physical binding @")
               << binding.getValue() << " must exactly match phys.qec_binding";
    }

    auto verifyOptionalReference = [&](StringRef field) -> LogicalResult {
      auto entryName = entry.getAs<StringAttr>(field);
      auto boundName = physicalBinding->getAttrOfType<FlatSymbolRefAttr>(field);
      if (static_cast<bool>(entryName) != static_cast<bool>(boundName) ||
          (entryName && entryName.getValue() != boundName.getValue()))
        return emitOpError()
               << field << " for physical binding @" << binding.getValue()
               << " must exactly match phys.qec_binding";
      return success();
    };
    if (failed(verifyOptionalReference("topology")) ||
        failed(verifyOptionalReference("patch_topology")))
      return failure();
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

  if (auto selections = getRppSelectionsAttr()) {
    if (getObjectiveFamily() != "pauli_product_rotation")
      return emitOpError(
          "rpp_selections requires pauli_product_rotation objective_family");
    for (NamedAttribute entry : selections) {
      StringRef adapterName = entry.getName().getValue();
      auto selection = dyn_cast<DictionaryAttr>(entry.getValue());
      if (adapterName.empty() || !selection || selection.size() != 2)
        return emitOpError(
            "rpp_selections entries require exactly strategy and "
            "implementation");
      auto strategy = selection.getAs<StringAttr>("strategy");
      auto implementation =
          selection.getAs<FlatSymbolRefAttr>("implementation");
      if (!strategy || !implementation ||
          (strategy.getValue() != "clifford" &&
           strategy.getValue() != "t_injection" &&
           strategy.getValue() != "native" &&
           strategy.getValue() != "rotation_state" &&
           strategy.getValue() != "synthesis"))
        return emitOpError(
            "rpp_selections entries require a canonical strategy and flat "
            "implementation reference");
      if (!dependencies.contains(adapterName))
        return emitOpError("RPP adapter @")
               << adapterName << " must be a declared dependency";
      Operation *adapter = SymbolTable::lookupNearestSymbolFrom(
          *this, FlatSymbolRefAttr::get(getContext(), adapterName));
      Operation *implementationTarget =
          SymbolTable::lookupNearestSymbolFrom(*this, implementation);
      if (!adapter || adapter->getName().getStringRef() != "fabric.protocol")
        return emitOpError("RPP selection adapter @")
               << adapterName << " must resolve to fabric.protocol";
      if (!implementationTarget ||
          (implementationTarget->getName().getStringRef() != "fabric.gadget" &&
           implementationTarget->getName().getStringRef() != "fabric.protocol"))
        return emitOpError("RPP selection implementation ")
               << implementation << " must resolve to fabric.gadget or "
               << "fabric.protocol";
    }
  }

  llvm::StringSet<> consumedResources;
  if (auto resources = getConsumesResources())
    for (Attribute value : *resources) {
      auto resource = dyn_cast<StringAttr>(value);
      if (!resource || resource.getValue().empty())
        return emitOpError(
            "consumes_resources must contain nonempty resource-kind names");
      if (!consumedResources.insert(resource.getValue()).second)
        return emitOpError("contains duplicate consumed resource kind ")
               << resource;
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
  static constexpr StringLiteral allowedStages[] = {"p0", "p1", "p2", "p3"};
  if (llvm::none_of(allowedStages,
                    [&](StringRef stage) { return getStage() == stage; }))
    return emitOpError("stage must be one of p0, p1, p2, or p3");

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
         {"qlx.resource_request", "event.test", "event.poll", "event.is",
          "event.select_ready", "event.try_take", "event.cancel", "event.await",
          "event.fence", "qlx.consume_resource"})
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
  case BuiltinAction::ccx:
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
  case BuiltinAction::ccx:
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

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qlx/Dialect/QLX/IR/QLXOps.cpp.inc"

//===----------------------------------------------------------------------===//
// QLX Dialect initialization
//===----------------------------------------------------------------------===//

void QLXDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "qlx/Dialect/QLX/IR/QLXTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "qlx/Dialect/QLX/IR/QLXAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "qlx/Dialect/QLX/IR/QLXOps.cpp.inc"
      >();
}
