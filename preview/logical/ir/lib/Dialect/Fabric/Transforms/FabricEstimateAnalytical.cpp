/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Fabric/Transforms/Passes.h"

#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include <cmath>
#include <limits>
#include <map>
#include <set>

using namespace mlir;
using namespace qlx;
using namespace qlx::fabric;

namespace qlx::fabric {
#define GEN_PASS_DEF_FABRICESTIMATEANALYTICAL
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"
} // namespace qlx::fabric

namespace {

static FailureOr<int64_t> dictionaryInteger(DictionaryAttr dictionary,
                                            StringRef name, Operation *source) {
  auto value = dictionary.getAs<IntegerAttr>(name);
  if (!value) {
    source->emitError() << "analytical estimate requires integer count '"
                        << name << "'";
    return failure();
  }
  return value.getInt();
}

struct SiteCounts {
  int64_t weighted = 0;
  int64_t idleRounds = 0;
};

struct FactorySummary {
  double resourceError = 0.0;
  double wallclock = 0.0;
  bool conservativeOutputError = false;
  bool singleShot = false;
  std::map<std::string, int64_t> physicalResources;
  llvm::SmallSetVector<StringRef, 4> producers;
};

struct RetryResourceAdjustment {
  StringAttr kind;
  SymbolRefAttr stream;
  double extraRequests = 0.0;
};

struct RetryEstimate {
  StringAttr attempt;
  StringAttr exhaustion;
  double successProbability = 1.0;
  double expectedAttempts = 1.0;
  double exhaustionProbability = 0.0;
  int64_t maxAttempts = 1;
  int64_t occurrences = 1;
  std::map<std::string, int64_t> requestsPerAttempt;
};

static LogicalResult verifyDeterministicTransportClosure(DictionaryAttr counts,
                                                         Operation *source) {
  auto operations = counts.getAs<DictionaryAttr>("operation_counts");
  auto transportCount =
      operations ? operations.getAs<IntegerAttr>("transport") : IntegerAttr{};
  if (!transportCount || transportCount.getInt() == 0)
    return success();

  auto root = source->getAttrOfType<FlatSymbolRefAttr>("root");
  auto module = source->getParentOfType<ModuleOp>();
  if (!root || !module) {
    source->emitError("analytical transport requires a retained P3 closure");
    return failure();
  }

  qlx::phys::GraphOp selected;
  for (auto graph : module.getOps<qlx::phys::GraphOp>()) {
    if (graph.getSourceProtocolAttr() != root)
      continue;
    if (selected) {
      source->emitError(
          "analytical transport requires one exact retained P3 graph");
      return failure();
    }
    selected = graph;
  }
  if (!selected) {
    source->emitError("native analytical resource model does not yet support "
                      "fabric.transport without an exact retained P3 graph");
    return failure();
  }

  SymbolTable symbols(module);
  bool sawTransport = false;
  bool valid = true;
  selected.walk([&](qlx::phys::TransportResourceOp operation) {
    sawTransport = true;
    auto modelReference = operation.getModelAttr();
    auto model = modelReference ? symbols.lookup<qlx::phys::TransportModelOp>(
                                      modelReference.getValue())
                                : qlx::phys::TransportModelOp{};
    if (!model || (model.getPolicy() != "guaranteed" &&
                   model.getPolicy() != "single_shot")) {
      operation.emitOpError(
          "Tier-3 analytical estimation requires an exact deterministic or "
          "single-shot transport model");
      valid = false;
    }
  });
  if (!sawTransport) {
    source->emitError(
        "Tier-1 transport demand has no retained P3 transport operation");
    return failure();
  }
  return success(valid);
}

struct RetrySummary {
  double expectedExtraOperationSites = 0.0;
  double acceptance = 1.0;
  double logicalError = 0.0;
  SmallVector<RetryResourceAdjustment> resourceAdjustments;
  SmallVector<RetryEstimate> estimates;
};

static FailureOr<double> finiteMetadataNumber(DictionaryAttr metadata,
                                              StringRef name,
                                              Operation *source) {
  double value = 0.0;
  Attribute raw = metadata ? metadata.get(name) : Attribute{};
  if (auto number = dyn_cast_or_null<FloatAttr>(raw))
    value = number.getValueAsDouble();
  else if (auto number = dyn_cast_or_null<IntegerAttr>(raw))
    value = static_cast<double>(number.getInt());
  else if (auto text = dyn_cast_or_null<StringAttr>(raw)) {
    if (text.getValue().getAsDouble(value))
      value = std::numeric_limits<double>::quiet_NaN();
  } else {
    value = std::numeric_limits<double>::quiet_NaN();
  }
  if (!std::isfinite(value)) {
    source->emitError() << "analytical factory metadata requires finite '"
                        << name << "'";
    return failure();
  }
  return value;
}

static FailureOr<int64_t> integerMetadataNumber(DictionaryAttr metadata,
                                                StringRef name,
                                                Operation *source) {
  auto value = finiteMetadataNumber(metadata, name, source);
  if (failed(value) || std::floor(*value) != *value || *value < 0.0 ||
      *value > static_cast<double>(std::numeric_limits<int64_t>::max())) {
    if (succeeded(value))
      source->emitError() << "analytical factory metadata requires integer '"
                          << name << "'";
    return failure();
  }
  return static_cast<int64_t>(*value);
}

static double independentUnion(double left, double right) {
  if (left <= 0.0)
    return right;
  if (right <= 0.0)
    return left;
  if (left >= 1.0 || right >= 1.0)
    return 1.0;
  return -std::expm1(std::log1p(-left) + std::log1p(-right));
}

static double independentFailure(double probability, double count) {
  if (probability <= 0.0 || count <= 0)
    return 0.0;
  if (probability >= 1.0)
    return 1.0;
  return -std::expm1(static_cast<double>(count) * std::log1p(-probability));
}

static FailureOr<RetrySummary> boundedRetrySummary(DictionaryAttr counts,
                                                   Operation *source) {
  RetrySummary summary;
  auto retries = counts.getAs<ArrayAttr>("retry_demands");
  if (!retries) {
    auto operations = counts.getAs<DictionaryAttr>("operation_counts");
    auto retryCount =
        operations ? operations.getAs<IntegerAttr>("retry") : IntegerAttr{};
    if (!retryCount || retryCount.getInt() == 0)
      return summary;
    source->emitError(
        "analytical estimate requires Tier-1 retry_demands for fabric.retry");
    return failure();
  }
  for (Attribute rawRetry : retries) {
    auto retry = dyn_cast<DictionaryAttr>(rawRetry);
    auto attempt =
        retry ? retry.getAs<FlatSymbolRefAttr>("attempt") : FlatSymbolRefAttr{};
    auto exhaustion =
        retry ? retry.getAs<StringAttr>("exhaustion") : StringAttr{};
    auto maximum =
        retry ? retry.getAs<IntegerAttr>("max_attempts") : IntegerAttr{};
    auto occurrences =
        retry ? retry.getAs<IntegerAttr>("occurrences") : IntegerAttr{};
    auto attemptSites =
        retry ? retry.getAs<IntegerAttr>("attempt_operation_sites")
              : IntegerAttr{};
    auto requests =
        retry ? retry.getAs<ArrayAttr>("resource_requests_per_attempt")
              : ArrayAttr{};
    if (!retry || !attempt || !exhaustion || !maximum || !occurrences ||
        !attemptSites || !requests || maximum.getInt() <= 0 ||
        occurrences.getInt() <= 0 || attemptSites.getInt() < 0) {
      source->emitError(
          "analytical retry demand has malformed static Tier-1 evidence");
      return failure();
    }

    auto probability = retry.getAs<FloatAttr>("success_probability");
    if (!probability) {
      if (!requests.empty()) {
        source->emitError()
            << "analytical retry @" << attempt.getValue()
            << " consumes resources but has no certified success_probability";
        return failure();
      }
      continue;
    }
    double p = probability.getValueAsDouble();
    if (!std::isfinite(p) || p <= 0.0 || p > 1.0 ||
        !retry.getAs<FlatSymbolRefAttr>("success_probability_source") ||
        !retry.getAs<StringAttr>("success_probability_evidence")) {
      source->emitError() << "analytical retry @" << attempt.getValue()
                          << " has invalid probability provenance";
      return failure();
    }

    double completion = 1.0;
    double exhaustionProbability = 0.0;
    if (p < 1.0) {
      double logExhaustion =
          static_cast<double>(maximum.getInt()) * std::log1p(-p);
      completion = -std::expm1(logExhaustion);
      exhaustionProbability = std::exp(logExhaustion);
    }
    double expectedAttempts = completion / p;
    double extraOccurrences =
        static_cast<double>(occurrences.getInt()) * (expectedAttempts - 1.0);
    double extraSites =
        extraOccurrences * static_cast<double>(attemptSites.getInt());
    if (!std::isfinite(extraSites) ||
        extraSites > std::numeric_limits<double>::max() -
                         summary.expectedExtraOperationSites) {
      source->emitError("analytical retry operation demand overflows f64");
      return failure();
    }
    summary.expectedExtraOperationSites += extraSites;

    RetryEstimate estimate;
    estimate.attempt =
        StringAttr::get(source->getContext(), attempt.getValue());
    estimate.exhaustion = exhaustion;
    estimate.successProbability = p;
    estimate.expectedAttempts = expectedAttempts;
    estimate.exhaustionProbability = exhaustionProbability;
    estimate.maxAttempts = maximum.getInt();
    estimate.occurrences = occurrences.getInt();
    for (Attribute rawRequest : requests) {
      auto request = dyn_cast<DictionaryAttr>(rawRequest);
      auto kind = request ? request.getAs<StringAttr>("kind") : StringAttr{};
      auto stream = request
                        ? dyn_cast_or_null<SymbolRefAttr>(request.get("stream"))
                        : SymbolRefAttr{};
      auto count =
          request ? request.getAs<IntegerAttr>("count") : IntegerAttr{};
      if (!request || !kind || !stream || !count || count.getInt() <= 0) {
        source->emitError(
            "analytical retry per-attempt resource demand is malformed");
        return failure();
      }
      double extraRequests =
          extraOccurrences * static_cast<double>(count.getInt());
      if (!std::isfinite(extraRequests)) {
        source->emitError("analytical retry resource demand overflows f64");
        return failure();
      }
      summary.resourceAdjustments.push_back(
          RetryResourceAdjustment{kind, stream, extraRequests});
      int64_t &perKind = estimate.requestsPerAttempt[kind.getValue().str()];
      if (count.getInt() > std::numeric_limits<int64_t>::max() - perKind) {
        source->emitError(
            "analytical retry per-kind resource demand overflows i64");
        return failure();
      }
      perKind += count.getInt();
    }

    StringRef policy = exhaustion.getValue();
    if (policy == "abort" || policy == "report_failure") {
      summary.acceptance *=
          std::pow(completion, static_cast<double>(occurrences.getInt()));
    } else if (policy == "return_last") {
      summary.logicalError = independentUnion(
          summary.logicalError,
          independentFailure(exhaustionProbability, occurrences.getInt()));
    } else {
      source->emitError() << "analytical retry @" << attempt.getValue()
                          << " has unsupported exhaustion policy '" << policy
                          << "'";
      return failure();
    }
    summary.estimates.push_back(std::move(estimate));
  }
  if (!std::isfinite(summary.acceptance) || summary.acceptance < 0.0 ||
      summary.acceptance > 1.0) {
    source->emitError("analytical retry acceptance is not a probability");
    return failure();
  }
  return summary;
}

static FailureOr<SiteCounts> operationSites(DictionaryAttr counts,
                                            Operation *source) {
  auto operations = counts.getAs<DictionaryAttr>("operation_counts");
  if (!operations) {
    source->emitError("analytical estimate requires Tier-1 operation_counts");
    return failure();
  }
  static constexpr StringLiteral structural[] = {
      "call",         "establish_support", "establish_topological_record",
      "map_children", "relocate",          "repeat"};
  SiteCounts result;
  for (NamedAttribute operation : operations) {
    if (llvm::is_contained(structural, operation.getName().strref()))
      continue;
    auto count = dyn_cast<IntegerAttr>(operation.getValue());
    if (!count || count.getInt() < 0) {
      source->emitError("analytical estimate found an invalid operation count");
      return failure();
    }
    if (operation.getName().strref() == "idle")
      continue;
    if (count.getInt() >
        std::numeric_limits<int64_t>::max() - result.weighted) {
      source->emitError("analytical operation-site count overflows signed i64");
      return failure();
    }
    result.weighted += count.getInt();
  }
  auto regions = counts.getAs<DictionaryAttr>("per_region");
  if (!regions) {
    source->emitError("analytical estimate requires Tier-1 per_region counts");
    return failure();
  }
  for (NamedAttribute region : regions) {
    auto data = dyn_cast<DictionaryAttr>(region.getValue());
    auto rounds =
        data ? data.getAs<DictionaryAttr>("rounds_by_kind") : DictionaryAttr{};
    auto idle = rounds ? rounds.getAs<IntegerAttr>("idle") : IntegerAttr{};
    if (!idle)
      continue;
    if (idle.getInt() < 0 ||
        idle.getInt() >
            std::numeric_limits<int64_t>::max() - result.idleRounds ||
        idle.getInt() > std::numeric_limits<int64_t>::max() - result.weighted) {
      source->emitError("analytical idle-round count overflows signed i64");
      return failure();
    }
    result.idleRounds += idle.getInt();
    result.weighted += idle.getInt();
  }
  return result;
}

static FailureOr<FactorySummary> resourceFactorySummary(
    DictionaryAttr counts, qlx::DeviceOp device,
    qlx::LogicalToQECBindingOp logicalToQec,
    qlx::QECToPhysicalBindingOp qecToPhysical, double physicalError,
    double cycleTime,
    const std::map<std::string, int64_t> &physicalUnitCapacity,
    const RetrySummary &retrySummary, Operation *source) {
  auto requests = counts.getAs<ArrayAttr>("resource_requests");
  auto operations = counts.getAs<DictionaryAttr>("operation_counts");
  if (!requests || !operations) {
    source->emitError(
        "analytical resource model requires exact Tier-1 resource_requests");
    return failure();
  }

  int64_t countedRequests = 0;
  if (auto count = operations.getAs<IntegerAttr>("resource_request"))
    countedRequests = count.getInt();
  int64_t detailedRequests = 0;
  for (Attribute raw : requests) {
    auto entry = dyn_cast<DictionaryAttr>(raw);
    auto count = entry ? entry.getAs<IntegerAttr>("count") : IntegerAttr{};
    if (!entry || !entry.getAs<StringAttr>("kind") ||
        !dyn_cast_or_null<SymbolRefAttr>(entry.get("stream")) || !count ||
        count.getInt() <= 0 ||
        count.getInt() >
            std::numeric_limits<int64_t>::max() - detailedRequests) {
      source->emitError(
          "analytical resource_requests entries require kind, stream, and "
          "positive nonoverflowing count");
      return failure();
    }
    detailedRequests += count.getInt();
  }
  if (detailedRequests != countedRequests) {
    source->emitError(
        "analytical resource_requests do not match Tier-1 operation counts");
    return failure();
  }

  // Resource production/consumption, Pauli-product rotations, retained
  // selections, deterministic compact transport, and certified bounded
  // retries have exact demand semantics in fabric-count. The analytical
  // result is conditional on selections; Tier 3 owns physical timing of the
  // selected branch. Transport is admitted only when the retained exact P3
  // closure binds every transfer to a verified compact model.
  if (failed(verifyDeterministicTransportClosure(counts, source)))
    return failure();
  static constexpr StringLiteral unsupported[] = {"inject", "event_try_take"};
  for (StringRef name : unsupported) {
    auto count = operations.getAs<IntegerAttr>(name);
    if (count && count.getInt() != 0) {
      source->emitError()
          << "native analytical resource model does not yet support fabric."
          << name;
      return failure();
    }
  }

  auto bindings = device->getAttrOfType<ArrayAttr>("resource_bindings");
  if (requests.empty())
    return FactorySummary{};
  if (!bindings) {
    device.emitOpError(
        "analytical resource demand requires typed resource_bindings");
    return failure();
  }

  auto logicalRegionToQec = [&](StringRef logical) -> StringAttr {
    for (Attribute raw : logicalToQec.getEntries()) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto candidate =
          entry ? entry.getAs<StringAttr>("logical") : StringAttr{};
      if (candidate && candidate.getValue() == logical)
        return entry.getAs<StringAttr>("qec");
    }
    return {};
  };
  auto qecBindingEntry = [&](StringRef qec) -> DictionaryAttr {
    for (Attribute raw : qecToPhysical.getEntries()) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto candidate = entry ? entry.getAs<StringAttr>("qec") : StringAttr{};
      if (candidate && candidate.getValue() == qec)
        return entry;
    }
    return {};
  };

  FactorySummary summary;
  std::map<std::string, double> cyclesByFactory;
  for (Attribute rawRequest : requests) {
    auto request = cast<DictionaryAttr>(rawRequest);
    auto kind = request.getAs<StringAttr>("kind");
    auto stream = cast<SymbolRefAttr>(request.get("stream"));
    double demand =
        static_cast<double>(request.getAs<IntegerAttr>("count").getInt());
    for (const RetryResourceAdjustment &adjustment :
         retrySummary.resourceAdjustments)
      if (adjustment.kind == kind && adjustment.stream == stream)
        demand += adjustment.extraRequests;
    if (!std::isfinite(demand) || demand <= 0.0) {
      source->emitError("analytical resource demand overflows finite f64");
      return failure();
    }

    DictionaryAttr binding;
    for (Attribute rawBinding : bindings) {
      auto candidate = dyn_cast<DictionaryAttr>(rawBinding);
      if (candidate && candidate.get("stream") == stream) {
        if (binding) {
          device.emitOpError("resource stream has several producer bindings");
          return failure();
        }
        binding = candidate;
      }
    }
    auto producerRef =
        binding ? dyn_cast_or_null<FlatSymbolRefAttr>(binding.get("producer"))
                : FlatSymbolRefAttr{};
    auto factoryRef =
        binding ? dyn_cast_or_null<SymbolRefAttr>(binding.get("factory"))
                : SymbolRefAttr{};
    if (!binding || !producerRef || !factoryRef) {
      device.emitOpError("resource stream ")
          << stream << " requires one backed factory producer";
      return failure();
    }
    Operation *streamTarget =
        SymbolTable::lookupNearestSymbolFrom(device, stream);
    auto produced =
        streamTarget
            ? streamTarget->getAttrOfType<FlatSymbolRefAttr>("produces")
            : FlatSymbolRefAttr{};
    if (!streamTarget ||
        streamTarget->getName().getStringRef() != "lvm.stream" || !produced ||
        produced.getValue() != kind.getValue()) {
      device.emitOpError("resource stream kind does not match Tier-1 demand");
      return failure();
    }

    Operation *factory =
        SymbolTable::lookupNearestSymbolFrom(device, factoryRef);
    auto laneCount = factory ? factory->getAttrOfType<IntegerAttr>("capacity")
                             : IntegerAttr{};
    if (!factory || factory->getName().getStringRef() != "lvm.space" ||
        !laneCount || laneCount.getInt() <= 0) {
      device.emitOpError(
          "resource factory must resolve to a positive-capacity lvm.space");
      return failure();
    }
    auto qecName = logicalRegionToQec(factoryRef.getLeafReference().getValue());
    auto physicalBinding =
        qecName ? qecBindingEntry(qecName.getValue()) : DictionaryAttr{};
    if (!qecName || !physicalBinding) {
      device.emitOpError(
          "analytical factory requires a selected physical QEC binding");
      return failure();
    }

    auto producer = dyn_cast_or_null<ProtocolOp>(
        SymbolTable::lookupNearestSymbolFrom(device, producerRef));
    if (!producer) {
      device.emitOpError(
          "resource stream producer must resolve to fabric.protocol");
      return failure();
    }
    auto metadata = producer.getMetadataAttr();
    auto factoryMode =
        metadata ? metadata.getAs<StringAttr>("factory_mode") : StringAttr{};
    auto produces =
        metadata ? metadata.getAs<StringAttr>("produces") : StringAttr{};
    bool hasAnalyticalModel = factoryMode &&
                              (factoryMode.getValue() == "analytical" ||
                               factoryMode.getValue() == "scheduled_macro") &&
                              produces &&
                              produces.getValue() == kind.getValue();

    auto intervalText = physicalBinding ? physicalBinding.getAs<StringAttr>(
                                              "factory_output_interval_cycles")
                                        : StringAttr{};
    auto policy =
        physicalBinding
            ? physicalBinding.getAs<StringAttr>("factory_model_policy")
            : StringAttr{};
    double intervalCycles = 0.0;
    if (!hasAnalyticalModel &&
        (!intervalText || !policy ||
         intervalText.getValue().getAsDouble(intervalCycles) ||
         !std::isfinite(intervalCycles) || intervalCycles <= 0.0 ||
         (policy.getValue() != "guaranteed" &&
          policy.getValue() != "single_shot"))) {
      device.emitOpError("analytical factory timing requires a closed typed "
                         "QEC-to-physical factory model");
      return failure();
    }

    int64_t depth = 1;
    double outputError = 1.0;
    int64_t qubits = 0;
    if (hasAnalyticalModel) {
      auto producerError =
          finiteMetadataNumber(metadata, "physical_error_rate", producer);
      auto metadataCycles =
          finiteMetadataNumber(metadata, "cycles_per_attempt", producer);
      auto metadataDepth =
          integerMetadataNumber(metadata, "pipeline_depth", producer);
      auto acceptance =
          finiteMetadataNumber(metadata, "acceptance_probability", producer);
      auto metadataOutputError =
          finiteMetadataNumber(metadata, "output_infidelity", producer);
      auto metadataQubits =
          integerMetadataNumber(metadata, "physical_qubits", producer);
      if (failed(producerError) || failed(metadataCycles) ||
          failed(metadataDepth) || failed(acceptance) ||
          failed(metadataOutputError) || failed(metadataQubits))
        return failure();
      depth = *metadataDepth;
      outputError = *metadataOutputError;
      qubits = *metadataQubits;
      if (*producerError != physicalError || *metadataCycles <= 0.0 ||
          depth <= 0 || *acceptance <= 0.0 || *acceptance > 1.0 ||
          outputError < 0.0 || outputError > 1.0 || qubits <= 0) {
        producer.emitOpError(
            "analytical factory metadata contradicts the selected operating "
            "point or typed producer model");
        return failure();
      }
      intervalCycles = *metadataCycles / *acceptance;
      if (!std::isfinite(intervalCycles) || intervalCycles <= 0.0) {
        producer.emitOpError(
            "analytical expected factory interval overflows finite f64");
        return failure();
      }
    } else if (auto sourceUnits = physicalBinding.getAs<IntegerAttr>(
                   "factory_source_physical_units")) {
      qubits = sourceUnits.getInt();
    }
    if (!hasAnalyticalModel) {
      summary.conservativeOutputError = true;
      summary.singleShot |= policy.getValue() == "single_shot";
    }
    auto resources = physicalBinding.getAs<ArrayAttr>("resources");
    std::string physicalResource;
    if (resources) {
      for (Attribute raw : resources) {
        auto name = dyn_cast<StringAttr>(raw);
        if (!name || !physicalUnitCapacity.count(name.getValue().str()))
          continue;
        if (!physicalResource.empty()) {
          device.emitOpError(
              "analytical factory requires one dedicated qubit resource");
          return failure();
        }
        physicalResource = name.getValue().str();
      }
    }
    if (physicalResource.empty()) {
      device.emitOpError(
          "analytical factory requires one dedicated physical resource");
      return failure();
    }
    int64_t selectedUnits = physicalUnitCapacity.at(physicalResource);
    if (qubits == 0) {
      if (selectedUnits % laneCount.getInt() != 0) {
        device.emitOpError(
            "declared factory model has no unambiguous per-lane footprint");
        return failure();
      }
      qubits = selectedUnits / laneCount.getInt();
    }
    if (qubits <= 0 ||
        qubits > std::numeric_limits<int64_t>::max() / laneCount.getInt() ||
        selectedUnits < qubits * laneCount.getInt()) {
      device.emitOpError(
          "factory physical resource must contain at least lane_count times "
          "the producer footprint");
      return failure();
    }

    std::string factoryName = factoryRef.getLeafReference().getValue().str();
    cyclesByFactory[factoryName] +=
        static_cast<double>(demand) * intervalCycles /
        static_cast<double>(laneCount.getInt() * depth);
    summary.resourceError = independentUnion(
        summary.resourceError, independentFailure(outputError, demand));
    summary.physicalResources.try_emplace(
        physicalResource, physicalUnitCapacity.at(physicalResource));
    summary.producers.insert(producer.getSymName());
  }
  for (const auto &[factory, cycles] : cyclesByFactory) {
    (void)factory;
    summary.wallclock = std::max(summary.wallclock, cycles * cycleTime);
  }
  if (!std::isfinite(summary.wallclock)) {
    source->emitError("analytical factory wallclock overflows finite f64");
    return failure();
  }
  return summary;
}

struct FabricEstimateAnalyticalPass
    : public qlx::fabric::impl::FabricEstimateAnalyticalBase<
          FabricEstimateAnalyticalPass> {
  using FabricEstimateAnalyticalBase::FabricEstimateAnalyticalBase;

  FabricEstimateAnalyticalPass(FabricEstimateAnalyticalOptions options,
                               bool countsAreFresh)
      : FabricEstimateAnalyticalBase(std::move(options)),
        countsAreFresh(countsAreFresh) {}

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SymbolTable symbols(module);
    auto countsResult =
        dyn_cast_or_null<EstimateResultOp>(symbols.lookup(countsSymbol));
    if (!countsResult || countsResult.getTier() != "static") {
      module.emitError("fabric-estimate-analytical counts @")
          << countsSymbol << " must resolve to a static qlx.estimate_result";
      return signalPassFailure();
    }
    auto countsMetadata = countsResult.getMetadata();
    auto countsProducer = countsMetadata
                              ? countsMetadata->getAs<StringAttr>("producer")
                              : StringAttr{};
    auto countsProducerVersion =
        countsMetadata ? countsMetadata->getAs<StringAttr>("producer_version")
                       : StringAttr{};
    if (countsResult.getSchema() != "qlx.fabric-counts/v1" ||
        !countsResult.getDeviceAttr() || !countsProducer ||
        countsProducer.getValue() != "fabric-count" || !countsProducerVersion ||
        countsProducerVersion.getValue() != "1") {
      countsResult.emitOpError(
          "analytical estimation requires an authenticated "
          "qlx.fabric-counts/v1 result produced by fabric-count v1");
      return signalPassFailure();
    }

    Operation *root = SymbolTable::lookupNearestSymbolFrom(
        countsResult, countsResult.getRootAttr());
    if (!rootSymbol.empty()) {
      root = symbols.lookup(rootSymbol);
      if (!root || !isa<GadgetOp, ProtocolOp, GadgetProfileOp>(root)) {
        module.emitError("fabric-estimate-analytical root @")
            << rootSymbol << " must resolve to an executable Fabric root";
        return signalPassFailure();
      }
      if (cast<StringAttr>(SymbolTable::getSymbolName(root)).getValue() !=
          countsResult.getRootAttr().getValue()) {
        module.emitError("fabric-estimate-analytical root does not match its "
                         "Tier-1 counts root");
        return signalPassFailure();
      }
    }
    if (!root || !isa<GadgetOp, ProtocolOp, GadgetProfileOp>(root)) {
      countsResult.emitOpError(
          "static result root must resolve to an executable Fabric root");
      return signalPassFailure();
    }

    if (!countsAreFresh) {
      // Producer labels are provenance metadata, not proof. Recompute Tier 1
      // from the selected executable/device closure and require byte-identical
      // typed counts before consuming a supplied result. The internal fused
      // constructor may skip this only when FabricCount created `counts` in
      // the same compiler-owned pass transaction.
      std::string recomputedSymbol = "__qlx_recomputed_static";
      while (SymbolTable::lookupSymbolIn(module, recomputedSymbol))
        recomputedSymbol.push_back('_');
      Attribute priorLegacyCounts = module->getAttr("fabric.counts");
      qlx::fabric::FabricCountOptions countOptions;
      countOptions.rootSymbol =
          cast<StringAttr>(SymbolTable::getSymbolName(root)).getValue().str();
      countOptions.deviceSymbol = countsResult.getDeviceAttr().getValue().str();
      countOptions.resultSymbol = recomputedSymbol;
      OpPassManager countPipeline(ModuleOp::getOperationName());
      countPipeline.addPass(qlx::fabric::createFabricCount(countOptions));
      if (failed(runPipeline(countPipeline, module)))
        return signalPassFailure();
      auto recomputed = dyn_cast_or_null<EstimateResultOp>(
          SymbolTable::lookupSymbolIn(module, recomputedSymbol));
      if (!recomputed || recomputed.getData() != countsResult.getData()) {
        if (recomputed)
          recomputed.erase();
        if (priorLegacyCounts)
          module->setAttr("fabric.counts", priorLegacyCounts);
        else
          module->removeAttr("fabric.counts");
        countsResult.emitOpError(
            "static counts do not match recomputation from the selected "
            "executable/device closure");
        return signalPassFailure();
      }
      recomputed.erase();
      if (priorLegacyCounts)
        module->setAttr("fabric.counts", priorLegacyCounts);
      else
        module->removeAttr("fabric.counts");
    }

    if (!std::isfinite(physicalError) || physicalError < 0.0 ||
        physicalError > 1.0) {
      module.emitError(
          "fabric-estimate-analytical requires finite p-phys in [0,1]");
      return signalPassFailure();
    }
    if (!std::isfinite(failureBudget) || failureBudget <= 0.0 ||
        failureBudget > 1.0) {
      module.emitError(
          "fabric-estimate-analytical requires finite failure-budget in (0,1]");
      return signalPassFailure();
    }
    if (!std::isfinite(cycleTime) || cycleTime <= 0.0 ||
        !std::isfinite(scalingPrefactor) || scalingPrefactor < 0.0 ||
        !std::isfinite(scalingThreshold) || scalingThreshold <= 0.0) {
      module.emitError("fabric-estimate-analytical requires finite positive "
                       "timing/scaling inputs");
      return signalPassFailure();
    }
    if (resultSymbol.empty() || symbols.lookup(resultSymbol)) {
      module.emitError("fabric-estimate-analytical result symbol must be "
                       "nonempty and unique");
      return signalPassFailure();
    }

    if (deviceSymbol.empty()) {
      module.emitError("fabric-estimate-analytical requires device= for an "
                       "authenticated physical closure");
      return signalPassFailure();
    }
    auto device = dyn_cast_or_null<qlx::DeviceOp>(symbols.lookup(deviceSymbol));
    if (!device) {
      module.emitError("fabric-estimate-analytical device @")
          << deviceSymbol << " must resolve to qlx.device";
      return signalPassFailure();
    }
    if (!countsResult.getDeviceAttr() ||
        countsResult.getDeviceAttr().getValue() != device.getSymName()) {
      module.emitError("fabric-estimate-analytical device does not match its "
                       "Tier-1 counts closure");
      return signalPassFailure();
    }
    auto gadgetRoot = dyn_cast<GadgetOp>(root);
    if (auto profile = dyn_cast<GadgetProfileOp>(root))
      gadgetRoot = symbols.lookup<GadgetOp>(profile.getGadgetAttr().getValue());
    if (!device.getQecAttr() || !device.getPhysicalAttr() ||
        !device.getLogicalToQecAttr() || !device.getQecToPhysicalAttr() ||
        !device.getOperatingPointAttr() ||
        (gadgetRoot && device.getQecAttr() != gadgetRoot.getDeviceAttr())) {
      device.emitOpError("analytical estimation requires matching logical, "
                         "QEC, physical, refinement, and operating-point "
                         "closure");
      return signalPassFailure();
    }
    auto logicalToQec = dyn_cast_or_null<qlx::LogicalToQECBindingOp>(
        symbols.lookup(device.getLogicalToQecAttr().getValue()));
    auto qecToPhysical = dyn_cast_or_null<qlx::QECToPhysicalBindingOp>(
        symbols.lookup(device.getQecToPhysicalAttr().getValue()));
    if (!logicalToQec || !qecToPhysical ||
        logicalToQec.getLogicalAttr() != device.getLogicalAttr() ||
        logicalToQec.getQecAttr() != device.getQecAttr() ||
        qecToPhysical.getQecAttr() != device.getQecAttr() ||
        qecToPhysical.getPhysicalAttr() != device.getPhysicalAttr()) {
      device.emitOpError(
          "analytical refinement bindings are unresolved or inconsistent");
      return signalPassFailure();
    }
    auto physical = dyn_cast_or_null<qlx::phys::ArchitectureOp>(
        symbols.lookup(device.getPhysicalAttr().getValue()));
    auto operatingPoint = dyn_cast_or_null<qlx::phys::OperatingPointOp>(
        symbols.lookup(device.getOperatingPointAttr().getValue()));
    if (!physical || !operatingPoint ||
        operatingPoint.getMachineAttr() != device.getPhysicalAttr()) {
      device.emitOpError(
          "analytical physical machine or operating point is unresolved");
      return signalPassFailure();
    }
    std::map<std::string, int64_t> physicalUnitCapacity;
    std::map<std::string, int64_t> resourceMemberCapacity;
    std::map<std::string, int64_t> physicalUnitsPerMember;
    for (auto resource :
         physical.getBody().getOps<qlx::phys::ResourceClassOp>()) {
      int64_t count = resource.getCountAttr().getInt();
      if (count < 0) {
        resource.emitOpError("physical resource capacity is invalid");
        return signalPassFailure();
      }
      int64_t memberUnits = 1;
      auto granularity = resource->getAttrOfType<StringAttr>("granularity");
      if (granularity && granularity.getValue() == "patch") {
        auto units = resource->getAttrOfType<IntegerAttr>("physical_units");
        if (!units || units.getInt() <= 0) {
          resource.emitOpError(
              "patch resource lacks a positive physical-unit footprint");
          return signalPassFailure();
        }
        memberUnits = units.getInt();
      } else if (resource.getKind() != "qubit") {
        continue;
      }
      if (count != 0 &&
          memberUnits > std::numeric_limits<int64_t>::max() / count) {
        resource.emitOpError("physical-unit footprint overflows signed i64");
        return signalPassFailure();
      }
      std::string name = resource.getSymName().str();
      resourceMemberCapacity[name] = count;
      physicalUnitsPerMember[name] = memberUnits;
      physicalUnitCapacity[name] = count * memberUnits;
    }
    FlatSymbolRefAttr deviceReference =
        FlatSymbolRefAttr::get(&getContext(), deviceSymbol);

    DictionaryAttr counts = countsResult.getData();
    auto retries = boundedRetrySummary(counts, countsResult);
    if (failed(retries))
      return signalPassFailure();
    FactorySummary factorySummary;
    if (auto extended =
            counts.getAs<BoolAttr>("requires_extended_analytical_model");
        extended && extended.getValue()) {
      auto summary = resourceFactorySummary(
          counts, device, logicalToQec, qecToPhysical, physicalError, cycleTime,
          physicalUnitCapacity, *retries, countsResult);
      if (failed(summary))
        return signalPassFailure();
      factorySummary = std::move(*summary);
    }
    auto logicalPeak = dictionaryInteger(counts, "logical_qubits_peak", root);
    auto sites = operationSites(counts, root);
    if (failed(logicalPeak) || failed(sites))
      return signalPassFailure();

    auto regions = counts.getAs<DictionaryAttr>("per_region");
    if (!regions) {
      countsResult.emitOpError("static result requires per_region evidence");
      return signalPassFailure();
    }
    std::map<std::string, std::string> physicalHomeByRegion;
    std::map<std::string, int64_t> patchCapacityByRegion;
    std::map<std::string, std::set<int64_t>> boundCarriersByRegion;
    std::map<std::string, SmallVector<int64_t>> carrierGroupSizesByRegion;
    std::set<std::string> regionsWithoutPatchTopology;
    for (Attribute raw : qecToPhysical.getEntries()) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto qec = entry ? entry.getAs<StringAttr>("qec") : StringAttr{};
      auto resources =
          entry ? entry.getAs<ArrayAttr>("resources") : ArrayAttr{};
      if (!qec || !resources)
        continue;
      std::string selected;
      for (Attribute value : resources) {
        auto name = dyn_cast<StringAttr>(value);
        if (!name || !physicalUnitCapacity.count(name.getValue().str()))
          continue;
        if (!selected.empty()) {
          qecToPhysical.emitOpError(
              "native analytical v1 requires exactly "
              "one physical resource class per QEC region");
          return signalPassFailure();
        }
        selected = name.getValue().str();
      }
      if (!selected.empty())
        physicalHomeByRegion[qec.getValue().str()] = selected;
      auto bindingName = entry.getAs<StringAttr>("binding");
      auto binding =
          bindingName
              ? dyn_cast_or_null<qlx::phys::QECBindingOp>(
                    SymbolTable(physical).lookup(bindingName.getValue()))
              : qlx::phys::QECBindingOp{};
      if (selected.empty() || !binding)
        continue;
      if (physicalUnitsPerMember.at(selected) > 1) {
        patchCapacityByRegion[qec.getValue().str()] =
            resourceMemberCapacity.at(selected);
        regionsWithoutPatchTopology.insert(qec.getValue().str());
        continue;
      }
      if (!binding.getPatchTopologyAttr()) {
        regionsWithoutPatchTopology.insert(qec.getValue().str());
        continue;
      }
      auto patch = dyn_cast_or_null<qlx::phys::PatchTopologyOp>(
          SymbolTable(physical).lookup(
              binding.getPatchTopologyAttr().getValue()));
      if (!patch) {
        binding.emitOpError("patch_topology is unresolved");
        return signalPassFailure();
      }
      patchCapacityByRegion[qec.getValue().str()] = patch.getCapacity();
      auto &carriers = boundCarriersByRegion[qec.getValue().str()];
      auto &groupSizes = carrierGroupSizesByRegion[qec.getValue().str()];
      for (Attribute rawGroup : patch.getCarrierGroups()) {
        auto group = dyn_cast<DenseI64ArrayAttr>(rawGroup);
        if (!group) {
          patch.emitOpError("carrier group is malformed");
          return signalPassFailure();
        }
        groupSizes.push_back(group.size());
        carriers.insert(group.asArrayRef().begin(), group.asArrayRef().end());
      }
    }

    llvm::SmallSetVector<StringRef, 4> codeNames;
    std::map<std::string, std::set<int64_t>> activeCarriersByResource;
    std::set<std::string> activeWholePoolResources;
    for (NamedAttribute region : regions) {
      auto data = dyn_cast<DictionaryAttr>(region.getValue());
      auto code = data ? data.getAs<StringAttr>("code") : StringAttr{};
      auto patches = data ? data.getAs<IntegerAttr>("patches") : IntegerAttr{};
      if (!code || !patches || patches.getInt() < 0) {
        countsResult.emitOpError(
            "static per_region entries require code and nonnegative patches");
        return signalPassFailure();
      }
      if (patches.getInt() == 0)
        continue;
      if (!physicalHomeByRegion.count(region.getName().str())) {
        countsResult.emitOpError("active QEC region '")
            << region.getName() << "' has no unique bound physical resource";
        return signalPassFailure();
      }
      if (auto capacity = patchCapacityByRegion.find(region.getName().str());
          capacity != patchCapacityByRegion.end() &&
          patches.getInt() > capacity->second) {
        countsResult.emitOpError("active QEC region '")
            << region.getName() << "' requires " << patches.getInt()
            << " simultaneous patches but its selected patch topology has "
               "capacity "
            << capacity->second;
        return signalPassFailure();
      }
      const std::string &resource =
          physicalHomeByRegion.at(region.getName().str());
      if (regionsWithoutPatchTopology.count(region.getName().str())) {
        activeWholePoolResources.insert(resource);
      } else if (auto carriers =
                     boundCarriersByRegion.find(region.getName().str());
                 carriers != boundCarriersByRegion.end()) {
        auto &active = activeCarriersByResource[resource];
        active.insert(carriers->second.begin(), carriers->second.end());
      }
      codeNames.insert(code.getValue());
    }
    if (codeNames.empty()) {
      module.emitError("fabric-estimate-analytical requires selected code "
                       "evidence in Tier-1 regions");
      return signalPassFailure();
    }

    int64_t distance = std::numeric_limits<int64_t>::max();
    llvm::StringMap<int64_t> carriersByCode;
    llvm::SmallSetVector<StringRef, 4> distanceEvidenceNames;
    std::string distanceStatus;
    bool allDistancesEstablished = true;
    for (StringRef name : codeNames) {
      auto code = symbols.lookup<CodeOp>(name);
      if (!code || code.getDistance() <= 0) {
        module.emitError("fabric-estimate-analytical missing usable code @")
            << name;
        return signalPassFailure();
      }
      distance = std::min(distance, static_cast<int64_t>(code.getDistance()));
      int64_t carriers = 0;
      for (NamedAttribute partition : code.getPartitions()) {
        auto count = dyn_cast<IntegerAttr>(partition.getValue());
        if (!count || count.getInt() < 0 ||
            count.getInt() > std::numeric_limits<int64_t>::max() - carriers) {
          code.emitOpError("partitions contain an invalid carrier count");
          return signalPassFailure();
        }
        carriers += count.getInt();
      }
      carriersByCode[name] = carriers;
      std::string status = "claimed";
      bool hasMethod = false;
      bool hasAuthenticatedProvenance = false;
      if (auto metadata = code.getMetadataAttr()) {
        if (auto value = metadata.getAs<StringAttr>("distance_method"))
          hasMethod = !value.getValue().empty();
        if (auto reference =
                metadata.getAs<FlatSymbolRefAttr>("distance_provenance")) {
          auto profile = symbols.lookup<CodeProfileOp>(reference.getValue());
          if (profile && profile.getCode() == name &&
              profile.getDistanceClaim() == code.getDistance() &&
              profile.getEvidenceAttr() && !profile.getEvidenceAttr().empty()) {
            status = profile.getDistanceStatus().value_or("claimed").str();
            hasAuthenticatedProvenance =
                status == "exact" || status == "lower_bound";
            if (hasAuthenticatedProvenance)
              distanceEvidenceNames.insert(profile.getSymName());
          }
        }
      }
      bool established = hasMethod && hasAuthenticatedProvenance;
      allDistancesEstablished &= established;
      if (distanceStatus.empty())
        distanceStatus = status;
      else if (distanceStatus != status)
        distanceStatus = "mixed";
    }
    if (requireEstablished && !allDistancesEstablished) {
      module.emitError("fabric-estimate-analytical distance status '")
          << distanceStatus << "' is not established";
      return signalPassFailure();
    }

    double siteError = 0.0;
    if (scalingPrefactor > 0.0 && physicalError > 0.0) {
      double exponent = (static_cast<double>(distance) + 1.0) / 2.0;
      double logSite =
          std::log(scalingPrefactor) +
          exponent * (std::log(physicalError) - std::log(scalingThreshold));
      siteError = logSite >= 0.0 ? 1.0 : std::exp(logSite);
    }
    if (!std::isfinite(siteError) || siteError < 0.0 || siteError > 1.0) {
      module.emitError("analytical scaling produced a non-finite probability");
      return signalPassFailure();
    }
    double expectedSites = static_cast<double>(sites->weighted) +
                           retries->expectedExtraOperationSites;
    if (!std::isfinite(expectedSites) || expectedSites < 0.0) {
      module.emitError("analytical expected operation sites overflow f64");
      return signalPassFailure();
    }
    double logicalError = independentFailure(siteError, expectedSites);
    logicalError = independentUnion(logicalError, retries->logicalError);
    logicalError = independentUnion(logicalError, factorySummary.resourceError);
    int64_t minimumPhysicalQubits = 0;
    std::map<std::string, int64_t> requiredByResource;
    for (NamedAttribute region : regions) {
      auto data = dyn_cast<DictionaryAttr>(region.getValue());
      auto code = data ? data.getAs<StringAttr>("code") : StringAttr{};
      auto patches = data ? data.getAs<IntegerAttr>("patches") : IntegerAttr{};
      if (!code || !patches || patches.getInt() < 0) {
        countsResult.emitOpError(
            "static per_region entries require code and nonnegative patches");
        return signalPassFailure();
      }
      if (patches.getInt() == 0)
        continue;
      auto carrier = carriersByCode.find(code.getValue());
      if (carrier == carriersByCode.end()) {
        countsResult.emitOpError("static region references an unresolved code");
        return signalPassFailure();
      }
      std::string resource = physicalHomeByRegion.at(region.getName().str());
      int64_t unitsPerPatch = physicalUnitsPerMember.at(resource) > 1
                                  ? physicalUnitsPerMember.at(resource)
                                  : carrier->second;
      if (patches.getInt() != 0 &&
          unitsPerPatch >
              std::numeric_limits<int64_t>::max() / patches.getInt()) {
        module.emitError(
            "analytical physical-qubit count overflows signed i64");
        return signalPassFailure();
      }
      int64_t contribution = unitsPerPatch * patches.getInt();
      if (physicalUnitsPerMember.at(resource) == 1) {
        if (auto groups =
                carrierGroupSizesByRegion.find(region.getName().str());
            groups != carrierGroupSizesByRegion.end()) {
          int64_t compatibleSlots =
              llvm::count_if(groups->second, [&](int64_t size) {
                return size >= carrier->second;
              });
          if (compatibleSlots < patches.getInt()) {
            countsResult.emitOpError("active QEC region '")
                << region.getName() << "' requires " << patches.getInt()
                << " patch slots with at least " << carrier->second
                << " carriers each, but its selected patch topology provides "
                << compatibleSlots;
            return signalPassFailure();
          }
        }
        if (auto bound = boundCarriersByRegion.find(region.getName().str());
            bound != boundCarriersByRegion.end() &&
            static_cast<uint64_t>(contribution) > bound->second.size()) {
          countsResult.emitOpError("active QEC region '")
              << region.getName() << "' requires " << contribution
              << " physical carriers but its selected patch topology binds "
              << bound->second.size();
          return signalPassFailure();
        }
      }
      if (contribution >
          std::numeric_limits<int64_t>::max() - minimumPhysicalQubits) {
        module.emitError(
            "analytical physical-qubit count overflows signed i64");
        return signalPassFailure();
      }
      minimumPhysicalQubits += contribution;
      // Several encoded regions may occupy disjoint portions of one physical
      // patch layout.  Summing every region's independent peak is a safe
      // joint-liveness upper bound; the capacity check below authenticates
      // that the shared class can hold that conservative total.
      int64_t &required = requiredByResource[resource];
      if (contribution > std::numeric_limits<int64_t>::max() - required) {
        module.emitError(
            "analytical bound-resource demand overflows signed i64");
        return signalPassFailure();
      }
      required += contribution;
    }
    int64_t configuredPhysicalQubits = 0;
    std::map<std::string, int64_t> selectedCapacity;
    for (const auto &[resource, required] : requiredByResource) {
      if (activeWholePoolResources.count(resource) &&
          activeCarriersByResource.count(resource)) {
        device.emitOpError("active bindings mix whole-pool and patch-topology "
                           "views of physical resource @")
            << resource;
        return signalPassFailure();
      }
      int64_t available = activeCarriersByResource.count(resource)
                              ? activeCarriersByResource.at(resource).size()
                              : physicalUnitCapacity.at(resource);
      if (available < required) {
        device.emitOpError("bound physical resource @")
            << resource << " provides " << available
            << " physical units but the selected QEC regions require "
            << required;
        return signalPassFailure();
      }
      if (available >
          std::numeric_limits<int64_t>::max() - configuredPhysicalQubits) {
        module.emitError("bound physical-qubit capacity overflows signed i64");
        return signalPassFailure();
      }
      selectedCapacity[resource] = available;
      configuredPhysicalQubits += available;
    }
    for (const auto &[resource, footprint] : factorySummary.physicalResources) {
      if (selectedCapacity.count(resource)) {
        device.emitOpError("factory resource @")
            << resource
            << " must be dedicated rather than shared with active compute";
        return signalPassFailure();
      }
      if (footprint >
              std::numeric_limits<int64_t>::max() - minimumPhysicalQubits ||
          footprint >
              std::numeric_limits<int64_t>::max() - configuredPhysicalQubits) {
        module.emitError("factory physical-qubit footprint overflows i64");
        return signalPassFailure();
      }
      minimumPhysicalQubits += footprint;
      configuredPhysicalQubits += footprint;
      selectedCapacity[resource] = footprint;
    }
    double wallclock = expectedSites * cycleTime;
    if (!std::isfinite(wallclock)) {
      module.emitError("analytical wallclock overflows finite f64");
      return signalPassFailure();
    }
    std::string bottleneck = "compute_limited";
    if (factorySummary.wallclock > wallclock) {
      wallclock = factorySummary.wallclock;
      bottleneck = "factory_limited";
    }

    MLIRContext *context = &getContext();
    OpBuilder builder(context);
    auto i64 = IntegerType::get(context, 64);
    SmallVector<NamedAttribute> capacityFields;
    for (const auto &[resource, capacity] : selectedCapacity) {
      capacityFields.emplace_back(builder.getStringAttr(resource),
                                  IntegerAttr::get(i64, capacity));
    }
    SmallVector<Attribute> retryFields;
    for (const RetryEstimate &retry : retries->estimates) {
      SmallVector<NamedAttribute> perAttempt;
      SmallVector<NamedAttribute> expectedRequests;
      SmallVector<NamedAttribute> maximumRequests;
      for (const auto &[kind, count] : retry.requestsPerAttempt) {
        perAttempt.emplace_back(builder.getStringAttr(kind),
                                IntegerAttr::get(i64, count));
        expectedRequests.emplace_back(
            builder.getStringAttr(kind),
            builder.getF64FloatAttr(static_cast<double>(retry.occurrences) *
                                    retry.expectedAttempts *
                                    static_cast<double>(count)));
        int64_t maximumPerOccurrence = 0;
        int64_t maximum = 0;
        if (llvm::MulOverflow(count, retry.maxAttempts, maximumPerOccurrence) ||
            llvm::MulOverflow(maximumPerOccurrence, retry.occurrences,
                              maximum)) {
          module.emitError(
              "analytical retry maximum resource demand overflows i64");
          return signalPassFailure();
        }
        maximumRequests.emplace_back(builder.getStringAttr(kind),
                                     IntegerAttr::get(i64, maximum));
      }
      retryFields.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("attempt", retry.attempt),
          builder.getNamedAttr("occurrences",
                               builder.getI64IntegerAttr(retry.occurrences)),
          builder.getNamedAttr(
              "success_probability",
              builder.getF64FloatAttr(retry.successProbability)),
          builder.getNamedAttr("max_attempts",
                               builder.getI64IntegerAttr(retry.maxAttempts)),
          builder.getNamedAttr("expected_attempts",
                               builder.getF64FloatAttr(retry.expectedAttempts)),
          builder.getNamedAttr(
              "exhaustion_probability",
              builder.getF64FloatAttr(retry.exhaustionProbability)),
          builder.getNamedAttr("exhaustion", retry.exhaustion),
          builder.getNamedAttr("resource_requests_per_attempt",
                               builder.getDictionaryAttr(perAttempt)),
          builder.getNamedAttr("expected_resource_requests",
                               builder.getDictionaryAttr(expectedRequests)),
          builder.getNamedAttr("maximum_resource_requests",
                               builder.getDictionaryAttr(maximumRequests)),
      }));
    }
    SmallVector<NamedAttribute> fields = {
        {builder.getStringAttr("counts"),
         FlatSymbolRefAttr::get(context, countsResult.getSymName())},
        {builder.getStringAttr("p_phys"),
         builder.getF64FloatAttr(physicalError)},
        {builder.getStringAttr("failure_budget"),
         builder.getF64FloatAttr(failureBudget)},
        {builder.getStringAttr("distance"), IntegerAttr::get(i64, distance)},
        {builder.getStringAttr("distance_status"),
         builder.getStringAttr(distanceStatus)},
        {builder.getStringAttr("logical_error"),
         builder.getF64FloatAttr(logicalError)},
        {builder.getStringAttr("resource_error"),
         builder.getF64FloatAttr(factorySummary.resourceError)},
        {builder.getStringAttr("physical_qubits_peak"),
         IntegerAttr::get(i64, minimumPhysicalQubits)},
        {builder.getStringAttr("physical_qubits_available"),
         IntegerAttr::get(i64, configuredPhysicalQubits)},
        {builder.getStringAttr("physical_qubit_capacity_by_resource"),
         DictionaryAttr::get(context, capacityFields)},
        {builder.getStringAttr("wallclock"),
         builder.getF64FloatAttr(wallclock)},
        {builder.getStringAttr("factory_wallclock"),
         builder.getF64FloatAttr(factorySummary.wallclock)},
        {builder.getStringAttr("cycle_time"),
         builder.getF64FloatAttr(cycleTime)},
        {builder.getStringAttr("scaling_model"),
         builder.getStringAttr("code-distance-power-law/v1")},
        {builder.getStringAttr("scaling_prefactor"),
         builder.getF64FloatAttr(scalingPrefactor)},
        {builder.getStringAttr("scaling_threshold"),
         builder.getF64FloatAttr(scalingThreshold)},
        {builder.getStringAttr("failure_aggregation_model"),
         builder.getStringAttr("independent-survival/v1")},
        {builder.getStringAttr("site_model"),
         builder.getStringAttr("weighted-fabric-operation-sites/v1")},
        {builder.getStringAttr("require_established_distance"),
         builder.getBoolAttr(requireEstablished)},
        {builder.getStringAttr("idle_round_sites"),
         IntegerAttr::get(i64, sites->idleRounds)},
        {builder.getStringAttr("acceptance"),
         builder.getF64FloatAttr(retries->acceptance)},
        {builder.getStringAttr("bottleneck"),
         builder.getStringAttr(bottleneck)},
        {builder.getStringAttr("budget_met"),
         builder.getBoolAttr(logicalError <= failureBudget)},
        {builder.getStringAttr("operation_sites"),
         IntegerAttr::get(i64, sites->weighted)},
        {builder.getStringAttr("expected_operation_sites"),
         builder.getF64FloatAttr(expectedSites)},
        {builder.getStringAttr("retry_demands"),
         builder.getArrayAttr(retryFields)},
        {builder.getStringAttr("resource_requests"),
         counts.getAs<ArrayAttr>("resource_requests")
             ? counts.getAs<ArrayAttr>("resource_requests")
             : builder.getArrayAttr({})},
    };
    builder.setInsertionPointToEnd(module.getBody());
    SmallVector<Attribute> evidence;
    for (StringRef name : codeNames)
      evidence.push_back(FlatSymbolRefAttr::get(context, name));
    for (StringRef name : distanceEvidenceNames)
      evidence.push_back(FlatSymbolRefAttr::get(context, name));
    for (StringRef name : factorySummary.producers)
      evidence.push_back(FlatSymbolRefAttr::get(context, name));
    evidence.push_back(FlatSymbolRefAttr::get(context, device.getSymName()));
    evidence.push_back(
        FlatSymbolRefAttr::get(context, qecToPhysical.getSymName()));
    evidence.push_back(FlatSymbolRefAttr::get(context, physical.getSymName()));
    evidence.push_back(
        FlatSymbolRefAttr::get(context, operatingPoint.getSymName()));
    SmallVector<Attribute> assumptions = {
        builder.getStringAttr("independent logical fault sites"),
        builder.getStringAttr("serial critical-path upper bound"),
        builder.getStringAttr("distance scaling model"),
    };
    if (!retries->estimates.empty()) {
      assumptions.push_back(builder.getStringAttr(
          "certified independent retry-attempt probabilities"));
      assumptions.push_back(builder.getStringAttr(
          "bounded retry resource demand includes failed attempts"));
    }
    if (!factorySummary.producers.empty()) {
      if (factorySummary.conservativeOutputError) {
        assumptions.push_back(builder.getStringAttr(
            "factory retry and output-error models are absent; analytical "
            "resource error is conservatively one"));
        if (factorySummary.singleShot)
          assumptions.push_back(builder.getStringAttr(
              "compact factory timing is conditional on retained selections"));
      } else {
        assumptions.push_back(
            builder.getStringAttr("guaranteed steady-state factory model"));
        assumptions.push_back(
            builder.getStringAttr("independent resource output infidelity"));
      }
      assumptions.push_back(builder.getStringAttr(
          "analytical factory wallclock is a throughput proxy; the P3 "
          "schedule is runtime authority"));
    }
    if (auto operations = counts.getAs<DictionaryAttr>("operation_counts"))
      if (auto selections = operations.getAs<IntegerAttr>("selection");
          selections && selections.getInt() > 0)
        assumptions.push_back(builder.getStringAttr(
            "analytical acceptance is conditional on retained P2 selections; "
            "no retry probability is inferred"));
    if (!requireEstablished)
      assumptions.push_back(builder.getStringAttr(
          "unestablished code distance explicitly permitted"));
    auto metadata = builder.getDictionaryAttr({
        builder.getNamedAttr(
            "producer", builder.getStringAttr("fabric-estimate-analytical")),
        builder.getNamedAttr("producer_version", builder.getStringAttr("1")),
        builder.getNamedAttr("physical_closure",
                             builder.getStringAttr("configured-device")),
    });
    EstimateResultOp::create(
        builder, root->getLoc(), builder.getStringAttr(resultSymbol),
        builder.getStringAttr("analytical"),
        FlatSymbolRefAttr::get(
            context,
            cast<StringAttr>(SymbolTable::getSymbolName(root)).getValue()),
        builder.getStringAttr("qlx.fabric-estimate/v1"),
        DictionaryAttr::get(context, fields), builder.getArrayAttr(assumptions),
        builder.getArrayAttr(evidence),
        FlatSymbolRefAttr::get(context, countsResult.getSymName()),
        deviceReference, metadata);
  }

private:
  bool countsAreFresh = false;
};

} // namespace

std::unique_ptr<mlir::Pass>
qlx::fabric::createFabricEstimateAnalyticalForFreshCounts(
    const FabricEstimateAnalyticalOptions &options) {
  return std::make_unique<FabricEstimateAnalyticalPass>(
      options,
      /*countsAreFresh=*/true);
}
