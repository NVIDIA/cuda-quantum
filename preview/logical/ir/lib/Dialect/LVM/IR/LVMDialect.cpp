/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/Event/IR/EventTypes.h"
#include "qlx/Dialect/LVM/IR/LVMAttrs.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/LVM/IR/LVMTypes.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace qlx::lvm;

#include "qlx/Dialect/LVM/IR/LVMDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/LVM/IR/LVMTypes.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "qlx/Dialect/LVM/IR/LVMAttrs.cpp.inc"

namespace {
class LVMDeviceBindingDialectInterface final
    : public qlx::DeviceBindingDialectInterface {
public:
  explicit LVMDeviceBindingDialectInterface(Dialect *dialect)
      : DeviceBindingDialectInterface(dialect) {}

  std::optional<StringRef>
  getLogicalCapabilityKey(Attribute attribute) const override {
    if (auto capability = dyn_cast<CapabilityAttr>(attribute))
      return capability.getKey();
    return std::nullopt;
  }
};
} // namespace

static DomainOp lookupDomain(Operation *from, FlatSymbolRefAttr reference) {
  for (Operation *scope = from; scope; scope = scope->getParentOp()) {
    if (!scope->hasTrait<OpTrait::SymbolTable>())
      continue;
    if (auto domain = dyn_cast_or_null<DomainOp>(
            SymbolTable::lookupSymbolIn(scope, reference.getValue())))
      return domain;
  }
  return {};
}

static Operation *lookupObjective(Operation *from,
                                  FlatSymbolRefAttr reference) {
  for (Operation *scope = from; scope; scope = scope->getParentOp()) {
    if (!scope->hasTrait<OpTrait::SymbolTable>())
      continue;
    if (Operation *symbol =
            SymbolTable::lookupSymbolIn(scope, reference.getValue()))
      return symbol;
  }
  return nullptr;
}

static LogicalResult rejectRetiredMachineCapabilityAttrs(Operation *owner) {
  if (owner->getAttr("serv"
                     "ice") ||
      owner->getAttr("serv"
                     "ices"))
    return owner->emitOpError(
        "contains a retired machine-capability attribute");
  return success();
}

static LogicalResult verifyMachineCapabilities(Operation *owner,
                                               ArrayAttr capabilities) {
  if (failed(rejectRetiredMachineCapabilityAttrs(owner)))
    return failure();
  llvm::StringSet<> seen;
  for (Attribute value : capabilities) {
    auto capability = dyn_cast<CapabilityAttr>(value);
    if (!capability)
      return owner->emitOpError(
          "capabilities entries must be #lvm.capability attributes");
    StringRef key = capability.getKey();
    size_t separator = key.find('/');
    if (separator == StringRef::npos || separator == 0 ||
        separator + 1 == key.size())
      return owner->emitOpError(
          "capability keys must be nonempty qualified names");
    StringRef nameSpace = key.take_front(separator);
    if (nameSpace.starts_with("qlx.") && nameSpace != "qlx.machine")
      return owner->emitOpError("reserved QLX machine capabilities must use "
                                "the qlx.machine namespace");
    if (!seen.insert(key).second)
      return owner->emitOpError("contains duplicate capability '")
             << key << "'";
  }
  return success();
}

LogicalResult DomainOp::verify() {
  SmallVector<PlacementOp> placements;
  getBody().walk(
      [&](PlacementOp placement) { placements.push_back(placement); });

  auto sameSlot = [](auto left, auto right) {
    return left.getSpaceAttr() == right.getSpaceAttr() &&
           left.getSlotAttr() == right.getSlotAttr();
  };
  for (auto [index, left] : llvm::enumerate(placements)) {
    for (PlacementOp right : llvm::drop_begin(placements, index + 1)) {
      if (!sameSlot(left, right))
        continue;
      return right.emitOpError(
          "slot is already occupied by another logical placement");
    }
  }

  // Authenticate generated P2 action-site provenance in one indexed pass.
  // Verifying each ActionSiteOp by walking every retained kernel/apply made
  // this proof quadratic in the number of sites (the Pinnacle L=16 body has
  // more than 200k).  The domain owns both the site declarations and their
  // placement namespace, so it can build the exact site-occurrence index once
  // without weakening any per-site check or diagnostic location.
  SmallVector<ActionSiteOp> sourcedActionSites;
  getBody().walk([&](ActionSiteOp site) {
    if (site.getSourceSiteAttr())
      sourcedActionSites.push_back(site);
  });
  if (sourcedActionSites.empty())
    return success();

  Operation *scope = getOperation()->getParentOp();
  if (!scope)
    return sourcedActionSites.front().emitOpError(
        "source_site requires a containing symbol-table scope");

  DenseMap<int64_t, SmallVector<ApplyOp, 1>> appliesBySite;
  scope->walk([&](KernelOp kernel) {
    if (lookupDomain(kernel.getOperation(), kernel.getDomainAttr()) != *this)
      return;
    kernel.getBody().walk([&](ApplyOp apply) {
      if (auto candidate = apply.getSiteAttr())
        appliesBySite[candidate.getInt()].push_back(apply);
    });
  });

  for (ActionSiteOp site : sourcedActionSites) {
    int64_t sourceSite = site.getSourceSiteAttr().getInt();
    if (sourceSite < 0)
      return site.emitOpError("source_site must be nonnegative");
    auto found = appliesBySite.find(sourceSite);
    if (found == appliesBySite.end())
      return site.emitOpError("source_site does not resolve to a retained "
                              "lvm.apply in this domain");
    ArrayRef<ApplyOp> matchingApplies = found->second;
    if (matchingApplies.size() != 1)
      return site.emitOpError(
          "source_site must resolve uniquely to one retained lvm.apply in "
          "this domain");

    ApplyOp source = matchingApplies.front();
    if (site.getKind() != "action")
      return site.emitOpError(
          "an lvm.apply source_site requires kind = action");
    if (site.getObjectiveAttr() != source.getActionAttr())
      return site.emitOpError(
          "objective does not match the source_site lvm.apply action");
    if (site.getPlacements() != source.getPlacements())
      return site.emitOpError(
          "placements do not match the source_site lvm.apply placements");
    auto sourceParameters = source.getParametersAttr();
    auto siteParameters = site.getParametersAttr();
    if (sourceParameters) {
      for (NamedAttribute parameter : sourceParameters) {
        if (!siteParameters ||
            siteParameters.get(parameter.getName()) != parameter.getValue())
          return site.emitOpError("parameter '")
                 << parameter.getName()
                 << "' does not match the source_site lvm.apply";
      }
    }

    Attribute derivedParameter;
    StringRef derivedParameterName;
    auto objective = dyn_cast<qlx::BuiltinActionAttr>(source.getActionAttr());
    if (objective &&
        objective.getValue() == qlx::BuiltinAction::pauli_rotation) {
      if (auto constant =
              source.getInputs().back().getDefiningOp<arith::ConstantOp>())
        derivedParameter = dyn_cast<FloatAttr>(constant.getValue());
      if (derivedParameter) {
        derivedParameterName = "angle";
      } else {
        derivedParameterName = "dynamic_angle";
        derivedParameter = BoolAttr::get(getContext(), true);
      }
      if (!siteParameters ||
          siteParameters.get(derivedParameterName) != derivedParameter)
        return site.emitOpError("derived parameter '")
               << derivedParameterName
               << "' does not match the source_site lvm.apply SSA angle";
    }

    size_t expectedParameterCount =
        sourceParameters ? sourceParameters.size() : 0;
    if (derivedParameter)
      ++expectedParameterCount;
    size_t actualParameterCount = siteParameters ? siteParameters.size() : 0;
    if (actualParameterCount != expectedParameterCount)
      return site.emitOpError(
          "parameters must exactly equal the source_site lvm.apply parameters "
          "plus canonical SSA-derived facts");
  }
  return success();
}

LogicalResult SpaceOp::verify() {
  return verifyMachineCapabilities(getOperation(), getCapabilities());
}

LogicalResult StreamOp::verify() {
  if (getProduces().empty())
    return emitOpError("produces must name a resource kind");
  if (auto capacity = getCapacityAttr())
    if (capacity.getInt() < 0)
      return emitOpError("capacity must be nonnegative");
  if (getExternal() && getBackingRegionAttr())
    return emitOpError(
        "a stream cannot be both external and backed by a region");
  auto domain = (*this)->getParentOfType<DomainOp>();
  if (auto backing = getBackingRegionAttr()) {
    auto space = dyn_cast_or_null<SpaceOp>(
        SymbolTable::lookupSymbolIn(domain, backing.getValue()));
    if (!space)
      return emitOpError("backing_region must resolve to an lvm.space");
    SmallVector<ChannelOp, 2> supplies;
    for (Operation &candidate : domain.getBody().front()) {
      auto channel = dyn_cast<ChannelOp>(candidate);
      if (!channel)
        continue;
      Operation *destination =
          SymbolTable::lookupSymbolIn(domain, channel.getToAttr().getValue());
      if (destination != getOperation())
        continue;
      bool transfersResource =
          llvm::any_of(channel.getCapabilities(), [](Attribute value) {
            auto capability = dyn_cast<CapabilityAttr>(value);
            return capability &&
                   capability.getKey() == "qlx.machine/resource_transfer";
          });
      if (transfersResource)
        supplies.push_back(channel);
    }
    if (supplies.size() != 1)
      return emitOpError(
          "a backed stream requires exactly one retained resource-transfer "
          "supply channel");
    Operation *source = SymbolTable::lookupSymbolIn(
        domain, supplies.front().getFromAttr().getValue());
    if (source != space.getOperation())
      return emitOpError(
          "backing_region must equal the retained supply channel source");
  }
  auto verifyProtocol = [&](FlatSymbolRefAttr reference,
                            StringRef label) -> LogicalResult {
    auto digest = (*this)->getAttrOfType<StringAttr>((label + "_sha256").str());
    if (!reference) {
      if (digest)
        return emitOpError()
               << label << "_sha256 requires the matching protocol identity";
      return success();
    }
    if (!digest || !digest.getValue().starts_with("sha256:") ||
        digest.getValue().size() != 71 ||
        !llvm::all_of(digest.getValue().drop_front(7), [](char value) {
          return llvm::isHexDigit(value) && !(value >= 'A' && value <= 'F');
        }))
      return emitOpError() << label
                           << " requires a canonical sha256 payload commitment";
    Operation *target = lookupObjective(*this, reference);
    // P1 commits the exact selected protocol identity and closure digest, but
    // the typed Fabric body is a P2 realization fact. P1-to-P2 linking
    // materializes the definition and authenticates it against this digest.
    if (!target)
      return success();
    if (target->getName().getStringRef() != "fabric.protocol")
      return emitOpError() << label
                           << " must resolve to a typed fabric.protocol";
    return success();
  };
  if (failed(verifyProtocol(getProducedByAttr(), "produced_by")) ||
      failed(verifyProtocol(getTransferAttr(), "transfer")))
    return failure();
  return success();
}

LogicalResult ChannelOp::verify() {
  return verifyMachineCapabilities(getOperation(), getCapabilities());
}

LogicalResult PlacementOp::verify() {
  if (getSlotAttr().getInt() < 0)
    return emitOpError("slot must be nonnegative");
  auto domain = (*this)->getParentOfType<DomainOp>();
  auto space = dyn_cast_or_null<SpaceOp>(
      SymbolTable::lookupSymbolIn(domain, getSpaceAttr().getValue()));
  if (!space)
    return emitOpError("references unknown space ") << getSpaceAttr();
  if (auto capacity = space.getCapacityAttr())
    if (getSlotAttr().getInt() >= capacity.getInt())
      return emitOpError("slot exceeds space capacity");

  if (auto binding = getBindingAttr()) {
    auto kind = dyn_cast_or_null<StringAttr>(binding.get("kind"));
    if (!kind)
      return emitOpError("binding requires a string kind");
    auto requireString = [&](StringRef key) -> LogicalResult {
      auto value = dyn_cast_or_null<StringAttr>(binding.get(key));
      if (!value || value.getValue().empty())
        return emitOpError()
               << "binding requires nonempty string '" << key << "'";
      return success();
    };
    auto requireArray = [&](StringRef key,
                            bool nonempty = true) -> FailureOr<ArrayAttr> {
      auto value = dyn_cast_or_null<ArrayAttr>(binding.get(key));
      if (!value || (nonempty && value.empty())) {
        emitOpError() << "binding requires " << (nonempty ? "nonempty " : "")
                      << "array '" << key << "'";
        return failure();
      }
      return value;
    };
    auto verifySpaces = [&](ArrayAttr spaces) -> LogicalResult {
      for (Attribute item : spaces) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(item);
        if (!reference || !isa_and_nonnull<SpaceOp>(SymbolTable::lookupSymbolIn(
                              domain, reference.getValue())))
          return emitOpError("binding references an unknown space");
      }
      return success();
    };
    StringRef name = kind.getValue();
    if (name == "local") {
      return emitOpError(
          "ordinary local residency must reference lvm.space directly");
    } else if (name == "distributed") {
      auto spaces = requireArray("spaces");
      auto views = requireArray("support_views");
      if (failed(spaces) || failed(views))
        return failure();
      if (spaces->size() != views->size())
        return emitOpError("distributed spaces and support_views must align");
      if (failed(verifySpaces(*spaces)) ||
          failed(requireString("ownership_witness")))
        return failure();
      if (!llvm::is_contained(*spaces, getSpaceAttr()))
        return emitOpError("distributed spaces must contain the primary space");
    } else if (name == "trajectory") {
      auto segments = requireArray("segments");
      auto transitions = requireArray("transition_events", false);
      if (failed(segments) || failed(transitions))
        return failure();
      if (failed(verifySpaces(*segments)) ||
          failed(requireString("continuity_witness")))
        return failure();
      if ((*segments)[0] != getSpaceAttr())
        return emitOpError("trajectory must begin in the primary space");
      if (transitions->size() + 1 != segments->size())
        return emitOpError(
            "trajectory requires one transition event between segments");
    } else if (name == "topological_record") {
      if (failed(requireString("record")) || failed(requireArray("frontier")) ||
          failed(requireString("support_witness")) ||
          failed(requireString("observable_witness")))
        return failure();
    } else if (name == "unresolved") {
      if (failed(requireArray("constraints")) ||
          failed(requireArray("dimensions")) || failed(requireString("reason")))
        return failure();
    } else {
      return emitOpError("binding kind must be local, distributed, trajectory, "
                         "topological_record, or unresolved");
    }
  }

  if (!getBindingAttr())
    return emitOpError(
        "ordinary local residency must reference lvm.space directly");
  return success();
}

static LogicalResult
verifyPlacementReference(Operation *op, SymbolRefAttr ref,
                         DomainOp expectedDomain,
                         SymbolTable *domainSymbols = nullptr) {
  if (!ref || ref.getNestedReferences().size() != 1)
    return op->emitOpError("placement references must be @domain::@placement");
  if (ref.getRootReference() != expectedDomain.getSymNameAttr())
    return op->emitOpError("placement reference ")
           << ref << " is not rooted in kernel domain @"
           << expectedDomain.getSymName();
  Operation *target =
      domainSymbols
          ? domainSymbols->lookup(ref.getLeafReference().getValue())
          : SymbolTable::lookupSymbolIn(expectedDomain, ref.getLeafReference());
  if (!isa_and_nonnull<SpaceOp, PlacementOp>(target))
    return op->emitOpError("references unknown logical space or placement ")
           << ref;
  return success();
}

static LogicalResult verifyPlacedType(Operation *op, Type type,
                                      SymbolRefAttr expected) {
  auto logical = dyn_cast<LogicalQubitType>(type);
  if (!logical)
    return success();
  if (logical.getPlacement() != expected)
    return op->emitOpError("logical-qubit placement type ")
           << logical.getPlacement() << " does not match operation placement "
           << expected;
  return success();
}

static LogicalResult verifyP0KernelBodyRefinement(KernelOp kernel,
                                                  qlx::ProgramOp portable);

LogicalResult KernelOp::verify() {
  DomainOp domain = lookupDomain(getOperation(), getDomainAttr());
  if (!domain)
    return emitOpError("references unknown lvm.domain ") << getDomainAttr();
  auto inputP0 = getInputP0Attr();
  if (getEstimateOnly() && !inputP0)
    return emitOpError("estimate_only requires an input_p0 refinement");
  qlx::ProgramOp inputProgram;
  if (inputP0) {
    inputProgram = dyn_cast_or_null<qlx::ProgramOp>(
        lookupObjective(getOperation(), inputP0));
    if (!inputProgram)
      return emitOpError("input_p0 must resolve to a qlx.program ") << inputP0;
    bool sourceEstimateOnly = inputProgram.getEstimateOnly().value_or(false);
    if (sourceEstimateOnly != getEstimateOnly().value_or(false))
      return emitOpError(
          "estimate_only must exactly match the input_p0 program");
    if (inputProgram->getAttr("specialization") !=
        (*this)->getAttr("specialization"))
      return emitOpError(
          "specialization must exactly match the input_p0 program");
  }
  auto placementCommitment =
      (*this)->getAttrOfType<StringAttr>("placement_witness_sha256");
  if (inputP0 && !placementCommitment)
    return emitOpError("with input_p0 requires placement_witness_sha256");
  if (!inputP0 && placementCommitment)
    return emitOpError("placement_witness_sha256 requires input_p0");
  if (placementCommitment) {
    StringRef value = placementCommitment.getValue();
    StringRef digest = value.consume_front("sha256:") ? value : StringRef();
    if (digest.size() != 64 || !llvm::all_of(digest, [](char character) {
          return (character >= '0' && character <= '9') ||
                 (character >= 'a' && character <= 'f');
        }))
      return emitOpError(
          "placement_witness_sha256 must be 'sha256:' followed by 64 "
          "lowercase hexadecimal digits");
  }
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one entry block");
  Block &entry = getBody().front();
  FunctionType type = getFunctionType();
  if (entry.getNumArguments() != type.getNumInputs() ||
      !llvm::equal(entry.getArgumentTypes(), type.getInputs()))
    return emitOpError("entry arguments do not match function_type inputs");
  auto ret = dyn_cast<ReturnOp>(entry.getTerminator());
  if (!ret || ret.getNumOperands() != type.getNumResults() ||
      !llvm::equal(ret.getOperandTypes(), type.getResults()))
    return emitOpError(
        "lvm.return operands do not match function_type results");

  LogicalResult result = success();
  llvm::DenseSet<int64_t> callScopes;
  // The placed body may contain hundreds of thousands of logical operands.
  // A standalone SymbolTable::lookupSymbolIn performs a direct symbol-table
  // lookup without retaining an index between calls.  Build one domain index
  // for the whole kernel proof so placement authentication remains exact and
  // linear in the body size.
  SymbolTable domainSymbols(domain);
  getBody().walk([&](Operation *nested) {
    if (failed(result))
      return;
    if (auto call = dyn_cast<CallOp>(nested)) {
      int64_t scope = call.getScopeAttr().getInt();
      if (!callScopes.insert(scope).second) {
        call.emitOpError("scope must be unique within its lvm.kernel");
        result = failure();
        return;
      }
    }
    for (Type valueType : nested->getOperandTypes()) {
      if (auto logical = dyn_cast<LogicalQubitType>(valueType))
        if (failed(verifyPlacementReference(nested, logical.getPlacement(),
                                            domain, &domainSymbols)))
          result = failure();
    }
    for (Type valueType : nested->getResultTypes()) {
      if (auto logical = dyn_cast<LogicalQubitType>(valueType))
        if (failed(verifyPlacementReference(nested, logical.getPlacement(),
                                            domain, &domainSymbols)))
          result = failure();
    }
  });
  if (succeeded(result) && inputProgram &&
      failed(verifyP0KernelBodyRefinement(*this, inputProgram)))
    return failure();
  return result;
}

static LogicalResult verifyP0Refinement(Operation *owner, TypeRange placed,
                                        TypeRange portable,
                                        StringRef description) {
  if (placed.size() != portable.size())
    return owner->emitOpError(description)
           << " arity does not match retained P0 signature";
  for (auto [actual, ideal] : llvm::zip(placed, portable)) {
    if (isa<qlx::LogicalQubitType>(ideal)) {
      if (!isa<LogicalQubitType>(actual))
        return owner->emitOpError(description)
               << " logical-qubit type does not refine P0";
      continue;
    }
    if (actual != ideal)
      return owner->emitOpError(description)
             << " non-quantum type does not match retained P0 signature";
  }
  return success();
}

namespace {

enum class PlacedBodyKind { Kernel, Call };

static bool isP0TypeRefinement(Type placed, Type portable) {
  if (isa<qlx::LogicalQubitType>(portable))
    return isa<LogicalQubitType>(placed);
  if (auto resource = dyn_cast<qlx::LogicalResourceType>(portable)) {
    auto bound = dyn_cast<LogicalResourceType>(placed);
    return bound && bound.getKind() == resource.getKind();
  }
  if (auto event = dyn_cast<qlx::event::HandleType>(portable)) {
    auto bound = dyn_cast<qlx::event::HandleType>(placed);
    return bound && bound.getOwnership() == event.getOwnership() &&
           isP0TypeRefinement(bound.getPayload(), event.getPayload());
  }
  if (auto frame = dyn_cast<qlx::LogicalFrameType>(portable)) {
    auto bound = dyn_cast<LogicalFrameType>(placed);
    return bound && bound.getDomain() == frame.getDomain();
  }
  return placed == portable;
}

static bool operationNamesRefine(StringRef placed, StringRef portable,
                                 PlacedBodyKind bodyKind) {
  if (portable == "qlx.return")
    return placed ==
           (bodyKind == PlacedBodyKind::Call ? "lvm.yield" : "lvm.return");
  if (portable.starts_with("qlx."))
    return placed.starts_with("lvm.") &&
           placed.drop_front(4) == portable.drop_front(4);
  return placed == portable;
}

static bool isDroppedP0BookkeepingAttribute(Operation *portable,
                                            StringRef name) {
  return isa<qlx::PrepareOp>(portable) &&
         (name == "allocation" || name == "value_index");
}

static bool isDerivedP1Attribute(StringRef operation, StringRef name) {
  if (operation == "lvm.call")
    return name == "scope";
  if (operation == "lvm.prepare")
    return name == "at" || name == "site" || name == "placement_owner" ||
           name == "placement_slot" || name == "source_allocation" ||
           name == "source_group" || name == "source_path";
  if (operation == "lvm.apply")
    return name == "placements" || name == "site";
  if (operation == "lvm.instrument")
    return name == "placements" || name == "site" || name == "channel" ||
           name == "channel_capability" || name == "endpoints";
  if (operation == "lvm.measure")
    return name == "at" || name == "site";
  if (operation == "lvm.idle" || operation == "lvm.discard")
    return name == "placements";
  if (operation == "lvm.resource_request")
    return name == "stream";
  if (operation == "lvm.consume_resource")
    return name == "resource_kind" || name == "resource_stream" ||
           name == "placements" || name == "site";
  return false;
}

static LogicalResult verifyP0BlockRefinement(Operation *owner, Block &portable,
                                             Block &placed,
                                             DenseMap<Value, Value> values,
                                             PlacedBodyKind bodyKind);

static LogicalResult verifyP0OperationRefinement(Operation *owner,
                                                 Operation *portable,
                                                 Operation *placed,
                                                 DenseMap<Value, Value> &values,
                                                 PlacedBodyKind bodyKind) {
  StringRef portableName = portable->getName().getStringRef();
  StringRef placedName = placed->getName().getStringRef();
  if (!operationNamesRefine(placedName, portableName, bodyKind))
    return owner->emitOpError("placed body does not refine retained P0: ")
           << "expected " << portableName << " but found " << placedName;

  if (portable->getNumOperands() != placed->getNumOperands())
    return owner->emitOpError(
        "placed body operation operand arity differs from retained P0");
  for (auto [portableOperand, placedOperand] :
       llvm::zip(portable->getOperands(), placed->getOperands())) {
    auto mapped = values.find(portableOperand);
    if (mapped == values.end() || mapped->second != placedOperand)
      return owner->emitOpError(
          "placed body SSA operand wiring differs from retained P0");
  }

  if (portable->getNumResults() != placed->getNumResults())
    return owner->emitOpError(
        "placed body operation result arity differs from retained P0");
  for (auto [portableResult, placedResult] :
       llvm::zip(portable->getResults(), placed->getResults())) {
    if (!isP0TypeRefinement(placedResult.getType(), portableResult.getType()))
      return owner->emitOpError(
          "placed body result type does not refine retained P0");
    values[portableResult] = placedResult;
  }

  for (NamedAttribute attribute : portable->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (isDroppedP0BookkeepingAttribute(portable, name))
      continue;
    if (placed->getAttr(attribute.getName()) != attribute.getValue())
      return owner->emitOpError("placed body attribute '")
             << name << "' differs from retained P0";
  }
  for (NamedAttribute attribute : placed->getAttrs()) {
    StringRef name = attribute.getName().strref();
    if (portable->getAttr(attribute.getName()) ||
        isDerivedP1Attribute(placedName, name))
      continue;
    return owner->emitOpError("placed body contains non-derived attribute '")
           << name << "' absent from retained P0";
  }

  if (portableName == "qlx.call") {
    auto call = dyn_cast<CallOp>(placed);
    if (!call)
      return owner->emitOpError(
          "placed nested call is not represented by lvm.call");
    return verifyP0CallBodyRefinement(call);
  }

  if (portable->getNumRegions() != placed->getNumRegions())
    return owner->emitOpError(
        "placed body region arity differs from retained P0");
  for (auto [portableRegion, placedRegion] :
       llvm::zip(portable->getRegions(), placed->getRegions())) {
    if (portableRegion.getBlocks().size() != placedRegion.getBlocks().size())
      return owner->emitOpError(
          "placed body block arity differs from retained P0");
    for (auto [portableBlock, placedBlock] :
         llvm::zip(portableRegion.getBlocks(), placedRegion.getBlocks()))
      if (failed(verifyP0BlockRefinement(owner, portableBlock, placedBlock,
                                         values, bodyKind)))
        return failure();
  }
  return success();
}

static LogicalResult verifyP0BlockRefinement(Operation *owner, Block &portable,
                                             Block &placed,
                                             DenseMap<Value, Value> values,
                                             PlacedBodyKind bodyKind) {
  if (portable.getNumArguments() != placed.getNumArguments())
    return owner->emitOpError(
        "placed body block-argument arity differs from retained P0");
  for (auto [portableArgument, placedArgument] :
       llvm::zip(portable.getArguments(), placed.getArguments())) {
    if (!isP0TypeRefinement(placedArgument.getType(),
                            portableArgument.getType()))
      return owner->emitOpError(
          "placed body block-argument type does not refine retained P0");
    values[portableArgument] = placedArgument;
  }
  if (portable.getOperations().size() != placed.getOperations().size())
    return owner->emitOpError(
        "placed body operation count differs from retained P0");
  for (auto [portableOperation, placedOperation] :
       llvm::zip(portable.getOperations(), placed.getOperations()))
    if (failed(verifyP0OperationRefinement(owner, &portableOperation,
                                           &placedOperation, values, bodyKind)))
      return failure();
  return success();
}

static LogicalResult verifyP0BodyRefinement(Operation *owner,
                                            qlx::ProgramOp portable,
                                            Region &placed,
                                            PlacedBodyKind bodyKind) {
  if (!llvm::hasSingleElement(portable.getBody()) ||
      !llvm::hasSingleElement(placed))
    return owner->emitOpError(
        "P0 and placed refinement bodies must each contain one block");
  DenseMap<Value, Value> values;
  return verifyP0BlockRefinement(owner, portable.getBody().front(),
                                 placed.front(), std::move(values), bodyKind);
}

} // namespace

LogicalResult qlx::lvm::verifyP0CallBodyRefinement(CallOp call) {
  auto callee = dyn_cast_or_null<qlx::ProgramOp>(
      lookupObjective(call.getOperation(), call.getCalleeAttr()));
  if (!callee)
    return call.emitOpError("references unknown qlx.program ")
           << call.getCalleeAttr();
  return verifyP0BodyRefinement(call.getOperation(), callee, call.getBody(),
                                PlacedBodyKind::Call);
}

static LogicalResult verifyP0KernelBodyRefinement(KernelOp kernel,
                                                  qlx::ProgramOp portable) {
  return verifyP0BodyRefinement(kernel.getOperation(), portable,
                                kernel.getBody(), PlacedBodyKind::Kernel);
}

LogicalResult CallOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("must appear inside lvm.kernel");
  if (getScopeAttr().getInt() < 0)
    return emitOpError("scope must be nonnegative");
  auto callee = dyn_cast_or_null<qlx::ProgramOp>(
      lookupObjective(getOperation(), getCalleeAttr()));
  if (!callee)
    return emitOpError("references unknown qlx.program ") << getCalleeAttr();
  if (callee.getEstimateOnly() && !kernel.getEstimateOnly())
    return emitOpError(
               "executable placed kernel cannot call estimate-only P0 helper ")
           << getCalleeAttr();
  FunctionType portable = callee.getFunctionType();
  if (failed(verifyP0Refinement(getOperation(), getInputs().getTypes(),
                                portable.getInputs(), "input")) ||
      failed(verifyP0Refinement(getOperation(), getResultTypes(),
                                portable.getResults(), "result")))
    return failure();
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("body must contain one block");
  Block &body = getBody().front();
  if (body.getArgumentTypes() != getInputs().getTypes())
    return emitOpError("body arguments must match placed call inputs");
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getOperandTypes() != getResultTypes())
    return emitOpError("yield types must match placed call results");
  return verifyP0CallBodyRefinement(*this);
}

LogicalResult PrepareOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("must appear inside lvm.kernel");
  DomainOp domain = lookupDomain(getOperation(), kernel.getDomainAttr());
  if (failed(verifyPlacementReference(getOperation(), getAtAttr(), domain)))
    return failure();
  if (failed(
          verifyPlacedType(getOperation(), getResult().getType(), getAtAttr())))
    return failure();
  if (getState().empty())
    return emitOpError("state must not be empty");
  return success();
}

static LogicalResult verifyPlacedCall(Operation *op, ArrayAttr placements,
                                      ValueRange inputs, TypeRange results,
                                      Attribute objective,
                                      bool expectInstrument) {
  SmallVector<Type> quantumInputs;
  for (Type type : inputs.getTypes())
    if (isa<LogicalQubitType>(type))
      quantumInputs.push_back(type);
  if (placements.size() != quantumInputs.size())
    return op->emitOpError("requires one placement per logical-qubit input");
  unsigned index = 0;
  for (Type type : inputs.getTypes()) {
    if (!isa<LogicalQubitType>(type))
      continue;
    auto placement = dyn_cast<SymbolRefAttr>(placements[index++]);
    if (!placement || failed(verifyPlacedType(op, type, placement)))
      return failure();
  }
  auto reference = dyn_cast<FlatSymbolRefAttr>(objective);
  if (!reference) {
    if (expectInstrument && !isa<qlx::BuiltinInstrumentAttr>(objective))
      return op->emitOpError(
          "instrument must be a built-in attribute or declaration reference");
    if (!expectInstrument && !isa<qlx::BuiltinActionAttr>(objective))
      return op->emitOpError(
          "action must be a built-in attribute or declaration reference");
    unsigned quantumResults = llvm::count_if(
        results, [](Type type) { return isa<LogicalQubitType>(type); });
    if (!expectInstrument && quantumInputs.size() != quantumResults)
      return op->emitOpError(
          "built-in logical actions must preserve logical owner arity");
    return success();
  }
  Operation *declaration = lookupObjective(op, reference);
  if (!declaration ||
      (expectInstrument && !isa<qlx::InstrumentDeclOp>(declaration)) ||
      (!expectInstrument && !isa<qlx::ActionOp>(declaration)))
    return op->emitOpError("references missing or wrong-kind objective ")
           << reference;
  auto functionType =
      expectInstrument
          ? cast<qlx::InstrumentDeclOp>(declaration).getFunctionType()
          : cast<qlx::ActionOp>(declaration).getFunctionType();
  if (inputs.size() != functionType.getNumInputs() ||
      !llvm::equal(inputs.getTypes(), functionType.getInputs()) ||
      results.size() != functionType.getNumResults() ||
      !llvm::equal(results, functionType.getResults())) {
    // P0 logical-qubit types refine to placement-parameterized LVM types, so
    // compare non-quantum types and arity while accepting that refinement.
    if (inputs.size() != functionType.getNumInputs() ||
        results.size() != functionType.getNumResults())
      return op->emitOpError("objective arity does not match placed call");
    for (auto [actual, ideal] :
         llvm::zip(inputs.getTypes(), functionType.getInputs()))
      if (!isa<LogicalQubitType>(actual) && actual != ideal)
        return op->emitOpError(
            "non-quantum input type does not match objective");
    for (auto [actual, ideal] : llvm::zip(results, functionType.getResults()))
      if (!isa<LogicalQubitType>(actual) && actual != ideal)
        return op->emitOpError(
            "non-quantum result type does not match objective");
  }
  return success();
}

LogicalResult ApplyOp::verify() {
  if (!(*this)->getParentOfType<KernelOp>())
    return emitOpError("must appear inside lvm.kernel");
  return verifyPlacedCall(getOperation(), getPlacements(), getInputs(),
                          getResultTypes(), getActionAttr(), false);
}

static FailureOr<FlatSymbolRefAttr>
resolveOwnerSpace(Operation *op, DomainOp domain, SymbolRefAttr placement) {
  if (!placement || placement.getNestedReferences().size() != 1 ||
      placement.getRootReference() != domain.getSymNameAttr()) {
    op->emitOpError("owner placements must be rooted in the kernel domain");
    return failure();
  }
  Operation *target =
      SymbolTable::lookupSymbolIn(domain, placement.getLeafReference());
  if (auto space = dyn_cast_or_null<SpaceOp>(target))
    return FlatSymbolRefAttr::get(op->getContext(), space.getSymName());
  if (auto binding = dyn_cast_or_null<PlacementOp>(target))
    return binding.getSpaceAttr();
  op->emitOpError("owner placement does not resolve to lvm.space or "
                  "lvm.placement");
  return failure();
}

static LogicalResult verifyRemoteMPPParameters(Operation *op,
                                               DictionaryAttr parameters) {
  if (!parameters)
    return op->emitOpError(
        "remote observable requires canonical MPP parameters");
  auto xMask = parameters.getAs<IntegerAttr>("x_mask");
  auto zMask = parameters.getAs<IntegerAttr>("z_mask");
  auto sign = parameters.getAs<IntegerAttr>("sign");
  if (!xMask || !zMask || !sign || parameters.size() != 3)
    return op->emitOpError(
        "remote observable requires exactly x_mask, z_mask, and sign");
  bool uniformXX = xMask.getInt() == 3 && zMask.getInt() == 0;
  bool uniformZZ = xMask.getInt() == 0 && zMask.getInt() == 3;
  if ((!uniformXX && !uniformZZ) || sign.getInt() != 1)
    return op->emitOpError(
        "remote observable supports only positive uniform two-body XX or ZZ "
        "products");
  return success();
}

static LogicalResult
verifyRemoteObservableChannel(Operation *owner, DomainOp domain,
                              ChannelOp selected, Attribute channelCapability,
                              ArrayRef<FlatSymbolRefAttr> endpoints) {
  StringRef direction = selected.getDirectionAttr()
                            ? selected.getDirectionAttr().getValue()
                            : "forward";
  if (direction != "bidirectional")
    return owner->emitOpError(
        "symmetric remote observable requires a bidirectional channel");
  if (auto capacity = selected.getCapacityAttr())
    if (capacity.getInt() <= 0)
      return owner->emitOpError(
          "remote observable channel requires positive capacity");

  unsigned eligible = 0;
  for (ChannelOp candidate : domain.getBody().front().getOps<ChannelOp>()) {
    bool joinsEndpoints = (candidate.getFromAttr() == endpoints[0] &&
                           candidate.getToAttr() == endpoints[1]) ||
                          (candidate.getFromAttr() == endpoints[1] &&
                           candidate.getToAttr() == endpoints[0]);
    StringRef candidateDirection = candidate.getDirectionAttr()
                                       ? candidate.getDirectionAttr().getValue()
                                       : "forward";
    bool hasCapacity = !candidate.getCapacityAttr() ||
                       candidate.getCapacityAttr().getInt() > 0;
    if (joinsEndpoints && candidateDirection == "bidirectional" &&
        hasCapacity &&
        llvm::is_contained(candidate.getCapabilities(), channelCapability))
      ++eligible;
  }
  if (eligible != 1)
    return owner->emitOpError(
        "remote observable requires one unique eligible bidirectional channel");
  return success();
}

static LogicalResult verifyCommunicationObligation(InstrumentOp op) {
  bool hasChannel = static_cast<bool>(op.getChannelAttr());
  bool hasChannelCapability = static_cast<bool>(op.getChannelCapabilityAttr());
  bool hasEndpoints = static_cast<bool>(op.getEndpointsAttr());
  unsigned present = static_cast<unsigned>(hasChannel) +
                     static_cast<unsigned>(hasChannelCapability) +
                     static_cast<unsigned>(hasEndpoints);
  auto kernel = op->getParentOfType<KernelOp>();
  if (!kernel)
    return op.emitOpError("must appear inside lvm.kernel");
  DomainOp domain = lookupDomain(op.getOperation(), kernel.getDomainAttr());
  if (!domain)
    return op.emitOpError("kernel references an unknown lvm.domain");

  llvm::SmallDenseSet<Attribute, 4> ownerSpaces;
  for (Attribute placementAttr : op.getPlacements()) {
    auto placement = dyn_cast<SymbolRefAttr>(placementAttr);
    auto ownerSpace = resolveOwnerSpace(op.getOperation(), domain, placement);
    if (failed(ownerSpace))
      return failure();
    ownerSpaces.insert(*ownerSpace);
  }
  if (present == 0) {
    if (ownerSpaces.size() > 1)
      return op.emitOpError(
          "cross-region instrument requires a communication obligation");
    return success();
  }
  if (present != 3)
    return op.emitOpError("communication obligation requires channel, channel "
                          "capability, and endpoints");
  auto builtin = dyn_cast<qlx::BuiltinInstrumentAttr>(op.getInstrumentAttr());
  if (!builtin || builtin.getValue() != qlx::BuiltinInstrument::mpp)
    return op.emitOpError(
        "communication-qualified instrument must be the built-in MPP");
  if (op.getChannelCapabilityAttr().getKey() != "qlx.machine/observable_remote")
    return op.emitOpError("communication-qualified MPP requires "
                          "observable_remote channel capability");
  if (failed(
          verifyRemoteMPPParameters(op.getOperation(), op.getParametersAttr())))
    return failure();

  auto channelRef = op.getChannelAttr();
  if (channelRef.getNestedReferences().size() != 1 ||
      channelRef.getRootReference() != domain.getSymNameAttr())
    return op.emitOpError("channel must be rooted in the kernel domain");
  auto channel = dyn_cast_or_null<ChannelOp>(
      SymbolTable::lookupSymbolIn(domain, channelRef.getLeafReference()));
  if (!channel)
    return op.emitOpError("channel does not resolve to lvm.channel");

  if (!llvm::is_contained(channel.getCapabilities(),
                          op.getChannelCapabilityAttr()))
    return op.emitOpError(
        "channel does not advertise the requested channel capability");

  ArrayAttr endpoints = op.getEndpointsAttr();
  if (endpoints.size() != 2)
    return op.emitOpError("remote observable requires exactly two endpoints");
  SmallVector<FlatSymbolRefAttr> endpointSpaces;
  for (Attribute item : endpoints) {
    auto endpoint = dyn_cast<SymbolRefAttr>(item);
    if (!endpoint || endpoint.getNestedReferences().size() != 1 ||
        endpoint.getRootReference() != domain.getSymNameAttr())
      return op.emitOpError(
          "endpoints must be space references rooted in the kernel domain");
    auto space = dyn_cast_or_null<SpaceOp>(
        SymbolTable::lookupSymbolIn(domain, endpoint.getLeafReference()));
    if (!space)
      return op.emitOpError("endpoint does not resolve to lvm.space");
    endpointSpaces.push_back(
        FlatSymbolRefAttr::get(op.getContext(), space.getSymName()));
  }
  if (endpointSpaces[0] == endpointSpaces[1])
    return op.emitOpError("remote observable endpoints must be distinct");
  if (failed(verifyRemoteObservableChannel(op.getOperation(), domain, channel,
                                           op.getChannelCapabilityAttr(),
                                           endpointSpaces)))
    return failure();

  auto isForward = endpointSpaces[0] == channel.getFromAttr() &&
                   endpointSpaces[1] == channel.getToAttr();
  auto isReverse = endpointSpaces[0] == channel.getToAttr() &&
                   endpointSpaces[1] == channel.getFromAttr();
  StringRef direction = channel.getDirectionAttr()
                            ? channel.getDirectionAttr().getValue()
                            : "forward";
  if (direction != "forward" && direction != "reverse" &&
      direction != "bidirectional")
    return op.emitOpError("channel direction must be forward, reverse, or "
                          "bidirectional");
  if ((direction == "forward" && !isForward) ||
      (direction == "reverse" && !isReverse) ||
      (direction == "bidirectional" && !isForward && !isReverse))
    return op.emitOpError("ordered endpoints are incompatible with channel "
                          "direction");

  SmallVector<Type> inputOwners;
  SmallVector<Type> resultOwners;
  for (Type type : op.getInputs().getTypes())
    if (isa<LogicalQubitType>(type))
      inputOwners.push_back(type);
  for (Type type : op.getResultTypes())
    if (isa<LogicalQubitType>(type))
      resultOwners.push_back(type);
  if (inputOwners.size() != 2)
    return op.emitOpError(
        "remote observable requires exactly two logical owners");
  if (inputOwners != resultOwners)
    return op.emitOpError(
        "remote observable must preserve owner types and order");
  if (op.getNumResults() == resultOwners.size())
    return op.emitOpError(
        "remote observable requires at least one non-owner outcome");

  for (auto [index, placementAttr] : llvm::enumerate(op.getPlacements())) {
    auto placement = dyn_cast<SymbolRefAttr>(placementAttr);
    auto ownerSpace = resolveOwnerSpace(op.getOperation(), domain, placement);
    if (failed(ownerSpace))
      return failure();
    if (*ownerSpace != endpointSpaces[index])
      return op.emitOpError(
          "owner placement does not match its ordered endpoint");
  }
  return success();
}

LogicalResult ActionSiteOp::verify() {
  if (failed(rejectRetiredMachineCapabilityAttrs(getOperation())))
    return failure();

  if (auto sourceSite = getSourceSiteAttr()) {
    if (sourceSite.getInt() < 0)
      return emitOpError("source_site must be nonnegative");

    auto domain = (*this)->getParentOfType<DomainOp>();
    if (!domain)
      return emitOpError("must appear inside lvm.domain");
  }

  bool hasChannel = static_cast<bool>(getChannelAttr());
  bool hasChannelCapability = static_cast<bool>(getChannelCapabilityAttr());
  bool hasEndpoints = static_cast<bool>(getEndpointsAttr());
  bool hasDirection = static_cast<bool>(getDirectionAttr());
  unsigned present = static_cast<unsigned>(hasChannel) +
                     static_cast<unsigned>(hasChannelCapability) +
                     static_cast<unsigned>(hasEndpoints) +
                     static_cast<unsigned>(hasDirection);
  if (present == 0)
    return success();
  if (present != 4)
    return emitOpError("communication action site requires channel, channel "
                       "capability, endpoints, "
                       "and direction");
  if (getKind() != "instrument")
    return emitOpError("communication action site must have kind = instrument");
  auto builtin =
      dyn_cast_or_null<qlx::BuiltinInstrumentAttr>(getObjectiveAttr());
  if (!builtin || builtin.getValue() != qlx::BuiltinInstrument::mpp)
    return emitOpError(
        "communication action site objective must be the built-in MPP");
  if (getChannelCapabilityAttr().getKey() != "qlx.machine/observable_remote")
    return emitOpError("communication action site requires observable_remote "
                       "channel capability");
  if (failed(verifyRemoteMPPParameters(getOperation(), getParametersAttr())))
    return failure();

  auto domain = (*this)->getParentOfType<DomainOp>();
  if (!domain)
    return emitOpError("must appear inside lvm.domain");
  auto channelRef = getChannelAttr();
  if (channelRef.getNestedReferences().size() != 1 ||
      channelRef.getRootReference().getValue() != domain.getSymName())
    return emitOpError("channel must be rooted in the containing domain (got ")
           << channelRef.getRootReference().getValue() << ", expected "
           << domain.getSymName() << ")";
  auto channel = dyn_cast_or_null<ChannelOp>(
      SymbolTable::lookupSymbolIn(domain, channelRef.getLeafReference()));
  if (!channel)
    return emitOpError("channel does not resolve to lvm.channel");
  if (!llvm::is_contained(channel.getCapabilities(),
                          getChannelCapabilityAttr()))
    return emitOpError(
        "channel does not advertise the requested channel capability");

  ArrayAttr endpoints = getEndpointsAttr();
  if (endpoints.size() != 2 || getPlacements().size() != 2)
    return emitOpError(
        "remote observable action site requires two endpoints and placements");
  SmallVector<FlatSymbolRefAttr> endpointSpaces;
  for (Attribute item : endpoints) {
    auto endpoint = dyn_cast<SymbolRefAttr>(item);
    if (!endpoint || endpoint.getNestedReferences().size() != 1 ||
        endpoint.getRootReference().getValue() != domain.getSymName())
      return emitOpError(
          "endpoints must be space references rooted in the domain");
    auto space = dyn_cast_or_null<SpaceOp>(
        SymbolTable::lookupSymbolIn(domain, endpoint.getLeafReference()));
    if (!space)
      return emitOpError("endpoint does not resolve to lvm.space");
    endpointSpaces.push_back(
        FlatSymbolRefAttr::get(getContext(), space.getSymName()));
  }
  if (endpointSpaces[0] == endpointSpaces[1])
    return emitOpError("remote observable endpoints must be distinct");
  if (failed(verifyRemoteObservableChannel(getOperation(), domain, channel,
                                           getChannelCapabilityAttr(),
                                           endpointSpaces)))
    return failure();

  bool isForward = endpointSpaces[0] == channel.getFromAttr() &&
                   endpointSpaces[1] == channel.getToAttr();
  bool isReverse = endpointSpaces[0] == channel.getToAttr() &&
                   endpointSpaces[1] == channel.getFromAttr();
  StringRef channelDirection = channel.getDirectionAttr()
                                   ? channel.getDirectionAttr().getValue()
                                   : "forward";
  if (getDirection() != channelDirection)
    return emitOpError("direction does not match canonical channel provenance");
  if ((channelDirection == "forward" && !isForward) ||
      (channelDirection == "reverse" && !isReverse) ||
      (channelDirection == "bidirectional" && !isForward && !isReverse))
    return emitOpError(
        "ordered endpoints are incompatible with channel direction");
  for (auto [placementAttr, endpoint] :
       llvm::zip(getPlacements(), endpointSpaces)) {
    auto placement = dyn_cast<SymbolRefAttr>(placementAttr);
    auto ownerSpace = resolveOwnerSpace(getOperation(), domain, placement);
    if (failed(ownerSpace))
      return failure();
    if (*ownerSpace != endpoint)
      return emitOpError("owner placement does not match its ordered endpoint");
  }
  return success();
}

LogicalResult InstrumentOp::verify() {
  if (failed(rejectRetiredMachineCapabilityAttrs(getOperation())))
    return failure();
  if (failed(verifyPlacedCall(getOperation(), getPlacements(), getInputs(),
                              getResultTypes(), getInstrumentAttr(), true)))
    return failure();
  return verifyCommunicationObligation(*this);
}

LogicalResult MeasureOp::verify() {
  auto kernel = (*this)->getParentOfType<KernelOp>();
  if (!kernel)
    return emitOpError("must appear inside lvm.kernel");
  DomainOp domain = lookupDomain(getOperation(), kernel.getDomainAttr());
  if (failed(verifyPlacementReference(getOperation(), getAtAttr(), domain)))
    return failure();
  return verifyPlacedType(getOperation(), getInput().getType(), getAtAttr());
}

LogicalResult IdleOp::verify() {
  if (!(*this)->getParentOfType<KernelOp>())
    return emitOpError("must appear inside lvm.kernel");
  if (getInputs().empty() || getInputs().size() != getResults().size())
    return emitOpError("requires one successor for each logical input");
  if (getPlacements().size() != getInputs().size())
    return emitOpError("requires one placement per logical input");
  for (auto [input, result, attr] :
       llvm::zip(getInputs(), getResults(), getPlacements())) {
    auto placement = dyn_cast<SymbolRefAttr>(attr);
    if (!placement ||
        failed(verifyPlacedType(getOperation(), input.getType(), placement)) ||
        failed(verifyPlacedType(getOperation(), result.getType(), placement)))
      return failure();
  }
  return success();
}

LogicalResult ConsumeResourceOp::verify() {
  auto type = getResource().getType();
  if (type.getKind() != getResourceKindAttr().getValue())
    return emitOpError(
        "resource_kind must match the typed logical resource kind");
  if (type.getStream() != getResourceStreamAttr())
    return emitOpError(
        "resource_stream must match the typed logical resource stream");
  auto stream = dyn_cast_or_null<StreamOp>(
      SymbolTable::lookupNearestSymbolFrom(*this, getResourceStreamAttr()));
  if (!stream)
    return emitOpError("resource_stream must resolve to an lvm.stream");
  if (stream.getProducesAttr() != getResourceKindAttr())
    return emitOpError(
        "resource_stream must produce the consumed resource kind");
  return success();
}

#define GET_OP_CLASSES
#include "qlx/Dialect/LVM/IR/LVMOps.cpp.inc"

void LVMDialect::initialize() {
  addInterfaces<LVMDeviceBindingDialectInterface>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "qlx/Dialect/LVM/IR/LVMTypes.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "qlx/Dialect/LVM/IR/LVMAttrs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "qlx/Dialect/LVM/IR/LVMOps.cpp.inc"
      >();
}
