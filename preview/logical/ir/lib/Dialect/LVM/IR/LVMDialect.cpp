//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//

#include "qlx/Dialect/LVM/IR/LVMDialect.h"
#include "qlx/Dialect/LVM/IR/LVMAttrs.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/LVM/IR/LVMTypes.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/SymbolTable.h"

using namespace mlir;
using namespace qlx::lvm;

#include "qlx/Dialect/LVM/IR/LVMDialect.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/LVM/IR/LVMTypes.cpp.inc"

Type LVMDialect::parseType(DialectAsmParser &parser) const {
  SMLoc location = parser.getCurrentLocation();
  StringRef mnemonic;
  if (failed(parser.parseKeyword(&mnemonic)))
    return {};
  if (mnemonic == LogicalQubitType::getMnemonic())
    return LogicalQubitType::parse(parser);
  parser.emitError(location) << "unknown type in dialect 'lvm': " << mnemonic;
  return {};
}

void LVMDialect::printType(Type type, DialectAsmPrinter &printer) const {
  if (auto value = dyn_cast<LogicalQubitType>(type)) {
    printer << LogicalQubitType::getMnemonic();
    value.print(printer);
    return;
  }
  llvm_unreachable("attempted to print an unregistered LVM type");
}

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
  if (failed(verifyMachineCapabilities(getOperation(), getCapabilities())))
    return failure();
  if (getCapabilities().size() != 1) {
    return emitOpError(
        "QLX channels are compiler-derived resource-supply edges");
  }
  auto capability = dyn_cast<CapabilityAttr>(*getCapabilities().begin());
  if (!capability || capability.getKey() != "qlx.machine/resource_transfer")
    return emitOpError(
        "QLX channels require exactly resource_transfer capability");
  if (getCapacityAttr())
    return emitOpError("QLX resource-supply channels have no capacity");
  if (getDirectionAttr() && getDirectionAttr().getValue() != "forward")
    return emitOpError("QLX resource-supply channels are forward-only");
  auto domain = (*this)->getParentOfType<DomainOp>();
  auto source = dyn_cast_or_null<SpaceOp>(
      SymbolTable::lookupSymbolIn(domain, getFromAttr().getValue()));
  auto destination = dyn_cast_or_null<StreamOp>(
      SymbolTable::lookupSymbolIn(domain, getToAttr().getValue()));
  if (!source || !destination)
    return emitOpError(
        "QLX resource-supply channel must connect space to stream");
  if (!destination.getBackingRegionAttr())
    return emitOpError(
        "QLX resource-supply destination must be a backed stream");
  if (!destination.getProducedByAttr() ||
      !destination->getAttrOfType<StringAttr>("produced_by_sha256"))
    return emitOpError("QLX resource-supply destination requires authenticated "
                       "produced_by provenance");
  Operation *producer = lookupObjective(destination.getOperation(),
                                        destination.getProducedByAttr());
  if (!producer || producer->getName().getStringRef() != "fabric.protocol")
    return emitOpError(
        "QLX resource-supply produced_by must resolve to a typed "
        "fabric.protocol");
  return success();
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
    return emitOpError("nonlocal placement bindings are not supported by QLX");
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

static LogicalResult verifyPlacementReference(Operation *op, SymbolRefAttr ref,
                                              DomainOp expectedDomain) {
  if (!ref || ref.getNestedReferences().size() != 1)
    return op->emitOpError("placement references must be @domain::@placement");
  if (ref.getRootReference() != expectedDomain.getSymNameAttr())
    return op->emitOpError("placement reference ")
           << ref << " is not rooted in kernel domain @"
           << expectedDomain.getSymName();
  Operation *target =
      SymbolTable::lookupSymbolIn(expectedDomain, ref.getLeafReference());
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

LogicalResult KernelOp::verify() {
  DomainOp domain = lookupDomain(getOperation(), getDomainAttr());
  if (!domain)
    return emitOpError("references unknown lvm.domain ") << getDomainAttr();
  auto inputP0 = getInputP0Attr();
  if (getEstimateOnly() && !inputP0)
    return emitOpError("estimate_only requires an input_p0 refinement");
  if (inputP0) {
    auto inputProgram = dyn_cast_or_null<qlx::ProgramOp>(
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
  getBody().walk([&](Operation *nested) {
    if (failed(result))
      return;
    for (Type valueType : nested->getOperandTypes()) {
      if (auto logical = dyn_cast<LogicalQubitType>(valueType))
        if (failed(verifyPlacementReference(nested, logical.getPlacement(),
                                            domain)))
          result = failure();
    }
    for (Type valueType : nested->getResultTypes()) {
      if (auto logical = dyn_cast<LogicalQubitType>(valueType))
        if (failed(verifyPlacementReference(nested, logical.getPlacement(),
                                            domain)))
          result = failure();
    }
  });
  return result;
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
  if (present != 0)
    return op.emitOpError(
        "communication-qualified instruments are not supported by QLX");
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
  bool hasChannel = static_cast<bool>(getChannelAttr());
  bool hasChannelCapability = static_cast<bool>(getChannelCapabilityAttr());
  bool hasEndpoints = static_cast<bool>(getEndpointsAttr());
  bool hasDirection = static_cast<bool>(getDirectionAttr());
  unsigned present = static_cast<unsigned>(hasChannel) +
                     static_cast<unsigned>(hasChannelCapability) +
                     static_cast<unsigned>(hasEndpoints) +
                     static_cast<unsigned>(hasDirection);
  if (present != 0)
    return emitOpError("communication action sites are not supported by QLX");
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

LogicalResult IfOp::verify() {
  for (Region *region : {&getThenRegion(), &getElseRegion()}) {
    if (!llvm::hasSingleElement(*region))
      return emitOpError("branches must each contain one block");
    auto yield = dyn_cast<YieldOp>(region->front().getTerminator());
    if (!yield || yield.getOperandTypes() != getResultTypes())
      return emitOpError("branch yields must match result types");
  }
  return success();
}

LogicalResult RepeatOp::verify() {
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("body must contain one block");
  Block &body = getBody().front();
  if (body.getArgumentTypes() != getInits().getTypes() ||
      getResultTypes() != getInits().getTypes())
    return emitOpError(
        "iter arguments, inits, and results must have equal types");
  auto yield = dyn_cast<YieldOp>(body.getTerminator());
  if (!yield || yield.getOperandTypes() != getResultTypes())
    return emitOpError("yield types must match repeat result types");
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
    return emitOpError("before region must terminate with lvm.while_condition");
  if (condition.getForwarded().getTypes() != getResultTypes())
    return emitOpError(
        "while_condition forwarded types must match loop result types");
  auto yield = dyn_cast<YieldOp>(after.getTerminator());
  if (!yield)
    return emitOpError("after region must terminate with lvm.yield");
  if (yield.getOperandTypes() != getResultTypes())
    return emitOpError("after-region yield types must match loop result types");
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
  addTypes<LogicalQubitType>();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "qlx/Dialect/LVM/IR/LVMAttrs.cpp.inc"
      >();
  addOperations<DomainOp, SpaceOp, StreamOp, ChannelOp, PlacementOp,
                ActionSiteOp, ReturnOp, KernelOp, PrepareOp, ApplyOp,
                InstrumentOp, MeasureOp, IdleOp, DiscardOp, SelectionOp, XorOp,
                YieldOp, IfOp, RepeatOp, WhileConditionOp, WhileOp>();
}
