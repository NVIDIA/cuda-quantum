/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Dialect/Fabric/IR/FabricDialect.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/Event/IR/EventOps.h"
#include "qlx/Dialect/Fabric/IR/FabricAttrs.h"
#include "qlx/Dialect/Fabric/IR/FabricInterfaces.h"
#include "qlx/Dialect/Fabric/IR/FabricOps.h"
#include "qlx/Dialect/Fabric/IR/FabricTypes.h"
#include "qlx/Dialect/Fabric/IR/ResourceContract.h"
#include "qlx/Dialect/LVM/IR/LVMAttrs.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/QLX/IR/QLXAttrs.h"
#include "qlx/Dialect/QLX/IR/QLXDialect.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <optional>
#include <set>
#include <string>

using namespace mlir;
using namespace qlx::fabric;

namespace {

struct ResourceContractRegistration {
  qlx::fabric::PackResourceVerifier pack;
  qlx::fabric::UnpackResourceVerifier unpack;
};

static std::mutex &resourceContractMutex() {
  static std::mutex mutex;
  return mutex;
}

static llvm::StringMap<ResourceContractRegistration> &resourceContracts() {
  static llvm::StringMap<ResourceContractRegistration> contracts;
  return contracts;
}

} // namespace

void qlx::fabric::registerResourceContractVerifier(
    StringRef resourceKind, PackResourceVerifier packVerifier,
    UnpackResourceVerifier unpackVerifier) {
  if (resourceKind.empty() || (!packVerifier && !unpackVerifier))
    llvm::report_fatal_error(
        "invalid QLX resource-contract verifier registration");
  std::lock_guard<std::mutex> lock(resourceContractMutex());
  auto [entry, inserted] = resourceContracts().try_emplace(
      resourceKind, ResourceContractRegistration{packVerifier, unpackVerifier});
  if (!inserted && (entry->second.pack != packVerifier ||
                    entry->second.unpack != unpackVerifier))
    llvm::report_fatal_error(
        "conflicting QLX resource-contract verifier registration");
}

LogicalResult qlx::fabric::verifyRegisteredResourcePack(PackResourceOp pack) {
  PackResourceVerifier verifier = nullptr;
  {
    std::lock_guard<std::mutex> lock(resourceContractMutex());
    auto found =
        resourceContracts().find(pack.getResourceKindAttr().getValue());
    if (found != resourceContracts().end())
      verifier = found->second.pack;
  }
  return verifier ? verifier(pack) : success();
}

LogicalResult
qlx::fabric::verifyRegisteredResourceUnpack(UnpackResourceOp unpack) {
  UnpackResourceVerifier verifier = nullptr;
  auto resource = cast<ResourceStateType>(unpack.getResource().getType());
  auto kind = dyn_cast<SymbolRefAttr>(resource.getKind());
  {
    std::lock_guard<std::mutex> lock(resourceContractMutex());
    auto found =
        kind ? resourceContracts().find(kind.getRootReference().getValue())
             : resourceContracts().end();
    if (found != resourceContracts().end())
      verifier = found->second.unpack;
  }
  return verifier ? verifier(unpack) : success();
}

bool qlx::fabric::arePhysicallyTemplateEquivalent(Operation *lhs,
                                                  Operation *rhs) {
  std::set<std::pair<Operation *, Operation *>> active;
  std::set<std::pair<Operation *, Operation *>> proven;
  std::function<bool(Operation *, Operation *)> equivalent =
      [&](Operation *left, Operation *right) -> bool {
    if (left == right)
      return true;
    if (!left || !right || left->getName() != right->getName())
      return false;
    std::pair<Operation *, Operation *> pair{left, right};
    if (proven.contains(pair) || active.contains(pair))
      return true;
    auto leftGenerated = left->getAttrOfType<FlatSymbolRefAttr>("generated_by");
    auto rightGenerated =
        right->getAttrOfType<FlatSymbolRefAttr>("generated_by");
    if (!leftGenerated || leftGenerated != rightGenerated)
      return false;
    active.insert(pair);
    llvm::scope_exit clearActive([&] { active.erase(pair); });

    SmallVector<CallOp> leftCalls;
    SmallVector<CallOp> rightCalls;
    left->walk([&](CallOp call) { leftCalls.push_back(call); });
    right->walk([&](CallOp call) { rightCalls.push_back(call); });
    if (leftCalls.size() != rightCalls.size())
      return false;
    for (auto [leftCall, rightCall] : llvm::zip(leftCalls, rightCalls)) {
      Operation *leftTarget = SymbolTable::lookupNearestSymbolFrom(
          leftCall, leftCall.getCalleeAttr());
      Operation *rightTarget = SymbolTable::lookupNearestSymbolFrom(
          rightCall, rightCall.getCalleeAttr());
      if (!equivalent(leftTarget, rightTarget))
        return false;
    }

    Operation *leftClone = left->clone();
    Operation *rightClone = right->clone();
    llvm::scope_exit cleanup([&] {
      leftClone->destroy();
      rightClone->destroy();
    });
    auto normalize = [](Operation *callable) {
      for (StringRef attribute :
           {SymbolTable::getSymbolAttrName(), StringRef("action_site"),
            StringRef("input_p1_kernel"), StringRef("input_p1_callee"),
            StringRef("input_p1_scope")})
        callable->removeAttr(attribute);
      unsigned callIndex = 0;
      callable->walk([&](Operation *nested) {
        if (auto call = dyn_cast<CallOp>(nested)) {
          call->setAttr(
              "callee",
              FlatSymbolRefAttr::get(
                  callable->getContext(),
                  ("__physical_template_callee_" + Twine(callIndex++)).str()));
        }
        if (auto unpack = dyn_cast<UnpackResourceOp>(nested))
          unpack->removeAttr("payload_logical_block_ids");
      });
    };
    normalize(leftClone);
    normalize(rightClone);
    if (!OperationEquivalence::isEquivalentTo(
            leftClone, rightClone, OperationEquivalence::IgnoreLocations))
      return false;
    proven.insert(pair);
    return true;
  };
  return equivalent(lhs, rhs);
}

namespace {
class FabricDeviceBindingDialectInterface final
    : public qlx::DeviceBindingDialectInterface {
public:
  explicit FabricDeviceBindingDialectInterface(Dialect *dialect)
      : DeviceBindingDialectInterface(dialect) {}

  FlatSymbolRefAttr getResourceKind(Type type) const override {
    auto resource = dyn_cast<ResourceStateType>(type);
    if (!resource)
      return {};
    return dyn_cast<FlatSymbolRefAttr>(resource.getKind());
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Generated dialect definition
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Fabric/IR/FabricDialect.cpp.inc"

#include "qlx/Dialect/Fabric/IR/FabricInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated enum definitions (all enums from FabricAttrs.td)
//===----------------------------------------------------------------------===//

#include "qlx/Dialect/Fabric/IR/FabricAttrEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Generated attribute definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricAttrs.cpp.inc"

//===----------------------------------------------------------------------===//
// ResourceStateType custom assembly:
//   alpha compatibility: !fabric.resource<T|CCZ|CS>
//   QLX open kind:   !fabric.resource<@kind>
//===----------------------------------------------------------------------===//

mlir::Type ResourceStateType::parse(mlir::AsmParser &parser) {
  if (parser.parseLess())
    return {};
  SymbolRefAttr symbol;
  auto optionalSymbol = parser.parseOptionalAttribute(symbol);
  if (optionalSymbol.has_value()) {
    if (failed(*optionalSymbol))
      return {};
    if (parser.parseGreater())
      return {};
    return ResourceStateType::get(parser.getContext(), symbol);
  }
  llvm::StringRef keyword;
  if (parser.parseKeyword(&keyword))
    return {};
  auto rt = symbolizeResourceType(keyword);
  if (!rt) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown resource type: " + keyword);
    return {};
  }
  if (parser.parseGreater())
    return {};
  return ResourceStateType::get(parser.getContext(), *rt);
}

void ResourceStateType::print(mlir::AsmPrinter &printer) const {
  printer << "<";
  if (hasLegacyResourceType())
    printer << stringifyResourceType(getResourceType());
  else
    printer.printAttribute(getKind());
  printer << ">";
}

//===----------------------------------------------------------------------===//
// FloorplanAttr custom assembly: #fabric.floorplan<checkerboard, [11, 11]>
//===----------------------------------------------------------------------===//

mlir::Attribute FloorplanAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  if (parser.parseLess())
    return {};
  llvm::StringRef layoutKw;
  if (parser.parseKeyword(&layoutKw))
    return {};
  auto layout = symbolizeLayout(layoutKw);
  if (!layout) {
    parser.emitError(parser.getCurrentLocation(),
                     "unknown layout: " + layoutKw);
    return {};
  }
  if (parser.parseComma())
    return {};
  llvm::SmallVector<int64_t> params;
  if (parser.parseLSquare())
    return {};
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        int64_t v;
        if (parser.parseInteger(v))
          return failure();
        params.push_back(v);
        return success();
      }))
    return {};
  if (parser.parseRSquare() || parser.parseGreater())
    return {};
  return FloorplanAttr::get(parser.getContext(), *layout, params);
}

void FloorplanAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<" << stringifyLayout(getLayout()) << ", [";
  llvm::interleaveComma(getParams(), printer, [&](int64_t v) { printer << v; });
  printer << "]>";
}

//===----------------------------------------------------------------------===//
// FlowAttr custom assembly: #fabric.flow<{x = "z", z = "x"}>
//===----------------------------------------------------------------------===//

mlir::Attribute FlowAttr::parse(mlir::AsmParser &parser, mlir::Type type) {
  if (parser.parseLess() || parser.parseLBrace())
    return {};
  if (parser.parseKeyword("x") || parser.parseEqual())
    return {};
  std::string xTo;
  if (parser.parseString(&xTo))
    return {};
  if (parser.parseComma())
    return {};
  if (parser.parseKeyword("z") || parser.parseEqual())
    return {};
  std::string zTo;
  if (parser.parseString(&zTo))
    return {};
  if (parser.parseRBrace() || parser.parseGreater())
    return {};
  return FlowAttr::get(parser.getContext(),
                       StringAttr::get(parser.getContext(), xTo),
                       StringAttr::get(parser.getContext(), zTo));
}

void FlowAttr::print(mlir::AsmPrinter &printer) const {
  printer << "<{x = \"" << getXTo().getValue() << "\", z = \""
          << getZTo().getValue() << "\"}>";
}

//===----------------------------------------------------------------------===//
// Custom format directives for ops
//===----------------------------------------------------------------------===//

// Parse bare partition keyword: data, sx, sz, all
static ParseResult parsePartitionKeyword(OpAsmParser &parser,
                                         PartitionAttr &attr) {
  llvm::StringRef kw;
  if (parser.parseKeyword(&kw))
    return failure();
  auto val = symbolizePartition(kw);
  if (!val) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected partition keyword (data, sx, sz, all)");
    return failure();
  }
  attr = PartitionAttr::get(parser.getContext(), *val);
  return success();
}

static void printPartitionKeyword(OpAsmPrinter &printer, Operation *,
                                  PartitionAttr attr) {
  printer << stringifyPartition(attr.getValue());
}

// Parse optional index list: [0, 2, 4]
static ParseResult parseOptionalIndices(OpAsmParser &parser,
                                        DenseI64ArrayAttr &attr) {
  attr = nullptr;
  if (failed(parser.parseOptionalLSquare()))
    return success();
  llvm::SmallVector<int64_t> indices;
  if (parser.parseCommaSeparatedList([&]() -> ParseResult {
        int64_t v;
        if (parser.parseInteger(v))
          return failure();
        indices.push_back(v);
        return success();
      }))
    return failure();
  if (parser.parseRSquare())
    return failure();
  attr = DenseI64ArrayAttr::get(parser.getContext(), indices);
  return success();
}

static void printOptionalIndices(OpAsmPrinter &printer, Operation *,
                                 DenseI64ArrayAttr attr) {
  if (!attr || attr.empty())
    return;
  printer << "[";
  llvm::interleaveComma(attr.asArrayRef(), printer,
                        [&](int64_t v) { printer << v; });
  printer << "]";
}

// Parse bare merge basis keyword: X, Z
static ParseResult parseMergeBasisKeyword(OpAsmParser &parser,
                                          MergeBasisAttr &attr) {
  llvm::StringRef kw;
  if (parser.parseKeyword(&kw))
    return failure();
  auto val = symbolizeMergeBasis(kw);
  if (!val) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected merge basis keyword (X, Z)");
    return failure();
  }
  attr = MergeBasisAttr::get(parser.getContext(), *val);
  return success();
}

static void printMergeBasisKeyword(OpAsmPrinter &printer, Operation *,
                                   MergeBasisAttr attr) {
  printer << stringifyMergeBasis(attr.getValue());
}

// Parse bare boundary keyword: north, east, south, west
static ParseResult parseBoundaryKeyword(OpAsmParser &parser,
                                        BoundaryAttr &attr) {
  llvm::StringRef kw;
  if (parser.parseKeyword(&kw))
    return failure();
  auto val = symbolizeBoundary(kw);
  if (!val) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected boundary keyword (north, east, south, west)");
    return failure();
  }
  attr = BoundaryAttr::get(parser.getContext(), *val);
  return success();
}

static void printBoundaryKeyword(OpAsmPrinter &printer, Operation *,
                                 BoundaryAttr attr) {
  printer << stringifyBoundary(attr.getValue());
}

//===----------------------------------------------------------------------===//
// GadgetOp custom assembly
//===----------------------------------------------------------------------===//
//
// fabric.gadget @name(%arg: type, ...) -> (result_types)
//     flow #fabric.flow<{...}>
// { body }
//
// fabric.gadget @name {entry} on @device () { body }
//

//===----------------------------------------------------------------------===//
// CircuitOp custom assembly
//===----------------------------------------------------------------------===//

void CircuitOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                      FunctionType type, ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addRegion();
  state.addAttributes(attrs);
}

ParseResult CircuitOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          OpAsmParser::Argument arg;
          if (parser.parseArgument(arg, /*allowType=*/true,
                                   /*allowAttrs=*/false))
            return failure();
          args.push_back(arg);
          return success();
        }) ||
        parser.parseRParen())
      return failure();
  }

  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    } else {
      Type type;
      if (parser.parseType(type))
        return failure();
      resultTypes.push_back(type);
    }
  }
  SmallVector<Type> argumentTypes;
  for (const auto &argument : args)
    argumentTypes.push_back(argument.type);
  result.addAttribute(getFunctionTypeAttrName(result.name),
                      TypeAttr::get(FunctionType::get(
                          parser.getContext(), argumentTypes, resultTypes)));

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, args, /*enableNameShadowing=*/false))
    return failure();
  ensureTerminator(*body, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void CircuitOp::print(OpAsmPrinter &printer) {
  printer << " @" << getSymName();
  Block &block = getBody().front();
  printer << "(";
  llvm::interleaveComma(
      block.getArguments(), printer,
      [&](BlockArgument argument) { printer.printRegionArgument(argument); });
  printer << ")";
  TypeRange results = getFunctionType().getResults();
  if (!results.empty()) {
    printer << " -> ";
    if (results.size() == 1)
      printer << results.front();
    else {
      printer << "(";
      llvm::interleaveComma(results, printer);
      printer << ")";
    }
  }
  printer << " ";
  printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                      /*printBlockTerminators=*/true);
  printer.printOptionalAttrDict(
      (*this)->getAttrs(),
      {SymbolTable::getSymbolAttrName(), getFunctionTypeAttrName()});
}

void GadgetOp::build(OpBuilder &builder, OperationState &state, StringRef name,
                     FunctionType type, ArrayRef<NamedAttribute> attrs) {
  state.addAttribute(SymbolTable::getSymbolAttrName(),
                     builder.getStringAttr(name));
  state.addAttribute(getFunctionTypeAttrName(state.name), TypeAttr::get(type));
  state.addRegion();
  state.addAttributes(attrs);
}

ParseResult GadgetOp::parse(OpAsmParser &parser, OperationState &result) {
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr, SymbolTable::getSymbolAttrName(),
                             result.attributes))
    return failure();

  // Optional {entry} on @device
  bool isEntry = false;
  if (succeeded(parser.parseOptionalLBrace())) {
    if (parser.parseKeyword("entry") || parser.parseRBrace())
      return failure();
    result.addAttribute("entry", parser.getBuilder().getUnitAttr());
    isEntry = true;
    if (parser.parseKeyword("on"))
      return failure();
    FlatSymbolRefAttr devRef;
    if (parser.parseAttribute(devRef))
      return failure();
    result.addAttribute("device", devRef);
  }

  // Parse argument list
  SmallVector<OpAsmParser::Argument> args;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    if (parser.parseCommaSeparatedList([&]() -> ParseResult {
          OpAsmParser::Argument arg;
          if (parser.parseArgument(arg, /*allowType=*/true,
                                   /*allowAttrs=*/false))
            return failure();
          args.push_back(arg);
          return success();
        }))
      return failure();
    if (parser.parseRParen())
      return failure();
  }

  // Parse optional result types
  SmallVector<Type> resultTypes;
  if (succeeded(parser.parseOptionalArrow())) {
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    } else {
      Type singleType;
      if (parser.parseType(singleType))
        return failure();
      resultTypes.push_back(singleType);
    }
  }

  // Build function type
  SmallVector<Type> argTypes;
  for (auto &arg : args)
    argTypes.push_back(arg.type);
  auto funcType = FunctionType::get(parser.getContext(), argTypes, resultTypes);
  result.addAttribute(GadgetOp::getFunctionTypeAttrName(result.name),
                      TypeAttr::get(funcType));

  if (succeeded(parser.parseOptionalKeyword("realization"))) {
    FlatSymbolRefAttr reference;
    if (parser.parseAttribute(reference))
      return failure();
    result.addAttribute("realization", reference);
  }

  // Parse optional flow attribute
  if (succeeded(parser.parseOptionalKeyword("flow"))) {
    FlowAttr flowAttr;
    if (parser.parseAttribute(flowAttr))
      return failure();
    result.addAttribute("flow", flowAttr);
  }

  // A referenced realization owns no inline body. Parse its trailing
  // attributes before synthesizing the argument-bearing signature stub; both
  // an MLIR region and an attribute dictionary start with `{`, so attempting
  // to parse an optional region first would misread attributes such as `spec`
  // and `realization_kind` as a body.
  auto *body = result.addRegion();
  if (result.attributes.get("realization")) {
    Block &block = body->emplaceBlock();
    for (auto &arg : args)
      block.addArgument(arg.type, result.location);
    ensureTerminator(*body, parser.getBuilder(), result.location);
    if (parser.parseOptionalAttrDict(result.attributes))
      return failure();
    return success();
  }

  // Inline gadgets continue to own and print their realization region.
  OptionalParseResult bodyResult =
      parser.parseOptionalRegion(*body, args, /*enableNameShadowing=*/false);
  if (bodyResult.has_value()) {
    if (failed(*bodyResult))
      return failure();
  } else {
    Block &block = body->emplaceBlock();
    for (auto &arg : args)
      block.addArgument(arg.type, result.location);
  }
  ensureTerminator(*body, parser.getBuilder(), result.location);
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  return success();
}

void GadgetOp::print(OpAsmPrinter &printer) {
  printer << " @" << getSymName();
  if (getEntry()) {
    printer << " {entry} on ";
    printer.printAttribute(getDeviceAttr());
  }

  // Print arguments
  auto funcType = getFunctionType();
  auto &entryBlock = getBody().front();
  printer << "(";
  for (unsigned i = 0, e = entryBlock.getNumArguments(); i < e; ++i) {
    if (i > 0)
      printer << ", ";
    printer.printRegionArgument(entryBlock.getArgument(i));
  }
  printer << ")";

  // Print result types
  auto resultTypes = funcType.getResults();
  if (!resultTypes.empty()) {
    printer << " -> ";
    if (resultTypes.size() == 1) {
      printer << resultTypes[0];
    } else {
      printer << "(";
      llvm::interleaveComma(resultTypes, printer);
      printer << ")";
    }
  }

  if (auto realization = getRealizationAttr()) {
    printer << " realization ";
    printer.printAttribute(realization);
  }

  // Print flow attribute
  if (auto flow = getFlow()) {
    printer << " flow ";
    printer.printAttribute(*flow);
  }

  // A referenced gadget owns only an empty signature stub. The reusable
  // circuit is the sole realization body and is printed independently.
  if (!getRealizationAttr()) {
    printer << " ";
    printer.printRegion(getBody(), /*printEntryBlockArgs=*/false,
                        /*printBlockTerminators=*/true);
  }

  // Elide known attributes from attr-dict
  SmallVector<StringRef, 8> elidedAttrs = {SymbolTable::getSymbolAttrName(),
                                           getFunctionTypeAttrName(),
                                           "entry",
                                           "device",
                                           "flow",
                                           "realization"};
  printer.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

//===----------------------------------------------------------------------===//
// MultiMeasureOp custom assembly
//===----------------------------------------------------------------------===//
//
// %p0_out, %p1_out, %bit = fabric.multi_measure %p0, %p1
//     corridors %c0
//     pauli_product = "ZZ"
//     : (!fabric.patch<@sc>, !fabric.patch<@sc>, !fabric.slot)
//     -> (!fabric.patch<@sc>, !fabric.patch<@sc>, i1)
//

ParseResult MultiMeasureOp::parse(OpAsmParser &parser, OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> patchOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> corridorOperands;

  // Parse patch operands
  if (parser.parseOperandList(patchOperands))
    return failure();

  // Parse corridors
  if (parser.parseKeyword("corridors"))
    return failure();
  if (parser.parseOperandList(corridorOperands))
    return failure();

  // Parse pauli_product attribute
  StringAttr pauliProductAttr;
  if (parser.parseKeyword("pauli_product") || parser.parseEqual() ||
      parser.parseAttribute(pauliProductAttr))
    return failure();
  result.addAttribute("pauli_product", pauliProductAttr);

  // Parse functional type
  FunctionType funcType;
  if (parser.parseColonType(funcType))
    return failure();

  // Resolve operands
  auto inputTypes = funcType.getInputs();
  unsigned numPatches = patchOperands.size();
  unsigned numCorridors = corridorOperands.size();
  if (inputTypes.size() != numPatches + numCorridors) {
    parser.emitError(parser.getCurrentLocation(),
                     "type mismatch in operand count");
    return failure();
  }
  if (parser.resolveOperands(patchOperands, inputTypes.take_front(numPatches),
                             parser.getCurrentLocation(), result.operands))
    return failure();
  if (parser.resolveOperands(corridorOperands,
                             inputTypes.drop_front(numPatches),
                             parser.getCurrentLocation(), result.operands))
    return failure();

  // Segment sizes
  result.addAttribute("operandSegmentSizes",
                      parser.getBuilder().getDenseI32ArrayAttr(
                          {static_cast<int32_t>(numPatches),
                           static_cast<int32_t>(numCorridors)}));

  result.addTypes(funcType.getResults());
  return success();
}

void MultiMeasureOp::print(OpAsmPrinter &printer) {
  printer << " ";
  printer.printOperands(getPatches());
  printer << " corridors ";
  printer.printOperands(getCorridors());
  printer << " pauli_product = ";
  printer.printAttribute(getPauliProductAttr());

  // Print functional type
  SmallVector<Type> inputTypes;
  for (auto v : getPatches())
    inputTypes.push_back(v.getType());
  for (auto v : getCorridors())
    inputTypes.push_back(v.getType());
  auto funcType = FunctionType::get(getContext(), inputTypes,
                                    getOperation()->getResultTypes());
  printer << " : " << funcType;
}

//===----------------------------------------------------------------------===//
// Verifiers
//===----------------------------------------------------------------------===//

static LogicalResult verifyEncodingQualifiedTypes(Operation *owner,
                                                  TypeRange types) {
  for (Type type : types) {
    FlatSymbolRefAttr code;
    FlatSymbolRefAttr encoding;
    FlatSymbolRefAttr epoch;
    if (auto patch = dyn_cast<PatchType>(type)) {
      code = patch.getCodeType();
      encoding = patch.getEncoding();
      epoch = patch.getEpoch();
    } else if (auto syndrome = dyn_cast<SyndromeType>(type)) {
      code = syndrome.getCodeType();
      encoding = syndrome.getEncoding();
      epoch = syndrome.getEpoch();
    } else {
      continue;
    }
    if (!encoding)
      continue; // Compatibility form for alpha and partially linked IR.
    Operation *target = SymbolTable::lookupNearestSymbolFrom(owner, encoding);
    if (!target)
      if (auto module = owner->getParentOfType<ModuleOp>())
        target = SymbolTable(module).lookup(encoding.getValue());
    if (!target)
      continue; // Link verification diagnoses unresolved external symbols.
    auto declaration = dyn_cast<EncodingOp>(target);
    if (!declaration)
      return owner->emitOpError("encoding-qualified type references @")
             << encoding.getValue() << ", which is not fabric.encoding";
    if (declaration.getCodeAttr() != code)
      return owner->emitOpError("encoding-qualified type code ")
             << code << " disagrees with encoding @" << encoding.getValue()
             << " code " << declaration.getCodeAttr();
    if (epoch) {
      Operation *epochTarget =
          SymbolTable::lookupNearestSymbolFrom(owner, epoch);
      if (!epochTarget)
        if (auto module = owner->getParentOfType<ModuleOp>())
          epochTarget = SymbolTable(module).lookup(epoch.getValue());
      if (epochTarget) {
        auto instance = dyn_cast<EncodingEpochOp>(epochTarget);
        if (!instance)
          return owner->emitOpError("epoch-qualified type references @")
                 << epoch.getValue() << ", which is not fabric.encoding_epoch";
        if (instance.getEncodingAttr() != encoding)
          return owner->emitOpError("epoch @")
                 << epoch.getValue() << " belongs to encoding "
                 << instance.getEncodingAttr() << ", not " << encoding;
      }
    }
  }
  return success();
}

LogicalResult InterconnectOp::verify() {
  if ((*this)->getAttr("serv"
                       "ice") ||
      (*this)->getAttr("serv"
                       "ices"))
    return emitOpError("contains a retired channel-capability attribute");
  auto device = (*this)->getParentOfType<DeviceOp>();
  if (!device)
    return emitOpError("must be nested in fabric.machine");

  SymbolTable symbols(device);
  auto resolveRegion = [&](FlatSymbolRefAttr reference,
                           StringRef attribute) -> FailureOr<RegionOp> {
    auto region =
        dyn_cast_or_null<RegionOp>(symbols.lookup(reference.getValue()));
    if (!region) {
      emitOpError() << attribute << " must resolve to fabric.region in this "
                    << "machine";
      return failure();
    }
    return region;
  };
  auto regionA = resolveRegion(getRegionAAttr(), "region_a");
  auto regionB = resolveRegion(getRegionBAttr(), "region_b");
  if (failed(regionA) || failed(regionB))
    return failure();
  if (getRegionAAttr() == getRegionBAttr())
    return emitOpError("must connect two distinct regions");

  auto verifyPort = [&](RegionOp region, int64_t port,
                        StringRef attribute) -> LogicalResult {
    if (port < 0)
      return emitOpError() << attribute << " must be nonnegative";
    if (auto capacity = region.getBlockCapacity())
      if (port >= *capacity)
        return emitOpError()
               << attribute << " must be less than region @"
               << region.getSymName() << " block_capacity " << *capacity;
    return success();
  };
  if (failed(verifyPort(*regionA, getPortA(), "port_a")) ||
      failed(verifyPort(*regionB, getPortB(), "port_b")))
    return failure();

  bool hasSelectedRefinement = static_cast<bool>(getLogicalChannel());
  bool hasRefinementDetail =
      getCapabilities() || getDirection() || getConcurrency() ||
      getProvider() || getPortAName() || getPortBName() ||
      getPortAConcurrency() || getPortBConcurrency() ||
      getPortACapabilities() || getPortBCapabilities() || getPortAProvider() ||
      getPortBProvider() || getPortAMetadata() || getPortBMetadata() ||
      getMetadata();
  if (!hasSelectedRefinement && hasRefinementDetail)
    return emitOpError(
        "P2 channel refinement attributes require logical_channel");
  if (!hasSelectedRefinement)
    return success(); // Compatibility transport-only declaration.

  auto selectedCapabilities = getCapabilities();
  if (!selectedCapabilities || selectedCapabilities->empty())
    return emitOpError("selected P2 channel requires nonempty capabilities");
  llvm::StringSet<> capabilities;
  for (Attribute value : *selectedCapabilities) {
    auto channelCapability = dyn_cast<qlx::lvm::CapabilityAttr>(value);
    if (!channelCapability)
      return emitOpError(
          "capabilities entries must be #lvm.capability attributes");
    if (!capabilities.insert(channelCapability.getKey()).second)
      return emitOpError("contains duplicate channel capability '")
             << channelCapability.getKey() << "'";
  }
  if (!getDirection() || getDirection()->empty())
    return emitOpError("direction must be nonempty");
  if (!getConcurrency() || *getConcurrency() <= 0)
    return emitOpError("concurrency must be positive");
  if (!getProvider() || getProvider()->empty())
    return emitOpError("provider must be nonempty");
  if (!getPortAName() || getPortAName()->empty() || !getPortBName() ||
      getPortBName()->empty())
    return emitOpError("selected P2 channel requires nonempty port names");
  if (!getPortAConcurrency() || *getPortAConcurrency() <= 0 ||
      !getPortBConcurrency() || *getPortBConcurrency() <= 0)
    return emitOpError("channel-port concurrency must be positive");
  if (*getConcurrency() > *getPortAConcurrency() ||
      *getConcurrency() > *getPortBConcurrency())
    return emitOpError(
        "channel concurrency cannot exceed either channel-port concurrency");
  if (!getPortACapabilities() || getPortACapabilities()->empty() ||
      !getPortBCapabilities() || getPortBCapabilities()->empty())
    return emitOpError(
        "selected P2 channel requires nonempty channel-port capabilities");
  if (!getPortAProvider() || getPortAProvider()->empty() ||
      !getPortBProvider() || getPortBProvider()->empty())
    return emitOpError(
        "selected P2 channel requires nonempty channel-port providers");

  auto collectPortCapabilities =
      [&](ArrayAttr values, StringRef attribute,
          llvm::StringSet<> &result) -> LogicalResult {
    for (Attribute value : values) {
      auto channelCapability = dyn_cast<qlx::lvm::CapabilityAttr>(value);
      if (!channelCapability)
        return emitOpError()
               << attribute << " entries must be #lvm.capability attributes";
      if (!result.insert(channelCapability.getKey()).second)
        return emitOpError()
               << attribute << " contains duplicate channel capability '"
               << channelCapability.getKey() << "'";
    }
    return success();
  };
  llvm::StringSet<> portACapabilities;
  llvm::StringSet<> portBCapabilities;
  if (failed(collectPortCapabilities(
          *getPortACapabilities(), "port_a_capabilities", portACapabilities)) ||
      failed(collectPortCapabilities(*getPortBCapabilities(),
                                     "port_b_capabilities", portBCapabilities)))
    return failure();
  for (auto &entry : capabilities)
    if (!portACapabilities.contains(entry.getKey()) ||
        !portBCapabilities.contains(entry.getKey()))
      return emitOpError("selected channel capability '")
             << entry.getKey()
             << "' must be offered by both persistent channel ports";

  auto logicalRef = *getLogicalChannel();
  auto module = (*this)->getParentOfType<ModuleOp>();
  if (!module || logicalRef.getNestedReferences().size() != 1)
    return emitOpError(
        "logical_channel must be a nested @domain::@channel reference");
  auto logicalDomain = dyn_cast_or_null<qlx::lvm::DomainOp>(
      SymbolTable::lookupSymbolIn(module, logicalRef.getRootReference()));
  auto logicalChannel =
      logicalDomain
          ? dyn_cast_or_null<qlx::lvm::ChannelOp>(SymbolTable::lookupSymbolIn(
                logicalDomain, logicalRef.getLeafReference()))
          : nullptr;
  if (!logicalChannel)
    return emitOpError("logical_channel must resolve to lvm.channel");
  for (Attribute channelCapability : *selectedCapabilities)
    if (!llvm::is_contained(logicalChannel.getCapabilities(),
                            channelCapability))
      return emitOpError("selected channel capability ")
             << channelCapability << " is absent from the P1 logical channel";
  for (ArrayAttr offered : {*getPortACapabilities(), *getPortBCapabilities()})
    for (Attribute channelCapability : offered)
      if (!llvm::is_contained(logicalChannel.getCapabilities(),
                              channelCapability))
        return emitOpError("channel-port capability ")
               << channelCapability << " is absent from the P1 logical channel";

  llvm::StringMap<SmallVector<Attribute, 7>> persistentPorts;
  LogicalResult consistentPorts = success();
  device.walk([&](InterconnectOp candidate) {
    if (failed(consistentPorts) || !candidate.getLogicalChannel())
      return;
    auto record = [&](StringAttr name, FlatSymbolRefAttr region,
                      IntegerAttr slot, ArrayAttr portCapabilities,
                      IntegerAttr concurrency, StringAttr provider,
                      DictionaryAttr metadata) {
      if (!name || !portCapabilities || !concurrency || !provider)
        return;
      SmallVector<Attribute, 7> facts = {
          region, slot, portCapabilities, concurrency, provider, metadata};
      auto [entry, inserted] =
          persistentPorts.try_emplace(name.getValue(), facts);
      if (!inserted && entry->second != facts) {
        emitOpError("persistent channel port '")
            << name.getValue()
            << "' has inconsistent region, slot, capability, concurrency, "
               "provider, or metadata facts";
        consistentPorts = failure();
      }
    };
    record(candidate.getPortANameAttr(), candidate.getRegionAAttr(),
           candidate.getPortAAttr(), candidate.getPortACapabilitiesAttr(),
           candidate.getPortAConcurrencyAttr(),
           candidate.getPortAProviderAttr(), candidate.getPortAMetadataAttr());
    record(candidate.getPortBNameAttr(), candidate.getRegionBAttr(),
           candidate.getPortBAttr(), candidate.getPortBCapabilitiesAttr(),
           candidate.getPortBConcurrencyAttr(),
           candidate.getPortBProviderAttr(), candidate.getPortBMetadataAttr());
  });
  if (failed(consistentPorts))
    return failure();
  StringRef logicalDirection =
      logicalChannel.getDirectionAttr()
          ? logicalChannel.getDirectionAttr().getValue()
          : "forward";
  if (*getDirection() != logicalDirection)
    return emitOpError("direction must match the P1 logical channel");
  if (auto logicalCapacity = logicalChannel.getCapacityAttr())
    if (*getConcurrency() > logicalCapacity.getInt())
      return emitOpError("concurrency exceeds the P1 logical channel capacity");

  bool resourceTransfer = llvm::any_of(*getCapabilities(), [](Attribute value) {
    auto channelCapability = dyn_cast<qlx::lvm::CapabilityAttr>(value);
    return channelCapability &&
           channelCapability.getKey() == "qlx.machine/resource_transfer";
  });
  Operation *protocolTarget = nullptr;
  if (auto protocol = dyn_cast_or_null<FlatSymbolRefAttr>(getProtocolAttr())) {
    protocolTarget = SymbolTable::lookupNearestSymbolFrom(*this, protocol);
    if (!protocolTarget)
      protocolTarget = SymbolTable(module).lookup(protocol.getValue());
    if (!protocolTarget ||
        (protocolTarget->getName().getStringRef() != "fabric.protocol" &&
         protocolTarget->getName().getStringRef() != "fabric.gadget"))
      return emitOpError("protocol must resolve to fabric.protocol or "
                         "fabric.gadget");
  }
  if (resourceTransfer) {
    auto protocol = dyn_cast_or_null<ProtocolOp>(protocolTarget);
    if (!protocol)
      return emitOpError(
          "resource-transfer refinement requires a typed fabric.protocol");
    SmallVector<qlx::lvm::StreamOp, 2> streams;
    for (FlatSymbolRefAttr endpoint :
         {logicalChannel.getFromAttr(), logicalChannel.getToAttr()})
      if (auto stream = dyn_cast_or_null<qlx::lvm::StreamOp>(
              SymbolTable(logicalDomain).lookup(endpoint.getValue())))
        streams.push_back(stream);
    if (streams.empty() ||
        llvm::any_of(streams, [&](qlx::lvm::StreamOp stream) {
          return stream.getProducesAttr() != streams.front().getProducesAttr();
        }))
      return emitOpError(
          "resource-transfer P1 endpoints must determine one resource kind");
    FunctionType signature = protocol.getFunctionType();
    if (signature.getNumInputs() != 1 || signature.getNumResults() != 1)
      return emitOpError(
          "resource-transfer protocol must consume and return one resource");
    auto input = dyn_cast<ResourceStateType>(signature.getInput(0));
    auto output = dyn_cast<ResourceStateType>(signature.getResult(0));
    if (!input || !output ||
        input.getKind() != streams.front().getProducesAttr() ||
        output.getKind() != streams.front().getProducesAttr())
      return emitOpError(
          "resource-transfer protocol resource kind must match the P1 stream");
  }

  // A selected P2 channel is a refinement, not an unrelated transport edge.
  // Close that claim against the retained qlx.device and logical_to_qec map.
  Operation *matchedDevice = nullptr;
  for (Operation &candidate : module.getBody()->getOperations()) {
    if (candidate.getName().getStringRef() != "qlx.device")
      continue;
    auto logical = candidate.getAttrOfType<FlatSymbolRefAttr>("logical");
    auto qec = candidate.getAttrOfType<FlatSymbolRefAttr>("qec");
    if (!logical || !qec ||
        logical.getValue() != logicalRef.getRootReference().getValue() ||
        qec.getValue() != device.getSymName())
      continue;
    if (matchedDevice)
      return emitOpError(
          "selected P2 channel has multiple matching qlx.device manifests");
    matchedDevice = &candidate;
  }
  if (!matchedDevice)
    return emitOpError(
        "selected P2 channel requires a matching retained qlx.device");

  auto bindingRef =
      matchedDevice->getAttrOfType<FlatSymbolRefAttr>("logical_to_qec");
  Operation *binding =
      bindingRef
          ? SymbolTable::lookupNearestSymbolFrom(matchedDevice, bindingRef)
          : nullptr;
  auto entries =
      binding ? binding->getAttrOfType<ArrayAttr>("entries") : ArrayAttr{};
  if (!binding || binding->getName().getStringRef() != "qlx.logical_to_qec" ||
      !entries)
    return emitOpError(
        "selected P2 channel requires a retained logical_to_qec map");

  auto endpointSpace = [&](FlatSymbolRefAttr endpoint,
                           StringRef label) -> FailureOr<std::string> {
    Operation *target = SymbolTable(logicalDomain).lookup(endpoint.getValue());
    if (!target) {
      emitOpError() << label << " endpoint does not resolve in the P1 domain";
      return failure();
    }
    if (target->getName().getStringRef() == "lvm.space")
      return endpoint.getValue().str();
    if (target->getName().getStringRef() != "lvm.stream") {
      emitOpError() << label << " endpoint must resolve to lvm.space or "
                    << "lvm.stream";
      return failure();
    }

    std::optional<std::string> owner;
    for (Operation &candidate : logicalDomain.getBody().front()) {
      auto supply = dyn_cast<qlx::lvm::ChannelOp>(candidate);
      if (!supply || supply.getToAttr() != endpoint)
        continue;
      bool transfersResource =
          llvm::any_of(supply.getCapabilities(), [](Attribute value) {
            auto channelCapability = dyn_cast<qlx::lvm::CapabilityAttr>(value);
            return channelCapability && channelCapability.getKey() ==
                                            "qlx.machine/resource_transfer";
          });
      Operation *source =
          SymbolTable(logicalDomain).lookup(supply.getFromAttr().getValue());
      if (!transfersResource || !source ||
          source->getName().getStringRef() != "lvm.space")
        continue;
      if (owner) {
        emitOpError() << label
                      << " stream endpoint has multiple owning P1 spaces";
        return failure();
      }
      owner = supply.getFromAttr().getValue().str();
    }
    if (!owner) {
      emitOpError() << label
                    << " stream endpoint has no unique owning P1 space";
      return failure();
    }
    return *owner;
  };

  auto logicalA = endpointSpace(logicalChannel.getFromAttr(), "source");
  auto logicalB = endpointSpace(logicalChannel.getToAttr(), "destination");
  if (failed(logicalA) || failed(logicalB))
    return failure();
  auto mappedRegion = [&](StringRef logical) -> std::optional<StringRef> {
    for (Attribute raw : entries) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto logicalName = entry ? entry.getAs<StringAttr>("logical") : nullptr;
      auto qecName = entry ? entry.getAs<StringAttr>("qec") : nullptr;
      if (logicalName && qecName && logicalName.getValue() == logical)
        return qecName.getValue();
    }
    return std::nullopt;
  };
  auto expectedA = mappedRegion(*logicalA);
  auto expectedB = mappedRegion(*logicalB);
  if (!expectedA || !expectedB)
    return emitOpError(
        "selected P2 channel endpoints are absent from logical_to_qec");
  if (*expectedA != getRegionAAttr().getValue() ||
      *expectedB != getRegionBAttr().getValue())
    return emitOpError(
        "P2 interconnect regions must refine the P1 channel endpoints");
  return success();
}

LogicalResult EncodingHierarchyOp::verify() {
  if (getMultiplicity() <= 0)
    return emitOpError("multiplicity must be positive");
  if (getDepth() <= 0)
    return emitOpError("depth must be positive");

  auto parsePort = [&](StringRef value, int64_t &child,
                       int64_t &port) -> LogicalResult {
    SmallVector<StringRef, 2> pieces;
    value.split(pieces, ':');
    if (pieces.size() != 2 || pieces[0].getAsInteger(10, child) ||
        pieces[1].getAsInteger(10, port) || child < 0 || port < 0)
      return emitOpError(
                 "port entries must be nonnegative 'child:port' strings; got '")
             << value << "'";
    return success();
  };

  llvm::SmallDenseSet<int64_t, 16> outerCarriers;
  llvm::StringSet<> mappedPorts;
  int64_t maxChild = -1;
  for (Attribute value : getCarrierMap()) {
    auto entry = dyn_cast<StringAttr>(value);
    if (!entry)
      return emitOpError("carrier_map entries must be strings");
    SmallVector<StringRef, 3> pieces;
    entry.getValue().split(pieces, ':');
    int64_t outer = -1, child = -1, port = -1;
    if (pieces.size() != 3 || pieces[0].getAsInteger(10, outer) ||
        pieces[1].getAsInteger(10, child) || pieces[2].getAsInteger(10, port) ||
        outer < 0 || child < 0 || port < 0)
      return emitOpError("carrier_map entries must be nonnegative "
                         "'outer:child:port' strings; got '")
             << entry.getValue() << "'";
    if (!outerCarriers.insert(outer).second)
      return emitOpError("carrier_map repeats outer carrier ") << outer;
    std::string key = (Twine(child) + ":" + Twine(port)).str();
    if (!mappedPorts.insert(key).second)
      return emitOpError("carrier_map repeats child logical port ") << key;
    maxChild = std::max(maxChild, child);
  }
  if (maxChild >= getMultiplicity())
    return emitOpError("carrier_map child index exceeds multiplicity");

  llvm::StringSet<> dispositions;
  auto addPorts = [&](std::optional<ArrayAttr> values,
                      StringRef label) -> LogicalResult {
    if (!values)
      return success();
    for (Attribute value : *values) {
      auto entry = dyn_cast<StringAttr>(value);
      int64_t child = -1, port = -1;
      if (!entry || failed(parsePort(entry.getValue(), child, port)))
        return failure();
      std::string key = (Twine(child) + ":" + Twine(port)).str();
      if (mappedPorts.contains(key))
        return emitOpError(label)
               << " port " << key << " is already mapped by carrier_map";
      if (!dispositions.insert(key).second)
        return emitOpError("child logical port ")
               << key << " has more than one unmapped disposition";
      if (child >= getMultiplicity())
        return emitOpError(label) << " child index exceeds multiplicity";
    }
    return success();
  };
  if (failed(addPorts(getExposedPorts(), "exposed")) ||
      failed(addPorts(getGaugePorts(), "gauge")))
    return failure();

  int64_t fixedCount = 0;
  if (auto fixed = getFixedPorts()) {
    for (Attribute value : *fixed) {
      auto entry = dyn_cast<DictionaryAttr>(value);
      auto child = entry ? entry.getAs<IntegerAttr>("child") : IntegerAttr{};
      auto port = entry ? entry.getAs<IntegerAttr>("port") : IntegerAttr{};
      auto basis = entry ? entry.getAs<StringAttr>("basis") : StringAttr{};
      auto eigenvalue =
          entry ? entry.getAs<IntegerAttr>("eigenvalue") : IntegerAttr{};
      auto evidence =
          entry ? entry.getAs<StringAttr>("evidence") : StringAttr{};
      if (!child || !port || !basis || !eigenvalue || !evidence ||
          child.getInt() < 0 || child.getInt() >= getMultiplicity() ||
          port.getInt() < 0 ||
          (basis.getValue() != "x" && basis.getValue() != "z") ||
          (eigenvalue.getInt() != -1 && eigenvalue.getInt() != 1) ||
          evidence.getValue().empty())
        return emitOpError(
            "fixed_ports entries require valid child, port, x/z basis, +/-1 "
            "eigenvalue, and nonempty evidence");
      std::string key =
          (Twine(child.getInt()) + ":" + Twine(port.getInt())).str();
      if (mappedPorts.contains(key) || !dispositions.insert(key).second)
        return emitOpError("fixed child logical port ")
               << key << " is mapped or has another disposition";
      ++fixedCount;
    }
  }

  bool invalidReference = false;
  auto resolveEncoding = [&](FlatSymbolRefAttr reference,
                             StringRef label) -> EncodingOp {
    auto *target = SymbolTable::lookupNearestSymbolFrom(*this, reference);
    if (!target)
      return {};
    auto encoding = dyn_cast<EncodingOp>(target);
    if (!encoding) {
      emitOpError() << label << " must resolve to fabric.encoding";
      invalidReference = true;
    }
    return encoding;
  };
  auto outer = resolveEncoding(getOuterAttr(), "outer");
  auto child = resolveEncoding(getChildAttr(), "child");
  auto flat = resolveEncoding(getFlatEncodingAttr(), "flat_encoding");
  if (invalidReference)
    return failure();
  if (!outer || !child || !flat)
    return success(); // Partial linked modules are checked after linking.
  auto *outerTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, outer.getCodeAttr());
  auto *childTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, child.getCodeAttr());
  auto *codeTarget = SymbolTable::lookupNearestSymbolFrom(*this, getCodeAttr());
  auto outerCode = dyn_cast_or_null<CodeOp>(outerTarget);
  auto childCode = dyn_cast_or_null<CodeOp>(childTarget);
  auto code = dyn_cast_or_null<CodeOp>(codeTarget);
  if (!outerCode || !childCode || !code)
    return emitOpError(
        "code, outer code, and child code must resolve to fabric.code");
  if (flat.getCodeAttr() != getCodeAttr())
    return emitOpError("flat_encoding must reference the hierarchy code");

  int64_t outerN = outerCode.getN().value_or(0);
  int64_t childN = childCode.getN().value_or(0);
  int64_t childK = childCode.getK().value_or(1);
  int64_t expectedPorts = getMultiplicity() * childK;
  if (static_cast<int64_t>(outerCarriers.size()) != outerN)
    return emitOpError(
        "carrier_map must map every outer code carrier exactly once");
  for (int64_t index = 0; index < outerN; ++index)
    if (!outerCarriers.contains(index))
      return emitOpError("carrier_map outer indices must be dense from zero");
  for (auto &entry : mappedPorts) {
    int64_t c = -1, p = -1;
    if (failed(parsePort(entry.getKey(), c, p)) || p >= childK)
      return emitOpError("carrier_map logical port exceeds child k");
  }
  for (auto &entry : dispositions) {
    int64_t c = -1, p = -1;
    if (failed(parsePort(entry.getKey(), c, p)) || p >= childK)
      return emitOpError("unmapped logical port exceeds child k");
  }
  if (static_cast<int64_t>(mappedPorts.size() + dispositions.size()) !=
      expectedPorts)
    return emitOpError(
        "every child logical port must be mapped, exposed, gauged, or fixed");
  if (code.getN().value_or(0) != getMultiplicity() * childN)
    return emitOpError("composite n must equal multiplicity times child n");
  int64_t exposedCount = getExposedPorts() ? getExposedPorts()->size() : 0;
  int64_t gaugeCount = getGaugePorts() ? getGaugePorts()->size() : 0;
  if (code.getK().value_or(1) != outerCode.getK().value_or(1) + exposedCount)
    return emitOpError("composite k must equal outer k plus exposed ports");
  if (code.getR().value_or(0) !=
      outerCode.getR().value_or(0) +
          getMultiplicity() * childCode.getR().value_or(0) + gaugeCount)
    return emitOpError(
        "composite r must include outer, child, and reclassified gauges");
  (void)fixedCount;
  return success();
}

LogicalResult EncodingOp::verify() {
  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getProfileAttr());
  if (!target)
    return success();
  auto profile = dyn_cast<CodeProfileOp>(target);
  if (!profile)
    return emitOpError("profile must resolve to fabric.code_profile");
  if (profile.getCodeAttr() != getCodeAttr())
    return emitOpError("profile and encoding must reference the same code");
  if (static_cast<bool>(getEpochSchema()) !=
      static_cast<bool>(getInitialEpoch()))
    return emitOpError(
        "epoch_schema and initial_epoch must be provided together");
  if (getEpochSchema()) {
    auto schema = SymbolTable::lookupNearestSymbolFrom(
        *this, FlatSymbolRefAttr::get(getContext(), *getEpochSchema()));
    if (schema && !isa<EncodingEpochSchemaOp>(schema))
      return emitOpError("epoch_schema must resolve to "
                         "fabric.encoding_epoch_schema");
    auto epoch = SymbolTable::lookupNearestSymbolFrom(
        *this, FlatSymbolRefAttr::get(getContext(), *getInitialEpoch()));
    if (epoch) {
      auto instance = dyn_cast<EncodingEpochOp>(epoch);
      if (!instance)
        return emitOpError(
            "initial_epoch must resolve to fabric.encoding_epoch");
      if (instance.getEncodingAttr().getValue() != getSymName())
        return emitOpError("initial_epoch must belong to this encoding");
      if (instance.getSchemaAttr().getValue() != *getEpochSchema())
        return emitOpError(
            "initial_epoch must use this encoding's epoch_schema");
    }
  }

  if (auto dynamicPhases = profile.getDynamicPhases()) {
    if (!getEpochSchema())
      return emitOpError(
          "dynamic code profile requires a derived encoding epoch schema");
    auto *schemaTarget = SymbolTable::lookupNearestSymbolFrom(
        *this, FlatSymbolRefAttr::get(getContext(), *getEpochSchema()));
    auto schema = dyn_cast_or_null<EncodingEpochSchemaOp>(schemaTarget);
    if (!schema)
      return emitOpError("dynamic code profile epoch_schema must resolve for "
                         "exact verification");

    SmallVector<Attribute> expectedPhases;
    SmallVector<Attribute> expectedTransitions;
    SmallVector<NamedAttribute> expectedLogicalMaps;
    for (Attribute raw : *dynamicPhases) {
      auto phase = dyn_cast<DictionaryAttr>(raw);
      auto name = phase ? phase.getAs<StringAttr>("name") : StringAttr{};
      auto input =
          phase ? phase.getAs<StringAttr>("input_epoch") : StringAttr{};
      auto output =
          phase ? phase.getAs<StringAttr>("output_epoch") : StringAttr{};
      auto logicalMap =
          phase ? phase.getAs<DictionaryAttr>("logical_map") : DictionaryAttr{};
      if (!name || !input || !output || !logicalMap)
        return emitOpError("dynamic profile phases must expose names, "
                           "transitions, and logical maps");
      expectedPhases.push_back(name);
      std::string edge = (input.getValue() + "->" + output.getValue()).str();
      expectedTransitions.push_back(StringAttr::get(getContext(), edge));
      expectedLogicalMaps.emplace_back(StringAttr::get(getContext(), edge),
                                       logicalMap);
    }
    auto expectedPhaseArray = ArrayAttr::get(getContext(), expectedPhases);
    auto expectedTransitionArray =
        ArrayAttr::get(getContext(), expectedTransitions);
    auto expectedMap = DictionaryAttr::get(getContext(), expectedLogicalMaps);
    if (schema.getPhases() != expectedPhaseArray)
      return emitOpError(
          "epoch_schema phases contradict the dynamic code profile");
    if (expectedPhases.empty() ||
        schema.getInitialAttr() != expectedPhases.front())
      return emitOpError(
          "epoch_schema initial phase contradicts the dynamic code profile");
    if (schema.getTransitions() != expectedTransitionArray)
      return emitOpError(
          "epoch_schema transitions contradict the dynamic code profile");
    auto logicalMaps = schema.getLogicalMaps();
    if (!logicalMaps || *logicalMaps != expectedMap)
      return emitOpError(
          "epoch_schema logical_maps contradict the dynamic code profile");
    if (!schema.getPeriodic())
      return emitOpError(
          "dynamic code profile requires a periodic epoch_schema");
    if (!profile.getPeriodClosureAttr() ||
        schema.getClosureAttr() != profile.getPeriodClosureAttr())
      return emitOpError(
          "epoch_schema closure contradicts the dynamic code profile");
  }
  return success();
}

static FailureOr<StringRef> patchRecordId(Operation *owner, Attribute value,
                                          StringRef label) {
  auto record = dyn_cast<DictionaryAttr>(value);
  if (!record)
    return owner->emitOpError() << label << " entries must be dictionaries";
  auto id = record.getAs<StringAttr>("id");
  if (!id || id.getValue().empty())
    return owner->emitOpError()
           << label << " entries require a nonempty string id";
  return id.getValue();
}

LogicalResult PatchGraphOp::verify() {
  llvm::StringSet<> ids;
  for (Attribute value : getNodes()) {
    auto id = patchRecordId(*this, value, "node");
    if (failed(id) || !ids.insert(*id).second)
      return failed(id) ? failure() : emitOpError("contains duplicate node id");
  }
  llvm::StringSet<> interactionIds;
  for (Attribute value : getInteractions()) {
    auto id = patchRecordId(*this, value, "interaction");
    if (failed(id))
      return failure();
    if (!interactionIds.insert(*id).second)
      return emitOpError("contains duplicate interaction id");
    auto record = cast<DictionaryAttr>(value);
    auto action = record.getAs<StringAttr>("action");
    auto patches = record.getAs<ArrayAttr>("patches");
    if (!action || action.getValue().empty() || !patches || patches.size() < 2)
      return emitOpError(
          "interactions require a nonempty action and at least two patches");
    for (Attribute endpoint : patches) {
      auto patch = dyn_cast<StringAttr>(endpoint);
      if (!patch || !ids.contains(patch.getValue()))
        return emitOpError(
            "interaction patches must name nodes in the patch graph");
    }
  }
  return success();
}

LogicalResult PatchMappingOp::verify() {
  Operation *graphTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getGraphAttr());
  auto graph = dyn_cast_or_null<PatchGraphOp>(graphTarget);
  if (!graph)
    return emitOpError("graph must resolve to fabric.patch_graph");
  llvm::StringSet<> graphPatches;
  for (Attribute value : graph.getNodes()) {
    auto id = patchRecordId(graph, value, "node");
    if (failed(id))
      return failure();
    graphPatches.insert(*id);
  }
  llvm::StringSet<> ids;
  llvm::StringSet<> mappedPatches;
  llvm::DenseSet<std::pair<Attribute, int64_t>> mappedSlots;
  for (Attribute value : getAssignments()) {
    auto id = patchRecordId(*this, value, "assignment");
    if (failed(id) || !ids.insert(*id).second)
      return failed(id) ? failure()
                        : emitOpError("contains duplicate assignment id");
    auto record = cast<DictionaryAttr>(value);
    auto patch = record.getAs<StringAttr>("patch");
    auto slot = record.getAs<IntegerAttr>("slot");
    auto topologyRef = record.getAs<SymbolRefAttr>("topology");
    if (!patch || patch.getValue().empty() || !slot || !topologyRef)
      return emitOpError(
          "assignments require patch, integer slot, and topology reference");
    Operation *topology =
        SymbolTable::lookupNearestSymbolFrom(*this, topologyRef);
    if (!topology ||
        topology->getName().getStringRef() != "phys.patch_topology")
      return emitOpError(
          "assignment topology must resolve to phys.patch_topology");
    auto capacity = topology->getAttrOfType<IntegerAttr>("capacity");
    if (!capacity || capacity.getInt() < 0)
      return emitOpError("patch topology requires a nonnegative capacity");
    if (!graphPatches.contains(patch.getValue()))
      return emitOpError("assignment patch '")
             << patch.getValue() << "' is absent from the patch graph";
    if (!mappedPatches.insert(patch.getValue()).second)
      return emitOpError("contains duplicate mapping for patch '")
             << patch.getValue() << "'";
    if (slot.getInt() < 0 || slot.getInt() >= capacity.getInt())
      return emitOpError("assignment slot ")
             << slot.getInt() << " is absent from the patch topology";
    if (!mappedSlots.insert({topologyRef, slot.getInt()}).second)
      return emitOpError("contains duplicate assignment to slot ")
             << slot.getInt() << " in " << topologyRef;
  }
  if (mappedPatches.size() != graphPatches.size())
    return emitOpError("must assign every patch graph node exactly once");
  return success();
}

static LogicalResult verifySymplecticClosure(Operation *op, Attribute value,
                                             StringRef label) {
  auto matrix = dyn_cast_or_null<DenseIntElementsAttr>(value);
  if (!matrix || matrix.getType().getRank() != 2 ||
      !matrix.getType().getElementType().isInteger(1))
    return op->emitOpError() << label << " must be a rank-2 i1 tensor";
  auto shape = matrix.getType().getShape();
  if (shape[0] <= 0 || shape[0] != shape[1] || shape[0] % 2)
    return op->emitOpError()
           << label << " must be a nonempty even-dimensional square matrix";

  SmallVector<llvm::SmallBitVector> rows;
  auto values = matrix.getValues<APInt>();
  auto iterator = values.begin();
  for (int64_t row = 0; row < shape[0]; ++row) {
    llvm::SmallBitVector bits(shape[1]);
    for (int64_t column = 0; column < shape[1]; ++column, ++iterator)
      if (!(*iterator).isZero())
        bits.set(column);
    rows.push_back(std::move(bits));
  }

  SmallVector<llvm::SmallBitVector> reduced(rows.begin(), rows.end());
  int64_t rank = 0;
  for (int64_t column = shape[1] - 1; column >= 0; --column) {
    auto pivot =
        llvm::find_if(llvm::drop_begin(reduced, rank),
                      [&](const auto &row) { return row.test(column); });
    if (pivot == reduced.end())
      continue;
    std::iter_swap(reduced.begin() + rank, pivot);
    for (int64_t index = 0; index < shape[0]; ++index)
      if (index != rank && reduced[index].test(column))
        reduced[index] ^= reduced[rank];
    if (++rank == shape[0])
      break;
  }
  if (rank != shape[0])
    return op->emitOpError() << label << " must be full rank";

  int64_t half = shape[0] / 2;
  for (int64_t left = 0; left < shape[0]; ++left) {
    for (int64_t right = 0; right < shape[0]; ++right) {
      bool product = false;
      for (int64_t index = 0; index < half; ++index)
        product ^= (rows[left].test(index) && rows[right].test(half + index)) ^
                   (rows[left].test(half + index) && rows[right].test(index));
      bool expected = (left < half && right == left + half) ||
                      (right < half && left == right + half);
      if (product != expected)
        return op->emitOpError()
               << label << " must preserve the canonical symplectic form";
    }
  }
  return success();
}

LogicalResult EncodingEpochSchemaOp::verify() {
  llvm::SmallDenseSet<StringRef, 8> phases;
  for (Attribute value : getPhases()) {
    auto phase = dyn_cast<StringAttr>(value);
    if (!phase || phase.getValue().empty())
      return emitOpError("phases must be nonempty strings");
    if (!phases.insert(phase.getValue()).second)
      return emitOpError("contains duplicate phase '")
             << phase.getValue() << "'";
  }
  if (phases.empty() || !phases.contains(getInitial()))
    return emitOpError("initial must name a declared phase");
  for (Attribute value : getTransitions()) {
    auto transition = dyn_cast<StringAttr>(value);
    if (!transition)
      return emitOpError("transitions must be 'source->target' strings");
    auto pair = transition.getValue().split("->");
    if (pair.first.empty() || pair.second.empty() ||
        !phases.contains(pair.first) || !phases.contains(pair.second))
      return emitOpError("transition '")
             << transition.getValue() << "' must connect declared phases";
  }
  if (getPeriodic()) {
    if (getTransitions().empty())
      return emitOpError(
          "periodic schema requires explicit closure transitions");
    if (!getClosureAttr())
      return emitOpError("periodic schema requires a closure matrix");
    if (failed(verifySymplecticClosure(getOperation(), getClosureAttr(),
                                       "closure")))
      return failure();
  } else if (getClosureAttr()) {
    return emitOpError("closure requires a periodic schema");
  }
  return success();
}

LogicalResult EncodingEpochOp::verify() {
  Operation *schemaTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getSchemaAttr());
  if (schemaTarget) {
    auto schema = dyn_cast<EncodingEpochSchemaOp>(schemaTarget);
    if (!schema)
      return emitOpError("schema must resolve to fabric.encoding_epoch_schema");
    bool found = llvm::any_of(schema.getPhases(), [&](Attribute value) {
      auto phase = dyn_cast<StringAttr>(value);
      return phase && phase.getValue() == getPhase();
    });
    if (!found)
      return emitOpError("phase must belong to the referenced epoch schema");
  }
  if (getIndex() < 0)
    return emitOpError("index must be nonnegative");
  Operation *encodingTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getEncodingAttr());
  if (encodingTarget) {
    auto encoding = dyn_cast<EncodingOp>(encodingTarget);
    if (!encoding)
      return emitOpError("encoding must resolve to fabric.encoding");
    if (encoding.getEpochSchema() &&
        *encoding.getEpochSchema() != getSchemaAttr().getValue())
      return emitOpError("schema must match the encoding's epoch_schema");
  }
  return success();
}

static std::optional<int64_t> frameSize(DictionaryAttr partitions) {
  int64_t total = 0;
  for (NamedAttribute value : partitions) {
    auto count = dyn_cast<IntegerAttr>(value.getValue());
    if (!count || count.getInt() < 0)
      return std::nullopt;
    total += count.getInt();
  }
  return total;
}

static LogicalResult verifySupport(Operation *op, ArrayRef<int64_t> support,
                                   int64_t width, int64_t expected,
                                   StringRef label) {
  if (static_cast<int64_t>(support.size()) != expected)
    return op->emitOpError(label)
           << " width must equal the referenced code n (" << expected << ")";
  llvm::SmallDenseSet<int64_t, 32> seen;
  for (int64_t index : support) {
    if (index < 0 || index >= width)
      return op->emitOpError(label)
             << " index " << index << " is outside the carrier frame";
    if (!seen.insert(index).second)
      return op->emitOpError(label) << " contains duplicate index " << index;
  }
  return success();
}

static LogicalResult verifyCarrierRoles(Operation *op, DictionaryAttr roles,
                                        ArrayRef<int64_t> support,
                                        int64_t frameWidth, StringRef label) {
  constexpr std::array<StringLiteral, 5> roleNames = {
      "active", "measured", "reset", "scratch", "dormant"};
  llvm::SmallDenseSet<StringRef, 8> allowed(roleNames.begin(), roleNames.end());
  llvm::SmallDenseMap<int64_t, StringRef, 32> assignments;
  for (NamedAttribute entry : roles) {
    if (!allowed.contains(entry.getName().getValue()))
      return op->emitOpError(label)
             << " contains unknown role '" << entry.getName().getValue() << "'";
  }
  for (StringLiteral name : roleNames) {
    auto values = roles.getAs<DenseI64ArrayAttr>(name);
    if (!values)
      return op->emitOpError(label) << " requires role '" << name << "'";
    llvm::SmallDenseSet<int64_t, 32> local;
    for (int64_t index : values.asArrayRef()) {
      if (index < 0 || index >= frameWidth)
        return op->emitOpError(label)
               << " role '" << name << "' index " << index
               << " is outside the carrier frame";
      if (!local.insert(index).second)
        return op->emitOpError(label)
               << " role '" << name << "' contains duplicate index " << index;
      if (auto existing = assignments.find(index);
          existing != assignments.end())
        return op->emitOpError(label)
               << " assigns carrier " << index << " to both '"
               << existing->second << "' and '" << name << "'";
      assignments[index] = name;
    }
  }
  auto active = roles.getAs<DenseI64ArrayAttr>("active");
  llvm::SmallDenseSet<int64_t, 32> activeSet(active.asArrayRef().begin(),
                                             active.asArrayRef().end());
  llvm::SmallDenseSet<int64_t, 32> supportSet(support.begin(), support.end());
  if (activeSet.size() != supportSet.size() ||
      llvm::any_of(supportSet,
                   [&](int64_t index) { return !activeSet.contains(index); }))
    return op->emitOpError(label)
           << " active role must equal the boundary support";
  return success();
}

LogicalResult PatchTransformOp::verify() {
  auto size = frameSize(getFramePartitions());
  if (!size || *size <= 0)
    return emitOpError("frame_partitions must define a nonempty carrier frame");
  auto resolveEncoding = [&](FlatSymbolRefAttr reference,
                             StringRef label) -> FailureOr<EncodingOp> {
    Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, reference);
    if (!target)
      return failure();
    auto encoding = dyn_cast<EncodingOp>(target);
    if (!encoding) {
      emitOpError(label) << " must resolve to fabric.encoding";
      return failure();
    }
    return encoding;
  };
  auto source = resolveEncoding(getSourceAttr(), "source");
  auto destination = resolveEncoding(getDestinationAttr(), "destination");
  if (failed(source) || failed(destination))
    return failure();
  auto codeN = [&](EncodingOp encoding) -> FailureOr<int64_t> {
    Operation *target =
        SymbolTable::lookupNearestSymbolFrom(*this, encoding.getCodeAttr());
    auto code = dyn_cast_or_null<CodeOp>(target);
    if (!code) {
      emitOpError("transform encoding code must resolve to fabric.code");
      return failure();
    }
    return code.getN().value_or(0);
  };
  auto sourceN = codeN(*source);
  auto destinationN = codeN(*destination);
  if (failed(sourceN) || failed(destinationN))
    return failure();
  if (failed(verifySupport(getOperation(), getSourceSupport(), *size, *sourceN,
                           "source_support")) ||
      failed(verifySupport(getOperation(), getDestinationSupport(), *size,
                           *destinationN, "destination_support")))
    return failure();
  if (failed(verifyCarrierRoles(getOperation(), getSourceRoles(),
                                getSourceSupport(), *size, "source_roles")) ||
      failed(verifyCarrierRoles(getOperation(), getDestinationRoles(),
                                getDestinationSupport(), *size,
                                "destination_roles")))
    return failure();
  // EncodingOps point at CodeOps; read the protected logical dimensions from
  // those code declarations rather than treating the encoding as algebra.
  auto *sourceCodeTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, (*source).getCodeAttr());
  auto *destinationCodeTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, (*destination).getCodeAttr());
  auto sourceCode = dyn_cast_or_null<CodeOp>(sourceCodeTarget);
  auto destinationCode = dyn_cast_or_null<CodeOp>(destinationCodeTarget);
  int64_t sourceK = sourceCode ? sourceCode.getK().value_or(1) : 1;
  int64_t destinationK =
      destinationCode ? destinationCode.getK().value_or(1) : 1;
  if (static_cast<int64_t>(getLogicalMap().size()) != sourceK)
    return emitOpError("logical_map width must equal source code k (")
           << sourceK << ")";
  llvm::SmallDenseSet<int64_t, 16> logicalTargets;
  for (int64_t index : getLogicalMap()) {
    if (index < 0 || index >= destinationK)
      return emitOpError("logical_map index ")
             << index << " is outside destination logical ports";
    if (!logicalTargets.insert(index).second)
      return emitOpError("logical_map contains duplicate destination index ")
             << index;
  }
  if (getEvidence().empty())
    return emitOpError("requires nonempty transform evidence");
  return success();
}

LogicalResult TransformBeginOp::verify() {
  auto frameType = cast<PatchFrameType>(getFrame().getType());
  if (frameType.getTransform() != getTransformAttr())
    return emitOpError("frame type must reference transform");
  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getTransformAttr());
  if (!target)
    return success();
  auto transform = dyn_cast<PatchTransformOp>(target);
  if (!transform)
    return emitOpError("transform must resolve to fabric.patch_transform");
  Operation *sourceTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, transform.getSourceAttr());
  auto encoding = dyn_cast_or_null<EncodingOp>(sourceTarget);
  auto patch = cast<PatchType>(getSource().getType());
  if (encoding && patch.getEncoding() != transform.getSourceAttr())
    return emitOpError("source patch encoding must match transform source");
  return success();
}

LogicalResult TransformEndOp::verify() {
  auto frameType = cast<PatchFrameType>(getFrame().getType());
  if (frameType.getTransform() != getTransformAttr())
    return emitOpError("frame type must reference transform");
  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getTransformAttr());
  if (!target)
    return success();
  auto transform = dyn_cast<PatchTransformOp>(target);
  if (!transform)
    return emitOpError("transform must resolve to fabric.patch_transform");
  auto patch = cast<PatchType>(getDestination().getType());
  if (patch.getEncoding() != transform.getDestinationAttr())
    return emitOpError(
        "destination patch encoding must match transform destination");
  return success();
}

LogicalResult EpochTransitionOp::verify() {
  auto sourceType = cast<PatchType>(getPatch().getType());
  auto targetType = cast<PatchType>(getResult().getType());
  if (sourceType.getCodeType() != targetType.getCodeType() ||
      sourceType.getEncoding() != targetType.getEncoding())
    return emitOpError("must preserve code and encoding identity");
  if (!sourceType.getEpoch() || !targetType.getEpoch())
    return emitOpError("requires explicit source and destination epochs");
  if (targetType.getEpoch() != getToEpochAttr())
    return emitOpError("result epoch must equal to_epoch");
  if (getEvidence().empty())
    return emitOpError("requires nonempty transition evidence");
  SmallVector<Type, 2> types{sourceType, targetType};
  if (failed(verifyEncodingQualifiedTypes(*this, types)))
    return failure();

  Operation *sourceTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, sourceType.getEpoch());
  Operation *destinationTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, targetType.getEpoch());
  if (!sourceTarget || !destinationTarget)
    return success();
  auto source = dyn_cast<EncodingEpochOp>(sourceTarget);
  auto destination = dyn_cast<EncodingEpochOp>(destinationTarget);
  if (!source || !destination)
    return emitOpError(
        "epoch references must resolve to fabric.encoding_epoch");
  if (source.getSchemaAttr() != destination.getSchemaAttr())
    return emitOpError("source and destination epochs must share one schema");
  Operation *schemaTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, source.getSchemaAttr());
  auto schema = dyn_cast_or_null<EncodingEpochSchemaOp>(schemaTarget);
  if (!schema)
    return success();
  std::string edge = (source.getPhase() + "->" + destination.getPhase()).str();
  bool allowed = llvm::any_of(schema.getTransitions(), [&](Attribute value) {
    auto transition = dyn_cast<StringAttr>(value);
    return transition && transition.getValue() == edge;
  });
  if (!allowed)
    return emitOpError("transition '")
           << edge << "' is not declared by epoch schema @"
           << schema.getSymName();

  if (sourceType.getEncoding()) {
    Operation *encodingTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, sourceType.getEncoding());
    auto encoding = dyn_cast_or_null<EncodingOp>(encodingTarget);
    Operation *profileTarget = encoding ? SymbolTable::lookupNearestSymbolFrom(
                                              *this, encoding.getProfileAttr())
                                        : nullptr;
    auto profile = dyn_cast_or_null<CodeProfileOp>(profileTarget);
    if (profile && profile.getDynamicPhases()) {
      auto logicalMaps = schema.getLogicalMaps();
      auto expected =
          logicalMaps ? dyn_cast_or_null<DictionaryAttr>(logicalMaps->get(edge))
                      : DictionaryAttr{};
      if (!expected)
        return emitOpError("dynamic epoch schema has no logical map for edge '")
               << edge << "'";
      auto actual = getLogicalMap();
      if ((!actual && !expected.empty()) || (actual && *actual != expected))
        return emitOpError("logical_map contradicts dynamic profile edge '")
               << edge << "'";
    }
  }
  return success();
}

LogicalResult CircuitOp::verify() {
  FunctionType functionType = getFunctionType();
  if (failed(verifyEncodingQualifiedTypes(*this, functionType.getInputs())) ||
      failed(verifyEncodingQualifiedTypes(*this, functionType.getResults())))
    return failure();
  if (!llvm::hasSingleElement(getBody()))
    return emitOpError("requires exactly one entry block");
  Block &block = getBody().front();
  if (block.getArgumentTypes() != functionType.getInputs())
    return emitOpError("entry block arguments must match function_type inputs");
  auto returnOp = dyn_cast<ReturnOp>(block.getTerminator());
  if (!returnOp)
    return emitOpError("must terminate with fabric.return");
  if (returnOp.getOperandTypes() != functionType.getResults())
    return returnOp.emitOpError(
        "operand types must match enclosing circuit result types");
  return success();
}

static bool containsString(ArrayAttr values, StringRef expected) {
  return values && llvm::any_of(values, [&](Attribute value) {
           auto string = dyn_cast<StringAttr>(value);
           return string && string.getValue() == expected;
         });
}

static LogicalResult verifyGeneratedRemoteProvenance(Operation *owner) {
  auto generatedBy = owner->getAttrOfType<FlatSymbolRefAttr>("generated_by");
  auto actionSite = owner->getAttrOfType<SymbolRefAttr>("action_site");
  if (!generatedBy)
    return success();

  Operation *lowering =
      SymbolTable::lookupNearestSymbolFrom(owner, generatedBy);
  if (!lowering || lowering->getName().getStringRef() != "qlx.qec_lowering")
    return success();
  auto family = lowering->getAttrOfType<StringAttr>("objective_family");
  bool remoteObservable =
      containsString(lowering->getAttrOfType<ArrayAttr>("requirements"),
                     "qlx.machine/observable_remote");
  if (!remoteObservable)
    return success(); // Noncommunication generator families own other schemas.

  Operation *site = nullptr;
  if (actionSite) {
    site = SymbolTable::lookupNearestSymbolFrom(owner, actionSite);
    if (!site || site->getName().getStringRef() != "lvm.action_site")
      return owner->emitOpError(
          "remote observable action_site must resolve to lvm.action_site");
  }
  auto protocol = dyn_cast<ProtocolOp>(owner);
  if (!protocol)
    return owner->emitOpError(
        "communication QEC lowering must generate a fabric.protocol");
  Attribute generatorObjective = lowering->getAttr("objective");
  if (!generatorObjective || !protocol.getObjectiveAttr() ||
      protocol.getObjectiveAttr() != generatorObjective)
    return owner->emitOpError(
        "generated protocol objective must equal its generator objective");
  if (site && site->getAttr("objective") != generatorObjective)
    return owner->emitOpError(
        "generated protocol objective must equal its action-site objective");
  if (!family || family.getValue() != "pauli_product_measurement")
    return owner->emitOpError("remote observable generator must select "
                              "pauli_product_measurement");

  auto dependencies = lowering->getAttrOfType<ArrayAttr>("dependencies");
  auto ownerName = owner->getAttrOfType<StringAttr>("sym_name");
  bool generatedCandidate =
      dependencies && ownerName &&
      llvm::any_of(dependencies, [&](Attribute value) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(value);
        return reference && reference.getValue() == ownerName.getValue();
      });
  if (!generatedCandidate)
    return owner->emitOpError(
        "remote observable protocol must be a declared generator dependency");

  auto specialization = owner->getAttrOfType<DictionaryAttr>("specialization");
  if (!specialization)
    return owner->emitOpError(
        "remote observable generation requires canonical specialization");
  if (site) {
    auto siteParameters = site->getAttrOfType<DictionaryAttr>("parameters");
    if (!siteParameters)
      return owner->emitOpError(
          "remote observable action_site requires canonical parameters");
    for (NamedAttribute parameter : siteParameters)
      if (specialization.get(parameter.getName()) != parameter.getValue())
        return owner->emitOpError("generated specialization does not preserve "
                                  "action-site parameter ")
               << parameter.getName();
  } else {
    auto xMask = specialization.getAs<IntegerAttr>("x_mask");
    auto zMask = specialization.getAs<IntegerAttr>("z_mask");
    auto sign = specialization.getAs<IntegerAttr>("sign");
    if (!xMask || !zMask || !sign || xMask.getInt() != 0 ||
        zMask.getInt() != 3 || sign.getInt() != 1)
      return owner->emitOpError(
          "remote observable specialization must be positive two-body ZZ");
  }

  auto codes = lowering->getAttrOfType<ArrayAttr>("codes");
  auto acceptsCode = [&](Type type) {
    auto patch = dyn_cast<PatchType>(type);
    return !patch ||
           (codes && (llvm::is_contained(codes, patch.getCodeType()) ||
                      (patch.getEncoding() &&
                       llvm::is_contained(codes, patch.getEncoding()))));
  };
  if (!llvm::all_of(protocol.getFunctionType().getInputs(), acceptsCode) ||
      !llvm::all_of(protocol.getFunctionType().getResults(), acceptsCode))
    return owner->emitOpError(
        "generated protocol patch code is not accepted by its generator");
  return success();
}

static LogicalResult
verifyRPPSpecializationSchema(Operation *owner, DictionaryAttr specialization) {
  auto strategy = specialization.getAs<StringAttr>("rpp_strategy");
  if (!strategy)
    return success(); // Other QEC compiler families own their parameter schema.
  StringRef strategyName = strategy.getValue();
  if (strategyName != "clifford" && strategyName != "t_injection" &&
      strategyName != "native" && strategyName != "rotation_state" &&
      strategyName != "synthesis")
    return owner->emitOpError("unknown RPP specialization strategy ")
           << strategy;
  auto convention = specialization.getAs<StringAttr>("angle_convention");
  if (!convention || convention.getValue() != "exp(-i*theta*P/2)")
    return owner->emitOpError(
        "RPP specialization requires canonical angle_convention");
  if (!specialization.getAs<FloatAttr>("angle") ||
      !specialization.getAs<FloatAttr>("effective_angle"))
    return owner->emitOpError(
        "RPP specialization requires angle and effective_angle f64 values");
  Attribute numeratorValue = specialization.get("angle_pi_numer");
  Attribute denominatorValue = specialization.get("angle_pi_denom");
  if (numeratorValue || denominatorValue) {
    auto numerator = dyn_cast_or_null<IntegerAttr>(numeratorValue);
    auto denominator = dyn_cast_or_null<IntegerAttr>(denominatorValue);
    if (!numerator || !denominator)
      return owner->emitOpError(
          "RPP specialization exact angle requires integer angle_pi_numer "
          "and angle_pi_denom");
    if (denominator.getValue().isNegative() || denominator.getValue().isZero())
      return owner->emitOpError(
          "RPP specialization angle_pi_denom must be positive");
  }
  if (auto precision = specialization.getAs<FloatAttr>("precision"))
    if (precision.getValueAsDouble() <= 0.0)
      return owner->emitOpError(
          "RPP specialization precision must be positive");
  return success();
}

static LogicalResult verifyGeneratedSpecialization(Operation *owner) {
  if (failed(verifyGeneratedRemoteProvenance(owner)))
    return failure();
  auto specialization = owner->getAttrOfType<DictionaryAttr>("specialization");
  if (!specialization)
    return success();
  if (specialization.getAs<StringAttr>("rpp_strategy") &&
      !owner->hasAttr("generated_by"))
    return owner->emitOpError(
        "RPP specialization requires generated_by provenance");
  return verifyRPPSpecializationSchema(owner, specialization);
}

LogicalResult ObjectiveOp::verify() {
  bool hasLogical = static_cast<bool>(getLogicalAttr());
  bool hasSubsystem = static_cast<bool>(getSubsystemFragmentAttr());
  if (hasLogical == hasSubsystem)
    return emitOpError(
        "requires exactly one semantic source: logical or subsystem_fragment");
  if (!hasSubsystem)
    return success();

  DictionaryAttr fragment = getSubsystemFragmentAttr();
  auto derivation = fragment.getAs<StringAttr>("derivation");
  if (!derivation || derivation.getValue() != "body_exact")
    return emitOpError("subsystem_fragment requires derivation = body_exact");
  for (StringRef field : {"protected_effect", "gauge_effect"}) {
    auto effect = fragment.getAs<StringAttr>(field);
    if (!effect ||
        (effect.getValue() != "identity" && effect.getValue() != "unitary" &&
         effect.getValue() != "instrument"))
      return emitOpError("subsystem_fragment requires typed ")
             << field << " in {identity, unitary, instrument}";
  }
  auto epoch = fragment.getAs<StringAttr>("epoch_effect");
  if (!epoch || (epoch.getValue() != "preserve" && epoch.getValue() != "cycle"))
    return emitOpError(
        "subsystem_fragment epoch_effect must be preserve or cycle");
  auto steps = fragment.getAs<ArrayAttr>("steps");
  if (!steps || steps.empty())
    return emitOpError("subsystem_fragment requires one or more exact steps");

  auto functionType = getFunctionType();
  bool hasDynamicStep = false;
  for (auto [stepIndex, raw] : llvm::enumerate(steps)) {
    auto step = dyn_cast<DictionaryAttr>(raw);
    if (!step)
      return emitOpError("subsystem_fragment step ")
             << stepIndex << " must be a dictionary";
    auto operation = step.getAs<StringAttr>("operation");
    if (!operation ||
        (operation.getValue() != "measure" && operation.getValue() != "pauli" &&
         operation.getValue() != "rotate" &&
         operation.getValue() != "measure_gauges" &&
         operation.getValue() != "epoch_transition"))
      return emitOpError("subsystem_fragment step ")
             << stepIndex << " has invalid operation";
    if (operation.getValue() == "measure_gauges") {
      hasDynamicStep = true;
      auto operators = step.getAs<DenseIntElementsAttr>("operators");
      auto phase = step.getAs<StringAttr>("phase");
      auto record = step.getAs<StringAttr>("record");
      auto inputEpoch = step.getAs<StringAttr>("input_epoch");
      if (!operators || operators.getType().getRank() != 2 ||
          !operators.getType().getElementType().isInteger(1) ||
          operators.getType().getShape()[0] <= 0 || !phase || phase.empty() ||
          !record || record.empty() || !inputEpoch || inputEpoch.empty())
        return emitOpError("subsystem_fragment measure_gauges step ")
               << stepIndex
               << " requires nonempty operators, phase, record, and "
                  "input_epoch";
      continue;
    }
    if (operation.getValue() == "epoch_transition") {
      hasDynamicStep = true;
      auto from = step.getAs<StringAttr>("from_epoch");
      auto to = step.getAs<StringAttr>("to_epoch");
      auto logicalMap = step.getAs<DictionaryAttr>("logical_map");
      auto evidence = step.getAs<StringAttr>("evidence");
      if (!from || from.empty() || !to || to.empty() || !logicalMap ||
          !evidence || evidence.empty())
        return emitOpError("subsystem_fragment epoch_transition step ")
               << stepIndex
               << " requires from_epoch, to_epoch, logical_map, and evidence";
      llvm::StringSet<> mapped;
      for (NamedAttribute entry : logicalMap) {
        auto target = dyn_cast<StringAttr>(entry.getValue());
        if (entry.getName().getValue().empty() || !target || target.empty() ||
            !mapped.insert(target.getValue()).second)
          return emitOpError("subsystem_fragment epoch_transition step ")
                 << stepIndex << " requires a typed bijective logical_map";
      }
      if (Attribute closure = step.get("period_closure"))
        if (failed(verifySymplecticClosure(getOperation(), closure,
                                           "step period_closure")))
          return failure();
      continue;
    }
    auto patches = step.getAs<DenseI64ArrayAttr>("patch_indices");
    auto kinds = step.getAs<ArrayAttr>("port_kinds");
    auto indices = step.getAs<DenseI64ArrayAttr>("port_indices");
    auto paulis = step.getAs<StringAttr>("paulis");
    auto sign = step.getAs<IntegerAttr>("sign");
    if (!patches || !kinds || !indices || !paulis || !sign)
      return emitOpError("subsystem_fragment step ")
             << stepIndex << " is missing typed Pauli support";
    if (patches.size() != kinds.size() || kinds.size() != indices.size() ||
        paulis.getValue().size() != kinds.size() || kinds.empty())
      return emitOpError("subsystem_fragment step ")
             << stepIndex << " has inconsistent support widths";
    if (sign.getInt() != -1 && sign.getInt() != 1)
      return emitOpError("subsystem_fragment step ")
             << stepIndex << " sign must be +1 or -1";
    for (auto [portIndex, rawKind] : llvm::enumerate(kinds)) {
      auto kind = dyn_cast<StringAttr>(rawKind);
      if (!kind ||
          (kind.getValue() != "protected" && kind.getValue() != "gauge"))
        return emitOpError("subsystem_fragment step ")
               << stepIndex << " port " << portIndex
               << " must be protected or gauge";
      char pauli = paulis.getValue()[portIndex];
      if (pauli != 'X' && pauli != 'Y' && pauli != 'Z')
        return emitOpError("subsystem_fragment step ")
               << stepIndex << " contains a non-Pauli factor";
      if (patches.asArrayRef()[portIndex] < 0 ||
          indices.asArrayRef()[portIndex] < 0)
        return emitOpError("subsystem_fragment step ")
               << stepIndex << " contains a negative port index";
    }
    if (auto result = step.getAs<IntegerAttr>("outcome_result")) {
      int64_t index = result.getInt();
      if (operation.getValue() != "measure")
        return emitOpError("only a measurement step may export an outcome");
      if (index < 0 ||
          static_cast<unsigned>(index) >= functionType.getNumResults() ||
          !functionType.getResult(index).isInteger(1))
        return emitOpError(
            "subsystem_fragment outcome_result must name an i1 result");
    }
  }

  if (epoch.getValue() == "preserve") {
    if (hasDynamicStep)
      return emitOpError("epoch-preserving subsystem_fragment cannot contain "
                         "dynamic phase steps");
    return success();
  }

  auto profileRef = fragment.getAs<FlatSymbolRefAttr>("profile");
  auto closure = fragment.getAs<DenseIntElementsAttr>("period_closure");
  if (!profileRef || !closure)
    return emitOpError(
        "cyclic subsystem_fragment requires profile and period_closure");
  Operation *profileTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, profileRef);
  auto profile = dyn_cast_or_null<CodeProfileOp>(profileTarget);
  if (!profile || !profile.getDynamicPhases())
    return emitOpError("cyclic subsystem_fragment profile must resolve to a "
                       "dynamic code profile");
  if (!profile.getPeriodClosureAttr() ||
      profile.getPeriodClosureAttr() != closure)
    return emitOpError("cyclic subsystem_fragment period_closure must equal "
                       "its dynamic profile");

  ArrayAttr phases = *profile.getDynamicPhases();
  if (steps.size() != 2 * phases.size())
    return emitOpError("cyclic subsystem_fragment must contain one "
                       "measure/transition pair per dynamic phase");
  for (auto [index, rawPhase] : llvm::enumerate(phases)) {
    auto phase = cast<DictionaryAttr>(rawPhase);
    auto measurement = cast<DictionaryAttr>(steps[2 * index]);
    auto transition = cast<DictionaryAttr>(steps[2 * index + 1]);
    auto measurementKind = measurement.getAs<StringAttr>("operation");
    auto transitionKind = transition.getAs<StringAttr>("operation");
    if (!measurementKind || measurementKind.getValue() != "measure_gauges" ||
        !transitionKind || transitionKind.getValue() != "epoch_transition")
      return emitOpError("cyclic subsystem_fragment steps must alternate "
                         "measure_gauges and epoch_transition");
    if (measurement.get("operators") != phase.get("measured_gauges") ||
        measurement.get("phase") != phase.get("name") ||
        measurement.get("input_epoch") != phase.get("input_epoch"))
      return emitOpError("cyclic subsystem_fragment measurement step ")
             << 2 * index << " contradicts dynamic profile phase " << index;
    if (transition.get("from_epoch") != phase.get("input_epoch") ||
        transition.get("to_epoch") != phase.get("output_epoch") ||
        transition.get("logical_map") != phase.get("logical_map"))
      return emitOpError("cyclic subsystem_fragment transition step ")
             << 2 * index << " contradicts dynamic profile phase " << index;
    Attribute stepClosure = transition.get("period_closure");
    if (index + 1 == phases.size()) {
      if (stepClosure != profile.getPeriodClosureAttr())
        return emitOpError("final cyclic subsystem_fragment transition must "
                           "carry the exact period_closure");
    } else if (stepClosure) {
      return emitOpError("only the final cyclic subsystem_fragment transition "
                         "may carry period_closure");
    }
  }
  auto firstPhase = cast<DictionaryAttr>(phases[0]);
  auto lastPhase = cast<DictionaryAttr>(phases[phases.size() - 1]);
  if (lastPhase.get("output_epoch") != firstPhase.get("input_epoch"))
    return emitOpError(
        "cyclic subsystem_fragment must return to the initial dynamic epoch");
  return success();
}

static LogicalResult
verifyGadgetSpecRealizationBoundary(GadgetOp gadget, GadgetSpecOp spec,
                                    bool requireWitness = false) {
  auto boundary = gadget.getRealizationBoundaryAttr();
  if (!boundary && requireWitness)
    return gadget.emitOpError(
        "a resolved gadget_spec requires a typed realization_boundary witness");
  if (!boundary)
    return success();
  if (boundary.getAs<ArrayAttr>("ports") != spec.getPortsAttr())
    return gadget.emitOpError(
        "referenced gadget spec ports do not match realization_boundary");
  if (boundary.getAs<ArrayAttr>("flows") != spec.getFlowsAttr())
    return gadget.emitOpError(
        "referenced gadget spec flows do not match realization_boundary");
  return success();
}

LogicalResult GadgetSpecOp::verify() {
  if ((*this)->hasAttr("selection"))
    return emitOpError(
        "selection policy belongs to fabric.retry or event.selection, not "
        "the gadget specification");
  if ((*this)->hasAttr("frame_map"))
    return emitOpError(
        "frame_map is not a gadget-spec semantic noun; use typed protocol "
        "frame operations or compiler-derived action equivalence");
  ObjectiveOp objective;
  if (auto *target =
          SymbolTable::lookupNearestSymbolFrom(*this, getObjectiveAttr())) {
    objective = dyn_cast<ObjectiveOp>(target);
    if (!objective)
      return emitOpError("objective must resolve to fabric.objective");
  }
  bool isEntrypoint = static_cast<bool>(getEntrypoint());
  if (isEntrypoint) {
    if (!objective)
      return emitOpError(
          "entrypoint GadgetSpec objective must resolve to fabric.objective");
    if (objective.getFunctionType() != getFunctionType())
      return emitOpError("entrypoint GadgetSpec objective and realization "
                         "function types must match exactly");
  }

  auto booleanOutcomes = [](FunctionType type) {
    return llvm::count_if(type.getResults(),
                          [](Type result) { return result.isInteger(1); });
  };
  int64_t realizationOutcomes = booleanOutcomes(getFunctionType());
  std::optional<int64_t> objectiveOutcomes;
  if (objective)
    objectiveOutcomes = booleanOutcomes(objective.getFunctionType());

  llvm::DenseSet<Attribute> declaredEncodings;
  for (Attribute raw : getEncodings()) {
    auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
    if (!reference)
      return emitOpError("encodings must contain symbol references");
    declaredEncodings.insert(reference);
    if (auto *target = SymbolTable::lookupNearestSymbolFrom(*this, reference))
      if (!isa<EncodingOp>(target))
        return emitOpError(
            "encoding reference must resolve to fabric.encoding");
  }

  llvm::StringSet<> recordSchema;
  if (auto schema = getRecordSchema())
    for (Attribute raw : *schema) {
      auto record = dyn_cast<StringAttr>(raw);
      if (!record || record.empty())
        return emitOpError(
            "record_schema must contain nonempty string record names");
      if (!recordSchema.insert(record.getValue()).second)
        return emitOpError("record_schema names must be unique");
    }

  auto verifyMapKeys = [&](DictionaryAttr map, ArrayRef<StringRef> allowed,
                           StringRef label) -> LogicalResult {
    for (NamedAttribute entry : map)
      if (!llvm::is_contained(allowed, entry.getName().getValue()))
        return emitOpError(label) << " contains unsupported field '"
                                  << entry.getName().getValue() << "'";
    return success();
  };

  auto verifyMapRecords = [&](DictionaryAttr map,
                              StringRef label) -> LogicalResult {
    auto records = map.getAs<ArrayAttr>("records");
    if (!records)
      return success();
    llvm::StringSet<> seen;
    for (Attribute raw : records) {
      auto record = dyn_cast<StringAttr>(raw);
      if (!record || record.empty())
        return emitOpError(label)
               << " records must contain nonempty string names";
      if (!seen.insert(record.getValue()).second)
        return emitOpError(label) << " record names must be unique";
      if (!getRecordSchema())
        return emitOpError(label) << " records require gadget record_schema";
      if (getRecordSchema() && !recordSchema.contains(record.getValue()))
        return emitOpError(label) << " record '" << record.getValue()
                                  << "' is absent from record_schema";
    }
    return success();
  };

  if (auto outcome = getOutcomeMap()) {
    if (failed(verifyMapKeys(
            *outcome,
            {"records", "rows", "constants", "input_syndromes", "roles"},
            "outcome_map")))
      return failure();
    auto records = outcome->getAs<ArrayAttr>("records");
    auto rows = outcome->getAs<DenseIntElementsAttr>("rows");
    auto constants = outcome->getAs<DenseI64ArrayAttr>("constants");
    auto inputSyndromes = outcome->getAs<ArrayAttr>("input_syndromes");
    auto roles = outcome->getAs<ArrayAttr>("roles");
    if (!records || !rows || rows.getType().getRank() != 2 ||
        !rows.getType().getElementType().isInteger(1) || !constants)
      return emitOpError(
          "outcome_map requires records, rank-2 i1 rows, and constants");
    auto shape = rows.getType().getShape();
    if (shape[1] != static_cast<int64_t>(records.size()) ||
        shape[0] != static_cast<int64_t>(constants.size()))
      return emitOpError(
          "outcome_map dimensions must match its records and constants");
    if ((realizationOutcomes != 0 && shape[0] != realizationOutcomes) ||
        (objectiveOutcomes && shape[0] != *objectiveOutcomes))
      return emitOpError(
          "outcome_map rows must exactly cover objective and realization "
          "Boolean outcomes");
    if (llvm::any_of(constants.asArrayRef(),
                     [](int64_t value) { return value != 0 && value != 1; }))
      return emitOpError("outcome_map constants must be binary");
    if (failed(verifyMapRecords(*outcome, "outcome_map")))
      return failure();

    // Both arrays distinguish the canonical role-aware affine schema from
    // the pre-role replay form.  Legacy maps omit both and retain their old
    // consumer-specific interpretation; canonical producers emit both.
    if (static_cast<bool>(inputSyndromes) != static_cast<bool>(roles))
      return emitOpError(
          "outcome_map input_syndromes and roles must be present together");
    if (inputSyndromes) {
      if (static_cast<int64_t>(inputSyndromes.size()) != shape[0] ||
          static_cast<int64_t>(roles.size()) != shape[0])
        return emitOpError(
            "outcome_map input_syndromes and roles must contain one array per "
            "outcome row");

      SmallVector<int64_t> inputWidths;
      if (auto ports = getPorts())
        for (Attribute rawPort : *ports) {
          auto port = dyn_cast<DictionaryAttr>(rawPort);
          auto direction =
              port ? port.getAs<StringAttr>("direction") : StringAttr{};
          if (!direction || (direction.getValue() != "input" &&
                             direction.getValue() != "inout"))
            continue;
          int64_t width = -1;
          auto encoding = port.getAs<FlatSymbolRefAttr>("encoding");
          auto encodingOp =
              encoding
                  ? dyn_cast_or_null<EncodingOp>(
                        SymbolTable::lookupNearestSymbolFrom(*this, encoding))
                  : EncodingOp{};
          auto profileOp = encodingOp
                               ? dyn_cast_or_null<CodeProfileOp>(
                                     SymbolTable::lookupNearestSymbolFrom(
                                         *this, encodingOp.getProfileAttr()))
                               : CodeProfileOp{};
          if (profileOp) {
            auto effective = profileOp.getEffectiveStabilizers();
            auto matrix = effective ? dyn_cast<DenseIntElementsAttr>(*effective)
                                    : DenseIntElementsAttr{};
            if (!matrix || matrix.getType().getRank() != 2)
              return emitOpError(
                  "outcome_map input endpoint profile has no canonical "
                  "effective syndrome basis");
            width = matrix.getType().getShape()[0];
          }
          inputWidths.push_back(width);
        }

      for (auto [row, rawTerms] : llvm::enumerate(inputSyndromes)) {
        auto terms = dyn_cast<ArrayAttr>(rawTerms);
        if (!terms)
          return emitOpError("outcome_map input_syndromes row ")
                 << row << " must be an array";
        llvm::StringSet<> seenTerms;
        SmallVector<std::pair<int64_t, int64_t>> parsedTerms;
        for (Attribute rawTerm : terms) {
          auto term = dyn_cast<DictionaryAttr>(rawTerm);
          if (!term)
            return emitOpError(
                "outcome_map input-syndrome terms must be dictionaries");
          if (failed(verifyMapKeys(term, {"port", "index"},
                                   "outcome_map input-syndrome term")))
            return failure();
          auto port = term.getAs<IntegerAttr>("port");
          auto index = term.getAs<IntegerAttr>("index");
          if (!port || !index || port.getInt() < 0 || index.getInt() < 0)
            return emitOpError(
                "outcome_map input-syndrome terms require nonnegative integer "
                "port and index");
          std::string key =
              (Twine(port.getInt()) + ":" + Twine(index.getInt())).str();
          if (!seenTerms.insert(key).second)
            return emitOpError(
                "outcome_map input-syndrome terms must be duplicate-free");
          parsedTerms.emplace_back(port.getInt(), index.getInt());
        }
        for (auto [port, index] : parsedTerms) {
          if (port >= static_cast<int64_t>(inputWidths.size()))
            return emitOpError(
                       "outcome_map input-syndrome term names absent input "
                       "endpoint ")
                   << port;
          int64_t width = inputWidths[port];
          if (width >= 0 && index >= width)
            return emitOpError("outcome_map input-syndrome term ")
                   << port << "[" << index
                   << "] exceeds effective syndrome width " << width;
        }
      }

      for (auto [row, rawRoles] : llvm::enumerate(roles)) {
        auto rowRoles = dyn_cast<ArrayAttr>(rawRoles);
        if (!rowRoles)
          return emitOpError("outcome_map roles row ")
                 << row << " must be an array";
        if (rowRoles.empty())
          return emitOpError("outcome_map row roles must be nonempty");
        llvm::StringSet<> seenRoles;
        for (Attribute rawRole : rowRoles) {
          auto role = dyn_cast<StringAttr>(rawRole);
          if (!role ||
              (role.getValue() != "result" && role.getValue() != "success"))
            return emitOpError(
                "outcome_map roles must be 'result' or 'success'");
          if (!seenRoles.insert(role.getValue()).second)
            return emitOpError("outcome_map row roles must be duplicate-free");
        }
      }
    }
  } else if (!isEntrypoint &&
             (realizationOutcomes != 0 ||
              (objectiveOutcomes && *objectiveOutcomes != 0))) {
    return emitOpError("Boolean objective and realization results require a "
                       "total outcome_map");
  }

  auto parameterArity = [](FunctionType type) {
    return llvm::count_if(type.getInputs(), [](Type input) {
      return isa<IntegerType, FloatType, IndexType>(input);
    });
  };
  int64_t realizationParameters = parameterArity(getFunctionType());
  std::optional<int64_t> objectiveParameters;
  if (objective)
    objectiveParameters = parameterArity(objective.getFunctionType());
  if (auto parameters = getParameterMap()) {
    llvm::StringSet<> objectiveNames;
    for (NamedAttribute entry : *parameters)
      if (auto objective = dyn_cast<StringAttr>(entry.getValue());
          !objective || objective.empty())
        return emitOpError(
            "parameter_map values must be nonempty objective parameter names");
      else if (!objectiveNames.insert(objective.getValue()).second)
        return emitOpError(
            "parameter_map must be a bijection onto objective parameters");
    if (static_cast<int64_t>(parameters->size()) != realizationParameters ||
        (objectiveParameters &&
         static_cast<int64_t>(parameters->size()) != *objectiveParameters))
      return emitOpError(
          "parameter_map must be total over objective and realization runtime "
          "parameters");
  }

  auto ports = getPorts();
  if (!ports) {
    if (getFlows() && !getFlows()->empty())
      return emitOpError("flows require the inferred or explicit port table");
    return success();
  }
  llvm::StringSet<> names;
  int64_t inputEndpointCount = 0;
  int64_t outputEndpointCount = 0;
  for (auto [index, raw] : llvm::enumerate(*ports)) {
    auto port = dyn_cast<DictionaryAttr>(raw);
    if (!port)
      return emitOpError("port ") << index << " must be a dictionary";
    auto name = port.getAs<StringAttr>("name");
    auto direction = port.getAs<StringAttr>("direction");
    auto ownership = port.getAs<StringAttr>("ownership");
    auto encoding = port.getAs<FlatSymbolRefAttr>("encoding");
    auto inputState = port.getAs<StringAttr>("input_state");
    auto outputState = port.getAs<StringAttr>("output_state");
    auto logicalArity = port.getAs<IntegerAttr>("logical_arity");
    auto dataWidth = port.getAs<IntegerAttr>("data_width");
    auto scratchWidth = port.getAs<IntegerAttr>("scratch_width");
    if (!name || name.getValue().empty() || !direction || !ownership ||
        !encoding || !inputState || !outputState || !logicalArity ||
        !dataWidth || !scratchWidth)
      return emitOpError("port ")
             << index << " is missing its typed boundary contract";
    if (!names.insert(name.getValue()).second)
      return emitOpError("port names must be unique");
    if (!declaredEncodings.contains(encoding))
      return emitOpError("port encoding must appear in encodings");
    StringRef d = direction.getValue();
    StringRef owner = ownership.getValue();
    StringRef input = inputState.getValue();
    StringRef output = outputState.getValue();
    if (d == "inout") {
      ++inputEndpointCount;
      ++outputEndpointCount;
      if (owner != "borrow" || input != "initialized" ||
          output != "initialized")
        return emitOpError(
            "inout ports must borrow initialized-to-initialized state");
    } else if (d == "input") {
      ++inputEndpointCount;
      if (owner != "consume" || input != "initialized" || output != "absent")
        return emitOpError(
            "input ports must consume initialized state to absent output");
    } else if (d == "output") {
      ++outputEndpointCount;
      if (owner != "produce" || input != "uninitialized" ||
          output != "initialized")
        return emitOpError("output ports must produce initialized state from "
                           "uninitialized carriers");
    } else {
      return emitOpError("port direction must be input, output, or inout");
    }
    if (logicalArity.getInt() < 0 || dataWidth.getInt() < 0 ||
        scratchWidth.getInt() < 0)
      return emitOpError("port widths must be nonnegative");
    if (auto *target = SymbolTable::lookupNearestSymbolFrom(*this, encoding)) {
      auto encodingOp = dyn_cast<EncodingOp>(target);
      if (!encodingOp)
        return emitOpError("port encoding must resolve to fabric.encoding");
      if (auto *codeTarget = SymbolTable::lookupNearestSymbolFrom(
              *this, encodingOp.getCodeAttr())) {
        auto code = dyn_cast<CodeOp>(codeTarget);
        if (!code)
          return emitOpError("port encoding code must resolve to fabric.code");
        if (logicalArity.getInt() != code.getK() ||
            dataWidth.getInt() != code.getN())
          return emitOpError("port logical/data widths disagree with its code");
      }
    }
  }

  if (auto flows = getFlows()) {
    SmallVector<bool> seenInputs(inputEndpointCount, false);
    SmallVector<bool> seenOutputs(outputEndpointCount, false);
    for (auto [flowIndex, raw] : llvm::enumerate(*flows)) {
      auto flow = dyn_cast<DictionaryAttr>(raw);
      auto kind = flow ? flow.getAs<StringAttr>("kind") : StringAttr{};
      auto inputs =
          flow ? flow.getAs<DenseI64ArrayAttr>("inputs") : DenseI64ArrayAttr{};
      auto outputs =
          flow ? flow.getAs<DenseI64ArrayAttr>("outputs") : DenseI64ArrayAttr{};
      if (!kind || kind.empty() || !inputs || !outputs)
        return emitOpError("flow ")
               << flowIndex << " requires kind, inputs, and outputs";
      if (inputs.empty() && outputs.empty())
        return emitOpError("flow ") << flowIndex << " has no endpoints";
      for (int64_t endpoint : inputs.asArrayRef()) {
        if (endpoint < 0 || endpoint >= inputEndpointCount)
          return emitOpError("flow ")
                 << flowIndex << " has invalid input endpoint " << endpoint;
        if (seenInputs[endpoint])
          return emitOpError("input endpoint ")
                 << endpoint << " appears in more than one flow";
        seenInputs[endpoint] = true;
      }
      for (int64_t endpoint : outputs.asArrayRef()) {
        if (endpoint < 0 || endpoint >= outputEndpointCount)
          return emitOpError("flow ")
                 << flowIndex << " has invalid output endpoint " << endpoint;
        if (seenOutputs[endpoint])
          return emitOpError("output endpoint ")
                 << endpoint << " appears in more than one flow";
        seenOutputs[endpoint] = true;
      }
    }
    if (llvm::is_contained(seenInputs, false) ||
        llvm::is_contained(seenOutputs, false))
      return emitOpError("flows must cover every input and output endpoint");
  }
  return success();
}

static FailureOr<CodeProfileOp>
resolveSelectedCodeProfile(Operation *owner, FlatSymbolRefAttr codeRef,
                           FlatSymbolRefAttr encodingRef,
                           FlatSymbolRefAttr epochRef,
                           FlatSymbolRefAttr expectedProfile, StringRef label) {
  if (!encodingRef)
    return owner->emitOpError(label)
           << " requires an encoding-qualified endpoint";
  auto encoding = dyn_cast_or_null<EncodingOp>(
      SymbolTable::lookupNearestSymbolFrom(owner, encodingRef));
  if (!encoding)
    return owner->emitOpError(label)
           << " encoding must resolve to fabric.encoding";
  if (encoding.getCodeAttr() != codeRef)
    return owner->emitOpError(label)
           << " encoding code contradicts the typed endpoint code";
  if (expectedProfile && encoding.getProfileAttr() != expectedProfile)
    return owner->emitOpError(label)
           << " profile must exactly equal the typed endpoint encoding's "
              "selected CodeProfile";
  auto profile = dyn_cast_or_null<CodeProfileOp>(
      SymbolTable::lookupNearestSymbolFrom(owner, encoding.getProfileAttr()));
  if (!profile || profile.getCodeAttr() != codeRef)
    return owner->emitOpError(label)
           << " selected profile must resolve to a CodeProfile for the typed "
              "endpoint code";
  if (epochRef) {
    auto epoch = dyn_cast_or_null<EncodingEpochOp>(
        SymbolTable::lookupNearestSymbolFrom(owner, epochRef));
    if (!epoch || epoch.getEncodingAttr() != encodingRef)
      return owner->emitOpError(label)
             << " epoch must belong to the typed endpoint encoding";
  }
  return profile;
}

struct RecursiveRecordManifest {
  SmallVector<std::string> measurementPaths;
  llvm::StringMap<unsigned> typedProducerCounts;
};

static bool isOutcomeMeasurementProducer(Operation *op) {
  return isa<MzOp, MeasureBasisOp, MppOp, MeasureProductOp, MeasureGaugesOp,
             ReadSyndromeAncillasOp, MultiMeasureOp, SplitOp>(op);
}

static FailureOr<RecursiveRecordManifest>
collectRecursiveRecordManifest(GadgetOp root, bool requireMeasurementRecords,
                               ArrayRef<std::string> demandedPaths = {}) {
  RecursiveRecordManifest manifest;
  auto qualify = [](ArrayRef<std::string> instancePath,
                    StringRef local) -> std::string {
    if (instancePath.empty())
      return local.str();
    std::string result;
    for (StringRef segment : instancePath) {
      if (!result.empty())
        result += ".";
      result += segment;
    }
    result += ".";
    result += local;
    return result;
  };
  auto addTypedPath = [&](ArrayRef<std::string> instancePath, Twine path) {
    std::string qualified = qualify(instancePath, path.str());
    ++manifest.typedProducerCounts[qualified];
    return qualified;
  };
  auto addMeasurementPath = [&](ArrayRef<std::string> instancePath,
                                Twine path) {
    manifest.measurementPaths.push_back(addTypedPath(instancePath, path));
  };
  auto addMeasurementFamily = [&](ArrayRef<std::string> instancePath,
                                  StringRef base, StringRef field,
                                  int64_t width) {
    for (int64_t index = 0; index < width; ++index)
      addMeasurementPath(instancePath,
                         Twine(base) + "." + field + Twine(index));
  };
  auto addTypedFamily = [&](ArrayRef<std::string> instancePath, StringRef base,
                            StringRef field, int64_t width) {
    for (int64_t index = 0; index < width; ++index)
      addTypedPath(instancePath, Twine(base) + "." + field + Twine(index));
  };
  auto tensorWidth = [](Value value) -> std::optional<int64_t> {
    auto type = dyn_cast<RankedTensorType>(value.getType());
    if (!type || type.getRank() != 1 || type.isDynamicDim(0))
      return std::nullopt;
    return type.getShape()[0];
  };
  auto syndromeWidth = [](Operation *op,
                          SyndromeType syndrome) -> std::optional<int64_t> {
    if (!syndrome.getEncoding())
      return std::nullopt;
    auto profile = resolveSelectedCodeProfile(
        op, syndrome.getCodeType(), syndrome.getEncoding(), syndrome.getEpoch(),
        FlatSymbolRefAttr{}, "syndrome record");
    if (failed(profile))
      return std::nullopt;
    auto effective = (*profile).getEffectiveStabilizers();
    auto matrix = effective ? dyn_cast<DenseIntElementsAttr>(*effective)
                            : DenseIntElementsAttr{};
    if (!matrix || matrix.getType().getRank() != 2 ||
        !matrix.getType().getElementType().isInteger(1))
      return std::nullopt;
    return matrix.getType().getShape()[0];
  };
  std::function<LogicalResult(Operation *, ArrayRef<std::string>, unsigned)>
      collectOperation;
  std::function<LogicalResult(Region &, ArrayRef<std::string>)> collectRegion;
  std::function<LogicalResult(GadgetOp, ArrayRef<std::string>)> collectGadget;
  llvm::DenseSet<Operation *> activeGadgets;
  collectRegion = [&](Region &region,
                      ArrayRef<std::string> instancePath) -> LogicalResult {
    for (Block &block : region) {
      unsigned ordinal = 0;
      for (Operation &operation : block) {
        if (failed(collectOperation(&operation, instancePath, ordinal)))
          return failure();
        ++ordinal;
      }
    }
    return success();
  };
  collectGadget = [&](GadgetOp gadget,
                      ArrayRef<std::string> instancePath) -> LogicalResult {
    if (!activeGadgets.insert(gadget.getOperation()).second) {
      if (!requireMeasurementRecords)
        return success();
      return gadget.emitOpError(
          "recursive gadget calls cannot define a finite outcome_response "
          "measurement table");
    }
    Operation *realizationOwner = gadget.getOperation();
    if (auto realization = gadget.getRealizationAttr()) {
      auto *target = SymbolTable::lookupNearestSymbolFrom(gadget, realization);
      auto circuit = dyn_cast_or_null<CircuitOp>(target);
      if (!circuit) {
        activeGadgets.erase(gadget.getOperation());
        if (!requireMeasurementRecords)
          return success();
        return gadget.emitOpError(
            "outcome_response measurement collection requires the referenced "
            "realization to resolve to fabric.circuit");
      }
      realizationOwner = circuit.getOperation();
    }
    for (Region &region : realizationOwner->getRegions())
      if (failed(collectRegion(region, instancePath))) {
        activeGadgets.erase(gadget.getOperation());
        return failure();
      }
    activeGadgets.erase(gadget.getOperation());
    return success();
  };
  collectOperation = [&](Operation *op, ArrayRef<std::string> instancePath,
                         unsigned operationOrdinal) -> LogicalResult {
    if (auto call = dyn_cast<CallOp>(op)) {
      auto *target =
          SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr());
      auto callee = dyn_cast_or_null<GadgetOp>(target);
      if (!callee) {
        if (!requireMeasurementRecords)
          return success();
        return call.emitOpError(
            "outcome_response measurement collection requires a resolved "
            "fabric.gadget callee");
      }
      SmallVector<std::string> nestedPath(instancePath.begin(),
                                          instancePath.end());
      nestedPath.push_back(
          (Twine("__qlx_call") + Twine(operationOrdinal)).str());
      return collectGadget(callee, nestedPath);
    }
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(op)) {
      auto collectIteration = [&](int64_t iteration) -> LogicalResult {
        SmallVector<std::string> nestedPath(instancePath.begin(),
                                            instancePath.end());
        nestedPath.push_back((Twine("__qlx_repeat") + Twine(operationOrdinal) +
                              "_" + Twine(iteration))
                                 .str());
        return collectRegion(repeat.getBody(), nestedPath);
      };
      if (requireMeasurementRecords) {
        for (int64_t iteration = 0; iteration < repeat.getCount(); ++iteration)
          if (failed(collectIteration(iteration)))
            return failure();
        return success();
      }

      // Stable schemas and detached profiles ask whether a finite set of
      // exact record paths resolve. Select those coordinates algebraically
      // instead of expanding the repeat's complete manifest.
      std::string repeatPrefix;
      for (StringRef segment : instancePath)
        repeatPrefix += (Twine(segment) + ".").str();
      repeatPrefix +=
          (Twine("__qlx_repeat") + Twine(operationOrdinal) + "_").str();
      std::set<int64_t> selectedIterations;
      if (repeat.getCount() > 0)
        selectedIterations.insert(0);
      for (const std::string &path : demandedPaths) {
        StringRef remainder(path);
        if (!remainder.consume_front(repeatPrefix))
          continue;
        size_t separator = remainder.find('.');
        if (separator == StringRef::npos)
          continue;
        int64_t iteration = -1;
        if (remainder.take_front(separator).getAsInteger(10, iteration) ||
            iteration < 0 || iteration >= repeat.getCount())
          continue;
        selectedIterations.insert(iteration);
      }
      for (int64_t iteration : selectedIterations)
        if (failed(collectIteration(iteration)))
          return failure();
      return success();
    }
    if (requireMeasurementRecords && op->getNumRegions() != 0) {
      bool reachesPathDependentProducer = false;
      op->walk([&](Operation *nested) -> WalkResult {
        if (nested == op)
          return WalkResult::advance();
        if (isa<CallOp, qlx::cflow::RepeatOp>(nested) ||
            isOutcomeMeasurementProducer(nested)) {
          reachesPathDependentProducer = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (reachesPathDependentProducer)
        return op->emitOpError(
            "outcome_response measurement collection cannot flatten a "
            "path-dependent or dynamically executed region containing a "
            "measurement, gadget call, or repeat; only fixed cflow.repeat "
            "has a finite manifest contract");
    }
    bool isMeasurement = isOutcomeMeasurementProducer(op);
    auto record = op->getAttrOfType<StringAttr>("record");
    if (isMeasurement && !record && requireMeasurementRecords)
      return op->emitOpError(
          "every measurement contributing to an outcome_response requires a "
          "stable record attribute");
    if (isMeasurement && !record)
      return success();
    if (isa<MultiMeasureOp, SplitOp>(op) && requireMeasurementRecords)
      return op->emitOpError(
          "outcome_response does not support this legacy measurement op "
          "because it has no stable record-layout contract");
    if (record && record.getValue().starts_with("__qlx_"))
      return op->emitOpError(
          "measurement record names beginning '__qlx_' are reserved for "
          "compiler-derived invocation qualification");
    if (auto measurement = dyn_cast<MzOp>(op)) {
      auto width = tensorWidth(measurement.getBits());
      if (!width)
        return op->emitOpError(
            "outcome_response requires a statically known measurement width");
      addMeasurementFamily(instancePath, record.getValue(),
                           stringifyPartition(measurement.getPartition()),
                           *width);
    } else if (auto measurement = dyn_cast<MeasureBasisOp>(op)) {
      auto width = tensorWidth(measurement.getBits());
      if (!width)
        return op->emitOpError(
            "outcome_response requires a statically known measurement width");
      addMeasurementFamily(instancePath, record.getValue(),
                           stringifyPartition(measurement.getPartition()),
                           *width);
    } else if (isa<MppOp, MeasureProductOp>(op)) {
      addMeasurementPath(instancePath, Twine(record.getValue()) + ".outcome");
    } else if (auto measurement = dyn_cast<MeasureGaugesOp>(op)) {
      auto operators =
          dyn_cast<DenseIntElementsAttr>(measurement.getOperators());
      if (!operators || operators.getType().getRank() != 2)
        return op->emitOpError(
            "outcome_response requires a statically known gauge-measurement "
            "width");
      addMeasurementFamily(instancePath, record.getValue(), "g",
                           operators.getType().getShape()[0]);
    } else if (auto read = dyn_cast<ReadSyndromeAncillasOp>(op)) {
      auto width =
          syndromeWidth(op, cast<SyndromeType>(read.getSyndrome().getType()));
      if (!width)
        return op->emitOpError(
            "outcome_response requires a statically known syndrome width");
      addMeasurementFamily(instancePath, record.getValue(), "s", *width);
    } else if (auto assemble = dyn_cast<AssembleSyndromeOp>(op)) {
      if (record) {
        auto width = syndromeWidth(
            op, cast<SyndromeType>(assemble.getSyndrome().getType()));
        if (!width)
          return op->emitOpError(
              "stable record collection requires a statically known "
              "assembled-syndrome width");
        addTypedFamily(instancePath, record.getValue(), "s", *width);
      }
    }
    for (Region &region : op->getRegions())
      if (failed(collectRegion(region, instancePath)))
        return failure();
    return success();
  };
  if (failed(collectGadget(root, {})))
    return failure();
  return manifest;
}

namespace {
using OutcomeSyndromeTerm = std::pair<int64_t, int64_t>;

static SmallVector<OutcomeSyndromeTerm>
getOutcomeSyndromeTerms(DictionaryAttr outcome, int64_t row) {
  SmallVector<OutcomeSyndromeTerm> result;
  auto rows = outcome.getAs<ArrayAttr>("input_syndromes");
  if (!rows)
    return result;
  auto terms = cast<ArrayAttr>(rows[row]);
  for (Attribute raw : terms) {
    auto term = cast<DictionaryAttr>(raw);
    result.emplace_back(term.getAs<IntegerAttr>("port").getInt(),
                        term.getAs<IntegerAttr>("index").getInt());
  }
  return result;
}

static bool outcomeRowHasRole(DictionaryAttr outcome, int64_t row,
                              StringRef role) {
  auto rows = outcome.getAs<ArrayAttr>("roles");
  if (!rows)
    return true; // Preserve the pre-role replay interpretation.
  for (Attribute raw : cast<ArrayAttr>(rows[row]))
    if (cast<StringAttr>(raw).getValue() == role)
      return true;
  return false;
}

static SmallVector<OutcomeSyndromeTerm>
getProfileSyndromeTerms(Operation *declaration) {
  SmallVector<OutcomeSyndromeTerm> result;
  auto terms = declaration->getAttrOfType<ArrayAttr>("input_syndromes");
  if (!terms)
    return result;
  for (Attribute raw : terms) {
    auto term = cast<DictionaryAttr>(raw);
    result.emplace_back(term.getAs<IntegerAttr>("port_index").getInt(),
                        term.getAs<IntegerAttr>("index").getInt());
  }
  return result;
}

struct RecordAffineExpression {
  llvm::StringMap<bool> records;
  bool constant = false;
};
} // namespace

static void xorRecordExpression(RecordAffineExpression &target,
                                const RecordAffineExpression &source) {
  target.constant ^= source.constant;
  for (const auto &entry : source.records) {
    auto found = target.records.find(entry.getKey());
    if (found == target.records.end())
      target.records.insert({entry.getKey(), true});
    else
      target.records.erase(found);
  }
}

static void qualifyRecordExpression(RecordAffineExpression &expression,
                                    StringRef prefix) {
  llvm::StringMap<bool> qualified;
  for (const auto &entry : expression.records)
    qualified.insert({(Twine(prefix) + entry.getKey()).str(), true});
  expression.records = std::move(qualified);
}

static unsigned operationOrdinal(Operation *operation) {
  unsigned ordinal = 0;
  for (Operation &candidate : *operation->getBlock()) {
    if (&candidate == operation)
      return ordinal;
    ++ordinal;
  }
  llvm_unreachable("operation must belong to its reported parent block");
}

static FailureOr<Value> finalRepeatYield(qlx::cflow::RepeatOp repeat,
                                         Value result) {
  auto opResult = dyn_cast<OpResult>(result);
  if (!opResult || opResult.getOwner() != repeat.getOperation())
    return failure();
  unsigned index = opResult.getResultNumber();
  if (index >= repeat.getInits().size())
    return failure();
  if (repeat.getCount() == 0)
    return repeat.getInits()[index];
  auto yield =
      cast<qlx::cflow::YieldOp>(repeat.getBody().front().getTerminator());
  Value yielded = yield.getOperands()[index];
  if (auto argument = dyn_cast<BlockArgument>(yielded)) {
    // Identity carries have the same value in every iteration. General
    // recurrences remain unsupported rather than being inferred from one body.
    if (argument.getOwner() != &repeat.getBody().front() ||
        argument.getArgNumber() != index)
      return failure();
    return repeat.getInits()[index];
  }
  return yielded;
}

static std::string finalRepeatRecordPrefix(qlx::cflow::RepeatOp repeat) {
  return (Twine("__qlx_repeat") +
          Twine(operationOrdinal(repeat.getOperation())) + "_" +
          Twine(repeat.getCount() - 1) + ".")
      .str();
}

static FailureOr<SmallVector<RecordAffineExpression>>
resolveTypedRecordBundle(Value value) {
  std::function<FailureOr<SmallVector<RecordAffineExpression>>(Value)> resolve =
      [&](Value current) -> FailureOr<SmallVector<RecordAffineExpression>> {
    auto type = dyn_cast<RankedTensorType>(current.getType());
    if (!type || type.getRank() != 1 || type.isDynamicDim(0) ||
        !type.getElementType().isInteger(1))
      return failure();

    Operation *producer = current.getDefiningOp();
    if (!producer)
      return failure();
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(producer)) {
      auto yielded = finalRepeatYield(repeat, current);
      if (failed(yielded))
        return failure();
      auto result = resolve(*yielded);
      if (failed(result))
        return failure();
      if (repeat.getCount() != 0) {
        std::string prefix = finalRepeatRecordPrefix(repeat);
        for (auto &expression : *result)
          qualifyRecordExpression(expression, prefix);
      }
      return result;
    }

    auto record = producer->getAttrOfType<StringAttr>("record");
    if (!record || record.empty())
      return failure();
    SmallVector<RecordAffineExpression> result(type.getDimSize(0));
    auto setFamily = [&](StringRef field) {
      for (auto [index, expression] : llvm::enumerate(result))
        expression.records.insert(
            {(Twine(record.getValue()) + "." + field + Twine(index)).str(),
             true});
    };
    if (auto measurement = dyn_cast<MzOp>(producer)) {
      if (measurement.getBits() != current)
        return failure();
      setFamily(stringifyPartition(measurement.getPartition()));
      return result;
    }
    if (auto measurement = dyn_cast<MeasureBasisOp>(producer)) {
      if (measurement.getBits() != current)
        return failure();
      setFamily(stringifyPartition(measurement.getPartition()));
      return result;
    }
    if (auto measurement = dyn_cast<MppOp>(producer)) {
      if (measurement.getBits() != current || result.size() != 1)
        return failure();
      result.front().records.insert(
          {(Twine(record.getValue()) + ".outcome").str(), true});
      return result;
    }
    return failure();
  };
  return resolve(value);
}

static FailureOr<RecordAffineExpression>
resolveTypedRecordExpression(Value value) {
  std::function<FailureOr<RecordAffineExpression>(Value)> resolve =
      [&](Value current) -> FailureOr<RecordAffineExpression> {
    if (!current.getType().isInteger(1))
      return failure();
    if (auto measurement = current.getDefiningOp<MeasureProductOp>()) {
      if (measurement.getOutcome() != current)
        return failure();
      auto record = measurement->getAttrOfType<StringAttr>("record");
      if (!record || record.empty())
        return failure();
      RecordAffineExpression result;
      result.records.insert(
          {(Twine(record.getValue()) + ".outcome").str(), true});
      return result;
    }
    if (auto constant = current.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer || !integer.getType().isInteger(1))
        return failure();
      return RecordAffineExpression{{}, !integer.getValue().isZero()};
    }
    if (auto repeat = current.getDefiningOp<qlx::cflow::RepeatOp>()) {
      auto yielded = finalRepeatYield(repeat, current);
      if (failed(yielded))
        return failure();
      auto result = resolve(*yielded);
      if (failed(result))
        return failure();
      if (repeat.getCount() != 0)
        qualifyRecordExpression(*result, finalRepeatRecordPrefix(repeat));
      return result;
    }

    Value lhs;
    Value rhs;
    if (auto xorOp = current.getDefiningOp<XorOp>()) {
      lhs = xorOp.getLhs();
      rhs = xorOp.getRhs();
    } else if (auto xorOp = current.getDefiningOp<arith::XOrIOp>()) {
      lhs = xorOp.getLhs();
      rhs = xorOp.getRhs();
    }
    if (lhs && rhs) {
      auto left = resolve(lhs);
      auto right = resolve(rhs);
      if (failed(left) || failed(right))
        return failure();
      RecordAffineExpression result = *left;
      xorRecordExpression(result, *right);
      return result;
    }

    if (auto parity = current.getDefiningOp<ParityOp>()) {
      RecordAffineExpression result;
      for (Value bits : parity.getBits()) {
        auto bundle = resolveTypedRecordBundle(bits);
        if (failed(bundle))
          return failure();
        for (const auto &entry : *bundle)
          xorRecordExpression(result, entry);
      }
      return result;
    }

    return failure();
  };
  return resolve(value);
}

static LogicalResult verifyOutcomeReturnProvenance(Operation *owner,
                                                   ReturnOp returnOp,
                                                   GadgetSpecOp spec) {
  if (spec.getEntrypoint())
    return success();
  auto outcome = spec.getOutcomeMap();
  if (!outcome)
    return success();
  auto records = outcome->getAs<ArrayAttr>("records");
  auto rows = outcome->getAs<DenseIntElementsAttr>("rows");
  auto constants = outcome->getAs<DenseI64ArrayAttr>("constants");
  if (!records || !rows || !constants)
    return owner->emitOpError(
        "linked GadgetSpec outcome_map must be canonical");

  llvm::StringMap<unsigned> recordColumns;
  for (auto [index, raw] : llvm::enumerate(records))
    recordColumns.insert(
        {cast<StringAttr>(raw).getValue(), static_cast<unsigned>(index)});
  SmallVector<Value> booleanResults;
  for (Value value : returnOp.getOperands())
    if (value.getType().isInteger(1))
      booleanResults.push_back(value);

  auto shape = rows.getType().getShape();
  auto denseValues = rows.getValues<APInt>();
  SmallVector<APInt> values(denseValues.begin(), denseValues.end());
  for (auto [row, value] : llvm::enumerate(booleanResults)) {
    if (!getOutcomeSyndromeTerms(*outcome, row).empty())
      return owner->emitOpError("Boolean gadget return ")
             << row
             << " has input-syndrome OutcomeMap terms that are not represented "
                "by its SSA provenance";
    auto expression = resolveTypedRecordExpression(value);
    if (failed(expression))
      return owner->emitOpError("Boolean gadget returns must have statically "
                                "provable affine typed-record provenance");
    llvm::SmallBitVector actual(records.size());
    for (const auto &record : expression->records) {
      auto column = recordColumns.find(record.getKey());
      if (column == recordColumns.end())
        return owner->emitOpError("Boolean gadget return references record '")
               << record.getKey() << "' absent from its GadgetSpec outcome_map";
      actual.set(column->second);
    }
    for (int64_t column = 0; column < shape[1]; ++column)
      if (actual.test(column) != !values[row * shape[1] + column].isZero())
        return owner->emitOpError("Boolean gadget return ")
               << row << " contradicts its GadgetSpec outcome_map row";
    bool declaredConstant = constants.asArrayRef()[row] != 0;
    if (expression->constant != declaredConstant)
      return owner->emitOpError("Boolean gadget return ")
             << row << " contradicts its GadgetSpec outcome_map constant";
  }
  return success();
}

static LogicalResult verifyTerminalMeasurementOwnership(Operation *owner) {
  auto verifyTerminalResult = [&](Operation *measurement, Value patchOut,
                                  Partition partition,
                                  DenseI64ArrayAttr indices) -> LogicalResult {
    if (partition != Partition::data || indices)
      return success();
    if (!patchOut.hasOneUse())
      return measurement->emitOpError(
          "complete data measurement must have exactly one terminal disposal "
          "use");
    Operation *user = *patchOut.getUsers().begin();
    if (isa<PatchType>(patchOut.getType())) {
      if (!isa<DeallocOp>(user))
        return measurement->emitOpError(
            "complete data measurement destroys the encoded state; its patch "
            "result may only feed fabric.dealloc");
      return success();
    }
    if (!isa<PatchFrameType>(patchOut.getType()))
      return measurement->emitOpError(
          "complete data measurement requires a patch or patch-frame result");
    auto transformEnd = dyn_cast<TransformEndOp>(user);
    if (!transformEnd || !transformEnd.getDestination().hasOneUse() ||
        !isa<DeallocOp>(*transformEnd.getDestination().getUsers().begin()))
      return measurement->emitOpError(
          "complete data measurement of a transformed patch may only close "
          "the carrier frame with fabric.transform_end and immediately "
          "fabric.dealloc the destination");
    return success();
  };

  WalkResult result = owner->walk([&](Operation *operation) -> WalkResult {
    LogicalResult verified = success();
    if (auto measurement = dyn_cast<MzOp>(operation))
      verified = verifyTerminalResult(operation, measurement.getPatchOut(),
                                      measurement.getPartition(),
                                      measurement.getIndicesAttr());
    else if (auto measurement = dyn_cast<MeasureBasisOp>(operation))
      verified = verifyTerminalResult(operation, measurement.getPatchOut(),
                                      measurement.getPartition(),
                                      measurement.getIndicesAttr());
    return failed(verified) ? WalkResult::interrupt() : WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static GadgetSpecOp resolvedGadgetSpec(GadgetOp gadget) {
  if (!gadget || !gadget.getSpecAttr())
    return {};
  return dyn_cast_or_null<GadgetSpecOp>(
      SymbolTable::lookupNearestSymbolFrom(gadget, gadget.getSpecAttr()));
}

static std::optional<StringRef> gadgetLogicalAction(GadgetOp gadget) {
  auto spec = resolvedGadgetSpec(gadget);
  if (!spec)
    return std::nullopt;
  auto objective = dyn_cast_or_null<ObjectiveOp>(
      SymbolTable::lookupNearestSymbolFrom(spec, spec.getObjectiveAttr()));
  if (!objective || !objective.getLogicalAttr())
    return std::nullopt;
  auto logical =
      dyn_cast_or_null<::qlx::ActionOp>(SymbolTable::lookupNearestSymbolFrom(
          objective, objective.getLogicalAttr()));
  return logical ? std::optional<StringRef>(logical.getKind()) : std::nullopt;
}

static bool hasGadgetEquivalence(GadgetOp gadget, StringRef expected) {
  auto spec = resolvedGadgetSpec(gadget);
  auto equivalence = spec
                         ? spec->getAttrOfType<StringAttr>("action_equivalence")
                         : StringAttr{};
  auto epoch =
      spec ? spec->getAttrOfType<StringAttr>("epoch_map") : StringAttr{};
  return equivalence && equivalence.getValue() == expected && epoch &&
         epoch.getValue() == "preserve";
}

static SmallVector<Operation *, 4> straightLineBody(Operation *callable) {
  SmallVector<Operation *, 4> result;
  Region &body = callable->getRegion(0);
  if (!llvm::hasSingleElement(body))
    return result;
  for (Operation &operation : body.front().without_terminator())
    result.push_back(&operation);
  return result;
}

static bool isFullIdentityDataRelation(Operation *operation, StringRef pairs,
                                       ValueRange operands) {
  if (operands.size() != 2)
    return false;
  auto left = dyn_cast<PatchType>(operands[0].getType());
  auto right = dyn_cast<PatchType>(operands[1].getType());
  if (!left || !right || left != right)
    return false;
  auto code = dyn_cast_or_null<CodeOp>(
      SymbolTable::lookupNearestSymbolFrom(operation, left.getCodeType()));
  if (!code || code.getK().value_or(1) != 1 || code.getR().value_or(0) != 0)
    return false;
  auto width = dyn_cast_or_null<IntegerAttr>(code.getPartitions().get("data"));
  if (!width || width.getInt() <= 0)
    return false;
  if (pairs == "index")
    return true;
  std::string expected;
  llvm::raw_string_ostream stream(expected);
  for (int64_t index = 0; index < width.getInt(); ++index) {
    if (index != 0)
      stream << ',';
    stream << index << ':' << index;
  }
  return pairs == stream.str();
}

static LogicalResult verifyDerivedCSSCX(GadgetOp gadget, GadgetSpecOp spec) {
  auto equivalence = spec ? spec.getActionEquivalence() : std::nullopt;
  if (!equivalence || *equivalence != "derived_exact_css_transversal_cx")
    return success();
  if (gadgetLogicalAction(gadget) != std::optional<StringRef>("cx"))
    return gadget.emitOpError(
        "derived CSS transversal CX requires the typed logical CX objective");
  auto semantic = straightLineBody(gadget);
  if (semantic.size() != 1)
    return gadget.emitOpError(
        "derived CSS transversal CX requires exactly one carrier operation");
  auto cx = dyn_cast<CXOp>(semantic.front());
  auto returnOp = dyn_cast<ReturnOp>(gadget.getBody().front().getTerminator());
  if (!cx || cx.getCtrl() != Partition::data ||
      cx.getTarg() != Partition::data || !cx.getPairs() ||
      !isFullIdentityDataRelation(cx, *cx.getPairs(), cx.getPatches()) ||
      cx.getPatches() != gadget.getBody().front().getArguments() || !returnOp ||
      returnOp.getOperands() != cx.getResults())
    return gadget.emitOpError(
        "derived CSS transversal CX requires full indexwise data coupling "
        "and ordered boundary successors");
  auto patch = dyn_cast<PatchType>(cx.getPatches().front().getType());
  auto code =
      patch ? dyn_cast_or_null<CodeOp>(SymbolTable::lookupNearestSymbolFrom(
                  gadget, patch.getCodeType()))
            : CodeOp{};
  bool isBare = code && code.getN().value_or(0) == 1 &&
                !code.getHx().has_value() && !code.getHz().has_value();
  if (!code || (!isBare && (!code.getHx() || !code.getHz())))
    return gadget.emitOpError(
        "derived CSS transversal CX requires a CSS code or a bare qubit");
  return success();
}

static LogicalResult verifyDerivedCSSHPermutation(GadgetOp gadget,
                                                  GadgetSpecOp spec) {
  auto equivalence = spec ? spec.getActionEquivalence() : std::nullopt;
  if (!equivalence ||
      *equivalence != "derived_exact_css_transversal_h_permutation")
    return success();
  if (gadgetLogicalAction(gadget) != std::optional<StringRef>("h"))
    return gadget.emitOpError(
        "derived CSS H-permutation requires the typed logical H objective");
  auto semantic = straightLineBody(gadget);
  auto h = semantic.empty() ? HOp{} : dyn_cast<HOp>(semantic.front());
  Block &entry = gadget.getBody().front();
  auto returnOp = dyn_cast<ReturnOp>(entry.getTerminator());
  if (!h || entry.getNumArguments() != 1 ||
      h.getPatch() != entry.getArgument(0) ||
      h.getPartition() != Partition::data || h.getIndicesAttr() ||
      h.getResult().getType() != entry.getArgument(0).getType() || !returnOp ||
      returnOp.getNumOperands() != 1)
    return gadget.emitOpError(
        "derived CSS H-permutation requires one full-data transversal H");

  auto patch = dyn_cast<PatchType>(entry.getArgument(0).getType());
  auto code =
      patch ? dyn_cast_or_null<CodeOp>(SymbolTable::lookupNearestSymbolFrom(
                  gadget, patch.getCodeType()))
            : CodeOp{};
  int64_t n = code ? code.getN().value_or(0) : 0;
  bool isBare = code && n == 1 && code.getK().value_or(1) == 1 &&
                code.getR().value_or(0) == 0 &&
                (!code.getHx() || code.getHx()->empty()) &&
                (!code.getHz() || code.getHz()->empty());
  if (!code || n <= 0 || code.getK().value_or(1) != 1 ||
      code.getR().value_or(0) != 0 ||
      (!isBare && (!code.getHx() || !code.getHz())))
    return gadget.emitOpError(
        "derived CSS H-permutation requires a bare qubit or one-logical CSS "
        "stabilizer code");

  auto parsePair =
      [](StringRef relation) -> std::optional<std::pair<int64_t, int64_t>> {
    if (relation.contains(','))
      return std::nullopt;
    auto [leftText, rightText] = relation.split(':');
    int64_t left = -1;
    int64_t right = -1;
    if (leftText.empty() || rightText.empty() || rightText.contains(':') ||
        leftText.getAsInteger(10, left) || rightText.getAsInteger(10, right))
      return std::nullopt;
    return std::pair<int64_t, int64_t>{left, right};
  };

  Value current = h.getResult();
  SmallVector<std::pair<int64_t, int64_t>> swaps;
  if ((semantic.size() - 1) % 3 != 0)
    return gadget.emitOpError(
        "derived CSS H-permutation requires complete three-CX SWAP groups");
  for (size_t offset = 1; offset < semantic.size(); offset += 3) {
    auto first = dyn_cast<CXOp>(semantic[offset]);
    auto second = dyn_cast<CXOp>(semantic[offset + 1]);
    auto third = dyn_cast<CXOp>(semantic[offset + 2]);
    if (!first || !second || !third)
      return gadget.emitOpError(
          "derived CSS H-permutation accepts only the exact SWAP CX network");
    std::array<CXOp, 3> group{first, second, third};
    std::array<std::pair<int64_t, int64_t>, 3> pairs;
    for (auto [index, cx] : llvm::enumerate(group)) {
      auto relation = cx.getPairs();
      auto parsed = relation ? parsePair(*relation) : std::nullopt;
      if (cx.getPatches().size() != 1 || cx.getPatches().front() != current ||
          cx.getNumResults() != 1 || cx.getCtrl() != Partition::data ||
          cx.getTarg() != Partition::data || !parsed || parsed->first < 0 ||
          parsed->second < 0 || parsed->first >= n || parsed->second >= n ||
          parsed->first == parsed->second)
        return gadget.emitOpError(
            "derived CSS H-permutation has an invalid SWAP carrier relation");
      pairs[index] = *parsed;
      current = cx.getResult(0);
    }
    if (pairs[1] !=
            std::pair<int64_t, int64_t>{pairs[0].second, pairs[0].first} ||
        pairs[2] != pairs[0])
      return gadget.emitOpError(
          "derived CSS H-permutation requires CX(a,b), CX(b,a), CX(a,b)");
    swaps.push_back(pairs[0]);
  }
  if (returnOp.getOperand(0) != current)
    return gadget.emitOpError(
        "derived CSS H-permutation must return the final SWAP successor");
  if (isBare)
    return success();

  SmallVector<int64_t> permutation;
  for (int64_t index = 0; index < n; ++index)
    permutation.push_back(index);
  for (auto [left, right] : swaps)
    for (int64_t &destination : permutation) {
      if (destination == left)
        destination = right;
      else if (destination == right)
        destination = left;
    }

  auto matrix =
      [&](StringRef name,
          int64_t rows) -> FailureOr<SmallVector<llvm::SmallBitVector>> {
    auto attr = code->getAttrOfType<DenseIntElementsAttr>(name);
    if (!attr || attr.getType().getRank() != 2 ||
        !attr.getType().getElementType().isInteger(1) ||
        attr.getType().getShape()[0] != rows ||
        attr.getType().getShape()[1] != 2 * n)
      return failure();
    SmallVector<llvm::SmallBitVector> result;
    auto values = attr.getValues<APInt>();
    auto iterator = values.begin();
    for (int64_t row = 0; row < rows; ++row) {
      llvm::SmallBitVector bits(2 * n);
      for (int64_t column = 0; column < 2 * n; ++column, ++iterator)
        if (!(*iterator).isZero())
          bits.set(column);
      result.push_back(std::move(bits));
    }
    return result;
  };
  auto stabilizers = matrix("stabilizer_basis", n - 1);
  auto logicalX = matrix("logical_x_basis", 1);
  auto logicalZ = matrix("logical_z_basis", 1);
  if (failed(stabilizers) || failed(logicalX) || failed(logicalZ))
    return gadget.emitOpError(
        "derived CSS H-permutation requires canonical code symplectic bases");

  auto rank = [](ArrayRef<llvm::SmallBitVector> source) {
    SmallVector<llvm::SmallBitVector> rows(source.begin(), source.end());
    int64_t value = 0;
    int64_t width = rows.empty() ? 0 : rows.front().size();
    for (int64_t column = width - 1; column >= 0; --column) {
      auto pivot =
          llvm::find_if(llvm::drop_begin(rows, value),
                        [&](const auto &row) { return row.test(column); });
      if (pivot == rows.end())
        continue;
      std::iter_swap(rows.begin() + value, pivot);
      for (int64_t index = 0; index < static_cast<int64_t>(rows.size());
           ++index)
        if (index != value && rows[index].test(column))
          rows[index] ^= rows[value];
      if (++value == static_cast<int64_t>(rows.size()))
        break;
    }
    return value;
  };
  auto transform = [&](const llvm::SmallBitVector &row) {
    llvm::SmallBitVector result(2 * n);
    for (int64_t source = 0; source < n; ++source) {
      int64_t destination = permutation[source];
      if (row.test(source))
        result.set(n + destination);
      if (row.test(n + source))
        result.set(destination);
    }
    return result;
  };
  int64_t stabilizerRank = rank(*stabilizers);
  auto rowsInStabilizerSpan = [&](ArrayRef<llvm::SmallBitVector> rows) {
    SmallVector<llvm::SmallBitVector> extended(stabilizers->begin(),
                                               stabilizers->end());
    llvm::append_range(extended, rows);
    return rank(extended) == stabilizerRank;
  };
  SmallVector<llvm::SmallBitVector> transformedStabilizers;
  transformedStabilizers.reserve(stabilizers->size());
  llvm::transform(*stabilizers, std::back_inserter(transformedStabilizers),
                  transform);
  if (!rowsInStabilizerSpan(transformedStabilizers))
    return gadget.emitOpError(
        "derived CSS H-permutation does not preserve the stabilizer group");
  llvm::SmallBitVector xDifference = transform(logicalX->front());
  xDifference ^= logicalZ->front();
  llvm::SmallBitVector zDifference = transform(logicalZ->front());
  zDifference ^= logicalX->front();
  std::array<llvm::SmallBitVector, 2> logicalDifferences{
      std::move(xDifference), std::move(zDifference)};
  if (!rowsInStabilizerSpan(logicalDifferences))
    return gadget.emitOpError(
        "derived CSS H-permutation does not implement logical H");
  return success();
}

static LogicalResult verifyDerivedBareCZ(GadgetOp gadget, GadgetSpecOp spec) {
  auto equivalence = spec ? spec.getActionEquivalence() : std::nullopt;
  if (!equivalence || *equivalence != "derived_exact_bare_physical_cz")
    return success();
  if (gadgetLogicalAction(gadget) != std::optional<StringRef>("cz"))
    return gadget.emitOpError(
        "derived bare physical CZ requires the typed logical CZ objective");
  auto semantic = straightLineBody(gadget);
  auto cz = semantic.size() == 1 ? dyn_cast<CZOp>(semantic.front()) : CZOp{};
  auto returnOp = dyn_cast<ReturnOp>(gadget.getBody().front().getTerminator());
  if (!cz || cz.getCtrl() != Partition::data ||
      cz.getTarg() != Partition::data || !cz.getPairs() ||
      !isFullIdentityDataRelation(cz, *cz.getPairs(), cz.getPatches()) ||
      cz.getPatches() != gadget.getBody().front().getArguments() || !returnOp ||
      returnOp.getOperands() != cz.getResults())
    return gadget.emitOpError(
        "derived bare physical CZ requires one full indexwise data CZ and "
        "ordered boundary successors");
  auto patch = dyn_cast<PatchType>(cz.getPatches().front().getType());
  auto code =
      patch ? dyn_cast_or_null<CodeOp>(SymbolTable::lookupNearestSymbolFrom(
                  gadget, patch.getCodeType()))
            : CodeOp{};
  if (!code || code.getN().value_or(0) != 1 || code.getK().value_or(1) != 1 ||
      code.getR().value_or(0) != 0 ||
      (code.getHx() && !code.getHx()->empty()) ||
      (code.getHz() && !code.getHz()->empty()))
    return gadget.emitOpError(
        "carrierwise CZ is a logical-CZ derivation only for a bare one-qubit "
        "code");
  return success();
}

static LogicalResult verifyDerivedCZCallComposition(GadgetOp gadget,
                                                    GadgetSpecOp spec) {
  auto equivalence = spec ? spec.getActionEquivalence() : std::nullopt;
  if (!equivalence || *equivalence != "derived_exact_clifford_call_composition")
    return success();
  if (gadgetLogicalAction(gadget) != std::optional<StringRef>("cz"))
    return gadget.emitOpError(
        "derived Clifford call composition requires logical CZ");
  auto semantic = straightLineBody(gadget);
  if (semantic.size() != 3 || !llvm::all_of(semantic, [](Operation *operation) {
        return isa<CallOp>(operation);
      }))
    return gadget.emitOpError(
        "derived logical CZ requires the exact logical H-CX-H call sequence");
  auto firstH = cast<CallOp>(semantic[0]);
  auto cx = cast<CallOp>(semantic[1]);
  auto secondH = cast<CallOp>(semantic[2]);
  auto resolve = [](CallOp call) {
    return dyn_cast_or_null<GadgetOp>(
        SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
  };
  auto hGadget = resolve(firstH);
  auto cxGadget = resolve(cx);
  auto secondHGadget = resolve(secondH);
  if (!hGadget || hGadget != secondHGadget || !cxGadget ||
      gadgetLogicalAction(hGadget) != std::optional<StringRef>("h") ||
      (!hasGadgetEquivalence(hGadget,
                             "derived_exact_signed_symplectic_match") &&
       !hasGadgetEquivalence(hGadget,
                             "derived_exact_css_transversal_h_permutation")) ||
      gadgetLogicalAction(cxGadget) != std::optional<StringRef>("cx") ||
      !hasGadgetEquivalence(cxGadget, "derived_exact_css_transversal_cx"))
    return gadget.emitOpError(
        "derived logical CZ requires independently verified logical H and "
        "CSS transversal CX callees");
  Block &entry = gadget.getBody().front();
  auto returnOp = dyn_cast<ReturnOp>(entry.getTerminator());
  if (entry.getNumArguments() != 2 || firstH.getNumOperands() != 1 ||
      firstH.getNumResults() != 1 ||
      firstH.getOperand(0) != entry.getArgument(1) ||
      cx.getNumOperands() != 2 || cx.getNumResults() != 2 ||
      cx.getOperand(0) != entry.getArgument(0) ||
      cx.getOperand(1) != firstH.getResult(0) ||
      secondH.getNumOperands() != 1 || secondH.getNumResults() != 1 ||
      secondH.getOperand(0) != cx.getResult(1) || !returnOp ||
      returnOp.getNumOperands() != 2 ||
      returnOp.getOperand(0) != cx.getResult(0) ||
      returnOp.getOperand(1) != secondH.getResult(0))
    return gadget.emitOpError(
        "derived logical CZ H-CX-H calls have incorrect owner wiring");
  return success();
}

static LogicalResult verifyAutomorphismObjective(Operation *realization,
                                                 Operation *diagnosticOwner,
                                                 GadgetSpecOp spec) {
  SmallVector<PermuteOp> permutations;
  realization->walk(
      [&](PermuteOp operation) { permutations.push_back(operation); });
  auto fail = [&](const Twine &message) {
    return diagnosticOwner->emitOpError() << message;
  };
  if (permutations.empty()) {
    auto equivalence = spec ? spec.getActionEquivalence() : std::nullopt;
    bool separatelyVerifiedClifford =
        equivalence &&
        (*equivalence == "derived_exact_css_transversal_cx" ||
         *equivalence == "derived_exact_css_transversal_h_permutation" ||
         *equivalence == "derived_exact_bare_physical_cz" ||
         *equivalence == "derived_exact_clifford_call_composition");
    if (spec && !separatelyVerifiedClifford &&
        (spec->hasAttr("action_equivalence") || spec->hasAttr("epoch_map")))
      return fail("action_equivalence and epoch_map require one verified "
                  "fabric.permute realization");
    return success();
  }
  if (permutations.size() != 1)
    return fail("automorphism realization must contain exactly one "
                "fabric.permute; compose permutations before declaring "
                "objective equivalence");
  // A partially linked or legacy gadget without a spec is still checked by
  // PermuteOp's local canonical-code verifier.  Objective equivalence remains
  // a link-time obligation until a GadgetSpec is present.
  if (!spec)
    return success();

  auto equivalence = spec->getAttrOfType<StringAttr>("action_equivalence");
  auto epochMap = spec->getAttrOfType<StringAttr>("epoch_map");
  if (!equivalence ||
      equivalence.getValue() != "derived_exact_signed_symplectic_match" ||
      !epochMap || epochMap.getValue() != "preserve")
    return fail("automorphism GadgetSpec requires derived exact signed-"
                "symplectic equivalence and a preserved epoch");

  PermuteOp permutation = permutations.front();
  auto binding =
      permutation->getAttrOfType<DictionaryAttr>("objective_binding");
  auto specBinding = spec->getAttrOfType<DictionaryAttr>("logical_ports");
  if (!binding || !specBinding || binding != specBinding)
    return fail("GadgetSpec logical_ports must equal the verified "
                "fabric.permute objective_binding");

  auto *objectiveTarget =
      SymbolTable::lookupNearestSymbolFrom(spec, spec.getObjectiveAttr());
  auto objective = dyn_cast_or_null<ObjectiveOp>(objectiveTarget);
  if (!objective || !objective.getLogicalAttr())
    return fail("automorphism GadgetSpec objective must resolve through "
                "fabric.objective to a typed logical action");
  auto *logicalTarget = SymbolTable::lookupNearestSymbolFrom(
      objective, objective.getLogicalAttr());
  auto logicalAction = dyn_cast_or_null<::qlx::ActionOp>(logicalTarget);
  auto clifford = logicalTarget
                      ? logicalTarget->getAttrOfType<::qlx::CliffordActionAttr>(
                            "clifford_action")
                      : ::qlx::CliffordActionAttr{};
  if (!logicalAction || !clifford)
    return fail("automorphism objective must carry a typed independently "
                "derived #qlx.clifford_action");

  auto patch = cast<PatchType>(permutation.getPatch().getType());
  if (!patch.getEncoding())
    return fail("automorphism objective verification requires an "
                "encoding-qualified patch");
  auto *encodingTarget =
      SymbolTable::lookupNearestSymbolFrom(permutation, patch.getEncoding());
  auto encoding = dyn_cast_or_null<EncodingOp>(encodingTarget);
  if (!encoding)
    return fail("automorphism patch encoding must resolve to "
                "fabric.encoding");

  auto declaredX = permutation->getAttrOfType<ArrayAttr>("logical_x_action");
  auto declaredZ = permutation->getAttrOfType<ArrayAttr>("logical_z_action");
  auto protectedAttr =
      permutation->getAttrOfType<IntegerAttr>("protected_logicals");
  auto gaugeAttr = permutation->getAttrOfType<IntegerAttr>("gauge_qubits");
  if (!declaredX || !declaredZ || !protectedAttr || !gaugeAttr)
    return fail("fabric.permute is missing typed derived action evidence");
  const int64_t protectedCount = protectedAttr.getInt();
  const int64_t gaugeCount = gaugeAttr.getInt();
  const int64_t totalCount = protectedCount + gaugeCount;
  if (static_cast<int64_t>(encoding.getLogicalPorts().size()) != protectedCount)
    return fail("encoding logical-port count contradicts the permutation's "
                "protected logical count");

  ArrayAttr objectivePorts = clifford.getPorts();
  const int64_t arity = objectivePorts.size();
  const int64_t objectiveWidth = 2 * arity;
  ArrayRef<int64_t> objectiveMatrix = clifford.getMatrix().asArrayRef();
  ArrayRef<int64_t> phases = clifford.getPhases().asArrayRef();
  if (static_cast<int64_t>(binding.size()) != arity)
    return fail("objective_binding must cover every typed objective port "
                "exactly once");

  SmallVector<int64_t> objectiveToCode(arity, -1);
  llvm::SmallDenseSet<int64_t, 8> selectedCodePorts;
  for (auto [objectiveIndex, rawPort] : llvm::enumerate(objectivePorts)) {
    auto objectivePort = dyn_cast<StringAttr>(rawPort);
    if (!objectivePort)
      return fail("typed Clifford objective ports must be strings");
    auto boundPort = binding.getAs<StringAttr>(objectivePort.getValue());
    if (!boundPort)
      return fail("objective_binding is missing typed objective port '" +
                  objectivePort.getValue() + "'");
    int64_t codeIndex = -1;
    for (auto [index, rawEncodingPort] :
         llvm::enumerate(encoding.getLogicalPorts())) {
      auto encodingPort = dyn_cast<StringAttr>(rawEncodingPort);
      if (encodingPort && encodingPort.getValue() == boundPort.getValue()) {
        codeIndex = index;
        break;
      }
    }
    if (codeIndex < 0)
      return fail("objective_binding names an unknown encoding logical port");
    if (!selectedCodePorts.insert(codeIndex).second)
      return fail("objective_binding must map to distinct encoding logical "
                  "ports");
    objectiveToCode[objectiveIndex] = codeIndex;
  }

  SmallVector<int64_t> codeToObjective(protectedCount, -1);
  for (auto [objectiveIndex, codeIndex] : llvm::enumerate(objectiveToCode))
    codeToObjective[codeIndex] = objectiveIndex;
  auto rowSupport = [&](ArrayAttr rows,
                        int64_t index) -> FailureOr<llvm::SmallBitVector> {
    auto row = dyn_cast<DenseI64ArrayAttr>(rows[index]);
    if (!row)
      return failure();
    llvm::SmallBitVector support(totalCount);
    for (int64_t entry : row.asArrayRef()) {
      if (entry < 0 || entry >= totalCount || support.test(entry))
        return failure();
      support.set(entry);
    }
    return support;
  };
  auto objectiveBit = [&](int64_t row, int64_t column) {
    return objectiveMatrix[row * objectiveWidth + column] != 0;
  };

  for (int64_t codeSource = 0; codeSource < protectedCount; ++codeSource) {
    const int64_t objectiveSource = codeToObjective[codeSource];
    for (int64_t basis = 0; basis < 2; ++basis) {
      auto actual = rowSupport(basis == 0 ? declaredX : declaredZ, codeSource);
      if (failed(actual))
        return fail("fabric.permute action rows must be unique in-range dense "
                    "i64 arrays");
      llvm::SmallBitVector expectedX(totalCount);
      llvm::SmallBitVector expectedZ(totalCount);
      int64_t expectedPhase = 0;
      if (objectiveSource < 0) {
        (basis == 0 ? expectedX : expectedZ).set(codeSource);
      } else {
        const int64_t objectiveRow =
            basis == 0 ? objectiveSource : arity + objectiveSource;
        expectedPhase = phases[objectiveRow];
        for (int64_t objectiveOutput = 0; objectiveOutput < arity;
             ++objectiveOutput) {
          const int64_t codeOutput = objectiveToCode[objectiveOutput];
          if (objectiveBit(objectiveRow, objectiveOutput))
            expectedX.set(codeOutput);
          if (objectiveBit(objectiveRow, arity + objectiveOutput))
            expectedZ.set(codeOutput);
        }
      }
      const llvm::SmallBitVector &sameBasis =
          basis == 0 ? expectedX : expectedZ;
      const llvm::SmallBitVector &mixedBasis =
          basis == 0 ? expectedZ : expectedX;
      if (expectedPhase != 0 || mixedBasis.any() || *actual != sameBasis)
        return fail("fabric.permute action does not implement the typed "
                    "objective's exact signed-symplectic action");
    }
  }
  for (int64_t gauge = protectedCount; gauge < totalCount; ++gauge) {
    for (ArrayAttr rows : {declaredX, declaredZ}) {
      auto support = rowSupport(rows, gauge);
      if (failed(support))
        return fail("fabric.permute gauge action row is malformed");
      for (int64_t logical = 0; logical < protectedCount; ++logical)
        if (support->test(logical))
          return fail("a gauge output may not hide a protected logical "
                      "action");
    }
  }
  return success();
}

LogicalResult GadgetOp::verify() {
  // If entry is set, device must also be set.
  if (getEntry() && !getDevice())
    return emitOpError("entry gadget must specify a device");

  // Verify return types match function_type results.
  auto funcType = getFunctionType();
  if (failed(verifyEncodingQualifiedTypes(*this, funcType.getInputs())) ||
      failed(verifyEncodingQualifiedTypes(*this, funcType.getResults())))
    return failure();
  if (failed(verifyGeneratedSpecialization(*this)))
    return failure();
  GadgetSpecOp spec;
  if (auto specRef = getSpecAttr()) {
    if (Operation *target =
            SymbolTable::lookupNearestSymbolFrom(*this, specRef)) {
      spec = dyn_cast<GadgetSpecOp>(target);
      if (!spec)
        return emitOpError("spec must resolve to fabric.gadget_spec");
      if (spec.getFunctionType() != funcType)
        return emitOpError(
            "gadget signature must exactly match its GadgetSpec realization "
            "signature");
      if (failed(verifyGadgetSpecRealizationBoundary(*this, spec)))
        return failure();
    }
  }

  auto verifyRecordProducers = [&]() -> LogicalResult {
    if (!spec)
      return success();
    // Keep the optional alive across the range loop. Dereferencing the
    // temporary accessor result would leave its contained ArrayAttr dangling.
    auto recordSchema = spec.getRecordSchema();
    if (!recordSchema)
      return success();
    SmallVector<std::string> demandedPaths;
    demandedPaths.reserve(recordSchema->size());
    for (Attribute raw : *recordSchema)
      demandedPaths.push_back(cast<StringAttr>(raw).getValue().str());
    auto manifest = collectRecursiveRecordManifest(*this, false, demandedPaths);
    if (failed(manifest))
      return failure();
    const llvm::StringMap<unsigned> &produced = manifest->typedProducerCounts;
    for (Attribute raw : *recordSchema) {
      auto record = cast<StringAttr>(raw);
      StringRef path = record.getValue();
      unsigned producerCount = produced.lookup(path);
      if (producerCount == 0)
        return emitOpError("GadgetSpec record '")
               << path
               << "' is not an exact output of a typed record-producing "
                  "operation in the realization";
      if (producerCount != 1)
        return emitOpError("GadgetSpec record '")
               << path << "' has " << producerCount
               << " typed producers; stable record paths require exactly one";
    }
    return success();
  };
  auto &body = getBody();
  if (!llvm::hasSingleElement(body))
    return emitOpError("requires exactly one signature block");
  if (body.front().getArgumentTypes() != funcType.getInputs())
    return emitOpError("entry block arguments must match function_type inputs");

  auto bodyReturn = dyn_cast<ReturnOp>(body.front().getTerminator());
  bool bodyHasOperations = !body.front().without_terminator().empty() ||
                           (bodyReturn && !bodyReturn.getOperands().empty());
  if (auto realization = getRealizationAttr()) {
    if (bodyHasOperations)
      return emitOpError(
          "has both an inline body and a realization circuit reference");
    Operation *target =
        SymbolTable::lookupNearestSymbolFrom(*this, realization);
    if (!target)
      return success(); // Partially linked modules retain the obligation.
    auto circuit = dyn_cast<CircuitOp>(target);
    if (!circuit)
      return emitOpError("realization must resolve to fabric.circuit");
    if (circuit.getFunctionType() != funcType)
      return emitOpError(
          "realization circuit signature must match the gadget signature");
    if (getRealizationKind() && getRealizationKind() != "circuit")
      return emitOpError(
          "a referenced realization requires realization_kind = circuit");
    if (failed(verifyRecordProducers()))
      return failure();
    if (failed(verifyTerminalMeasurementOwnership(circuit.getOperation())))
      return failure();
    if (failed(verifyDerivedCSSCX(*this, spec)) ||
        failed(verifyDerivedCSSHPermutation(*this, spec)) ||
        failed(verifyDerivedBareCZ(*this, spec)) ||
        failed(verifyDerivedCZCallComposition(*this, spec)))
      return failure();
    if (failed(verifyAutomorphismObjective(circuit.getOperation(),
                                           getOperation(), spec)))
      return failure();
    auto circuitReturn =
        dyn_cast<ReturnOp>(circuit.getBody().front().getTerminator());
    if (spec && circuitReturn &&
        failed(
            verifyOutcomeReturnProvenance(getOperation(), circuitReturn, spec)))
      return failure();
    return success();
  }

  auto returnOp = dyn_cast<ReturnOp>(body.front().getTerminator());
  if (!returnOp)
    return success();

  auto returnTypes = returnOp.getOperandTypes();
  auto resultTypes = funcType.getResults();
  if (returnTypes.size() != resultTypes.size())
    return returnOp.emitOpError("return operand count (")
           << returnTypes.size() << ") does not match gadget result count ("
           << resultTypes.size() << ")";

  for (unsigned i = 0; i < returnTypes.size(); ++i) {
    if (returnTypes[i] != resultTypes[i])
      return returnOp.emitOpError("return operand type mismatch at index ")
             << i << ": expected " << resultTypes[i] << ", got "
             << returnTypes[i];
  }
  if (failed(verifyRecordProducers()))
    return failure();
  if (failed(verifyTerminalMeasurementOwnership(getOperation())))
    return failure();
  if (failed(verifyDerivedCSSCX(*this, spec)) ||
      failed(verifyDerivedCSSHPermutation(*this, spec)) ||
      failed(verifyDerivedBareCZ(*this, spec)) ||
      failed(verifyDerivedCZCallComposition(*this, spec)))
    return failure();
  if (failed(verifyAutomorphismObjective(getOperation(), getOperation(), spec)))
    return failure();
  if (spec &&
      failed(verifyOutcomeReturnProvenance(getOperation(), returnOp, spec)))
    return failure();
  return success();
}

static std::optional<unsigned> builtinActionArity(qlx::BuiltinAction action);

static LogicalResult verifyRetainedStreamProtocols(ProtocolOp protocol) {
  auto module = protocol->getParentOfType<ModuleOp>();
  if (!module)
    return success();

  SmallVector<qlx::lvm::StreamOp, 2> producerStreams;
  SmallVector<qlx::lvm::StreamOp, 2> transferStreams;
  // Streams are direct members of their lvm.domain symbol table.  Walking the
  // complete module once for every protocol made verification quadratic in
  // large QEC-lowered programs even though the stream inventory is tiny.
  // Restrict discovery to the typed domain boundary while retaining the exact
  // same symbol-resolution proof for every stream.
  for (auto domain : module.getOps<qlx::lvm::DomainOp>()) {
    for (Operation &candidate : domain.getBody().front()) {
      auto stream = dyn_cast<qlx::lvm::StreamOp>(candidate);
      if (!stream)
        continue;
      auto resolvesToProtocol = [&](FlatSymbolRefAttr reference) {
        if (!reference)
          return false;
        for (Operation *scope = stream; scope; scope = scope->getParentOp()) {
          if (!scope->hasTrait<OpTrait::SymbolTable>())
            continue;
          if (Operation *target =
                  SymbolTable::lookupSymbolIn(scope, reference.getValue()))
            return target == protocol.getOperation();
        }
        return false;
      };
      if (resolvesToProtocol(stream.getProducedByAttr()))
        producerStreams.push_back(stream);
      if (resolvesToProtocol(stream.getTransferAttr()))
        transferStreams.push_back(stream);
    }
  }
  if (producerStreams.empty() && transferStreams.empty())
    return success();

  Attribute objectiveAttr = protocol.getObjectiveAttr();
  auto objectiveRef = dyn_cast_or_null<SymbolRefAttr>(objectiveAttr);
  auto *objectiveTarget =
      objectiveRef
          ? SymbolTable::lookupNearestSymbolFrom(protocol, objectiveRef)
          : nullptr;
  auto objective = dyn_cast_or_null<::qlx::ActionOp>(objectiveTarget);
  auto builtinObjective =
      dyn_cast_or_null<qlx::BuiltinActionAttr>(objectiveAttr);
  if (!objective && !builtinObjective)
    return protocol.emitOpError(
        "a retained stream protocol requires a typed action objective");
  if (objective && objective.getFunctionType() != protocol.getFunctionType())
    return protocol.emitOpError(
        "a retained stream protocol function type must exactly match its "
        "qlx.action objective");

  auto verifyRole = [&](qlx::lvm::StreamOp stream, StringRef role,
                        bool producer) -> LogicalResult {
    FunctionType signature = protocol.getFunctionType();
    auto matchesResource = [&](Type type) {
      auto resource = dyn_cast<ResourceStateType>(type);
      return resource && resource.getKind() == stream.getProducesAttr();
    };
    if (producer || objective) {
      bool validBoundary = producer
                               ? signature.getNumInputs() == 0 &&
                                     signature.getNumResults() == 1 &&
                                     matchesResource(signature.getResult(0))
                               : signature.getNumInputs() == 1 &&
                                     signature.getNumResults() == 1 &&
                                     matchesResource(signature.getInput(0)) &&
                                     matchesResource(signature.getResult(0));
      if (!objective || !validBoundary)
        return protocol.emitOpError()
               << "retained stream " << role
               << " function type must have the exact resource-flow boundary "
                  "for "
               << stream.getProducesAttr();

      std::string expectedKind = (producer ? "produce_" : "transport_") +
                                 stream.getProducesAttr().getValue().str();
      if (objective.getKind() != expectedKind)
        return protocol.emitOpError()
               << "retained stream " << role << " objective must be the exact '"
               << expectedKind << "' qlx.action";
      return success();
    }

    auto arity = builtinActionArity(builtinObjective.getValue());
    if (!arity)
      return protocol.emitOpError(
          "retained stream transfer requires a fixed-arity consumer action");
    unsigned resourceInputs = 0;
    unsigned patchInputs = 0;
    bool validBoundary = true;
    for (Type type : signature.getInputs()) {
      if (isa<ResourceStateType>(type)) {
        ++resourceInputs;
        validBoundary &= matchesResource(type);
      } else {
        validBoundary &= isa<PatchType>(type);
        patchInputs += isa<PatchType>(type);
      }
    }
    unsigned patchResults = 0;
    for (Type type : signature.getResults()) {
      validBoundary &= isa<PatchType>(type);
      patchResults += isa<PatchType>(type);
    }
    validBoundary &=
        resourceInputs == 1 && patchInputs == *arity && patchResults == *arity;
    if (!validBoundary)
      return protocol.emitOpError()
             << "retained stream " << role
             << " consumer protocol must consume exactly one matching resource "
                "and preserve the typed action patch arity";

    LogicalResult exactObjective = success();
    module.walk([&](qlx::lvm::ConsumeResourceOp consume) {
      Operation *target = SymbolTable::lookupNearestSymbolFrom(
          consume, consume.getResourceStreamAttr());
      if (target == stream.getOperation() &&
          consume.getActionAttr() != builtinObjective)
        exactObjective = failure();
    });
    if (failed(exactObjective))
      return protocol.emitOpError(
          "retained stream transfer objective must exactly equal its P1 "
          "resource-consume action");
    return success();
  };

  for (auto stream : producerStreams)
    if (failed(verifyRole(stream, "produced_by", true)))
      return failure();
  for (auto stream : transferStreams)
    if (failed(verifyRole(stream, "transfer", false)))
      return failure();

  for (auto stream : producerStreams) {
    auto backing = stream.getBackingRegionAttr();
    if (!backing)
      continue;
    auto domain = stream->getParentOfType<qlx::lvm::DomainOp>();
    Operation *backingTarget =
        domain ? SymbolTable::lookupSymbolIn(domain, backing.getValue())
               : nullptr;
    if (!isa_and_nonnull<qlx::lvm::SpaceOp>(backingTarget))
      return protocol.emitOpError(
          "retained producer backing_region must resolve to lvm.space");
    SmallVector<Operation *, 2> factoryReferences;
    protocol.getBody().walk([&](Operation *nested) {
      StringRef name = nested->getName().getStringRef();
      if (name != "fabric.alloc" && name != "fabric.produce_resource")
        return;
      if (auto region = nested->getAttrOfType<FlatSymbolRefAttr>("region"))
        factoryReferences.push_back(
            SymbolTable::lookupSymbolIn(domain, region.getValue()));
    });
    if (factoryReferences.empty())
      return protocol.emitOpError(
          "a backed retained stream producer must carry an authoritative "
          "factory-region reference in its protocol body");
    if (llvm::any_of(factoryReferences, [&](Operation *region) {
          return region != backingTarget;
        }))
      return protocol.emitOpError(
          "retained producer factory-region references must exactly equal "
          "the stream backing_region");
  }
  return success();
}

// Forward declarations: both are defined further below (the shared helper's
// natural home stays with `RetryOp::verify()`'s call site; the embedded-
// selection walk stays with the deleted `Fabric_SelectionOp::verify()`'s old
// location), but `ProtocolOp::verify()` needs to call the latter here.
static LogicalResult verifySelectedPredicateSemantics(
    Operation *owner, Value predicate, ValueRange carries,
    FlatSymbolRefAttr attemptAttr, FlatSymbolRefAttr profileAttr,
    GadgetOp gadget, GadgetProfileOp profile, GadgetSpecOp spec,
    StringRef operation);
static LogicalResult verifyEmbeddedSelections(Operation *protocolBody);
static LogicalResult verifyLocalGeneratedRPPProtocol(ProtocolOp protocol);
static LogicalResult verifyLocalGeneratedRPPInvocations(ProtocolOp protocol);

LogicalResult ProtocolOp::verify() {
  auto functionType = getFunctionType();
  if (failed(verifyEncodingQualifiedTypes(*this, functionType.getInputs())) ||
      failed(verifyEncodingQualifiedTypes(*this, functionType.getResults())))
    return failure();
  unsigned p1CallAttrs =
      static_cast<unsigned>(static_cast<bool>(getInputP1KernelAttr())) +
      static_cast<unsigned>(static_cast<bool>(getInputP1CalleeAttr())) +
      static_cast<unsigned>(static_cast<bool>(getInputP1ScopeAttr()));
  if (p1CallAttrs != 0 && p1CallAttrs != 3)
    return emitOpError(
        "P1 call provenance requires input_p1_kernel, input_p1_callee, and "
        "input_p1_scope together");
  if (p1CallAttrs == 3) {
    if (getInputP1ScopeAttr().getInt() < 0)
      return emitOpError("input_p1_scope must be nonnegative");
    auto kernel = dyn_cast_or_null<qlx::lvm::KernelOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, getInputP1KernelAttr()));
    if (!kernel)
      return emitOpError("input_p1_kernel must resolve to lvm.kernel");
    auto callee = dyn_cast_or_null<qlx::ProgramOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, getInputP1CalleeAttr()));
    if (!callee)
      return emitOpError("input_p1_callee must resolve to qlx.program");
    unsigned matches = 0;
    qlx::lvm::CallOp matchedCall;
    kernel.getBody().walk([&](qlx::lvm::CallOp call) {
      if (call.getScope() == getInputP1ScopeAttr().getInt() &&
          call.getCalleeAttr() == getInputP1CalleeAttr()) {
        ++matches;
        matchedCall = call;
      }
    });
    if (matches != 1)
      return emitOpError(
                 "P1 call provenance must resolve to exactly one retained "
                 "lvm.call; found ")
             << matches;
    if (failed(qlx::lvm::verifyP0CallBodyRefinement(matchedCall)))
      return emitOpError(
          "P1 call provenance resolves to a body that does not refine P0");
  }
  if (failed(verifyRetainedStreamProtocols(*this)))
    return failure();
  if (auto objective = dyn_cast_or_null<SymbolRefAttr>(getObjectiveAttr())) {
    if (auto *target = SymbolTable::lookupNearestSymbolFrom(*this, objective)) {
      auto typeAttr = target->getAttrOfType<TypeAttr>("function_type");
      auto objectiveType = typeAttr
                               ? dyn_cast<FunctionType>(typeAttr.getValue())
                               : FunctionType();
      bool producesResource =
          objectiveType && objectiveType.getNumInputs() == 0 &&
          objectiveType.getNumResults() == 1 &&
          isa<ResourceStateType>(objectiveType.getResult(0));
      if (producesResource) {
        if (functionType.getResults() != objectiveType.getResults())
          return emitOpError(
              "result types must match the production objective");
      }
    }
  }
  if (failed(verifyGeneratedSpecialization(*this)))
    return failure();
  if (failed(verifyLocalGeneratedRPPProtocol(*this)))
    return failure();
  auto metadata = getMetadataAttr();
  Attribute rawInputP1 = metadata ? metadata.get("input_p1") : Attribute();
  Attribute rawSelectionCommitment =
      metadata ? metadata.get("qec_selection_sha256") : Attribute();
  auto inputP1 = rawInputP1 ? dyn_cast<StringAttr>(rawInputP1) : StringAttr();
  auto selectionCommitment = rawSelectionCommitment
                                 ? dyn_cast<StringAttr>(rawSelectionCommitment)
                                 : StringAttr();
  if (rawInputP1 && (!inputP1 || inputP1.getValue().empty()))
    return emitOpError("metadata input_p1 must be a nonempty string");
  if (rawSelectionCommitment && !selectionCommitment)
    return emitOpError("metadata qec_selection_sha256 must be a string");
  if (inputP1 && !selectionCommitment)
    return emitOpError("metadata with input_p1 requires qec_selection_sha256");
  if (!inputP1 && selectionCommitment)
    return emitOpError("metadata qec_selection_sha256 requires input_p1");
  if (selectionCommitment) {
    StringRef value = selectionCommitment.getValue();
    StringRef digest = value.consume_front("sha256:") ? value : StringRef();
    if (digest.size() != 64 || !llvm::all_of(digest, [](char character) {
          return (character >= '0' && character <= '9') ||
                 (character >= 'a' && character <= 'f');
        }))
      return emitOpError(
          "metadata qec_selection_sha256 must be 'sha256:' followed by 64 "
          "lowercase hexadecimal digits");
  }
  if (getBody().empty())
    return emitOpError("requires one entry block");
  auto &block = getBody().front();
  if (block.getArgumentTypes() != functionType.getInputs())
    return emitOpError("entry block arguments must match function_type inputs");
  auto returnOp = dyn_cast<ProtocolReturnOp>(block.getTerminator());
  if (!returnOp)
    return emitOpError("must terminate with fabric.protocol_return");
  if (returnOp.getOperandTypes() != functionType.getResults())
    return returnOp.emitOpError(
        "operand types must match enclosing protocol result types");
  bool hasPredicateGadget = static_cast<bool>(getPredicateGadgetAttr());
  bool hasPredicateProfile = static_cast<bool>(getPredicateProfileAttr());
  bool hasPredicateResult = static_cast<bool>(getPredicateResultAttr());
  if ((hasPredicateGadget || hasPredicateProfile || hasPredicateResult) &&
      !(hasPredicateGadget && hasPredicateProfile && hasPredicateResult))
    return emitOpError(
        "predicate provenance requires gadget, profile, and result together");
  if (hasPredicateGadget) {
    int64_t resultIndex = getPredicateResultAttr().getInt();
    if (resultIndex < 0 || resultIndex >= functionType.getNumResults() ||
        !functionType.getResult(resultIndex).isInteger(1))
      return emitOpError("predicate_result must name an i1 protocol result");
    auto gadget = dyn_cast_or_null<GadgetOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, getPredicateGadgetAttr()));
    auto profile = dyn_cast_or_null<GadgetProfileOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, getPredicateProfileAttr()));
    if (!gadget || !profile ||
        profile.getGadgetAttr() != getPredicateGadgetAttr())
      return emitOpError(
          "predicate provenance must name a profile for its exact gadget");
    Value returned = returnOp.getOperand(resultIndex);
    auto result = dyn_cast<OpResult>(returned);
    auto call = returned.getDefiningOp<CallOp>();
    if (!result || !call || call.getCalleeAttr() != getPredicateGadgetAttr() ||
        call.getProfileAttr() != getPredicateProfileAttr())
      return emitOpError(
          "predicate result must be returned directly from the selected "
          "gadget/profile call");
    int64_t protocolBooleanOrdinal = 0;
    for (int64_t index = 0; index < resultIndex; ++index)
      if (functionType.getResult(index).isInteger(1))
        ++protocolBooleanOrdinal;
    int64_t gadgetBooleanOrdinal = 0;
    for (unsigned index = 0; index < result.getResultNumber(); ++index)
      if (call.getResult(index).getType().isInteger(1))
        ++gadgetBooleanOrdinal;
    if (protocolBooleanOrdinal != gadgetBooleanOrdinal)
      return emitOpError(
          "predicate result must preserve the selected gadget Boolean ordinal");
  }
  if (failed(verifyEmbeddedSelections(getOperation())))
    return failure();
  return verifyLocalGeneratedRPPInvocations(*this);
}

struct BooleanAffineExpression {
  SmallVector<Value> terms;
  bool constant = false;
};

static FailureOr<BooleanAffineExpression> normalizeBooleanAffine(Value value) {
  std::function<FailureOr<BooleanAffineExpression>(Value)> normalize =
      [&](Value current) -> FailureOr<BooleanAffineExpression> {
    if (!current.getType().isInteger(1))
      return failure();
    if (isa_and_nonnull<CallOp>(current.getDefiningOp()))
      return BooleanAffineExpression{{current}, false};
    if (auto constant = current.getDefiningOp<arith::ConstantOp>()) {
      auto integer = dyn_cast<IntegerAttr>(constant.getValue());
      if (!integer || integer.getType().getIntOrFloatBitWidth() != 1)
        return failure();
      return BooleanAffineExpression{{}, !integer.getValue().isZero()};
    }

    Value lhs;
    Value rhs;
    if (auto xorOp = current.getDefiningOp<XorOp>()) {
      lhs = xorOp.getLhs();
      rhs = xorOp.getRhs();
    } else if (auto xorOp = current.getDefiningOp<arith::XOrIOp>()) {
      lhs = xorOp.getLhs();
      rhs = xorOp.getRhs();
    } else {
      return failure();
    }
    auto left = normalize(lhs);
    auto right = normalize(rhs);
    if (failed(left) || failed(right))
      return failure();
    BooleanAffineExpression result = *left;
    result.constant ^= right->constant;
    for (Value term : right->terms) {
      auto found = llvm::find(result.terms, term);
      if (found == result.terms.end())
        result.terms.push_back(term);
      else
        result.terms.erase(found);
    }
    return result;
  };
  return normalize(value);
}

static LogicalResult verifySelectedPredicateSemantics(
    Operation *owner, Value predicate, ValueRange carries,
    FlatSymbolRefAttr attemptAttr, FlatSymbolRefAttr profileAttr,
    GadgetOp gadget, GadgetProfileOp profile, GadgetSpecOp spec,
    StringRef operation) {
  SmallVector<SuccessOp> successes;
  profile.getBody().walk(
      [&](SuccessOp success) { successes.push_back(success); });

  auto outcome = spec.getOutcomeMap();
  if (!outcome)
    return owner->emitOpError()
           << operation << " attempt requires a total GadgetSpec outcome_map";
  auto outcomeRecords = outcome->getAs<ArrayAttr>("records");
  auto rows = outcome->getAs<DenseIntElementsAttr>("rows");
  auto constants = outcome->getAs<DenseI64ArrayAttr>("constants");
  if (!outcomeRecords || !rows || !constants)
    return owner->emitOpError()
           << operation
           << " attempt outcome_map must use canonical records, rows, and "
              "constants";

  StringRef gadgetName = gadget.getSymName();
  auto shape = rows.getType().getShape();
  auto denseValues = rows.getValues<APInt>();
  SmallVector<APInt> values(denseValues.begin(), denseValues.end());
  struct SuccessBinding {
    int64_t outcomeRow;
    bool profileConstant;
  };
  SmallVector<SuccessBinding> bindings;
  llvm::SmallDenseSet<int64_t> claimedRows;
  SmallVector<int64_t> authoritativeSuccessRows;
  for (int64_t row = 0; row < shape[0]; ++row)
    if (outcomeRowHasRole(*outcome, row, "success"))
      authoritativeSuccessRows.push_back(row);
  if (authoritativeSuccessRows.empty() && successes.empty())
    return owner->emitOpError("selected ")
           << operation
           << " profile requires an OutcomeMap success role or an explicit "
              "profile success row";
  if (!authoritativeSuccessRows.empty() && !successes.empty() &&
      outcome->getAs<ArrayAttr>("roles") &&
      successes.size() != authoritativeSuccessRows.size())
    return owner->emitOpError()
           << operation << " profile declares " << successes.size()
           << " success row(s), but the OutcomeMap tags "
           << authoritativeSuccessRows.size() << " authoritative row(s)";
  if (successes.empty()) {
    for (int64_t row : authoritativeSuccessRows)
      bindings.push_back({row, constants.asArrayRef()[row] != 0});
  }
  for (SuccessOp profileSuccess : successes) {
    SmallVector<StringRef> successRecords;
    if (auto records = profileSuccess.getRecords())
      for (Attribute raw : *records) {
        auto record = dyn_cast<StringAttr>(raw);
        if (!record)
          return owner->emitOpError()
                 << operation << " success records must be stable string paths";
        StringRef path = record.getValue();
        if (!path.consume_front(gadgetName) || !path.consume_front("."))
          return owner->emitOpError()
                 << operation
                 << " success records must be owned by the attempt gadget";
        successRecords.push_back(path);
      }
    SmallVector<OutcomeSyndromeTerm> successTerms =
        getProfileSyndromeTerms(profileSuccess);
    bool profileConstant = false;
    if (auto constant = profileSuccess.getConstantAttr())
      profileConstant = constant.getValue();

    std::optional<int64_t> matchingRow;
    for (int64_t row = 0; row < shape[0]; ++row) {
      if (!authoritativeSuccessRows.empty() &&
          !outcomeRowHasRole(*outcome, row, "success"))
        continue;
      if (authoritativeSuccessRows.empty() &&
          outcomeRowHasRole(*outcome, row, "result"))
        continue;
      SmallVector<StringRef> rowRecords;
      for (int64_t column = 0; column < shape[1]; ++column)
        if (!values[row * shape[1] + column].isZero())
          rowRecords.push_back(
              cast<StringAttr>(outcomeRecords[column]).getValue());
      if (rowRecords != successRecords ||
          getOutcomeSyndromeTerms(*outcome, row) != successTerms)
        continue;
      if (matchingRow)
        return owner->emitOpError()
               << operation
               << " success semantics match more than one OutcomeMap row";
      matchingRow = row;
    }
    if (!matchingRow)
      return owner->emitOpError()
             << operation
             << " success semantics do not match an exact ordered OutcomeMap "
                "row";
    if (!claimedRows.insert(*matchingRow).second)
      return owner->emitOpError()
             << operation
             << " profile success rows must map one-to-one to distinct "
                "OutcomeMap rows";
    if (!authoritativeSuccessRows.empty() &&
        outcome->getAs<ArrayAttr>("roles") &&
        *matchingRow != authoritativeSuccessRows[bindings.size()])
      return owner->emitOpError()
             << operation
             << " profile success rows must preserve OutcomeMap role order";

    bindings.push_back({*matchingRow, profileConstant});
  }

  CallOp selectedCall;
  auto verifyEvent = [&](Value event, SuccessBinding binding,
                         bool trueOnAccept) -> LogicalResult {
    auto expression = normalizeBooleanAffine(event);
    if (failed(expression) || expression->terms.size() != 1)
      return owner->emitOpError()
             << operation
             << " predicate events must be provable affine call-result "
                "Booleans";
    Value term = expression->terms.front();
    auto result = dyn_cast<OpResult>(term);
    auto call = term.getDefiningOp<CallOp>();
    if (!result || !call || call.getCalleeAttr() != attemptAttr ||
        call.getProfileAttr() != profileAttr)
      return owner->emitOpError()
             << operation
             << " predicate must derive from the selected attempt/profile "
                "call";
    if (selectedCall && selectedCall != call)
      return owner->emitOpError()
             << "all " << operation
             << " predicate events must derive from the same selected attempt "
                "call";
    selectedCall = call;

    int64_t booleanOrdinal = 0;
    bool foundResult = false;
    for (auto [index, type] : llvm::enumerate(call.getResultTypes())) {
      if (index == result.getResultNumber()) {
        if (!type.isInteger(1))
          return owner->emitOpError()
                 << operation << " predicate call result must be i1";
        foundResult = true;
        break;
      }
      if (type.isInteger(1))
        ++booleanOrdinal;
    }
    if (!foundResult || booleanOrdinal != binding.outcomeRow)
      return owner->emitOpError()
             << operation
             << " predicate events must preserve profile success-row order "
                "and identity";

    bool outcomeConstant = constants.asArrayRef()[binding.outcomeRow] != 0;
    bool expectedConstant =
        outcomeConstant ^ binding.profileConstant ^ trueOnAccept;
    if (expression->constant != expectedConstant)
      return owner->emitOpError()
             << operation
             << " predicate polarity contradicts profile success semantics";
    return success();
  };

  if (auto allFalse = predicate.getDefiningOp<AllFalseOp>()) {
    if (allFalse.getEvents().size() != bindings.size())
      return owner->emitOpError("fabric.all_false ")
             << operation
             << " predicates must cover every selected profile success row "
                "exactly once";
    for (auto [event, binding] : llvm::zip(allFalse.getEvents(), bindings))
      if (failed(verifyEvent(event, binding, /*trueOnAccept=*/false)))
        return failure();
  } else {
    if (bindings.size() != 1)
      return owner->emitOpError("multi-row ")
             << operation
             << " success requires an ordered fabric.all_false predicate";
    if (failed(verifyEvent(predicate, bindings.front(),
                           /*trueOnAccept=*/true)))
      return failure();
  }

  llvm::SmallDenseSet<Value, 4> carriedResults;
  for (Value carry : carries) {
    if (!selectedCall || carry.getDefiningOp() != selectedCall.getOperation())
      return owner->emitOpError("every ")
             << operation
             << " carry must originate from the selected attempt call";
    if (!carriedResults.insert(carry).second)
      return owner->emitOpError(operation)
             << " carries must contain each selected attempt patch result "
                "exactly once";
  }
  if (operation == "retry") {
    int64_t patchResultCount = 0;
    for (Value result : selectedCall.getResults())
      if (isa<PatchType>(result.getType())) {
        ++patchResultCount;
        if (!carriedResults.contains(result))
          return owner->emitOpError(operation)
                 << " must carry every linear patch result from the selected "
                    "attempt; sibling cleanup is not a closed replay boundary";
      }
    if (patchResultCount != static_cast<int64_t>(carries.size()))
      return owner->emitOpError(operation)
             << " carries must exactly equal the selected attempt patch "
                "results";
  }
  return success();
}

LogicalResult RetryOp::verify() {
  if (getMaxAttempts() <= 0)
    return emitOpError("max_attempts must be positive");
  if (getExhaustion() != "report_failure" && getExhaustion() != "abort" &&
      getExhaustion() != "return_last")
    return emitOpError(
        "exhaustion must be report_failure, abort, or return_last");
  if (getSuccessProbabilityAttr()) {
    double probability = getSuccessProbabilityAttr().getValueAsDouble();
    if (!std::isfinite(probability) || probability <= 0.0 || probability > 1.0)
      return emitOpError(
          "success_probability must be finite and lie in (0, 1]");
  } else if (getSuccessProbabilitySourceAttr() ||
             getSuccessProbabilityEvidenceAttr()) {
    return emitOpError("probability provenance requires success_probability");
  }
  if (getCarries().size() != getResults().size())
    return emitOpError("must return every linear carry");
  for (auto [input, result] : llvm::zip(getCarries(), getResults()))
    if (input.getType() != result.getType())
      return emitOpError("carry/result patch types must match");
  if (!getAttemptAttr())
    return emitOpError("requires an explicit attempt gadget");
  Operation *attempt =
      SymbolTable::lookupNearestSymbolFrom(*this, getAttemptAttr());
  if (!attempt)
    return emitOpError("attempt must resolve to fabric.gadget or a contracted "
                       "fabric.protocol");
  auto protocol = dyn_cast<ProtocolOp>(attempt);
  auto gadget = dyn_cast<GadgetOp>(attempt);
  if (!gadget && !protocol)
    return emitOpError("attempt must resolve to fabric.gadget or a contracted "
                       "fabric.protocol");
  if (protocol && (!protocol.getPredicateGadgetAttr() ||
                   !protocol.getPredicateProfileAttr() ||
                   !protocol.getPredicateResultAttr()))
    return emitOpError(
        "protocol retry attempts are unsupported until protocols carry a "
        "typed selection and predicate-provenance contract");

  if (!getProfileAttr())
    return emitOpError("requires an explicit selected gadget profile");
  Operation *profileTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, getProfileAttr());
  auto profile = dyn_cast_or_null<GadgetProfileOp>(profileTarget);
  if (!profile)
    return emitOpError("profile must resolve to fabric.gadget_profile");
  if (protocol) {
    if (protocol.getPredicateProfileAttr() != getProfileAttr())
      return emitOpError(
          "retry profile must equal the protocol predicate profile");
    gadget = dyn_cast_or_null<GadgetOp>(SymbolTable::lookupNearestSymbolFrom(
        *this, protocol.getPredicateGadgetAttr()));
    if (!gadget || profile.getGadgetAttr() != protocol.getPredicateGadgetAttr())
      return emitOpError(
          "protocol predicate profile must analyze its contracted gadget");
  } else if (profile.getGadgetAttr() != getAttemptAttr()) {
    return emitOpError("profile must analyze the retry attempt");
  }
  if (!gadget.getSpecAttr())
    return emitOpError("retry requires an attempt gadget with a spec");
  auto *specTarget =
      SymbolTable::lookupNearestSymbolFrom(*this, gadget.getSpecAttr());
  auto spec = dyn_cast_or_null<GadgetSpecOp>(specTarget);
  if (!spec)
    return emitOpError("attempt spec must resolve to fabric.gadget_spec");
  if (spec.getFunctionType() != gadget.getFunctionType())
    return emitOpError(
        "attempt gadget signature must exactly match its GadgetSpec");
  if (failed(verifyGadgetSpecRealizationBoundary(gadget, spec,
                                                 /*requireWitness=*/true)))
    return failure();
  if (auto commitPoint = getCommitPoint()) {
    StringRef value = *commitPoint;
    if (value == "before_output") {
      // The attempt specification itself is the typed output boundary.
    } else if (value.consume_front("before_output:")) {
      if (value.empty())
        return emitOpError(
            "qualified before_output commit point requires an endpoint name");
      bool found = false;
      if (auto ports = spec.getPorts())
        for (Attribute raw : *ports) {
          auto port = dyn_cast<DictionaryAttr>(raw);
          auto name = port ? port.getAs<StringAttr>("name") : StringAttr{};
          auto direction =
              port ? port.getAs<StringAttr>("direction") : StringAttr{};
          if (name && name.getValue() == value && direction &&
              (direction.getValue() == "output" ||
               direction.getValue() == "inout")) {
            found = true;
            break;
          }
        }
      if (!found)
        return emitOpError("commit point names no output endpoint in the "
                           "selected attempt specification: ")
               << value;
    } else if (value == "pack_resource") {
      if (getResults().empty())
        return emitOpError(
            "pack_resource commit point requires a carried patch");
      for (Value result : getResults())
        if (!result.hasOneUse() ||
            !isa<PackResourceOp>(*result.getUsers().begin()))
          return emitOpError(
              "pack_resource commit point requires every retry result to be "
              "consumed directly by fabric.pack_resource");
    } else {
      return emitOpError(
          "commit_point must be before_output, before_output:<endpoint>, or "
          "pack_resource");
    }
  }
  if (failed(verifySelectedPredicateSemantics(
          getOperation(), getSuccess(), getCarries(), getAttemptAttr(),
          getProfileAttr(), gadget, profile, spec, "retry")))
    return failure();

  if (getSuccessProbabilityAttr()) {
    auto sourceAttr = getSuccessProbabilitySourceAttr();
    auto evidenceAttr = getSuccessProbabilityEvidenceAttr();
    if (!sourceAttr || !evidenceAttr || evidenceAttr.getValue().empty())
      return emitOpError(
          "success_probability requires a source and nonempty evidence");
    if (sourceAttr != getAttemptAttr() && sourceAttr != getProfileAttr())
      return emitOpError(
          "success_probability source must be the attempt or selected profile");
    DictionaryAttr metadata;
    if (sourceAttr == getAttemptAttr())
      metadata = protocol ? protocol.getMetadataAttr() : spec.getMetadataAttr();
    else if (sourceAttr == getProfileAttr())
      metadata = profile.getMetadataAttr();
    auto probabilityText =
        metadata ? metadata.getAs<StringAttr>("success_probability")
                 : StringAttr{};
    auto evidenceText =
        metadata ? metadata.getAs<StringAttr>("success_probability_evidence")
                 : StringAttr{};
    double establishedProbability = 0.0;
    if (!probabilityText ||
        probabilityText.getValue().getAsDouble(establishedProbability) ||
        establishedProbability !=
            getSuccessProbabilityAttr().getValueAsDouble())
      return emitOpError(
          "success_probability must equal the value established by its source");
    if (!evidenceText || evidenceText.getValue() != evidenceAttr.getValue())
      return emitOpError(
          "success_probability evidence must match its source metadata");

    StringRef evidence = evidenceAttr.getValue();
    constexpr StringLiteral synthesisPrefix = "synthesis:sha256:";
    constexpr StringLiteral analysisPrefix = "analysis:";
    if (evidence.starts_with(synthesisPrefix)) {
      StringRef digest = evidence.drop_front(synthesisPrefix.size());
      auto sourceDigest = metadata
                              ? metadata.getAs<StringAttr>("synthesis_sha256")
                              : StringAttr{};
      if (sourceAttr != getAttemptAttr() || digest.size() != 64 ||
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
      return emitOpError("success_probability evidence must be "
                         "synthesis:sha256:... or analysis:...");
    }
  }
  return success();
}

LogicalResult PermuteOp::verify() {
  auto values = getPerm();
  auto patch = cast<PatchType>(getPatch().getType());
  auto *target =
      SymbolTable::lookupNearestSymbolFrom(*this, patch.getCodeType());
  auto code = dyn_cast_or_null<CodeOp>(target);
  auto nAttr = target ? target->getAttrOfType<IntegerAttr>("n") : IntegerAttr{};
  if (nAttr && values.size() != static_cast<size_t>(nAttr.getInt()))
    return emitOpError(
        "permutation length must equal the code's carrier count");

  llvm::SmallDenseSet<int64_t, 16> seen;
  auto limit = nAttr ? nAttr.getInt() : static_cast<int64_t>(values.size());
  for (auto value : values) {
    if (value < 0 || value >= limit)
      return emitOpError("permutation entries must lie in [0, n)");
    if (!seen.insert(value).second)
      return emitOpError("permutation entries must be unique");
  }

  if (!code || !nAttr)
    return emitOpError(
        "requires a resolvable fabric.code with canonical carrier count");
  const int64_t n = nAttr.getInt();
  const int64_t k = code.getK().value_or(1);
  const int64_t r = code.getR().value_or(0);
  const int64_t normalizerWidth = k + r;

  auto matrix = [&](StringRef name, int64_t expectedRows)
      -> FailureOr<SmallVector<llvm::SmallBitVector>> {
    auto attr = code->getAttrOfType<DenseIntElementsAttr>(name);
    if (!attr || attr.getType().getRank() != 2 ||
        !attr.getType().getElementType().isInteger(1) ||
        attr.getType().getShape()[0] != expectedRows ||
        attr.getType().getShape()[1] != 2 * n) {
      emitOpError() << "requires canonical code " << name << " with shape "
                    << expectedRows << "x" << 2 * n;
      return failure();
    }
    SmallVector<llvm::SmallBitVector> rows;
    auto entries = attr.getValues<APInt>();
    auto iterator = entries.begin();
    for (int64_t row = 0; row < expectedRows; ++row) {
      llvm::SmallBitVector bits(2 * n);
      for (int64_t column = 0; column < 2 * n; ++column, ++iterator)
        if (!(*iterator).isZero())
          bits.set(column);
      rows.push_back(std::move(bits));
    }
    return rows;
  };

  const int64_t stabilizerCount = n - k - r;
  auto stabilizers = matrix("stabilizer_basis", stabilizerCount);
  auto logicalX = matrix("logical_x_basis", k);
  auto logicalZ = matrix("logical_z_basis", k);
  auto gaugeX = matrix("gauge_x_basis", r);
  auto gaugeZ = matrix("gauge_z_basis", r);
  if (failed(stabilizers) || failed(logicalX) || failed(logicalZ) ||
      failed(gaugeX) || failed(gaugeZ))
    return failure();

  auto rowRank = [](ArrayRef<llvm::SmallBitVector> source) {
    SmallVector<llvm::SmallBitVector> rows(source.begin(), source.end());
    int64_t rank = 0;
    const int64_t width = rows.empty() ? 0 : rows.front().size();
    for (int64_t column = width - 1; column >= 0; --column) {
      auto pivot =
          llvm::find_if(llvm::drop_begin(rows, rank),
                        [&](const auto &row) { return row.test(column); });
      if (pivot == rows.end())
        continue;
      std::iter_swap(rows.begin() + rank, pivot);
      for (int64_t index = 0; index < static_cast<int64_t>(rows.size());
           ++index)
        if (index != rank && rows[index].test(column))
          rows[index] ^= rows[rank];
      ++rank;
      if (rank == static_cast<int64_t>(rows.size()))
        break;
    }
    return rank;
  };
  auto transform = [&](const llvm::SmallBitVector &row) {
    llvm::SmallBitVector result(2 * n);
    for (int64_t carrier = 0; carrier < n; ++carrier) {
      const int64_t destination = values[carrier];
      if (row.test(carrier))
        result.set(destination);
      if (row.test(n + carrier))
        result.set(n + destination);
    }
    return result;
  };
  auto preservesSpan = [&](ArrayRef<llvm::SmallBitVector> span,
                           ArrayRef<llvm::SmallBitVector> rows) {
    const int64_t originalRank = rowRank(span);
    SmallVector<llvm::SmallBitVector> extended(span.begin(), span.end());
    llvm::append_range(extended, rows);
    return rowRank(extended) == originalRank;
  };

  SmallVector<llvm::SmallBitVector> transformedStabilizers;
  llvm::transform(*stabilizers, std::back_inserter(transformedStabilizers),
                  transform);
  if (!preservesSpan(*stabilizers, transformedStabilizers))
    return emitOpError(
        "permutation does not preserve the canonical stabilizer group");

  SmallVector<llvm::SmallBitVector> gaugeGroup(stabilizers->begin(),
                                               stabilizers->end());
  llvm::append_range(gaugeGroup, *gaugeX);
  llvm::append_range(gaugeGroup, *gaugeZ);
  SmallVector<llvm::SmallBitVector> transformedGauge;
  for (const auto &row : *gaugeX)
    transformedGauge.push_back(transform(row));
  for (const auto &row : *gaugeZ)
    transformedGauge.push_back(transform(row));
  if (!preservesSpan(gaugeGroup, transformedGauge))
    return emitOpError(
        "permutation does not preserve the canonical gauge group");

  SmallVector<llvm::SmallBitVector> xBasis(logicalX->begin(), logicalX->end());
  llvm::append_range(xBasis, *gaugeX);
  SmallVector<llvm::SmallBitVector> zBasis(logicalZ->begin(), logicalZ->end());
  llvm::append_range(zBasis, *gaugeZ);
  SmallVector<llvm::SmallBitVector> coordinateBasis(stabilizers->begin(),
                                                    stabilizers->end());
  llvm::append_range(coordinateBasis, xBasis);
  llvm::append_range(coordinateBasis, zBasis);

  auto coordinates = [&](const llvm::SmallBitVector &source)
      -> FailureOr<llvm::SmallBitVector> {
    const int64_t basisSize = coordinateBasis.size();
    SmallVector<llvm::SmallBitVector> pivotRows(2 * n,
                                                llvm::SmallBitVector(2 * n));
    SmallVector<llvm::SmallBitVector> pivotCoefficients(
        2 * n, llvm::SmallBitVector(basisSize));
    llvm::SmallBitVector occupied(2 * n);
    for (auto [index, rawRow] : llvm::enumerate(coordinateBasis)) {
      llvm::SmallBitVector row = rawRow;
      llvm::SmallBitVector coefficients(basisSize);
      coefficients.set(index);
      while (row.any()) {
        int64_t pivot = -1;
        for (int64_t column = 2 * n - 1; column >= 0; --column)
          if (row.test(column)) {
            pivot = column;
            break;
          }
        if (!occupied.test(pivot)) {
          occupied.set(pivot);
          pivotRows[pivot] = std::move(row);
          pivotCoefficients[pivot] = std::move(coefficients);
          break;
        }
        row ^= pivotRows[pivot];
        coefficients ^= pivotCoefficients[pivot];
      }
    }
    llvm::SmallBitVector row = source;
    llvm::SmallBitVector result(basisSize);
    while (row.any()) {
      int64_t pivot = -1;
      for (int64_t column = 2 * n - 1; column >= 0; --column)
        if (row.test(column)) {
          pivot = column;
          break;
        }
      if (!occupied.test(pivot))
        return failure();
      row ^= pivotRows[pivot];
      result ^= pivotCoefficients[pivot];
    }
    return result;
  };

  auto declaredX = (*this)->getAttrOfType<ArrayAttr>("logical_x_action");
  auto declaredZ = (*this)->getAttrOfType<ArrayAttr>("logical_z_action");
  auto protectedLogicals =
      (*this)->getAttrOfType<IntegerAttr>("protected_logicals");
  auto gaugeQubits = (*this)->getAttrOfType<IntegerAttr>("gauge_qubits");
  auto derivation = (*this)->getAttrOfType<StringAttr>("derivation");
  auto binding = (*this)->getAttrOfType<DictionaryAttr>("objective_binding");
  if (!declaredX || !declaredZ || !protectedLogicals || !gaugeQubits ||
      !derivation || !binding)
    return emitOpError(
        "requires typed derived action and objective-binding evidence");
  if (protectedLogicals.getInt() != k || gaugeQubits.getInt() != r)
    return emitOpError(
        "protected_logicals and gauge_qubits must equal the canonical code");
  if (derivation.getValue() != "verified_code_automorphism")
    return emitOpError("derivation must be verified_code_automorphism");
  if (declaredX.size() != static_cast<size_t>(normalizerWidth) ||
      declaredZ.size() != static_cast<size_t>(normalizerWidth))
    return emitOpError(
        "logical action evidence must contain one X and Z row per protected "
        "or gauge qubit");

  auto verifyAction = [&](ArrayRef<llvm::SmallBitVector> source,
                          ArrayAttr declared, bool xAction) -> LogicalResult {
    for (auto [rowIndex, row] : llvm::enumerate(source)) {
      auto coordinate = coordinates(transform(row));
      if (failed(coordinate))
        return emitOpError(
            "permuted normalizer generator leaves the canonical code basis");
      SmallVector<int64_t> expected;
      const int64_t offset = stabilizerCount;
      for (int64_t index = 0; index < normalizerWidth; ++index) {
        bool sameBasis =
            coordinate->test(offset + (xAction ? 0 : normalizerWidth) + index);
        bool mixedBasis =
            coordinate->test(offset + (xAction ? normalizerWidth : 0) + index);
        if (mixedBasis)
          return emitOpError(
              "permutation mixes canonical X and Z generators; use a typed "
              "Clifford realization rather than fabric.permute");
        if (sameBasis)
          expected.push_back(index);
      }
      auto actual = dyn_cast<DenseI64ArrayAttr>(declared[rowIndex]);
      if (!actual || actual.asArrayRef() != ArrayRef(expected))
        return emitOpError(xAction ? "logical_x_action row "
                                   : "logical_z_action row ")
               << rowIndex
               << " contradicts the action independently derived from the "
                  "canonical code";
    }
    return success();
  };
  if (failed(verifyAction(xBasis, declaredX, true)) ||
      failed(verifyAction(zBasis, declaredZ, false)))
    return failure();
  return success();
}

LogicalResult ReadSyndromeAncillasOp::verify() {
  auto patch = cast<PatchType>(getPatch().getType());
  auto syndrome = cast<SyndromeType>(getSyndrome().getType());
  if (patch.getCodeType() != syndrome.getCodeType() ||
      patch.getEncoding() != syndrome.getEncoding() ||
      patch.getEpoch() != syndrome.getEpoch())
    return emitOpError(
        "syndrome must preserve the patch code and encoding qualification");
  SmallVector<Type, 2> types{getPatch().getType(), getSyndrome().getType()};
  return verifyEncodingQualifiedTypes(*this, types);
}

Value MzOp::getMeasurementPatchOut() { return getPatchOut(); }

PartitionAttr MzOp::getMeasurementPartition() { return getPartitionAttr(); }

DenseI64ArrayAttr MzOp::getMeasurementIndices() { return getIndicesAttr(); }

Prep MzOp::getMeasurementBasis() { return Prep::z; }

Value MeasureBasisOp::getMeasurementPatchOut() { return getPatchOut(); }

PartitionAttr MeasureBasisOp::getMeasurementPartition() {
  return getPartitionAttr();
}

DenseI64ArrayAttr MeasureBasisOp::getMeasurementIndices() {
  return getIndicesAttr();
}

Prep MeasureBasisOp::getMeasurementBasis() { return getBasis(); }

LogicalResult AssembleSyndromeOp::verify() {
  auto patch = cast<PatchType>(getPatch().getType());
  auto syndrome = cast<SyndromeType>(getSyndrome().getType());
  if (getRecord().empty())
    return emitOpError("record must be nonempty");
  if (patch.getCodeType() != syndrome.getCodeType() ||
      patch.getEncoding() != syndrome.getEncoding() ||
      patch.getEpoch() != syndrome.getEpoch())
    return emitOpError(
        "syndrome must preserve the patch code and encoding qualification");

  auto verifyBits = [&](Value bits, StringRef label,
                        int64_t expected) -> LogicalResult {
    auto type = dyn_cast<RankedTensorType>(bits.getType());
    if (!type || type.getRank() != 1 || !type.getElementType().isInteger(1))
      return emitOpError() << label << " must be a rank-1 i1 tensor";
    if (type.isDynamicDim(0))
      return emitOpError() << label << " must have statically known width";
    if (expected >= 0 && type.getDimSize(0) != expected)
      return emitOpError() << label << " width " << type.getDimSize(0)
                           << " does not match code partition width "
                           << expected;
    return success();
  };

  int64_t sxWidth = -1;
  int64_t szWidth = -1;
  auto *target =
      SymbolTable::lookupNearestSymbolFrom(*this, patch.getCodeType());
  if (auto code = dyn_cast_or_null<CodeOp>(target)) {
    auto partitions = code.getPartitions();
    if (auto value = dyn_cast_or_null<IntegerAttr>(partitions.get("sx")))
      sxWidth = value.getInt();
    if (auto value = dyn_cast_or_null<IntegerAttr>(partitions.get("sz")))
      szWidth = value.getInt();
  }
  if (failed(verifyBits(getSxBits(), "sx_bits", sxWidth)) ||
      failed(verifyBits(getSzBits(), "sz_bits", szWidth)))
    return failure();

  auto sxMeasurement =
      getSxBits().getDefiningOp<MeasurementBundleOpInterface>();
  auto szMeasurement =
      getSzBits().getDefiningOp<MeasurementBundleOpInterface>();
  if (!sxMeasurement || sxMeasurement.getMeasurementBasis() != Prep::x ||
      sxMeasurement.getMeasurementPartition().getValue() != Partition::sx)
    return emitOpError("sx_bits must be produced by an X-basis measurement of "
                       "the sx partition");
  if (!szMeasurement || szMeasurement.getMeasurementBasis() != Prep::z ||
      szMeasurement.getMeasurementPartition().getValue() != Partition::sz)
    return emitOpError("sz_bits must be produced by a Z-basis measurement of "
                       "the sz partition");
  if (sxMeasurement.getMeasurementIndices() ||
      szMeasurement.getMeasurementIndices())
    return emitOpError(
        "sx_bits and sz_bits must measure complete partitions in canonical "
        "order without explicit indices");
  if (sxMeasurement.getMeasurementPatchOut().getType() !=
          getPatch().getType() ||
      szMeasurement.getMeasurementPatchOut().getType() != getPatch().getType())
    return emitOpError(
        "measurement bundles must preserve this patch qualification");

  auto isPatchAncestor = [](Value ancestor, Value descendant) {
    SmallVector<Value, 8> pending{descendant};
    llvm::DenseSet<Value> visited;
    while (!pending.empty()) {
      Value value = pending.pop_back_val();
      if (value == ancestor)
        return true;
      if (!visited.insert(value).second)
        continue;
      Operation *producer = value.getDefiningOp();
      if (!producer)
        continue;
      // Assembly provenance is deliberately local. Region-bearing operations
      // and calls may reinterpret or reorder ownership behind their boundary,
      // so they require a future operation-specific lineage contract.
      if (producer->getNumRegions() != 0 ||
          isa<CallOp, CallOpInterface>(producer))
        continue;
      SmallVector<Value, 4> patchOperands;
      SmallVector<Value, 4> patchResults;
      for (Value operand : producer->getOperands())
        if (isa<PatchType>(operand.getType()))
          patchOperands.push_back(operand);
      for (Value result : producer->getResults())
        if (isa<PatchType>(result.getType()))
          patchResults.push_back(result);
      // The ordinary owner-preserving form is one patch in and the one patch
      // out currently being traced. In particular, do not treat every result
      // of resource unpacking or another one-to-many producer as the anchor.
      if (patchOperands.size() == 1 && patchResults.size() == 1 &&
          patchResults.front() == value) {
        pending.push_back(patchOperands.front());
        continue;
      }
      // CX and CZ explicitly preserve positional ownership for their
      // two-patch form. No other multi-owner operation is inferred here.
      if (!isa<CXOp, CZOp>(producer))
        continue;
      if (patchOperands.size() != patchResults.size())
        continue;
      auto result = llvm::find(patchResults, value);
      if (result != patchResults.end())
        pending.push_back(
            patchOperands[std::distance(patchResults.begin(), result)]);
    }
    return false;
  };
  if (!isPatchAncestor(sxMeasurement.getMeasurementPatchOut(), getPatch()) ||
      !isPatchAncestor(szMeasurement.getMeasurementPatchOut(), getPatch()))
    return emitOpError(
        "sx_bits and sz_bits must come from this patch's SSA lineage");

  SmallVector<Type, 2> types{getPatch().getType(), getSyndrome().getType()};
  return verifyEncodingQualifiedTypes(*this, types);
}

static FailureOr<DenseIntElementsAttr>
requireGF2Matrix(Operation *op, Attribute value, StringRef label) {
  auto matrix = dyn_cast_or_null<DenseIntElementsAttr>(value);
  if (!matrix || matrix.getType().getRank() != 2 ||
      !matrix.getType().getElementType().isInteger(1)) {
    op->emitOpError() << label << " must be a rank-2 i1 tensor";
    return failure();
  }
  return matrix;
}

static SmallVector<llvm::SmallBitVector>
denseGF2Rows(DenseIntElementsAttr matrix) {
  auto shape = matrix.getType().getShape();
  SmallVector<llvm::SmallBitVector> rows;
  auto values = matrix.getValues<APInt>();
  auto iterator = values.begin();
  for (int64_t row = 0; row < shape[0]; ++row) {
    llvm::SmallBitVector bits(shape[1]);
    for (int64_t column = 0; column < shape[1]; ++column, ++iterator)
      if (!(*iterator).isZero())
        bits.set(column);
    rows.push_back(std::move(bits));
  }
  return rows;
}

static llvm::SmallBitVector
combineGF2Rows(ArrayRef<llvm::SmallBitVector> source,
               const llvm::SmallBitVector &coefficients, int64_t width) {
  llvm::SmallBitVector result(width);
  for (int index : coefficients.set_bits())
    result ^= source[index];
  return result;
}

LogicalResult MeasureGaugesOp::verify() {
  auto patch = cast<PatchType>(getPatch().getType());
  auto output = cast<PatchType>(getPatchOut().getType());
  auto records = cast<GaugeRecordsType>(getRecords().getType());
  if (patch != output)
    return emitOpError("must preserve patch ownership type");
  if (!patch.getEncoding() || !patch.getEpoch())
    return emitOpError("requires an encoding- and epoch-qualified patch");
  if (patch.getCodeType() != records.getCodeType() ||
      patch.getEncoding() != records.getEncoding() ||
      patch.getEpoch() != records.getEpoch())
    return emitOpError(
        "gauge records must preserve patch code, encoding, and epoch");
  auto operators = requireGF2Matrix(*this, getOperators(), "operators");
  if (failed(operators))
    return failure();
  auto shape = operators->getType().getShape();
  if (shape[0] <= 0)
    return emitOpError("requires at least one measured gauge operator");
  if (getRecord().empty())
    return emitOpError("record must be nonempty");
  if (auto *target =
          SymbolTable::lookupNearestSymbolFrom(*this, patch.getCodeType())) {
    if (auto n = target->getAttrOfType<IntegerAttr>("n");
        n && shape[1] != 2 * n.getInt())
      return emitOpError("operator matrix must have symplectic width 2n");
  }
  if (auto value = getStabilizerMap()) {
    auto map = requireGF2Matrix(*this, *value, "stabilizer_map");
    if (failed(map))
      return failure();
    if (map->getType().getShape()[1] != shape[0])
      return emitOpError(
          "stabilizer_map width must equal measured gauge count");
  }
  SmallVector<Type, 3> types{patch, output, records};
  return verifyEncodingQualifiedTypes(*this, types);
}

LogicalResult ResourceRequestOp::verify() {
  auto payload = dyn_cast<ResourceStateType>(getEvent().getType().getPayload());
  if (!payload)
    return emitOpError("event payload must be a Fabric resource");
  auto resource = dyn_cast<SymbolRefAttr>(payload.getKind());
  if (!resource || resource.getLeafReference().getValue() != getKind())
    return emitOpError("kind must match the event payload resource");
  auto stream = getStreamAttr();
  SmallVector<Operation *> selectedDevices;
  if (auto module = (*this)->getParentOfType<ModuleOp>())
    module.walk([&](Operation *operation) {
      if (operation->getName().getStringRef() == "qlx.device")
        selectedDevices.push_back(operation);
    });
  if (selectedDevices.empty())
    return success();
  if (selectedDevices.size() != 1)
    return emitOpError(
        "selected-device requests require exactly one qlx.device");
  if (stream.getNestedReferences().empty())
    return emitOpError(
        "selected-device requests require a fully qualified stream reference");
  auto logical =
      selectedDevices.front()->getAttrOfType<FlatSymbolRefAttr>("logical");
  if (!logical || stream.getRootReference().getValue() != logical.getValue())
    return emitOpError(
        "stream must belong to the selected device logical domain");
  Operation *target = SymbolTable::lookupNearestSymbolFrom(*this, stream);
  if (!target || target->getName().getStringRef() != "lvm.stream")
    return emitOpError("stream must resolve to lvm.stream");
  auto produces = target->getAttrOfType<SymbolRefAttr>("produces");
  if (!produces || produces.getLeafReference().getValue() != getKind())
    return emitOpError("stream must produce the requested resource kind");
  return success();
}

// Alpha QLX lowered logical gates by threading a syndrome value beside every
// patch through fabric.call, even when the called gadget's declared function
// type mentioned patches only.  QLX uses exact callable signatures and
// keeps boundary records explicit, but the old form remains valid import IR.
// Keep the compatibility narrow: it applies only to an unclaimed legacy
// gadget, every syndrome must immediately follow a matching patch, and the
// call must pass every threaded value through with an identical type.
static bool isLegacySyndromeThreadedCall(CallOp call, GadgetOp gadget,
                                         FunctionType signature) {
  if (gadget->hasAttr("spec"))
    return false;
  if (call.getNumOperands() != call.getNumResults())
    return false;

  SmallVector<Type> semanticInputs;
  SmallVector<Type> semanticResults;
  bool sawSyndrome = false;
  for (auto [index, types] : llvm::enumerate(
           llvm::zip(call.getOperandTypes(), call.getResultTypes()))) {
    Type inputType = std::get<0>(types);
    Type resultType = std::get<1>(types);
    if (inputType != resultType)
      return false;
    if (auto syndrome = dyn_cast<SyndromeType>(inputType)) {
      sawSyndrome = true;
      if (index == 0)
        return false;
      auto patch = dyn_cast<PatchType>(call.getOperand(index - 1).getType());
      if (!patch || patch.getCodeType() != syndrome.getCodeType() ||
          patch.getEncoding() != syndrome.getEncoding() ||
          patch.getEpoch() != syndrome.getEpoch())
        return false;
      continue;
    }
    semanticInputs.push_back(inputType);
    semanticResults.push_back(resultType);
  }
  return sawSyndrome && semanticInputs == signature.getInputs() &&
         semanticResults == signature.getResults();
}

static bool isRegisteredLVMAttribute(Attribute attribute, StringRef mnemonic) {
  return attribute && attribute.getAbstractAttribute().getName() ==
                          ("lvm." + mnemonic).str();
}

namespace {

enum class OwnerOriginKind : uint8_t { boundary, fresh, unknown };

struct OwnerOrigin {
  OwnerOriginKind kind = OwnerOriginKind::unknown;
  unsigned boundary = 0;

  static OwnerOrigin fromBoundary(unsigned ordinal) {
    return {OwnerOriginKind::boundary, ordinal};
  }
  static OwnerOrigin fresh() { return {OwnerOriginKind::fresh, 0}; }
  static OwnerOrigin unknown() { return {}; }
};

/// Derive patch-owner lineage from the verified Fabric SSA graph.  The summary
/// of a callable is expressed relative to its patch boundary arguments, so a
/// nested fabric.call can map the summary back through its actual operands.
/// Unsupported ownership-changing or structured operations remain unknown and
/// therefore fail closed when a communication obligation depends on them.
class OwnerLineageAnalysis {
public:
  OwnerOrigin traceCallableResult(Operation *callable, unsigned resultIndex) {
    auto key = std::make_pair(callable, resultIndex);
    if (auto found = summaries.find(key); found != summaries.end())
      return found->second;
    if (!active.insert(key).second)
      return OwnerOrigin::unknown();

    OwnerOrigin origin = traceCallableResultImpl(callable, resultIndex);
    active.erase(key);
    summaries.try_emplace(key, origin);
    return origin;
  }

private:
  static bool isPositionallyOwnerPreserving(Operation *operation) {
    return llvm::StringSwitch<bool>(operation->getName().getStringRef())
        .Cases({"fabric.epoch_transition", "fabric.relocate",
                "fabric.establish_support"},
               true)
        .Case("fabric.establish_topological_record", true)
        .Cases({"fabric.prep_z", "fabric.prep_x", "fabric.h"}, true)
        .Cases({"fabric.s", "fabric.sdg", "fabric.x"}, true)
        .Cases({"fabric.z", "fabric.t", "fabric.tdg"}, true)
        .Cases({"fabric.reset", "fabric.permute"}, true)
        .Cases({"fabric.mz", "fabric.mpp", "fabric.read_syndrome_ancillas"},
               true)
        .Cases({"fabric.cx", "fabric.cz"}, true)
        .Cases({"fabric.transversal_cx", "fabric.multi_measure",
                "fabric.measure_product"},
               true)
        .Cases({"fabric.rotate_product", "fabric.resource_rotate_product",
                "fabric.idle"},
               true)
        .Default(false);
  }

  static Value patchOperand(Operation *operation, unsigned patchOrdinal) {
    unsigned current = 0;
    for (Value operand : operation->getOperands()) {
      if (!isa<PatchType>(operand.getType()))
        continue;
      if (current++ == patchOrdinal)
        return operand;
    }
    return {};
  }

  static std::optional<unsigned> patchResultOrdinal(OpResult result) {
    unsigned current = 0;
    for (Value candidate : result.getOwner()->getResults()) {
      if (!isa<PatchType>(candidate.getType()))
        continue;
      if (candidate == result)
        return current;
      ++current;
    }
    return std::nullopt;
  }

  static unsigned patchCount(ValueRange values) {
    return llvm::count_if(
        values, [](Value value) { return isa<PatchType>(value.getType()); });
  }

  OwnerOrigin traceValue(Value value, Block &entry) {
    if (!isa<PatchType>(value.getType()))
      return OwnerOrigin::unknown();

    if (auto argument = dyn_cast<BlockArgument>(value)) {
      if (argument.getOwner() != &entry)
        return OwnerOrigin::unknown();
      unsigned ordinal = 0;
      for (BlockArgument candidate : entry.getArguments()) {
        if (!isa<PatchType>(candidate.getType()))
          continue;
        if (candidate == argument)
          return OwnerOrigin::fromBoundary(ordinal);
        ++ordinal;
      }
      return OwnerOrigin::unknown();
    }

    auto result = dyn_cast<OpResult>(value);
    if (!result)
      return OwnerOrigin::unknown();
    Operation *definition = result.getOwner();
    if (isa<AllocOp>(definition))
      return OwnerOrigin::fresh();

    if (auto call = dyn_cast<CallOp>(definition)) {
      Operation *target = SymbolTable::lookupNearestSymbolFrom(
          call.getOperation(), call.getCalleeAttr());
      if (!target)
        return OwnerOrigin::unknown();
      OwnerOrigin nested =
          traceCallableResult(target, result.getResultNumber());
      if (nested.kind != OwnerOriginKind::boundary)
        return nested;
      Value operand = patchOperand(definition, nested.boundary);
      return operand ? traceValue(operand, entry) : OwnerOrigin::unknown();
    }

    if (!isPositionallyOwnerPreserving(definition))
      return OwnerOrigin::unknown();
    auto ordinal = patchResultOrdinal(result);
    if (!ordinal || patchCount(definition->getOperands()) !=
                        patchCount(definition->getResults()))
      return OwnerOrigin::unknown();
    Value operand = patchOperand(definition, *ordinal);
    return operand ? traceValue(operand, entry) : OwnerOrigin::unknown();
  }

  OwnerOrigin traceCallableResultImpl(Operation *callable,
                                      unsigned resultIndex) {
    if (auto gadget = dyn_cast<GadgetOp>(callable)) {
      if (auto realization = gadget.getRealizationAttr()) {
        Operation *target =
            SymbolTable::lookupNearestSymbolFrom(callable, realization);
        auto circuit = dyn_cast_or_null<CircuitOp>(target);
        if (!circuit || circuit.getFunctionType() != gadget.getFunctionType())
          return OwnerOrigin::unknown();
        return traceCallableResult(circuit, resultIndex);
      }
    }

    Region *body = nullptr;
    if (auto gadget = dyn_cast<GadgetOp>(callable))
      body = &gadget.getBody();
    else if (auto protocol = dyn_cast<ProtocolOp>(callable))
      body = &protocol.getBody();
    else if (auto circuit = dyn_cast<CircuitOp>(callable))
      body = &circuit.getBody();
    else
      return OwnerOrigin::unknown();

    if (!body || !llvm::hasSingleElement(*body))
      return OwnerOrigin::unknown();
    Block &entry = body->front();
    Operation *terminator = entry.getTerminator();
    if (!terminator || resultIndex >= terminator->getNumOperands())
      return OwnerOrigin::unknown();
    return traceValue(terminator->getOperand(resultIndex), entry);
  }

  llvm::DenseMap<std::pair<Operation *, unsigned>, OwnerOrigin> summaries;
  llvm::DenseSet<std::pair<Operation *, unsigned>> active;
};

} // namespace

static LogicalResult verifyCommunicationObligation(CallOp op) {
  bool hasChannel = static_cast<bool>(op.getChannelAttr());
  bool hasChannelCapability = static_cast<bool>(op.getChannelCapabilityAttr());
  bool hasEndpoints = static_cast<bool>(op.getEndpointsAttr());
  bool hasActionSite = static_cast<bool>(op.getActionSiteAttr());
  bool hasGeneratedBy = static_cast<bool>(op.getGeneratedByAttr());
  unsigned present = static_cast<unsigned>(hasChannel) +
                     static_cast<unsigned>(hasChannelCapability) +
                     static_cast<unsigned>(hasEndpoints) +
                     static_cast<unsigned>(hasActionSite) +
                     static_cast<unsigned>(hasGeneratedBy);
  if (present == 0)
    return success();
  if (present != 5)
    return op.emitOpError("communication obligation requires channel, channel "
                          "capability, endpoints, "
                          "action_site, and generated_by");

  if (!isRegisteredLVMAttribute(op.getChannelCapabilityAttr(), "capability"))
    return op.emitOpError("communication channel capability must be a "
                          "registered #lvm.capability attribute");
  ArrayAttr endpoints = op.getEndpointsAttr();
  if (endpoints.size() != 2)
    return op.emitOpError("remote observable requires exactly two endpoints");
  SmallVector<SymbolRefAttr> endpointRefs;
  for (Attribute item : endpoints) {
    auto endpoint = dyn_cast<SymbolRefAttr>(item);
    if (!endpoint || endpoint.getNestedReferences().size() != 1)
      return op.emitOpError(
          "endpoints must contain nested logical-space symbol references");
    Operation *target =
        SymbolTable::lookupNearestSymbolFrom(op.getOperation(), endpoint);
    if (!target || target->getName().getStringRef() != "lvm.space")
      return op.emitOpError("endpoint must resolve to lvm.space");
    endpointRefs.push_back(endpoint);
  }
  if (endpointRefs[0] == endpointRefs[1])
    return op.emitOpError("remote observable endpoints must be distinct");

  Operation *channel = SymbolTable::lookupNearestSymbolFrom(
      op.getOperation(), op.getChannelAttr());
  if (!channel || channel->getName().getStringRef() != "lvm.channel")
    return op.emitOpError("channel must resolve to lvm.channel");
  auto capabilities = channel->getAttrOfType<ArrayAttr>("capabilities");
  if (!capabilities ||
      !llvm::is_contained(capabilities, op.getChannelCapabilityAttr()))
    return op.emitOpError(
        "channel does not advertise the requested channel capability");
  auto from = channel->getAttrOfType<FlatSymbolRefAttr>("from");
  auto to = channel->getAttrOfType<FlatSymbolRefAttr>("to");
  if (!from || !to)
    return op.emitOpError("resolved lvm.channel has no valid endpoints");
  auto first = endpointRefs[0].getLeafReference();
  auto second = endpointRefs[1].getLeafReference();
  bool isForward = first == from.getAttr() && second == to.getAttr();
  bool isReverse = first == to.getAttr() && second == from.getAttr();
  auto direction = channel->getAttrOfType<StringAttr>("direction");
  StringRef directionName = direction ? direction.getValue() : "forward";
  if (directionName != "forward" && directionName != "reverse" &&
      directionName != "bidirectional")
    return op.emitOpError("channel direction must be forward, reverse, or "
                          "bidirectional");
  if ((directionName == "forward" && !isForward) ||
      (directionName == "reverse" && !isReverse) ||
      (directionName == "bidirectional" && !isForward && !isReverse))
    return op.emitOpError("ordered endpoints are incompatible with channel "
                          "direction");

  Operation *actionSite = SymbolTable::lookupNearestSymbolFrom(
      op.getOperation(), op.getActionSiteAttr());
  if (!actionSite || actionSite->getName().getStringRef() != "lvm.action_site")
    return op.emitOpError("action_site must resolve to lvm.action_site");
  auto siteChannel = actionSite->getAttrOfType<SymbolRefAttr>("channel");
  Attribute siteChannelCapability = actionSite->getAttr("channel_capability");
  auto siteEndpoints = actionSite->getAttrOfType<ArrayAttr>("endpoints");
  if (!siteChannel || !siteChannelCapability || !siteEndpoints)
    return op.emitOpError("action_site must retain channel, channel "
                          "capability, and endpoints provenance");
  if (siteChannel != op.getChannelAttr())
    return op.emitOpError("channel does not match action_site provenance");
  if (siteChannelCapability != op.getChannelCapabilityAttr())
    return op.emitOpError(
        "channel capability does not match action_site provenance");
  if (siteEndpoints != endpoints)
    return op.emitOpError("endpoints do not match action_site provenance");

  Operation *lowering = SymbolTable::lookupNearestSymbolFrom(
      op.getOperation(), op.getGeneratedByAttr());
  if (!lowering || lowering->getName().getStringRef() != "qlx.qec_lowering")
    return op.emitOpError("generated_by must resolve to qlx.qec_lowering");
  auto family = lowering->getAttrOfType<StringAttr>("objective_family");
  if (!family || family.getValue() != "pauli_product_measurement")
    return op.emitOpError("generated_by must select pauli_product_measurement");
  if (!containsString(lowering->getAttrOfType<ArrayAttr>("requirements"),
                      "qlx.machine/observable_remote"))
    return op.emitOpError("generated_by must require observable_remote");
  Attribute siteObjective = actionSite->getAttr("objective");
  if (!siteObjective || lowering->getAttr("objective") != siteObjective)
    return op.emitOpError(
        "generated_by objective does not match action_site provenance");

  Operation *callee = SymbolTable::lookupNearestSymbolFrom(op.getOperation(),
                                                           op.getCalleeAttr());
  auto protocol = dyn_cast_or_null<ProtocolOp>(callee);
  if (!protocol)
    return op.emitOpError(
        "remote-observable callee must be a generated fabric.protocol");
  if (protocol.getGeneratedByAttr() != op.getGeneratedByAttr())
    return op.emitOpError(
        "callee generation provenance does not match the call");
  if (!protocol.getObjectiveAttr() ||
      protocol.getObjectiveAttr() != siteObjective)
    return op.emitOpError(
        "callee objective does not match action_site provenance");

  SmallVector<Type> inputOwners;
  SmallVector<Type> resultOwners;
  for (Type type : op.getOperandTypes()) {
    if (!isa<PatchType>(type))
      return op.emitOpError(
          "remote-observable call operands must all be patch owners");
    inputOwners.push_back(type);
  }
  for (Type type : op.getResultTypes())
    if (isa<PatchType>(type))
      resultOwners.push_back(type);
  if (inputOwners.size() != 2)
    return op.emitOpError(
        "remote-observable call requires exactly two patch owners");
  if (inputOwners != resultOwners)
    return op.emitOpError(
        "remote-observable call must preserve owner types and order");
  auto leftOwner = cast<PatchType>(inputOwners[0]);
  auto rightOwner = cast<PatchType>(inputOwners[1]);
  if (leftOwner.getCodeType() != rightOwner.getCodeType() ||
      leftOwner.getEncoding() != rightOwner.getEncoding())
    return op.emitOpError(
        "remote-observable owners must use the same code and encoding");
  auto acceptedCodes = lowering->getAttrOfType<ArrayAttr>("codes");
  for (Type type : inputOwners) {
    auto patch = cast<PatchType>(type);
    bool accepted = acceptedCodes &&
                    (llvm::is_contained(acceptedCodes, patch.getCodeType()) ||
                     (patch.getEncoding() &&
                      llvm::is_contained(acceptedCodes, patch.getEncoding())));
    if (!accepted)
      return op.emitOpError(
          "remote-observable owner code is not accepted by generated_by");
  }
  OwnerLineageAnalysis ownerLineage;
  for (unsigned owner = 0; owner < 2; ++owner) {
    OwnerOrigin origin = ownerLineage.traceCallableResult(protocol, owner);
    if (origin.kind != OwnerOriginKind::boundary || origin.boundary != owner)
      return op.emitOpError(
          "remote-observable protocol must preserve ordered boundary owner "
          "identity");
  }
  if (op.getNumResults() == resultOwners.size())
    return op.emitOpError(
        "remote-observable call requires at least one non-owner outcome");
  return success();
}

static LogicalResult
verifyLocalGeneratedRPPProvenance(Operation *owner, Operation *callee,
                                  SymbolRefAttr actionSiteRef,
                                  SymbolTableCollection &symbolTables) {
  Operation *actionSite =
      symbolTables.lookupNearestSymbolFrom(owner, actionSiteRef);
  if (!actionSite || actionSite->getName().getStringRef() != "lvm.action_site")
    return owner->emitOpError(
        "local generated RPP action_site must resolve to lvm.action_site");
  if (actionSite->hasAttr("channel") ||
      actionSite->hasAttr("channel_capability") ||
      actionSite->hasAttr("endpoints") || actionSite->hasAttr("direction"))
    return owner->emitOpError(
        "local generated RPP action_site must not carry communication "
        "provenance");

  auto siteKind = actionSite->getAttrOfType<StringAttr>("kind");
  auto siteObjective = dyn_cast_or_null<qlx::BuiltinActionAttr>(
      actionSite->getAttr("objective"));
  if (!siteKind || siteKind.getValue() != "action" || !siteObjective ||
      siteObjective.getValue() != qlx::BuiltinAction::pauli_rotation)
    return owner->emitOpError(
        "local generated RPP action_site must select pauli_rotation");

  auto generatedBy = callee->getAttrOfType<FlatSymbolRefAttr>("generated_by");
  if (!generatedBy)
    return owner->emitOpError(
        "local generated RPP callee requires generated_by provenance");
  Operation *lowering =
      symbolTables.lookupNearestSymbolFrom(owner, generatedBy);
  if (!lowering || lowering->getName().getStringRef() != "qlx.qec_lowering")
    return owner->emitOpError(
        "local generated RPP callee generated_by must resolve to "
        "qlx.qec_lowering");
  auto family = lowering->getAttrOfType<StringAttr>("objective_family");
  if (!family || family.getValue() != "pauli_product_rotation")
    return owner->emitOpError(
        "local generated RPP callee must select pauli_product_rotation");
  Attribute loweringObjective = lowering->getAttr("objective");
  if (loweringObjective && loweringObjective != siteObjective)
    return owner->emitOpError(
        "local generated RPP lowering objective must match action_site "
        "pauli_rotation");

  auto calleeName = callee->getAttrOfType<StringAttr>("sym_name");
  auto dependencies = lowering->getAttrOfType<ArrayAttr>("dependencies");
  bool isDependency =
      dependencies && calleeName &&
      llvm::any_of(dependencies, [&](Attribute value) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(value);
        return reference && reference.getValue() == calleeName.getValue();
      });
  if (!isDependency)
    return owner->emitOpError(
        "local generated RPP callee must be a declared generator dependency");

  auto specialization = callee->getAttrOfType<DictionaryAttr>("specialization");
  if (!specialization || !specialization.getAs<StringAttr>("rpp_strategy"))
    return owner->emitOpError(
        "local generated RPP callee requires canonical RPP specialization");
  auto siteParameters = actionSite->getAttrOfType<DictionaryAttr>("parameters");
  if (!siteParameters)
    return owner->emitOpError(
        "local generated RPP action_site requires canonical parameters");
  for (NamedAttribute parameter : siteParameters) {
    StringRef name = parameter.getName().getValue();
    if (name == "angle_pi_numer" || name == "angle_pi_denom")
      continue;
    if (specialization.get(parameter.getName()) != parameter.getValue())
      return owner->emitOpError(
                 "local generated RPP specialization does not preserve "
                 "action-site parameter ")
             << parameter.getName();
  }

  auto siteAngle = siteParameters.getAs<FloatAttr>("angle");
  auto effectiveAngle = specialization.getAs<FloatAttr>("effective_angle");
  if (!siteAngle || !effectiveAngle)
    return owner->emitOpError(
        "local generated RPP requires action-site angle and specialization "
        "effective_angle f64 values");
  int64_t sign = 1;
  if (auto siteSign = siteParameters.getAs<IntegerAttr>("sign"))
    sign = siteSign.getInt();
  auto expectedEffectiveAngle =
      FloatAttr::get(siteAngle.getType(), siteAngle.getValueAsDouble() * sign);
  if (effectiveAngle != expectedEffectiveAngle)
    return owner->emitOpError(
        "local generated RPP effective_angle must equal the signed "
        "action-site angle");

  auto protocol = dyn_cast<ProtocolOp>(callee);
  if (!protocol)
    return owner->emitOpError(
        "local generated RPP callee must be a product-rotation protocol "
        "adapter");
  auto metadata = protocol->getAttrOfType<DictionaryAttr>("metadata");
  auto compiler = metadata ? metadata.getAs<StringAttr>("compiler") : nullptr;
  auto strategy = metadata ? metadata.getAs<StringAttr>("strategy") : nullptr;
  auto implementation =
      metadata ? metadata.getAs<StringAttr>("implementation") : nullptr;
  auto specializationStrategy =
      specialization.getAs<StringAttr>("rpp_strategy");
  if (!compiler || compiler.getValue() != "cudaq.logical.product_rotation" ||
      !strategy || !implementation || !specializationStrategy)
    return owner->emitOpError(
        "local generated RPP callee requires canonical product-rotation "
        "adapter evidence");
  if (strategy != specializationStrategy)
    return owner->emitOpError(
        "local generated RPP strategy does not match its executable adapter");

  auto selections = lowering->getAttrOfType<DictionaryAttr>("rpp_selections");
  auto selection = selections && calleeName
                       ? dyn_cast_or_null<DictionaryAttr>(
                             selections.get(calleeName.getValue()))
                       : DictionaryAttr{};
  auto selectedStrategy =
      selection ? selection.getAs<StringAttr>("strategy") : nullptr;
  auto selectedImplementation =
      selection ? selection.getAs<FlatSymbolRefAttr>("implementation")
                : FlatSymbolRefAttr{};
  if (!selectedStrategy || !selectedImplementation)
    return owner->emitOpError(
        "local generated RPP callee requires a typed lowering selection "
        "witness");
  if (selectedStrategy != specializationStrategy ||
      selectedStrategy != strategy)
    return owner->emitOpError(
        "local generated RPP strategy contradicts its lowering selection "
        "witness");
  if (selectedImplementation.getValue() != implementation.getValue())
    return owner->emitOpError(
        "local generated RPP implementation contradicts its lowering "
        "selection witness");

  SmallVector<CallOp> implementationCalls;
  protocol.getBody().walk(
      [&](CallOp call) { implementationCalls.push_back(call); });
  if (implementationCalls.size() != 1 ||
      implementationCalls.front().getCalleeAttr().getValue() !=
          implementation.getValue() ||
      implementationCalls.front().getCalleeAttr() != selectedImplementation)
    return owner->emitOpError(
        "local generated RPP adapter must call exactly its recorded "
        "implementation");

  Attribute siteNumerValue = siteParameters.get("angle_pi_numer");
  Attribute siteDenomValue = siteParameters.get("angle_pi_denom");
  if (siteNumerValue || siteDenomValue) {
    auto siteNumer = dyn_cast_or_null<IntegerAttr>(siteNumerValue);
    auto siteDenom = dyn_cast_or_null<IntegerAttr>(siteDenomValue);
    if (!siteNumer || !siteDenom)
      return owner->emitOpError(
          "local generated RPP exact angle requires integer angle_pi_numer "
          "and angle_pi_denom on action_site");

    auto specializationNumer =
        dyn_cast_or_null<IntegerAttr>(specialization.get("angle_pi_numer"));
    auto specializationDenom =
        dyn_cast_or_null<IntegerAttr>(specialization.get("angle_pi_denom"));
    if (!specializationNumer || !specializationDenom)
      return owner->emitOpError(
          "local generated RPP exact angle requires integer angle_pi_numer "
          "and angle_pi_denom on specialization");
    if (siteDenom.getValue().isNegative() || siteDenom.getValue().isZero() ||
        specializationDenom.getValue().isNegative() ||
        specializationDenom.getValue().isZero())
      return owner->emitOpError(
          "local generated RPP exact angle denominators must be positive");

    auto siteSign = siteParameters.getAs<IntegerAttr>("sign");
    if (siteParameters.get("sign") && !siteSign)
      return owner->emitOpError(
          "local generated RPP exact angle requires an integer action-site "
          "sign");

    // P1 stores sign*n/d while the generated specialization stores the
    // already-signed canonical numerator modulo 2*pi. Widen every factor
    // before multiplication so the proof cannot overflow source integers.
    unsigned siteSignWidth = siteSign ? siteSign.getValue().getBitWidth() : 1;
    unsigned leftWidth = siteSignWidth + siteNumer.getValue().getBitWidth() +
                         specializationDenom.getValue().getBitWidth();
    unsigned rightWidth = specializationNumer.getValue().getBitWidth() +
                          siteDenom.getValue().getBitWidth();
    unsigned proofWidth = std::max(leftWidth, rightWidth) + 2;
    llvm::APInt sign = siteSign ? siteSign.getValue().sext(proofWidth)
                                : llvm::APInt(proofWidth, 1);
    llvm::APInt siteN = siteNumer.getValue().sext(proofWidth);
    llvm::APInt siteD = siteDenom.getValue().sext(proofWidth);
    llvm::APInt specializationN =
        specializationNumer.getValue().sext(proofWidth);
    llvm::APInt specializationD =
        specializationDenom.getValue().sext(proofWidth);
    llvm::APInt difference =
        sign * siteN * specializationD - specializationN * siteD;
    llvm::APInt twoPiDenominator = (siteD * specializationD).shl(1);
    if (!difference.srem(twoPiDenominator).isZero())
      return owner->emitOpError(
          "local generated RPP specialization exact angle must equal the "
          "signed action-site angle modulo 2*pi");
  }

  if (!protocol.getObjectiveAttr() ||
      protocol.getObjectiveAttr() != siteObjective)
    return owner->emitOpError(
        "local generated RPP protocol objective must match action_site "
        "pauli_rotation");
  return success();
}

static LogicalResult verifyLocalGeneratedRPPProtocol(ProtocolOp protocol) {
  auto specialization = protocol.getSpecializationAttr();
  auto actionSite = protocol.getActionSiteAttr();
  if (!actionSite || !specialization ||
      !specialization.getAs<StringAttr>("rpp_strategy"))
    return success();
  SymbolTableCollection symbolTables;
  return verifyLocalGeneratedRPPProvenance(protocol.getOperation(),
                                           protocol.getOperation(), actionSite,
                                           symbolTables);
}

static LogicalResult verifyLocalGeneratedRPPInvocations(ProtocolOp protocol) {
  SymbolTableCollection symbolTables;
  LogicalResult result = success();
  protocol.getBody().walk([&](CallOp call) {
    if (failed(result))
      return;
    bool hasLocalActionSite = static_cast<bool>(call.getActionSiteAttr());
    bool hasRemoteTrigger =
        static_cast<bool>(call.getChannelAttr()) ||
        static_cast<bool>(call.getChannelCapabilityAttr()) ||
        static_cast<bool>(call.getEndpointsAttr()) ||
        static_cast<bool>(call.getGeneratedByAttr());
    if (!hasLocalActionSite || hasRemoteTrigger)
      return;

    Operation *target =
        symbolTables.lookupNearestSymbolFrom(call, call.getCalleeAttr());
    if (!target) {
      result = call.emitOpError(
          "provenance-bound callee must resolve to fabric.gadget or "
          "fabric.protocol");
      return;
    }
    FunctionType signature;
    if (auto gadget = dyn_cast<GadgetOp>(target))
      signature = gadget.getFunctionType();
    else if (auto selectedProtocol = dyn_cast<ProtocolOp>(target))
      signature = selectedProtocol.getFunctionType();
    else {
      result = call.emitOpError(
          "callee must resolve to fabric.gadget or fabric.protocol");
      return;
    }
    if (call.getOperandTypes() != signature.getInputs() ||
        call.getResultTypes() != signature.getResults()) {
      result = call.emitOpError(
          "operand/result types must match the callee signature");
      return;
    }
    if (auto profileRef = call.getProfileAttr()) {
      Operation *profileTarget =
          symbolTables.lookupNearestSymbolFrom(call, profileRef);
      if (profileTarget) {
        auto profile = dyn_cast<GadgetProfileOp>(profileTarget);
        if (!profile) {
          result =
              call.emitOpError("profile must resolve to fabric.gadget_profile");
          return;
        }
        if (auto gadget = dyn_cast<GadgetOp>(target)) {
          if (profile.getGadgetAttr() != call.getCalleeAttr()) {
            result = call.emitOpError("profile must analyze the called gadget");
            return;
          }
        } else {
          auto selectedProtocol = cast<ProtocolOp>(target);
          if (!selectedProtocol.getPredicateProfileAttr() ||
              !selectedProtocol.getPredicateGadgetAttr() ||
              selectedProtocol.getPredicateProfileAttr() != profileRef ||
              profile.getGadgetAttr() !=
                  selectedProtocol.getPredicateGadgetAttr()) {
            result = call.emitOpError(
                "protocol call profile must equal its typed predicate profile");
            return;
          }
        }
      }
    }
    result = verifyLocalGeneratedRPPProvenance(
        call.getOperation(), target, call.getActionSiteAttr(), symbolTables);
  });
  return result;
}

LogicalResult CallOp::verify() {
  bool hasLocalActionSite = static_cast<bool>(getActionSiteAttr());
  bool hasRemoteTrigger = static_cast<bool>(getChannelAttr()) ||
                          static_cast<bool>(getChannelCapabilityAttr()) ||
                          static_cast<bool>(getEndpointsAttr()) ||
                          static_cast<bool>(getGeneratedByAttr());
  bool isLocalGeneratedInvocation = hasLocalActionSite && !hasRemoteTrigger;
  bool hasProvenance = hasLocalActionSite || hasRemoteTrigger;

  // A generated RPP call references one action site in the enclosing P1
  // domain.  The enclosing protocol verifies all such calls with one shared
  // SymbolTableCollection; resolving the 200k-site domain independently from
  // every CallOp is quadratic.  Keep the obligation parent-owned and fail
  // closed if the call has no protocol proof boundary.
  if (isLocalGeneratedInvocation) {
    if (!(*this)->getParentOfType<ProtocolOp>())
      return emitOpError(
          "local generated RPP invocation must appear inside fabric.protocol");
    return success();
  }

  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!target) {
    if (hasProvenance)
      return emitOpError(
          "provenance-bound callee must resolve to fabric.gadget or "
          "fabric.protocol");
    return success(); // A partially linked ordinary call retains the
                      // obligation.
  }

  FunctionType signature;
  if (auto gadget = dyn_cast<GadgetOp>(target))
    signature = gadget.getFunctionType();
  else if (auto protocol = dyn_cast<ProtocolOp>(target))
    signature = protocol.getFunctionType();
  else
    return emitOpError(
        "callee must resolve to fabric.gadget or fabric.protocol");

  bool exactSignature = getOperandTypes() == signature.getInputs() &&
                        getResultTypes() == signature.getResults();
  bool legacySyndromeThreading =
      !exactSignature && isa<GadgetOp>(target) &&
      isLegacySyndromeThreadedCall(*this, cast<GadgetOp>(target), signature);
  if (!exactSignature &&
      (isLocalGeneratedInvocation || !legacySyndromeThreading))
    return emitOpError("operand/result types must match the callee signature");

  if (auto profileRef = getProfileAttr()) {
    Operation *profileTarget =
        SymbolTable::lookupNearestSymbolFrom(*this, profileRef);
    if (profileTarget) {
      auto profile = dyn_cast<GadgetProfileOp>(profileTarget);
      if (!profile)
        return emitOpError("profile must resolve to fabric.gadget_profile");
      if (auto gadget = dyn_cast<GadgetOp>(target)) {
        if (profile.getGadgetAttr() != getCalleeAttr())
          return emitOpError("profile must analyze the called gadget");
      } else {
        auto protocol = cast<ProtocolOp>(target);
        if (!protocol.getPredicateProfileAttr() ||
            !protocol.getPredicateGadgetAttr() ||
            protocol.getPredicateProfileAttr() != profileRef ||
            profile.getGadgetAttr() != protocol.getPredicateGadgetAttr())
          return emitOpError(
              "protocol call profile must equal its typed predicate profile");
      }
    }
    // Link verification owns unresolved external profiles, but an unresolved
    // optional profile does not relax this call's local semantic obligations.
  }
  if (hasRemoteTrigger)
    return verifyCommunicationObligation(*this);
  return success();
}

LogicalResult RelocateOp::verify() {
  if (getSourceAttr() == getDestinationAttr())
    return emitOpError("source and destination spaces must differ");
  if (getTransition().empty() || getContinuityWitness().empty())
    return emitOpError("transition and continuity_witness must be nonempty");
  if (getStepAttr().getInt() < 0)
    return emitOpError("step must be nonnegative");
  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!target)
    return success(); // Partial linked modules resolve the callee later.
  FunctionType signature;
  if (auto gadget = dyn_cast<GadgetOp>(target))
    signature = gadget.getFunctionType();
  else if (auto protocol = dyn_cast<ProtocolOp>(target))
    signature = protocol.getFunctionType();
  else
    return emitOpError(
        "callee must resolve to fabric.gadget or fabric.protocol");
  if (signature.getNumInputs() != 1 || signature.getNumResults() != 1 ||
      signature.getInput(0) != getPatch().getType() ||
      signature.getResult(0) != getResult().getType())
    return emitOpError("callee must preserve exactly one compatible patch");
  return success();
}

LogicalResult EstablishSupportOp::verify() {
  if (getSpaces().empty())
    return emitOpError("requires at least one support space");
  if (getSpaces().size() != getSupportViews().size())
    return emitOpError("spaces and support_views must align");

  SmallVector<Attribute> seenSpaces;
  for (auto [index, value] : llvm::enumerate(getSpaces())) {
    auto space = dyn_cast<SymbolRefAttr>(value);
    if (!space)
      return emitOpError("spaces must contain symbol references");
    if (llvm::is_contained(seenSpaces, value))
      return emitOpError("support spaces must be unique");
    seenSpaces.push_back(value);
    auto view = dyn_cast<StringAttr>(getSupportViews()[index]);
    if (!view || view.getValue().empty())
      return emitOpError("support_views must contain nonempty strings");
  }
  if (getSpaces()[0] != getPrimaryAttr())
    return emitOpError("primary must be the first distributed support space");
  if (getOwnershipWitness().empty())
    return emitOpError("ownership_witness must be nonempty");
  for (Attribute value : getLinkObligations()) {
    auto obligation = dyn_cast<StringAttr>(value);
    if (!obligation || obligation.getValue().empty())
      return emitOpError("link_obligations must contain nonempty strings");
  }

  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!target)
    return success(); // A partially linked module retains the obligation.
  FunctionType signature;
  if (auto gadget = dyn_cast<GadgetOp>(target))
    signature = gadget.getFunctionType();
  else if (auto protocol = dyn_cast<ProtocolOp>(target))
    signature = protocol.getFunctionType();
  else
    return emitOpError(
        "callee must resolve to fabric.gadget or fabric.protocol");
  auto patchType = getPatch().getType();
  if (TypeRange(signature.getInputs()) != TypeRange{patchType} ||
      TypeRange(signature.getResults()) != TypeRange{patchType})
    return emitOpError(
        "distributed-support realization must preserve exactly one patch");
  return success();
}

LogicalResult EstablishTopologicalRecordOp::verify() {
  if (getRecord().empty())
    return emitOpError("record must be nonempty");
  if (getFrontier().empty())
    return emitOpError("frontier must be nonempty");
  SmallVector<StringRef> seen;
  for (Attribute value : getFrontier()) {
    auto item = dyn_cast<StringAttr>(value);
    if (!item || item.getValue().empty())
      return emitOpError("frontier must contain nonempty strings");
    if (llvm::is_contained(seen, item.getValue()))
      return emitOpError("frontier entries must be unique");
    seen.push_back(item.getValue());
  }
  if (getSupportWitness().empty() || getObservableWitness().empty())
    return emitOpError(
        "support_witness and observable_witness must be nonempty");

  Operation *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (!target)
    return success(); // A partially linked module retains the obligation.
  FunctionType signature;
  if (auto gadget = dyn_cast<GadgetOp>(target))
    signature = gadget.getFunctionType();
  else if (auto protocol = dyn_cast<ProtocolOp>(target))
    signature = protocol.getFunctionType();
  else
    return emitOpError(
        "callee must resolve to fabric.gadget or fabric.protocol");
  auto patchType = getPatch().getType();
  if (TypeRange(signature.getInputs()) != TypeRange{patchType} ||
      TypeRange(signature.getResults()) != TypeRange{patchType})
    return emitOpError(
        "topological-record realization must preserve exactly one patch");
  return success();
}

// event.selection's shared verifier (EventDialect.cpp) only checks mode/
// accept_when and that attempt/profile resolve to some symbol, since the
// shared `event` dialect cannot depend on `fabric` to check that attempt is
// specifically a fabric.gadget, that profile is its fabric.gadget_profile,
// or the outcome-map contract verifySelectedPredicateSemantics enforces.
// Walking every embedded event.selection in a protocol body from here
// restores those checks for the one dialect whose selections carry them.
static LogicalResult verifyEmbeddedSelections(Operation *protocolBody) {
  LogicalResult result = success();
  protocolBody->walk([&](qlx::event::SelectionOp selection) {
    if (failed(result))
      return;
    auto attemptAttr = selection.getAttemptAttr();
    if (!attemptAttr)
      return;
    Operation *attempt =
        SymbolTable::lookupNearestSymbolFrom(selection, attemptAttr);
    auto gadget = dyn_cast_or_null<GadgetOp>(attempt);
    if (!gadget) {
      selection.emitOpError("attempt must resolve to fabric.gadget");
      result = failure();
      return;
    }
    auto profileAttr = selection.getProfileAttr();
    if (!profileAttr)
      return;
    auto profile = dyn_cast_or_null<GadgetProfileOp>(
        SymbolTable::lookupNearestSymbolFrom(selection, profileAttr));
    if (!profile) {
      selection.emitOpError("profile must resolve to fabric.gadget_profile");
      result = failure();
      return;
    }
    if (profile.getGadgetAttr() != attemptAttr) {
      selection.emitOpError("profile must analyze the selection attempt");
      result = failure();
      return;
    }
    if (!gadget.getSpecAttr()) {
      selection.emitOpError("selected attempt requires a gadget spec");
      result = failure();
      return;
    }
    auto spec = dyn_cast_or_null<GadgetSpecOp>(
        SymbolTable::lookupNearestSymbolFrom(selection, gadget.getSpecAttr()));
    if (!spec) {
      selection.emitOpError("attempt spec must resolve to fabric.gadget_spec");
      result = failure();
      return;
    }
    if (failed(verifySelectedPredicateSemantics(
            selection.getOperation(), selection.getPredicate(), ValueRange{},
            attemptAttr, profileAttr, gadget, profile, spec, "selection")))
      result = failure();
  });
  return result;
}

LogicalResult ProduceResourceOp::verify() {
  bool hasLegacy = static_cast<bool>(getResourceAttr());
  bool hasSymbolic = static_cast<bool>(getResourceKindAttr());
  if (hasLegacy == hasSymbolic)
    return emitOpError(
        "requires exactly one of legacy resource or symbolic resource_kind");
  if (hasSymbolic) {
    auto type = cast<ResourceStateType>(getResourceState().getType());
    auto kind = dyn_cast<SymbolRefAttr>(type.getKind());
    if (!kind || kind != getResourceKindAttr())
      return emitOpError(
          "resource_kind must match the symbolic result resource type");
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

static LogicalResult verifyResourcePayloadRoles(Operation *owner,
                                                ArrayAttr roles,
                                                unsigned payloadCount) {
  if (!roles)
    return success();
  if (roles.size() != payloadCount)
    return owner->emitOpError(
        "payload_roles must exactly cover the ordered payloads");
  llvm::StringSet<> unique;
  for (Attribute value : roles) {
    auto role = dyn_cast<StringAttr>(value);
    if (!role || role.getValue().empty())
      return owner->emitOpError("payload_roles must contain nonempty strings");
    if (!unique.insert(role.getValue()).second)
      return owner->emitOpError("payload_roles must be unique");
  }
  return success();
}

LogicalResult UnpackResourceOp::verify() {
  unsigned count = getAnchors().size();
  if (count == 0)
    return emitOpError("requires at least one live anchor patch");
  if (getOutputs().size() != count * 2)
    return emitOpError("requires one successor and one payload per anchor");
  for (unsigned index = 0; index < count; ++index) {
    auto anchor = cast<PatchType>(getAnchors()[index].getType());
    auto successor = cast<PatchType>(getOutputs()[index].getType());
    if (anchor != successor)
      return emitOpError("must preserve every anchor patch type");
    auto payload = cast<PatchType>(getOutputs()[count + index].getType());
    if (!payload.getCodeType() || !payload.getEncoding() || !payload.getEpoch())
      return emitOpError(
          "every payload patch must name a code, encoding, and epoch");
  }

  if (failed(verifyResourcePayloadRoles(*this, getPayloadRolesAttr(), count)))
    return failure();

  auto blocks = getPayloadLogicalBlocksAttr();
  auto ports = getPayloadLogicalPortsAttr();
  auto blockIDs = getPayloadLogicalBlockIdsAttr();
  auto action = getPayloadActionAttr();
  if (static_cast<bool>(blocks) != static_cast<bool>(ports))
    return emitOpError(
        "payload logical block and port maps must appear together");
  if (count > 1 && (!blocks || !action))
    return emitOpError(
        "multi-payload unpack requires an action and exact logical maps");
  if (blocks && !action)
    return emitOpError("payload logical maps require a typed payload_action");
  if (action &&
      (action.getValue() == qlx::BuiltinAction::ccz ||
       action.getValue() == qlx::BuiltinAction::ccx) &&
      !blockIDs) {
    auto protocol = (*this)->getParentOfType<ProtocolOp>();
    auto objective = protocol ? dyn_cast_or_null<qlx::BuiltinActionAttr>(
                                    protocol.getObjectiveAttr())
                              : qlx::BuiltinActionAttr{};
    if (!protocol || protocol.getActionSiteAttr() || objective)
      return emitOpError(
          "selected three-qubit magic-state handoff requires exact QEC block "
          "identities");
  }
  if (blockIDs && (!blocks || blockIDs.size() != blocks.size()))
    return emitOpError(
        "payload logical block identities must exactly cover the logical map");
  if (!blocks)
    return verifyRegisteredResourceUnpack(*this);
  if (blocks.size() == 0 || blocks.size() != ports.size())
    return emitOpError(
        "payload logical block and port maps must have equal nonzero length");
  auto arity = builtinActionArity(action.getValue());
  if (!arity)
    return emitOpError("payload_action must have a fixed logical arity");
  if (blocks.size() != *arity)
    return emitOpError(
        "payload logical maps must exactly cover the action arity");
  if (auto protocol = (*this)->getParentOfType<ProtocolOp>()) {
    if (auto objective = dyn_cast_or_null<qlx::BuiltinActionAttr>(
            protocol.getObjectiveAttr());
        objective && objective != action)
      return emitOpError(
          "payload_action must match the enclosing protocol objective");
  }
  llvm::SmallDenseSet<std::pair<int64_t, int64_t>, 8> mappings;
  llvm::SmallDenseSet<int64_t, 8> coveredBlocks;
  llvm::StringMap<int64_t> indexByBlockID;
  SmallVector<StringRef, 8> blockIDByIndex(count);
  auto rawBlockIDs = blockIDs ? blockIDs.getValue() : ArrayRef<Attribute>{};
  for (auto [ordinal, pair] :
       llvm::enumerate(llvm::zip(blocks.asArrayRef(), ports.asArrayRef()))) {
    auto [block, port] = pair;
    if (block < 0 || block >= static_cast<int64_t>(count))
      return emitOpError("payload logical block index is out of range");
    if (port < 0)
      return emitOpError("payload logical port index must be nonnegative");
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
        return emitOpError(
            "one payload block maps to several selected QEC block identities");
    }
    auto payload = cast<PatchType>(getOutputs()[count + block].getType());
    auto *target =
        SymbolTable::lookupNearestSymbolFrom(*this, payload.getEncoding());
    auto encoding = dyn_cast_or_null<EncodingOp>(target);
    if (!encoding)
      return emitOpError(
          "mapped payload encoding must resolve to fabric.encoding");
    if (port >= static_cast<int64_t>(encoding.getLogicalPorts().size()))
      return emitOpError(
          "payload logical port index exceeds the encoding logical capacity");
  }
  if (!getPayloadRolesAttr() && coveredBlocks.size() != count)
    return emitOpError("payload logical maps must cover every payload block");
  if (blockIDs && indexByBlockID.size() != coveredBlocks.size())
    return emitOpError("payload logical block identities must exactly name "
                       "the action-bearing payload owners");
  return verifyRegisteredResourceUnpack(*this);
}

LogicalResult PackResourceOp::verify() {
  auto resource = cast<ResourceStateType>(getResource().getType());
  auto kind = dyn_cast<SymbolRefAttr>(resource.getKind());
  if (!kind)
    return emitOpError(
        "canonical pack_resource requires an open symbolic resource kind");
  if (kind != getResourceKindAttr())
    return emitOpError(
        "resource_kind must match the packed result resource type");
  if (getPayloads().empty())
    return emitOpError("requires at least one encoded payload patch");
  if (getPayloadEncodings().size() != getPayloads().size())
    return emitOpError(
        "payload_encodings must exactly cover the consumed payload patches");
  if (failed(verifyResourcePayloadRoles(*this, getPayloadRolesAttr(),
                                        getPayloads().size())))
    return failure();
  for (auto [value, encodingAttr] :
       llvm::zip(getPayloads(), getPayloadEncodings())) {
    auto patch = cast<PatchType>(value.getType());
    if (!patch.getEncoding())
      return emitOpError("packed payload must name its concrete encoding");
    auto encoding = dyn_cast<FlatSymbolRefAttr>(encodingAttr);
    if (!encoding || encoding != patch.getEncoding())
      return emitOpError(
          "each payload_encodings entry must match its consumed patch type");
  }
  return verifyRegisteredResourcePack(*this);
}

LogicalResult AllZeroOp::verify() {
  auto tensor = cast<RankedTensorType>(getBits().getType());
  if (tensor.getRank() != 1 || !tensor.getElementType().isInteger(1))
    return emitOpError("requires a tensor<Nxi1> measurement bundle");
  return success();
}

LogicalResult ParityOp::verify() {
  if (getBits().empty())
    return emitOpError("requires one or more measurement bundles");
  for (Value bits : getBits()) {
    auto tensor = dyn_cast<RankedTensorType>(bits.getType());
    if (!tensor || tensor.getRank() != 1 ||
        !tensor.getElementType().isInteger(1))
      return emitOpError("requires tensor<Nxi1> measurement bundles");
  }
  return success();
}

LogicalResult AllFalseOp::verify() {
  if (getEvents().empty())
    return emitOpError("requires one or more classical events");
  return success();
}

LogicalResult MppOp::verify() {
  auto indices = getIndices();
  StringRef paulis = getPaulis();
  if (indices.empty())
    return emitOpError("requires at least one physical carrier");
  // A single optional leading '-' encodes a negated product (sign -1); the
  // recorded outcome is the complement of the unsigned product's outcome.
  paulis.consume_front("-");
  if (paulis.empty())
    return emitOpError("paulis must contain at least one Pauli after the "
                       "optional leading '-' sign");
  if (paulis.size() != indices.size())
    return emitOpError(
        "paulis width must equal indices width (ignoring an optional "
        "leading '-')");
  if (llvm::any_of(paulis, [](char value) {
        return value != 'X' && value != 'Y' && value != 'Z';
      }))
    return emitOpError(
        "paulis must contain only X, Y, and Z after the optional leading '-'");
  auto bits = dyn_cast<RankedTensorType>(getBits().getType());
  if (!bits || bits.getRank() != 1 || bits.getDimSize(0) != 1 ||
      !bits.getElementType().isInteger(1))
    return emitOpError("must return tensor<1xi1>");
  return success();
}

LogicalResult InjectOp::verify() {
  auto module = (*this)->getParentOfType<ModuleOp>();
  auto stages =
      module ? module->getAttrOfType<ArrayAttr>("qlx.stages") : ArrayAttr();
  if (stages) {
    for (Attribute value : stages) {
      auto stage = dyn_cast<StringAttr>(value);
      if (stage && (stage.getValue() == "p2" || stage.getValue() == "p3"))
        return emitOpError(
            "is legacy logical intent and is not legal in canonical P2/P3; "
            "select a concrete protocol using fabric.unpack_resource");
    }
    return success();
  }
  auto profiles =
      module ? module->getAttrOfType<ArrayAttr>("qlx.profiles") : ArrayAttr();
  if (!profiles)
    return success(); // Unprofiled alpha/import modules remain parseable.
  for (Attribute value : profiles) {
    auto profile = dyn_cast<StringAttr>(value);
    if (profile &&
        (profile.getValue().starts_with("p2") || profile.getValue() == "p3"))
      return emitOpError(
          "is legacy logical intent and is not legal in canonical P2/P3; "
          "select a concrete protocol using fabric.unpack_resource");
  }
  return success();
}

LogicalResult EncodingUnpackOp::verify() {
  auto bundle = cast<PatchBundleType>(getChildren().getType());
  if (bundle.getHierarchy() != getHierarchyAttr())
    return emitOpError("result bundle hierarchy must match 'hierarchy'");
  if (bundle.getSlotGroup() != getSlotGroupAttr())
    return emitOpError("result bundle slot group must match 'slot_group'");
  auto *target =
      SymbolTable::lookupNearestSymbolFrom(*this, getHierarchyAttr());
  if (!target)
    return success();
  auto hierarchy = dyn_cast<EncodingHierarchyOp>(target);
  if (!hierarchy)
    return emitOpError("hierarchy must resolve to fabric.encoding_hierarchy");
  auto parent = cast<PatchType>(getParent().getType());
  if (parent.getCodeType() != hierarchy.getCodeAttr())
    return emitOpError("parent patch code must match hierarchy composite code");
  return success();
}

LogicalResult MapChildrenOp::verify() {
  auto bundle = cast<PatchBundleType>(getChildren().getType());
  if (bundle.getSlotGroup() != getSlotGroupAttr())
    return emitOpError("bundle slot group must match 'slot_group'");
  auto *callee = SymbolTable::lookupNearestSymbolFrom(*this, getCalleeAttr());
  if (callee && !isa<GadgetOp, ProtocolOp>(callee))
    return emitOpError(
        "callee must resolve to fabric.gadget or fabric.protocol");
  return success();
}

LogicalResult EncodingPackOp::verify() {
  auto bundle = cast<PatchBundleType>(getChildren().getType());
  auto *target = SymbolTable::lookupNearestSymbolFrom(*this, getEncodingAttr());
  if (!target)
    return success();
  auto encoding = dyn_cast<EncodingOp>(target);
  if (!encoding)
    return emitOpError("encoding must resolve to fabric.encoding");
  if (!encoding.getHierarchy() ||
      *encoding.getHierarchy() != bundle.getHierarchy().getValue())
    return emitOpError("bundle hierarchy must match target encoding hierarchy");
  auto parent = cast<PatchType>(getParent().getType());
  if (parent.getCodeType() != encoding.getCodeAttr())
    return emitOpError("parent patch code must match target encoding code");
  return success();
}

LogicalResult RegionOp::verify() {
  if (getEpochAttr() && !getEncodingAttr())
    return emitOpError("epoch requires an encoding-qualified region");
  if (!getEncodingAttr())
    return success();
  auto type = PatchType::get(getContext(), getCodeAttr(), getEncodingAttr(),
                             getEpochAttr());
  return verifyEncodingQualifiedTypes(*this, TypeRange{type});
}

LogicalResult AllocOp::verify() {
  auto patch = getResult().getType();
  if (getCodeAttr() != patch.getCodeType())
    return emitOpError("code must match the allocated patch code");
  if (failed(verifyEncodingQualifiedTypes(*this, TypeRange{patch})))
    return failure();

  // An unmarked allocation carries a provider-facing hint, not a concrete
  // claim on an ambient machine pool.  Only strict allocations resolve here.
  if (!getStrictRegionAttr())
    return success();

  auto module = getOperation()->getParentOfType<ModuleOp>();
  if (!module)
    return success();

  SmallVector<RegionOp, 2> matches;
  bool hasMachine = false;
  for (auto machine : module.getOps<DeviceOp>()) {
    hasMachine = true;
    for (auto region : machine.getBody().getOps<RegionOp>())
      if (region.getSymName() == getRegion())
        matches.push_back(region);
  }
  // Detached reusable definitions remain legal before a device is linked.
  if (!hasMachine)
    return success();
  if (matches.empty()) {
    return emitOpError("region @")
           << getRegion() << " does not resolve in the linked fabric.machine";
  }
  if (matches.size() != 1)
    return emitOpError("region @")
           << getRegion() << " is ambiguous across linked fabric.machine ops";

  RegionOp region = matches.front();
  if (region.getCodeAttr() != getCodeAttr())
    return emitOpError("code @")
           << getCode() << " does not match resolved region @" << getRegion()
           << " code " << region.getCodeAttr();
  if (auto encoding = region.getEncodingAttr()) {
    if (!patch.getEncoding())
      return emitOpError("resolved region @")
             << getRegion() << " requires an encoding-qualified patch";
    if (encoding != patch.getEncoding())
      return emitOpError("patch encoding ")
             << patch.getEncoding() << " does not match resolved region @"
             << getRegion() << " encoding " << encoding;
  }
  if (auto epoch = region.getEpochAttr()) {
    if (!patch.getEpoch())
      return emitOpError("resolved region @")
             << getRegion() << " requires an epoch-qualified patch";
    if (epoch != patch.getEpoch())
      return emitOpError("patch epoch ")
             << patch.getEpoch() << " does not match resolved region @"
             << getRegion() << " epoch " << epoch;
  }
  return success();
}

// Custom parser shared by CXOp and CZOp.
//
// Single-patch form (existing):
//   %r = fabric.cx %p sx -> data {schedule = "hx"} : !fabric.patch<@c>
//
// Two-patch (cross) form:
//   %a', %b' = fabric.cx %a, %b sx -> data {pairs = "0:0,1:1"}
//              : (!fabric.patch<@a>, !fabric.patch<@b>)
//                  -> (!fabric.patch<@a>, !fabric.patch<@b>)
//
// We decide which form to parse by peeking at the type list after the
// final ':'. A leading '(' means functional-type (two-patch); anything
// else falls back to a bare type (single-patch).
static ParseResult parseTwoPatchOpForm(OpAsmParser &parser,
                                       OperationState &result) {
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  if (parser.parseOperandList(operands))
    return failure();

  // Parse `ctrl_partition -> targ_partition` using the same keyword
  // syntax as the existing single-patch form did.
  PartitionAttr ctrlAttr, targAttr;
  if (parsePartitionKeyword(parser, ctrlAttr))
    return failure();
  result.addAttribute("ctrl", ctrlAttr);
  if (parser.parseArrow())
    return failure();
  if (parsePartitionKeyword(parser, targAttr))
    return failure();
  result.addAttribute("targ", targAttr);

  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();
  if (parser.parseColon())
    return failure();

  // Peek: '(' means functional-type form; otherwise bare type.
  if (succeeded(parser.parseOptionalLParen())) {
    SmallVector<Type> inputTypes, resultTypes;
    if (parser.parseTypeList(inputTypes) || parser.parseRParen() ||
        parser.parseArrow())
      return failure();
    if (succeeded(parser.parseOptionalLParen())) {
      if (parser.parseTypeList(resultTypes) || parser.parseRParen())
        return failure();
    } else {
      Type single;
      if (parser.parseType(single))
        return failure();
      resultTypes.push_back(single);
    }
    if (operands.size() != inputTypes.size())
      return parser.emitError(parser.getCurrentLocation(),
                              "operand count must match input type count");
    if (parser.resolveOperands(operands, inputTypes,
                               parser.getCurrentLocation(), result.operands))
      return failure();
    result.addTypes(resultTypes);
    return success();
  }

  // Single-type form: accept both the dialect-stripped `<@sym>` shorthand
  // and the full `!fabric.patch<@sym>` form, mirroring `type($result)`
  // in the auto-generated tablegen parsers.
  PatchType singleTy;
  if (parser.parseCustomTypeWithFallback(singleTy))
    return failure();
  if (operands.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "single-type form requires exactly one patch "
                            "operand");
  if (parser.resolveOperands(operands, {Type(singleTy)},
                             parser.getCurrentLocation(), result.operands))
    return failure();
  result.addTypes({Type(singleTy)});
  return success();
}

static void printTwoPatchOpForm(OpAsmPrinter &printer, Operation *op,
                                ValueRange operands, ResultRange results,
                                Attribute ctrl, Attribute targ,
                                ArrayRef<StringRef> elidedAttrs) {
  printer << " ";
  printer.printOperands(operands);
  printer << " ";
  if (auto p = llvm::dyn_cast<PartitionAttr>(ctrl))
    printer << stringifyPartition(p.getValue());
  printer << " -> ";
  if (auto p = llvm::dyn_cast<PartitionAttr>(targ))
    printer << stringifyPartition(p.getValue());
  printer.printOptionalAttrDict(op->getAttrs(), elidedAttrs);
  printer << " : ";
  if (operands.size() == 1 && results.size() == 1) {
    // Mirror tablegen `type($result)` behaviour: strip the dialect
    // prefix so a Fabric patch type prints as ``<@sym>`` instead of
    // ``!fabric.patch<@sym>``.
    Type ty = operands[0].getType();
    if (auto pt = llvm::dyn_cast<PatchType>(ty)) {
      printer.printStrippedAttrOrType(pt);
      return;
    }
    // Patch-frame values use the functional form below.  Unlike PatchType,
    // PatchFrameType has no legacy stripped shorthand accepted by this
    // custom parser.
  }
  printer << "(";
  llvm::interleaveComma(operands, printer.getStream(),
                        [&](Value v) { printer << v.getType(); });
  printer << ") -> (";
  llvm::interleaveComma(results, printer.getStream(),
                        [&](Value v) { printer << v.getType(); });
  printer << ")";
}

ParseResult CXOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTwoPatchOpForm(parser, result);
}

void CXOp::print(OpAsmPrinter &printer) {
  printTwoPatchOpForm(printer, getOperation(), getPatches(), getResults(),
                      getCtrlAttr(), getTargAttr(),
                      /*elidedAttrs=*/{"ctrl", "targ"});
}

ParseResult CZOp::parse(OpAsmParser &parser, OperationState &result) {
  return parseTwoPatchOpForm(parser, result);
}

void CZOp::print(OpAsmPrinter &printer) {
  printTwoPatchOpForm(printer, getOperation(), getPatches(), getResults(),
                      getCtrlAttr(), getTargAttr(),
                      /*elidedAttrs=*/{"ctrl", "targ"});
}

static LogicalResult verifyCXLike(Operation *op, ValueRange operands,
                                  ResultRange results, bool hasSchedule,
                                  std::optional<StringRef> pairs,
                                  Partition ctrl, Partition targ) {
  bool hasPairs = pairs.has_value();
  unsigned n = operands.size();
  if (n != 1 && n != 2)
    return op->emitOpError("expected 1 or 2 patch operands, got ") << n;
  if (results.size() != n)
    return op->emitOpError("result count (")
           << results.size() << ") must equal operand count (" << n << ")";
  for (unsigned i = 0; i < n; ++i)
    if (operands[i].getType() != results[i].getType())
      return op->emitOpError("operand/result type mismatch at index ") << i;
  // Single-patch: exactly one of schedule/pairs. Cross-patch: pairs only
  // — schedule= "hx"/"hz" are intra-patch concepts that resolve via the
  // single patch's parity matrices.
  if (n == 1) {
    if (hasSchedule == hasPairs)
      return op->emitOpError("exactly one of 'schedule' or 'pairs' must be set "
                             "for single-patch form");
  } else {
    if (hasSchedule)
      return op->emitOpError("two-patch form does not accept 'schedule'; use "
                             "'pairs' to enumerate cross-patch CX pairs");
    if (!hasPairs)
      return op->emitOpError("two-patch form requires 'pairs'");
  }

  if (!pairs)
    return success();

  auto partitionWidth = [&](Type type,
                            Partition partition) -> std::optional<int64_t> {
    auto patch = dyn_cast<PatchType>(type);
    if (!patch)
      return std::nullopt;
    auto *target =
        SymbolTable::lookupNearestSymbolFrom(op, patch.getCodeType());
    auto code = dyn_cast_or_null<CodeOp>(target);
    if (!code)
      return std::nullopt;
    DictionaryAttr partitions = code.getPartitions();
    if (partition == Partition::all) {
      int64_t total = 0;
      for (NamedAttribute entry : partitions) {
        auto value = dyn_cast<IntegerAttr>(entry.getValue());
        if (!value)
          return std::nullopt;
        total += value.getInt();
      }
      return total;
    }
    auto value = dyn_cast_or_null<IntegerAttr>(
        partitions.get(stringifyPartition(partition)));
    if (!value)
      return std::nullopt;
    return value.getInt();
  };

  auto ctrlWidth = partitionWidth(operands.front().getType(), ctrl);
  auto targWidth = partitionWidth(operands[n == 1 ? 0 : 1].getType(), targ);
  auto partitionBase = [&](Type type,
                           Partition partition) -> std::optional<int64_t> {
    auto patch = dyn_cast<PatchType>(type);
    if (!patch)
      return std::nullopt;
    auto *target =
        SymbolTable::lookupNearestSymbolFrom(op, patch.getCodeType());
    auto code = dyn_cast_or_null<CodeOp>(target);
    if (!code)
      return std::nullopt;
    DictionaryAttr partitions = code.getPartitions();
    auto size = [&](StringRef name) -> std::optional<int64_t> {
      auto value = dyn_cast_or_null<IntegerAttr>(partitions.get(name));
      return value ? std::optional<int64_t>(value.getInt()) : std::nullopt;
    };
    switch (partition) {
    case Partition::data:
    case Partition::all:
      return 0;
    case Partition::sx:
      return size("data");
    case Partition::sz: {
      auto data = size("data");
      auto sx = size("sx");
      if (!data || !sx)
        return std::nullopt;
      return *data + *sx;
    }
    }
    return std::nullopt;
  };
  auto ctrlBase = partitionBase(operands.front().getType(), ctrl);
  auto targBase = partitionBase(operands[n == 1 ? 0 : 1].getType(), targ);
  auto rejectSelfAlias = [&](int64_t control, int64_t target) -> LogicalResult {
    if (n == 1 && ctrlBase && targBase &&
        *ctrlBase + control == *targBase + target)
      return op->emitOpError(
                 "'pairs' must select two distinct carriers; control index ")
             << control << " and target index " << target
             << " alias the same patch carrier";
    return success();
  };
  StringRef authoredRelation = *pairs;
  StringRef relation = authoredRelation.trim();
  if (relation != authoredRelation)
    return op->emitOpError("'pairs' must not contain surrounding whitespace");
  if (relation == "index") {
    if (ctrlWidth && targWidth && std::min(*ctrlWidth, *targWidth) == 0)
      return op->emitOpError(
          "'index' pairs relation selects no carrier interactions");
    if (ctrlWidth && targWidth)
      for (int64_t index = 0; index < std::min(*ctrlWidth, *targWidth); ++index)
        if (failed(rejectSelfAlias(index, index)))
          return failure();
    return success();
  }
  if (relation.empty())
    return op->emitOpError(
        "'pairs' must be 'index' or a nonempty c:t comma-separated relation");
  if (relation.starts_with(',') || relation.ends_with(','))
    return op->emitOpError(
        "'pairs' entries must use the canonical c:t grammar");

  while (!relation.empty()) {
    auto [entry, rest] = relation.split(',');
    relation = rest;
    if (entry != entry.trim())
      return op->emitOpError(
          "'pairs' entries must use the canonical c:t grammar");
    size_t colon = entry.find(':');
    if (colon == StringRef::npos ||
        entry.find(':', colon + 1) != StringRef::npos)
      return op->emitOpError(
          "'pairs' entries must use the canonical c:t grammar");
    int64_t control = -1;
    int64_t target = -1;
    StringRef controlText = entry.take_front(colon);
    StringRef targetText = entry.drop_front(colon + 1);
    if (controlText != controlText.trim() || targetText != targetText.trim() ||
        controlText.getAsInteger(10, control) ||
        targetText.getAsInteger(10, target) || control < 0 || target < 0)
      return op->emitOpError(
          "'pairs' entries must contain nonnegative integer indices");
    if (ctrlWidth && control >= *ctrlWidth)
      return op->emitOpError("'pairs' control index ")
             << control << " is outside partition width " << *ctrlWidth;
    if (targWidth && target >= *targWidth)
      return op->emitOpError("'pairs' target index ")
             << target << " is outside partition width " << *targWidth;
    if (failed(rejectSelfAlias(control, target)))
      return failure();
  }
  return success();
}

LogicalResult CXOp::verify() {
  return verifyCXLike(getOperation(), getPatches(), getResults(),
                      getSchedule().has_value(), getPairs(), getCtrl(),
                      getTarg());
}

LogicalResult CZOp::verify() {
  return verifyCXLike(getOperation(), getPatches(), getResults(),
                      getSchedule().has_value(), getPairs(), getCtrl(),
                      getTarg());
}

LogicalResult BarrierOp::verify() {
  if (getPatches().size() != getResults().size())
    return emitOpError("number of patch inputs (")
           << getPatches().size() << ") must equal number of results ("
           << getResults().size() << ")";
  for (unsigned i = 0; i < getPatches().size(); ++i) {
    if (getPatches()[i].getType() != getResults()[i].getType())
      return emitOpError("patch/result type mismatch at index ") << i;
  }
  return success();
}

static LogicalResult verifyProductTermEncoding(Operation *op,
                                               OperandRange patches,
                                               ResultRange patchResults,
                                               ArrayRef<int64_t> patchIdx,
                                               ArrayRef<int64_t> logicalIdx,
                                               StringRef pauliProduct) {
  if (patches.empty())
    return op->emitOpError("requires at least one patch operand");
  if (patchResults.size() != patches.size())
    return op->emitOpError("number of patch results (")
           << patchResults.size() << ") must equal number of patch operands ("
           << patches.size() << ")";
  for (auto [i, pair] : llvm::enumerate(llvm::zip(patches, patchResults))) {
    auto inTy = std::get<0>(pair).getType();
    auto outTy = std::get<1>(pair).getType();
    if (inTy != outTy)
      return op->emitOpError("patch result type mismatch at index ")
             << i << ": expected " << inTy << ", got " << outTy;
  }

  // A single optional leading '-' encodes a negated product (sign -1). The
  // remaining characters are the per-term Pauli labels.
  pauliProduct.consume_front("-");
  if (pauliProduct.empty())
    return op->emitOpError(
        "pauli_product must contain at least one Pauli term after the "
        "optional leading '-' sign");
  if (patchIdx.size() != logicalIdx.size() ||
      patchIdx.size() != pauliProduct.size()) {
    return op->emitOpError(
               "patch_indices, logical_indices, and pauli_product must have "
               "equal length (ignoring an optional leading '-'); got ")
           << patchIdx.size() << ", " << logicalIdx.size() << ", "
           << pauliProduct.size();
  }
  for (auto [i, idx] : llvm::enumerate(patchIdx)) {
    if (idx < 0 || idx >= static_cast<int64_t>(patches.size()))
      return op->emitOpError("patch_indices[")
             << i << "] = " << idx << " out of range for " << patches.size()
             << " patch operand(s)";
  }
  SmallVector<CodeOp> termCodes;
  termCodes.reserve(logicalIdx.size());
  llvm::DenseSet<std::pair<int64_t, int64_t>> addressedTerms;
  for (auto [i, idx] : llvm::enumerate(logicalIdx)) {
    if (idx < 0)
      return op->emitOpError("logical_indices[")
             << i << "] must be non-negative, got " << idx;
    auto patchType = dyn_cast<PatchType>(patches[patchIdx[i]].getType());
    auto *target =
        SymbolTable::lookupNearestSymbolFrom(op, patchType.getCodeType());
    if (!target) {
      // Reusable fragments may carry an unresolved external code reference.
      // Link verification owns resolution; checks that require the code's
      // protected/gauge split below still fail closed.
      termCodes.push_back(CodeOp{});
      if (!addressedTerms.insert({patchIdx[i], idx}).second)
        return op->emitOpError("product repeats logical address (patch ")
               << patchIdx[i] << ", logical " << idx << ")";
      continue;
    }
    auto code = dyn_cast_or_null<CodeOp>(target);
    if (!code)
      return op->emitOpError("patch term ")
             << i << " code " << patchType.getCodeType()
             << " must resolve to fabric.code";
    int64_t k = code.getK().value_or(1);
    int64_t r = code.getR().value_or(0);
    if (idx >= k + r)
      return op->emitOpError("logical_indices[")
             << i << "] = " << idx << " out of range for code @"
             << code.getSymName() << " protected/gauge ports [0, " << k + r
             << ")";
    if (!addressedTerms.insert({patchIdx[i], idx}).second)
      return op->emitOpError("product repeats logical address (patch ")
             << patchIdx[i] << ", logical " << idx << ")";
    termCodes.push_back(code);
  }
  Attribute subsystemKindsAttr = op->getAttr("subsystem_kinds");
  Attribute subsystemIndicesAttr = op->getAttr("subsystem_indices");
  if (static_cast<bool>(subsystemKindsAttr) !=
      static_cast<bool>(subsystemIndicesAttr))
    return op->emitOpError(
        "subsystem_kinds and subsystem_indices must be specified together");
  if (subsystemKindsAttr) {
    auto subsystemKinds = dyn_cast<ArrayAttr>(subsystemKindsAttr);
    auto subsystemIndices = dyn_cast<DenseI64ArrayAttr>(subsystemIndicesAttr);
    if (!subsystemKinds || !subsystemIndices)
      return op->emitOpError(
          "subsystem_kinds must be a string array and subsystem_indices must "
          "be a dense i64 array");
    if (subsystemKinds.size() != logicalIdx.size() ||
        subsystemIndices.size() != logicalIdx.size())
      return op->emitOpError(
                 "subsystem_kinds and subsystem_indices must match the product "
                 "term count; got ")
             << subsystemKinds.size() << " and " << subsystemIndices.size()
             << " for " << logicalIdx.size() << " term(s)";
    for (auto [i, pair] : llvm::enumerate(
             llvm::zip(subsystemKinds, subsystemIndices.asArrayRef()))) {
      auto kind = dyn_cast<StringAttr>(std::get<0>(pair));
      int64_t subsystemIndex = std::get<1>(pair);
      if (!kind)
        return op->emitOpError("subsystem_kinds[") << i << "] must be a string";
      if (!termCodes[i])
        return op->emitOpError("patch term ")
               << i << " code "
               << cast<PatchType>(patches[patchIdx[i]].getType()).getCodeType()
               << " must resolve to fabric.code when subsystem metadata is "
                  "present";
      int64_t k = termCodes[i].getK().value_or(1);
      int64_t r = termCodes[i].getR().value_or(0);
      int64_t expectedLogicalIndex = -1;
      int64_t subsystemCapacity = -1;
      if (kind.getValue() == "protected") {
        expectedLogicalIndex = subsystemIndex;
        subsystemCapacity = k;
      } else if (kind.getValue() == "gauge") {
        expectedLogicalIndex = k + subsystemIndex;
        subsystemCapacity = r;
      } else {
        return op->emitOpError("subsystem_kinds[")
               << i << "] must be 'protected' or 'gauge', got '"
               << kind.getValue() << "'";
      }
      if (subsystemIndex < 0 || subsystemIndex >= subsystemCapacity)
        return op->emitOpError("subsystem_indices[")
               << i << "] = " << subsystemIndex << " out of range for "
               << kind.getValue() << " ports [0, " << subsystemCapacity << ")";
      if (logicalIdx[i] != expectedLogicalIndex)
        return op->emitOpError("logical_indices[")
               << i << "] = " << logicalIdx[i] << " is inconsistent with "
               << kind.getValue() << " subsystem index " << subsystemIndex
               << "; expected " << expectedLogicalIndex;
    }
  }
  for (char c : pauliProduct) {
    if (c != 'X' && c != 'Y' && c != 'Z')
      return op->emitOpError("pauli_product must contain only X/Y/Z after "
                             "the optional leading '-'; got '")
             << c << "'";
  }
  return success();
}

LogicalResult MeasureProductOp::verify() {
  return verifyProductTermEncoding(*this, getPatches(), getPatchResults(),
                                   getPatchIndices(), getLogicalIndices(),
                                   getPauliProduct());
}

LogicalResult RotateProductOp::verify() {
  StringRef synthesis = getSynthesis();
  if (synthesis != "auto" && synthesis != "native" && synthesis != "decompose")
    return emitOpError("synthesis must be one of auto, native, decompose; got ")
           << synthesis;
  return verifyProductTermEncoding(*this, getPatches(), getPatchResults(),
                                   getPatchIndices(), getLogicalIndices(),
                                   getPauliProduct());
}

LogicalResult ResourceRotateProductOp::verify() {
  auto resource = dyn_cast<ResourceStateType>(getResource().getType());
  if (!resource)
    return emitOpError("requires one typed resource owner");
  return verifyProductTermEncoding(*this, getPatches(), getPatchResults(),
                                   getPatchIndices(), getLogicalIndices(),
                                   getPauliProduct());
}

/// GF(2) row rank of equal-width bit rows. Shared by the canonical
/// symplectic checks and the CSS support-array checks below.
static int64_t gf2RowRank(ArrayRef<llvm::SmallBitVector> source) {
  SmallVector<llvm::SmallBitVector> rows(source.begin(), source.end());
  int64_t result = 0;
  int64_t width = rows.empty() ? 0 : rows.front().size();
  for (int64_t column = width - 1; column >= 0; --column) {
    auto pivot =
        llvm::find_if(llvm::drop_begin(rows, result),
                      [&](const auto &row) { return row.test(column); });
    if (pivot == rows.end())
      continue;
    std::iter_swap(rows.begin() + result, pivot);
    for (int64_t index = 0; index < static_cast<int64_t>(rows.size()); ++index)
      if (index != result && rows[index].test(column))
        rows[index] ^= rows[result];
    ++result;
    if (result == static_cast<int64_t>(rows.size()))
      break;
  }
  return result;
}

/// Verify the CSS support-array adapter form of a fabric.code that carries
/// no canonical symplectic attributes: hx/hz/gx/gz (and lx/lz) list the
/// data-qubit supports of X-type and Z-type operators. This is the only
/// algebra such a declaration carries, so it is checked directly over GF(2)
/// instead of trusting imported IR to be "normalized later". Two same-basis
/// CSS operators always commute; an X-type and a Z-type operator commute iff
/// their supports intersect evenly, and canonical logical/gauge partners must
/// anticommute exactly on the diagonal.
static LogicalResult verifyCssSupportAlgebra(CodeOp op, int64_t k,
                                             int64_t dataSize) {
  std::optional<ArrayAttr> hx = op.getHx();
  std::optional<ArrayAttr> hz = op.getHz();
  std::optional<ArrayAttr> gx = op.getGx();
  std::optional<ArrayAttr> gz = op.getGz();
  std::optional<ArrayAttr> lx = op.getLx();
  std::optional<ArrayAttr> lz = op.getLz();
  if (!hx && !hz && !gx && !gz)
    return success(); // No CSS adapter payload declared.

  // Supports are bounded by n (falling back to the data partition size).
  // When neither is declared the widest index defines the row width so the
  // intersection parities below stay well defined.
  int64_t bound = op.getN().value_or(dataSize);
  int64_t width = bound;
  if (width < 0) {
    for (std::optional<ArrayAttr> family : {hx, hz, gx, gz, lx, lz})
      if (family)
        for (Attribute entry : *family)
          if (auto support = dyn_cast<DenseI64ArrayAttr>(entry))
            for (int64_t index : support.asArrayRef())
              width = std::max(width, index + 1);
    width = std::max<int64_t>(width, 1);
  }

  auto supportRows = [&](StringRef label, std::optional<ArrayAttr> family)
      -> FailureOr<SmallVector<llvm::SmallBitVector>> {
    SmallVector<llvm::SmallBitVector> rows;
    if (!family)
      return rows;
    for (auto [i, entry] : llvm::enumerate(*family)) {
      auto support = dyn_cast<DenseI64ArrayAttr>(entry);
      if (!support) {
        op.emitOpError(label)
            << "[" << i << "] must be a dense i64 array (array<i64: ...>)";
        return failure();
      }
      if (support.empty()) {
        op.emitOpError(label) << "[" << i << "] must be non-empty";
        return failure();
      }
      llvm::SmallBitVector row(width);
      for (auto [j, index] : llvm::enumerate(support.asArrayRef())) {
        if (index < 0 || (bound >= 0 && index >= bound)) {
          op.emitOpError(label)
              << "[" << i << "][" << j << "] qubit index " << index
              << " out of range for " << bound << " data qubit(s)";
          return failure();
        }
        if (row.test(index)) {
          op.emitOpError(label)
              << "[" << i << "] repeats qubit index " << index;
          return failure();
        }
        row.set(index);
      }
      rows.push_back(std::move(row));
    }
    return rows;
  };
  auto hxRows = supportRows("hx", hx);
  auto hzRows = supportRows("hz", hz);
  auto gxRows = supportRows("gx", gx);
  auto gzRows = supportRows("gz", gz);
  auto lxRows = supportRows("lx", lx);
  auto lzRows = supportRows("lz", lz);
  if (failed(hxRows) || failed(hzRows) || failed(gxRows) || failed(gzRows) ||
      failed(lxRows) || failed(lzRows))
    return failure();

  auto oddIntersection = [](const llvm::SmallBitVector &left,
                            const llvm::SmallBitVector &right) {
    llvm::SmallBitVector meet = left;
    meet &= right;
    return (meet.count() % 2) == 1;
  };
  auto checkFamilies = [&](ArrayRef<llvm::SmallBitVector> left,
                           ArrayRef<llvm::SmallBitVector> right,
                           StringRef equation, bool paired) -> LogicalResult {
    for (auto [i, lhs] : llvm::enumerate(left))
      for (auto [j, rhs] : llvm::enumerate(right))
        if (oddIntersection(lhs, rhs) != (paired && i == j))
          return op.emitOpError("css checks violate ")
                 << equation << " at " << i << "," << j;
    return success();
  };
  for (auto [left, right, equation] :
       {std::tuple<ArrayRef<llvm::SmallBitVector>,
                   ArrayRef<llvm::SmallBitVector>, StringRef>{*hxRows, *hzRows,
                                                              "Hx.Hz^T = 0"},
        {*lxRows, *hzRows, "Lx.Hz^T = 0"},
        {*lzRows, *hxRows, "Lz.Hx^T = 0"},
        {*gxRows, *hzRows, "Gx.Hz^T = 0"},
        {*gzRows, *hxRows, "Gz.Hx^T = 0"},
        {*lxRows, *gzRows, "Lx.Gz^T = 0"},
        {*lzRows, *gxRows, "Lz.Gx^T = 0"}}) {
    if (failed(checkFamilies(left, right, equation, /*paired=*/false)))
      return failure();
  }
  // Pairings are only checkable when both partner families are declared;
  // verifyLogicalOps already pinned lx/lz to k rows each.
  if (lx && lz &&
      failed(checkFamilies(*lxRows, *lzRows, "Lx.Lz^T = I", /*paired=*/true)))
    return failure();
  if (gx && gz) {
    if (gxRows->size() != gzRows->size())
      return op.emitOpError(
                 "gx and gz must declare the same number of gauge pairs; "
                 "got ")
             << gxRows->size() << " and " << gzRows->size();
    if (failed(checkFamilies(*gxRows, *gzRows, "Gx.Gz^T = I",
                             /*paired=*/true)))
      return failure();
  }

  // Row counts and ranks must stay consistent with the declared n/k/r.
  if (auto declaredR = op.getR()) {
    if (gx && static_cast<int64_t>(gxRows->size()) != *declaredR)
      return op.emitOpError("gx must declare exactly r = ")
             << *declaredR << " gauge rows, got " << gxRows->size();
    if (gz && static_cast<int64_t>(gzRows->size()) != *declaredR)
      return op.emitOpError("gz must declare exactly r = ")
             << *declaredR << " gauge rows, got " << gzRows->size();
  }
  if (auto declaredN = op.getN()) {
    int64_t n = *declaredN;
    int64_t r = op.getR().value_or(0);
    if (n < 1 || r < 0 || k + r > n)
      return op.emitOpError("requires n>=1, r>=0, and k+r<=n");
    int64_t stabilizerRank = gf2RowRank(*hxRows) + gf2RowRank(*hzRows);
    if (stabilizerRank != n - k - r)
      return op.emitOpError("css checks have GF(2) stabilizer rank ")
             << stabilizerRank << " but n/k/r imply " << n - k - r
             << " independent stabilizers";
  }
  return success();
}

LogicalResult CodeOp::verify() {
  if ((*this)->hasAttr("noise"))
    return emitOpError("noise was removed from fabric.code");
  // Extract k (default 1 when absent).
  int64_t k = getK().value_or(1);
  if (k < 0)
    return emitOpError("k must be >= 0, got ") << k;

  // Read data partition size for bounds-checking logical operator supports.
  auto partitions = getPartitions();
  int64_t dataSize = -1;
  if (auto dataAttr = partitions.get("data")) {
    if (auto dataInt = dyn_cast<IntegerAttr>(dataAttr))
      dataSize = dataInt.getInt();
  }

  auto verifyLogicalOps = [&](StringRef name, ArrayAttr arr) -> LogicalResult {
    if ((int64_t)arr.size() != k)
      return emitOpError(name) << " must have exactly k = " << k
                               << " entries, got " << arr.size();
    for (auto [i, entry] : llvm::enumerate(arr)) {
      auto dense = dyn_cast<DenseI64ArrayAttr>(entry);
      if (!dense)
        return emitOpError(name)
               << "[" << i << "] must be a dense i64 array (array<i64: ...>)";
      auto indices = dense.asArrayRef();
      if (indices.empty())
        return emitOpError(name) << "[" << i << "] must be non-empty";
      for (auto [j, qIdx] : llvm::enumerate(indices)) {
        if (qIdx < 0)
          return emitOpError(name)
                 << "[" << i << "][" << j
                 << "] qubit index must be non-negative, got " << qIdx;
        if (dataSize >= 0 && qIdx >= dataSize)
          return emitOpError(name)
                 << "[" << i << "][" << j << "] qubit index " << qIdx
                 << " out of range for data partition size " << dataSize;
      }
    }
    return success();
  };

  if (auto lx = getLx()) {
    if (failed(verifyLogicalOps("lx", *lx)))
      return failure();
  }
  if (auto lz = getLz()) {
    if (failed(verifyLogicalOps("lz", *lz)))
      return failure();
  }

  auto stabilizerBasis = getStabilizerBasis();
  auto logicalXBasis = getLogicalXBasis();
  auto logicalZBasis = getLogicalZBasis();
  auto gaugeXBasis = getGaugeXBasis();
  auto gaugeZBasis = getGaugeZBasis();
  auto antiStabilizers = getAntiStabilizers();
  auto encodingClifford = getEncodingClifford();
  bool hasCanonical = stabilizerBasis || logicalXBasis || logicalZBasis ||
                      gaugeXBasis || gaugeZBasis || antiStabilizers ||
                      encodingClifford;
  if (!hasCanonical) {
    // Legacy/imported declarations carry the CSS support-array form only.
    // Normalization happens later, but the algebra must already hold: an
    // import is never a licence to accept an invalid code.
    return verifyCssSupportAlgebra(*this, k, dataSize);
  }
  if (!stabilizerBasis || !logicalXBasis || !logicalZBasis || !gaugeXBasis ||
      !gaugeZBasis || !antiStabilizers || !encodingClifford)
    return emitOpError(
        "canonical code algebra requires stabilizer, logical, gauge, "
        "anti-stabilizer, and encoding-Clifford matrices together");

  int64_t n = getN().value_or(dataSize);
  int64_t r = getR().value_or(0);
  if (n < 1 || r < 0 || k + r > n)
    return emitOpError("requires n>=1, r>=0, and k+r<=n");
  int64_t s = n - k - r;
  auto require = [&](Attribute value, StringRef label,
                     int64_t rows) -> FailureOr<DenseIntElementsAttr> {
    auto matrix = requireGF2Matrix(*this, value, label);
    if (failed(matrix))
      return failure();
    auto shape = matrix->getType().getShape();
    if (shape[0] != rows || shape[1] != 2 * n) {
      emitOpError() << label << " must have shape " << rows << "x" << 2 * n;
      return failure();
    }
    return matrix;
  };
  auto stabilizers = require(*stabilizerBasis, "stabilizer_basis", s);
  auto logicalX = require(*logicalXBasis, "logical_x_basis", k);
  auto logicalZ = require(*logicalZBasis, "logical_z_basis", k);
  auto gaugeX = require(*gaugeXBasis, "gauge_x_basis", r);
  auto gaugeZ = require(*gaugeZBasis, "gauge_z_basis", r);
  auto anti = require(*antiStabilizers, "anti_stabilizers", s);
  auto encoding = require(*encodingClifford, "encoding_clifford", 2 * n);
  if (failed(stabilizers) || failed(logicalX) || failed(logicalZ) ||
      failed(gaugeX) || failed(gaugeZ) || failed(anti) || failed(encoding))
    return failure();

  auto rowsOf = [](DenseIntElementsAttr matrix) {
    auto shape = matrix.getType().getShape();
    SmallVector<llvm::SmallBitVector> rows;
    auto values = matrix.getValues<APInt>();
    auto iterator = values.begin();
    for (int64_t row = 0; row < shape[0]; ++row) {
      llvm::SmallBitVector bits(shape[1]);
      for (int64_t column = 0; column < shape[1]; ++column, ++iterator)
        if (!(*iterator).isZero())
          bits.set(column);
      rows.push_back(std::move(bits));
    }
    return rows;
  };
  auto symplectic = [n](const llvm::SmallBitVector &left,
                        const llvm::SmallBitVector &right) {
    bool value = false;
    for (int64_t index = 0; index < n; ++index)
      value ^= (left.test(index) && right.test(n + index)) ^
               (left.test(n + index) && right.test(index));
    return value;
  };
  auto sRows = rowsOf(*stabilizers);
  auto lxRows = rowsOf(*logicalX);
  auto lzRows = rowsOf(*logicalZ);
  auto gxRows = rowsOf(*gaugeX);
  auto gzRows = rowsOf(*gaugeZ);
  auto tRows = rowsOf(*anti);
  auto encodingRows = rowsOf(*encoding);
  auto checkPair = [&](ArrayRef<llvm::SmallBitVector> left,
                       ArrayRef<llvm::SmallBitVector> right, StringRef label,
                       bool paired = false) -> LogicalResult {
    for (auto [i, lhs] : llvm::enumerate(left))
      for (auto [j, rhs] : llvm::enumerate(right))
        if (symplectic(lhs, rhs) != (paired && i == j))
          return emitOpError(label)
                 << " violates canonical commutation at " << i << "," << j;
    return success();
  };
  if (gf2RowRank(sRows) != s || gf2RowRank(encodingRows) != 2 * n)
    return emitOpError("canonical code basis does not have the required rank");
  for (auto [left, right, label, paired] :
       {std::tuple<ArrayRef<llvm::SmallBitVector>,
                   ArrayRef<llvm::SmallBitVector>, StringRef, bool>{
            sRows, sRows, "stabilizers", false},
        {sRows, lxRows, "stabilizer/logical-X", false},
        {sRows, lzRows, "stabilizer/logical-Z", false},
        {sRows, gxRows, "stabilizer/gauge-X", false},
        {sRows, gzRows, "stabilizer/gauge-Z", false},
        {lxRows, lxRows, "logical-X", false},
        {lzRows, lzRows, "logical-Z", false},
        {lxRows, lzRows, "logical pairs", true},
        {gxRows, gxRows, "gauge-X", false},
        {gzRows, gzRows, "gauge-Z", false},
        {gxRows, gzRows, "gauge pairs", true},
        {lxRows, gxRows, "logical-X/gauge-X", false},
        {lxRows, gzRows, "logical-X/gauge-Z", false},
        {lzRows, gxRows, "logical-Z/gauge-X", false},
        {lzRows, gzRows, "logical-Z/gauge-Z", false},
        {sRows, tRows, "stabilizer/anti-stabilizer pairs", true},
        {tRows, tRows, "anti-stabilizers", false},
        {tRows, lxRows, "anti-stabilizer/logical-X", false},
        {tRows, lzRows, "anti-stabilizer/logical-Z", false},
        {tRows, gxRows, "anti-stabilizer/gauge-X", false},
        {tRows, gzRows, "anti-stabilizer/gauge-Z", false}}) {
    if (failed(checkPair(left, right, label, paired)))
      return failure();
  }
  SmallVector<llvm::SmallBitVector> expectedEncoding;
  llvm::append_range(expectedEncoding, tRows);
  llvm::append_range(expectedEncoding, lxRows);
  llvm::append_range(expectedEncoding, gxRows);
  llvm::append_range(expectedEncoding, sRows);
  llvm::append_range(expectedEncoding, lzRows);
  llvm::append_range(expectedEncoding, gzRows);
  if (expectedEncoding != encodingRows)
    return emitOpError("encoding_clifford must use canonical rows "
                       "[T,LX,GX,S,LZ,GZ]");
  return success();
}

LogicalResult CodeProfileOp::verify() {
  auto *target = SymbolTable::lookupNearestSymbolFrom(*this, getCodeAttr());
  if (!target)
    return success(); // A partial linked module verifies after linking.
  auto code = dyn_cast<CodeOp>(target);
  if (!code)
    return emitOpError("code reference must resolve to fabric.code");
  int64_t carriers = code.getN().value_or(0);
  auto asMatrix = [&](Attribute value,
                      StringRef label) -> FailureOr<DenseIntElementsAttr> {
    auto matrix = dyn_cast_or_null<DenseIntElementsAttr>(value);
    if (!matrix || matrix.getType().getRank() != 2 ||
        !matrix.getType().getElementType().isInteger(1)) {
      emitOpError() << label << " must be a rank-2 i1 tensor";
      return failure();
    }
    return matrix;
  };
  auto denseRows = [&](DenseIntElementsAttr matrix) {
    auto shape = matrix.getType().getShape();
    SmallVector<llvm::SmallBitVector> rows;
    auto values = matrix.getValues<APInt>();
    auto iterator = values.begin();
    for (int64_t row = 0; row < shape[0]; ++row) {
      llvm::SmallBitVector bits(shape[1]);
      for (int64_t column = 0; column < shape[1]; ++column, ++iterator)
        if (!(*iterator).isZero())
          bits.set(column);
      rows.push_back(std::move(bits));
    }
    return rows;
  };
  auto rankRows = [](ArrayRef<llvm::SmallBitVector> source) {
    SmallVector<llvm::SmallBitVector> rows(source.begin(), source.end());
    int64_t result = 0;
    int64_t width = rows.empty() ? 0 : rows.front().size();
    for (int64_t column = width - 1; column >= 0; --column) {
      auto pivot =
          llvm::find_if(llvm::drop_begin(rows, result),
                        [&](const auto &row) { return row.test(column); });
      if (pivot == rows.end())
        continue;
      std::iter_swap(rows.begin() + result, pivot);
      for (int64_t index = 0; index < static_cast<int64_t>(rows.size());
           ++index)
        if (index != result && rows[index].test(column))
          rows[index] ^= rows[result];
      ++result;
      if (result == static_cast<int64_t>(rows.size()))
        break;
    }
    return result;
  };
  auto inSpan = [&](const llvm::SmallBitVector &row,
                    ArrayRef<llvm::SmallBitVector> basis) {
    SmallVector<llvm::SmallBitVector> extended(basis.begin(), basis.end());
    int64_t basisRank = rankRows(extended);
    extended.push_back(row);
    return rankRows(extended) == basisRank;
  };
  auto combineSelected = [](ArrayRef<llvm::SmallBitVector> source,
                            const llvm::SmallBitVector &coefficients,
                            int64_t width) {
    llvm::SmallBitVector result(width);
    for (int index : coefficients.set_bits())
      result ^= source[index];
    return result;
  };
  auto effectiveStabilizers = getEffectiveStabilizers();
  auto decompositionValue = getDecomposition();
  auto effectiveMetachecks = getEffectiveMetachecks();
  auto keptFromEffectiveValue = getKeptFromEffective();
  bool hasCanonicalProfile = effectiveStabilizers || decompositionValue ||
                             effectiveMetachecks || keptFromEffectiveValue;
  if (hasCanonicalProfile) {
    if (!effectiveStabilizers || !decompositionValue || !effectiveMetachecks ||
        !keptFromEffectiveValue)
      return emitOpError(
          "canonical code profile requires effective_stabilizers, "
          "decomposition, effective_metachecks, and kept_from_effective "
          "together");
    auto effective = asMatrix(*effectiveStabilizers, "effective_stabilizers");
    auto decomposition = asMatrix(*decompositionValue, "decomposition");
    auto relations = asMatrix(*effectiveMetachecks, "effective_metachecks");
    auto codeStabilizerBasis = code.getStabilizerBasis();
    if (!codeStabilizerBasis)
      return emitOpError(
          "canonical code profile requires a canonical code stabilizer basis");
    auto kept = asMatrix(*codeStabilizerBasis, "code stabilizer_basis");
    if (failed(effective) || failed(decomposition) || failed(relations) ||
        failed(kept))
      return failure();
    auto effectiveShape = effective->getType().getShape();
    auto decompositionShape = decomposition->getType().getShape();
    auto relationShape = relations->getType().getShape();
    auto keptShape = kept->getType().getShape();
    if (effectiveShape[1] != 2 * carriers ||
        decompositionShape[0] != effectiveShape[0] ||
        decompositionShape[1] != keptShape[0] ||
        relationShape[1] != effectiveShape[0])
      return emitOpError("canonical code-profile matrix dimensions disagree");
    auto effectiveRows = denseRows(*effective);
    auto decompositionRows = denseRows(*decomposition);
    auto relationRows = denseRows(*relations);
    auto keptRows = denseRows(*kept);
    auto combine = [](ArrayRef<llvm::SmallBitVector> source,
                      const llvm::SmallBitVector &coefficients, int64_t width) {
      llvm::SmallBitVector result(width);
      for (int index : coefficients.set_bits())
        result ^= source[index];
      return result;
    };
    if (rankRows(effectiveRows) != keptShape[0])
      return emitOpError(
          "effective stabilizers must span the complete kept stabilizer basis");
    for (auto [index, coefficients] : llvm::enumerate(decompositionRows))
      if (combine(keptRows, coefficients, 2 * carriers) != effectiveRows[index])
        return emitOpError("decomposition row ")
               << index << " does not reproduce its effective generator";
    if (rankRows(relationRows) != relationShape[0] ||
        relationShape[0] != effectiveShape[0] - keptShape[0])
      return emitOpError("effective metachecks must be an independent complete "
                         "relation basis");
    for (auto [index, relation] : llvm::enumerate(relationRows)) {
      if (combine(effectiveRows, relation, 2 * carriers).any() ||
          combine(decompositionRows, relation, keptShape[0]).any())
        return emitOpError("effective metacheck ")
               << index << " does not cancel in both representations";
    }

    auto keptFromEffective =
        asMatrix(*keptFromEffectiveValue, "kept_from_effective");
    if (failed(keptFromEffective))
      return failure();
    auto keptFromRows = denseRows(*keptFromEffective);
    auto keptFromShape = keptFromEffective->getType().getShape();
    if (keptFromShape[0] != keptShape[0] ||
        keptFromShape[1] != effectiveShape[0])
      return emitOpError("kept_from_effective has wrong shape");
    for (int64_t row = 0; row < keptShape[0]; ++row) {
      auto recovered =
          combine(decompositionRows, keptFromRows[row], keptShape[0]);
      for (int64_t column = 0; column < keptShape[0]; ++column)
        if (recovered.test(column) != (row == column))
          return emitOpError(
              "kept_from_effective is not a left inverse of decomposition");
    }
  }
  auto verifyRelations =
      [&](StringRef family, DenseIntElementsAttr matrix,
          ArrayRef<llvm::SmallBitVector> checks) -> LogicalResult {
    auto shape = matrix.getType().getShape();
    if (shape[1] != static_cast<int64_t>(checks.size()))
      return emitOpError(family)
             << " metacheck width must equal effective check count = "
             << checks.size();
    auto coefficients = denseRows(matrix);
    SmallVector<llvm::SmallBitVector> independent;
    for (auto [rowIndex, row] : llvm::enumerate(coefficients)) {
      if (row.none())
        return emitOpError(family)
               << " metacheck row " << rowIndex << " must be nonzero";
      llvm::SmallBitVector relation(checks.empty() ? 0 : checks.front().size());
      for (int column : row.set_bits())
        relation ^= checks[column];
      if (relation.any())
        return emitOpError(family) << " metacheck row " << rowIndex
                                   << " is not a relation among code checks";
      llvm::SmallBitVector reduced = row;
      for (const auto &basis : independent) {
        int pivot = basis.find_last();
        if (pivot >= 0 && reduced.test(pivot))
          reduced ^= basis;
      }
      if (reduced.none())
        return emitOpError(family)
               << " metacheck row " << rowIndex << " is linearly dependent";
      int pivot = reduced.find_last();
      for (auto &basis : independent)
        if (basis.test(pivot))
          basis ^= reduced;
      independent.push_back(std::move(reduced));
      llvm::sort(independent, [](const auto &left, const auto &right) {
        return left.find_last() > right.find_last();
      });
    }
    return success();
  };
  auto supportRows =
      [&](StringRef family,
          ArrayAttr checks) -> FailureOr<SmallVector<llvm::SmallBitVector>> {
    SmallVector<llvm::SmallBitVector> rows;
    for (Attribute attribute : checks) {
      auto support = dyn_cast<DenseI64ArrayAttr>(attribute);
      if (!support) {
        code.emitOpError(family) << " checks must use dense i64 support arrays";
        return failure();
      }
      llvm::SmallBitVector row(carriers);
      for (int64_t index : support.asArrayRef()) {
        if (index < 0 || index >= carriers) {
          code.emitOpError(family) << " check support exceeds carrier count";
          return failure();
        }
        row.flip(index);
      }
      rows.push_back(std::move(row));
    }
    return rows;
  };

  DenseIntElementsAttr gaugeOperators;
  DenseIntElementsAttr gaugeMap;
  if (auto gauges = getGaugeMeasurements()) {
    auto operators = asMatrix(gauges->get("operators"), "gauge operators");
    auto map =
        asMatrix(gauges->get("stabilizer_map"), "gauge-to-stabilizer map");
    if (failed(operators) || failed(map))
      return failure();
    gaugeOperators = *operators;
    gaugeMap = *map;
    auto operatorShape = gaugeOperators.getType().getShape();
    auto mapShape = gaugeMap.getType().getShape();
    if (operatorShape[1] != 2 * carriers)
      return emitOpError("gauge operators must have symplectic width 2n");
    if (mapShape[1] != operatorShape[0])
      return emitOpError(
          "gauge-to-stabilizer width must equal gauge measurement count");
  }

  if (auto metachecks = getMetachecks()) {
    for (NamedAttribute entry : *metachecks) {
      StringRef family = entry.getName().strref();
      auto matrix = asMatrix(entry.getValue(), family);
      if (failed(matrix))
        return failure();
      if (family == "gauge") {
        if (!gaugeOperators)
          return emitOpError("gauge metachecks require gauge measurements");
        auto rows = denseRows(gaugeOperators);
        if (failed(verifyRelations(family, *matrix, rows)))
          return failure();
        continue;
      }
      ArrayAttr checks;
      if (family == "x")
        checks = code.getHx().value_or(ArrayAttr{});
      else if (family == "z")
        checks = code.getHz().value_or(ArrayAttr{});
      else
        return emitOpError("unknown metacheck family '") << family << "'";
      auto rows = supportRows(family, checks);
      if (failed(rows) || failed(verifyRelations(family, *matrix, *rows)))
        return failure();
    }
  }

  auto dynamicPhases = getDynamicPhases();
  auto recordLogicals = getRecordLogicals();
  int64_t codeK = code.getK().value_or(0);
  int64_t codeR = code.getR().value_or(0);
  SmallVector<std::string> protectedNames;
  llvm::StringSet<> protectedNameSet;
  if (recordLogicals) {
    auto names = recordLogicals->getAs<ArrayAttr>("names");
    if (!names || names.empty())
      return emitOpError(
          "record_logicals requires a nonempty protected-basis name array");
    for (Attribute raw : names) {
      auto name = dyn_cast<StringAttr>(raw);
      if (!name || name.empty())
        return emitOpError("record_logicals names must be nonempty strings");
      if (!protectedNameSet.insert(name.getValue()).second)
        return emitOpError("record_logicals names must be unique");
      protectedNames.push_back(name.getValue().str());
    }
    auto gaugePairIndices =
        recordLogicals->getAs<DenseI64ArrayAttr>("gauge_pair_indices");
    if (codeK > 0) {
      if (protectedNames.size() != static_cast<size_t>(codeK))
        return emitOpError(
            "record logical pairs must equal the linked code's protected "
            "logical-pair count");
      for (int64_t index = 0; index < codeK; ++index)
        if (protectedNames[index] != ("q" + Twine(index)).str())
          return emitOpError(
                     "stabilizer-code record logical names must use the exact "
                     "canonical protected order q0..q{k-1}; entry ")
                 << index << " is '" << protectedNames[index] << "'";
      if (gaugePairIndices && !gaugePairIndices.empty())
        return emitOpError(
            "stabilizer-code record logicals may not select gauge pairs");
    } else {
      if (!gaugePairIndices || gaugePairIndices.size() != protectedNames.size())
        return emitOpError(
            "a k=0 dynamic profile must bind every record logical to a "
            "distinct canonical gauge pair");
      llvm::SmallDenseSet<int64_t, 8> selectedPairs;
      for (int64_t index : gaugePairIndices.asArrayRef()) {
        if (index < 0 || index >= codeR)
          return emitOpError(
              "record logical gauge-pair selector is outside the linked "
              "code's canonical gauge basis");
        if (!selectedPairs.insert(index).second)
          return emitOpError(
              "record logical gauge-pair selectors must be unique");
      }
    }
  } else {
    for (int64_t index = 0; index < codeK; ++index) {
      protectedNames.push_back(("q" + Twine(index)).str());
      protectedNameSet.insert(protectedNames.back());
    }
  }

  int64_t phaseRecordCount = 0;
  llvm::StringSet<> phaseNames;
  SmallVector<StringRef> orderedPhaseNames;
  SmallVector<StringRef> orderedPhaseInputs;
  SmallVector<StringRef> orderedPhaseOutputs;
  SmallVector<llvm::SmallBitVector> allPhaseMeasurements;
  SmallVector<SmallVector<llvm::SmallBitVector>> phaseMeasurements;
  SmallVector<SmallVector<llvm::SmallBitVector>> phaseIsgs;
  SmallVector<DenseIntElementsAttr> phaseLogicalActions;
  SmallVector<llvm::SmallBitVector> codeStabilizers;
  SmallVector<llvm::SmallBitVector> codeGaugeGroup;
  auto periodClosure = getPeriodClosureAttr();
  if (dynamicPhases) {
    if (Attribute value = code.getStabilizerBasis().value_or(Attribute{})) {
      auto stabilizerBasis = asMatrix(value, "code stabilizer_basis");
      if (failed(stabilizerBasis))
        return failure();
      codeStabilizers = denseRows(*stabilizerBasis);
    } else {
      auto appendCssRows =
          [&](std::optional<ArrayAttr> supports, int64_t offset,
              StringRef label,
              SmallVectorImpl<llvm::SmallBitVector> &rows) -> LogicalResult {
        if (!supports)
          return success();
        for (Attribute raw : *supports) {
          auto support = dyn_cast<DenseI64ArrayAttr>(raw);
          if (!support)
            return emitOpError(label) << " must use dense i64 support arrays";
          llvm::SmallBitVector row(2 * carriers);
          for (int64_t index : support.asArrayRef()) {
            if (index < 0 || index >= carriers)
              return emitOpError(label)
                     << " support exceeds the linked code carrier count";
            row.flip(offset + index);
          }
          rows.push_back(std::move(row));
        }
        return success();
      };
      if (failed(appendCssRows(code.getHx(), 0, "code hx", codeStabilizers)) ||
          failed(appendCssRows(code.getHz(), carriers, "code hz",
                               codeStabilizers)))
        return failure();
      if (codeStabilizers.empty())
        return emitOpError(
            "dynamic profiles require a canonical stabilizer_basis or CSS "
            "hx/hz declarations on the linked code");
      SmallVector<std::pair<int, llvm::SmallBitVector>> pivots;
      SmallVector<llvm::SmallBitVector> independent;
      for (const auto &candidate : codeStabilizers) {
        llvm::SmallBitVector reduced = candidate;
        while (reduced.any()) {
          int pivot = reduced.find_last();
          auto existing = llvm::find_if(
              pivots, [&](const auto &entry) { return entry.first == pivot; });
          if (existing == pivots.end()) {
            pivots.emplace_back(pivot, reduced);
            independent.push_back(candidate);
            break;
          }
          reduced ^= existing->second;
        }
      }
      codeStabilizers = std::move(independent);
    }
    llvm::append_range(codeGaugeGroup, codeStabilizers);
    auto symplectic = [carriers](const llvm::SmallBitVector &left,
                                 const llvm::SmallBitVector &right) {
      bool value = false;
      for (int64_t index = 0; index < carriers; ++index)
        value ^= (left.test(index) && right.test(carriers + index)) ^
                 (left.test(carriers + index) && right.test(index));
      return value;
    };
    bool hasCanonicalGaugeBasis = false;
    for (auto [value, label] : {std::pair<Attribute, StringRef>{
                                    code.getGaugeXBasis().value_or(Attribute{}),
                                    "code gauge_x_basis"},
                                {code.getGaugeZBasis().value_or(Attribute{}),
                                 "code gauge_z_basis"}}) {
      if (!value)
        continue;
      auto basis = asMatrix(value, label);
      if (failed(basis))
        return failure();
      llvm::append_range(codeGaugeGroup, denseRows(*basis));
      hasCanonicalGaugeBasis = true;
    }
    if (!hasCanonicalGaugeBasis) {
      auto appendGaugeCssRows = [&](std::optional<ArrayAttr> supports,
                                    int64_t offset,
                                    StringRef label) -> LogicalResult {
        if (!supports)
          return success();
        for (Attribute raw : *supports) {
          auto support = dyn_cast<DenseI64ArrayAttr>(raw);
          if (!support)
            return emitOpError(label) << " must use dense i64 support arrays";
          llvm::SmallBitVector row(2 * carriers);
          for (int64_t index : support.asArrayRef()) {
            if (index < 0 || index >= carriers)
              return emitOpError(label)
                     << " support exceeds the linked code carrier count";
            row.flip(offset + index);
          }
          codeGaugeGroup.push_back(std::move(row));
        }
        return success();
      };
      if (failed(appendGaugeCssRows(code.getGx(), 0, "code gx")) ||
          failed(appendGaugeCssRows(code.getGz(), carriers, "code gz")))
        return failure();
    }

    auto loadPairedFamily =
        [&](Attribute canonical, std::optional<ArrayAttr> css, int64_t offset,
            StringRef label) -> FailureOr<SmallVector<llvm::SmallBitVector>> {
      if (canonical) {
        auto matrix = asMatrix(canonical, label);
        if (failed(matrix))
          return failure();
        if (matrix->getType().getShape()[1] != 2 * carriers) {
          emitOpError() << label << " must have symplectic width 2n";
          return failure();
        }
        return denseRows(*matrix);
      }
      if (!css) {
        emitOpError() << "dynamic profile requires linked " << label;
        return failure();
      }
      SmallVector<llvm::SmallBitVector> rows;
      for (Attribute raw : *css) {
        auto support = dyn_cast<DenseI64ArrayAttr>(raw);
        if (!support) {
          emitOpError() << label << " must use dense i64 support arrays";
          return failure();
        }
        llvm::SmallBitVector row(2 * carriers);
        for (int64_t index : support.asArrayRef()) {
          if (index < 0 || index >= carriers) {
            emitOpError() << label << " support exceeds carrier count";
            return failure();
          }
          row.flip(offset + index);
        }
        rows.push_back(std::move(row));
      }
      return rows;
    };

    SmallVector<llvm::SmallBitVector> protectedRows;
    if (codeK > 0) {
      auto logicalX =
          loadPairedFamily(code.getLogicalXBasis().value_or(Attribute{}),
                           code.getLx(), 0, "code logical_x_basis");
      auto logicalZ =
          loadPairedFamily(code.getLogicalZBasis().value_or(Attribute{}),
                           code.getLz(), carriers, "code logical_z_basis");
      if (failed(logicalX) || failed(logicalZ))
        return failure();
      llvm::append_range(protectedRows, *logicalX);
      llvm::append_range(protectedRows, *logicalZ);
    } else {
      auto gaugeX =
          loadPairedFamily(code.getGaugeXBasis().value_or(Attribute{}),
                           code.getGx(), 0, "code gauge_x_basis");
      auto gaugeZ =
          loadPairedFamily(code.getGaugeZBasis().value_or(Attribute{}),
                           code.getGz(), carriers, "code gauge_z_basis");
      if (failed(gaugeX) || failed(gaugeZ))
        return failure();
      auto selected =
          recordLogicals
              ? recordLogicals->getAs<DenseI64ArrayAttr>("gauge_pair_indices")
              : DenseI64ArrayAttr{};
      if (!selected)
        return emitOpError(
            "k=0 dynamic profile requires bound record-logical gauge pairs");
      for (int64_t index : selected.asArrayRef())
        protectedRows.push_back((*gaugeX)[index]);
      for (int64_t index : selected.asArrayRef())
        protectedRows.push_back((*gaugeZ)[index]);
    }

    int64_t protectedCount = static_cast<int64_t>(protectedNames.size());
    if (static_cast<int64_t>(protectedRows.size()) != 2 * protectedCount)
      return emitOpError("protected periodic representatives do not match the "
                         "bound paired basis");
    for (int64_t left = 0; left < 2 * protectedCount; ++left)
      for (int64_t right = 0; right < 2 * protectedCount; ++right) {
        bool expected =
            (left < protectedCount && right == left + protectedCount) ||
            (right < protectedCount && left == right + protectedCount);
        if (symplectic(protectedRows[left], protectedRows[right]) != expected)
          return emitOpError("protected periodic representatives must form a "
                             "canonical paired symplectic basis");
      }

    auto phases = dynamicPhases;
    if (phases->empty())
      return emitOpError("dynamic_phases must be nonempty when present");
    if (!periodClosure)
      return emitOpError("periodic dynamic profile requires period_closure");
    if (failed(verifySymplecticClosure(getOperation(), periodClosure,
                                       "period_closure")))
      return failure();
    int64_t expectedClosureWidth = 2 * protectedNames.size();
    if (cast<DenseIntElementsAttr>(periodClosure).getType().getShape()[0] !=
        expectedClosureWidth)
      return emitOpError(
                 "period_closure dimension must equal the protected periodic "
                 "basis width = ")
             << expectedClosureWidth;
    for (Attribute value : *phases) {
      auto phase = dyn_cast<DictionaryAttr>(value);
      if (!phase)
        return emitOpError("dynamic phase must be a dictionary");
      auto name = dyn_cast_or_null<StringAttr>(phase.get("name"));
      auto inputEpoch = dyn_cast_or_null<StringAttr>(phase.get("input_epoch"));
      auto outputEpoch =
          dyn_cast_or_null<StringAttr>(phase.get("output_epoch"));
      auto measured =
          asMatrix(phase.get("measured_gauges"), "phase measured_gauges");
      auto isg = asMatrix(phase.get("instantaneous_stabilizers"),
                          "phase instantaneous_stabilizers");
      if (!name || name.getValue().empty() || failed(measured) || failed(isg))
        return emitOpError("dynamic phase is incomplete");
      if (!phaseNames.insert(name.getValue()).second)
        return emitOpError("dynamic phase names must be unique");
      if (!inputEpoch || inputEpoch.getValue().empty() || !outputEpoch ||
          outputEpoch.getValue().empty())
        return emitOpError(
            "dynamic phase must declare input_epoch and output_epoch");
      orderedPhaseNames.push_back(name.getValue());
      orderedPhaseInputs.push_back(inputEpoch.getValue());
      orderedPhaseOutputs.push_back(outputEpoch.getValue());
      auto measuredShape = measured->getType().getShape();
      auto isgShape = isg->getType().getShape();
      if (measuredShape[1] != 2 * carriers || isgShape[1] != 2 * carriers)
        return emitOpError("dynamic phase operators must have width 2n");
      auto measuredRows = denseRows(*measured);
      auto isgRows = denseRows(*isg);
      for (const auto &row : measuredRows)
        if (row.none() || !inSpan(row, codeGaugeGroup))
          return emitOpError(
              "phase measurement must be a nonidentity member of the linked "
              "code gauge group");
      for (const auto &row : isgRows)
        if (row.none() || !inSpan(row, codeGaugeGroup))
          return emitOpError(
              "phase instantaneous stabilizer lies outside the linked code "
              "gauge group");
      for (auto [leftIndex, left] : llvm::enumerate(isgRows))
        for (const auto &right : llvm::drop_begin(isgRows, leftIndex + 1))
          if (symplectic(left, right))
            return emitOpError(
                "phase instantaneous stabilizers must mutually commute");
      llvm::append_range(allPhaseMeasurements, measuredRows);
      phaseMeasurements.push_back(measuredRows);
      phaseIsgs.push_back(isgRows);

      auto logicalMap = phase.getAs<DictionaryAttr>("logical_map");
      if (!logicalMap || logicalMap.size() != protectedNames.size())
        return emitOpError(
            "phase logical_map must be total over the protected periodic "
            "basis");
      llvm::StringSet<> mappedNames;
      for (NamedAttribute entry : logicalMap) {
        auto mapped = dyn_cast<StringAttr>(entry.getValue());
        if (!protectedNameSet.contains(entry.getName().getValue()) || !mapped ||
            !protectedNameSet.contains(mapped.getValue()) ||
            !mappedNames.insert(mapped.getValue()).second)
          return emitOpError(
              "phase logical_map must be a bijection over the protected "
              "periodic basis");
      }
      auto logicalAction =
          asMatrix(phase.get("logical_action"), "phase logical_action");
      if (failed(logicalAction))
        return failure();
      auto actionShape = logicalAction->getType().getShape();
      if (actionShape[0] != 2 * protectedCount ||
          actionShape[1] != 2 * protectedCount)
        return emitOpError("phase logical_action dimension must equal the "
                           "protected periodic basis width");
      if (failed(verifySymplecticClosure(getOperation(), *logicalAction,
                                         "phase logical_action")))
        return failure();
      phaseLogicalActions.push_back(*logicalAction);
      phaseRecordCount += measuredShape[0];
      if (Attribute recoveryValue = phase.get("temporal_recovery")) {
        auto recovery = asMatrix(recoveryValue, "phase temporal recovery");
        if (failed(recovery))
          return failure();
        auto shape = recovery->getType().getShape();
        if (shape[1] != measuredShape[0] || shape[0] != isgShape[0])
          return emitOpError(
              "phase recovery must map measured gauges to the ISG");
        auto recoveryRows = denseRows(*recovery);
        for (auto [rowIndex, coefficients] : llvm::enumerate(recoveryRows)) {
          llvm::SmallBitVector recovered(2 * carriers);
          for (int column : coefficients.set_bits())
            recovered ^= measuredRows[column];
          if (recovered != isgRows[rowIndex])
            return emitOpError("phase temporal recovery row ")
                   << rowIndex << " does not reproduce its ISG row";
        }
      }
    }
    for (size_t index = 0; index < orderedPhaseNames.size(); ++index) {
      if (orderedPhaseInputs[index] != orderedPhaseNames[index])
        return emitOpError("dynamic phase name must equal its input_epoch");
      StringRef expectedOutput =
          orderedPhaseNames[(index + 1) % orderedPhaseNames.size()];
      if (orderedPhaseOutputs[index] != expectedOutput)
        return emitOpError(
            "dynamic phases must form one ordered periodic epoch cycle");
    }
    if (recordLogicals) {
      auto x = asMatrix(recordLogicals->get("x"), "record_logicals.x");
      auto z = asMatrix(recordLogicals->get("z"), "record_logicals.z");
      if (failed(x) || failed(z))
        return failure();
      auto xShape = x->getType().getShape();
      auto zShape = z->getType().getShape();
      if (xShape[0] != static_cast<int64_t>(protectedNames.size()) ||
          zShape[0] != static_cast<int64_t>(protectedNames.size()) ||
          xShape[1] != phaseRecordCount || zShape[1] != phaseRecordCount)
        return emitOpError(
            "record_logicals X/Z maps must align the paired protected basis "
            "with every phase gauge record");
    }

    auto makeBoundaryBasis = [&](ArrayRef<llvm::SmallBitVector> phaseIsg) {
      SmallVector<llvm::SmallBitVector> boundary;
      for (const auto &candidate : llvm::concat<const llvm::SmallBitVector>(
               codeStabilizers, phaseIsg)) {
        SmallVector<llvm::SmallBitVector> extended(boundary.begin(),
                                                   boundary.end());
        int64_t before = rankRows(extended);
        extended.push_back(candidate);
        if (rankRows(extended) != before)
          boundary.push_back(candidate);
      }
      return boundary;
    };
    auto verifyProtectedBoundary =
        [&](ArrayRef<llvm::SmallBitVector> representatives,
            ArrayRef<llvm::SmallBitVector> boundary,
            StringRef label) -> LogicalResult {
      for (int64_t left = 0; left < 2 * protectedCount; ++left)
        for (int64_t right = 0; right < 2 * protectedCount; ++right) {
          bool expected =
              (left < protectedCount && right == left + protectedCount) ||
              (right < protectedCount && left == right + protectedCount);
          if (symplectic(representatives[left], representatives[right]) !=
              expected)
            return emitOpError(label)
                   << " protected representatives do not form a canonical "
                      "paired symplectic basis";
        }
      for (const auto &logical : representatives)
        for (const auto &stabilizer : boundary)
          if (symplectic(logical, stabilizer))
            return emitOpError(label)
                   << " protected representatives do not commute with the "
                      "kept/next-phase ISG basis";
      SmallVector<llvm::SmallBitVector> quotient(representatives.begin(),
                                                 representatives.end());
      llvm::append_range(quotient, boundary);
      if (rankRows(quotient) != static_cast<int64_t>(quotient.size()))
        return emitOpError(label)
               << " protected representatives are not independent modulo "
                  "the kept/next-phase ISG basis";
      return success();
    };

    SmallVector<llvm::SmallBitVector> initialIsg =
        makeBoundaryBasis(phaseIsgs.front());
    if (failed(verifyProtectedBoundary(protectedRows, initialIsg,
                                       "initial dynamic boundary")))
      return failure();

    SmallVector<llvm::SmallBitVector> quotientBasis(protectedRows.begin(),
                                                    protectedRows.end());
    llvm::append_range(quotientBasis, initialIsg);
    if (rankRows(quotientBasis) != static_cast<int64_t>(quotientBasis.size()))
      return emitOpError("protected periodic representatives must be "
                         "independent modulo the initial kept/ISG basis");

    SmallVector<llvm::SmallBitVector> recordUpdates;
    if (recordLogicals) {
      auto x = cast<DenseIntElementsAttr>(recordLogicals->get("x"));
      auto z = cast<DenseIntElementsAttr>(recordLogicals->get("z"));
      llvm::append_range(recordUpdates, denseRows(x));
      llvm::append_range(recordUpdates, denseRows(z));
    } else {
      for (int64_t index = 0; index < 2 * protectedCount; ++index)
        recordUpdates.emplace_back(phaseRecordCount);
    }

    SmallVector<llvm::SmallBitVector> transportedRows(protectedRows.begin(),
                                                      protectedRows.end());
    int64_t recordOffset = 0;
    for (auto [phaseIndex, measurements] : llvm::enumerate(phaseMeasurements)) {
      for (const auto &logical : transportedRows)
        for (const auto &measurement : measurements)
          if (symplectic(logical, measurement))
            return emitOpError("dynamic boundary before phase '")
                   << orderedPhaseNames[phaseIndex]
                   << "' protected representatives do not commute with the "
                      "active measured_gauges";
      for (auto [transported, updates] :
           llvm::zip(transportedRows, recordUpdates))
        for (int64_t local = 0;
             local < static_cast<int64_t>(measurements.size()); ++local)
          if (updates.test(recordOffset + local))
            transported ^= measurements[local];
      recordOffset += static_cast<int64_t>(measurements.size());

      auto nextBoundary =
          makeBoundaryBasis(phaseIsgs[(phaseIndex + 1) % phaseIsgs.size()]);
      std::string boundaryLabel = (Twine("dynamic boundary after phase '") +
                                   orderedPhaseNames[phaseIndex] + "'")
                                      .str();
      if (failed(verifyProtectedBoundary(transportedRows, nextBoundary,
                                         boundaryLabel)))
        return failure();
    }

    auto coordinatesInBasis = [&](const llvm::SmallBitVector &targetRow)
        -> FailureOr<llvm::SmallBitVector> {
      int64_t width = 2 * carriers;
      SmallVector<int64_t> pivotOwner(width, -1);
      SmallVector<llvm::SmallBitVector> pivotRows;
      SmallVector<llvm::SmallBitVector> pivotCoordinates;
      for (auto [index, basisRow] : llvm::enumerate(quotientBasis)) {
        llvm::SmallBitVector reduced = basisRow;
        llvm::SmallBitVector coordinates(quotientBasis.size());
        coordinates.set(index);
        while (reduced.any()) {
          int pivot = reduced.find_last();
          int64_t owner = pivotOwner[pivot];
          if (owner < 0) {
            pivotOwner[pivot] = static_cast<int64_t>(pivotRows.size());
            pivotRows.push_back(std::move(reduced));
            pivotCoordinates.push_back(std::move(coordinates));
            break;
          }
          reduced ^= pivotRows[owner];
          coordinates ^= pivotCoordinates[owner];
        }
      }
      llvm::SmallBitVector reduced = targetRow;
      llvm::SmallBitVector coordinates(quotientBasis.size());
      while (reduced.any()) {
        int pivot = reduced.find_last();
        int64_t owner = pivotOwner[pivot];
        if (owner < 0)
          return failure();
        reduced ^= pivotRows[owner];
        coordinates ^= pivotCoordinates[owner];
      }
      return coordinates;
    };

    SmallVector<llvm::SmallBitVector> derivedClosure;
    for (const auto &transported : transportedRows) {
      auto coordinates = coordinatesInBasis(transported);
      if (failed(coordinates))
        return emitOpError("record-defined logical transport does not close in "
                           "the initial protected/kept/ISG quotient");
      llvm::SmallBitVector row(2 * protectedCount);
      for (int64_t column = 0; column < 2 * protectedCount; ++column)
        row[column] = (*coordinates)[column];
      derivedClosure.push_back(std::move(row));
    }
    if (rankRows(derivedClosure) != 2 * protectedCount)
      return emitOpError("derived period_closure must be full rank");
    auto closureSymplectic =
        [protectedCount](const llvm::SmallBitVector &left,
                         const llvm::SmallBitVector &right) {
          bool value = false;
          for (int64_t index = 0; index < protectedCount; ++index)
            value ^= (left.test(index) && right.test(protectedCount + index)) ^
                     (left.test(protectedCount + index) && right.test(index));
          return value;
        };
    for (int64_t left = 0; left < 2 * protectedCount; ++left)
      for (int64_t right = 0; right < 2 * protectedCount; ++right) {
        bool expected =
            (left < protectedCount && right == left + protectedCount) ||
            (right < protectedCount && left == right + protectedCount);
        if (closureSymplectic(derivedClosure[left], derivedClosure[right]) !=
            expected)
          return emitOpError("derived period_closure is not symplectic");
      }
    SmallVector<llvm::SmallBitVector> actionProduct;
    for (int64_t row = 0; row < 2 * protectedCount; ++row) {
      llvm::SmallBitVector identity(2 * protectedCount);
      identity.set(row);
      actionProduct.push_back(std::move(identity));
    }
    for (DenseIntElementsAttr action : phaseLogicalActions) {
      SmallVector<llvm::SmallBitVector> composed;
      auto actionRows = denseRows(action);
      for (const auto &coefficients : actionProduct)
        composed.push_back(
            combineSelected(actionRows, coefficients, 2 * protectedCount));
      actionProduct = std::move(composed);
    }
    if (actionProduct != derivedClosure)
      return emitOpError("phase logical actions contradict the exact physical "
                         "record/ISG transport derivation");
    if (denseRows(cast<DenseIntElementsAttr>(periodClosure)) != derivedClosure)
      return emitOpError("period_closure contradicts the exact physical "
                         "record/ISG transport derivation");
  } else {
    if (periodClosure)
      return emitOpError("period_closure requires a periodic dynamic profile");
    if (recordLogicals)
      return emitOpError("record_logicals require a periodic dynamic profile");
  }
  if (auto value = getTemporalRecovery()) {
    auto recovery = asMatrix(*value, "temporal recovery");
    if (failed(recovery))
      return failure();
    if (recovery->getType().getShape()[1] != phaseRecordCount)
      return emitOpError("temporal recovery must span every phase record");
    Attribute targetValue = (*this)->getAttr("temporal_recovery_targets");
    auto targets = asMatrix(targetValue, "temporal recovery targets");
    if (failed(targets))
      return failure();
    auto recoveryShape = recovery->getType().getShape();
    auto targetShape = targets->getType().getShape();
    if (targetShape[0] != recoveryShape[0] ||
        targetShape[1] != static_cast<int64_t>(codeStabilizers.size()))
      return emitOpError(
          "temporal recovery targets must map every recovery row into the "
          "linked code's kept stabilizer basis");
    auto targetRows = denseRows(*targets);
    for (auto [rowIndex, coefficients] :
         llvm::enumerate(denseRows(*recovery))) {
      auto recovered =
          combineSelected(allPhaseMeasurements, coefficients, 2 * carriers);
      if (recovered.none() || !inSpan(recovered, codeStabilizers))
        return emitOpError("temporal recovery row ")
               << rowIndex << " must recover one nonidentity kept stabilizer";
      auto declaredTarget =
          combineSelected(codeStabilizers, targetRows[rowIndex], 2 * carriers);
      if (declaredTarget != recovered)
        return emitOpError("temporal recovery target row ")
               << rowIndex << " does not equal the recovered kept stabilizer";
    }
  } else if ((*this)->hasAttr("temporal_recovery_targets")) {
    return emitOpError("temporal_recovery_targets require temporal_recovery");
  }

  return success();
}

LogicalResult TransportOp::verify() {
  auto inType = cast<ResourceStateType>(getResource().getType());
  auto outType = cast<ResourceStateType>(getResult().getType());
  if (inType != outType)
    return emitOpError("input and output resource types must match");
  if (getSrcRegion() == getDstRegion())
    return emitOpError("source and destination regions must differ");
  if (auto route = getRouteAttr()) {
    auto *target = SymbolTable::lookupNearestSymbolFrom(*this, route);
    auto interconnect = dyn_cast_or_null<InterconnectOp>(target);
    if (!interconnect)
      return emitOpError(
          "route must resolve to the selected fabric.interconnect");
    if (getSrcRegionAttr() != interconnect.getRegionAAttr() ||
        getDstRegionAttr() != interconnect.getRegionBAttr())
      return emitOpError(
          "transport endpoints must exactly match the selected route ports");
    if (!interconnect.getProtocolAttr() ||
        getProtocolAttr() != interconnect.getProtocolAttr())
      return emitOpError(
          "transport protocol must exactly match the selected route protocol");
  }
  return success();
}

LogicalResult SuccessOp::verify() {
  if (!(*this)->getParentOfType<GadgetProfileOp>())
    return emitOpError("must be nested in a fabric.gadget_profile");
  if ((*this)->hasAttr("expected") || (*this)->hasAttr("scope") ||
      (*this)->hasAttr("label"))
    return emitOpError(
        "success rows contain only affine mismatch parity; fold polarity into "
        "constant");
  bool hasRecords = getRecords() && !getRecords()->empty();
  bool hasInputs = getInputSyndromes() && !getInputSyndromes()->empty();
  if (!hasRecords && !hasInputs && !getConstant())
    return emitOpError("success parity must contain at least one affine term");
  return success();
}

LogicalResult OutputSyndromeOp::verify() {
  if (!(*this)->getParentOfType<GadgetProfileOp>())
    return emitOpError("must be nested in a fabric.gadget_profile");
  if (getPortIndex() < 0 || getIndex() < 0)
    return emitOpError("requires nonnegative port_index and index");
  return success();
}

LogicalResult GadgetProfileOp::verify() {
  if ((*this)->hasAttr("selection"))
    return emitOpError(
        "selection policy belongs to fabric.retry or event.selection, not "
        "the analysis profile");
  auto *target = SymbolTable::lookupNearestSymbolFrom(*this, getGadgetAttr());
  if (!target)
    return success(); // A partial linked module may resolve this at link time.
  auto gadget = dyn_cast<GadgetOp>(target);
  if (!gadget)
    return emitOpError("gadget reference must resolve to fabric.gadget");

  SmallVector<PatchType> physicalInputs;
  SmallVector<PatchType> physicalOutputs;
  for (Type type : gadget.getFunctionType().getInputs())
    if (auto patch = dyn_cast<PatchType>(type))
      physicalInputs.push_back(patch);
  for (Type type : gadget.getFunctionType().getResults())
    if (auto patch = dyn_cast<PatchType>(type))
      physicalOutputs.push_back(patch);

  // A produced patch is represented by an uninitialized physical carrier in
  // the realization signature, but it is not a semantic input boundary.  Use
  // the GadgetSpec port directions, when available, to distinguish that case
  // from input and inout endpoints.
  SmallVector<PatchType> inputEndpoints;
  SmallVector<PatchType> outputEndpoints;
  GadgetSpecOp spec;
  if (auto specRef = gadget.getSpecAttr())
    spec = dyn_cast_or_null<GadgetSpecOp>(
        SymbolTable::lookupNearestSymbolFrom(*this, specRef));
  if (gadget.getSpecAttr() && !spec)
    return emitOpError(
        "linked gadget has an unresolved GadgetSpec, so semantic endpoint "
        "directions are ambiguous");
  std::optional<ArrayAttr> specPorts = spec ? spec.getPorts() : std::nullopt;
  if (specPorts) {
    size_t physicalInputIndex = 0;
    size_t physicalOutputIndex = 0;
    for (Attribute raw : *specPorts) {
      auto port = dyn_cast<DictionaryAttr>(raw);
      auto direction =
          port ? port.getAs<StringAttr>("direction") : StringAttr{};
      if (!direction || physicalInputIndex >= physicalInputs.size())
        return emitOpError(
            "linked GadgetSpec port table does not align with the gadget's "
            "physical patch inputs");
      PatchType carrier = physicalInputs[physicalInputIndex++];
      if (direction.getValue() == "input" || direction.getValue() == "inout")
        inputEndpoints.push_back(carrier);
      if (direction.getValue() == "output" || direction.getValue() == "inout") {
        if (physicalOutputIndex >= physicalOutputs.size())
          return emitOpError(
              "linked GadgetSpec port table does not align with the gadget's "
              "physical patch outputs");
        outputEndpoints.push_back(physicalOutputs[physicalOutputIndex++]);
      }
    }
    if (physicalInputIndex != physicalInputs.size() ||
        physicalOutputIndex != physicalOutputs.size())
      return emitOpError(
          "linked GadgetSpec port table must cover every physical patch "
          "input and output");
  } else {
    inputEndpoints = std::move(physicalInputs);
    outputEndpoints = std::move(physicalOutputs);
  }

  SmallVector<int64_t> inputWidths;
  SmallVector<int64_t> outputWidths;
  auto parseProfiles = [&](std::optional<ArrayAttr> values, StringRef label,
                           ArrayRef<PatchType> endpoints,
                           SmallVectorImpl<int64_t> &widths) -> LogicalResult {
    bool hasQualifiedEndpoint = llvm::any_of(endpoints, [](PatchType endpoint) {
      return static_cast<bool>(endpoint.getEncoding());
    });
    if (!values && hasQualifiedEndpoint)
      return emitOpError(label)
             << " must be present and exactly cover every encoding-qualified "
                "semantic patch endpoint";
    if (!values)
      return success();
    if (values->size() != endpoints.size())
      return emitOpError(label)
             << " entries must exactly cover every typed patch endpoint";
    for (Attribute raw : *values) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto endpoint =
          entry ? entry.getAs<IntegerAttr>("endpoint") : IntegerAttr{};
      auto reference = entry ? entry.getAs<FlatSymbolRefAttr>("profile")
                             : FlatSymbolRefAttr{};
      if (!endpoint ||
          endpoint.getInt() != static_cast<int64_t>(widths.size()) ||
          !reference)
        return emitOpError(label)
               << " entries require contiguous endpoint ordinals from zero "
                  "and a code-profile symbol";
      PatchType endpointType = endpoints[endpoint.getInt()];
      auto profile =
          resolveSelectedCodeProfile(getOperation(), endpointType.getCodeType(),
                                     endpointType.getEncoding(),
                                     endpointType.getEpoch(), reference, label);
      if (failed(profile))
        return failure();
      auto effective = (*profile).getEffectiveStabilizers();
      auto matrix = effective ? dyn_cast<DenseIntElementsAttr>(*effective)
                              : DenseIntElementsAttr{};
      if (!matrix || matrix.getType().getRank() != 2)
        return emitOpError(label)
               << " profile " << reference
               << " has no canonical effective syndrome basis";
      widths.push_back(matrix.getType().getShape()[0]);
    }
    return success();
  };
  if (failed(
          parseProfiles(getInputs(), "inputs", inputEndpoints, inputWidths)) ||
      failed(parseProfiles(getOutputs(), "outputs", outputEndpoints,
                           outputWidths)))
    return failure();

  StringRef gadgetName = gadget.getSymName();
  SmallVector<std::string> demandedPaths;
  getBody().walk([&](Operation *declaration) {
    if (!isa<SuccessOp, OutputSyndromeOp>(declaration))
      return;
    auto records = declaration->getAttrOfType<ArrayAttr>("records");
    if (!records)
      return;
    for (Attribute raw : records) {
      auto record = dyn_cast<StringAttr>(raw);
      if (!record)
        continue;
      StringRef path = record.getValue();
      if (path.consume_front(gadgetName) && path.consume_front(".") &&
          !path.empty())
        demandedPaths.push_back(path.str());
    }
  });
  auto recordManifest =
      collectRecursiveRecordManifest(gadget, false, demandedPaths);
  if (failed(recordManifest))
    return failure();
  const llvm::StringMap<unsigned> &produced =
      recordManifest->typedProducerCounts;
  auto checkPath = [&](Operation *declaration,
                       StringAttr record) -> LogicalResult {
    StringRef path = record.getValue();
    if (!path.consume_front(gadgetName) || !path.consume_front("."))
      return declaration->emitOpError("record '")
             << record.getValue() << "' is not owned by gadget @" << gadgetName;
    unsigned producerCount = produced.lookup(path);
    if (producerCount == 0) {
      return declaration->emitOpError("record '")
             << record.getValue()
             << "' is not an exact output of a typed record-producing "
                "operation in gadget @"
             << gadgetName;
    }
    if (producerCount != 1)
      return declaration->emitOpError("record '")
             << record.getValue() << "' has " << producerCount
             << " typed producers in gadget @" << gadgetName
             << "; stable record paths require exactly one";
    return success();
  };
  auto checkRecords = [&](Operation *declaration) -> LogicalResult {
    auto records = declaration->getAttrOfType<ArrayAttr>("records");
    if (records) {
      llvm::StringSet<> seenRecords;
      for (Attribute value : records) {
        auto record = dyn_cast<StringAttr>(value);
        if (!record)
          return declaration->emitOpError(
              "stable record references must currently be string attributes");
        if (!seenRecords.insert(record.getValue()).second)
          return declaration->emitOpError(
              "scalar profile record support must be duplicate-free");
      }
      for (Attribute value : records) {
        auto record = cast<StringAttr>(value);
        if (failed(checkPath(declaration, record)))
          return failure();
      }
    }
    return success();
  };

  auto checkInputSyndromes = [&](Operation *declaration) -> LogicalResult {
    auto inputs = declaration->getAttrOfType<ArrayAttr>("input_syndromes");
    if (!inputs)
      return success();
    llvm::StringSet<> seenTerms;
    SmallVector<std::pair<int64_t, int64_t>> terms;
    for (Attribute raw : inputs) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto port =
          entry ? entry.getAs<IntegerAttr>("port_index") : IntegerAttr{};
      auto index = entry ? entry.getAs<IntegerAttr>("index") : IntegerAttr{};
      if (!port || !index)
        return declaration->emitOpError(
            "input syndrome terms require port_index and index");
      std::string key =
          (Twine(port.getInt()) + ":" + Twine(index.getInt())).str();
      if (!seenTerms.insert(key).second)
        return declaration->emitOpError(
            "scalar profile input-syndrome support must be duplicate-free");
      terms.emplace_back(port.getInt(), index.getInt());
    }
    for (auto [port, index] : terms)
      if (port < 0 || port >= static_cast<int64_t>(inputWidths.size()) ||
          index < 0 || index >= inputWidths[port])
        return declaration->emitOpError("invalid input syndrome term ")
               << port << "[" << index << "]";
    return success();
  };

  llvm::StringSet<> outputAssignments;
  int64_t expectedOutputAssignments = 0;
  for (int64_t width : outputWidths) {
    if (width < 0 ||
        width > std::numeric_limits<int64_t>::max() - expectedOutputAssignments)
      return emitOpError(
          "aggregate output syndrome width exceeds the supported signed i64 "
          "range");
    expectedOutputAssignments += width;
  }
  auto checkOutputSyndrome = [&](OutputSyndromeOp assignment) -> LogicalResult {
    if (assignment.getPortIndex() < 0 ||
        assignment.getPortIndex() >=
            static_cast<int64_t>(outputWidths.size()) ||
        assignment.getIndex() < 0 ||
        assignment.getIndex() >= outputWidths[assignment.getPortIndex()])
      return assignment.emitOpError("invalid output syndrome target ")
             << assignment.getPortIndex() << "[" << assignment.getIndex()
             << "]";
    std::string key =
        (Twine(assignment.getPortIndex()) + ":" + Twine(assignment.getIndex()))
            .str();
    if (!outputAssignments.insert(key).second)
      return assignment.emitOpError("duplicates output syndrome target ")
             << key;
    return success();
  };

  WalkResult result = getBody().walk([&](Operation *op) -> WalkResult {
    if (isa<SuccessOp, OutputSyndromeOp>(op)) {
      if (failed(checkRecords(op)) || failed(checkInputSyndromes(op)))
        return WalkResult::interrupt();
    }
    if (auto assignment = dyn_cast<OutputSyndromeOp>(op))
      if (failed(checkOutputSyndrome(assignment)))
        return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  // A canonical role-tagged OutcomeMap is the affine authority for success
  // rows. If a profile authors them, its complete ordered support, input
  // terms, and constants must agree with the linked GadgetSpec.
  if (spec && spec.getOutcomeMap()) {
    DictionaryAttr outcome = *spec.getOutcomeMap();
    auto roleRows = outcome.getAs<ArrayAttr>("roles");
    auto outcomeRecords = outcome.getAs<ArrayAttr>("records");
    auto rows = outcome.getAs<DenseIntElementsAttr>("rows");
    auto constants = outcome.getAs<DenseI64ArrayAttr>("constants");
    if (roleRows && (!outcomeRecords || !rows || !constants))
      return emitOpError(
          "linked role-tagged GadgetSpec outcome_map is not canonical");
    if (roleRows) {
      auto shape = rows.getType().getShape();
      auto denseValues = rows.getValues<APInt>();
      SmallVector<APInt> values(denseValues.begin(), denseValues.end());
      SmallVector<SuccessOp> successes;
      getBody().walk([&](Operation *op) {
        if (auto success = dyn_cast<SuccessOp>(op))
          successes.push_back(success);
      });

      auto verifyRole = [&](ArrayRef<Operation *> declarations,
                            StringRef role) -> LogicalResult {
        SmallVector<int64_t> selectedRows;
        for (int64_t row = 0; row < shape[0]; ++row)
          if (outcomeRowHasRole(outcome, row, role))
            selectedRows.push_back(row);
        if (selectedRows.empty() || declarations.empty())
          return success();
        if (declarations.size() != selectedRows.size())
          return emitOpError("declares ")
                 << declarations.size() << " " << role
                 << " row(s), but the linked GadgetSpec outcome_map tags "
                 << selectedRows.size() << " authoritative row(s)";

        for (auto [ordinal, row] : llvm::enumerate(selectedRows)) {
          SmallVector<StringRef> expectedRecords;
          for (int64_t column = 0; column < shape[1]; ++column)
            if (!values[row * shape[1] + column].isZero())
              expectedRecords.push_back(
                  cast<StringAttr>(outcomeRecords[column]).getValue());
          SmallVector<StringRef> actualRecords;
          if (auto records =
                  declarations[ordinal]->getAttrOfType<ArrayAttr>("records"))
            for (Attribute raw : records) {
              auto record = dyn_cast<StringAttr>(raw);
              if (!record)
                return emitOpError("profile ")
                       << role << " records must be stable string paths";
              StringRef path = record.getValue();
              if (!path.consume_front(gadgetName) || !path.consume_front("."))
                return emitOpError("profile ")
                       << role << " record '" << record.getValue()
                       << "' is not owned by gadget @" << gadgetName;
              actualRecords.push_back(path);
            }
          bool actualConstant = false;
          if (auto constant =
                  declarations[ordinal]->getAttrOfType<BoolAttr>("constant"))
            actualConstant = constant.getValue();
          bool expectedConstant = constants.asArrayRef()[row] != 0;
          if (actualRecords != expectedRecords ||
              getProfileSyndromeTerms(declarations[ordinal]) !=
                  getOutcomeSyndromeTerms(outcome, row) ||
              actualConstant != expectedConstant)
            return emitOpError("profile ")
                   << role << " row " << ordinal
                   << " disagrees with linked GadgetSpec outcome_map row "
                   << row;
        }
        return success();
      };

      SmallVector<Operation *> successDeclarations;
      for (SuccessOp success : successes)
        successDeclarations.push_back(success.getOperation());
      if (failed(verifyRole(successDeclarations, "success")))
        return failure();
    }
  }

  if (getBoundaryComplete() && static_cast<int64_t>(outputAssignments.size()) !=
                                   expectedOutputAssignments)
    return emitOpError("boundary_complete profile assigns ")
           << outputAssignments.size() << " of " << expectedOutputAssignments
           << " output syndrome components";

  return success();
}

//===----------------------------------------------------------------------===//
// Generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "qlx/Dialect/Fabric/IR/FabricOps.cpp.inc"

//===----------------------------------------------------------------------===//
// Fabric Dialect initialization
//===----------------------------------------------------------------------===//

void FabricDialect::initialize() {
  addInterfaces<FabricDeviceBindingDialectInterface>();
  addTypes<
#define GET_TYPEDEF_LIST
#include "qlx/Dialect/Fabric/IR/FabricTypes.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "qlx/Dialect/Fabric/IR/FabricAttrs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "qlx/Dialect/Fabric/IR/FabricOps.cpp.inc"
      >();
}
