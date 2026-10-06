/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "qlx/Conversion/Passes.h"
#include "qlx/Dialect/Cflow/IR/CflowOps.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/LVM/IR/LVMTypes.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Parser/Parser.h"

#include <limits>
#include <set>

namespace qlx {
#define GEN_PASS_DEF_QLXTOLVM
#include "qlx/Conversion/Passes.h.inc"
} // namespace qlx

using namespace mlir;

namespace {

static std::string sha256(StringRef text) {
  llvm::SHA256 digest;
  digest.update(text);
  auto bytes = digest.final();
  std::string result = "sha256:";
  llvm::raw_string_ostream stream(result);
  static constexpr char hex[] = "0123456789abcdef";
  for (uint8_t byte : bytes)
    stream << hex[byte >> 4] << hex[byte & 0xf];
  return result;
}

static std::string jsonString(StringRef text) {
  return llvm::formatv("{0}", llvm::json::Value(text.str())).str();
}

class NativePlacer {
public:
  NativePlacer(ModuleOp module, qlx::ProgramOp program,
               qlx::lvm::DomainOp domain, StringRef resultName,
               StringRef witnessOutput)
      : module(module), program(program), domain(domain), builder(module),
        resultName(resultName.str()), witnessOutput(witnessOutput.str()) {}

  LogicalResult run() {
    if (failed(indexSpaces()) || failed(preflight()) ||
        failed(indexCompatibility()) || failed(createKernel()))
      return failure();
    setModuleProfile();
    return success();
  }

private:
  struct SpaceState {
    qlx::lvm::SpaceOp op;
    std::optional<int64_t> capacity;
    std::set<int64_t> occupied;
    std::set<int64_t> recycled;
    int64_t nextUnused = 0;
    bool compatible = false;
  };

  struct Allocation {
    unsigned space = 0;
    int64_t slot = 0;
    SymbolRefAttr reference;
  };

  ModuleOp module;
  qlx::ProgramOp program;
  qlx::lvm::DomainOp domain;
  OpBuilder builder;
  std::string resultName;
  std::string witnessOutput;
  SmallVector<SpaceState> spaces;
  DenseMap<Value, Value> values;
  DenseMap<Value, SymbolRefAttr> placements;
  DenseMap<Value, Allocation> allocations;
  struct BindingRecord {
    std::string source;
    std::string space;
    int64_t slot;
  };
  SmallVector<BindingRecord> bindingRecords;
  llvm::StringSet<> requiredCapabilities;
  int64_t nextSite = 0;

  LogicalResult indexSpaces() {
    for (auto space : domain.getBody().getOps<qlx::lvm::SpaceOp>()) {
      SpaceState state{space};
      if (auto capacity = space.getCapacityAttr())
        state.capacity = capacity.getInt();
      spaces.push_back(state);
    }
    if (spaces.empty())
      return domain.emitOpError(
          "native placement requires at least one lvm.space");
    return success();
  }

  bool supports(const SpaceState &space) const { return space.compatible; }

  LogicalResult indexCompatibility() {
    for (SpaceState &space : spaces) {
      llvm::StringSet<> available;
      for (Attribute raw : space.op.getCapabilities())
        if (auto capability = dyn_cast<qlx::lvm::CapabilityAttr>(raw))
          available.insert(capability.getKey());
      space.compatible =
          llvm::all_of(requiredCapabilities, [&](const auto &required) {
            return available.contains(required.getKey());
          });
    }
    return success();
  }

  /// Take the lowest available slot without rescanning the occupied prefix.
  /// Virgin slots form a monotonic frontier; released slots are kept ordered.
  /// Allocation is therefore O(log n), rather than O(n) per live owner.
  FailureOr<int64_t> takeSlot(SpaceState &space) {
    int64_t slot = 0;
    if (!space.recycled.empty()) {
      auto first = space.recycled.begin();
      slot = *first;
      space.recycled.erase(first);
    } else {
      if (space.capacity && space.nextUnused >= *space.capacity)
        return failure();
      slot = space.nextUnused++;
    }
    if (!space.occupied.insert(slot).second)
      return failure();
    return slot;
  }

  FailureOr<Allocation> allocate(Location location, StringRef source) {
    for (auto [spaceIndex, space] : llvm::enumerate(spaces)) {
      if (!supports(space))
        continue;
      auto slot = takeSlot(space);
      if (failed(slot))
        continue;
      auto reference =
          SymbolRefAttr::get(builder.getContext(), domain.getSymName(),
                             {FlatSymbolRefAttr::get(builder.getContext(),
                                                     space.op.getSymName())});
      bindingRecords.push_back(
          {source.str(), space.op.getSymName().str(), *slot});
      return Allocation{static_cast<unsigned>(spaceIndex), *slot, reference};
    }
    emitError(location)
        << "native P0-to-P1 placement found no logical space satisfying "
           "required capabilities and peak-live capacity in lvm.domain @"
        << domain.getSymName();
    return failure();
  }

  LogicalResult preflight() {
    program.walk([&](Operation *operation) {
      if (isa<qlx::PrepareOp>(operation))
        requiredCapabilities.insert("qlx.machine/logical_reset");
      else if (auto apply = dyn_cast<qlx::ApplyOp>(operation)) {
        requiredCapabilities.insert("qlx.machine/logical_compute");
        if (isa<FlatSymbolRefAttr>(apply.getActionAttr()) &&
            (llvm::any_of(apply.getOperandTypes(),
                          [&](Type type) { return isQubit(type); }) ||
             llvm::any_of(apply.getResultTypes(),
                          [&](Type type) { return isQubit(type); })))
          apply.emitOpError("native placement requires a typed ownership map "
                            "for custom quantum actions");
      } else if (isa<qlx::MeasureOp>(operation))
        requiredCapabilities.insert("qlx.machine/logical_measurement");
      else if (isa<qlx::IdleOp>(operation))
        requiredCapabilities.insert("qlx.machine/logical_memory");
    });
    bool customError = false;
    program.walk([&](qlx::ApplyOp apply) {
      if (isa<FlatSymbolRefAttr>(apply.getActionAttr()) &&
          (llvm::any_of(apply.getOperandTypes(),
                        [&](Type type) { return isQubit(type); }) ||
           llvm::any_of(apply.getResultTypes(),
                        [&](Type type) { return isQubit(type); })))
        customError = true;
    });
    return failure(customError);
  }

  bool isQubit(Type type) const { return isa<qlx::LogicalQubitType>(type); }

  FailureOr<Type> convertType(Type type, SymbolRefAttr placement = {}) {
    if (!isQubit(type))
      return type;
    if (!placement)
      return failure();
    return qlx::lvm::LogicalQubitType::get(builder.getContext(), placement);
  }

  FailureOr<Value> mapped(Value value) {
    auto found = values.find(value);
    if (found == values.end()) {
      emitError(value.getLoc())
          << "native P0-to-P1 has no mapping for value " << value;
      return failure();
    }
    return found->second;
  }

  FailureOr<SymbolRefAttr> placementOf(Value value) {
    auto found = placements.find(value);
    if (found != placements.end())
      return found->second;

    Operation *owner = value.getDefiningOp();
    if (!owner)
      return failure();
    if (isa<qlx::ApplyOp, qlx::InstrumentOp, qlx::IdleOp>(owner)) {
      SmallVector<Value> quantumInputs;
      for (Value input : owner->getOperands())
        if (isQubit(input.getType()))
          quantumInputs.push_back(input);
      SmallVector<Value> quantumResults;
      for (Value result : owner->getResults())
        if (isQubit(result.getType()))
          quantumResults.push_back(result);
      auto it = llvm::find(quantumResults, value);
      if (it == quantumResults.end())
        return failure();
      unsigned index = std::distance(quantumResults.begin(), it);
      if (index >= quantumInputs.size())
        return failure();
      return placementOf(quantumInputs[index]);
    }
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(owner)) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      if (index < repeat.getInits().size())
        return placementOf(repeat.getInits()[index]);
    }
    if (auto whileOp = dyn_cast<qlx::cflow::WhileOp>(owner)) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      if (index < whileOp.getInits().size())
        return placementOf(whileOp.getInits()[index]);
    }
    if (auto ifOp = dyn_cast<qlx::cflow::IfOp>(owner)) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      auto thenYield = dyn_cast<qlx::cflow::YieldOp>(
          ifOp.getThenRegion().front().getTerminator());
      auto elseYield = dyn_cast<qlx::cflow::YieldOp>(
          ifOp.getElseRegion().front().getTerminator());
      if (!thenYield || !elseYield || index >= thenYield.getNumOperands() ||
          index >= elseYield.getNumOperands())
        return failure();
      auto left = placementOf(thenYield.getOperand(index));
      auto right = placementOf(elseYield.getOperand(index));
      if (failed(left) || failed(right) || *left != *right) {
        ifOp.emitOpError("branch join changes logical placement");
        return failure();
      }
      return *left;
    }
    return failure();
  }

  FailureOr<Allocation> allocationOf(Value value) {
    auto found = allocations.find(value);
    if (found != allocations.end())
      return found->second;
    Operation *owner = value.getDefiningOp();
    if (!owner)
      return failure();
    SmallVector<Value> quantumInputs;
    if (isa<qlx::ApplyOp, qlx::InstrumentOp, qlx::IdleOp>(owner)) {
      for (Value input : owner->getOperands())
        if (isQubit(input.getType()))
          quantumInputs.push_back(input);
      SmallVector<Value> quantumResults;
      for (Value result : owner->getResults())
        if (isQubit(result.getType()))
          quantumResults.push_back(result);
      auto it = llvm::find(quantumResults, value);
      unsigned index = std::distance(quantumResults.begin(), it);
      if (it == quantumResults.end() || index >= quantumInputs.size())
        return failure();
      return allocationOf(quantumInputs[index]);
    }
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(owner))
      return allocationOf(
          repeat.getInits()[cast<OpResult>(value).getResultNumber()]);
    if (auto ifOp = dyn_cast<qlx::cflow::IfOp>(owner)) {
      unsigned index = cast<OpResult>(value).getResultNumber();
      auto thenYield = cast<qlx::cflow::YieldOp>(
          ifOp.getThenRegion().front().getTerminator());
      auto elseYield = cast<qlx::cflow::YieldOp>(
          ifOp.getElseRegion().front().getTerminator());
      auto left = allocationOf(thenYield.getOperand(index));
      auto right = allocationOf(elseYield.getOperand(index));
      if (failed(left) || failed(right) || left->space != right->space ||
          left->slot != right->slot) {
        ifOp.emitOpError("branch join changes logical slot assignment");
        return failure();
      }
      return *left;
    }
    return failure();
  }

  LogicalResult release(Value value) {
    auto allocation = allocationOf(value);
    if (failed(allocation))
      return emitError(value.getLoc(),
                       "native placement cannot release logical owner");
    SpaceState &space = spaces[allocation->space];
    if (!space.occupied.erase(allocation->slot))
      return emitError(value.getLoc(),
                       "native placement detected a non-live logical owner");
    if (!space.recycled.insert(allocation->slot).second)
      return emitError(value.getLoc(),
                       "native placement recycled a logical slot twice");
    return success();
  }

  Operation *create(StringRef name, Location location, ValueRange operands,
                    TypeRange results, ArrayRef<NamedAttribute> attributes,
                    unsigned regions = 0) {
    OperationState state(location, name);
    state.addOperands(operands);
    state.addTypes(results);
    state.addAttributes(attributes);
    for (unsigned i = 0; i < regions; ++i)
      state.addRegion();
    return builder.create(state);
  }

  LogicalResult createKernel() {
    Block &source = program.getBody().front();
    SmallVector<Type> inputTypes;
    for (auto [index, argument] : llvm::enumerate(source.getArguments())) {
      if (!isQubit(argument.getType())) {
        inputTypes.push_back(argument.getType());
        continue;
      }
      auto placement =
          allocate(argument.getLoc(), ("argument:" + Twine(index)).str());
      if (failed(placement))
        return failure();
      placements[argument] = placement->reference;
      allocations[argument] = *placement;
      inputTypes.push_back(qlx::lvm::LogicalQubitType::get(
          builder.getContext(), placement->reference));
    }

    SmallVector<Type> resultTypes;
    for (Type type : program.getFunctionType().getResults()) {
      if (isQubit(type))
        return program.emitOpError(
            "native placement currently requires classical program results; "
            "use qlx.place(...) for a quantum-valued P1 ABI");
      resultTypes.push_back(type);
    }
    auto functionType = builder.getFunctionType(inputTypes, resultTypes);
    std::string selectedName = resultName.empty()
                                   ? (program.getSymName() + "_placed").str()
                                   : resultName;
    if (SymbolTable::lookupSymbolIn(module, selectedName))
      return program.emitOpError(
                 "native placement result symbol already exists @")
             << selectedName;

    builder.setInsertionPointAfter(domain);
    OperationState state(program.getLoc(),
                         qlx::lvm::KernelOp::getOperationName());
    state.addAttribute(SymbolTable::getSymbolAttrName(),
                       builder.getStringAttr(selectedName));
    state.addAttribute("domain", FlatSymbolRefAttr::get(builder.getContext(),
                                                        domain.getSymName()));
    state.addAttribute("function_type", TypeAttr::get(functionType));
    state.addAttribute(
        "input_p0",
        FlatSymbolRefAttr::get(builder.getContext(), program.getSymName()));
    state.addAttribute("qlx.profile", builder.getStringAttr("p1"));
    state.addAttribute("qlx.stage", builder.getStringAttr("p1"));
    if (auto specialization = program.getSpecializationAttr())
      state.addAttribute("specialization", specialization);
    state.addRegion();
    Operation *created = builder.create(state);
    auto kernel = cast<qlx::lvm::KernelOp>(created);
    Block *target = builder.createBlock(
        &kernel.getBody(), {}, inputTypes,
        SmallVector<Location>(inputTypes.size(), program.getLoc()));
    for (auto [oldValue, newValue] :
         llvm::zip(source.getArguments(), target->getArguments()))
      values[oldValue] = newValue;

    builder.setInsertionPointToStart(target);
    for (Operation &operation : source.getOperations())
      if (failed(convert(&operation))) {
        kernel.erase();
        return failure();
      }

    std::string programText;
    llvm::raw_string_ostream programStream(programText);
    program.print(programStream, OpPrintingFlags().enableDebugInfo(false));
    std::string domainText;
    llvm::raw_string_ostream domainStream(domainText);
    domain.print(domainStream, OpPrintingFlags().enableDebugInfo(false));
    std::string canonical;
    llvm::raw_string_ostream witness(canonical);
    witness << "{\"schema\":\"qlx.placement-witness/v1\",\"root\":"
            << jsonString(program.getSymName())
            << ",\"domain\":" << jsonString(domain.getSymName())
            << ",\"p0_sha256\":" << jsonString(sha256(programText))
            << ",\"domain_sha256\":" << jsonString(sha256(domainText))
            << ",\"objective\":\"first_fit\",\"tie_break\":"
               "\"declaration_order\",\"bindings\":[";
    for (auto [index, binding] : llvm::enumerate(bindingRecords)) {
      if (index)
        witness << ',';
      witness << "{\"source\":" << jsonString(binding.source)
              << ",\"space\":" << jsonString(binding.space)
              << ",\"slot\":" << binding.slot << '}';
    }
    witness << "]}\n";
    kernel->setAttr("placement_witness_sha256",
                    builder.getStringAttr(sha256(canonical)));
    kernel->setAttr("qlx.placement_policy",
                    builder.getStringAttr("native-first-fit/v3"));
    if (!witnessOutput.empty()) {
      std::error_code error;
      llvm::ToolOutputFile output(witnessOutput, error, llvm::sys::fs::OF_None);
      if (error) {
        kernel.emitOpError("cannot create detached placement witness '")
            << witnessOutput << "': " << error.message();
        kernel.erase();
        return failure();
      }
      output.os() << canonical;
      output.os().flush();
      if (output.os().has_error()) {
        output.os().clear_error();
        kernel.emitOpError("failed to write detached placement witness '")
            << witnessOutput << "'";
        kernel.erase();
        return failure();
      }
      output.keep();
    }
    return success();
  }

  LogicalResult mapResults(Operation *source, Operation *target,
                           ArrayRef<SymbolRefAttr> resultPlacements = {}) {
    unsigned qIndex = 0;
    for (auto [oldValue, newValue] :
         llvm::zip(source->getResults(), target->getResults())) {
      values[oldValue] = newValue;
      if (isQubit(oldValue.getType())) {
        if (qIndex >= resultPlacements.size())
          return source->emitOpError(
              "native placement could not determine a result placement");
        placements[oldValue] = resultPlacements[qIndex++];
      }
    }
    return success();
  }

  LogicalResult mapAllocatedResults(Operation *source, Operation *target,
                                    ArrayRef<Allocation> resultAllocations) {
    unsigned qIndex = 0;
    for (auto [oldValue, newValue] :
         llvm::zip(source->getResults(), target->getResults())) {
      values[oldValue] = newValue;
      if (!isQubit(oldValue.getType()))
        continue;
      if (qIndex >= resultAllocations.size())
        return source->emitOpError(
            "native placement could not determine a result slot");
      const Allocation &allocation = resultAllocations[qIndex++];
      placements[oldValue] = allocation.reference;
      allocations[oldValue] = allocation;
    }
    return success();
  }

  FailureOr<SmallVector<Value>> mapOperands(ValueRange source) {
    SmallVector<Value> result;
    for (Value value : source) {
      auto converted = mapped(value);
      if (failed(converted))
        return failure();
      result.push_back(*converted);
    }
    return result;
  }

  LogicalResult convertRegion(Region &source, Region &target,
                              ValueRange initValues = {}) {
    OpBuilder::InsertionGuard guard(builder);
    Block &sourceBlock = source.front();
    SmallVector<Type> argumentTypes;
    SmallVector<Location> locations;
    for (auto [index, argument] : llvm::enumerate(sourceBlock.getArguments())) {
      SymbolRefAttr placement;
      if (isQubit(argument.getType())) {
        if (index >= initValues.size())
          return emitError(argument.getLoc(),
                           "native placement cannot bind region argument");
        auto inferred = placementOf(initValues[index]);
        auto allocation = allocationOf(initValues[index]);
        if (failed(inferred) || failed(allocation))
          return failure();
        placement = *inferred;
        placements[argument] = placement;
        allocations[argument] = *allocation;
      }
      auto type = convertType(argument.getType(), placement);
      if (failed(type))
        return failure();
      argumentTypes.push_back(*type);
      locations.push_back(argument.getLoc());
    }
    Block *targetBlock =
        builder.createBlock(&target, {}, argumentTypes, locations);
    for (auto [oldValue, newValue] :
         llvm::zip(sourceBlock.getArguments(), targetBlock->getArguments()))
      values[oldValue] = newValue;
    builder.setInsertionPointToStart(targetBlock);
    for (Operation &operation : sourceBlock.getOperations())
      if (failed(convert(&operation)))
        return failure();
    return success();
  }

  LogicalResult convert(Operation *operation) {
    Location location = operation->getLoc();
    if (auto ret = dyn_cast<qlx::ReturnOp>(operation)) {
      auto operands = mapOperands(ret.getOperands());
      if (failed(operands))
        return failure();
      create(qlx::lvm::ReturnOp::getOperationName(), location, *operands, {},
             {});
      return success();
    }
    if (auto constant = dyn_cast<arith::ConstantOp>(operation)) {
      Operation *target =
          create(arith::ConstantOp::getOperationName(), location, {},
                 operation->getResultTypes(), operation->getAttrs());
      return mapResults(operation, target);
    }
    if (auto prepare = dyn_cast<qlx::PrepareOp>(operation)) {
      std::string source = "prepare:" + std::to_string(nextSite);
      if (auto allocation = prepare.getAllocationAttr())
        source = ("allocation:" + Twine(allocation.getInt()) + ":" +
                  Twine(prepare.getValueIndexAttr()
                            ? prepare.getValueIndexAttr().getInt()
                            : 0))
                     .str();
      auto placement = allocate(location, source);
      if (failed(placement))
        return failure();
      auto type = qlx::lvm::LogicalQubitType::get(builder.getContext(),
                                                  placement->reference);
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("state", prepare.getStateAttr()),
          builder.getNamedAttr("at", placement->reference),
          builder.getNamedAttr("site", builder.getI64IntegerAttr(nextSite++)),
      };
      Operation *target = create(qlx::lvm::PrepareOp::getOperationName(),
                                 location, {}, {type}, attrs);
      return mapAllocatedResults(operation, target, {*placement});
    }
    if (auto apply = dyn_cast<qlx::ApplyOp>(operation)) {
      auto operands = mapOperands(apply.getInputs());
      if (failed(operands))
        return failure();
      SmallVector<SymbolRefAttr> quantumPlacements;
      SmallVector<Allocation> quantumAllocations;
      for (Value input : apply.getInputs())
        if (isQubit(input.getType())) {
          auto placement = placementOf(input);
          auto allocation = allocationOf(input);
          if (failed(placement) || failed(allocation))
            return failure();
          quantumPlacements.push_back(*placement);
          quantumAllocations.push_back(*allocation);
        }
      if (!llvm::all_equal(quantumPlacements))
        return apply.emitOpError(
            "native first-fit rejects cross-space actions; use qlx.place(...) "
            "with an explicit distributed protocol");
      SmallVector<Type> resultTypes;
      unsigned qIndex = 0;
      for (Type type : apply.getResultTypes()) {
        auto converted =
            convertType(type, isQubit(type) ? quantumPlacements[qIndex++]
                                            : SymbolRefAttr{});
        if (failed(converted))
          return failure();
        resultTypes.push_back(*converted);
      }
      SmallVector<Attribute> placementAttrs(quantumPlacements.begin(),
                                            quantumPlacements.end());
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("action", apply.getActionAttr()),
          builder.getNamedAttr("placements",
                               builder.getArrayAttr(placementAttrs)),
          builder.getNamedAttr("site", builder.getI64IntegerAttr(nextSite++)),
      };
      if (auto parameters = apply.getParametersAttr())
        attrs.push_back(builder.getNamedAttr("parameters", parameters));
      Operation *target = create(qlx::lvm::ApplyOp::getOperationName(),
                                 location, *operands, resultTypes, attrs);
      return mapAllocatedResults(operation, target, quantumAllocations);
    }
    if (auto measure = dyn_cast<qlx::MeasureOp>(operation)) {
      auto input = mapped(measure.getInput());
      auto placement = placementOf(measure.getInput());
      if (failed(input) || failed(placement))
        return failure();
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("basis", measure.getBasisAttr()),
          builder.getNamedAttr("at", *placement),
          builder.getNamedAttr("site", builder.getI64IntegerAttr(nextSite++)),
      };
      Operation *target =
          create(qlx::lvm::MeasureOp::getOperationName(), location, {*input},
                 operation->getResultTypes(), attrs);
      if (failed(release(measure.getInput())))
        return failure();
      return mapResults(operation, target);
    }
    if (auto discard = dyn_cast<qlx::DiscardOp>(operation)) {
      auto operands = mapOperands(discard.getInputs());
      if (failed(operands))
        return failure();
      SmallVector<Attribute> refs;
      for (Value input : discard.getInputs()) {
        auto placement = placementOf(input);
        if (failed(placement))
          return failure();
        refs.push_back(*placement);
      }
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("placements", builder.getArrayAttr(refs))};
      if (auto reason = discard.getReasonAttr())
        attrs.push_back(builder.getNamedAttr("reason", reason));
      create(qlx::lvm::DiscardOp::getOperationName(), location, *operands, {},
             attrs);
      for (Value input : discard.getInputs())
        if (failed(release(input)))
          return failure();
      return success();
    }
    if (auto idle = dyn_cast<qlx::IdleOp>(operation)) {
      auto operands = mapOperands(idle->getOperands());
      if (failed(operands))
        return failure();
      SmallVector<SymbolRefAttr> refs;
      SmallVector<Allocation> resultAllocations;
      SmallVector<Type> types;
      for (auto [input, result] :
           llvm::zip(idle.getInputs(), idle.getResults())) {
        auto placement = placementOf(input);
        if (failed(placement))
          return failure();
        refs.push_back(*placement);
        auto allocation = allocationOf(input);
        if (failed(allocation))
          return failure();
        resultAllocations.push_back(*allocation);
        types.push_back(
            qlx::lvm::LogicalQubitType::get(builder.getContext(), *placement));
      }
      SmallVector<Attribute> attrs(refs.begin(), refs.end());
      Operation *target = create(
          qlx::lvm::IdleOp::getOperationName(), location, *operands, types,
          {builder.getNamedAttr("placements", builder.getArrayAttr(attrs))});
      return mapAllocatedResults(operation, target, resultAllocations);
    }
    if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
      auto operands = mapOperands(repeat.getInits());
      if (failed(operands))
        return failure();
      SmallVector<Type> types;
      SmallVector<SymbolRefAttr> refs;
      SmallVector<Allocation> resultAllocations;
      for (auto [init, result] :
           llvm::zip(repeat.getInits(), repeat.getResults())) {
        SymbolRefAttr placement;
        if (isQubit(result.getType())) {
          auto inferred = placementOf(init);
          if (failed(inferred))
            return failure();
          placement = *inferred;
          refs.push_back(placement);
          auto allocation = allocationOf(init);
          if (failed(allocation))
            return failure();
          resultAllocations.push_back(*allocation);
        }
        auto type = convertType(result.getType(), placement);
        if (failed(type))
          return failure();
        types.push_back(*type);
      }
      Operation *target = create(
          qlx::cflow::RepeatOp::getOperationName(), location, *operands, types,
          {builder.getNamedAttr("count", repeat.getCountAttr())}, 1);
      auto entrySpaces = spaces;
      if (failed(convertRegion(repeat.getBody(), target->getRegion(0),
                               repeat.getInits())))
        return failure();
      if (repeat.getCount() == 0) {
        spaces = entrySpaces;
      } else {
        for (auto [before, after] : llvm::zip(entrySpaces, spaces))
          if (before.occupied != after.occupied)
            return repeat.emitOpError(
                "native placement requires a live-slot fixed point across "
                "repeat iterations");
        auto yield =
            cast<qlx::cflow::YieldOp>(repeat.getBody().front().getTerminator());
        unsigned quantumIndex = 0;
        for (auto [index, init] : llvm::enumerate(repeat.getInits())) {
          if (!isQubit(init.getType()))
            continue;
          auto initial = allocationOf(init);
          auto loopBack = allocationOf(yield.getOperand(index));
          if (failed(initial) || failed(loopBack) ||
              initial->space != loopBack->space ||
              initial->slot != loopBack->slot)
            return repeat.emitOpError(
                "native placement requires each repeat yield to preserve "
                "its carried owner's exact logical slot");
          resultAllocations[quantumIndex++] = *loopBack;
        }
      }
      return mapAllocatedResults(operation, target, resultAllocations);
    }
    if (auto ifOp = dyn_cast<qlx::cflow::IfOp>(operation)) {
      auto condition = mapped(ifOp.getCondition());
      if (failed(condition))
        return failure();
      auto thenYield = cast<qlx::cflow::YieldOp>(
          ifOp.getThenRegion().front().getTerminator());
      auto elseYield = cast<qlx::cflow::YieldOp>(
          ifOp.getElseRegion().front().getTerminator());
      SmallVector<Type> types;
      SmallVector<SymbolRefAttr> refs;
      SmallVector<Allocation> resultAllocations;
      auto previewSpaces = spaces;
      auto previewAllocate = [&]() -> FailureOr<Allocation> {
        for (auto [spaceIndex, space] : llvm::enumerate(previewSpaces)) {
          if (!supports(space))
            continue;
          auto slot = takeSlot(space);
          if (failed(slot))
            continue;
          auto reference = SymbolRefAttr::get(
              builder.getContext(), domain.getSymName(),
              {FlatSymbolRefAttr::get(builder.getContext(),
                                      space.op.getSymName())});
          return Allocation{static_cast<unsigned>(spaceIndex), *slot,
                            reference};
        }
        return failure();
      };
      for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
        SymbolRefAttr placement;
        if (isQubit(result.getType())) {
          auto left = allocationOf(thenYield.getOperand(index));
          auto right = allocationOf(elseYield.getOperand(index));
          Allocation selected;
          if (succeeded(left) && succeeded(right)) {
            if (left->space != right->space || left->slot != right->slot)
              return ifOp.emitOpError(
                  "branch join changes logical slot assignment");
            selected = *left;
          } else if (succeeded(left)) {
            selected = *left;
          } else if (succeeded(right)) {
            selected = *right;
          } else {
            auto preview = previewAllocate();
            if (failed(preview))
              return ifOp.emitOpError(
                  "native placement cannot reserve a branch-result slot");
            selected = *preview;
          }
          placement = selected.reference;
          refs.push_back(placement);
          resultAllocations.push_back(selected);
        }
        auto type = convertType(result.getType(), placement);
        if (failed(type))
          return failure();
        types.push_back(*type);
      }
      Operation *target = create(qlx::cflow::IfOp::getOperationName(), location,
                                 {*condition}, types, {}, 2);
      auto entrySpaces = spaces;
      if (failed(convertRegion(ifOp.getThenRegion(), target->getRegion(0))))
        return failure();
      auto thenSpaces = spaces;
      spaces = entrySpaces;
      if (failed(convertRegion(ifOp.getElseRegion(), target->getRegion(1))))
        return failure();
      for (auto [thenState, elseState] : llvm::zip(thenSpaces, spaces)) {
        if (thenState.occupied != elseState.occupied)
          return ifOp.emitOpError(
              "native placement requires identical live slots at branch "
              "join");
        // Both alternatives are compiled, so retain a path-independent
        // high-water frontier and make every unoccupied prior slot reusable.
        // This keeps post-join allocation deterministic even when one branch
        // used more temporaries than the other.
        if (thenState.nextUnused > elseState.nextUnused) {
          elseState.nextUnused = thenState.nextUnused;
          elseState.recycled = thenState.recycled;
        }
      }
      unsigned quantumIndex = 0;
      for (auto [index, result] : llvm::enumerate(ifOp.getResults())) {
        if (!isQubit(result.getType()))
          continue;
        auto left = allocationOf(thenYield.getOperand(index));
        auto right = allocationOf(elseYield.getOperand(index));
        const Allocation &expected = resultAllocations[quantumIndex++];
        if (failed(left) || failed(right) || left->space != right->space ||
            left->slot != right->slot || left->space != expected.space ||
            left->slot != expected.slot)
          return ifOp.emitOpError(
              "native placement requires branches to yield the exact "
              "reserved logical slot");
      }
      return mapAllocatedResults(operation, target, resultAllocations);
    }
    if (auto yield = dyn_cast<qlx::cflow::YieldOp>(operation)) {
      auto operands = mapOperands(yield.getOperands());
      if (failed(operands))
        return failure();
      create(qlx::cflow::YieldOp::getOperationName(), location, *operands, {},
             {});
      return success();
    }
    if (auto xorOp = dyn_cast<qlx::XorOp>(operation)) {
      auto operands = mapOperands(xorOp->getOperands());
      if (failed(operands))
        return failure();
      Operation *target = create(qlx::lvm::XorOp::getOperationName(), location,
                                 *operands, operation->getResultTypes(), {});
      return mapResults(operation, target);
    }
    if (isa<qlx::CallOp>(operation))
      return operation->emitOpError(
          "native placement requires P0 calls to be inlined");
    return operation->emitOpError(
        "is not supported by hardened native P0-to-P1 placement; use the "
        "explicit Python qlx.place(...) route for experimental policies");
  }

  void setModuleProfile() {
    auto profiles = module->getAttrOfType<ArrayAttr>("qlx.profiles");
    SmallVector<Attribute> values;
    if (profiles)
      values.append(profiles.begin(), profiles.end());
    if (!llvm::any_of(values, [](Attribute value) {
          auto string = dyn_cast<StringAttr>(value);
          return string && string.getValue() == "p1";
        }))
      values.push_back(builder.getStringAttr("p1"));
    module->setAttr("qlx.profiles", builder.getArrayAttr(values));
  }
};

struct QLXToLVMPass : public qlx::impl::QLXToLVMBase<QLXToLVMPass> {
  using QLXToLVMBase::QLXToLVMBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<Operation *> importedDomains;
    if (!archModule.empty()) {
      auto buffer = llvm::MemoryBuffer::getFile(archModule);
      if (!buffer) {
        module.emitError("qlx-to-lvm cannot read arch-module file: ")
            << archModule;
        return signalPassFailure();
      }
      auto architecture = parseSourceString<ModuleOp>((*buffer)->getBuffer(),
                                                      module.getContext());
      if (!architecture) {
        module.emitError("qlx-to-lvm cannot parse arch-module file: ")
            << archModule;
        return signalPassFailure();
      }
      OpBuilder builder(module.getContext());
      builder.setInsertionPointToStart(module.getBody());
      unsigned imported = 0;
      llvm::StringSet<> incomingNames;
      for (auto source : architecture->getOps<qlx::lvm::DomainOp>()) {
        if (!incomingNames.insert(source.getSymName()).second ||
            SymbolTable::lookupSymbolIn(module, source.getSymName())) {
          source.emitOpError(
              "arch-module domain collides with an input or imported symbol");
          return signalPassFailure();
        }
      }
      for (auto source : architecture->getOps<qlx::lvm::DomainOp>()) {
        Operation *clone = source->clone();
        builder.insert(clone);
        importedDomains.push_back(clone);
        ++imported;
      }
      if (imported == 0) {
        module.emitError("qlx-to-lvm arch-module contains no lvm.domain");
        return signalPassFailure();
      }
    }
    SmallVector<qlx::ProgramOp> programs;
    SmallVector<qlx::lvm::DomainOp> domains;
    module.getBodyRegion().walk([&](qlx::ProgramOp op) {
      if (op->getParentOp() == module)
        programs.push_back(op);
    });
    module.getBodyRegion().walk([&](qlx::lvm::DomainOp op) {
      if (op->getParentOp() == module)
        domains.push_back(op);
    });

    auto selectProgram = [&]() -> qlx::ProgramOp {
      if (!rootSymbol.empty())
        return dyn_cast_or_null<qlx::ProgramOp>(
            SymbolTable::lookupSymbolIn(module, rootSymbol));
      return programs.size() == 1 ? programs.front() : qlx::ProgramOp{};
    };
    auto selectDomain = [&]() -> qlx::lvm::DomainOp {
      if (!domainSymbol.empty())
        return dyn_cast_or_null<qlx::lvm::DomainOp>(
            SymbolTable::lookupSymbolIn(module, domainSymbol));
      return domains.size() == 1 ? domains.front() : qlx::lvm::DomainOp{};
    };
    qlx::ProgramOp program = selectProgram();
    qlx::lvm::DomainOp domain = selectDomain();
    auto rollbackImports = [&] {
      for (Operation *imported : llvm::reverse(importedDomains))
        imported->erase();
      importedDomains.clear();
    };
    if (!program) {
      module.emitError(
          rootSymbol.empty()
              ? "qlx-to-lvm requires exactly one qlx.program or an "
                "explicit root option"
              : "qlx-to-lvm root does not name a qlx.program");
      rollbackImports();
      return signalPassFailure();
    }
    if (!domain) {
      module.emitError(domainSymbol.empty()
                           ? "qlx-to-lvm requires exactly one lvm.domain or an "
                             "explicit domain option"
                           : "qlx-to-lvm domain does not name an lvm.domain");
      rollbackImports();
      return signalPassFailure();
    }
    if (failed(
            NativePlacer(module, program, domain, resultSymbol, witnessOutput)
                .run())) {
      rollbackImports();
      signalPassFailure();
    }
  }
};

} // namespace
