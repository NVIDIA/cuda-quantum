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
#include "qlx/Dialect/Fabric/Transforms/Passes.h"
#include "qlx/Dialect/LVM/IR/LVMOps.h"
#include "qlx/Dialect/Phys/IR/PhysDialect.h"
#include "qlx/Dialect/Phys/IR/PhysOps.h"
#include "qlx/Dialect/QLX/IR/QLXOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include <cassert>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>

namespace qlx {
namespace fabric {
#define GEN_PASS_DEF_FABRICTOPHYS
#include "qlx/Dialect/Fabric/Transforms/Passes.h.inc"
} // namespace fabric
} // namespace qlx

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

static bool isLinearPhysicalDefinitionType(Type type) {
  if (isa<qlx::phys::StateType, qlx::phys::ResourcePayloadType>(type))
    return true;
  auto event = dyn_cast<qlx::event::HandleType>(type);
  return event && event.getOwnership() == "linear";
}

static int64_t countLinearPhysicalDefinitions(Operation *owner) {
  int64_t count = 0;
  owner->walk([&](Operation *operation) {
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        count += llvm::count_if(block.getArgumentTypes(),
                                isLinearPhysicalDefinitionType);
    count += llvm::count_if(operation->getResultTypes(),
                            isLinearPhysicalDefinitionType);
  });
  return count;
}

/// A projected P2 aggregate is copied frequently across pass-through SSA
/// boundaries but mutated only when a physical event changes one of its
/// carriers. Share the contiguous value vector until that first mutation.
/// This preserves cheap ordered iteration for MLIR construction while avoiding
/// eager O(width) copies for every retained P2 value.
class PhysicalGroup {
  using Storage = SmallVector<Value>;

public:
  using const_iterator = Storage::const_iterator;
  /// Opaque proof that two groups have the same ordered physical type layout.
  /// The token is copied only across operations that preserve every result
  /// type in place.  Structural edits detach it, so pointer equality is a
  /// sufficient (but deliberately not necessary) exact-layout proof.
  using LayoutIdentity = std::shared_ptr<const char>;

  PhysicalGroup()
      : storage(std::make_shared<Storage>()),
        layout(std::make_shared<const char>(0)) {}

  PhysicalGroup(std::initializer_list<Value> values) : PhysicalGroup() {
    reserve(values.size());
    append(values.begin(), values.end());
  }

  explicit PhysicalGroup(Value value) : PhysicalGroup({value}) {}

  template <typename Iterator>
  PhysicalGroup(Iterator begin, Iterator end) : PhysicalGroup() {
    append(begin, end);
  }

  template <typename Iterator>
  PhysicalGroup(Iterator begin, Iterator end, LayoutIdentity identity)
      : storage(std::make_shared<Storage>()), layout(std::move(identity)) {
    storage->append(begin, end);
  }

  size_t size() const { return storage->size(); }
  bool empty() const { return storage->empty(); }
  Value front() const { return storage->front(); }
  Value operator[](size_t index) const { return (*storage)[index]; }
  const_iterator begin() const { return storage->begin(); }
  const_iterator end() const { return storage->end(); }
  ValueRange values() const { return *storage; }
  LayoutIdentity getLayoutIdentity() const { return layout; }

  void reserve(size_t count) {
    ensureUniqueStorage();
    storage->reserve(count);
  }

  void push_back(Value value) {
    ensureUniqueStorage();
    ensureUniqueLayout();
    storage->push_back(value);
  }

  template <typename Iterator>
  void append(Iterator begin, Iterator end) {
    ensureUniqueStorage();
    ensureUniqueLayout();
    storage->append(begin, end);
  }

  template <typename Iterator>
  void insert(const_iterator position, Iterator begin, Iterator end) {
    size_t offset = position - storage->begin();
    ensureUniqueStorage();
    ensureUniqueLayout();
    storage->insert(storage->begin() + offset, begin, end);
  }

  void set(size_t index, Value value) {
    ensureUniqueStorage();
    assert((*storage)[index].getType() == value.getType() &&
           "owner-threading operation changed physical layout");
    (*storage)[index] = value;
  }

private:
  void ensureUniqueStorage() {
    if (storage.use_count() != 1)
      storage = std::make_shared<Storage>(*storage);
  }

  void ensureUniqueLayout() {
    if (layout.use_count() != 1)
      layout = std::make_shared<const char>(0);
  }

  std::shared_ptr<Storage> storage;
  LayoutIdentity layout;
};

using ProjectedValues = DenseMap<Value, PhysicalGroup>;

static PhysicalGroup takeProjected(ProjectedValues &projected, Value source,
                                   Operation *consumer) {
  auto found = projected.find(source);
  assert(found != projected.end() && "source value was not projected");
  if (source.hasOneUse() && source.use_begin()->getOwner() == consumer) {
    PhysicalGroup result = std::move(found->second);
    projected.erase(found);
    return result;
  }
  return found->second;
}

static StringRef symbolName(Operation *operation) {
  if (auto symbol = operation->getAttrOfType<StringAttr>(
          SymbolTable::getSymbolAttrName()))
    return symbol.getValue();
  return {};
}

static Operation *createGeneric(OpBuilder &builder, Location location,
                                StringRef name, ValueRange operands = {},
                                TypeRange results = {},
                                ArrayRef<NamedAttribute> attributes = {},
                                unsigned regions = 0) {
  OperationState state(location, name);
  state.addOperands(operands);
  state.addTypes(results);
  state.addAttributes(attributes);
  for (unsigned i = 0; i < regions; ++i)
    state.addRegion();
  return builder.create(state);
}

static StringAttr stringAttr(MLIRContext *context, StringRef value) {
  return StringAttr::get(context, value);
}

static FlatSymbolRefAttr symbolAttr(MLIRContext *context, StringRef value) {
  return FlatSymbolRefAttr::get(context, value);
}

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

struct RegionBinding {
  std::string resourceClass;
  std::string physicalBinding;
  std::string kind;
  std::string granularity;
  int64_t capacity = 0;
  qlx::phys::QECBindingOp binding;
};

enum class PatchMacroKind { Action, Prepare, Measure };

struct PatchMacro {
  PatchMacroKind kind;
  std::string physicalOperation;
};

using CarrierHomes = SmallVector<StringAttr, 1>;
using CarrierHomeMap = DenseMap<Value, CarrierHomes>;

struct ResourceInfo {
  std::string resourceClass;
  int64_t index = 0;
};

struct AllocationPlanStep {
  bool acquire = false;
  std::string resourceClass;
  int64_t capacity = 0;
  SmallVector<int64_t> indices;
};

struct CanonicalCallTemplate {
  Operation *sourceCall = nullptr;
  FlatSymbolRefAttr callee;
  FlatSymbolRefAttr profile;
  // A call whose result directly seeds fabric.retry must retain its complete
  // state boundary.  Such calls may share only with another retry-closing
  // call, never with a canonical body whose pass-through states were elided.
  bool closesRetryBoundary = false;
  // Full source-call boundary used to prove that another invocation is the
  // same physical specialization.  The emitted phys.call boundary below is
  // smaller: state pairs proven to pass through untouched are not consumed
  // and reproduced merely to preserve source arity.
  size_t inputCount = 0;
  size_t outputCount = 0;
  llvm::BitVector retainedInputs;
  llvm::BitVector retainedOutputs;
  SmallVector<Type> retainedInputTypes;
  SmallVector<Type> retainedOutputTypes;
  // Group-granularity exact-layout proofs avoid rediscovering hundreds of
  // identical carrier resource identities at every shared invocation.  A
  // mismatch merely falls back to the complete typed alias proof below.
  SmallVector<PhysicalGroup::LayoutIdentity> inputGroupLayouts;
  SmallVector<PhysicalGroup::LayoutIdentity> outputGroupLayouts;
  StringAttr event;
  std::string instance;
  llvm::StringMap<SmallVector<std::string>> recordProjection;
  SmallVector<AllocationPlanStep> allocationPlan;
};

struct StateAliasCacheKey {
  size_t templateIndex = 0;
  SmallVector<uintptr_t, 4> layouts;

  bool operator<(const StateAliasCacheKey &other) const {
    if (templateIndex != other.templateIndex)
      return templateIndex < other.templateIndex;
    return std::lexicographical_compare(layouts.begin(), layouts.end(),
                                        other.layouts.begin(),
                                        other.layouts.end());
  }
};

struct AllocationRecord {
  std::string name;
  std::string resourceClass;
  SmallVector<Attribute> resources;
  SmallVector<int64_t> indices;
  std::string acquire;
  std::string release;
  SmallVector<std::string> after;
};

struct PhysicalPauliProduct {
  SmallVector<std::pair<unsigned, int64_t>> carriers;
  SmallVector<char> paulis;
  bool invert = false;
};

struct PendingSidecar {
  std::string kind;
  Operation *declaration = nullptr;
  FlatSymbolRefAttr sourceProfile;
  int64_t sourceRow = 0;
  std::string sourceInstance;
  SmallVector<std::string> sourceRecords;
  SmallVector<std::string> physicalRecords;
};

using RecordProjectionKey = std::pair<std::string, std::string>;
using RecordProjectionLane = std::tuple<std::string, std::string, std::string>;
using RepeatContext = SmallVector<std::pair<std::string, int64_t>, 2>;

struct RecordProjectionState {
  SmallVector<std::string> records;
};

using RecordProjectionMap =
    std::map<RecordProjectionKey, RecordProjectionState>;

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
constexpr StringLiteral kNativePauliProductRotation =
    "qlx.physical/native_pauli_product_rotation";

struct SurfaceFactoryRecurrenceLayout {
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

struct SurfaceFactoryStageShape {
  int64_t level1QuarterLayers;
  int64_t level2Layers;
  int64_t autoCCZLayers;
};

using SpacetimeProviderKind = qlx::spacetime::BuiltinProviderKind;

struct RecordProjectionScope {
  DenseSet<const RecordProjectionState *> seen;
  SmallVector<std::pair<const RecordProjectionMap::value_type *, size_t>>
      starts;
};

struct TransformFrameProjection {
  PhysicalGroup source;
  FlatSymbolRefAttr transform;
};

class NativeProjection {
public:
  NativeProjection(ModuleOp module, StringRef rootName, StringRef deviceName,
                   StringRef requestedGraphName, bool useParallelPlanning,
                   bool preflightOnly, bool deferOutputVerification)
      : module(module), context(module.getContext()), symbols(module),
        rootName(rootName.str()), deviceName(deviceName.str()),
        graphName(requestedGraphName.empty() ? (rootName + "_physical").str()
                                             : requestedGraphName.str()),
        useParallelPlanning(useParallelPlanning), preflightOnly(preflightOnly),
        deferOutputVerification(deferOutputVerification) {}

  LogicalResult run() {
    const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
    auto phase = [&](StringRef name, auto &&action) -> LogicalResult {
      const auto started = std::chrono::steady_clock::now();
      LogicalResult result = action();
      if (profile)
        llvm::errs() << "fabric-to-phys: phase " << name << ' '
                     << std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - started)
                            .count()
                     << "s\n";
      return result;
    };
    if (failed(phase("resolve", [&] { return resolveClosure(); })) ||
        failed(phase("discover", [&] { return discoverReachable(); })) ||
        failed(phase("capabilities",
                     [&] { return validatePhysicalCapabilities(); })) ||
        failed(phase("preflight", [&] { return preflight(); })))
      return failure();
    if (preflightOnly)
      return success();
    if (SymbolTable::lookupSymbolIn(module, graphName))
      return module.emitError("fabric-to-phys output symbol @")
             << graphName << " already exists";
    if (profile)
      llvm::errs() << "fabric-to-phys: projection-start reachable="
                   << reachable.size() << "\n";
    return phase("emit", [&] { return emitProjection(); });
  }

private:
  LogicalResult resolveClosure() {
    root = symbols.lookup(rootName);
    if (!root || !isa<qlx::fabric::ProtocolOp, qlx::fabric::GadgetOp>(root)) {
      module.emitError("fabric-to-phys root @")
          << rootName << " is not a fabric.protocol or fabric.gadget";
      return failure();
    }
    device = dyn_cast_or_null<qlx::DeviceOp>(symbols.lookup(deviceName));
    if (!device || !device.getPhysicalAttr() ||
        !device.getQecToPhysicalAttr()) {
      module.emitError("fabric-to-phys device @")
          << deviceName
          << " must reference physical and qec_to_physical definitions";
      return failure();
    }
    architecture = dyn_cast_or_null<qlx::phys::ArchitectureOp>(
        symbols.lookup(device.getPhysicalAttr().getValue()));
    qecToPhysical = dyn_cast_or_null<qlx::QECToPhysicalBindingOp>(
        symbols.lookup(device.getQecToPhysicalAttr().getValue()));
    if (!architecture || !qecToPhysical ||
        qecToPhysical.getPhysicalAttr() != device.getPhysicalAttr()) {
      device.emitOpError("has an unresolved or inconsistent physical closure");
      return failure();
    }

    for (auto plan : module.getOps<qlx::phys::SpacetimePlanOp>()) {
      if (plan.getProvider() != "qlx.component-model")
        continue;
      if (plan.getArchitectureAttr() != device.getPhysicalAttr() ||
          plan.getOperatingPointAttr() != device.getOperatingPointAttr())
        continue;
      StringRef source = plan.getSourceProtocol();
      auto [entry, inserted] = componentSpacetimePlans.try_emplace(
          source, symbolAttr(context, plan.getSymName()));
      if (!inserted)
        return plan.emitOpError("duplicates a component spacetime model for "
                                "source protocol @")
               << source;
      spacetimePlans[source] = entry->second;
    }

    SymbolTable architectureSymbols(architecture);
    for (auto resource :
         architecture.getBody().getOps<qlx::phys::ResourceClassOp>()) {
      resourceClasses[resource.getSymName()] = resource;
      FlatSymbolRefAttr resourceRef =
          symbolAttr(context, resource.getSymName());
      for (Attribute rawAction : resource.getNativeActions())
        if (auto actionRef = dyn_cast<FlatSymbolRefAttr>(rawAction))
          advertisedNativeActions.insert({resourceRef, actionRef});
    }
    for (auto action : module.getOps<qlx::phys::ActionOp>())
      physicalActions[action.getSymName()] = action;
    for (auto action : architecture.getBody().getOps<qlx::phys::ActionOp>())
      physicalActions[action.getSymName()] = action;
    if (auto operatingPointRef = device.getOperatingPointAttr()) {
      operatingPoint = operatingPointRef;
      auto operatingPoint = dyn_cast_or_null<qlx::phys::OperatingPointOp>(
          symbols.lookup(operatingPointRef.getValue()));
      if (!operatingPoint ||
          operatingPoint.getMachineAttr() != device.getPhysicalAttr()) {
        device.emitOpError("has an inconsistent operating point");
        return failure();
      }
      if (auto timing = operatingPoint.getTimingAttr()) {
        operatingTiming = timing;
        auto cycle = numericValue(timing.get("cycle_ns"));
        if (!cycle)
          cycle = numericValue(timing.get("surface_cycle_ns"));
        if (cycle) {
          cycleNanoseconds = *cycle;
          hasExplicitCycleNanoseconds = true;
        }
        if (auto reaction = numericValue(timing.get("reaction_time_ns"))) {
          reactionNanoseconds = *reaction;
          hasExplicitReactionNanoseconds = true;
        }
      }
    }
    for (Attribute raw : qecToPhysical.getEntries()) {
      auto entry = dyn_cast<DictionaryAttr>(raw);
      auto qec = entry ? entry.getAs<StringAttr>("qec") : StringAttr{};
      auto physicalBinding =
          entry ? entry.getAs<StringAttr>("binding") : StringAttr{};
      auto resources =
          entry ? entry.getAs<ArrayAttr>("resources") : ArrayAttr{};
      if (!qec || !physicalBinding || !resources) {
        qecToPhysical.emitOpError("contains a malformed region binding");
        return failure();
      }
      auto concreteBinding = dyn_cast_or_null<qlx::phys::QECBindingOp>(
          architectureSymbols.lookup(physicalBinding.getValue()));
      if (!concreteBinding) {
        qecToPhysical.emitOpError("region binding '")
            << physicalBinding.getValue()
            << "' does not resolve to phys.qec_binding";
        return failure();
      }
      std::optional<RegionBinding> selected;
      for (Attribute rawResource : resources) {
        auto resourceName = dyn_cast<StringAttr>(rawResource);
        auto resource =
            resourceName
                ? dyn_cast_or_null<qlx::phys::ResourceClassOp>(
                      architectureSymbols.lookup(resourceName.getValue()))
                : qlx::phys::ResourceClassOp{};
        if (!resource)
          continue;
        auto granularityAttr =
            resource->getAttrOfType<StringAttr>("granularity");
        StringRef granularity =
            granularityAttr ? granularityAttr.getValue() : StringRef("carrier");
        bool eligible =
            granularity == "patch" ||
            (granularity == "carrier" && resource.getKind() == "qubit");
        if (!eligible)
          continue;
        if (selected) {
          qecToPhysical.emitOpError(
              "native projection requires exactly one patch or qubit resource "
              "class "
              "per QEC region");
          return failure();
        }
        selected = RegionBinding{
            resource.getSymName().str(),      physicalBinding.getValue().str(),
            resource.getKind().str(),         granularity.str(),
            resource.getCountAttr().getInt(), concreteBinding};
      }
      if (!selected) {
        qecToPhysical.emitOpError("QEC region '")
            << qec.getValue()
            << "' has no patch or physical-qubit resource class";
        return failure();
      }
      regionBindings[qec.getValue()] = *selected;
    }
    return success();
  }

  Region &callableBody(Operation *callable) const {
    if (auto protocol = dyn_cast<qlx::fabric::ProtocolOp>(callable))
      return protocol.getBody();
    return cast<qlx::fabric::GadgetOp>(callable).getBody();
  }

  std::optional<SpacetimeProviderKind>
  spacetimeProviderFor(Operation *callable) const {
    auto protocol = dyn_cast<qlx::fabric::ProtocolOp>(callable);
    if (!protocol)
      return std::nullopt;
    return qlx::spacetime::providerFor(protocol);
  }

  LogicalResult discoverReachable() {
    llvm::StringSet<> active;
    std::function<LogicalResult(Operation *)> visit = [&](Operation *callable) {
      StringRef name = symbolName(callable);
      if (active.contains(name)) {
        callable->emitOpError("recursive Fabric call graph through @") << name;
        return failure();
      }
      if (!reachableNames.insert(name).second)
        return success();
      active.insert(name);
      llvm::scope_exit eraseActive([&] { active.erase(name); });
      reachable.push_back(callable);
      SmallVector<qlx::fabric::CallOp> calls;
      callableBody(callable).walk(
          [&](qlx::fabric::CallOp call) { calls.push_back(call); });
      for (qlx::fabric::CallOp call : calls) {
        Operation *callee = symbols.lookup(call.getCallee());
        if (!callee ||
            !isa<qlx::fabric::ProtocolOp, qlx::fabric::GadgetOp>(callee)) {
          call.emitOpError("references unresolved callable @")
              << call.getCallee();
          return failure();
        }
        if (failed(visit(callee)))
          return failure();
      }
      return success();
    };
    return visit(root);
  }

  LogicalResult validatePhysicalCapabilities() {
    const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
    auto lastCheckpoint = std::chrono::steady_clock::now();
    auto checkpoint = [&](StringRef name) {
      if (!profile)
        return;
      const auto now = std::chrono::steady_clock::now();
      llvm::errs()
          << "fabric-to-phys: capabilities-" << name << ' '
          << std::chrono::duration<double>(now - lastCheckpoint).count()
          << "s\n";
      lastCheckpoint = now;
    };
    CarrierHomeMap homes;
    llvm::StringSet<> usedHomes;
    SmallVector<Value> pendingHomes;
    auto isCarrier = [](Type type) {
      return isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(type);
    };
    auto retainHome = [&](Value value, StringRef region) {
      StringAttr home = stringAttr(context, region);
      CarrierHomes &selected = homes[value];
      if (!llvm::is_contained(selected, home)) {
        selected.push_back(home);
        pendingHomes.push_back(value);
      }
      usedHomes.insert(region);
    };

    SmallVector<Operation *> operations;
    for (auto [index, callable] : llvm::enumerate(reachable)) {
      bool exactPatchMacro = false;
      if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callable);
          gadget && succeeded(patchMacroFor(gadget)))
        exactPatchMacro = true;
      if (!exactPatchMacro)
        callableBody(callable).walk(
            [&](Operation *operation) { operations.push_back(operation); });
      if (profile && (index == 0 || (index + 1) % 50 == 0 ||
                      index + 1 == reachable.size()))
        llvm::errs() << "fabric-to-phys: capabilities-collect-progress "
                     << index + 1 << '/' << reachable.size()
                     << " operations=" << operations.size() << " callable=@"
                     << symbolName(callable) << "\n";
    }
    if (profile)
      llvm::errs() << "fabric-to-phys: capabilities-operations "
                   << operations.size() << "\n";
    checkpoint("collect");
    for (Operation *operation : operations)
      if (auto allocation = dyn_cast<qlx::fabric::AllocOp>(operation))
        retainHome(allocation.getResult(), allocation.getRegion());
    for (BlockArgument argument : callableBody(root).front().getArguments()) {
      auto patch = dyn_cast<qlx::fabric::PatchType>(argument.getType());
      if (!patch)
        continue;
      auto region = rootPatchRegion(patch);
      if (failed(region))
        return root->emitOpError(
            "native projection cannot resolve a root patch's exact QEC home");
      retainHome(argument, *region);
    }
    checkpoint("seed");

    // Build the exact carrier-home dataflow once, then propagate from concrete
    // allocations with a worklist.  This remains linear in the reachable SSA
    // graph even for deep reusable call hierarchies.
    DenseMap<Value, SmallVector<Value, 2>> homeSuccessors;
    auto addFlow = [&](Value source, Value target) {
      SmallVector<Value, 2> &successors = homeSuccessors[source];
      if (!llvm::is_contained(successors, target))
        successors.push_back(target);
    };
    for (Operation *operation : operations) {
      if (auto call = dyn_cast<qlx::fabric::CallOp>(operation)) {
        Operation *callee = symbols.lookup(call.getCallee());
        if (!callee)
          continue;
        bool exactPatchMacro = false;
        if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callee);
            gadget && succeeded(patchMacroFor(gadget)))
          exactPatchMacro = true;
        if (exactPatchMacro) {
          SmallVector<Value> carrierOperands;
          SmallVector<Value> carrierResults;
          for (Value operand : call.getOperands())
            if (isCarrier(operand.getType()))
              carrierOperands.push_back(operand);
          for (Value result : call.getResults())
            if (isCarrier(result.getType()))
              carrierResults.push_back(result);
          if (carrierOperands.size() == carrierResults.size())
            for (auto [operand, result] :
                 llvm::zip(carrierOperands, carrierResults))
              addFlow(operand, result);
          else
            for (Value result : carrierResults)
              for (Value operand : carrierOperands)
                addFlow(operand, result);
          continue;
        }
        Block &calleeBlock = callableBody(callee).front();
        for (auto [operand, argument] :
             llvm::zip(call.getOperands(), calleeBlock.getArguments()))
          if (isCarrier(argument.getType()))
            addFlow(operand, argument);
        for (auto [result, yielded] : llvm::zip(
                 call.getResults(), calleeBlock.getTerminator()->getOperands()))
          if (isCarrier(result.getType()))
            addFlow(yielded, result);
        continue;
      }
      if (auto unpack = dyn_cast<qlx::fabric::UnpackResourceOp>(operation)) {
        size_t count = unpack.getAnchors().size();
        if (unpack.getOutputs().size() != count * 2)
          continue;
        for (size_t index = 0; index < count; ++index) {
          addFlow(unpack.getAnchors()[index], unpack.getOutputs()[index]);
          addFlow(unpack.getAnchors()[index],
                  unpack.getOutputs()[count + index]);
        }
        continue;
      }
      if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
        Block &body = repeat.getBody().front();
        for (auto [init, argument] :
             llvm::zip(repeat.getInits(), body.getArguments()))
          if (isCarrier(argument.getType()))
            addFlow(init, argument);
        for (auto [argument, yielded, result] :
             llvm::zip(body.getArguments(), body.getTerminator()->getOperands(),
                       repeat.getResults())) {
          if (!isCarrier(result.getType()))
            continue;
          addFlow(yielded, argument);
          addFlow(yielded, result);
        }
        continue;
      }
      if (auto conditional = dyn_cast<qlx::cflow::IfOp>(operation)) {
        for (Region *region :
             {&conditional.getThenRegion(), &conditional.getElseRegion()})
          for (auto [result, yielded] :
               llvm::zip(conditional.getResults(),
                         region->front().getTerminator()->getOperands()))
            if (isCarrier(result.getType()))
              addFlow(yielded, result);
        continue;
      }
      if (auto retry = dyn_cast<qlx::fabric::RetryOp>(operation)) {
        for (auto [carry, result] :
             llvm::zip(retry.getCarries(), retry.getResults()))
          if (isCarrier(result.getType()))
            addFlow(carry, result);
        continue;
      }

      SmallVector<Value> carrierOperands;
      SmallVector<Value> carrierResults;
      for (Value operand : operation->getOperands())
        if (isCarrier(operand.getType()))
          carrierOperands.push_back(operand);
      for (Value result : operation->getResults())
        if (isCarrier(result.getType()))
          carrierResults.push_back(result);
      if (carrierOperands.size() == carrierResults.size()) {
        for (auto [result, operand] :
             llvm::zip(carrierResults, carrierOperands))
          addFlow(operand, result);
      } else {
        // This is outside the native subset's usual one-owner-per-result
        // boundary. Retaining every possible source remains fail-closed.
        for (Value result : carrierResults)
          for (Value operand : carrierOperands)
            addFlow(operand, result);
      }
    }
    if (profile)
      llvm::errs() << "fabric-to-phys: capabilities-flow-nodes "
                   << homeSuccessors.size() << "\n";
    checkpoint("flow");
    // Propagation inserts one entry for each reached SSA value. Reserve the
    // complete known flow domain up front, and copy the source's tiny home set
    // before inserting a target. DenseMap insertion may otherwise rehash and
    // invalidate selectedHomes while it is still being iterated.
    homes.reserve(homes.size() + homeSuccessors.size());
    for (size_t cursor = 0; cursor < pendingHomes.size(); ++cursor) {
      Value source = pendingHomes[cursor];
      auto selectedHomes = homes.find(source);
      auto successors = homeSuccessors.find(source);
      if (selectedHomes == homes.end() || successors == homeSuccessors.end())
        continue;
      CarrierHomes sourceHomes = selectedHomes->second;
      for (Value target : successors->second) {
        CarrierHomes &targetHomes = homes[target];
        for (StringAttr home : sourceHomes)
          if (!llvm::is_contained(targetHomes, home)) {
            targetHomes.push_back(home);
            pendingHomes.push_back(target);
          }
      }
    }
    if (profile)
      llvm::errs() << "fabric-to-phys: capabilities-home-values "
                   << homes.size() << " queue=" << pendingHomes.size() << "\n";
    checkpoint("propagate");

    auto bindingFor = [&](StringAttr home) -> RegionBinding * {
      auto found = regionBindings.find(home.getValue());
      return found == regionBindings.end() ? nullptr : &found->second;
    };
    auto homesFor = [&](Operation *operation,
                        Value value) -> FailureOr<ArrayRef<StringAttr>> {
      auto found = homes.find(value);
      if (found == homes.end() || found->second.empty()) {
        auto diagnostic = operation->emitOpError(
            "native projection cannot resolve the exact physical home of an "
            "action operand");
        for (Operation *parent = operation->getParentOp(); parent;
             parent = parent->getParentOp()) {
          if (!isa<qlx::fabric::ProtocolOp, qlx::fabric::GadgetOp>(parent))
            continue;
          diagnostic << " in callable @" << symbolName(parent);
          break;
        }
        return failure();
      }
      return ArrayRef<StringAttr>(found->second);
    };
    auto validateAction = [&](Operation *operation, Value value,
                              StringRef action) -> LogicalResult {
      auto selectedHomes = homesFor(operation, value);
      if (failed(selectedHomes))
        return failure();
      for (StringAttr home : *selectedHomes) {
        RegionBinding *binding = bindingFor(home);
        if (!binding) {
          operation->emitOpError("has no physical binding for QEC region @")
              << home.getValue();
          return failure();
        }
        if (!resourceAdvertises(binding->resourceClass, action)) {
          operation->emitOpError("physical resource class @")
              << binding->resourceClass << " does not advertise native action @"
              << action;
          return failure();
        }
      }
      return success();
    };
    auto validateProductRotation = [&](Operation *operation,
                                       Value value) -> LogicalResult {
      auto selectedHomes = homesFor(operation, value);
      if (failed(selectedHomes))
        return failure();
      for (StringAttr home : *selectedHomes) {
        RegionBinding *binding = bindingFor(home);
        if (!binding) {
          operation->emitOpError("has no physical binding for QEC region @")
              << home.getValue();
          return failure();
        }
        if (!resourceClassHasCapability(binding->resourceClass,
                                        kNativePauliProductRotation) &&
            !resourceAdvertises(binding->resourceClass, "rpp")) {
          operation->emitOpError("physical resource class @")
              << binding->resourceClass
              << " lacks typed native Pauli-product-rotation capability '"
              << kNativePauliProductRotation << "'";
          return failure();
        }
      }
      return success();
    };
    auto rejectTopologySensitive = [&](Operation *operation,
                                       Value value) -> LogicalResult {
      auto selectedHomes = homesFor(operation, value);
      if (failed(selectedHomes))
        return failure();
      for (StringAttr home : *selectedHomes) {
        RegionBinding *binding = bindingFor(home);
        if (binding && binding->binding.getTopologyAttr()) {
          operation->emitOpError(
              "native projection does not yet map or route topology-sensitive "
              "physical actions; use the reference physical projector");
          return failure();
        }
      }
      return success();
    };

    // Compact factory projection carries one binding, not a pooled engine
    // instance. Reject ambiguous homes during the read-only preflight rather
    // than discovering them after graph construction has started.
    for (Operation *operation : operations) {
      auto request = dyn_cast<qlx::fabric::ResourceRequestOp>(operation);
      if (!request)
        continue;
      auto stream = dyn_cast_or_null<qlx::lvm::StreamOp>(
          SymbolTable::lookupNearestSymbolFrom(request,
                                               request.getStreamAttr()));
      if (!stream || !stream.getProducedByAttr() || stream.getExternalAttr())
        continue;
      auto provider = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
          symbols.lookup(stream.getProducedByAttr().getValue()));
      FlatSymbolRefAttr region = stream.getBackingRegionAttr();
      DictionaryAttr metadata =
          provider ? provider->getAttrOfType<DictionaryAttr>("metadata")
                   : DictionaryAttr{};
      auto factoryMode =
          metadata ? metadata.getAs<StringAttr>("factory_mode") : StringAttr{};
      auto footprintText = metadata
                               ? metadata.getAs<StringAttr>("physical_qubits")
                               : StringAttr{};
      if (!region || !factoryMode ||
          factoryMode.getValue() != "scheduled_macro" || !footprintText)
        continue;
      int64_t footprint = 0;
      if (footprintText.getValue().getAsInteger(10, footprint) ||
          footprint <= 0)
        continue;
      auto binding = regionBindings.find(region.getValue());
      if (binding != regionBindings.end() &&
          binding->second.capacity != footprint)
        return request.emitOpError(
                   "scheduled resource provider physical binding has ")
               << binding->second.capacity << " qubits but requires "
               << footprint
               << " qubits (exactly one engine footprint); pooled or "
                  "multi-engine homes are unsupported";
    }
    checkpoint("factory");

    // A selected patch topology fixes slot-to-carrier groups.  The native
    // allocator currently allocates from a homogeneous pool and cannot emit
    // that exact mapping, even for an otherwise unary-only workload.
    for (const auto &home : usedHomes) {
      auto found = regionBindings.find(home.getKey());
      if (found != regionBindings.end() &&
          found->second.binding.getPatchTopologyAttr()) {
        found->second.binding.emitOpError(
            "native projection does not yet implement selected patch-topology "
            "mapping; use the reference physical projector");
        return failure();
      }
    }
    checkpoint("topology");

    static const llvm::StringMap<StringRef> unaryActions{
        {"fabric.h", "h"},     {"fabric.s", "s"}, {"fabric.sdg", "sdg"},
        {"fabric.x", "x"},     {"fabric.z", "z"}, {"fabric.t", "t"},
        {"fabric.tdg", "tdg"},
    };
    for (Operation *operation : operations) {
      StringRef name = operation->getName().getStringRef();
      if (auto found = unaryActions.find(name); found != unaryActions.end()) {
        if (failed(validateAction(operation, operation->getOperand(0),
                                  found->second)))
          return failure();
        continue;
      }
      StringRef action;
      if (isa<qlx::fabric::CXOp>(operation))
        action = "cx";
      else if (isa<qlx::fabric::CZOp>(operation))
        action = "cz";
      if (!action.empty()) {
        for (Value operand : operation->getOperands())
          if (isCarrier(operand.getType()) &&
              (failed(validateAction(operation, operand, action)) ||
               failed(rejectTopologySensitive(operation, operand))))
            return failure();
        continue;
      }
      if (isa<qlx::fabric::RotateProductOp>(operation)) {
        for (Value operand : operation->getOperands())
          if (isCarrier(operand.getType()) &&
              (failed(validateProductRotation(operation, operand)) ||
               failed(rejectTopologySensitive(operation, operand))))
            return failure();
        continue;
      }
      if (isa<qlx::fabric::ResourceRotateProductOp>(operation))
        for (Value operand : operation->getOperands())
          if (isCarrier(operand.getType()) &&
              failed(rejectTopologySensitive(operation, operand)))
            return failure();
    }
    checkpoint("actions");
    return success();
  }

  std::string validateBlock(Block &block) const {
    static const llvm::StringSet<> unaryActions{
        "fabric.h", "fabric.s", "fabric.sdg", "fabric.x",
        "fabric.z", "fabric.t", "fabric.tdg"};
    for (BlockArgument argument : block.getArguments())
      if (!isSupportedProjectedType(argument.getType()))
        return "native v1 has no physical type projection for a block argument";
    for (Operation &operation : block) {
      if (operation.getName().getStringRef() == "arith.constant") {
        if (operation.getNumOperands() != 0 || operation.getNumResults() != 1 ||
            failed(physicalScalarType(operation.getResult(0).getType())))
          return "native v1 requires scalar arith.constant values";
        continue;
      }
      if (auto allocation = dyn_cast<qlx::fabric::AllocOp>(operation)) {
        if (allocation.getPrepAttr())
          return "native projection rejects the legacy fabric.alloc prep "
                 "attribute; use a selected preparation realization";
        continue;
      }
      if (isa<qlx::fabric::CXOp, qlx::fabric::CZOp>(operation)) {
        if (!operation.getAttrOfType<StringAttr>("pairs"))
          return "native v1 requires explicit CX/CZ interaction pairs";
        continue;
      }
      if (isa<qlx::fabric::AllocOp, qlx::fabric::PrepZOp, qlx::fabric::PrepXOp,
              qlx::fabric::DeallocOp, qlx::fabric::CallOp,
              qlx::fabric::ReturnOp, qlx::fabric::ProtocolReturnOp,
              qlx::cflow::YieldOp, qlx::fabric::InitBasisOp,
              qlx::fabric::ResetOp, qlx::fabric::IdleOp, qlx::fabric::TickOp,
              qlx::fabric::TransformBeginOp, qlx::fabric::TransformEndOp,
              qlx::fabric::MzOp, qlx::fabric::ParityOp, qlx::fabric::AllZeroOp,
              qlx::fabric::XorOp, qlx::fabric::AllFalseOp,
              qlx::event::SelectionOp, qlx::fabric::ProduceResourceOp,
              qlx::fabric::ResourceRequestOp, qlx::event::AwaitOp,
              qlx::fabric::UnpackResourceOp, qlx::fabric::PackResourceOp,
              qlx::fabric::DiscardResourceOp, qlx::fabric::MeasureProductOp,
              qlx::fabric::RotateProductOp,
              qlx::fabric::ResourceRotateProductOp, qlx::fabric::RetryOp,
              qlx::fabric::SuccessOp>(operation))
        continue;
      if (unaryActions.contains(operation.getName().getStringRef()))
        continue;
      if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
        for (Type type : repeat.getInits().getTypes())
          if (!isSupportedProjectedType(type))
            return "native v1 has no physical repeat carry projection";
        std::string nested = validateBlock(repeat.getBody().front());
        if (!nested.empty())
          return nested;
        continue;
      }
      if (auto conditional = dyn_cast<qlx::cflow::IfOp>(operation)) {
        for (Region *region :
             {&conditional.getThenRegion(), &conditional.getElseRegion()}) {
          std::string nested = validateBlock(region->front());
          if (!nested.empty())
            return nested;
        }
        continue;
      }
      return ("native v1 does not project " +
              operation.getName().getStringRef())
          .str();
    }
    Operation *terminator = block.getTerminator();
    for (Value value : terminator->getOperands())
      if (!isSupportedProjectedType(value.getType()))
        return "native v1 has no physical type projection for a block result";
    return {};
  }

  std::string validateCallable(Operation *callable) const {
    if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callable)) {
      if (gadget.getRealizationAttr())
        return "native v1 does not yet project a detached fabric.circuit "
               "realization body";
      // A patch macro's selected physical representation is a property of its
      // concrete operands, not of the encoded patch type. The same code may be
      // bound to patch-granular compute regions and a carrier-granular factory
      // region. Emission validates the actual operand granularity and either
      // emits one authenticated patch event or expands the retained body.
      if (succeeded(patchMacroFor(gadget)))
        return {};
    }
    if (auto metadata = callable->getAttrOfType<DictionaryAttr>("metadata")) {
      if (Attribute rawReplay = metadata.get("runtime_replay")) {
        auto replay = dyn_cast<StringAttr>(rawReplay);
        if (!replay)
          return "native projection requires metadata.runtime_replay to be a "
                 "string";
        if (replay.getValue() == "unsupported")
          return "native projection rejects metadata.runtime_replay = "
                 "\"unsupported\": P3 attempt replay is not implemented";
        if (replay.getValue() != "bounded_p3_retry")
          return ("native projection does not recognize "
                  "metadata.runtime_replay = \"" +
                  replay.getValue() + "\"; expected \"bounded_p3_retry\"")
              .str();
      }
    }
    return validateBlock(callableBody(callable).front());
  }

  LogicalResult preflight() {
    SmallVector<std::string> errors(reachable.size());
    auto validate = [&](size_t index) {
      errors[index] = validateCallable(reachable[index]);
      return success();
    };
    if (useParallelPlanning) {
      if (failed(
              failableParallelForEachN(context, 0, reachable.size(), validate)))
        return failure();
    } else {
      for (size_t index = 0; index < reachable.size(); ++index)
        (void)validate(index);
    }
    for (auto [index, error] : llvm::enumerate(errors)) {
      if (error.empty())
        continue;
      reachable[index]->emitOpError(error);
      return failure();
    }
    return success();
  }

  FailureOr<DictionaryAttr> patchPartitions(Type type) const {
    if (auto patch = dyn_cast<qlx::fabric::PatchType>(type)) {
      auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
          symbols.lookup(patch.getCodeType().getValue()));
      if (!code)
        return failure();
      return code.getPartitions();
    }
    if (auto frame = dyn_cast<qlx::fabric::PatchFrameType>(type)) {
      auto transform = dyn_cast_or_null<qlx::fabric::PatchTransformOp>(
          symbols.lookup(frame.getTransform().getValue()));
      if (!transform)
        return failure();
      return transform.getFramePartitions();
    }
    return failure();
  }

  FailureOr<int64_t> patchWidth(Type type) const {
    auto partitions = patchPartitions(type);
    if (failed(partitions))
      return failure();
    int64_t width = 0;
    for (NamedAttribute partition : *partitions) {
      auto count = dyn_cast<IntegerAttr>(partition.getValue());
      if (!count || count.getInt() < 0)
        return failure();
      width += count.getInt();
    }
    return width;
  }

  FailureOr<int64_t> regionCodeDistance(StringRef region,
                                        Location location) const {
    auto qecRef = device->getAttrOfType<FlatSymbolRefAttr>("qec");
    auto machine = dyn_cast_or_null<qlx::fabric::DeviceOp>(
        qecRef ? symbols.lookup(qecRef.getValue()) : nullptr);
    if (!machine) {
      emitError(location) << "selected device has no resolvable QEC machine";
      return failure();
    }
    qlx::fabric::RegionOp selected;
    for (auto candidate : machine.getBody().getOps<qlx::fabric::RegionOp>())
      if (candidate.getSymName() == region) {
        if (selected) {
          emitError(location) << "QEC region @" << region
                              << " is ambiguous while deriving code distance";
          return failure();
        }
        selected = candidate;
      }
    auto code = selected ? dyn_cast_or_null<qlx::fabric::CodeOp>(symbols.lookup(
                               selected.getCodeAttr().getValue()))
                         : qlx::fabric::CodeOp{};
    if (!selected || !code || code.getDistance() <= 0) {
      emitError(location) << "QEC region @" << region
                          << " must resolve to a positive code distance";
      return failure();
    }
    return code.getDistance();
  }

  FailureOr<StringRef>
  patchResourceGranularity(qlx::fabric::PatchType patch) const {
    auto qecRef = device->getAttrOfType<FlatSymbolRefAttr>("qec");
    auto machine = dyn_cast_or_null<qlx::fabric::DeviceOp>(
        qecRef ? symbols.lookup(qecRef.getValue()) : nullptr);
    if (!qecRef || !machine)
      return failure();
    std::optional<StringRef> selected;
    for (auto region : machine.getBody().getOps<qlx::fabric::RegionOp>()) {
      auto binding = regionBindings.find(region.getSymName());
      if (binding == regionBindings.end() ||
          region.getCodeAttr() != patch.getCodeType())
        continue;
      if (patch.getEncoding() && region.getEncodingAttr() &&
          patch.getEncoding() != region.getEncodingAttr())
        continue;
      StringRef candidate = binding->second.granularity;
      if (selected && *selected != candidate)
        return failure();
      selected = candidate;
    }
    if (!selected)
      return failure();
    return *selected;
  }

  FailureOr<SmallVector<std::pair<std::string, int64_t>>>
  orderedPartitionSizes(Type type) const {
    auto partitions = patchPartitions(type);
    if (failed(partitions))
      return failure();
    std::map<std::string, int64_t> counts;
    for (NamedAttribute partition : *partitions) {
      auto count = dyn_cast<IntegerAttr>(partition.getValue());
      if (!count || count.getInt() < 0)
        return failure();
      counts[partition.getName().strref().str()] = count.getInt();
    }
    SmallVector<std::pair<std::string, int64_t>> ordered;
    for (StringRef name :
         {StringRef("data"), StringRef("sx"), StringRef("sz")}) {
      auto found = counts.find(name.str());
      if (found == counts.end())
        continue;
      ordered.push_back(*found);
      counts.erase(found);
    }
    llvm::append_range(ordered, counts);
    return ordered;
  }

  FailureOr<FlatSymbolRefAttr>
  resourceKind(qlx::fabric::ResourceStateType resource) const {
    auto kind = dyn_cast<FlatSymbolRefAttr>(resource.getKind());
    if (!kind)
      return failure();
    return kind;
  }

  FailureOr<Type> physicalScalarType(Type type) const {
    if (auto resource = dyn_cast<qlx::fabric::ResourceStateType>(type)) {
      auto kind = resourceKind(resource);
      if (failed(kind))
        return failure();
      return Type(qlx::phys::ResourcePayloadType::get(context, *kind));
    }
    if (auto event = dyn_cast<qlx::event::HandleType>(type)) {
      auto payload = physicalScalarType(event.getPayload());
      if (failed(payload))
        return failure();
      return Type(qlx::event::HandleType::get(
          context, *payload, event.getOwnership(), event.getStream()));
    }
    if (isa<IntegerType, IndexType, FloatType>(type))
      return type;
    return failure();
  }

  FailureOr<int64_t> projectedWidth(Type type) const {
    if (auto patch = dyn_cast<qlx::fabric::PatchType>(type)) {
      auto granularity = patchResourceGranularity(patch);
      if (failed(granularity))
        return failure();
      if (*granularity == "patch")
        return 1;
      return patchWidth(type);
    }
    if (isa<qlx::fabric::PatchFrameType>(type))
      return patchWidth(type);
    if (auto tensor = dyn_cast<RankedTensorType>(type)) {
      if (!tensor.hasStaticShape() || !tensor.getElementType().isInteger(1))
        return failure();
      return tensor.getNumElements();
    }
    if (succeeded(physicalScalarType(type)))
      return 1;
    return failure();
  }

  bool isSupportedProjectedType(Type type) const {
    if (isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(type)) {
      auto width = patchWidth(type);
      return succeeded(width) && *width > 0;
    }
    auto width = projectedWidth(type);
    return succeeded(width) && *width >= 0;
  }

  FailureOr<StringRef> projectedRegion(const PhysicalGroup &values) const {
    std::optional<StringRef> selected;
    for (Value value : values.values()) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      auto found = state ? resourceRegions.find(state.getResource().getValue())
                         : resourceRegions.end();
      if (found == resourceRegions.end())
        return failure();
      StringRef region = found->second;
      if (selected && *selected != region)
        return failure();
      selected = region;
    }
    if (!selected)
      return failure();
    return *selected;
  }

  FailureOr<PhysicalGroup> allocatePatch(Type patchType, StringRef region,
                                         Location location,
                                         OpBuilder &builder) {
    auto bindingIt = regionBindings.find(region);
    if (bindingIt == regionBindings.end()) {
      emitError(location) << "has no physical binding for QEC region @"
                          << region;
      return failure();
    }
    FailureOr<int64_t> carrierWidth = patchWidth(patchType);
    if (failed(carrierWidth)) {
      emitError(location) << "has an unresolved or malformed patch code";
      return failure();
    }
    FailureOr<int64_t> codeDistance = regionCodeDistance(region, location);
    if (failed(codeDistance))
      return failure();
    RegionBinding &binding = bindingIt->second;
    int64_t width = binding.granularity == "patch" ? 1 : *carrierWidth;
    std::set<int64_t> &active = activeIndices[binding.resourceClass];
    std::set<int64_t> &ever = everAllocatedIndices[binding.resourceClass];
    SmallVector<int64_t> selected;
    for (int64_t index = 0;
         index < binding.capacity && selected.size() < size_t(width); ++index)
      if (!active.count(index) && !ever.count(index))
        selected.push_back(index);
    for (int64_t index = 0;
         index < binding.capacity && selected.size() < size_t(width); ++index)
      if (!active.count(index) && !llvm::is_contained(selected, index))
        selected.push_back(index);
    if (selected.size() != size_t(width)) {
      emitError(location) << "physical resource class @"
                          << binding.resourceClass << " capacity "
                          << binding.capacity
                          << " cannot satisfy an allocation that needs "
                          << width << " carriers";
      return failure();
    }

    SmallVector<Attribute> references;
    SmallVector<Type> resultTypes;
    for (int64_t index : selected) {
      active.insert(index);
      ever.insert(index);
      std::string resourceName =
          llvm::formatv("{0}_{1}_alloc{2}", binding.resourceClass, index,
                        resourceIdentity++)
              .str();
      FlatSymbolRefAttr reference = symbolAttr(context, resourceName);
      if (!resources.count(reference)) {
        OpBuilder topBuilder(module.getBodyRegion());
        topBuilder.setInsertionPoint(graph);
        SmallVector<NamedAttribute> resourceAttrs{
            topBuilder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                    stringAttr(context, resourceName)),
            topBuilder.getNamedAttr("kind", stringAttr(context, binding.kind)),
            topBuilder.getNamedAttr(
                "architecture", symbolAttr(context, architecture.getSymName())),
            topBuilder.getNamedAttr("resource_class",
                                    symbolAttr(context, binding.resourceClass)),
            topBuilder.getNamedAttr("index",
                                    topBuilder.getI64IntegerAttr(index)),
            topBuilder.getNamedAttr("qec_region",
                                    binding.binding.getQecRegionAttr()),
            topBuilder.getNamedAttr(
                "code_distance", topBuilder.getI64IntegerAttr(*codeDistance)),
        };
        auto resourceClass = resourceClasses.find(binding.resourceClass);
        if (resourceClass != resourceClasses.end())
          if (auto capabilities = resourceClass->second.getCapabilities())
            resourceAttrs.push_back(
                topBuilder.getNamedAttr("capabilities", *capabilities));
        createGeneric(topBuilder, location, "phys.resource", {}, {},
                      resourceAttrs);
        resources[reference] = ResourceInfo{binding.resourceClass, index};
      }
      resourceByIdentity[{binding.resourceClass, index}] = reference;
      references.push_back(reference);
      resourceRegions[reference.getValue()] = region.str();
      resultTypes.push_back(qlx::phys::StateType::get(context, reference));
    }
    StringAttr acquireEvent = nextEvent("acquire");
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("resources", builder.getArrayAttr(references)),
        builder.getNamedAttr("event_id", acquireEvent),
    };
    Operation *acquire = createGeneric(builder, location, "phys.acquire", {},
                                       resultTypes, attrs);
    AllocationRecord record;
    record.name = llvm::formatv("alloc{0}", allocationIdentity++).str();
    record.resourceClass = binding.resourceClass;
    record.indices = selected;
    record.acquire = acquireEvent.getValue().str();
    for (int64_t index : selected) {
      auto previous = lastReleaseByBinding.find(
          std::pair<std::string, int64_t>{binding.resourceClass, index});
      if (previous != lastReleaseByBinding.end() &&
          !llvm::is_contained(record.after, previous->second))
        record.after.push_back(previous->second);
    }
    record.resources = references;
    for (Attribute resource : record.resources)
      resourceToAllocation[cast<FlatSymbolRefAttr>(resource).getValue()] =
          allocations.size();
    allocations.push_back(std::move(record));
    allocationPlanTrace.push_back(AllocationPlanStep{
        true, binding.resourceClass, binding.capacity, selected});
    PhysicalGroup values(acquire->getResults().begin(),
                         acquire->getResults().end());
    return values;
  }

  FailureOr<PhysicalGroup> allocatePatch(qlx::fabric::AllocOp allocation,
                                         OpBuilder &builder) {
    if (allocation.getPrepAttr()) {
      allocation.emitOpError(
          "native projection requires preparation as a selected P2 "
          "realization, not the legacy alloc prep attribute");
      return failure();
    }
    return allocatePatch(allocation.getResult().getType(),
                         allocation.getRegion(), allocation.getLoc(), builder);
  }

  FailureOr<qlx::phys::FactoryModelOp>
  factoryModelFor(qlx::fabric::ResourceRequestOp request) {
    qlx::phys::FactoryModelOp selected;
    for (auto candidate : module.getOps<qlx::phys::FactoryModelOp>()) {
      if (candidate.getStreamAttr() != request.getStreamAttr() ||
          candidate.getResourceKindAttr().getValue() != request.getKind())
        continue;
      if (selected) {
        request.emitOpError(
            "matches several deterministic physical factory models");
        return failure();
      }
      selected = candidate;
    }
    return selected;
  }

  FailureOr<SmallVector<Operation *>> callableClosure(Operation *source) const {
    SmallVector<Operation *> closure;
    llvm::StringSet<> visited;
    llvm::StringSet<> active;
    std::function<LogicalResult(Operation *)> visit =
        [&](Operation *callable) -> LogicalResult {
      StringRef name = symbolName(callable);
      if (name.empty() || active.contains(name))
        return callable->emitOpError(
            "cannot derive a spacetime plan from a recursive or unnamed "
            "Fabric callable closure");
      if (!visited.insert(name).second)
        return success();
      active.insert(name);
      llvm::scope_exit eraseActive([&] { active.erase(name); });
      closure.push_back(callable);
      SmallVector<qlx::fabric::CallOp> calls;
      callableBody(callable).walk(
          [&](qlx::fabric::CallOp call) { calls.push_back(call); });
      for (qlx::fabric::CallOp call : calls) {
        Operation *callee = symbols.lookup(call.getCallee());
        if (!callee ||
            !isa<qlx::fabric::ProtocolOp, qlx::fabric::GadgetOp>(callee))
          return call.emitOpError(
              "has no resolved callable for spacetime-plan derivation");
        if (failed(visit(callee)))
          return failure();
      }
      return success();
    };
    if (failed(visit(source)))
      return failure();
    return closure;
  }

  FailureOr<int64_t> closureCodeDistance(ArrayRef<Operation *> closure) const {
    std::optional<int64_t> selected;
    auto inspect = [&](Type type, Operation *owner) -> LogicalResult {
      auto patch = dyn_cast<qlx::fabric::PatchType>(type);
      if (!patch)
        return success();
      auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
          symbols.lookup(patch.getCodeType().getValue()));
      if (!code || code.getDistance() <= 0)
        return owner->emitOpError(
            "spacetime-plan patch code must resolve with positive distance");
      if (selected && *selected != code.getDistance())
        return owner->emitOpError(
            "spacetime plan requires one selected code distance across its "
            "callable closure");
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
      WalkResult result =
          callableBody(callable).walk([&](Operation *operation) {
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
    if (!selected) {
      closure.front()->emitOpError(
          "spacetime-plan callable closure contains no encoded patch type");
      return failure();
    }
    return *selected;
  }

  FailureOr<SmallVector<int64_t>>
  closureCodeDistances(ArrayRef<Operation *> closure) const {
    std::set<int64_t> selected;
    auto inspect = [&](Type type, Operation *owner) -> LogicalResult {
      auto patch = dyn_cast<qlx::fabric::PatchType>(type);
      if (!patch)
        return success();
      auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
          symbols.lookup(patch.getCodeType().getValue()));
      if (!code || code.getDistance() <= 0)
        return owner->emitOpError(
            "factory-model patch code must resolve with positive distance");
      selected.insert(code.getDistance());
      return success();
    };
    for (Operation *callable : closure) {
      for (Type type : callable->getOperandTypes())
        if (failed(inspect(type, callable)))
          return failure();
      for (Type type : callable->getResultTypes())
        if (failed(inspect(type, callable)))
          return failure();
      WalkResult result =
          callableBody(callable).walk([&](Operation *operation) {
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
    if (selected.empty()) {
      closure.front()->emitOpError(
          "factory-model closure contains no encoded patch type");
      return failure();
    }
    return SmallVector<int64_t>(selected.begin(), selected.end());
  }

  FailureOr<SmallVector<std::string>>
  closureWorkspaceClasses(ArrayRef<Operation *> closure,
                          ValueRange physicalInputs) const {
    llvm::StringSet<> selected;
    for (Value value : physicalInputs) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      auto resource =
          state ? resources.find(state.getResource()) : resources.end();
      if (state && resource == resources.end()) {
        closure.front()->emitOpError(
            "spacetime-plan input uses an unresolved physical resource");
        return failure();
      }
      if (resource != resources.end())
        selected.insert(resource->second.resourceClass);
    }
    for (Operation *callable : closure)
      callableBody(callable).walk([&](qlx::fabric::AllocOp allocation) {
        auto binding = regionBindings.find(allocation.getRegion());
        if (binding != regionBindings.end())
          selected.insert(binding->second.resourceClass);
      });
    SmallVector<std::string> result;
    for (const auto &entry : selected)
      result.push_back(entry.getKey().str());
    llvm::sort(result);
    if (result.empty()) {
      closure.front()->emitOpError(
          "spacetime-plan closure has no selected workspace resource class");
      return failure();
    }
    return result;
  }

  FailureOr<qlx::phys::FactoryModelOp>
  closureFactoryModel(ArrayRef<Operation *> closure) {
    qlx::phys::FactoryModelOp selected;
    for (Operation *callable : closure) {
      bool failedLookup = false;
      callableBody(callable).walk([&](qlx::fabric::ResourceRequestOp request) {
        auto model = factoryModelFor(request);
        if (failed(model) || !*model) {
          if (succeeded(model))
            request.emitOpError(
                "spacetime derivation request has no selected physical "
                "factory model");
          failedLookup = true;
          return WalkResult::interrupt();
        }
        if (selected && selected != *model) {
          request.emitOpError(
              "spacetime derivation uses several physical factory models");
          failedLookup = true;
          return WalkResult::interrupt();
        }
        selected = *model;
        return WalkResult::advance();
      });
      if (failedLookup)
        return failure();
    }
    if (!selected) {
      closure.front()->emitOpError(
          "spacetime derivation has no authenticated factory demand");
      return failure();
    }
    return selected;
  }

  FailureOr<SmallVector<qlx::phys::FactoryModelOp>> projectionFactoryModels() {
    SmallVector<qlx::phys::FactoryModelOp> models;
    auto retain = [&](qlx::phys::FactoryModelOp model) {
      if (model && !llvm::is_contained(models, model))
        models.push_back(model);
    };
    for (Operation *callable : reachable) {
      bool failedLookup = false;
      callable->walk([&](qlx::fabric::ResourceRequestOp request) {
        auto model = factoryModelFor(request);
        if (failed(model)) {
          failedLookup = true;
          return WalkResult::interrupt();
        }
        retain(*model);
        return WalkResult::advance();
      });
      if (failedLookup)
        return failure();
    }
    return models;
  }

  FailureOr<SurfaceFactoryRecurrenceLayout>
  surfaceFactoryRecurrenceLayout(qlx::fabric::ProtocolOp protocol,
                                 ArrayRef<Operation *> closure) {
    std::map<int64_t, std::string> classByDistance;
    std::map<int64_t, int64_t> capacityByDistance;
    bool failedAllocation = false;
    for (Operation *callable : closure) {
      callableBody(callable).walk([&](qlx::fabric::AllocOp allocation) {
        auto code = dyn_cast_or_null<qlx::fabric::CodeOp>(
            symbols.lookup(allocation.getCodeAttr().getValue()));
        auto binding = regionBindings.find(allocation.getRegion());
        if (!code || code.getDistance() <= 0 ||
            binding == regionBindings.end()) {
          allocation.emitOpError(
              "surface factory recurrence requires every allocation to "
              "resolve to a positive-distance code and physical binding");
          failedAllocation = true;
          return WalkResult::interrupt();
        }
        auto [entry, inserted] = classByDistance.try_emplace(
            code.getDistance(), binding->second.resourceClass);
        if (!inserted && entry->second != binding->second.resourceClass) {
          allocation.emitOpError(
              "surface factory recurrence requires one physical resource "
              "class per code distance");
          failedAllocation = true;
          return WalkResult::interrupt();
        }
        auto qecReference = binding->second.binding.getQecRegionAttr();
        auto qecMachine = dyn_cast_or_null<qlx::fabric::DeviceOp>(
            symbols.lookup(qecReference.getRootReference().getValue()));
        auto region = qecMachine
                          ? SymbolTable(qecMachine)
                                .lookup<qlx::fabric::RegionOp>(
                                    qecReference.getLeafReference().getValue())
                          : qlx::fabric::RegionOp{};
        auto capacity = region ? region.getBlockCapacity() : std::nullopt;
        if (!region || !capacity || *capacity <= 0) {
          allocation.emitOpError(
              "surface factory recurrence requires positive QEC-region "
              "block capacities for every selected code distance");
          failedAllocation = true;
          return WalkResult::interrupt();
        }
        auto [capacityEntry, capacityInserted] =
            capacityByDistance.try_emplace(code.getDistance(), *capacity);
        if (!capacityInserted && capacityEntry->second != *capacity) {
          allocation.emitOpError(
              "surface factory recurrence has inconsistent QEC-region "
              "capacity for one code distance");
          failedAllocation = true;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (failedAllocation)
        return failure();
    }
    if (classByDistance.size() != 2)
      return protocol.emitOpError(
          "surface factory recurrence requires exactly two selected code "
          "distances and physical patch layout");

    auto first = classByDistance.begin();
    auto second = std::next(first);
    int64_t level1Distance = first->first;
    int64_t level2Distance = second->first;
    auto level1 = resourceClasses.find(first->second);
    auto level2 = resourceClasses.find(second->second);
    auto validPatchClass = [](qlx::phys::ResourceClassOp resource) {
      auto granularity =
          resource ? resource->getAttrOfType<StringAttr>("granularity")
                   : StringAttr{};
      return resource && granularity && granularity.getValue() == "patch" &&
             resource.getCount() > 0;
    };
    if (level1 == resourceClasses.end() || level2 == resourceClasses.end() ||
        !validPatchClass(level1->second) || !validPatchClass(level2->second))
      return protocol.emitOpError(
          "surface factory recurrence requires two positive-capacity patch "
          "bindings");

    constexpr int64_t level1PatchesPerLane = 5;
    constexpr int64_t cczPatches = 11;
    constexpr int64_t autoCCZPatches = 9;
    int64_t level1Capacity = capacityByDistance[level1Distance];
    int64_t level2Capacity = capacityByDistance[level2Distance];
    if (level1Capacity % level1PatchesPerLane != 0)
      return protocol.emitOpError(
          "surface factory level-1 patch capacity must contain a whole "
          "number of five-patch 15-to-1 lanes");
    int64_t level1Lanes = level1Capacity / level1PatchesPerLane;
    if (level1Lanes <= 0 || level2Capacity <= cczPatches ||
        (level2Capacity - cczPatches) % autoCCZPatches != 0)
      return protocol.emitOpError(
          "surface factory level-2 patch capacity must contain one "
          "eleven-patch CCZ stage and whole nine-patch AutoCCZ workspaces");
    int64_t fixupBoxes = (level2Capacity - cczPatches) / autoCCZPatches;
    if (level1Lanes != 6)
      return protocol.emitOpError(
          "the registered Gidney--Fowler layout requires exactly six "
          "five-patch level-1 lanes");
    if (fixupBoxes != 2)
      return protocol.emitOpError(
          "the registered AutoCCZ layout requires exactly two nine-patch "
          "fixup workspaces");

    int64_t factoryPatches = level1->second.getCount();
    auto distanceWidth = checkedSum({level2Distance, 1});
    auto patchPhysicalUnits =
        distanceWidth ? checkedProduct({2, *distanceWidth, *distanceWidth})
                      : std::nullopt;
    auto occupiedPatches = checkedSum({level1Capacity, level2Capacity});
    auto footprint = level1->second.getPhysicalUnitsAttr();
    if (first->second != second->second || !patchPhysicalUnits ||
        !occupiedPatches || !footprint ||
        footprint.getInt() != *patchPhysicalUnits ||
        *occupiedPatches > factoryPatches)
      return protocol.emitOpError(
          "the registered Gidney--Fowler layout requires one shared "
          "distance-2 footprint patch class large enough for its exact "
          "level-1 and level-2 regions");

    return SurfaceFactoryRecurrenceLayout{
        level1Distance,
        level2Distance,
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

  FailureOr<double> surfaceFactoryTiming(qlx::fabric::ProtocolOp protocol,
                                         StringRef action,
                                         int64_t distance) const {
    if (!operatingTiming || action.empty() || distance <= 0)
      return protocol.emitOpError(
          "surface factory recurrence requires an explicit selected-device "
          "timing profile");
    std::string qualified =
        (action + "_d" + std::to_string(distance) + "_ns").str();
    Attribute raw = operatingTiming.get(qualified);
    if (!raw)
      raw = operatingTiming.get((action + "_ns").str());
    auto value = numericValue(raw);
    if (!value || !std::isfinite(*value) || *value <= 0.0)
      return protocol.emitOpError("surface factory recurrence requires finite "
                                  "positive selected-device timing for '")
             << action << "' at code distance " << distance;
    return *value;
  }

  FailureOr<SurfaceFactoryStageShape>
  surfaceFactoryStageShape(qlx::fabric::ProtocolOp protocol,
                           ArrayRef<Operation *> closure) const {
    qlx::fabric::ProtocolOp level1;
    qlx::fabric::ProtocolOp level2;
    for (Operation *callable : closure) {
      auto candidate = dyn_cast<qlx::fabric::ProtocolOp>(callable);
      if (!candidate || candidate.getFunctionType().getNumResults() != 1)
        continue;
      auto resource = dyn_cast<qlx::fabric::ResourceStateType>(
          candidate.getFunctionType().getResult(0));
      auto kind = resource ? dyn_cast<FlatSymbolRefAttr>(resource.getKind())
                           : FlatSymbolRefAttr{};
      if (!kind)
        continue;
      if (kind.getValue() == kTState)
        level1 = candidate;
      else if (kind.getValue() == kCCZState)
        level2 = candidate;
    }
    if (!level1 || !level2)
      return protocol.emitOpError(
          "surface factory recurrence cannot derive its authenticated stage "
          "shape");

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
    for (auto call : protocol.getBody().front().getOps<qlx::fabric::CallOp>()) {
      Operation *callee = symbols.lookup(call.getCallee());
      auto candidate = dyn_cast_or_null<qlx::fabric::GadgetOp>(callee);
      if (!candidate || !candidate.getSpecAttr())
        continue;
      auto spec = dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
          symbols.lookup(candidate.getSpecAttr().getValue()));
      auto objective =
          spec ? dyn_cast_or_null<qlx::fabric::ObjectiveOp>(
                     symbols.lookup(spec.getObjectiveAttr().getValue()))
               : qlx::fabric::ObjectiveOp{};
      auto logical = objective && objective.getLogicalAttr()
                         ? dyn_cast_or_null<qlx::ActionOp>(symbols.lookup(
                               objective.getLogicalAttr().getValue()))
                         : qlx::ActionOp{};
      if (logical && logical.getKind() == "cz")
        ++ringEdges;
    }
    if (rotations <= 0 || checks <= 0 || ringEdges < 3)
      return protocol.emitOpError(
          "surface factory recurrence has no complete authenticated physical "
          "stage shape");

    // The registered layout maps every 15-to-1 product rotation to two
    // quarter-distance slices and uses one final quarter slice.  Its level-2
    // block has one layer per measured check plus the output layer.  The exact
    // odd CZ ring needs three disjoint matchings; derive that chromatic depth
    // from the authenticated ring rather than copying a cadence scalar.
    int64_t quarterLayers = 2 * rotations + 1;
    int64_t level2Layers = checks + 1;
    int64_t autoCCZLayers = ringEdges % 2 == 0 ? 2 : 3;
    return SurfaceFactoryStageShape{quarterLayers, level2Layers, autoCCZLayers};
  }

  FailureOr<FlatSymbolRefAttr>
  surfaceFactoryRecurrencePlan(qlx::fabric::ProtocolOp protocol) {
    auto existing = spacetimePlans.find(protocol.getSymName());
    if (existing != spacetimePlans.end())
      return existing->second;
    if (!operatingPoint || !hasExplicitCycleNanoseconds ||
        !std::isfinite(cycleNanoseconds) || cycleNanoseconds <= 0.0)
      return protocol.emitOpError(
          "surface factory recurrence requires finite positive "
          "surface_cycle_ns at the selected operating point");
    auto closure = callableClosure(protocol);
    if (failed(closure))
      return failure();
    auto layout = surfaceFactoryRecurrenceLayout(protocol, *closure);
    auto stageShape = surfaceFactoryStageShape(protocol, *closure);
    auto rawModel = closureFactoryModel(*closure);
    if (failed(layout) || failed(stageShape) || failed(rawModel))
      return failure();
    if ((*rawModel).getResourceKindAttr().getValue() != kRawTState)
      return protocol.emitOpError(
          "surface factory recurrence requires one authenticated raw-T "
          "physical supply");

    SmallVector<qlx::phys::PackResourceOp, 1> outputs;
    graph->walk([&](qlx::phys::PackResourceOp pack) {
      if (pack.getResourceKindAttr().getValue() == kAutoCCZState)
        outputs.push_back(pack);
    });
    if (outputs.size() != 1 || !outputs.front().getEventIdAttr())
      return protocol.emitOpError(
          "surface factory recurrence requires exactly one materialized "
          "physical AutoCCZ output event");

    double rawInterval = (*rawModel).getOutputIntervalNs().convertToDouble();
    if (!std::isfinite(rawInterval) || rawInterval <= 0.0)
      return protocol.emitOpError(
          "surface factory recurrence requires a finite positive raw-T "
          "output interval");
    auto level1RotationTiming =
        surfaceFactoryTiming(protocol, "resource_rpp", layout->level1Distance);
    auto level2LayerTiming =
        surfaceFactoryTiming(protocol, "resource_rpp", layout->level2Distance);
    auto autoCCZLayerTiming =
        surfaceFactoryTiming(protocol, "cz", layout->level2Distance);
    if (failed(level1RotationTiming) || failed(level2LayerTiming) ||
        failed(autoCCZLayerTiming))
      return failure();

    std::string name = (protocol.getSymName() + "_recurrence_plan").str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return protocol.emitOpError(
                 "derived surface-factory recurrence symbol already exists @")
             << name;
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToEnd(module.getBody());
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                             stringAttr(context, name)),
        builder.getNamedAttr("architecture", device.getPhysicalAttr()),
        builder.getNamedAttr("source_protocol",
                             symbolAttr(context, protocol.getSymName())),
        builder.getNamedAttr("operating_point", operatingPoint),
        builder.getNamedAttr("provider",
                             stringAttr(context, kSpacetimeProvider)),
        builder.getNamedAttr("provider_version",
                             stringAttr(context, kSpacetimeProviderVersion)),
        builder.getNamedAttr("derivation",
                             stringAttr(context, kSurfaceFactoryRecurrence)),
        builder.getNamedAttr(
            "derivation_version",
            builder.getI64IntegerAttr(kSurfaceFactoryRecurrenceVersion)),
        builder.getNamedAttr(
            "geometry", DenseI64ArrayAttr::get(context, layout->geometry())),
        builder.getNamedAttr(
            "evidence", stringAttr(context, kSurfaceFactoryRecurrenceEvidence)),
        builder.getNamedAttr("recurrence_resource_kind",
                             symbolAttr(context, kAutoCCZState)),
        builder.getNamedAttr("recurrence_output_event",
                             outputs.front().getEventIdAttr()),
    };
    Operation *plan = createGeneric(builder, protocol.getLoc(),
                                    "phys.spacetime_plan", {}, {}, attrs, 1);
    Block *body = new Block();
    plan->getRegion(0).push_back(body);
    OpBuilder eventBuilder = OpBuilder::atBlockBegin(body);
    auto resourceReference = [&](StringRef resource) -> Attribute {
      return SymbolRefAttr::get(context, architecture.getSymName(),
                                {FlatSymbolRefAttr::get(context, resource)});
    };
    auto event = [&](StringRef eventName, StringRef kind, int64_t iteration,
                     double start, double duration, StringRef resourceClass,
                     int64_t resourceOffset, int64_t resourceCount,
                     qlx::phys::FactoryModelOp model, int64_t factoryUnits,
                     ArrayRef<std::string> after, StringRef outputKind) {
      SmallVector<Attribute> dependencies;
      for (const std::string &dependency : after)
        dependencies.push_back(symbolAttr(context, dependency));
      SmallVector<NamedAttribute> eventAttrs{
          eventBuilder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                    stringAttr(context, eventName)),
          eventBuilder.getNamedAttr("kind", stringAttr(context, kind)),
          eventBuilder.getNamedAttr("iteration",
                                    eventBuilder.getI64IntegerAttr(iteration)),
          eventBuilder.getNamedAttr("start_ns",
                                    eventBuilder.getF64FloatAttr(start)),
          eventBuilder.getNamedAttr("duration_ns",
                                    eventBuilder.getF64FloatAttr(duration)),
          eventBuilder.getNamedAttr("after",
                                    eventBuilder.getArrayAttr(dependencies)),
      };
      if (!resourceClass.empty()) {
        eventAttrs.push_back(eventBuilder.getNamedAttr(
            "resource_class", resourceReference(resourceClass)));
        eventAttrs.push_back(eventBuilder.getNamedAttr(
            "resource_offset", eventBuilder.getI64IntegerAttr(resourceOffset)));
        eventAttrs.push_back(eventBuilder.getNamedAttr(
            "resource_count", eventBuilder.getI64IntegerAttr(resourceCount)));
      }
      if (model) {
        eventAttrs.push_back(eventBuilder.getNamedAttr(
            "factory_model", symbolAttr(context, model.getSymName())));
        eventAttrs.push_back(eventBuilder.getNamedAttr(
            "factory_units", eventBuilder.getI64IntegerAttr(factoryUnits)));
      }
      if (!outputKind.empty())
        eventAttrs.push_back(eventBuilder.getNamedAttr(
            "output_resource_kind", symbolAttr(context, outputKind)));
      createGeneric(eventBuilder, protocol.getLoc(), "phys.spacetime_event", {},
                    {}, eventAttrs);
    };

    // Materialize three consecutive outputs of the published interleaved
    // patch-macro layout. The final cadence is deliberately absent: generic
    // characterization derives it from the typed output-event separation.
    constexpr int64_t outputsToProve = 3;
    constexpr int64_t tStatesPerOutput = 8;
    const double level1Duration =
        static_cast<double>(stageShape->level1QuarterLayers) *
        *level1RotationTiming / 4.0;
    const double level2LayerDuration = *level2LayerTiming;
    const double autoCCZLayerDuration = *autoCCZLayerTiming;
    const double rawStartup = (*rawModel).getStartupNs().convertToDouble();
    if (!std::isfinite(level1Duration) || level1Duration <= 0.0 ||
        !std::isfinite(level2LayerDuration) || level2LayerDuration <= 0.0 ||
        !std::isfinite(autoCCZLayerDuration) || autoCCZLayerDuration <= 0.0 ||
        !std::isfinite(rawStartup) || rawStartup < 0.0)
      return protocol.emitOpError(
          "surface factory recurrence derives invalid physical event timing");

    SmallVector<double, 6> laneReady(layout->level1Lanes, rawStartup);
    SmallVector<std::string, 24> tEvents;
    SmallVector<double, 24> tReady;
    for (int64_t index = 0; index < outputsToProve * tStatesPerOutput;
         ++index) {
      int64_t lane = index % layout->level1Lanes;
      double start = laneReady[lane];
      double finish = start + level1Duration;
      std::string eventName = llvm::formatv("level1_t{0:00}", index).str();
      event(eventName, "level1_15to1", index / tStatesPerOutput, start,
            level1Duration, layout->level1ResourceClass,
            layout->level1ResourceOffset + lane * layout->level1PatchesPerLane,
            layout->level1PatchesPerLane, *rawModel,
            /*factoryUnits=*/15, {}, kTState);
      laneReady[lane] = finish;
      tEvents.push_back(eventName);
      tReady.push_back(finish);
    }

    double coreReady = 0.0;
    SmallVector<double, 2> fixupReady(layout->fixupBoxes, 0.0);
    SmallVector<std::string, 2> fixupTail(layout->fixupBoxes);
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
        std::string eventName =
            llvm::formatv("ccz_{0}_layer_{1}", iteration, layer).str();
        SmallVector<std::string> after =
            layer == 0 ? dependencies : SmallVector<std::string>{previous};
        event(eventName, "level2_ccz_layer", iteration, layerStart,
              level2LayerDuration, layout->level2ResourceClass,
              layout->level2ResourceOffset, layout->cczPatches, {}, 0, after,
              layer + 1 == stageShape->level2Layers ? StringRef(kCCZState)
                                                    : StringRef{});
        previous = eventName;
        layerStart += level2LayerDuration;
      }
      coreReady = layerStart;

      int64_t workspace = iteration % layout->fixupBoxes;
      layerStart = std::max(coreReady, fixupReady[workspace]);
      previous.clear();
      for (int64_t layer = 0; layer < stageShape->autoCCZLayers; ++layer) {
        std::string eventName =
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
        event(eventName, "autoccz_ring_layer", iteration, layerStart,
              autoCCZLayerDuration, layout->level2ResourceClass,
              layout->level2ResourceOffset + layout->cczPatches +
                  workspace * layout->autoCCZPatches,
              layout->autoCCZPatches, {}, 0, after,
              layer + 1 == stageShape->autoCCZLayers ? StringRef(kAutoCCZState)
                                                     : StringRef{});
        previous = eventName;
        layerStart += autoCCZLayerDuration;
      }
      fixupReady[workspace] = layerStart;
      fixupTail[workspace] = previous;
    }
    auto reference = symbolAttr(context, name);
    spacetimePlans[protocol.getSymName()] = reference;
    return reference;
  }

  FailureOr<FlatSymbolRefAttr>
  surfaceAutoCCZApplicationPlan(qlx::fabric::ProtocolOp protocol) {
    auto existing = spacetimePlans.find(protocol.getSymName());
    if (existing != spacetimePlans.end())
      return existing->second;
    if (!operatingPoint || !hasExplicitReactionNanoseconds ||
        !std::isfinite(reactionNanoseconds) || reactionNanoseconds <= 0.0)
      return protocol.emitOpError(
          "surface AutoCCZ application requires finite positive "
          "reaction_time_ns at the selected operating point");
    auto closure = callableClosure(protocol);
    if (failed(closure))
      return failure();
    auto model = closureFactoryModel(*closure);
    if (failed(model))
      return failure();
    if ((*model).getResourceKindAttr().getValue() != kAutoCCZState)
      return protocol.emitOpError(
          "surface AutoCCZ application requires one authenticated AutoCCZ "
          "physical supply");

    SmallVector<qlx::fabric::AllocOp, 6> allocations;
    for (auto allocation :
         protocol.getBody().front().getOps<qlx::fabric::AllocOp>())
      allocations.push_back(allocation);
    if (allocations.size() != 6)
      return protocol.emitOpError(
          "surface AutoCCZ application requires six routing allocations");
    auto binding = regionBindings.find(allocations.front().getRegion());
    if (binding == regionBindings.end() ||
        llvm::any_of(allocations,
                     [&](qlx::fabric::AllocOp allocation) {
                       return allocation.getRegion() !=
                              allocations.front().getRegion();
                     }) ||
        binding->second.granularity != "patch" || binding->second.capacity < 6)
      return protocol.emitOpError(
          "surface AutoCCZ application requires one patch-granularity "
          "routing binding with capacity six");
    int64_t lanes = binding->second.capacity / 6;
    double factoryInterval = (*model).getOutputIntervalNs().convertToDouble();
    if (!std::isfinite(factoryInterval) || factoryInterval <= 0.0)
      return protocol.emitOpError(
          "surface AutoCCZ application requires a finite positive factory "
          "output interval");
    double interval = std::max(reactionNanoseconds / static_cast<double>(lanes),
                               factoryInterval);

    std::string name = (protocol.getSymName() + "_application_plan").str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return protocol.emitOpError(
                 "derived surface-AutoCCZ application symbol already exists @")
             << name;
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToEnd(module.getBody());
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                             stringAttr(context, name)),
        builder.getNamedAttr("architecture", device.getPhysicalAttr()),
        builder.getNamedAttr("source_protocol",
                             symbolAttr(context, protocol.getSymName())),
        builder.getNamedAttr("operating_point", operatingPoint),
        builder.getNamedAttr("provider",
                             stringAttr(context, kSpacetimeProvider)),
        builder.getNamedAttr("provider_version",
                             stringAttr(context, kSpacetimeProviderVersion)),
        builder.getNamedAttr("derivation",
                             stringAttr(context, kSurfaceAutoCCZApplication)),
        builder.getNamedAttr(
            "derivation_version",
            builder.getI64IntegerAttr(kSurfaceAutoCCZApplicationVersion)),
        builder.getNamedAttr(
            "evidence",
            stringAttr(context, kSurfaceAutoCCZApplicationEvidence)),
        builder.getNamedAttr("forwarding_latency_ns",
                             builder.getF64FloatAttr(reactionNanoseconds)),
        builder.getNamedAttr("initiation_interval_ns",
                             builder.getF64FloatAttr(interval)),
    };
    Operation *plan = createGeneric(builder, protocol.getLoc(),
                                    "phys.spacetime_plan", {}, {}, attrs, 1);
    Block *body = new Block();
    plan->getRegion(0).push_back(body);
    OpBuilder phaseBuilder = OpBuilder::atBlockBegin(body);
    auto resource = SymbolRefAttr::get(
        context, architecture.getSymName(),
        {FlatSymbolRefAttr::get(context, binding->second.resourceClass)});
    createGeneric(
        phaseBuilder, protocol.getLoc(), "phys.spacetime_phase", {}, {},
        {
            phaseBuilder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                      stringAttr(context, "reaction")),
            phaseBuilder.getNamedAttr("steps",
                                      phaseBuilder.getI64IntegerAttr(1)),
            phaseBuilder.getNamedAttr(
                "step_duration_ns",
                phaseBuilder.getF64FloatAttr(reactionNanoseconds)),
            phaseBuilder.getNamedAttr("resource_classes",
                                      phaseBuilder.getArrayAttr({resource})),
            phaseBuilder.getNamedAttr("factory_models",
                                      phaseBuilder.getArrayAttr({symbolAttr(
                                          context, (*model).getSymName())})),
            phaseBuilder.getNamedAttr("after", phaseBuilder.getArrayAttr({})),
        });
    FlatSymbolRefAttr reference = symbolAttr(context, name);
    spacetimePlans[protocol.getSymName()] = reference;
    return reference;
  }

  FailureOr<FlatSymbolRefAttr>
  surfaceSpacelikeCallablePlan(qlx::fabric::ProtocolOp protocol,
                               qlx::spacetime::SurfaceSpacelikeShape shape) {
    auto existing = spacetimePlans.find(protocol.getSymName());
    if (existing != spacetimePlans.end())
      return existing->second;
    if (!operatingPoint || !hasExplicitReactionNanoseconds ||
        !hasExplicitCycleNanoseconds || !std::isfinite(reactionNanoseconds) ||
        !std::isfinite(cycleNanoseconds) || reactionNanoseconds <= 0.0 ||
        cycleNanoseconds <= 0.0)
      return protocol.emitOpError(
          "surface spacelike callable requires finite positive reaction and "
          "surface-cycle timing at the selected operating point");
    auto closure = callableClosure(protocol);
    if (failed(closure))
      return failure();
    auto model = closureFactoryModel(*closure);
    auto distance = closureCodeDistance(*closure);
    if (failed(model) || failed(distance))
      return failure();
    if ((*model).getResourceKindAttr().getValue() != kAutoCCZState)
      return protocol.emitOpError(
          "surface spacelike callable requires authenticated AutoCCZ "
          "physical supply");

    SmallVector<qlx::fabric::ProtocolOp, 4> applications;
    for (Operation *callable : *closure) {
      auto candidate = dyn_cast<qlx::fabric::ProtocolOp>(callable);
      auto provider = candidate ? spacetimeProviderFor(candidate)
                                : std::optional<SpacetimeProviderKind>{};
      if (!provider ||
          *provider != SpacetimeProviderKind::SurfaceAutoCCZApplication)
        continue;
      applications.push_back(candidate);
    }
    if (applications.empty())
      return protocol.emitOpError(
          "surface spacelike callable reaches no exact AutoCCZ application");
    if (applications.size() != shape.reactionApplications)
      return protocol.emitOpError(
          "surface spacelike structural demand differs from its exact "
          "AutoCCZ closure");
    SmallVector<qlx::fabric::AllocOp, 6> allocations;
    for (qlx::fabric::ProtocolOp application : applications) {
      SmallVector<qlx::fabric::AllocOp, 6> current;
      for (auto allocation :
           application.getBody().front().getOps<qlx::fabric::AllocOp>())
        current.push_back(allocation);
      if (current.size() != 6)
        return protocol.emitOpError(
            "surface spacelike callable requires six routing allocations "
            "per AutoCCZ application");
      llvm::append_range(allocations, current);
    }
    if (allocations.size() != 6 * shape.reactionApplications)
      return protocol.emitOpError(
          "surface spacelike callable has inconsistent routing demand");
    auto binding = regionBindings.find(allocations.front().getRegion());
    if (binding == regionBindings.end() ||
        llvm::any_of(allocations,
                     [&](qlx::fabric::AllocOp allocation) {
                       return allocation.getRegion() !=
                              allocations.front().getRegion();
                     }) ||
        binding->second.granularity != "patch" ||
        binding->second.capacity <
            static_cast<int64_t>(6 * shape.peakReactionWidth))
      return protocol.emitOpError(
          "surface spacelike callable requires one patch-granularity "
          "routing binding with capacity six per concurrent AutoCCZ "
          "application");
    int64_t lanes = binding->second.capacity / 6;
    if (shape.peakReactionWidth > static_cast<unsigned>(lanes))
      return protocol.emitOpError(
          "surface spacelike callable exceeds selected routing "
          "concurrency");
    double factoryInterval = (*model).getOutputIntervalNs().convertToDouble();
    if (!std::isfinite(factoryInterval) || factoryInterval <= 0.0)
      return protocol.emitOpError(
          "surface spacelike callable requires a finite positive factory "
          "output interval");
    double access = static_cast<double>(shape.accessLayers) *
                    static_cast<double>(*distance) * cycleNanoseconds / 2.0;
    double reactionDuration =
        static_cast<double>(shape.reactionDepth) * reactionNanoseconds;
    double forwarding = std::max(reactionDuration, access);
    double interval = std::max(
        {access, reactionDuration,
         static_cast<double>(shape.reactionApplications) * factoryInterval});

    std::string templateKey =
        llvm::formatv(
            "owners={0};access={1};reaction_depth={2};applications={3};"
            "peak_width={4};distance={5};model={6};routing={7}",
            shape.boundaryOwners, shape.accessLayers, shape.reactionDepth,
            shape.reactionApplications, shape.peakReactionWidth, *distance,
            (*model).getSymName(), binding->second.resourceClass)
            .str();
    auto shared = spacelikePlanTemplates.find(templateKey);
    if (shared != spacelikePlanTemplates.end()) {
      spacetimePlans[protocol.getSymName()] = shared->second;
      return shared->second;
    }

    std::string name = (protocol.getSymName() + "_spacelike_plan").str();
    if (SymbolTable::lookupSymbolIn(module, name))
      return protocol.emitOpError(
                 "derived surface spacelike symbol already exists @")
             << name;
    OpBuilder builder(module.getBodyRegion());
    builder.setInsertionPointToEnd(module.getBody());
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                             stringAttr(context, name)),
        builder.getNamedAttr("architecture", device.getPhysicalAttr()),
        builder.getNamedAttr("source_protocol",
                             symbolAttr(context, protocol.getSymName())),
        builder.getNamedAttr("operating_point", operatingPoint),
        builder.getNamedAttr("provider",
                             stringAttr(context, kSpacetimeProvider)),
        builder.getNamedAttr("provider_version",
                             stringAttr(context, kSpacetimeProviderVersion)),
        builder.getNamedAttr("derivation",
                             stringAttr(context, kSurfaceSpacelikeCallable)),
        builder.getNamedAttr(
            "derivation_version",
            builder.getI64IntegerAttr(kSurfaceSpacelikeCallableVersion)),
        builder.getNamedAttr(
            "evidence", stringAttr(context, kSurfaceSpacelikeCallableEvidence)),
        builder.getNamedAttr("forwarding_latency_ns",
                             builder.getF64FloatAttr(forwarding)),
        builder.getNamedAttr("initiation_interval_ns",
                             builder.getF64FloatAttr(interval)),
    };
    Operation *plan = createGeneric(builder, protocol.getLoc(),
                                    "phys.spacetime_plan", {}, {}, attrs, 1);
    Block *body = new Block();
    plan->getRegion(0).push_back(body);
    OpBuilder phaseBuilder = OpBuilder::atBlockBegin(body);
    auto resource = SymbolRefAttr::get(
        context, architecture.getSymName(),
        {FlatSymbolRefAttr::get(context, binding->second.resourceClass)});
    createGeneric(
        phaseBuilder, protocol.getLoc(), "phys.spacetime_phase", {}, {},
        {
            phaseBuilder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                      stringAttr(context, "pipeline")),
            phaseBuilder.getNamedAttr("steps",
                                      phaseBuilder.getI64IntegerAttr(1)),
            phaseBuilder.getNamedAttr("step_duration_ns",
                                      phaseBuilder.getF64FloatAttr(forwarding)),
            phaseBuilder.getNamedAttr("resource_classes",
                                      phaseBuilder.getArrayAttr({resource})),
            phaseBuilder.getNamedAttr("factory_models",
                                      phaseBuilder.getArrayAttr({symbolAttr(
                                          context, (*model).getSymName())})),
            phaseBuilder.getNamedAttr("after", phaseBuilder.getArrayAttr({})),
        });
    FlatSymbolRefAttr reference = symbolAttr(context, name);
    spacetimePlans[protocol.getSymName()] = reference;
    spacelikePlanTemplates[templateKey] = reference;
    return reference;
  }

  FailureOr<StringRef> rootPatchRegion(qlx::fabric::PatchType patch) {
    auto machine = dyn_cast_or_null<qlx::fabric::DeviceOp>(
        symbols.lookup(device.getQecAttr().getValue()));
    if (!machine)
      return failure();
    qlx::fabric::RegionOp selected;
    for (auto region : machine.getBody().getOps<qlx::fabric::RegionOp>()) {
      if (region.getCodeAttr() != patch.getCodeType() ||
          !regionBindings.count(region.getSymName()))
        continue;
      if (patch.getEncoding() && region.getEncodingAttr() &&
          patch.getEncoding() != region.getEncodingAttr())
        continue;
      if (selected)
        return failure();
      selected = region;
    }
    if (!selected)
      return failure();
    return selected.getSymName();
  }

  PhysicalGroup prepare(ValueRange inputs, StringRef state, Location location,
                        OpBuilder &builder) {
    SmallVector<Type> types;
    llvm::transform(inputs, std::back_inserter(types),
                    [](Value value) { return value.getType(); });
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("state", stringAttr(context, state)),
        builder.getNamedAttr("event_id", nextEvent("prepare")),
    };
    Operation *operation =
        createGeneric(builder, location, "phys.prepare", inputs, types, attrs);
    return PhysicalGroup(operation->getResults().begin(),
                         operation->getResults().end());
  }

  LogicalResult retireAllocation(const PhysicalGroup &values,
                                 StringRef releaseEvent) {
    std::optional<size_t> allocationIndex;
    for (Value value : values) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      auto found =
          state ? resourceToAllocation.find(state.getResource().getValue())
                : resourceToAllocation.end();
      if (found == resourceToAllocation.end() ||
          (allocationIndex && *allocationIndex != found->second))
        return failure();
      allocationIndex = found->second;
    }
    if (!allocationIndex)
      return failure();
    AllocationRecord &record = allocations[*allocationIndex];
    if (!record.release.empty())
      return failure();
    record.release = releaseEvent.str();
    for (int64_t index : record.indices)
      lastReleaseByBinding[{record.resourceClass, index}] = record.release;
    for (Value value : values) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      auto info = state ? resources.find(state.getResource()) : resources.end();
      if (info == resources.end())
        return failure();
      activeIndices[info->second.resourceClass].erase(info->second.index);
      resourceToAllocation.erase(state.getResource().getValue());
    }
    allocationPlanTrace.push_back(
        AllocationPlanStep{false, record.resourceClass, 0, record.indices});
    return success();
  }

  LogicalResult release(const PhysicalGroup &values, Location location,
                        OpBuilder &builder) {
    StringAttr releaseEvent = nextEvent("release");
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("event_id", releaseEvent),
    };
    createGeneric(builder, location, "phys.release", values.values(), {},
                  attrs);
    return retireAllocation(values, releaseEvent.getValue());
  }

  void emitAllocationMapping(OpBuilder &builder, Location location) {
    SmallVector<Attribute> entries;
    for (const AllocationRecord &record : allocations) {
      SmallVector<NamedAttribute> fields{
          builder.getNamedAttr("allocation", stringAttr(context, record.name)),
          builder.getNamedAttr("resource_class",
                               symbolAttr(context, record.resourceClass)),
          builder.getNamedAttr("resources",
                               builder.getArrayAttr(record.resources)),
          builder.getNamedAttr("indices",
                               builder.getDenseI64ArrayAttr(record.indices)),
          builder.getNamedAttr("acquire", stringAttr(context, record.acquire)),
      };
      if (!record.release.empty())
        fields.push_back(builder.getNamedAttr(
            "release", stringAttr(context, record.release)));
      if (!record.after.empty())
        fields.push_back(builder.getNamedAttr(
            "after", builder.getArrayAttr(llvm::map_to_vector(
                         record.after, [&](const std::string &event) {
                           return Attribute(stringAttr(context, event));
                         }))));
      entries.push_back(builder.getDictionaryAttr(fields));
    }
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                             stringAttr(context, graphName + "_allocations")),
        builder.getNamedAttr("graph", symbolAttr(context, graphName)),
        builder.getNamedAttr("entries", builder.getArrayAttr(entries)),
    };
    createGeneric(builder, location, "phys.allocation_mapping", {}, {}, attrs);
  }

  void emitPatchMapping(OpBuilder &builder, Location location) {
    std::string patchGraphName = graphName + "_patch_graph";
    SmallVector<Attribute> nodes;
    SmallVector<Attribute> carrierRecords;
    for (const AllocationRecord &record : allocations) {
      nodes.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("id", stringAttr(context, record.name)),
      }));
      for (auto [ordinal, resource] : llvm::enumerate(record.resources)) {
        std::string role =
            llvm::formatv("{0}[{1}]", record.name, ordinal).str();
        carrierRecords.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("role", stringAttr(context, role)),
            builder.getNamedAttr("resource", resource),
        }));
      }
    }
    createGeneric(
        builder, location, "fabric.patch_graph", {}, {},
        {
            builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                 stringAttr(context, patchGraphName)),
            builder.getNamedAttr("root", symbolAttr(context, rootName)),
            builder.getNamedAttr("nodes", builder.getArrayAttr(nodes)),
            builder.getNamedAttr("interactions", builder.getArrayAttr({})),
        });
    ArrayAttr carrierRecordArray = builder.getArrayAttr(carrierRecords);
    createGeneric(
        builder, location, "phys.mapping", {}, {},
        {
            builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                 stringAttr(context, graphName + "_mapping")),
            builder.getNamedAttr("graph", symbolAttr(context, graphName)),
            builder.getNamedAttr("source_graph",
                                 symbolAttr(context, patchGraphName)),
            builder.getNamedAttr("initial", carrierRecordArray),
            builder.getNamedAttr("final", carrierRecordArray),
        });
  }

  FailureOr<std::pair<int64_t, int64_t>> partitionRange(Type type,
                                                        StringRef selected) {
    auto partitions = patchPartitions(type);
    if (failed(partitions))
      return failure();
    std::map<std::string, int64_t> counts;
    for (NamedAttribute partition : *partitions) {
      auto count = dyn_cast<IntegerAttr>(partition.getValue());
      if (!count || count.getInt() < 0)
        return failure();
      counts[partition.getName().strref().str()] = count.getInt();
    }
    SmallVector<std::string> order;
    for (StringRef name : {StringRef("data"), StringRef("sx"), StringRef("sz")})
      if (counts.count(name.str()))
        order.push_back(name.str());
    for (const auto &[name, count] : counts)
      if (!llvm::is_contained(order, name))
        order.push_back(name);
    if (selected == "all") {
      int64_t total = 0;
      for (const std::string &name : order)
        total += counts[name];
      return std::pair<int64_t, int64_t>{0, total};
    }
    int64_t offset = 0;
    for (const std::string &name : order) {
      if (name == selected)
        return std::pair<int64_t, int64_t>{offset, counts[name]};
      offset += counts[name];
    }
    return failure();
  }

  FailureOr<SmallVector<int64_t>> selectedPositions(Operation *operation,
                                                    Type patchType) {
    auto partition =
        operation->getAttrOfType<qlx::fabric::PartitionAttr>("partition");
    if (!partition)
      return failure();
    StringRef name = qlx::fabric::stringifyPartition(partition.getValue());
    auto range = partitionRange(patchType, name);
    if (failed(range))
      return failure();
    auto [offset, count] = *range;
    SmallVector<int64_t> positions;
    if (auto indices = operation->getAttrOfType<DenseI64ArrayAttr>("indices")) {
      for (int64_t index : indices.asArrayRef()) {
        if (index < 0 || index >= count)
          return failure();
        positions.push_back(offset + index);
      }
    } else {
      for (int64_t index = 0; index < count; ++index)
        positions.push_back(offset + index);
    }
    if (auto patch = dyn_cast<qlx::fabric::PatchType>(patchType)) {
      auto granularity = patchResourceGranularity(patch);
      if (failed(granularity))
        return failure();
      if (*granularity == "patch" && !positions.empty())
        return SmallVector<int64_t>{0};
    }
    return positions;
  }

  FailureOr<SmallVector<int64_t>> wholePartitionPositions(Type patchType,
                                                          StringRef partition) {
    auto range = partitionRange(patchType, partition);
    if (failed(range))
      return failure();
    if (auto patch = dyn_cast<qlx::fabric::PatchType>(patchType)) {
      auto granularity = patchResourceGranularity(patch);
      if (failed(granularity))
        return failure();
      if (*granularity == "patch")
        return SmallVector<int64_t>{0};
    }
    return llvm::to_vector(
        llvm::seq(range->first, range->first + range->second));
  }

  FailureOr<PhysicalGroup> prepareSelected(PhysicalGroup values,
                                           ArrayRef<int64_t> positions,
                                           StringRef state, Location location,
                                           OpBuilder &builder) {
    SmallVector<Value> selected;
    for (int64_t position : positions) {
      if (position < 0 || size_t(position) >= values.size())
        return failure();
      selected.push_back(values[position]);
    }
    PhysicalGroup updated = prepare(selected, state, location, builder);
    for (auto [position, value] : llvm::zip(positions, updated))
      values.set(position, value);
    return values;
  }

  FailureOr<PhysicalGroup> resetSelected(PhysicalGroup values,
                                         ArrayRef<int64_t> positions,
                                         Location location,
                                         OpBuilder &builder) {
    SmallVector<Value> selected;
    SmallVector<Type> types;
    for (int64_t position : positions) {
      if (position < 0 || size_t(position) >= values.size())
        return failure();
      selected.push_back(values[position]);
      types.push_back(values[position].getType());
    }
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("state", stringAttr(context, "zero")),
        builder.getNamedAttr("event_id", nextEvent("reset")),
    };
    Operation *reset =
        createGeneric(builder, location, "phys.reset", selected, types, attrs);
    for (auto [position, value] : llvm::zip(positions, reset->getResults()))
      values.set(position, value);
    return values;
  }

  bool resourceAdvertises(StringRef resourceClass, StringRef action) {
    return advertisedNativeActions.contains(
        {symbolAttr(context, resourceClass), symbolAttr(context, action)});
  }

  bool resourceClassHasCapability(StringRef resourceClass,
                                  StringRef capability) {
    auto found = resourceClasses.find(resourceClass);
    if (found == resourceClasses.end())
      return false;
    auto capabilities = found->second.getCapabilities();
    return capabilities && llvm::any_of(*capabilities, [&](Attribute value) {
             auto key = dyn_cast<StringAttr>(value);
             return key && key.getValue() == capability;
           });
  }

  FailureOr<qlx::phys::InstrumentOp> sharedNativeInstrument(ValueRange carriers,
                                                            StringRef kind) {
    qlx::phys::InstrumentOp selected;
    for (Value carrier : carriers) {
      auto state = dyn_cast<qlx::phys::StateType>(carrier.getType());
      auto info = state ? resources.find(state.getResource()) : resources.end();
      if (info == resources.end())
        return failure();
      auto resourceClass = resourceClasses.find(info->second.resourceClass);
      if (resourceClass == resourceClasses.end())
        return failure();
      qlx::phys::InstrumentOp match;
      for (Attribute raw :
           resourceClass->second.getNativeInstruments().value_or(ArrayAttr{})) {
        auto reference = dyn_cast<FlatSymbolRefAttr>(raw);
        auto instrument = reference ? dyn_cast_or_null<qlx::phys::InstrumentOp>(
                                          symbols.lookup(reference.getValue()))
                                    : qlx::phys::InstrumentOp{};
        if (!instrument || instrument.getKind() != kind)
          continue;
        if (match)
          return failure();
        match = instrument;
      }
      if (!match || (selected && selected != match))
        return failure();
      selected = match;
    }
    if (!selected)
      return failure();
    return selected;
  }

  FailureOr<PhysicalGroup> applyAction(PhysicalGroup values,
                                       ArrayRef<int64_t> positions,
                                       StringRef action, Location location,
                                       OpBuilder &builder) {
    actionFailure.clear();
    auto foundAction = physicalActions.find(action);
    qlx::phys::ActionOp actionOp = foundAction == physicalActions.end()
                                       ? qlx::phys::ActionOp{}
                                       : foundAction->second;
    SmallVector<Value> selected;
    SmallVector<Attribute> resourceRefs;
    for (int64_t position : positions) {
      if (position < 0 || size_t(position) >= values.size())
        return failure();
      auto state = dyn_cast<qlx::phys::StateType>(values[position].getType());
      auto info = state ? resources.find(state.getResource()) : resources.end();
      if (info == resources.end()) {
        actionFailure =
            llvm::formatv("physical action @{0} references an unknown carrier "
                          "resource",
                          action)
                .str();
        return failure();
      }
      if (!resourceAdvertises(info->second.resourceClass, action)) {
        actionFailure =
            llvm::formatv("physical resource class @{0} does not advertise "
                          "native action @{1}",
                          info->second.resourceClass, action)
                .str();
        return failure();
      }
      selected.push_back(values[position]);
      resourceRefs.push_back(state.getResource());
    }
    if (selected.empty())
      return values;

    int64_t batchLanes = 1;
    if (actionOp) {
      int64_t arity = actionOp.getArityAttr().getInt();
      if (!actionOp.getBroadcastAttr()) {
        if (arity <= 0 || int64_t(selected.size()) % arity != 0)
          return failure();
        batchLanes = int64_t(selected.size()) / arity;
      }
    }
    // Compatibility action names deliberately have no standalone phys.action
    // declaration. They still apply once to the complete selected set, which
    // is also the only grouping emitted for declared actions.
    SmallVector<Type> types = llvm::map_to_vector(
        selected, [](Value value) { return value.getType(); });
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("action", symbolAttr(context, action)),
        builder.getNamedAttr("resources", builder.getArrayAttr(resourceRefs)),
        builder.getNamedAttr("event_id", nextEvent(action)),
    };
    if (batchLanes > 1)
      attrs.push_back(builder.getNamedAttr(
          "batch_lanes", builder.getI64IntegerAttr(batchLanes)));
    Operation *apply =
        createGeneric(builder, location, "phys.apply", selected, types, attrs);
    ++applyEvents;
    llvm::copy(apply->getResults(), selected.begin());
    for (auto [position, value] : llvm::zip(positions, selected))
      values.set(position, value);
    return values;
  }

  FailureOr<SmallVector<std::pair<int64_t, int64_t>>>
  explicitPairs(Operation *operation, int64_t controlCount,
                int64_t targetCount) const {
    auto pairs = operation->getAttrOfType<StringAttr>("pairs");
    if (!pairs)
      return failure();
    if (pairs.getValue() == "index") {
      if (controlCount != targetCount)
        return failure();
      SmallVector<std::pair<int64_t, int64_t>> result;
      for (int64_t index = 0; index < controlCount; ++index)
        result.push_back({index, index});
      return result;
    }
    SmallVector<StringRef> entries;
    pairs.getValue().split(entries, ',', /*MaxSplit=*/-1,
                           /*KeepEmpty=*/false);
    SmallVector<std::pair<int64_t, int64_t>> result;
    for (StringRef entry : entries) {
      auto [left, right] = entry.trim().split(':');
      int64_t control = -1;
      int64_t target = -1;
      if (right.empty() || left.trim().getAsInteger(10, control) ||
          right.trim().getAsInteger(10, target) || control < 0 ||
          control >= controlCount || target < 0 || target >= targetCount)
        return failure();
      result.push_back({control, target});
    }
    if (result.empty())
      return failure();
    return result;
  }

  LogicalResult applyTwoPatch(Operation *operation, StringRef action,
                              ProjectedValues &projected, OpBuilder &builder) {
    if (operation->getNumOperands() < 1 || operation->getNumOperands() > 2 ||
        operation->getNumResults() != operation->getNumOperands())
      return failure();
    auto controlPartition =
        operation->getAttrOfType<qlx::fabric::PartitionAttr>("ctrl");
    auto targetPartition =
        operation->getAttrOfType<qlx::fabric::PartitionAttr>("targ");
    if (!controlPartition || !targetPartition)
      return failure();
    StringRef controlName =
        qlx::fabric::stringifyPartition(controlPartition.getValue());
    StringRef targetName =
        qlx::fabric::stringifyPartition(targetPartition.getValue());
    auto controlRange =
        partitionRange(operation->getOperand(0).getType(), controlName);
    unsigned targetOperand = operation->getNumOperands() - 1;
    auto targetRange = partitionRange(
        operation->getOperand(targetOperand).getType(), targetName);
    if (failed(controlRange) || failed(targetRange))
      return failure();
    auto pairs =
        explicitPairs(operation, controlRange->second, targetRange->second);
    if (failed(pairs))
      return failure();

    SmallVector<PhysicalGroup> groups;
    for (Value operand : operation->getOperands())
      groups.push_back(takeProjected(projected, operand, operation));
    SmallVector<std::pair<int64_t, int64_t>> positions;
    llvm::DenseSet<Value> batchedStates;
    bool disjoint = true;
    for (auto [control, target] : *pairs) {
      int64_t controlPosition = controlRange->first + control;
      int64_t targetPosition = targetRange->first + target;
      if (targetOperand == 0 && controlPosition == targetPosition)
        return failure();
      if (controlPosition < 0 || size_t(controlPosition) >= groups[0].size() ||
          targetPosition < 0 ||
          size_t(targetPosition) >= groups[targetOperand].size())
        return failure();
      positions.emplace_back(controlPosition, targetPosition);
      disjoint &= batchedStates.insert(groups[0][controlPosition]).second;
      disjoint &=
          batchedStates.insert(groups[targetOperand][targetPosition]).second;
    }
    if (disjoint) {
      PhysicalGroup batched;
      for (auto [controlPosition, targetPosition] : positions) {
        batched.push_back(groups[0][controlPosition]);
        batched.push_back(groups[targetOperand][targetPosition]);
      }
      auto applied = applyAction(
          batched, llvm::to_vector(llvm::seq<int64_t>(0, batched.size())),
          action, operation->getLoc(), builder);
      if (failed(applied))
        return failure();
      for (auto [lane, position] : llvm::enumerate(positions)) {
        groups[0].set(position.first, (*applied)[2 * lane]);
        groups[targetOperand].set(position.second, (*applied)[2 * lane + 1]);
      }
    } else {
      for (auto [controlPosition, targetPosition] : positions) {
        PhysicalGroup pair{groups[0][controlPosition],
                           groups[targetOperand][targetPosition]};
        auto applied =
            applyAction(pair, {0, 1}, action, operation->getLoc(), builder);
        if (failed(applied))
          return failure();
        groups[0].set(controlPosition, (*applied)[0]);
        groups[targetOperand].set(targetPosition, (*applied)[1]);
      }
    }
    for (auto [result, group] : llvm::zip(operation->getResults(), groups))
      projected[result] = std::move(group);
    return success();
  }

  FlatSymbolRefAttr ensureMeasureZInstrument(Location location) {
    constexpr StringLiteral name = "measure_z_instrument";
    if (!SymbolTable::lookupSymbolIn(module, name)) {
      OpBuilder topBuilder(module.getBodyRegion());
      topBuilder.setInsertionPoint(graph);
      createGeneric(
          topBuilder, location, "phys.instrument", {}, {},
          {
              topBuilder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                      stringAttr(context, name)),
              topBuilder.getNamedAttr("kind", stringAttr(context, "measure")),
              topBuilder.getNamedAttr("arity", topBuilder.getI64IntegerAttr(1)),
              topBuilder.getNamedAttr("record_schema",
                                      stringAttr(context, "bit")),
              topBuilder.getNamedAttr("preserves_inputs",
                                      UnitAttr::get(context)),
              topBuilder.getNamedAttr(
                  "process",
                  stringAttr(context,
                             "{\"kind\":\"builtin\",\"name\":\"measure_z\","
                             "\"parameters\":{}}")),
          });
    }
    return symbolAttr(context, name);
  }

  FailureOr<std::pair<PhysicalGroup, PhysicalGroup>>
  measureZ(PhysicalGroup values, ArrayRef<int64_t> positions,
           StringRef recordBase, StringRef sourceSymbol,
           StringRef sourceInstance, StringRef lanePrefix, Location location,
           OpBuilder &builder) {
    FlatSymbolRefAttr instrument = ensureMeasureZInstrument(location);
    PhysicalGroup records;
    for (auto [ordinal, position] : llvm::enumerate(positions)) {
      if (position < 0 || size_t(position) >= values.size())
        return failure();
      Value input = values[position];
      std::string recordId = llvm::formatv("{0}.{1}.{2}", graphName, recordBase,
                                           measurementRecord++)
                                 .str();
      auto recordType =
          qlx::phys::RecordType::get(context, symbolAttr(context, "bit"));
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("measurement", instrument),
          builder.getNamedAttr("record_id", stringAttr(context, recordId)),
          builder.getNamedAttr("event_id", nextEvent("measure")),
      };
      Operation *measurement =
          createGeneric(builder, location, "phys.measure", input,
                        TypeRange{input.getType(), recordType}, attrs);
      values.set(position, measurement->getResult(0));
      records.push_back(measurement->getResult(1));
      std::string sourceRecord = llvm::formatv("{0}.{1}.{2}{3}", sourceSymbol,
                                               recordBase, lanePrefix, ordinal)
                                     .str();
      // Stable-record projection is consumed only by a selected detached
      // gadget profile. Retaining every unprofiled helper invocation here
      // creates millions of unreachable provenance strings without adding
      // P3 evidence.
      if (currentProfile)
        mutateRecordProjection({sourceInstance.str(), std::move(sourceRecord)})
            .push_back(recordId);
    }
    return std::pair<PhysicalGroup, PhysicalGroup>{std::move(values),
                                                   std::move(records)};
  }

  FailureOr<Value> condition(Value value, bool expected, Location location,
                             OpBuilder &builder, bool forceEvent = false) {
    if (value.getType().isInteger(1) && !forceEvent)
      return value;
    if (!value.getType().isInteger(1) &&
        !isa<qlx::phys::RecordType>(value.getType()))
      return failure();
    Operation *predicate = createGeneric(
        builder, location, "phys.condition", value, builder.getI1Type(),
        {
            builder.getNamedAttr("expected", builder.getBoolAttr(expected)),
            builder.getNamedAttr("event_id", nextEvent("condition")),
        });
    return predicate->getResult(0);
  }

  FailureOr<Value> parity(ValueRange sourceValues, Location location,
                          OpBuilder &builder) {
    if (sourceValues.empty())
      return failure();
    SmallVector<Value> predicates;
    for (Value value : sourceValues) {
      auto projected = condition(value, /*expected=*/true, location, builder);
      if (failed(projected))
        return failure();
      predicates.push_back(*projected);
    }
    Value result = predicates.front();
    for (Value next : ArrayRef(predicates).drop_front()) {
      Operation *xored =
          createGeneric(builder, location, "phys.xor", ValueRange{result, next},
                        builder.getI1Type(),
                        {builder.getNamedAttr("event_id", nextEvent("xor"))});
      result = xored->getResult(0);
    }
    return result;
  }

  FailureOr<PhysicalGroup>
  projectedTemplate(Value value, const ProjectedValues &projected,
                    llvm::SmallPtrSetImpl<Operation *> &active) const {
    auto found = projected.find(value);
    if (found != projected.end())
      return found->second;
    Operation *owner = value.getDefiningOp();
    if (!owner || !active.insert(owner).second)
      return failure();
    llvm::scope_exit erase([&] { active.erase(owner); });
    const bool patchValue =
        isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(
            value.getType());
    if (patchValue) {
      SmallVector<Value> patchOperands;
      SmallVector<Value> patchResults;
      for (Value operand : owner->getOperands())
        if (isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(
                operand.getType()))
          patchOperands.push_back(operand);
      for (Value result : owner->getResults())
        if (isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(
                result.getType()))
          patchResults.push_back(result);
      if (patchOperands.size() == patchResults.size()) {
        auto position = llvm::find(patchResults, value);
        if (position != patchResults.end()) {
          auto traced = projectedTemplate(
              patchOperands[std::distance(patchResults.begin(), position)],
              projected, active);
          if (succeeded(traced))
            return traced;
        }
      }
      if (auto conditional = dyn_cast<qlx::cflow::IfOp>(owner)) {
        unsigned position = cast<OpResult>(value).getResultNumber();
        std::optional<PhysicalGroup> selected;
        for (Region *region :
             {&conditional.getThenRegion(), &conditional.getElseRegion()}) {
          auto traced = projectedTemplate(
              region->front().getTerminator()->getOperand(position), projected,
              active);
          if (failed(traced))
            return failure();
          if (!selected) {
            selected = *traced;
            continue;
          }
          if (selected->size() != traced->size() ||
              !llvm::equal(selected->values(), traced->values(),
                           [](Value left, Value right) {
                             return left.getType() == right.getType();
                           }))
            return failure();
        }
        if (selected)
          return *selected;
      }
    } else {
      auto expected = projectedWidth(value.getType());
      if (failed(expected))
        return failure();
      for (Value operand : owner->getOperands()) {
        auto width = projectedWidth(operand.getType());
        if (failed(width) || *width != *expected)
          continue;
        auto traced = projectedTemplate(operand, projected, active);
        if (succeeded(traced))
          return traced;
      }
    }
    return failure();
  }

  FailureOr<PhysicalGroup>
  projectedTemplate(Value value, const ProjectedValues &projected) const {
    llvm::SmallPtrSet<Operation *, 8> active;
    return projectedTemplate(value, projected, active);
  }

  /// Decide whether a source-level i1 is the projection of a physical
  /// measurement record rather than an ordinary classical scalar.  Fabric
  /// intentionally presents both with the same logical type, while Phys keeps
  /// records typed so scheduling, provenance, and downstream sampling cannot
  /// confuse an acquired outcome with an authored boolean.
  FailureOr<bool>
  projectsToPhysicalRecord(Value value,
                           llvm::SmallPtrSetImpl<Operation *> &active) const {
    auto result = dyn_cast<OpResult>(value);
    Operation *owner = value.getDefiningOp();
    if (!result || !owner || !active.insert(owner).second)
      return false;
    llvm::scope_exit erase([&] { active.erase(owner); });
    if (auto measurement = dyn_cast<qlx::fabric::MeasureProductOp>(owner))
      return value == measurement.getOutcome();
    if (auto call = dyn_cast<qlx::fabric::CallOp>(owner)) {
      Operation *callee = symbols.lookup(call.getCallee());
      if (!callee)
        return failure();
      Block &body = callableBody(callee).front();
      if (result.getResultNumber() >= body.getTerminator()->getNumOperands())
        return failure();
      return projectsToPhysicalRecord(
          body.getTerminator()->getOperand(result.getResultNumber()), active);
    }
    if (auto conditional = dyn_cast<qlx::cflow::IfOp>(owner)) {
      std::optional<bool> expected;
      for (Region *region :
           {&conditional.getThenRegion(), &conditional.getElseRegion()}) {
        auto candidate = projectsToPhysicalRecord(
            region->front().getTerminator()->getOperand(
                result.getResultNumber()),
            active);
        if (failed(candidate) || (expected && *expected != *candidate))
          return failure();
        expected = *candidate;
      }
      return expected.value_or(false);
    }
    return false;
  }

  FailureOr<bool> projectsToPhysicalRecord(Value value) const {
    llvm::SmallPtrSet<Operation *, 8> active;
    return projectsToPhysicalRecord(value, active);
  }

  LogicalResult emitConditional(qlx::cflow::IfOp conditional,
                                ProjectedValues &projected,
                                OpBuilder &builder) {
    PhysicalGroup conditionValues = projected[conditional.getCondition()];
    if (conditionValues.size() != 1)
      return failure();
    auto predicate = condition(conditionValues.front(), /*expected=*/true,
                               conditional.getLoc(), builder);
    if (failed(predicate))
      return failure();

    Block &thenSource = conditional.getThenRegion().front();
    SmallVector<Type> resultTypes;
    SmallVector<int64_t> resultWidths;
    for (auto [result, yielded] :
         llvm::zip(conditional.getResults(),
                   thenSource.getTerminator()->getOperands())) {
      auto values = projectedTemplate(yielded, projected);
      if (succeeded(values) && !values->empty()) {
        resultWidths.push_back(values->size());
        for (Value value : *values)
          resultTypes.push_back(value.getType());
      } else if (auto tensor = dyn_cast<RankedTensorType>(result.getType())) {
        resultWidths.push_back(tensor.getNumElements());
        for (int64_t index = 0; index < tensor.getNumElements(); ++index)
          resultTypes.push_back(
              qlx::phys::RecordType::get(context, symbolAttr(context, "bit")));
      } else {
        auto type = physicalScalarType(result.getType());
        if (failed(type))
          return failure();
        resultWidths.push_back(1);
        resultTypes.push_back(*type);
      }
    }
    Operation *physical = createGeneric(
        builder, conditional.getLoc(), "cflow.if", *predicate, resultTypes,
        {builder.getNamedAttr("event_id", nextEvent("if"))}, 2);
    for (auto [source, target] :
         llvm::zip(conditional->getRegions(), physical->getRegions())) {
      Block *block = new Block();
      target.push_back(block);
      ProjectedValues nested = projected;
      OpBuilder nestedBuilder = OpBuilder::atBlockBegin(block);
      if (failed(emitBlock(source.front(), nested, nestedBuilder)))
        return failure();
      PhysicalGroup yields;
      for (Value value : source.front().getTerminator()->getOperands())
        llvm::append_range(yields, nested[value]);
      if (yields.size() != resultTypes.size())
        return failure();
      createGeneric(nestedBuilder, conditional.getLoc(), "cflow.yield",
                    yields.values());
    }
    size_t offset = 0;
    for (auto [result, width] :
         llvm::zip(conditional.getResults(), resultWidths)) {
      auto results = physical->getResults().slice(offset, width);
      projected[result] = PhysicalGroup(results.begin(), results.end());
      offset += width;
    }
    return success();
  }

  LogicalResult emitRetry(qlx::fabric::RetryOp retry,
                          ProjectedValues &projected, OpBuilder &builder) {
    if (retry.getCarries().empty() || !retry.getAttemptAttr() ||
        !retry.getProfileAttr())
      return retry.emitOpError(
          "requires carries plus attempt and profile symbols");
    PhysicalGroup states;
    SmallVector<int64_t> resultWidths;
    Operation *sourceAttempt = nullptr;
    for (Value carry : retry.getCarries()) {
      Operation *owner = carry.getDefiningOp();
      if (!owner || (sourceAttempt && sourceAttempt != owner))
        return retry.emitOpError(
            "requires all carries to come from one Fabric attempt call");
      sourceAttempt = owner;
      PhysicalGroup values = projected[carry];
      if (values.empty() || !llvm::all_of(values, [](Value value) {
            return isa<qlx::phys::StateType>(value.getType());
          }))
        return retry.emitOpError(
            "requires every carry to project to physical states");
      resultWidths.push_back(values.size());
      llvm::append_range(states, values);
    }
    auto projectedAttempt = projectedCallEvents.find(sourceAttempt);
    if (projectedAttempt == projectedCallEvents.end())
      return retry.emitOpError(
          "requires an authenticated projected attempt call");
    PhysicalGroup successes = projected[retry.getSuccess()];
    if (successes.size() != 1 ||
        (!successes.front().getType().isInteger(1) &&
         !isa<qlx::phys::RecordType>(successes.front().getType())))
      return retry.emitOpError(
          "requires its success value to project to one physical predicate");
    StringAttr attemptEvent = projectedAttempt->second;
    Value decided = successes.front();
    Operation *decision = decided.getDefiningOp();
    StringAttr decisionId =
        decision ? decision->getAttrOfType<StringAttr>("event_id")
                 : StringAttr{};
    if (!decisionId || decisionId == attemptEvent) {
      auto decision = condition(decided, /*expected=*/true, retry.getLoc(),
                                builder, /*forceEvent=*/true);
      if (failed(decision))
        return retry.emitOpError(
            "cannot derive a physical predicate from the attempt record");
      decided = *decision;
    }
    decision = decided.getDefiningOp();
    auto decisionEvent = decision
                             ? decision->getAttrOfType<StringAttr>("event_id")
                             : StringAttr{};
    if (!attemptEvent || !decisionEvent)
      return retry.emitOpError(
          "requires event identities on its attempt and decision");
    SmallVector<Value> operands(states.begin(), states.end());
    operands.push_back(decided);
    SmallVector<Type> resultTypes = llvm::map_to_vector(
        states, [](Value value) { return value.getType(); });
    auto exhaustion = retry->getAttrOfType<StringAttr>("exhaustion");
    const bool aborts = exhaustion && exhaustion.getValue() == "abort";
    auto emitAbortBoundary = [&] {
      SmallVector<NamedAttribute> barrierAttrs{
          builder.getNamedAttr(
              "domains",
              builder.getArrayAttr({builder.getStringAttr("clock"),
                                    builder.getStringAttr("factory")})),
          builder.getNamedAttr("event_id", nextEvent("barrier")),
      };
      createGeneric(builder, retry.getLoc(), "phys.barrier", {}, {},
                    barrierAttrs);
    };
    if (aborts)
      emitAbortBoundary();
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("max_attempts", retry.getMaxAttemptsAttr()),
        builder.getNamedAttr("attempt", retry.getAttemptAttr()),
        builder.getNamedAttr("profile", retry.getProfileAttr()),
        builder.getNamedAttr("attempt_event", attemptEvent),
        builder.getNamedAttr("decision_event", decisionEvent),
        builder.getNamedAttr("event_id", nextEvent("retry")),
    };
    for (StringRef name : {StringRef("exhaustion"), StringRef("commit_point"),
                           StringRef("success_probability"),
                           StringRef("success_probability_source"),
                           StringRef("success_probability_evidence")})
      if (Attribute value = retry->getAttr(name))
        attrs.push_back(builder.getNamedAttr(name, value));
    Operation *physical = createGeneric(builder, retry.getLoc(), "phys.retry",
                                        operands, resultTypes, attrs);
    if (aborts)
      emitAbortBoundary();
    size_t offset = 0;
    for (auto [result, width] : llvm::zip(retry.getResults(), resultWidths)) {
      auto results = physical->getResults().slice(offset, width);
      projected[result] = PhysicalGroup(results.begin(), results.end());
      offset += width;
    }
    return success();
  }

  FailureOr<SmallVector<int64_t>> supportRow(Operation *code, StringRef name,
                                             int64_t row) const {
    auto matrix = code->getAttrOfType<ArrayAttr>(name);
    if (!matrix || row < 0 || size_t(row) >= matrix.size())
      return failure();
    Attribute raw = matrix[row];
    if (auto dense = dyn_cast<DenseI64ArrayAttr>(raw))
      return llvm::to_vector(dense.asArrayRef());
    auto array = dyn_cast<ArrayAttr>(raw);
    if (!array)
      return failure();
    SmallVector<int64_t> result;
    for (Attribute value : array) {
      auto integer = dyn_cast<IntegerAttr>(value);
      if (!integer)
        return failure();
      result.push_back(integer.getInt());
    }
    return result;
  }

  static FailureOr<std::pair<int, std::optional<char>>>
  multiplyPauli(std::optional<char> left, char right) {
    if (!left)
      return std::pair<int, std::optional<char>>{0, right};
    if (*left == right)
      return std::pair<int, std::optional<char>>{0, std::nullopt};
    if (*left == 'X' && right == 'Y')
      return std::pair<int, std::optional<char>>{1, 'Z'};
    if (*left == 'Y' && right == 'X')
      return std::pair<int, std::optional<char>>{3, 'Z'};
    if (*left == 'X' && right == 'Z')
      return std::pair<int, std::optional<char>>{3, 'Y'};
    if (*left == 'Z' && right == 'X')
      return std::pair<int, std::optional<char>>{1, 'Y'};
    if (*left == 'Y' && right == 'Z')
      return std::pair<int, std::optional<char>>{1, 'X'};
    if (*left == 'Z' && right == 'Y')
      return std::pair<int, std::optional<char>>{3, 'X'};
    return failure();
  }

  FailureOr<PhysicalPauliProduct>
  physicalPauliProduct(Operation *operation,
                       ArrayRef<PhysicalGroup> patches) const {
    auto patchIndices =
        operation->getAttrOfType<DenseI64ArrayAttr>("patch_indices");
    auto logicalIndices =
        operation->getAttrOfType<DenseI64ArrayAttr>("logical_indices");
    auto encoded = operation->getAttrOfType<StringAttr>("pauli_product");
    if (!patchIndices || !logicalIndices || !encoded)
      return failure();
    StringRef word = encoded.getValue();
    int phase = 0;
    if (word.consume_front("-"))
      phase = 2;
    if (patchIndices.size() != logicalIndices.size() ||
        patchIndices.size() != word.size())
      return failure();

    std::map<std::pair<unsigned, int64_t>, char> factors;
    auto add = [&](unsigned patch, int64_t carrier,
                   char pauli) -> LogicalResult {
      if (patch >= patches.size() || carrier < 0 ||
          size_t(carrier) >= patches[patch].size())
        return failure();
      auto key = std::pair<unsigned, int64_t>{patch, carrier};
      auto found = factors.find(key);
      auto product = multiplyPauli(found == factors.end()
                                       ? std::optional<char>{}
                                       : std::optional<char>{found->second},
                                   pauli);
      if (failed(product))
        return failure();
      phase = (phase + product->first) & 3;
      if (!product->second)
        factors.erase(key);
      else
        factors[key] = *product->second;
      return success();
    };

    for (size_t term = 0; term < word.size(); ++term) {
      int64_t patchIndex = patchIndices.asArrayRef()[term];
      int64_t logical = logicalIndices.asArrayRef()[term];
      char pauli = word[term];
      if (patchIndex < 0 || size_t(patchIndex) >= patches.size() ||
          (pauli != 'X' && pauli != 'Y' && pauli != 'Z'))
        return failure();
      auto patchType = dyn_cast<qlx::fabric::PatchType>(
          operation
              ->getOperand(operation->getNumOperands() - patches.size() +
                           patchIndex)
              .getType());
      auto code = patchType
                      ? dyn_cast_or_null<qlx::fabric::CodeOp>(
                            symbols.lookup(patchType.getCodeType().getValue()))
                      : qlx::fabric::CodeOp{};
      if (!code)
        return failure();
      int64_t protectedCount = 1;
      if (auto k = code->getAttrOfType<IntegerAttr>("k"))
        protectedCount = k.getInt();
      int64_t basis = logical;
      StringRef xName = "lx";
      StringRef zName = "lz";
      if (logical >= protectedCount) {
        basis = logical - protectedCount;
        xName = "gx";
        zName = "gz";
      }
      auto granularity =
          projectedGroupGranularity(patches[patchIndex].values());
      if (failed(granularity))
        return failure();
      if (*granularity == "patch") {
        // A patch-granularity physical resource is one authenticated encoded
        // state, not a truncated carrier array. Validate the requested logical
        // Pauli against the retained code supports, then preserve it as one
        // aggregate P3 Pauli on that encoded-state resource.
        if ((pauli == 'X' || pauli == 'Y') &&
            failed(supportRow(code, xName, basis)))
          return failure();
        if ((pauli == 'Z' || pauli == 'Y') &&
            failed(supportRow(code, zName, basis)))
          return failure();
        if (failed(add(patchIndex, 0, pauli)))
          return failure();
        continue;
      }
      if (pauli == 'X' || pauli == 'Y') {
        if (pauli == 'Y')
          phase = (phase + 1) & 3;
        auto support = supportRow(code, xName, basis);
        if (failed(support))
          return failure();
        for (int64_t carrier : *support)
          if (failed(add(patchIndex, carrier, 'X')))
            return failure();
      }
      if (pauli == 'Z' || pauli == 'Y') {
        auto support = supportRow(code, zName, basis);
        if (failed(support))
          return failure();
        for (int64_t carrier : *support)
          if (failed(add(patchIndex, carrier, 'Z')))
            return failure();
      }
    }
    if (factors.empty() || (phase & 1))
      return failure();
    PhysicalPauliProduct result;
    for (auto [carrier, pauli] : factors) {
      result.carriers.push_back(carrier);
      result.paulis.push_back(pauli);
    }
    result.invert = phase == 2;
    return result;
  }

  LogicalResult rotateProduct(Operation *operation, Value resource,
                              ValueRange patchOperands, ValueRange patchResults,
                              ProjectedValues &projected, OpBuilder &builder) {
    SmallVector<PhysicalGroup> patches;
    for (Value patch : patchOperands)
      patches.push_back(takeProjected(projected, patch, operation));
    auto product = physicalPauliProduct(operation, patches);
    auto rawAngle = operation->getAttrOfType<FloatAttr>("angle");
    if (failed(product) || !rawAngle ||
        patchResults.size() != patchOperands.size())
      return failure();
    SmallVector<Value> inputs;
    SmallVector<Type> resultTypes;
    SmallVector<Attribute> paulis;
    for (auto [carrier, pauli] :
         llvm::zip(product->carriers, product->paulis)) {
      Value value = patches[carrier.first][carrier.second];
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      auto info = state ? resources.find(state.getResource()) : resources.end();
      if (info == resources.end() ||
          (!resource &&
           !resourceClassHasCapability(info->second.resourceClass,
                                       kNativePauliProductRotation) &&
           !resourceAdvertises(info->second.resourceClass, "rpp")))
        return failure();
      inputs.push_back(value);
      resultTypes.push_back(value.getType());
      paulis.push_back(stringAttr(context, StringRef(&pauli, 1)));
    }
    double angle = rawAngle.getValueAsDouble();
    if (product->invert)
      angle = -angle;
    SmallVector<Value> operands;
    StringRef name = "phys.rotate_product";
    StringRef eventName = "rpp";
    if (resource) {
      operands.push_back(resource);
      name = "phys.resource_rotate_product";
      eventName = "resource_rpp";
    }
    llvm::append_range(operands, inputs);
    Operation *physical = createGeneric(
        builder, operation->getLoc(), name, operands, resultTypes,
        {
            builder.getNamedAttr("paulis", builder.getArrayAttr(paulis)),
            builder.getNamedAttr("angle", builder.getF64FloatAttr(angle)),
            builder.getNamedAttr("event_id", nextEvent(eventName)),
        });
    for (auto [carrier, replacement] :
         llvm::zip(product->carriers, physical->getResults()))
      patches[carrier.first].set(carrier.second, replacement);
    for (auto [result, patch] : llvm::zip(patchResults, patches))
      projected[result] = std::move(patch);
    return success();
  }

  LogicalResult measureProduct(qlx::fabric::MeasureProductOp measurement,
                               ProjectedValues &projected, OpBuilder &builder) {
    actionFailure.clear();
    SmallVector<PhysicalGroup> patches;
    patches.reserve(measurement.getPatches().size());
    for (Value patch : measurement.getPatches())
      patches.push_back(takeProjected(projected, patch, measurement));
    auto product = physicalPauliProduct(measurement, patches);
    if (failed(product) || measurement.getPatchResults().size() !=
                               measurement.getPatches().size()) {
      actionFailure = "has an invalid physical Pauli projection";
      return failure();
    }

    SmallVector<Value> inputs;
    SmallVector<Type> resultTypes;
    SmallVector<Attribute> paulis;
    for (auto [carrier, pauli] :
         llvm::zip(product->carriers, product->paulis)) {
      Value value = patches[carrier.first][carrier.second];
      inputs.push_back(value);
      resultTypes.push_back(value.getType());
      paulis.push_back(stringAttr(context, StringRef(&pauli, 1)));
    }

    StringRef stableRecord = measurement.getRecord().value_or("mpp");
    std::string recordId = llvm::formatv("{0}.{1}.{2}", currentInstance,
                                         stableRecord, measurementRecord++)
                               .str();
    Operation *physical = nullptr;
    auto scalar =
        inputs.size() == 1 && product->paulis.front() == 'Z' && !product->invert
            ? sharedNativeInstrument(inputs, "measure")
            : FailureOr<qlx::phys::InstrumentOp>(failure());
    if (succeeded(scalar)) {
      auto recordType = qlx::phys::RecordType::get(
          context, symbolAttr(context, scalar->getRecordSchema()));
      resultTypes.push_back(recordType);
      physical = createGeneric(
          builder, measurement.getLoc(), "phys.measure", inputs, resultTypes,
          {
              builder.getNamedAttr("measurement",
                                   symbolAttr(context, scalar->getSymName())),
              builder.getNamedAttr("record_id", stringAttr(context, recordId)),
              builder.getNamedAttr("event_id", nextEvent("measure")),
          });
    } else if (auto instrument =
                   sharedNativeInstrument(inputs, "measure_product");
               succeeded(instrument)) {
      auto recordType = qlx::phys::RecordType::get(
          context, symbolAttr(context, instrument->getRecordSchema()));
      resultTypes.push_back(recordType);
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("instrument",
                               symbolAttr(context, instrument->getSymName())),
          builder.getNamedAttr("paulis", builder.getArrayAttr(paulis)),
          builder.getNamedAttr("record_id", stringAttr(context, recordId)),
          builder.getNamedAttr("event_id", nextEvent("mpp")),
      };
      if (product->invert)
        attrs.push_back(builder.getNamedAttr("invert", UnitAttr::get(context)));
      physical =
          createGeneric(builder, measurement.getLoc(), "phys.measure_product",
                        inputs, resultTypes, attrs);
    } else {
      actionFailure = "requires one shared native measure_product instrument";
      return failure();
    }

    for (auto [carrier, replacement] :
         llvm::zip(product->carriers, physical->getResults().drop_back()))
      patches[carrier.first].set(carrier.second, replacement);
    for (auto [result, patch] :
         llvm::zip(measurement.getPatchResults(), patches))
      projected[result] = std::move(patch);
    projected[measurement.getOutcome()] =
        PhysicalGroup{physical->getResults().back()};

    if (currentProfile) {
      std::string qualified = currentCallable + "." + stableRecord.str();
      mutateRecordProjection({currentInstance, qualified}).push_back(recordId);
      mutateRecordProjection({currentInstance, qualified + ".outcome"})
          .push_back(recordId);
    }
    return success();
  }

  FailureOr<PhysicalGroup> delay(PhysicalGroup values, int64_t rounds,
                                 Location location, OpBuilder &builder) {
    if (rounds < 0)
      return failure();
    SmallVector<Type> types;
    llvm::transform(values, std::back_inserter(types),
                    [](Value value) { return value.getType(); });
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr(
            "duration_ns", builder.getF64FloatAttr(rounds * cycleNanoseconds)),
        builder.getNamedAttr("event_id", nextEvent("idle")),
    };
    Operation *operation = createGeneric(builder, location, "phys.delay",
                                         values.values(), types, attrs);
    return PhysicalGroup(operation->getResults().begin(),
                         operation->getResults().end(),
                         values.getLayoutIdentity());
  }

  LogicalResult momentBarrier(ProjectedValues &projected, Location location,
                              OpBuilder &builder) {
    DenseMap<Attribute, Value> currentByResource;
    for (const auto &[source, group] : projected) {
      (void)source;
      for (Value value : group) {
        auto state = dyn_cast<qlx::phys::StateType>(value.getType());
        if (!state)
          continue;
        auto [entry, inserted] =
            currentByResource.try_emplace(state.getResource(), value);
        if (!inserted && entry->second != value)
          return emitError(location)
                 << "fabric.tick has multiple live physical SSA owners for "
                 << state.getResource();
      }
    }
    if (currentByResource.empty())
      return success();

    SmallVector<std::pair<FlatSymbolRefAttr, Value>> ordered;
    ordered.reserve(currentByResource.size());
    for (const auto &[resource, value] : currentByResource)
      ordered.emplace_back(cast<FlatSymbolRefAttr>(resource), value);
    llvm::sort(ordered, [](const auto &left, const auto &right) {
      return left.first.getValue() < right.first.getValue();
    });
    SmallVector<Value> inputs = llvm::map_to_vector(
        ordered, [](const auto &entry) { return entry.second; });
    SmallVector<Type> types = llvm::map_to_vector(
        inputs, [](Value value) { return value.getType(); });
    SmallVector<NamedAttribute> attrs{
        builder.getNamedAttr("event_id", nextEvent("tick")),
    };
    Operation *barrier =
        createGeneric(builder, location, "phys.barrier", inputs, types, attrs);
    DenseMap<Value, Value> replacements;
    for (auto [input, output] : llvm::zip(inputs, barrier->getResults()))
      replacements[input] = output;
    for (auto &[source, group] : projected) {
      (void)source;
      for (size_t index = 0; index < group.size(); ++index)
        if (auto replacement = replacements.find(group[index]);
            replacement != replacements.end())
          group.set(index, replacement->second);
    }
    return success();
  }

  FailureOr<std::string> physicalRecord(Value value) const {
    auto measurement = value.getDefiningOp<qlx::phys::MeasureOp>();
    if (!measurement || measurement.getRecord() != value ||
        measurement.getRecordId().empty())
      return failure();
    return measurement.getRecordId().str();
  }

  FailureOr<StringRef> projectedGroupGranularity(ValueRange values) const {
    std::optional<StringRef> selected;
    for (Value value : values) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      auto resource =
          state ? resources.find(state.getResource()) : resources.end();
      if (resource == resources.end())
        return failure();
      auto resourceClass = resourceClasses.find(resource->second.resourceClass);
      if (resourceClass == resourceClasses.end())
        return failure();
      auto raw =
          resourceClass->second->getAttrOfType<StringAttr>("granularity");
      StringRef candidate = raw ? raw.getValue() : StringRef("carrier");
      if (selected && *selected != candidate)
        return failure();
      selected = candidate;
    }
    if (!selected)
      return failure();
    return *selected;
  }

  FailureOr<PatchMacro> patchMacroFor(qlx::fabric::GadgetOp gadget) const {
    auto spec = gadget.getSpecAttr()
                    ? dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
                          symbols.lookup(gadget.getSpecAttr().getValue()))
                    : qlx::fabric::GadgetSpecOp{};
    auto objective =
        spec ? dyn_cast_or_null<qlx::fabric::ObjectiveOp>(
                   symbols.lookup(spec.getObjectiveAttr().getValue()))
             : qlx::fabric::ObjectiveOp{};
    Operation *logical =
        objective && objective.getLogicalAttr()
            ? symbols.lookup(objective.getLogicalAttr().getValue())
            : nullptr;
    if (!spec || !objective || !logical)
      return failure();

    if (auto action = dyn_cast<qlx::ActionOp>(logical)) {
      StringRef kind = action.getKind();
      if (!llvm::is_contained(ArrayRef<StringRef>{"h", "s", "sdg", "x", "y",
                                                  "z", "cx", "cz", "swap"},
                              kind))
        return failure();
      return PatchMacro{PatchMacroKind::Action, kind.str()};
    }

    auto instrument = dyn_cast<qlx::InstrumentDeclOp>(logical);
    if (!instrument)
      return failure();
    if (instrument.getKind() == "prepare_zero")
      return PatchMacro{PatchMacroKind::Prepare, "zero"};
    if (instrument.getKind() == "prepare_plus")
      return PatchMacro{PatchMacroKind::Prepare, "plus"};
    if (instrument.getKind() != "composite" || !instrument.getSemanticsAttr())
      return failure();
    auto semanticsRef =
        dyn_cast<FlatSymbolRefAttr>(instrument.getSemanticsAttr());
    auto semantics = dyn_cast_or_null<qlx::ObjectiveBodyOp>(
        semanticsRef ? symbols.lookup(semanticsRef.getValue()) : nullptr);
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
      return PatchMacro{PatchMacroKind::Measure, "measure_x"};
    if (measurement.getBasis() == qlx::Pauli::Z)
      return PatchMacro{PatchMacroKind::Measure, "measure"};
    return failure();
  }

  FailureOr<SmallVector<unsigned>>
  transformOutputSources(qlx::fabric::GadgetSpecOp spec, unsigned inputCount,
                         unsigned outputCount) const {
    SmallVector<std::optional<unsigned>> sources(outputCount);
    auto flows = spec.getFlows();
    if (!flows)
      return failure();
    for (Attribute raw : *flows) {
      auto flow = dyn_cast<DictionaryAttr>(raw);
      auto kind = flow ? flow.getAs<StringAttr>("kind") : StringAttr{};
      auto inputs =
          flow ? flow.getAs<DenseI64ArrayAttr>("inputs") : DenseI64ArrayAttr{};
      auto outputs =
          flow ? flow.getAs<DenseI64ArrayAttr>("outputs") : DenseI64ArrayAttr{};
      if (!kind || kind.getValue() != "transform" || !inputs || !outputs ||
          inputs.size() != outputs.size())
        return failure();
      for (auto [input, output] :
           llvm::zip(inputs.asArrayRef(), outputs.asArrayRef())) {
        if (input < 0 || output < 0 || unsigned(input) >= inputCount ||
            unsigned(output) >= outputCount || sources[output])
          return failure();
        sources[output] = unsigned(input);
      }
    }
    if (llvm::any_of(sources, [](const auto &source) { return !source; }))
      return failure();
    return llvm::map_to_vector(sources,
                               [](const auto &source) { return *source; });
  }

  FailureOr<SmallVector<PhysicalGroup>>
  emitPatchMacro(qlx::fabric::GadgetOp gadget, const PatchMacro &macro,
                 ArrayRef<PhysicalGroup> arguments, OpBuilder &builder) {
    auto spec = dyn_cast_or_null<qlx::fabric::GadgetSpecOp>(
        symbols.lookup(gadget.getSpecAttr().getValue()));
    Block &block = gadget.getBody().front();
    if (!spec || arguments.size() != block.getNumArguments() ||
        llvm::any_of(
            block.getArgumentTypes(),
            [](Type type) { return !isa<qlx::fabric::PatchType>(type); }) ||
        llvm::any_of(arguments, [](const PhysicalGroup &group) {
          return group.size() != 1;
        }))
      return failure();

    PhysicalGroup values;
    for (const PhysicalGroup &group : arguments)
      values.push_back(group.front());
    SmallVector<PhysicalGroup> outputs;
    if (macro.kind == PatchMacroKind::Action) {
      auto sources = transformOutputSources(
          spec, arguments.size(), block.getTerminator()->getNumOperands());
      SmallVector<int64_t> positions =
          llvm::to_vector(llvm::seq<int64_t>(0, values.size()));
      auto applied = applyAction(values, positions, macro.physicalOperation,
                                 gadget.getLoc(), builder);
      if (failed(sources) || failed(applied))
        return failure();
      for (unsigned source : *sources)
        outputs.push_back(PhysicalGroup((*applied)[source]));
      return outputs;
    }

    if (macro.kind == PatchMacroKind::Prepare) {
      if (values.size() != 1 || block.getTerminator()->getNumOperands() != 1)
        return failure();
      PhysicalGroup prepared = prepare(values.values(), macro.physicalOperation,
                                       gadget.getLoc(), builder);
      outputs.push_back(std::move(prepared));
      return outputs;
    }

    if (values.size() != 1 || block.getTerminator()->getNumOperands() != 1)
      return failure();
    auto instrument =
        sharedNativeInstrument(values.values(), macro.physicalOperation);
    auto schema = spec.getRecordSchema().value_or(ArrayAttr{});
    auto stableRecord =
        schema.size() == 1 ? dyn_cast<StringAttr>(schema[0]) : StringAttr{};
    if (failed(instrument) || !stableRecord || stableRecord.getValue().empty())
      return failure();
    std::string recordId =
        llvm::formatv("{0}.{1}.{2}", graphName, stableRecord.getValue(),
                      measurementRecord++)
            .str();
    auto recordType = qlx::phys::RecordType::get(
        context, symbolAttr(context, instrument->getRecordSchema()));
    SmallVector<Type> resultTypes{recordType};
    StringAttr measureEvent = nextEvent("measure");
    Operation *measurement = createGeneric(
        builder, gadget.getLoc(), "phys.measure", values.values(), resultTypes,
        {
            builder.getNamedAttr("measurement",
                                 symbolAttr(context, instrument->getSymName())),
            builder.getNamedAttr("record_id", stringAttr(context, recordId)),
            builder.getNamedAttr("event_id", measureEvent),
            builder.getNamedAttr("destructive", UnitAttr::get(context)),
        });
    if (failed(retireAllocation(values, measureEvent.getValue())))
      return failure();
    std::string stableName = stableRecord.getValue().str();
    auto &local =
        mutateRecordProjection(std::make_pair(currentInstance, stableName));
    local.push_back(recordId);
    auto &qualified = mutateRecordProjection(
        std::make_pair(currentInstance, currentCallable + "." + stableName));
    qualified = local;
    outputs.push_back(PhysicalGroup{measurement->getResult(0)});
    return outputs;
  }

  FailureOr<SmallVector<Operation *> *>
  profileSidecarDeclarations(FlatSymbolRefAttr profileRef) {
    auto cached = profileSidecarDeclarationCache.find(profileRef);
    if (cached != profileSidecarDeclarationCache.end())
      return &cached->second;
    auto profile = dyn_cast_or_null<qlx::fabric::GadgetProfileOp>(
        symbols.lookup(profileRef.getValue()));
    if (!profile)
      return failure();
    SmallVector<Operation *> declarations;
    for (Operation &declaration : profile.getBody().front())
      if (isa<qlx::fabric::SuccessOp>(declaration))
        declarations.push_back(&declaration);
    auto [inserted, didInsert] = profileSidecarDeclarationCache.try_emplace(
        profileRef, std::move(declarations));
    assert(didInsert && "profile sidecar cache changed during insertion");
    return &inserted->second;
  }

  LogicalResult collectProfileSidecars(FlatSymbolRefAttr profileRef,
                                       StringRef instance) {
    auto declarations = profileSidecarDeclarations(profileRef);
    if (failed(declarations))
      return emitError(UnknownLoc::get(context))
             << "physical call references missing gadget profile @"
             << profileRef.getValue();
    std::map<std::string, int64_t> rowOrdinals;
    for (Operation *declarationPointer : **declarations) {
      Operation &declaration = *declarationPointer;
      StringRef operationName = declaration.getName().getStringRef();
      StringRef kind;
      if (operationName == "fabric.success")
        kind = "selection";
      else
        continue;

      auto sourceAttr = declaration.getAttrOfType<ArrayAttr>("records");
      SmallVector<std::string> sources;
      if (sourceAttr)
        for (Attribute raw : sourceAttr) {
          auto source = dyn_cast<StringAttr>(raw);
          if (!source || source.getValue().empty())
            return declaration.emitOpError(
                "profile row has a malformed stable record");
          sources.push_back(source.getValue().str());
        }
      if (sources.empty())
        return declaration.emitOpError(
            "profile selection requires stable records");

      SmallVector<ArrayRef<std::string>> supports;
      size_t multiplicity = 1;
      for (const std::string &source : sources) {
        auto found = recordProjection.find({instance.str(), source});
        if (found == recordProjection.end() || found->second.records.empty())
          return declaration.emitOpError("stable record '")
                 << source << "' has no physical projection for instance '"
                 << instance << "'";
        supports.push_back(found->second.records);
        multiplicity = std::max(multiplicity, found->second.records.size());
      }
      for (ArrayRef<std::string> support : supports)
        if (support.size() != 1 && support.size() != multiplicity)
          return declaration.emitOpError(
              "profile row has incompatible folded record multiplicities");

      int64_t sourceRow = rowOrdinals[kind.str()]++;
      for (size_t occurrence = 0; occurrence < multiplicity; ++occurrence) {
        PendingSidecar pending;
        pending.kind = kind.str();
        pending.declaration = &declaration;
        pending.sourceProfile = profileRef;
        pending.sourceRow = sourceRow;
        pending.sourceInstance = instance.str();
        pending.sourceRecords = sources;
        for (ArrayRef<std::string> support : supports)
          pending.physicalRecords.push_back(
              support.size() == 1 ? support.front() : support[occurrence]);
        pendingSidecars.push_back(std::move(pending));
      }
    }
    return success();
  }

  bool profilesShareGadget(FlatSymbolRefAttr left, FlatSymbolRefAttr right,
                           FlatSymbolRefAttr leftCallee,
                           FlatSymbolRefAttr rightCallee) const {
    if (left == right)
      return true;
    if (!left || !right)
      return false;
    auto leftProfile = dyn_cast_or_null<qlx::fabric::GadgetProfileOp>(
        symbols.lookup(left.getValue()));
    auto rightProfile = dyn_cast_or_null<qlx::fabric::GadgetProfileOp>(
        symbols.lookup(right.getValue()));
    if (!leftProfile || !rightProfile ||
        leftProfile.getGadgetAttr() != leftCallee ||
        rightProfile.getGadgetAttr() != rightCallee)
      return false;
    // Profiles classify and post-process records; they do not change the
    // selected gadget's physical body.  A template invocation may therefore
    // choose another profile for the same gadget.  collectProfileSidecars()
    // below proves that every selected row resolves against the canonical
    // body's exact record projection, and the Phys verifier independently
    // authenticates those aliases.
    return leftProfile.getGadgetAttr() == rightProfile.getGadgetAttr();
  }

  StringRef callTemplateBucket(Operation *callee,
                               FlatSymbolRefAttr calleeRef) const {
    auto generatedBy =
        callee ? callee->getAttrOfType<FlatSymbolRefAttr>("generated_by")
               : FlatSymbolRefAttr{};
    auto inputP1Callee =
        callee ? callee->getAttrOfType<FlatSymbolRefAttr>("input_p1_callee")
               : FlatSymbolRefAttr{};
    if (inputP1Callee)
      return inputP1Callee.getValue();
    return generatedBy ? generatedBy.getValue() : calleeRef.getValue();
  }

  uint64_t physicalTemplateFingerprint(Operation *callable) {
    auto found = physicalTemplateFingerprints.find(callable);
    if (found != physicalTemplateFingerprints.end())
      return found->second;

    DenseSet<Operation *> active;
    std::function<uint64_t(Operation *)> compute =
        [&](Operation *current) -> uint64_t {
      if (!current)
        return 0;
      auto cached = physicalTemplateFingerprints.find(current);
      if (cached != physicalTemplateFingerprints.end())
        return cached->second;
      if (!active.insert(current).second)
        return 1;
      llvm::scope_exit clearActive([&] { active.erase(current); });

      uint64_t fingerprint = 0;
      auto combine = [&](auto value) {
        fingerprint =
            static_cast<size_t>(llvm::hash_combine(fingerprint, value));
      };
      std::function<void(Operation *, bool)> hashBody =
          [&](Operation *operation, bool isRoot) {
            combine(operation->getName().getStringRef());
            combine(operation->getNumOperands());
            for (Type type : operation->getOperandTypes())
              combine(type);
            combine(operation->getNumResults());
            for (Type type : operation->getResultTypes())
              combine(type);
            for (NamedAttribute attribute : operation->getAttrs()) {
              StringRef name = attribute.getName().getValue();
              if (isRoot &&
                  llvm::is_contained({SymbolTable::getSymbolAttrName(),
                                      StringRef("action_site"),
                                      StringRef("input_p1_kernel"),
                                      StringRef("input_p1_callee"),
                                      StringRef("input_p1_scope")},
                                     name))
                continue;
              if (isa<qlx::fabric::CallOp>(operation) && name == "callee")
                continue;
              if (isa<qlx::fabric::UnpackResourceOp>(operation) &&
                  name == "payload_logical_block_ids")
                continue;
              combine(name);
              combine(attribute.getValue());
            }
            combine(operation->getNumRegions());
            for (Region &region : operation->getRegions()) {
              combine(region.getBlocks().size());
              for (Block &block : region) {
                combine(block.getNumArguments());
                for (BlockArgument argument : block.getArguments())
                  combine(argument.getType());
                combine(block.getOperations().size());
                for (Operation &child : block)
                  hashBody(&child, false);
              }
            }
          };
      hashBody(current, true);

      SmallVector<qlx::fabric::CallOp> calls;
      current->walk([&](qlx::fabric::CallOp call) { calls.push_back(call); });
      for (qlx::fabric::CallOp call : calls) {
        Operation *target =
            SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr());
        fingerprint = static_cast<size_t>(
            llvm::hash_combine(fingerprint, compute(target)));
      }
      physicalTemplateFingerprints.try_emplace(current, fingerprint);
      return fingerprint;
    };
    return compute(callable);
  }

  std::optional<SmallVector<Attribute>>
  stateAliasesFor(const CanonicalCallTemplate &candidate, ValueRange inputs,
                  ArrayRef<Type> outputTypes, OpBuilder &builder) const {
    if (candidate.inputCount != inputs.size() ||
        candidate.outputCount != outputTypes.size())
      return std::nullopt;
    DenseMap<Attribute, Attribute> aliases;
    DenseMap<Attribute, Attribute> inverse;
    auto align = [&](Type templateType, Type invocationType) {
      if (templateType == invocationType)
        return true;
      auto templateState = dyn_cast<qlx::phys::StateType>(templateType);
      auto invocationState = dyn_cast<qlx::phys::StateType>(invocationType);
      if (!templateState || !invocationState)
        return false;
      FlatSymbolRefAttr templateResourceRef = templateState.getResource();
      FlatSymbolRefAttr invocationResourceRef = invocationState.getResource();
      auto templateInfo = resources.find(templateResourceRef);
      auto invocationInfo = resources.find(invocationResourceRef);
      if (templateInfo == resources.end() ||
          invocationInfo == resources.end() ||
          templateInfo->second.resourceClass !=
              invocationInfo->second.resourceClass)
        return false;
      auto [forward, inserted] =
          aliases.try_emplace(templateResourceRef, invocationResourceRef);
      if (!inserted && forward->second != invocationResourceRef)
        return false;
      auto [reverse, reverseInserted] =
          inverse.try_emplace(invocationResourceRef, templateResourceRef);
      return reverseInserted || reverse->second == templateResourceRef;
    };
    // Only the effectful physical boundary participates in template
    // compatibility.  Omitted positions were proven canonical pass-through
    // states and remain owned by this invocation's caller; scanning or
    // aliasing them would recreate the very O(full patch width) boundary that
    // the compact form removes.
    size_t retained = 0;
    for (int64_t index = candidate.retainedInputs.find_first(); index >= 0;
         index = candidate.retainedInputs.find_next(index))
      if (!align(candidate.retainedInputTypes[retained++],
                 inputs[index].getType()))
        return std::nullopt;
    retained = 0;
    for (int64_t index = candidate.retainedOutputs.find_first(); index >= 0;
         index = candidate.retainedOutputs.find_next(index))
      if (!align(candidate.retainedOutputTypes[retained++], outputTypes[index]))
        return std::nullopt;

    SmallVector<std::pair<FlatSymbolRefAttr, FlatSymbolRefAttr>> orderedAliases;
    orderedAliases.reserve(aliases.size());
    for (const auto &[templateResource, invocationResource] : aliases)
      orderedAliases.emplace_back(cast<FlatSymbolRefAttr>(templateResource),
                                  cast<FlatSymbolRefAttr>(invocationResource));
    llvm::sort(orderedAliases, [](const auto &left, const auto &right) {
      return left.first.getValue() < right.first.getValue();
    });
    SmallVector<Attribute> result;
    result.reserve(orderedAliases.size());
    for (const auto &[templateResource, invocationResource] : orderedAliases)
      result.push_back(builder.getDictionaryAttr({
          builder.getNamedAttr("template", templateResource),
          builder.getNamedAttr("alias", invocationResource),
      }));
    return result;
  }

  static bool
  supportsElidedStateBoundary(const CanonicalCallTemplate &candidate) {
    return !candidate.retainedInputTypes.empty() &&
           candidate.retainedInputTypes == candidate.retainedOutputTypes &&
           llvm::all_of(candidate.retainedInputTypes, [](Type type) {
             return isa<qlx::phys::StateType>(type);
           });
  }

  static bool exactGroupLayoutsFor(
      const CanonicalCallTemplate &candidate,
      ArrayRef<PhysicalGroup> inputGroups,
      ArrayRef<PhysicalGroup::LayoutIdentity> outputGroupLayouts) {
    if (!supportsElidedStateBoundary(candidate) ||
        candidate.inputGroupLayouts.size() != inputGroups.size() ||
        candidate.outputGroupLayouts.size() != outputGroupLayouts.size())
      return false;
    for (auto [expected, actual] :
         llvm::zip(candidate.inputGroupLayouts, inputGroups))
      if (expected != actual.getLayoutIdentity())
        return false;
    return llvm::equal(candidate.outputGroupLayouts, outputGroupLayouts);
  }

  static StateAliasCacheKey stateAliasCacheKeyFor(
      size_t templateIndex, ArrayRef<PhysicalGroup> inputGroups,
      ArrayRef<PhysicalGroup::LayoutIdentity> outputGroupLayouts) {
    StateAliasCacheKey key;
    key.templateIndex = templateIndex;
    key.layouts.reserve(inputGroups.size() + outputGroupLayouts.size() + 1);
    key.layouts.push_back(inputGroups.size());
    for (const PhysicalGroup &group : inputGroups)
      key.layouts.push_back(
          reinterpret_cast<uintptr_t>(group.getLayoutIdentity().get()));
    for (const PhysicalGroup::LayoutIdentity &layout : outputGroupLayouts)
      key.layouts.push_back(reinterpret_cast<uintptr_t>(layout.get()));
    return key;
  }

  LogicalResult
  appendCallTemplateAttributes(const CanonicalCallTemplate &candidate,
                               FlatSymbolRefAttr invocationCallee,
                               FlatSymbolRefAttr profileRef,
                               StringAttr instance, ArrayAttr stateAliases,
                               SmallVectorImpl<NamedAttribute> &attrs,
                               OpBuilder &builder, Location location) {
    SmallVector<Attribute> recordAliases;
    if (profileRef) {
      std::set<std::string> selectedRecordSources;
      auto declarations = profileSidecarDeclarations(profileRef);
      if (failed(declarations)) {
        emitError(location) << "references an unresolved gadget profile";
        return failure();
      }
      for (Operation *declaration : **declarations) {
        if (auto records = declaration->getAttrOfType<ArrayAttr>("records"))
          for (Attribute raw : records)
            if (auto record = dyn_cast<StringAttr>(raw))
              selectedRecordSources.insert(record.getValue().str());
      }
      for (const std::string &sourceRecord : selectedRecordSources) {
        auto selected = candidate.recordProjection.find(sourceRecord);
        if (selected == candidate.recordProjection.end())
          continue;
        auto &aliases =
            mutateRecordProjection({instance.getValue().str(), sourceRecord});
        for (const std::string &templateRecord : selected->second) {
          std::string alias =
              llvm::formatv("{0}.template_record{1}", instance.getValue(),
                            templateRecordIdentity++)
                  .str();
          aliases.push_back(alias);
          recordAliases.push_back(builder.getDictionaryAttr({
              builder.getNamedAttr("alias", stringAttr(context, alias)),
              builder.getNamedAttr("template",
                                   stringAttr(context, templateRecord)),
          }));
        }
      }
    }
    attrs.push_back(builder.getNamedAttr("template_event", candidate.event));
    if (!stateAliases.empty())
      attrs.push_back(builder.getNamedAttr("state_aliases", stateAliases));
    if (candidate.callee != invocationCallee) {
      SmallVector<Attribute> substitutions;
      substitutions.reserve(stateAliases.size());
      for (Attribute raw : stateAliases) {
        auto alias = cast<DictionaryAttr>(raw);
        substitutions.push_back(builder.getDictionaryAttr({
            builder.getNamedAttr("from",
                                 alias.getAs<FlatSymbolRefAttr>("template")),
            builder.getNamedAttr("to", alias.getAs<FlatSymbolRefAttr>("alias")),
        }));
      }
      if (!substitutions.empty())
        attrs.push_back(builder.getNamedAttr(
            "resource_substitutions", builder.getArrayAttr(substitutions)));
    }
    if (!recordAliases.empty())
      attrs.push_back(builder.getNamedAttr(
          "record_aliases", builder.getArrayAttr(recordAliases)));
    return success();
  }

  bool physicallyTemplateEquivalent(Operation *left, Operation *right) {
    if (left == right)
      return true;
    auto key = std::make_pair(left, right);
    auto cached = physicalTemplateEquivalenceCache.find(key);
    if (cached != physicalTemplateEquivalenceCache.end()) {
      ++templateEquivalenceCacheHits;
      return cached->second;
    }
    bool equivalent = qlx::fabric::arePhysicallyTemplateEquivalent(left, right);
    physicalTemplateEquivalenceCache.try_emplace(key, equivalent);
    ++templateEquivalenceCacheMisses;
    return equivalent;
  }

  bool hasSourceDataDependency(qlx::fabric::CallOp call,
                               Operation *sourceCall) {
    if (!sourceCall || sourceCall == call.getOperation())
      return false;
    auto source = sourceDataDependencySources.find(sourceCall);
    if (source == sourceDataDependencySources.end())
      return false;
    auto dependenciesFor = [&](Value root) -> const SmallVector<uint64_t, 1> & {
      SmallVector<Value, 16> pending{root};
      while (!pending.empty()) {
        Value value = pending.back();
        auto cached = sourceDataDependencies.find(value);
        if (cached != sourceDataDependencies.end()) {
          ++sourceDataDependencyCacheHits;
          pending.pop_back();
          continue;
        }
        ++sourceDataDependencyVisits;
        Operation *definition = value.getDefiningOp();
        if (!definition) {
          sourceDataDependencies.try_emplace(value);
          pending.pop_back();
          continue;
        }
        bool queued = false;
        for (Value operand : definition->getOperands()) {
          if (!sourceDataDependencies.count(operand)) {
            pending.push_back(operand);
            queued = true;
            break;
          }
        }
        if (queued)
          continue;
        SmallVector<uint64_t, 1> dependencies;
        for (Value operand : definition->getOperands()) {
          const auto &operandDependencies =
              sourceDataDependencies.find(operand)->second;
          if (dependencies.size() < operandDependencies.size())
            dependencies.resize(operandDependencies.size());
          for (auto [word, bits] : llvm::enumerate(operandDependencies))
            dependencies[word] |= bits;
        }
        auto ownSource = sourceDataDependencySources.find(definition);
        if (ownSource != sourceDataDependencySources.end()) {
          size_t word = ownSource->second / 64;
          if (dependencies.size() <= word)
            dependencies.resize(word + 1);
          dependencies[word] |= uint64_t{1} << (ownSource->second % 64);
        }
        sourceDataDependencies.try_emplace(value, std::move(dependencies));
        pending.pop_back();
      }
      return sourceDataDependencies.find(root)->second;
    };
    size_t word = source->second / 64;
    uint64_t mask = uint64_t{1} << (source->second % 64);
    return llvm::any_of(call.getOperands(), [&](Value operand) {
      const auto &dependencies = dependenciesFor(operand);
      return word < dependencies.size() && (dependencies[word] & mask);
    });
  }

  bool allocationPlanMatches(const CanonicalCallTemplate &candidate,
                             ArrayRef<Attribute> stateAliases,
                             bool preferExactReuse) const {
    if (candidate.allocationPlan.empty())
      return true;
    using ResourceIdentity = std::pair<std::string, int64_t>;
    std::map<ResourceIdentity, ResourceIdentity> aliases;
    for (Attribute rawAlias : stateAliases) {
      auto alias = dyn_cast<DictionaryAttr>(rawAlias);
      auto templateResource = alias ? alias.getAs<FlatSymbolRefAttr>("template")
                                    : FlatSymbolRefAttr{};
      auto invocationResource =
          alias ? alias.getAs<FlatSymbolRefAttr>("alias") : FlatSymbolRefAttr{};
      if (!templateResource || !invocationResource)
        return false;
      auto templateInfo = resources.find(templateResource);
      auto invocationInfo = resources.find(invocationResource);
      if (templateInfo == resources.end() || invocationInfo == resources.end())
        return false;
      aliases[{templateInfo->second.resourceClass,
               templateInfo->second.index}] = {
          invocationInfo->second.resourceClass, invocationInfo->second.index};
    }

    auto mappedIdentity = [&](StringRef resourceClass,
                              int64_t index) -> ResourceIdentity {
      ResourceIdentity identity{resourceClass.str(), index};
      auto found = aliases.find(identity);
      return found == aliases.end() ? identity : found->second;
    };
    std::map<std::string, std::set<int64_t>> localActive;
    std::map<std::string, std::set<int64_t>> localEver;
    std::map<std::string, std::set<int64_t>> releasedBase;
    for (const AllocationPlanStep &step : candidate.allocationPlan) {
      SmallVector<ResourceIdentity> expected;
      expected.reserve(step.indices.size());
      for (int64_t index : step.indices)
        expected.push_back(mappedIdentity(step.resourceClass, index));
      if (llvm::any_of(expected, [&](const ResourceIdentity &identity) {
            return identity.first != step.resourceClass;
          }))
        return false;

      std::set<int64_t> &active = localActive[step.resourceClass];
      std::set<int64_t> &ever = localEver[step.resourceClass];
      auto baseActive = activeIndices.find(step.resourceClass);
      auto baseEver = everAllocatedIndices.find(step.resourceClass);
      auto isActive = [&](int64_t index) {
        return active.count(index) ||
               (baseActive != activeIndices.end() &&
                baseActive->second.count(index) &&
                !releasedBase[step.resourceClass].count(index));
      };
      auto wasEverAllocated = [&](int64_t index) {
        return ever.count(index) || (baseEver != everAllocatedIndices.end() &&
                                     baseEver->second.count(index));
      };
      if (!step.acquire) {
        for (const ResourceIdentity &identity : expected) {
          if (active.erase(identity.second))
            continue;
          if (baseActive == activeIndices.end() ||
              !baseActive->second.count(identity.second) ||
              !releasedBase[step.resourceClass].insert(identity.second).second)
            return false;
        }
        continue;
      }

      SmallVector<int64_t> selected;
      if (preferExactReuse) {
        for (const ResourceIdentity &identity : expected) {
          if (identity.second < 0 || identity.second >= step.capacity ||
              isActive(identity.second) ||
              llvm::is_contained(selected, identity.second))
            return false;
          selected.push_back(identity.second);
        }
      } else {
        for (int64_t index = 0;
             index < step.capacity && selected.size() < step.indices.size();
             ++index)
          if (!isActive(index) && !wasEverAllocated(index))
            selected.push_back(index);
        for (int64_t index = 0;
             index < step.capacity && selected.size() < step.indices.size();
             ++index)
          if (!isActive(index) && !llvm::is_contained(selected, index))
            selected.push_back(index);
      }
      if (selected.size() != expected.size())
        return false;
      for (auto [selectedIndex, expectedIdentity] :
           llvm::zip(selected, expected))
        if (selectedIndex != expectedIdentity.second)
          return false;
      active.insert(selected.begin(), selected.end());
      ever.insert(selected.begin(), selected.end());
    }
    return llvm::all_of(localActive,
                        [](const auto &entry) { return entry.second.empty(); });
  }

  LogicalResult replayAllocationPlan(const CanonicalCallTemplate &candidate,
                                     ArrayRef<Attribute> stateAliases,
                                     StringRef completionEvent) {
    using ResourceIdentity = std::pair<std::string, int64_t>;
    std::map<ResourceIdentity, ResourceIdentity> aliases;
    for (Attribute rawAlias : stateAliases) {
      auto alias = dyn_cast<DictionaryAttr>(rawAlias);
      auto templateResource = alias ? alias.getAs<FlatSymbolRefAttr>("template")
                                    : FlatSymbolRefAttr{};
      auto invocationResource =
          alias ? alias.getAs<FlatSymbolRefAttr>("alias") : FlatSymbolRefAttr{};
      auto templateInfo =
          templateResource ? resources.find(templateResource) : resources.end();
      auto invocationInfo = invocationResource
                                ? resources.find(invocationResource)
                                : resources.end();
      if (templateInfo == resources.end() || invocationInfo == resources.end())
        return failure();
      aliases[{templateInfo->second.resourceClass,
               templateInfo->second.index}] = {
          invocationInfo->second.resourceClass, invocationInfo->second.index};
    }
    auto mappedIdentity = [&](StringRef resourceClass,
                              int64_t index) -> ResourceIdentity {
      ResourceIdentity identity{resourceClass.str(), index};
      auto found = aliases.find(identity);
      return found == aliases.end() ? identity : found->second;
    };

    std::set<ResourceIdentity> acquiredHere;
    for (const AllocationPlanStep &step : candidate.allocationPlan) {
      SmallVector<ResourceIdentity> identities;
      identities.reserve(step.indices.size());
      SmallVector<int64_t> replayedIndices;
      replayedIndices.reserve(step.indices.size());
      for (int64_t index : step.indices) {
        ResourceIdentity identity = mappedIdentity(step.resourceClass, index);
        if (identity.first != step.resourceClass)
          return failure();
        identities.push_back(identity);
        replayedIndices.push_back(identity.second);
      }

      if (step.acquire) {
        for (const ResourceIdentity &identity : identities) {
          if (!activeIndices[identity.first].insert(identity.second).second)
            return failure();
          everAllocatedIndices[identity.first].insert(identity.second);
          acquiredHere.insert(identity);
        }
      } else {
        auto active = activeIndices.find(step.resourceClass);
        if (active == activeIndices.end() ||
            llvm::any_of(identities, [&](const ResourceIdentity &identity) {
              return !active->second.count(identity.second);
            }))
          return failure();

        bool releasesLocal =
            llvm::all_of(identities, [&](const ResourceIdentity &identity) {
              return acquiredHere.count(identity);
            });
        if (!releasesLocal &&
            llvm::any_of(identities, [&](const ResourceIdentity &identity) {
              return acquiredHere.count(identity);
            }))
          return failure();

        if (releasesLocal) {
          for (const ResourceIdentity &identity : identities) {
            active->second.erase(identity.second);
            acquiredHere.erase(identity);
          }
        } else {
          // One allocation record owns every carrier in a patch. Validate the
          // complete release before mutating it, then retire the record once
          // rather than once per carrier.
          std::optional<size_t> allocationIndex;
          SmallVector<FlatSymbolRefAttr> resourceReferences;
          resourceReferences.reserve(identities.size());
          for (const ResourceIdentity &identity : identities) {
            auto resource = resourceByIdentity.find(identity);
            if (resource == resourceByIdentity.end())
              return failure();
            auto allocation =
                resourceToAllocation.find(resource->second.getValue());
            if (allocation == resourceToAllocation.end() ||
                (allocationIndex && *allocationIndex != allocation->second))
              return failure();
            allocationIndex = allocation->second;
            resourceReferences.push_back(resource->second);
          }
          if (!allocationIndex)
            return failure();
          AllocationRecord &record = allocations[*allocationIndex];
          if (!record.release.empty())
            return failure();

          for (const ResourceIdentity &identity : identities)
            active->second.erase(identity.second);
          record.release = completionEvent.str();
          for (auto [identity, resource] :
               llvm::zip(identities, resourceReferences)) {
            lastReleaseByBinding[identity] = record.release;
            resourceToAllocation.erase(resource.getValue());
          }
        }
      }
      allocationPlanTrace.push_back(
          AllocationPlanStep{step.acquire, step.resourceClass, step.capacity,
                             std::move(replayedIndices)});
    }
    return acquiredHere.empty() ? success() : failure();
  }

  void emitAnalysisSidecars(OpBuilder &builder, Location location) {
    std::map<std::tuple<std::string, std::string, std::string>, int64_t>
        projectionIndices;
    SmallVector<Attribute> projectionEntries;
    int64_t sidecarIdentity = 0;
    std::string projectionName = graphName + "_record_projection";
    for (const PendingSidecar &pending : pendingSidecars) {
      SmallVector<Attribute> records = llvm::map_to_vector(
          pending.physicalRecords, [&](const std::string &record) {
            return Attribute(stringAttr(context, record));
          });
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr(
              SymbolTable::getSymbolAttrName(),
              stringAttr(context, llvm::formatv("{0}_{1}{2}", graphName,
                                                pending.kind, sidecarIdentity++)
                                      .str())),
          builder.getNamedAttr("graph", symbolAttr(context, graphName)),
          builder.getNamedAttr("records", builder.getArrayAttr(records)),
      };
      Operation *declaration = pending.declaration;
      if (pending.sourceProfile) {
        SmallVector<Attribute> sources = llvm::map_to_vector(
            pending.sourceRecords, [&](const std::string &record) {
              return Attribute(stringAttr(context, record));
            });
        SmallVector<int64_t> indices;
        for (auto [source, physical] :
             llvm::zip(pending.sourceRecords, pending.physicalRecords)) {
          auto key = std::tuple{pending.sourceInstance, source, physical};
          auto [found, inserted] = projectionIndices.try_emplace(
              key, static_cast<int64_t>(projectionEntries.size()));
          if (inserted) {
            SmallVector<NamedAttribute> projectionAttrs{
                builder.getNamedAttr(
                    "instance", stringAttr(context, pending.sourceInstance)),
                builder.getNamedAttr("source_record",
                                     stringAttr(context, source)),
                builder.getNamedAttr("physical_record",
                                     stringAttr(context, physical)),
            };
            auto repeat = recordProjectionRepeats.find(key);
            if (repeat != recordProjectionRepeats.end() &&
                !repeat->second.empty()) {
              SmallVector<Attribute> events;
              SmallVector<int64_t> counts;
              for (const auto &[event, count] : repeat->second) {
                events.push_back(stringAttr(context, event));
                counts.push_back(count);
              }
              projectionAttrs.push_back(builder.getNamedAttr(
                  "repeat_events", builder.getArrayAttr(events)));
              projectionAttrs.push_back(builder.getNamedAttr(
                  "repeat_counts", builder.getDenseI64ArrayAttr(counts)));
            }
            projectionEntries.push_back(
                builder.getDictionaryAttr(projectionAttrs));
          }
          indices.push_back(found->second);
        }
        attrs.push_back(
            builder.getNamedAttr("source_profile", pending.sourceProfile));
        attrs.push_back(builder.getNamedAttr("source_kind",
                                             stringAttr(context, "profile")));
        attrs.push_back(builder.getNamedAttr(
            "source_row", builder.getI64IntegerAttr(pending.sourceRow)));
        attrs.push_back(builder.getNamedAttr(
            "source_instance", stringAttr(context, pending.sourceInstance)));
        attrs.push_back(builder.getNamedAttr(
            "record_projection", symbolAttr(context, projectionName)));
        attrs.push_back(builder.getNamedAttr(
            "projection_indices", builder.getDenseI64ArrayAttr(indices)));
        attrs.push_back(builder.getNamedAttr("source_records",
                                             builder.getArrayAttr(sources)));
      }
      if (declaration) {
        for (StringRef name :
             {StringRef("input_syndromes"), StringRef("constant"),
              StringRef("label"), StringRef("expected"), StringRef("scope")})
          if (Attribute attribute = declaration->getAttr(name))
            attrs.push_back(builder.getNamedAttr(name, attribute));
      }
      if (!declaration || !declaration->hasAttr("input_syndromes"))
        attrs.push_back(
            builder.getNamedAttr("input_syndromes", builder.getArrayAttr({})));
      if (!declaration || !declaration->hasAttr("constant"))
        attrs.push_back(
            builder.getNamedAttr("constant", builder.getBoolAttr(false)));
      if (pending.kind == "selection") {
        llvm::erase_if(attrs, [](NamedAttribute attribute) {
          return attribute.getName().strref() == "expected";
        });
        attrs.push_back(
            builder.getNamedAttr("expected", builder.getBoolAttr(false)));
      }
      std::string sidecarName = "phys." + pending.kind + "_sidecar";
      createGeneric(builder, location, sidecarName, {}, {}, attrs);
    }
    if (!projectionEntries.empty())
      createGeneric(
          builder, location, "phys.record_projection", {}, {},
          {
              builder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                   stringAttr(context, projectionName)),
              builder.getNamedAttr("graph", symbolAttr(context, graphName)),
              builder.getNamedAttr("source_protocol",
                                   symbolAttr(context, rootName)),
              builder.getNamedAttr("entries",
                                   builder.getArrayAttr(projectionEntries)),
          });
  }

  FailureOr<TransformFrameProjection> frameProjection(Value value) {
    if (auto direct = transformFrames.find(value);
        direct != transformFrames.end())
      return direct->second;

    // A large WSC frame is an ordinary SSA chain thousands of operations long.
    // Recursive provenance discovery overflows the native stack and repeats
    // prefixes for later queries. Evaluate the same DAG relation explicitly in
    // postorder and memoize every derived frame origin.
    struct WorkItem {
      Value value;
      bool expanded;
    };
    SmallVector<WorkItem, 32> work{{value, false}};
    DenseSet<Value> active;
    while (!work.empty()) {
      WorkItem item = work.pop_back_val();
      if (transformFrames.contains(item.value))
        continue;
      Operation *producer = item.value.getDefiningOp();
      if (!producer)
        return failure();
      if (!item.expanded) {
        if (!active.insert(item.value).second)
          return failure();
        work.push_back({item.value, true});
        bool hasFrameOperand = false;
        for (Value operand : llvm::reverse(producer->getOperands())) {
          if (!isa<qlx::fabric::PatchFrameType>(operand.getType()))
            continue;
          hasFrameOperand = true;
          if (!transformFrames.contains(operand))
            work.push_back({operand, false});
        }
        if (!hasFrameOperand)
          return failure();
        continue;
      }

      active.erase(item.value);
      std::optional<TransformFrameProjection> origin;
      for (Value operand : producer->getOperands()) {
        if (!isa<qlx::fabric::PatchFrameType>(operand.getType()))
          continue;
        auto candidate = transformFrames.find(operand);
        if (candidate == transformFrames.end())
          return failure();
        if (origin && origin->transform != candidate->second.transform)
          return failure();
        origin = candidate->second;
      }
      if (!origin)
        return failure();
      transformFrames[item.value] = *origin;
    }
    return transformFrames.lookup(value);
  }

  FailureOr<PhysicalGroup> beginTransform(qlx::fabric::TransformBeginOp begin,
                                          PhysicalGroup source) {
    auto transform = dyn_cast_or_null<qlx::fabric::PatchTransformOp>(
        symbols.lookup(begin.getTransform()));
    if (!transform)
      return failure();
    auto sourceData = partitionRange(begin.getSource().getType(), "data");
    auto frameWidth = patchWidth(begin.getFrame().getType());
    if (failed(sourceData) || failed(frameWidth))
      return failure();
    auto [dataOffset, dataCount] = *sourceData;
    ArrayRef<int64_t> support = transform.getSourceSupport();
    if (dataCount != int64_t(support.size()) || dataOffset < 0 ||
        dataOffset + dataCount > int64_t(source.size()))
      return failure();

    SmallVector<std::optional<Value>> frame(*frameWidth);
    for (auto [ordinal, index] : llvm::enumerate(support)) {
      if (index < 0 || index >= *frameWidth || frame[index])
        return failure();
      frame[index] = source[dataOffset + ordinal];
    }
    SmallVector<Value> remaining;
    for (auto [index, value] : llvm::enumerate(source))
      if (index < size_t(dataOffset) || index >= size_t(dataOffset + dataCount))
        remaining.push_back(value);
    size_t cursor = 0;
    PhysicalGroup exposed;
    exposed.reserve(*frameWidth);
    for (std::optional<Value> &value : frame) {
      if (!value) {
        if (cursor >= remaining.size())
          return failure();
        value = remaining[cursor++];
      }
      exposed.push_back(*value);
    }
    transformFrames[begin.getFrame()] =
        TransformFrameProjection{std::move(source), begin.getTransformAttr()};
    return exposed;
  }

  FailureOr<PhysicalGroup> endTransform(qlx::fabric::TransformEndOp end,
                                        PhysicalGroup frame) {
    auto origin = frameProjection(end.getFrame());
    if (failed(origin) || origin->transform != end.getTransformAttr())
      return failure();
    auto transform = dyn_cast_or_null<qlx::fabric::PatchTransformOp>(
        symbols.lookup(end.getTransform()));
    if (!transform)
      return failure();

    std::map<std::string, Value> current;
    for (Value value : frame) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      if (!state)
        return failure();
      current[state.getResource().getValue().str()] = value;
    }
    PhysicalGroup restored;
    restored.reserve(origin->source.size());
    for (Value value : origin->source) {
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      if (!state)
        return failure();
      auto replacement = current.find(state.getResource().getValue().str());
      restored.push_back(replacement == current.end() ? value
                                                      : replacement->second);
    }

    ArrayRef<int64_t> support = transform.getDestinationSupport();
    auto destinationPartitions =
        orderedPartitionSizes(end.getDestination().getType());
    if (failed(destinationPartitions))
      return failure();
    auto data =
        llvm::find_if(*destinationPartitions, [](const auto &partition) {
          return partition.first == "data";
        });
    if (data == destinationPartitions->end() ||
        data->second != int64_t(support.size()))
      return failure();

    PhysicalGroup destinationData;
    std::set<std::string> dataResources;
    for (int64_t index : support) {
      if (index < 0 || index >= int64_t(frame.size()))
        return failure();
      Value value = frame[index];
      auto state = dyn_cast<qlx::phys::StateType>(value.getType());
      if (!state)
        return failure();
      destinationData.push_back(value);
      dataResources.insert(state.getResource().getValue().str());
    }
    PhysicalGroup remainder;
    for (Value value : restored) {
      auto state = cast<qlx::phys::StateType>(value.getType());
      if (!dataResources.count(state.getResource().getValue().str()))
        remainder.push_back(value);
    }

    PhysicalGroup destination;
    size_t cursor = 0;
    for (const auto &[name, count] : *destinationPartitions) {
      if (name == "data") {
        llvm::append_range(destination, destinationData);
        continue;
      }
      if (count < 0 || cursor + size_t(count) > remainder.size())
        return failure();
      destination.append(remainder.begin() + cursor,
                         remainder.begin() + cursor + count);
      cursor += count;
    }
    return destination;
  }

  LogicalResult emitBlock(Block &block, ProjectedValues &projected,
                          OpBuilder &builder) {
    int64_t operationOrdinal = 0;
    for (Operation &operation : block.without_terminator()) {
      int64_t sourceOrdinal = operationOrdinal++;
      if (operation.getName().getStringRef() == "arith.constant") {
        auto resultType = physicalScalarType(operation.getResult(0).getType());
        if (failed(resultType))
          return operation.emitOpError(
              "has no supported physical scalar projection");
        SmallVector<NamedAttribute> attrs(operation.getAttrs().begin(),
                                          operation.getAttrs().end());
        Operation *physical =
            createGeneric(builder, operation.getLoc(), "arith.constant", {},
                          TypeRange{*resultType}, attrs);
        projected[operation.getResult(0)] =
            PhysicalGroup{physical->getResult(0)};
        continue;
      }
      if (isa<qlx::fabric::SuccessOp>(operation)) {
        continue;
      }
      if (auto produced = dyn_cast<qlx::fabric::ProduceResourceOp>(operation)) {
        auto resultType =
            physicalScalarType(produced.getResourceState().getType());
        auto kind = resourceKind(cast<qlx::fabric::ResourceStateType>(
            produced.getResourceState().getType()));
        Attribute protocol = produced->getAttr("protocol");
        if (failed(resultType) || failed(kind) || !protocol)
          return produced.emitOpError(
              "requires a symbolic resource kind and selected protocol");
        FlatSymbolRefAttr selectedKind = produced.getResourceKindAttr();
        if (!selectedKind)
          selectedKind = *kind;
        Operation *physical = createGeneric(
            builder, produced.getLoc(), "phys.produce_resource", {},
            TypeRange{*resultType},
            {
                builder.getNamedAttr("region", produced.getRegionAttr()),
                builder.getNamedAttr("resource_kind", selectedKind),
                builder.getNamedAttr("protocol", protocol),
                builder.getNamedAttr("event_id", nextEvent("produce_resource")),
            });
        projected[produced.getResourceState()] =
            PhysicalGroup{physical->getResult(0)};
        continue;
      }
      if (auto request = dyn_cast<qlx::fabric::ResourceRequestOp>(operation)) {
        auto resultType = physicalScalarType(request.getEvent().getType());
        if (failed(resultType))
          return request.emitOpError("has an unsupported event payload type");

        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("kind", request.getKindAttr()),
            builder.getNamedAttr("stream", request.getStreamAttr()),
            builder.getNamedAttr("event_id", nextEvent("resource_request")),
        };
        auto stream = dyn_cast_or_null<qlx::lvm::StreamOp>(
            SymbolTable::lookupNearestSymbolFrom(request,
                                                 request.getStreamAttr()));
        if (stream && stream.getExternalAttr()) {
          attrs.push_back(
              builder.getNamedAttr("external", builder.getUnitAttr()));
        } else if (stream && stream.getProducedByAttr()) {
          FlatSymbolRefAttr providerRef = stream.getProducedByAttr();
          auto provider = dyn_cast_or_null<qlx::fabric::ProtocolOp>(
              symbols.lookup(providerRef.getValue()));
          if (!provider)
            return request.emitOpError(
                "has no retained typed resource-provider protocol");
          attrs.push_back(builder.getNamedAttr("provider", providerRef));
          {
            FlatSymbolRefAttr region = stream.getBackingRegionAttr();
            if (!region)
              return request.emitOpError(
                  "backed resource stream has no selected QEC region");
            auto binding = regionBindings.find(region.getValue());
            if (binding == regionBindings.end())
              return request.emitOpError(
                  "backed resource stream has no physical QEC binding");

            auto selectedModel = factoryModelFor(request);
            if (failed(selectedModel))
              return failure();
            if (*selectedModel) {
              SymbolRefAttr physicalBinding = SymbolRefAttr::get(
                  context, architecture.getSymName(),
                  {FlatSymbolRefAttr::get(
                      context,
                      (*selectedModel).getQecBindingAttr().getValue())});
              attrs.push_back(builder.getNamedAttr(
                  "factory_model",
                  symbolAttr(context, (*selectedModel).getSymName())));
              attrs.push_back(builder.getNamedAttr(
                  "region", (*selectedModel).getRegionAttr()));
              attrs.push_back(
                  builder.getNamedAttr("physical_binding", physicalBinding));
              if (auto transfer = stream.getTransferAttr())
                attrs.push_back(builder.getNamedAttr("transfer", transfer));

              Operation *physical = createGeneric(
                  builder, request.getLoc(), "phys.resource_request", {},
                  TypeRange{*resultType}, attrs);
              projected[request.getEvent()] =
                  PhysicalGroup{physical->getResult(0)};
              continue;
            }

            DictionaryAttr metadata =
                provider->getAttrOfType<DictionaryAttr>("metadata");
            auto factoryMode = metadata
                                   ? metadata.getAs<StringAttr>("factory_mode")
                                   : StringAttr{};
            auto cyclesText =
                metadata ? metadata.getAs<StringAttr>("cycles_per_attempt")
                         : StringAttr{};
            auto depthText = metadata
                                 ? metadata.getAs<StringAttr>("pipeline_depth")
                                 : StringAttr{};
            auto acceptanceText =
                metadata ? metadata.getAs<StringAttr>("acceptance_probability")
                         : StringAttr{};
            auto footprintText =
                metadata ? metadata.getAs<StringAttr>("physical_qubits")
                         : StringAttr{};
            double cycles = 0.0;
            double acceptance = 0.0;
            int64_t depth = 0;
            int64_t footprint = 0;
            if (!factoryMode || factoryMode.getValue() != "scheduled_macro" ||
                !cyclesText || cyclesText.getValue().getAsDouble(cycles) ||
                !depthText || depthText.getValue().getAsInteger(10, depth) ||
                !acceptanceText ||
                acceptanceText.getValue().getAsDouble(acceptance) ||
                !footprintText ||
                footprintText.getValue().getAsInteger(10, footprint) ||
                !std::isfinite(cycles) || cycles <= 0.0 || depth <= 0 ||
                !std::isfinite(acceptance) || acceptance <= 0.0 ||
                acceptance > 1.0 || footprint <= 0)
              return request.emitOpError(
                  "scheduled resource provider has invalid or incomplete "
                  "factory evidence");
            if (!hasExplicitCycleNanoseconds ||
                !std::isfinite(cycleNanoseconds) || cycleNanoseconds <= 0.0)
              return request.emitOpError(
                  "scheduled resource model requires an explicit positive "
                  "cycle_ns or surface_cycle_ns operating-point fact");
            if (binding->second.capacity != footprint)
              return request.emitOpError(
                         "scheduled resource provider physical binding has ")
                     << binding->second.capacity << " qubits but requires "
                     << footprint
                     << " qubits (exactly one engine footprint); pooled or "
                        "multi-engine homes are unsupported";

            double attemptDuration = cycles * cycleNanoseconds;
            double duration =
                attemptDuration / (static_cast<double>(depth) * acceptance);
            SymbolRefAttr physicalBinding = SymbolRefAttr::get(
                context, architecture.getSymName(),
                {FlatSymbolRefAttr::get(context,
                                        binding->second.physicalBinding)});
            attrs.push_back(builder.getNamedAttr("region", region));
            attrs.push_back(
                builder.getNamedAttr("physical_binding", physicalBinding));
            attrs.push_back(builder.getNamedAttr(
                "duration_ns", builder.getF64FloatAttr(duration)));
            attrs.push_back(
                builder.getNamedAttr("factory_attempt_duration_ns",
                                     builder.getF64FloatAttr(attemptDuration)));
            attrs.push_back(
                builder.getNamedAttr("factory_acceptance_probability",
                                     builder.getF64FloatAttr(acceptance)));
            attrs.push_back(builder.getNamedAttr(
                "factory_pipeline_depth", builder.getI64IntegerAttr(depth)));
            attrs.push_back(builder.getNamedAttr("factory_mode", factoryMode));
          }
          if (auto transfer = stream.getTransferAttr())
            attrs.push_back(builder.getNamedAttr("transfer", transfer));
        }
        Operation *physical =
            createGeneric(builder, request.getLoc(), "phys.resource_request",
                          {}, TypeRange{*resultType}, attrs);
        projected[request.getEvent()] = PhysicalGroup{physical->getResult(0)};
        continue;
      }
      if (auto await = dyn_cast<qlx::event::AwaitOp>(operation)) {
        PhysicalGroup event = projected[await.getEvent()];
        auto resultType = physicalScalarType(await.getPayload().getType());
        if (event.size() != 1 || failed(resultType))
          return await.emitOpError("has an unsupported physical event payload");
        Operation *physical = createGeneric(
            builder, await.getLoc(), "event.await", event.values(),
            TypeRange{*resultType},
            {builder.getNamedAttr("event_id", nextEvent("await"))});
        projected[await.getPayload()] = PhysicalGroup{physical->getResult(0)};
        continue;
      }
      if (auto unpack = dyn_cast<qlx::fabric::UnpackResourceOp>(operation)) {
        size_t count = unpack.getAnchors().size();
        if (count == 0 || unpack.getOutputs().size() != count * 2)
          return unpack.emitOpError(
              "requires one successor and one payload per anchor patch");
        PhysicalGroup resource =
            takeProjected(projected, unpack.getResource(), unpack);
        if (resource.size() != 1 ||
            !isa<qlx::phys::ResourcePayloadType>(resource.front().getType()))
          return unpack.emitOpError(
              "has no exact projected physical resource payload");

        SmallVector<PhysicalGroup> anchors;
        SmallVector<PhysicalGroup> payloads;
        SmallVector<Value> payloadStates;
        SmallVector<int64_t> segments{0};
        FlatSymbolRefAttr encoding;
        anchors.reserve(count);
        payloads.reserve(count);
        for (size_t index = 0; index < count; ++index) {
          PhysicalGroup anchor =
              takeProjected(projected, unpack.getAnchors()[index], unpack);
          auto region = projectedRegion(anchor);
          auto patch = dyn_cast<qlx::fabric::PatchType>(
              unpack.getOutputs()[count + index].getType());
          if (failed(region) || !patch || !patch.getEncoding())
            return unpack.emitOpError(
                "cannot derive the payload encoding and exact anchor home");
          if (encoding && encoding != patch.getEncoding())
            return unpack.emitOpError(
                "requires one common encoding for every payload patch");
          encoding = patch.getEncoding();
          auto payload =
              allocatePatch(patch, *region, unpack.getLoc(), builder);
          if (failed(payload))
            return failure();
          llvm::append_range(payloadStates, payload->values());
          segments.push_back(payloadStates.size());
          anchors.push_back(std::move(anchor));
          payloads.push_back(std::move(*payload));
        }

        SmallVector<Value> physicalInputs{resource.front()};
        llvm::append_range(physicalInputs, payloadStates);
        SmallVector<Type> resultTypes = llvm::map_to_vector(
            payloadStates, [](Value value) { return value.getType(); });
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("encoding", encoding),
            builder.getNamedAttr("payload_carrier_segments",
                                 builder.getDenseI64ArrayAttr(segments)),
            builder.getNamedAttr("event_id", nextEvent("unpack_resource")),
        };
        for (StringRef name :
             {StringRef("payload_action"),
              StringRef("payload_logical_block_ids"),
              StringRef("payload_logical_blocks"),
              StringRef("payload_logical_ports"), StringRef("payload_roles")})
          if (Attribute attribute = unpack->getAttr(name))
            attrs.push_back(builder.getNamedAttr(name, attribute));
        Operation *physical =
            createGeneric(builder, unpack.getLoc(), "phys.unpack_resource",
                          physicalInputs, resultTypes, attrs);
        size_t offset = 0;
        for (size_t index = 0; index < count; ++index) {
          projected[unpack.getOutputs()[index]] = std::move(anchors[index]);
          size_t width = payloads[index].size();
          projected[unpack.getOutputs()[count + index]] =
              PhysicalGroup(physical->getResults().begin() + offset,
                            physical->getResults().begin() + offset + width);
          offset += width;
        }
        continue;
      }
      if (auto pack = dyn_cast<qlx::fabric::PackResourceOp>(operation)) {
        if (pack.getPayloads().empty())
          return pack.emitOpError("requires at least one payload patch");
        SmallVector<Value> payloadStates;
        SmallVector<PhysicalGroup> payloadGroups;
        SmallVector<int64_t> segments{0};
        for (Value payload : pack.getPayloads()) {
          PhysicalGroup group = takeProjected(projected, payload, pack);
          if (group.empty())
            return pack.emitOpError("has an empty physical payload group");
          llvm::append_range(payloadStates, group.values());
          segments.push_back(payloadStates.size());
          payloadGroups.push_back(std::move(group));
        }
        auto encodings = pack->getAttrOfType<ArrayAttr>("payload_encodings");
        FlatSymbolRefAttr encoding;
        if (!encodings || encodings.empty())
          return pack.emitOpError("has no typed payload encoding evidence");
        for (Attribute raw : encodings) {
          auto candidate = dyn_cast<FlatSymbolRefAttr>(raw);
          if (!candidate || (encoding && candidate != encoding))
            return pack.emitOpError(
                "requires one common encoding for every payload patch");
          encoding = candidate;
        }
        auto resultType = physicalScalarType(pack.getResource().getType());
        if (!encoding || failed(resultType))
          return pack.emitOpError("has no physical resource payload type");
        StringAttr packEvent = nextEvent("pack_resource");
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("resource_kind", pack.getResourceKindAttr()),
            builder.getNamedAttr("encoding", encoding),
            builder.getNamedAttr("payload_carrier_segments",
                                 builder.getDenseI64ArrayAttr(segments)),
            builder.getNamedAttr("event_id", packEvent),
        };
        if (Attribute roles = pack->getAttr("payload_roles"))
          attrs.push_back(builder.getNamedAttr("payload_roles", roles));
        Operation *physical =
            createGeneric(builder, pack.getLoc(), "phys.pack_resource",
                          payloadStates, TypeRange{*resultType}, attrs);
        for (const PhysicalGroup &group : payloadGroups)
          if (failed(retireAllocation(group, packEvent.getValue())))
            return pack.emitOpError(
                "payload patch does not own one live physical allocation");
        projected[pack.getResource()] = PhysicalGroup{physical->getResult(0)};
        continue;
      }
      if (auto discard = dyn_cast<qlx::fabric::DiscardResourceOp>(operation)) {
        PhysicalGroup resource =
            takeProjected(projected, discard.getResource(), discard);
        if (resource.size() != 1 ||
            !isa<qlx::phys::ResourcePayloadType>(resource.front().getType()))
          return discard.emitOpError(
              "has no exact projected physical resource payload");
        createGeneric(
            builder, discard.getLoc(), "phys.discard_resource_payload",
            resource.values(), {},
            {builder.getNamedAttr("event_id", nextEvent("discard_resource"))});
        continue;
      }
      if (auto measurement =
              dyn_cast<qlx::fabric::MeasureProductOp>(operation)) {
        if (failed(measureProduct(measurement, projected, builder)))
          return measurement.emitOpError(
              actionFailure.empty()
                  ? "has an invalid Pauli projection or no compatible native "
                    "measurement instrument"
                  : actionFailure);
        continue;
      }
      if (isa<qlx::fabric::RotateProductOp>(operation)) {
        if (failed(rotateProduct(&operation, {}, operation.getOperands(),
                                 operation.getResults(), projected, builder)))
          return operation.emitOpError(
              "has an invalid or unsupported physical Pauli product");
        continue;
      }
      if (auto rotation =
              dyn_cast<qlx::fabric::ResourceRotateProductOp>(operation)) {
        PhysicalGroup resource = projected[rotation.getResource()];
        if (resource.size() != 1 ||
            failed(rotateProduct(
                &operation, resource.front(), rotation.getPatches(),
                rotation.getPatchResults(), projected, builder)))
          return rotation.emitOpError(
              "has an invalid resource-assisted physical Pauli product");
        continue;
      }
      if (auto allocation = dyn_cast<qlx::fabric::AllocOp>(operation)) {
        auto values = allocatePatch(allocation, builder);
        if (failed(values))
          return failure();
        projected[allocation.getResult()] = *values;
        continue;
      }
      if (auto prep = dyn_cast<qlx::fabric::PrepZOp>(operation)) {
        auto positions =
            wholePartitionPositions(prep.getPatch().getType(), "data");
        if (failed(positions))
          return prep.emitOpError("has no data partition");
        auto prepared =
            prepareSelected(takeProjected(projected, prep.getPatch(), prep),
                            *positions, "zero", prep.getLoc(), builder);
        if (failed(prepared))
          return prep.emitOpError("failed physical data preparation");
        projected[prep.getResult()] = *prepared;
        continue;
      }
      if (auto prep = dyn_cast<qlx::fabric::PrepXOp>(operation)) {
        auto positions =
            wholePartitionPositions(prep.getPatch().getType(), "data");
        if (failed(positions))
          return prep.emitOpError("has no data partition");
        auto prepared =
            prepareSelected(takeProjected(projected, prep.getPatch(), prep),
                            *positions, "plus", prep.getLoc(), builder);
        if (failed(prepared))
          return prep.emitOpError("failed physical data preparation");
        projected[prep.getResult()] = *prepared;
        continue;
      }
      if (auto dealloc = dyn_cast<qlx::fabric::DeallocOp>(operation)) {
        if (failed(release(projected[dealloc.getPatch()], dealloc.getLoc(),
                           builder))) {
          dealloc.emitOpError("does not own projected physical states");
          return failure();
        }
        continue;
      }
      if (auto init = dyn_cast<qlx::fabric::InitBasisOp>(operation)) {
        auto positions = selectedPositions(init, init.getPatch().getType());
        if (failed(positions))
          return init.emitOpError("has an invalid partition selection");
        StringRef state =
            init.getBasis() == qlx::fabric::Prep::x ? "plus" : "zero";
        auto prepared =
            prepareSelected(takeProjected(projected, init.getPatch(), init),
                            *positions, state, init.getLoc(), builder);
        if (failed(prepared))
          return init.emitOpError("failed physical partition preparation");
        projected[init.getResult()] = *prepared;
        continue;
      }
      if (auto begin = dyn_cast<qlx::fabric::TransformBeginOp>(operation)) {
        auto frame = beginTransform(
            begin, takeProjected(projected, begin.getSource(), begin));
        if (failed(frame))
          return begin.emitOpError("has an invalid physical frame projection");
        projected[begin.getFrame()] = std::move(*frame);
        continue;
      }
      if (auto end = dyn_cast<qlx::fabric::TransformEndOp>(operation)) {
        auto destination =
            endTransform(end, takeProjected(projected, end.getFrame(), end));
        if (failed(destination))
          return end.emitOpError("has an invalid physical frame commit");
        projected[end.getDestination()] = std::move(*destination);
        continue;
      }
      if (auto measure = dyn_cast<qlx::fabric::MzOp>(operation)) {
        auto positions =
            selectedPositions(measure, measure.getPatch().getType());
        if (failed(positions))
          return measure.emitOpError("has an invalid partition selection");
        StringRef record = measure.getRecord().value_or("mz");
        StringRef lane =
            qlx::fabric::stringifyPartition(measure.getPartition());
        auto measured =
            measureZ(takeProjected(projected, measure.getPatch(), measure),
                     *positions, record, currentCallable, currentInstance, lane,
                     measure.getLoc(), builder);
        if (failed(measured))
          return measure.emitOpError("failed physical Z measurement");
        projected[measure.getPatchOut()] = std::move(measured->first);
        projected[measure.getBits()] = std::move(measured->second);
        continue;
      }
      if (auto parityOp = dyn_cast<qlx::fabric::ParityOp>(operation)) {
        PhysicalGroup inputs;
        for (Value operand : parityOp.getBits())
          llvm::append_range(inputs, projected[operand]);
        auto result = parity(inputs.values(), parityOp.getLoc(), builder);
        if (failed(result))
          return parityOp.emitOpError("has no projectable measurement bits");
        projected[parityOp.getResult()] = PhysicalGroup{*result};
        continue;
      }
      if (auto allZero = dyn_cast<qlx::fabric::AllZeroOp>(operation)) {
        PhysicalGroup inputs = projected[allZero.getBits()];
        SmallVector<Value> predicates;
        for (Value input : inputs) {
          auto predicate =
              condition(input, /*expected=*/true, allZero.getLoc(), builder);
          if (failed(predicate))
            return allZero.emitOpError("has a non-record physical projection");
          predicates.push_back(*predicate);
        }
        if (predicates.empty())
          return allZero.emitOpError("requires projected measurement bits");
        Operation *result = createGeneric(
            builder, allZero.getLoc(), "phys.all_false", predicates,
            builder.getI1Type(),
            {builder.getNamedAttr("event_id", nextEvent("all_false"))});
        projected[allZero.getResult()] = PhysicalGroup{result->getResult(0)};
        continue;
      }
      if (auto xored = dyn_cast<qlx::fabric::XorOp>(operation)) {
        PhysicalGroup left = projected[xored.getLhs()];
        PhysicalGroup right = projected[xored.getRhs()];
        if (left.size() != 1 || right.size() != 1)
          return xored.emitOpError("requires scalar physical predicates");
        auto leftPredicate =
            condition(left.front(), /*expected=*/true, xored.getLoc(), builder);
        auto rightPredicate = condition(right.front(), /*expected=*/true,
                                        xored.getLoc(), builder);
        if (failed(leftPredicate) || failed(rightPredicate))
          return xored.emitOpError(
              "requires scalar booleans or physical records");
        Operation *result = createGeneric(
            builder, xored.getLoc(), "phys.xor",
            ValueRange{*leftPredicate, *rightPredicate}, builder.getI1Type(),
            {builder.getNamedAttr("event_id", nextEvent("xor"))});
        projected[xored.getResult()] = PhysicalGroup{result->getResult(0)};
        continue;
      }
      if (auto allFalse = dyn_cast<qlx::fabric::AllFalseOp>(operation)) {
        PhysicalGroup predicates;
        for (Value operand : allFalse.getEvents()) {
          PhysicalGroup values = projected[operand];
          if (values.size() != 1)
            return allFalse.emitOpError(
                "requires scalar booleans or physical records");
          auto predicate = condition(values.front(), /*expected=*/true,
                                     allFalse.getLoc(), builder);
          if (failed(predicate))
            return allFalse.emitOpError(
                "requires scalar booleans or physical records");
          predicates.push_back(*predicate);
        }
        Operation *result = createGeneric(
            builder, allFalse.getLoc(), "phys.all_false", predicates.values(),
            builder.getI1Type(),
            {builder.getNamedAttr("event_id", nextEvent("all_false"))});
        projected[allFalse.getResult()] = PhysicalGroup{result->getResult(0)};
        continue;
      }
      if (auto selection = dyn_cast<qlx::event::SelectionOp>(operation)) {
        PhysicalGroup values = projected[selection.getPredicate()];
        if (values.size() != 1)
          return selection.emitOpError(
              "requires one scalar physical predicate");
        auto predicate = condition(values.front(), /*expected=*/true,
                                   selection.getLoc(), builder);
        if (failed(predicate))
          return selection.emitOpError(
              "requires a scalar boolean or physical record");
        createGeneric(
            builder, selection.getLoc(), "event.selection",
            ValueRange{*predicate}, {},
            {
                builder.getNamedAttr("mode", selection.getModeAttr()),
                builder.getNamedAttr("accept_when",
                                     selection.getAcceptWhenAttr()),
                builder.getNamedAttr("event_id", nextEvent("selection")),
            });
        continue;
      }
      if (auto conditional = dyn_cast<qlx::cflow::IfOp>(operation)) {
        if (failed(emitConditional(conditional, projected, builder)))
          return conditional.emitOpError(
              "has an unsupported physical conditional boundary");
        continue;
      }
      if (auto retry = dyn_cast<qlx::fabric::RetryOp>(operation)) {
        if (failed(emitRetry(retry, projected, builder)))
          return retry.emitOpError(
              "has no exact projected attempt/decision boundary");
        continue;
      }
      if (isa<qlx::fabric::CXOp>(operation)) {
        if (failed(applyTwoPatch(&operation, "cx", projected, builder)))
          return operation.emitOpError(
              "requires explicit valid pairs and native physical @cx");
        continue;
      }
      if (isa<qlx::fabric::CZOp>(operation)) {
        if (failed(applyTwoPatch(&operation, "cz", projected, builder)))
          return operation.emitOpError(
              "requires explicit valid pairs and native physical @cz");
        continue;
      }
      StringRef name = operation.getName().getStringRef();
      static const llvm::StringMap<StringRef> actions{
          {"fabric.h", "h"},     {"fabric.s", "s"}, {"fabric.sdg", "sdg"},
          {"fabric.x", "x"},     {"fabric.z", "z"}, {"fabric.t", "t"},
          {"fabric.tdg", "tdg"},
      };
      if (auto found = actions.find(name); found != actions.end()) {
        Value input = operation.getOperand(0);
        auto positions = selectedPositions(&operation, input.getType());
        if (failed(positions))
          return operation.emitOpError("has an invalid partition selection");
        auto applied =
            applyAction(takeProjected(projected, input, &operation), *positions,
                        found->second, operation.getLoc(), builder);
        if (failed(applied))
          return operation.emitOpError(actionFailure.empty()
                                           ? "failed native physical action"
                                           : actionFailure);
        projected[operation.getResult(0)] = *applied;
        continue;
      }
      if (auto reset = dyn_cast<qlx::fabric::ResetOp>(operation)) {
        auto positions = selectedPositions(reset, reset.getPatch().getType());
        if (failed(positions))
          return reset.emitOpError("has an invalid partition selection");
        auto updated =
            resetSelected(takeProjected(projected, reset.getPatch(), reset),
                          *positions, reset.getLoc(), builder);
        if (failed(updated))
          return reset.emitOpError("failed physical reset");
        projected[reset.getResult()] = *updated;
        continue;
      }
      if (auto idle = dyn_cast<qlx::fabric::IdleOp>(operation)) {
        auto delayed = delay(takeProjected(projected, idle.getPatch(), idle),
                             idle.getRounds(), idle.getLoc(), builder);
        if (failed(delayed))
          return idle.emitOpError("has an invalid physical duration");
        projected[idle.getResult()] = *delayed;
        continue;
      }
      if (isa<qlx::fabric::TickOp>(operation)) {
        if (failed(momentBarrier(projected, operation.getLoc(), builder)))
          return failure();
        continue;
      }
      if (auto repeat = dyn_cast<qlx::cflow::RepeatOp>(operation)) {
        // Record the pre-body size of a projection lane only when the body
        // actually mutates it.  The previous implementation copied and later
        // rescanned the complete, monotonically growing projection map for
        // every nested repeat.  Large WSC protocols contain many repeats and
        // stable records, turning exact folded-record bookkeeping into
        // quadratic work unrelated to the emitted P3 graph.
        RecordProjectionScope projectionScope;
        activeRecordProjectionScopes.push_back(&projectionScope);
        size_t sidecarStart = pendingSidecars.size();
        SmallVector<Value> inputs;
        SmallVector<PhysicalGroup> templates;
        for (Value init : repeat.getInits()) {
          PhysicalGroup group = takeProjected(projected, init, repeat);
          llvm::append_range(inputs, group);
          templates.push_back(std::move(group));
        }
        SmallVector<Type> types = llvm::map_to_vector(
            inputs, [](Value value) { return value.getType(); });
        StringAttr repeatEvent = nextEvent("repeat");
        SmallVector<NamedAttribute> attrs{
            builder.getNamedAttr("count", repeat.getCountAttr()),
            builder.getNamedAttr("event_id", repeatEvent),
        };
        Operation *physical = createGeneric(
            builder, repeat.getLoc(), "cflow.repeat", inputs, types, attrs, 1);
        Block *body = new Block();
        physical->getRegion(0).push_back(body);
        for (Type type : types)
          body->addArgument(type, repeat.getLoc());
        ProjectedValues nested = projected;
        size_t offset = 0;
        Block &sourceBody = repeat.getBody().front();
        for (auto [argument, group] :
             llvm::zip(sourceBody.getArguments(), templates)) {
          auto arguments = body->getArguments().slice(offset, group.size());
          nested[argument] = PhysicalGroup(arguments.begin(), arguments.end());
          offset += group.size();
        }
        OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
        LogicalResult emittedBody = emitBlock(sourceBody, nested, bodyBuilder);
        activeRecordProjectionScopes.pop_back();
        if (failed(emittedBody))
          return failure();

        // The body is emitted once, while its stable Fabric records denote a
        // bounded dynamic family. Records authored directly by this callable
        // have iteration-qualified P2 names, so retain the exact physical
        // iteration in their record spelling. Records owned by nested calls
        // keep their stable source name and carry the enclosing folded repeat
        // as structural projection evidence. Neither case expands the repeat
        // body or duplicates a detached analysis row.
        struct ProjectionDelta {
          RecordProjectionKey key;
          size_t start;
          SmallVector<std::string> records;
        };
        SmallVector<ProjectionDelta> deltas;
        for (const auto &[entry, start] : projectionScope.starts) {
          const RecordProjectionKey &key = entry->first;
          auto found = recordProjection.find(key);
          if (found == recordProjection.end() ||
              found->second.records.size() < start)
            return repeat.emitOpError(
                "record-projection scope changed while lowering its body");
          const auto &records = found->second.records;
          if (start < records.size())
            deltas.push_back({key, start,
                              SmallVector<std::string>(records.begin() + start,
                                                       records.end())});
        }
        int64_t count = repeat.getCount();
        std::string sourcePrefix = currentCallable + ".";
        // Detached profiles name the finite stable-record coordinates they
        // consume. Select those coordinates algebraically instead of
        // enumerating the repeat's dynamic extent.
        std::set<std::string> selectedProfileRecords;
        if (currentProfile) {
          auto profile = dyn_cast_or_null<qlx::fabric::GadgetProfileOp>(
              symbols.lookup(currentProfile.getValue()));
          if (!profile)
            return repeat.emitOpError(
                "selected profile is unresolved while projecting records");
          for (Operation &declaration : profile.getBody().front()) {
            if (!isa<qlx::fabric::SuccessOp>(declaration))
              continue;
            if (auto records = declaration.getAttrOfType<ArrayAttr>("records"))
              for (Attribute raw : records)
                if (auto record = dyn_cast<StringAttr>(raw))
                  selectedProfileRecords.insert(record.getValue().str());
          }
        }
        for (const ProjectionDelta &delta : deltas) {
          auto found = recordProjection.find(delta.key);
          if (found == recordProjection.end() ||
              found->second.records.size() < delta.start)
            return repeat.emitOpError(
                "record-projection scope changed while lowering its body");
          if (count == 0) {
            mutateRecordProjection(delta.key).resize(delta.start);
            for (const std::string &record : delta.records)
              recordProjectionRepeats.erase(
                  {delta.key.first, delta.key.second, record});
            continue;
          }
          bool direct = delta.key.first == currentInstance &&
                        StringRef(delta.key.second).starts_with(sourcePrefix);
          if (!direct) {
            for (const std::string &record : delta.records)
              recordProjectionRepeats[{delta.key.first, delta.key.second,
                                       record}]
                  .push_back({repeatEvent.getValue().str(), count});
            continue;
          }

          mutateRecordProjection(delta.key).resize(delta.start);
          StringRef relative =
              StringRef(delta.key.second).drop_front(sourcePrefix.size());
          std::string relativeSuffix = (Twine(".") + relative).str();
          std::string repeatSegment =
              (Twine("__qlx_repeat") + Twine(sourceOrdinal) + "_").str();
          std::set<int64_t> selectedIterations;
          for (const std::string &selected : selectedProfileRecords) {
            StringRef candidate(selected);
            if (!candidate.consume_front(sourcePrefix) ||
                !candidate.consume_back(relativeSuffix))
              continue;
            size_t segmentStart = candidate.rfind('.');
            StringRef segment = segmentStart == StringRef::npos
                                    ? candidate
                                    : candidate.drop_front(segmentStart + 1);
            if (!segment.consume_front(repeatSegment))
              continue;
            int64_t iteration = -1;
            if (segment.getAsInteger(10, iteration) || iteration < 0 ||
                iteration >= count)
              continue;
            selectedIterations.insert(iteration);
          }
          for (int64_t iteration : selectedIterations) {
            std::string qualifiedSource =
                llvm::formatv("{0}.__qlx_repeat{1}_{2}.{3}", currentCallable,
                              sourceOrdinal, iteration, relative)
                    .str();
            auto &qualifiedRecords = mutateRecordProjection(
                {delta.key.first, std::move(qualifiedSource)});
            for (const std::string &record : delta.records) {
              std::string qualifiedPhysical =
                  llvm::formatv("{0}.__qlx_repeat[{1}][{2}]", record,
                                repeatEvent.getValue(), iteration)
                      .str();
              qualifiedRecords.push_back(qualifiedPhysical);
              auto previousContext = recordProjectionRepeats.find(
                  {delta.key.first, delta.key.second, record});
              if (previousContext != recordProjectionRepeats.end())
                recordProjectionRepeats[{delta.key.first, qualifiedSource,
                                         qualifiedPhysical}] =
                    previousContext->second;
            }
          }
          for (const std::string &record : delta.records)
            recordProjectionRepeats.erase(
                {delta.key.first, delta.key.second, record});
        }
        if (count == 0)
          pendingSidecars.resize(sidecarStart);

        SmallVector<Value> yields;
        for (auto [operand, templateValues] :
             llvm::zip(sourceBody.getTerminator()->getOperands(), templates)) {
          PhysicalGroup group = nested[operand];
          if (group.size() != templateValues.size())
            return repeat.emitOpError(
                "physical carry boundary changed in body");

          // A packed logical carry projects to a complete physical code block.
          // Calls inside the body are free to return that block in a different
          // member order, but cflow.repeat requires the yielded state types to
          // match the init/result order exactly.  Restore the per-carry
          // resource-identity order without changing any produced SSA value.
          // Do this per logical carry rather than globally because packed
          // aliases can project overlapping physical groups.
          SmallVector<bool> used(group.size(), false);
          for (Value templateValue : templateValues) {
            std::optional<size_t> matched;
            for (size_t index = 0; index < group.size(); ++index)
              if (!used[index] &&
                  group[index].getType() == templateValue.getType()) {
                matched = index;
                break;
              }
            if (!matched)
              return repeat.emitOpError(
                  "physical carry resource identity changed in body");
            used[*matched] = true;
            yields.push_back(group[*matched]);
          }
        }
        if (yields.size() != types.size())
          return repeat.emitOpError("physical carry boundary changed in body");
        createGeneric(bodyBuilder, repeat.getLoc(), "cflow.yield", yields);
        offset = 0;
        for (auto [result, templateValues] :
             llvm::zip(repeat.getResults(), templates)) {
          size_t width = templateValues.size();
          auto results = physical->getResults().slice(offset, width);
          projected[result] = PhysicalGroup(results.begin(), results.end());
          offset += width;
        }
        continue;
      }
      auto call = dyn_cast<qlx::fabric::CallOp>(operation);
      if (!call)
        return operation.emitOpError(
            "passed native preflight but has no emitter");
      SmallVector<PhysicalGroup> callArguments;
      callArguments.reserve(call.getNumOperands());
      size_t inputCount = 0;
      for (Value operand : call.getOperands()) {
        PhysicalGroup group = takeProjected(projected, operand, call);
        inputCount += group.size();
        callArguments.push_back(std::move(group));
      }
      Operation *callee = symbols.lookup(call.getCallee());
      if (!callee)
        return call.emitOpError("references an unresolved Fabric callee");
      StringAttr instance = stringAttr(
          context,
          (Twine(call.getCallee()) + ".call" + Twine(callInstance++)).str());
      StringAttr callEvent = nextEvent("call");
      projectedCallEvents[call.getOperation()] = callEvent;
      FlatSymbolRefAttr callProfile = call.getProfileAttr();
      auto callProvider = spacetimeProviderFor(callee);
      auto componentPlan = componentSpacetimePlans.find(call.getCallee());
      bool closesRetryBoundary =
          llvm::any_of(call.getResults(), [](Value result) {
            return llvm::any_of(result.getUsers(), [](Operation *user) {
              return isa<qlx::fabric::RetryOp>(user);
            });
          });
      StringRef templateBucket =
          callTemplateBucket(callee, call.getCalleeAttr());
      uint64_t templateFingerprint = physicalTemplateFingerprint(callee);

      // The dominant compiler-lowered path threads complete physical patches
      // through a previously verified template. Once a layout pair has been
      // proven, emit the compact resource-summary invocation without
      // rebuilding O(code-distance^2) flattened type/result boundaries.
      bool patchOnlyBoundary =
          !callProvider && componentPlan == componentSpacetimePlans.end() &&
          call.getNumOperands() == call.getNumResults() &&
          llvm::all_of(
              call.getOperandTypes(),
              [](Type type) {
                return isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(
                    type);
              }) &&
          llvm::all_of(call.getResultTypes(), [](Type type) {
            return isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(
                type);
          });
      if (patchOnlyBoundary) {
        SmallVector<PhysicalGroup::LayoutIdentity> outputLayouts;
        outputLayouts.reserve(callArguments.size());
        for (const PhysicalGroup &group : callArguments)
          outputLayouts.push_back(group.getLayoutIdentity());
        CanonicalCallTemplate *compatible = nullptr;
        ArrayAttr stateAliases;
        auto indexedTemplates = callTemplateIndices.find(templateBucket);
        if (indexedTemplates != callTemplateIndices.end()) {
          auto fingerprintedTemplates =
              indexedTemplates->second.find(templateFingerprint);
          if (fingerprintedTemplates != indexedTemplates->second.end())
            for (size_t index : fingerprintedTemplates->second) {
              ++templateCompatibilityProbes;
              CanonicalCallTemplate &candidate = callTemplates[index];
              if (candidate.closesRetryBoundary != closesRetryBoundary)
                continue;
              if (!physicallyTemplateEquivalent(
                      symbols.lookup(candidate.callee.getValue()), callee)) {
                ++templateEquivalenceRejects;
                continue;
              }
              if (!profilesShareGadget(candidate.profile, callProfile,
                                       candidate.callee,
                                       call.getCalleeAttr())) {
                ++templateProfileRejects;
                continue;
              }
              if (exactGroupLayoutsFor(candidate, callArguments,
                                       outputLayouts)) {
                stateAliases = builder.getArrayAttr({});
                ++templateExactLayoutHits;
              } else {
                StateAliasCacheKey cacheKey =
                    stateAliasCacheKeyFor(index, callArguments, outputLayouts);
                auto cached = stateAliasCache.find(cacheKey);
                if (cached == stateAliasCache.end())
                  continue;
                stateAliases = cached->second;
                ++templateAliasCacheHits;
              }
              if (!stateAliases) {
                ++templateAliasRejects;
                continue;
              }
              bool preferExactReuse = candidate.callee != call.getCalleeAttr();
              if (!preferExactReuse && !candidate.allocationPlan.empty())
                preferExactReuse =
                    hasSourceDataDependency(call, candidate.sourceCall);
              if (!allocationPlanMatches(candidate, stateAliases.getValue(),
                                         preferExactReuse)) {
                ++templateAllocationRejects;
                continue;
              }
              compatible = &candidate;
              break;
            }
        }
        if (compatible) {
          SmallVector<NamedAttribute> attrs{
              builder.getNamedAttr("callee", call.getCalleeAttr()),
              builder.getNamedAttr("instance", instance),
              builder.getNamedAttr("event_id", callEvent),
              builder.getNamedAttr("state_boundary_elided",
                                   UnitAttr::get(context)),
          };
          if (callProfile)
            attrs.push_back(builder.getNamedAttr("profile", callProfile));
          if (failed(appendCallTemplateAttributes(
                  *compatible, call.getCalleeAttr(), callProfile, instance,
                  stateAliases, attrs, builder, call.getLoc())))
            return failure();
          createGeneric(builder, call.getLoc(), "phys.call_template", {}, {},
                        attrs);
          ++templateInvocations;
          ++elidedTemplateInvocations;
          if (!stateAliases.empty())
            ++aliasedTemplateInvocations;
          if (callProfile &&
              failed(collectProfileSidecars(callProfile, instance.getValue())))
            return failure();
          for (auto [result, group] :
               llvm::zip(call.getResults(), callArguments))
            projected[result] = std::move(group);
          if (failed(replayAllocationPlan(*compatible, stateAliases.getValue(),
                                          callEvent.getValue())))
            return call.emitOpError(
                "cannot replay the canonical physical allocation effects");
          ++templateFastPathInvocations;
          continue;
        }
      }
      SmallVector<Value> flattenedInputs;
      flattenedInputs.reserve(inputCount);
      for (const PhysicalGroup &group : callArguments)
        llvm::append_range(flattenedInputs, group);
      SmallVector<Type> resultTypes;
      SmallVector<unsigned> stateInputPositions;
      stateInputPositions.reserve(flattenedInputs.size());
      for (auto [position, value] : llvm::enumerate(flattenedInputs)) {
        Type type = value.getType();
        if (isa<qlx::phys::StateType>(type))
          stateInputPositions.push_back(position);
      }
      size_t stateOffset = 0;
      SmallVector<int64_t> resultWidths;
      resultWidths.reserve(call.getNumResults());
      size_t resultCount = 0;
      SmallVector<size_t> patchInputWidths;
      SmallVector<PhysicalGroup::LayoutIdentity> patchInputLayouts;
      for (auto [operand, values] :
           llvm::zip(call.getOperands(), callArguments))
        if (isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(
                operand.getType())) {
          patchInputWidths.push_back(values.size());
          patchInputLayouts.push_back(values.getLayoutIdentity());
        }
      size_t patchResult = 0;
      SmallVector<PhysicalGroup::LayoutIdentity> resultGroupLayouts;
      resultGroupLayouts.reserve(call.getNumResults());
      for (Type type : call.getResultTypes()) {
        if (isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(type)) {
          if (patchResult >= patchInputWidths.size())
            return call.emitOpError(
                "creates physical patch states across a call boundary");
          resultWidths.push_back(patchInputWidths[patchResult]);
          resultGroupLayouts.push_back(patchInputLayouts[patchResult++]);
        } else {
          auto width = projectedWidth(type);
          if (failed(width))
            return call.emitOpError("has an unresolved physical result type");
          resultWidths.push_back(*width);
          resultGroupLayouts.push_back(PhysicalGroup().getLayoutIdentity());
        }
        resultCount += resultWidths.back();
      }
      resultTypes.reserve(resultCount);
      SmallVector<int64_t> outputInputPositions;
      outputInputPositions.reserve(resultCount);
      for (size_t resultIndex = 0; resultIndex < call.getNumResults();
           ++resultIndex) {
        Type type = call.getResult(resultIndex).getType();
        int64_t width = resultWidths[resultIndex];
        if (isa<qlx::fabric::PatchType, qlx::fabric::PatchFrameType>(type)) {
          if (stateOffset + width > stateInputPositions.size())
            return call.emitOpError(
                "creates physical patch states across a call boundary");
          for (int64_t index = 0; index < width; ++index) {
            unsigned inputPosition = stateInputPositions[stateOffset + index];
            resultTypes.push_back(flattenedInputs[inputPosition].getType());
            outputInputPositions.push_back(inputPosition);
          }
          stateOffset += width;
          continue;
        }
        if (auto tensor = dyn_cast<RankedTensorType>(type)) {
          for (int64_t index = 0; index < tensor.getNumElements(); ++index) {
            resultTypes.push_back(qlx::phys::RecordType::get(
                context, symbolAttr(context, "bit")));
            outputInputPositions.push_back(-1);
          }
          continue;
        }
        if (type.isInteger(1)) {
          auto record = projectsToPhysicalRecord(call.getResult(resultIndex));
          if (failed(record))
            return call.emitOpError(
                "has an ambiguous physical scalar result boundary");
          if (*record) {
            resultTypes.push_back(qlx::phys::RecordType::get(
                context, symbolAttr(context, "bit")));
            outputInputPositions.push_back(-1);
            continue;
          }
        }
        auto physical = physicalScalarType(type);
        if (failed(physical))
          return call.emitOpError("has an unsupported physical result type");
        resultTypes.push_back(*physical);
        outputInputPositions.push_back(-1);
      }
      if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callee)) {
        auto macro = patchMacroFor(gadget);
        auto granularity = projectedGroupGranularity(flattenedInputs);
        if (succeeded(macro) && macro->kind == PatchMacroKind::Measure &&
            succeeded(granularity) && *granularity == "patch") {
          auto instrument =
              sharedNativeInstrument(flattenedInputs, macro->physicalOperation);
          if (failed(instrument) || resultTypes.size() != 1)
            return call.emitOpError(
                "has no valid physical measurement-macro result type");
          resultTypes.front() = qlx::phys::RecordType::get(
              context, symbolAttr(context, instrument->getRecordSchema()));
        }
      }
      if (componentPlan != componentSpacetimePlans.end()) {
        auto protocol = dyn_cast<qlx::fabric::ProtocolOp>(callee);
        if (!protocol)
          return call.emitOpError(
              "component spacetime models may refine only fabric.protocol");
        if (flattenedInputs.size() != resultTypes.size() ||
            !llvm::equal(
                llvm::map_range(flattenedInputs,
                                [](Value value) { return value.getType(); }),
                resultTypes))
          return call.emitOpError(
              "component spacetime model v1 requires a boundary-preserving "
              "physical call");
        Operation *physicalCall = createGeneric(
            builder, call.getLoc(), "phys.spacetime_call", flattenedInputs,
            resultTypes,
            {
                builder.getNamedAttr("plan", componentPlan->second),
                builder.getNamedAttr(
                    "source_protocol",
                    symbolAttr(context, protocol.getSymName())),
                builder.getNamedAttr("instance", instance),
                builder.getNamedAttr("event_id", callEvent),
            });
        size_t resultOffset = 0;
        size_t resultIndex = 0;
        for (auto [result, width] :
             llvm::zip(call.getResults(), resultWidths)) {
          PhysicalGroup group(physicalCall->getResults().begin() + resultOffset,
                              physicalCall->getResults().begin() +
                                  resultOffset + width,
                              resultGroupLayouts[resultIndex++]);
          resultOffset += width;
          projected[result] = std::move(group);
        }
        continue;
      }
      if (callProvider &&
          (*callProvider == SpacetimeProviderKind::SurfaceAutoCCZApplication ||
           *callProvider == SpacetimeProviderKind::SurfaceSpacelikeCallable) &&
          hasExplicitReactionNanoseconds) {
        auto protocol = cast<qlx::fabric::ProtocolOp>(callee);
        FailureOr<FlatSymbolRefAttr> plan = failure();
        if (*callProvider == SpacetimeProviderKind::SurfaceAutoCCZApplication) {
          plan = surfaceAutoCCZApplicationPlan(protocol);
        } else {
          auto shape = qlx::spacetime::surfaceSpacelikeShape(protocol);
          if (!shape)
            return call.emitOpError(
                "selected surface spacelike provider lost its exact P2 "
                "structural proof");
          plan = surfaceSpacelikeCallablePlan(protocol, *shape);
        }
        if (failed(plan))
          return failure();
        if (flattenedInputs.size() != resultTypes.size() ||
            !llvm::equal(
                llvm::map_range(flattenedInputs,
                                [](Value value) { return value.getType(); }),
                resultTypes))
          return call.emitOpError(
              "surface spacetime call must preserve its complete physical "
              "owner boundary");
        Operation *physicalCall =
            createGeneric(builder, call.getLoc(), "phys.spacetime_call",
                          flattenedInputs, resultTypes,
                          {
                              builder.getNamedAttr("plan", *plan),
                              builder.getNamedAttr(
                                  "source_protocol",
                                  symbolAttr(context, protocol.getSymName())),
                              builder.getNamedAttr("instance", instance),
                              builder.getNamedAttr("event_id", callEvent),
                          });
        size_t resultOffset = 0;
        size_t resultIndex = 0;
        for (auto [result, width] :
             llvm::zip(call.getResults(), resultWidths)) {
          PhysicalGroup group(physicalCall->getResults().begin() + resultOffset,
                              physicalCall->getResults().begin() +
                                  resultOffset + width,
                              resultGroupLayouts[resultIndex++]);
          resultOffset += width;
          projected[result] = std::move(group);
        }
        continue;
      }
      CanonicalCallTemplate *compatible = nullptr;
      ArrayAttr stateAliases;
      auto indexedTemplates = callTemplateIndices.find(templateBucket);
      if (indexedTemplates != callTemplateIndices.end()) {
        auto fingerprintedTemplates =
            indexedTemplates->second.find(templateFingerprint);
        if (fingerprintedTemplates != indexedTemplates->second.end())
          for (size_t index : fingerprintedTemplates->second) {
            ++templateCompatibilityProbes;
            CanonicalCallTemplate &candidate = callTemplates[index];
            if (candidate.closesRetryBoundary != closesRetryBoundary)
              continue;
            if (!physicallyTemplateEquivalent(
                    symbols.lookup(candidate.callee.getValue()), callee)) {
              ++templateEquivalenceRejects;
              continue;
            }
            if (!profilesShareGadget(candidate.profile, callProfile,
                                     candidate.callee, call.getCalleeAttr())) {
              ++templateProfileRejects;
              continue;
            }
            ArrayAttr aliases;
            if (exactGroupLayoutsFor(candidate, callArguments,
                                     resultGroupLayouts)) {
              aliases = builder.getArrayAttr({});
              ++templateExactLayoutHits;
            } else {
              StateAliasCacheKey cacheKey = stateAliasCacheKeyFor(
                  index, callArguments, resultGroupLayouts);
              auto cached = stateAliasCache.find(cacheKey);
              if (cached != stateAliasCache.end()) {
                aliases = cached->second;
                ++templateAliasCacheHits;
              } else {
                auto computed = stateAliasesFor(candidate, flattenedInputs,
                                                resultTypes, builder);
                if (computed)
                  aliases = builder.getArrayAttr(*computed);
                stateAliasCache.emplace(std::move(cacheKey), aliases);
                ++templateAliasCacheMisses;
              }
            }
            if (!aliases) {
              ++templateAliasRejects;
              continue;
            }
            bool preferExactReuse = candidate.callee != call.getCalleeAttr();
            if (!preferExactReuse && !candidate.allocationPlan.empty())
              preferExactReuse =
                  hasSourceDataDependency(call, candidate.sourceCall);
            if (!allocationPlanMatches(candidate, aliases.getValue(),
                                       preferExactReuse)) {
              if (std::getenv("QLX_PROFILE_P2_TO_P3") &&
                  templateAllocationRejects < 8) {
                llvm::errs()
                    << "fabric-to-phys: allocation-reject canonical=@"
                    << candidate.callee.getValue() << " invocation=@"
                    << call.getCallee() << " exact=" << preferExactReuse
                    << " aliases=" << aliases.size()
                    << " steps=" << candidate.allocationPlan.size() << "\n";
                for (const AllocationPlanStep &step :
                     candidate.allocationPlan) {
                  llvm::errs()
                      << "  " << (step.acquire ? "acquire " : "release ")
                      << step.resourceClass << " capacity=" << step.capacity
                      << " indices-count=" << step.indices.size();
                  if (!step.indices.empty())
                    llvm::errs() << " indices-first=" << step.indices.front()
                                 << " indices-last=" << step.indices.back();
                  auto active = activeIndices.find(step.resourceClass);
                  llvm::errs() << " active-count=";
                  if (active != activeIndices.end())
                    llvm::errs() << active->second.size();
                  else
                    llvm::errs() << 0;
                  llvm::errs() << "\n";
                }
              }
              ++templateAllocationRejects;
              continue;
            }
            compatible = &candidate;
            stateAliases = aliases;
            break;
          }
      }
      SmallVector<NamedAttribute> attrs{
          builder.getNamedAttr("callee", call.getCalleeAttr()),
          builder.getNamedAttr("instance", instance),
          builder.getNamedAttr("event_id", callEvent),
      };
      if (callProfile)
        attrs.push_back(builder.getNamedAttr("profile", callProfile));
      bool elideStateBoundary =
          compatible && supportsElidedStateBoundary(*compatible);
      if (elideStateBoundary)
        attrs.push_back(builder.getNamedAttr("state_boundary_elided",
                                             UnitAttr::get(context)));
      SmallVector<Value> physicalInputs;
      SmallVector<Type> physicalResultTypes;
      if (compatible) {
        if (!elideStateBoundary) {
          for (int64_t position = compatible->retainedInputs.find_first();
               position >= 0;
               position = compatible->retainedInputs.find_next(position))
            physicalInputs.push_back(flattenedInputs[position]);
          for (int64_t position = compatible->retainedOutputs.find_first();
               position >= 0;
               position = compatible->retainedOutputs.find_next(position))
            physicalResultTypes.push_back(resultTypes[position]);
        }
      } else {
        llvm::append_range(physicalInputs, flattenedInputs);
        llvm::append_range(physicalResultTypes, resultTypes);
      }
      Operation *physicalCall = nullptr;
      if (compatible) {
        if (failed(appendCallTemplateAttributes(
                *compatible, call.getCalleeAttr(), callProfile, instance,
                stateAliases, attrs, builder, call.getLoc())))
          return failure();
        physicalCall =
            createGeneric(builder, call.getLoc(), "phys.call_template",
                          physicalInputs, physicalResultTypes, attrs);
        ++templateInvocations;
        templateBoundaryValues +=
            physicalInputs.size() + physicalResultTypes.size();
        maxTemplateBoundaryValues = std::max<int64_t>(
            maxTemplateBoundaryValues,
            physicalInputs.size() + physicalResultTypes.size());
        if (elideStateBoundary)
          ++elidedTemplateInvocations;
        if (!stateAliases.empty())
          ++aliasedTemplateInvocations;
      } else {
        physicalCall =
            createGeneric(builder, call.getLoc(), "phys.call", physicalInputs,
                          physicalResultTypes, attrs, 1);
      }
      size_t offset = 0;
      if (compatible) {
        if (callProfile &&
            failed(collectProfileSidecars(callProfile, instance.getValue())))
          return failure();
        SmallVector<Value> fullResults(resultTypes.size());
        size_t retainedResult = 0;
        for (size_t position = 0; position < resultTypes.size(); ++position) {
          if (compatible->retainedOutputs.test(position)) {
            if (elideStateBoundary) {
              int64_t input = outputInputPositions[position];
              if (input < 0)
                return call.emitOpError(
                    "cannot elide a non-state shared call result");
              fullResults[position] = flattenedInputs[input];
            } else {
              fullResults[position] = physicalCall->getResult(retainedResult++);
            }
            continue;
          }
          int64_t input = outputInputPositions[position];
          if (input < 0)
            return call.emitOpError(
                "omits a non-state physical result from a shared boundary");
          fullResults[position] = flattenedInputs[input];
        }
        size_t resultIndex = 0;
        for (auto [result, width] :
             llvm::zip(call.getResults(), resultWidths)) {
          PhysicalGroup group(fullResults.begin() + offset,
                              fullResults.begin() + offset + width,
                              resultGroupLayouts[resultIndex++]);
          offset += width;
          projected[result] = std::move(group);
        }
        if (failed(replayAllocationPlan(*compatible, stateAliases.getValue(),
                                        callEvent.getValue())))
          return call.emitOpError(
              "cannot replay the canonical physical allocation effects");
        continue;
      }
      SmallVector<Type> inputTypes = llvm::map_to_vector(
          flattenedInputs, [](Value value) { return value.getType(); });
      Block *body = new Block();
      physicalCall->getRegion(0).push_back(body);
      for (Type type : inputTypes)
        body->addArgument(type, call.getLoc());
      SmallVector<PhysicalGroup> bodyArguments;
      for (PhysicalGroup &group : callArguments) {
        auto arguments = body->getArguments().slice(offset, group.size());
        bodyArguments.emplace_back(arguments.begin(), arguments.end(),
                                   group.getLayoutIdentity());
        offset += group.size();
      }
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
      size_t allocationPlanStart = allocationPlanTrace.size();
      auto yielded = emitCallable(callee, bodyArguments, bodyBuilder,
                                  callProfile, instance.getValue());
      if (failed(yielded))
        return failure();
      SmallVector<Value> flattenedYields;
      for (PhysicalGroup &group : *yielded)
        llvm::append_range(flattenedYields, group);
      if (flattenedYields.size() != resultTypes.size()) {
        call.emitOpError("projected callee result boundary mismatch");
        return failure();
      }
      Operation *physicalYield = createGeneric(bodyBuilder, call.getLoc(),
                                               "phys.yield", flattenedYields);

      llvm::BitVector retainedInputs(inputTypes.size(), true);
      llvm::BitVector retainedOutputs(resultTypes.size(), true);
      for (size_t position = 0; position < resultTypes.size(); ++position) {
        int64_t input = outputInputPositions[position];
        if (input < 0)
          continue;
        BlockArgument argument = body->getArgument(input);
        if (flattenedYields[position] != argument || !argument.hasOneUse())
          continue;
        OpOperand &use = *argument.use_begin();
        if (use.getOwner() != physicalYield ||
            use.getOperandNumber() != position)
          continue;
        if (closesRetryBoundary)
          continue;
        retainedInputs.reset(input);
        retainedOutputs.reset(position);
      }

      SmallVector<Value> fullResults(resultTypes.size());
      if (!closesRetryBoundary && retainedInputs.count() != inputTypes.size()) {
        SmallVector<Value> compactInputs;
        SmallVector<Type> compactResults;
        for (int64_t position = retainedInputs.find_first(); position >= 0;
             position = retainedInputs.find_next(position))
          compactInputs.push_back(flattenedInputs[position]);
        for (int64_t position = retainedOutputs.find_first(); position >= 0;
             position = retainedOutputs.find_next(position))
          compactResults.push_back(resultTypes[position]);

        llvm::BitVector erasedOutputs = retainedOutputs;
        erasedOutputs.flip();
        physicalYield->eraseOperands(erasedOutputs);
        llvm::BitVector erasedInputs = retainedInputs;
        erasedInputs.flip();
        body->eraseArguments(erasedInputs);

        OpBuilder compactBuilder(physicalCall);
        Operation *compactCall =
            createGeneric(compactBuilder, call.getLoc(), "phys.call",
                          compactInputs, compactResults, attrs, 1);
        compactCall->getRegion(0).takeBody(physicalCall->getRegion(0));
        physicalCall->erase();
        physicalCall = compactCall;
      }
      size_t retainedResult = 0;
      for (size_t position = 0; position < resultTypes.size(); ++position) {
        if (retainedOutputs.test(position)) {
          fullResults[position] = physicalCall->getResult(retainedResult++);
          continue;
        }
        fullResults[position] = flattenedInputs[outputInputPositions[position]];
      }
      llvm::StringMap<SmallVector<std::string>> canonicalRecords;
      RecordProjectionKey first{instance.getValue().str(), {}};
      for (auto found = recordProjection.lower_bound(first);
           found != recordProjection.end() &&
           found->first.first == instance.getValue();
           ++found)
        canonicalRecords.try_emplace(found->first.second,
                                     found->second.records);
      SmallVector<Type> retainedInputTypes;
      for (int64_t position = retainedInputs.find_first(); position >= 0;
           position = retainedInputs.find_next(position))
        retainedInputTypes.push_back(inputTypes[position]);
      SmallVector<Type> retainedOutputTypes;
      for (int64_t position = retainedOutputs.find_first(); position >= 0;
           position = retainedOutputs.find_next(position))
        retainedOutputTypes.push_back(resultTypes[position]);
      SmallVector<PhysicalGroup::LayoutIdentity> inputGroupLayouts;
      inputGroupLayouts.reserve(callArguments.size());
      for (const PhysicalGroup &group : callArguments)
        inputGroupLayouts.push_back(group.getLayoutIdentity());
      SmallVector<AllocationPlanStep> canonicalAllocationPlan(
          allocationPlanTrace.begin() + allocationPlanStart,
          allocationPlanTrace.end());
      callTemplates.push_back(CanonicalCallTemplate{
          call.getOperation(), call.getCalleeAttr(), callProfile,
          closesRetryBoundary, inputTypes.size(), resultTypes.size(),
          std::move(retainedInputs), std::move(retainedOutputs),
          std::move(retainedInputTypes), std::move(retainedOutputTypes),
          std::move(inputGroupLayouts), resultGroupLayouts, callEvent,
          instance.getValue().str(), std::move(canonicalRecords),
          std::move(canonicalAllocationPlan)});
      if (!callTemplates.back().allocationPlan.empty())
        sourceDataDependencySources.try_emplace(
            call.getOperation(), sourceDataDependencySources.size());
      callTemplateIndices[templateBucket][templateFingerprint].push_back(
          callTemplates.size() - 1);
      if (std::getenv("QLX_PROFILE_P2_TO_P3") &&
          callTemplates.size() % 50 == 0) {
        llvm::StringMap<int64_t> specializations;
        int64_t maximumSpecializations = 0;
        for (const CanonicalCallTemplate &candidate : callTemplates)
          maximumSpecializations =
              std::max(maximumSpecializations,
                       ++specializations[candidate.callee.getValue()]);
        llvm::errs() << "fabric-to-phys: templates=" << callTemplates.size()
                     << " bucket=" << templateBucket
                     << " callees=" << specializations.size()
                     << " max-specializations=" << maximumSpecializations
                     << " calls=" << callInstance
                     << " equivalence-rejects=" << templateEquivalenceRejects
                     << " profile-rejects=" << templateProfileRejects
                     << " alias-rejects=" << templateAliasRejects
                     << " allocation-rejects=" << templateAllocationRejects
                     << " records=" << recordProjection.size()
                     << " events=" << event << " apply=" << applyEvents
                     << " measure=" << measurementRecord << "\n";
      }
      offset = 0;
      size_t resultIndex = 0;
      for (auto [result, width] : llvm::zip(call.getResults(), resultWidths)) {
        PhysicalGroup group(fullResults.begin() + offset,
                            fullResults.begin() + offset + width,
                            resultGroupLayouts[resultIndex++]);
        offset += width;
        projected[result] = std::move(group);
      }
    }

    return success();
  }

  FailureOr<SmallVector<PhysicalGroup>>
  emitCallable(Operation *callable, ArrayRef<PhysicalGroup> arguments,
               OpBuilder &builder, FlatSymbolRefAttr analysisProfile = {},
               StringRef instance = {}) {
    Block &block = callableBody(callable).front();
    if (arguments.size() != block.getNumArguments()) {
      callable->emitOpError("physical call boundary arity mismatch");
      return failure();
    }

    // Patch-frame provenance is relative to one physical invocation.  The
    // Fabric SSA values below are shared when a generated callable is reused,
    // but their projected physical states are not.  Retaining a memoized
    // derived frame across invocations therefore reconnects a later
    // transform_end to the first call's carriers.  Invalidate only this
    // callable's frame SSA values before seeding the current invocation; frame
    // values owned by an enclosing callable remain available to that scope.
    for (BlockArgument argument : block.getArguments())
      if (isa<qlx::fabric::PatchFrameType>(argument.getType()))
        transformFrames.erase(argument);
    callable->walk([&](Operation *operation) {
      if (operation == callable)
        return;
      for (Value result : operation->getResults())
        if (isa<qlx::fabric::PatchFrameType>(result.getType()))
          transformFrames.erase(result);
    });

    ProjectedValues projected;
    for (auto [argument, values] : llvm::zip(block.getArguments(), arguments))
      projected[argument] = values;
    std::string previousCallable = std::move(currentCallable);
    std::string previousInstance = std::move(currentInstance);
    FlatSymbolRefAttr previousProfile = currentProfile;
    currentCallable = symbolName(callable).str();
    currentInstance = instance.empty() ? currentCallable : instance.str();
    currentProfile = analysisProfile;
    llvm::scope_exit restoreContext([&] {
      currentCallable = std::move(previousCallable);
      currentInstance = std::move(previousInstance);
      currentProfile = previousProfile;
    });
    if (auto gadget = dyn_cast<qlx::fabric::GadgetOp>(callable)) {
      auto macro = patchMacroFor(gadget);
      if (succeeded(macro)) {
        bool hasPatchGranularity = false;
        bool hasCarrierGranularity = false;
        for (auto [argument, values] :
             llvm::zip(block.getArguments(), arguments)) {
          if (!isa<qlx::fabric::PatchType>(argument.getType()))
            continue;
          auto granularity = projectedGroupGranularity(values.values());
          if (failed(granularity)) {
            gadget.emitOpError(
                "cannot derive one physical granularity for a macro operand");
            return failure();
          }
          hasPatchGranularity |= *granularity == "patch";
          hasCarrierGranularity |= *granularity == "carrier";
        }
        if (hasPatchGranularity && hasCarrierGranularity) {
          gadget.emitOpError(
              "selected patch macro mixes carrier- and patch-granularity "
              "operands");
          return failure();
        }
        if (hasPatchGranularity) {
          actionFailure.clear();
          auto outputs = emitPatchMacro(gadget, *macro, arguments, builder);
          if (failed(outputs)) {
            if (!actionFailure.empty())
              gadget.emitOpError(actionFailure);
            else
              gadget.emitOpError(
                  "has no valid physical patch-macro realization on the "
                  "selected resource class");
            return failure();
          }
          return outputs;
        }
      }
    }
    if (failed(emitBlock(block, projected, builder)))
      return failure();
    if (analysisProfile && isa<qlx::fabric::GadgetOp>(callable) &&
        failed(collectProfileSidecars(analysisProfile, currentInstance)))
      return failure();

    SmallVector<PhysicalGroup> outputs;
    for (Value operand : block.getTerminator()->getOperands())
      outputs.push_back(projected[operand]);
    return outputs;
  }

  StringAttr nextEvent(StringRef kind) {
    return stringAttr(context, (Twine(kind) + Twine(event++)).str());
  }

  SmallVector<std::string> &mutateRecordProjection(RecordProjectionKey key) {
    auto [found, inserted] = recordProjection.try_emplace(std::move(key));
    for (RecordProjectionScope *scope : activeRecordProjectionScopes)
      if (scope->seen.insert(&found->second).second)
        scope->starts.emplace_back(&*found, found->second.records.size());
    if (inserted && std::getenv("QLX_PROFILE_P2_TO_P3") &&
        recordProjection.size() % 100000 == 0)
      llvm::errs() << "fabric-to-phys: records=" << recordProjection.size()
                   << " scopes=" << activeRecordProjectionScopes.size()
                   << " templates=" << callTemplates.size()
                   << " template-probes=" << templateCompatibilityProbes
                   << " calls=" << callInstance << " events=" << event
                   << " apply=" << applyEvents
                   << " measure=" << measurementRecord
                   << " sidecars=" << pendingSidecars.size() << "\n";
    return found->second.records;
  }

  LogicalResult verifyProjectedGraph() {
    const bool profileKinds =
        std::getenv("QLX_PROFILE_P3_VERIFY_KINDS") != nullptr;
    // Recursive verification is bottom-up. Preserve that ordering between
    // depths while verifying independent operations at one depth in parallel;
    // the graph verifier then discharges the whole-region invariants serially.
    SmallVector<SmallVector<Operation *, 0>, 8> operationsByDepth;
    using PendingOperation = std::pair<Operation *, size_t>;
    SmallVector<PendingOperation, 32> pending;
    auto pushChildren = [&](Operation *parent, size_t depth) {
      SmallVector<Operation *, 16> children;
      for (Region &region : parent->getRegions())
        for (Block &block : region)
          for (Operation &child : block)
            children.push_back(&child);
      for (Operation *child : llvm::reverse(children))
        pending.emplace_back(child, depth);
    };
    pushChildren(graph, 0);
    while (!pending.empty()) {
      auto [operation, depth] = pending.pop_back_val();
      if (operationsByDepth.size() <= depth)
        operationsByDepth.resize(depth + 1);
      operationsByDepth[depth].push_back(operation);
      pushChildren(operation, depth + 1);
    }
    for (size_t depth = operationsByDepth.size(); depth-- > 0;) {
      ArrayRef<Operation *> operations = operationsByDepth[depth];
      auto verifyOperations = [&](ArrayRef<Operation *> selected) {
        size_t shards = 1;
        if (context->isMultithreadingEnabled())
          shards = std::min({size_t{64},
                             static_cast<size_t>(
                                 context->getThreadPool().getMaxConcurrency()),
                             std::max<size_t>(1, selected.size())});
        auto verifyShard = [&](size_t shard) -> LogicalResult {
          size_t begin = selected.size() * shard / shards;
          size_t end = selected.size() * (shard + 1) / shards;
          for (Operation *operation : selected.slice(begin, end - begin))
            if (failed(mlir::verify(operation, /*verifyRecursively=*/false)))
              return failure();
          return success();
        };
        return failableParallelForEachN(context, 0, shards, verifyShard);
      };
      // Keep expensive verifier kinds evenly distributed across workers.
      // Projected graphs are emitted in large structural clusters, so a
      // contiguous mixed-kind partition leaves one shard with most calls or
      // resource requests while its peers finish cheap apply operations.
      // Same-depth operations are independent by construction; verifying one
      // kind at a time preserves the required bottom-up depth order.
      std::map<std::string, SmallVector<Operation *, 0>> byName;
      for (Operation *operation : operations)
        byName[operation->getName().getStringRef().str()].push_back(operation);
      for (auto &[name, selected] : byName) {
        auto started = std::chrono::steady_clock::now();
        if (failed(verifyOperations(selected)))
          return failure();
        if (profileKinds)
          llvm::errs() << "fabric-to-phys: verify-depth=" << depth
                       << " op=" << name << " count=" << selected.size()
                       << " time="
                       << std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - started)
                              .count()
                       << "s\n";
      }
    }
    return mlir::verify(graph, /*verifyRecursively=*/false);
  }

  LogicalResult emitProjection() {
    const bool profile = std::getenv("QLX_PROFILE_P2_TO_P3") != nullptr;
    auto previousPhase = std::chrono::steady_clock::now();
    auto reportPhase = [&](StringRef name) {
      if (!profile)
        return;
      auto now = std::chrono::steady_clock::now();
      llvm::errs() << "fabric-to-phys: projection " << name << ' '
                   << std::chrono::duration<double>(now - previousPhase).count()
                   << "s\n";
      previousPhase = now;
    };
    OpBuilder topBuilder(module.getBodyRegion());
    topBuilder.setInsertionPointToEnd(module.getBody());
    Block &rootBlock = callableBody(root).front();
    SmallVector<Type> externalTypes;
    for (BlockArgument argument : rootBlock.getArguments()) {
      if (isa<qlx::fabric::PatchType>(argument.getType()))
        continue;
      auto type = physicalScalarType(argument.getType());
      if (failed(type))
        return root->emitOpError(
            "has an unsupported native physical root argument");
      externalTypes.push_back(*type);
    }
    auto functionType = topBuilder.getFunctionType(externalTypes, {});
    SmallVector<NamedAttribute> attrs{
        topBuilder.getNamedAttr(SymbolTable::getSymbolAttrName(),
                                stringAttr(context, graphName)),
        topBuilder.getNamedAttr("function_type", TypeAttr::get(functionType)),
        topBuilder.getNamedAttr("architecture", device.getPhysicalAttr()),
        topBuilder.getNamedAttr("source_protocol",
                                symbolAttr(context, rootName)),
    };
    if (operatingPoint)
      attrs.push_back(
          topBuilder.getNamedAttr("operating_point", operatingPoint));
    graph = createGeneric(topBuilder, root->getLoc(), "phys.graph", {}, {},
                          attrs, 1);
    Block *body = new Block();
    graph->getRegion(0).push_back(body);
    for (Type type : externalTypes)
      body->addArgument(type, root->getLoc());
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(body);
    auto startedModels = projectionFactoryModels();
    if (failed(startedModels)) {
      graph->erase();
      graph = nullptr;
      return failure();
    }
    for (qlx::phys::FactoryModelOp model : *startedModels)
      createGeneric(
          bodyBuilder, root->getLoc(), "phys.factory_start", {}, {},
          {
              bodyBuilder.getNamedAttr(
                  "factory_model",
                  FlatSymbolRefAttr::get(context, model.getSymName())),
              bodyBuilder.getNamedAttr("event_id", nextEvent("factory_start")),
          });
    SmallVector<PhysicalGroup> arguments;
    size_t external = 0;
    for (BlockArgument argument : rootBlock.getArguments()) {
      if (auto patch = dyn_cast<qlx::fabric::PatchType>(argument.getType())) {
        auto region = rootPatchRegion(patch);
        if (failed(region)) {
          root->emitOpError(
              "patch root argument has no unique compatible QEC region");
          graph->erase();
          graph = nullptr;
          return failure();
        }
        auto allocated =
            allocatePatch(patch, *region, root->getLoc(), bodyBuilder);
        if (failed(allocated)) {
          graph->erase();
          graph = nullptr;
          return failure();
        }
        arguments.push_back(std::move(*allocated));
      } else {
        arguments.push_back(PhysicalGroup{body->getArgument(external++)});
      }
    }
    reportPhase("setup");
    auto outputs = emitCallable(root, arguments, bodyBuilder, {}, rootName);
    if (failed(outputs)) {
      graph->erase();
      graph = nullptr;
      return failure();
    }
    reportPhase("callable");
    SmallVector<Value> flattened;
    for (PhysicalGroup &group : *outputs)
      llvm::append_range(flattened, group);
    createGeneric(bodyBuilder, root->getLoc(), "phys.return", flattened);
    auto finalType = topBuilder.getFunctionType(
        externalTypes, llvm::map_to_vector(flattened, [](Value value) {
          return value.getType();
        }));
    graph->setAttr("function_type", TypeAttr::get(finalType));
    reportPhase("root-finish");
    topBuilder.setInsertionPointAfter(graph);
    emitAnalysisSidecars(topBuilder, root->getLoc());
    reportPhase("analysis-sidecars");
    emitPatchMapping(topBuilder, root->getLoc());
    reportPhase("patch-mapping");
    emitAllocationMapping(topBuilder, root->getLoc());
    reportPhase("allocation-mapping");
    if (auto provider = spacetimeProviderFor(root);
        provider && *provider == SpacetimeProviderKind::SurfaceAutoCCZFactory) {
      auto plan =
          surfaceFactoryRecurrencePlan(cast<qlx::fabric::ProtocolOp>(root));
      if (failed(plan))
        return failure();
    }
    reportPhase("spacetime-recurrence");

    graph->setAttr(
        "linear_value_count",
        topBuilder.getI64IntegerAttr(countLinearPhysicalDefinitions(graph)));

    if (std::getenv("QLX_PROFILE_P2_TO_P3")) {
      llvm::StringMap<int64_t> operationCounts;
      graph->walk([&](Operation *operation) {
        ++operationCounts[operation->getName().getStringRef()];
      });
      llvm::errs() << "fabric-to-phys: before-verify records="
                   << recordProjection.size()
                   << " templates=" << callTemplates.size()
                   << " template-probes=" << templateCompatibilityProbes
                   << " equivalence-rejects=" << templateEquivalenceRejects
                   << " equivalence-cache-hits=" << templateEquivalenceCacheHits
                   << " equivalence-cache-misses="
                   << templateEquivalenceCacheMisses
                   << " profile-rejects=" << templateProfileRejects
                   << " alias-rejects=" << templateAliasRejects
                   << " allocation-rejects=" << templateAllocationRejects
                   << " exact-layout-hits=" << templateExactLayoutHits
                   << " alias-cache-hits=" << templateAliasCacheHits
                   << " alias-cache-misses=" << templateAliasCacheMisses
                   << " fast-path-invocations=" << templateFastPathInvocations
                   << " dependency-visits=" << sourceDataDependencyVisits
                   << " dependency-cache-hits=" << sourceDataDependencyCacheHits
                   << " template-invocations=" << templateInvocations
                   << " elided-invocations=" << elidedTemplateInvocations
                   << " aliased-invocations=" << aliasedTemplateInvocations
                   << " template-boundary-values=" << templateBoundaryValues
                   << " max-template-boundary=" << maxTemplateBoundaryValues
                   << " calls=" << callInstance << " events=" << event
                   << " apply=" << applyEvents
                   << " measure=" << measurementRecord
                   << " sidecars=" << pendingSidecars.size()
                   << " resources=" << operationCounts["phys.resource"]
                   << " acquire=" << operationCounts["phys.acquire"]
                   << " release=" << operationCounts["phys.release"]
                   << " produce=" << operationCounts["phys.produce_resource"]
                   << " consume=" << operationCounts["phys.consume_resource"]
                   << " retry=" << operationCounts["phys.retry"] << "\n";
    }
    reportPhase("profile-counts");

    SmallVector<Attribute> profiles;
    if (auto existing = module->getAttrOfType<ArrayAttr>("qlx.profiles"))
      llvm::append_range(profiles, existing);
    if (!llvm::is_contained(profiles, stringAttr(context, "p3")))
      profiles.push_back(stringAttr(context, "p3"));
    module->setAttr("qlx.profiles", ArrayAttr::get(context, profiles));
    if (deferOutputVerification) {
      // The input Build and the projection-preparation increment are already
      // authenticated. Verify only the graph and top-level sidecars appended
      // by this pass; recursively verifying the complete retained P2 closure
      // would repeat an earlier stage obligation and dominate large builds.
      SmallVector<Operation *, 16> sidecars;
      auto verifySidecars = [&]() -> LogicalResult {
        if (sidecars.empty())
          return success();
        auto verifyStarted = std::chrono::steady_clock::now();
        if (failed(qlx::phys::verifySidecarBatch(sidecars)))
          return failure();
        if (profile)
          llvm::errs() << "fabric-to-phys: verify-top phys.sidecar-batch "
                       << std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - verifyStarted)
                              .count()
                       << "s count=" << sidecars.size() << "\n";
        sidecars.clear();
        return success();
      };
      for (Operation *created = graph; created;
           created = created->getNextNode()) {
        StringRef name = created->getName().getStringRef();
        if (name == "phys.selection_sidecar") {
          sidecars.push_back(created);
          continue;
        }
        if (failed(verifySidecars()))
          return failure();
        auto verifyStarted = std::chrono::steady_clock::now();
        LogicalResult verified = created == graph
                                     ? verifyProjectedGraph()
                                     : mlir::verify(created,
                                                    /*verifyRecursively=*/true);
        if (failed(verified)) {
          created->emitOpError(
              "native physical projection increment failed verification");
          return failure();
        }
        if (profile)
          llvm::errs() << "fabric-to-phys: verify-top "
                       << created->getName().getStringRef() << ' '
                       << std::chrono::duration<double>(
                              std::chrono::steady_clock::now() - verifyStarted)
                              .count()
                       << "s\n";
      }
      if (failed(verifySidecars()))
        return failure();
    } else if (failed(module.verify())) {
      graph->emitOpError("native physical projection failed verification");
      return failure();
    }
    reportPhase("verification");
    return success();
  }

  ModuleOp module;
  MLIRContext *context;
  SymbolTable symbols;
  std::string rootName;
  std::string deviceName;
  std::string graphName;
  bool useParallelPlanning;
  bool preflightOnly;
  bool deferOutputVerification;
  Operation *root = nullptr;
  qlx::DeviceOp device;
  qlx::phys::ArchitectureOp architecture;
  qlx::QECToPhysicalBindingOp qecToPhysical;
  FlatSymbolRefAttr operatingPoint;
  SmallVector<Operation *> reachable;
  llvm::StringSet<> reachableNames;
  llvm::StringMap<RegionBinding> regionBindings;
  llvm::StringMap<qlx::phys::ResourceClassOp> resourceClasses;
  DenseSet<std::pair<Attribute, Attribute>> advertisedNativeActions;
  llvm::StringMap<qlx::phys::ActionOp> physicalActions;
  std::map<std::string, std::set<int64_t>> activeIndices;
  std::map<std::string, std::set<int64_t>> everAllocatedIndices;
  DenseMap<Attribute, ResourceInfo> resources;
  std::map<std::pair<std::string, int64_t>, FlatSymbolRefAttr>
      resourceByIdentity;
  llvm::StringMap<std::string> resourceRegions;
  SmallVector<AllocationRecord, 0> allocations;
  SmallVector<AllocationPlanStep, 0> allocationPlanTrace;
  llvm::StringMap<size_t> resourceToAllocation;
  std::map<std::pair<std::string, int64_t>, std::string> lastReleaseByBinding;
  SmallVector<CanonicalCallTemplate, 0> callTemplates;
  llvm::StringMap<DenseMap<uint64_t, SmallVector<size_t, 1>>>
      callTemplateIndices;
  DenseMap<Operation *, uint64_t> physicalTemplateFingerprints;
  DenseMap<std::pair<Operation *, Operation *>, bool>
      physicalTemplateEquivalenceCache;
  std::map<StateAliasCacheKey, ArrayAttr> stateAliasCache;
  DenseMap<Operation *, unsigned> sourceDataDependencySources;
  DenseMap<Value, SmallVector<uint64_t, 1>> sourceDataDependencies;
  DenseMap<Attribute, SmallVector<Operation *>> profileSidecarDeclarationCache;
  int64_t templateCompatibilityProbes = 0;
  int64_t templateEquivalenceRejects = 0;
  int64_t templateEquivalenceCacheHits = 0;
  int64_t templateEquivalenceCacheMisses = 0;
  int64_t templateProfileRejects = 0;
  int64_t templateAliasRejects = 0;
  int64_t templateAllocationRejects = 0;
  int64_t templateExactLayoutHits = 0;
  int64_t templateAliasCacheHits = 0;
  int64_t templateAliasCacheMisses = 0;
  int64_t templateFastPathInvocations = 0;
  int64_t sourceDataDependencyVisits = 0;
  int64_t sourceDataDependencyCacheHits = 0;
  int64_t templateInvocations = 0;
  int64_t elidedTemplateInvocations = 0;
  int64_t aliasedTemplateInvocations = 0;
  int64_t templateBoundaryValues = 0;
  int64_t maxTemplateBoundaryValues = 0;
  RecordProjectionMap recordProjection;
  SmallVector<RecordProjectionScope *, 2> activeRecordProjectionScopes;
  std::map<RecordProjectionLane, RepeatContext> recordProjectionRepeats;
  SmallVector<PendingSidecar> pendingSidecars;
  std::string currentCallable;
  std::string currentInstance;
  FlatSymbolRefAttr currentProfile;
  DenseMap<Value, TransformFrameProjection> transformFrames;
  DenseMap<Operation *, StringAttr> projectedCallEvents;
  std::string actionFailure;
  Operation *graph = nullptr;
  DictionaryAttr operatingTiming;
  double cycleNanoseconds = 1.0;
  bool hasExplicitCycleNanoseconds = false;
  double reactionNanoseconds = 0.0;
  bool hasExplicitReactionNanoseconds = false;
  llvm::StringMap<FlatSymbolRefAttr> spacetimePlans;
  llvm::StringMap<FlatSymbolRefAttr> componentSpacetimePlans;
  llvm::StringMap<FlatSymbolRefAttr> spacelikePlanTemplates;
  int64_t event = 0;
  int64_t applyEvents = 0;
  int64_t callInstance = 0;
  int64_t resourceIdentity = 0;
  int64_t allocationIdentity = 0;
  int64_t measurementRecord = 0;
  int64_t templateRecordIdentity = 0;
};

class FabricToPhysPass
    : public qlx::fabric::impl::FabricToPhysBase<FabricToPhysPass> {
public:
  using FabricToPhysBase::FabricToPhysBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (rootSymbol.empty() || deviceSymbol.empty()) {
      module.emitError("fabric-to-phys requires root-symbol and device-symbol");
      return signalPassFailure();
    }
    NativeProjection projection(module, rootSymbol, deviceSymbol, graphSymbol,
                                parallelPlanning, preflightOnly,
                                deferOutputVerification);
    if (failed(projection.run()))
      signalPassFailure();
  }
};

} // namespace
