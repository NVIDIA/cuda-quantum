/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "cudaq/Optimizer/Transforms/Passes.h"
#include "cudaq/Support/Device.h"
#include "cudaq/Support/Placement.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/ScopedPrinter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/TopologicalSortUtils.h"

#define DEBUG_TYPE "quantum-mapper"

using namespace cudaq;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Generated logic
//===----------------------------------------------------------------------===//

namespace cudaq::opt {
#define GEN_PASS_DEF_MAPPINGPASS
#include "cudaq/Optimizer/Transforms/Passes.h.inc"
} // namespace cudaq::opt

namespace {

//===----------------------------------------------------------------------===//
// Placement
//===----------------------------------------------------------------------===//

void identityPlacement(Placement &placement) {
  for (unsigned i = 0, end = placement.getNumVirtualQ(); i < end; ++i)
    placement.map(Placement::VirtualQ(i), Placement::DeviceQ(i));
}

//===----------------------------------------------------------------------===//
// Routing
//===----------------------------------------------------------------------===//

/// This class encapsulates an quake operation that uses wires with information
/// about the virtual qubits these wires correspond.
struct VirtualOp {
  quake::WireInterface iface;
  SmallVector<Placement::VirtualQ, 2> qubits;

  VirtualOp(quake::WireInterface iface, ArrayRef<Placement::VirtualQ> qubits)
      : iface(iface), qubits(qubits) {}
};

class SabreRouter {
  using WireMap = DenseMap<Value, Placement::VirtualQ>;
  using Swap = std::pair<Placement::DeviceQ, Placement::DeviceQ>;

public:
  SabreRouter(const Device &device, WireMap &wireMap, Placement &placement)
      : device(device), wireToVirtualQ(wireMap), placement(placement),
        phyDecay(device.getNumQubits(), 1.0), phyToWire(device.getNumQubits()) {
  }

  void route(Block &block, ArrayRef<quake::NullWireOp> sources);

private:
  void visitUsers(ResultRange::user_range users,
                  SmallVectorImpl<VirtualOp> &layer,
                  SmallVectorImpl<Operation *> *incremented = nullptr);

  LogicalResult mapOperation(VirtualOp &op);

  LogicalResult mapFrontLayer();

  void selectExtendedLayer();

  double computeLayerCost(ArrayRef<VirtualOp> layer);

  Swap chooseSwap();

private:
  const Device &device;
  WireMap &wireToVirtualQ;
  Placement &placement;

  // Parameters
  unsigned extendedLayerSize = 20;
  float extendedLayerWeight = 0.5;
  float decayDelta = 0.5;
  unsigned roundsDecayReset = 5;

  // Internal data
  SmallVector<VirtualOp> frontLayer;
  SmallVector<VirtualOp> extendedLayer;
  llvm::SmallSet<Placement::DeviceQ, 32> involvedPhy;
  SmallVector<float> phyDecay;

  SmallVector<Value> phyToWire;

  /// Keeps track of how many times an operation was visited.
  DenseMap<Operation *, unsigned> visited;

#ifndef NDEBUG
  /// A logger used to emit diagnostics during the maping process.
  llvm::ScopedPrinter logger{llvm::dbgs()};
#endif
};

void SabreRouter::visitUsers(ResultRange::user_range users,
                             SmallVectorImpl<VirtualOp> &layer,
                             SmallVectorImpl<Operation *> *incremented) {
  for (auto user : users) {
    auto [entry, created] = visited.try_emplace(user, 1);
    if (!created)
      entry->second += 1;
    if (incremented)
      incremented->push_back(user);

    auto optor = dyn_cast<quake::WireInterface>(user);
    assert(optor && "TODO");

    auto wires = optor.getWireOperands();
    if (entry->second == wires.size()) {
      SmallVector<Placement::VirtualQ, 2> qubits;
      for (auto wire : wires)
        qubits.push_back(wireToVirtualQ[wire]);
      layer.emplace_back(optor, qubits);
    }
  }
}

LogicalResult SabreRouter::mapOperation(VirtualOp &op) {
  // Take the device qubits from this operation.
  SmallVector<Placement::DeviceQ, 2> deviceQubits;
  for (auto vr : op.qubits)
    deviceQubits.push_back(placement.getPhy(vr));

  // An operation cannot be mapped if it is not a measurement and uses two
  // qubits virtual qubit that are no adjacently placed.
  if (!op.iface->hasTrait<QuantumMeasure>() && deviceQubits.size() == 2 &&
      !device.areConnected(deviceQubits[0], deviceQubits[1]))
    return failure();

  // Rewire the operation.
  SmallVector<Value, 2> newOpWires;
  for (auto phy : deviceQubits)
    newOpWires.push_back(phyToWire[phy.index]);
  op.iface.setWireOperands(newOpWires);

  if (isa<quake::SinkOp>(op.iface))
    return success();

  // Update the mapping between device qubits and wires.
  for (auto &&[w, q] : llvm::zip_equal(op.iface.getWireResults(), deviceQubits))
    phyToWire[q.index] = w;

  return success();
}

LogicalResult SabreRouter::mapFrontLayer() {
  bool mappedAtLeastOne = false;
  SmallVector<VirtualOp> newFrontLayer;

  LLVM_DEBUG({
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
  });
  for (auto op : frontLayer) {
    LLVM_DEBUG({
      logger.startLine() << "* ";
      op.iface->print(logger.getOStream(),
                      OpPrintingFlags().printGenericOpForm());
    });
    if (failed(mapOperation(op))) {
      LLVM_DEBUG(logger.getOStream() << " --> FAILURE\n");
      newFrontLayer.push_back(op);
      for (auto vr : op.qubits)
        involvedPhy.insert(placement.getPhy(vr));
      LLVM_DEBUG({
        auto phy0 = placement.getPhy(op.qubits[0]);
        auto phy1 = placement.getPhy(op.qubits[1]);
        logger.indent();
        logger.startLine() << "+ virtual qubits: " << op.qubits[0] << ", "
                           << op.qubits[1] << '\n';
        logger.startLine() << "+ device qubits: " << phy0 << ", " << phy1
                           << '\n';
        logger.unindent();
      });
      continue;
    }
    LLVM_DEBUG(logger.getOStream() << " --> SUCCESS\n");
    mappedAtLeastOne = true;
    visitUsers(op.iface->getUsers(), newFrontLayer);
  }
  LLVM_DEBUG(logger.unindent());
  frontLayer = std::move(newFrontLayer);
  return mappedAtLeastOne ? success() : failure();
}

void SabreRouter::selectExtendedLayer() {
  extendedLayer.clear();
  SmallVector<Operation *, 20> incremented;
  SmallVector<VirtualOp> tmpLayer = frontLayer;
  while (!tmpLayer.empty() && extendedLayer.size() < extendedLayerSize) {
    SmallVector<VirtualOp> newTmpLayer;
    for (VirtualOp &op : tmpLayer)
      visitUsers(op.iface->getUsers(), newTmpLayer, &incremented);
    for (VirtualOp &op : newTmpLayer)
      // We only add operations that can influence placement to the extended
      // frontlayer, i.e., quantum operators that use two qubits.
      if (!op.iface->hasTrait<QuantumMeasure>() &&
          op.iface.getWireOperands().size() == 2)
        extendedLayer.emplace_back(op);
    tmpLayer = std::move(newTmpLayer);
  }

  for (auto op : incremented)
    visited[op] -= 1;
}

double SabreRouter::computeLayerCost(ArrayRef<VirtualOp> layer) {
  double cost = 0.0;
  for (VirtualOp const &op : layer) {
    auto phy0 = placement.getPhy(op.qubits[0]);
    auto phy1 = placement.getPhy(op.qubits[1]);
    cost += device.getDistance(phy0, phy1) - 1;
  }
  return cost / layer.size();
}

SabreRouter::Swap SabreRouter::chooseSwap() {
  // Obtain SWAP candidates
  SmallVector<Swap> candidates;
  for (auto phy0 : involvedPhy)
    for (auto phy1 : device.getNeighbours(phy0))
      candidates.emplace_back(phy0, phy1);

  if (extendedLayerSize)
    selectExtendedLayer();

  // Compute cost
  SmallVector<double> cost;
  for (auto [phy0, phy1] : candidates) {
    placement.swap(phy0, phy1);
    double swapCost = computeLayerCost(frontLayer);
    double maxDecay = std::max(phyDecay[phy0.index], phyDecay[phy1.index]);

    if (!extendedLayer.empty()) {
      double extendedLayerCost =
          computeLayerCost(extendedLayer) / extendedLayer.size();
      swapCost /= frontLayer.size();
      swapCost += extendedLayerWeight * extendedLayerCost;
    }

    cost.emplace_back(maxDecay * swapCost);
    placement.swap(phy0, phy1);
  }

  // Find and return the swap with minimal cost
  std::size_t minIdx = 0u;
  for (std::size_t i = 1u, end = cost.size(); i < end; ++i)
    if (cost[i] < cost[minIdx])
      minIdx = i;

  LLVM_DEBUG({
    logger.startLine() << "Choosing a swap:\n";
    logger.indent();
    logger.startLine() << "Involved device qubits:";
    for (auto phy : involvedPhy)
      logger.getOStream() << " " << phy;
    logger.getOStream() << "\n";
    logger.startLine() << "Swap candidates:\n";
    logger.indent();
    for (auto &&[qubits, c] : llvm::zip_equal(candidates, cost))
      logger.startLine() << "* " << qubits.first << ", " << qubits.second
                         << " (cost = " << c << ")\n";
    logger.getOStream() << "\n";
    logger.unindent();
    logger.startLine() << "Selected swap: " << candidates[minIdx].first << ", "
                       << candidates[minIdx].second << '\n';
    logger.unindent();
  });
  return candidates[minIdx];
}

void SabreRouter::route(Block &block, ArrayRef<quake::NullWireOp> sources) {
#ifndef NDEBUG
  const char *logLineComment =
      "//===-------------------------------------------===//\n";
#endif

  LLVM_DEBUG({
    logger.getOStream() << "\n";
    logger.startLine() << logLineComment;
    logger.startLine() << "Mapping front layer:\n";
    logger.indent();
    for (auto op : sources)
      logger.startLine() << "* " << op << " --> SUCCESS\n";
    logger.unindent();
    logger.startLine() << logLineComment;
  });

  // The source ops can always be mapped.
  for (quake::NullWireOp nullWire : sources) {
    visitUsers(nullWire->getUsers(), frontLayer);
    Value wire = nullWire.getResult();
    auto phy = placement.getPhy(wireToVirtualQ[wire]);
    phyToWire[phy.index] = wire;
  }

  OpBuilder builder(&block, block.begin());
  auto wireType = builder.getType<quake::WireType>();
  auto addSwap = [&](Placement::DeviceQ q0, Placement::DeviceQ q1) {
    placement.swap(q0, q1);
    auto swap = builder.create<quake::SwapOp>(
        builder.getUnknownLoc(), TypeRange{wireType, wireType}, false,
        ValueRange{}, ValueRange{},
        ValueRange{phyToWire[q0.index], phyToWire[q1.index]},
        DenseBoolArrayAttr{});
    phyToWire[q0.index] = swap.getResult(0);
    phyToWire[q1.index] = swap.getResult(1);
  };

  uint32_t numSwapSearches = 0u;
  while (!frontLayer.empty()) {
    LLVM_DEBUG({
      logger.getOStream() << "\n";
      logger.startLine() << logLineComment;
    });

    if (succeeded(mapFrontLayer()))
      continue;

    LLVM_DEBUG(logger.getOStream() << "\n";);

    // Add a swap
    numSwapSearches += 1u;
    auto [phy0, phy1] = chooseSwap();
    addSwap(phy0, phy1);
    involvedPhy.clear();

    // Update decay
    if ((numSwapSearches % roundsDecayReset) == 0) {
      std::fill(phyDecay.begin(), phyDecay.end(), 1.0);
    } else {
      phyDecay[phy0.index] += decayDelta;
      phyDecay[phy1.index] += decayDelta;
    }
  }
  LLVM_DEBUG(logger.startLine() << '\n' << logLineComment << '\n';);
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct Mapper : public cudaq::opt::impl::MappingPassBase<Mapper> {
  using MappingPassBase::MappingPassBase;

  void runOnOperation() override {
    auto func = getOperation();
    auto &blocks = func.getBlocks();

    // Current limitations:
    //  * Can only map a entry-point kernel
    //  * The kernel can only have one block

    // FIXME: Add the ability to handle multiple blocks.
    if (blocks.size() > 1) {
      func.emitError("The mapper cannot handle multiple blocks");
      signalPassFailure();
    }

    // Get device
    StringRef deviceDef = device;
    StringRef name = deviceDef.take_front(deviceDef.find_first_of('('));
    std::size_t deviceDim[2] = {0, 0};
    if (name.size() < deviceDef.size())
      deviceDef = deviceDef.drop_front(name.size());

    if (deviceDef.consume_front("(")) {
      deviceDef = deviceDef.ltrim();
      deviceDef.consumeInteger(10, deviceDim[0]);
      if (deviceDef.trim().consume_front(","))
        deviceDef.consumeInteger(10, deviceDim[1]);
      if (!deviceDef.trim().consume_front(")")) {
        func.emitError("Missing closing ')' in device option");
        signalPassFailure();
      }
    }

    // Sanity checks and create a wire to virtual qubit mapping.
    Block &block = *blocks.begin();
    SmallVector<quake::NullWireOp> sources;
    DenseMap<Value, Placement::VirtualQ> wireToVirtualQ;
    for (Operation &op : block.getOperations()) {
      if (auto qop = dyn_cast<quake::NullWireOp>(op)) {
        // Assing a new virtual qubit to the resulting wire.
        wireToVirtualQ[qop.getResult()] = Placement::VirtualQ(sources.size());
        sources.push_back(qop);
      } else if (auto iface = dyn_cast<quake::WireInterface>(op)) {
        // Make sure the operation is using value semantics.
        if (!quake::isValueSSAForm(&op)) {
          func.emitError("The mapper requires value semantics.");
          signalPassFailure();
          return;
        }

        // Since `quake.sink` operations do not generate new wires, we don't
        // need to further analyze.
        if (isa<quake::SinkOp>(op))
          continue;

        // Get the wire operands and check if the operatos uses at most two
        // qubits. N.B: Measurements do not have this restriction.
        auto wireOperands = iface.getWireOperands();
        if (!iface->hasTrait<QuantumMeasure>() && wireOperands.size() > 2) {
          func.emitError("Cannot map a kernel with operators that use more "
                         "than two qubits.");
          signalPassFailure();
          return;
        }

        // Map the result wires to the appropriate virtual qubits.
        for (auto &&[wire, newWire] :
             llvm::zip_equal(wireOperands, iface.getWireResults()))
          wireToVirtualQ[newWire] = wireToVirtualQ[wire];
      }
    }

    std::size_t deviceNumQubits =
        name == "grid" ? deviceDim[0] * deviceDim[1] : deviceDim[0];

    if (deviceNumQubits && sources.size() > deviceNumQubits) {
      signalPassFailure();
      return;
    }

    if (!deviceNumQubits) {
      deviceDim[0] =
          name == "grid" ? std::sqrt(sources.size()) : sources.size();
      deviceDim[1] = deviceDim[0];
    }

    Device d = llvm::StringSwitch<Device>(name)
                   .Case("path", Device::path(deviceDim[0]))
                   .Case("ring", Device::ring(deviceDim[0]))
                   .Case("star", Device::star(deviceDim[0]))
                   .Case("grid", Device::grid(deviceDim[0], deviceDim[1]))
                   .Default(Device());

    if (d.getNumQubits() == 0) {
      func.emitError("Trying to target an empty device.");
      signalPassFailure();
      return;
    }

    // Place
    Placement placement(sources.size(), d.getNumQubits());
    identityPlacement(placement);

    // Route
    SabreRouter router(d, wireToVirtualQ, placement);
    router.route(*blocks.begin(), sources);
    sortTopologically(&block);
  }
};

} // namespace
