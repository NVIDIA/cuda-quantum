/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq/Optimizer/CodeGen/QIRCodeGenUtils.h"
#include "cudaq/Optimizer/CodeGen/QIRAttributeNames.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include <algorithm>
#include <numeric>

using namespace mlir;
using cudaq::opt::StartingOffsetAttrName;

bool cudaq::opt::claimPhysicalQubits(Operation *op,
                                     llvm::DenseSet<std::size_t> &claimed,
                                     std::size_t offset, std::size_t size) {
  for (std::size_t i = 0; i < size; ++i) {
    auto id = offset + i;
    if (claimed.contains(id)) {
      op->emitOpError("overlaps physical qubit id ") << id;
      return false;
    }
  }
  for (std::size_t i = 0; i < size; ++i)
    claimed.insert(offset + i);
  return true;
}

std::optional<std::size_t> cudaq::opt::getStartingOffset(mlir::Operation *op) {
  auto attr =
      dyn_cast_if_present<IntegerAttr>(op->getAttr(StartingOffsetAttrName));
  if (!attr)
    return std::nullopt;
  return static_cast<std::size_t>(attr.getValue().getLimitedValue());
}

void cudaq::opt::propagateStartingOffset(mlir::Operation *dst,
                                         mlir::Operation *src) {
  if (auto attr = src->getAttr(StartingOffsetAttrName))
    dst->setAttr(StartingOffsetAttrName, attr);
}

static constexpr llvm::StringLiteral mappedWireSetName("mapped_wireset");

llvm::SmallVector<std::size_t>
cudaq::opt::collectMappedDeviceQubits(mlir::Operation *op) {
  llvm::SmallVector<std::size_t> ids;
  op->walk([&](cudaq::quake::BorrowWireOp borrowWire) {
    if (borrowWire.getSetName() == mappedWireSetName)
      ids.push_back(borrowWire.getIdentity());
  });
  op->walk([&](cudaq::quake::AllocaOp alloca) {
    if (auto offset = getStartingOffset(alloca))
      ids.push_back(*offset);
  });
  llvm::sort(ids);
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids;
}

cudaq::opt::TargetQubitMappingValues
cudaq::opt::collectMappedTargetQubitMapping(mlir::Operation *op) {
  TargetQubitMappingValues mapping;
  auto ids = collectMappedDeviceQubits(op);
  mapping.reserve(ids.size());
  for (auto id : ids)
    mapping.emplace_back("QB" + std::to_string(id + 1), id);
  return mapping;
}

nlohmann::json cudaq::opt::buildEnrichedOutputNamesJson(
    const ResultQubitVals &resultQubitVals,
    const std::vector<std::size_t> &localIndexToOutputOrder) {
  std::vector<std::size_t> resultIds;
  std::vector<std::size_t> outputOrders;
  resultIds.reserve(resultQubitVals.size());
  outputOrders.reserve(resultQubitVals.size());
  for (const auto &[resultId, qubitAndName] : resultQubitVals) {
    auto localIndex = qubitAndName.first;
    auto outputOrder = localIndex;
    if (!localIndexToOutputOrder.empty() &&
        localIndex < localIndexToOutputOrder.size())
      outputOrder = localIndexToOutputOrder[localIndex];
    resultIds.push_back(resultId);
    outputOrders.push_back(outputOrder);
  }

  std::vector<std::size_t> rank(resultIds.size());
  std::iota(rank.begin(), rank.end(), 0);
  std::sort(rank.begin(), rank.end(), [&](std::size_t lhs, std::size_t rhs) {
    if (outputOrders[lhs] != outputOrders[rhs])
      return outputOrders[lhs] < outputOrders[rhs];
    return resultIds[lhs] < resultIds[rhs];
  });

  std::vector<std::size_t> outputPositions(resultIds.size());
  for (std::size_t position = 0; position < rank.size(); ++position)
    outputPositions[rank[position]] = position;

  auto entries = nlohmann::json::array();
  std::size_t entryIndex = 0;
  for (const auto &[resultId, qubitAndName] : resultQubitVals) {
    entries.push_back({resultId,
                       {qubitAndName.first, qubitAndName.second,
                        outputPositions[entryIndex]}});
    ++entryIndex;
  }
  return nlohmann::json::array({std::move(entries)});
}
