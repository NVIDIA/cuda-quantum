/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "nlohmann/json.hpp"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Operation.h"
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace cudaq::opt {

/// Returns false and emits an op diagnostic if [offset, offset+size) overlaps
/// any qubit already in claimed. On success, inserts the range into claimed and
/// returns true.
[[nodiscard]] bool claimPhysicalQubits(mlir::Operation *op,
                                       llvm::DenseSet<std::size_t> &claimed,
                                       std::size_t offset, std::size_t size);

/// Returns the StartingOffset integer attribute on op, or std::nullopt when
/// the attribute is absent.
std::optional<std::size_t> getStartingOffset(mlir::Operation *op);

/// Copies the StartingOffset attribute from `src` to `dst`. Has no effect when
/// `src` carries no StartingOffset attribute.
void propagateStartingOffset(mlir::Operation *dst, mlir::Operation *src);

/// Collect all mapped device qubit ids in op and its nested regions. Mapped
/// qubits can appear either as @mapped_wireset BorrowWireOps or as quake.alloca
/// operations carrying StartingOffset after the RegToMem pass. Returns a
/// sorted, de-duplicated vector of device qubit identities. Accepts any
/// operation as root (e.g., FuncOp or ModuleOp).
llvm::SmallVector<std::size_t> collectMappedDeviceQubits(mlir::Operation *op);

using TargetQubitMappingValues =
    std::vector<std::pair<std::string, std::size_t>>;

/// Collect the target-code logical qubit name for each mapped device qubit.
/// Names follow the IQM mapped-wire convention: device identity N maps to
/// logical name QB(N + 1).
TargetQubitMappingValues collectMappedTargetQubitMapping(mlir::Operation *op);

/// Map of result id to its (qubit local index, register name) pair, as recorded
/// by the QIR profile lowering passes.
using ResultQubitVals =
    std::map<std::size_t, std::pair<std::size_t, std::string>>;

/// Build the output_names JSON value as it is recorded on the
/// QIROutputNamesAttrName attribute. Each entry is
/// [resultId, [qubitNum, registerName, outputPosition]] wrapped in the
/// single-element outer array the runtime parser expects.
///
/// outputPosition is the per-result user-visible output order. It is derived
/// from localIndexToOutputOrder, which assigns an output order to each qubit
/// local index from the borrow lowering order. When localIndexToOutputOrder is
/// empty, the output order falls back to the qubit local index itself, which
/// matches the non-mapped full-QIR lowering. Output positions are the dense
/// ranks of the (outputOrder, resultId) ordering, so they remain compact even
/// when the borrow order skips local indices.
nlohmann::json buildEnrichedOutputNamesJson(
    const ResultQubitVals &resultQubitVals,
    const std::vector<std::size_t> &localIndexToOutputOrder);

} // namespace cudaq::opt
