/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "cudaq/Optimizer/Dialect/Quake/QuakeOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/Builders.h"
#include <cassert>
#include <cstddef>
#include <optional>

namespace cudaq::opt {

/// The controls and polarities resulting from expanding statically sized
/// vector controls. Controls with unresolved vector sizes remain intact for
/// callers that can lower them without making the predicate scalar.
struct ExpandedControlVeqs {
  llvm::SmallVector<mlir::Value> controls;
  llvm::SmallVector<bool> polarities;
  bool didExpand = false;
};

inline llvm::SmallVector<bool>
getControlPolarities(mlir::ValueRange controls,
                     std::optional<llvm::ArrayRef<bool>> negatedControls = {}) {
  llvm::SmallVector<bool> polarities(controls.size(), false);
  if (negatedControls)
    for (auto [index, value] : llvm::enumerate(*negatedControls))
      polarities[index] = value;
  return polarities;
}

/// Return true when any control vector cannot be expanded into a statically
/// known number of scalar references.
inline bool hasUnresolvedControlVeq(mlir::ValueRange controls) {
  return llvm::any_of(controls, [](mlir::Value control) {
    return mlir::isa<cudaq::quake::VeqType>(control.getType()) &&
           !cudaq::quake::getVeqSize(control);
  });
}

/// Expand controls with statically known vector sizes into scalar references,
/// including vectors whose known size is visible through RelaxSizeOp. Unknown
/// vector controls are preserved for callers that support them.
inline ExpandedControlVeqs
expandKnownSizedControlVeqs(mlir::OpBuilder &builder, mlir::Location location,
                            mlir::ValueRange controls,
                            llvm::ArrayRef<bool> polarities) {
  assert(controls.size() == polarities.size() &&
         "every control must have a corresponding polarity");

  ExpandedControlVeqs expanded;
  for (auto [index, control] : llvm::enumerate(controls)) {
    if (!mlir::isa<cudaq::quake::VeqType>(control.getType())) {
      expanded.controls.push_back(control);
      expanded.polarities.push_back(polarities[index]);
      continue;
    }

    auto size = cudaq::quake::getVeqSize(control);
    if (!size) {
      expanded.controls.push_back(control);
      expanded.polarities.push_back(polarities[index]);
      continue;
    }

    // extract_ref requires the sized source of a relaxed vector.
    mlir::Value vector = control;
    if (auto relax = control.getDefiningOp<cudaq::quake::RelaxSizeOp>())
      vector = relax.getInputVec();
    for (std::size_t i = 0; i < *size; ++i) {
      expanded.controls.push_back(
          cudaq::quake::ExtractRefOp::create(builder, location, vector, i));
      expanded.polarities.push_back(polarities[index]);
    }
    expanded.didExpand = true;
  }
  return expanded;
}

/// Return the wire result types for a Quake operator with the given controls
/// and targets. Quake orders wire results by controls first, then targets.
inline llvm::SmallVector<mlir::Type>
getWireResultTypes(mlir::OpBuilder &builder, mlir::ValueRange controls,
                   mlir::ValueRange targets) {
  auto wireType = cudaq::quake::WireType::get(builder.getContext());
  llvm::SmallVector<mlir::Type> resultTypes;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      resultTypes.push_back(wireType);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      resultTypes.push_back(wireType);
  return resultTypes;
}

/// Update controls and targets to the corresponding wire results of a newly
/// created Quake operator.
template <typename Op>
inline void threadWireResults(Op op,
                              llvm::MutableArrayRef<mlir::Value> controls,
                              llvm::MutableArrayRef<mlir::Value> targets) {
  unsigned result = 0;
  for (mlir::Value &control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      control = op.getWires()[result++];
  for (mlir::Value &target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      target = op.getWires()[result++];
  assert(result == op.getWires().size() &&
         "gate result count does not match its wire operands");
}

/// Collect threaded values in Quake's wire-result order.
inline llvm::SmallVector<mlir::Value> getWireValues(mlir::ValueRange controls,
                                                    mlir::ValueRange targets) {
  llvm::SmallVector<mlir::Value> values;
  for (mlir::Value control : controls)
    if (mlir::isa<cudaq::quake::WireType>(control.getType()))
      values.push_back(control);
  for (mlir::Value target : targets)
    if (mlir::isa<cudaq::quake::WireType>(target.getType()))
      values.push_back(target);
  return values;
}

} // namespace cudaq::opt
