/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "StaticQubitTarget.h"

using namespace mlir;

std::optional<cudaq::opt::StaticQubitTarget>
cudaq::opt::planStaticQubitTarget(Value target, std::size_t sourceIndex) {
  if (cudaq::quake::isScalarQubitTarget(target))
    return StaticQubitTarget{target, sourceIndex, std::nullopt};
  if (auto size = cudaq::quake::getVeqSize(target); size && *size != 0)
    return StaticQubitTarget{target, sourceIndex, *size - 1};
  return std::nullopt;
}

std::optional<cudaq::opt::StaticQubitTarget>
cudaq::opt::findLastStaticQubitTarget(ValueRange targets) {
  return findLastStaticQubitTarget(
      targets, [](const StaticQubitTarget &) { return true; });
}

Value cudaq::opt::materializeStaticQubitTarget(
    OpBuilder &builder, Location location, const StaticQubitTarget &target) {
  if (!target.elementIndex)
    return target.source;
  return cudaq::quake::ExtractRefOp::create(builder, location, target.source,
                                            *target.elementIndex)
      .getResult();
}
