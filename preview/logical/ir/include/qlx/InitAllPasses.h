/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifndef QLX_INITALLPASSES_H
#define QLX_INITALLPASSES_H

#include "qlx/Conversion/Passes.h"
#ifdef QLX_HAS_CUDAQ_QUAKE
#include "qlx/Conversion/QuakeToQLXPasses.h"
#endif
#include "qlx/Compiler/SpacetimeProviders.h"
#include "qlx/Dialect/Fabric/Transforms/Passes.h"
#include "qlx/Dialect/Phys/Transforms/Passes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"

namespace qlx {

/// Register every native QLX pass with the global MLIR pass registry.
inline void registerAllQLXPasses() {
  spacetime::registerBuiltinProviders();
  registerQLXConversionPasses();
#ifdef QLX_HAS_CUDAQ_QUAKE
  registerQuakeToQLXPasses();
  registerPrepareQuakeForQLXPipeline();
#endif
  fabric::registerFabricTransformPasses();
  phys::registerPhysTransformPasses();
  registerQLXTransformPasses();
}

} // namespace qlx

#endif // QLX_INITALLPASSES_H
