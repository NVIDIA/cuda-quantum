/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "qlx/Conversion/Passes.h"
#include "cudaq/Optimizer/Transforms/VerifyLinearOwnership.h"
#ifdef QLX_HAS_CUDAQ_QUAKE
#include "qlx/Conversion/PhysToQuakePasses.h"
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
  registerPhysToQuakePasses();
#endif
  fabric::registerFabricTransformPasses();
  phys::registerPhysTransformPasses();
  registerQLXTransformPasses();
  // The linear-ownership verifier lives in CUDA-Q core. Registering it again
  // next to the core passes is harmless (MLIR only rejects a second
  // registration that maps the same argument to a different pass).
  cudaq::opt::registerLinearOwnershipVerifierPass();
}

} // namespace qlx
