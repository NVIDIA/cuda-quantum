//===- InitAllPasses.h - QLX pass registration composition ------*- C++ -*-===//
//
// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
//
// This source code and the accompanying materials are made available under
// the terms of the Apache License 2.0 which accompanies this distribution.
//
//===----------------------------------------------------------------------===//

#ifndef QLX_INITALLPASSES_H
#define QLX_INITALLPASSES_H

#include "qlx/Conversion/Passes.h"
#include "qlx/Conversion/QuakeToQLXPasses.h"
#include "qlx/Dialect/Fabric/Transforms/Passes.h"
#include "qlx/Dialect/QLX/Transforms/Passes.h"

namespace qlx {

/// Register every native QLX pass with the global MLIR pass registry.
inline void registerAllQLXPasses() {
  registerQLXConversionPasses();
  registerQuakeToQLXPasses();
  registerPrepareQuakeForQLXPipeline();
  fabric::registerFabricTransformPasses();
  registerQLXTransformPasses();
}

} // namespace qlx

#endif // QLX_INITALLPASSES_H
