/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"

#include <string>

namespace qlx {

#define GEN_PASS_DECL
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "qlx/Conversion/QuakeToQLXPasses.h.inc"

/// Options for the `quake-to-qlx-pipeline` named MLIR pipeline.
struct QuakeToQLXPipelineOptions
    : public mlir::PassPipelineOptions<QuakeToQLXPipelineOptions> {
  Option<std::string> entryPoint{
      *this, "entry-point",
      llvm::cl::desc("Source entry point to import; forwarded to "
                     "convert-quake-to-qlx"),
      llvm::cl::init("")};
};

/// Add the QLX-side boundary cleanup (`prepare-quake-for-qlx`) to `pm`. Its
/// input must already have passed through CUDA-Q's `convert-to-linear-values`
/// preparation.
void buildPrepareQuakeForQLXPipeline(mlir::OpPassManager &pm);

/// Add the complete Quake-to-P0 import to `pm`: `prepare-quake-for-qlx`
/// followed by `convert-quake-to-qlx`, with `options.entryPoint` forwarded to
/// the conversion. Callers should use this rather than composing the two
/// stages themselves. The conversion rejects input that has not been prepared.
void buildQuakeToQLXPipeline(mlir::OpPassManager &pm,
                             const QuakeToQLXPipelineOptions &options);

/// Register the `prepare-quake-for-qlx` and `quake-to-qlx-pipeline` named MLIR
/// pipelines. The individual passes stay registered separately (see
/// `registerQuakeToQLXPasses`) for tests that exercise only one stage.
void registerQuakeToQLXPipelines();

} // namespace qlx
