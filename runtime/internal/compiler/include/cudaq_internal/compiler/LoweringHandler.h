/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "cudaq/Target/Profile/AllProfiles.h"
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {
struct CompileOptions;
}

namespace cudaq_internal::compiler {

/// The passes one compilation stage adds to the JIT pipeline. The pipeline
/// runs `canonicalize`, prep, deploy, and finalize once for the whole target,
/// and each field says where a stage's passes go:
///
/// \code
///   canonicalize
///   prep pipeline
///     preDeployPasses       (first stage only)
///   jit-deploy-pipeline
///     postDeployPasses
///   jit-finalize-pipeline
///     postFinalizePasses    (last stage only)
/// \endcode
///
/// The first-stage and last-stage limits keep each stage's passes after those
/// of the stages before it. Without them, a second stage's passes before
/// deploy would run ahead of the first stage's passes after deploy.
/// `getPassPipeline` checks each stage and rejects a target that breaks either
/// limit. It also uses this struct to collect the passes of all stages, in
/// stage order.
struct StagePasses {
  /// Passes that run after prep and before deploy.
  std::string preDeployPasses;
  /// Passes that run after deploy and before finalize.
  std::string postDeployPasses;
  /// Passes that run after finalize.
  std::string postFinalizePasses;
};

/// Interface for lowering between IR profiles.
///
/// Registering lowering handlers with the compiler is what enables the JIT
/// compiler to lower between different IR profiles. Whenever new optimization
/// pipelines or support for new IR profiles is added, a corresponding lowering
/// handler must be defined and registered.
class LoweringHandler {
public:
  virtual ~LoweringHandler() = default;

  /// Whether this handler can lower \p input into \p output.
  ///
  /// Defaults to calling `getStagePasses` and checking for a non-`nullopt`
  /// result. Override for handlers that can answer this more efficiently.
  virtual bool supports(const cudaq::config::Profile &input,
                        const cudaq::config::Profile &output,
                        const cudaq::CompileOptions &options) const {
    return getStagePasses(input, output, options).has_value();
  }

  /// Return the passes this handler adds to lower \p input into \p output, or
  /// `nullopt` if this handler cannot perform that lowering.
  virtual std::optional<StagePasses>
  getStagePasses(const cudaq::config::Profile &input,
                 const cudaq::config::Profile &output,
                 const cudaq::CompileOptions &options) const = 0;
};

/// Return the first handler in \p handlers supporting the conversion
/// \p input -> \p output, or nullptr if none does.
const LoweringHandler *selectLoweringHandler(
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    const cudaq::config::Profile &input, const cudaq::config::Profile &output,
    const cudaq::CompileOptions &options);

/// The compiler's built-in lowering handlers, in selection order.
std::vector<std::unique_ptr<LoweringHandler>> makeBuiltinLoweringHandlers();

/// The built-in generic handler lowering to the quake-reference-semantics
/// profile.
std::unique_ptr<LoweringHandler> makeQuakeReferenceSemanticsHandler();

/// The built-in handler lowering to the `quake-unmapped` profile.
std::unique_ptr<LoweringHandler> makeQuakeUnmappedHandler();

} // namespace cudaq_internal::compiler
