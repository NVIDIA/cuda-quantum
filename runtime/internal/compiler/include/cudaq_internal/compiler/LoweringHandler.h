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

namespace cudaq_internal::compiler {

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
  /// Defaults to calling \p getPipeline and checking for non-`nullopt` return
  /// value. Override for handlers that can answer this more efficiently.
  virtual bool supports(const cudaq::config::Profile &input,
                        const cudaq::config::Profile &output) const {
    return getPipeline(input, output).has_value();
  }

  /// Return the pass pipeline lowering \p input into \p output, or `nullopt` if
  /// this handler cannot perform that lowering.
  virtual std::optional<std::string>
  getPipeline(const cudaq::config::Profile &input,
              const cudaq::config::Profile &output) const = 0;
};

/// Return the first handler in \p handlers supporting the conversion
/// \p input -> \p output, or nullptr if none does.
const LoweringHandler *selectLoweringHandler(
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    const cudaq::config::Profile &input, const cudaq::config::Profile &output);

/// The compiler's built-in lowering handlers, in selection order.
///
/// \p emulate and \p emissionFormat are the compiler settings handlers
/// capture for common pipeline preparation.
std::vector<std::unique_ptr<LoweringHandler>>
makeBuiltinLoweringHandlers(bool emulate, std::string emissionFormat);

/// The built-in generic handler lowering to the quake-reference-semantics
/// profile.
std::unique_ptr<LoweringHandler>
makeQuakeReferenceSemanticsHandler(bool emulate, std::string emissionFormat);

} // namespace cudaq_internal::compiler
