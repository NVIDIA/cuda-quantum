/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "cudaq_internal/compiler/LoweringHandler.h"

const cudaq_internal::compiler::LoweringHandler *
cudaq_internal::compiler::selectLoweringHandler(
    const std::vector<std::unique_ptr<LoweringHandler>> &handlers,
    const cudaq::config::Profile &input, const cudaq::config::Profile &output,
    const cudaq::CompileOptions &options) {
  for (const auto &handler : handlers)
    if (handler->supports(input, output, options))
      return handler.get();
  return nullptr;
}

std::vector<std::unique_ptr<cudaq_internal::compiler::LoweringHandler>>
cudaq_internal::compiler::makeBuiltinLoweringHandlers() {
  std::vector<std::unique_ptr<LoweringHandler>> handlers;
  handlers.push_back(makeQuakeReferenceSemanticsHandler());
  return handlers;
}
