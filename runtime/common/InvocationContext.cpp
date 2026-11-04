/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "InvocationContext.h"
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>

namespace cudaq::detail {
/// The invocation context handed from an algorithm to the next kernel launch on
/// this thread.
thread_local std::unique_ptr<InvocationContext> pendingInvocationContext =
    nullptr;
} // namespace cudaq::detail

//======= push/pop ==========

[[nodiscard]] cudaq::detail::LiveInvocationContext
cudaq::pushInvocationContext(InvocationContext ctx) {
  return detail::LiveInvocationContext(std::move(ctx));
}

std::optional<cudaq::InvocationContext> cudaq::popInvocationContext() noexcept {
  std::unique_ptr<InvocationContext> claimed =
      std::move(detail::pendingInvocationContext);
  if (!claimed)
    return std::nullopt;
  return std::optional<InvocationContext>(std::move(*claimed));
}

//======= LiveInvocationContext ==========

cudaq::detail::LiveInvocationContext::LiveInvocationContext(
    InvocationContext ctx) {
  if (pendingInvocationContext)
    throw std::logic_error(
        "Nested kernel launch: an invocation context is already pending on "
        "this thread. Algorithms cannot be called before the kernel of an "
        "enclosing algorithm has been launched.");
  pendingInvocationContext =
      std::make_unique<InvocationContext>(std::move(ctx));
}

cudaq::detail::LiveInvocationContext::~LiveInvocationContext() {
  pendingInvocationContext.reset();
}
