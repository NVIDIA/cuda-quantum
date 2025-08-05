/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "RuntimeTarget.h"
#include "ServerHelper.h"

namespace cudaq {

/// @brief An abstract base class for providing extra payload to the server.
class ExtraPayloadProvider {
public:
  ExtraPayloadProvider() = default;
  virtual ~ExtraPayloadProvider() = default;

  /// @brief Get the name of the extra payload provider.
  virtual const std::string name() const = 0;

  /// @brief Inject extra payload into the server message.
  /// @param target The RuntimeTarget to generate the payload for.
  /// @param msg The server message to inject extra payload into.
  virtual void injectExtraPayload(const RuntimeTarget &target,
                                  ServerMessage &msg) = 0;
};

/// @brief Register an extra payload provider.
/// @param provider The extra payload provider to register.
void registerExtraPayloadProvider(
    std::unique_ptr<ExtraPayloadProvider> provider);

/// @brief Get a list of all extra payload providers.
/// @return A list of all extra payload providers.
const std::vector<std::unique_ptr<ExtraPayloadProvider>> &
getExtraPayloadProviders();

} // namespace cudaq
