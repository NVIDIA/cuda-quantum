/****************************************************************-*- C++ -*-****
 * Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include <memory>
#include <vector>

namespace cudaq {
// Forward declaration
struct RuntimeTarget;
/// @brief An abstract base class for providing extra payload to the server.
class ExtraPayloadProvider {
public:
  ExtraPayloadProvider() = default;
  virtual ~ExtraPayloadProvider() = default;

  /// @brief Get the name of the extra payload provider.
  virtual std::string name() const = 0;

  /// @brief Get the type of the extra payload.
  // This would help the server determine how to handle the payload.
  virtual std::string getPayloadType() const = 0;

  /// @brief Get extra payload for the target's service request.
  /// @param target The RuntimeTarget to generate the payload for.
  // Note: the target's server helper will use the extra payload in accordance
  // with the service API.
  virtual std::string getExtraPayload(const RuntimeTarget &target) = 0;
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
