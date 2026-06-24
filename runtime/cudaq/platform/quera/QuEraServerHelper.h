/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BraketServerHelper.h"

namespace cudaq {
class QuEraServerHelper : public cudaq::BraketServerHelper {
public:
  /// @brief Returns the name of the server helper.
  const std::string name() const override { return "quera"; }

  /// @brief Initializes the server helper with the provided backend
  /// configuration.
  void initialize(BackendConfig config) override;

  /// @brief Creates a quantum computation job using the provided kernel
  /// executions and returns the corresponding payload.
  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override;

  /// @brief Given a successful job and the success response,
  /// retrieve the results and map them to a sample_result.
  cudaq::sample_result processResults(ServerMessage &postJobResponse,
                                      std::string &jobId) override;
};

} // namespace cudaq
