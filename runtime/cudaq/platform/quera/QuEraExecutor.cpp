/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "QuEraServerHelper.h"
#include "common/BraketExecutor.h"

namespace cudaq {
/// @brief The Executor subclass for QuEra target
class QuEraExecutor : public BraketExecutor {
public:
  ~QuEraExecutor() = default;

  ServerJobPayload checkHelperAndCreateJob(
      std::vector<KernelExecution> &codesToExecute) override {
    auto queraServerHelper = dynamic_cast<QuEraServerHelper *>(serverHelper);
    assert(queraServerHelper);
    queraServerHelper->setShots(shots);

    auto config = queraServerHelper->getConfig();
    CUDAQ_INFO("Backend config: {}, shots {}", config, shots);
    config.insert({"shots", std::to_string(shots)});

    return queraServerHelper->createJob(codesToExecute);
  }
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::QuEraExecutor, quera);
