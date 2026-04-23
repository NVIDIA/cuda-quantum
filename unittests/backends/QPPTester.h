/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include "cudaq/algorithms/sample/policy.h"
#include <string>
#include <vector>

/// Wrapper class for QppCircuitSimulator instances to expose testing related
/// methods.
template <typename Simulator>
class QppCircuitSimulatorTester : public Simulator {
public:
  std::string getSampledBitString(std::vector<std::size_t> &&qubits) {
    cudaq::ExecutionContext ctx("sample", 1);
    // a quick set-reset to trigger sampling
    this->configureExecutionContext(ctx);
    cudaq::detail::setExecutionContext(&ctx);
    auto sampleResults =
        this->finalizeExecutionContext(cudaq::sample_policy{}, ctx);
    cudaq::detail::resetExecutionContext();

    return sampleResults.begin()->first;
  }

  auto getStateVector() {
    this->flushGateQueue();
    return this->state;
  }
};
