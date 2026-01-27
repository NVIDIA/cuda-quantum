/****************************************************************-*- C++ -*-****
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/ExecutionContext.h"
#include <string>
#include <vector>

/// Wrapper class for QppCircuitSimulator instances to expose testing related
/// methods.
template <typename Simulator>
class QppCircuitSimulatorTester : public Simulator {
public:
  std::string getSampledBitString(std::vector<std::size_t> &&qubits) {
    cudaq::ExecutionContext ctx("sample", 1);
    this->setExecutionContext(&ctx);
    // TODO: replace this custom code with a simulator-provided method, once
    // resetExecutionContext is split into a "finalize" and a "reset" phase.
    this->sampleQubits.resize(this->getNumQubits());
    std::iota(this->sampleQubits.begin(), this->sampleQubits.end(), 0);
    this->flushGateQueue();
    this->flushAnySamplingTasks();
    this->sampleQubits.clear();
    this->executionContext = nullptr;
    auto sampleResults = ctx.result;
    return sampleResults.begin()->first;
  }

  auto getStateVector() {
    this->flushGateQueue();
    return this->state;
  }
};
