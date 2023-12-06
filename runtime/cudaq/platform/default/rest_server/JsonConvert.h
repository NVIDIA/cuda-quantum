/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/ExecutionContext.h"
#include "nlohmann/json.hpp"
#include <deque>
using json = nlohmann::json;

namespace std {
template <class T>
void to_json(json &j, const std::complex<T> &p) {
  j = json{p.real(), p.imag()};
}

template <class T>
void from_json(const json &j, std::complex<T> &p) {
  p.real(j.at(0));
  p.imag(j.at(1));
}
} // namespace std

namespace cudaq {
void to_json(json &j, const ExecutionContext &context) {
  j = json{{"name", context.name},
           {"shots", context.shots},
           {"hasConditionalsOnMeasureResults",
            context.hasConditionalsOnMeasureResults}};

  j["result"] = context.result.serialize();
  if (context.expectationValue.has_value()) {
    j["expectationValue"] = context.expectationValue.value();
  }
  j["simulationData"] = json();
  j["simulationData"]["dim"] = std::get<0>(context.simulationData);
  j["simulationData"]["data"] = std::get<1>(context.simulationData);
  if (context.spin.has_value() && context.spin.value() != nullptr) {
    const std::vector<double> spinOpRepr =
        context.spin.value()->getDataRepresentation();
    const auto spinOpN = context.spin.value()->num_qubits();
    j["spin"] = json();
    j["spin"]["num_qubits"] = spinOpN;
    j["spin"]["data"] = spinOpRepr;
  }
}

void from_json(const json &j, ExecutionContext &context) {
  j.at("shots").get_to(context.shots);
  j.at("hasConditionalsOnMeasureResults")
      .get_to(context.hasConditionalsOnMeasureResults);

  std::vector<std::size_t> sampleData;
  j["result"].get_to(sampleData);
  context.result.deserialize(sampleData);
  if (j.contains("expectationValue")) {
    double expectationValue;
    j["expectationValue"].get_to(expectationValue);
    context.expectationValue = expectationValue;
  }

  if (j.contains("spin")) {
    std::vector<double> spinData;
    j["spin"]["data"].get_to(spinData);
    const std::size_t nQubits = j["spin"]["num_qubits"];
    // Static container of reconstructed spin_op instances for proper cleanup.
    // Use std::deque to prevent pointer invalidation.
    static thread_local std::deque<cudaq::spin_op> cacheSerializedSpinOps;
    cacheSerializedSpinOps.emplace_back(spinData, nQubits);
    context.spin = &cacheSerializedSpinOps.back();
  }

  std::vector<std::size_t> stateDim;
  std::vector<std::complex<double>> stateData;
  j["simulationData"]["dim"].get_to(stateDim);
  j["simulationData"]["data"].get_to(stateData);
  context.simulationData =
      std::make_tuple(std::move(stateDim), std::move(stateData));
}
} // namespace cudaq