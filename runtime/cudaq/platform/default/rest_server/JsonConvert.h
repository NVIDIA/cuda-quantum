/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "nlohmann/json.hpp"

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
inline void to_json(json &j, const ExecutionResult &result) {
  j = json{{"counts", result.counts},
           {"registerName", result.registerName},
           {"sequentialData", result.sequentialData}};
  if (result.expectationValue.has_value())
    j["expectationValue"] = result.expectationValue.value();
}

inline void from_json(const json &j, ExecutionResult &result) {
  j.at("counts").get_to(result.counts);
  j.at("registerName").get_to(result.registerName);
  j.at("sequentialData").get_to(result.sequentialData);
  double expVal = 0.0;
  if (j.contains("expectationValue")) {
    j.at("expectationValue").get_to(expVal);
    result.expectationValue = expVal;
  }
}

inline void to_json(json &j, const ExecutionContext &context) {
  j = json{{"name", context.name},
           {"shots", context.shots},
           {"hasConditionalsOnMeasureResults",
            context.hasConditionalsOnMeasureResults}};

  const auto &regNames = context.result.register_names();
  // Here, we serialize the full lists of ExecutionResult records so that
  // expectation values are captured.
  std::vector<ExecutionResult> results;
  for (const auto &regName : regNames) {
    ExecutionResult result;
    result.registerName = regName;
    result.counts = context.result.to_map(regName);
    result.sequentialData = context.result.sequential_data(regName);
    if (context.result.has_expectation(regName))
      result.expectationValue = context.result.expectation(regName);
    results.emplace_back(std::move(result));
  }
  j["result"] = results;

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

inline void from_json(const json &j, ExecutionContext &context) {
  j.at("shots").get_to(context.shots);
  j.at("hasConditionalsOnMeasureResults")
      .get_to(context.hasConditionalsOnMeasureResults);
  std::vector<ExecutionResult> results;
  j.at("result").get_to(results);
  context.result = sample_result(results);

  if (j.contains("expectationValue")) {
    double expectationValue;
    j["expectationValue"].get_to(expectationValue);
    context.expectationValue = expectationValue;
  }

  if (j.contains("spin")) {
    std::vector<double> spinData;
    j["spin"]["data"].get_to(spinData);
    const std::size_t nQubits = j["spin"]["num_qubits"];
    auto serializedSpinOps = std::make_unique<spin_op>(spinData, nQubits);
    context.spin = serializedSpinOps.release();
  }

  std::vector<std::size_t> stateDim;
  std::vector<std::complex<double>> stateData;
  j["simulationData"]["dim"].get_to(stateDim);
  j["simulationData"]["data"].get_to(stateData);
  context.simulationData =
      std::make_tuple(std::move(stateDim), std::move(stateData));
}

enum CodeFormat { MLIR, LLVM };

NLOHMANN_JSON_SERIALIZE_ENUM(CodeFormat, {
                                             {MLIR, "MLIR"},
                                             {LLVM, "LLVM"},
                                         });

class RestRequest {
private:
  /// Holder of the reconstructed execution context.
  std::unique_ptr<ExecutionContext> m_deserializedContext;
  /// Holder of the reconstructed `spin_op`.
  std::unique_ptr<spin_op> m_deserializedSpinOp;
  std::vector<uint8_t> code;
  // Version number of this payload.
  // This needs to be bumped whenever a breaking change is introduced.
  // e.g., adding/removing non-optional fields, changing field names, etc.
  static constexpr std::size_t SCHEMA_VERSION = 1;

public:
  RestRequest(ExecutionContext &context)
      : executionContext(context), version(SCHEMA_VERSION) {}
  RestRequest(const json &j)
      : m_deserializedContext(
            std::make_unique<ExecutionContext>(j["executionContext"]["name"])),
        executionContext(*m_deserializedContext) {
    from_json(j, *this);
    // Take the ownership of the spin_op pointer for proper cleanup.
    if (executionContext.spin.has_value() && executionContext.spin.value())
      m_deserializedSpinOp.reset(executionContext.spin.value());
    // If the incoming JSON payload has a different version than the one this is
    // compiled with, throw an error. Note: we don't support automatically
    // versioning the payload (converting payload between different versions) at
    // the moment.
    if (version != SCHEMA_VERSION)
      throw std::runtime_error(
          fmt::format("Incompatible JSON schema detected: expected version {}, "
                      "got version {}",
                      SCHEMA_VERSION, version));
  }

  void setCode(const std::string &codeStr) {
    code.assign(codeStr.begin(), codeStr.end());
  }

  std::string_view getCode() const {
    return std::string_view(reinterpret_cast<const char *>(code.data()),
                            code.size());
  }

  std::string entryPoint;
  std::string simulator;
  ExecutionContext &executionContext;
  CodeFormat format;
  std::size_t seed;
  std::vector<std::string> passes;
  std::vector<uint8_t> args;
  std::size_t version;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RestRequest, version, entryPoint, simulator,
                                 executionContext, code, args, format, seed,
                                 passes);
};
} // namespace cudaq
