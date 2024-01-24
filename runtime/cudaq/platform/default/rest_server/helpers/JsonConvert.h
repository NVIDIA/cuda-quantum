/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/EigenDense.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "nlohmann/json.hpp"

/*! \file JsonConvert.h
    \brief Utility to support JSON serialization between the client and server.
*/

using json = nlohmann::json;

namespace std {
// Complex data serialization.
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
struct RemoteJsonSimulationState : public SimulationState {
  std::vector<std::size_t> m_shape;
  std::vector<std::complex<double>> m_data;
  RemoteJsonSimulationState(const std::vector<std::size_t> &shape,
                            const std::vector<std::complex<double>> &data)
      : m_shape(shape), m_data(data) {}

  std::size_t getNumQubits() const override {
    if (m_shape.size() == 1)
      return std::log2(m_data.size());

    return std::log2(m_shape[0]);
  }

  std::vector<std::size_t> getDataShape() const override { return m_shape; }

  double overlap(const cudaq::SimulationState &other) override {
    if (other.getDataShape() != getDataShape())
      throw std::runtime_error(
          "[remote json state] overlap error - other state "
          "dimension not equal to this state dimension.");

    if (other.isDeviceData())
      throw std::runtime_error("remote qpu simulation data cannot compute "
                               "overlap with GPU state data.");

    if (m_shape.size() == 1) {
      return Eigen::Map<Eigen::VectorXcd>(
                 const_cast<std::complex<double> *>(m_data.data()), m_shape[0])
          .transpose()
          .dot(Eigen::Map<Eigen::VectorXcd>(
              reinterpret_cast<cudaq::complex *>(other.ptr()),
              other.getDataShape()[0]))
          .real();
    }

    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        m_data.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
        reinterpret_cast<cudaq::complex *>(other.ptr()),
        other.getDataShape()[0], other.getDataShape()[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(const std::vector<cudaq::complex> &data) override {
    if (data.size() != getDataShape()[0])
      throw std::runtime_error(
          "[remote json state] overlap error - other state "
          "dimension not equal to this state dimension.");
    if (m_shape.size() == 1) {
      return Eigen::Map<Eigen::VectorXcd>(
                 const_cast<std::complex<double> *>(m_data.data()), m_shape[0])
          .transpose()
          .dot(Eigen::Map<Eigen::VectorXcd>(
              reinterpret_cast<cudaq::complex *>(
                  const_cast<cudaq::complex *>(data.data())),
              m_shape[0]))
          .real();
    }

    // Create rho and sigma matrices
    Eigen::MatrixXcd rho = Eigen::Map<Eigen::MatrixXcd>(
        m_data.data(), getDataShape()[0], getDataShape()[1]);
    Eigen::MatrixXcd sigma = Eigen::Map<Eigen::MatrixXcd>(
        reinterpret_cast<cudaq::complex *>(
            const_cast<cudaq::complex *>(data.data())),
        m_shape[0], m_shape[1]);

    // For qubit systems, F(rho,sigma) = tr(rho*sigma) + 2 *
    // sqrt(det(rho)*det(sigma))
    auto detprod = rho.determinant() * sigma.determinant();
    return (rho * sigma).trace().real() + 2 * std::sqrt(detprod.real());
  }

  double overlap(const std::vector<std::complex<float>> &data) override {
    throw std::runtime_error("remote rest json state vector requires FP64 data "
                             "for overlap computation.");
  }

  double overlap(void *data) override {
    throw std::runtime_error(
        "[remote json state] overlap with pointer is not supported.");
  }

  cudaq::complex vectorElement(std::size_t idx) override {
    if (m_shape.size() != 1)
      throw std::runtime_error("[remote rest json] vectorElement not supported "
                               "for density matrix data.");
    return m_data[idx];
  }
  cudaq::complex matrixElement(std::size_t idx, std::size_t jdx) override {
    if (m_shape.size() != 2)
      throw std::runtime_error("[remote rest json] matrixElement not supported "
                               "for state vector data.");
    return Eigen::Map<Eigen::MatrixXcd>(m_data.data(), m_shape[0],
                                        m_shape[1])(idx, jdx);
  }

  void dump(std::ostream &os) const override {
    if (m_shape.size() == 1)
      os << Eigen::Map<Eigen::VectorXcd>(
                const_cast<std::complex<double> *>(m_data.data()), m_shape[0])
         << "\n";
  }

  void *ptr() const override {
    return reinterpret_cast<void *>(
        const_cast<std::complex<double> *>(m_data.data()));
  }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {}
};

// `ExecutionResult` serialization.
// Here, we capture full data (not just bit string statistics) since the remote
// platform can populate simulator-only data, such as `expectationValue`.
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

// `ExecutionContext` serialization.
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
  if (context.simulationState) {
    j["simulationData"]["dim"] = context.simulationState->getDataShape();
    std::complex<double> *hostPtr = nullptr;
    if (context.simulationState->isDeviceData())
      hostPtr = reinterpret_cast<std::complex<double> *>(
          context.simulationState->toHost());
    else
      hostPtr = reinterpret_cast<std::complex<double> *>(
          context.simulationState->ptr());

    j["simulationData"]["data"] = std::vector<std::complex<double>>(
        hostPtr, hostPtr + context.simulationState->getNumElements());
  } else {
    j["simulationData"]["dim"] = std::vector<std::size_t>{};
    j["simulationData"]["data"] = std::vector<std::complex<double>>{};
  }

  if (context.spin.has_value() && context.spin.value() != nullptr) {
    const std::vector<double> spinOpRepr =
        context.spin.value()->getDataRepresentation();
    const auto spinOpN = context.spin.value()->num_qubits();
    j["spin"] = json();
    j["spin"]["num_qubits"] = spinOpN;
    j["spin"]["data"] = spinOpRepr;
  }
  j["registerNames"] = context.registerNames;
}

inline void from_json(const json &j, ExecutionContext &context) {
  j.at("shots").get_to(context.shots);
  j.at("hasConditionalsOnMeasureResults")
      .get_to(context.hasConditionalsOnMeasureResults);

  if (j.contains("result")) {
    std::vector<ExecutionResult> results;
    j.at("result").get_to(results);
    context.result = sample_result(results);
  }

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

  if (j.contains("simulationData")) {
    std::vector<std::size_t> stateDim;
    std::vector<std::complex<double>> stateData;
    j["simulationData"]["dim"].get_to(stateDim);
    j["simulationData"]["data"].get_to(stateData);

    // Create the simulation specific SimulationState
    context.simulationState =
        std::make_unique<RemoteJsonSimulationState>(stateDim, stateData);
  }

  if (j.contains("registerNames"))
    j["registerNames"].get_to(context.registerNames);
}

// Enum data to denote the payload format.
enum class CodeFormat { MLIR, LLVM };

NLOHMANN_JSON_SERIALIZE_ENUM(CodeFormat, {
                                             {CodeFormat::MLIR, "MLIR"},
                                             {CodeFormat::LLVM, "LLVM"},
                                         });

// Payload from client to server for a kernel execution.
class RestRequest {
private:
  /// Holder of the reconstructed execution context.
  std::unique_ptr<ExecutionContext> m_deserializedContext;
  /// Holder of the reconstructed `spin_op`.
  std::unique_ptr<spin_op> m_deserializedSpinOp;
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

  // Underlying code (IR) payload as a Base64 string.
  std::string code;
  // Name of the entry-point kernel.
  std::string entryPoint;
  // Name of the NVQIR simulator to use.
  std::string simulator;
  // The ExecutionContext to run the simulation.
  // The server will execute in this context, and populate simulation data in
  // this context, to be sending back to the client once finished.
  ExecutionContext &executionContext;
  // Format of the code buffer.
  CodeFormat format;
  // Simulation random seed.
  std::size_t seed;
  // List of MLIR passes to be applied on the code before execution.
  std::vector<std::string> passes;
  // Serialized kernel arguments.
  std::vector<uint8_t> args;
  // Version of this schema for compatibility check.
  std::size_t version;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RestRequest, version, entryPoint, simulator,
                                 executionContext, code, args, format, seed,
                                 passes);
};

/// NVCF function version status
enum class FunctionStatus { ACTIVE, DEPLOYING, ERROR, INACTIVE, DELETED };
NLOHMANN_JSON_SERIALIZE_ENUM(FunctionStatus,
                             {
                                 {FunctionStatus::ACTIVE, "ACTIVE"},
                                 {FunctionStatus::DEPLOYING, "DEPLOYING"},
                                 {FunctionStatus::ERROR, "ERROR"},
                                 {FunctionStatus::INACTIVE, "INACTIVE"},
                                 {FunctionStatus::DELETED, "DELETED"},
                             });

// Encapsulates a function version info
// Note: we only parse a subset of required fields (always present). There may
// be other fields, which are not required.
struct NvcfFunctionVersionInfo {
  // Function Id
  std::string id;
  // NVIDIA NGC Org Id (NCA Id)
  std::string ncaId;
  // Version Id
  std::string versionId;
  // Function name
  std::string name;
  // Status of this particular function version
  FunctionStatus status;
  // Function version creation timestamp (ISO 8601 string)
  // e.g., "2024-02-05T00:09:51.154Z"
  std::string createdAt;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(NvcfFunctionVersionInfo, id, ncaId, versionId,
                                 name, status, createdAt);
};

} // namespace cudaq
