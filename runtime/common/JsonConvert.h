/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "GPUInfo.h"
#include "common/ExecutionContext.h"
#include "common/FmtCore.h"
#include "cudaq/Support/Version.h"
#include "cudaq/optimizers.h"
#include "nlohmann/json.hpp"
/*! \file
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

  if (context.expectationValue.has_value())
    j["expectationValue"] = context.expectationValue.value();
  if (context.optResult.has_value())
    j["optResult"] = context.optResult.value();
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

  if (j.contains("optResult"))
    context.optResult = j["optResult"];

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
    context.simulationData =
        std::make_tuple(std::move(stateDim), std::move(stateData));
  }

  if (j.contains("registerNames"))
    j["registerNames"].get_to(context.registerNames);
}

// Enum data to denote the payload format.
enum class CodeFormat { MLIR, LLVM };

#define JSON_ENUM(enum_class, val)                                             \
  { enum_class::val, #val }

NLOHMANN_JSON_SERIALIZE_ENUM(CodeFormat, {JSON_ENUM(CodeFormat, MLIR),
                                          JSON_ENUM(CodeFormat, LLVM)});

// Enum data for the Optimizer. Serializer/deserializer below.
enum class Optimizer { COBYLA, NELDERMEAD, LBFGS, SPSA, ADAM, GRAD_DESC, SGD };

NLOHMANN_JSON_SERIALIZE_ENUM(
    Optimizer, {JSON_ENUM(Optimizer, COBYLA), JSON_ENUM(Optimizer, NELDERMEAD),
                JSON_ENUM(Optimizer, LBFGS), JSON_ENUM(Optimizer, SPSA),
                JSON_ENUM(Optimizer, ADAM), JSON_ENUM(Optimizer, GRAD_DESC),
                JSON_ENUM(Optimizer, SGD)});

inline Optimizer get_optimizer_type(const cudaq::optimizer &p) {
  if (dynamic_cast<const cudaq::optimizers::cobyla *>(&p))
    return Optimizer::COBYLA;
  if (dynamic_cast<const cudaq::optimizers::neldermead *>(&p))
    return Optimizer::NELDERMEAD;
  if (dynamic_cast<const cudaq::optimizers::lbfgs *>(&p))
    return Optimizer::LBFGS;
  if (dynamic_cast<const cudaq::optimizers::spsa *>(&p))
    return Optimizer::SPSA;
  if (dynamic_cast<const cudaq::optimizers::adam *>(&p))
    return Optimizer::ADAM;
  if (dynamic_cast<const cudaq::optimizers::gradient_descent *>(&p))
    return Optimizer::GRAD_DESC;
  if (dynamic_cast<const cudaq::optimizers::sgd *>(&p))
    return Optimizer::SGD;
  __builtin_unreachable();
}

inline void to_json(json &j, const cudaq::optimizers::BaseEnsmallen &p) {
// Macro to help reduce redundant field typing
#define TO_JSON_OPT_HELPER(field)                                              \
  do {                                                                         \
    if (p.field)                                                               \
      j[#field] = *p.field;                                                    \
  } while (0)
  TO_JSON_OPT_HELPER(max_eval);
  TO_JSON_OPT_HELPER(initial_parameters);
  TO_JSON_OPT_HELPER(lower_bounds);
  TO_JSON_OPT_HELPER(upper_bounds);
  TO_JSON_OPT_HELPER(f_tol);
  TO_JSON_OPT_HELPER(step_size);
#undef TO_JSON_OPT_HELPER
}

inline void to_json(json &j, const cudaq::optimizers::base_nlopt &p) {
// Macro to help reduce redundant field typing
#define TO_JSON_OPT_HELPER(field)                                              \
  do {                                                                         \
    if (p.field)                                                               \
      j[#field] = *p.field;                                                    \
  } while (0)
  TO_JSON_OPT_HELPER(max_eval);
  TO_JSON_OPT_HELPER(initial_parameters);
  TO_JSON_OPT_HELPER(lower_bounds);
  TO_JSON_OPT_HELPER(upper_bounds);
  TO_JSON_OPT_HELPER(f_tol);
#undef TO_JSON_OPT_HELPER
}

inline void to_json(json &j, const cudaq::optimizer &p) {
  if (auto *base_ensmallen =
          dynamic_cast<const cudaq::optimizers::BaseEnsmallen *>(&p))
    j = json(*base_ensmallen);
  else if (auto *base_nlopt =
               dynamic_cast<const cudaq::optimizers::base_nlopt *>(&p))
    j = json(*base_nlopt);
}

inline void from_json(const nlohmann::json &j,
                      cudaq::optimizers::BaseEnsmallen &p) {
// Macro to help reduce redundant field typing
#define FROM_JSON_OPT_HELPER(field)                                            \
  do {                                                                         \
    if (j.contains(#field))                                                    \
      p.field = j[#field];                                                     \
  } while (0)
  FROM_JSON_OPT_HELPER(max_eval);
  FROM_JSON_OPT_HELPER(initial_parameters);
  FROM_JSON_OPT_HELPER(lower_bounds);
  FROM_JSON_OPT_HELPER(upper_bounds);
  FROM_JSON_OPT_HELPER(f_tol);
  FROM_JSON_OPT_HELPER(step_size);
#undef FROM_JSON_OPT_HELPER
}

inline void from_json(const nlohmann::json &j,
                      cudaq::optimizers::base_nlopt &p) {
// Macro to help reduce redundant field typing
#define FROM_JSON_OPT_HELPER(field)                                            \
  do {                                                                         \
    if (j.contains(#field))                                                    \
      p.field = j[#field];                                                     \
  } while (0)
  FROM_JSON_OPT_HELPER(max_eval);
  FROM_JSON_OPT_HELPER(initial_parameters);
  FROM_JSON_OPT_HELPER(lower_bounds);
  FROM_JSON_OPT_HELPER(upper_bounds);
  FROM_JSON_OPT_HELPER(f_tol);
#undef FROM_JSON_OPT_HELPER
}

inline std::unique_ptr<cudaq::optimizer>
make_optimizer_from_json(const nlohmann::json &j, Optimizer optimizer_type) {
  if (optimizer_type == Optimizer::COBYLA) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::cobyla>();
    from_json(j, dynamic_cast<cudaq::optimizers::base_nlopt &>(*ret_ptr));
    return ret_ptr;
  }
  if (optimizer_type == Optimizer::NELDERMEAD) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::neldermead>();
    from_json(j, dynamic_cast<cudaq::optimizers::base_nlopt &>(*ret_ptr));
    return ret_ptr;
  }
  if (optimizer_type == Optimizer::LBFGS) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::lbfgs>();
    from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(*ret_ptr));
    return ret_ptr;
  }
  if (optimizer_type == Optimizer::SPSA) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::spsa>();
    from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(*ret_ptr));
    return ret_ptr;
  }
  if (optimizer_type == Optimizer::ADAM) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::adam>();
    from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(*ret_ptr));
    return ret_ptr;
  }
  if (optimizer_type == Optimizer::GRAD_DESC) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::gradient_descent>();
    from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(*ret_ptr));
    return ret_ptr;
  }
  if (optimizer_type == Optimizer::SGD) {
    auto ret_ptr = std::make_unique<cudaq::optimizers::sgd>();
    from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(*ret_ptr));
    return ret_ptr;
  }
  return nullptr;
}

// inline void to_json(const nlohmann::json &j, cudaq::optimizers::lbfgs p) {
//   to_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen>(p));
// }

// Payload from client to server for a kernel execution.
class RestRequest {
private:
  /// Holder of the reconstructed execution context.
  std::unique_ptr<ExecutionContext> m_deserializedContext;
  /// Holder of the reconstructed `spin_op`.
  std::unique_ptr<spin_op> m_deserializedSpinOp;
  // Version string identifying the client version.
  static inline const std::string CUDA_QUANTUM_VERSION = []() {
    std::stringstream ss;
    ss << "CUDA-Q Version " << cudaq::getVersion() << " ("
       << cudaq::getFullRepositoryVersion() << ")";
    return ss.str();
  }();

public:
  // Version number of this payload.
  // This needs to be bumped whenever a breaking change is introduced, which
  // causes incompatibility.
  //
  // For example,
  //
  // (1) Breaking Json schema changes,
  // e.g., adding/removing non-optional fields, changing field names, etc.,
  //     which introduce parsing incompatibility.
  // (2) Breaking changes in the runtime, which make JIT execution incompatible,
  //     e.g., changing the simulator names (.so files), changing signatures of
  //     QIR functions, etc.
  // IMPORTANT: When a new version is defined, a new NVQC deployment will be
  // needed.
  static constexpr std::size_t REST_PAYLOAD_VERSION = 1;
  RestRequest(ExecutionContext &context, int versionNumber)
      : executionContext(context), optimizer_n_params(0),
        version(versionNumber), clientVersion(CUDA_QUANTUM_VERSION) {}
  RestRequest(const json &j)
      : m_deserializedContext(
            std::make_unique<ExecutionContext>(j["executionContext"]["name"])),
        executionContext(*m_deserializedContext) {
    from_json(j, *this);
    // Take the ownership of the spin_op pointer for proper cleanup.
    if (executionContext.spin.has_value() && executionContext.spin.value())
      m_deserializedSpinOp.reset(executionContext.spin.value());
    // Customized processing for optional optimizer parameters
    if (j.contains("optimizer_type")) {
      j["optimizer_type"].get_to(this->optimizer_type);
      j["optimizer_n_params"].get_to(this->optimizer_n_params);
      j["optimizer"].get_to(this->optimizer);
    }
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
  // CUDA-Q optimizer enum
  Optimizer optimizer_type;
  // Serialized optimizer (JSON)
  std::string optimizer;
  // Number of parameters for VQE
  std::size_t optimizer_n_params;
  // List of MLIR passes to be applied on the code before execution.
  std::vector<std::string> passes;
  // Serialized kernel arguments.
  std::vector<uint8_t> args;
  // Version of this schema for compatibility check.
  std::size_t version;
  // Version of the runtime client submitting the request.
  std::string clientVersion;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE(RestRequest, version, entryPoint, simulator,
                                 executionContext, code, args, format, seed,
                                 optimizer_type, optimizer, optimizer_n_params,
                                 passes, clientVersion);
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

// NVCF execution metadata.
struct NvcfExecutionInfo {
  // Time point (milliseconds since epoch) when the request handling starts.
  std::size_t requestStart;
  // Time point (milliseconds since epoch) when the execution starts (JIT
  // completed).
  std::size_t simulationStart;
  // Time point (milliseconds since epoch) when the execution finishes.
  std::size_t simulationEnd;
  CudaDeviceProperties deviceProps;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(NvcfExecutionInfo, requestStart,
                                 simulationStart, simulationEnd, deviceProps);
};
} // namespace cudaq
