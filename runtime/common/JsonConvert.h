/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "GPUInfo.h"
#include "common/ExecutionContext.h"
#include "cudaq/Support/Version.h"
#include "cudaq/gradients.h"
#include "cudaq/optimizers.h"
#include "cudaq/simulators.h"
#include "nlohmann/json.hpp"
/*! \file
    \brief Utility to support JSON serialization between the client and server.
*/

using json = nlohmann::json;

namespace std {
// Complex data serialization.
template <class T>
inline void to_json(json &j, const std::complex<T> &p) {
  j = json{p.real(), p.imag()};
}

template <class T>
inline void from_json(const json &j, std::complex<T> &p) {
  p.real(j.at(0));
  p.imag(j.at(1));
}
} // namespace std

// Macros to help reduce redundant field typing for optional fields
#define TO_JSON_OPT_HELPER(field)                                              \
  do {                                                                         \
    if (p.field)                                                               \
      j[#field] = *p.field;                                                    \
  } while (0)

#define FROM_JSON_OPT_HELPER(field)                                            \
  do {                                                                         \
    if (j.contains(#field))                                                    \
      p.field = j[#field];                                                     \
  } while (0)

// Macros to help reduce redundant field typing for non-optional fields
#define TO_JSON_HELPER(field) j[#field] = p.field
#define FROM_JSON_HELPER(field) j[#field].get_to(p.field)

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

  if (context.simulationState) {
    j["simulationData"] = json();
    if (context.simulationState->isArrayLike()) {
      j["simulationData"]["dim"] = context.simulationState->getTensor().extents;
    } else {
      // Tensor-network like states: we serialize the flattened state vector.
      j["simulationData"]["dim"] = std::vector<std::size_t>{
          1ULL << context.simulationState->getNumQubits()};
    }
    const auto hostDataSize =
        context.simulationState->isArrayLike()
            ? context.simulationState->getNumElements()
            : 1ULL << context.simulationState->getNumQubits();
    if (context.simulationState->isDeviceData()) {
      if (context.simulationState->getPrecision() ==
          cudaq::SimulationState::precision::fp32) {
        std::vector<std::complex<float>> hostData(hostDataSize);
        context.simulationState->toHost(hostData.data(), hostData.size());
        std::vector<std::complex<double>> converted(hostData.begin(),
                                                    hostData.end());
        j["simulationData"]["data"] = converted;
      } else {
        std::vector<std::complex<double>> hostData(hostDataSize);
        context.simulationState->toHost(hostData.data(), hostData.size());
        j["simulationData"]["data"] = hostData;
      }
    } else {
      auto *ptr = reinterpret_cast<std::complex<double> *>(
          context.simulationState->getTensor().data);
      j["simulationData"]["data"] = std::vector<std::complex<double>>(
          ptr, ptr + context.simulationState->getNumElements());
    }
  }

  if (context.spin.has_value()) {
    const std::vector<double> spinOpRepr =
        context.spin.value().get_data_representation();
    j["spin"] = json();
    j["spin"]["data"] = spinOpRepr;
  }
  j["registerNames"] = context.registerNames;
  if (context.overlapResult.has_value())
    j["overlapResult"] = context.overlapResult.value();

  if (context.amplitudeMaps.has_value())
    j["amplitudeMaps"] = context.amplitudeMaps.value();

  if (!context.invocationResultBuffer.empty())
    j["invocationResultBuffer"] = context.invocationResultBuffer;
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

  if (j.contains("expectationValue"))
    context.expectationValue = j["expectationValue"];

  if (j.contains("optResult"))
    context.optResult = j["optResult"];

  if (j.contains("spin")) {
    std::vector<double> spinData;
    j["spin"]["data"].get_to(spinData);
    auto serializedSpinOps = spin_op(spinData);
    context.spin = std::move(serializedSpinOps);
    assert(cudaq::spin_op::canonicalize(context.spin.value()) ==
           context.spin.value());
  }

  if (j.contains("simulationData")) {
    std::vector<std::size_t> stateDim;
    std::vector<std::complex<double>> stateData;
    j["simulationData"]["dim"].get_to(stateDim);
    j["simulationData"]["data"].get_to(stateData);

    // Note: before `SimulationState` was added, `simulationData` contains a
    // flat pair of dimensions and data, whereby an empty dimension array
    // represents no state data in the context.
    if (!stateDim.empty()) {
      // Create the simulation specific SimulationState
      auto *simulator = cudaq::get_simulator();
      if (simulator->isSinglePrecision()) {
        // If the host (local) simulator is single-precision, convert the type
        // before loading the state vector.
        std::vector<std::complex<float>> converted(stateData.begin(),
                                                   stateData.end());
        context.simulationState = simulator->createStateFromData(
            std::make_pair(converted.data(), stateDim[0]));
      } else {
        context.simulationState = simulator->createStateFromData(
            std::make_pair(stateData.data(), stateDim[0]));
      }
    }
  }

  if (j.contains("registerNames"))
    j["registerNames"].get_to(context.registerNames);

  if (j.contains("overlapResult"))
    context.overlapResult = j["overlapResult"];

  if (j.contains("amplitudeMaps"))
    context.amplitudeMaps = j["amplitudeMaps"];

  if (j.contains("invocationResultBuffer"))
    context.invocationResultBuffer = j["invocationResultBuffer"];
}

// Enum data to denote the payload format.
enum class CodeFormat { MLIR, LLVM };

#define JSON_ENUM(enum_class, val)                                             \
  { enum_class::val, #val }

NLOHMANN_JSON_SERIALIZE_ENUM(CodeFormat, {JSON_ENUM(CodeFormat, MLIR),
                                          JSON_ENUM(CodeFormat, LLVM)});

// ----- cudaq::optimizer serialization/deserialization support below

// Enum data for the OptimizerEnum.
enum class OptimizerEnum {
  COBYLA,
  NELDERMEAD,
  LBFGS,
  SPSA,
  ADAM,
  GRAD_DESC,
  SGD
};

NLOHMANN_JSON_SERIALIZE_ENUM(
    OptimizerEnum,
    {JSON_ENUM(OptimizerEnum, COBYLA), JSON_ENUM(OptimizerEnum, NELDERMEAD),
     JSON_ENUM(OptimizerEnum, LBFGS), JSON_ENUM(OptimizerEnum, SPSA),
     JSON_ENUM(OptimizerEnum, ADAM), JSON_ENUM(OptimizerEnum, GRAD_DESC),
     JSON_ENUM(OptimizerEnum, SGD)});

inline OptimizerEnum get_optimizer_type(const cudaq::optimizer &p) {
  if (dynamic_cast<const cudaq::optimizers::cobyla *>(&p))
    return OptimizerEnum::COBYLA;
  if (dynamic_cast<const cudaq::optimizers::neldermead *>(&p))
    return OptimizerEnum::NELDERMEAD;
  if (dynamic_cast<const cudaq::optimizers::lbfgs *>(&p))
    return OptimizerEnum::LBFGS;
  if (dynamic_cast<const cudaq::optimizers::spsa *>(&p))
    return OptimizerEnum::SPSA;
  if (dynamic_cast<const cudaq::optimizers::adam *>(&p))
    return OptimizerEnum::ADAM;
  if (dynamic_cast<const cudaq::optimizers::gradient_descent *>(&p))
    return OptimizerEnum::GRAD_DESC;
  if (dynamic_cast<const cudaq::optimizers::sgd *>(&p))
    return OptimizerEnum::SGD;
  // This shouldn't happen, but gracefully handle it if it does.
  return OptimizerEnum::COBYLA;
}

inline void to_json(json &j, const cudaq::optimizers::BaseEnsmallen &p) {
  TO_JSON_OPT_HELPER(max_eval);
  TO_JSON_OPT_HELPER(initial_parameters);
  TO_JSON_OPT_HELPER(lower_bounds);
  TO_JSON_OPT_HELPER(upper_bounds);
  TO_JSON_OPT_HELPER(f_tol);
  TO_JSON_OPT_HELPER(step_size);
}

inline void to_json(json &j, const cudaq::optimizers::lbfgs &p) {
  TO_JSON_OPT_HELPER(max_line_search_trials);
  to_json(j, dynamic_cast<const cudaq::optimizers::BaseEnsmallen &>(p));
}

inline void to_json(json &j, const cudaq::optimizers::spsa &p) {
  TO_JSON_OPT_HELPER(alpha);
  TO_JSON_OPT_HELPER(gamma);
  TO_JSON_OPT_HELPER(eval_step_size);
  to_json(j, dynamic_cast<const cudaq::optimizers::BaseEnsmallen &>(p));
}

inline void to_json(json &j, const cudaq::optimizers::adam &p) {
  TO_JSON_OPT_HELPER(batch_size);
  TO_JSON_OPT_HELPER(beta1);
  TO_JSON_OPT_HELPER(beta2);
  TO_JSON_OPT_HELPER(eps);
  TO_JSON_OPT_HELPER(step_size);
  TO_JSON_OPT_HELPER(f_tol);
  to_json(j, dynamic_cast<const cudaq::optimizers::BaseEnsmallen &>(p));
}

inline void to_json(json &j, const cudaq::optimizers::gradient_descent &p) {
  to_json(j, dynamic_cast<const cudaq::optimizers::BaseEnsmallen &>(p));
}

inline void to_json(json &j, const cudaq::optimizers::sgd &p) {
  TO_JSON_OPT_HELPER(batch_size);
  TO_JSON_OPT_HELPER(step_size);
  TO_JSON_OPT_HELPER(f_tol);
  to_json(j, dynamic_cast<const cudaq::optimizers::BaseEnsmallen &>(p));
}

inline void to_json(json &j, const cudaq::optimizers::base_nlopt &p) {
  TO_JSON_OPT_HELPER(max_eval);
  TO_JSON_OPT_HELPER(initial_parameters);
  TO_JSON_OPT_HELPER(lower_bounds);
  TO_JSON_OPT_HELPER(upper_bounds);
  TO_JSON_OPT_HELPER(f_tol);
}

inline void to_json(json &j, const cudaq::optimizer &p) {
  if (auto *p2 = dynamic_cast<const cudaq::optimizers::lbfgs *>(&p))
    j = json(*p2);
  else if (auto *p2 = dynamic_cast<const cudaq::optimizers::spsa *>(&p))
    j = json(*p2);
  else if (auto *p2 = dynamic_cast<const cudaq::optimizers::adam *>(&p))
    j = json(*p2);
  else if (auto *p2 =
               dynamic_cast<const cudaq::optimizers::gradient_descent *>(&p))
    j = json(*p2);
  else if (auto *p2 = dynamic_cast<const cudaq::optimizers::sgd *>(&p))
    j = json(*p2);
  else if (auto *base_nlopt =
               dynamic_cast<const cudaq::optimizers::base_nlopt *>(&p))
    j = json(*base_nlopt);
}

inline void from_json(const nlohmann::json &j,
                      cudaq::optimizers::BaseEnsmallen &p) {
  FROM_JSON_OPT_HELPER(max_eval);
  FROM_JSON_OPT_HELPER(initial_parameters);
  FROM_JSON_OPT_HELPER(lower_bounds);
  FROM_JSON_OPT_HELPER(upper_bounds);
  FROM_JSON_OPT_HELPER(f_tol);
  FROM_JSON_OPT_HELPER(step_size);
}

inline void from_json(const nlohmann::json &j, cudaq::optimizers::lbfgs &p) {
  from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(p));
  FROM_JSON_OPT_HELPER(max_line_search_trials);
}

inline void from_json(const nlohmann::json &j, cudaq::optimizers::spsa &p) {
  from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(p));
  FROM_JSON_OPT_HELPER(alpha);
  FROM_JSON_OPT_HELPER(gamma);
  FROM_JSON_OPT_HELPER(eval_step_size);
}

inline void from_json(const nlohmann::json &j, cudaq::optimizers::adam &p) {
  from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(p));
  FROM_JSON_OPT_HELPER(batch_size);
  FROM_JSON_OPT_HELPER(beta1);
  FROM_JSON_OPT_HELPER(beta2);
  FROM_JSON_OPT_HELPER(eps);
  FROM_JSON_OPT_HELPER(step_size);
  FROM_JSON_OPT_HELPER(f_tol);
}

inline void from_json(const nlohmann::json &j,
                      cudaq::optimizers::gradient_descent &p) {
  from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(p));
}

inline void from_json(const nlohmann::json &j, cudaq::optimizers::sgd &p) {
  from_json(j, dynamic_cast<cudaq::optimizers::BaseEnsmallen &>(p));
  FROM_JSON_OPT_HELPER(batch_size);
  FROM_JSON_OPT_HELPER(step_size);
  FROM_JSON_OPT_HELPER(f_tol);
}

inline void from_json(const nlohmann::json &j,
                      cudaq::optimizers::base_nlopt &p) {
  FROM_JSON_OPT_HELPER(max_eval);
  FROM_JSON_OPT_HELPER(initial_parameters);
  FROM_JSON_OPT_HELPER(lower_bounds);
  FROM_JSON_OPT_HELPER(upper_bounds);
  FROM_JSON_OPT_HELPER(f_tol);
}

inline std::unique_ptr<cudaq::optimizer>
make_optimizer_from_json(const nlohmann::json &j,
                         const OptimizerEnum optimizer_type) {
  switch (optimizer_type) {
  case OptimizerEnum::COBYLA: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::cobyla>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case OptimizerEnum::NELDERMEAD: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::neldermead>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case OptimizerEnum::LBFGS: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::lbfgs>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case OptimizerEnum::SPSA: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::spsa>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case OptimizerEnum::ADAM: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::adam>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case OptimizerEnum::GRAD_DESC: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::gradient_descent>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case OptimizerEnum::SGD: {
    auto ret_ptr = std::make_unique<cudaq::optimizers::sgd>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  }
  // This shouldn't happen, but gracefully handle it if it does.
  return std::make_unique<cudaq::optimizers::cobyla>();
}

// ----- cudaq::gradient serialization/deserialization support below

enum class GradientEnum { CENTRAL_DIFF, FORWARD_DIFF, PARAMETER_SHIFT };
NLOHMANN_JSON_SERIALIZE_ENUM(GradientEnum,
                             {JSON_ENUM(GradientEnum, CENTRAL_DIFF),
                              JSON_ENUM(GradientEnum, FORWARD_DIFF),
                              JSON_ENUM(GradientEnum, PARAMETER_SHIFT)});

inline GradientEnum get_gradient_type(const cudaq::gradient &p) {
  if (dynamic_cast<const cudaq::gradients::central_difference *>(&p))
    return GradientEnum::CENTRAL_DIFF;
  if (dynamic_cast<const cudaq::gradients::forward_difference *>(&p))
    return GradientEnum::FORWARD_DIFF;
  if (dynamic_cast<const cudaq::gradients::parameter_shift *>(&p))
    return GradientEnum::PARAMETER_SHIFT;
  // This shouldn't happen, but handle it gracefully if it does.
  return GradientEnum::CENTRAL_DIFF;
}

// These do not attempt to serialize or deserialize the quantum kernel
// (ansatz_functor). That is intentional because those will be handled via the
// RestRequest.code.
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(cudaq::gradients::central_difference, step);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(cudaq::gradients::forward_difference, step);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(cudaq::gradients::parameter_shift,
                                   shiftScalar);

inline void to_json(json &j, const cudaq::gradient &p) {
  if (auto *central_difference =
          dynamic_cast<const cudaq::gradients::central_difference *>(&p))
    j = json(*central_difference);
  else if (auto *forward_difference =
               dynamic_cast<const cudaq::gradients::forward_difference *>(&p))
    j = json(*forward_difference);
  else if (auto *parameter_shift =
               dynamic_cast<const cudaq::gradients::parameter_shift *>(&p))
    j = json(*parameter_shift);
}

inline std::unique_ptr<cudaq::gradient>
make_gradient_from_json(const nlohmann::json &j,
                        const GradientEnum gradient_type) {
  switch (gradient_type) {
  case GradientEnum::CENTRAL_DIFF: {
    auto ret_ptr = std::make_unique<cudaq::gradients::central_difference>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case GradientEnum::FORWARD_DIFF: {
    auto ret_ptr = std::make_unique<cudaq::gradients::forward_difference>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  case GradientEnum::PARAMETER_SHIFT: {
    auto ret_ptr = std::make_unique<cudaq::gradients::parameter_shift>();
    from_json(j, *ret_ptr);
    return ret_ptr;
  }
  }
  // This shouldn't happen, but handle it gracefully if it does.
  return std::make_unique<cudaq::gradients::central_difference>();
}

// ----- Optional optimizer serialization/deserialization support below

struct RestRequestOptFields {
  std::optional<std::size_t> optimizer_n_params;
  std::optional<OptimizerEnum> optimizer_type;
  std::optional<GradientEnum> gradient_type;

  // Used on the server
  std::unique_ptr<cudaq::optimizer> optimizer;
  std::unique_ptr<cudaq::gradient> gradient;

  // Used on the client
  cudaq::optimizer *optimizer_ptr = nullptr;
  cudaq::gradient *gradient_ptr = nullptr;
};

inline void to_json(json &j, const RestRequestOptFields &p) {
  if (p.optimizer_ptr)
    j["optimizer"] = *p.optimizer_ptr;
  if (p.gradient_ptr)
    j["gradient"] = *p.gradient_ptr;
  TO_JSON_OPT_HELPER(optimizer_n_params);
  TO_JSON_OPT_HELPER(optimizer_type);
  TO_JSON_OPT_HELPER(gradient_type);
}

inline void from_json(const json &j, RestRequestOptFields &p) {
  FROM_JSON_OPT_HELPER(optimizer_n_params);
  FROM_JSON_OPT_HELPER(optimizer_type);
  if (p.optimizer_type)
    p.optimizer = make_optimizer_from_json(j["optimizer"], *p.optimizer_type);
  FROM_JSON_OPT_HELPER(gradient_type);
  if (p.gradient_type)
    p.gradient = make_gradient_from_json(j["gradient"], *p.gradient_type);
}

// Encapsulate the IR payload
struct IRPayLoad {
  // Underlying code (IR) payload as a Base64 string.
  std::string ir;

  // Name of the entry-point kernel.
  std::string entryPoint;

  // Serialized kernel arguments.
  std::vector<uint8_t> args;
  NLOHMANN_DEFINE_TYPE_INTRUSIVE(IRPayLoad, ir, entryPoint, args);
};

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
  static constexpr std::size_t REST_PAYLOAD_VERSION = 1;
  static constexpr std::size_t REST_PAYLOAD_MINOR_VERSION = 1;
  RestRequest(ExecutionContext &context, int versionNumber)
      : executionContext(context), version(versionNumber),
        clientVersion(CUDA_QUANTUM_VERSION) {}
  RestRequest(const json &j)
      : m_deserializedContext(
            std::make_unique<ExecutionContext>(j["executionContext"]["name"])),
        executionContext(*m_deserializedContext) {
    from_json(j, *this);
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
  // Optional optimizer fields
  std::optional<RestRequestOptFields> opt;
  // Optional kernel to compute the overlap with
  std::optional<IRPayLoad> overlapKernel;
  // List of MLIR passes to be applied on the code before execution.
  std::vector<std::string> passes;
  // Serialized kernel arguments.
  std::vector<uint8_t> args;
  // Version of this schema for compatibility check.
  std::size_t version;
  // Version of the runtime client submitting the request.
  std::string clientVersion;

  friend void to_json(json &j, const RestRequest &p) {
    TO_JSON_HELPER(version);
    TO_JSON_HELPER(entryPoint);
    TO_JSON_HELPER(simulator);
    TO_JSON_HELPER(executionContext);
    TO_JSON_HELPER(code);
    TO_JSON_HELPER(args);
    TO_JSON_HELPER(format);
    TO_JSON_OPT_HELPER(opt);
    TO_JSON_OPT_HELPER(overlapKernel);
    TO_JSON_HELPER(seed);
    TO_JSON_HELPER(passes);
    TO_JSON_HELPER(clientVersion);
  }

  friend void from_json(const json &j, RestRequest &p) {
    FROM_JSON_HELPER(version);
    FROM_JSON_HELPER(entryPoint);
    FROM_JSON_HELPER(simulator);
    FROM_JSON_HELPER(executionContext);
    FROM_JSON_HELPER(code);
    FROM_JSON_HELPER(args);
    FROM_JSON_HELPER(format);
    FROM_JSON_OPT_HELPER(opt);
    FROM_JSON_OPT_HELPER(overlapKernel);
    FROM_JSON_HELPER(seed);
    FROM_JSON_HELPER(passes);
    FROM_JSON_HELPER(clientVersion);
  }
};

} // namespace cudaq
