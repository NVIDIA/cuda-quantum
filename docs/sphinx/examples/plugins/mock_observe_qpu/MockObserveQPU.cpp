/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRemoteRESTQPU.h"
#include "common/ObservableUserData.h"
#include "common/ServerHelper.h"
#include "cudaq_internal/compiler/Compiler.h"
#include "nlohmann/json.hpp"
#include "cudaq/algorithms/observe/policy.h"
#include <cstdlib>
#include <string>

namespace cudaq {

/// Reference ServerHelper for server-side observe plugins.
/// Reads `user_data["observable"]` and returns a configured expectation value.
class MockObserveServerHelper : public ServerHelper {
  std::string url = "http://localhost:62454";
  double value = 0.41;

public:
  const std::string name() const override { return "mock_observe_qpu"; }

  void initialize(BackendConfig config) override {
    backendConfig = std::move(config);
    parseConfigForCommonParams(backendConfig);
    if (auto iter = backendConfig.find("url"); iter != backendConfig.end())
      url = iter->second;
    if (auto iter = backendConfig.find("shots");
        iter != backendConfig.end() && !iter->second.empty())
      setShots(std::stoul(iter->second));
    if (auto iter = backendConfig.find("value");
        iter != backendConfig.end() && !iter->second.empty())
      value = std::stod(iter->second);
  }

  RestHeaders getHeaders() override { return {}; }

  ServerJobPayload
  createJob(std::vector<KernelExecution> &circuitCodes) override {
    if (circuitCodes.size() != 1)
      throw std::runtime_error(
          "mock_observe_qpu expects a single preparation circuit");

    auto &circuit = circuitCodes.front();
    if (!circuit.user_data->contains("observable"))
      throw std::runtime_error(
          "mock_observe_qpu requires user_data[\"observable\"]");

    ServerMessage task;
    task["kernel"] = circuit.name;
    task["code"] = circuit.code;
    task["shots"] = shots;
    task["observable"] = circuit.user_data->at("observable");
    task["value"] = value;

    return std::make_tuple(url + "/jobs", RestHeaders(),
                           std::vector<ServerMessage>{std::move(task)});
  }

  std::string extractJobId(ServerMessage &postResponse) override {
    if (postResponse.is_array() && !postResponse.empty())
      return postResponse.front().value("id", "mock-observe-job");
    return postResponse.value("id", "mock-observe-job");
  }

  std::string constructGetJobPath(std::string &jobId) override {
    return url + "/jobs/" + jobId;
  }

  std::string constructGetJobPath(ServerMessage &postResponse) override {
    auto jobId = extractJobId(postResponse);
    return constructGetJobPath(jobId);
  }

  bool jobIsDone(ServerMessage &getJobResponse) override {
    const auto status = getJobResponse.value("status", "done");
    return status == "done" || status == "completed" || status == "succeeded";
  }

  sample_result processResults(ServerMessage &postJobResponse,
                               std::string &jobId) override {
    // Prefer an expectation returned by the mock server; otherwise use the
    // configured target argument.
    double expectation = value;
    if (postJobResponse.contains("value")) {
      if (postJobResponse["value"].is_number())
        expectation = postJobResponse["value"].get<double>();
      else if (postJobResponse["value"].is_string()) {
        try {
          expectation = std::stod(postJobResponse["value"].get<std::string>());
        } catch (...) {
          // Keep configured value when the echo server returns a bitstring.
        }
      }
    }
    return sample_result(ExecutionResult(expectation));
  }
};

/// Fermioniq-style custom QPU: no Pauli split, full spin_op on the wire.
class MockObserveQPU : public BaseRemoteRESTQPU {
public:
  ~MockObserveQPU() override = default;

  bool isRemote() override { return true; }
  bool isEmulated() override { return false; }

  void setNoiseModel(const cudaq::noise_model *model) override {
    if (model)
      throw std::runtime_error("Noise modeling is not allowed on this backend");
  }

  using BaseRemoteRESTQPU::getCompileTarget;
  std::unique_ptr<CompileTarget>
  getCompileTarget(const observe_policy &policy) override {
    auto target = BaseRemoteRESTQPU::getCompileTarget(policy);
    target->pauliTermSplitObservable = std::nullopt;
    return target;
  }

  using QPU::launchKernel;

  observe_result launchKernel(const observe_policy &policy,
                              const CompiledModule &module,
                              KernelArgs args) override {
    if (module.getMlirArtifacts().empty())
      throw std::runtime_error(
          "QPU does not support launching a CompiledModule without MLIR "
          "artifacts.");

    cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy));
    auto codes = compiler.emitKernelExecutions(module);
    if (codes.size() != 1)
      throw std::runtime_error(
          "mock_observe_qpu expects a single preparation circuit");

    attachObservableUserData(codes[0], policy.spin);
    async_observe_policy asyncPolicy{policy};
    return completeLaunchKernel(asyncPolicy, module.getName(), std::move(codes))
        .get();
  }

  async_observe_result launchKernel(const async_observe_policy &policy,
                                    const CompiledModule &module,
                                    KernelArgs args) override {
    if (module.getMlirArtifacts().empty())
      throw std::runtime_error(
          "QPU does not support launching a CompiledModule without MLIR "
          "artifacts.");

    cudaq_internal::compiler::Compiler compiler(getCompileTarget(policy.inner));
    auto codes = compiler.emitKernelExecutions(module);
    if (codes.size() != 1)
      throw std::runtime_error(
          "mock_observe_qpu expects a single preparation circuit");

    attachObservableUserData(codes[0], policy.inner.spin);
    return completeLaunchKernel(policy, module.getName(), std::move(codes));
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::QPU, cudaq::MockObserveQPU, mock_observe_qpu)
CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::MockObserveServerHelper,
                    mock_observe_qpu)
