/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PasqalQrmiServerHelper.h"
#ifdef CUDAQ_PASQAL_QRMI_ENABLED
#include "PasqalUtils.h"
#include "QrmiUtils.h"

#include <cstdint>
#include <cstdlib>
#include <utility>

namespace cudaq {

PasqalQrmiServerHelper::ModeConfig PasqalQrmiServerHelper::resolveConfig() {
  ModeConfig modeConfig;
  auto resources = qrmi::splitList(
      qrmi::jobEnv("QRMI_JOB_QPU_RESOURCES", "SLURM_JOB_QPU_RESOURCES"));
  if (resources.empty()) {
    throw std::runtime_error(
        "Pasqal QRMI mode requires QRMI_JOB_QPU_RESOURCES or legacy "
        "SLURM_JOB_QPU_RESOURCES.");
  }
  // QRMI can expose multiple requested resources. CUDA-Q's Pasqal integration
  // currently targets a single backend, so use the first resource for now.
  modeConfig.backendName = resources.front();
  if (modeConfig.backendName.empty())
    throw std::runtime_error("Pasqal QRMI mode requires a backend name, but "
                             "QRMI_JOB_QPU_RESOURCES is empty.");

  return modeConfig;
}

std::vector<ServerMessage>
PasqalQrmiServerHelper::createJobs(std::vector<KernelExecution> &circuitCodes,
                                   std::size_t shots,
                                   const std::string &backendName) {
  if (!std::in_range<std::int32_t>(shots))
    throw std::runtime_error("Pasqal QRMI mode requires shots <= INT32_MAX.");

  const auto jobRuns = static_cast<std::int32_t>(shots);
  std::vector<ServerMessage> tasks;
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["machine"] = backendName;
    message["job_runs"] = jobRuns;
    message["sequence"] = nlohmann::json::parse(circuitCode.code);
    tasks.push_back(std::move(message));
  }
  return tasks;
}

ExecutionResult PasqalQrmiServerHelper::parseCountsFromTaskResult(
    const std::string &taskResultJson) {
  return pasqal::parseExecutionResultFromTaskResult(taskResultJson);
}

} // namespace cudaq
#endif
