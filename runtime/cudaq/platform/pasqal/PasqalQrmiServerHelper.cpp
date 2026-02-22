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

namespace cudaq {

PasqalQrmiServerHelper::ModeConfig PasqalQrmiServerHelper::resolveConfig() {
  ModeConfig modeConfig;
  // QRMI mode currently always runs under Slurm, and SLURM_JOB_QPU_RESOURCES is
  // present. Backend name is taken only from that environment variable.
  if (auto *slurmResources = std::getenv("SLURM_JOB_QPU_RESOURCES")) {
    std::string resources = slurmResources;
    auto firstComma = resources.find(',');
    modeConfig.backendName = resources.substr(0, firstComma);
    if (modeConfig.backendName.empty())
      throw std::runtime_error("Pasqal QRMI mode requires a backend name, but "
                               "SLURM_JOB_QPU_RESOURCES is empty.");
  } else {
    throw std::runtime_error(
        "Pasqal QRMI mode requires SLURM_JOB_QPU_RESOURCES.");
  }

  return modeConfig;
}

std::vector<ServerMessage>
PasqalQrmiServerHelper::createJobs(std::vector<KernelExecution> &circuitCodes,
                                   std::size_t shots,
                                   const std::string &backendName) {
  std::vector<ServerMessage> tasks;
  for (auto &circuitCode : circuitCodes) {
    ServerMessage message;
    message["machine"] = backendName;
    message["job_runs"] = static_cast<std::int32_t>(shots);
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
