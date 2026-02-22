/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

/*
 * Unified Pasqal executor:
 * - direct mode delegates to normal CUDA-Q REST Executor flow
 * - machine=qrmi mode uses QRMI task lifecycle directly
 *
 * Why:
 *
 * The standard remote flow for backends like `pasqal` is implemented in the
 * shared `runtime/common` Executor + Future stack. Tt uses RestClient to
 * POST/GET job payloads and polls until terminal status before calling
 * ServerHelper::processResults().
 *
 * QRMI is a C API (`qrmi_resource_task_start/status/result/...`) wrapping its
 * own REST calls, but it is not itself a REST API. It has its own resource/task
 * lifecycle. So it cannot drive QRMI tasks directly.
 *
 * Because of that, `pasqal_qrmi` performs task lifecycle orchestration here:
 * build payload -> start task -> poll status -> fetch result -> map to CUDA-Q
 * sample_result.
 *
 * We could have implemented a a QRMIClient at the same level as RestClient,
 * which could reduce QRMI backend implementations to just building payloads
 * and parsing results, but this would be out of scope for this work.
 */

#include "PasqalServerHelper.h"
#include "common/Executor.h"
#ifdef CUDAQ_PASQAL_QRMI_ENABLED
#include "PasqalQrmiServerHelper.h"
#include "QrmiUtils.h"
#endif

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <future>

namespace cudaq {

class PasqalExecutor : public Executor {
public:
  ~PasqalExecutor() override = default;

  details::future execute(std::vector<KernelExecution> &codesToExecute,
                          cudaq::details::ExecutionContextType execType =
                              cudaq::details::ExecutionContextType::sample,
                          std::vector<char> *rawOutput = nullptr) override {
    auto *helper = dynamic_cast<PasqalServerHelper *>(serverHelper);
    if (!helper) {
      throw std::runtime_error("PasqalExecutor expected PasqalServerHelper but "
                               "got incompatible helper.");
    }

    auto helperConfig = helper->getConfig();
    std::string machine =
        helperConfig.contains("machine") ? helperConfig.at("machine") : "";
    std::transform(
        machine.begin(), machine.end(), machine.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    const bool useQrmiMode = machine == "qrmi";

    // If not using QRMI mode, delegate to normal REST Executor flow
    // delegating to Executor::execute() will use the PasqalServerHelper
    // REST-based implementation of processResults() as normally.
    if (!useQrmiMode) {
      return Executor::execute(codesToExecute, execType, rawOutput);
    }
    // if we reach this point, we are in QRMI mode.
#ifndef CUDAQ_PASQAL_QRMI_ENABLED
    throw std::runtime_error( // so throw if QRMI mode requested but not
                              // supported by this build
        "Pasqal QRMI mode requested (machine=qrmi), but CUDA-Q was built "
        "without QRMI support.");
#else  // QRMI mode implementation below
    if (execType != cudaq::details::ExecutionContextType::sample) {
      throw std::runtime_error(
          "Pasqal QRMI mode supports sampling execution only.");
    }
    if (rawOutput) {
      throw std::runtime_error(
          "Pasqal QRMI mode does not support run()-style raw output.");
    }

    helper->setShots(shots);
    auto modeConfig = PasqalQrmiServerHelper::resolveConfig();
    auto jobs = PasqalQrmiServerHelper::createJobs(codesToExecute, shots,
                                                   modeConfig.backendName);
    auto worker = std::async(
        std::launch::async,
        [backendName = modeConfig.backendName,
         jobs = std::move(jobs)]() mutable -> sample_result {
          qrmi::ResourceSession session(backendName);
          std::vector<ExecutionResult> results;
          results.reserve(jobs.size());
          // Keep loop structure for future batching support.
          // Today we only submit a single circuit per job.

          for (auto &job : jobs) {
            std::string sequence = job.at("sequence").dump();
            auto jobRuns = job.at("job_runs").get<std::int32_t>();

            QrmiPayload payload{};
            payload.tag = QRMI_PAYLOAD_PASQAL_CLOUD;
            payload.PASQAL_CLOUD.sequence =
                const_cast<char *>(sequence.c_str());
            payload.PASQAL_CLOUD.job_runs = jobRuns;

            auto taskId = session.startTask(payload);
            auto taskStatus = session.waitForTaskTerminalStatus(taskId);
            session.ensureTaskCompleted(taskId, taskStatus);

            results.push_back(PasqalQrmiServerHelper::parseCountsFromTaskResult(
                session.taskResult(taskId)));
          }

          return sample_result(results);
        });

    return details::future(std::move(worker));
#endif // end if QRMI mode
  }
};

} // namespace cudaq

// unified registration for PasqalExecutor, which will handle both REST and QRMI
// modes based on config
CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::PasqalExecutor, pasqal);
