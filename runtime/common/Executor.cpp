/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Executor.h"
#include "common/Logger.h"

namespace cudaq {
details::future Executor::execute(std::vector<KernelExecution> &codesToExecute,
                                  bool isObserve) {

  serverHelper->setShots(shots);

  cudaq::info("Executor creating {} jobs to execute with the {} helper.",
              codesToExecute.size(), serverHelper->name());

  // Create the Job Payload, composed of job post path, headers,
  // and the job json messages themselves
  auto [jobPostPath, headers, jobs] = serverHelper->createJob(codesToExecute);

  auto config = serverHelper->getConfig();

  std::vector<details::future::Job> ids;

  auto *extraPayloadProvider = [&]() -> cudaq::ExtraPayloadProvider * {
    const auto extraPayloadProvider =
        runtimeTarget.runtimeConfig.find("extra_payload_provider");
    if (extraPayloadProvider == runtimeTarget.runtimeConfig.end())
      return nullptr;
    const auto &extraPayloadProviderName = extraPayloadProvider->second;

    auto &extraProviders = cudaq::getExtraPayloadProviders();
    const auto it = std::find_if(
        extraProviders.begin(), extraProviders.end(), [&](const auto &entry) {
          return entry->name() == extraPayloadProviderName;
        });
    if (it == extraProviders.end())
      throw std::runtime_error("ExtraPayloadProvider with name " +
                               extraPayloadProviderName + " not found.");
    cudaq::info("Setting ExtraPayloadProvider to {}", extraPayloadProviderName);
    return it->get();
  }();

  for (std::size_t i = 0; auto &job : jobs) {
    cudaq::info("Job (name={}) created, posting to {}", codesToExecute[i].name,
                jobPostPath);

    // Add extra payload if any
    if (extraPayloadProvider) {
      extraPayloadProvider->injectExtraPayload(runtimeTarget, job);
    }

    // Post it, get the response
    auto response = client.post(jobPostPath, "", job, headers, true, false,
                                serverHelper->getCookies());
    cudaq::info("Job (name={}) posted, response was {}", codesToExecute[i].name,
                response.dump());

    // Add the job id and the job name.
    auto task_id = serverHelper->extractJobId(response);
    if (task_id.empty()) {
      nlohmann::json tmp(job.at("tasks"));
      serverHelper->constructGetJobPath(tmp[0]);
      task_id = tmp[0].at("task_id");
    }
    cudaq::info("Task ID is {}", task_id);
    ids.emplace_back(task_id, codesToExecute[i].name);
    config["output_names." + task_id] = codesToExecute[i].output_names.dump();

    nlohmann::json jReorder = codesToExecute[i].mapping_reorder_idx;
    config["reorderIdx." + task_id] = jReorder.dump();

    i++;
  }

  config.insert({"shots", std::to_string(shots)});
  std::string name = serverHelper->name();
  return details::future(ids, name, config, isObserve);
}
} // namespace cudaq

LLVM_INSTANTIATE_REGISTRY(cudaq::Executor::RegistryType)
