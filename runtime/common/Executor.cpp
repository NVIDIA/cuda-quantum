/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "Executor.h"
#include "common/Logger.h"

namespace cudaq {
details::future
Executor::execute(std::vector<KernelExecution> &codesToExecute) {

  serverHelper->setShots(shots);

  cudaq::info("Executor creating {} jobs to execute with the {} helper.",
              codesToExecute.size(), serverHelper->name());

  // Create the Job Payload, composed of job post path, headers,
  // and the job json messages themselves
  auto [jobPostPath, headers, jobs] = serverHelper->createJob(codesToExecute);

  auto config = serverHelper->getConfig();

  std::vector<details::future::Job> ids;
  for (std::size_t i = 0; auto &job : jobs) {
    cudaq::info("Job (name={}) created, posting to {}", codesToExecute[i].name,
                jobPostPath);

    // Post it, get the response
    auto response = client.post(jobPostPath, "", job, headers);
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
  return details::future(ids, name, config);
}

cudaq::orca::details::Orcafuture
Executor::execute(cudaq::orca::TBIParameters params,
                  const std::string &kernelName) {

  serverHelper->setShots(shots);

  cudaq::info("Executor creating job to execute with the {} helper.",
              serverHelper->name());

  // Create the Job Payload, composed of job post path, headers,
  // and the job json messages themselves
  auto [jobPostPath, headers, jobs] = serverHelper->createJob(params);
  auto job = jobs[0];
  auto config = serverHelper->getConfig();

  std::vector<cudaq::orca::details::Orcafuture::Job> ids;
  // for (std::size_t i = 0; auto &job : jobs) {
  cudaq::info("Job created, posting to {}", jobPostPath);

  // Post it, get the response
  auto response = client.post(jobPostPath, "", job, headers);
  cudaq::info("Job posted, response was {}", response.dump());

  // // Add the job id and the job name.
  auto job_id = serverHelper->extractJobId(response);
  if (job_id.empty()) {
    nlohmann::json tmp(job.at("job_id"));
    serverHelper->constructGetJobPath(tmp[0]);
    job_id = tmp[0].at("job_id");
  }
  ids.emplace_back(job_id, kernelName);
  config["output_names." + job_id] = kernelName;

  // nlohmann::json jReorder = codesToExecute[i].mapping_reorder_idx;
  // config["reorderIdx." + task_id] = jReorder.dump();

  // i++;
  // }

  config.insert({"shots", std::to_string(shots)});
  std::string name = serverHelper->name();
  return cudaq::orca::details::Orcafuture(ids, name, config);
}
} // namespace cudaq