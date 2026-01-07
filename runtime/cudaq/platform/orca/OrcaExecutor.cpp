/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "OrcaExecutor.h"
#include "OrcaServerHelper.h"
#include "common/Logger.h"

namespace cudaq {

details::future OrcaExecutor::execute(cudaq::orca::TBIParameters params,
                                      const std::string &kernelName) {
  auto orcaServerHelper = dynamic_cast<OrcaServerHelper *>(serverHelper);
  assert(orcaServerHelper);
  orcaServerHelper->setShots(shots);
  CUDAQ_INFO("Executor creating job to execute with the {} helper.",
             orcaServerHelper->name());
  // Create the Job Payload, composed of job post path, headers,
  // and the job json messages themselves
  auto [jobPostPath, headers, jobs] = orcaServerHelper->createJob(params);
  auto job = jobs[0];
  auto config = orcaServerHelper->getConfig();
  std::vector<cudaq::details::future::Job> ids;
  CUDAQ_INFO("Job created, posting to {}", jobPostPath);
  // Post it, get the response
  auto response = client.post(jobPostPath, "", job, headers);
  CUDAQ_INFO("Job posted, response was {}", response.dump());
  // Add the job id and the job name.
  auto job_id = orcaServerHelper->extractJobId(response);
  if (job_id.empty()) {
    nlohmann::json tmp(job.at("job_id"));
    orcaServerHelper->constructGetJobPath(tmp[0]);
    job_id = tmp[0].at("job_id");
  }
  ids.emplace_back(job_id, kernelName);
  config["output_names." + job_id] = kernelName;

  config.insert({"shots", std::to_string(shots)});
  std::string name = orcaServerHelper->name();
  return cudaq::details::future(ids, name, config);
}

} // namespace cudaq
