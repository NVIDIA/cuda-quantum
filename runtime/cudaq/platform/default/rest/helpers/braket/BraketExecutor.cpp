/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BraketExecutor.h"
#include "BraketServerHelper.h"

namespace cudaq {

details::future
BraketExecutor::execute(std::vector<KernelExecution> &codesToExecute) {
  auto braketServerHelper = dynamic_cast<BraketServerHelper *>(serverHelper);
  assert(braketServerHelper);
  braketServerHelper->setShots(shots);

  auto [dummy1, dummy2, messages] =
      braketServerHelper->createJob(codesToExecute);

  cudaq::debug("messages[0] = {}", messages[0].dump());

  auto const &message = messages[0];

  CountsDictionary counts;

  std::string const defaultBucket = defaultBucketFuture.get();
  std::string const defaultPrefix = "tasks-cudaq";

  Aws::Braket::Model::CreateQuantumTaskRequest req;

  auto config = braketServerHelper->getConfig();
  cudaq::info("Backend config: {}, shots {}", config, shots);
  config.insert({"shots", std::to_string(shots)});

  req.SetAction(message["action"]);
  req.SetDeviceArn(message["deviceArn"]);
  req.SetShots(message["shots"]);
  req.SetOutputS3Bucket(defaultBucket);
  req.SetOutputS3KeyPrefix(defaultPrefix);

  auto response = braketClient.CreateQuantumTask(req);

  if (response.IsSuccess()) {
    std::string taskArn = response.GetResult().GetQuantumTaskArn();
    cudaq::info("Created {}", taskArn);
    Aws::Braket::Model::QuantumTaskStatus taskStatus;
    Aws::Braket::Model::GetQuantumTaskRequest getTaskReq;
    getTaskReq.SetQuantumTaskArn(taskArn);

    auto r = braketClient.GetQuantumTask(getTaskReq);
    do {
      std::this_thread::sleep_for(std::chrono::milliseconds{100});
      r = braketClient.GetQuantumTask(getTaskReq);
      taskStatus = r.GetResult().GetStatus();
    } while (taskStatus != Aws::Braket::Model::QuantumTaskStatus::COMPLETED &&
             taskStatus != Aws::Braket::Model::QuantumTaskStatus::FAILED &&
             taskStatus != Aws::Braket::Model::QuantumTaskStatus::CANCELLED);

    std::string outBucket = r.GetResult().GetOutputS3Bucket();
    std::string outPrefix = r.GetResult().GetOutputS3Directory();

    cudaq::info("results at {}/{}", outBucket, outPrefix);
    Aws::S3Crt::Model::GetObjectRequest resultRequest;
    resultRequest.SetBucket(outBucket);
    resultRequest.SetKey(fmt::format("{}/results.json", outPrefix));

    auto results = s3Client.GetObject(resultRequest);

    auto parsedResult =
        nlohmann::json::parse(results.GetResultWithOwnership().GetBody());

    auto measurements = parsedResult.at("measurements");

    for (auto const &m : measurements) {
      std::string bitString = "";
      for (int bit : m) {
        bitString += std::to_string(bit);
      }
      counts[bitString] += 1;
    }

  } else {
    std::cout << "Create error\n" << response.GetError() << "\n";
  }

  ExecutionResult ex_r{counts};
  sample_result result{ex_r};

  std::promise<sample_result> p;
  p.set_value(result);
  return {p.get_future()};
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::BraketExecutor, braket);