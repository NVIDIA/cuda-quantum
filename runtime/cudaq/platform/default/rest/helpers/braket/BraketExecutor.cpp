/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BraketExecutor.h"
#include "BraketServerHelper.h"

namespace {
std::string getDeviceRegion(std::string deviceArn) {
  auto start = deviceArn.begin();
  for (int i = 0; i < 3; ++i) {
    start = std::find(start, deviceArn.end(), ':');
    if (start != deviceArn.end()) {
      ++start;
    } else {
      throw std::runtime_error("Improper device arn: " + deviceArn);
    }
  }
  auto end = std::find(start, deviceArn.end(), ':');
  return std::string{start, end};
}
} // namespace

namespace cudaq {
BraketExecutor::BraketExecutor()
    : api(options), jobToken(std::getenv("AMZN_BRAKET_JOB_TOKEN")) {}

/// @brief Set the server helper
void BraketExecutor::setServerHelper(ServerHelper *helper) {
  Executor::setServerHelper(helper);
  std::string region = getDeviceRegion(helper->getConfig().at("deviceArn"));

  Aws::Client::ClientConfiguration clientConfig;
  clientConfig.verifySSL = false;
  Aws::S3Crt::ClientConfiguration s3ClientConfig;
  s3ClientConfig.verifySSL = false;
  if (!region.empty()) {
    if (region != clientConfig.region) {
      cudaq::info("Auto-routing to AWS region {}", region);
      clientConfig.region = region;
      s3ClientConfig.region = region;
    }
  } else {
    region = clientConfig.region;
  }

  braketClientPtr = std::make_unique<Aws::Braket::BraketClient>(clientConfig);
  stsClientPtr = std::make_unique<Aws::STS::STSClient>(clientConfig);
  s3ClientPtr = std::make_unique<Aws::S3Crt::S3CrtClient>(s3ClientConfig);

  defaultBucketFuture = std::async(std::launch::async, [this, region] {
    auto response = stsClientPtr->GetCallerIdentity();
    std::string bucketName;
    if (response.IsSuccess()) {
      bucketName = fmt::format("amazon-braket-{}-{}", region,
                               response.GetResult().GetAccount());
      cudaq::info("Braket task results will use S3 bucket \"{}\"", bucketName);
      return bucketName;
    } else {
      throw std::runtime_error(response.GetError().GetMessage());
    }
  });
}

details::future
BraketExecutor::execute(std::vector<KernelExecution> &codesToExecute) {
  auto braketServerHelper = dynamic_cast<BraketServerHelper *>(serverHelper);
  assert(braketServerHelper);
  braketServerHelper->setShots(shots);

  auto [dummy1, dummy2, messages] =
      braketServerHelper->createJob(codesToExecute);

  std::string const defaultBucket = defaultBucketFuture.get();
  std::string const defaultPrefix = "tasks";

  auto config = braketServerHelper->getConfig();
  cudaq::info("Backend config: {}, shots {}", config, shots);
  config.insert({"shots", std::to_string(shots)});

  std::vector<Aws::Braket::Model::CreateQuantumTaskOutcomeCallable>
      createOutcomes;

  for (const auto &message : messages) {
    Aws::Braket::Model::CreateQuantumTaskRequest req;
    req.SetAction(message["action"]);
    req.SetDeviceArn(message["deviceArn"]);
    req.SetShots(message["shots"]);
    if (jobToken)
      req.SetJobToken(jobToken);
    req.SetOutputS3Bucket(defaultBucket);
    req.SetOutputS3KeyPrefix(defaultPrefix);

    createOutcomes.push_back(braketClientPtr->CreateQuantumTaskCallable(req));
  }

  return std::async(
      std::launch::async,
      [this](std::vector<Aws::Braket::Model::CreateQuantumTaskOutcomeCallable>
                 createOutcomes) {
        std::vector<ExecutionResult> results;
        for (auto &outcome : createOutcomes) {
          auto createResponse = outcome.get();
          if (!createResponse.IsSuccess()) {
            throw std::runtime_error(createResponse.GetError().GetMessage());
          }
          std::string taskArn = createResponse.GetResult().GetQuantumTaskArn();
          cudaq::info("Created Braket quantum task {}", taskArn);

          Aws::Braket::Model::GetQuantumTaskRequest req;
          req.SetQuantumTaskArn(taskArn);
          auto getResponse = braketClientPtr->GetQuantumTask(req);
          if (!getResponse.IsSuccess()) {
            throw std::runtime_error(getResponse.GetError().GetMessage());
          }
          auto taskStatus = getResponse.GetResult().GetStatus();
          while (
              taskStatus != Aws::Braket::Model::QuantumTaskStatus::COMPLETED &&
              taskStatus != Aws::Braket::Model::QuantumTaskStatus::FAILED &&
              taskStatus != Aws::Braket::Model::QuantumTaskStatus::CANCELLED) {
            std::this_thread::sleep_for(pollingInterval);

            getResponse = braketClientPtr->GetQuantumTask(req);
            if (!getResponse.IsSuccess()) {
              throw std::runtime_error(getResponse.GetError().GetMessage());
            }
            taskStatus = getResponse.GetResult().GetStatus();
          }

          auto getResult = getResponse.GetResult();
          if (taskStatus != Aws::Braket::Model::QuantumTaskStatus::COMPLETED) {
            // Task terminated without results
            throw std::runtime_error(
                fmt::format("Braket task {} terminated without results. {}",
                            taskArn, getResult.GetFailureReason()));
          }

          std::string outBucket = getResult.GetOutputS3Bucket();
          std::string outPrefix = getResult.GetOutputS3Directory();

          cudaq::info("Fetching braket quantum task {} results from "
                      "s3://{}/{}/results.json",
                      taskArn, outBucket, outPrefix);

          Aws::S3Crt::Model::GetObjectRequest resultsJsonRequest;
          resultsJsonRequest.SetBucket(outBucket);
          resultsJsonRequest.SetKey(fmt::format("{}/results.json", outPrefix));
          auto s3Response = s3ClientPtr->GetObject(resultsJsonRequest);
          if (!s3Response.IsSuccess()) {
            throw std::runtime_error(s3Response.GetError().GetMessage());
          }
          auto resultsJson = nlohmann::json::parse(
              s3Response.GetResultWithOwnership().GetBody());
          auto c = serverHelper->processResults(resultsJson, taskArn);

          for (auto &regName : c.register_names()) {
            results.emplace_back(c.to_map(regName), regName);
            results.back().sequentialData = c.sequential_data(regName);
          }
        }

        return sample_result(results);
      },
      std::move(createOutcomes));
};
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::BraketExecutor, braket);