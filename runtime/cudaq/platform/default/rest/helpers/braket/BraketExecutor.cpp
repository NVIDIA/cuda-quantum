/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BraketExecutor.h"
#include "common/BraketServerHelper.h"
#include "common/FmtCore.h"

#include <aws/braket/model/Association.h>
#include <aws/braket/model/AssociationType.h>
#include <aws/braket/model/CreateQuantumTaskRequest.h>
#include <aws/braket/model/GetQuantumTaskRequest.h>
#include <aws/braket/model/QuantumTaskStatus.h>

#include <aws/s3-crt/model/CreateBucketRequest.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#include <aws/s3-crt/model/PutBucketPolicyRequest.h>
#include <aws/s3-crt/model/PutPublicAccessBlockRequest.h>

#include <aws/core/utils/ARN.h>

namespace {
void tryCreateBucket(Aws::S3Crt::S3CrtClient &client, std::string const &region,
                     std::string const &bucketName) {
  Aws::S3Crt::Model::CreateBucketRequest createReq;
  createReq.SetBucket(bucketName);
  Aws::S3Crt::Model::CreateBucketConfiguration config;
  if (region != Aws::Region::US_EAST_1) {
    config.SetLocationConstraint(
        Aws::S3Crt::Model::BucketLocationConstraintMapper::
            GetBucketLocationConstraintForName(region));
  }
  createReq.SetCreateBucketConfiguration(config);
  CUDAQ_INFO("Attempting to create S3 bucket \"s3://{}\"", bucketName);
  auto createResponse = client.CreateBucket(createReq);
  if (!createResponse.IsSuccess()) {
    auto error = createResponse.GetError();
    if (error.GetErrorType() ==
        Aws::S3Crt::S3CrtErrors::BUCKET_ALREADY_OWNED_BY_YOU) {
      CUDAQ_INFO("\"s3://{}\" already exists", bucketName);
      return;
    } else if (error.GetErrorType() ==
               Aws::S3Crt::S3CrtErrors::BUCKET_ALREADY_EXISTS) {
      throw std::runtime_error("default bucket name \"" + bucketName +
                               "\" already exists in another account. Please "
                               "supply an alternative bucket name.");
    } else {
      throw std::runtime_error(error.GetMessage());
    }
  }

  Aws::S3Crt::Model::PutPublicAccessBlockRequest publicReq;
  publicReq.SetBucket(bucketName);
  Aws::S3Crt::Model::PublicAccessBlockConfiguration publicConfig;
  publicConfig.SetBlockPublicAcls(true);
  publicConfig.SetIgnorePublicAcls(true);
  publicConfig.SetBlockPublicPolicy(true);
  publicConfig.SetRestrictPublicBuckets(true);
  publicReq.SetPublicAccessBlockConfiguration(publicConfig);

  auto publicResponse = client.PutPublicAccessBlock(publicReq);
  if (!publicResponse.IsSuccess()) {
    auto error = publicResponse.GetError();
    throw std::runtime_error(error.GetMessage());
  }

  std::string policy = fmt::format(R"({{
    "Version": "2012-10-17",
    "Statement": [
        {{
            "Effect": "Allow",
            "Principal": {{
                "Service": [
                    "braket.amazonaws.com"
                ]
            }},
            "Action": "s3:*",
            "Resource": [
                "arn:aws:s3:::{0}",
                "arn:aws:s3:::{0}/*"
            ]
        }}
    ]
}})",
                                   bucketName);

  Aws::S3Crt::Model::PutBucketPolicyRequest policyReq;
  policyReq.SetBucket(bucketName);
  policyReq.SetBody(std::make_shared<Aws::StringStream>(policy));

  auto policyResponse = client.PutBucketPolicy(policyReq);
  if (!policyResponse.IsSuccess()) {
    auto error = policyResponse.GetError();
    throw std::runtime_error(error.GetMessage());
  }
}

} // namespace

namespace cudaq {
BraketExecutor::BraketExecutor()
    : api(options), jobToken(std::getenv("AMZN_BRAKET_JOB_TOKEN")),
      reservationArn(std::getenv("AMZN_BRAKET_RESERVATION_TIME_WINDOW_ARN")) {}

/// @brief Set the server helper
void BraketExecutor::setServerHelper(ServerHelper *helper) {
  Executor::setServerHelper(helper);

  std::string region =
      Aws::Utils::ARN(helper->getConfig().at("deviceArn")).GetRegion();
  std::string defaultBucket = helper->getConfig().at("defaultBucket");

  if (helper->getConfig().contains("polling_interval_ms")) {
    long pollingIntervalMs{
        std::stol(helper->getConfig().at("polling_interval_ms"))};
    if (pollingIntervalMs <= 0) {
      throw std::runtime_error(
          "polling_interval_ms must be a positive integer.");
    }
    pollingInterval = std::chrono::milliseconds{pollingIntervalMs};
  }

  Aws::Client::ClientConfiguration clientConfig;
  clientConfig.verifySSL = false;
  Aws::S3Crt::ClientConfiguration s3ClientConfig;
  s3ClientConfig.verifySSL = false;
  if (!region.empty()) {
    if (region != clientConfig.region) {
      CUDAQ_INFO("Auto-routing to AWS region {}", region);
      clientConfig.region = region;
      s3ClientConfig.region = region;
    }
  } else {
    region = clientConfig.region;
  }

  braketClientPtr = std::make_unique<Aws::Braket::BraketClient>(clientConfig);
  stsClientPtr = std::make_unique<Aws::STS::STSClient>(clientConfig);
  s3ClientPtr = std::make_unique<Aws::S3Crt::S3CrtClient>(s3ClientConfig);

  defaultBucketFuture =
      std::async(std::launch::async, [this, region, defaultBucket] {
        std::string bucketName = defaultBucket;
        if (bucketName.empty()) {
          auto response = stsClientPtr->GetCallerIdentity();
          if (response.IsSuccess()) {
            bucketName = fmt::format("amazon-braket-{}-{}", region,
                                     response.GetResult().GetAccount());
          } else {
            throw std::runtime_error(response.GetError().GetMessage());
          }
        }
        tryCreateBucket(*s3ClientPtr, region, bucketName);
        CUDAQ_INFO("Braket task results will use S3 bucket \"s3://{}\"",
                   bucketName);
        return bucketName;
      }).share();
}

ServerJobPayload BraketExecutor::checkHelperAndCreateJob(
    std::vector<KernelExecution> &codesToExecute) {
  auto braketServerHelper = dynamic_cast<BraketServerHelper *>(serverHelper);
  assert(braketServerHelper);
  braketServerHelper->setShots(shots);

  auto config = braketServerHelper->getConfig();
  CUDAQ_INFO("Backend config: {}, shots {}", config, shots);
  config.insert({"shots", std::to_string(shots)});

  return braketServerHelper->createJob(codesToExecute);
}

void BraketExecutor::setOutputNames(const KernelExecution &codeToExecute,
                                    const std::string &taskId) {
  auto braketServerHelper = dynamic_cast<BraketServerHelper *>(serverHelper);
  assert(braketServerHelper);
  auto config = braketServerHelper->getConfig();

  auto output_names = codeToExecute.output_names.dump();
  config["output_names." + taskId] = output_names;

  braketServerHelper->setOutputNames(taskId, output_names);
}

details::future
BraketExecutor::execute(std::vector<KernelExecution> &codesToExecute,
                        cudaq::details::ExecutionContextType execType,
                        std::vector<char> *rawOutput) {
  const bool isObserve =
      execType == cudaq::details::ExecutionContextType::observe;
  auto [dummy1, dummy2, messages] = checkHelperAndCreateJob(codesToExecute);

  std::string const defaultBucket = defaultBucketFuture.get();
  std::string const defaultPrefix = "tasks";

  std::vector<Aws::Braket::Model::CreateQuantumTaskOutcomeCallable>
      createOutcomes;

  for (const auto &message : messages) {
    Aws::Braket::Model::CreateQuantumTaskRequest req;
    req.SetAction(message["action"]);
    req.SetDeviceArn(message["deviceArn"]);
    req.SetShots(message["shots"]);
    if (jobToken)
      req.SetJobToken(jobToken);

    if (reservationArn) {
      Aws::Braket::Model::Association assoc;
      assoc.SetArn(reservationArn);
      assoc.SetType(
          Aws::Braket::Model::AssociationType::RESERVATION_TIME_WINDOW_ARN);
      req.AddAssociations(std::move(assoc));
    }

    req.SetOutputS3Bucket(defaultBucket);
    req.SetOutputS3KeyPrefix(defaultPrefix);

    createOutcomes.push_back(braketClientPtr->CreateQuantumTaskCallable(req));
  }

  return std::async(
      std::launch::async,
      [this, codesToExecute, isObserve](
          std::vector<Aws::Braket::Model::CreateQuantumTaskOutcomeCallable>
              createOutcomes) {
        std::vector<ExecutionResult> results;
        for (std::size_t i = 0; auto &outcome : createOutcomes) {
          auto createResponse = outcome.get();
          if (!createResponse.IsSuccess()) {
            throw std::runtime_error(createResponse.GetError().GetMessage());
          }
          std::string taskArn = createResponse.GetResult().GetQuantumTaskArn();

          CUDAQ_INFO("Created Braket quantum task {}", taskArn);
          setOutputNames(codesToExecute[i], taskArn);

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

          CUDAQ_INFO("Fetching braket quantum task {} results from "
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

          if (isObserve) {
            // Use the job name instead of the global register.
            results.emplace_back(c.to_map(), codesToExecute[i].name);
            results.back().sequentialData = c.sequential_data();
          } else {
            // For each register, add the results into result.
            for (auto &regName : c.register_names()) {
              results.emplace_back(c.to_map(regName), regName);
              results.back().sequentialData = c.sequential_data(regName);
            }
          }
          i++;
        }

        return sample_result(results);
      },
      std::move(createOutcomes));
}
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::BraketExecutor, braket);
