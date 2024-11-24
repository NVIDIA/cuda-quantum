/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BraketExecutor.h"
#include "common/BraketServerHelper.h"

#include <aws/braket/model/CreateQuantumTaskRequest.h>
#include <aws/braket/model/GetQuantumTaskRequest.h>
#include <aws/braket/model/QuantumTaskStatus.h>

#include <aws/s3-crt/model/CreateBucketRequest.h>
#include <aws/s3-crt/model/GetObjectRequest.h>
#include <aws/s3-crt/model/PutBucketPolicyRequest.h>
#include <aws/s3-crt/model/PutPublicAccessBlockRequest.h>

#include <aws/core/utils/ARN.h>

#include <iostream>

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
  cudaq::info("Attempting to create S3 bucket \"s3://{}\"", bucketName);
  auto createResponse = client.CreateBucket(createReq);
  if (!createResponse.IsSuccess()) {
    auto error = createResponse.GetError();
    if (error.GetErrorType() ==
        Aws::S3Crt::S3CrtErrors::BUCKET_ALREADY_OWNED_BY_YOU) {
      cudaq::info("\"s3://{}\" already exists", bucketName);
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
    : api(options), jobToken(std::getenv("AMZN_BRAKET_JOB_TOKEN")) {}

/// @brief Set the server helper
void BraketExecutor::setServerHelper(ServerHelper *helper) {
  Executor::setServerHelper(helper);

  std::string region =
      Aws::Utils::ARN(helper->getConfig().at("deviceArn")).GetRegion();
  std::string defaultBucket = helper->getConfig().at("defaultBucket");

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
        cudaq::info("Braket task results will use S3 bucket \"s3://{}\"",
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
  cudaq::info("Backend config: {}, shots {}", config, shots);
  config.insert({"shots", std::to_string(shots)});

  return braketServerHelper->createJob(codesToExecute);
}

details::future
BraketExecutor::execute(std::vector<KernelExecution> &codesToExecute) {
  // auto [dummy1, dummy2, messages] = checkHelperAndCreateJob(codesToExecute);

  // std::string const defaultBucket = defaultBucketFuture.get();
  // std::string const defaultPrefix = "tasks";

  // std::vector<Aws::Braket::Model::CreateQuantumTaskOutcomeCallable>
  //     createOutcomes;

  // for (const auto &message : messages) {
  //   Aws::Braket::Model::CreateQuantumTaskRequest req;
  //   req.SetAction(message["action"]);
  //   req.SetDeviceArn(message["deviceArn"]);
  //   req.SetShots(message["shots"]);
  //   if (jobToken)
  //     req.SetJobToken(jobToken);
  //   req.SetOutputS3Bucket(defaultBucket);
  //   req.SetOutputS3KeyPrefix(defaultPrefix);

  //   createOutcomes.push_back(braketClientPtr->CreateQuantumTaskCallable(req));
  // }

  return std::async(
      std::launch::async,
      [this](/*std::vector<Aws::Braket::Model::CreateQuantumTaskOutcomeCallable>
                 createOutcomes*/) {
        std::vector<ExecutionResult> results;
        //for (auto &outcome : createOutcomes) {
          // auto createResponse = outcome.get();
          // if (!createResponse.IsSuccess()) {
          //   throw std::runtime_error(createResponse.GetError().GetMessage());
          // }
          // std::string taskArn = createResponse.GetResult().GetQuantumTaskArn();
          
          // cudaq::info("Created Braket quantum task {}", taskArn);

          // Aws::Braket::Model::GetQuantumTaskRequest req;
          // req.SetQuantumTaskArn(taskArn);
          // auto getResponse = braketClientPtr->GetQuantumTask(req);
          // if (!getResponse.IsSuccess()) {
          //   throw std::runtime_error(getResponse.GetError().GetMessage());
          // }
          // auto taskStatus = getResponse.GetResult().GetStatus();
          // while (
          //     taskStatus != Aws::Braket::Model::QuantumTaskStatus::COMPLETED &&
          //     taskStatus != Aws::Braket::Model::QuantumTaskStatus::FAILED &&
          //     taskStatus != Aws::Braket::Model::QuantumTaskStatus::CANCELLED) {
          //   std::this_thread::sleep_for(pollingInterval);

          //   getResponse = braketClientPtr->GetQuantumTask(req);
          //   if (!getResponse.IsSuccess()) {
          //     throw std::runtime_error(getResponse.GetError().GetMessage());
          //   }
          //   taskStatus = getResponse.GetResult().GetStatus();
          // }

          // auto getResult = getResponse.GetResult();
          // if (taskStatus != Aws::Braket::Model::QuantumTaskStatus::COMPLETED) {
          //   // Task terminated without results
          //   throw std::runtime_error(
          //       fmt::format("Braket task {} terminated without results. {}",
          //                   taskArn, getResult.GetFailureReason()));
          // }

          // std::string outBucket = getResult.GetOutputS3Bucket();
          // std::string outPrefix = getResult.GetOutputS3Directory();

          // cudaq::info("Fetching braket quantum task {} results from "
          //             "s3://{}/{}/results.json",
          //             taskArn, outBucket, outPrefix);

          // Aws::S3Crt::Model::GetObjectRequest resultsJsonRequest;
          // resultsJsonRequest.SetBucket(outBucket);
          // resultsJsonRequest.SetKey(fmt::format("{}/results.json", outPrefix));
          // auto s3Response = s3ClientPtr->GetObject(resultsJsonRequest);
          // if (!s3Response.IsSuccess()) {
          //   throw std::runtime_error(s3Response.GetError().GetMessage());
          // }
          // auto resultsJson = nlohmann::json::parse(
          //     s3Response.GetResultWithOwnership().GetBody());
          std::string taskArn = "arn:aws:braket:us-east-1:783764578061:quantum-task/8bad9d49-b546-4ed8-8517-1b23c1eb929e";
          auto resultsJson = nlohmann::json::parse("{\"additionalMetadata\":{\"action\":{\"braketSchemaHeader\":{\"name\":\"braket.ir.openqasm.program\",\"version\":\"1\"},\"inputs\":{},\"source\":\"// Code generated by NVIDIA's nvq++ compiler\\nOPENQASM 2.0;\\n\\n\\n\\nqreg var0[4];\\nx var0[1];\\nx var0[2];\\ncreg var3[4];\\nmeasure var0 -> var3;\\n\"},\"simulatorMetadata\":{\"braketSchemaHeader\":{\"name\":\"braket.task_result.simulator_metadata\",\"version\":\"1\"},\"executionDuration\":2}},\"braketSchemaHeader\":{\"name\":\"braket.task_result.gate_model_task_result\",\"version\":\"1\"},\"measuredQubits\":[0,1,2,3],\"measurements\":[[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]],\"resultTypes\":[],\"taskMetadata\":{\"braketSchemaHeader\":{\"name\":\"braket.task_result.task_metadata\",\"version\":\"1\"},\"createdAt\":\"2024-11-23T19:16:08.872Z\",\"deviceId\":\"arn:aws:braket:::device/quantum-simulator/amazon/sv1\",\"deviceParameters\":{\"braketSchemaHeader\":{\"name\":\"braket.device_schema.simulators.gate_model_simulator_device_parameters\",\"version\":\"1\"},\"paradigmParameters\":{\"braketSchemaHeader\":{\"name\":\"braket.device_schema.gate_model_parameters\",\"version\":\"1\"},\"disableQubitRewiring\":false,\"qubitCount\":4}},\"endedAt\":\"2024-11-23T19:16:10.138Z\",\"id\":\"arn:aws:braket:us-east-1:783764578061:quantum-task/8bad9d49-b546-4ed8-8517-1b23c1eb929e\",\"shots\":100,\"status\":\"COMPLETED\"}}");
          std::cout << "Results: " << resultsJson << std::endl;
          auto c = serverHelper->processResults(resultsJson, taskArn);

          for (auto &regName : c.register_names()) {
            std::cout << "Register name: " << regName << std::endl;
            results.emplace_back(c.to_map(regName), regName);
            std::cout << "Sequential data: " << regName << std::endl;
            for(auto d: c.sequential_data(regName)) {
              std::cout << d << std::endl;
            }
            results.back().sequentialData = c.sequential_data(regName);
          }
        //}

        return sample_result(results);
      }/*,
      std::move(createOutcomes)*/);
}
} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::Executor, cudaq::BraketExecutor, braket);