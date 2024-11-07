/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "common/Executor.h"
#include "common/FmtCore.h"
#include "common/MeasureCounts.h"
#include "cudaq.h"

#include <chrono>
#include <iostream>

#include <aws/core/Aws.h>

#include <aws/braket/BraketClient.h>
#include <aws/braket/model/CreateQuantumTaskRequest.h>
#include <aws/braket/model/GetQuantumTaskRequest.h>
#include <aws/braket/model/QuantumTaskStatus.h>

#include <aws/sts/STSClient.h>

#include <aws/s3-crt/S3CrtClient.h>
#include <aws/s3-crt/model/GetObjectRequest.h>

#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/logging/LogLevel.h>

#include "common/BraketServerHelper.h"
#include "common/Logger.h"

#include <nlohmann/json.hpp>
#include <regex>
#include <string>
#include <thread>

namespace cudaq {
/// @brief The Executor subclass for Amazon Braket
class BraketExecutor : public Executor {
protected:
  Aws::SDKOptions options;

  class ScopedApi {
    Aws::SDKOptions &options;

  public:
    ScopedApi(Aws::SDKOptions &options) : options(options) {
      cudaq::debug("Initializing AWS API");
      /// FIXME: Allow setting following flag via CUDA-Q frontend
      // options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Debug;
      Aws::InitAPI(options);
    }
    ~ScopedApi() { Aws::ShutdownAPI(options); }
  };

  ScopedApi api;
  Aws::Braket::BraketClient braketClient;
  Aws::STS::STSClient stsClient;
  Aws::S3Crt::S3CrtClient s3Client;

  std::future<std::string> defaultBucketFuture;
  char const *jobToken;

  std::chrono::microseconds pollingInterval = std::chrono::milliseconds{100};

  static auto getClientConfig() {
    Aws::Client::ClientConfiguration clientConfig;
    clientConfig.verifySSL = false;
    return clientConfig;
  }

  static auto getS3ClientConfig() {
    Aws::S3Crt::ClientConfiguration clientConfig;
    clientConfig.verifySSL = false;
    return clientConfig;
  }

  /// @brief Utility function to check the type of ServerHelper and use it to
  /// create job
  virtual ServerJobPayload
  checkHelperAndCreateJob(std::vector<KernelExecution> &codesToExecute);

public:
  BraketExecutor()
      : api(options), braketClient(getClientConfig()),
        stsClient(getClientConfig()), s3Client(getS3ClientConfig()),
        jobToken(std::getenv("AMZN_BRAKET_JOB_TOKEN")) {
    cudaq::debug("Creating BraketExecutor");

    defaultBucketFuture = std::async(std::launch::async, [this] {
      auto response = stsClient.GetCallerIdentity();
      std::string bucketName;
      if (response.IsSuccess()) {
        bucketName =
            fmt::format("amazon-braket-{}-{}", getClientConfig().region,
                        response.GetResult().GetAccount());
        cudaq::info("Braket task results will use S3 bucket \"{}\"",
                    bucketName);
        return bucketName;
      } else {
        throw std::runtime_error(response.GetError().GetMessage());
      }
    });
  }

  ~BraketExecutor() = default;

  /// @brief Execute the provided Braket task
  details::future
  execute(std::vector<KernelExecution> &codesToExecute) override;
};
} // namespace cudaq