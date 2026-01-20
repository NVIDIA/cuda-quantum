/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/BraketServerHelper.h"
#include "common/Executor.h"
#include "common/Logger.h"
#include "common/SampleResult.h"
#include "cudaq.h"
#include <aws/braket/BraketClient.h>
#include <aws/core/Aws.h>
#include <aws/core/utils/logging/AWSLogging.h>
#include <aws/core/utils/logging/ConsoleLogSystem.h>
#include <aws/core/utils/logging/LogLevel.h>
#include <aws/s3-crt/S3CrtClient.h>
#include <aws/sts/STSClient.h>
#include <chrono>
#include <iostream>
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
      CUDAQ_DBG("Initializing AWS API");
      /// FIXME: Allow setting following flag via CUDA-Q frontend
      // options.loggingOptions.logLevel = Aws::Utils::Logging::LogLevel::Debug;
      Aws::InitAPI(options);
    }
    ~ScopedApi() { Aws::ShutdownAPI(options); }
  };

  ScopedApi api;

  std::unique_ptr<Aws::Braket::BraketClient> braketClientPtr;
  std::unique_ptr<Aws::STS::STSClient> stsClientPtr;
  std::unique_ptr<Aws::S3Crt::S3CrtClient> s3ClientPtr;

  std::shared_future<std::string> defaultBucketFuture;
  char const *jobToken;
  char const *reservationArn;

  std::chrono::microseconds pollingInterval = std::chrono::milliseconds{2000};

  /// @brief Utility function to check the type of ServerHelper and use it to
  /// create job
  virtual ServerJobPayload
  checkHelperAndCreateJob(std::vector<KernelExecution> &codesToExecute);

  /// @brief Utility function to set the output qubits for a task.
  void setOutputNames(const KernelExecution &codeToExecute,
                      const std::string &taskId);

public:
  BraketExecutor();

  ~BraketExecutor() = default;

  /// @brief Execute the provided Braket task
  details::future execute(std::vector<KernelExecution> &codesToExecute,
                          cudaq::details::ExecutionContextType execType,
                          std::vector<char> *rawOutput) override;

  /// @brief Set the server helper
  void setServerHelper(ServerHelper *helper) override;
};
} // namespace cudaq
