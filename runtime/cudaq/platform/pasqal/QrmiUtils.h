/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ServerHelper.h"
#include "qrmi.h"

#include <chrono>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace cudaq::qrmi {

struct ResourceDeleter {
  void operator()(QrmiQuantumResource *resource) const {
    if (resource)
      qrmi_resource_free(resource);
  }
};

struct StringDeleter {
  void operator()(char *value) const {
    if (value)
      qrmi_string_free(value);
  }
};

using ResourcePtr = std::unique_ptr<QrmiQuantumResource, ResourceDeleter>;
using StringPtr = std::unique_ptr<char, StringDeleter>;

inline std::string lastError() {
  if (auto *error = qrmi_get_last_error())
    return std::string(error);
  return "QRMI call failed without an error message.";
}

inline void ensureSuccess(QrmiReturnCode rc, std::string_view context) {
  if (rc != QRMI_RETURN_CODE_SUCCESS)
    throw std::runtime_error(std::string(context) + ": " + lastError());
}

inline std::vector<std::string> splitCsv(const char *value) {
  if (!value)
    return {};
  std::vector<std::string> out;
  std::string csv(value);
  std::size_t begin = 0;
  while (begin <= csv.size()) {
    auto end = csv.find(',', begin);
    out.push_back(
        csv.substr(begin, end == std::string::npos ? end : end - begin));
    if (end == std::string::npos)
      break;
    begin = end + 1;
  }
  return out;
}

inline QrmiResourceType parseResourceType(const std::string &resourceType) {
  if (resourceType == "pasqal-cloud")
    return QRMI_RESOURCE_TYPE_PASQAL_CLOUD;
  if (resourceType == "pasqal-local")
    return QRMI_RESOURCE_TYPE_PASQAL_LOCAL;

  throw std::runtime_error("Unsupported QRMI resource type '" + resourceType +
                           "' in SLURM_JOB_QPU_TYPES. Supported values are "
                           "pasqal-cloud and pasqal-local.");
}

inline QrmiResourceType resolveResourceType(const std::string &backendName) {
  auto resources = splitCsv(std::getenv("SLURM_JOB_QPU_RESOURCES"));
  if (resources.empty()) {
    throw std::runtime_error(
        "QRMI mode requires SLURM_JOB_QPU_RESOURCES to resolve backend type.");
  }

  auto types = splitCsv(std::getenv("SLURM_JOB_QPU_TYPES"));
  if (types.empty()) {
    throw std::runtime_error(
        "QRMI mode requires SLURM_JOB_QPU_TYPES to resolve backend type.");
  }

  for (std::size_t idx = 0; idx < resources.size(); ++idx) {
    if (resources[idx] == backendName) {
      if (idx >= types.size()) {
        throw std::runtime_error("QRMI backend '" + backendName +
                                 "' has no matching entry in "
                                 "SLURM_JOB_QPU_TYPES.");
      }
      return parseResourceType(types[idx]);
    }
  }

  throw std::runtime_error("QRMI backend '" + backendName +
                           "' is not present in SLURM_JOB_QPU_RESOURCES.");
}

inline bool taskInProgress(QrmiTaskStatus status);
inline std::string taskStatusAsString(QrmiTaskStatus status);

class ResourceSession {
public:
  ResourceSession(const std::string &backendName)
      : backendName_(backendName),
        resource_(qrmi_resource_new(backendName.c_str(),
                                    resolveResourceType(backendName_))) {
    if (!resource_) {
      throw std::runtime_error("Failed to create QRMI resource for '" +
                               backendName_ + "': " + lastError());
    }

    char *acquisitionTokenRaw = nullptr;
    ensureSuccess(qrmi_resource_acquire(resource_.get(), &acquisitionTokenRaw),
                  "qrmi_resource_acquire()");
    acquisitionToken_.reset(acquisitionTokenRaw);
  }

  ResourceSession(const ResourceSession &) = delete;
  ResourceSession &operator=(const ResourceSession &) = delete;
  ResourceSession(ResourceSession &&) = default;
  ResourceSession &operator=(ResourceSession &&) = default;

  ~ResourceSession() {
    if (!resource_ || !acquisitionToken_)
      return;
    auto rc = qrmi_resource_release(resource_.get(), acquisitionToken_.get());
    if (rc == QRMI_RETURN_CODE_SUCCESS)
      acquisitionToken_.reset();
  }

  std::string startTask(QrmiPayload &payload) {
    char *taskIdRaw = nullptr;
    ensureSuccess(
        qrmi_resource_task_start(resource_.get(), &payload, &taskIdRaw),
        "qrmi_resource_task_start()");
    if (!taskIdRaw)
      throw std::runtime_error(
          "qrmi_resource_task_start() returned a null task id.");

    StringPtr taskId(taskIdRaw);
    return std::string(taskId.get());
  }

  QrmiTaskStatus waitForTaskTerminalStatus(
      const std::string &taskId,
      std::chrono::seconds pollInterval = std::chrono::seconds(1),
      std::chrono::seconds timeout = std::chrono::seconds(60 * 60 * 24 * 30)) {
    if (auto timeoutStr = std::getenv("CUDAQ_QRMI_TIMEOUT_SEC"))
      timeout = std::chrono::seconds(std::atoi(timeoutStr));
    if (timeout <= std::chrono::seconds::zero())
      throw std::runtime_error("CUDAQ_QRMI_TIMEOUT_SEC must be positive.");

    const auto start = std::chrono::steady_clock::now();
    QrmiTaskStatus status = QRMI_TASK_STATUS_QUEUED;
    while (true) {
      ensureSuccess(
          qrmi_resource_task_status(resource_.get(), taskId.c_str(), &status),
          "qrmi_resource_task_status()");
      if (!taskInProgress(status))
        return status;
      if (std::chrono::steady_clock::now() - start >= timeout)
        throw std::runtime_error(
            "QRMI task '" + taskId +
            "' timed out while waiting for terminal status.");
      std::this_thread::sleep_for(pollInterval);
    }
  }

  std::string taskLogs(const std::string &taskId) const {
    char *taskLogsRaw = nullptr;
    if (qrmi_resource_task_logs(resource_.get(), taskId.c_str(),
                                &taskLogsRaw) != QRMI_RETURN_CODE_SUCCESS ||
        !taskLogsRaw)
      return {};

    StringPtr taskLogs(taskLogsRaw);
    return std::string(taskLogs.get());
  }

  std::string taskResult(const std::string &taskId) const {
    char *taskResultRaw = nullptr;
    ensureSuccess(qrmi_resource_task_result(resource_.get(), taskId.c_str(),
                                            &taskResultRaw),
                  "qrmi_resource_task_result()");
    if (!taskResultRaw)
      throw std::runtime_error(
          "qrmi_resource_task_result() returned a null payload.");

    StringPtr taskResult(taskResultRaw);
    return std::string(taskResult.get());
  }

  void ensureTaskCompleted(const std::string &taskId,
                           QrmiTaskStatus status) const {
    if (status == QRMI_TASK_STATUS_COMPLETED)
      return;

    auto logs = taskLogs(taskId);
    throw std::runtime_error("QRMI task '" + taskId + "' ended with status " +
                             taskStatusAsString(status) +
                             (logs.empty() ? "" : (": " + logs)));
  }

private:
  std::string backendName_;
  ResourcePtr resource_;
  StringPtr acquisitionToken_;
};

inline bool taskInProgress(QrmiTaskStatus status) {
  return status == QRMI_TASK_STATUS_QUEUED ||
         status == QRMI_TASK_STATUS_RUNNING;
}

inline std::string taskStatusAsString(QrmiTaskStatus status) {
  switch (status) {
  case QRMI_TASK_STATUS_QUEUED:
    return "QUEUED";
  case QRMI_TASK_STATUS_RUNNING:
    return "RUNNING";
  case QRMI_TASK_STATUS_COMPLETED:
    return "COMPLETED";
  case QRMI_TASK_STATUS_FAILED:
    return "FAILED";
  case QRMI_TASK_STATUS_CANCELLED:
    return "CANCELLED";
  }
  return "UNKNOWN";
}

} // namespace cudaq::qrmi
