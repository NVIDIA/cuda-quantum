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
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>

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

inline bool taskInProgress(QrmiTaskStatus status);
inline std::string taskStatusAsString(QrmiTaskStatus status);

class ResourceSession {
public:
  ResourceSession( // TODO: I think resource type should not be needed by
                   // caller. fix in qrmi
      const std::string &backendName,
      QrmiResourceType resourceType = QRMI_RESOURCE_TYPE_PASQAL_CLOUD)
      : backendName_(backendName),
        resource_(qrmi_resource_new(backendName.c_str(), resourceType)) {
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
      std::chrono::seconds pollInterval = std::chrono::seconds(1)) {
    QrmiTaskStatus status = QRMI_TASK_STATUS_QUEUED;
    while (true) {
      ensureSuccess(
          qrmi_resource_task_status(resource_.get(), taskId.c_str(), &status),
          "qrmi_resource_task_status()");
      if (!taskInProgress(status))
        return status;
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
  default:
    return "UNKNOWN";
  }
}

} // namespace cudaq::qrmi
