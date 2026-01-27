/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "nlohmann/json.hpp"
#include <string>

using json = nlohmann::json;

namespace cudaq::qaas::v1alpha1 {

struct Platform {
  std::string id;
  std::string version;
  std::string name;
  std::string provider_name;
  std::string backend_name;
  std::string type;
  std::string technology;
  int64_t max_qubit_count;
  int64_t max_shot_count;
  int64_t max_circuit_count;
  std::string availability;
  std::string metadata;
  std::string description;
  std::string documentation_url;
  bool is_bookable = false;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      Platform, id, version, name, provider_name, backend_name, type,
      technology, max_qubit_count, max_shot_count, max_circuit_count,
      availability, metadata, description, documentation_url, is_bookable)
};

struct Session {
  std::string id;
  std::string name;
  std::string platform_id;
  std::string created_at;
  std::string started_at;
  std::string updated_at;
  std::string terminated_at;
  std::string max_idle_duration;
  std::string max_duration;
  int64_t waiting_job_count;
  int64_t finished_job_count;
  std::string status;
  std::string project_id;
  std::string deduplication_id;
  std::string origin_type;
  std::string origin_id;
  std::string progress_message;
  std::string booking_id;
  std::string model_id;
  std::string parameters;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      Session, id, name, platform_id, created_at, started_at, updated_at,
      terminated_at, max_idle_duration, max_duration, waiting_job_count,
      finished_job_count, status, project_id, deduplication_id, origin_type,
      origin_id, progress_message, booking_id, model_id, parameters)
};

struct Model {
  std::string id;
  std::string created_at;
  std::string url;
  std::string project_id;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(Model, id, created_at, url,
                                              project_id)
};

struct Job {
  std::string id;
  std::string name;
  std::string session_id;
  std::string created_at;
  std::string started_at;
  std::string updated_at;
  std::string status;
  std::string progress_message;
  std::string job_duration;
  std::string result_distribution;
  std::string model_id;
  std::string parameters;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(
      Job, id, name, session_id, created_at, started_at, updated_at, status,
      progress_message, job_duration, result_distribution, model_id, parameters)

  inline bool is_finished() const {
    return status == "completed" || status == "error" || status == "cancelled";
  }
};

struct JobResult {
  std::string job_id;
  std::string result;
  std::string url;
  std::string created_at;

  NLOHMANN_DEFINE_TYPE_INTRUSIVE_WITH_DEFAULT(JobResult, job_id, result, url,
                                              created_at)

  inline bool has_inline_result() const {
    return !result.empty();
  }

  inline bool has_download_url() const {
    return !url.empty();
  }
};
} // namespace cudaq::qaas::v1alpha1
