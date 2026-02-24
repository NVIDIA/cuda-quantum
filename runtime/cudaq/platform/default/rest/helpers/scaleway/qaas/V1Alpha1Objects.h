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

/// @brief Common API objects used to interact with Scaleway QaaS.
/// Custom serialization / deserialization methods are implemented because of
/// 'null' field values. Objects such as Booking, Process and Application are
/// not implemented because not handled by CUDA-Q.
namespace cudaq::qaas::v1alpha1 {

template <typename T>
void get_safe(const json &j, const char *key, T &target,
              const T &default_val = T{}) {
  if (j.contains(key) && !j[key].is_null()) {
    j.at(key).get_to(target);
  } else {
    target = default_val;
  }
}

struct Platform {
  std::string id = "";
  std::string version = "";
  std::string name = "";
  std::string provider_name = "";
  std::string backend_name = "";
  std::string type = "";
  std::string technology = "";
  int64_t max_qubit_count = 0;
  int64_t max_shot_count = 0;
  int64_t max_circuit_count = 0;
  std::string availability = "";
  std::string metadata = "";
};

inline void from_json(const json &j, Platform &p) {
  get_safe(j, "id", p.id);
  get_safe(j, "name", p.name);
  get_safe(j, "version", p.version);
  get_safe(j, "provider_name", p.provider_name);
  get_safe(j, "backend_name", p.backend_name);
  get_safe(j, "type", p.type);
  get_safe(j, "technology", p.technology);
  get_safe(j, "availability", p.availability);
  get_safe(j, "metadata", p.metadata);
  get_safe(j, "max_qubit_count", p.max_qubit_count);
  get_safe(j, "max_shot_count", p.max_shot_count);
  get_safe(j, "max_circuit_count", p.max_circuit_count);
}

inline void to_json(json &j, const Platform &p) {
  j = json{
      {"id", p.id},
      {"name", p.name},
      {"version", p.version},
      {"provider_name", p.provider_name},
      {"backend_name", p.backend_name},
      {"type", p.type},
      {"technology", p.technology},
      {"availability", p.availability},
      {"metadata", p.metadata},
      {"max_qubit_count", p.max_qubit_count},
      {"max_shot_count", p.max_shot_count},
      {"max_circuit_count", p.max_circuit_count},
  };
}

struct Session {
  std::string id = "";
  std::string name = "";
  std::string platform_id = "";
  std::string created_at = "";
  std::string started_at = "";
  std::string updated_at = "";
  std::string terminated_at = "";
  std::string max_idle_duration = "";
  std::string max_duration = "";
  std::string status = "";
  std::string project_id = "";
  std::string deduplication_id = "";
  std::string progress_message = "";
  std::string parameters = "";
};

inline void from_json(const json &j, Session &p) {
  get_safe(j, "id", p.id);
  get_safe(j, "name", p.name);
  get_safe(j, "platform_id", p.platform_id);
  get_safe(j, "created_at", p.created_at);
  get_safe(j, "started_at", p.started_at);
  get_safe(j, "updated_at", p.updated_at);
  get_safe(j, "terminated_at", p.terminated_at);
  get_safe(j, "max_duration", p.max_duration);
  get_safe(j, "max_idle_duration", p.max_idle_duration);
  get_safe(j, "status", p.status);
  get_safe(j, "project_id", p.project_id);
  get_safe(j, "deduplication_id", p.deduplication_id);
  get_safe(j, "progress_message", p.progress_message);
  get_safe(j, "parameters", p.parameters);
}

inline void to_json(json &j, const Session &p) {
  j = json{{"id", p.id},
           {"name", p.name},
           {"platform_id", p.platform_id},
           {"created_at", p.created_at},
           {"started_at", p.started_at},
           {"updated_at", p.updated_at},
           {"terminated_at", p.terminated_at},
           {"max_idle_duration", p.max_idle_duration},
           {"max_duration", p.max_duration},
           {"status", p.status},
           {"project_id", p.project_id},
           {"deduplication_id", p.deduplication_id},
           {"progress_message", p.progress_message},
           {"parameters", p.parameters}};
}

struct Model {
  std::string id = "";
  std::string created_at = "";
  std::string url = "";
  std::string project_id = "";
};

inline void from_json(const json &j, Model &p) {
  get_safe(j, "id", p.id);
  get_safe(j, "project_id", p.project_id);
  get_safe(j, "url", p.url);
  get_safe(j, "created_at", p.created_at);
}

inline void to_json(json &j, const Model &p) {
  j = json{{"id", p.id},
           {"created_at", p.created_at},
           {"url", p.url},
           {"project_id", p.project_id}};
}

struct Job {
  std::string id = "";
  std::string name = "";
  std::string session_id = "";
  std::string created_at = "";
  std::string started_at = "";
  std::string updated_at = "";
  std::string status = "";
  std::string progress_message = "";
  std::string model_id = "";
  std::string parameters = "";

  inline bool is_finished() const {
    return status == "completed" || status == "error" || status == "cancelled";
  }
};

inline void from_json(const json &j, Job &p) {
  get_safe(j, "id", p.id);
  get_safe(j, "name", p.name);
  get_safe(j, "session_id", p.session_id);
  get_safe(j, "created_at", p.created_at);
  get_safe(j, "started_at", p.started_at);
  get_safe(j, "updated_at", p.updated_at);
  get_safe(j, "status", p.status);
  get_safe(j, "progress_message", p.progress_message);
  get_safe(j, "model_id", p.model_id);
  get_safe(j, "parameters", p.parameters);
}

inline void to_json(json &j, const Job &p) {
  j = json{{"id", p.id},
           {"name", p.name},
           {"session_id", p.session_id},
           {"created_at", p.created_at},
           {"started_at", p.started_at},
           {"updated_at", p.updated_at},
           {"status", p.status},
           {"progress_message", p.progress_message},
           {"model_id", p.model_id},
           {"parameters", p.parameters}};
}

struct JobResult {
  std::string job_id = "";
  std::string result = "";
  std::string url = "";
  std::string created_at = "";

  inline bool has_inline_result() const { return !result.empty(); }

  inline bool has_download_url() const { return !url.empty(); }
};

inline void from_json(const json &j, JobResult &p) {
  get_safe(j, "job_id", p.job_id);
  get_safe(j, "result", p.result);
  get_safe(j, "url", p.url);
  get_safe(j, "created_at", p.created_at);
}

inline void to_json(json &j, const JobResult &p) {
  j = json{{"job_id", p.job_id},
           {"result", p.result},
           {"url", p.url},
           {"created_at", p.created_at}};
}

} // namespace cudaq::qaas::v1alpha1
