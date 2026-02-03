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

  NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE_WITH_DEFAULT(
      Platform, id, version, name, provider_name, backend_name, type,
      technology, max_qubit_count, max_shot_count, max_circuit_count,
      availability, metadata)
};

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

void from_json(const json& j, Session& p) {
  auto get_safe = [&](const char* key, std::string& target) {
      if (j.contains(key) && !j[key].is_null()) {
          j.at(key).get_to(target);
      } else {
          target = "";
      }
  };

  get_safe("id", p.id);
  get_safe("name", p.name);
  get_safe("platform_id", p.platform_id);
  get_safe("created_at", p.created_at);
  get_safe("started_at", p.started_at);
  get_safe("updated_at", p.updated_at);
  get_safe("terminated_at", p.terminated_at);
  get_safe("max_duration", p.max_duration);
  get_safe("max_idle_duration", p.max_idle_duration);
  get_safe("status", p.status);
  get_safe("project_id", p.project_id);
  get_safe("deduplication_id", p.deduplication_id);
  get_safe("progress_message", p.progress_message);
  get_safe("parameters", p.parameters);
}

struct Model {
  std::string id = "";
  std::string created_at = "";
  std::string url = "";
  std::string project_id = "";
};

void from_json(const json& j, Model& p) {
  auto get_safe = [&](const char* key, std::string& target) {
      if (j.contains(key) && !j[key].is_null()) {
          j.at(key).get_to(target);
      } else {
          target = "";
      }
  };

  get_safe("id", p.id);
  get_safe("project_id", p.project_id);
  get_safe("url", p.url);
  get_safe("created_at", p.created_at);
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

void from_json(const json& j, Job& p) {
    auto get_safe = [&](const char* key, std::string& target) {
        if (j.contains(key) && !j[key].is_null()) {
            j.at(key).get_to(target);
        } else {
            target = "";
        }
    };

    get_safe("id", p.id);
    get_safe("name", p.name);
    get_safe("session_id", p.session_id);
    get_safe("created_at", p.created_at);
    get_safe("started_at", p.started_at);
    get_safe("updated_at", p.updated_at);
    get_safe("status", p.status);
    get_safe("progress_message", p.progress_message);
    get_safe("model_id", p.model_id);
    get_safe("parameters", p.parameters);
}

struct JobResult {
  std::string job_id = "";
  std::string result = "";
  std::string url = "";
  std::string created_at = "";

  inline bool has_inline_result() const {
    return !result.empty();
  }

  inline bool has_download_url() const {
    return !url.empty();
  }
};

void from_json(const json& j, JobResult& p) {
  auto get_safe = [&](const char* key, std::string& target) {
      if (j.contains(key) && !j[key].is_null()) {
          j.at(key).get_to(target);
      } else {
          target = "";
      }
  };

  get_safe("job_id", p.job_id);
  get_safe("result", p.result);
  get_safe("url", p.url);
  get_safe("created_at", p.created_at);
}
} // namespace cudaq::qaas::v1alpha1
