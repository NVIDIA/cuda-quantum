/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/Logger.h"
#include "common/RestClient.h"
#include "common/ServerHelper.h"
#include "cudaq/utils/cudaq_utils.h"
#include "nlohmann/json.hpp"
#include <iostream>
#include <map>

using json = nlohmann::json;

namespace cudaq {

class ScalewayServerHelper : public ServerHelper {
  private:
    std::string m_baseUrl = "https://api.scaleway.com/qaas/v1alpha1";
    std::string m_basePlatformName = "EMU-CUDAQ-H100";
    std::string m_projectId = "";
    std::string m_sessionId = "";
    std::string m_secretKey = "";

  void ensureSessionActive() {
       if (!m_sessionId.empty()) {
        try {
          std::string response = performInternalRequest("GET", "/sessions/" + sessionId);
          auto j = json::parse(response);
          std::string status = j.value("status", "unknown");

          if (status == "error" || status == "stopped" || status == "stopping") {
            m_sessionId = "";
          } else {
            return;
          }
        } catch (...) {
          m_sessionId = "";
        }
      }

      if (m_sessionId.empty()) {
        m_platformName = getOption("machine", m_basePlatformName);
        m_projectId = getOption("project_id");

        json sessionPayload;
        sessionPayload["name"] = "cudaq-session-" + std::to_string(std::rand());
        sessionPayload["platform_id"] = m_platformName; // TODO: get platform id
        if (!m_projectId.empty()) {
          sessionPayload["project_id"] = m_projectId;
        }

        // Endpoint sans région
        std::string response = performInternalRequest("POST", "/sessions", sessionPayload);

        auto j = json::parse(response);
        if (j.contains("id")) {
          m_sessionId = j["id"].get<std::string>();
        } else {
          throw std::runtime_error("Echec création session Scaleway: " + response);
        }
      }
    }

  std::string uploadModel(const std::string& name, const std::string& openqasm) {
      json circuitPayload;
      circuitPayload["name"] = name;
      circuitPayload["definition"]["openqasm"] = openqasm;
      if (!projectId.empty()) {
          circuitPayload["project_id"] = projectId;
      }

      std::string response = performInternalRequest("POST", "/model", circuitPayload);

      auto j = json::parse(response);
      if (j.contains("id")) {
        return j["id"].get<std::string>();
      }
      throw std::runtime_error("Echec upload circuit: " + response);
    }

  public:
  ScalewayServerHelper() : ServerHelper() {}

  virtual std::string constructPostEndpoint() override {
    return m_baseUrl + "/jobs";
  }

  virtual std::string constructGetEndpoint(const std::string &jobId) override {
    return m_baseUrl + "/jobs/" + jobId;
  }

  virtual std::map<std::string, std::string> constructHeaders() override {
    std::map<std::string, std::string> headers;
    std::string apiKey = getOption("api_key");
    if (apiKey.empty()) {
         if (const char* envKey = std::getenv("SCW_SECRET_KEY")) apiKey = envKey;
    }
    headers["X-Auth-Token"] = apiKey;
    headers["Content-Type"] = "application/json";
    return headers;
  }

  virtual std::string
  constructPostMessage(const std::string &circuitName,
                       const std::string &openqasmString,
                       const int shots) override {
    ensureSessionActive();

    std::string finalName = circuitName.empty() ? "cudaq-job" : circuitName;
    std::string circuitId = uploadCircuit(finalName, openqasmString);

    json payload;
    payload["session_id"] = sessionId;
    payload["model_id"] = modelId;
    payload["name"] = finalName;
    payload["sampling_count"] = shots;

    return payload.dump();
  }

  virtual std::string extractJobId(const std::string &postResponse) override {
    auto j = json::parse(postResponse);
    if (j.contains("id")) return j["id"].get<std::string>();
    throw std::runtime_error("Job submission failed: " + postResponse);
  }

  virtual bool isJobCompleted(const std::string &getResponse) override {
    auto j = json::parse(getResponse);
    std::string status = j.value("status", "unknown");
    if (status == "error") {
        std::string err = j.contains("result") ? j["result"].value("error_message", "Unknown error") : "Unknown";
        throw std::runtime_error("Scaleway Job Error: " + err);
    }
    return (status == "completed" || status == "canceled");
  }

  /**
   * @brief Traitement des résultats avec logique de fallback URL (Object Storage).
   */
  virtual void processResults(const std::string &getResponse,
                              sample_counts &counts) override {
    auto j = json::parse(getResponse);

    // On cherche la liste "job_results"
    if (!j.contains("job_results") || j["job_results"].empty()) {
        throw std::runtime_error("Job terminé mais aucun résultat (job_results vide).");
    }

    // On prend le premier résultat
    auto firstResult = j["job_results"][0];

    std::string rawPayload;

    // Cas 1 : Le résultat est directement dans le JSON
    if (firstResult.contains("result") && !firstResult["result"].is_null() && firstResult["result"] != "") {
        rawPayload = firstResult["result"].get<std::string>();
    }
    // Cas 2 : Le résultat est accessible via une URL (Object Storage)
    else if (firstResult.contains("url") && !firstResult["url"].is_null()) {
        std::string downloadUrl = firstResult["url"].get<std::string>();
        // Téléchargement synchrone via notre helper interne (mode URL absolue)
        rawPayload = performInternalRequest("GET", downloadUrl, {}, true);
    }
    else {
        throw std::runtime_error("Format de résultat invalide : ni champ 'result' ni champ 'url'.");
    }

    // Une fois le payload récupéré (depuis JSON ou S3), on le parse
    // On s'attend à ce que rawPayload soit un JSON contenant "execution_result" ou directement les counts
    try {
        auto resultJson = json::parse(rawPayload);

        // Selon la structure exacte retournée par le backend (AQT, Sim, Pasqal...)
        // On assume ici la structure standard QaaS : { "execution_result": { "00": 12, ... } }
        if (resultJson.contains("execution_result")) {
            auto executionData = resultJson["execution_result"];
            for (auto it = executionData.begin(); it != executionData.end(); ++it) {
                counts.emplace(it.key(), it.value().get<int>());
            }
        } else {
             // Fallback: peut-être que le rawPayload EST directement les counts ?
             for (auto it = resultJson.begin(); it != resultJson.end(); ++it) {
                if (it.value().is_number()) {
                    counts.emplace(it.key(), it.value().get<int>());
                }
             }
        }

    } catch (const std::exception& e) {
        throw std::runtime_error("Erreur parsing du résultat final Scaleway: " + std::string(e.what()) + " | Payload: " + rawPayload);
    }
  }
};

} // namespace cudaq

CUDAQ_REGISTER_TYPE(cudaq::ServerHelper, cudaq::ScalewayServerHelper, scaleway)