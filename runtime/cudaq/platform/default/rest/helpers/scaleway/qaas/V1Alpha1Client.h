/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once
#include "V1Alpha1Objects.h"

#include "common/Logger.h"
#include "common/RestClient.h"
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <thread>

namespace cudaq::qaas::v1alpha1 {

class V1Alpha1Client {
public:
    void initialize(const std::string projectId, const std::string secretKey, std::string url);

    Platform getPlatform(const std::string& platformId);
    std::vector<Platform> listPlatforms(const std::string platformName);

    Session createSession(
        const std::string& platformId,
        std::string name = "",
        std::string deduplicationId = "",
        const std::string& modelId = "",
        std::string maxDuration = "59m",
        std::string maxIdleDuration = "59m",
        std::string parameters = "");
    Session getSession(const std::string& sessionId);

    Job createJob(const std::string& sessionId, const std::string& modelId, std::string name = "");
    Job getJob(const std::string& jobId);
    std::vector<JobResult> listJobResults(const std::string& jobId);

    Model createModel(const std::string& payload);

protected:
    RestHeaders getHeader() const;
private:
    RestClient m_client;
    std::string m_baseUrl = "https://api.scaleway.com/qaas/v1alpha1";
    std::string m_projectId = "";
    std::string m_secretKey = "";
    bool m_secure = true;
    bool m_logging = false;
};

} // namespace cudaq::qaas::v1alpha1
