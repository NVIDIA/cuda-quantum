/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRestRemoteClient.h"

using namespace mlir;

namespace {
class RemoteRestRuntimeClient : public cudaq::BaseRemoteRestRuntimeClient {
public:
  /// @brief The constructor
  RemoteRestRuntimeClient() : BaseRemoteRestRuntimeClient() {}
};

/// REST client submitting jobs to NVCF-hosted `cudaq-qpud` service.
class NvcfRuntimeClient : public RemoteRestRuntimeClient {
private:
  // API key for authentication
  std::string m_apiKey;
  // Rest client to send HTTP request
  cudaq::RestClient m_restClient;
  // NVCF function Id to use
  std::string m_functionId;
  // NVCF version Id of that function to use
  std::string m_functionVersionId;
  // Base URL for NVCF APIs
  static inline const std::string m_baseUrl = "api.nvcf.nvidia.com/v2";
  // Return the URL to invoke the function specified in this client
  std::string nvcfInvocationUrl() const {
    return fmt::format("https://{}/nvcf/exec/functions/{}/versions/{}",
                       m_baseUrl, m_functionId, m_functionVersionId);
  }
  // Return the URL to request an Asset upload link
  std::string nvcfAssetUrl() const {
    return fmt::format("https://{}/nvcf/assets", m_baseUrl);
  }
  // Return the URL to retrieve status/result of an NVCF request.
  std::string
  nvcfInvocationStatus(const std::string &invocationRequestId) const {
    return fmt::format("https://{}/nvcf/exec/status/{}", m_baseUrl,
                       invocationRequestId);
  }
  // Construct the REST headers for calling NVCF REST APIs
  std::map<std::string, std::string> getHeaders() const {
    std::map<std::string, std::string> header{
        {"Authorization", fmt::format("Bearer {}", m_apiKey)},
        {"Content-type", "application/json"}};
    return header;
  };
  // Helper to retrieve the list of all available versions of the specified
  // function Id.
  std::vector<cudaq::NvcfFunctionVersionInfo> getFunctionVersions() {
    auto headers = getHeaders();
    auto versionDataJs =
        m_restClient.get(fmt::format("https://{}/nvcf/functions/{}", m_baseUrl,
                                     m_functionId, /*enableSsl=*/true),
                         "/versions", headers);
    cudaq::info("Version data: {}", versionDataJs.dump());
    std::vector<cudaq::NvcfFunctionVersionInfo> versions;
    versionDataJs["functions"].get_to(versions);
    return versions;
  }

public:
  virtual void setConfig(
      const std::unordered_map<std::string, std::string> &configs) override {
    {
      const auto apiKeyIter = configs.find("api-key");
      if (apiKeyIter != configs.end())
        m_apiKey = apiKeyIter->second;
      if (m_apiKey.empty())
        throw std::runtime_error("No NVCF API key is provided.");
    }
    {
      const auto funcIdIter = configs.find("function-id");
      if (funcIdIter != configs.end())
        m_functionId = funcIdIter->second;
      if (m_functionId.empty())
        throw std::runtime_error("No NVCF function Id is provided.");
    }
    {
      auto versions = getFunctionVersions();
      // Check if a version Id is set
      const auto versionIdIter = configs.find("version-id");
      if (versionIdIter != configs.end()) {
        m_functionVersionId = versionIdIter->second;
        // Do a sanity check that this is an active version (i.e., usable).
        const auto versionInfoIter =
            std::find_if(versions.begin(), versions.end(),
                         [&](const cudaq::NvcfFunctionVersionInfo &info) {
                           return info.versionId == m_functionVersionId;
                         });
        // Invalid version Id.
        if (versionInfoIter == versions.end())
          throw std::runtime_error(
              fmt::format("Version Id '{}' is not valid for NVCF function Id "
                          "'{}'. Please check your NVCF configurations.",
                          m_functionVersionId, m_functionId));
        // The version is not active/deployed.
        if (versionInfoIter->status != cudaq::FunctionStatus::ACTIVE)
          throw std::runtime_error(
              fmt::format("Version Id '{}' of NVCF function Id "
                          "'{}' is not ACTIVE. Please check your NVCF "
                          "configurations or contact support.",
                          m_functionVersionId, m_functionId));
      } else {
        // No version Id is set. Just pick the latest version of the function
        // Id. The timestamp is an ISO 8601 string, e.g.,
        // 2024-01-25T04:14:46.360Z. To sort it from latest to oldest, we can
        // use string sorting.
        std::sort(versions.begin(), versions.end(),
                  [](const auto &a, const auto &b) {
                    return a.createdAt > b.createdAt;
                  });
        for (const auto &versionInfo : versions)
          cudaq::info("Found version Id {}, created at {}",
                      versionInfo.versionId, versionInfo.createdAt);

        auto activeVersions =
            versions |
            std::ranges::views::filter(
                [](const cudaq::NvcfFunctionVersionInfo &info) {
                  return info.status == cudaq::FunctionStatus::ACTIVE;
                });

        if (activeVersions.empty())
          throw std::runtime_error(
              fmt::format("No active version available for NVCF function Id "
                          "'{}'. Please check your function Id.",
                          m_functionId));

        m_functionVersionId = activeVersions.front().versionId;
        cudaq::info("Selected the latest version Id {} for function Id {}",
                    m_functionVersionId, m_functionId);
      }
    }
  }
  virtual bool
  sendRequest(MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
              const std::string &backendSimName, const std::string &kernelName,
              void (*kernelFunc)(void *), void *kernelArgs,
              std::uint64_t argsSize, std::string *optionalErrorMsg) override {
    // Construct the base `cudaq-qpud` request payload.
    cudaq::RestRequest request =
        constructJobRequest(mlirContext, io_context, backendSimName, kernelName,
                            kernelFunc, kernelArgs, argsSize);

    if (request.code.empty()) {
      if (optionalErrorMsg)
        *optionalErrorMsg =
            std::string(
                "Failed to construct/retrieve kernel IR for kernel named ") +
            kernelName;
      return false;
    }
    // Max message size that we can send in the body
    constexpr std::size_t MAX_SIZE_BYTES = 250000; // 250 KB
    json requestJson;
    auto jobHeader = getHeaders();
    std::optional<std::string> assetId;
    // Make sure that we delete the asset that we've uploaded when this
    // `sendRequest` function exits (success or not).
    ScopeExit deleteAssetOnExit([&]() {
      if (assetId.has_value()) {
        cudaq::info("Deleting NVCF Asset Id {}", assetId.value());
        auto headers = getHeaders();
        m_restClient.del(nvcfAssetUrl(), std::string("/") + assetId.value(),
                         headers, /*enableLogging=*/false, /*enableSsl=*/true);
      }
    });

    // Upload this request as an NVCF asset if needed.
    if (request.code.size() > MAX_SIZE_BYTES) {
      assetId = uploadRequest(request);
      if (!assetId.has_value()) {
        if (optionalErrorMsg)
          *optionalErrorMsg = "Failed to upload request as NVCF assets";
        return false;
      }
      json requestBody;
      // Use NVCF `inputAssetReferences` field to specify the asset that needs
      // to be pulled in when invoking this function.
      requestBody["inputAssetReferences"] =
          std::vector<std::string>{assetId.value()};
      requestJson["requestBody"] = requestBody;
      requestJson["requestHeader"] = requestBody;
    } else {
      requestJson["requestBody"] = request;
    }

    try {
      // Making the request
      cudaq::debug("Sending NVCF request to {}", nvcfInvocationUrl());
      auto resultJs =
          m_restClient.post(nvcfInvocationUrl(), "", requestJson, jobHeader,
                            /*enableLogging=*/false, /*enableSsl=*/true);
      cudaq::debug("Response: {}", resultJs.dump());
      while (resultJs.contains("status") &&
             resultJs["status"] == "pending-evaluation") {
        const std::string reqId = resultJs["reqId"];
        cudaq::info("Polling result data for Request Id {}", reqId);
        // Wait 1 sec then poll the result
        std::this_thread::sleep_for(std::chrono::seconds(1));
        resultJs = m_restClient.get(nvcfInvocationStatus(reqId), "", jobHeader,
                                    /*enableSsl=*/true);
      }

      if (!resultJs.contains("status") || resultJs["status"] != "fulfilled") {
        if (optionalErrorMsg)
          *optionalErrorMsg =
              std::string(
                  "Failed to complete the simulation request. Status: ") +
              (resultJs.contains("status") ? std::string(resultJs["status"])
                                           : std::string("unknown"));
        return false;
      }

      // If there is a `responseReference` field, this is a large response.
      // Hence, need to download result .zip file from the provided URL.
      if (resultJs.contains("responseReference")) {
        // This is a large response that needs to be downloaded
        const std::string downloadUrl = resultJs["responseReference"];
        const std::string reqId = resultJs["reqId"];
        cudaq::info("Download result for Request Id {} at {}", reqId,
                    downloadUrl);
        llvm::SmallString<32> tempDir;
        llvm::sys::path::system_temp_directory(/*ErasedOnReboot*/ true,
                                               tempDir);
        std::filesystem::path resultFilePath =
            std::filesystem::path(tempDir.c_str()) / (reqId + ".zip");
        m_restClient.download(downloadUrl, resultFilePath.string(),
                              /*enableLogging=*/false, /*enableSsl=*/true);
        cudaq::info("Downloaded zip file {}", resultFilePath.string());
        std::filesystem::path unzipDir =
            std::filesystem::path(tempDir.c_str()) / reqId;
        // Unzip the response
        cudaq::utils::unzip(resultFilePath, unzipDir);
        std::filesystem::path resultJsonFile =
            unzipDir / (reqId + "_result.json");
        if (!std::filesystem::exists(resultJsonFile)) {
          if (optionalErrorMsg)
            *optionalErrorMsg =
                "Unexpected response file: missing the result JSON file.";
          return false;
        }
        std::ifstream t(resultJsonFile.string());
        std::string resultJsonFromFile((std::istreambuf_iterator<char>(t)),
                                       std::istreambuf_iterator<char>());
        try {
          resultJs["response"] = json::parse(resultJsonFromFile);
        } catch (...) {
          if (optionalErrorMsg)
            *optionalErrorMsg =
                fmt::format("Failed to parse the response JSON from file '{}'.",
                            resultJsonFile.string());
          return false;
        }
        cudaq::info(
            "Delete response zip file {} and its inflated contents in {}",
            resultFilePath.c_str(), unzipDir.c_str());
        std::filesystem::remove(resultFilePath);
        std::filesystem::remove_all(unzipDir);
      }

      if (!resultJs.contains("response")) {
        if (optionalErrorMsg)
          *optionalErrorMsg = "Unexpected response from the NVCF invocation. "
                              "Missing the 'response' field.";
        return false;
      }
      if (!resultJs["response"].contains("executionContext")) {
        if (optionalErrorMsg)
          *optionalErrorMsg = "Unexpected response from the NVCF response. "
                              "Missing the required field 'executionContext'.";
        return false;
      }
      resultJs["response"]["executionContext"].get_to(io_context);
      return true;
    } catch (std::exception &e) {
      if (optionalErrorMsg)
        *optionalErrorMsg = e.what();
      return false;
    }
  }

  // Upload a job request as an NVCF asset.
  // Return asset Id on success. Otherwise, return null.
  std::optional<std::string>
  uploadRequest(const cudaq::RestRequest &jobRequest) {
    json requestJson;
    requestJson["contentType"] = "application/json";
    requestJson["description"] = "cudaq-nvcf-job";
    try {
      auto headers = getHeaders();
      auto resultJs =
          m_restClient.post(nvcfAssetUrl(), "", requestJson, headers,
                            /*enableLogging=*/false, /*enableSsl=*/true);
      const std::string uploadUrl = resultJs["uploadUrl"];
      const std::string assetId = resultJs["assetId"];
      cudaq::info("Upload NVCF Asset Id {} to {}", assetId, uploadUrl);
      std::map<std::string, std::string> uploadHeader;
      // This must match the request to create the upload link
      uploadHeader["Content-Type"] = "application/json";
      uploadHeader["x-amz-meta-nvcf-asset-description"] = "cudaq-nvcf-job";
      json jobRequestJs = jobRequest;
      m_restClient.put(uploadUrl, "", jobRequestJs, uploadHeader,
                       /*enableLogging=*/false, /*enableSsl=*/true);
      return assetId;
    } catch (...) {
      return {};
    }
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, RemoteRestRuntimeClient, rest)
CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, NvcfRuntimeClient, NVCF)
