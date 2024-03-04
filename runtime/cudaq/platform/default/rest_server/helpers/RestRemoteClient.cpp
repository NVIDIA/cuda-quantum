/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/BaseRestRemoteClient.h"
#include "common/NvqcConfig.h"

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
  // None: Don't log; Info: basic info; Trace: Timing data per invocation.
  enum class LogLevel : int { None = 0, Info, Trace };
  // NVQC logging level
  // Enabled high-level info log by default (can be set by an environment
  // variable)
  LogLevel m_logLevel = LogLevel::Info;
  // API key for authentication
  std::string m_apiKey;
  // Rest client to send HTTP request
  cudaq::RestClient m_restClient;
  // NVCF function Id to use
  std::string m_functionId;
  // NVCF version Id of that function to use
  std::string m_functionVersionId;
  // Available functions: function Id to number of GPUs mapping
  using DeploymentInfo = std::unordered_map<std::string, std::size_t>;
  DeploymentInfo m_availableFuncs;
  const std::string CUDAQ_NCA_ID = cudaq::getNvqcNcaId();
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
    auto versionDataJs = m_restClient.get(
        fmt::format("https://{}/nvcf/functions/{}", m_baseUrl, m_functionId),
        "/versions", headers, /*enableSsl=*/true);
    cudaq::info("Version data: {}", versionDataJs.dump());
    std::vector<cudaq::NvcfFunctionVersionInfo> versions;
    versionDataJs["functions"].get_to(versions);
    return versions;
  }
  DeploymentInfo getAllAvailableDeployments() {
    auto headers = getHeaders();
    auto allVisibleFunctions =
        m_restClient.get(fmt::format("https://{}/nvcf/functions", m_baseUrl),
                         "", headers, /*enableSsl=*/true);
    const auto queryDeploymentInfo =
        [&](const std::string &functionNamePrefix) -> DeploymentInfo {
      DeploymentInfo info;
      for (auto funcInfo : allVisibleFunctions["functions"]) {
        if (funcInfo["ncaId"].get<std::string>() == CUDAQ_NCA_ID &&
            funcInfo["status"].get<std::string>() == "ACTIVE" &&
            funcInfo["name"].get<std::string>().starts_with(
                functionNamePrefix)) {
          const std::size_t numGpus = [&]() {
            if (funcInfo.contains("containerEnvironment")) {
              for (auto it : funcInfo["containerEnvironment"]) {
                if (it["key"].get<std::string>() == "NUM_GPUS")
                  return std::stoi(it["value"].get<std::string>());
              }
            }
            return 1;
          }();

          info[funcInfo["id"].get<std::string>()] = numGpus;
        }
      }
      return info;
    };

    const std::string baseCudaqNvcfFuncNamePrefix = "cuda_quantum";
    const std::string versionedCudaqNvcfFuncNamePrefix =
        baseCudaqNvcfFuncNamePrefix + "_" +
        std::to_string(cudaq::RestRequest::REST_PAYLOAD_VERSION);

    // First, query the functions that have an exact version match
    DeploymentInfo info = queryDeploymentInfo(versionedCudaqNvcfFuncNamePrefix);
    if (!info.empty())
      return info;

    cudaq::info("Unable to find any active NVQC deployments for REST request "
                "version '{}' under NVIDIA Cloud Account Id {}.",
                cudaq::RestRequest::REST_PAYLOAD_VERSION, CUDAQ_NCA_ID);
    cudaq::info("Try to find generic deployments under the same NCA Id.");
    return queryDeploymentInfo(baseCudaqNvcfFuncNamePrefix);
  }

  std::optional<std::size_t> getQueueDepth(const std::string &funcId,
                                           const std::string &verId) {
    auto headers = getHeaders();
    try {
      auto queueDepthInfo = m_restClient.get(
          fmt::format("https://{}/nvcf/queues/functions/{}/versions/{}",
                      m_baseUrl, funcId, verId),
          "", headers, /*enableSsl=*/true);

      if (queueDepthInfo.contains("functionId") &&
          queueDepthInfo["functionId"] == funcId &&
          queueDepthInfo.contains("queues")) {
        for (auto queueInfo : queueDepthInfo["queues"]) {
          if (queueInfo.contains("functionVersionId") &&
              queueInfo["functionVersionId"] == verId &&
              queueInfo.contains("queueDepth")) {
            return queueInfo["queueDepth"].get<std::size_t>();
          }
        }
      }
      return std::nullopt;
    } catch (...) {
      // Make this non-fatal. Returns null, i.e., unknown.
      return std::nullopt;
    }
  }

public:
  virtual void setConfig(
      const std::unordered_map<std::string, std::string> &configs) override {
    {
      // Check if user set a specific log level (e.g., disable logging)
      if (auto logConfigEnv = std::getenv("NVQC_LOG_LEVEL")) {
        auto logConfig = std::string(logConfigEnv);
        std::transform(logConfig.begin(), logConfig.end(), logConfig.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        if (logConfig == "0" || logConfig == "off" || logConfig == "false" ||
            logConfig == "no" || logConfig == "none")
          m_logLevel = LogLevel::None;
        if (logConfig == "trace")
          m_logLevel = LogLevel::Trace;
        if (logConfig == "info")
          m_logLevel = LogLevel::Info;
      }
    }
    {
      const auto apiKeyIter = configs.find("api-key");
      if (apiKeyIter != configs.end())
        m_apiKey = apiKeyIter->second;
      if (m_apiKey.empty())
        throw std::runtime_error("No NVQC API key is provided.");
    }

    m_availableFuncs = getAllAvailableDeployments();
    for (const auto &[funcId, numGpus] : m_availableFuncs)
      cudaq::info("Function Id {} has {} GPUs.", funcId, numGpus);
    {
      const auto funcIdIter = configs.find("function-id");
      if (funcIdIter != configs.end()) {
        // User overrides a specific function Id.
        m_functionId = funcIdIter->second;
        if (m_logLevel > LogLevel::None) {
          // Print out the configuration
          cudaq::log("Submitting jobs to NVQC using function Id {}.",
                     m_functionId);
        }
      } else {
        // Determine the function Id based on the number of GPUs
        const auto nGpusIter = configs.find("ngpus");
        // Default is 1 GPU if none specified
        const std::size_t numGpusRequested =
            (nGpusIter != configs.end()) ? std::stoi(nGpusIter->second) : 1;
        cudaq::info("Looking for an NVQC deployment that has {} GPUs.",
                    numGpusRequested);
        for (const auto &[funcId, numGpus] : m_availableFuncs) {
          if (numGpus == numGpusRequested) {
            m_functionId = funcId;
            if (m_logLevel > LogLevel::None) {
              // Print out the configuration
              cudaq::log("Submitting jobs to NVQC service with {} GPU(s).",
                         numGpus);
            }
            break;
          }
        }
        if (m_functionId.empty()) {
          // Make sure that we sort the GPU count list
          std::set<std::size_t> gpuCounts;
          for (const auto &[funcId, numGpus] : m_availableFuncs) {
            gpuCounts.emplace(numGpus);
          }
          std::stringstream ss;
          ss << "Unable to find NVQC deployment with " << numGpusRequested
             << " GPUs.\nAvailable deployments have ";
          ss << fmt::format("{}", gpuCounts) << " GPUs.\n";
          ss << "Please check your 'ngpus' value (Python) or `--nvqc-ngpus` "
                "value (C++).\n";
          throw std::runtime_error(ss.str());
        }
      }
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
              fmt::format("Version Id '{}' is not valid for NVQC function Id "
                          "'{}'. Please check your NVQC configurations.",
                          m_functionVersionId, m_functionId));
        // The version is not active/deployed.
        if (versionInfoIter->status != cudaq::FunctionStatus::ACTIVE)
          throw std::runtime_error(
              fmt::format("Version Id '{}' of NVQC function Id "
                          "'{}' is not ACTIVE. Please check your NVQC "
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
              fmt::format("No active version available for NVQC function Id "
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
    static const std::vector<std::string> MULTI_GPU_BACKENDS = {"tensornet",
                                                                "nvidia-mgpu"};
    {
      // Print out a message if users request a multi-GPU deployment while
      // setting the backend to a single-GPU one. Only print once in case this
      // is a execution loop.
      static bool printOnce = false;
      if (m_availableFuncs[m_functionId] > 1 &&
          std::find(MULTI_GPU_BACKENDS.begin(), MULTI_GPU_BACKENDS.end(),
                    backendSimName) == MULTI_GPU_BACKENDS.end() &&
          !printOnce) {
        std::cout << "The requested backend simulator (" << backendSimName
                  << ") is not capable of using all "
                  << m_availableFuncs[m_functionId] << " GPUs requested.\n";
        std::cout << "Only one GPU will be used for simulation.\n";
        std::cout << "Please refer to CUDA Quantum documentation for a list of "
                     "multi-GPU capable simulator backends.\n";
        printOnce = true;
      }
    }
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

    if (request.format != cudaq::CodeFormat::MLIR) {
      // The `.config` file may have been tampered with.
      std::cerr << "Internal error: unsupported kernel IR detected.\nThis may "
                   "indicate a corrupted CUDA Quantum installation.";
      std::abort();
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
        cudaq::info("Deleting NVQC Asset Id {}", assetId.value());
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
          *optionalErrorMsg = "Failed to upload request to NVQC as NVCF assets";
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
      cudaq::debug("Sending NVQC request to {}", nvcfInvocationUrl());
      if (m_logLevel > LogLevel::None) {
        auto queueDepth = getQueueDepth(m_functionId, m_functionVersionId);
        if (queueDepth.has_value() && queueDepth.value() > 0) {
          cudaq::log("Number of jobs ahead of yours in the NVQC queue: {}.",
                     queueDepth.value());
        }
      }

      if (m_logLevel > LogLevel::None)
        cudaq::log("Posting NVQC request now");
      auto resultJs =
          m_restClient.post(nvcfInvocationUrl(), "", requestJson, jobHeader,
                            /*enableLogging=*/false, /*enableSsl=*/true);
      cudaq::debug("Response: {}", resultJs.dump());
      while (resultJs.contains("status") &&
             resultJs["status"] == "pending-evaluation") {
        const std::string reqId = resultJs["reqId"];
        if (m_logLevel > LogLevel::None)
          cudaq::log("Polling NVQC result data for Request Id {}", reqId);
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
          *optionalErrorMsg = "Unexpected response from the NVQC invocation. "
                              "Missing the 'response' field.";
        return false;
      }
      if (!resultJs["response"].contains("executionContext")) {
        if (optionalErrorMsg) {
          if (resultJs["response"].contains("errorMessage")) {
            *optionalErrorMsg = fmt::format(
                "NVQC failed to handle request. Server error: {}",
                resultJs["response"]["errorMessage"].get<std::string>());
          } else {
            *optionalErrorMsg =
                "Unexpected response from the NVQC response. "
                "Missing the required field 'executionContext'.";
          }
        }
        return false;
      }
      if (m_logLevel > LogLevel::None &&
          resultJs["response"].contains("executionInfo")) {
        try {
          // We only print GPU device info once if logging is not disabled.
          static bool printDeviceInfoOnce = false;
          cudaq::NvcfExecutionInfo info;
          resultJs["response"]["executionInfo"].get_to(info);
          if (!printDeviceInfoOnce) {
            std::size_t totalWidth = 50;
            std::string message = "NVQC Device Info";
            auto strLen = message.size() + 2; // Account for surrounding spaces
            auto leftSize = (totalWidth - strLen) / 2;
            auto rightSize = (totalWidth - strLen) - leftSize;
            std::string leftSide(leftSize, '=');
            std::string rightSide(rightSize, '=');
            fmt::print("\n{} {} {}\n", leftSide, message, rightSide);
            fmt::print("GPU Device Name: \"{}\"\n",
                       info.deviceProps.deviceName);
            fmt::print("CUDA Driver Version / Runtime Version: {}.{} / {}.{}\n",
                       info.deviceProps.driverVersion / 1000,
                       (info.deviceProps.driverVersion % 100) / 10,
                       info.deviceProps.runtimeVersion / 1000,
                       (info.deviceProps.runtimeVersion % 100) / 10);
            fmt::print("Total global memory (GB): {:.1f}\n",
                       (float)(info.deviceProps.totalGlobalMemMbytes) / 1024.0);
            fmt::print("Memory Clock Rate (MHz): {:.3f}\n",
                       info.deviceProps.memoryClockRateMhz);
            fmt::print("GPU Clock Rate (MHz): {:.3f}\n",
                       info.deviceProps.clockRateMhz);
            fmt::print("{}\n", std::string(totalWidth, '='));
            // Only print this device info once.
            printDeviceInfoOnce = true;
          }

          // If trace logging mode is enabled, log timing data for each request.
          if (m_logLevel == LogLevel::Trace) {
            fmt::print("\n===== NVQC Execution Timing ======\n");
            fmt::print(" - Pre-processing: {} milliseconds \n",
                       info.simulationStart - info.requestStart);
            fmt::print(" - Execution: {} milliseconds \n",
                       info.simulationEnd - info.simulationStart);
            fmt::print("==================================\n");
          }
        } catch (...) {
          fmt::print("Unable to parse NVQC execution info metadata.\n");
        }
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
    requestJson["description"] = "cudaq-nvqc-job";
    try {
      auto headers = getHeaders();
      auto resultJs =
          m_restClient.post(nvcfAssetUrl(), "", requestJson, headers,
                            /*enableLogging=*/false, /*enableSsl=*/true);
      const std::string uploadUrl = resultJs["uploadUrl"];
      const std::string assetId = resultJs["assetId"];
      cudaq::info("Upload NVQC job request as NVCF Asset Id {} to {}", assetId,
                  uploadUrl);
      std::map<std::string, std::string> uploadHeader;
      // This must match the request to create the upload link
      uploadHeader["Content-Type"] = "application/json";
      uploadHeader["x-amz-meta-nvcf-asset-description"] = "cudaq-nvqc-job";
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
