/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "common/RuntimeMLIR.h"
#include "cudaq/Frontend/nvqpp/AttributeNames.h"
#include "cudaq/Optimizer/CodeGen/OpenQASMEmitter.h"
#include "cudaq/Optimizer/CodeGen/Passes.h"
#include "cudaq/Optimizer/Dialect/CC/CCDialect.h"
#include "cudaq/Optimizer/Dialect/Quake/QuakeDialect.h"
#include "cudaq/Optimizer/Transforms/Passes.h"

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/Base64.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "JsonConvert.h"
#include "common/Logger.h"
#include "common/RemoteKernelExecutor.h"
#include "common/RestClient.h"
#include "cudaq.h"
#include <dlfcn.h>
#include <fstream>
#include <streambuf>
namespace {
struct ScopeExit {
  ScopeExit(std::function<void()> &&func) : m_atExitFunc(std::move(func)) {}
  ~ScopeExit() noexcept { m_atExitFunc(); }
  ScopeExit(const ScopeExit &) = delete;
  ScopeExit &operator=(const ScopeExit &) = delete;
  ScopeExit(ScopeExit &&other) = delete;
  ScopeExit &operator=(ScopeExit &&other) = delete;

private:
  std::function<void()> m_atExitFunc;
};

using namespace mlir;
class RemoteRestRuntimeClient : public cudaq::RemoteRuntimeClient {
  std::string m_url;
  static inline const std::vector<std::string> clientPasses = {
      "func.func(unwind-lowering)",
      "func.func(indirect-to-direct-calls)",
      "inline",
      "canonicalize",
      "apply-op-specialization",
      "func.func(memtoreg{quantum=0})",
      "canonicalize",
      "expand-measurements",
      "cc-loop-normalize",
      "cc-loop-unroll",
      "canonicalize",
      "func.func(add-dealloc)",
      "func.func(quake-add-metadata)",
      "canonicalize",
      "func.func(lower-to-cfg)",
      "func.func(combine-quantum-alloc)",
      "canonicalize",
      "cse",
      "quake-to-qir"};
  static inline const std::vector<std::string> serverPasses = {};

public:
  virtual void setConfig(
      const std::unordered_map<std::string, std::string> &configs) override {
    const auto urlIter = configs.find("url");
    if (urlIter != configs.end())
      m_url = urlIter->second;
  }

  std::string constructKernelPayload(MLIRContext &mlirContext,
                                     const std::string &name,
                                     void (*kernelFunc)(void *), void *args,
                                     std::uint64_t voidStarSize) {
    if (cudaq::__internal__::isLibraryMode(name)) {
      // Library mode: retrieve the embedded bitcode in the executable.
      const auto path = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
      // Load the object file
      auto [objBin, objBuffer] =
          llvm::cantFail(llvm::object::ObjectFile::createObjectFile(path))
              .takeBinary();
      if (!objBin)
        throw std::runtime_error("Failed to load binary object file");
      for (const auto &section : objBin->sections()) {
        // Get the bitcode section
        if (section.isBitcode()) {
          llvm::MemoryBufferRef llvmBc(llvm::cantFail(section.getContents()),
                                       "Bitcode");
          return llvm::encodeBase64(llvmBc.getBuffer());
        }
      }
      return "";
    } else {
      // Get the quake representation of the kernel
      auto quakeCode = cudaq::get_quake_by_name(name);
      auto module = parseSourceString<ModuleOp>(quakeCode, &mlirContext);
      if (!module)
        throw std::runtime_error("module cannot be parsed");

      // Extract the kernel name
      auto func = module->lookupSymbol<mlir::func::FuncOp>(
          std::string("__nvqpp__mlirgen__") + name);

      // Create a new Module to clone the function into
      auto location = FileLineColLoc::get(&mlirContext, "<builder>", 1, 1);
      ImplicitLocOpBuilder builder(location, &mlirContext);
      // Add entry-point attribute if not already set.
      if (!func->hasAttr(cudaq::entryPointAttrName))
        func->setAttr(cudaq::entryPointAttrName, builder.getUnitAttr());
      auto moduleOp = builder.create<ModuleOp>();
      moduleOp.push_back(func.clone());

      if (args) {
        cudaq::info("Run Quake Synth.\n");
        PassManager pm(&mlirContext);
        pm.addPass(cudaq::opt::createQuakeSynthesizer(name, args));
        if (failed(pm.run(moduleOp)))
          throw std::runtime_error("Could not successfully apply quake-synth.");
      }

      // Client-side passes
      if (!clientPasses.empty()) {
        PassManager pm(&mlirContext);
        std::string errMsg;
        llvm::raw_string_ostream os(errMsg);
        const std::string pipeline =
            std::accumulate(clientPasses.begin(), clientPasses.end(),
                            std::string(), [](const auto &ss, const auto &s) {
                              return ss.empty() ? s : ss + "," + s;
                            });
        if (failed(parsePassPipeline(pipeline, pm, os)))
          throw std::runtime_error(
              "Remote rest platform failed to add passes to pipeline (" +
              errMsg + ").");

        if (failed(pm.run(moduleOp)))
          throw std::runtime_error(
              "Remote rest platform: applying IR passes failed.");
      }
      std::string mlirCode;
      llvm::raw_string_ostream outStr(mlirCode);
      mlir::OpPrintingFlags opf;
      opf.enableDebugInfo(/*enable=*/true,
                          /*pretty=*/false);
      moduleOp.print(outStr, opf);
      return llvm::encodeBase64(mlirCode);
    }
  }
  cudaq::RestRequest constructJobRequest(
      MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
      const std::string &backendSimName, const std::string &kernelName,
      void (*kernelFunc)(void *), void *kernelArgs, std::uint64_t argsSize) {

    cudaq::RestRequest request(io_context);
    request.entryPoint = kernelName;
    if (cudaq::__internal__::isLibraryMode(kernelName)) {
      request.format = cudaq::CodeFormat::LLVM;
      if (kernelArgs && argsSize > 0) {
        cudaq::info("Serialize {} bytes of args.", argsSize);
        request.args.resize(argsSize);
        std::memcpy(request.args.data(), kernelArgs, argsSize);
      }

      if (kernelFunc) {
        ::Dl_info info;
        ::dladdr(reinterpret_cast<void *>(kernelFunc), &info);
        const auto funcName = cudaq::quantum_platform::demangle(info.dli_sname);
        cudaq::info("RemoteSimulatorQPU: retrieve name '{}' for kernel {}",
                    funcName, kernelName);
        request.entryPoint = funcName;
      }
    } else {
      request.passes = serverPasses;
      request.format = cudaq::CodeFormat::MLIR;
    }

    request.code = constructKernelPayload(mlirContext, kernelName, kernelFunc,
                                          kernelArgs, argsSize);
    request.simulator = backendSimName;
    request.seed = cudaq::get_random_seed();
    return request;
  }

  virtual bool
  sendRequest(MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
              const std::string &backendSimName, const std::string &kernelName,
              void (*kernelFunc)(void *), void *kernelArgs,
              std::uint64_t argsSize, std::string *optionalErrorMsg) override {
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

    // Don't let curl adding "Expect: 100-continue" header, which is not
    // suitable for large requests, e.g., bitcode in the JSON request.
    //  Ref: https://gms.tf/when-curl-sends-100-continue.html
    std::map<std::string, std::string> headers{
        {"Expect:", ""}, {"Content-type", "application/json"}};
    json requestJson = request;
    try {
      cudaq::RestClient restClient;
      auto resultJs =
          restClient.post(m_url, "job", requestJson, headers, false);

      if (!resultJs.contains("executionContext")) {
        if (optionalErrorMsg)
          *optionalErrorMsg = "Unexpected response from the REST server. "
                              "Missing the required field 'executionContext'.";
        return false;
      }
      resultJs["executionContext"].get_to(io_context);
      return true;
    } catch (std::exception &e) {
      if (optionalErrorMsg)
        *optionalErrorMsg = e.what();
      return false;
    }
  }
};

class NvcfRuntimeClient : public RemoteRestRuntimeClient {
private:
  std::string m_apiKey;
  cudaq::RestClient m_restClient;
  std::string m_functionId;
  std::string m_functionVersionId;
  static inline const std::string m_baseUrl = "api.nvcf.nvidia.com/v2";
  std::string nvcfInvocationUrl() const {
    return fmt::format("https://{}/nvcf/exec/functions/{}/versions/{}",
                       m_baseUrl, m_functionId, m_functionVersionId);
  }
  std::string nvcfAssetUrl() const {
    return fmt::format("https://{}/nvcf/assets", m_baseUrl);
  }

  std::string
  nvcfInvocationStatus(const std::string &invocationRequestId) const {
    return fmt::format("https://{}/nvcf/exec/status/{}", m_baseUrl,
                       invocationRequestId);
  }

  std::map<std::string, std::string> getHeaders() const {
    std::map<std::string, std::string> header{
        {"Authorization", fmt::format("Bearer {}", m_apiKey)},
        {"Content-type", "application/json"}};
    return header;
  };

  std::vector<cudaq::NvcfFunctionVersionInfo> getFunctionVersions() {
    auto headers = getHeaders();
    auto versionDataJs = m_restClient.get(
        fmt::format("https://{}/nvcf/functions/{}", m_baseUrl, m_functionId),
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
      // The timestamp is an ISO 8601 string, e.g., 2024-01-25T04:14:46.360Z.
      // To sort it from latest to oldest, we can use string sorting.
      std::sort(versions.begin(), versions.end(),
                [](const auto &a, const auto &b) {
                  return a.createdAt > b.createdAt;
                });
      for (const auto &versionInfo : versions)
        cudaq::info("Found version Id {}, created at {}", versionInfo.versionId,
                    versionInfo.createdAt);

      auto activeVersions =
          versions | std::ranges::views::filter(
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
  virtual bool
  sendRequest(MLIRContext &mlirContext, cudaq::ExecutionContext &io_context,
              const std::string &backendSimName, const std::string &kernelName,
              void (*kernelFunc)(void *), void *kernelArgs,
              std::uint64_t argsSize, std::string *optionalErrorMsg) override {
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
    constexpr std::size_t MAX_SIZE_BYTES = 250 * 1ULL << 10; // 250KB
    json requestJson;
    auto jobHeader = getHeaders();
    std::optional<std::string> assetId;
    ScopeExit deleteAssetOnExit([&]() {
      if (assetId.has_value()) {
        cudaq::info("Deleting NVCF Asset Id {}", assetId.value());
        auto headers = getHeaders();
        m_restClient.del(nvcfAssetUrl(), std::string("/") + assetId.value(),
                         headers, false);
      }
    });

    if (request.code.size() > MAX_SIZE_BYTES) {
      assetId = uploadRequest(request);
      if (!assetId.has_value()) {
        if (optionalErrorMsg)
          *optionalErrorMsg = "Failed to upload request as NVCF assets";
        return false;
      }
      json requestBody;
      requestBody["inputAssetReferences"] =
          std::vector<std::string>{assetId.value()};
      requestJson["requestBody"] = requestBody;
      requestJson["requestHeader"] = requestBody;
    } else {
      requestJson["requestBody"] = request;
    }

    try {
      cudaq::debug("Sending NVCF request to {}", nvcfInvocationUrl());
      auto resultJs = m_restClient.post(nvcfInvocationUrl(), "", requestJson,
                                        jobHeader, false);
      cudaq::debug("Response: {}", resultJs.dump());
      while (resultJs.contains("status") &&
             resultJs["status"] == "pending-evaluation") {
        const std::string reqId = resultJs["reqId"];
        cudaq::info("Polling result data for Request Id {}", reqId);
        // Wait 1 sec then poll the result
        std::this_thread::sleep_for(std::chrono::seconds(1));
        resultJs = m_restClient.get(nvcfInvocationStatus(reqId), "", jobHeader);
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
        const bool downloadOk =
            m_restClient.download(downloadUrl, resultFilePath.string());
        cudaq::info("Download zip file {}", resultFilePath.string());
        if (!downloadOk) {
          if (optionalErrorMsg)
            *optionalErrorMsg = "Failed to download large-response result.";
          return false;
        }
        // FIXME: use system "unzip" command.
        // libz has a `minizip` addon
        // (https://github.com/madler/zlib/tree/develop/contrib/minizip) which
        // could do unzipping.
        auto unzipExe = llvm::sys::findProgramByName("unzip");
        if (!unzipExe) {
          if (optionalErrorMsg)
            *optionalErrorMsg = "Unable to find 'unzip' command to unzip the "
                                "large response file. Please install 'unzip'.";
          return false;
        }
        std::filesystem::path unzipDir =
            std::filesystem::path(tempDir.c_str()) / reqId;
        std::vector<llvm::StringRef> unzipArgs = {
            unzipExe.get(), resultFilePath.c_str(), "-d", unzipDir.c_str()};
        std::string errorMsg;
        bool unzipFailed = false;
        std::optional<llvm::StringRef> redirects[] = {{""}, {""}, {""}};
        llvm::sys::ExecuteAndWait(unzipExe.get(), unzipArgs, std::nullopt,
                                  redirects, 0, 0, &errorMsg, &unzipFailed);
        if (unzipFailed) {
          if (optionalErrorMsg)
            *optionalErrorMsg = "Failed to unzip the large response zip file.";
          return false;
        }
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
            *optionalErrorMsg = "Failed to parse the response JSON from file.";
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

  // Upload IR as an NVCF asset.
  // Return asset Id
  std::optional<std::string>
  uploadRequest(const cudaq::RestRequest &jobRequest) {
    json requestJson;
    requestJson["contentType"] = "application/json";
    requestJson["description"] = "cudaq-nvcf-job";
    try {
      auto headers = getHeaders();
      auto resultJs =
          m_restClient.post(nvcfAssetUrl(), "", requestJson, headers, false);
      const std::string uploadUrl = resultJs["uploadUrl"];
      const std::string assetId = resultJs["assetId"];
      cudaq::info("Upload NVCF Asset Id {} to {}", assetId, uploadUrl);
      std::map<std::string, std::string> uploadHeader;
      // This must match the request to create the upload link
      uploadHeader["Content-Type"] = "application/json";
      uploadHeader["x-amz-meta-nvcf-asset-description"] = "cudaq-nvcf-job";
      json jobRequestJs = jobRequest;
      m_restClient.put(uploadUrl, "", jobRequestJs, uploadHeader, false);
      return assetId;
    } catch (...) {
      return {};
    }
  }
};
} // namespace

CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, RemoteRestRuntimeClient, rest)
CUDAQ_REGISTER_TYPE(cudaq::RemoteRuntimeClient, NvcfRuntimeClient, NVCF)
