/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#ifdef __GNUC__
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "qpud_client.h"
#include "nlohmann/json.hpp"
#include "rpc/client.h"
#include "rpc/rpc_error.h"
#include "llvm/Support/Program.h"

#include <fmt/core.h>
#include <fstream>
#include <iostream>
#include <random>

#if defined(__APPLE__) && defined(__MACH__)
#include <mach-o/dyld.h>
#else
#include <link.h>
#endif

namespace cudaq {
std::string get_quake_by_name(const std::string &);
static std::unique_ptr<qpud_client> qpudClient = nullptr;

qpud_client &get_qpud_client() {
  if (!qpudClient)
    qpudClient = std::make_unique<qpud_client>();
  return *qpudClient.get();
}

/// @brief Simple struct for extracting the library
/// names this executable is linked to
struct NVQIRLibraryData {
  std::string path;
};

// We need to get the NVQIR Backend Library that we are linked to
// so we can give that information to the qpud process.
#if defined(__APPLE__) && defined(__MACH__)
// https://stackoverflow.com/questions/10009043/dl-iterate-phdr-equivalent-on-mac
static void getNVQIRLibraryPath(NVQIRLibraryData *data) {
  auto nLibs = _dyld_image_count();
  for (uint32_t i = 0; i < nLibs; i++) {
    auto ptr = _dyld_get_image_name(i);
    std::string libName(ptr);
    if (libName.find("nvqir-") != std::string::npos) {
      auto casted = static_cast<NVQIRLibraryData *>(data);
      casted->path = std::string(ptr);
    }
  }
}
#else
/// @brief Extract the NVQIR backend library path
static int getNVQIRLibraryPath(struct dl_phdr_info *info, size_t size,
                               void *data) {
  std::string libraryName(info->dlpi_name);
  if (libraryName.find("nvqir-") != std::string::npos) {
    auto casted = static_cast<NVQIRLibraryData *>(data);
    casted->path = std::string(info->dlpi_name);
  }
  return 0;
}
#endif

/// @brief Pointer to captured runtime arguments
std::unique_ptr<std::vector<std::string>> capturedArgs;
/// @brief Capture any user-provided runtime args so we can forward to QPUD
/// @param argc
/// @param argv
/// @param
void captureHostCommandLineArgs(int argc, char **argv, char **) {
  capturedArgs =
      std::make_unique<std::vector<std::string>>(argv + 1, argv + argc);
}

#if defined(__APPLE__) && defined(__MACH__)
#define INIT_ARRAY section("__DATA, __mod_init_func")
#else
#define INIT_ARRAY section(".init_array")
#endif

[[maybe_unused]] __attribute__((INIT_ARRAY)) typeof(captureHostCommandLineArgs)
    *__captureHostCommandLineArgs = captureHostCommandLineArgs;

#undef INIT_ARRAY

/// @brief Invoke the call, catch any exceptions, stop the server if we hit an
/// error
template <typename... Args>
auto invokeCall(rpc::client *client, const std::string &functionName,
                Args &...args) {
  try {
    // Load the quake code to the QPU
    return client->call(functionName, args...);
  } catch (rpc::rpc_error &e) {
    client->call("stopServer");
    std::string msg = "[qpud::" + e.get_function_name() + "] " +
                      e.get_error().as<std::string>();
    throw std::runtime_error(msg);
  }
}

llvm::sys::ProcessInfo qpud_client::startDaemon() {
  // We need to know what NVQIR backend we were compiled
  // with. Here we loop over all linked libraries to get the nvqir backend
  // library
  NVQIRLibraryData data;
#if defined(__APPLE__) && defined(__MACH__)
  getNVQIRLibraryPath(&data);
#else
  dl_iterate_phdr(getNVQIRLibraryPath, &data);
#endif
  qpudJITExtraLibraries.push_back(data.path);

  std::random_device rd;
  std::mt19937 mt(rd());
  std::uniform_int_distribution<int> dist(10000, 65534);
  port = dist(mt);

  std::filesystem::path nvqirPath{data.path};
  auto installLibPath = nvqirPath.parent_path();
  auto installPath = installLibPath.parent_path();
  auto qpudExePath = installPath / "bin" / "qpud";

  std::string qpudError = "";

  std::vector<llvm::StringRef> qpudArgs{qpudExePath.string(), "--port",
                                        std::to_string(port), "--qpu",
                                        std::to_string(qpu_id)};
  // Forward the captured args.
  if (capturedArgs)
    for (auto &arg : *capturedArgs) {
      qpudArgs.push_back(arg);
    }

  bool execFailed = false;
  auto qpudProcInfo =
      llvm::sys::ExecuteNoWait(qpudExePath.string(), qpudArgs, std::nullopt, {},
                               0, &qpudError, &execFailed);
  if (execFailed) {
    std::cerr << "Failed to launch qpud process on port " << port << ":\n"
              << qpudError << "\n";
    throw std::runtime_error("Could not create qpud process.");
  }

  return qpudProcInfo;
}

rpc::client *qpud_client::getClient(bool startServer) {
  if (!rpcClient && startServer) {
    auto qpudProcInfo = startDaemon();
    // Since the client is starting the server and connecting to it.  We might
    // run into the problem that when we try to connect to it, the server didn't
    // had enough time to initialize. So here, we try to connect 10 times with
    // a waiting time between each try.
    rpc::client::connection_state state;
    for (auto i = 0; i < 10; ++i) {
      rpcClient = std::make_unique<rpc::client>(url, port);
      state = rpcClient->get_connection_state();
      // Upon construction, the client is at state `initial` and tries to
      // connect to the server using an asynchronous call.  This is basically
      // a spin lock to wait for the return of this call, which should be
      // either a `connected` or `disconnected` state.
      while (state == rpc::client::connection_state::initial) {
        state = rpcClient->get_connection_state();
      };
      if (state == rpc::client::connection_state::connected)
        break;
      std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    if (state == rpc::client::connection_state::disconnected) {
      if (qpudProcInfo.Pid) {
        // If the QPU daemon stated, but we were not able to connect to it, we
        // kill it. (This call needs a SecondsToWait > 0 to kill the process)
        llvm::sys::Wait(qpudProcInfo, /*SecondsToWait=*/1, nullptr, nullptr);
      }
      throw std::runtime_error(fmt::format(
          "Could not connect to remote qpud process at {}:{}", url, port));
    }
  }

  return rpcClient.get();
}

void qpud_client::jitQuakeIfUnseen(const std::string &kernelName) {
  rpc::client *client = getClient();
  // Upload the kernel mlir code if we haven't already.
  if (std::find(launchedKernels.begin(), launchedKernels.end(), kernelName) ==
      std::end(launchedKernels)) {
    auto quakeCode = cudaq::get_quake_by_name(kernelName);
    invokeCall(client, "loadQuakeCode", kernelName, quakeCode,
               qpudJITExtraLibraries);
    launchedKernels.push_back(kernelName);
  }
}

/// By default, let startDaemon create the qpud proc
/// This will also allocate a random port
qpud_client::qpud_client() {}

void qpud_client::set_backend(const std::string &backend) {
  // Get the client, if not set, create it
  rpc::client *client = getClient();

  // invoke the setTargetBackend function
  invokeCall(client, "setTargetBackend", backend);
}

bool qpud_client::is_simulator() {
  // Get the client, if not set, create it
  rpc::client *client = getClient();
  return invokeCall(client, "getIsSimulator").as<bool>();
}

bool qpud_client::supports_conditional_feedback() {
  rpc::client *client = getClient();
  return invokeCall(client, "getSupportsConditionalFeedback").as<bool>();
}

void qpud_client::execute(const std::string &kernelName, void *runtimeArgs,
                          std::uint64_t argsSize, std::uint64_t resultOff) {
  rpc::client *client = getClient();

  // Tell the QPUD to JIT compile the code
  jitQuakeIfUnseen(kernelName);

  // Map the runtime args to a vector<uint8_t>
  uint8_t *buf = (uint8_t *)runtimeArgs;
  std::vector<uint8_t> vec_buf(buf, buf + argsSize);

  // No context has been set, just calling base execute
  auto updatedArgs = invokeCall(client, "executeKernel", kernelName, vec_buf)
                         .as<std::vector<uint8_t>>();

  if (updatedArgs.size() > argsSize) {
    assert(resultOff != NoResultOffset && "result offset must be given");
    // The return buffer is longer than the argument buffer, therefore the
    // return buffer includes dynamically allocated result values. Return these
    // in a new heap allocated buffer.
    const std::uint64_t dynResSize = updatedArgs.size() - argsSize;
    char *resBuff = reinterpret_cast<char *>(std::malloc(dynResSize));
    std::memcpy(resBuff, &updatedArgs[argsSize], dynResSize);
    // Update the pointer to the new buffer in updatedArgs before copying it.
    void **resultPtr = reinterpret_cast<void **>(&updatedArgs[resultOff]);
    *resultPtr = resBuff;
    assert(dynResSize == *(reinterpret_cast<uint64_t *>(
                             &updatedArgs[resultOff + sizeof(void *)])));
  }

  // If this function has a return type, it has been
  // set as part of the void* args, set it here.
  std::memcpy(runtimeArgs, updatedArgs.data(), argsSize);
}

sample_result qpud_client::sample(const std::string &kernelName,
                                  const std::size_t shots, void *runtimeArgs,
                                  std::size_t argsSize) {
  using ResultType = std::vector<std::size_t>;
  rpc::client *client = getClient();

  // Tell the QPUD to JIT compile the code
  jitQuakeIfUnseen(kernelName);

  // Map the runtime args to a vector<uint8_t>
  uint8_t *buf = (uint8_t *)runtimeArgs;
  std::vector<uint8_t> vec_buf(buf, buf + argsSize);

  // Tell teh QPUD to sample
  auto countsData =
      invokeCall(client, "sampleKernel", kernelName, shots, vec_buf)
          .as<ResultType>();

  // Deserialize the result and return
  sample_result counts;
  counts.deserialize(countsData);
  return counts;
}

detached_job qpud_client::sample_detach(const std::string &kernelName,
                                        const std::size_t shots,
                                        void *runtimeArgs,
                                        std::size_t argsSize) {
  using ResultType = std::tuple<std::string, std::string>;

  rpc::client *client = getClient();

  // Tell the QPUD to JIT compile the code
  jitQuakeIfUnseen(kernelName);

  // Map the runtime args to a vector<uint8_t>
  uint8_t *buf = (uint8_t *)runtimeArgs;
  std::vector<uint8_t> vec_buf(buf, buf + argsSize);

  using ResultType = std::tuple<std::string, std::string>;
  // Invoke the sample function and detach
  auto retJob =
      invokeCall(client, "sampleKernelDetach", kernelName, shots, vec_buf)
          .as<ResultType>();

  detached_job job;
  auto id = std::get<0>(retJob);
  auto name = std::get<1>(retJob);
  job.emplace_back(name, id);
  return job;
}

sample_result qpud_client::sample(detached_job &job) {
  using ResultType = std::vector<std::size_t>;
  rpc::client *client = getClient();
  auto countsData =
      invokeCall(client, "sampleKernelFromJobId", job[0].id).as<ResultType>();
  sample_result counts;
  counts.deserialize(countsData);
  return counts;
}

observe_result qpud_client::observe(const std::string &kernelName,
                                    cudaq::spin_op &spinOp, void *runtimeArgs,
                                    std::size_t argsSize, std::size_t shots) {
  using ResultType = std::tuple<double, std::vector<std::size_t>>;
  rpc::client *client = getClient();

  // Tell the QPUD to JIT compile the code
  jitQuakeIfUnseen(kernelName);

  // Serialize the spin op
  std::vector<double> H_data = spinOp.getDataRepresentation();

  // Map the runtime args to a vector<uint8_t>
  uint8_t *buf = (uint8_t *)runtimeArgs;
  std::vector<uint8_t> vec_buf(buf, buf + argsSize);

  // Invoke the observation function
  auto result =
      invokeCall(client, "observeKernel", kernelName, H_data, shots, vec_buf)
          .as<ResultType>();
  // Handle counts
  sample_result data;
  data.deserialize(std::get<1>(result));
  return observe_result(std::get<0>(result), spinOp, data);
}

detached_job qpud_client::observe_detach(const std::string &kernelName,
                                         cudaq::spin_op &spinOp,
                                         void *runtimeArgs,
                                         std::size_t argsSize,
                                         std::size_t shots) {
  using ResultType =
      std::tuple<std::vector<std::string>, std::vector<std::string>>;

  rpc::client *client = getClient();

  // Tell the QPUD to JIT compile the code
  jitQuakeIfUnseen(kernelName);

  // Serialize the spin op
  std::vector<double> H_data = spinOp.getDataRepresentation();

  // Map the runtime args to a vector<uint8_t>
  uint8_t *buf = (uint8_t *)runtimeArgs;
  std::vector<uint8_t> vec_buf(buf, buf + argsSize);

  // Invoke the observation function and detach
  auto retJobs = invokeCall(client, "observeKernelDetach", kernelName, H_data,
                            shots, vec_buf)
                     .as<ResultType>();

  detached_job jobs;
  auto ids = std::get<0>(retJobs);
  for (auto [i, id] : cudaq::enumerate(ids)) {
    jobs.emplace_back(std::get<1>(retJobs)[i], id);
  }
  return jobs;
}

observe_result qpud_client::observe(cudaq::spin_op &spinOp, detached_job &job) {
  using ResultType = std::tuple<double, std::vector<std::size_t>>;
  rpc::client *client = getClient();
  int counter = 0;
  double sum = 0.0;
  sample_result global;
  std::vector<ExecutionResult> sampleResults;
  for (std::size_t i = 0; i < spinOp.n_terms(); i++) {
    auto term = spinOp[i];
    auto realCoeff = term.get_term_coefficient(0).real();
    if (term.is_identity())
      sum += realCoeff;
    else {
      auto result =
          invokeCall(client, "observeKernelFromJobId", job[counter].id)
              .as<ResultType>();
      // Handle counts
      sample_result m;
      m.deserialize(std::get<1>(result));

      sampleResults.emplace_back(m.to_map(), term.to_string(false));

      sum += realCoeff * std::get<0>(result);
      counter++;
    }
  }
  sample_result m(sampleResults);
  return observe_result(sum, spinOp, m);
}

void qpud_client::stop_qpud() {
  if (!stopRequested &&
      (rpcClient && rpcClient->get_connection_state() ==
                        rpc::client::connection_state::connected)) {
    stopRequested = true;
    rpcClient->call("stopServer");
  }
}

qpud_client::~qpud_client() {
  // Upon destruction, stop the server
  stop_qpud();
}

void detached_job::serialize(const std::string &fileName,
                             const std::vector<double> params) {
  std::ofstream out(fileName);
  nlohmann::ordered_json j;
  if (!params.empty())
    j["parameters"] = params;

  for (auto &job : *this) {
    j[job.name] = job.id;
  }
  out << j.dump(4);
  out.close();
}

void detached_job::deserialize(const std::string &fileName) {
  std::ifstream i(fileName);
  nlohmann::ordered_json j;
  i >> j;

  for (auto &element : j.items()) {
    if (element.key() == "parameters")
      params = element.value().get<std::vector<double>>();
    else
      emplace_back(element.key(), element.value().get<std::string>());
  }
}
} // namespace cudaq
