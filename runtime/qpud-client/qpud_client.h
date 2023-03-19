/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "common/ObserveResult.h"
#include "cudaq/spin_op.h"

#include <memory>
#include <stack>

// Forward declare the RPC client type
namespace rpc {
class client;
}

namespace llvm::sys {
struct ProcessInfo;
}

namespace cudaq {

static constexpr std::size_t NoResultOffset = ~0u >> 1;

/// @brief An observe_job contains metadata describing
/// a detached cudaq"::"observe invocation. By detached, we mean
/// this job has been submitted to a remote queuing system. This type
/// describes the Job Ids for each term evaluation.
class detached_job {
private:
  std::vector<double> params;

  /// @brief A job has a name and an ID
  struct TermJob {
    TermJob(const std::string &n, const std::string &i) : name(n), id(i) {}
    std::string name;
    std::string id;
  };

  /// @brief For a SpinOp, we'll have N ansatz+measure evaluations
  /// each of these is a single job on the remote processor
  std::vector<TermJob> jobs;

public:
  /// Thin wrapper around vector"::"emplace_back
  template <typename... Args>
  void emplace_back(Args &&...args) {
    jobs.emplace_back(args...);
  }

  std::string id_from_name(const std::string &name) {
    for (auto &job : jobs) {
      if (job.name == name)
        return job.id;
    }
    throw std::runtime_error(
        "Invalid name, no job ID corresponding to that name.");
  }

  /// @brief  Thin wrapper around vector"::"operator[]
  TermJob &operator[](const std::size_t idx) { return jobs[idx]; }

  /// @brief Enable range based iteration
  auto begin() { return jobs.begin(); }

  /// @brief Enable range-based iteration
  auto end() { return jobs.end(); }

  /// @brief Return the parameters the ansatz was evaluated at
  std::vector<double> parameters() { return params; }

  /// @brief Serialize this observe_job to file, can optionally provide
  /// the parameters used to evaluate the ansatz
  void serialize(const std::string &fileName,
                 const std::vector<double> params = {});

  /// @brief Read in this observe_job from file
  void deserialize(const std::string &fileName);
};

/// Typedef the KernelArgs Creator Function
typedef std::size_t (*Creator)(void **, void **);

/// Retrieve the kernel args creator function for the kernel name
Creator getArgsCreator(const std::string &);
/// @brief Utility function for mapping variadic args to qpud required void*,
/// size_t. Note clients of this function own the allocated rawArgs.
template <typename... Args>
std::pair<void *, std::size_t> mapToRawArgs(const std::string &kernelName,
                                            Args &&...args) {
  void *rawArgs = nullptr;
  auto argsCreator = getArgsCreator(kernelName);
  void *argPointers[sizeof...(Args)] = {&args...};
  auto argsSize = argsCreator(argPointers, &rawArgs);
  return std::make_pair(rawArgs, argsSize);
}

/// @brief The QPUD Client provides a high-level API for interacting
/// with a external qpud process.
class qpud_client {
protected:
  /// Observed kernel names, stored here so we don't
  /// call loadQuakeCode more than once.
  std::vector<std::string> launchedKernels;

  // The RPC Client, which submits function invocations
  // to the remote qpud server
  std::unique_ptr<rpc::client> rpcClient;

  // Extra libraries that our qpud JIT engine will need
  std::vector<std::string> qpudJITExtraLibraries;

  /// @brief The url of the remote qpud proc
  const std::string url = "127.0.0.1";

  /// @brief the port for the remote qpud proc
  int port = 0;

  /// @brief The QPU that we are targeting on the remote qpud proc
  int qpu_id = 0;

  /// @brief Bool indicating if a stop of qpud has been requested
  bool stopRequested = false;

  /// @brief Return a raw pointer to the rpc client.
  /// @param connectClient
  /// @return
  rpc::client *getClient(bool connectClient = true);

  /// @brief Utility function for starting the qpud proc
  llvm::sys::ProcessInfo startDaemon();

  /// @brief Utility function for JIT compiling the quakeCode once
  /// @param kernelName
  void jitQuakeIfUnseen(const std::string &kernelName);

public:
  /// @brief The constructor
  qpud_client();

  /// @brief The constructor, does not create the qpud proc but
  /// instead connects to an existing one
  qpud_client(const std::string &qpudUrl, const int qpudPort);

  /// @brief Set the qpud proc target backend
  void set_backend(const std::string &backend);

  /// @brief Return true if the current backend is a simulator
  bool is_simulator();

  /// @brief Return true if the current backend supports conditional feedback
  bool supports_conditional_feedback();

  /// Execute a circuit and return the results
  void execute(const std::string &kernelName, void *runtimeArgs,
               std::uint64_t argsSize, std::uint64_t resultOffset);

  /// @brief Execute a circuit and return the results, automate the args
  /// processing
  template <typename ArgsType>
  void execute(const std::string &kernelName, ArgsType &argsTypeInstance) {
    auto [rawArgs, size, resultOff] = process_args(argsTypeInstance);
    return execute(kernelName, rawArgs, size, resultOff);
  }

  /// @brief Sample the circuit generated by the quakeCode for the given kernel
  /// name.
  sample_result sample(const std::string &kernelName, const std::size_t shots,
                       void *runtimeArgs, std::size_t argsSize);

  /// @brief Sample the circuit generated by the quakeCode for the given kernel
  /// name. Automate the args processing
  template <typename ArgsType>
  sample_result sample(const std::string &kernelName, const std::size_t shots,
                       ArgsType &argsTypeInstance) {
    auto [rawArgs, size, resultOff] = process_args(argsTypeInstance);
    return sample(kernelName, shots, rawArgs, size);
  }

  /// @brief Sample the circuit generated by the given quantum kernel
  template <typename QuantumKernel, typename... Args,
            typename R = typename std::invoke_result_t<QuantumKernel, Args...>,
            typename = std::enable_if_t<std::is_void_v<R>>>
  sample_result sample(QuantumKernel &&kernel, const std::size_t shots,
                       Args &&...args) {
    auto kernelName = cudaq::getKernelName(kernel);
    auto [rawArgs, size] = mapToRawArgs(kernelName, args...);
    return sample(kernelName, shots, rawArgs, size);
  }

  /// @brief Launch a sampling job and detach, returning a unique job id.
  detached_job sample_detach(const std::string &kernelName,
                             const std::size_t shots, void *runtimeArgs,
                             std::size_t argsSize);
  /// @brief Launch a sampling job and detach, returning a job id. Automate the
  /// args processing.
  template <typename ArgsType>
  detached_job sample_detach(const std::string &kernelName,
                             const std::size_t shots,
                             ArgsType &argsTypeInstance) {
    auto [rawArgs, size, resultOff] = process_args(argsTypeInstance);
    return sample_detach(kernelName, shots, rawArgs, size);
  }

  /// @brief Return the measure count result from a detached sample job.
  sample_result sample(detached_job &job);

  /// @brief Observe the state generated by the quakeCode at the given kernel
  /// name with respect to the given spin op.
  observe_result observe(const std::string &kernelName, cudaq::spin_op &spinOp,
                         void *runtimeArgs, std::size_t argsSize,
                         std::size_t shots = 0);

  /// @brief Observe the state generated by the quakeCode at the given kernel
  /// name with respect to the given spin op. Automate the args processing
  template <typename ArgsType>
  observe_result observe(const std::string &kernelName, cudaq::spin_op &spinOp,
                         ArgsType &argsTypeInstance) {
    auto [rawArgs, size, resultOff] = process_args(argsTypeInstance);
    return observe(kernelName, spinOp, rawArgs, size);
  }

  /// @brief Observe the state generated by the given kernel
  /// with respect to the given spin op. Automate the args processing
  template <
      typename QuantumKernel, typename... Args,
      typename = std::enable_if_t<std::is_invocable_v<QuantumKernel, Args...>>>
  observe_result observe(QuantumKernel &&kernel, cudaq::spin_op &spinOp,
                         Args &&...args) {
    auto kernelName = cudaq::getKernelName(kernel);
    auto [rawArgs, size] = mapToRawArgs(kernelName, args...);
    return observe(kernelName, spinOp, rawArgs, size);
  }

  /// @brief Observe the state generated by the given kernel
  /// with respect to the given spin op. Automate the args processing
  template <
      typename QuantumKernel, typename... Args,
      typename = std::enable_if_t<std::is_invocable_v<QuantumKernel, Args...>>>
  observe_result observe(QuantumKernel &&kernel, std::size_t shots,
                         cudaq::spin_op &spinOp, Args &&...args) {
    auto kernelName = cudaq::getKernelName(kernel);
    auto [rawArgs, size] = mapToRawArgs(kernelName, args...);
    return observe(kernelName, spinOp, rawArgs, size, shots);
  }

  /// @brief Invoke an observe task, but detach and return the job id
  detached_job observe_detach(const std::string &kernelName,
                              cudaq::spin_op &spinOp, void *runtimeArgs,
                              std::size_t argsSize, std::size_t shots = 0);

  /// @brief Invoke an observe task, but detach and return the job id.
  /// Automate the args processing
  template <typename ArgsType>
  detached_job observe_detach(const std::string &kernelName,
                              cudaq::spin_op &spinOp,
                              ArgsType &argsTypeInstance) {
    auto [rawArgs, size, resultOff] = process_args(argsTypeInstance);
    return observe_detach(kernelName, spinOp, rawArgs, size);
  }

  /// @brief Return the observe result based on a detached job
  observe_result observe(cudaq::spin_op &spinOp, detached_job &job);

  /// @brief Convert a user specified kernel argument struct to a raw void
  /// pointer and its associated size.
  template <typename ArgsType>
  std::tuple<void *, std::uint64_t, std::uint64_t>
  process_args(ArgsType &argsTypeInstance) {
    return std::make_tuple(reinterpret_cast<void *>(&argsTypeInstance),
                           sizeof(ArgsType), NoResultOffset);
  }

  /// @brief Manually stop the qpud proc. If called, one must create a new
  /// qpud_client to continue interacting with a qpud proc
  void stop_qpud();

  /// The destructor, will automatically stop the qpud proc
  ~qpud_client();
};

qpud_client &get_qpud_client();

/// @brief Launch a sampling job and detach, returning the job id. Takes the
/// kernel runtime arguments as a variadic parameter pack.
template <typename... Args>
detached_job sample_detach(const std::string &kernelName,
                           const std::size_t shots, Args &&...args) {
  auto [rawArgs, argsSize] = mapToRawArgs(kernelName, args...);
  auto &client = get_qpud_client();
  auto job = client.sample_detach(kernelName, shots, rawArgs, argsSize);
  std::free(rawArgs);
  return job;
}

/// @brief Return the sample result for the given detached job id.
/// @param job
/// @return
sample_result sample(detached_job &job) {
  auto &client = get_qpud_client();
  return client.sample(job);
}

/// @brief Observe the state generated by the kernel with given spinOp
/// asynchronously, detach and return the job id.
template <typename... Args>
detached_job observe_detach(const std::string &kernelName, spin_op &spinOp,
                            Args &&...args) {
  auto [rawArgs, argsSize] = mapToRawArgs(kernelName, args...);
  auto &client = get_qpud_client();
  auto job = client.observe_detach(kernelName, spinOp, rawArgs, argsSize);
  std::free(rawArgs);
  return job;
}

/// @brief Observe the state generated by the kernel with given spinOp
/// asynchronously, detach and return the job id. Set the number of shots
/// explicitly
template <typename... Args>
detached_job observe_detach(const std::string &kernelName, std::size_t shots,
                            spin_op &spinOp, Args &&...args) {
  auto [rawArgs, argsSize] = mapToRawArgs(kernelName, args...);
  auto &client = get_qpud_client();
  return client.observe_detach(kernelName, spinOp, rawArgs, argsSize, shots);
}

/// @brief Return the observe result for the given detached job
observe_result observe(spin_op &spinOp, detached_job &detachedJob) {
  auto &client = get_qpud_client();
  return client.observe(spinOp, detachedJob);
}
} // namespace cudaq
