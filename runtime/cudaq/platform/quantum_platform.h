/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/CodeGenConfig.h"
#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "common/ObserveResult.h"
#include "common/ThunkInterface.h"
#include "cudaq/remote_capabilities.h"
#include "cudaq/utils/cudaq_utils.h"
#include <cstring>
#include <cxxabi.h>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace cudaq {

class QPU;
class gradient;
class optimizer;
struct RuntimeTarget;

/// Typedefs for defining the connectivity structure of a QPU
using QubitEdge = std::pair<std::size_t, std::size_t>;
using QubitConnectivity = std::vector<QubitEdge>;

/// A sampling tasks takes no input arguments and returns
/// a sample_result instance.
using KernelExecutionTask = std::function<sample_result()>;

/// An observation tasks takes no input arguments and returns
/// a double expectation value.
using ObserveTask = std::function<observe_result()>;

/// An ID for a QPU on this platform, defaulting to this thread's current QPU.
using QpuId = std::optional<std::size_t>;

namespace detail {
/// Temporary per-thread execution context storage.
/// Will be removed when executionContext is eliminated.
struct PerThreadExecCtx {
  PerThreadExecCtx();
  ~PerThreadExecCtx();
  ExecutionContext *get() const;
  void set(ExecutionContext *ctx);

  struct Impl;
  std::unique_ptr<Impl> impl;
};
} // namespace detail

/// The quantum_platform corresponds to a specific quantum architecture.
/// The quantum_platform exposes a public API for programmers to
/// query specific information about the targeted QPU(s) (e.g. number
/// of qubits, qubit connectivity, etc.). This type is meant to
/// be subclassed for concrete realizations of quantum platforms, which
/// are intended to populate this platformQPUs member of this base class.
class quantum_platform {
public:
  quantum_platform() = default;
  virtual ~quantum_platform() = default;

  /// Fetch the connectivity info
  std::optional<QubitConnectivity> connectivity();

  /// Get the number of qubits for the QPU with ID qpu_id.
  std::size_t get_num_qubits(QpuId qpu_id = std::nullopt) const;

  /// @brief Return true if this platform exposes multiple QPUs and
  /// supports parallel distribution of quantum tasks.
  virtual bool supports_task_distribution() const { return false; }

  /// Getter for the shots. This will be deprecated once `set_shots` and
  /// `clear_shots` are removed.
  std::optional<int> get_shots() { return platformNumShots; }

  /// Setter for the shots
  [[deprecated("Specify the number of shots in the using the overloaded "
               "sample() and observe() functions")]] virtual void
  set_shots(int numShots) {
    platformNumShots = numShots;
  }

  /// Reset shots
  [[deprecated("Specify the number of shots in the using the overloaded "
               "sample() and observe() functions")]] virtual void
  clear_shots() {
    platformNumShots = std::nullopt;
  }

  /// Specify the execution context for this platform.
  void set_exec_ctx(ExecutionContext *ctx);

  /// Return the current execution context
  ExecutionContext *get_exec_ctx() const { return executionContext.get(); }

  /// Reset the execution context for this platform.
  void reset_exec_ctx();

  ///  Get the number of QPUs available with this platform.
  std::size_t num_qpus() const { return platformNumQPUs; }

  /// Return whether this platform is simulating the architecture.
  bool is_simulator(QpuId qpu_id = std::nullopt) const;

  /// @brief Return whether the QPU has conditional feedback support
  bool supports_conditional_feedback(QpuId qpu_id = std::nullopt) const;

  /// @brief Return whether the QPU supports explicit measurements.
  bool supports_explicit_measurements(QpuId qpu_id = std::nullopt) const;

  /// The name of the platform, which also corresponds to the name of the
  /// platform file.
  std::string name() const { return platformName; }

  /// Get the ID of the QPU in the current execution context.
  std::size_t get_current_qpu() const;

  /// @brief Return true if the QPU is remote.
  ///
  /// If @p qpu_id is not specified, the QPU in the current execution context is
  /// used.
  bool is_remote(QpuId qpu_id = std::nullopt) const;

  /// @brief Return true if QPU is locally emulating a remote QPU
  ///
  /// If @p qpu_id is not specified, the QPU in the current execution context is
  /// used.
  bool is_emulated(QpuId qpu_id = std::nullopt) const;

  /// @brief Set the noise model for @p qpu_id on this platform.
  ///
  /// If @p qpu_id is not specified, the QPU in the current execution context is
  /// used.
  void set_noise(const noise_model *model, QpuId qpu_id = std::nullopt);

  /// @brief Return the noise model for @p qpu_id on this platform.
  ///
  /// If @p qpu_id is not specified, the QPU in the current execution context is
  /// used.
  const noise_model *get_noise(QpuId qpu_id = std::nullopt);

  /// @brief Get the remote capabilities (only applicable for remote platforms)
  ///
  /// If @p qpu_id is not specified, the QPU in the current execution context is
  /// used.
  RemoteCapabilities get_remote_capabilities(QpuId qpuId = std::nullopt) const;

  /// Get code generation configuration values
  CodeGenConfig get_codegen_config();

  /// Get runtime target information
  // This includes information about the target configuration (config file) and
  // any other user-defined settings (nvq++ target option compile flags or
  // `set_target` arguments).
  const RuntimeTarget *get_runtime_target() const;

  /// @brief Turn off any noise models.
  ///
  /// If @p qpu_id is not specified, the QPU in the current execution context is
  /// used.
  void reset_noise(QpuId qpu_id = std::nullopt);

  /// Enqueue an asynchronous sampling task.
  std::future<sample_result> enqueueAsyncTask(const std::size_t qpu_id,
                                              KernelExecutionTask &t);

  /// @brief Enqueue a general task that runs on the specified QPU
  void enqueueAsyncTask(const std::size_t qpu_id, std::function<void()> &f);

  /// @brief Launch a VQE operation on the platform.
  void launchVQE(const std::string kernelName, const void *kernelArgs,
                 cudaq::gradient *gradient, const cudaq::spin_op &H,
                 cudaq::optimizer &optimizer, const int n_params,
                 const std::size_t shots, QpuId qpu_id = std::nullopt);

  // This method is the hook for the kernel rewrites to invoke quantum kernels.
  [[nodiscard]] KernelThunkResultType
  launchKernel(const std::string &kernelName, KernelThunkType kernelFunc,
               void *args, std::uint64_t voidStarSize,
               std::uint64_t resultOffset, const std::vector<void *> &rawArgs,
               QpuId qpu_id = std::nullopt);
  void launchKernel(const std::string &kernelName, const std::vector<void *> &,
                    QpuId qpu_id = std::nullopt);

  /// List all available platforms
  static std::vector<std::string> list_platforms();

  static std::string demangle(char const *mangled) {
    auto ptr = std::unique_ptr<char, decltype(&std::free)>{
        abi::__cxa_demangle(mangled, nullptr, nullptr, nullptr), std::free};
    return {ptr.get()};
  }

  /// @brief Set the target backend, by default do nothing, let subclasses
  /// override
  /// @param name
  virtual void setTargetBackend(const std::string &name) {}

  /// @brief Called by the runtime to notify that a new random seed value is
  /// set.
  virtual void onRandomSeedSet(std::size_t seed);

  /// @brief Turn off any custom logging stream.
  void resetLogStream();

  /// @brief Get the stream for info logging.
  // Returns null if no specific stream was set.
  std::ostream *getLogStream();

  /// @brief Set the info logging stream.
  void setLogStream(std::ostream &logStream);

protected:
  /// The runtime target settings
  std::unique_ptr<RuntimeTarget> runtimeTarget;

  /// Code generation configuration
  std::optional<CodeGenConfig> codeGenConfig;

  /// The Platform QPUs, populated by concrete subtypes
  std::vector<std::unique_ptr<QPU>> platformQPUs;

  /// Name of the platform.
  std::string platformName;

  /// Number of QPUs in the platform.
  std::size_t platformNumQPUs;

  /// Optional number of shots.
  std::optional<int> platformNumShots;

  /// Keep a per-thread pointer to the current execution context.
  // TODO: Remove this
  detail::PerThreadExecCtx executionContext;

  /// Optional logging stream for platform output.
  // If set, the platform and its QPUs will print info log to this stream.
  // Otherwise, default output stream (std::cout) will be used.
  std::ostream *platformLogStream = nullptr;

private:
  // Helper to validate QPU Id
  void validateQpuId(int qpuId) const;

  /// Resolve @p qpu_id to a QPU ID for this platform.
  ///
  /// 1. If @p qpu_id is specified, return it.
  /// 2. If no @p qpu_id is specified but @p ctx is provided, return the QPU ID
  /// of @p ctx.
  /// 3. If no @p qpu_id is specified and no @p ctx is provided, return the
  ///    QPU ID of the current execution context.
  ///
  /// This always returns a valid QPU ID.
  std::size_t resolveQpuId(QpuId qpu_id, ExecutionContext *ctx = nullptr) const;
};

/// Entry point for the auto-generated kernel execution path. TODO: Needs to be
/// tied to the quantum platform instance somehow. Note that the compiler cannot
/// provide that information.
extern "C" {
// Client-server (legacy) interface.
[[nodiscard]] KernelThunkResultType
altLaunchKernel(const char *kernelName, KernelThunkType kernel, void *args,
                std::uint64_t argsSize, std::uint64_t resultOffset);

// Streamlined interface for launching kernels. Argument synthesis and JIT
// compilation *must* happen on the local machine.
[[nodiscard]] KernelThunkResultType
streamlinedLaunchKernel(const char *kernelName,
                        const std::vector<void *> &rawArgs);

// Hybrid of the client-server and streamlined approaches. Letting JIT
// compilation happen either early or late and can handle return values from
// each kernel launch.
[[nodiscard]] KernelThunkResultType
hybridLaunchKernel(const char *kernelName, KernelThunkType kernel, void *args,
                   std::uint64_t argsSize, std::uint64_t resultOffset,
                   const std::vector<void *> &rawArgs);
} // extern "C"
} // namespace cudaq

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define CUDAQ_REGISTER_PLATFORM(NAME, PRINTED_NAME)                            \
  extern "C" {                                                                 \
  cudaq::quantum_platform *getQuantumPlatform() {                              \
    thread_local static std::unique_ptr<cudaq::quantum_platform> m_platform =  \
        std::make_unique<NAME>();                                              \
    return m_platform.get();                                                   \
  }                                                                            \
  cudaq::quantum_platform *CONCAT(getQuantumPlatform_, PRINTED_NAME)() {       \
    thread_local static std::unique_ptr<cudaq::quantum_platform> m_platform =  \
        std::make_unique<NAME>();                                              \
    return m_platform.get();                                                   \
  }                                                                            \
  }
