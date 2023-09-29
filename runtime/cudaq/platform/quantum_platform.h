/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/ExecutionContext.h"
#include "common/NoiseModel.h"
#include "common/ObserveResult.h"
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

/// Typedefs for defining the connectivity structure of a QPU
using QubitEdge = std::pair<std::size_t, std::size_t>;
using QubitConnectivity = std::vector<QubitEdge>;

/// A sampling tasks takes no input arguments and returns
/// a sample_result instance.
using KernelExecutionTask = std::function<sample_result()>;

/// An observation tasks takes no input arguments and returns
/// a double expectation value.
using ObserveTask = std::function<observe_result()>;

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

  /// Get the number of qubits for the current QPU
  std::size_t get_num_qubits();

  /// @brief Return true if this platform exposes multiple QPUs and
  /// supports parallel distribution of quantum tasks.
  virtual bool supports_task_distribution() const { return false; }

  /// Get the number of qubits for the QPU with ID qpu_id
  std::size_t get_num_qubits(std::size_t qpu_id);

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
  void set_exec_ctx(cudaq::ExecutionContext *ctx, std::size_t qpu_id = 0);

  /// Return the current execution context
  ExecutionContext *get_exec_ctx() const { return executionContext; }

  /// Reset the execution context for this platform.
  void reset_exec_ctx(std::size_t qpu_id = 0);

  ///  Get the number of QPUs available with this platform.
  std::size_t num_qpus() const { return platformNumQPUs; }

  /// Return whether this platform is simulating the architecture.
  bool is_simulator(const std::size_t qpu_id = 0) const;

  /// @brief Return whether the qpu has conditional feedback support
  bool supports_conditional_feedback(const std::size_t qpu_id = 0) const;

  /// The name of the platform, which also corresponds to the name of the
  /// platform file.
  std::string name() const { return platformName; }

  /// Get the ID of the current QPU.
  std::size_t get_current_qpu();

  /// Set the current QPU via its device ID.
  void set_current_qpu(const std::size_t device_id);

  /// @brief Return true if the QPU is remote.
  bool is_remote(const std::size_t qpuId = 0);

  /// @brief Return true if QPU is locally emulating a remote QPU
  bool is_emulated(const std::size_t qpuId = 0) const;

  /// @brief Set the noise model for future invocations of
  /// quantum kernels.
  void set_noise(const noise_model *model);

  /// @brief Turn off any noise models.
  void reset_noise();

  /// Enqueue an asynchronous sampling task.
  std::future<sample_result> enqueueAsyncTask(const std::size_t qpu_id,
                                              KernelExecutionTask &t);

  /// @brief Enqueue a general task that runs on the specified QPU
  void enqueueAsyncTask(const std::size_t qpu_id, std::function<void()> &f);

  // This method is the hook for the kernel rewrites to invoke
  // quantum kernels.
  void launchKernel(std::string kernelName, void (*kernelFunc)(void *),
                    void *args, std::uint64_t voidStarSize,
                    std::uint64_t resultOffset);

  /// List all available platforms, which correspond to .qplt files in the
  /// platform directory.
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

protected:
  /// The Platform QPUs, populated by concrete subtypes
  std::vector<std::unique_ptr<QPU>> platformQPUs;

  /// Name of the platform.
  std::string platformName;

  /// Number of QPUs in the platform.
  std::size_t platformNumQPUs;

  /// The current QPU.
  std::size_t platformCurrentQPU = 0;

  /// @brief Store the mapping of thread ids to the QPU id
  /// that it is running in a multi-QPU context.
  std::unordered_map<std::size_t, std::size_t> threadToQpuId;

  /// Optional number of shots.
  std::optional<int> platformNumShots;

  ExecutionContext *executionContext = nullptr;
};

/// Entry point for the auto-generated kernel execution path. TODO: Needs to be
/// tied to the quantum platform instance somehow. Note that the compiler cannot
/// provide that information.
extern "C" {
void altLaunchKernel(const char *kernelName, void (*kernel)(void *), void *args,
                     std::uint64_t argsSize, std::uint64_t resultOffset);
}

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
