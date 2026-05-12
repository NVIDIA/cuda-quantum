/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "QuantumExecutionQueue.h"
#include "common/CompiledModule.h"
#include "common/KernelArgs.h"
#include "common/Registry.h"
#include "common/ThunkInterface.h"
#include "cudaq/remote_capabilities.h"

namespace mlir {
class Type;
} // namespace mlir

namespace cudaq {
class gradient;
class optimizer;
class noise_model;
class ExecutionContext;

// forward declare the spin_op type
template <typename T>
class sum_op;
class spin_handler;
typedef sum_op<spin_handler> spin_op;

/// A CUDA-Q QPU is an abstraction on the quantum processing unit which executes
/// quantum kernel expressions. The QPU exposes certain information about the
/// QPU being targeting, such as the number of available qubits, the logical ID
/// for this QPU in a set of available QPUs, and its qubit connectivity. The QPU
/// keeps track of an execution queue for enqueuing asynchronous tasks that
/// execute quantum kernel expressions. The QPU also tracks the client-provided
/// execution context to enable quantum kernel related tasks such as sampling
/// and observation.
///
/// This type is meant to be subtyped by concrete quantum_platform subtypes.
class QPU : public registry::RegisteredType<QPU> {
protected:
  /// The logical id of this QPU in the platform set of QPUs
  std::size_t qpu_id = 0;
  std::size_t numQubits = 30;
  std::optional<std::vector<std::pair<std::size_t, std::size_t>>> connectivity;
  std::unique_ptr<QuantumExecutionQueue> execution_queue;

  /// @brief Noise model specified for QPU execution.
  const noise_model *noiseModel = nullptr;

  /// @brief Check if the current execution context is a `spin_op` observation
  /// and perform state-preparation circuit measurement based on the `spin_op`
  /// terms.
  void handleObservation(ExecutionContext &context) const;

  [[nodiscard]] static KernelThunkResultType
  runJITCompiledModule(const CompiledModule &compiled, KernelArgs args);

public:
  /// The constructor, initializes the execution queue
  QPU() : execution_queue(std::make_unique<QuantumExecutionQueue>()) {}
  /// The constructor, sets the current QPU Id and initializes the execution
  /// queue
  QPU(std::size_t _qpuId)
      : qpu_id(_qpuId),
        execution_queue(std::make_unique<QuantumExecutionQueue>()) {}
  /// Move constructor
  QPU(QPU &&) = default;
  /// The destructor
  virtual ~QPU() = default;
  /// Set the current QPU Id
  void setId(std::size_t _qpuId) { qpu_id = _qpuId; }

  /// Get id of the thread this QPU's queue executes on.
  // If no execution_queue has been constructed, returns a 'null' id (does not
  // represent a thread of execution).
  std::thread::id getExecutionThreadId() const {
    return execution_queue ? execution_queue->getExecutionThreadId()
                           : std::thread::id();
  }

  virtual void setNoiseModel(const noise_model *model) { noiseModel = model; }
  virtual const noise_model *getNoiseModel() { return noiseModel; }

  /// Return the number of qubits
  std::size_t getNumQubits() { return numQubits; }
  /// Return the qubit connectivity
  auto getConnectivity() { return connectivity; }
  /// Is this QPU a simulator ?
  virtual bool isSimulator() { return true; }

  /// @brief Return whether this QPU supports explicit measurements
  virtual bool supportsExplicitMeasurements() { return true; }

  /// @brief Return the remote capabilities for this platform.
  virtual RemoteCapabilities getRemoteCapabilities() const {
    return RemoteCapabilities(/*initValues=*/false);
  }

  /// Base class handling of shots is do-nothing,
  /// subclasses can handle as they wish
  virtual void setShots(int _nShots) {}
  virtual void clearShots() {}

  virtual bool isRemote() { return false; }

  /// Is this a local emulator of a remote QPU?
  virtual bool isEmulated() { return false; }

  /// Enqueue a quantum task on the asynchronous execution queue.
  virtual void
  enqueue(QuantumTask &task) = 0; //{ execution_queue->enqueue(task); }

  /// @brief Configure the execution context for this QPU.
  virtual void configureExecutionContext(ExecutionContext &context) const {}

  /// @brief Post-process the execution results stored in @p context for this
  /// QPU.
  virtual void finalizeExecutionContext(ExecutionContext &context) const {}

  /// @brief Prepare the QPU for a new execution.
  ///
  /// This is called after the execution context has been configured and is
  /// already set.
  virtual void beginExecution() {}

  /// @brief Clean up after an execution on this QPU.
  ///
  /// This is called after the execution context has been finalized and before
  /// the execution context is reset.
  virtual void endExecution() {}

  virtual void setTargetBackend(const std::string &backend) {}

  virtual void launchVQE(const std::string &name, const void *kernelArgs,
                         cudaq::gradient *gradient, const cudaq::spin_op &H,
                         cudaq::optimizer &optimizer, const int n_params,
                         const std::size_t shots) {}

  [[nodiscard]] virtual KernelThunkResultType
  unifiedLaunchModule(const AnyModule &module, KernelArgs args);

  [[nodiscard]] virtual CompiledModule
  compileModule(const SourceModule &src, KernelArgs args, bool isEntryPoint);

  /// @brief Notify the QPU that a new random seed value is set.
  /// By default do nothing, let subclasses override.
  virtual void onRandomSeedSet(std::size_t seed) {}
};

struct ModuleLauncher : public registry::RegisteredType<ModuleLauncher> {
  virtual ~ModuleLauncher() = default;

  /// Compile (specialize + JIT) a kernel module and return a ready-to-execute
  /// CompiledModule.
  virtual CompiledModule compileModule(const SourceModule &src, KernelArgs args,
                                       bool isEntryPoint) = 0;
};

} // namespace cudaq
