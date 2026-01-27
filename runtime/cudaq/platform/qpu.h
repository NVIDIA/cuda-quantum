/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "QuantumExecutionQueue.h"
#include "common/Logger.h"
#include "common/Registry.h"
#include "common/ThunkInterface.h"
#include "common/Timing.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/remote_capabilities.h"
#include "cudaq/utils/cudaq_utils.h"
#include "mlir/IR/BuiltinOps.h"

namespace cudaq {
class gradient;
class optimizer;

/// Expose the function that will return the current ExecutionManager
ExecutionManager *getExecutionManager();

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

  /// @brief The current execution context.
  ExecutionContext *executionContext = nullptr;

  /// @brief Noise model specified for QPU execution.
  const noise_model *noiseModel = nullptr;

  /// @brief Check if the current execution context is a `spin_op` observation
  /// and perform state-preparation circuit measurement based on the `spin_op`
  /// terms.
  void handleObservation(ExecutionContext *localContext) {
    // The reason for the 2 if checks is simply to do a flushGateQueue() before
    // initiating the trace.
    bool execute = localContext && localContext->name == "observe";
    if (execute) {
      ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                             "handleObservation flushGateQueue()");
      getExecutionManager()->flushGateQueue();
    }
    if (execute) {
      ScopedTraceWithContext(cudaq::TIMING_OBSERVE,
                             "QPU::handleObservation (after flush)");
      double sum = 0.0;
      if (!localContext->spin.has_value())
        throw std::runtime_error("[QPU] Observe ExecutionContext specified "
                                 "without a cudaq::spin_op.");

      std::vector<cudaq::ExecutionResult> results;
      cudaq::spin_op &H = localContext->spin.value();
      assert(cudaq::spin_op::canonicalize(H) == H);

      // If the backend supports the observe task, let it compute the
      // expectation value instead of manually looping over terms, applying
      // basis change ops, and computing <ZZ..ZZZ>
      if (localContext->canHandleObserve) {
        auto [exp, data] = cudaq::measure(H);
        localContext->expectationValue = exp;
        localContext->result = data;
      } else {

        // Loop over each term and compute coeff * <term>
        for (const auto &term : H) {
          if (term.is_identity())
            sum += term.evaluate_coefficient().real();
          else {
            // This takes a longer time for the first iteration unless
            // flushGateQueue() is called above.
            auto [exp, data] = cudaq::measure(term);
            results.emplace_back(data.to_map(), term.get_term_id(), exp);
            sum += term.evaluate_coefficient().real() * exp;
          }
        };

        localContext->expectationValue = sum;
        localContext->result = cudaq::sample_result(sum, results);
      }
    }
  }

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

  /// @brief Return whether this QPU has conditional feedback support
  virtual bool supportsConditionalFeedback() { return false; }

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

  /// Set the execution context, meant for subtype specification
  virtual void setExecutionContext(ExecutionContext *context) = 0;
  /// Reset the execution context, meant for subtype specification
  virtual void resetExecutionContext() = 0;
  virtual void setTargetBackend(const std::string &backend) {}

  virtual void launchVQE(const std::string &name, const void *kernelArgs,
                         cudaq::gradient *gradient, const cudaq::spin_op &H,
                         cudaq::optimizer &optimizer, const int n_params,
                         const std::size_t shots) {}

  /// Launch the kernel with given name (to extract its Quake representation).
  /// The raw function pointer is also provided, as are the runtime arguments,
  /// as a struct-packed void pointer and its corresponding size.
  [[nodiscard]] virtual KernelThunkResultType
  launchKernel(const std::string &name, KernelThunkType kernelFunc, void *args,
               std::uint64_t, std::uint64_t,
               const std::vector<void *> &rawArgs) = 0;

  /// Launch the kernel with given name and argument arrays.
  // This is intended for any QPUs whereby we need to JIT-compile the kernel
  // with argument synthesis. The QPU implementation must override this.
  virtual void launchKernel(const std::string &name,
                            const std::vector<void *> &rawArgs) {
    if (!isRemote())
      throw std::runtime_error("Wrong kernel launch point: Attempt to launch "
                               "kernel in streamlined for JIT mode on local "
                               "simulated QPU. This is not supported.");
  }

  [[nodiscard]] virtual KernelThunkResultType
  launchModule(const std::string &name, mlir::ModuleOp module,
               const std::vector<void *> &rawArgs, mlir::Type resultTy);

  [[nodiscard]] virtual void *
  specializeModule(const std::string &name, mlir::ModuleOp module,
                   const std::vector<void *> &rawArgs, mlir::Type resultTy,
                   void *cachedEngine);

  /// @brief Notify the QPU that a new random seed value is set.
  /// By default do nothing, let subclasses override.
  virtual void onRandomSeedSet(std::size_t seed) {}
};

struct ModuleLauncher : public registry::RegisteredType<ModuleLauncher> {
  virtual ~ModuleLauncher() = default;

  virtual KernelThunkResultType launchModule(const std::string &name,
                                             mlir::ModuleOp module,
                                             const std::vector<void *> &rawArgs,
                                             mlir::Type resultTy) = 0;
  virtual void *specializeModule(const std::string &name, mlir::ModuleOp module,
                                 const std::vector<void *> &rawArgs,
                                 mlir::Type resultTy, void *cachedEngine) = 0;
};

} // namespace cudaq
