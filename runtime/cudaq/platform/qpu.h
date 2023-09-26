/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "QuantumExecutionQueue.h"
#include "common/Registry.h"
#include "cudaq/qis/execution_manager.h"
#include "cudaq/qis/qubit_qis.h"
#include "cudaq/utils/cudaq_utils.h"

#include <optional>

namespace cudaq {

/// Expose the function that will return the current ExecutionManager
ExecutionManager *getExecutionManager();

/// A CUDA Quantum QPU is an abstraction on the quantum processing
/// unit which executes quantum kernel expressions. The QPU exposes
/// certain information about the QPU being targeting, such as the
/// number of available qubits, the logical ID for this QPU in a set
/// of available QPUs, and its qubit connectivity. The QPU keeps
/// track of an execution queue for enqueuing asynchronous tasks
/// that execute quantum kernel expressions. The QPU also tracks the
/// client-provided execution context to enable quantum kernel
/// related tasks such as sampling and observation.
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

  /// @brief Check if the current execution context is a `spin_op`
  /// observation and perform state-preparation circuit measurement
  /// based on the `spin_op` terms.
  void handleObservation(ExecutionContext *localContext) {
    if (localContext && localContext->name == "observe") {
      double sum = 0.0;
      if (!localContext->spin.has_value())
        throw std::runtime_error("[QPU] Observe ExecutionContext specified "
                                 "without a cudaq::spin_op.");

      std::vector<cudaq::ExecutionResult> results;
      cudaq::spin_op &H = *localContext->spin.value();

      // If the backend supports the observe task,
      // let it compute the expectation value instead of
      // manually looping over terms, applying basis change ops,
      // and computing <ZZ..ZZZ>
      if (localContext->canHandleObserve) {
        auto [exp, data] = cudaq::measure(H);
        results.emplace_back(data.to_map(), H.to_string(false), exp);
        localContext->expectationValue = exp;
        localContext->result = cudaq::sample_result(results);
      } else {

        // Loop over each term and compute coeff * <term>
        H.for_each_term([&](cudaq::spin_op &term) {
          if (term.is_identity())
            sum += term.get_coefficient().real();
          else {
            auto [exp, data] = cudaq::measure(term);
            results.emplace_back(data.to_map(), term.to_string(false), exp);
            sum += term.get_coefficient().real() * exp;
          }
        });

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

  virtual void setNoiseModel(const noise_model *model) { noiseModel = model; }

  /// Return the number of qubits
  std::size_t getNumQubits() { return numQubits; }
  /// Return the qubit connectivity
  auto getConnectivity() { return connectivity; }
  /// Is this QPU a simulator ?
  virtual bool isSimulator() { return true; }

  /// @brief Return whether this qpu has conditional feedback support
  virtual bool supportsConditionalFeedback() { return false; }

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

  /// Launch the kernel with given name (to extract its Quake representation).
  /// The raw function pointer is also provided, as are the runtime arguments,
  /// as a struct-packed void pointer and its corresponding size.
  virtual void launchKernel(const std::string &name, void (*kernelFunc)(void *),
                            void *args, std::uint64_t, std::uint64_t) = 0;
};
} // namespace cudaq
