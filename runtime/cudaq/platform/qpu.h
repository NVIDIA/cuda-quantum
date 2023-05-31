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

  ExecutionContext *executionContext = nullptr;
  noise_model *noiseModel = nullptr;

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

  virtual void setNoiseModel(noise_model *model) { noiseModel = model; }

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
