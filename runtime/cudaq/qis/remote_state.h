/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "common/SimulationState.h"
#include "cudaq.h"
#include "cudaq/utils/cudaq_utils.h"

namespace cudaq {
/// Implementation of `SimulationState` for remote simulator backends.
// The state is represented by a quantum kernel.
// For accessor APIs, we may resolve the state to a state vector by executing
// the kernel on the remote simulator. For overlap API b/w 2 remote states, we
// can send both kernels to the remote backend for execution and compute the
// overlap.
class RemoteSimulationState : public cudaq::SimulationState {
protected:
  std::string kernelName;
  // Lazily-evaluated state data (just keeping the kernel name and arguments).
  // e.g., to be evaluated at amplitude accessor APIs (const APIs, hence needs
  // to be mutable) or overlap calculation with another remote state (combining
  // the IR of both states for remote evaluation)
  mutable std::unique_ptr<cudaq::SimulationState> state;
  mutable std::vector<char> argsBuffer;
  // Cache log messages from the remote execution.
  // Mutable to support lazy execution during `const` API calls.
  mutable std::string platformExecutionLog;

public:
  /// @brief Constructor
  template <typename QuantumKernel, typename... Args>
  RemoteSimulationState(QuantumKernel &&kernel, Args &&...args) {
    if constexpr (has_name<QuantumKernel>::value) {
      // kernel_builder kernel: need to JIT code to get it registered.
      static_cast<cudaq::details::kernel_builder_base &>(kernel).jitCode();
      kernelName = kernel.name();
    } else {
      kernelName = cudaq::getKernelName(kernel);
    }

    argsBuffer = cudaq::serializeArgs(std::forward<Args>(args)...);
  }
  RemoteSimulationState() = default;
  virtual ~RemoteSimulationState();
  /// @brief Triggers remote execution to resolve the state data.
  virtual void execute() const;

  /// @brief Helper to retrieve (kernel name, `args` pointer and `args` size)
  virtual std::tuple<std::string, void *, std::size_t> getKernelInfo() const;

  /// @brief Return the number of qubits this state represents.
  std::size_t getNumQubits() const override;

  /// @brief Compute the overlap of this state representation with
  /// the provided `other` state, e.g. `<this | other>`.
  std::complex<double> overlap(const cudaq::SimulationState &other) override;

  /// @brief Return the amplitude of the given computational
  /// basis state.
  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  /// @brief Return the amplitudes of the given list of computational
  /// basis states.
  std::vector<std::complex<double>>
  getAmplitudes(const std::vector<std::vector<int>> &basisState) override;

  /// @brief Return the tensor at the given index. Throws
  /// for an invalid tensor index.
  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override;

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override;

  /// @brief Return the element from the tensor at the
  /// given tensor index and at the given indices.
  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;

  /// @brief Create a new subclass specific SimulationState
  /// from the user provided data set.
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override;

  /// @brief Dump a representation of the state to the
  /// given output stream.
  void dump(std::ostream &os) const override;

  /// @brief Return the floating point precision used by the simulation state.
  precision getPrecision() const override;

  /// @brief Destroy the state representation, frees all associated memory.
  void destroyState() override;

  /// @brief Return true if this `SimulationState` wraps data on the GPU.
  bool isDeviceData() const override;

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  void toHost(std::complex<double> *clientAllocatedData,
              std::size_t numElements) const override;

  /// @brief Transfer data from device to host, return the data
  /// to the pointer provided by the client. Clients must specify the number of
  /// elements.
  void toHost(std::complex<float> *clientAllocatedData,
              std::size_t numElements) const override;

private:
  /// @brief Return the qubit count threshold where the full remote state should
  /// be flattened and returned.
  static std::size_t maxQubitCountForFullStateTransfer();
};
} // namespace cudaq
