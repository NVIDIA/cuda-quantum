/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "cutensornet.h"
#include "tensornet_utils.h"
#include "timing_utils.h"
#include <unordered_map>

#include "common/EigenDense.h"
#include "common/SimulationState.h"

namespace nvqir {

/// @brief An MPSTensor is a representation
/// of a MPS tensor and encapsulates the
/// tensor devide data and the tensor extents.
struct MPSTensor {
  void *deviceData = nullptr;
  std::vector<int64_t> extents;
};

/// @brief Wrapper of cutensornetState_t to provide convenient API's for CUDAQ
/// simulator implementation.
class TensorNetState {
public:
  std::size_t m_numQubits;
  cutensornetHandle_t m_cutnHandle;
  cutensornetState_t m_quantumState;

  /// @brief Constructor
  TensorNetState(std::size_t numQubits, cutensornetHandle_t handle);

  /// @brief Apply a unitary gate
  /// @param qubitIds Qubit operands
  /// @param gateDeviceMem Gate unitary matrix in device memory
  /// @param adjoint Apply the adjoint of gate matrix if true
  void applyGate(const std::vector<int32_t> &qubitIds, void *gateDeviceMem,
                 bool adjoint = false);

  /// @brief Apply a projector matrix (non-unitary)
  /// @param proj_d Projector matrix (expected a 2x2 matrix in column major)
  /// @param qubitIdx Qubit operand
  void applyQubitProjector(void *proj_d, int32_t qubitIdx);

  /// @brief Accessor to the underlying `cutensornetState_t`
  cutensornetState_t getInternalState() { return m_quantumState; }

  /// @brief Perform measurement sampling on the quantum state.
  std::unordered_map<std::string, size_t>
  sample(const std::vector<int32_t> &measuredBitIds, int32_t shots);

  /// @brief Contract the tensor network representation to retrieve the state
  /// vector.
  std::vector<std::complex<double>> getStateVector();

  /// @brief Compute the reduce density matrix on a set of qubits
  ///
  /// The order of the specified qubits (`cutensornet` open state modes) will be
  /// respected when computing the RDM.
  std::vector<std::complex<double>>
  computeRDM(const std::vector<int32_t> &qubits);

  /// Factorize the `cutensornetState_t` into matrix product state form.
  /// Returns MPS tensors in GPU device memory.
  /// Note: the caller assumes the ownership of these pointers, thus needs to
  /// clean them up properly (with cudaFree).
  std::vector<MPSTensor> factorizeMPS(
      int64_t maxExtent, double absCutoff, double relCutoff,
      cutensornetTensorSVDAlgo_t algo = CUTENSORNET_TENSOR_SVD_ALGO_GESVDJ);

  /// @brief  Compute the expectation value w.r.t. a
  /// `cutensornetNetworkOperator_t`
  ///
  /// The `cutensornetNetworkOperator_t` can be constructed from
  /// `cudaq::spin_op`, i.e., representing a sum of Pauli products with
  /// different coefficients.
  std::complex<double>
  computeExpVal(cutensornetNetworkOperator_t tensorNetworkOperator);

  /// @brief Number of qubits that this state represents.
  std::size_t getNumQubits() const { return m_numQubits; }

  /// @brief Destructor
  ~TensorNetState();
};

/// @brief Default TensorNet SimulationState. Defaults to
/// just extracting a state-vector representation for users.
class TensorNetSimulationState : public cudaq::SimulationState {
protected:
  /// @brief Reference to the `TensorNetState` pointer. We
  /// take ownership of it here.
  TensorNetState *state;

  /// @brief FIXME We should remove this. For now it is for backward
  /// compatibility
  std::vector<cudaq::complex128> stateData;

public:
  TensorNetSimulationState(TensorNetState *inState,
                           bool generateStateVector = true)
      : state(inState) {

    // Subtypes can specify that the state vector
    // generation is not used.
    if (generateStateVector)
      stateData = state->getStateVector();
  }

  std::size_t getNumQubits() const override { return state->getNumQubits(); }

  /// @brief Return the shape of the data.
  std::vector<std::size_t> getDataShape() const override {
    // FIXME I imagine this should be different
    return {getNumQubits()};
  }

  /// @brief Compute the overlap of this state with the provided one.
  /// If the other state is not on GPU device, this function will
  /// copy the data from host.
  double overlap(const cudaq::SimulationState &other) override {
    if (other.getDataShape() != getDataShape())
      throw std::runtime_error("[tensornet-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    if (other.isDeviceData())
      throw std::runtime_error("[tensornet-state] cannot compute "
                               "overlap with GPU state data yet.");

    return std::abs(
        Eigen::Map<Eigen::VectorXcd>(
            const_cast<cudaq::complex128 *>(stateData.data()), stateData.size())
            .transpose()
            .dot(Eigen::Map<Eigen::VectorXcd>(
                reinterpret_cast<cudaq::complex128 *>(other.ptr()),
                stateData.size()))
            .real());
  }

  /// @brief Compute the overlap of this state with the data provided as a
  /// `std::vector<double>`. If this device state is not FP64, throw an
  /// exception. This overload requires an explicit copy from host memory.
  double overlap(const std::vector<cudaq::complex128> &other) override {
    if (stateData.size() != other.size())
      throw std::runtime_error("[tensornet-state] overlap error - other state "
                               "dimension not equal to this state dimension.");
    return std::abs(
        Eigen::Map<Eigen::VectorXcd>(
            const_cast<cudaq::complex128 *>(stateData.data()), stateData.size())
            .transpose()
            .dot(Eigen::Map<Eigen::VectorXcd>(
                reinterpret_cast<cudaq::complex128 *>(
                    const_cast<cudaq::complex128 *>(other.data())),
                other.size()))
            .real());
  }

  /// @brief Compute the overlap of this state with the data provided as a
  /// `std::vector<float>`. If this device state is not FP32, throw an
  /// exception. This overload requires an explicit copy from host memory.
  double overlap(const std::vector<cudaq::complex64> &other) override {
    throw std::runtime_error(
        "[tensornet-state] requires FP64 data for overlap computation.");
  }

  /// @brief Compute the overlap of this state with the data provided as a raw
  /// pointer. This overload will check if this pointer corresponds to a device
  /// pointer. It will copy the data from host to device if necessary.
  double overlap(cudaq::complex128 *other, std::size_t numElements) override {
    if (stateData.size() != numElements)
      throw std::runtime_error("[tensornet-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    return std::abs(
        Eigen::Map<Eigen::VectorXcd>(
            const_cast<cudaq::complex128 *>(stateData.data()), stateData.size())
            .transpose()
            .dot(Eigen::Map<Eigen::VectorXcd>(other, numElements))
            .real());
  }

  double overlap(cudaq::complex64 *other, std::size_t numElements) override {
    throw std::runtime_error(
        "[tensornet-state] requires FP64 data for overlap computation.");
  }

  /// @brief Return the vector element at the given index.
  cudaq::complex128 vectorElement(std::size_t idx) override {
    return stateData[idx];
  }

  /// @brief Dump the state to the given output stream
  void dump(std::ostream &os) const override {
    os << Eigen::Map<Eigen::VectorXcd>(
              const_cast<cudaq::complex128 *>(stateData.data()),
              stateData.size())
       << "\n";
  }

  /// @brief This state is GPU device data, always return true.
  bool isDeviceData() const override { return false; }

  /// @brief Return the raw pointer to the device data.
  void *ptr() const override {
    return const_cast<cudaq::complex128 *>(stateData.data());
  }

  /// @brief Return the precision of the state data elements.
  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  /// @brief Free the device data.
  void destroyState() override { delete state; }
};
} // namespace nvqir
