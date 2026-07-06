/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuStateVecState.h"
#include "common/SimulationState.h"

#include <complex>
#include <optional>
#include <vector>

namespace cudaq::cusv {
/// @brief Adapts an internal `CuStateVecState` to `cudaq::SimulationState`.
///
/// The adapter takes ownership of a completed simulator state and provides the
/// user-facing amplitude, overlap, tensor, precision, and host-transfer APIs.
/// It normalizes the `cuStateVecEx` wire ordering before exposing contiguous
/// state-vector data.
template <typename Scalar>
class CuStateVecSimulationState : public cudaq::SimulationState {
public:
  explicit CuStateVecSimulationState(CuStateVecState<Scalar> &&state)
      : m_state(std::move(state)) {}

  ~CuStateVecSimulationState() override { destroyState(); }

  /// Create a state from host- or device-resident amplitudes. A zero-qubit
  /// state owns its sole amplitude in a standalone device allocation because
  /// cuStateVecEx cannot represent a zero-wire state vector.
  static std::unique_ptr<CuStateVecSimulationState>
  create(std::size_t size, const void *data, bool allowFp32Emulation);

  /// Return the number of amplitudes in the global state vector.
  std::size_t getNumElements() const override {
    if (m_state)
      return std::size_t{1} << m_state->numWires();
    return m_scalarDevicePtr ? 1 : 0;
  }

  /// Return the number of qubits represented by this state.
  std::size_t getNumQubits() const override;

  /// Compute the device-resident inner product with a state of the same
  /// precision and storage layout. Distributed partial results are all-reduced.
  std::complex<double> overlap(const SimulationState &other) override;

  /// Return the amplitude of a computational-basis state. For a distributed
  /// state, the owning rank broadcasts the amplitude to all ranks.
  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  /// Write the complete state vector to the given output stream.
  void dump(std::ostream &stream) const override;

  /// Return the precision of the state-vector amplitudes.
  precision getPrecision() const override;

  /// The live state is backed by GPU-accessible cuStateVecEx resources.
  bool isDeviceData() const override { return true; }
  bool isArrayLike() const override {
    return m_scalarDevicePtr ||
           (m_state &&
            m_state->distributionType() ==
                CUSTATEVEC_EX_SV_DISTRIBUTION_SINGLE_DEVICE &&
            m_state->numMigrationWires() == 0);
  }
  /// Release the owned cuStateVecEx descriptor and cached host data.
  void destroyState() override;

  /// Return the sole device tensor. Migrated states cannot expose one
  /// contiguous device tensor and therefore reject this operation.
  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  /// Return all tensors representing this state.
  std::vector<Tensor> getTensors() const override;

  /// A state vector is represented as one logical tensor.
  std::size_t getNumTensors() const override { return 1; }

  /// Return an amplitude by logical tensor index.
  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;

  /// Copy into a same-precision host buffer. For a distributed state, a
  /// rank-local-sized buffer receives only this rank's sub-states, while a
  /// global-sized buffer gathers and orders all sub-states.
  void toHost(std::complex<double> *data,
              std::size_t numElements) const override;
  void toHost(std::complex<float> *data,
              std::size_t numElements) const override;

  /// Return the internal state-vector owner.
  const CuStateVecState<Scalar> &state() const;

protected:
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *data,
                       std::size_t dataType) override;

private:
  explicit CuStateVecSimulationState(void *scalarDevicePtr)
      : m_scalarDevicePtr(scalarDevicePtr) {}

  std::size_t numElements() const;
  std::size_t physicalIndex(std::size_t logicalIndex) const;
  std::complex<double> amplitudeAt(std::size_t logicalIndex);
  std::vector<std::complex<Scalar>> localState() const;
  template <typename HostScalar>
  void copyToHost(std::complex<HostScalar> *data,
                  std::size_t numElementsRequested) const;
  void normalizeWireOrdering() const;

  mutable std::optional<CuStateVecState<Scalar>> m_state;
  void *m_scalarDevicePtr = nullptr;
};

extern template class CuStateVecSimulationState<float>;
extern template class CuStateVecSimulationState<double>;

} // namespace cudaq::cusv
