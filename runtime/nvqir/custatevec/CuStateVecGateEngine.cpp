/*******************************************************************************
 * Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuStateVecGateEngine.h"

#include "CuStateVecError.h"

#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {

using cudaq::cusv::dataOrNull;

template <typename Scalar>
void enqueueNoise(custatevecExSVUpdaterDescriptor_t updater,
                  const cudaq::cusv::NoiseTask<Scalar> &task) {
  if (task.matrices.empty())
    throw std::invalid_argument("A noise channel must contain a matrix.");
  if (task.matrixTypes.size() != task.matrices.size())
    throw std::invalid_argument(
        "A noise channel must provide one type per matrix.");

  std::vector<const void *> matrices;
  matrices.reserve(task.matrices.size());
  for (const auto &matrix : task.matrices)
    matrices.push_back(matrix.data());

  if (task.kind == cudaq::cusv::NoiseChannelKind::MixedUnitary) {
    if (task.probabilities.size() != task.matrices.size())
      throw std::invalid_argument(
          "A mixed-unitary channel must provide one probability per matrix.");
    HANDLE_CUSTATEVEC_ERROR(custatevecExSVUpdaterEnqueueUnitaryChannel(
        /*svUpdater=*/updater, /*unitaries=*/matrices.data(),
        /*unitariesDataType=*/cudaq::cusv::complexDataType<double>(),
        /*exMatrixTypes=*/task.matrixTypes.data(),
        /*numUnitaries=*/static_cast<int32_t>(matrices.size()),
        /*layout=*/CUSTATEVEC_MATRIX_LAYOUT_ROW,
        /*probabilities=*/task.probabilities.data(),
        /*channelWires=*/dataOrNull(task.wires),
        /*numChannelWires=*/static_cast<int32_t>(task.wires.size())));
    return;
  }

  HANDLE_CUSTATEVEC_ERROR(custatevecExSVUpdaterEnqueueGeneralChannel(
      /*svUpdater=*/updater, /*matrices=*/matrices.data(),
      /*matrixDataType=*/cudaq::cusv::complexDataType<double>(),
      /*exMatrixTypes=*/task.matrixTypes.data(),
      /*numMatrices=*/static_cast<int32_t>(matrices.size()),
      /*layout=*/CUSTATEVEC_MATRIX_LAYOUT_ROW,
      /*channelWires=*/dataOrNull(task.wires),
      /*numChannelWires=*/static_cast<int32_t>(task.wires.size())));
}

template <typename Scalar>
void enqueueTask(custatevecExSVUpdaterDescriptor_t updater,
                 const cudaq::cusv::SimulationTask<Scalar> &task) {
  std::visit(
      [updater](const auto &operation) {
        using Operation = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<Operation,
                                     cudaq::cusv::MatrixTask<Scalar>>) {
          HANDLE_CUSTATEVEC_ERROR(custatevecExSVUpdaterEnqueueMatrix(
              /*svUpdater=*/updater, /*matrix=*/operation.matrix.data(),
              /*matrixDataType=*/cudaq::cusv::complexDataType<Scalar>(),
              /*exMatrixType=*/operation.matrixType,
              /*layout=*/operation.layout, /*adjoint=*/operation.adjoint,
              /*targets=*/dataOrNull(operation.targets),
              /*numTargets=*/static_cast<int32_t>(operation.targets.size()),
              /*controls=*/dataOrNull(operation.controls),
              /*controlBitValues=*/dataOrNull(operation.controlValues),
              /*numControls=*/static_cast<int32_t>(operation.controls.size())));
        } else if constexpr (std::is_same_v<Operation,
                                            cudaq::cusv::NoiseTask<Scalar>>) {
          enqueueNoise(updater, operation);
        } else {
          // Unreachable: FusedGateEngine::apply intercepts PauliRotationTask
          // and applies it directly (Pauli rotations have no fused
          // representation), so one never reaches enqueueTask. This guards
          // against a future caller/refactor forwarding one here by mistake.
          throw std::logic_error(
              "internal error: a PauliRotationTask reached enqueueTask; Pauli "
              "rotations cannot be fused and must be applied directly.");
        }
      },
      task);
}

void applyUpdater(custatevecExSVUpdaterDescriptor_t updater,
                  custatevecExStateVectorDescriptor_t state,
                  std::span<const double> randomNumbers) {
  int32_t required = 0;
  HANDLE_CUSTATEVEC_ERROR(custatevecExSVUpdaterGetMaxNumRequiredRandnums(
      /*svUpdater=*/updater, /*maxNumRequiredRandnums=*/&required));
  if (randomNumbers.size() < static_cast<std::size_t>(required))
    throw std::invalid_argument(
        "Insufficient random numbers for queued noise channels.");
  HANDLE_CUSTATEVEC_ERROR(custatevecExSVUpdaterApply(
      /*svUpdater=*/updater, /*stateVector=*/state,
      /*randnums=*/randomNumbers.data(), /*numRandnums=*/required));
}

} // namespace

namespace cudaq::cusv {

template <typename Scalar>
void DirectGateEngine<Scalar>::apply(CuStateVecState<Scalar> &state,
                                     const SimulationTask<Scalar> &task,
                                     std::span<const double>) {
  std::visit(
      [&state](const auto &operation) {
        using Operation = std::decay_t<decltype(operation)>;
        if constexpr (std::is_same_v<Operation, MatrixTask<Scalar>>) {
          HANDLE_CUSTATEVEC_ERROR(custatevecExApplyMatrix(
              /*stateVector=*/state.descriptor(),
              /*matrix=*/operation.matrix.data(),
              /*matrixDataType=*/complexDataType<Scalar>(),
              /*exMatrixType=*/operation.matrixType,
              /*layout=*/operation.layout, /*adjoint=*/operation.adjoint,
              /*targets=*/dataOrNull(operation.targets),
              /*numTargets=*/static_cast<int32_t>(operation.targets.size()),
              /*controls=*/dataOrNull(operation.controls),
              /*controlBitValues=*/dataOrNull(operation.controlValues),
              /*numControls=*/static_cast<int32_t>(operation.controls.size())));
        } else if constexpr (std::is_same_v<Operation, PauliRotationTask>) {
          HANDLE_CUSTATEVEC_ERROR(custatevecExApplyPauliRotation(
              /*stateVector=*/state.descriptor(), /*theta=*/operation.angle,
              /*paulis=*/dataOrNull(operation.paulis),
              /*targets=*/dataOrNull(operation.targets),
              /*numTargets=*/static_cast<int32_t>(operation.targets.size()),
              /*controls=*/dataOrNull(operation.controls),
              /*controlBitValues=*/dataOrNull(operation.controlValues),
              /*numControls=*/static_cast<int32_t>(operation.controls.size())));
        } else {
          // cuStateVecEx exposes noise channels only through the SV updater, so
          // noise is applied by the fused engine. The direct engine applies
          // gates immediately with no updater and does not handle noise.
          throw std::invalid_argument(
              "The direct gate engine does not apply noise channels. Direct "
              "mode is active because fusion was disabled "
              "(CUDAQ_FUSION_MAX_QUBITS "
              "or CUDAQ_MGPU_FUSE set to a non-positive value); set a positive "
              "fusion size to run noisy circuits on the fused updater engine.");
        }
      },
      task);
}

template <typename Scalar>
FusedGateEngine<Scalar>::FusedGateEngine(const CuStateVecConfig &config) {
  std::vector<custatevecExSVUpdaterConfigItem_t> items;
  items.push_back({CUSTATEVEC_EX_SVUPDATER_CONFIG_MAX_NUM_HOST_THREADS,
                   {config.hostThreads}});
  items.push_back({CUSTATEVEC_EX_SVUPDATER_CONFIG_DENSE_FUSION_SIZE,
                   {config.denseFusionQubits}});
  if (config.diagonalFusionQubits >= 0)
    items.push_back({CUSTATEVEC_EX_SVUPDATER_CONFIG_DIAGONAL_FUSION_SIZE,
                     {config.diagonalFusionQubits}});

  HANDLE_CUSTATEVEC_ERROR(custatevecExConfigureSVUpdater(
      /*svUpdaterConfig=*/&m_configuration,
      /*dataType=*/complexDataType<Scalar>(), /*configItems=*/items.data(),
      /*numConfigItems=*/static_cast<int32_t>(items.size())));
  try {
    HANDLE_CUSTATEVEC_ERROR(custatevecExSVUpdaterCreate(
        /*svUpdater=*/&m_updater, /*svUpdaterConfig=*/m_configuration,
        /*resourceManager=*/nullptr));
  } catch (...) {
    custatevecExDictionaryDestroy(/*dictionary=*/m_configuration);
    m_configuration = nullptr;
    throw;
  }
}

template <typename Scalar>
FusedGateEngine<Scalar>::~FusedGateEngine() {
  if (m_updater)
    custatevecExSVUpdaterDestroy(/*svUpdater=*/m_updater);
  if (m_configuration)
    custatevecExDictionaryDestroy(/*dictionary=*/m_configuration);
}

template <typename Scalar>
void FusedGateEngine<Scalar>::apply(CuStateVecState<Scalar> &state,
                                    const SimulationTask<Scalar> &task,
                                    std::span<const double> randomNumbers) {
  // Pauli rotations have no fused representation: the SV updater only enqueues
  // matrices and channels, and cuStateVecEx applies rotations through the
  // dedicated custatevecExApplyPauliRotation primitive. A rotation therefore
  // acts as a fusion barrier -- flush the pending fused gates, apply it
  // directly via the direct engine.
  if (std::holds_alternative<PauliRotationTask>(task)) {
    flush(state, randomNumbers);
    DirectGateEngine<Scalar> direct;
    direct.apply(state, task, randomNumbers);
    return;
  }

  enqueueTask(m_updater, task);
  ++m_pendingTaskCount;
}

template <typename Scalar>
void FusedGateEngine<Scalar>::flush(CuStateVecState<Scalar> &state,
                                    std::span<const double> randomNumbers) {
  if (m_pendingTaskCount == 0)
    return;
  try {
    applyUpdater(m_updater, state.descriptor(), randomNumbers);
    // Clear is required after every application to reinitialize the updater's
    // fusion state before accepting the next circuit segment.
    HANDLE_CUSTATEVEC_ERROR(
        custatevecExSVUpdaterClear(/*svUpdater=*/m_updater));
    m_pendingTaskCount = 0;
  } catch (...) {
    custatevecExSVUpdaterClear(/*svUpdater=*/m_updater);
    m_pendingTaskCount = 0;
    throw;
  }
}

template <typename Scalar>
std::unique_ptr<GateEngine<Scalar>>
createGateEngine(const CuStateVecConfig &config) {
  if (config.gateMode == GateExecutionMode::Direct ||
      config.denseFusionQubits <= 0)
    return std::make_unique<DirectGateEngine<Scalar>>();
  return std::make_unique<FusedGateEngine<Scalar>>(config);
}

template struct DirectGateEngine<float>;
template struct DirectGateEngine<double>;
template class FusedGateEngine<float>;
template class FusedGateEngine<double>;
template std::unique_ptr<GateEngine<float>>
createGateEngine(const CuStateVecConfig &);
template std::unique_ptr<GateEngine<double>>
createGateEngine(const CuStateVecConfig &);

} // namespace cudaq::cusv
