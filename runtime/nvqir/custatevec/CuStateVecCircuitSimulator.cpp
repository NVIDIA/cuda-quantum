/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "CuStateVecCircuitSimulator.h"
#include "CircuitSimulator.h"
#include "CuStateVecState.h"
#include "Gates.h"
#include "Timing.h"
#include "cuComplex.h"
#include "custatevec.h"
#include "device_launch_parameters.h"
#include <bitset>
#include <complex>
#include <iostream>
#include <random>
#include <set>

namespace {

/// @brief The CuStateVecCircuitSimulator implements the CircuitSimulator
/// base class to provide a simulator that delegates to the NVIDIA CuStateVec
/// GPU-accelerated library.
template <typename ScalarType = double>
class CuStateVecCircuitSimulator
    : public nvqir::CircuitSimulatorBase<ScalarType> {
protected:
  // This type by default uses FP64
  using DataType = std::complex<ScalarType>;
  using DataVector = std::vector<DataType>;
  using CudaDataType = std::conditional_t<std::is_same_v<ScalarType, float>,
                                          cuFloatComplex, cuDoubleComplex>;

  using nvqir::CircuitSimulatorBase<ScalarType>::tracker;
  using nvqir::CircuitSimulatorBase<ScalarType>::nQubitsAllocated;
  using nvqir::CircuitSimulatorBase<ScalarType>::stateDimension;
  using nvqir::CircuitSimulatorBase<ScalarType>::calculateStateDim;
  using nvqir::CircuitSimulatorBase<ScalarType>::executionContext;
  using nvqir::CircuitSimulatorBase<ScalarType>::gateToString;
  using nvqir::CircuitSimulatorBase<ScalarType>::x;
  using nvqir::CircuitSimulatorBase<ScalarType>::flushGateQueue;
  using nvqir::CircuitSimulatorBase<ScalarType>::previousStateDimension;
  using nvqir::CircuitSimulatorBase<ScalarType>::shouldObserveFromSampling;
  using nvqir::CircuitSimulatorBase<ScalarType>::summaryData;

  /// @brief The statevector that cuStateVec manipulates on the GPU
  void *deviceStateVector = nullptr;

  /// @brief The cuStateVec handle
  custatevecHandle_t handle = nullptr;

  /// @brief Pointer to potentially needed extra memory
  void *extraWorkspace = nullptr;

  /// @brief The size of the extra workspace
  size_t extraWorkspaceSizeInBytes = 0;

  custatevecComputeType_t cuStateVecComputeType = CUSTATEVEC_COMPUTE_64F;
  cudaDataType_t cuStateVecCudaDataType = CUDA_C_64F;
  std::random_device randomDevice;
  std::mt19937 randomEngine;
  bool ownsDeviceVector = true;

  /// @brief Generate a vector of random values
  std::vector<double> randomValues(uint64_t num_samples, double max_value) {
    std::vector<double> rs;
    rs.reserve(num_samples);
    std::uniform_real_distribution<double> distr(0.0, max_value);
    for (uint64_t i = 0; i < num_samples; ++i) {
      rs.emplace_back(distr(randomEngine));
    }
    std::sort(rs.begin(), rs.end());
    return rs;
  }

  /// @brief Convert the pauli rotation gate name to a CUSTATEVEC_PAULI Type
  /// @param type
  /// @return
  custatevecPauli_t pauliStringToEnum(const std::string_view type) {
    if (type == "rx") {
      return CUSTATEVEC_PAULI_X;
    } else if (type == "ry") {
      return CUSTATEVEC_PAULI_Y;
    } else if (type == "rz") {
      return CUSTATEVEC_PAULI_Z;
    }
    printf("Error, should not be here with pauli.\n");
    exit(1);
  }

  /// @brief Apply the matrix to the state vector on the GPU
  /// @param matrix The matrix data as a 1-d array, row-major
  /// @param controls Possible control qubits, can be empty
  /// @param targets Target qubits
  void applyGateMatrix(const DataVector &matrix,
                       const std::vector<int> &controls,
                       const std::vector<int> &targets) {
    HANDLE_ERROR(custatevecApplyMatrixGetWorkspaceSize(
        handle, cuStateVecCudaDataType, nQubitsAllocated, matrix.data(),
        cuStateVecCudaDataType, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets.size(),
        controls.size(), cuStateVecComputeType, &extraWorkspaceSizeInBytes));

    if (extraWorkspaceSizeInBytes > 0)
      HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    auto localNQubitsAllocated =
        stateDimension > 0 ? std::log2(stateDimension) : 0;

    // apply gate
    HANDLE_ERROR(custatevecApplyMatrix(
        handle, deviceStateVector, cuStateVecCudaDataType,
        localNQubitsAllocated, matrix.data(), cuStateVecCudaDataType,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets.data(), targets.size(),
        controls.empty() ? nullptr : controls.data(), nullptr, controls.size(),
        cuStateVecComputeType, extraWorkspace, extraWorkspaceSizeInBytes));

    if (extraWorkspace) {
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
      extraWorkspace = nullptr;
    }
  }

  /// @brief Utility function for applying one-target-qubit rotation operations
  template <typename RotationGateT>
  void oneQubitOneParamApply(const double angle,
                             const std::vector<std::size_t> &controls,
                             const std::size_t qubitIdx) {
    RotationGateT gate;
    std::vector<int> controls32;
    for (auto c : controls)
      controls32.push_back((int)c);
    custatevecPauli_t pauli[] = {pauliStringToEnum(gate.name())};
    int targets[] = {(int)qubitIdx};
    HANDLE_ERROR(custatevecApplyPauliRotation(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        -0.5 * angle, pauli, targets, 1, controls32.data(), nullptr,
        controls32.size()));
  }

  /// @brief Nice utility function to have to print the state vector contents on
  /// GPU.
  void printStateFromGPU(const std::string &name, void *ptr, std::size_t size) {
    std::vector<std::complex<ScalarType>> tmp(size);
    cudaMemcpy(tmp.data(), ptr, size * sizeof(std::complex<ScalarType>),
               cudaMemcpyDeviceToHost);
    for (auto &r : tmp)
      printf("%s: (%.12lf, %.12lf)\n", name.c_str(), r.real(), r.imag());
    printf("\n");
  }

  /// @brief Increase the state size by the given number of qubits.
  void addQubitsToState(std::size_t count, const void *stateIn) override {
    ScopedTraceWithContext("CuStateVecCircuitSimulator::addQubitsToState",
                           count);
    if (count == 0)
      return;

    // Cast the state, at this point an error would
    // have been thrown if it is not of the right floating point type
    std::complex<ScalarType> *state =
        reinterpret_cast<std::complex<ScalarType> *>(
            const_cast<void *>(stateIn));

    int dev;
    HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
    CUDAQ_INFO("GPU {} Allocating new qubit array of size {}.", dev, count);

    constexpr int32_t threads_per_block = 256;
    uint32_t n_blocks =
        (stateDimension + threads_per_block - 1) / threads_per_block;

    // Check if this is the first time to allocate, if so
    // the allocation is much easier
    if (!deviceStateVector) {
      // Create the memory and the handle
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&deviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      HANDLE_ERROR(custatevecCreate(&handle));
      ownsDeviceVector = true;
      // If no state provided, initialize to the zero state
      if (state == nullptr) {
        nvqir::initializeDeviceStateVector<CudaDataType>(
            n_blocks, threads_per_block, deviceStateVector, stateDimension);
        HANDLE_CUDA_ERROR(cudaGetLastError());
        return;
      }

      // Check if the pointer is a device pointer
      cudaPointerAttributes attributes;
      HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, state));

      if (attributes.type == cudaMemoryTypeDevice) {
        throw std::invalid_argument(
            "[CuStateVecCircuitSimulator] Incompatible host pointer");
      }

      // First allocation, so just set the user provided data here
      ScopedTraceWithContext(
          "CuStateVecCircuitSimulator::addQubitsToState cudaMemcpy",
          stateDimension * sizeof(CudaDataType));
      HANDLE_CUDA_ERROR(cudaMemcpy(deviceStateVector, state,
                                   stateDimension * sizeof(CudaDataType),
                                   cudaMemcpyHostToDevice));

      return;
    }

    // State already exists, need to allocate new state and compute
    // kronecker product with existing state

    // Allocate new vector to place the kron prod result
    void *newDeviceStateVector;
    HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                 stateDimension * sizeof(CudaDataType)));
    HANDLE_CUDA_ERROR(cudaMemset(newDeviceStateVector, 0,
                                 stateDimension * sizeof(CudaDataType)));
    // Place the state data on device. Could be that
    // we just need the zero state, or the user could have provided one
    void *otherState;
    HANDLE_CUDA_ERROR(cudaMalloc((void **)&otherState,
                                 (1UL << count) * sizeof(CudaDataType)));
    if (state == nullptr) {
      nvqir::initializeDeviceStateVector<CudaDataType>(
          n_blocks, threads_per_block, otherState, (1UL << count));
      HANDLE_CUDA_ERROR(cudaGetLastError());
    } else {
      // Check if the pointer is a device pointer
      cudaPointerAttributes attributes;
      HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, state));

      if (attributes.type == cudaMemoryTypeDevice) {
        throw std::invalid_argument(
            "[CuStateVecCircuitSimulator] Incompatible host pointer");
      }

      HANDLE_CUDA_ERROR(cudaMemcpy(otherState, state,
                                   (1UL << count) * sizeof(CudaDataType),
                                   cudaMemcpyHostToDevice));
    }

    {
      ScopedTraceWithContext(
          "CuStateVecCircuitSimulator::addQubitsToState kronprod");
      // Compute the kronecker product
      nvqir::kronprod<CudaDataType>(
          n_blocks, threads_per_block, previousStateDimension,
          deviceStateVector, (1UL << count), otherState, newDeviceStateVector);
      HANDLE_CUDA_ERROR(cudaGetLastError());
    }
    // Free the old vectors we don't need anymore.
    HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    HANDLE_CUDA_ERROR(cudaFree(otherState));
    deviceStateVector = newDeviceStateVector;
  }

  void addQubitsToState(const cudaq::SimulationState &in_state) override {
    const cudaq::CusvState<ScalarType> *const casted =
        dynamic_cast<const cudaq::CusvState<ScalarType> *>(&in_state);
    if (!casted)
      throw std::invalid_argument(
          "[CuStateVecCircuitSimulator] Incompatible state input");

    if (!deviceStateVector) {
      // Create the memory and the handle
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&deviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      ownsDeviceVector = true;
      HANDLE_ERROR(custatevecCreate(&handle));
      ScopedTraceWithContext(
          "CuStateVecCircuitSimulator::addQubitsToState cudaMemcpy");
      // First allocation, so just copy the user provided data (device mem) here
      HANDLE_CUDA_ERROR(cudaMemcpy(
          deviceStateVector, casted->getDevicePointer(),
          stateDimension * sizeof(CudaDataType), cudaMemcpyDeviceToDevice));
      return;
    }

    // Expanding the state
    // Allocate new vector to place the kron prod result
    void *newDeviceStateVector;
    HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                 stateDimension * sizeof(CudaDataType)));
    HANDLE_CUDA_ERROR(cudaMemset(newDeviceStateVector, 0,
                                 stateDimension * sizeof(CudaDataType)));
    constexpr int32_t threads_per_block = 256;
    uint32_t n_blocks =
        (stateDimension + threads_per_block - 1) / threads_per_block;
    {
      ScopedTraceWithContext(
          "CuStateVecCircuitSimulator::addQubitsToState kronprod");
      // Compute the kronecker product
      nvqir::kronprod<CudaDataType>(
          n_blocks, threads_per_block, previousStateDimension,
          deviceStateVector, (1UL << in_state.getNumQubits()),
          casted->getDevicePointer(), newDeviceStateVector);
      HANDLE_CUDA_ERROR(cudaGetLastError());
    }
    // Free the old state we don't need anymore.
    // Note: the devicePtr of the input state is owned by the caller.
    HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    deviceStateVector = newDeviceStateVector;
  }

  /// @brief Increase the state size by one qubit.
  void addQubitToState() override {
    ScopedTraceWithContext("CuStateVecCircuitSimulator::addQubitToState");
    // Update the state vector
    if (!deviceStateVector) {
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&deviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      nvqir::initializeDeviceStateVector<CudaDataType>(
          n_blocks, threads_per_block, deviceStateVector, stateDimension);
      HANDLE_ERROR(custatevecCreate(&handle));
    } else {
      // Allocate new state..
      void *newDeviceStateVector;
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      nvqir::setFirstNElements<CudaDataType>(
          n_blocks, threads_per_block, newDeviceStateVector, deviceStateVector,
          previousStateDimension);
      HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
      deviceStateVector = newDeviceStateVector;
    }
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    if (deviceStateVector)
      HANDLE_ERROR(custatevecDestroy(handle));
    if (deviceStateVector && ownsDeviceVector) {
      HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    }
    if (extraWorkspace) {
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
      extraWorkspace = nullptr;
    }
    deviceStateVector = nullptr;
    extraWorkspaceSizeInBytes = 0;
  }

  /// @brief Apply the given GateApplicationTask
  void applyGate(const typename nvqir::CircuitSimulatorBase<
                 ScalarType>::GateApplicationTask &task) override {
    std::vector<int> controls, targets;
    std::transform(task.controls.begin(), task.controls.end(),
                   std::back_inserter(controls),
                   [](std::size_t idx) { return static_cast<int>(idx); });
    std::transform(task.targets.begin(), task.targets.end(),
                   std::back_inserter(targets),
                   [](std::size_t idx) { return static_cast<int>(idx); });
    // If we have no parameters, just apply the matrix.
    if (task.parameters.empty()) {
      applyGateMatrix(task.matrix, controls, targets);
      return;
    }

    // If we have parameters, it may be more efficient to
    // compute with custatevecApplyPauliRotation
    if (task.operationName == "rx") {
      oneQubitOneParamApply<nvqir::rx<ScalarType>>(
          task.parameters[0], task.controls, task.targets[0]);
    } else if (task.operationName == "ry") {
      oneQubitOneParamApply<nvqir::ry<ScalarType>>(
          task.parameters[0], task.controls, task.targets[0]);
    } else if (task.operationName == "rz") {
      oneQubitOneParamApply<nvqir::rz<ScalarType>>(
          task.parameters[0], task.controls, task.targets[0]);
    } else {
      // Fallback to just applying the gate.
      applyGateMatrix(task.matrix, controls, targets);
    }
  }

  /// @brief Set the state back to the |0> state on the
  /// current number of qubits
  void setToZeroState() override {
    constexpr int32_t threads_per_block = 256;
    uint32_t n_blocks =
        (stateDimension + threads_per_block - 1) / threads_per_block;
    nvqir::initializeDeviceStateVector<CudaDataType>(
        n_blocks, threads_per_block, deviceStateVector, stateDimension);
    HANDLE_CUDA_ERROR(cudaGetLastError());
  }

public:
  /// @brief The constructor
  CuStateVecCircuitSimulator() {
    if constexpr (std::is_same_v<ScalarType, float>) {
      cuStateVecComputeType = CUSTATEVEC_COMPUTE_32F;
      cuStateVecCudaDataType = CUDA_C_32F;
    }

    // Populate the correct name so it is printed correctly during
    // deconstructor.
    summaryData.name = name();

    HANDLE_CUDA_ERROR(cudaFree(0));
    randomEngine = std::mt19937(randomDevice());
  }

  /// The destructor
  virtual ~CuStateVecCircuitSimulator() = default;

  void setRandomSeed(std::size_t randomSeed) override {
    randomEngine = std::mt19937(randomSeed);
  }

  /// @brief Device synchronization
  void synchronize() override { HANDLE_CUDA_ERROR(cudaDeviceSynchronize()); }

  /// @brief Measure operation
  /// @param qubitIdx
  /// @return
  bool measureQubit(const std::size_t qubitIdx) override {
    const int basisBits[] = {(int)qubitIdx};
    int parity;
    double rand = randomValues(1, 1.0)[0];
    HANDLE_ERROR(custatevecMeasureOnZBasis(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        &parity, basisBits, /*N Bits*/ 1, rand,
        CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO));
    CUDAQ_INFO("Measured qubit {} -> {}", qubitIdx, parity);
    return parity == 1 ? true : false;
  }

  /// @brief Reset the qubit
  /// @param qubitIdx
  void resetQubit(const std::size_t qubitIdx) override {
    flushGateQueue();
    this->flushAnySamplingTasks();
    const int basisBits[] = {(int)qubitIdx};
    int parity;
    double rand = randomValues(1, 1.0)[0];
    HANDLE_ERROR(custatevecMeasureOnZBasis(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        &parity, basisBits, /*N Bits*/ 1, rand,
        CUSTATEVEC_COLLAPSE_NORMALIZE_AND_ZERO));
    if (parity) {
      x(qubitIdx);
    }
  }

  /// @brief Override base class functionality for a general Pauli
  /// rotation to delegate to the performant custatevecApplyPauliRotation.
  void applyExpPauli(double theta, const std::vector<std::size_t> &controlIds,
                     const std::vector<std::size_t> &qubits,
                     const cudaq::spin_op_term &term) override {
    if (this->isInTracerMode()) {
      nvqir::CircuitSimulator::applyExpPauli(theta, controlIds, qubits, term);
      return;
    }
    flushGateQueue();
    CUDAQ_INFO(" [cusv decomposing] exp_pauli({}, {})", theta,
               term.to_string());
    std::vector<int> controls, targets;
    for (const auto &bit : controlIds)
      controls.emplace_back(static_cast<int>(bit));
    std::vector<custatevecPauli_t> paulis;
    if (term.num_ops() != qubits.size())
      throw std::runtime_error(
          "incorrect number of qubits for exp_pauli - expecting " +
          std::to_string(term.num_ops()) + " qubits");

    std::size_t idx = 0;
    for (const auto &op : term) {
      auto pauli = op.as_pauli();
      if (pauli == cudaq::pauli::I)
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_I);
      else if (pauli == cudaq::pauli::X)
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_X);
      else if (pauli == cudaq::pauli::Y)
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_Y);
      else
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_Z);

      targets.push_back(qubits[idx++]);
    }

    HANDLE_ERROR(custatevecApplyPauliRotation(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        theta, paulis.data(), targets.data(), targets.size(), controls.data(),
        nullptr, controls.size()));
  }

  /// @brief Compute the operator expectation value, with respect to
  /// the current state vector, directly on GPU with the
  /// given the operator matrix and target qubit indices.
  auto getExpectationFromOperatorMatrix(const std::complex<double> *matrix,
                                        const std::vector<std::size_t> &tgts) {
    // Convert the size_t tgts into ints
    std::vector<int> tgtsInt(tgts.size());
    std::transform(tgts.begin(), tgts.end(), tgtsInt.begin(),
                   [&](std::size_t x) { return static_cast<int>(x); });
    // our bit ordering is reversed.
    size_t nIndexBits = nQubitsAllocated;

    // check the size of external workspace
    HANDLE_ERROR(custatevecComputeExpectationGetWorkspaceSize(
        handle, cuStateVecCudaDataType, nIndexBits, matrix,
        cuStateVecCudaDataType, CUSTATEVEC_MATRIX_LAYOUT_ROW, tgts.size(),
        cuStateVecComputeType, &extraWorkspaceSizeInBytes));

    if (extraWorkspaceSizeInBytes > 0)
      HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    double expect;

    // compute expectation
    HANDLE_ERROR(custatevecComputeExpectation(
        handle, deviceStateVector, cuStateVecCudaDataType, nIndexBits, &expect,
        CUDA_R_64F, nullptr, matrix, cuStateVecCudaDataType,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, tgtsInt.data(), tgts.size(),
        cuStateVecComputeType, extraWorkspace, extraWorkspaceSizeInBytes));

    if (extraWorkspace) {
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
      extraWorkspace = nullptr;
    }

    return expect;
  }

  /// @brief We can compute Observe from the matrix for a
  /// reasonable number of qubits, otherwise we should compute it
  /// via sampling
  bool canHandleObserve() override {
    // Do not compute <H> from matrix if shots based sampling requested
    // i.e., a valid shots count value was set.
    // Note: -1 is also used to denote non-sampling execution. Hence, we need to
    // check for this particular -1 value as being casted to an unsigned type.
    if (executionContext && executionContext->shots > 0 &&
        executionContext->shots != static_cast<std::size_t>(-1)) {
      return false;
    }

    // If no shots requested (exact expectation calulation), don't use
    // term-by-term observe as the default since
    // `CuStateVecCircuitSimulator::observe` will do a batched expectation value
    // calculation to compute all expectation values for all terms at once.
    return !shouldObserveFromSampling(/*defaultConfig=*/false);
  }

  /// @brief Compute the expected value from the observable matrix.
  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    // Use batched custatevecComputeExpectationsOnPauliBasis to compute all term
    // expectation values in one go
    uint32_t nPauliOperatorArrays = op.num_terms();
    assert(cudaq::spin_op::canonicalize(op) == op);

    // custatevecComputeExpectationsOnPauliBasis will throw errors if
    // nPauliOperatorArrays is 0, so catch that case early.
    if (nPauliOperatorArrays == 0)
      return cudaq::observe_result{};

    // Stable holders of vectors since we need to send vectors of pointers to
    // custatevec
    std::deque<std::vector<custatevecPauli_t>> pauliOperatorsArrayHolder;
    std::deque<std::vector<int32_t>> basisBitsArrayHolder;
    std::vector<const custatevecPauli_t *> pauliOperatorsArray;
    std::vector<const int32_t *> basisBitsArray;
    std::vector<std::complex<double>> coeffs;
    std::vector<uint32_t> nBasisBitsArray;
    pauliOperatorsArray.reserve(nPauliOperatorArrays);
    basisBitsArray.reserve(nPauliOperatorArrays);
    coeffs.reserve(nPauliOperatorArrays);
    nBasisBitsArray.reserve(nPauliOperatorArrays);
    // Helper to convert Pauli enums
    const auto cudaqToCustateVec = [](cudaq::pauli pauli) -> custatevecPauli_t {
      switch (pauli) {
      case cudaq::pauli::I:
        return CUSTATEVEC_PAULI_I;
      case cudaq::pauli::X:
        return CUSTATEVEC_PAULI_X;
      case cudaq::pauli::Y:
        return CUSTATEVEC_PAULI_Y;
      case cudaq::pauli::Z:
        return CUSTATEVEC_PAULI_Z;
      }
      __builtin_unreachable();
    };

    // Contruct data to send on to custatevec
    std::vector<std::string> termStrs;
    termStrs.reserve(nPauliOperatorArrays);
    for (const auto &term : op) {
      coeffs.emplace_back(term.evaluate_coefficient());
      std::vector<custatevecPauli_t> paulis;
      std::vector<int32_t> idxs;
      paulis.reserve(term.num_ops());
      idxs.reserve(term.num_ops());
      for (const auto &p : term) {
        auto pauli = p.as_pauli();
        if (pauli != cudaq::pauli::I) {
          auto target = p.target();
          paulis.emplace_back(cudaqToCustateVec(pauli));
          idxs.emplace_back(target);
          // Only X and Y pauli's translate to applied gates
          if (pauli != cudaq::pauli::Z) {
            // One operation for applying the term
            summaryData.svGateUpdate(/*nControls=*/0, /*nTargets=*/1,
                                     stateDimension,
                                     stateDimension * sizeof(DataType));
            // And one operation for un-applying the term
            summaryData.svGateUpdate(/*nControls=*/0, /*nTargets=*/1,
                                     stateDimension,
                                     stateDimension * sizeof(DataType));
          }
        }
      }
      pauliOperatorsArrayHolder.emplace_back(std::move(paulis));
      basisBitsArrayHolder.emplace_back(std::move(idxs));
      pauliOperatorsArray.emplace_back(pauliOperatorsArrayHolder.back().data());
      basisBitsArray.emplace_back(basisBitsArrayHolder.back().data());
      nBasisBitsArray.emplace_back(pauliOperatorsArrayHolder.back().size());
      termStrs.emplace_back(term.get_term_id());
    }
    std::vector<double> expectationValues(nPauliOperatorArrays);
    HANDLE_ERROR(custatevecComputeExpectationsOnPauliBasis(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        expectationValues.data(), pauliOperatorsArray.data(),
        nPauliOperatorArrays, basisBitsArray.data(), nBasisBitsArray.data()));
    std::complex<double> expVal = 0.0;
    std::vector<cudaq::ExecutionResult> results;
    results.reserve(nPauliOperatorArrays);
    for (uint32_t i = 0; i < nPauliOperatorArrays; ++i) {
      expVal += coeffs[i] * expectationValues[i];
      results.emplace_back(
          cudaq::ExecutionResult({}, termStrs[i], expectationValues[i]));
    }
    cudaq::sample_result perTermData(static_cast<double>(expVal.real()),
                                     results);
    return cudaq::observe_result(static_cast<double>(expVal.real()), op,
                                 perTermData);
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
    ScopedTraceWithContext(cudaq::TIMING_SAMPLE, "CuStateVecSimulator::sample");
    double expVal = 0.0;
    // cudaq::CountsDictionary counts;
    std::vector<custatevecPauli_t> z_pauli;
    std::vector<int> measuredBits32;
    for (auto m : measuredBits) {
      measuredBits32.push_back(m);
      z_pauli.push_back(CUSTATEVEC_PAULI_Z);
    }

    if (shots < 1) {
      // Just compute the expected value on <Z...Z>
      const uint32_t nBasisBitsArray[] = {(uint32_t)measuredBits.size()};
      const int *basisBitsArray[] = {measuredBits32.data()};
      const custatevecPauli_t *pauliArray[] = {z_pauli.data()};
      double expectationValues[1];
      HANDLE_ERROR(custatevecComputeExpectationsOnPauliBasis(
          handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
          expectationValues, pauliArray, 1, basisBitsArray, nBasisBitsArray));
      expVal = expectationValues[0];
      CUDAQ_INFO("Computed expectation value = {}", expVal);
      return cudaq::ExecutionResult{expVal};
    }

    // Grab some random seed values and create the sampler
    auto randomValues_ = randomValues(shots, 1.0);
    custatevecSamplerDescriptor_t sampler;
    HANDLE_ERROR(custatevecSamplerCreate(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        &sampler, shots, &extraWorkspaceSizeInBytes));
    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0)
      HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    // Run the sampling preprocess step.
    HANDLE_ERROR(custatevecSamplerPreprocess(handle, sampler, extraWorkspace,
                                             extraWorkspaceSizeInBytes));

    // Sample!
    std::vector<custatevecIndex_t> bitstrings0(shots);
    HANDLE_ERROR(custatevecSamplerSample(
        handle, sampler, bitstrings0.data(), measuredBits32.data(),
        measuredBits32.size(), randomValues_.data(), shots,
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

    if (extraWorkspace) {
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
      extraWorkspace = nullptr;
    }

    std::vector<std::string> sequentialData;
    sequentialData.reserve(shots);

    cudaq::ExecutionResult counts;

    // We've sampled, convert the results to our ExecutionResult counts
    for (int i = 0; i < shots; ++i) {
      auto bitstring = std::bitset<64>(bitstrings0[i])
                           .to_string()
                           .erase(0, 64 - measuredBits.size());
      std::reverse(bitstring.begin(), bitstring.end());
      counts.appendResult(bitstring, 1);
      sequentialData.push_back(std::move(bitstring));
    }

    // Compute the expectation value from the counts
    for (auto &kv : counts.counts) {
      auto par = cudaq::sample_result::has_even_parity(kv.first);
      auto p = kv.second / (double)shots;
      if (!par) {
        p = -p;
      }
      expVal += p;
    }

    counts.expectationValue = expVal;

    HANDLE_ERROR(custatevecSamplerDestroy(sampler));

    return counts;
  }

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    ownsDeviceVector = false;
    return std::make_unique<cudaq::CusvState<ScalarType>>(stateDimension,
                                                          deviceStateVector);
  }

  bool isStateVectorSimulator() const override { return true; }

  std::string name() const override;
  NVQIR_SIMULATOR_CLONE_IMPL(CuStateVecCircuitSimulator<ScalarType>)
};
} // namespace

#ifndef __NVQIR_CUSTATEVEC_TOGGLE_CREATE
template <>
std::string CuStateVecCircuitSimulator<double>::name() const {
  return "custatevec-fp64";
}
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(CuStateVecCircuitSimulator<>, custatevec_fp64)
#endif
