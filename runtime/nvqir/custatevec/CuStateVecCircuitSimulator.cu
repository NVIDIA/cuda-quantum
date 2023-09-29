/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma nv_diag_suppress = unsigned_compare_with_zero
#pragma nv_diag_suppress = unrecognized_gcc_pragma

#include "CircuitSimulator.h"
#include "Gates.h"
#include "cuComplex.h"
#include "custatevec.h"
#include <bitset>
#include <complex>
#include <iostream>
#include <random>
#include <set>

namespace {

#define HANDLE_ERROR(x)                                                        \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != CUSTATEVEC_STATUS_SUCCESS) {                                    \
      throw std::runtime_error(fmt::format("[custatevec] %{} in {} (line {})", \
                                           custatevecGetErrorString(err),      \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  };

#define HANDLE_CUDA_ERROR(x)                                                   \
  {                                                                            \
    const auto err = x;                                                        \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(fmt::format("[custatevec] %{} in {} (line {})", \
                                           cudaGetErrorString(err),            \
                                           __FUNCTION__, __LINE__));           \
    }                                                                          \
  };

/// @brief Initialize the device state vector to the |0...0> state
/// @param sv
/// @param dim
/// @return
template <typename CudaDataType>
__global__ void initializeDeviceStateVector(CudaDataType *sv, int64_t dim) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i == 0) {
    sv[i].x = 1.0;
    sv[i].y = 0.0;
  } else if (i < dim) {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

/// @brief Kernel to set the first N elements of the state vector sv equal to
/// the
// elements provided by the vector sv2. N is the number of elements to set.
// Size of sv must be greater than size of sv2.
/// @param sv
/// @param sv2
/// @param N
/// @return
template <typename T>
__global__ void setFirstNElements(T *sv, const T *__restrict__ sv2, int64_t N) {
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < N) {
    sv[i].x = sv2[i].x;
    sv[i].y = sv2[i].y;
  } else {
    sv[i].x = 0.0;
    sv[i].y = 0.0;
  }
}

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

  /// @brief The statevector that cuStateVec manipulates on the GPU
  void *deviceStateVector = nullptr;

  /// @brief The cuStateVec handle
  custatevecHandle_t handle;

  /// @brief Pointer to potentially needed extra memory
  void *extraWorkspace = nullptr;

  /// @brief The size of the extra workspace
  size_t extraWorkspaceSizeInBytes = 0;

  /// @brief Count the number of resets.
  int nResets = 0;

  custatevecComputeType_t cuStateVecComputeType = CUSTATEVEC_COMPUTE_64F;
  cudaDataType_t cuStateVecCudaDataType = CUDA_C_64F;
  std::random_device randomDevice;
  std::mt19937 randomEngine;

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

    // When we perform a deallocation we apply a
    // qubit reset, and the state does not shrink (trying to minimize device
    // memory manipulations), but nQubitsAllocated decrements.
    auto localNQubitsAllocated = nQubitsAllocated + nResets;

    // apply gate
    HANDLE_ERROR(custatevecApplyMatrix(
        handle, deviceStateVector, cuStateVecCudaDataType,
        localNQubitsAllocated, matrix.data(), cuStateVecCudaDataType,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, 0, targets.data(), targets.size(),
        controls.empty() ? nullptr : controls.data(), nullptr, controls.size(),
        cuStateVecComputeType, extraWorkspace, extraWorkspaceSizeInBytes));
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
    custatevecApplyPauliRotation(handle, deviceStateVector,
                                 cuStateVecCudaDataType, nQubitsAllocated,
                                 -0.5 * angle, pauli, targets, 1,
                                 controls32.data(), nullptr, controls32.size());
  }

  /// @brief Increase the state size by the given number of qubits.
  void addQubitsToState(std::size_t count) override {
    if (count == 0)
      return;

    int dev;
    cudaGetDevice(&dev);
    cudaq::info("GPU {} Allocating new qubit array of size {}.", dev, count);

    if (!deviceStateVector) {
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&deviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      initializeDeviceStateVector<<<n_blocks, threads_per_block>>>(
          reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension);
      HANDLE_ERROR(custatevecCreate(&handle));
    } else {
      // Allocate new state..
      void *newDeviceStateVector;
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      setFirstNElements<<<n_blocks, threads_per_block>>>(
          reinterpret_cast<CudaDataType *>(newDeviceStateVector),
          reinterpret_cast<CudaDataType *>(deviceStateVector),
          previousStateDimension);
      cudaFree(deviceStateVector);
      deviceStateVector = newDeviceStateVector;
    }
  }

  /// @brief Increase the state size by one qubit.
  void addQubitToState() override {
    // Update the state vector
    if (!deviceStateVector) {
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&deviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      initializeDeviceStateVector<<<n_blocks, threads_per_block>>>(
          reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension);
      HANDLE_ERROR(custatevecCreate(&handle));
    } else {
      // Allocate new state..
      void *newDeviceStateVector;
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      setFirstNElements<<<n_blocks, threads_per_block>>>(
          reinterpret_cast<CudaDataType *>(newDeviceStateVector),
          reinterpret_cast<CudaDataType *>(deviceStateVector),
          previousStateDimension);
      cudaFree(deviceStateVector);
      deviceStateVector = newDeviceStateVector;
    }
  }

  /// @brief Reset the qubit state.
  void deallocateStateImpl() override {
    HANDLE_ERROR(custatevecDestroy(handle));
    HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    if (extraWorkspaceSizeInBytes)
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
    deviceStateVector = nullptr;
    extraWorkspaceSizeInBytes = 0;
    nResets = 0;
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
    initializeDeviceStateVector<<<n_blocks, threads_per_block>>>(
        reinterpret_cast<CudaDataType *>(deviceStateVector), stateDimension);
  }

public:
  /// @brief The constructor
  CuStateVecCircuitSimulator() {
    if constexpr (std::is_same_v<ScalarType, float>) {
      cuStateVecComputeType = CUSTATEVEC_COMPUTE_32F;
      cuStateVecCudaDataType = CUDA_C_32F;
    }

    cudaFree(0);
    randomEngine = std::mt19937(randomDevice());
  }

  /// The destructor
  virtual ~CuStateVecCircuitSimulator() = default;

  void setRandomSeed(std::size_t randomSeed) override {
    randomEngine = std::mt19937(randomSeed);
  }

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
    cudaq::info("Measured qubit {} -> {}", qubitIdx, parity);
    return parity == 1 ? true : false;
  }

  /// @brief Reset the qubit
  /// @param qubitIdx
  void resetQubit(const std::size_t qubitIdx) override {
    flushGateQueue();
    nResets++;
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
  void applyExpPauli(double theta, const std::vector<std::size_t> &qubits,
                     const cudaq::spin_op &op) override {
    flushGateQueue();
    cudaq::info(" [cusv decomposing] exp_pauli({}, {})", theta,
                op.to_string(false));
    std::vector<int> controls, targets;
    std::vector<custatevecPauli_t> paulis;
    op.for_each_pauli([&](cudaq::pauli p, std::size_t i) {
      if (p == cudaq::pauli::I)
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_I);
      else if (p == cudaq::pauli::X)
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_X);
      else if (p == cudaq::pauli::Y)
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_Y);
      else
        paulis.push_back(custatevecPauli_t::CUSTATEVEC_PAULI_Z);

      targets.push_back(qubits[i]);
    });

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
    void *extraWorkspace = nullptr;
    size_t extraWorkspaceSizeInBytes = 0;

    // Convert the size_t tgts into ints
    std::vector<int> tgtsInt(tgts.size());
    std::transform(tgts.begin(), tgts.end(), tgtsInt.begin(),
                   [&](std::size_t x) { return static_cast<int>(x); });
    // our bit ordering is reversed.
    std::reverse(tgtsInt.begin(), tgtsInt.end());
    size_t nIndexBits = nQubitsAllocated;

    // check the size of external workspace
    HANDLE_ERROR(custatevecComputeExpectationGetWorkspaceSize(
        handle, cuStateVecCudaDataType, nIndexBits, matrix,
        cuStateVecCudaDataType, CUSTATEVEC_MATRIX_LAYOUT_ROW, tgts.size(),
        cuStateVecComputeType, &extraWorkspaceSizeInBytes));

    if (extraWorkspaceSizeInBytes > 0) {
      HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
    }

    double expect;

    // compute expectation
    HANDLE_ERROR(custatevecComputeExpectation(
        handle, deviceStateVector, cuStateVecCudaDataType, nIndexBits, &expect,
        CUDA_R_64F, nullptr, matrix, cuStateVecCudaDataType,
        CUSTATEVEC_MATRIX_LAYOUT_ROW, tgtsInt.data(), tgts.size(),
        cuStateVecComputeType, extraWorkspace, extraWorkspaceSizeInBytes));
    if (extraWorkspaceSizeInBytes)
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));

    return expect;
  }

  /// @brief We can compute Observe from the matrix for a
  /// reasonable number of qubits, otherwise we should compute it
  /// via sampling
  bool canHandleObserve() override {
    // Do not compute <H> from matrix if shots based sampling requested
    if (executionContext &&
        executionContext->shots != static_cast<std::size_t>(-1)) {
      return false;
    }

    /// Seems that FP32 is faster with
    /// custatevecComputeExpectationsOnPauliBasis
    if constexpr (std::is_same_v<ScalarType, float>) {
      return false;
    }

    return !shouldObserveFromSampling();
  }

  /// @brief Compute the expected value from the observable matrix.
  cudaq::ExecutionResult observe(const cudaq::spin_op &op) override {

    flushGateQueue();

    // The op is on the following target bits.
    std::set<std::size_t> targets;
    op.for_each_term([&](cudaq::spin_op &term) {
      term.for_each_pauli(
          [&](cudaq::pauli p, std::size_t idx) { targets.insert(idx); });
    });

    std::vector<std::size_t> targetsVec(targets.begin(), targets.end());

    // Get the matrix
    auto matrix = op.to_matrix();
    /// Compute the expectation value.
    auto ee = getExpectationFromOperatorMatrix(matrix.data(), targetsVec);
    return cudaq::ExecutionResult({}, ee);
  }

  /// @brief Sample the multi-qubit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &measuredBits,
                                const int shots) override {
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
      cudaq::info("Computed expectation value = {}", expVal);
      return cudaq::ExecutionResult{expVal};
    }

    // Grab some random seed values and create the sampler
    auto randomValues_ = randomValues(shots, 1.0);
    custatevecSamplerDescriptor_t sampler;
    HANDLE_ERROR(custatevecSamplerCreate(
        handle, deviceStateVector, cuStateVecCudaDataType, nQubitsAllocated,
        &sampler, shots, &extraWorkspaceSizeInBytes));
    // allocate external workspace if necessary
    if (extraWorkspaceSizeInBytes > 0) {
      HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));
    }

    // Run the sampling preprocess step.
    HANDLE_ERROR(custatevecSamplerPreprocess(handle, sampler, extraWorkspace,
                                             extraWorkspaceSizeInBytes));

    // Sample!
    custatevecIndex_t bitstrings0[shots];
    HANDLE_ERROR(custatevecSamplerSample(
        handle, sampler, bitstrings0, measuredBits32.data(),
        measuredBits32.size(), randomValues_.data(), shots,
        CUSTATEVEC_SAMPLER_OUTPUT_ASCENDING_ORDER));

    std::vector<std::string> sequentialData;

    cudaq::ExecutionResult counts;

    // We've sampled, convert the results to our ExecutionResult counts
    for (int i = 0; i < shots; ++i) {
      auto bitstring = std::bitset<64>(bitstrings0[i])
                           .to_string()
                           .erase(0, 64 - measuredBits.size());
      std::reverse(bitstring.begin(), bitstring.end());
      sequentialData.push_back(bitstring);
      counts.appendResult(bitstring, 1);
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
    return counts;
  }

  cudaq::State getStateData() override {
    // Handle empty state (e.g., no qubit allocation)
    if (stateDimension == 0)
      return cudaq::State{{stateDimension}, {}};

    std::vector<std::complex<ScalarType>> tmp(stateDimension);
    // Use custatevec accessor to retrieve the view
    custatevecAccessorDescriptor_t accessor;
    const uint32_t nIndexBits = std::log2(stateDimension);
    // Note: we use MSB bit ordering when reporting the state vector
    // hence, bit ordering vector = [N-1, N-2, ..., 0]
    std::vector<int32_t> bitOrdering(nIndexBits);
    std::iota(std::rbegin(bitOrdering), std::rend(bitOrdering), 0);
    std::size_t extraWorkspaceSizeInBytes = 0;
    // create accessor view
    HANDLE_ERROR(custatevecAccessorCreateView(
        handle, deviceStateVector, cuStateVecCudaDataType, nIndexBits,
        &accessor, bitOrdering.data(), bitOrdering.size(),
        /*maskBitString*/ nullptr, /*maskOrdering*/ nullptr,
        /*maskLen*/ 0, &extraWorkspaceSizeInBytes));
    // allocate external workspace if necessary
    void *extraWorkspace = nullptr;
    if (extraWorkspaceSizeInBytes > 0)
      HANDLE_CUDA_ERROR(cudaMalloc(&extraWorkspace, extraWorkspaceSizeInBytes));

    // set external workspace
    HANDLE_ERROR(custatevecAccessorSetExtraWorkspace(
        handle, accessor, extraWorkspace, extraWorkspaceSizeInBytes));

    // get all state vector components: [0, stateDimension)
    HANDLE_ERROR(custatevecAccessorGet(handle, accessor, tmp.data(),
                                       /*begin*/ 0,
                                       /*end*/
                                       stateDimension));
    // destroy descriptor
    HANDLE_ERROR(custatevecAccessorDestroy(accessor));
    // free extra workspace if allocated
    if (extraWorkspaceSizeInBytes > 0)
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));

    if constexpr (std::is_same_v<ScalarType, float>) {
      std::vector<std::complex<double>> data;
      std::transform(tmp.begin(), tmp.end(), std::back_inserter(data),
                     [](std::complex<float> &el) -> std::complex<double> {
                       return {static_cast<double>(el.real()),
                               static_cast<double>(el.imag())};
                     });
      return cudaq::State{{stateDimension}, data};
    } else {
      return cudaq::State{{stateDimension}, tmp};
    }
  }

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
