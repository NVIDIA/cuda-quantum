/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

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

/// @brief Generate a vector of random values
/// @param num_samples
/// @param max_value
/// @return
static std::vector<double> randomValues(uint64_t num_samples,
                                        double max_value) {
  std::vector<double> rs;
  rs.reserve(num_samples);
  std::random_device rd;
  std::mt19937 rgen(rd());
  std::uniform_real_distribution<double> distr(0.0, max_value);
  for (uint64_t i = 0; i < num_samples; ++i) {
    rs.emplace_back(distr(rgen));
  }
  std::sort(rs.begin(), rs.end());
  return rs;
}

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
class CuStateVecCircuitSimulator : public nvqir::CircuitSimulator {
protected:
  // This type by default uses FP64
  using DataType = std::complex<ScalarType>;
  using DataVector = std::vector<DataType>;

  using CudaDataType = std::conditional_t<std::is_same_v<ScalarType, float>,
                                          cuFloatComplex, cuDoubleComplex>;

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

  /// @brief Return true if the bit string has even parity
  /// @param x
  /// @return
  bool hasEvenParity(const std::string &x) {
    int c = std::count(x.begin(), x.end(), '1');
    return c % 2 == 0;
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
        handle, deviceStateVector, cuStateVecCudaDataType, localNQubitsAllocated,
        matrix.data(), cuStateVecCudaDataType, CUSTATEVEC_MATRIX_LAYOUT_ROW, 0,
        targets.data(), targets.size(),
        controls.empty() ? nullptr : controls.data(), nullptr, controls.size(),
        cuStateVecComputeType, extraWorkspace, extraWorkspaceSizeInBytes));
  }

  /// @brief Utility function for applying one-target-qubit operations with
  /// optional control qubits
  /// @tparam GateT The instruction type, must be QppInstruction derived
  /// @param controls The control qubits, can be empty
  /// @param qubitIdx The target qubit
  template <typename GateT>
  void oneQubitApply(const std::vector<std::size_t> &controls,
                     const std::size_t qubitIdx) {
    GateT gate;
    cudaq::info(gateToString(gate.name(), controls, {}, {qubitIdx}));
    DataVector matrix = gate.getGate();
    std::vector<int> targets{(int)qubitIdx}, ctrls32;
    for (auto &c : controls)
      ctrls32.push_back(c);
    applyGateMatrix(matrix, ctrls32, targets);
  }

  /// @brief Utility function for applying one-target-qubit rotation operations
  /// @tparam RotationGateT The instruction type, must be QppInstruction derived
  /// @param angle The rotation angle
  /// @param controls The control qubits, can be empty
  /// @param qubitIdx The target qubit
  template <typename RotationGateT>
  void oneQubitOneParamApply(const double angle,
                             const std::vector<std::size_t> &controls,
                             const std::size_t qubitIdx) {
    RotationGateT gate;
    cudaq::info(gateToString(gate.name(), controls, {angle}, {qubitIdx}));
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

  /// @brief It's more efficient for us to allocate the whole state vector
  /// and if we are in sampling or observe contexts, we will likely allocate
  /// a chunk of qubits at once. Override the base class here and allocate
  /// the state vector on GPU.
  std::vector<std::size_t> allocateQubits(std::size_t count) override {
    std::vector<std::size_t> qubits;
    for (std::size_t i = 0; i < count; i++)
      qubits.emplace_back(tracker.getNextIndex());

    int dev;
    cudaGetDevice(&dev);
    cudaq::info("GPU {} Allocating new qubit array of size {}.", dev, count);

    // Increment the number of qubits and set
    // the new state dimension
    nQubitsAllocated += count;
    auto oldStateDimension = stateDimension;
    stateDimension = calculateStateDim(nQubitsAllocated);

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
          oldStateDimension);
      cudaFree(deviceStateVector);
      deviceStateVector = newDeviceStateVector;
    }

    return qubits;
  }

  /// @brief Grow the state vector by one qubit.
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
      int64_t oldDimension = calculateStateDim(std::log2(stateDimension) - 1);
      // Allocate new state..
      void *newDeviceStateVector;
      HANDLE_CUDA_ERROR(cudaMalloc((void **)&newDeviceStateVector,
                                   stateDimension * sizeof(CudaDataType)));
      constexpr int32_t threads_per_block = 256;
      uint32_t n_blocks =
          (stateDimension + threads_per_block - 1) / threads_per_block;
      setFirstNElements<<<n_blocks, threads_per_block>>>(
          reinterpret_cast<CudaDataType *>(newDeviceStateVector),
          reinterpret_cast<CudaDataType *>(deviceStateVector), oldDimension);
      cudaFree(deviceStateVector);
      deviceStateVector = newDeviceStateVector;
    }
  }

  /// @brief Reset the qubit state.
  void resetQubitStateImpl() override {
    HANDLE_ERROR(custatevecDestroy(handle));
    HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    if (extraWorkspaceSizeInBytes)
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
    deviceStateVector = nullptr;
    extraWorkspaceSizeInBytes = 0;
    nResets = 0;
  }

public:
  /// @brief The constructor
  CuStateVecCircuitSimulator() {
    if constexpr (std::is_same_v<ScalarType, float>) {
      cuStateVecComputeType = CUSTATEVEC_COMPUTE_32F;
      cuStateVecCudaDataType = CUDA_C_32F;
    }

    cudaFree(0);
  }

  /// The destructor
  virtual ~CuStateVecCircuitSimulator() = default;

/// The one-qubit overrides
#define QPP_ONE_QUBIT_METHOD_OVERRIDE(NAME)                                    \
  using CircuitSimulator::NAME;                                                \
  void NAME(const std::vector<std::size_t> &controls,                          \
            const std::size_t qubitIdx) override {                             \
    oneQubitApply<nvqir::NAME<ScalarType>>(controls, qubitIdx);                \
  }

  QPP_ONE_QUBIT_METHOD_OVERRIDE(x)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(y)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(z)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(h)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(s)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(t)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(sdg)
  QPP_ONE_QUBIT_METHOD_OVERRIDE(tdg)

/// The one-qubit parameterized overrides
#define QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(NAME)                          \
  using CircuitSimulator::NAME;                                                \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::size_t qubitIdx) override {                             \
    oneQubitOneParamApply<nvqir::NAME<ScalarType>>(angle, controls, qubitIdx); \
  }

  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(rx)
  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(ry)
  QPP_ONE_QUBIT_ONE_PARAM_METHOD_OVERRIDE(rz)

  using CircuitSimulator::r1;
  /// @brief The r1 gate
  /// @param angle
  /// @param controls
  /// @param qubitIdx
  void r1(const double angle, const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    cudaq::info(gateToString("r1", controls, {}, {qubitIdx}));
    DataVector matrix{
        {1.0, 0.0},
        {0.0, 0.0},
        {0.0, 0.0},
        std::exp(nvqir::im<ScalarType> * static_cast<ScalarType>(angle))};
    std::vector<int> targets{(int)qubitIdx}, ctrls32;
    for (auto &c : controls)
      ctrls32.push_back(c);

    applyGateMatrix(matrix, ctrls32, targets);
  }

  using CircuitSimulator::u2;
  /// @brief The u2 gate
  /// @param phi
  /// @param lambda
  /// @param controls
  /// @param qubitIdx
  void u2(const double phi, const double lambda,
          const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    ScalarType castedPhi = static_cast<ScalarType>(phi);
    ScalarType castedLambda = static_cast<ScalarType>(lambda);
    cudaq::info(gateToString("u2", controls, {phi, lambda}, {qubitIdx}));
    auto matrix = nvqir::getGateByName<ScalarType>(nvqir::GateName::U2,
                                                   {castedPhi, castedLambda});
    std::vector<int> targets{(int)qubitIdx}, ctrls32;
    for (auto &c : controls)
      ctrls32.push_back(c);
    applyGateMatrix(matrix, ctrls32, targets);
  }

  using CircuitSimulator::u3;
  /// @brief The u3 gate.
  /// @param theta
  /// @param phi
  /// @param lambda
  /// @param controls
  /// @param qubitIdx
  void u3(const double theta, const double phi, const double lambda,
          const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    auto castedTheta = static_cast<ScalarType>(theta);
    auto castedPhi = static_cast<ScalarType>(phi);
    auto castedLambda = static_cast<ScalarType>(lambda);
    cudaq::info(gateToString("u3", controls, {theta, phi, lambda}, {qubitIdx}));
    auto matrix = nvqir::getGateByName<ScalarType>(
        nvqir::GateName::U3, {castedTheta, castedPhi, castedLambda});
    std::vector<int> targets{(int)qubitIdx}, ctrls32;
    for (auto &c : controls)
      ctrls32.push_back(c);
    applyGateMatrix(matrix, ctrls32, targets);
  }

  using CircuitSimulator::u1;
  /// @brief The u1 gate
  /// @param angle
  /// @param controls
  /// @param qubitIdx
  void u1(const double angle, const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    r1(angle, controls, qubitIdx);
  }

  /// @brief Swap operation
  using CircuitSimulator::swap;
  void swap(const std::vector<std::size_t> &ctrlBits, const std::size_t srcIdx,
            const std::size_t tgtIdx) override {
    cudaq::info(gateToString("swap", ctrlBits, {}, {srcIdx, tgtIdx}));
    DataVector matrix{{1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                      {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0},
                      {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
                      {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    std::vector<int> targets{(int)srcIdx, (int)tgtIdx}, ctrls32;
    for (auto &c : ctrlBits)
      ctrls32.push_back(c);
    applyGateMatrix(matrix, ctrls32, targets);
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

  /// @brief Sample the multi-qubit state.
  /// @param measuredBits
  /// @param shots
  /// @return
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
      auto par = hasEvenParity(kv.first);
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
    if constexpr (std::is_same_v<ScalarType, float>) {
      throw std::runtime_error(
          "CustateVec F32 does not support getStateData().");
    } else {
      std::vector<std::complex<ScalarType>> data(stateDimension);
      cudaMemcpy(data.data(), deviceStateVector,
                 stateDimension * sizeof(CudaDataType), cudaMemcpyDeviceToHost);
      return cudaq::State{{stateDimension}, data};
    }
  }

  std::string name() const override;
  NVQIR_SIMULATOR_CLONE_IMPL(CuStateVecCircuitSimulator<ScalarType>)
};
} // namespace

#ifndef __NVQIR_CUSTATEVEC_TOGGLE_CREATE
template <>
std::string CuStateVecCircuitSimulator<double>::name() const {
  return "custatevec";
}
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(CuStateVecCircuitSimulator<>, custatevec)
#endif
