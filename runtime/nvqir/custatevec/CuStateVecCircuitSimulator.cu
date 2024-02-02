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
#include "CuStateVecState.h"

#include "cuComplex.h"
#include "custatevec.h"
#include <bitset>
#include <complex>
#include <iostream>
#include <random>
#include <set>

namespace {

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

// template <typename T>
// using ThrustComplex = thrust::complex<T>;

// /// @brief Custom functor for the thrust inner product.
// template <typename T>
// struct AdotConjB
//     : public thrust::binary_function<ThrustComplex<T>, ThrustComplex<T>,
//                                      ThrustComplex<T>> {
//   __host__ __device__ ThrustComplex<T> operator()(ThrustComplex<T> a,
//                                                   ThrustComplex<T> b) {
//     return a * thrust::conj(b);
//   };
// };

// // /// @brief CusvState provides an implementation of `SimulationState` that
// /// encapsulates the state data for the Custatevec Circuit Simulator. It
// /// attempts to keep the simulation data on GPU device and care is taken 
// /// to ensure operations and comparisons with other states operate 
// /// on compatible floating point element types.
// template <typename ScalarType>
// struct CusvState : public cudaq::SimulationState {
// private:
//   /// @brief Size of the state data array on GPU.
//   std::size_t size = 0;

//   /// @brief State device data pointer.
//   void *devicePtr = nullptr;

//   /// @brief Check that we are currently
//   /// using the correct CUDA device, set it
//   /// to the correct one if not
//   void checkAndSetDevice() const {
//     int dev = 0;
//     HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
//     auto currentDevice = deviceFromPointer(devicePtr);
//     if (dev != currentDevice)
//       HANDLE_CUDA_ERROR(cudaSetDevice(currentDevice));
//   }

//   /// @brief Extract state vector amplitudes from the
//   /// given range.
//   void extractValues(std::complex<ScalarType> *value, std::size_t start,
//                      std::size_t end) const {
//     checkAndSetDevice();
//     HANDLE_CUDA_ERROR(cudaMemcpy(
//         value, reinterpret_cast<std::complex<ScalarType> *>(devicePtr) + start,
//         (end - start) * sizeof(std::complex<ScalarType>),
//         cudaMemcpyDeviceToHost));
//   }

//   /// @brief Return true if the given pointer is a GPU device pointer
//   bool isDevicePointer(void *ptr) const {
//     cudaPointerAttributes attributes;
//     HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
//     return attributes.type > 1;
//   }

//   /// @brief Given a GPU device pointer, get the CUDA device it is on.
//   std::size_t deviceFromPointer(void *ptr) const {
//     cudaPointerAttributes attributes;
//     HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
//     return attributes.device;
//   }

//   /// @brief Internal utility method for computing overlap from
//   /// validated pointer. This method combines common code from
//   /// the `overlap(T*,size_t)` overloads.
//   template <typename T>
//   double internalOverlapVectorImpl(const std::vector<std::complex<T>> &other) {
//     // Cast our data pointer to be compatible with Thrust.
//     auto *castedDevicePtr = reinterpret_cast<ThrustComplex<T> *>(devicePtr);
//     thrust::device_ptr<ThrustComplex<T>> thrustDevPtrABegin(castedDevicePtr);
//     thrust::device_ptr<ThrustComplex<T>> thrustDevPtrAEnd(castedDevicePtr +
//                                                           size);

//     // Here we explicitly copy the data to the GPU
//     thrust::device_vector<ThrustComplex<T>> otherDevPtr(other);
//     return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
//                                  otherDevPtr.begin(), ThrustComplex<T>(0.0),
//                                  thrust::plus<ThrustComplex<T>>(),
//                                  AdotConjB<T>())
//         .real();
//   }

//   /// @brief Internal utility method for computing overlap from
//   /// validated pointer. This method combines common code from
//   /// the `overlap(T*,size_t)` overloads.
//   template <typename T>
//   double internalOverlapPointerImpl(std::complex<T> *other) {
//     // Cast the data to a Thrust compatible type
//     auto *castedOther = reinterpret_cast<ThrustComplex<T> *>(other);
//     auto *castedDevicePtr = reinterpret_cast<ThrustComplex<T> *>(devicePtr);
//     thrust::device_ptr<ThrustComplex<T>> thrustDevPtrABegin(castedDevicePtr);
//     thrust::device_ptr<ThrustComplex<T>> thrustDevPtrAEnd(castedDevicePtr +
//                                                           size);

//     // Check that the other pointer is on GPU device
//     if (!isDevicePointer(other)) {
//       // here we have to copy the data
//       thrust::device_vector<ThrustComplex<T>> otherDevPtr(castedOther,
//                                                           castedOther + size);
//       return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
//                                    otherDevPtr.begin(), ThrustComplex<T>(0.0),
//                                    thrust::plus<ThrustComplex<T>>(),
//                                    AdotConjB<T>())
//           .real();
//     }

//     // We have two device pointers, make sure they are on the same CUDA device
//     if (deviceFromPointer(devicePtr) != deviceFromPointer(other))
//       throw std::runtime_error("[custatevec-state] overlap requested for "
//                                "device pointers on separate GPU devices.");

//     // Compute the overlap
//     thrust::device_ptr<ThrustComplex<T>> thrustDevPtrBBegin(castedOther);
//     return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
//                                  &thrustDevPtrBBegin[0], ThrustComplex<T>(0.0),
//                                  thrust::plus<ThrustComplex<T>>(),
//                                  AdotConjB<T>())
//         .real();
//   }

// public:
//   CusvState(std::size_t s, void *ptr) : size(s), devicePtr(ptr) {}

//   /// @brief Return the number of qubits this state models
//   std::size_t getNumQubits() const override { return std::log2(size); }

//   /// @brief Return the shape of the data.
//   std::vector<std::size_t> getDataShape() const override { return {size}; }

//   /// @brief Compute the overlap of this state with the provided one.
//   /// If the other state is not on GPU device, this function will
//   /// copy the data from host.
//   double overlap(const cudaq::SimulationState &other) override {
//     if (getDataShape() != other.getDataShape())
//       throw std::runtime_error("[custatevec-state] overlap error - other state "
//                                "dimension not equal to this state dimension.");

//     if (other.getPrecision() != getPrecision()) {
//       throw std::runtime_error(
//           "[custatevec-state] overlap error - precision mismatch.");
//     }

//     // Cast our data pointer to be compatible with Thrust.
//     auto *castedDevicePtr =
//         reinterpret_cast<ThrustComplex<ScalarType> *>(devicePtr);
//     thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrABegin(
//         castedDevicePtr);
//     thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrAEnd(
//         castedDevicePtr + size);

//     // Make sure other is on GPU device already
//     if (isDevicePointer(other.ptr())) {
//       if (deviceFromPointer(devicePtr) != deviceFromPointer(other.ptr()))
//         throw std::runtime_error(
//             "overlap requested for device pointers on separate GPU devices.");
//       // other is a device pointer
//       thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrBBegin(
//           reinterpret_cast<ThrustComplex<ScalarType> *>(other.ptr()));
//       return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
//                                    thrustDevPtrBBegin,
//                                    ThrustComplex<ScalarType>(0.0),
//                                    thrust::plus<ThrustComplex<ScalarType>>(),
//                                    AdotConjB<ScalarType>())
//           .real();
//     }

//     // If we reach here, then we have to copy the data from host.
//     cudaq::info(
//         "[custatevec-state] overlap computation requested with a state that is "
//         "in host memory. Host data will be copied to GPU.");

//     // Cast the other pointer to be compatible with Thrust.
//     auto *castedOtherPtr =
//         reinterpret_cast<std::complex<ScalarType> *>(other.ptr());
//     std::vector<std::complex<ScalarType>> dataAsVec(castedOtherPtr,
//                                                     castedOtherPtr + size);
//     thrust::device_vector<ThrustComplex<ScalarType>> otherDevPtr(dataAsVec);
//     return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
//                                  otherDevPtr.begin(),
//                                  ThrustComplex<ScalarType>(0.0),
//                                  thrust::plus<ThrustComplex<ScalarType>>(),
//                                  AdotConjB<ScalarType>())
//         .real();
//   }

//   /// @brief Compute the overlap of this state with the data provided as a
//   /// `std::vector<double>`. If this device state is not FP64, throw an
//   /// exception. This overload requires an explicit copy from host memory.
//   double overlap(const std::vector<cudaq::complex128> &other) override {
//     // We must use compatible element types
//     if constexpr (std::is_same_v<ScalarType, float>) {
//       throw std::runtime_error("simulation precision is FP32 but overlap "
//                                "requested with FP64 state data.");
//     }

//     // Beyond here, ScalarType can only be == double

//     // Check that our shapes are correct
//     if (getDataShape()[0] != other.size())
//       throw std::runtime_error("[custatevec-state] overlap error - other state "
//                                "dimension not equal to this state dimension.");

//     return internalOverlapVectorImpl<double>(other);
//   }

//   /// @brief Compute the overlap of this state with the data provided as a
//   /// `std::vector<float>`. If this device state is not FP32, throw an
//   /// exception. This overload requires an explicit copy from host memory.
//   double overlap(const std::vector<cudaq::complex64> &other) override {
//     if constexpr (std::is_same_v<ScalarType, double>) {
//       throw std::runtime_error("simulation precision is FP64 but overlap "
//                                "requested with FP32 state data.");
//     }

//     // Beyond here, ScalarType can only be == float

//     // Check that are shapes are correct
//     if (getDataShape()[0] != other.size())
//       throw std::runtime_error("[custatevec-state] overlap error - other state "
//                                "dimension not equal to this state dimension.");

//     return internalOverlapVectorImpl<float>(other);
//   }

//   /// @brief Compute the overlap of this state with the data provided as a raw
//   /// pointer. This overload will check if this pointer corresponds to a device
//   /// pointer. It will copy the data from host to device if necessary.
//   double overlap(cudaq::complex128 *other, std::size_t numElements) override {
//     // Must have the correct precision
//     if constexpr (std::is_same_v<ScalarType, float>) {
//       throw std::runtime_error("simulation precision is FP32 but overlap "
//                                "requested with FP64 state data.");
//     }

//     // Must have the correct number of elements.
//     if (numElements != size)
//       throw std::runtime_error("[custatevec-state] overlap with pointer, "
//                                "invalid number of elements specified.");

//     return internalOverlapPointerImpl<double>(other);
//   }

//   double overlap(cudaq::complex64 *other, std::size_t numElements) override {
//     // Must have the correct precision
//     if constexpr (std::is_same_v<ScalarType, double>) {
//       throw std::runtime_error("simulation precision is FP64 but overlap "
//                                "requested with FP32 state data.");
//     }

//     // Must have the correct number of elements.
//     if (numElements != size)
//       throw std::runtime_error("[custatevec-state] overlap with pointer, "
//                                "invalid number of elements specified.");

//     return internalOverlapPointerImpl<float>(other);
//   }

//   /// @brief Return the vector element at the given index.
//   cudaq::complex128 vectorElement(std::size_t idx) override {
//     std::complex<ScalarType> value;
//     extractValues(&value, idx, idx + 1);
//     return {value.real(), value.imag()};
//   }

//   /// @brief Dump the state to the given output stream
//   void dump(std::ostream &os) const override {
//     // get state data from device to print
//     std::vector<std::complex<ScalarType>> tmp(size);
//     HANDLE_CUDA_ERROR(cudaMemcpy(tmp.data(), devicePtr,
//                                  size * sizeof(std::complex<ScalarType>),
//                                  cudaMemcpyDeviceToHost));
//     for (auto &t : tmp)
//       os << t << "\n";
//   }

//   /// @brief This state is GPU device data, always return true.
//   bool isDeviceData() const override { return true; }

//   /// @brief Copy the state device data to the user-provided host data pointer.
//   void toHost(cudaq::complex128 *userData,
//               std::size_t numElements) const override {
//     // Must have the correct precision
//     if constexpr (std::is_same_v<ScalarType, float>) {
//       throw std::runtime_error("simulation precision is FP32 but overlap "
//                                "requested with FP64 state data.");
//     }
//     // Must have the correct number of elements.
//     if (numElements != size)
//       throw std::runtime_error("[custatevec-state] provided toHost pointer has "
//                                "invalid number of elements specified.");

//     extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
//                   size);
//     return;
//   }

//   /// @brief Copy the state device data to the user-provided host data pointer.
//   void toHost(cudaq::complex64 *userData,
//               std::size_t numElements) const override {
//     // Must have the correct precision
//     if constexpr (std::is_same_v<ScalarType, float>) {
//       throw std::runtime_error("simulation precision is FP32 but overlap "
//                                "requested with FP64 state data.");
//     }
//     // Must have the correct number of elements.
//     if (numElements != size)
//       throw std::runtime_error("[custatevec-state] provided toHost pointer has "
//                                "invalid number of elements specified.");

//     extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
//                   size);
//     return;
//   }

//   /// @brief Return the raw pointer to the device data.
//   void *ptr() const override { return devicePtr; }

//   /// @brief Return the precision of the state data elements.
//   precision getPrecision() const override {
//     if constexpr (std::is_same_v<ScalarType, float>)
//       return cudaq::SimulationState::precision::fp32;
//     return cudaq::SimulationState::precision::fp64;
//   }

//   /// @brief Free the device data.
//   void destroyState() override {
//     cudaq::info("custatevec-state destroying state vector handle.");
//     HANDLE_CUDA_ERROR(cudaFree(devicePtr));
//   }
// };

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
    HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
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
    if (deviceStateVector) {
      HANDLE_ERROR(custatevecDestroy(handle));
      if (ownsDeviceVector)
        HANDLE_CUDA_ERROR(cudaFree(deviceStateVector));
    }
    if (extraWorkspaceSizeInBytes)
      HANDLE_CUDA_ERROR(cudaFree(extraWorkspace));
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
                     const cudaq::spin_op &op) override {
    flushGateQueue();
    cudaq::info(" [cusv decomposing] exp_pauli({}, {})", theta,
                op.to_string(false));
    std::vector<int> controls, targets;
    for (const auto &bit : controlIds)
      controls.emplace_back(static_cast<int>(bit));
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

  std::unique_ptr<cudaq::SimulationState> getSimulationState() override {
    flushGateQueue();
    ownsDeviceVector = false;
    return std::make_unique<cudaq::CusvState<ScalarType>>(stateDimension,
                                                   deviceStateVector);
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
