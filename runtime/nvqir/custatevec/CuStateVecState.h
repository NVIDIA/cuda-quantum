/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/SimulationState.h"

#include <thrust/complex.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/inner_product.h>

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

namespace cudaq {
template <typename T>
using ThrustComplex = thrust::complex<T>;

/// @brief Custom functor for the thrust inner product.
template <typename T>
struct AdotConjB
    : public thrust::binary_function<ThrustComplex<T>, ThrustComplex<T>,
                                     ThrustComplex<T>> {
  __host__ __device__ ThrustComplex<T> operator()(ThrustComplex<T> a,
                                                  ThrustComplex<T> b) {
    return a * thrust::conj(b);
  };
};

/// @brief CusvState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Custatevec Circuit Simulator. It
/// attempts to keep the simulation data on GPU device and care is taken
/// to ensure operations and comparisons with other states operate
/// on compatible floating point element types.
template <typename ScalarType>
struct CusvState : public cudaq::SimulationState {
private:
  /// @brief Size of the state data array on GPU.
  std::size_t size = 0;

  /// @brief State device data pointer.
  void *devicePtr = nullptr;

  /// @brief Check that we are currently
  /// using the correct CUDA device, set it
  /// to the correct one if not
  void checkAndSetDevice() const {
    int dev = 0;
    HANDLE_CUDA_ERROR(cudaGetDevice(&dev));
    auto currentDevice = deviceFromPointer(devicePtr);
    if (dev != currentDevice)
      HANDLE_CUDA_ERROR(cudaSetDevice(currentDevice));
  }

  /// @brief Extract state vector amplitudes from the
  /// given range.
  void extractValues(std::complex<ScalarType> *value, std::size_t start,
                     std::size_t end) const {
    checkAndSetDevice();
    HANDLE_CUDA_ERROR(cudaMemcpy(
        value, reinterpret_cast<std::complex<ScalarType> *>(devicePtr) + start,
        (end - start) * sizeof(std::complex<ScalarType>),
        cudaMemcpyDeviceToHost));
  }

  /// @brief Return true if the given pointer is a GPU device pointer
  bool isDevicePointer(void *ptr) const {
    cudaPointerAttributes attributes;
    HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
    return attributes.type > 1;
  }

  /// @brief Given a GPU device pointer, get the CUDA device it is on.
  std::size_t deviceFromPointer(void *ptr) const {
    cudaPointerAttributes attributes;
    HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
    return attributes.device;
  }

  /// @brief Internal utility method for computing overlap from
  /// validated pointer. This method combines common code from
  /// the `overlap(T*,size_t)` overloads.
  template <typename T>
  double internalOverlapVectorImpl(const std::vector<std::complex<T>> &other) {
    // Cast our data pointer to be compatible with Thrust.
    auto *castedDevicePtr = reinterpret_cast<ThrustComplex<T> *>(devicePtr);
    thrust::device_ptr<ThrustComplex<T>> thrustDevPtrABegin(castedDevicePtr);
    thrust::device_ptr<ThrustComplex<T>> thrustDevPtrAEnd(castedDevicePtr +
                                                          size);

    // Here we explicitly copy the data to the GPU
    thrust::device_vector<ThrustComplex<T>> otherDevPtr(other);
    return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
                                 otherDevPtr.begin(), ThrustComplex<T>(0.0),
                                 thrust::plus<ThrustComplex<T>>(),
                                 AdotConjB<T>())
        .real();
  }

  /// @brief Internal utility method for computing overlap from
  /// validated pointer. This method combines common code from
  double internalOverlapPointerImpl(void *other) {
    // Cast the data to a Thrust compatible type
    auto *castedOther = reinterpret_cast<ThrustComplex<ScalarType> *>(other);
    auto *castedDevicePtr =
        reinterpret_cast<ThrustComplex<ScalarType> *>(devicePtr);
    thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrABegin(
        castedDevicePtr);
    thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrAEnd(
        castedDevicePtr + size);

    // Check that the other pointer is on GPU device
    if (!isDevicePointer(other)) {
      // here we have to copy the data
      thrust::device_vector<ThrustComplex<ScalarType>> otherDevPtr(
          castedOther, castedOther + size);
      return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
                                   otherDevPtr.begin(),
                                   ThrustComplex<ScalarType>(0.0),
                                   thrust::plus<ThrustComplex<ScalarType>>(),
                                   AdotConjB<ScalarType>())
          .real();
    }

    // We have two device pointers, make sure they are on the same CUDA device
    if (deviceFromPointer(devicePtr) != deviceFromPointer(other))
      throw std::runtime_error("[custatevec-state] overlap requested for "
                               "device pointers on separate GPU devices.");

    // Compute the overlap
    thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrBBegin(
        castedOther);
    return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
                                 &thrustDevPtrBBegin[0],
                                 ThrustComplex<ScalarType>(0.0),
                                 thrust::plus<ThrustComplex<ScalarType>>(),
                                 AdotConjB<ScalarType>())
        .real();
  }

public:
  CusvState(std::size_t s, void *ptr) : size(s), devicePtr(ptr) {}

  /// @brief Return the number of qubits this state models
  std::size_t getNumQubits() const override { return std::log2(size); }

  /// @brief Return the shape of the data.
  std::vector<std::size_t> getDataShape() const override { return {size}; }

  /// @brief Compute the overlap of this state with the provided one.
  /// If the other state is not on GPU device, this function will
  /// copy the data from host.
  double overlap(const cudaq::SimulationState &other) override {
    if (getDataShape() != other.getDataShape())
      throw std::runtime_error("[custatevec-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    if (other.getPrecision() != getPrecision()) {
      throw std::runtime_error(
          "[custatevec-state] overlap error - precision mismatch.");
    }

    // Cast our data pointer to be compatible with Thrust.
    auto *castedDevicePtr =
        reinterpret_cast<ThrustComplex<ScalarType> *>(devicePtr);
    thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrABegin(
        castedDevicePtr);
    thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrAEnd(
        castedDevicePtr + size);

    // Make sure other is on GPU device already
    if (isDevicePointer(other.ptr())) {
      if (deviceFromPointer(devicePtr) != deviceFromPointer(other.ptr()))
        throw std::runtime_error(
            "overlap requested for device pointers on separate GPU devices.");
      // other is a device pointer
      thrust::device_ptr<ThrustComplex<ScalarType>> thrustDevPtrBBegin(
          reinterpret_cast<ThrustComplex<ScalarType> *>(other.ptr()));
      return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
                                   thrustDevPtrBBegin,
                                   ThrustComplex<ScalarType>(0.0),
                                   thrust::plus<ThrustComplex<ScalarType>>(),
                                   AdotConjB<ScalarType>())
          .real();
    }

    // If we reach here, then we have to copy the data from host.
    cudaq::info(
        "[custatevec-state] overlap computation requested with a state that is "
        "in host memory. Host data will be copied to GPU.");

    // Cast the other pointer to be compatible with Thrust.
    auto *castedOtherPtr =
        reinterpret_cast<std::complex<ScalarType> *>(other.ptr());
    std::vector<std::complex<ScalarType>> dataAsVec(castedOtherPtr,
                                                    castedOtherPtr + size);
    thrust::device_vector<ThrustComplex<ScalarType>> otherDevPtr(dataAsVec);
    return thrust::inner_product(thrustDevPtrABegin, thrustDevPtrAEnd,
                                 otherDevPtr.begin(),
                                 ThrustComplex<ScalarType>(0.0),
                                 thrust::plus<ThrustComplex<ScalarType>>(),
                                 AdotConjB<ScalarType>())
        .real();
  }

  /// @brief Compute the overlap of this state with the data provided as a
  /// `std::vector<double>`. If this device state is not FP64, throw an
  /// exception. This overload requires an explicit copy from host memory.
  double overlap(const std::vector<cudaq::complex128> &other) override {
    // We must use compatible element types
    if constexpr (std::is_same_v<ScalarType, float>) {
      throw std::runtime_error("simulation precision is FP32 but overlap "
                               "requested with FP64 state data.");
    }

    // Beyond here, ScalarType can only be == double

    // Check that our shapes are correct
    if (getDataShape()[0] != other.size())
      throw std::runtime_error("[custatevec-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    return internalOverlapVectorImpl<double>(other);
  }

  /// @brief Compute the overlap of this state with the data provided as a
  /// `std::vector<float>`. If this device state is not FP32, throw an
  /// exception. This overload requires an explicit copy from host memory.
  double overlap(const std::vector<cudaq::complex64> &other) override {
    if constexpr (std::is_same_v<ScalarType, double>) {
      throw std::runtime_error("simulation precision is FP64 but overlap "
                               "requested with FP32 state data.");
    }

    // Beyond here, ScalarType can only be == float

    // Check that are shapes are correct
    if (getDataShape()[0] != other.size())
      throw std::runtime_error("[custatevec-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    return internalOverlapVectorImpl<float>(other);
  }

  /// @brief Compute the overlap of this state with the data provided as a raw
  /// pointer. This overload will check if this pointer corresponds to a device
  /// pointer. It will copy the data from host to device if necessary.
  double overlap(cudaq::complex128 *other, std::size_t numElements) override {
    // Must have the correct precision
    if constexpr (std::is_same_v<ScalarType, float>) {
      throw std::runtime_error("simulation precision is FP32 but overlap "
                               "requested with FP64 state data.");
    }

    // Must have the correct number of elements.
    if (numElements != size)
      throw std::runtime_error("[custatevec-state] overlap with pointer, "
                               "invalid number of elements specified.");
    return internalOverlapPointerImpl(other);
  }

  double overlap(cudaq::complex64 *other, std::size_t numElements) override {
    // Must have the correct precision
    if constexpr (std::is_same_v<ScalarType, double>) {
      throw std::runtime_error("simulation precision is FP64 but overlap "
                               "requested with FP32 state data.");
    }

    // Must have the correct number of elements.
    if (numElements != size)
      throw std::runtime_error("[custatevec-state] overlap with pointer, "
                               "invalid number of elements specified.");

    return internalOverlapPointerImpl(other);
  }

  /// @brief Return the vector element at the given index.
  cudaq::complex128 vectorElement(std::size_t idx) override {
    std::complex<ScalarType> value;
    extractValues(&value, idx, idx + 1);
    return {value.real(), value.imag()};
  }

  /// @brief Dump the state to the given output stream
  void dump(std::ostream &os) const override {
    // get state data from device to print
    std::vector<std::complex<ScalarType>> tmp(size);
    HANDLE_CUDA_ERROR(cudaMemcpy(tmp.data(), devicePtr,
                                 size * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyDeviceToHost));
    for (auto &t : tmp)
      os << t << "\n";
  }

  /// @brief This state is GPU device data, always return true.
  bool isDeviceData() const override { return true; }

  /// @brief Copy the state device data to the user-provided host data pointer.
  void toHost(cudaq::complex128 *userData,
              std::size_t numElements) const override {
    // Must have the correct precision
    if constexpr (std::is_same_v<ScalarType, float>) {
      throw std::runtime_error("simulation precision is FP32 but overlap "
                               "requested with FP64 state data.");
    }
    // Must have the correct number of elements.
    if (numElements != size)
      throw std::runtime_error("[custatevec-state] provided toHost pointer has "
                               "invalid number of elements specified.");

    extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
                  size);
    return;
  }

  /// @brief Copy the state device data to the user-provided host data pointer.
  void toHost(cudaq::complex64 *userData,
              std::size_t numElements) const override {
    // Must have the correct precision
    if constexpr (std::is_same_v<ScalarType, double>) {
      throw std::runtime_error("simulation precision is FP32 but overlap "
                               "requested with FP64 state data.");
    }
    // Must have the correct number of elements.
    if (numElements != size)
      throw std::runtime_error("[custatevec-state] provided toHost pointer has "
                               "invalid number of elements specified.");

    extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
                  size);
    return;
  }

  /// @brief Return the raw pointer to the device data.
  void *ptr() const override { return devicePtr; }

  /// @brief Return the precision of the state data elements.
  precision getPrecision() const override {
    if constexpr (std::is_same_v<ScalarType, float>)
      return cudaq::SimulationState::precision::fp32;
    return cudaq::SimulationState::precision::fp64;
  }

  /// @brief Free the device data.
  void destroyState() override {
    cudaq::info("custatevec-state destroying state vector handle.");
    HANDLE_CUDA_ERROR(cudaFree(devicePtr));
  }
};

} // namespace cudaq