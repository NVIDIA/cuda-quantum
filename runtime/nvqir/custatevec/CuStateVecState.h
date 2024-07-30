/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
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

/// @brief CusvState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Custatevec Circuit Simulator. It
/// attempts to keep the simulation data on GPU device and care is taken
/// to ensure operations and comparisons with other states operate
/// on compatible floating point element types.
template <typename ScalarType>
class CusvState : public cudaq::SimulationState {
private:
  /// @brief Size of the state data array on GPU.
  std::size_t size = 0;

  /// @brief State device data pointer.
  void *devicePtr = nullptr;

  /// @brief Flag indicating ownership of the state data.
  bool ownsDevicePtr = true;

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
  int deviceFromPointer(void *ptr) const {
    cudaPointerAttributes attributes;
    HANDLE_CUDA_ERROR(cudaPointerGetAttributes(&attributes, ptr));
    return attributes.device;
  }

  /// @brief Check the input data pointer and if it is
  /// host data, copy it to the GPU.
  auto maybeCopyToDevice(std::size_t size, void *dataPtr) {
    if (isDevicePointer(dataPtr))
      return dataPtr;

    std::complex<ScalarType> *ptr = nullptr;
    HANDLE_CUDA_ERROR(
        cudaMalloc((void **)&ptr, size * sizeof(std::complex<ScalarType>)));
    HANDLE_CUDA_ERROR(cudaMemcpy(ptr, dataPtr,
                                 size * sizeof(std::complex<ScalarType>),
                                 cudaMemcpyHostToDevice));
    return reinterpret_cast<void *>(ptr);
  };

public:
  CusvState(std::size_t s, void *ptr) : size(s), devicePtr(ptr) {}
  CusvState(std::size_t s, void *ptr, bool owns)
      : size(s), devicePtr(ptr), ownsDevicePtr(owns) {}

  /// @brief Return the number of qubits this state models
  std::size_t getNumQubits() const override { return std::log2(size); }

  /// @brief Compute the overlap of this state with the provided one.
  /// If the other state is not on GPU device, this function will
  /// copy the data from host.
  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    if (getTensor().extents != other.getTensor().extents)
      throw std::runtime_error("[custatevec-state] overlap error - other state "
                               "dimension not equal to this state dimension.");

    if (other.getPrecision() != getPrecision()) {
      throw std::runtime_error(
          "[custatevec-state] overlap error - precision mismatch.");
    }

    // It could be that the current set Device is not
    // where the data resides.
    int currentDev;
    cudaGetDevice(&currentDev);
    auto dataDev = deviceFromPointer(devicePtr);
    if (currentDev != dataDev)
      cudaSetDevice(dataDev);

    // Make sure other is on GPU device already
    if (isDevicePointer(other.getTensor().data)) {
      if (deviceFromPointer(devicePtr) !=
          deviceFromPointer(other.getTensor().data))
        throw std::runtime_error(
            "overlap requested for device pointers on separate GPU devices.");

      auto cmplx = nvqir::innerProduct<ScalarType>(
          devicePtr, other.getTensor().data, size, false);
      return std::abs(std::complex<ScalarType>(cmplx.real, cmplx.imaginary));
    } else {
      // If we reach here, then we have to copy the data from host.
      cudaq::info("[custatevec-state] overlap computation requested with a "
                  "state that is "
                  "in host memory. Host data will be copied to GPU.");

      auto cmplx = nvqir::innerProduct<ScalarType>(
          devicePtr, other.getTensor().data, size, true);
      return std::abs(std::complex<ScalarType>(cmplx.real, cmplx.imaginary));
    }
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(
          fmt::format("[custatevec-state] getAmplitude with an invalid number "
                      "of bits in the "
                      "basis state: expected {}, provided {}.",
                      getNumQubits(), basisState.size()));
    if (std::any_of(basisState.begin(), basisState.end(),
                    [](int x) { return x != 0 && x != 1; }))
      throw std::runtime_error(
          "[custatevec-state] getAmplitude with an invalid basis state: only "
          "qubit state (0 or 1) is supported.");

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [](std::size_t acc, int bit) { return (acc << 1) + bit; });
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

  /// @brief Return the device pointer
  const void *getDevicePointer() const { return devicePtr; }

  /// @brief Return the precision of the state data elements.
  precision getPrecision() const override {
    if constexpr (std::is_same_v<ScalarType, float>)
      return cudaq::SimulationState::precision::fp32;

    return cudaq::SimulationState::precision::fp64;
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t type) override {
    // If the data is provided as a pointer / size, then
    // we assume we do not own it.
    bool weOwnTheData = type < 2 ? true : false;
    ptr = maybeCopyToDevice(size, ptr);
    return std::make_unique<CusvState<ScalarType>>(size, ptr, weOwnTheData);
  }

  /// @brief Return the tensor at the given index. Throws
  /// for an invalid tensor index.
  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[cusv-state] invalid tensor requested.");
    return Tensor{devicePtr, std::vector<std::size_t>{size}, getPrecision()};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[cusv-state] invalid tensor requested.");

    if (indices.size() != 1)
      throw std::runtime_error("[cusv-state] invalid element extraction.");

    auto idx = indices[0];
    std::complex<ScalarType> value;
    extractValues(&value, idx, idx + 1);
    return {value.real(), value.imag()};
  }

  /// @brief Copy the state device data to the user-provided host data pointer.
  void toHost(std::complex<double> *userData,
              std::size_t numElements) const override {
    // Must have the correct precision
    if constexpr (std::is_same_v<ScalarType, float>)
      throw std::runtime_error("simulation precision is FP32 but toHost "
                               "requested with FP64 host buffer.");

    // Must have the correct number of elements.
    if (numElements != size)
      throw std::runtime_error("[custatevec-state] provided toHost pointer has "
                               "invalid number of elements specified.");

    extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
                  size);
    return;
  }

  /// @brief Copy the state device data to the user-provided host data pointer.
  void toHost(std::complex<float> *userData,
              std::size_t numElements) const override {
    // Must have the correct precision
    if constexpr (std::is_same_v<ScalarType, double>)
      throw std::runtime_error("simulation precision is FP64 but toHost "
                               "requested with FP32 host buffer.");

    // Must have the correct number of elements.
    if (numElements != size)
      throw std::runtime_error("[custatevec-state] provided toHost pointer has "
                               "invalid number of elements specified.");

    extractValues(reinterpret_cast<std::complex<ScalarType> *>(userData), 0,
                  size);
    return;
  }

  /// @brief Free the device data.
  void destroyState() override {
    if (!ownsDevicePtr)
      return;

    int currentDev;
    cudaGetDevice(&currentDev);
    auto device = deviceFromPointer(devicePtr);
    if (currentDev != device)
      cudaSetDevice(device);

    cudaGetDevice(&currentDev);
    cudaq::info("custatevec-state destroying state vector handle (devicePtr "
                "GPU = {}, currentDevice = {}).",
                device, currentDev);

    HANDLE_CUDA_ERROR(cudaFree(devicePtr));
  }
};

} // namespace cudaq