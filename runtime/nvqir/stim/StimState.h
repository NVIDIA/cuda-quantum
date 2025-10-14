/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2025 NVIDIA Corporation & Affiliates.                         *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "common/SimulationState.h"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <variant>
#include <vector>

namespace cudaq {

/// @brief Provides stabilizer simulation state representation using StimData.
class StimState : public SimulationState, public ClonableState {
private:
  StimData data_;

  template <typename T, typename Variant>
  struct variant_type_index;

  template <typename T, typename... Types>
  struct variant_type_index<T, std::variant<Types...>> {
    static constexpr std::size_t value = []() {
      std::size_t index = 0;
      bool found =
          ((std::is_same<T, Types>::value ? true : (++index, false)) || ...);
      return found ? index : throw "Type not in variant";
    }();
  };

  // Disable copying to prevent shallow copy issues
  StimState(const StimState&) = delete;
  StimState& operator=(const StimState&) = delete;
  
  // Allow moving
  StimState(StimState&&) = delete;
  StimState& operator=(StimState&&) = delete;

public:
  /// @brief Construct from StimData (may copy).
  explicit StimState(const StimData &d) : data_(d) {}

  /// @brief Construct from an rvalue StimData
  explicit StimState(StimData &&d) : data_(std::move(d)) {}

  /// @brief Factory for this type from state_data.
  std::unique_ptr<SimulationState>
  createFromData(const state_data &d) override {
    if (!std::holds_alternative<StimData>(d))
      throw std::runtime_error(
          "[StimState] only supports StimData for initialization.");
    return std::make_unique<StimState>(std::get<StimData>(d));
  }

protected:
  /// @brief Create from data pointer.
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t, void *ptr, std::size_t dataType) override {
    if (dataType != variant_type_index<StimData, state_data>::value)
      throw std::runtime_error(
          "[StimState] only supports StimData for initialization.");

    auto stim_data = static_cast<StimData *>(ptr);
    return std::make_unique<StimState>(*stim_data);
  }

public:
  std::unique_ptr<SimulationState> clone() const override {
    // Note: This performs shallow copy of pointers in StimData
    // Caller must ensure proper lifetime management of underlying data
    return std::make_unique<StimState>(data_);
  }

  /// @brief This simulator is not array-like (must use Pauli frame APIs).
  bool isArrayLike() const override { return false; }

  /// @brief Return the number of qubits.
  std::size_t getNumQubits() const override {
    if (data_.empty() || !data_[0].first || data_[0].second == 0)
      throw std::runtime_error(
          "[StimState] Invalid StimData: missing num_qubits.");
    return *static_cast<const std::size_t *>(data_[0].first);
  }

  /// @brief Get MSM error count.
  std::size_t getMsmErrorCount() const {
    if (data_.size() < 2 || !data_[1].first || data_[1].second == 0)
      throw std::runtime_error(
          "[StimState] Invalid StimData: missing msm_err_count.");
    return *static_cast<const std::size_t *>(data_[1].first);
  }

/// @brief Get batch size (number of shots)
std::size_t getBatchSize() const {
    std::size_t nq = getNumQubits();
    if (data_.size() > 2 && data_[2].first && data_[2].second > 0) {
        return data_[2].second / nq;
    }
    return 0;
}

/// @brief Get X value for a specific shot and qubit
uint8_t getXValue(std::size_t shot, std::size_t qubit) const {
    if (data_.size() <= 2 || !data_[2].first) {
        throw std::runtime_error("[StimState] No X output data");
    }
    std::size_t nq = getNumQubits();
    std::size_t idx = qubit + shot * nq;
    if (idx >= data_[2].second) {
        throw std::out_of_range("[StimState] Index out of bounds");
    }
    const auto* x_data = static_cast<const uint8_t*>(data_[2].first);
    return x_data[idx];
}

/// @brief Get Z value for a specific shot and qubit
uint8_t getZValue(std::size_t shot, std::size_t qubit) const {
    if (data_.size() <= 3 || !data_[3].first) {
        throw std::runtime_error("[StimState] No Z output data");
    }
    std::size_t nq = getNumQubits();
    std::size_t idx = qubit + shot * nq;
    if (idx >= data_[3].second) {
        throw std::out_of_range("[StimState] Index out of bounds");
    }
    const auto* z_data = static_cast<const uint8_t*>(data_[3].first);
    return z_data[idx];
}

  /// @brief Tensor interface not supported for StimState.
  Tensor getTensor(std::size_t idx = 0) const override {
    throw std::runtime_error("[StimState] Tensor interface not supported.");
  }

  std::vector<Tensor> getTensors() const override { return {}; }
  std::size_t getNumTensors() const override { return 0; }

  /// @brief Overlap is not implemented for stabilizer states.
  std::complex<double> overlap(const SimulationState &other) override {
    throw std::runtime_error(
        "[StimState] overlap not implemented for stabilizer data.");
  }

  /// @brief Amplitude access not supported for StimState.
  std::complex<double> getAmplitude(const std::vector<int> &) override {
    throw std::runtime_error(
        "[StimState] amplitudes not supported for stabilizer states.");
  }

void dump(std::ostream &os) const override {
  if (data_.size() < 2) {
    os << "StimState { Invalid/empty data }";
    return;
  }

  os << "StimState { qubits=" << getNumQubits()
     << ", msm_err_count=" << getMsmErrorCount();
  
  // Display batch size if data is available
  std::size_t batch_size = getBatchSize();
  if (batch_size > 0) {
    os << ", batch_size=" << batch_size;
  }
  os << " }\n";

  // Display X and Z output sizes
  if (data_.size() > 2 && data_[2].first) {
    os << "X output size: " << data_[2].second << "\n";
  }
  if (data_.size() > 3 && data_[3].first) {
    os << "Z output size: " << data_[3].second << "\n";
  }

  // Display sample data for first few shots (if available)
  if (batch_size > 0) {
    std::size_t nq = getNumQubits();
    std::size_t shots_to_display = std::min(batch_size, std::size_t(5));
    
    os << "\nSample data (first " << shots_to_display << " shots):\n";
    for (std::size_t shot = 0; shot < shots_to_display; ++shot) {
      os << "Shot " << shot << ": X=[";
      for (std::size_t qubit = 0; qubit < nq; ++qubit) {
        if (qubit > 0) os << " ";
        os << static_cast<int>(getXValue(shot, qubit));
      }
      os << "] Z=[";
      for (std::size_t qubit = 0; qubit < nq; ++qubit) {
        if (qubit > 0) os << " ";
        os << static_cast<int>(getZValue(shot, qubit));
      }
      os << "]\n\n\n";
    }
  }
}


  /// @brief Precision is always double for stabilizer/Stim data.
  precision getPrecision() const override { return precision::fp64; }

  /// @brief Destroy any resources.
  void destroyState() override {       // Free all allocated memory in StimData
      for (auto& [ptr, size] : data_) {
          if (ptr) {
              // Elements 0 and 1 are single std::size_t values
              if (&ptr == &data_[0].first || &ptr == &data_[1].first) {
                  delete static_cast<std::size_t*>(ptr);
              }
              // Elements 2 and 3 are uint8_t arrays
              else if ((&ptr == &data_[2].first || &ptr == &data_[3].first) && size > 0) {
                  delete[] static_cast<uint8_t*>(ptr);
              }
          }
      }
      std::cout << "StimState destroyed and memory freed.\n"; }
};

} // namespace cudaq