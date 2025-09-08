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
#include <variant>
#include <type_traits>
#include <stdexcept>
#include <vector>

namespace cudaq {

/// @brief Provides stabilizer simulation state representation using StimData.
class StimState : public SimulationState {
public:
  /// @brief Helper structure to represent a tableau row with proper interface
  struct TableauRow {
    const void* data_ptr;
    std::size_t num_qubits;
    std::size_t row_size;

    TableauRow(const void* ptr, std::size_t nq, std::size_t rs) 
      : data_ptr(ptr), num_qubits(nq), row_size(rs) {}

    // Helper methods to access bits - assumes data is stored as bytes/bits
    bool getBit(std::size_t index) const {
      if (index >= row_size) return false;
      const auto* bytes = static_cast<const uint8_t*>(data_ptr);
      return (bytes[index / 8] >> (index % 8)) & 1;
    }
  };

  /// @brief Helper structure to represent the full tableau
  struct TableauWrapper {
    std::vector<TableauRow> x_output;
    std::vector<TableauRow> z_output;
    std::size_t num_qubits;

    TableauWrapper(const StimData& data, std::size_t nq) : num_qubits(nq) {
      // Build x_output rows
      if (data.size() > 3 && data[3].first && data[3].second > 0) {
        const auto* x_data = static_cast<const uint8_t*>(data[3].first);
        std::size_t x_size = data[3].second;
        
        // Assuming each row is stored sequentially in the x_output array
        // Row size includes stabilizers + destabilizers + phase bits
        std::size_t row_size = (2 * nq + 1 + 7) / 8; // Round up to byte boundary
        std::size_t num_rows = x_size / row_size;
        
        for (std::size_t i = 0; i < num_rows; ++i) {
          x_output.emplace_back(x_data + i * row_size, nq, row_size * 8);
        }
      }

      // Build z_output rows
      if (data.size() > 4 && data[4].first && data[4].second > 0) {
        const auto* z_data = static_cast<const uint8_t*>(data[4].first);
        std::size_t z_size = data[4].second;
        
        std::size_t row_size = (2 * nq + 1 + 7) / 8; // Round up to byte boundary
        std::size_t num_rows = z_size / row_size;
        
        for (std::size_t i = 0; i < num_rows; ++i) {
          z_output.emplace_back(z_data + i * row_size, nq, row_size * 8);
        }
      }
    }
  };

  /// @brief Helper structure to represent Pauli frame
  struct PauliFrameWrapper {
    const void* x_data;
    const void* z_data;
    std::size_t num_qubits;
    std::size_t total_size;

    PauliFrameWrapper(const StimData& data, std::size_t nq) : num_qubits(nq) {
      if (data.size() > 5 && data[5].first && data[5].second > 0) {
        const auto* frame_data = static_cast<const uint8_t*>(data[5].first);
        total_size = data[5].second;
        
        // Assuming frame is stored as [x_bits, z_bits] consecutively
        std::size_t x_size = (nq + 7) / 8; // Round up to byte boundary
        x_data = frame_data;
        z_data = frame_data + x_size;
      } else {
        x_data = nullptr;
        z_data = nullptr;
        total_size = 0;
      }
    }

    // Access methods for x and z components
    std::vector<bool> getXBits() const {
      std::vector<bool> result(num_qubits, false);
      if (x_data) {
        const auto* bytes = static_cast<const uint8_t*>(x_data);
        for (std::size_t i = 0; i < num_qubits; ++i) {
          result[i] = (bytes[i / 8] >> (i % 8)) & 1;
        }
      }
      return result;
    }

    std::vector<bool> getZBits() const {
      std::vector<bool> result(num_qubits, false);
      if (z_data) {
        const auto* bytes = static_cast<const uint8_t*>(z_data);
        for (std::size_t i = 0; i < num_qubits; ++i) {
          result[i] = (bytes[i / 8] >> (i % 8)) & 1;
        }
      }
      return result;
    }
  };

private:
  StimData data_;
  mutable std::unique_ptr<TableauWrapper> cached_tableau_;
  mutable std::unique_ptr<PauliFrameWrapper> cached_frame_;

  template <typename T, typename Variant>
  struct variant_type_index;

  template <typename T, typename... Types>
  struct variant_type_index<T, std::variant<Types...>> {
      static constexpr std::size_t value = []() {
          std::size_t index = 0;
          bool found = ((std::is_same<T, Types>::value ? true : (++index, false)) || ...);
          return found ? index : throw "Type not in variant";
      }();
  };

public:
  /// @brief Construct from StimData (may copy).
  explicit StimState(const StimData& d) : data_(d) {}

  /// @brief Construct from an rvalue StimData
  explicit StimState(StimData&& d) : data_(std::move(d)) {}

  /// @brief Factory for this type from state_data.
  std::unique_ptr<SimulationState>
  createFromData(const state_data& d) override {
    if (!std::holds_alternative<StimData>(d))
      throw std::runtime_error("[StimState] only supports StimData for initialization.");
    return std::make_unique<StimState>(std::get<StimData>(d));
  }

protected:
  /// @brief Create from data pointer.
  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t, void* ptr, std::size_t dataType) override {
  if (dataType != variant_type_index<StimData, state_data>::value)
    throw std::runtime_error("[StimState] only supports StimData for initialization.");

    auto stim_data = static_cast<StimData*>(ptr);
    return std::make_unique<StimState>(*stim_data);
  }

public:
  /// @brief This simulator is not array-like (must use Pauli frame/tableau APIs).
  bool isArrayLike() const override { return false; }

  /// @brief Return the number of qubits.
  std::size_t getNumQubits() const override { 
    if (data_.empty() || data_[0].second == 0)
      throw std::runtime_error("[StimState] Invalid StimData: missing num_qubits.");
    return *static_cast<std::size_t*>(data_[0].first);
  }

  /// @brief Tensor interface not supported for StimState.
  Tensor getTensor(std::size_t idx = 0) const override {
    throw std::runtime_error("[StimState] Tensor interface not supported.");
  }

  std::vector<Tensor> getTensors() const override { return {}; }
  std::size_t getNumTensors() const override { return 0; }

  /// @brief Overlap is not implemented for stabilizer states.
  std::complex<double> overlap(const SimulationState& other) override {
    throw std::runtime_error("[StimState] overlap not implemented for stabilizer data.");
  }

  /// @brief Amplitude access not supported for StimState.
  std::complex<double> getAmplitude(const std::vector<int>&) override {
    throw std::runtime_error("[StimState] amplitudes not supported for stabilizer states.");
  }

  /// @brief Dump stabilizer state summary.
  void dump(std::ostream &os) const override {
    if (data_.size() < 3) {
      os << "StimState { Invalid/empty data }";
      return;
    }

    auto num_qubits = getNumQubits();
    auto msm_err_count = getMsmErrorCount();
    auto num_stabilizers = getNumStabilizers();

    os << "StimState { qubits=" << num_qubits
       << ", msm_err_count=" << msm_err_count
       << ", num_stabilizers=" << num_stabilizers << " }";

    const auto& tableau = getTableau();
    os << "\nTableau X_output rows: " << tableau.x_output.size();
    os << "\nTableau Z_output rows: " << tableau.z_output.size();
    
    const auto& frame = getPauliFrame();
    os << "\nPauli frame size: " << frame.total_size;
    os << "\n";
  }

  /// @brief Precision is always double for stabilizer/Stim data.
  precision getPrecision() const override { return precision::fp64; }

  /// @brief Destroy any resources.
  void destroyState() override {
    cached_tableau_.reset();
    cached_frame_.reset();
  }

  /// @brief Get MSM error count.
  std::size_t getMsmErrorCount() const {
    if (data_.size() < 2 || data_[1].second == 0)
      throw std::runtime_error("[StimState] Invalid StimData: missing msm_err_count.");
    return *static_cast<std::size_t*>(data_[1].first);
  }

  /// @brief Get number of stabilizers.
  std::size_t getNumStabilizers() const {
    if (data_.size() < 3 || data_[2].second == 0)
      throw std::runtime_error("[StimState] Invalid StimData: missing num_stabilizers.");
    return *static_cast<std::size_t*>(data_[2].first);
  }

  /// @brief Returns a reference to the tableau (stabilizer generator).
  const TableauWrapper& getTableau() const {
    if (!cached_tableau_) {
      cached_tableau_ = std::make_unique<TableauWrapper>(data_, getNumQubits());
    }
    return *cached_tableau_;
  }

  /// @brief Returns a reference to the Pauli frame.
  const PauliFrameWrapper& getPauliFrame() const {
    if (!cached_frame_) {
      cached_frame_ = std::make_unique<PauliFrameWrapper>(data_, getNumQubits());
    }
    return *cached_frame_;
  }

  /// @brief Clone this state.
  std::unique_ptr<SimulationState> clone() const {
    // Since StimData is now just pointers, we need to be careful about copying
    // For a deep copy, the caller would need to manage the actual data copying
    return std::make_unique<StimState>(data_);
  }
};

} // namespace cudaq