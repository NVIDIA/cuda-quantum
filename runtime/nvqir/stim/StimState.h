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
#include <stdexcept>

namespace cudaq {

/// @brief Provides stabilizer simulation state representation using StimData.
class StimState : public SimulationState {
public:
  /// @brief Construct from StimData (may copy).
  explicit StimState(const StimData& d) : data_(d.copy()) {}

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
    if (dataType != state_data::variant_type_index<StimData>())
      throw std::runtime_error("[StimState] only supports StimData for initialization.");
    auto stim_data = static_cast<StimData*>(ptr);
    return std::make_unique<StimState>(*stim_data);
  }

public:
  /// @brief This simulator is not array-like (must use Pauli frame/tableau APIs).
  bool isArrayLike() const override { return false; }

  /// @brief Return the number of qubits.
  std::size_t getNumQubits() const override { return data_.num_qubits; }

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
    os << "StimState { qubits=" << data_.num_qubits
       << ", msm_err_count=" << data_.msm_err_count
       << ", current_size=" << data_.current_size << " }";
    // Optionally list the tableau or Pauli frame if desired
    os << "\nTableau X_output:\n";
    for (const auto& row : data_.tableau.x_output) {
      for (bool b : row) os << (b ? '1' : '0');
      os << "\n";
    }
    os << "Tableau Z_output:\n";
    for (const auto& row : data_.tableau.z_output) {
      for (bool b : row) os << (b ? '1' : '0');
      os << "\n";
    }
    os << "PauliFrame X:\n";
    for (bool b : data_.frame.x) os << (b ? '1' : '0');
    os << "\nPauliFrame Z:\n";
    for (bool b : data_.frame.z) os << (b ? '1' : '0');
    os << "\n";
  }

  /// @brief Precision is always double for stabilizer/Stim data.
  precision getPrecision() const override { return precision::fp64; }

  /// @brief Destroy any resources (none needed here).
  void destroyState() override {
    // No-op: All managed by RAII.
  }

  /// @brief Returns a const reference to the tableau (stabilizer generator).
  const StimData::TableauClone& getTableau() const { return data_.tableau; }

  /// @brief Returns a const reference to the Pauli frame.
  const StimData::PauliFrameClone& getPauliFrame() const { return data_.frame; }

  /// @brief Access StimData internals, if needed.
  const StimData& stim_data() const { return data_; }

    void set_tableau(const StimData::TableauClone& t) { data_.set_tableau(t); }
    void set_pauli_frame(const StimData::PauliFrameClone& f) { data_.set_pauli_frame(f); }
    void set_current_size(std::size_t s) { data_.set_current_size(s); }
    void set_msm_err_count(std::size_t c) { data_.set_msm_err_count(c); }
    void set_num_qubits(uint64_t n) { data_.set_num_qubits(n); }

    


private:
  StimData data_;
};

} // namespace cudaq
