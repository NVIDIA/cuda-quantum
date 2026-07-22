/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

// [Begin Documentation]
#include "nvqir/CircuitSimulator.h"

#include <cmath>
#include <complex>
#include <random>
#include <vector>

namespace {

/// @brief A minimal, from-scratch, dense state-vector simulator. It exists to
/// demonstrate the `CircuitSimulatorBase` extension points; it is not
/// optimized and should not be used for anything beyond a handful of qubits.
class MySimulator : public nvqir::CircuitSimulatorBase<double> {

  /// @brief The dense state vector. Qubit `i` always occupies bit `i` of the
  /// basis-state index, i.e. amplitude index `idx` describes the qubits with
  /// qubit `i` in state `(idx >> i) & 1`.
  std::vector<std::complex<double>> state;

  /// @brief Simulator-local random engine, reseedable via `setRandomSeed`.
  std::mt19937_64 rng{std::random_device{}()};

protected:
  /// @brief Grow the state by one qubit, initialized to |0>. New qubits are
  /// always appended as the new highest-order bit.
  void addQubitToState() override {
    std::vector<std::complex<double>> grown(stateDimension, {0.0, 0.0});
    if (state.empty())
      grown[0] = 1.0;
    else
      std::copy(state.begin(), state.end(), grown.begin());
    state = std::move(grown);
  }

  /// @brief Clear the state entirely (invoked when all qubits are
  /// deallocated).
  void deallocateStateImpl() override { state.clear(); }

  /// @brief Apply `task.matrix` to `task.targets`, conditional on all of
  /// `task.controls` being `|1>`. This recomputes every amplitude from
  /// scratch against the *previous* state (rather than updating `state` in
  /// place) -- it does some redundant work, but there is no risk of reading
  /// an amplitude that this same call already overwrote.
  void applyGate(const GateApplicationTask &task) override {
    const std::size_t numTargets = task.targets.size();
    const std::size_t blockDim = 1ULL << numTargets;

    std::size_t controlMask = 0;
    for (auto c : task.controls)
      controlMask |= (1ULL << c);
    std::size_t targetMask = 0;
    for (auto t : task.targets)
      targetMask |= (1ULL << t);

    std::vector<std::complex<double>> newState(state.size());
    for (std::size_t idx = 0; idx < state.size(); idx++) {
      // Controls not satisfied: this amplitude is untouched.
      if ((idx & controlMask) != controlMask) {
        newState[idx] = state[idx];
        continue;
      }

      // idx's bits, with the target bits stripped out, identify which
      // "row block" of the full state this amplitude belongs to.
      const std::size_t base = idx & ~targetMask;
      // idx's target bits (repacked into 0..numTargets-1) select the row
      // of `task.matrix` this amplitude corresponds to.
      std::size_t row = 0;
      for (std::size_t b = 0; b < numTargets; b++)
        if ((idx >> task.targets[b]) & 1)
          row |= (1ULL << b);

      std::complex<double> acc{0.0, 0.0};
      for (std::size_t col = 0; col < blockDim; col++) {
        std::size_t full = base;
        for (std::size_t b = 0; b < numTargets; b++)
          if ((col >> b) & 1)
            full |= (1ULL << task.targets[b]);
        acc += task.matrix[row * blockDim + col] * state[full];
      }
      newState[idx] = acc;
    }
    state = std::move(newState);
  }

  /// @brief Measure the qubit, collapsing and re-normalizing the state.
  bool measureQubit(const std::size_t qubitIdx) override {
    const std::size_t mask = 1ULL << qubitIdx;
    double probOne = 0.0;
    for (std::size_t idx = 0; idx < state.size(); idx++)
      if (idx & mask)
        probOne += std::norm(state[idx]);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    const bool result = dist(rng) < probOne;

    double norm = 0.0;
    for (std::size_t idx = 0; idx < state.size(); idx++) {
      const bool bitIsSet = idx & mask;
      if (bitIsSet != result)
        state[idx] = {0.0, 0.0};
      else
        norm += std::norm(state[idx]);
    }
    const double scale = 1.0 / std::sqrt(norm);
    for (auto &amplitude : state)
      amplitude *= scale;

    return result;
  }

public:
  MySimulator() { summaryData.name = name(); }
  virtual ~MySimulator() = default;

  /// @brief Reseed this simulator's random engine.
  void setRandomSeed(std::size_t seed) override { rng.seed(seed); }

  /// @brief Reset a single qubit's state to |0> in place.
  void resetQubit(const std::size_t qubitIdx) override {
    flushGateQueue();
    bool measured = measureQubit(qubitIdx);
    if (measured) {
      const std::size_t mask = 1ULL << qubitIdx;
      for (std::size_t idx = 0; idx < state.size(); idx++) {
        if ((idx & mask) == 0) {
          std::swap(state[idx], state[idx | mask]);
        }
      }
    }
  }

  /// @brief Reset the full state back to |0...0>.
  void setToZeroState() override {
    state.assign(stateDimension, {0.0, 0.0});
    state[0] = 1.0;
  }

  /// @brief Draw `shots` samples of `qubits` from the current (uncollapsed)
  /// state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qubits,
                                const int shots,
                                bool includeSequentialData = true) override {
    std::unordered_map<std::string, std::size_t> tally;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int shot = 0; shot < shots; shot++) {
      const double r = dist(rng);
      double cumulative = 0.0;
      std::size_t chosen = state.size() - 1;
      for (std::size_t idx = 0; idx < state.size(); idx++) {
        cumulative += std::norm(state[idx]);
        if (r <= cumulative) {
          chosen = idx;
          break;
        }
      }
      std::string bits;
      for (auto q : qubits)
        bits += ((chosen >> q) & 1) ? '1' : '0';
      tally[bits]++;
    }

    cudaq::ExecutionResult result;
    for (auto &[bits, count] : tally)
      result.appendResult(bits, count);
    return result;
  }

  /// @brief This must be the same name used with `nvq++ --target NAME` (or
  /// `cudaq.set_target('NAME')` in Python).
  std::string name() const override { return "MySimulator"; }

  /// @brief This example does not support constructing a simulator-specific
  /// `SimulationState` from user-supplied data; a simulator that does should
  /// implement its own `cudaq::SimulationState` subtype and return it here.
  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &) override {
    throw std::runtime_error(
        "MySimulator does not support constructing a state from data.");
  }

  /// @brief Generates `clone()`, which NVQIR uses to hand each execution
  /// thread its own simulator instance.
  NVQIR_SIMULATOR_CLONE_IMPL(MySimulator)
};

} // namespace

/// Register this simulator with NVQIR.
NVQIR_REGISTER_SIMULATOR(MySimulator, MySimulator)
// [End Documentation]
