/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#include "common/FmtCore.h"
#include "common/Logger.h"
#include "cudaq/operators.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/utils/cudaq_utils.h"
#include "qpp.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

namespace cudaq {

struct PhotonicsState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  /// @brief The qudit-levels (`qumodes`)
  std::size_t levels;

  PhotonicsState(qpp::ket &&data, std::size_t lvl)
      : state(std::move(data)), levels(lvl) {}

  /// TODO: Rename the API to be generic
  std::size_t getNumQubits() const override {
    return (std::log2(state.size()) / std::log2(levels));
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    throw "not supported for this photonics simulator";
  }

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override {
    if (getNumQubits() != basisState.size())
      throw std::runtime_error(fmt::format(
          "[photonics] getAmplitude with an invalid number of bits in the "
          "basis state: expected {}, provided {}.",
          getNumQubits(), basisState.size()));

    // Convert the basis state to an index value
    const std::size_t idx = std::accumulate(
        std::make_reverse_iterator(basisState.end()),
        std::make_reverse_iterator(basisState.begin()), 0ull,
        [&](std::size_t acc, int qudit) { return (acc * levels) + qudit; });
    return state[idx];
  }

  Tensor getTensor(std::size_t tensorIdx = 0) const override {
    if (tensorIdx != 0)
      throw std::runtime_error("[photonics] invalid tensor requested.");
    return Tensor{
        reinterpret_cast<void *>(
            const_cast<std::complex<double> *>(state.data())),
        std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
        getPrecision()};
  }

  /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override {
    if (tensorIdx != 0)
      throw std::runtime_error("[photonics] invalid tensor requested.");
    if (indices.size() != 1)
      throw std::runtime_error("[photonics] invalid element extraction.");
    return state[indices[0]];
  }

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw "not supported for this photonics simulator";
  }

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
};

/// @brief The `PhotonicsExecutionManager` implements allocation, deallocation,
/// and quantum instruction application for the photonics execution manager.
class PhotonicsExecutionManager : public cudaq::BasicExecutionManager {
private:
  /// @brief Current state
  qpp::ket state;

  /// @brief The qudit-levels (`qumodes`)
  std::size_t levels;

  /// @brief Instructions are stored in a map
  std::unordered_map<std::string, std::function<void(const Instruction &)>>
      instructions;

  /// @brief Qudits to be sampled
  std::vector<cudaq::QuditInfo> sampleQudits;

protected:
  /// @brief Qudit allocation method: a zeroState is first initialized, the
  /// following ones are added via kron operators
  void allocateQudit(const cudaq::QuditInfo &q) override {
    if (state.size() == 0) {
      // qubit will give [1,0], qutrit will give [1,0,0] and so on...
      state = qpp::ket::Zero(q.levels);
      state(0) = 1.0;
      levels = q.levels;
      return;
    }

    qpp::ket zeroState = qpp::ket::Zero(q.levels);
    zeroState(0) = 1.0;
    state = qpp::kron(state, zeroState);
  }

  /// @brief Allocate a set of `qudits` (`qumodes`) with a single call.
  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    for (auto &q : qudits)
      allocateQudit(q);
  }

  void initializeState(const std::vector<cudaq::QuditInfo> &targets,
                       const void *state,
                       simulation_precision precision) override {
    throw std::runtime_error("initializeState not implemented.");
  }

  virtual void initializeState(const std::vector<QuditInfo> &targets,
                               const SimulationState *state) override {
    throw std::runtime_error("initializeState not implemented.");
  }

  /// @brief Qudit deallocation method
  void deallocateQudit(const cudaq::QuditInfo &q) override {}

  /// @brief Deallocate a set of `qudits` (`qumodes`) with a single call.
  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {}

  /// @brief Handler for when the photonics execution context changes
  void handleExecutionContextChanged() override {
    if (!executionContext)
      throw std::runtime_error(
          "Execution context is not set for the PhotonicsExecutionManager.");

    if (!(executionContext->name == "sample" ||
          executionContext->name == "extract-state" ||
          executionContext->name == "tracer"))
      throw std::runtime_error(executionContext->name +
                               " is not supported on this target");
  }

  /// @brief Handler for when the current execution context has ended. It
  /// returns samples to the execution context if it is "sample".
  void handleExecutionContextEnded() override {
    if (executionContext) {
      std::vector<std::size_t> ids;
      for (auto &s : sampleQudits) {
        ids.push_back(s.id);
      }
      if (executionContext->name == "sample") {
        CUDAQ_INFO("Sampling");
        auto shots = executionContext->shots;
        auto sampleResult =
            qpp::sample(shots, state, ids, sampleQudits.begin()->levels);
        cudaq::ExecutionResult counts;
        for (auto [result, count] : sampleResult) {
          std::stringstream bitstring;
          for (const auto &quditRes : result) {
            bitstring << quditRes;
          }
          // Add to the sample result
          // in mid-circ sampling mode this will append 1 bitstring
          counts.appendResult(bitstring.str(), count);
          // Reset the string.
          bitstring.str("");
          bitstring.clear();
        }
        executionContext->result.append(counts);
      } else if (executionContext->name == "extract-state") {
        CUDAQ_INFO("Extracting state");
        // If here, then we care about the result qudit, so compute it.
        for (auto &q : sampleQudits) {
          const auto measurement_tuple = qpp::measure(
              state, qpp::cmat::Identity(q.levels, q.levels), {q.id},
              /*qudit dimension=*/q.levels, /*destructive measmt=*/false);
          const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
          const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
          const auto &collapsed_state = post_meas_states[measurement_result];
          state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                             collapsed_state.size());
        }

        executionContext->simulationState =
            std::make_unique<cudaq::PhotonicsState>(std::move(state), levels);
      }
      // Reset the state and qudits
      state.resize(0);
      sampleQudits.clear();
    }
  }

  /// @brief Method for executing instructions.
  void executeInstruction(const Instruction &instruction) override {
    auto operation = instructions[std::get<0>(instruction)];
    operation(instruction);
  }

  /// @brief Method for performing qudit measurement.
  int measureQudit(const cudaq::QuditInfo &q,
                   const std::string &registerName) override {
    if (executionContext && executionContext->name == "sample") {
      sampleQudits.push_back(q);
      return 0;
    }

    if (executionContext && executionContext->name == "extract-state") {
      sampleQudits.push_back(q);
      return 0;
    }

    // If here, then we care about the result qudit, so compute it.
    const auto measurement_tuple = qpp::measure(
        state, qpp::cmat::Identity(q.levels, q.levels), {q.id},
        /*qudit dimension=*/q.levels, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                       collapsed_state.size());

    CUDAQ_INFO("Measured qubit {} -> {}", q.id, measurement_result);
    return measurement_result;
  }

  /// @brief Measure the state in the basis described by the given `spin_op`.
  void measureSpinOp(const cudaq::spin_op &) override {}

  /// @brief Method for performing qudit reset.
  void resetQudit(const cudaq::QuditInfo &id) override {}

  /// @brief Returns a precomputed factorial for n up tp 30
  double _fast_factorial(int n) {
    static std::vector<double> FACTORIAL_TABLE = {
        1.,
        1.,
        2.,
        6.,
        24.,
        120.,
        720.,
        5040.,
        40320.,
        362880.,
        3628800.,
        39916800.,
        479001600.,
        6227020800.,
        87178291200.,
        1307674368000.,
        20922789888000.,
        355687428096000.,
        6402373705728000.,
        121645100408832000.,
        2432902008176640000.,
        51090942171709440000.,
        1124000727777607680000.,
        25852016738884976640000.,
        620448401733239439360000.,
        15511210043330985984000000.,
        403291461126605635584000000.,
        10888869450418352160768000000.,
        304888344611713860501504000000.,
        8841761993739701954543616000000.,
        265252859812191058636308480000000.,
    };
    if (n >
        30) { // We do not expect to get 30 photons in the loop at the same time
      throw std::invalid_argument("received invalid value, n <= 30");
    }
    return FACTORIAL_TABLE[n];
  }

  /// @brief Computes a single element in the matrix representing a beam
  /// splitter gate
  double _calc_beam_splitter_elem(int N1, int N2, int n1, int n2,
                                  double theta) {

    const double t = cos(theta); // transmission coefficient
    const double r = sin(theta); // reflection coefficient
    double sum = 0;
    for (int k = 0; k <= n1; ++k) {
      int l = N1 - k;
      if (l >= 0 && l <= n2) {
        double term1 = pow(r, (n1 - k + l)) * pow(t, (n2 + k - l));
        if (term1 == 0) {
          continue;
        }
        double term2 = pow((-1), (l)) *
                       (sqrt(_fast_factorial(n1)) * sqrt(_fast_factorial(n2)) *
                        sqrt(_fast_factorial(N1)) * sqrt(_fast_factorial(N2)));
        double term3 = (_fast_factorial(k) * _fast_factorial(n1 - k) *
                        _fast_factorial(l) * _fast_factorial(n2 - l));
        double term = term1 * term2 / term3;
        sum += term;
      } else {
        continue;
      }
    }

    return sum;
  }

  /// @brief Computes matrix representing a beam splitter gate
  void beam_splitter(const double theta, qpp::cmat &BS) {
    int d = sqrt(BS.rows());
    //     """Returns a matrix representing a beam splitter
    for (int n1 = 0; n1 < d; ++n1) {
      for (int n2 = 0; n2 < d; ++n2) {
        int nxx = n1 + n2;
        int nxd = std::min(nxx + 1, d);
        for (int N1 = 0; N1 < nxd; ++N1) {
          int N2 = nxx - N1;
          if (N2 >= nxd) {
            continue;
          } else {

            BS(n1 * d + n2, N1 * d + N2) =
                _calc_beam_splitter_elem(N1, N2, n1, n2, theta);
          }
        }
      }
    }
  }

public:
  PhotonicsExecutionManager() {

    instructions.emplace("create", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      int d = target.levels;
      qpp::cmat u{qpp::cmat::Zero(d, d)};
      u(d - 1, d - 1) = 1;
      for (int i = 1; i < d; i++) {
        u(i, i - 1) = 1;
      }
      CUDAQ_INFO("Applying create on {}<{}>", target.id, target.levels);
      state = qpp::apply(state, u, {target.id}, target.levels);
    });

    instructions.emplace("annihilate", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      int d = target.levels;
      qpp::cmat u{qpp::cmat::Zero(d, d)};
      u(0, 0) = 1;
      for (int i = 0; i < d - 1; i++) {
        u(i, i + 1) = 1;
      }
      CUDAQ_INFO("Applying annihilate on {}<{}>", target.id, target.levels);
      state = qpp::apply(state, u, {target.id}, target.levels);
    });

    instructions.emplace("plus", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      int d = target.levels;
      qpp::cmat u{qpp::cmat::Zero(d, d)};
      u(0, d - 1) = 1;
      for (int i = 1; i < d; i++) {
        u(i, i - 1) = 1;
      }
      CUDAQ_INFO("Applying plus on {}<{}>", target.id, target.levels);
      state = qpp::apply(state, u, {target.id}, target.levels);
    });

    instructions.emplace("beam_splitter", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target1 = qudits[0];
      auto target2 = qudits[1];
      size_t d = target1.levels;
      const double theta = params[0];
      qpp::cmat BS{qpp::cmat::Zero(d * d, d * d)};
      beam_splitter(theta, BS);
      CUDAQ_INFO("Applying beam_splitter on {}<{}> and {}<{}>", target1.id,
                 target1.levels, target2.id, target2.levels);
      state = qpp::apply(state, BS, {target1.id, target2.id}, d);
    });

    instructions.emplace("phase_shift", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      size_t d = target.levels;
      const double phi = params[0];
      qpp::cmat PS{qpp::cmat::Identity(d, d)};
      const std::complex<double> i(0.0, 1.0);
      for (size_t n = 0; n < d; n++) {
        PS(n, n) = std::exp(n * phi * i);
      }
      CUDAQ_INFO("Applying phase_shift on {}<{}>", target.id, target.levels);
      state = qpp::apply(state, PS, {target.id}, target.levels);
    });
  }

  virtual ~PhotonicsExecutionManager() = default;

  cudaq::SpinMeasureResult measure(const cudaq::spin_op &op) override {
    throw "spin_op observation (cudaq::observe()) is not supported for this "
          "photonics simulator";
  }

}; // PhotonicsExecutionManager

} // namespace cudaq

CUDAQ_REGISTER_EXECUTION_MANAGER(PhotonicsExecutionManager, photonics)
