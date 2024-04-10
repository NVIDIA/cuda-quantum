/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/
#include "common/Logger.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include "qpp.h"
#include <cmath>
#include <complex>
#include <cstring>
#include <functional>
#include <iostream>
#include <sstream>

namespace cudaq {

/// @brief The `PhotonicsExecutionManager` implements allocation, deallocation,
/// and quantum instruction application for the photonics execution manager.
class PhotonicsExecutionManager : public cudaq::BasicExecutionManager {
private:
  /// @brief Current state
  qpp::ket state;

  /// @brief Instructions are strored in a map
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
      return;
    }

    qpp::ket zeroState = qpp::ket::Zero(q.levels);
    zeroState(0) = 1.0;
    state = qpp::kron(state, zeroState);
  }

  /// @brief Allocate a set of `qudits` with a single call.
  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    for (auto &q : qudits)
      allocateQudit(q);
  }

  void initializeState(const std::vector<cudaq::QuditInfo> &targets,
                       const void *state,
                       simulation_precision precision) override {
    throw std::runtime_error("initializeState not implemented.");
  }

  /// @brief Qudit deallocation method
  void deallocateQudit(const cudaq::QuditInfo &q) override {}

  /// @brief Deallocate a set of `qudits` with a single call.
  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {}

  /// @brief Handler for when the photonics execution context changes
  void handleExecutionContextChanged() override {}

  /// @brief Handler for when the current execution context has ended. It
  /// returns samples to the execution context if it is "sample".
  void handleExecutionContextEnded() override {
    if (executionContext && executionContext->name == "sample") {
      std::vector<std::size_t> ids;
      for (auto &s : sampleQudits) {
        ids.push_back(s.id);
      }
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

    // If here, then we care about the result bit, so compute it.
    const auto measurement_tuple = qpp::measure(
        state, qpp::cmat::Identity(q.levels, q.levels), {q.id},
        /*qudit dimension=*/q.levels, /*destructive measmt=*/false);
    const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
    const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
    const auto &collapsed_state = post_meas_states[measurement_result];
    state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                       collapsed_state.size());

    cudaq::info("Measured qubit {} -> {}", q.id, measurement_result);
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

  /// @brief Computes the kronecker delta of two values
  int _kron(int a, int b) {
    if (a == b)
      return 1;
    else
      return 0;
  }

  /// @brief Computes if two double values are within some absolute and relative
  /// tolerance
  bool _isclose(double a, double b, double rtol = 1e-08, double atol = 1e-9) {
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
  }

  /// @brief Computes a single element in the matrix representing a beam
  /// splitter gate
  double _calc_beamsplitter_elem(int N1, int N2, int n1, int n2, double theta) {

    const double t = cos(theta); // transmission coeffient
    const double r = sin(theta); // reflection coeffient
    double sum = 0;
    for (int k = 0; k <= n1; ++k) {
      int l = N1 - k;
      if (l >= 0 && l <= n2) {
        // int term4 = _kron(N1, k + l); //* kron(N1 + N2, n1 + n2);

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
  void beamsplitter(const double theta, qpp::cmat &BS) {
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
                _calc_beamsplitter_elem(N1, N2, n1, n2, theta);
          }
        }
      }
    }
  }

public:
  PhotonicsExecutionManager() {

    instructions.emplace("plusGate", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      int d = target.levels;
      qpp::cmat u{qpp::cmat::Zero(d, d)};
      u(0, d - 1) = 1;
      for (int i = 1; i < d; i++) {
        u(i, i - 1) = 1;
      }
      cudaq::info("Applying plusGate on {}<{}>", target.id, target.levels);
      state = qpp::apply(state, u, {target.id}, target.levels);
    });

    instructions.emplace("beamSplitterGate", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target1 = qudits[0];
      auto target2 = qudits[1];
      size_t d = target1.levels;
      const double theta = params[0];
      qpp::cmat BS{qpp::cmat::Zero(d * d, d * d)};
      beamsplitter(theta, BS);
      cudaq::info("Applying beamSplitterGate on {}<{}> and {}<{}>", target1.id,
                  target1.levels, target2.id, target2.levels);
      state = qpp::apply(state, BS, {target1.id, target2.id}, d);
    });

    instructions.emplace("phaseShiftGate", [&](const Instruction &inst) {
      auto &[gateName, params, controls, qudits, spin_op] = inst;
      auto target = qudits[0];
      size_t d = target.levels;
      const double phi = params[0];
      qpp::cmat PS{qpp::cmat::Identity(d, d)};
      const std::complex<double> i(0.0, 1.0);
      for (size_t n = 0; n < d; n++) {
        PS(n, n) = std::exp(n * phi * i);
      }
      cudaq::info("Applying phaseShiftGate on {}<{}>", target.id,
                  target.levels);
      state = qpp::apply(state, PS, {target.id}, target.levels);
    });
  }

  virtual ~PhotonicsExecutionManager() = default;

  cudaq::SpinMeasureResult measure(cudaq::spin_op &op) override {
    throw "spin_op observation (cudaq::observe()) is not supported for this "
          "photonics simulator";
  }

}; // PhotonicsExecutionManager

} // namespace cudaq

CUDAQ_REGISTER_EXECUTION_MANAGER(PhotonicsExecutionManager)
