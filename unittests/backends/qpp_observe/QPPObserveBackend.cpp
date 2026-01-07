/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#define __NVQIR_QPP_TOGGLE_CREATE
#include "qpp/QppCircuitSimulator.cpp"
#undef __NVQIR_QPP_TOGGLE_CREATE
#include "cudaq/operators.h"

namespace cudaq {

/// @brief Create a new CircuitSimulator backend for testing
/// that provides its own observe functionality.
class QppObserveTester : public nvqir::QppCircuitSimulator<qpp::ket> {
public:
  NVQIR_SIMULATOR_CLONE_IMPL(QppObserveTester)
  bool canHandleObserve() override { return true; }
  cudaq::observe_result observe(const cudaq::spin_op &op) override {
    assert(cudaq::spin_op::canonicalize(op) == op);
    flushGateQueue();

    ::qpp::cmat X = ::qpp::Gates::get_instance().X;
    ::qpp::cmat Y = ::qpp::Gates::get_instance().Y;
    ::qpp::cmat Z = ::qpp::Gates::get_instance().Z;

    double sum = 0.0;

    // Want to loop over all terms in op and
    // compute E_i = coeff_i * < psi | Term | psi >
    // = coeff_i * sum_k <psi | Pauli_k psi>
    for (const auto &term : op) {
      if (!term.is_identity()) {
        ::qpp::ket cached = state;
        auto bsf = term.get_binary_symplectic_form();
        auto max_target = bsf.size() / 2;
        for (std::size_t i = 0; i < max_target; i++) {
          if (bsf[i] && bsf[i + max_target])
            cached = ::qpp::apply(cached, Y, {convertQubitIndex(i)});
          else if (bsf[i])
            cached = ::qpp::apply(cached, X, {convertQubitIndex(i)});
          else if (bsf[i + max_target])
            cached = ::qpp::apply(cached, Z, {convertQubitIndex(i)});
        }

        sum += term.evaluate_coefficient().real() *
               state.transpose().dot(cached).real();
      } else {
        sum += term.evaluate_coefficient().real();
      }
    }

    return cudaq::observe_result(
        sum, op,
        cudaq::sample_result(cudaq::ExecutionResult({}, op.to_string(), sum)));
  }

  std::string name() const override { return "qpp-test"; }
};
} // namespace cudaq

NVQIR_REGISTER_SIMULATOR(cudaq::QppObserveTester, qpp_test)
