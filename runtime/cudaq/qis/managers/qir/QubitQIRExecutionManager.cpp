/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "QIRForwards.h"
#include "common/Logger.h"
#include "cudaq/qis/managers/BasicExecutionManager.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/spin_op.h"
#include "cudaq/utils/cudaq_utils.h"
#include <complex>
#include <cstring>
#include <functional>
#include <map>
#include <queue>
#include <sstream>
#include <stack>

namespace {
Array *spinToArray(const cudaq::spin_op &);

/// The QIRQubitQISManager will implement allocation, deallocation, and
/// quantum instruction application via calls to the extern declared QIR
/// runtime library functions.
class QIRExecutionManager : public cudaq::BasicExecutionManager {
private:
  /// Each CUDA Quantum qubit id will map to a QIR Qubit pointer
  std::map<std::size_t, Qubit *> qubits;

  /// The QIR function application map. Each element of
  /// this map exposes a functor that takes a parameter vector, control
  /// qubits as an Array*, and the target QIR Qubits.
  std::map<std::string_view, std::function<void(std::vector<double>, Array *,
                                                std::vector<Qubit *> &)>>
      qir_qis_q_funcs{
          {"h",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__h__ctl(a, q[0])
                          : __quantum__qis__h(q[0]);
           }},
          {"x",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__x__ctl(a, q[0])
                          : __quantum__qis__x(q[0]);
           }},
          {"y",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__y__ctl(a, q[0])
                          : __quantum__qis__y(q[0]);
           }},
          {"z",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__z__ctl(a, q[0])
                          : __quantum__qis__z(q[0]);
           }},

          {"t",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__t__ctl(a, q[0])
                          : __quantum__qis__t(q[0]);
           }},
          {"s",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__s__ctl(a, q[0])
                          : __quantum__qis__s(q[0]);
           }},
          {"tdg",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__tdg__ctl(a, q[0])
                          : __quantum__qis__tdg(q[0]);
           }},
          {"sdg",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__sdg__ctl(a, q[0])
                          : __quantum__qis__sdg(q[0]);
           }},
          {"rx",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__rx__ctl(d[0], a, q[0])
                          : __quantum__qis__rx(d[0], q[0]);
           }},
          {"ry",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__ry__ctl(d[0], a, q[0])
                          : __quantum__qis__ry(d[0], q[0]);
           }},
          {"rz",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__rz__ctl(d[0], a, q[0])
                          : __quantum__qis__rz(d[0], q[0]);
           }},
          {"r1",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__r1__ctl(d[0], a, q[0])
                          : __quantum__qis__r1(d[0], q[0]);
           }},
          {"swap",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             a != nullptr ? __quantum__qis__swap__ctl(a, q[0], q[1])
                          : __quantum__qis__swap(q[0], q[1]);
           }}};

  /// Utility to convert a vector of qubits into an opaque Array pointer
  Array *vectorToArray(const std::vector<cudaq::QuditInfo> &ctrls) {
    if (ctrls.empty()) {
      return nullptr;
    }

    auto a = __quantum__rt__array_create_1d(sizeof(Qubit *), ctrls.size());

    // For each qubit in the tuple, add it to the Array
    for (std::size_t i = 0; i < ctrls.size(); i++) {
      Qubit **qq = reinterpret_cast<Qubit **>(
          __quantum__rt__array_get_element_ptr_1d(a, i));
      *qq = qubits[ctrls[i].id];
    }

    // Return
    return a;
  }

  /// Clean up the Array * created in vectorToArray
  void clearArray(Array *a, std::size_t size) {
    if (a == nullptr) {
      return;
    }

    for (std::size_t i = 0; i < size; i++) {
      auto ptr = __quantum__rt__array_get_element_ptr_1d(a, i);
      *reinterpret_cast<Qubit **>(ptr) = nullptr;
    }
    __quantum__rt__array_release(a);
  }

protected:
  /// @brief To improve `qudit` allocation, we defer
  /// single `qudit` allocation requests until the first
  /// encountered `apply` call.
  std::vector<cudaq::QuditInfo> requestedAllocations;

  /// @brief Allocate all requested `qudits`.
  void flushRequestedAllocations() {
    if (requestedAllocations.empty())
      return;

    allocateQudits(requestedAllocations);
    requestedAllocations.clear();
  }

  void allocateQudit(const cudaq::QuditInfo &q) override {
    requestedAllocations.emplace_back(2, q.id);
  }

  void allocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    auto *qa = __quantum__rt__qubit_allocate_array(qudits.size());
    for (std::size_t i = 0; i < qudits.size(); i++) {
      Qubit **qq = reinterpret_cast<Qubit **>(
          __quantum__rt__array_get_element_ptr_1d(qa, i));
      qubits.insert({qudits[i].id, *qq});
    }
  }

  void deallocateQudit(const cudaq::QuditInfo &q) override {
    if (!qubits.count(q.id))
      return;
    __quantum__rt__qubit_release(qubits[q.id]);
    qubits.erase(q.id);
  }

  void deallocateQudits(const std::vector<cudaq::QuditInfo> &qudits) override {
    std::vector<std::size_t> local;
    std::transform(qudits.begin(), qudits.end(), std::back_inserter(local),
                   [](auto &&el) { return el.id; });

    __quantum__rt__deallocate_all(local.size(), local.data());

    // remove from the qubits map
    for (auto &q : qudits)
      qubits.erase(q.id);
  }

  void handleExecutionContextChanged() override {
    requestedAllocations.clear();
    __quantum__rt__setExecutionContext(executionContext);
  }

  void handleExecutionContextEnded() override {
    __quantum__rt__resetExecutionContext();
  }

  void executeInstruction(const Instruction &instruction) override {
    flushRequestedAllocations();

    // Get the data, create the Qubit* targets
    auto [gateName, p, c, q] = instruction;

    std::vector<Qubit *> qqs;
    std::transform(
        q.begin(), q.end(), std::back_inserter(qqs),
        [&, qqs](const cudaq::QuditInfo &q) mutable { return qubits[q.id]; });

    auto ctmp = vectorToArray(c);

    // Run the QIR QIS function
    qir_qis_q_funcs[gateName](p, ctmp, qqs);
    if (ctmp != nullptr) {
      auto s = __quantum__rt__array_get_size_1d(ctmp);
      clearArray(ctmp, s);
    }
  }

  int measureQudit(const cudaq::QuditInfo &q) override {
    auto res_ptr = __quantum__qis__mz(qubits[q.id]);
    auto res = *reinterpret_cast<bool *>(res_ptr);
    return res ? 1 : 0;
  }

  void measureSpinOp(const cudaq::spin_op &op) override {
    Array *term_arr = spinToArray(op);
    __quantum__qis__measure__body(term_arr, nullptr);
  }

public:
  QIRExecutionManager() = default;
  virtual ~QIRExecutionManager() {}

  void resetQudit(const cudaq::QuditInfo &id) override {
    __quantum__qis__reset(qubits[id.id]);
  }
};

Array *spinToArray(const cudaq::spin_op &op) {
  // How to pack the data???
  // add all term data as correct pointer to double for x,y,z,or I.
  // After each term add a pointer to real part of term coeff,
  // add imag part of coeff.
  // End the data array with the number of terms in the list
  // x0 y1 - y0 x1 would be
  // 1 3 coeff.real coeff.imag 3 1 coeff.real coeff.imag NTERMS
  auto n_qubits = op.num_qubits(); // data[0].size() / 2.;
  auto n_terms = op.num_terms();   // data.size();
  auto [data, coeffs] = op.get_raw_data();

  auto arr = __quantum__rt__array_create_1d(
      sizeof(double), n_qubits * n_terms + 2 * n_terms + 1);

  for (std::size_t i = 0; i < n_terms; i++) {
    auto term_data = data[i];
    std::size_t row_size = n_qubits + 2;
    for (std::size_t j = 0; j < row_size; j++) {
      int8_t *ptr =
          __quantum__rt__array_get_element_ptr_1d(arr, i * row_size + j);
      auto ptr_el = reinterpret_cast<double *>(ptr);
      if (j == n_qubits) {
        *ptr_el = coeffs[i].real();
        continue;
      }
      if (j == n_qubits + 1) {
        *ptr_el = coeffs[i].imag();
        break;
      }

      if (term_data[j] && term_data[j + n_qubits]) {
        // Y term
        *ptr_el = 3.0;
      } else if (term_data[j]) {
        // X term
        *ptr_el = 1.0;
      } else if (term_data[j + n_qubits]) {
        // Z term
        *ptr_el = 2.0;
      } else {
        *ptr_el = 0.0;
      }
    }
  }

  int8_t *ptr = __quantum__rt__array_get_element_ptr_1d(
      arr, n_qubits * n_terms + 2 * n_terms);
  auto ptr_el = reinterpret_cast<double *>(ptr);
  *ptr_el = n_terms;
  return arr;
}

} // namespace

CUDAQ_REGISTER_EXECUTION_MANAGER(QIRExecutionManager)
