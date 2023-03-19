/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#include "common/ExecutionContext.h"
#include "common/Logger.h"

#include "cudaq/qis/execution_manager.h"
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

// Define some stubs for the QIR opaque types
class Array;
class Qubit;
class Result;
using TuplePtr = int8_t *;

/// QIR QIS Extern Declarations
extern "C" {
Array *__quantum__rt__array_concatenate(Array *, Array *);
void __quantum__rt__array_release(Array *);
int8_t *__quantum__rt__array_get_element_ptr_1d(Array *q, uint64_t);
int64_t __quantum__rt__array_get_size_1d(Array *);
Array *__quantum__rt__array_create_1d(int, int64_t);
void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits);
void __quantum__qis__measure__body(Array *, Array *);
Qubit *__quantum__rt__qubit_allocate();
void __quantum__rt__qubit_release(Qubit *);
void __quantum__rt__setExecutionContext(cudaq::ExecutionContext *);
void __quantum__rt__resetExecutionContext();

void __quantum__qis__reset(Qubit *);

void __quantum__qis__h(Qubit *q);
void __quantum__qis__h__ctl(Array *ctls, Qubit *q);

void __quantum__qis__x(Qubit *q);
void __quantum__qis__x__ctl(Array *ctls, Qubit *q);

void __quantum__qis__y(Qubit *q);
void __quantum__qis__y__ctl(Array *ctls, Qubit *q);

void __quantum__qis__z(Qubit *q);
void __quantum__qis__z__ctl(Array *ctls, Qubit *q);

void __quantum__qis__t(Qubit *q);
void __quantum__qis__t__ctl(Array *ctls, Qubit *q);
void __quantum__qis__tdg(Qubit *q);
void __quantum__qis__tdg__ctl(Array *ctls, Qubit *q);

void __quantum__qis__s(Qubit *q);
void __quantum__qis__s__ctl(Array *ctls, Qubit *q);
void __quantum__qis__sdg(Qubit *q);
void __quantum__qis__sdg__ctl(Array *ctls, Qubit *q);

void __quantum__qis__rx(double, Qubit *q);
void __quantum__qis__rx__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__ry(double, Qubit *q);
void __quantum__qis__ry__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__rz(double, Qubit *q);
void __quantum__qis__rz__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__r1(double, Qubit *q);
void __quantum__qis__r1__ctl(double, Array *ctls, Qubit *q);

void __quantum__qis__swap(Qubit *, Qubit *);
void __quantum__qis__cphase(double x, Qubit *src, Qubit *tgt);

Result *__quantum__qis__mz(Qubit *);
}

namespace {
Array *spinToArray(cudaq::spin_op &);

/// The QIRQubitQISManager will implement allocation, deallocation, and
/// quantum instruction application via calls to the extern declared QIR
/// runtime library functions.
class QIRQubitQISManager : public cudaq::ExecutionManager {
private:
  using Instruction = std::tuple<std::string, std::vector<double>, Array *,
                                 std::vector<std::size_t>>;
  using InstructionQueue = std::queue<Instruction>;
  cudaq::ExecutionContext *ctx;
  std::vector<std::size_t> contextQubitIdsForDeletion;

  /// Each CUDA Quantum qubit id will map to a QIR Qubit pointer
  std::map<std::size_t, Qubit *> qubits;

  InstructionQueue instructionQueue;
  std::stack<InstructionQueue> adjointQueueStack;

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
             __quantum__qis__swap(q[0], q[1]);
           }},
          {"cphase",
           [](std::vector<double> d, Array *a, std::vector<Qubit *> &q) {
             __quantum__qis__cphase(d[0], q[0], q[1]);
           }}};

  /// Utility to convert a vector of qubits into an opaque Array pointer
  Array *vectorToArray(const std::vector<std::size_t> &ctrls) {
    if (ctrls.empty()) {
      return nullptr;
    }

    auto a = __quantum__rt__array_create_1d(sizeof(Qubit *), ctrls.size());

    // For each qubit in the tuple, add it to the Array
    for (std::size_t i = 0; i < ctrls.size(); i++) {
      Qubit **qq = reinterpret_cast<Qubit **>(
          __quantum__rt__array_get_element_ptr_1d(a, i));
      *qq = qubits[ctrls[i]];
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

public:
  virtual ~QIRQubitQISManager() {}

  void setExecutionContext(cudaq::ExecutionContext *_ctx) override {
    ctx = _ctx;
    __quantum__rt__setExecutionContext(_ctx);
    // If we set a new exec context, make sure we clear any old instructions.
    while (!instructionQueue.empty()) {
      instructionQueue.pop();
    }
  }

  void resetExecutionContext() override {
    synchronize();
    std::string_view ctx_name = "";
    if (ctx)
      ctx_name = ctx->name;

    if (ctx_name == "observe" || ctx_name == "sample" ||
        ctx_name == "extract-state") {
      for (auto &q : contextQubitIdsForDeletion) {
        __quantum__rt__qubit_release(qubits[q]);
        qubits.erase(q);
        returnIndex(q);
      }
      contextQubitIdsForDeletion.clear();
    }

    __quantum__rt__resetExecutionContext();
    ctx = nullptr;
  }

  /// Override getAvailableIndex to allocate the Qubit *
  std::size_t getAvailableIndex() override {
    auto new_id = getNextIndex();
    Qubit *qubit = __quantum__rt__qubit_allocate();

    qubits.insert({new_id, qubit});
    return new_id;
  }

  /// Overriding returnQubit in order to release the Qubit *
  void returnQubit(const std::size_t &qid) override {
    if (!ctx) {
      __quantum__rt__qubit_release(qubits[qid]);
      qubits.erase(qid);
      returnIndex(qid);
      return;
    }

    std::string_view ctx_name = "";
    if (ctx)
      ctx_name = ctx->name;

    // Handle the case where we are sampling with an implicit
    // measure on the entire register.
    if (ctx && (ctx_name == "observe" || ctx_name == "sample" ||
                ctx_name == "extract-state")) {
      contextQubitIdsForDeletion.push_back(qid);
      return;
    }

    __quantum__rt__qubit_release(qubits[qid]);
    qubits.erase(qid);
    returnIndex(qid);
    if (numAvailable() == totalNumQudits()) {
      if (ctx && ctx_name == "observe") {
        while (!instructionQueue.empty())
          instructionQueue.pop();
      }
    }
  }

  std::vector<std::size_t> extra_control_qubit_ids;
  bool inAdjointRegion = false;
  void startAdjointRegion() override { adjointQueueStack.emplace(); }

  void endAdjointRegion() override {
    // Get the top queue and remove it
    auto adjointQueue = adjointQueueStack.top();
    adjointQueueStack.pop();

    // Reverse it
    [](InstructionQueue &q) {
      std::stack<Instruction> s;
      while (!q.empty()) {
        s.push(q.front());
        q.pop();
      }
      while (!s.empty()) {
        q.push(s.top());
        s.pop();
      }
    }(adjointQueue);

    while (!adjointQueue.empty()) {
      auto front = adjointQueue.front();
      adjointQueue.pop();
      if (adjointQueueStack.empty()) {
        instructionQueue.push(front);
      } else {
        adjointQueueStack.top().push(front);
      }
    }
  }

  void startCtrlRegion(std::vector<std::size_t> &control_qubits) override {
    for (auto &c : control_qubits) {
      extra_control_qubit_ids.push_back(c);
    }
  }

  void endCtrlRegion(const std::size_t n_controls) override {
    // extra_control_qubits.erase(end - n_controls, end);
    extra_control_qubit_ids.resize(extra_control_qubit_ids.size() - n_controls);
  }

  /// The goal for apply is to create a new element of the
  /// instruction queue (a tuple).
  void apply(const std::string_view gateName,
             const std::vector<double> &&params,
             std::span<std::size_t> controls, std::span<std::size_t> targets,
             bool isAdjoint = false) override {
    cudaq::ScopedTrace trace("QIRExecManager::apply", gateName, params,
                             controls, targets, isAdjoint);

    // Make a copy of the name that we can mutate if necessary
    std::string mutable_name(gateName);

    // Make a copy of the params that we can mutate
    std::vector<double> mutable_params = params;

    // Create an array of controls, we will
    // prepend any extra controls if in a control region
    std::vector<std::size_t> mutable_controls;
    for (auto &e : extra_control_qubit_ids) {
      mutable_controls.push_back(e);
    }
    for (auto &e : controls) {
      mutable_controls.push_back(e);
    }

    std::vector<std::size_t> mutable_targets;
    for (auto &t : targets) {
      mutable_targets.push_back(t);
    }

    // Get the ctrls Array* , could be nullptr
    auto ctrls_a = vectorToArray(mutable_controls);

    if (!qir_qis_q_funcs.count(mutable_name)) {
      std::stringstream ss;
      ss << mutable_name << " is an invalid quantum instruction.";
      throw std::invalid_argument(ss.str());
    }

    if (isAdjoint || !adjointQueueStack.empty()) {
      for (std::size_t i = 0; i < params.size(); i++) {
        mutable_params[i] = -1.0 * params[i];
      }
      if (mutable_name == "t") {
        mutable_name = "tdg";
      } else if (mutable_name == "s") {
        mutable_name = "sdg";
      }
    }
    if (!adjointQueueStack.empty()) {
      // Add to the adjoint instruction queue
      adjointQueueStack.top().emplace(std::make_tuple(
          mutable_name, mutable_params, ctrls_a, mutable_targets));
    } else {
      // Add to the instruction queue
      instructionQueue.emplace(std::make_tuple(
          std::move(mutable_name), mutable_params, ctrls_a, mutable_targets));
    }
  }

  void synchronize() override {
    while (!instructionQueue.empty()) {
      auto instruction = instructionQueue.front();

      // Get the data, create the Qubit* targets
      auto [gateName, p, c, q] = instruction;

      std::vector<Qubit *> qqs;
      std::transform(
          q.begin(), q.end(), std::back_inserter(qqs),
          [&, qqs](const std::size_t &q) mutable { return qubits[q]; });

      // Run the QIR QIS function
      qir_qis_q_funcs[gateName](p, c, qqs);
      if (c != nullptr) {
        auto s = __quantum__rt__array_get_size_1d(c);
        clearArray(c, s);
      }
      instructionQueue.pop();
    }
  }

  int measure(const std::size_t &target) override {
    // We hit a measure, need to exec / clear instruction queue
    synchronize();

    // Instruction executed, run the measure call
    auto res_ptr = __quantum__qis__mz(qubits[target]);
    auto res = *reinterpret_cast<bool *>(res_ptr);
    return res ? 1 : 0;
  }

  cudaq::SpinMeasureResult measure(cudaq::spin_op &op) override {
    synchronize();
    // FIXME need to remove QIR things from spin_op
    Array *term_arr = spinToArray(op);
    __quantum__qis__measure__body(term_arr, nullptr);
    // auto counts_raw = ctx->extract_results();
    auto exp = ctx->expectationValue;
    auto data = ctx->result;
    return std::make_pair(exp.value(), data);
  }

  void resetQubit(const std::size_t &id) override {
    __quantum__qis__reset(qubits[id]);
  }

  void exp(std::vector<std::size_t> &&q, double theta,
           cudaq::spin_op &op) override {
    synchronize();
    Array *term = spinToArray(op);
    auto qubits = vectorToArray(q);
    __quantum__qis__exp__body(term, theta, qubits);
    clearArray(qubits, q.size());
  }
};

Array *spinToArray(cudaq::spin_op &op) {
  // How to pack the data???
  // add all term data as correct pointer to double for x,y,z,or I.
  // After each term add a pointer to real part of term coeff,
  // add imag part of coeff.
  // End the data array with the number of terms in the list
  // x0 y1 - y0 x1 would be
  // 1 3 coeff.real coeff.imag 3 1 coeff.real coeff.imag NTERMS
  auto n_qubits = op.n_qubits(); // data[0].size() / 2.;
  auto n_terms = op.n_terms();   // data.size();
  auto data = op.get_bsf();

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
        *ptr_el = op.get_term_coefficient(i).real();
        continue;
      }
      if (j == n_qubits + 1) {
        *ptr_el = op.get_term_coefficient(i).imag();
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

CUDAQ_REGISTER_EXECUTION_MANAGER(QIRQubitQISManager)
