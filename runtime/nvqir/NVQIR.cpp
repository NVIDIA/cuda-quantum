/*******************************************************************************
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "QIRTypes.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "cudaq/spin_op.h"
#include <cmath>
#include <complex>
#include <string>
#include <vector>

/// This file implements the primary QIR quantum-classical runtime API used
/// by the CUDA Quantum compilation platform.

// Useful preprocessor defines for building up the
// NVQIR quantum instruction functions
#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define QIS_FUNCTION_NAME(GATENAME) CONCAT(__quantum__qis__, GATENAME)
#define QIS_FUNCTION_CTRL_NAME(GATENAME)                                       \
  CONCAT(CONCAT(__quantum__qis__, GATENAME), __ctl)
#define QIS_FUNCTION_BODY_NAME(GATENAME)                                       \
  CONCAT(CONCAT(__quantum__qis__, GATENAME), __body)

// Is the library initialized?
thread_local bool initialized = false;
thread_local nvqir::CircuitSimulator *simulator;
inline static constexpr std::string_view GetCircuitSimulatorSymbol =
    "getCircuitSimulator";

/// @brief Provide a holder for externally created
/// CircuitSimulator pointers (like from Python) that
/// will invoke clone on the simulator when requested, which
/// in turn will create the simulator if there isn't one on the
/// current thread, otherwise it will reuse the existing one
struct ExternallyProvidedSimGenerator {
  nvqir::CircuitSimulator *simulator;
  ExternallyProvidedSimGenerator(nvqir::CircuitSimulator *sim)
      : simulator(sim) {}
  auto operator()() { return simulator->clone(); }
};
static std::unique_ptr<ExternallyProvidedSimGenerator> externSimGenerator;

extern "C" {
void __nvqir__setCircuitSimulator(nvqir::CircuitSimulator *sim) {
  simulator = sim;
  // If we had been given one before, reset the holder
  if (externSimGenerator) {
    auto ptr = externSimGenerator.release();
    delete ptr;
  }
  externSimGenerator = std::make_unique<ExternallyProvidedSimGenerator>(sim);
  cudaq::info("[runtime] Setting the circuit simulator to {}.", sim->name());
}
}

namespace nvqir {

/// @brief Return the single simulation backend pointer, create if not created
/// already.
/// @return
CircuitSimulator *getCircuitSimulatorInternal() {
  if (simulator)
    return simulator;

  if (externSimGenerator) {
    simulator = (*externSimGenerator)();
    return simulator;
  }

  simulator = cudaq::getUniquePluginInstance<CircuitSimulator>(
      GetCircuitSimulatorSymbol);
  cudaq::info("Creating the {} backend.", simulator->name());
  return simulator;
}

void setRandomSeed(std::size_t seed) {
  getCircuitSimulatorInternal()->setRandomSeed(seed);
}

thread_local static bool isBaseProfile = false;
void toggleBaseProfile() { isBaseProfile = !isBaseProfile; }

/// @brief Tell the simulator we are about to finalize MPI.
void tearDownBeforeMPIFinalize() {
  getCircuitSimulatorInternal()->tearDownBeforeMPIFinalize();
}

/// @brief Store allocated Array pointers
thread_local static std::vector<std::unique_ptr<Array>> allocatedArrays;

/// @brief Store allocated Qubit pointers
thread_local static std::vector<std::unique_ptr<Qubit>> allocatedSingleQubits;

/// @brief Utility function mapping qubit ids to a QIR Array pointer
Array *vectorSizetToArray(std::vector<std::size_t> &idxs) {
  auto newArray = std::make_unique<Array>(idxs.size(), sizeof(std::size_t));
  for (std::size_t i = 0; i < idxs.size(); i++) {
    auto qbit = new Qubit{idxs[i]};
    auto arrayPtr = (*newArray)[i];
    *reinterpret_cast<Qubit **>(arrayPtr) = qbit;
  }
  nvqir::allocatedArrays.emplace_back(std::move(newArray));
  return nvqir::allocatedArrays.back().get();
}

/// @brief Utility function mapping a QIR Array pointer to a vector of ids
std::vector<std::size_t> arrayToVectorSizeT(Array *arr) {
  std::vector<std::size_t> ret;
  for (std::size_t i = 0; i < arr->size(); i++) {
    auto arrayPtr = (*arr)[i];
    Qubit *idxVal = *reinterpret_cast<Qubit **>(arrayPtr);
    ret.push_back(idxVal->idx);
  }
  return ret;
}

/// @brief Utility function mapping a QIR Qubit pointer to its id
std::size_t qubitToSizeT(Qubit *q) {
  if (isBaseProfile)
    return (intptr_t)q;

  return q->idx;
}

} // namespace nvqir

using namespace nvqir;

extern "C" {

/// @brief QIR Initialization function
void __quantum__rt__initialize(int argc, int8_t **argv) {
  if (!initialized) {
    // We may need this init function later....
    initialized = true;
  }
}

/// @brief Finalize the NVQIR library
void __quantum__rt__finalize() {
  // retaining this, may want it later
}

/// @brief Set the Execution Context
void __quantum__rt__setExecutionContext(cudaq::ExecutionContext *ctx) {
  __quantum__rt__initialize(0, nullptr);

  if (ctx) {
    cudaq::ScopedTrace trace("NVQIR::setExecutionContext", ctx->name);
    cudaq::info("Setting execution context: {}{}", ctx ? ctx->name : "basic",
                ctx->hasConditionalsOnMeasureResults ? " with conditionals"
                                                     : "");
    nvqir::getCircuitSimulatorInternal()->setExecutionContext(ctx);
  }
}

/// @brief Reset the Execution Context
void __quantum__rt__resetExecutionContext() {
  cudaq::ScopedTrace trace("NVQIR::resetExecutionContext");
  cudaq::info("Resetting execution context.");
  nvqir::getCircuitSimulatorInternal()->resetExecutionContext();
}

/// @brief QIR function for allocated a qubit array
Array *__quantum__rt__qubit_allocate_array(uint64_t size) {
  cudaq::ScopedTrace trace("NVQIR::qubit_allocate_array", size);
  __quantum__rt__initialize(0, nullptr);
  auto qubitIdxs = nvqir::getCircuitSimulatorInternal()->allocateQubits(size);
  return vectorSizetToArray(qubitIdxs);
}

/// @brief Once done, release the QIR qubit array
void __quantum__rt__qubit_release_array(Array *arr) {
  cudaq::ScopedTrace trace("NVQIR::qubit_release_array", arr->size());
  for (std::size_t i = 0; i < arr->size(); i++) {
    auto arrayPtr = (*arr)[i];
    Qubit *idxVal = *reinterpret_cast<Qubit **>(arrayPtr);
    nvqir::getCircuitSimulatorInternal()->deallocate(idxVal->idx);
    delete idxVal;
  }
  auto begin = nvqir::allocatedArrays.begin();
  auto end = nvqir::allocatedArrays.end();
  nvqir::allocatedArrays.erase(
      std::remove_if(
          begin, end,
          [&](std::unique_ptr<Array> &array) { return arr == array.get(); }),
      end);
  return;
}

/// @brief Allocate a single QIR Qubit
Qubit *__quantum__rt__qubit_allocate() {
  cudaq::ScopedTrace trace("NVQIR::allocate_qubit");
  __quantum__rt__initialize(0, nullptr);
  auto qubitIdx = nvqir::getCircuitSimulatorInternal()->allocateQubit();
  auto qubit = std::make_unique<Qubit>(qubitIdx);
  nvqir::allocatedSingleQubits.emplace_back(std::move(qubit));
  return nvqir::allocatedSingleQubits.back().get();
}

/// @brief Once done, release that qubit
void __quantum__rt__qubit_release(Qubit *q) {
  cudaq::ScopedTrace trace("NVQIR::release_qubit");
  nvqir::getCircuitSimulatorInternal()->deallocate(q->idx);
  auto begin = nvqir::allocatedSingleQubits.begin();
  auto end = nvqir::allocatedSingleQubits.end();
  nvqir::allocatedSingleQubits.erase(
      std::remove_if(begin, end,
                     [&](std::unique_ptr<Qubit> &qq) { return q == qq.get(); }),
      end);
}

void __quantum__rt__deallocate_all(const std::size_t numQubits,
                                   const std::size_t *qubitIdxs) {
  std::vector<std::size_t> qubits(qubitIdxs, qubitIdxs + numQubits);
  nvqir::getCircuitSimulatorInternal()->deallocateQubits(qubits);
}

#define ONE_QUBIT_QIS_FUNCTION(GATENAME)                                       \
  void QIS_FUNCTION_NAME(GATENAME)(Qubit * qubit) {                            \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    cudaq::ScopedTrace trace("NVQIR::" + std::string(#GATENAME), targetIdx);   \
    nvqir::getCircuitSimulatorInternal()->GATENAME(targetIdx);                 \
  }                                                                            \
  void QIS_FUNCTION_CTRL_NAME(GATENAME)(Array * ctrlQubits, Qubit * qubit) {   \
    auto ctrlIdxs = arrayToVectorSizeT(ctrlQubits);                            \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    cudaq::ScopedTrace trace("NVQIR::ctrl-" + std::string(#GATENAME),          \
                             ctrlIdxs, targetIdx);                             \
    nvqir::getCircuitSimulatorInternal()->GATENAME(ctrlIdxs, targetIdx);       \
  }                                                                            \
  void QIS_FUNCTION_BODY_NAME(GATENAME)(Qubit * qubit) {                       \
    QIS_FUNCTION_NAME(GATENAME)(qubit);                                        \
  }

ONE_QUBIT_QIS_FUNCTION(h);
ONE_QUBIT_QIS_FUNCTION(x);
ONE_QUBIT_QIS_FUNCTION(y);
ONE_QUBIT_QIS_FUNCTION(z);
ONE_QUBIT_QIS_FUNCTION(t);
ONE_QUBIT_QIS_FUNCTION(s);
ONE_QUBIT_QIS_FUNCTION(tdg);
ONE_QUBIT_QIS_FUNCTION(sdg);

void __quantum__qis__t__adj(Qubit *qubit) {
  auto targetIdx = qubitToSizeT(qubit);
  nvqir::getCircuitSimulatorInternal()->tdg(targetIdx);
}

void __quantum__qis__s__adj(Qubit *qubit) {
  auto targetIdx = qubitToSizeT(qubit);
  nvqir::getCircuitSimulatorInternal()->sdg(targetIdx);
}

#define ONE_QUBIT_PARAM_QIS_FUNCTION(GATENAME)                                 \
  void QIS_FUNCTION_NAME(GATENAME)(double param, Qubit *qubit) {               \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    cudaq::ScopedTrace trace("NVQIR::" + std::string(#GATENAME), param,        \
                             targetIdx);                                       \
    nvqir::getCircuitSimulatorInternal()->GATENAME(param, targetIdx);          \
  }                                                                            \
  void QIS_FUNCTION_BODY_NAME(GATENAME)(double param, Qubit *qubit) {          \
    QIS_FUNCTION_NAME(GATENAME)(param, qubit);                                 \
  }                                                                            \
  void QIS_FUNCTION_CTRL_NAME(GATENAME)(double param, Array *ctrlQubits,       \
                                        Qubit *qubit) {                        \
    auto ctrlIdxs = arrayToVectorSizeT(ctrlQubits);                            \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    cudaq::ScopedTrace trace("NVQIR::" + std::string(#GATENAME), param,        \
                             ctrlIdxs, targetIdx);                             \
    nvqir::getCircuitSimulatorInternal()->GATENAME(param, ctrlIdxs,            \
                                                   targetIdx);                 \
  }

ONE_QUBIT_PARAM_QIS_FUNCTION(rx);
ONE_QUBIT_PARAM_QIS_FUNCTION(ry);
ONE_QUBIT_PARAM_QIS_FUNCTION(rz);
ONE_QUBIT_PARAM_QIS_FUNCTION(r1);

void __quantum__qis__swap(Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  cudaq::ScopedTrace trace("NVQIR::swap", qI, rI);
  nvqir::getCircuitSimulatorInternal()->swap(qI, rI);
}

void __quantum__qis__swap__ctl(Array *ctrls, Qubit *q, Qubit *r) {
  auto ctrlIdxs = arrayToVectorSizeT(ctrls);
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  nvqir::getCircuitSimulatorInternal()->swap(ctrlIdxs, qI, rI);
}

void __quantum__qis__swap__body(Qubit *q, Qubit *r) {
  __quantum__qis__swap(q, r);
}

void __quantum__qis__cphase(double d, Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  std::vector<std::size_t> ctrls{qI};
  nvqir::getCircuitSimulatorInternal()->r1(d, ctrls, rI);
}

void __quantum__qis__phased_rx(double theta, double phi, Qubit *q) {
  auto qI = qubitToSizeT(q);
  std::complex<double> i(0, 1.);
  std::vector<std::complex<double>> matrix{
      std::cos(theta / 2.), -i * std::exp(-i * phi) * std::sin(theta / 2.),
      -i * std::exp(i * phi) * std::sin(theta / 2.), std::cos(theta / 2.)};
  nvqir::getCircuitSimulatorInternal()->applyCustomOperation(matrix, {}, {qI});
}

void __quantum__qis__cnot(Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  cudaq::ScopedTrace trace("NVQIR::cnot", qI, rI);
  std::vector<std::size_t> controls{qI};
  nvqir::getCircuitSimulatorInternal()->x(controls, rI);
}

void __quantum__qis__cnot__body(Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  cudaq::ScopedTrace trace("NVQIR::cnot", qI, rI);
  std::vector<std::size_t> controls{qI};
  nvqir::getCircuitSimulatorInternal()->x(controls, rI);
}

void __quantum__qis__reset(Qubit *q) {
  auto qI = qubitToSizeT(q);
  cudaq::ScopedTrace trace("NVQIR::reset", qI);
  nvqir::getCircuitSimulatorInternal()->resetQubit(qI);
}

Result *__quantum__qis__mz(Qubit *q) {
  auto qI = qubitToSizeT(q);
  cudaq::ScopedTrace trace("NVQIR::mz", qI);
  auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, "");
  return b ? ResultOne : ResultZero;
}

Result *__quantum__qis__mz__body(Qubit *q) {
  auto qI = qubitToSizeT(q);
  cudaq::ScopedTrace trace("NVQIR::mz", qI);
  auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, "");
  return b ? ResultOne : ResultZero;
}

Result *__quantum__qis__mz__to__register(Qubit *q, const char *name) {
  std::string regName(name);
  auto qI = qubitToSizeT(q);
  cudaq::ScopedTrace trace("NVQIR::mz", qI, regName);
  auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, regName);
  return b ? ResultOne : ResultZero;
}

void __quantum__qis__exp_pauli(double theta, Array *qubits, char *pauliWord) {
  std::string pauliWordStr(pauliWord);
  auto qubitsVec = arrayToVectorSizeT(qubits);
  nvqir::getCircuitSimulatorInternal()->applyExpPauli(
      theta, qubitsVec, cudaq::spin_op::from_word(pauliWordStr));
  return;
}

void __quantum__rt__result_record_output(Result *, int8_t *) {}

/// @brief Map an Array pointer containing Paulis to a vector of Paulis.
/// @param paulis
/// @return
static std::vector<Pauli> extractPauliTermIds(Array *paulis) {
  std::vector<Pauli> pauliIds;
  // size - 3 bc we don't want coeff.real coeff.imag or nterms
  for (std::size_t i = 0; i < paulis->size() - 3; ++i) {
    auto ptr = (*paulis)[i];
    double *casted_and_deref = reinterpret_cast<double *>(ptr);
    Pauli tmp = static_cast<Pauli>(*casted_and_deref);
    pauliIds.emplace_back(tmp);
  }
  return pauliIds;
}

/// @brief QIR function measuring the qubit state in the given Pauli basis.
/// @param pauli_arr
/// @param qubits
/// @return
Result *__quantum__qis__measure__body(Array *pauli_arr, Array *qubits) {
  cudaq::info("NVQIR measuring in pauli basis");
  cudaq::ScopedTrace trace("NVQIR::observe_measure_body");

  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  auto currentContext = circuitSimulator->getExecutionContext();

  // Some backends may better handle the observe task.
  // Let's give them that opportunity.
  if (currentContext->canHandleObserve) {
    circuitSimulator->flushGateQueue();
    auto result = circuitSimulator->observe(*currentContext->spin.value());
    currentContext->expectationValue = result.expectationValue;
    currentContext->result = cudaq::sample_result(result);
    return ResultZero;
  }

  const auto paulis = extractPauliTermIds(pauli_arr);
  std::vector<std::size_t> qubits_to_measure;
  std::vector<std::pair<std::string, std::size_t>> reverser;
  for (size_t i = 0; i < paulis.size(); ++i) {
    const auto pauli = paulis[i];
    switch (pauli) {
    case Pauli::Pauli_I:
      break;
    case Pauli::Pauli_X: {

      circuitSimulator->h(i);
      qubits_to_measure.push_back(i);
      reverser.push_back({"X", i});
      break;
    }
    case Pauli::Pauli_Y: {
      double angle = M_PI_2;
      circuitSimulator->rx(angle, i);
      qubits_to_measure.push_back(i);
      reverser.push_back({"Y", i});

      break;
    }
    case Pauli::Pauli_Z: {
      qubits_to_measure.push_back(i);
      break;
    }
    }
  }

  circuitSimulator->flushGateQueue();
  int shots = 0;
  if (currentContext->shots > 0) {
    shots = currentContext->shots;
  }

  // Sample and give the data to the context
  cudaq::ExecutionResult result =
      circuitSimulator->sample(qubits_to_measure, shots);
  currentContext->expectationValue = result.expectationValue;
  currentContext->result = cudaq::sample_result(result);

  // Reverse the measurements bases change.
  if (!reverser.empty()) {
    cudaq::info("NVQIR reverse pauli bases change for measurement.");
    for (auto it = reverser.rbegin(); it != reverser.rend(); ++it) {
      if (it->first == "X") {
        circuitSimulator->h(it->second);
      } else if (it->first == "Y") {
        double angle = -M_PI_2;
        circuitSimulator->rx(angle, it->second);
      }
    }
    circuitSimulator->flushGateQueue();
  }

  return ResultZero;
}

/// @brief Implementation of first order trotterization
/// enables exp( i * angle * H), where H = Sum (PauliTensorProduct)
/// @param paulis
/// @param angle
/// @param qubits
void __quantum__qis__exp__body(Array *paulis, double angle, Array *qubits) {
  auto n_qubits = qubits->size();
  cudaq::ScopedTrace trace("NVQIR::exp_body");

  // if identity, do nothing
  std::vector<int> test;
  for (std::size_t i = 0; i < paulis->size() - 1; i += n_qubits + 2) {
    for (std::size_t j = 0; j < n_qubits; j++) {
      auto ptr = (*paulis)[i + j];
      double *casted_and_deref = reinterpret_cast<double *>(ptr);
      int val = (int)*casted_and_deref;
      test.push_back(val);
    }
  }

  if (std::all_of(test.begin(), test.end(), [](int i) { return i == 0; })) {
    // do nothing since this is the ID term
    std::string msg = "Applying exp (i theta H), where H is the identity";
    msg += "(warning, no non-identity terms in H. Not applying exp())";
    cudaq::info(msg.c_str());
    return;
  }

  enum QISInstructionType { H, Rx, CX, Rz };
  using QISInstruction =
      std::tuple<QISInstructionType, std::vector<Qubit *>, double>;
  // Want to map logical qubit idxs to those in the qubits Array
  std::map<std::size_t, std::size_t> qubit_map;
  std::map<std::size_t, Qubit *> idx_to_qptr;
  for (std::size_t i = 0; i < n_qubits; i++) {
    Qubit *q = *reinterpret_cast<Qubit **>((*qubits)[i]);
    qubit_map.insert({i, q->idx});
    idx_to_qptr.insert({q->idx, q});
  }

  // size -1 bc last element is NTERMS
  for (std::size_t i = 0; i < paulis->size() - 1; i += n_qubits + 2) {
    std::vector<QISInstruction> basis_back;
    std::vector<std::size_t> qIdxs;

    for (std::size_t j = 0; j < n_qubits; j++) {
      // Get what type of pauli this is
      auto ptr = (*paulis)[i + j];
      double *casted_and_deref = reinterpret_cast<double *>(ptr);
      int val = (int)*casted_and_deref;
      qIdxs.push_back(qubit_map[j]);

      // Get the Qubit pointer
      Qubit *q = idx_to_qptr[qubit_map[j]];
      if (val == 1) {

        nvqir::getCircuitSimulatorInternal()->h(q->idx);
        basis_back.emplace_back(std::make_tuple(QISInstructionType::H,
                                                std::vector<Qubit *>{q}, 0.0));
      } else if (val == 3) {
        double param = M_PI / 2.;
        nvqir::getCircuitSimulatorInternal()->rx(param, q->idx);
        basis_back.emplace_back(std::make_tuple(
            QISInstructionType::Rx, std::vector<Qubit *>{q}, -M_PI / 2.));
      }
    }

    std::vector<std::vector<int>> cnot_pairs(
        2, std::vector<int>(qIdxs.size() - 1));
    for (std::size_t i = 0; i < qIdxs.size() - 1; i++) {
      cnot_pairs[0][i] = qIdxs[i];
    }
    for (std::size_t i = 0; i < qIdxs.size() - 1; i++) {
      cnot_pairs[1][i] = qIdxs[i + 1];
    }

    std::vector<QISInstruction> cnot_back;
    for (std::size_t i = 0; i < qIdxs.size() - 1; i++) {
      auto c = cnot_pairs[0][i];
      auto t = cnot_pairs[1][i];
      auto q1 = idx_to_qptr[c];
      auto q2 = idx_to_qptr[t];
      std::vector<std::size_t> controls{(q1->idx)};
      nvqir::getCircuitSimulatorInternal()->x(controls, q2->idx);
    }

    for (int i = qIdxs.size() - 2; i >= 0; i--) {
      auto c = cnot_pairs[0][i];
      auto t = cnot_pairs[1][i];
      auto q1 = idx_to_qptr[c];
      auto q2 = idx_to_qptr[t];

      cnot_back.emplace_back(std::make_tuple(
          QISInstructionType::CX, std::vector<Qubit *>{q1, q2}, 0.0));
    }

    auto ptr = (*paulis)[i + n_qubits];
    double *casted_and_deref_real = reinterpret_cast<double *>(ptr);
    ptr = (*paulis)[i + n_qubits + 1];
    double *casted_and_deref_imag = reinterpret_cast<double *>(ptr);

    std::complex<double> coeff(*casted_and_deref_real, *casted_and_deref_imag);
    double param = coeff.real() * angle;
    nvqir::getCircuitSimulatorInternal()->rz(param,
                                             idx_to_qptr[qIdxs.back()]->idx);

    for (auto cxb : cnot_back) {
      auto qs = std::get<1>(cxb);
      std::vector<std::size_t> controls{(qs[0]->idx)};
      nvqir::getCircuitSimulatorInternal()->x(controls, qs[1]->idx);
    }

    for (auto bb : basis_back) {
      auto type = std::get<0>(bb);
      auto qs = std::get<1>(bb);
      auto ps = std::get<2>(bb);
      if (type == QISInstructionType::H) {
        nvqir::getCircuitSimulatorInternal()->h(qs[0]->idx);
      } else {
        nvqir::getCircuitSimulatorInternal()->rx(ps, qs[0]->idx);
      }
    }
  }
}

/// @brief Utility function used by Quake->QIR to pack a single Qubit pointer
/// into an Array pointer.
Array *packSingleQubitInArray(Qubit *q) {
  auto newArray = std::make_unique<Array>(1, sizeof(std::size_t));
  auto arrayPtr = (*newArray)[0];
  *reinterpret_cast<Qubit **>(arrayPtr) = q;
  nvqir::allocatedArrays.emplace_back(std::move(newArray));
  return nvqir::allocatedArrays.back().get();
}

/// @brief Utility function used by Quake->QIR to release any created Array from
/// Qubit packing after its been used
void releasePackedQubitArray(Array *a) {
  auto begin = nvqir::allocatedArrays.begin();
  auto end = nvqir::allocatedArrays.end();
  nvqir::allocatedArrays.erase(
      std::remove_if(
          begin, end,
          [&](std::unique_ptr<Array> &array) { return a == array.get(); }),
      end);
  return;
}

/// @brief Utility function used by Quake->QIR to invoke a QIR QIS function
/// with a variadic list of control qubits.
/// @param nControls
/// @param QISFunction
/// @param
void invokeWithControlQubits(const std::size_t nControls,
                             void (*QISFunction)(Array *, Qubit *), ...) {
  // Create the Control Array *, This should
  // be deallocated upon function exit.
  auto ctrlArray = std::make_unique<Array>(nControls, sizeof(std::size_t));

  // Start up the variadic arg processing
  va_list args;
  va_start(args, QISFunction);

  std::size_t nSetPointers = 0;
  while (nSetPointers < nControls) {
    // For each variadic arg, get it (its a Qubit*)
    // and set it on the array.
    Qubit *ctrli = va_arg(args, Qubit *);
    auto ctrliRawPtr =
        __quantum__rt__array_get_element_ptr_1d(ctrlArray.get(), nSetPointers);
    *reinterpret_cast<Qubit **>(ctrliRawPtr) = ctrli;
    nSetPointers++;
  }

  // The last one will be the target
  Qubit *target = va_arg(args, Qubit *);
  // Invoke the function
  QISFunction(ctrlArray.get(), target);

  // End the var args processing and release the array.
  va_end(args);
}
}
