/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CircuitSimulator.h"
#include "NVQIRUtil.h"
#include "QIRTypes.h"
#include "common/Logger.h"
#include "common/PluginUtils.h"
#include "cudaq/qis/qudit.h"
#include "cudaq/qis/state.h"
// TODO: do we want to avoid including this here?
#include "resourcecounter/ResourceCounter.h"
#include <cmath>
#include <complex>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

/// This file implements the primary QIR quantum-classical runtime API used
/// by the CUDA-Q compilation platform.

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
thread_local bool using_resource_counter = false;
thread_local nvqir::CircuitSimulator *simulator;
inline static constexpr std::string_view GetCircuitSimulatorSymbol =
    "getCircuitSimulator";

// The following maps are used to map Qubits to Results, and Results to boolean
// values. The pointer values may be integers if they are referring to Base
// Profile or Adaptive Profile QIR programs, so it is generally not safe to
// dereference them.
static thread_local std::map<Qubit *, Result *> measQB2Res;
static thread_local std::map<Result *, Qubit *> measRes2QB;
static thread_local std::map<Result *, Result> measRes2Val;

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
  CUDAQ_INFO("[runtime] Setting the circuit simulator to {}.", sim->name());
}
}

namespace nvqir {

/// @brief Return the single simulation backend pointer, create if not created
/// already.
/// @return
CircuitSimulator *getCircuitSimulatorInternal() {
  if (using_resource_counter)
    return getResourceCounterSimulator();

  if (simulator)
    return simulator;

  if (externSimGenerator) {
    simulator = (*externSimGenerator)();
    return simulator;
  }

  simulator = cudaq::getUniquePluginInstance<CircuitSimulator>(
      GetCircuitSimulatorSymbol);
  CUDAQ_INFO("Creating the {} backend.", simulator->name());
  return simulator;
}

void switchToResourceCounterSimulator() { using_resource_counter = true; }

void stopUsingResourceCounterSimulator() {
  using_resource_counter = false;
  getResourceCounterSimulator()->setToZeroState();
}

bool isUsingResourceCounterSimulator() { return using_resource_counter; }

void setRandomSeed(std::size_t seed) {
  getCircuitSimulatorInternal()->setRandomSeed(seed);
}

/// @brief The QIR spec allows for dynamic qubit management, where the qubit
/// pointers are true pointers, but the Base Profile and Adaptive profiles
/// require that qubits are identified by an integer value that is bitcast to a
/// pointer.
thread_local static bool qubitPtrIsIndex = false;
void toggleDynamicQubitManagement() { qubitPtrIsIndex = !qubitPtrIsIndex; }

/// @brief Tell the simulator we are about to finalize MPI.
void tearDownBeforeMPIFinalize() {
  getCircuitSimulatorInternal()->tearDownBeforeMPIFinalize();
}

/// @brief Store allocated Qubit pointers
thread_local static std::vector<std::unique_ptr<Qubit>> allocatedSingleQubits;

/// @brief Utility function mapping qubit ids to a QIR Array pointer
// Qubits are newly constructed in the returned array. Hence, the caller must
// manage their lifetime.
// For example, this can be used for allocating qubit arrays whereby the
// corresponding deallocation (`__quantum__rt__qubit_release_array`) would clean
// up the qubits.
//
// IMPORTANT: don't use this for borrowed qubit indices as that
// will result in leak.
Array *vectorSizetToArray(std::vector<std::size_t> &idxs) {
  Array *newArray = new Array(idxs.size(), sizeof(std::size_t));
  for (std::size_t i = 0; i < idxs.size(); i++) {
    auto qbit = new Qubit{idxs[i]};
    auto arrayPtr = (*newArray)[i];
    *reinterpret_cast<Qubit **>(arrayPtr) = qbit;
  }
  nvqir::ArrayTracker::getInstance().track(newArray);
  return newArray;
}

/// @brief Utility function mapping a QIR Array pointer to a vector of ids
std::vector<std::size_t> arrayToVectorSizeT(Array *arr) {
  assert(arr && "array must not be null");
  std::vector<std::size_t> ret;
  const auto arrSize = arr->size();
  for (std::size_t i = 0; i < arrSize; ++i) {
    auto arrayPtr = (*arr)[i];
    Qubit *idxVal = *reinterpret_cast<Qubit **>(arrayPtr);
    if (qubitPtrIsIndex)
      ret.push_back(reinterpret_cast<intptr_t>(idxVal));
    else
      ret.push_back(idxVal->idx);
  }
  return ret;
}

/// @brief Utility function mapping a QIR Qubit pointer to its id
std::size_t qubitToSizeT(Qubit *q) {
  if (qubitPtrIsIndex)
    return (intptr_t)q;
  assert(q && "qubit must not be null");
  return q->idx;
}

template <typename T>
concept FloatType = std::is_same<T, float>::value;

template <typename T>
concept DoubleType = std::is_same<T, double>::value;

template <typename T>
concept SimPrecisionType = FloatType<T> || DoubleType<T>;

template <FloatType T>
constexpr std::string_view typeName() {
  return "float";
}

template <DoubleType T>
constexpr std::string_view typeName() {
  return "double";
}

/// Input was complex<float>/complex<double> but we prefer
/// complex<double>/complex<float>. Make a copy, extending or truncating the
/// values.
template <SimPrecisionType To, SimPrecisionType From>
std::unique_ptr<std::complex<To>[]> convertToComplex(std::complex<From> *data,
                                                     std::size_t numQubits) {
  // The state size is `2^numQubits`
  auto size = pow(2, numQubits);
  constexpr auto toType = typeName<To>();
  constexpr auto fromType = typeName<From>();
  CUDAQ_INFO("copying {} complex<{}> values to complex<{}>", size, fromType,
             toType);

  auto convertData = std::make_unique<std::complex<To>[]>(size);
  for (std::size_t i = 0; i < size; ++i)
    convertData[i] = std::complex<To>{static_cast<To>(data[i].real()),
                                      static_cast<To>(data[i].imag())};
  return convertData;
}

/// Input was float/double but we prefer complex<float>/complex<double>.
/// Make a copy, extending or truncating the values.
template <SimPrecisionType To, SimPrecisionType From>
std::unique_ptr<std::complex<To>[]> convertToComplex(From *data,
                                                     std::size_t numQubits) {
  // The state size is `2^numQubits`
  auto size = pow(2, numQubits);
  constexpr auto toType = typeName<To>();
  constexpr auto fromType = typeName<From>();
  CUDAQ_INFO("copying {} {} values to complex<{}>", size, fromType, toType);

  auto convertData = std::make_unique<std::complex<To>[]>(size);
  for (std::size_t i = 0; i < size; ++i)
    convertData[i] =
        std::complex<To>{static_cast<To>(data[i]), static_cast<To>(0.0)};
  return convertData;
}

// Util function to access the current QIR output.
// Note: as the QIR output is attached to a specific simulator backend instance,
// the QIR output must be retrieved from the same thread as each thread will
// have a different simulator instance, e.g., async. execution.
std::string_view getQirOutputLog() {
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  return circuitSimulator->outputLog;
}
void clearQirOutputLog() {
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  return circuitSimulator->outputLog.clear();
}
} // namespace nvqir

using namespace nvqir;

template <typename VAL>
void quantumRTGenericRecordOutput(const char *type, VAL val,
                                  const char *label) {
  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  std::ostringstream ss;
  ss << "OUTPUT\t" << type << "\t" << val << '\t';
  if (label)
    ss << label;
  ss << '\n';
  circuitSimulator->outputLog += ss.str();
}

extern "C" {

void print_i64(const char *msg, std::size_t i) { printf(msg, i); }
void print_f64(const char *msg, double f) { printf(msg, f); }

/// @brief Return whether or not the NVQIR runtime is running with dynamic qubit
/// management (qubits are pointers) or not (qubits are integers).
bool __quantum__rt__is_dynamic_qubit_management() { return !qubitPtrIsIndex; }

/// @brief Set whether or not the NVQIR runtime is running with dynamic qubit
/// management (qubits are pointers) or not (qubits are integers).
void __quantum__rt__set_dynamic_qubit_management(bool isDynamic) {
  qubitPtrIsIndex = !isDynamic;
}

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
    ScopedTraceWithContext("NVQIR::setExecutionContext", ctx->name);
    CUDAQ_INFO("Setting execution context: {}{}", ctx ? ctx->name : "basic",
               ctx->hasConditionalsOnMeasureResults ? " with conditionals"
                                                    : "");
    nvqir::getCircuitSimulatorInternal()->setExecutionContext(ctx);
  }
}

/// @brief Reset the Execution Context
void __quantum__rt__resetExecutionContext() {
  ScopedTraceWithContext("NVQIR::resetExecutionContext");
  CUDAQ_INFO("Resetting execution context.");
  nvqir::getCircuitSimulatorInternal()->resetExecutionContext();
}

/// @brief QIR function for allocated a qubit array
Array *__quantum__rt__qubit_allocate_array(std::uint64_t numQubits) {
  ScopedTraceWithContext("NVQIR::qubit_allocate_array", numQubits);
  __quantum__rt__initialize(0, nullptr);
  auto qubitIdxs =
      nvqir::getCircuitSimulatorInternal()->allocateQubits(numQubits);
  return vectorSizetToArray(qubitIdxs);
}

Array *__quantum__rt__qubit_allocate_array_with_state_complex32(
    std::uint64_t numQubits, std::complex<float> *data);

Array *__quantum__rt__qubit_allocate_array_with_state_complex64(
    std::uint64_t numQubits, std::complex<double> *data) {
  ScopedTraceWithContext("NVQIR::qubit_allocate_array_with_data_complex64",
                         numQubits);
  __quantum__rt__initialize(0, nullptr);
  if (nvqir::getCircuitSimulatorInternal()->isDoublePrecision()) {
    auto qubitIdxs = nvqir::getCircuitSimulatorInternal()->allocateQubits(
        numQubits, data, cudaq::simulation_precision::fp64);
    return vectorSizetToArray(qubitIdxs);
  }
  auto convertData = convertToComplex<float>(data, numQubits);
  auto qubitIdxs = nvqir::getCircuitSimulatorInternal()->allocateQubits(
      numQubits, convertData.get(), cudaq::simulation_precision::fp32);
  return vectorSizetToArray(qubitIdxs);
}

Array *
__quantum__rt__qubit_allocate_array_with_state_fp64(std::uint64_t numQubits,
                                                    double *data) {
  ScopedTraceWithContext("NVQIR::qubit_allocate_array_with_data_fp64",
                         numQubits);
  if (nvqir::getCircuitSimulatorInternal()->isDoublePrecision()) {
    auto convertData = convertToComplex<double>(data, numQubits);
    return __quantum__rt__qubit_allocate_array_with_state_complex64(
        numQubits, convertData.get());
  }
  auto convertData = convertToComplex<float>(data, numQubits);
  return __quantum__rt__qubit_allocate_array_with_state_complex32(
      numQubits, convertData.get());
}

Array *__quantum__rt__qubit_allocate_array_with_state_ptr(
    cudaq::SimulationState *state) {
  if (!state)
    throw std::invalid_argument("[NVQIR] Invalid simulation state encountered "
                                "in qubit array allocation.");
  ScopedTraceWithContext(
      "NVQIR::__quantum__rt__qubit_allocate_array_with_state_ptr",
      state->getNumQubits());

  __quantum__rt__initialize(0, nullptr);
  auto qubitIdxs = nvqir::getCircuitSimulatorInternal()->allocateQubits(
      state->getNumQubits(), state);
  return vectorSizetToArray(qubitIdxs);
}

Array *
__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr(std::uint64_t,
                                                         cudaq::state *state) {
  if (!state)
    throw std::invalid_argument("[NVQIR] Invalid state encountered "
                                "in qubit array allocation.");
  ScopedTraceWithContext(
      "NVQIR::__quantum__rt__qubit_allocate_array_with_cudaq_state_ptr",
      state->get_num_qubits());

  auto simStatePtr = cudaq::state_helper::getSimulationState(state);
  return __quantum__rt__qubit_allocate_array_with_state_ptr(simStatePtr);
}

Array *__quantum__rt__qubit_allocate_array_with_state_complex32(
    uint64_t numQubits, std::complex<float> *data) {
  ScopedTraceWithContext("NVQIR::qubit_allocate_array_with_data_complex32",
                         numQubits);
  __quantum__rt__initialize(0, nullptr);
  if (nvqir::getCircuitSimulatorInternal()->isSinglePrecision()) {
    auto qubitIdxs = nvqir::getCircuitSimulatorInternal()->allocateQubits(
        numQubits, data, cudaq::simulation_precision::fp32);
    return vectorSizetToArray(qubitIdxs);
  }
  auto convertData = convertToComplex<double>(data, numQubits);
  auto qubitIdxs = nvqir::getCircuitSimulatorInternal()->allocateQubits(
      numQubits, convertData.get(), cudaq::simulation_precision::fp64);
  return vectorSizetToArray(qubitIdxs);
}

Array *__quantum__rt__qubit_allocate_array_with_state_fp32(uint64_t numQubits,
                                                           float *data) {
  ScopedTraceWithContext("NVQIR::qubit_allocate_array_with_data_fp32",
                         numQubits);
  if (nvqir::getCircuitSimulatorInternal()->isSinglePrecision()) {
    auto convertData = convertToComplex<float>(data, numQubits);
    return __quantum__rt__qubit_allocate_array_with_state_complex32(
        numQubits, convertData.get());
  }
  auto convertData = convertToComplex<double>(data, numQubits);
  return __quantum__rt__qubit_allocate_array_with_state_complex64(
      numQubits, convertData.get());
}

/// @brief Once done, release the QIR qubit array
void __quantum__rt__qubit_release_array(Array *arr) {
  ScopedTraceWithContext("NVQIR::qubit_release_array", arr->size());
  for (std::size_t i = 0; i < arr->size(); i++) {
    auto arrayPtr = (*arr)[i];
    Qubit *idxVal = *reinterpret_cast<Qubit **>(arrayPtr);
    nvqir::getCircuitSimulatorInternal()->deallocate(idxVal->idx);
    delete idxVal;
  }
  delete arr;
  nvqir::ArrayTracker::getInstance().untrack(arr);
  return;
}

/// @brief Allocate a single QIR Qubit
Qubit *__quantum__rt__qubit_allocate() {
  ScopedTraceWithContext("NVQIR::allocate_qubit");
  __quantum__rt__initialize(0, nullptr);
  auto qubitIdx = nvqir::getCircuitSimulatorInternal()->allocateQubit();
  auto qubit = std::make_unique<Qubit>(qubitIdx);
  nvqir::allocatedSingleQubits.emplace_back(std::move(qubit));
  return nvqir::allocatedSingleQubits.back().get();
}

/// @brief Once done, release that qubit
void __quantum__rt__qubit_release(Qubit *q) {
  ScopedTraceWithContext("NVQIR::release_qubit");
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

void __quantum__rt__bool_record_output(bool val, const char *label) {
  quantumRTGenericRecordOutput("BOOL", (val ? "true" : "false"), label);
}

void __quantum__rt__int_record_output(std::int64_t val, const char *label) {
  quantumRTGenericRecordOutput("INT", val, label);
}

void __quantum__rt__double_record_output(double val, const char *label) {
  quantumRTGenericRecordOutput("DOUBLE", val, label);
}

void __quantum__rt__tuple_record_output(std::uint64_t len, const char *label) {
  quantumRTGenericRecordOutput("TUPLE", len, label);
}

void __quantum__rt__array_record_output(std::uint64_t len, const char *label) {
  quantumRTGenericRecordOutput("ARRAY", len, label);
}

#define ONE_QUBIT_QIS_FUNCTION(GATENAME)                                       \
  void QIS_FUNCTION_NAME(GATENAME)(Qubit * qubit) {                            \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    ScopedTraceWithContext("NVQIR::" + std::string(#GATENAME), targetIdx);     \
    nvqir::getCircuitSimulatorInternal()->GATENAME(targetIdx);                 \
  }                                                                            \
  void QIS_FUNCTION_CTRL_NAME(GATENAME)(Array * ctrlQubits, Qubit * qubit) {   \
    auto ctrlIdxs = arrayToVectorSizeT(ctrlQubits);                            \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    ScopedTraceWithContext("NVQIR::ctrl-" + std::string(#GATENAME), ctrlIdxs,  \
                           targetIdx);                                         \
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
    ScopedTraceWithContext("NVQIR::" + std::string(#GATENAME), param,          \
                           targetIdx);                                         \
    nvqir::getCircuitSimulatorInternal()->GATENAME(param, targetIdx);          \
  }                                                                            \
  void QIS_FUNCTION_BODY_NAME(GATENAME)(double param, Qubit *qubit) {          \
    QIS_FUNCTION_NAME(GATENAME)(param, qubit);                                 \
  }                                                                            \
  void QIS_FUNCTION_CTRL_NAME(GATENAME)(double param, Array *ctrlQubits,       \
                                        Qubit *qubit) {                        \
    auto ctrlIdxs = arrayToVectorSizeT(ctrlQubits);                            \
    auto targetIdx = qubitToSizeT(qubit);                                      \
    ScopedTraceWithContext("NVQIR::" + std::string(#GATENAME), param,          \
                           ctrlIdxs, targetIdx);                               \
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
  ScopedTraceWithContext("NVQIR::swap", qI, rI);
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
  nvqir::getCircuitSimulatorInternal()->phased_rx(theta, phi, qI);
}

void __quantum__qis__phased_rx__body(double theta, double phi, Qubit *q) {
  __quantum__qis__phased_rx(theta, phi, q);
}

auto u3_matrix = [](double theta, double phi, double lambda) {
  std::complex<double> i(0, 1.);
  std::vector<std::complex<double>> matrix{
      std::cos(theta / 2.), -std::exp(i * lambda) * std::sin(theta / 2.),
      std::exp(i * phi) * std::sin(theta / 2.),
      std::exp(i * (lambda + phi)) * std::cos(theta / 2.)};
  return matrix;
};

void __quantum__qis__u3(double theta, double phi, double lambda, Qubit *q) {
  auto qI = qubitToSizeT(q);
  nvqir::getCircuitSimulatorInternal()->u3(theta, phi, lambda, {}, qI);
}

void __quantum__qis__u3__ctl(double theta, double phi, double lambda,
                             Array *ctrls, Qubit *q) {
  auto ctrlIdxs = arrayToVectorSizeT(ctrls);
  auto qI = qubitToSizeT(q);
  nvqir::getCircuitSimulatorInternal()->u3(theta, phi, lambda, ctrlIdxs, qI);
}

// ASKME: Do we need `__quantum__qis__u3__body(...)`?

void __quantum__qis__cnot(Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  ScopedTraceWithContext("NVQIR::cnot", qI, rI);
  std::vector<std::size_t> controls{qI};
  nvqir::getCircuitSimulatorInternal()->x(controls, rI);
}

void __quantum__qis__cnot__body(Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  ScopedTraceWithContext("NVQIR::cnot", qI, rI);
  std::vector<std::size_t> controls{qI};
  nvqir::getCircuitSimulatorInternal()->x(controls, rI);
}

void __quantum__qis__cz__body(Qubit *q, Qubit *r) {
  auto qI = qubitToSizeT(q);
  auto rI = qubitToSizeT(r);
  ScopedTraceWithContext("NVQIR::cz", qI, rI);
  std::vector<std::size_t> controls{qI};
  nvqir::getCircuitSimulatorInternal()->z(controls, rI);
}

void __quantum__qis__reset(Qubit *q) {
  auto qI = qubitToSizeT(q);
  ScopedTraceWithContext("NVQIR::reset", qI);
  nvqir::getCircuitSimulatorInternal()->resetQubit(qI);
}

void __quantum__qis__reset__body(Qubit *q) { __quantum__qis__reset(q); }

Result *__quantum__qis__mz(Qubit *q) {
  auto qI = qubitToSizeT(q);
  ScopedTraceWithContext("NVQIR::mz", qI);
  auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, "");
  return b ? ResultOne : ResultZero;
}

Result *__quantum__qis__mz__body(Qubit *q, Result *r) {
  measQB2Res[q] = r;
  measRes2QB[r] = q;
  auto qI = qubitToSizeT(q);
  ScopedTraceWithContext("NVQIR::mz", qI);
  auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, "");
  measRes2Val[r] = b;
  return b ? ResultOne : ResultZero;
}

bool __quantum__qis__read_result__body(Result *result) {
  ScopedTraceWithContext("NVQIR::read_result");
  auto iter = measRes2Val.find(result);
  if (iter != measRes2Val.end())
    return iter->second;
  return ResultZeroVal;
}

bool __quantum__rt__read_result(Result *result) {
  return __quantum__qis__read_result__body(result);
}

Result *__quantum__qis__mz__to__register(Qubit *q, const char *name) {
  std::string regName(name);
  auto qI = qubitToSizeT(q);
  ScopedTraceWithContext("NVQIR::mz", qI, regName);
  auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, regName);
  return b ? ResultOne : ResultZero;
}

void __quantum__qis__exp_pauli(double theta, Array *qubits, char *pauliWord) {
  struct CLikeString {
    char *ptr = nullptr;
    int64_t length = 0;
  };
  auto *castedString = reinterpret_cast<CLikeString *>(pauliWord);
  std::string pauliWordStr(castedString->ptr, castedString->length);
  auto qubitsVec = arrayToVectorSizeT(qubits);
  nvqir::getCircuitSimulatorInternal()->applyExpPauli(
      theta, {}, qubitsVec, cudaq::spin_op::from_word(pauliWordStr));
  return;
}

void __quantum__qis__exp_pauli__ctl(double theta, Array *ctrls, Array *qubits,
                                    char *pauliWord) {
  struct CLikeString {
    char *ptr = nullptr;
    int64_t length = 0;
  };
  auto *castedString = reinterpret_cast<CLikeString *>(pauliWord);
  std::string pauliWordStr(castedString->ptr, castedString->length);
  auto ctrlQubitsVec = arrayToVectorSizeT(ctrls);
  auto qubitsVec = arrayToVectorSizeT(qubits);
  nvqir::getCircuitSimulatorInternal()->applyExpPauli(
      theta, ctrlQubitsVec, qubitsVec, cudaq::spin_op::from_word(pauliWordStr));
  return;
}

void __quantum__qis__exp_pauli__body(double theta, Array *qubits,
                                     char *pauliWord) {
  return __quantum__qis__exp_pauli(theta, qubits, pauliWord);
}

void __quantum__rt__result_record_output(Result *r, int8_t *name) {
  auto *ctx = nvqir::getCircuitSimulatorInternal()->getExecutionContext();
  if (ctx && ctx->name == "run") {

    std::string regName(reinterpret_cast<const char *>(name));
    auto qI = qubitToSizeT(measRes2QB[r]);
    auto b = nvqir::getCircuitSimulatorInternal()->mz(qI, regName);
    quantumRTGenericRecordOutput("RESULT", (b ? 1 : 0), regName.c_str());
    return;
  }

  if (name && qubitPtrIsIndex)
    __quantum__qis__mz__to__register(measRes2QB[r],
                                     reinterpret_cast<const char *>(name));
}

static std::vector<std::size_t> safeArrayToVectorSizeT(Array *arr) {
  if (!arr)
    return {};
  return arrayToVectorSizeT(arr);
}

// It may not always be possible for the compiler to reduce a program fully to
// QIR. In such cases, code generation may elect to produce a trap in the
// kernel, which calls this function. The trap should explain the issue to the
// user and about the kernel when executed.
void __quantum__qis__trap(std::int64_t code) {
  if (code == 0)
    throw std::runtime_error("could not autogenerate the adjoint of a kernel");
  if (code == 1)
    throw std::runtime_error("unsupported return type from entry-point kernel");
  throw std::runtime_error("code generation failure for target");
}

void __quantum__qis__apply_kraus_channel_double(std::int64_t krausChannelKey,
                                                double *params,
                                                std::size_t numParams,
                                                Array *qubits) {

  auto *ctx = nvqir::getCircuitSimulatorInternal()->getExecutionContext();
  if (!ctx)
    return;

  auto *noise = ctx->noiseModel;
  // per-spec, no noise model provided, emit warning, no application
  if (!noise)
    return cudaq::details::warn(
        "apply_noise called but no noise model provided.");

  std::vector<double> paramVec(params, params + numParams);
  auto channel = noise->get_channel(krausChannelKey, paramVec);
  nvqir::getCircuitSimulatorInternal()->applyNoise(channel,
                                                   arrayToVectorSizeT(qubits));
}

static void
__quantum__qis__apply_kraus_channel_float(std::int64_t krausChannelKey,
                                          float *params, std::size_t numParams,
                                          Array *qubits) {

  auto *ctx = nvqir::getCircuitSimulatorInternal()->getExecutionContext();
  if (!ctx)
    return;

  auto *noise = ctx->noiseModel;
  // per-spec, no noise model provided, emit warning, no application
  if (!noise)
    return cudaq::details::warn(
        "apply_noise called but no noise model provided.");

  std::vector<float> paramVec(params, params + numParams);
  auto channel = noise->get_channel(krausChannelKey, paramVec);
  nvqir::getCircuitSimulatorInternal()->applyNoise(channel,
                                                   arrayToVectorSizeT(qubits));
}

// The dataKind encoding is defined in QIRFunctionNames.h. 0 is float, 1 is
// double.
void __quantum__qis__apply_kraus_channel_generalized(
    std::int64_t dataKind, std::int64_t krausChannelKey, std::size_t numSpans,
    std::size_t numParams, std::size_t numTargets, ...) {
  va_list args;
  va_start(args, numTargets);

  auto vapplyKrausChannel = [&]<typename REAL>() {
    struct basic_span {
      REAL *_0;
      std::size_t _1;
    };

    REAL *params;
    std::size_t totalParams;

    // We assume either a span OR a list of REALs, but not both (for now).
    if (numSpans) {
      // 1a. A set of basic spans, `{ptr, i64}`. Pop the varargs and build the
      // spans.
      if (numSpans != 1)
        throw std::invalid_argument("Too many spans (> 1), not supported");
      basic_span *spans =
          reinterpret_cast<basic_span *>(alloca(numSpans * sizeof(basic_span)));
      for (std::size_t i = 0; i < numSpans; ++i) {
        auto *dataPtr = va_arg(args, REAL *);
        auto dataLen = va_arg(args, std::size_t);
        spans[i] = basic_span{dataPtr, dataLen};
      }

      // There can be only one.
      params = spans[0]._0;
      totalParams = spans[0]._1;
    } else {
      // 1b. A set of parameters. Pop the varargs as REAL values.
      params = reinterpret_cast<REAL *>(alloca(numParams * sizeof(REAL)));
      for (std::size_t i = 0; i < numParams; ++i) {
        auto *dblPtr = va_arg(args, REAL *);
        params[i] = *dblPtr;
      }

      totalParams = numParams;
    }

    // 2. A set of qubits. Pop the varargs as qubit* values.
    std::vector<Array *> qubits(numTargets);
    for (std::size_t i = 0; i < numTargets; ++i) {
      auto *qbPtr = va_arg(args, Array *);
      qubits[i] = qbPtr;
    }
    // There can be only one.
    Array *asArray = qubits[0];

    if constexpr (std::is_same_v<REAL, float>) {
      __quantum__qis__apply_kraus_channel_float(krausChannelKey, params,
                                                totalParams, asArray);
    } else {
      __quantum__qis__apply_kraus_channel_double(krausChannelKey, params,
                                                 totalParams, asArray);
    }
  };

  switch (dataKind) {
  case 0:
    vapplyKrausChannel.template operator()<float>();
    break;
  case 1:
    vapplyKrausChannel.template operator()<double>();
    break;
  default:
    throw std::runtime_error("apply_noise: unknown data kind.");
  }
  va_end(args);
}

namespace details {
struct FakeQubit {
  std::int8_t *id;
  bool negated;
};
static_assert(sizeof(FakeQubit) == sizeof(cudaq::qudit<2>) &&
              "FakeQubit must have the same memory layout as cudaq::qudit<>");
} // namespace details

std::vector<details::FakeQubit> *
__quantum__qis__convert_array_to_stdvector(Array *arr) {
  const std::size_t size = arr->size();
  std::vector<details::FakeQubit> *result = new std::vector<details::FakeQubit>;
  result->reserve(size);
  for (std::size_t i = 0; i < size; ++i) {
    (*result)[i].id = (*arr)[i];
    (*result)[i].negated = false;
  }
  return result;
}

void __quantum__qis__free_converted_stdvector(
    std::vector<details::FakeQubit> *veq) {
  delete veq;
}

void __quantum__qis__custom_unitary(std::complex<double> *unitary,
                                    Array *controls, Array *targets,
                                    const char *name) {
  auto ctrlsVec = safeArrayToVectorSizeT(controls);
  auto tgtsVec = arrayToVectorSizeT(targets);
  auto numQubits = tgtsVec.size();
  if (numQubits >= 64)
    throw std::invalid_argument("Too many qubits (>=64), not supported");
  auto nToPowTwo = (1ULL << numQubits);
  auto numElements = nToPowTwo * nToPowTwo;
  std::vector<std::complex<double>> unitaryMatrix(unitary,
                                                  unitary + numElements);
  nvqir::getCircuitSimulatorInternal()->applyCustomOperation(
      unitaryMatrix, ctrlsVec, tgtsVec, name);
}

void __quantum__qis__custom_unitary__adj(std::complex<double> *unitary,
                                         Array *controls, Array *targets,
                                         const char *name) {
  auto ctrlsVec = safeArrayToVectorSizeT(controls);
  auto tgtsVec = arrayToVectorSizeT(targets);
  auto numQubits = tgtsVec.size();
  if (numQubits >= 64)
    throw std::invalid_argument("Too many qubits (>=64), not supported");
  auto nToPowTwo = (1ULL << numQubits);

  std::vector<std::vector<std::complex<double>>> unitaryConj2D;
  for (std::size_t r = 0; r < nToPowTwo; r++) {
    std::vector<std::complex<double>> row;
    for (std::size_t c = 0; c < nToPowTwo; c++)
      row.push_back(std::conj(unitary[r * nToPowTwo + c]));
    unitaryConj2D.push_back(row);
  }
  for (std::size_t r = 0; r < nToPowTwo; r++)
    for (std::size_t c = 0; c < r; c++)
      std::swap(unitaryConj2D[r][c], unitaryConj2D[c][r]);
  std::vector<std::complex<double>> unitaryFlattened;
  for (auto const &row : unitaryConj2D)
    unitaryFlattened.insert(unitaryFlattened.end(), row.begin(), row.end());

  nvqir::getCircuitSimulatorInternal()->applyCustomOperation(
      unitaryFlattened, ctrlsVec, tgtsVec, name);
}

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
  CUDAQ_INFO("NVQIR measuring in pauli basis");
  ScopedTraceWithContext("NVQIR::observe_measure_body");

  auto *circuitSimulator = nvqir::getCircuitSimulatorInternal();
  auto *currentContext = circuitSimulator->getExecutionContext();

  // Some backends may better handle the observe task.
  // Let's give them that opportunity.
  if (currentContext->canHandleObserve) {
    circuitSimulator->flushGateQueue();
    auto result = circuitSimulator->observe(currentContext->spin.value());
    currentContext->expectationValue = result.expectation();
    currentContext->result = result.raw_data();
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
    CUDAQ_INFO("NVQIR reverse pauli bases change for measurement.");
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
  ScopedTraceWithContext("NVQIR::exp_body");

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
    CUDAQ_INFO(msg.c_str());
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

/// @brief Cleanup an result maps at the end of a QIR program to avoid leaking
/// results into the next program.
void __quantum__rt__clear_result_maps() {
  measQB2Res.clear();
  measRes2QB.clear();
  measRes2Val.clear();
}

/// This is the generalized version of invoke that does not use a va_list
/// argument. It provides a general interface to allow invoking a general
/// quantum operation, which may contain some number of rotation arguments
/// (double), control arguments (either qubits or arrays), and target arguments
/// (qubits). \p numRotationOperands and \p numTargetOperands must be no more
/// than 2. \p numTargetOperands must be at least 1. The arguments are passed as
/// arrays (built by the caller on the stack) as: \p params, \p controls, and \p
/// targets. \p isArrayAndLength is a buffer used to determine the type of the
/// control arguments and must be present if \p numControlOperands is non-zero.
/// The length of \p isArrayAndLength must also be \p numControlOperands.
static void commonInvokeWithRotationsControlsTargets(
    std::size_t numRotationOperands, double *params,
    std::size_t numControlOperands, std::size_t *isArrayAndLength,
    Qubit **controls, std::size_t numTargetOperands, Qubit **targets,
    void (*QISFunction)()) {
  if (numRotationOperands > 3)
    throw std::runtime_error("Invoke has invalid number of rotations.");
  if (numTargetOperands < 1 || numTargetOperands > 2)
    throw std::runtime_error("Invoke has invalid number of targets.");
  assert(numRotationOperands == 0 || params);
  assert(numControlOperands == 0 || (isArrayAndLength && controls));
  assert(numTargetOperands && targets);

  std::size_t numControls = 0;
  for (std::size_t i = 0; i < numControlOperands; i++)
    numControls += isArrayAndLength[i] ? isArrayAndLength[i] : 1;

  // Create the Control Array *, This should
  // be deallocated upon function exit.
  auto ctrlArray = std::make_unique<Array>(numControls, sizeof(std::size_t));

  for (std::size_t counter = 0, i = 0; i < numControlOperands; i++) {
    if (auto numQubitsInArray = isArrayAndLength[i]) {
      // this is an array
      Array *array = reinterpret_cast<Array *>(controls[i]);
      for (std::size_t k = 0; k < numQubitsInArray; k++) {
        auto qubitK = __quantum__rt__array_get_element_ptr_1d(array, k);
        Qubit **ctrliRawPtr =
            reinterpret_cast<Qubit **>(__quantum__rt__array_get_element_ptr_1d(
                ctrlArray.get(), counter++));
        *ctrliRawPtr = *reinterpret_cast<Qubit **>(qubitK);
      }
    } else {
      // this is a qubit
      Qubit *ctrli = controls[i];
      Qubit **ctrliRawPtr = reinterpret_cast<Qubit **>(
          __quantum__rt__array_get_element_ptr_1d(ctrlArray.get(), counter++));
      *ctrliRawPtr = ctrli;
    }
  }

  // Should be one more arg in there

  // Invoke the function. Only the control arguments are passed as a group to a
  // QIR function. That implies 6 cases must be generated.
  switch (numRotationOperands) {
  case 0: // No rotations.
    if (numTargetOperands == 1)
      reinterpret_cast<void (*)(Array *, Qubit *)>(QISFunction)(ctrlArray.get(),
                                                                targets[0]);
    else
      reinterpret_cast<void (*)(Array *, Qubit *, Qubit *)>(QISFunction)(
          ctrlArray.get(), targets[0], targets[1]);
    break;
  case 1: // One rotation.
    if (numTargetOperands == 1)
      reinterpret_cast<void (*)(double, Array *, Qubit *)>(QISFunction)(
          params[0], ctrlArray.get(), targets[0]);
    else
      reinterpret_cast<void (*)(double, Array *, Qubit *, Qubit *)>(
          QISFunction)(params[0], ctrlArray.get(), targets[0], targets[1]);
    break;
  case 2: // Two rotations.
    if (numTargetOperands == 1)
      reinterpret_cast<void (*)(double, double, Array *, Qubit *)>(QISFunction)(
          params[0], params[1], ctrlArray.get(), targets[0]);
    else
      reinterpret_cast<void (*)(double, double, Array *, Qubit *, Qubit *)>(
          QISFunction)(params[0], params[1], ctrlArray.get(), targets[0],
                       targets[1]);
    break;
  case 3: // Three rotations.
    if (numTargetOperands == 1)
      reinterpret_cast<void (*)(double, double, double, Array *, Qubit *)>(
          QISFunction)(params[0], params[1], params[2], ctrlArray.get(),
                       targets[0]);
    else
      reinterpret_cast<void (*)(double, double, double, Array *, Qubit *,
                                Qubit *)>(QISFunction)(
          params[0], params[1], params[2], ctrlArray.get(), targets[0],
          targets[1]);
    break;
  }
}

void generalizedInvokeWithRotationsControlsTargets(
    std::size_t numRotationOperands, std::size_t numControlArrayOperands,
    std::size_t numControlQubitOperands, std::size_t numTargetOperands,
    void (*QISFunction)(...), ...) {
  const std::size_t totalControls =
      numControlArrayOperands + numControlQubitOperands;
  double parameters[numRotationOperands];
  std::size_t arrayAndLength[totalControls];
  Qubit *controls[totalControls];
  Qubit *targets[numTargetOperands];
  std::size_t i;
  va_list args;
  va_start(args, QISFunction);
  for (i = 0; i < numRotationOperands; ++i)
    parameters[i] = va_arg(args, double);
  for (i = 0; i < numControlArrayOperands; ++i) {
    arrayAndLength[i] = va_arg(args, std::size_t);
    controls[i] = va_arg(args, Qubit *);
  }
  for (i = 0; i < numControlQubitOperands; ++i) {
    arrayAndLength[numControlArrayOperands + i] = 0;
    controls[numControlArrayOperands + i] = va_arg(args, Qubit *);
  }
  for (i = 0; i < numTargetOperands; ++i)
    targets[i] = va_arg(args, Qubit *);
  va_end(args);

  commonInvokeWithRotationsControlsTargets(
      numRotationOperands, parameters, totalControls, arrayAndLength, controls,
      numTargetOperands, targets, reinterpret_cast<void (*)()>(QISFunction));
}

/// @brief Utility function used by Quake->QIR to invoke a QIR QIS function
/// with a variadic list of control qubits.
void invokeWithControlQubits(const std::size_t numControlOperands,
                             void (*QISFunction)(Array *, Qubit *), ...) {
  // Start up the variadic arg processing
  va_list args;
  va_start(args, QISFunction);
  Qubit *targets[1];
  auto **controls =
      reinterpret_cast<Qubit **>(alloca(numControlOperands * sizeof(Qubit *)));
  auto *isArrayAndLength = reinterpret_cast<std::size_t *>(
      alloca(numControlOperands * sizeof(std::size_t)));
  for (std::size_t i = 0; i < numControlOperands; ++i) {
    controls[i] = va_arg(args, Qubit *);
    isArrayAndLength[i] = 0;
  }
  targets[0] = va_arg(args, Qubit *);
  va_end(args);

  // Invoke the function
  commonInvokeWithRotationsControlsTargets(
      /*rotations=*/0, nullptr, numControlOperands, isArrayAndLength, controls,
      /*targets=*/1, targets, reinterpret_cast<void (*)()>(QISFunction));
}

/// @brief Utility function used by Quake->QIR to invoke a QIR QIS function with
/// a variadic list of "quantum" arguments, where the control arguments can be
/// either Array or Qubit types.
void invokeWithControlRegisterOrQubits(std::size_t numControlOperands,
                                       std::size_t *isArrayAndLength,
                                       std::size_t numTargetOperands,
                                       void (*QISFunction)(Array *, Qubit *),
                                       ...) {
  va_list args;
  va_start(args, QISFunction);
  Qubit *targets[2];
  auto **controls =
      reinterpret_cast<Qubit **>(alloca(numControlOperands * sizeof(Qubit *)));
  for (std::size_t i = 0; i < numControlOperands; ++i)
    controls[i] = va_arg(args, Qubit *);
  assert(numTargetOperands >= 1 && numTargetOperands <= 2);
  targets[0] = va_arg(args, Qubit *);
  if (numTargetOperands == 2)
    targets[1] = va_arg(args, Qubit *);
  va_end(args);
  commonInvokeWithRotationsControlsTargets(
      /*rotations=*/0, nullptr, numControlOperands, isArrayAndLength, controls,
      numTargetOperands, targets, reinterpret_cast<void (*)()>(QISFunction));
}

/// @brief Utility function used by Quake->QIR to invoke a QIR QIS function with
/// a variadic list of "quantum" arguments, where the control arguments can be
/// either Array or Qubit types. This function is to be used for controlled
/// rotations.
void invokeRotationWithControlQubits(
    double param, const std::size_t numControlOperands,
    std::size_t *isArrayAndLength,
    void (*QISFunction)(double, Array *, Qubit *), ...) {
  va_list args;
  va_start(args, QISFunction);
  double params[1] = {param};
  Qubit *targets[1];
  auto **controls =
      reinterpret_cast<Qubit **>(alloca(numControlOperands * sizeof(Qubit *)));
  for (std::size_t i = 0; i < numControlOperands; ++i)
    controls[i] = va_arg(args, Qubit *);
  targets[0] = va_arg(args, Qubit *);
  va_end(args);
  commonInvokeWithRotationsControlsTargets(
      /*rotations=*/1, params, numControlOperands, isArrayAndLength, controls,
      /*targets=*/1, targets, reinterpret_cast<void (*)()>(QISFunction));
}

/// @brief Utility function same as `invokeRotationWithControlQubits`, but used
/// for U3 controlled rotations.
void invokeU3RotationWithControlQubits(
    double theta, double phi, double lambda,
    const std::size_t numControlOperands, std::size_t *isArrayAndLength,
    void (*QISFunction)(double, double, double, Array *, Qubit *), ...) {
  va_list args;
  va_start(args, QISFunction);
  double params[3] = {theta, phi, lambda};
  Qubit *targets[1];
  auto **controls =
      reinterpret_cast<Qubit **>(alloca(numControlOperands * sizeof(Qubit *)));
  for (std::size_t i = 0; i < numControlOperands; ++i)
    controls[i] = va_arg(args, Qubit *);
  targets[0] = va_arg(args, Qubit *);
  va_end(args);
  commonInvokeWithRotationsControlsTargets(
      /*rotations=*/3, params, numControlOperands, isArrayAndLength, controls,
      /*targets=*/1, targets, reinterpret_cast<void (*)()>(QISFunction));
}
}
