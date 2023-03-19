/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "Logger.h"
#include "MeasureCounts.h"
#include "NoiseModel.h"
#include "QIRTypes.h"
#include <cstdarg>
#include <cstddef>
#include <sstream>
#include <string>

///
/// This file defines the CircuitSimulator, which is meant to be
/// the base class for all simulators provided to CUDA Quantum via the 
/// NVQIR library.
///

namespace nvqir {

/// The CircuitSimulator defines a base class for all simulators
/// that are available to CUDA Quantum via the NVQIR library.
/// This base class handles Qubit allocation and deallocation,
/// execution context handling, and defines all quantum operations pure
/// virtual methods that subtypes must implement. Subtypes should be responsible
/// for evolution of the concrete wave function representation (e.g.,
/// statevector), sampling, and measurements.
class CircuitSimulator {
protected:
  /// The current Execution Context (typically this is null,
  /// sampling, or spin_op observation.
  cudaq::ExecutionContext *executionContext = nullptr;

  /// A tracker for allocating and deallocating qubit ids
  nvqir::QubitIdTracker tracker;

  /// The number of qubits that have been allocated
  std::size_t nQubitsAllocated = 0;

  /// The dimension of the multi-qubit state.
  std::size_t stateDimension = 0;

  /// Vector containing qubit ids that are to be sampled
  std::vector<std::size_t> sampleQubits;

  /// @brief Map of register name to observed bit results for mid-circuit
  /// sampling
  std::unordered_map<std::string, std::vector<std::string>>
      midCircuitSampleResults;

  /// @brief Store the last observed register name, this will help us
  /// know if we are writing to a classical bit vector
  std::string lastMidCircuitRegisterName = "";

  /// @brief Vector storing register names that are bit vectors
  std::vector<std::string> vectorRegisters;

  /// Under certain execution contexts, we'll deallocate
  /// before we are actually done with the execution task,
  /// this vector keeps track of qubit ids that are to be
  /// deallocated at a later time.
  std::vector<std::size_t> deferredDeallocation;

  /// Return the current multi-qubit state dimension
  std::size_t calculateStateDim(const int n_qubits) { return 1ULL << n_qubits; }

  /// Add a new qubit to the state representation.
  /// This is subclass specific.
  virtual void addQubitToState() = 0;

  /// Reset the qubit state back to dim = 0.
  void resetQubitState() {
    this->resetQubitStateImpl();
    nQubitsAllocated = 0;
    stateDimension = 0;
  }
  /// Subclass specific part of resetQubitState().
  /// It will be invoked by resetQubitState()
  virtual void resetQubitStateImpl() = 0;

  /// Convert from little endian to big endian.
  std::size_t bigEndian(const int n_qubits, const int bit) {
    return n_qubits - bit - 1;
  }

  /// @brief Perform the actual mechanics of measuring a qubit,
  /// left as a task for concrete subtypes.
  virtual bool measureQubit(const std::size_t qubitIdx) = 0;

  /// @brief Return true if this CircuitSimulator can
  /// handle <psi | H | psi> instead of NVQIR applying measure
  /// basis quantum gates to change to the Z basis and sample.
  virtual bool canHandleObserve() { return false; }

  /// @brief Return the internal state representation. This
  /// is meant for subtypes to override
  virtual cudaq::State getStateData() { return {}; }

  /// @brief Handle basic sampling tasks by storing the qubit index for
  /// processing in resetExecutionContext. Return true to indicate this is
  /// sampling and to exit early. False otherwise.
  /// @param qubitIdx
  /// @return
  bool handleBasicSampling(const std::size_t qubitIdx) {
    if (executionContext && executionContext->name == "sample" &&
        !executionContext->hasConditionalsOnMeasureResults) {
      sampleQubits.push_back(qubitIdx);
      return true;
    }
    return false;
  }

  /// @brief This function handles sampling in the presence of conditional
  /// statements on qubit measurement results. Specifically, it will keep
  /// track of a classical register for all measures encountered in the program
  /// and store mid-circuit measures in the corresponding register.
  /// @param qubitIdx
  /// @param bitResult
  void handleSamplingWithConditionals(const std::size_t qubitIdx,
                                      const std::string bitResult,
                                      const std::string &registerName) {
    // We still care about what qubit we are measuring if in the
    // sample-conditional context
    if (executionContext && executionContext->name == "sample" &&
        executionContext->hasConditionalsOnMeasureResults) {

      // If no registerName, we'll just sample normally
      if (registerName.empty()) {
        sampleQubits.push_back(qubitIdx);
        return;
      }

      // See if we've observed this register before, if not
      // start a vector of bit results, if we have, add the
      // bit result to the existing vector
      auto iter = midCircuitSampleResults.find(registerName);
      if (iter == midCircuitSampleResults.end())
        midCircuitSampleResults.emplace(registerName,
                                        std::vector<std::string>{bitResult});
      else
        iter->second.push_back(bitResult);

      // If this register is the same as last time, then we are
      // writing to a bit vector register (auto var = mz(qreg))
      if (lastMidCircuitRegisterName == registerName)
        vectorRegisters.push_back(registerName);

      // Store the last register name
      lastMidCircuitRegisterName = registerName;
    }
  }

  /// Utility function that returns a string-view of the current
  /// quantum instruction, intended for logging purposes.
  std::string gateToString(const std::string_view gateName,
                           const std::vector<std::size_t> &controls,
                           const std::vector<double> &parameters,
                           const std::vector<std::size_t> &targets) {
    std::string angleStr = "";
    if (!parameters.empty()) {
      angleStr = std::to_string(parameters[0]);
      for (std::size_t i = 1; i < parameters.size(); i++)
        angleStr += ", " + std::to_string(parameters[i]);
      angleStr += ", ";
    }

    std::stringstream bits, ret;
    if (!controls.empty()) {
      bits << controls[0];
      for (size_t i = 1; i < controls.size(); i++) {
        bits << ", " << controls[i];
      }
      bits << ", " << targets[0];
      for (size_t i = 1; i < targets.size(); i++) {
        bits << ", " << targets[i];
      }
      ret << "(apply) ctrl-" << gateName << "(" << angleStr << bits.str()
          << ")";
    } else {
      bits << targets[0];
      for (size_t i = 1; i < targets.size(); i++) {
        bits << ", " << targets[i];
      }
      ret << "(apply) " << gateName << "(" << angleStr << bits.str() << ")";
    }
    return ret.str();
  }

public:
  CircuitSimulator() = default;
  virtual ~CircuitSimulator() = default;

  virtual void setNoiseModel(cudaq::noise_model &noise) {
    // Fixme consider this as a warning instead of a hard error
    throw std::runtime_error(
        "The current backend does not support noise modeling.");
  }

  /// @brief Compute the expected value of the given spin op
  /// with respect to the current state, <psi | H | psi>.
  virtual cudaq::ExecutionResult observe(const cudaq::spin_op &op) {
    throw std::runtime_error("This CircuitSimulator does not implement "
                             "observe(const cudaq::spin_op &).");
  }

  /// @brief Allocate a single qubit, return the qubit as a logical index
  /// @return qubit idx
  virtual std::size_t allocateQubit() {
    // Get a new qubit index
    auto newIdx = tracker.getNextIndex();

    cudaq::info("Allocating new qubit with idx {} (nQ={}, dim={})", newIdx,
                nQubitsAllocated, stateDimension);

    // Increment the number of qubits and set
    // the new state dimension
    nQubitsAllocated++;
    stateDimension = calculateStateDim(nQubitsAllocated);

    // Tell the subtype to grow the state representation
    addQubitToState();

    // return the new qubit index
    return newIdx;
  }

  /// @brief Allocate `count` qubits.
  /// @param count number of qubits to allocate
  /// @return  vector of allocated qubit ids
  virtual std::vector<std::size_t> allocateQubits(const std::size_t count) {
    std::vector<std::size_t> qubits;
    for (std::size_t i = 0; i < count; i++) {
      qubits.push_back(allocateQubit());
    }
    return qubits;
  }

  /// @brief Deallocate the qubit with give idx
  /// @param qubitIdx  The id of the qubit to deallocate
  virtual void deallocate(const std::size_t qubitIdx) {
    if (executionContext) {
      cudaq::info("Deferring qubit {} deallocation", qubitIdx);
      deferredDeallocation.push_back(qubitIdx);
      return;
    }

    cudaq::info("Deallocating qubit {}", qubitIdx);

    // Reset the qubit
    resetQubit(qubitIdx);

    // Return the index to the tracker
    tracker.returnIndex(qubitIdx);
    --nQubitsAllocated;

    // Reset the state if we've deallocated all qubits.
    if (tracker.numAvailable() == tracker.totalNumQubits()) {
      cudaq::info("Deallocated all qubits, reseting state vector.");
      // all qubits deallocated,
      resetQubitState();
    }
  }

  /// @brief Reset the current execution context.
  virtual void resetExecutionContext() {
    // If null, do nothing
    if (!executionContext)
      return;

    // Get the ExecutionContext name
    auto execContextName = executionContext->name;

    // If we are sampling...
    if (execContextName.find("sample") != std::string::npos) {
      // Sample the state over the specified number of shots
      if (sampleQubits.empty()) {
        sampleQubits.resize(nQubitsAllocated);
        std::iota(sampleQubits.begin(), sampleQubits.end(), 0);
      }

      sampleQubits.erase(std::unique(sampleQubits.begin(), sampleQubits.end()),
                         sampleQubits.end());

      cudaq::info("Sampling the current state, with measure qubits = {}",
                  sampleQubits);

      // Sample and give the results to the ExecutionContext
      auto sampleResult =
          sample(sampleQubits, executionContext->hasConditionalsOnMeasureResults
                                   ? 1
                                   : executionContext->shots);
      executionContext->result.append(sampleResult);

      for (auto &m : midCircuitSampleResults) {
        // Get the register name and the vector of bit results
        auto regName = m.first;
        auto bitResults = m.second;
        cudaq::ExecutionResult counts(regName);

        if (std::find(vectorRegisters.begin(), vectorRegisters.end(),
                      regName) != vectorRegisters.end()) {
          // this is a vector register
          std::string bitStr = "";
          for (std::size_t j = 0; j < bitResults.size(); j++)
            bitStr += bitResults[j];

          counts.appendResult(bitStr, 1);

        } else {
          // Not a vector, collate all bits into a 1 qubit counts dict
          for (std::size_t j = 0; j < bitResults.size(); j++) {
            counts.appendResult(bitResults[j], 1);
          }
        }
        executionContext->result.append(counts);
      }

      // Clear the sample bits for the next run
      sampleQubits.clear();
      midCircuitSampleResults.clear();
      lastMidCircuitRegisterName = "";
    }

    // Set the state data if requested.
    if (executionContext->name == "extract-state")
      executionContext->simulationData = getStateData();

    executionContext = nullptr;

    // Deallocate the deferred qubits, but do so
    // without explicit qubit reset.
    for (auto &deferred : deferredDeallocation)
      tracker.returnIndex(deferred);

    // Reset the state if we've deallocated all qubits.
    if (tracker.numAvailable() == tracker.totalNumQubits()) {
      cudaq::info("Deallocated all qubits, reseting state vector.");
      // all qubits deallocated,
      resetQubitState();
    }

    deferredDeallocation.clear();
  }

  /// @brief Set the execution context
  /// @param context The execution context (sampling, observe)
  virtual void setExecutionContext(cudaq::ExecutionContext *context) {
    executionContext = context;
    executionContext->canHandleObserve = canHandleObserve();
  }

  /// @brief Return the current execution context
  /// @return context
  cudaq::ExecutionContext *getExecutionContext() { return executionContext; }

  /// The following pragmas define the virtual and pure virtual
  /// quantum instruction methods for the CircuitSimulator.

#define CIRCUIT_SIMULATOR_ONE_QUBIT(NAME)                                      \
  virtual void NAME(const std::size_t qubitIdx) {                              \
    std::vector<std::size_t> tmp;                                              \
    NAME(tmp, qubitIdx);                                                       \
  }                                                                            \
  virtual void NAME(const std::vector<std::size_t> &controls,                  \
                    const std::size_t qubitIdx) = 0;

#define CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(NAME)                            \
  virtual void NAME(const double angle, const std::size_t qubitIdx) {          \
    std::vector<std::size_t> tmp;                                              \
    NAME(angle, tmp, qubitIdx);                                                \
  }                                                                            \
  virtual void NAME(const double angle,                                        \
                    const std::vector<std::size_t> &controls,                  \
                    const std::size_t qubitIdx) = 0;

  /// @brief The X gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(x)
  /// @brief The Y gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(y)
  /// @brief The Z gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(z)
  /// @brief The H gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(h)
  /// @brief The S gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(s)
  /// @brief The T gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(t)
  /// @brief The Sdg gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(sdg)
  /// @brief The Tdg gate
  CIRCUIT_SIMULATOR_ONE_QUBIT(tdg)
  /// @brief The RX gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(rx)
  /// @brief The RY gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(ry)
  /// @brief The RZ gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(rz)
  /// @brief The Phase gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(r1)
  /// @brief The IBM U1 gate
  CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(u1)

// Undef those preprocessor defines.
#undef CIRCUIT_SIMULATOR_ONE_QUBIT
#undef CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM

  virtual void u2(const double phi, const double lambda,
                  const std::size_t qubitIdx) {
    std::vector<std::size_t> controls;
    u2(phi, lambda, controls, qubitIdx);
  }

  virtual void u2(const double phi, const double lambda,
                  const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) = 0;

  virtual void u3(const double theta, const double phi, const double lambda,
                  const std::size_t qubitIdx) {
    std::vector<std::size_t> controls;
    u3(theta, phi, lambda, controls, qubitIdx);
  }

  virtual void u3(const double theta, const double phi, const double lambda,
                  const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) = 0;

  /// @brief  Invoke the SWAP gate
  virtual void swap(const std::size_t srcIdx, const std::size_t tgtIdx) {
    std::vector<std::size_t> tmp;
    swap(tmp, srcIdx, tgtIdx);
  }

  /// @brief Invoke a general multi-control swap gate
  /// @param ctrlBits
  /// @param srcIdx
  /// @param tgtIdx
  virtual void swap(const std::vector<std::size_t> &ctrlBits,
                    const std::size_t srcIdx, const std::size_t tgtIdx) = 0;
  /// @brief Measure the qubit with given index
  /// @param qubitIdx The unique id for the qubit
  /// @return the measurement result
  virtual bool mz(const std::size_t qubitIdx) { return mz(qubitIdx, ""); }

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qubit, which we then use to do full state sampling when
  /// resetExecutionContext() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (mz(q)) and we measure the qubit,
  /// collapse the state, and then store the sample qubit for final full state
  /// sampling. We also return the bit result. If no execution context, just
  /// measure, collapse, and return the bit.
  virtual bool mz(const std::size_t qubitIdx, const std::string &registerName) {
    // If sampling, just store the bit, do nothing else.
    if (handleBasicSampling(qubitIdx))
      return true;

    // Get the actual measurement from the subtype measureQubit implementation
    auto measureResult = measureQubit(qubitIdx);
    auto bitResult = measureResult == true ? "1" : "0";

    // If this kernel has conditional statements on measure results
    // then we want to handle the sampling a bit differently.
    handleSamplingWithConditionals(qubitIdx, bitResult, registerName);

    // Return the result
    return measureResult;
  }

  /// @brief Reset the qubit to the |0> state
  /// @param qubitIdx the qubit idx
  virtual void resetQubit(const std::size_t qubitIdx) = 0;

  /// @brief Is this a valid backend. Subtypes can override this to
  /// check conditions for their usage.
  /// @return valid
  virtual bool isValidBackend() { return true; }

  /// @brief Sample the current multi-qubit state on the given qubit indices
  /// over a certain number of shots
  /// @param qubitIdxs The qubits to sample
  /// @param shots The shot count
  /// @return result Results containing counts map and expected value
  virtual cudaq::ExecutionResult
  sample(const std::vector<std::size_t> &qubitIdxs, const int shots) = 0;

  virtual std::string name() const = 0;

  /// @brief Return a thread_local pointer to this CircuitSimulator
  /// @return
  virtual CircuitSimulator *clone() = 0;
};
} // namespace nvqir

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define NVQIR_REGISTER_SIMULATOR(CLASSNAME, PRINTED_NAME)                      \
  extern "C" {                                                                 \
  nvqir::CircuitSimulator *getCircuitSimulator() {                             \
    thread_local static std::unique_ptr<nvqir::CircuitSimulator> simulator =   \
        std::make_unique<CLASSNAME>();                                         \
    return simulator.get();                                                    \
  }                                                                            \
  nvqir::CircuitSimulator *CONCAT(getCircuitSimulator_, PRINTED_NAME)() {      \
    thread_local static std::unique_ptr<nvqir::CircuitSimulator> simulator =   \
        std::make_unique<CLASSNAME>();                                         \
    return simulator.get();                                                    \
  }                                                                            \
  }

#define NVQIR_SIMULATOR_CLONE_IMPL(CLASSNAME)                                  \
  CircuitSimulator *clone() override {                                         \
    thread_local static std::unique_ptr<nvqir::CircuitSimulator> simulator =   \
        std::make_unique<CLASSNAME>();                                         \
    return simulator.get();                                                    \
  }
