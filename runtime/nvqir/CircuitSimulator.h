/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 *******************************************************************************/

#pragma once

#include "Gates.h"
#include "Logger.h"
#include "MeasureCounts.h"
#include "NoiseModel.h"
#include "QIRTypes.h"

#include <cstdarg>
#include <cstddef>
#include <queue>
#include <sstream>
#include <string>

namespace nvqir {

/// @brief The CircuitSimulator defines a base class for all
/// simulators that are available to CUDAQ via the NVQIR library.
/// This base class handles Qubit allocation and deallocation,
/// execution context handling, and defines all quantum operations pure
/// virtual methods that subtypes must implement. Subtypes should be responsible
/// for evolution of the concrete wave function representation (e.g.,
/// statevector), sampling, and measurements.
class CircuitSimulator {
protected:
  /// @brief Flush the current queue of gates, i.e.
  /// apply them to the state. Internal and meant for
  /// subclasses to implement
  virtual void flushGateQueueImpl() = 0;

public:
  /// @brief The constructor
  CircuitSimulator() = default;
  /// @brief The destructor
  virtual ~CircuitSimulator() = default;

  /// @brief Flush the current queue of gates, i.e.
  /// apply them to the state.
  void flushGateQueue() { flushGateQueueImpl(); }

  /// @brief Set the current noise model to consider when
  /// simulating the state. This should be overridden by
  /// simulation strategies that support noise modeling.
  virtual void setNoiseModel(cudaq::noise_model &noise) = 0;

  /// @brief Compute the expected value of the given spin op
  /// with respect to the current state, <psi | H | psi>.
  virtual cudaq::ExecutionResult observe(const cudaq::spin_op &term) = 0;

  /// @brief Allocate a single qubit, return the qubit as a logical index
  virtual std::size_t allocateQubit() = 0;

  /// @brief Allocate `count` qubits.
  virtual std::vector<std::size_t> allocateQubits(const std::size_t count) = 0;

  /// @brief Deallocate the qubit with give idx
  virtual void deallocate(const std::size_t qubitIdx) = 0;

  /// @brief Reset the current execution context.
  virtual void resetExecutionContext() = 0;

  /// @brief Set the execution context
  virtual void setExecutionContext(cudaq::ExecutionContext *context) = 0;

  /// @brief Return the current execution context
  virtual cudaq::ExecutionContext *getExecutionContext() = 0;

  /// @brief Apply a custom operation described by a matrix of data
  /// represented as 1-D vector of elements in row-major order, as well
  /// as the the control qubit and target indices
  virtual void
  applyCustomOperation(const std::vector<std::complex<double>> &matrix,
                       const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &targets) = 0;

#define CIRCUIT_SIMULATOR_ONE_QUBIT(NAME)                                      \
  void NAME(const std::size_t qubitIdx) {                                      \
    std::vector<std::size_t> tmp;                                              \
    NAME(tmp, qubitIdx);                                                       \
  }                                                                            \
  virtual void NAME(const std::vector<std::size_t> &controls,                  \
                    const std::size_t qubitIdx) = 0;

#define CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(NAME)                            \
  void NAME(const double angle, const std::size_t qubitIdx) {                  \
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

  void u2(const double phi, const double lambda, const std::size_t qubitIdx) {
    std::vector<std::size_t> controls;
    u2(phi, lambda, controls, qubitIdx);
  }

  virtual void u2(const double phi, const double lambda,
                  const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) = 0;

  void u3(const double theta, const double phi, const double lambda,
          const std::size_t qubitIdx) {
    std::vector<std::size_t> controls;
    u3(theta, phi, lambda, controls, qubitIdx);
  }

  virtual void u3(const double theta, const double phi, const double lambda,
                  const std::vector<std::size_t> &controls,
                  const std::size_t qubitIdx) = 0;

  /// @brief Invoke the SWAP gate
  void swap(const std::size_t srcIdx, const std::size_t tgtIdx) {
    std::vector<std::size_t> tmp;
    swap(tmp, srcIdx, tgtIdx);
  }

  /// @brief Invoke a general multi-control swap gate
  virtual void swap(const std::vector<std::size_t> &ctrlBits,
                    const std::size_t srcIdx, const std::size_t tgtIdx) = 0;

  /// @brief Measure the qubit with given index
  virtual bool mz(const std::size_t qubitIdx) = 0;

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qubit, which we then use to do full state sampling when
  /// flushAnySamplingTask() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (mz(q)) and we measure the qubit,
  /// collapse the state, and then store the sample qubit for final full state
  /// sampling. We also return the bit result. If no execution context, just
  /// measure, collapse, and return the bit.
  virtual bool mz(const std::size_t qubitIdx,
                  const std::string &registerName) = 0;

  /// @brief Reset the qubit to the |0> state
  virtual void resetQubit(const std::size_t qubitIdx) = 0;

  /// @brief Sample the current multi-qubit state on the given qubit indices
  /// over a certain number of shots
  virtual cudaq::ExecutionResult
  sample(const std::vector<std::size_t> &qubitIdxs, const int shots) = 0;

  /// @brief Return the name of this CircuitSimulator
  virtual std::string name() const = 0;

  /// @brief Return a thread_local pointer to this CircuitSimulator
  virtual CircuitSimulator *clone() = 0;
};

/// @brief The CircuitSimulatorBase is the type that is meant to
/// be subclassed for new simulation strategies. The separation of
/// CircuitSimulator from CircuitSimulatorBase allows simulation sub-types
/// to specify the floating point precision for the simulation
template <typename ScalarType>
class CircuitSimulatorBase : public CircuitSimulator {
private:
  /// @brief Reference to the current circuit name.
  std::string currentCircuitName = "";

protected:
  /// @brief The current Execution Context (typically this is null,
  /// sampling, or spin_op observation.
  cudaq::ExecutionContext *executionContext = nullptr;

  /// @brief A tracker for allocating and deallocating qubit ids
  nvqir::QubitIdTracker tracker;

  /// @brief The number of qubits that have been allocated
  std::size_t nQubitsAllocated = 0;

  /// @brief The dimension of the multi-qubit state.
  std::size_t stateDimension = 0;

  /// @brief Vector containing qubit ids that are to be sampled
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

  /// @brief Under certain execution contexts, we'll deallocate
  /// before we are actually done with the execution task,
  /// this vector keeps track of qubit ids that are to be
  /// deallocated at a later time.
  std::vector<std::size_t> deferredDeallocation;

  /// @brief Map bit register names to the qubits that make it up
  std::unordered_map<std::string, std::vector<std::size_t>>
      registerNameToMeasuredQubit;

  /// @brief A GateApplicationTask consists of a
  /// matrix describing the quantum operation, a set of
  /// possible control qubit indices, and a set of target indices.
  struct GateApplicationTask {
    const std::string operationName;
    const std::vector<std::complex<ScalarType>> matrix;
    const std::vector<std::size_t> controls;
    const std::vector<std::size_t> targets;
    GateApplicationTask(const std::string &name,
                        const std::vector<std::complex<ScalarType>> &m,
                        const std::vector<std::size_t> &c,
                        const std::vector<std::size_t> &t)
        : operationName(name), matrix(m), controls(c), targets(t) {}
  };

  /// @brief The current queue of operations to execute
  std::queue<GateApplicationTask> gateQueue;

  /// @brief Get the name of the current circuit being executed.
  std::string getCircuitName() const { return currentCircuitName; }

  /// @brief Return the current multi-qubit state dimension
  std::size_t calculateStateDim(const std::size_t numQubits) {
    assert(numQubits < 64);
    return 1ULL << numQubits;
  }

  /// @brief Add a new qubit to the state representation.
  /// This is subclass specific.
  virtual void addQubitToState() = 0;

  /// @brief Subclass specific part of resetQubitState().
  /// It will be invoked by resetQubitState()
  virtual void resetQubitStateImpl() = 0;

  /// @brief Reset the qubit state back to dim = 0.
  void resetQubitState() {
    resetQubitStateImpl();
    nQubitsAllocated = 0;
    stateDimension = 0;
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
  bool handleBasicSampling(const std::size_t qubitIdx,
                           const std::string &regName) {
    if (executionContext && executionContext->name == "sample" &&
        !executionContext->hasConditionalsOnMeasureResults) {
      // Add the qubit to the sampling list
      sampleQubits.push_back(qubitIdx);

      // Configure the register name so we can operate on it
      std::string mutableName = regName;
      if (regName.empty())
        mutableName = cudaq::GlobalRegisterName;

      // Insert the sample qubit into the register name map
      auto iter = registerNameToMeasuredQubit.find(mutableName);
      if (iter == registerNameToMeasuredQubit.end())
        registerNameToMeasuredQubit.emplace(mutableName,
                                            std::vector<std::size_t>{qubitIdx});
      else
        iter->second.push_back(qubitIdx);

      return true;
    }

    return false;
  }

  /// @brief This function handles sampling in the presence of conditional
  /// statements on qubit measurement results. Specifically, it will keep
  /// track of a classical register for all measures encountered in the program
  /// and store mid-circuit measures in the corresponding register.
  void handleSamplingWithConditionals(const std::size_t qubitIdx,
                                      const std::string bitResult,
                                      const std::string &registerName) {
    // We still care about what qubit we are measuring if in the
    // sample-conditional context
    if (executionContext && executionContext->name == "sample" &&
        executionContext->hasConditionalsOnMeasureResults) {

      cudaq::info("Handling Sampling With Conditionals: {}, {}, {}", qubitIdx,
                  bitResult, registerName);
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

  /// @brief Utility function that returns a string-view of the current
  /// quantum instruction, intended for logging purposes.
  std::string gateToString(const std::string_view gateName,
                           const std::vector<std::size_t> &controls,
                           const std::vector<ScalarType> &parameters,
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

  /// @brief Execute a sampling task with the current set of sample qubits.
  void flushAnySamplingTasks(bool force = false) {
    if (sampleQubits.empty())
      return;

    if (executionContext->hasConditionalsOnMeasureResults && !force)
      return;

    // Sort the qubit indices
    std::sort(sampleQubits.begin(), sampleQubits.end());
    std::unique(sampleQubits.begin(), sampleQubits.end());

    cudaq::info("Sampling the current state, with measure qubits = {}",
                sampleQubits);

    // Ask the subtype to sample the current state
    auto execResult =
        sample(sampleQubits, executionContext->hasConditionalsOnMeasureResults
                                 ? 1
                                 : executionContext->shots);

    if (registerNameToMeasuredQubit.empty()) {
      executionContext->result.append(execResult);
    } else {

      for (auto &[regName, qubits] : registerNameToMeasuredQubit) {
        // Find the position of the qubits we have in the result bit string
        // Create a map of qubit to bit string location
        std::unordered_map<std::size_t, std::size_t> qubitLocMap;
        for (std::size_t i = 0; i < qubits.size(); i++) {
          auto iter =
              std::find(sampleQubits.begin(), sampleQubits.end(), qubits[i]);
          auto idx = std::distance(sampleQubits.begin(), iter);
          qubitLocMap.insert({qubits[i], idx});
        }

        cudaq::ExecutionResult tmp(regName);
        for (auto &[bits, count] : execResult.counts) {
          std::string b = "";
          for (auto &qb : qubits)
            b += bits[qubitLocMap[qb]];
          tmp.appendResult(b, count);
        }

        executionContext->result.append(tmp);
      }
    }

    sampleQubits.clear();
    registerNameToMeasuredQubit.clear();
  }

  /// @brief Enqueue a new gate application task
  void enqueueGate(const std::string name,
                   const std::vector<std::complex<ScalarType>> &matrix,
                   const std::vector<std::size_t> &controls,
                   const std::vector<std::size_t> &targets) {
    gateQueue.emplace(name, matrix, controls, targets);
  }

  /// @brief This pure virtual method is meant for subtypes
  /// to implement, and its goal is to apply the gate described
  /// by the GateApplicationTask to the subtype-specific state
  /// data representation.
  virtual void applyGate(const GateApplicationTask &task) = 0;

  /// @brief Provide a base-class method that can be invoked
  /// after every gate application and will apply any noise
  /// channels after the gate invocation based on a user-provided noise
  /// model. Unimplemented on the base class, sub-types can implement noise
  /// modeling.
  virtual void applyNoiseChannel(const std::string_view gateName,
                                 const std::vector<std::size_t> &qubits) {}

  /// @brief Flush the gate queue, run all queued gate
  /// application tasks.
  void flushGateQueueImpl() override {
    while (!gateQueue.empty()) {
      auto &next = gateQueue.front();
      applyGate(next);
      if (executionContext && executionContext->noiseModel) {
        std::vector<std::size_t> noiseQubits{next.controls.begin(),
                                             next.controls.end()};
        noiseQubits.insert(noiseQubits.end(), next.targets.begin(),
                           next.targets.end());
        applyNoiseChannel(next.operationName, noiseQubits);
      }
      gateQueue.pop();
    }
  }

public:
  /// @brief The constructor
  CircuitSimulatorBase() = default;
  /// @brief The destructor
  virtual ~CircuitSimulatorBase() = default;

  /// @brief Set the current noise model to consider when
  /// simulating the state. This should be overridden by
  /// simulation strategies that support noise modeling.
  void setNoiseModel(cudaq::noise_model &noise) override {
    // Fixme consider this as a warning instead of a hard error
    throw std::runtime_error(
        "The current backend does not support noise modeling.");
  }

  /// @brief Compute the expected value of the given spin op
  /// with respect to the current state, <psi | H | psi>.
  cudaq::ExecutionResult observe(const cudaq::spin_op &term) override {
    throw std::runtime_error("This CircuitSimulator does not implement "
                             "observe(const cudaq::spin_op &).");
  }

  /// @brief Allocate a single qubit, return the qubit as a logical index
  std::size_t allocateQubit() override {
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
  std::vector<std::size_t> allocateQubits(const std::size_t count) override {
    std::vector<std::size_t> qubits;
    for (std::size_t i = 0; i < count; i++) {
      qubits.push_back(allocateQubit());
    }
    return qubits;
  }

  /// @brief Deallocate the qubit with give idx
  void deallocate(const std::size_t qubitIdx) override {
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
      while (!gateQueue.empty())
        gateQueue.pop();
    }
  }

  /// @brief Reset the current execution context.
  void resetExecutionContext() override {
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

      // Flush the queue if there are any gates to apply
      flushGateQueue();

      // Flush any queued up sampling tasks
      flushAnySamplingTasks(/*force this*/ true);

      // Handle the processing for any mid circuit measurements
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
      currentCircuitName = "";
    }

    // Set the state data if requested.
    if (executionContext->name == "extract-state") {
      flushGateQueue();
      executionContext->simulationData = getStateData();
    }

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
  void setExecutionContext(cudaq::ExecutionContext *context) override {
    executionContext = context;
    executionContext->canHandleObserve = canHandleObserve();
    currentCircuitName = context->kernelName;
    cudaq::info("Setting current circuit name to {}", currentCircuitName);
  }

  /// @brief Return the current execution context
  cudaq::ExecutionContext *getExecutionContext() override {
    return executionContext;
  }

  /// @brief Apply a custom quantum operation
  void applyCustomOperation(const std::vector<std::complex<double>> &matrix,
                            const std::vector<std::size_t> &controls,
                            const std::vector<std::size_t> &targets) override {
    flushAnySamplingTasks();
    std::vector<std::complex<ScalarType>> actual;
    std::transform(matrix.begin(), matrix.end(), std::back_inserter(actual),
                   [](auto &&element) -> std::complex<ScalarType> {
                     if (!std::is_same_v<double, ScalarType>) {
                       return static_cast<std::complex<ScalarType>>(element);
                     } else {
                       return std::complex<ScalarType>(element.real(),
                                                       element.imag());
                     }
                   });
    enqueueGate("custom", actual, controls, targets);
  }

  template <typename QuantumOperation>
  void enqueueQuantumOperation(const std::vector<ScalarType> &angles,
                               const std::vector<std::size_t> &controls,
                               const std::vector<std::size_t> &targets) {
    flushAnySamplingTasks();
    QuantumOperation gate;
    cudaq::info(gateToString(gate.name(), controls, angles, targets));
    enqueueGate(gate.name(), gate.getGate(angles), controls, targets);
    if (executionContext && executionContext->noiseModel) {
      std::vector<std::size_t> noiseQubits{controls.begin(), controls.end()};
      noiseQubits.insert(noiseQubits.end(), targets.begin(), targets.end());
      applyNoiseChannel(gate.name(), noiseQubits);
    }
  }

#define CIRCUIT_SIMULATOR_ONE_QUBIT(NAME)                                      \
  using CircuitSimulator::NAME;                                                \
  void NAME(const std::vector<std::size_t> &controls,                          \
            const std::size_t qubitIdx) override {                             \
    enqueueQuantumOperation<nvqir::NAME<ScalarType>>(                          \
        {}, controls, std::vector<std::size_t>{qubitIdx});                     \
  }

#define CIRCUIT_SIMULATOR_ONE_QUBIT_ONE_PARAM(NAME)                            \
  using CircuitSimulator::NAME;                                                \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::size_t qubitIdx) override {                             \
    enqueueQuantumOperation<nvqir::NAME<ScalarType>>(                          \
        {static_cast<ScalarType>(angle)}, controls,                            \
        std::vector<std::size_t>{qubitIdx});                                   \
  }

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

  using CircuitSimulator::u2;
  void u2(const double phi, const double lambda,
          const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    std::vector<ScalarType> tmp{static_cast<ScalarType>(phi),
                                static_cast<ScalarType>(lambda)};

    enqueueQuantumOperation<nvqir::u2<ScalarType>>(
        tmp, controls, std::vector<std::size_t>{qubitIdx});
  }

  using CircuitSimulator::u3;
  void u3(const double theta, const double phi, const double lambda,
          const std::vector<std::size_t> &controls,
          const std::size_t qubitIdx) override {
    std::vector<ScalarType> tmp{static_cast<ScalarType>(theta),
                                static_cast<ScalarType>(phi),
                                static_cast<ScalarType>(lambda)};
    enqueueQuantumOperation<nvqir::u3<ScalarType>>(
        tmp, controls, std::vector<std::size_t>{qubitIdx});
  }

  using CircuitSimulator::swap;
  /// @brief Invoke a general multi-control swap gate
  void swap(const std::vector<std::size_t> &ctrlBits, const std::size_t srcIdx,
            const std::size_t tgtIdx) override {
    flushAnySamplingTasks();
    cudaq::info(gateToString("swap", ctrlBits, {}, {srcIdx, tgtIdx}));
    std::vector<std::complex<ScalarType>> matrix{
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}, {0.0, 0.0}, {0.0, 0.0},
        {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {1.0, 0.0}};
    enqueueGate("swap", matrix, ctrlBits,
                std::vector<std::size_t>{srcIdx, tgtIdx});
  }

  bool mz(const std::size_t qubitIdx) override { return mz(qubitIdx, ""); }

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qubit, which we then use to do full state sampling when
  /// flushAnySamplingTask() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (mz(q)) and we measure the qubit,
  /// collapse the state, and then store the sample qubit for final full state
  /// sampling. We also return the bit result. If no execution context, just
  /// measure, collapse, and return the bit.
  bool mz(const std::size_t qubitIdx,
          const std::string &registerName) override {
    // Flush the Gate Queue
    flushGateQueue();

    // If sampling, just store the bit, do nothing else.
    if (handleBasicSampling(qubitIdx, registerName))
      return true;

    // Get the actual measurement from the subtype measureQubit implementation
    auto measureResult = measureQubit(qubitIdx);
    auto bitResult = measureResult == true ? "1" : "0";

    // If this CUDAQ kernel has conditional statements on measure results
    // then we want to handle the sampling a bit differently.
    handleSamplingWithConditionals(qubitIdx, bitResult, registerName);

    // Return the result
    return measureResult;
  }
}; // namespace nvqir
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
  nvqir::CircuitSimulator *clone() override {                                  \
    thread_local static std::unique_ptr<nvqir::CircuitSimulator> simulator =   \
        std::make_unique<CLASSNAME>();                                         \
    return simulator.get();                                                    \
  }
