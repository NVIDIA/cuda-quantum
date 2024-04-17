/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "Gates.h"
#include "QIRTypes.h"
#include "common/Logger.h"
#include "common/MeasureCounts.h"
#include "common/NoiseModel.h"
#include "common/Timing.h"
#include "cudaq/host_config.h"
#include <cstdarg>
#include <cstddef>
#include <queue>
#include <sstream>
#include <string>
#include <variant>

namespace nvqir {

// @brief Collect summary data and print upon simulator termination
struct SummaryData {
  std::size_t gateCount = 0;
  std::size_t controlCount = 0;
  std::size_t targetCount = 0;
  std::size_t svIO = 0;
  std::size_t svFLOPs = 0;
  bool enabled = false;
  std::string name;
  SummaryData() {
    if (cudaq::isTimingTagEnabled(cudaq::TIMING_GATE_COUNT))
      enabled = true;
  }

  /// @brief Update state-vector-based statistics for a logic gate
  void svGateUpdate(const std::size_t nControls, const std::size_t nTargets,
                    const std::size_t stateDimension,
                    const std::size_t stateVectorSizeBytes) {
    assert(nControls <= 63);
    if (enabled) {
      gateCount++;
      controlCount += nControls;
      targetCount += nTargets;
      // Times 2 because operating on the state vector requires both reading
      // and writing.
      svIO += (2 * stateVectorSizeBytes) / (1 << nControls);
      // For each element of the state vector, 2 complex multiplies and 1
      // complex accumulate is needed. This is reduced if there if this is a
      // controlled operation.
      // Each complex multiply is 6 real ops.
      // So 2 complex multiplies and 1 complex addition is 2*6+2 = 14 ops.
      svFLOPs += stateDimension * (14 * nTargets) / (1 << nControls);
    }
  }

  ~SummaryData() {
    if (enabled) {
      cudaq::log("CircuitSimulator '{}' Total Program Metrics [tag={}]:", name,
                 cudaq::TIMING_GATE_COUNT);
      cudaq::log("Gate Count = {}", gateCount);
      cudaq::log("Control Count = {}", controlCount);
      cudaq::log("Target Count = {}", targetCount);
      cudaq::log("State Vector I/O (GB) = {:.6f}",
                 static_cast<double>(svIO) / 1e9);
      cudaq::log("State Vector GFLOPs = {:.6f}",
                 static_cast<double>(svFLOPs) / 1e9);
    }
  }
};

/// @brief The CircuitSimulator defines a base class for all
/// simulators that are available to CUDAQ via the NVQIR library.
/// This base class handles Qubit allocation and deallocation,
/// execution context handling, and defines all quantum operations pure
/// virtual methods that subtypes must implement. Subtypes should be responsible
/// for evolution of the concrete wave function representation (e.g.,
/// state vector), sampling, and measurements.
class CircuitSimulator {
protected:
  /// @brief Flush the current queue of gates, i.e.
  /// apply them to the state. Internal and meant for
  /// subclasses to implement
  virtual void flushGateQueueImpl() = 0;

  /// @brief Statistics collected over the life of the simulator.
  SummaryData summaryData;

public:
  /// @brief The constructor
  CircuitSimulator() = default;
  /// @brief The destructor
  virtual ~CircuitSimulator() = default;

  /// @brief Flush the current queue of gates, i.e.
  /// apply them to the state.
  void flushGateQueue() { flushGateQueueImpl(); }

  /// @brief Provide an opportunity for any tear-down
  /// tasks before MPI Finalize is invoked. Here we leave
  /// this unimplemented, it is meant for subclasses.
  virtual void tearDownBeforeMPIFinalize() {
    // do nothing
  }

  /// @brief Provide a mechanism for simulators to
  /// create and return a `SimulationState` instance from
  /// a user-specified data set.
  virtual std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &) = 0;

  /// @brief Set the current noise model to consider when
  /// simulating the state. This should be overridden by
  /// simulation strategies that support noise modeling.
  virtual void setNoiseModel(cudaq::noise_model &noise) = 0;

  virtual void setRandomSeed(std::size_t seed) {
    // do nothing
  }

  /// @brief Perform any flushing or synchronization to force that all
  /// previously applied gates have truly been applied by the underlying
  /// simulator.
  virtual void synchronize() {}

  /// @brief Apply exp(-i theta PauliTensorProd) to the underlying state.
  /// This must be provided by subclasses.
  virtual void applyExpPauli(double theta,
                             const std::vector<std::size_t> &controls,
                             const std::vector<std::size_t> &qubitIds,
                             const cudaq::spin_op &op) {
    if (op.is_identity()) {
      if (controls.empty()) {
        // exp(i*theta*Id) is noop if this is not a controlled gate.
        return;
      } else {
        // Throw an error if this exp_pauli(i*theta*Id) becomes a non-trivial
        // gate due to control qubits.
        // FIXME: revisit this once
        // https://github.com/NVIDIA/cuda-quantum/issues/483 is implemented.
        throw std::logic_error("Applying controlled global phase via exp_pauli "
                               "of identity operator is not supported");
      }
    }
    flushGateQueue();
    cudaq::info(" [CircuitSimulator decomposing] exp_pauli({}, {})", theta,
                op.to_string(false));
    std::vector<std::size_t> qubitSupport;
    std::vector<std::function<void(bool)>> basisChange;
    op.for_each_pauli([&](cudaq::pauli type, std::size_t qubitIdx) {
      if (type != cudaq::pauli::I)
        qubitSupport.push_back(qubitIds[qubitIdx]);

      if (type == cudaq::pauli::Y)
        basisChange.emplace_back([&, qubitIdx](bool reverse) {
          rx(!reverse ? M_PI_2 : -M_PI_2, qubitIds[qubitIdx]);
        });
      else if (type == cudaq::pauli::X)
        basisChange.emplace_back(
            [&, qubitIdx](bool) { h(qubitIds[qubitIdx]); });
    });

    if (!basisChange.empty())
      for (auto &basis : basisChange)
        basis(false);

    std::vector<std::pair<std::size_t, std::size_t>> toReverse;
    for (std::size_t i = 0; i < qubitSupport.size() - 1; i++) {
      x({qubitSupport[i]}, qubitSupport[i + 1]);
      toReverse.emplace_back(qubitSupport[i], qubitSupport[i + 1]);
    }

    // Since this is a compute-action-uncompute type circuit, we only need to
    // apply control on this rz gate.
    rz(-2.0 * theta, controls, qubitSupport.back());

    std::reverse(toReverse.begin(), toReverse.end());
    for (auto &[i, j] : toReverse)
      x({i}, j);

    if (!basisChange.empty()) {
      std::reverse(basisChange.begin(), basisChange.end());
      for (auto &basis : basisChange)
        basis(true);
    }
  }

  /// @brief Compute the expected value of the given spin op
  /// with respect to the current state, <psi | H | psi>.
  virtual cudaq::observe_result observe(const cudaq::spin_op &term) = 0;

  /// @brief Allocate a single qubit, return the qubit as a logical index
  virtual std::size_t allocateQubit() = 0;

  /// @brief Allocate `count` qubits.
  virtual std::vector<std::size_t>
  allocateQubits(std::size_t count, const void *state = nullptr,
                 cudaq::simulation_precision precision =
                     cudaq::simulation_precision::fp32) = 0;

  /// @brief Deallocate the qubit with give unique index
  virtual void deallocate(const std::size_t qubitIdx) = 0;

  /// @brief Deallocate all the provided qubits.
  virtual void deallocateQubits(const std::vector<std::size_t> &qubits) = 0;

  /// @brief Reset the current execution context.
  virtual void resetExecutionContext() = 0;

  /// @brief Set the execution context
  virtual void setExecutionContext(cudaq::ExecutionContext *context) = 0;

  /// @brief Return the current execution context
  virtual cudaq::ExecutionContext *getExecutionContext() = 0;

  /// @brief Whether or not this is a state vector simulator
  virtual bool isStateVectorSimulator() const { return false; }

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

  void phased_rx(const double phi, const double lambda,
                 const std::size_t qubitIdx) {
    std::vector<std::size_t> controls;
    phased_rx(phi, lambda, controls, qubitIdx);
  }

  virtual void phased_rx(const double phi, const double lambda,
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
  /// then we have a circuit that contains if (`mz(q)`) and we measure the
  /// qubit, collapse the state, and then store the sample qubit for final full
  /// state sampling. We also return the bit result. If no execution context,
  /// just measure, collapse, and return the bit.
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

  /// @brief A tracker for qubit allocation
  cudaq::QuditIdTracker tracker;

  /// @brief The number of qubits that have been allocated
  std::size_t nQubitsAllocated = 0;

  /// @brief The dimension of the multi-qubit state.
  std::size_t stateDimension = 0;

  /// @brief Keep track of the previous state dimension
  /// as we grow the state.
  std::size_t previousStateDimension = 0;

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

  /// @brief Keep track of the current number of qubits in batch mode
  std::size_t batchModeCurrentNumQubits = 0;

  /// @brief Environment variable name that allows a programmer to
  /// specify how expectation values should be computed. This
  /// defaults to true.
  constexpr static const char observeSamplingEnvVar[] =
      "CUDAQ_OBSERVE_FROM_SAMPLING";

  /// @brief A GateApplicationTask consists of a
  /// matrix describing the quantum operation, a set of
  /// possible control qubit indices, and a set of target indices.
  struct GateApplicationTask {
    const std::string operationName;
    const std::vector<std::complex<ScalarType>> matrix;
    const std::vector<std::size_t> controls;
    const std::vector<std::size_t> targets;
    const std::vector<ScalarType> parameters;
    GateApplicationTask(const std::string &name,
                        const std::vector<std::complex<ScalarType>> &m,
                        const std::vector<std::size_t> &c,
                        const std::vector<std::size_t> &t,
                        const std::vector<ScalarType> &params)
        : operationName(name), matrix(m), controls(c), targets(t),
          parameters(params) {}
  };

  /// @brief The current queue of operations to execute
  std::queue<GateApplicationTask> gateQueue;

  /// @brief Get the name of the current circuit being executed.
  std::string getCircuitName() const { return currentCircuitName; }

  /// @brief Return the current multi-qubit state dimension
  virtual std::size_t calculateStateDim(const std::size_t numQubits) {
    assert(numQubits < 64);
    return 1ULL << numQubits;
  }

  /// @brief Add a new qubit to the state representation.
  /// This is subclass specific.
  virtual void addQubitToState() = 0;

  /// @brief Subclass specific part of deallocateState().
  /// It will be invoked by deallocateState()
  virtual void deallocateStateImpl() = 0;

  /// @brief Reset the qubit state back to dim = 0.
  void deallocateState() {
    deallocateStateImpl();
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
  virtual std::unique_ptr<cudaq::SimulationState> getSimulationState() {
    throw std::runtime_error(
        "Simulation data not available for this simulator backend.");
  }

  /// @brief Handle basic sampling tasks by storing the qubit index for
  /// processing in resetExecutionContext. Return true to indicate this is
  /// sampling and to exit early. False otherwise.
  bool handleBasicSampling(const std::size_t qubitIdx,
                           const std::string &regName) {
    if (executionContext && executionContext->name == "sample" &&
        !executionContext->hasConditionalsOnMeasureResults) {
      // Add the qubit to the sampling list
      sampleQubits.push_back(qubitIdx);

      auto processForRegName = [&](const std::string &regStr) {
        // Insert the sample qubit into the register name map
        auto iter = registerNameToMeasuredQubit.find(regStr);
        if (iter == registerNameToMeasuredQubit.end())
          registerNameToMeasuredQubit.emplace(
              regStr, std::vector<std::size_t>{qubitIdx});
        else if (std::find(iter->second.begin(), iter->second.end(),
                           qubitIdx) == iter->second.end())
          iter->second.push_back(qubitIdx);
      };

      // Insert into global register and named register (if it exists)
      processForRegName(cudaq::GlobalRegisterName);
      if (!regName.empty())
        processForRegName(regName);

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
      std::string mutableRegisterName = registerName;

      // If no registerName, we'll just sample normally
      if (registerName.empty()) {
        // Either this is library mode and we have register names attached
        // to the execution context
        if (midCircuitSampleResults.size() <
            executionContext->registerNames.size()) {
          mutableRegisterName =
              executionContext->registerNames[midCircuitSampleResults.size()];
        } else {
          // or no register names, in which case we'll just treat it as
          // a regular sampled qubit and drop out
          sampleQubits.push_back(qubitIdx);
          return;
        }
      }

      cudaq::info("Handling Sampling With Conditionals: {}, {}, {}", qubitIdx,
                  bitResult, mutableRegisterName);
      // See if we've observed this register before, if not
      // start a vector of bit results, if we have, add the
      // bit result to the existing vector
      auto iter = midCircuitSampleResults.find(mutableRegisterName);
      if (iter == midCircuitSampleResults.end())
        midCircuitSampleResults.emplace(mutableRegisterName,
                                        std::vector<std::string>{bitResult});
      else
        iter->second.push_back(bitResult);

      // If this register is the same as last time, then we are
      // writing to a bit vector register (auto var = mz(qreg))
      if (lastMidCircuitRegisterName == mutableRegisterName)
        vectorRegisters.push_back(mutableRegisterName);

      // Store the last register name
      lastMidCircuitRegisterName = mutableRegisterName;
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

  /// @brief Return true if the current execution is in batch mode
  bool isInBatchMode() {
    if (!executionContext)
      return false;

    if (executionContext->totalIterations == 0)
      return false;

    return true;
  }

  /// @brief Return true if the current execution is the
  /// last execution of batch mode.
  bool isLastBatch() {
    return executionContext && executionContext->batchIteration > 0 &&
           executionContext->batchIteration ==
               executionContext->totalIterations - 1;
  }

  /// @brief Add the given number of qubits to the state.
  virtual void addQubitsToState(std::size_t count,
                                const void *state = nullptr) {
    if (state != nullptr)
      throw std::runtime_error("State initialization must be handled by "
                               "subclasses, override addQubitsToState.");
    for (std::size_t i = 0; i < count; i++)
      addQubitToState();
  }

  /// @brief Execute a sampling task with the current set of sample qubits.
  void flushAnySamplingTasks(bool force = false) {
    if (sampleQubits.empty())
      return;

    if (executionContext->hasConditionalsOnMeasureResults && !force)
      return;

    // Sort the qubit indices
    std::sort(sampleQubits.begin(), sampleQubits.end());
    auto last = std::unique(sampleQubits.begin(), sampleQubits.end());
    sampleQubits.erase(last, sampleQubits.end());

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
        // Measurements are sorted according to qubit allocation order
        std::sort(qubits.begin(), qubits.end());
        auto last = std::unique(qubits.begin(), qubits.end());
        qubits.erase(last, qubits.end());

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

  /// @brief Add a new gate application task to the queue
  void enqueueGate(const std::string name,
                   const std::vector<std::complex<ScalarType>> &matrix,
                   const std::vector<std::size_t> &controls,
                   const std::vector<std::size_t> &targets,
                   const std::vector<ScalarType> &params) {
    if (executionContext && executionContext->name == "tracer") {
      std::vector<cudaq::QuditInfo> controlsInfo, targetsInfo;
      for (auto &c : controls)
        controlsInfo.emplace_back(2, c);
      for (auto &t : targets)
        targetsInfo.emplace_back(2, t);

      std::vector<double> anglesProcessed;
      if constexpr (std::is_same_v<ScalarType, double>)
        anglesProcessed = params;
      else {
        for (auto &a : params)
          anglesProcessed.push_back(static_cast<ScalarType>(a));
      }

      executionContext->kernelTrace.appendInstruction(
          name, anglesProcessed, controlsInfo, targetsInfo);
      return;
    }

    gateQueue.emplace(name, matrix, controls, targets, params);
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
      if (isStateVectorSimulator() && summaryData.enabled)
        summaryData.svGateUpdate(
            next.controls.size(), next.targets.size(), stateDimension,
            stateDimension * sizeof(std::complex<ScalarType>));
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
    // For CUDA-based simulators, this calls cudaDeviceSynchronize()
    synchronize();
  }

  /// @brief Set the current state to the |0> state,
  /// retaining the current number of qubits.
  virtual void setToZeroState() = 0;

  /// @brief Return true if expectation values should be computed from
  /// sampling + parity of bit strings.
  /// Default is to enable observe from sampling, i.e., simulating the
  /// change-of-basis circuit for each term.
  ///
  /// The environment variable "CUDAQ_OBSERVE_FROM_SAMPLING" can be used to turn
  /// on or off this setting.
  bool shouldObserveFromSampling(bool defaultConfig = true) {
    if (auto envVar = std::getenv(observeSamplingEnvVar); envVar) {
      std::string asString = envVar;
      std::transform(asString.begin(), asString.end(), asString.begin(),
                     [](auto c) { return std::tolower(c); });
      if (asString == "false" || asString == "off" || asString == "0")
        return false;
      if (asString == "true" || asString == "on" || asString == "1")
        return true;
    }

    return defaultConfig;
  }

public:
  /// @brief The constructor
  CircuitSimulatorBase() = default;
  /// @brief The destructor
  virtual ~CircuitSimulatorBase() = default;

  /// @brief Create a simulation-specific SimulationState
  /// instance from a user-provided data set.
  std::unique_ptr<cudaq::SimulationState>
  createStateFromData(const cudaq::state_data &data) override {
    return getSimulationState()->createFromData(data);
  }

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
  cudaq::observe_result observe(const cudaq::spin_op &term) override {
    throw std::runtime_error("This CircuitSimulator does not implement "
                             "observe(const cudaq::spin_op &).");
  }

  /// @brief Allocate a single qubit, return the qubit as a logical index
  std::size_t allocateQubit() override {
    // Get a new qubit index
    auto newIdx = tracker.getNextIndex();

    if (isInBatchMode()) {
      batchModeCurrentNumQubits++;
      // In batch mode, we might already have an allocated state that
      // has been set to |0..0>. We can reuse it as is, if the next qubit
      // index is smaller than number of qubits of this allocated state.
      if (newIdx < nQubitsAllocated)
        return newIdx;
    }

    cudaq::info("Allocating new qubit with idx {} (nQ={}, dim={})", newIdx,
                nQubitsAllocated, stateDimension);

    // Increment the number of qubits and set
    // the new state dimension
    previousStateDimension = stateDimension;
    nQubitsAllocated++;
    stateDimension = calculateStateDim(nQubitsAllocated);

    // Tell the subtype to grow the state representation
    addQubitToState();

    // May be that the state grows enough that we
    // want to handle observation via sampling
    if (executionContext)
      executionContext->canHandleObserve = canHandleObserve();

    // return the new qubit index
    return newIdx;
  }

  /// @brief Allocate `count` qubits.
  std::vector<std::size_t>
  allocateQubits(std::size_t count, const void *state = nullptr,
                 cudaq::simulation_precision precision =
                     cudaq::simulation_precision::fp32) override {
    // Make sure if someone gives us state data, that the precision
    // is correct for this simulation.
    if (state != nullptr) {
      if constexpr (std::is_same_v<ScalarType, float>) {
        if (precision == cudaq::simulation_precision::fp64)
          throw std::runtime_error(
              "Invalid user-provided state data. Simulator "
              "is FP32 but state data is FP64.");
      } else {
        if (precision == cudaq::simulation_precision::fp32)
          throw std::runtime_error(
              "Invalid user-provided state data. Simulator "
              "is FP64 but state data is FP32.");
      }
    }

    std::vector<std::size_t> qubits;
    for (std::size_t i = 0; i < count; i++)
      qubits.emplace_back(tracker.getNextIndex());

    if (isInBatchMode()) {
      // Store the current number of qubits requested
      batchModeCurrentNumQubits += count;

      // We have an allocated state, it has been set to |0>,
      // we want to reuse it as is. If the state needs to grow, then
      // we will ask the subtype to add more qubits.
      if (qubits.back() < nQubitsAllocated)
        count = 0;
      else
        count = qubits.back() + 1 - nQubitsAllocated;
    }

    cudaq::info("Allocating {} new qubits.", count);

    previousStateDimension = stateDimension;
    nQubitsAllocated += count;
    stateDimension = calculateStateDim(nQubitsAllocated);

    // Tell the subtype to allocate more qubits
    addQubitsToState(count, state);

    // May be that the state grows enough that we
    // want to handle observation via sampling
    if (executionContext)
      executionContext->canHandleObserve = canHandleObserve();

    return qubits;
  }

  /// @brief Deallocate the qubit with give index
  void deallocate(const std::size_t qubitIdx) override {
    if (executionContext && executionContext->name != "tracer") {
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
    if (tracker.allDeallocated()) {
      cudaq::info("Deallocated all qubits, reseting state vector.");
      // all qubits deallocated,
      deallocateState();
      while (!gateQueue.empty())
        gateQueue.pop();
    }
  }

  /// @brief Deallocate all requested qubits. If the number of qubits
  /// is equal to the number of allocated qubits, then clear the entire
  /// state at once.
  void deallocateQubits(const std::vector<std::size_t> &qubits) override {
    // Do nothing if there are no allocated qubits.
    if (nQubitsAllocated == 0)
      return;

    if (executionContext) {
      for (auto &qubitIdx : qubits) {
        cudaq::info("Deferring qubit {} deallocation", qubitIdx);
        deferredDeallocation.push_back(qubitIdx);
      }
      return;
    }

    if (qubits.size() == tracker.numAllocated()) {
      cudaq::info("Deallocate all qubits.");
      deallocateState();
      for (auto &q : qubits)
        tracker.returnIndex(q);
      return;
    }

    for (auto &q : qubits)
      deallocate(q);
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
        if (isInBatchMode())
          sampleQubits.resize(batchModeCurrentNumQubits);
        else
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

      // Reorder the global register (if necessary). This might be necessary if
      // the mapping pass had run and we want to undo the shuffle that occurred
      // during mapping.
      if (!executionContext->reorderIdx.empty()) {
        executionContext->result.reorder(executionContext->reorderIdx);
        executionContext->reorderIdx.clear();
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
      executionContext->simulationState = getSimulationState();
    }

    // Deallocate the deferred qubits, but do so
    // without explicit qubit reset.
    for (auto &deferred : deferredDeallocation)
      tracker.returnIndex(deferred);

    bool shouldSetToZero = isInBatchMode() && !isLastBatch();
    executionContext = nullptr;

    // Reset the state if we've deallocated all qubits.
    if (tracker.allDeallocated()) {
      if (shouldSetToZero) {
        cudaq::info("In batch mode currently, reset state to |0>");
        // Do not deallocate the state, but reset it to |0>
        setToZeroState();
      } else {
        cudaq::info("Deallocated all qubits, reseting state vector.");
        // all qubits deallocated,
        deallocateState();
      }
    }

    batchModeCurrentNumQubits = 0;
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
    cudaq::info(gateToString("custom_unitary", controls, {}, targets));
    enqueueGate("custom", actual, controls, targets, {});
  }

  template <typename QuantumOperation>
  void enqueueQuantumOperation(const std::vector<ScalarType> &angles,
                               const std::vector<std::size_t> &controls,
                               const std::vector<std::size_t> &targets) {
    flushAnySamplingTasks();
    QuantumOperation gate;
    // This is a very hot section of code. Don't form the log string unless
    // we're actually going to use it.
    if (cudaq::details::should_log(cudaq::details::LogLevel::info))
      cudaq::info(gateToString(gate.name(), controls, angles, targets));
    enqueueGate(gate.name(), gate.getGate(angles), controls, targets, angles);
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

  using CircuitSimulator::phased_rx;
  void phased_rx(const double phi, const double lambda,
                 const std::vector<std::size_t> &controls,
                 const std::size_t qubitIdx) override {
    std::vector<ScalarType> tmp{static_cast<ScalarType>(phi),
                                static_cast<ScalarType>(lambda)};
    enqueueQuantumOperation<nvqir::phased_rx<ScalarType>>(
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
                std::vector<std::size_t>{srcIdx, tgtIdx}, {});
  }

  bool mz(const std::size_t qubitIdx) override { return mz(qubitIdx, ""); }

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qubit, which we then use to do full state sampling when
  /// flushAnySamplingTask() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (`mz(q)`) and we measure the
  /// qubit, collapse the state, and then store the sample qubit for final
  /// full state sampling. We also return the bit result. If no execution
  /// context, just measure, collapse, and return the bit.
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
