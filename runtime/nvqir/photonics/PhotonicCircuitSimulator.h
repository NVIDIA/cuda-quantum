/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "PhotonicGates.h"
#include "PhotonicState.h"

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/MeasureCounts.h"
#include "common/Timing.h"
#include "cudaq/host_config.h"

#include <cstdarg>
#include <cstddef>
#include <queue>
#include <sstream>
#include <string>
#include <variant>

namespace nvqir {

enum class QuditOrdering { lsb, msb };

// @brief Collect summary data and print upon simulator termination
struct PSSummaryData {
  std::size_t gateCount = 0;
  std::size_t controlCount = 0;
  std::size_t targetCount = 0;
  std::size_t svIO = 0;
  std::size_t svFLOPs = 0;
  bool enabled = false;
  std::string name;
  PSSummaryData() {
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

  ~PSSummaryData() {
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

/// @brief The PhotonicCircuitSimulator defines a base class for all photonic
/// simulators that are available to CUDA-Q via the NVQIR library.
/// This base class handles Qudit allocation and deallocation,
/// execution context handling, and defines all quantum operations pure
/// virtual methods that subtypes must implement. Subtypes should be responsible
/// for evolution of the concrete wave function representation (e.g.,
/// state vector), sampling, and measurements.
class PhotonicCircuitSimulator {
protected:
  /// @brief The number of levels for the qudits
  std::size_t levels;

  /// @brief Flush the current queue of gates, i.e.
  /// apply them to the state. Internal and meant for
  /// subclasses to implement
  virtual void flushGateQueueImpl() = 0;

  /// @brief Statistics collected over the life of the simulator.
  PSSummaryData summaryData;

public:
  /// @brief The constructor
  PhotonicCircuitSimulator() = default;
  /// @brief The destructor
  virtual ~PhotonicCircuitSimulator() = default;

  void setLevels(std::size_t newLevels) { levels = newLevels; }

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
  /// create and return a `PhotonicState` instance from
  /// a user-specified data set.
  virtual std::unique_ptr<PhotonicState>
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

  /// @brief Compute the expected value of the given spin op
  /// with respect to the current state, <psi | H | psi>.
  // virtual cudaq::observe_result observe(const cudaq::spin_op &term) = 0;

  /// @brief Allocate a single qudit, return the qudit as a logical index
  virtual std::size_t allocateQudit() = 0;

  /// @brief Allocate `count` qudits.
  virtual std::vector<std::size_t>
  allocateQudits(std::size_t count, const void *state = nullptr,
                 cudaq::simulation_precision precision =
                     cudaq::simulation_precision::fp32) = 0;

  virtual std::vector<std::size_t>
  allocateQudits(std::size_t count, const PhotonicState *state) = 0;

  /// @brief Deallocate the qudit with give unique index
  virtual void deallocate(const std::size_t quditIdx) = 0;

  /// @brief Deallocate all the provided qudits.
  virtual void deallocateQudits(const std::vector<std::size_t> &qudits) = 0;

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
  /// as the the control qudit and target indices
  virtual void
  applyCustomOperation(const std::vector<std::complex<double>> &matrix,
                       const std::vector<std::size_t> &controls,
                       const std::vector<std::size_t> &targets,
                       const std::string_view customUnitaryName = "") = 0;

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT(NAME)                             \
  void NAME(const std::size_t quditIdx) {                                      \
    std::vector<std::size_t> tmp;                                              \
    NAME(tmp, quditIdx);                                                       \
  }                                                                            \
  virtual void NAME(const std::vector<std::size_t> &controls,                  \
                    const std::size_t quditIdx) = 0;

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM(NAME)                   \
  void NAME(const double angle, const std::size_t quditIdx) {                  \
    std::vector<std::size_t> tmp;                                              \
    NAME(angle, tmp, quditIdx);                                                \
  }                                                                            \
  virtual void NAME(const double angle,                                        \
                    const std::vector<std::size_t> &controls,                  \
                    const std::size_t quditIdx) = 0;

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_TWO_PARAM(NAME)                   \
  void NAME(const double angle, const std::vector<std::size_t> quditIdxs) {    \
    std::vector<std::size_t> tmp;                                              \
    NAME(angle, tmp, quditIdxs);                                               \
  }                                                                            \
  virtual void NAME(const double angle,                                        \
                    const std::vector<std::size_t> &controls,                  \
                    const std::vector<std::size_t> quditIdxs) = 0;

  /// @brief The plus gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT(plus)
  /// @brief The phase_shift gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM(phase_shift)
  /// @brief The beam_splitter gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_TWO_PARAM(beam_splitter)

// Undef those preprocessor defines.
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_TWO_PARAM

  /// @brief Measure the qudit with given index
  virtual std::size_t mz(const std::size_t quditIdx) = 0;

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qudit, which we then use to do full state sampling when
  /// flushAnySamplingTask() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (`mz(q)`) and we measure the
  /// qudit, collapse the state, and then store the sample qudit for final full
  /// state sampling. We also return the bit result. If no execution context,
  /// just measure, collapse, and return the bit.
  virtual std::size_t mz(const std::size_t quditIdx,
                         const std::string &registerName) = 0;

  /// @brief Reset the qudit to the |0> state
  virtual void resetQudit(const std::size_t quditIdx) = 0;

  /// @brief Sample the current multi-qudit state on the given qudit indices
  /// over a certain number of shots
  virtual cudaq::ExecutionResult
  sample(const std::vector<std::size_t> &quditIdxs, const int shots) = 0;

  /// @brief Return the name of this PhotonicCircuitSimulator
  virtual std::string name() const = 0;

  /// @brief Return a thread_local pointer to this PhotonicCircuitSimulator
  virtual PhotonicCircuitSimulator *clone() = 0;

  /// Determine the (preferred) precision of the simulator.
  virtual bool isSinglePrecision() const = 0;

  bool isDoublePrecision() const { return !isSinglePrecision(); }

}; // PhotonicCircuitSimulator

/// @brief The PhotonicCircuitSimulatorBase is the type that is meant to
/// be subclassed for new simulation strategies. The separation of
/// PhotonicCircuitSimulator from PhotonicCircuitSimulatorBase allows simulation
/// sub-types to specify the floating point precision for the simulation
template <typename ScalarType>
class PhotonicCircuitSimulatorBase : public PhotonicCircuitSimulator {

private:
  /// @brief Reference to the current circuit name.
  std::string currentCircuitName = "";

  /// @brief Return true if the simulator is in the tracer mode.
  bool isInTracerMode() const {
    return executionContext && executionContext->name == "tracer";
  }

protected:
  /// @brief The current Execution Context (typically this is null,
  /// sampling, or spin_op observation.
  cudaq::ExecutionContext *executionContext = nullptr;

  /// @brief A tracker for qudit allocation
  cudaq::QuditIdTracker tracker;

  /// @brief The number of qudits that have been allocated
  std::size_t nQuditsAllocated = 0;

  /// @brief The dimension of the multi-qudit state.
  std::size_t stateDimension = 0;

  /// @brief Keep track of the previous state dimension
  /// as we grow the state.
  std::size_t previousStateDimension = 0;

  /// @brief Vector containing qudit ids that are to be sampled
  std::vector<std::size_t> sampleQudits;

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
  /// this vector keeps track of qudit ids that are to be
  /// deallocated at a later time.
  std::vector<std::size_t> deferredDeallocation;

  /// @brief Map bit register names to the qudits that make it up
  std::unordered_map<std::string, std::vector<std::size_t>>
      registerNameToMeasuredQudit;

  /// @brief Keep track of the current number of qudits in batch mode
  std::size_t batchModeCurrentNumQudits = 0;

  /// @brief Environment variable name that allows a programmer to
  /// specify how expectation values should be computed. This
  /// defaults to true.
  static constexpr const char observeSamplingEnvVar[] =
      "CUDAQ_OBSERVE_FROM_SAMPLING";

  /// @brief A GateApplicationTask consists of a
  /// matrix describing the quantum operation, a set of
  /// possible control qudit indices, and a set of target indices.
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

  /// @brief Return the current multi-qudit state dimension
  virtual std::size_t calculateStateDim(const std::size_t numQudits) {
    assert(numQudits < 30);
    cudaq::info("[calculateStateDim] levels {},  numQudits {}.", levels,
                numQudits);
    return std::pow(levels, numQudits);
  }

  /// @brief Grow the state vector by one qudit.
  virtual void addQuditToState() = 0;

  /// @brief Reset the qudit state.
  virtual void deallocateStateImpl() = 0;

  /// @brief Reset the qudit state back to dim = 0.
  void deallocateState() {
    deallocateStateImpl();
    nQuditsAllocated = 0;
    stateDimension = 0;
  }
  /// @brief Measure the qudit and return the result. Collapse the
  /// state vector.
  virtual int measureQudit(const std::size_t quditIdx) = 0;

  /// @brief Return true if this PhotonicCircuitSimulator can
  /// handle <psi | H | psi> instead of NVQIR applying measure
  /// basis quantum gates to change to the Z basis and sample.
  virtual bool canHandleObserve() { return false; }

  /// @brief Return the internal state representation. This
  /// is meant for subtypes to override
  virtual std::unique_ptr<PhotonicState> getSimulationState() {
    throw std::runtime_error(
        "Simulation data not available for this simulator backend.");
  }

  /// @brief Handle basic sampling tasks by storing the qudit index for
  /// processing in resetExecutionContext. Return true to indicate this is
  /// sampling and to exit early. False otherwise.
  bool handleBasicSampling(const std::size_t quditIdx,
                           const std::string &regName) {
    if (executionContext && executionContext->name == "sample" &&
        !executionContext->hasConditionalsOnMeasureResults) {
      // Add the qudit to the sampling list
      sampleQudits.push_back(quditIdx);

      auto processForRegName = [&](const std::string &regStr) {
        // Insert the sample qudit into the register name map
        auto iter = registerNameToMeasuredQudit.find(regStr);
        if (iter == registerNameToMeasuredQudit.end())
          registerNameToMeasuredQudit.emplace(
              regStr, std::vector<std::size_t>{quditIdx});
        else if (std::find(iter->second.begin(), iter->second.end(),
                           quditIdx) == iter->second.end())
          iter->second.push_back(quditIdx);
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
  /// statements on qudit measurement results. Specifically, it will keep
  /// track of a classical register for all measures encountered in the
  /// program and store mid-circuit measures in the corresponding register.
  void handleSamplingWithConditionals(const std::size_t quditIdx,
                                      const std::string bitResult,
                                      const std::string &registerName) {
    // We still care about what qudit we are measuring if in the
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
          // a regular sampled qudit and drop out
          sampleQudits.push_back(quditIdx);
          return;
        }
      }

      cudaq::info("Handling Sampling With Conditionals: {}, {}, {}", quditIdx,
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

  /// @brief Override the default sized allocation of qudits
  /// here to be a bit more efficient than the default implementation
  virtual void addQuditsToState(std::size_t count,
                                const void *state = nullptr) {
    if (state != nullptr)
      throw std::runtime_error("State initialization must be handled by "
                               "subclasses, override addQuditsToState.");
    for (std::size_t i = 0; i < count; i++)
      addQuditToState();
  }

  virtual void addQuditsToState(const PhotonicState &state) {
    throw std::runtime_error("State initialization must be handled by "
                             "subclasses, override addQubitsToState.");
  }

  /// @brief Execute a sampling task with the current set of sample qudits.
  void flushAnySamplingTasks(bool force = false) {
    if (sampleQudits.empty())
      return;

    if (executionContext->hasConditionalsOnMeasureResults && !force)
      return;

    // Sort the qudit indices
    std::sort(sampleQudits.begin(), sampleQudits.end());
    auto last = std::unique(sampleQudits.begin(), sampleQudits.end());
    sampleQudits.erase(last, sampleQudits.end());

    cudaq::info("Sampling the current state, with measure qudits = {}",
                sampleQudits);

    // Ask the subtype to sample the current state
    auto execResult =
        sample(sampleQudits, executionContext->hasConditionalsOnMeasureResults
                                 ? 1
                                 : executionContext->shots);

    if (registerNameToMeasuredQudit.empty()) {
      executionContext->result.append(execResult);
    } else {

      for (auto &[regName, qudits] : registerNameToMeasuredQudit) {
        // Measurements are sorted according to qudit allocation order
        std::sort(qudits.begin(), qudits.end());
        auto last = std::unique(qudits.begin(), qudits.end());
        qudits.erase(last, qudits.end());

        // Find the position of the qudits we have in the result bit string
        // Create a map of qudit to bit string location
        std::unordered_map<std::size_t, std::size_t> quditLocMap;
        for (std::size_t i = 0; i < qudits.size(); i++) {
          auto iter =
              std::find(sampleQudits.begin(), sampleQudits.end(), qudits[i]);
          auto idx = std::distance(sampleQudits.begin(), iter);
          quditLocMap.insert({qudits[i], idx});
        }

        cudaq::ExecutionResult tmp(regName);
        for (auto &[bits, count] : execResult.counts) {
          std::string b = "";
          for (auto &qb : qudits)
            b += bits[quditLocMap[qb]];
          tmp.appendResult(b, count);
        }

        executionContext->result.append(tmp);
      }
    }
    deallocateQudits(sampleQudits);
    sampleQudits.clear();
    registerNameToMeasuredQudit.clear();
  }

  /// @brief Add a new gate application task to the queue
  void enqueueGate(const std::string name,
                   const std::vector<std::complex<ScalarType>> &matrix,
                   const std::vector<std::size_t> &controls,
                   const std::vector<std::size_t> &targets,
                   const std::vector<ScalarType> &params) {
    if (isInTracerMode()) {
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

  /// @brief This pure method is meant for subtypes
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
                                 const std::vector<std::size_t> &qudits) {}

  /// @brief Flush the gate queue, run all queued gate
  /// application tasks.
  void flushGateQueueImpl() override {
    while (!gateQueue.empty()) {
      auto &next = gateQueue.front();
      if (isStateVectorSimulator() && summaryData.enabled)
        summaryData.svGateUpdate(
            next.controls.size(), next.targets.size(), stateDimension,
            stateDimension * sizeof(std::complex<ScalarType>));
      try {
        applyGate(next);
      } catch (std::exception &e) {
        while (!gateQueue.empty())
          gateQueue.pop();
        throw e;
      } catch (...) {
        while (!gateQueue.empty())
          gateQueue.pop();
        throw std::runtime_error("Unknown exception in applyGate");
      }
      // if (executionContext && executionContext->noiseModel) {
      //   std::vector<std::size_t> noiseQudits{next.controls.begin(),
      //                                        next.controls.end()};
      //   noiseQudits.insert(noiseQudits.end(), next.targets.begin(),
      //                      next.targets.end());
      //   applyNoiseChannel(next.operationName, noiseQudits);
      // }
      gateQueue.pop();
    }
    // // For CUDA-based simulators, this calls cudaDeviceSynchronize()
    synchronize();
  }

  /// @brief Set the current state to the |0> state,
  /// retaining the current number of qudits.
  virtual void setToZeroState() = 0;

  /// @brief Return true if expectation values should be computed from
  /// sampling + parity of bit strings.
  /// Default is to enable observe from sampling, i.e., simulating the
  /// change-of-basis circuit for each term.
  ///
  /// The environment variable "CUDAQ_OBSERVE_FROM_SAMPLING" can be used to
  /// turn on or off this setting.
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

  bool isSinglePrecision() const override {
    return std::is_same_v<ScalarType, float>;
  }

  /// @brief Return this simulator's qudit ordering.
  QuditOrdering getQuditOrdering() const { return QuditOrdering::lsb; }

public:
  /// @brief The constructor
  PhotonicCircuitSimulatorBase() = default;

  /// @brief The destructor
  virtual ~PhotonicCircuitSimulatorBase() = default;

  /// @brief Create a simulation-specific PhotonicState
  /// instance from a user-provided data set.
  std::unique_ptr<PhotonicState>
  createStateFromData(const cudaq::state_data &data) override {
    return getSimulationState()->createPSFromData(data);
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
  // cudaq::observe_result observe(const cudaq::spin_op &term) override {
  //   throw std::runtime_error("This PhotonicCircuitSimulator does not
  //   implement "
  //                            "observe(const cudaq::spin_op &).");
  // }

  /// @brief Allocate a single qudit, return the qudit as a logical index
  std::size_t allocateQudit() override {
    // Get a new qudit index
    auto newIdx = tracker.getNextIndex();

    if (isInBatchMode()) {
      batchModeCurrentNumQudits++;
      // In batch mode, we might already have an allocated state that
      // has been set to |0..0>. We can reuse it as is, if the next qudit
      // index is smaller than number of qudits of this allocated state.
      if (newIdx < nQuditsAllocated)
        return newIdx;
    }

    cudaq::info("Allocating new qudit with idx {} (nQ={}, dim={})", newIdx,
                nQuditsAllocated, stateDimension);

    // Increment the number of qudits and set
    // the new state dimension
    previousStateDimension = stateDimension;
    nQuditsAllocated++;
    stateDimension = calculateStateDim(nQuditsAllocated);

    if (!isInTracerMode())
      // Tell the subtype to grow the state representation
      addQuditToState();

    // May be that the state grows enough that we
    // want to handle observation via sampling
    if (executionContext)
      executionContext->canHandleObserve = canHandleObserve();

    // return the new qudit index
    return newIdx;
  }

  /// @brief Allocate `count` qudits in a specific state.
  std::vector<std::size_t>
  allocateQudits(std::size_t count, const void *state = nullptr,
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

    std::vector<std::size_t> qudits;
    for (std::size_t i = 0; i < count; i++)
      qudits.emplace_back(tracker.getNextIndex());

    if (isInBatchMode()) {
      // Store the current number of qudits requested
      batchModeCurrentNumQudits += count;

      // We have an allocated state, it has been set to |0>,
      // we want to reuse it as is. If the state needs to grow, then
      // we will ask the subtype to add more qudits.
      if (qudits.back() < nQuditsAllocated)
        count = 0;
      else
        count = qudits.back() + 1 - nQuditsAllocated;
    }

    cudaq::info("Allocating {} new qudits.", count);

    previousStateDimension = stateDimension;
    nQuditsAllocated += count;
    stateDimension = calculateStateDim(nQuditsAllocated);

    if (!isInTracerMode())
      // Tell the subtype to allocate more qudits
      addQuditsToState(count, state);

    // May be that the state grows enough that we
    // want to handle observation via sampling
    if (executionContext)
      executionContext->canHandleObserve = canHandleObserve();

    return qudits;
  }

  /// @brief Allocate `count` qudits in a specific state.
  std::vector<std::size_t> allocateQudits(std::size_t count,
                                          const PhotonicState *state) override {
    if (!state)
      return allocateQudits(count);

    if (!isInTracerMode() && count != state->getNumQudits())
      throw std::invalid_argument("Dimension mismatch: the input state doesn't "
                                  "match the number of qudits");

    std::vector<std::size_t> qudits;
    for (std::size_t i = 0; i < count; i++)
      qudits.emplace_back(tracker.getNextIndex());

    if (isInBatchMode()) {
      // Store the current number of qudits requested
      batchModeCurrentNumQudits += count;

      // We have an allocated state, it has been set to |0>,
      // we want to reuse it as is. If the state needs to grow, then
      // we will ask the subtype to add more qudits.
      if (qudits.back() < nQuditsAllocated)
        count = 0;
      else
        count = qudits.back() + 1 - nQuditsAllocated;
    }

    cudaq::info("Allocating {} new qudits.", count);

    previousStateDimension = stateDimension;
    nQuditsAllocated += count;
    stateDimension = calculateStateDim(nQuditsAllocated);

    if (!isInTracerMode())
      // Tell the subtype to allocate more qudits
      addQuditsToState(*state);

    // May be that the state grows enough that we
    // want to handle observation via sampling
    if (executionContext)
      executionContext->canHandleObserve = canHandleObserve();

    return qudits;
  }

  /// @brief Deallocate the qudit with give index
  void deallocate(const std::size_t quditIdx) override {
    if (executionContext && executionContext->name != "tracer") {
      cudaq::info("Deferring qudit {} deallocation", quditIdx);
      deferredDeallocation.push_back(quditIdx);
      return;
    }

    cudaq::info("Deallocating qudit {}", quditIdx);

    // Reset the qudit
    if (!isInTracerMode())
      resetQudit(quditIdx);

    // Return the index to the tracker
    tracker.returnIndex(quditIdx);
    --nQuditsAllocated;

    // Reset the state if we've deallocated all qudits.
    if (tracker.allDeallocated()) {
      cudaq::info("Deallocated all qudits, reseting state vector.");
      // all qudits deallocated,
      deallocateState();
      while (!gateQueue.empty())
        gateQueue.pop();
    }
  }

  /// @brief Deallocate all requested qudits. If the number of qudits
  /// is equal to the number of allocated qudits, then clear the entire
  /// state at once.
  void deallocateQudits(const std::vector<std::size_t> &qudits) override {
    // Do nothing if there are no allocated qudits.
    if (nQuditsAllocated == 0)
      return;

    if (executionContext) {
      for (auto &quditIdx : qudits) {
        cudaq::info("Deferring qudit {} deallocation", quditIdx);
        deferredDeallocation.push_back(quditIdx);
      }
      return;
    }

    if (qudits.size() == tracker.numAllocated()) {
      cudaq::info("Deallocate all qudits.");
      deallocateState();
      for (auto &q : qudits)
        tracker.returnIndex(q);
      return;
    }

    for (auto &q : qudits)
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
      if (sampleQudits.empty()) {
        if (isInBatchMode())
          sampleQudits.resize(batchModeCurrentNumQudits);
        else
          sampleQudits.resize(nQuditsAllocated);
        std::iota(sampleQudits.begin(), sampleQudits.end(), 0);
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
          // Not a vector, collate all bits into a 1 qudit counts dict
          for (std::size_t j = 0; j < bitResults.size(); j++) {
            counts.appendResult(bitResults[j], 1);
          }
        }
        executionContext->result.append(counts);
      }

      // Reorder the global register (if necessary). This might be necessary
      // if the mapping pass had run and we want to undo the shuffle that
      // occurred during mapping.
      if (!executionContext->reorderIdx.empty()) {
        executionContext->result.reorder(executionContext->reorderIdx);
        executionContext->reorderIdx.clear();
      }

      // Clear the sample bits for the next run
      sampleQudits.clear();
      midCircuitSampleResults.clear();
      lastMidCircuitRegisterName = "";
      currentCircuitName = "";
    }

    // Set the state data if requested.
    if (executionContext->name == "extract-state") {
      if (sampleQudits.empty()) {
        if (isInBatchMode())
          sampleQudits.resize(batchModeCurrentNumQudits);
        else
          sampleQudits.resize(nQuditsAllocated);
        std::iota(sampleQudits.begin(), sampleQudits.end(), 0);
      }
      flushGateQueue();
      executionContext->simulationState = getSimulationState();

      deallocateQudits(sampleQudits);
      sampleQudits.clear();
    }

    // Deallocate the deferred qudits, but do so
    // without explicit qudit reset.
    for (auto &deferred : deferredDeallocation)
      tracker.returnIndex(deferred);

    bool shouldSetToZero = isInBatchMode() && !isLastBatch();
    executionContext = nullptr;

    // Reset the state if we've deallocated all qudits.
    if (tracker.allDeallocated()) {
      if (shouldSetToZero) {
        cudaq::info("In batch mode currently, reset state to |0>");
        // Do not deallocate the state, but reset it to |0>
        setToZeroState();
      } else {
        cudaq::info("Deallocated all qudits, reseting state vector.");
        // all qudits deallocated,
        deallocateState();
      }
    }

    batchModeCurrentNumQudits = 0;
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
                            const std::vector<std::size_t> &targets,
                            const std::string_view customName) override {

    flushAnySamplingTasks();
    auto numRows = std::sqrt(matrix.size());
    auto numQudits = std::log2(numRows) / std::log2(levels);
    cudaq::info("Applying custom operation with matrix of size {}x{}", numRows,
                numRows);
    std::vector<std::complex<ScalarType>> actual;
    // if (numQudits > 1 && getQuditOrdering() != QuditOrdering::msb) {
    //   // Convert the matrix to LSB qudit ordering
    //   auto convertOrdering = [](std::size_t numQudits, std::size_t idx) {
    //     std::size_t newIdx = 0;
    //     // (std::log2(stateDimension) / std::log2(levels)) - quditIndex - 1;
    //     for (std::size_t i = 0; i < numQudits; ++i)
    //       if (idx & (1ULL << i))
    //         newIdx |= (1ULL << ((numQudits - 1) - i));
    //     return newIdx;
    //   };
    //   actual.resize(matrix.size());
    //   for (std::size_t i = 0; i < numRows; i++) {
    //     for (std::size_t j = 0; j < numRows; j++) {
    //       auto k = convertOrdering(numQudits, i);
    //       auto l = convertOrdering(numQudits, j);
    //       if (!std::is_same_v<double, ScalarType>) {
    //         actual[i * numRows + j] =
    //             static_cast<std::complex<ScalarType>>(matrix[k * numRows +
    //             l]);
    //       } else {
    //         auto element = matrix[k * numRows + l];
    //         actual[i * numRows + j] =
    //             std::complex<ScalarType>(element.real(), element.imag());
    //       }
    //     }
    //   }
    // } else {
    std::transform(matrix.begin(), matrix.end(), std::back_inserter(actual),
                   [](auto &&element) -> std::complex<ScalarType> {
                     if (!std::is_same_v<double, ScalarType>) {
                       return static_cast<std::complex<ScalarType>>(element);
                     } else {
                       return std::complex<ScalarType>(element.real(),
                                                       element.imag());
                     }
                   });
    // }
    if (cudaq::details::should_log(cudaq::details::LogLevel::info))
      cudaq::info(gateToString(customName.empty() ? "unknown op" : customName,
                               controls, {}, targets) +
                      " = {}",
                  matrix);
    enqueueGate(customName.empty() ? "unknown op" : customName.data(), actual,
                controls, targets, {});
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
    enqueueGate(gate.name(), gate.getGate(levels, angles), controls, targets,
                angles);
  }

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT(NAME)                             \
  using PhotonicCircuitSimulator::NAME;                                        \
  void NAME(const std::vector<std::size_t> &controls,                          \
            const std::size_t quditIdx) override {                             \
    enqueueQuantumOperation<nvqir::NAME<ScalarType>>(                          \
        {}, controls, std::vector<std::size_t>{quditIdx});                     \
  }

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM(NAME)                   \
  using PhotonicCircuitSimulator::NAME;                                        \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::size_t quditIdx) override {                             \
    enqueueQuantumOperation<nvqir::NAME<ScalarType>>(                          \
        {static_cast<ScalarType>(angle)}, controls,                            \
        std::vector<std::size_t>{quditIdx});                                   \
  }

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_TWO_PARAM(NAME)                   \
  using PhotonicCircuitSimulator::NAME;                                        \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::vector<std::size_t> quditsIdxs) override {              \
    enqueueQuantumOperation<nvqir::NAME<ScalarType>>(                          \
        {static_cast<ScalarType>(angle)}, controls, quditsIdxs);               \
  }

  /// @brief The plus gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT(plus)
  /// @brief The phase_shift gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM(phase_shift)
  /// @brief The beam_splitter gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_TWO_PARAM(beam_splitter)

// Undef those preprocessor defines.
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_TWO_PARAM

  std::size_t mz(const std::size_t quditIdx) { return mz(quditIdx, ""); }

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qudit, which we then use to do full state sampling when
  /// flushAnySamplingTask() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (`mz(q)`) and we measure the
  /// qudit, collapse the state, and then store the sample qudit for final
  /// full state sampling. We also return the bit result. If no execution
  /// context, just measure, collapse, and return the bit.
  std::size_t mz(const std::size_t quditIdx,
                 const std::string &registerName) override {
    // Flush the Gate Queue
    flushGateQueue();

    // If sampling, just store the bit, do nothing else.
    if (handleBasicSampling(quditIdx, registerName))
      return true;

    if (isInTracerMode())
      return true;

    // Get the actual measurement from the subtype measureQudit implementation
    auto measureResult = measureQudit(quditIdx);
    auto bitResult = std::to_string(measureResult);
    // If this CUDA-Q kernel has conditional statements on measure results
    // then we want to handle the sampling a bit differently.
    handleSamplingWithConditionals(quditIdx, bitResult, registerName);

    // Return the result
    return measureResult;
  }

}; // PhotonicCircuitSimulatorBase

} // namespace nvqir

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define NVQIR_REGISTER_PHOTONIC_SIMULATOR(CLASSNAME, PRINTED_NAME)             \
  extern "C" {                                                                 \
  nvqir::PhotonicCircuitSimulator *getPhotonicCircuitSimulator() {             \
    thread_local static std::unique_ptr<nvqir::PhotonicCircuitSimulator>       \
        photonic_simulator = std::make_unique<CLASSNAME>();                    \
    return photonic_simulator.get();                                           \
  }                                                                            \
  nvqir::PhotonicCircuitSimulator *CONCAT(getPhotonicCircuitSimulator_,        \
                                          PRINTED_NAME)() {                    \
    thread_local static std::unique_ptr<nvqir::PhotonicCircuitSimulator>       \
        photonic_simulator = std::make_unique<CLASSNAME>();                    \
    return photonic_simulator.get();                                           \
  }                                                                            \
  }

#define NVQIR_PHOTONIC_SIMULATOR_CLONE_IMPL(CLASSNAME)                         \
  nvqir::PhotonicCircuitSimulator *clone() override {                          \
    thread_local static std::unique_ptr<nvqir::PhotonicCircuitSimulator>       \
        photonic_simulator = std::make_unique<CLASSNAME>();                    \
    return photonic_simulator.get();                                           \
  }
