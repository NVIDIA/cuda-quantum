/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/
#pragma once

#include "PhotonicGates.h"

#include "common/ExecutionContext.h"
#include "common/Logger.h"
#include "common/MeasureCounts.h"
#include "common/SimulationState.h"
#include "cudaq/host_config.h"
#include "nvqir/CircuitSimulator.h"

#include "qpp.h"

#include <iostream>
#include <set>
#include <span>

#define NVQIR_PHOTONIC_SIMULATOR_CLONE_IMPL(CLASSNAME)                         \
  nvqir::PhotonicCircuitSimulator *clone() {                                   \
    thread_local static std::unique_ptr<nvqir::PhotonicCircuitSimulator>       \
        simulator = std::make_unique<CLASSNAME>();                             \
    return simulator.get();                                                    \
  }

using namespace cudaq;

namespace nvqir {

enum class QuditOrdering { lsb, msb };

/// @brief PhotonicState provides an implementation of `SimulationState` that
/// encapsulates the state data for the Photonic Circuit Simulator.
struct PhotonicState : public cudaq::SimulationState {
  /// @brief The state. This class takes ownership move semantics.
  qpp::ket state;

  /// @brief The qudit-levels (`qumodes`)
  std::size_t levels;

  PhotonicState(qpp::ket &&data, std::size_t lvl);
  PhotonicState(const std::vector<std::size_t> &shape,
                const std::vector<std::complex<double>> &data, std::size_t lvl);

  std::size_t getNumQudits() const;

  std::size_t getNumQubits() const override {
    throw "not supported for this photonics simulator";
  }

  std::complex<double> overlap(const cudaq::SimulationState &other) override {
    throw "not supported for this photonics simulator";
  };

  std::complex<double>
  getAmplitude(const std::vector<int> &basisState) override;

  Tensor getTensor(std::size_t tensorIdx = 0) const override;

  // /// @brief Return all tensors that represent this state
  std::vector<Tensor> getTensors() const override { return {getTensor()}; }

  // /// @brief Return the number of tensors that represent this state.
  std::size_t getNumTensors() const override { return 1; }

  std::complex<double>
  operator()(std::size_t tensorIdx,
             const std::vector<std::size_t> &indices) override;

  std::unique_ptr<SimulationState>
  createFromSizeAndPtr(std::size_t size, void *ptr, std::size_t) override {
    throw "not supported for this photonics simulator";
  }

  std::unique_ptr<PhotonicState>
  createPSFromSizeAndPtr(std::size_t size, void *ptr, std::size_t dataType);

  void dump(std::ostream &os) const override { os << state << "\n"; }

  precision getPrecision() const override {
    return cudaq::SimulationState::precision::fp64;
  }

  void destroyState() override {
    qpp::ket k;
    state = k;
  }
};

/// @brief The PhotonicCircuitSimulator implements the class to provide a
/// simulator delegating to the Q++ library from
/// https://github.com/softwareqinc/qpp.
// class PhotonicCircuitSimulator : public CircuitSimulator {
class PhotonicCircuitSimulator {

private:
  /// @brief Reference to the current circuit name.
  std::string currentCircuitName = "";

  /// @brief Return true if the simulator is in the tracer mode.
  bool isInTracerMode() const {
    return executionContext && executionContext->name == "tracer";
  }

protected:
  /// The QPP state representation (qpp::ket)
  qpp::ket state;

  // The levels of the qudits
  std::size_t levels;

  // PhotonicState photonic_state(state, levels);

  /// @brief Statistics collected over the life of the simulator.
  SummaryData summaryData;

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
    const std::vector<std::complex<double>> matrix;
    const std::vector<std::size_t> controls;
    const std::vector<std::size_t> targets;
    const std::vector<double> parameters;
    GateApplicationTask(const std::string &name,
                        const std::vector<std::complex<double>> &m,
                        const std::vector<std::size_t> &c,
                        const std::vector<std::size_t> &t,
                        const std::vector<double> &params)
        : operationName(name), matrix(m), controls(c), targets(t),
          parameters(params) {}
  };

  /// @brief The current queue of operations to execute
  std::queue<GateApplicationTask> gateQueue;

  /// @brief Get the name of the current circuit being executed.
  std::string getCircuitName() const { return currentCircuitName; }

  /// @brief Return the current multi-qudit state dimension
  std::size_t calculateStateDim(const std::size_t numQudits);

  /// @brief Grow the state vector by one qudit.
  void addQuditToState() { addQuditsToState(1); }

  /// @brief Override the default sized allocation of qudits
  /// here to be a bit more efficient than the default implementation
  void addQuditsToState(std::size_t quditCount,
                        const void *stateDataIn = nullptr);

  void addQuditsToState(const PhotonicState &in_state);

  /// @brief Reset the qudit state.
  void deallocateStateImpl();

  /// @brief Reset the qudit state back to dim = 0.
  void deallocateState();

  /// @brief Measure the qudit and return the result. Collapse the
  /// state vector.
  bool measureQudit(const std::size_t index);

  /// @brief Return true if this CircuitSimulator can
  /// handle <psi | H | psi> instead of NVQIR applying measure
  /// basis quantum gates to change to the Z basis and sample.
  bool canHandleObserve() { return false; }

  /// @brief Return the internal state representation. This
  /// is meant for subtypes to override
  std::unique_ptr<PhotonicState> getSimulationState();

  /// @brief Handle basic sampling tasks by storing the qudit index for
  /// processing in resetExecutionContext. Return true to indicate this is
  /// sampling and to exit early. False otherwise.
  bool handleBasicSampling(const std::size_t quditIdx,
                           const std::string &regName);

  /// @brief This function handles sampling in the presence of conditional
  /// statements on qudit measurement results. Specifically, it will keep
  /// track of a classical register for all measures encountered in the
  /// program and store mid-circuit measures in the corresponding register.
  void handleSamplingWithConditionals(const std::size_t quditIdx,
                                      const std::string bitResult,
                                      const std::string &registerName);

  /// @brief Utility function that returns a string-view of the current

  /// quantum instruction, intended for logging purposes.
  std::string gateToString(const std::string_view gateName,
                           const std::vector<std::size_t> &controls,
                           const std::vector<double> &parameters,
                           const std::vector<std::size_t> &targets);

  /// @brief Return true if the current execution is in batch mode
  bool isInBatchMode();

  /// @brief Return true if the current execution is the
  /// last execution of batch mode.
  bool isLastBatch();

  /// @brief Execute a sampling task with the current set of sample qudits.
  void flushAnySamplingTasks(bool force = false);

  /// @brief Add a new gate application task to the queue
  void enqueueGate(const std::string name,
                   const std::vector<std::complex<double>> &matrix,
                   const std::vector<std::size_t> &controls,
                   const std::vector<std::size_t> &targets,
                   const std::vector<double> &params);

  /// @brief This pure method is meant for subtypes
  /// to implement, and its goal is to apply the gate described
  /// by the GateApplicationTask to the subtype-specific state
  /// data representation.
  void applyGate(const GateApplicationTask &task);

  /// @brief Provide a base-class method that can be invoked
  /// after every gate application and will apply any noise
  /// channels after the gate invocation based on a user-provided noise
  /// model. Unimplemented on the base class, sub-types can implement noise
  /// modeling.
  void applyNoiseChannel(const std::string_view gateName,
                         const std::vector<std::size_t> &qudits) {}

  /// @brief Flush the gate queue, run all queued gate
  /// application tasks.
  void flushGateQueueImpl();

  /// @brief Set the current state to the |0> state,
  /// retaining the current number of qudits.
  void setToZeroState();

  /// @brief Return true if expectation values should be computed from
  /// sampling + parity of bit strings.
  /// Default is to enable observe from sampling, i.e., simulating the
  /// change-of-basis circuit for each term.
  ///
  /// The environment variable "CUDAQ_OBSERVE_FROM_SAMPLING" can be used to
  /// turn on or off this setting.
  bool shouldObserveFromSampling(bool defaultConfig = true);

  bool isSinglePrecision() const;

  /// @brief Return this simulator's qudit ordering.
  QuditOrdering getQuditOrdering() const { return QuditOrdering::lsb; }

  ////////////////////////////////////////////////////////////////////////////////////
  /// @brief Convert internal qudit index to Q++ qudit index.
  ///
  /// In Q++, qudits are indexed from left to right, and thus q0 is the
  /// leftmost qudit. Internally, in CUDA-Q, qudits are index from right to
  /// left, hence q0 is the rightmost qudit. Example:
  /// ```
  ///   Q++ indices:  0  1  2  3
  ///                |0>|0>|0>|0>
  ///                 3  2  1  0 : CUDA-Q indices
  ///   Q++ indices:  0  1  2  3
  ///                |0>|0>|0>|0>
  ///                 0  1  2  3 : CUDA-Q photonics indices
  /// ```
  std::size_t convertQuditIndex(std::size_t quditIndex);

  qpp::cmat toQppMatrix(const std::vector<std::complex<double>> &data,
                        std::size_t nTargets);

  qpp::cmat toQppMatrix(const std::vector<std::complex<double>> &data,
                        std::size_t nControls, std::size_t nTargets);

public:
  PhotonicCircuitSimulator();
  virtual ~PhotonicCircuitSimulator() = default;

  /// @brief Flush the current queue of gates, i.e.
  /// apply them to the state.
  void flushGateQueue() { flushGateQueueImpl(); }

  void setRandomSeed(std::size_t seed) {
    qpp::RandomDevices::get_instance().get_prng().seed(seed);
  }

  cudaq::observe_result observe(const cudaq::spin_op &op) {
    throw "not supported for this photonics simulator";
  }

  /// @brief Allocate a single qudit, return the qudit as a logical index
  std::size_t allocateQudit();

  /// @brief Allocate `count` qudits in a specific state.
  std::vector<std::size_t>
  allocateQudits(std::size_t count, const void *state,
                 cudaq::simulation_precision precision =
                     cudaq::simulation_precision::fp64);

  /// @brief Allocate `count` qudits in a specific state.
  std::vector<std::size_t> allocateQudits(std::size_t count,
                                          const PhotonicState *state);

  /// @brief Allocate `count` qudits.
  std::vector<std::size_t> allocateQudits(std::size_t count);

  /// @brief Deallocate the qudit with give index
  void deallocate(const std::size_t quditIdx);

  /// @brief Deallocate all requested qudits. If the number of qudits
  /// is equal to the number of allocated qudits, then clear the entire
  /// state at once.
  void deallocateQudits(const std::vector<std::size_t> &qudits);

  /// @brief Reset the current execution context.
  void resetExecutionContext();

  /// @brief Set the execution context
  void setExecutionContext(cudaq::ExecutionContext *context);

  /// @brief Apply a custom quantum operation
  void applyCustomOperation(const std::vector<std::complex<double>> &matrix,
                            const std::vector<std::size_t> &controls,
                            const std::vector<std::size_t> &targets,
                            const std::string_view customName);

  template <typename QuantumOperation>
  void enqueueQuantumOperation(const std::vector<double> &angles,
                               const std::vector<std::size_t> &controls,
                               const std::vector<std::size_t> &targets);

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT(NAME)                             \
  void NAME(const std::vector<std::size_t> &controls,                          \
            const std::size_t quditIdx) {                                      \
    enqueueQuantumOperation<nvqir::NAME>({}, controls,                         \
                                         std::vector<std::size_t>{quditIdx});  \
  }

#define PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM(NAME)                   \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::size_t quditIdx) {                                      \
    enqueueQuantumOperation<nvqir::NAME>({static_cast<double>(angle)},         \
                                         controls,                             \
                                         std::vector<std::size_t>{quditIdx});  \
  }

#define PHOTONIC_CIRCUIT_SIMULATOR_TWO_QUDIT_ONE_PARAM(NAME)                   \
  void NAME(const double angle, const std::vector<std::size_t> &controls,      \
            const std::vector<std::size_t> quditsIdxs) {                       \
    enqueueQuantumOperation<nvqir::NAME>({static_cast<double>(angle)},         \
                                         controls, quditsIdxs);                \
  }

  /// @brief The plus gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT(plus)
  /// @brief The phase_shift gate
  PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM(phase_shift)
  /// @brief The beam_splitter gate
  PHOTONIC_CIRCUIT_SIMULATOR_TWO_QUDIT_ONE_PARAM(beam_splitter)

// Undef those preprocessor defines.
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT
#undef PHOTONIC_CIRCUIT_SIMULATOR_ONE_QUDIT_ONE_PARAM
#undef PHOTONIC_CIRCUIT_SIMULATOR_TWO_QUDIT_ONE_PARAM

  bool mz(const std::size_t quditIdx) { return mz(quditIdx, ""); }

  /// @brief Measure operation. Here we check what the current execution
  /// context is. If the context is sample, then we do nothing but store the
  /// measure qudit, which we then use to do full state sampling when
  /// flushAnySamplingTask() is called. If the context is sample-conditional,
  /// then we have a circuit that contains if (`mz(q)`) and we measure the
  /// qudit, collapse the state, and then store the sample qudit for final
  /// full state sampling. We also return the bit result. If no execution
  /// context, just measure, collapse, and return the bit.
  bool mz(const std::size_t quditIdx, const std::string &registerName) {
    // Flush the Gate Queue
    flushGateQueue();

    // If sampling, just store the bit, do nothing else.
    if (handleBasicSampling(quditIdx, registerName))
      return true;

    if (isInTracerMode())
      return true;

    // Get the actual measurement from the subtype measureQudit implementation
    auto measureResult = measureQudit(quditIdx);
    auto bitResult = measureResult == true ? "1" : "0";

    // If this CUDA-Q kernel has conditional statements on measure results
    // then we want to handle the sampling a bit differently.
    handleSamplingWithConditionals(quditIdx, bitResult, registerName);

    // Return the result
    return measureResult;
  }

  /// @brief Return the current execution context
  cudaq::ExecutionContext *getExecutionContext() { return executionContext; }

  bool isStateVectorSimulator() const { return true; }

  /// @brief Reset the qudit
  /// @param index 0-based index of qudit to reset
  void resetQudit(const std::size_t index);

  /// @brief Sample the multi-qudit state.
  cudaq::ExecutionResult sample(const std::vector<std::size_t> &qudits,
                                const int shots);

  /// @brief Primarily used for testing.
  auto getStateVector() {
    flushGateQueue();
    return state;
  }

  std::string name() const { return "photonics"; }

  NVQIR_PHOTONIC_SIMULATOR_CLONE_IMPL(PhotonicCircuitSimulator)

}; // PhotonicCircuitSimulator

} // namespace nvqir

// /// Register this Simulator class with NVQIR under name "tensornet"
// extern "C" {
// nvqir::PhotonicCircuitSimulator *getPhotonicCircuitSimulator_photonics() {
//   thread_local static auto simulator =
//       std::make_unique<nvqir::PhotonicCircuitSimulator>();
//   // Handle multiple runtime __nvqir__setCircuitSimulator calls before/after
//   // MPI initialization. If the static simulator instance was created before
//   // MPI initialization, it needs to be reset to support MPI if needed. if
//   // (cudaq::mpi::is_initialized() && !simulator->m_cutnMpiInitialized) {
//   //   // Reset the static instance to pick up MPI.
//   //   simulator.reset(new nvqir::PhotonicCircuitSimulator());
//   // }
//   return simulator.get();
// }
// nvqir::PhotonicCircuitSimulator *getPhotonicCircuitSimulator() {
//   return getPhotonicCircuitSimulator_photonics();
// }
// }

#define CONCAT(a, b) CONCAT_INNER(a, b)
#define CONCAT_INNER(a, b) a##b
#define NVQIR_REGISTER_PHOTONIC_SIMULATOR(CLASSNAME, PRINTED_NAME)             \
  extern "C" {                                                                 \
  nvqir::PhotonicCircuitSimulator *getPhotonicCircuitSimulator() {             \
    thread_local static std::unique_ptr<nvqir::PhotonicCircuitSimulator>       \
        simulator = std::make_unique<CLASSNAME>();                             \
    return simulator.get();                                                    \
  }                                                                            \
  nvqir::PhotonicCircuitSimulator *CONCAT(getPhotonicCircuitSimulator,         \
                                          PRINTED_NAME)() {                    \
    thread_local static std::unique_ptr<nvqir::PhotonicCircuitSimulator>       \
        simulator = std::make_unique<CLASSNAME>();                             \
    return simulator.get();                                                    \
  }                                                                            \
  }

// #ifndef __NVQIR_QPP_TOGGLE_CREATE
/// Register this Simulator with NVQIR.
NVQIR_REGISTER_PHOTONIC_SIMULATOR(nvqir::PhotonicCircuitSimulator, photonics)
// #endif
