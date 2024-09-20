/*******************************************************************************
 * Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "PhotonicCircuitSimulator.h"

using namespace cudaq;

namespace nvqir {

////////////////////////////////////////////////////////////////////////////////
// PhotonicState
////////////////////////////////////////////////////////////////////////////////

PhotonicState::PhotonicState(qpp::ket &&data, std::size_t lvl)
    : state(std::move(data)), levels(lvl) {}
PhotonicState::PhotonicState(const std::vector<std::size_t> &shape,
                             const std::vector<std::complex<double>> &data,
                             std::size_t lvl) {
  if (shape.size() != 1)
    throw std::runtime_error(
        "PhotonicState must be created from data with 1D shape.");

  state = Eigen::Map<qpp::ket>(const_cast<std::complex<double> *>(data.data()),
                               shape[0]);

  levels = lvl;
}

std::size_t PhotonicState::getNumQudits() const {
  return (std::log2(state.size()) / std::log2(levels));
}

std::complex<double>
PhotonicState::getAmplitude(const std::vector<int> &basisState) {
  if (getNumQudits() != basisState.size())
    throw std::runtime_error(fmt::format(
        "[qpp-state] getAmplitude with an invalid number of bits in the "
        "basis state: expected {}, provided {}.",
        getNumQudits(), basisState.size()));
  // if (std::any_of(basisState.begin(), basisState.end(),
  //                 [](int x) { return x != 0 && x != 1; }))
  //   throw std::runtime_error(
  //       "[qpp-state] getAmplitude with an invalid basis state: only "
  //       "qudit state (0 or 1) is supported.");

  // Convert the basis state to an index value
  const std::size_t idx = std::accumulate(
      std::make_reverse_iterator(basisState.end()),
      std::make_reverse_iterator(basisState.begin()), 0ull,
      [&](std::size_t acc, int digit) { return (acc * levels) + digit; });
  return state[idx];
}

cudaq::SimulationState::Tensor
PhotonicState::getTensor(std::size_t tensorIdx) const {
  if (tensorIdx != 0)
    throw std::runtime_error("[qpp-state] invalid tensor requested.");
  return Tensor{
      reinterpret_cast<void *>(
          const_cast<std::complex<double> *>(state.data())),
      std::vector<std::size_t>{static_cast<std::size_t>(state.size())},
      getPrecision()};
}

std::complex<double>
PhotonicState::operator()(std::size_t tensorIdx,
                          const std::vector<std::size_t> &indices) {
  if (tensorIdx != 0)
    throw std::runtime_error("[qpp-state] invalid tensor requested.");
  if (indices.size() != 1)
    throw std::runtime_error("[qpp-state] invalid element extraction.");

  return state[indices[0]];
}

std::unique_ptr<PhotonicState>
PhotonicState::createPSFromSizeAndPtr(std::size_t size, void *ptr,
                                      std::size_t dataType) {
  return std::make_unique<PhotonicState>(
      Eigen::Map<qpp::ket>(reinterpret_cast<std::complex<double> *>(ptr), size),
      levels);
}

////////////////////////////////////////////////////////////////////////////////
// PhotonicCircuitSimulator
////////////////////////////////////////////////////////////////////////////////

std::size_t
PhotonicCircuitSimulator::calculateStateDim(const std::size_t numQudits) {
  assert(numQudits < 30);
  return std::pow(levels, numQudits);
}

void PhotonicCircuitSimulator::addQuditsToState(std::size_t quditCount,
                                                const void *stateDataIn) {
  if (quditCount == 0)
    return;

  auto *stateData =
      reinterpret_cast<std::complex<double> *>(const_cast<void *>(stateDataIn));

  if (state.size() == 0) {
    // If this is the first time, allocate the state
    if (stateData == nullptr) {
      state = qpp::ket::Zero(stateDimension);
      state(0) = 1.0;
    } else
      state = qpp::ket::Map(stateData, stateDimension);
    return;
  }
  // If we are resizing an existing, allocate
  // a zero state on a n qudit, and Kron-prod
  // that with the existing state.
  if (stateData == nullptr) {
    qpp::ket zero_state = qpp::ket::Zero(calculateStateDim(quditCount));
    zero_state(0) = 1.0;
    state = qpp::kron(zero_state, state);
  } else {
    qpp::ket initState =
        qpp::ket::Map(stateData, calculateStateDim(quditCount));
    state = qpp::kron(initState, state);
  }
  return;
}

void PhotonicCircuitSimulator::addQuditsToState(const PhotonicState &in_state) {
  const PhotonicState *const casted =
      dynamic_cast<const PhotonicState *>(&in_state);
  if (!casted)
    throw std::invalid_argument(
        "[PhotonicCircuitSimulator] Incompatible state input");

  if (state.size() == 0)
    state = casted->state;
  else
    state = qpp::kron(casted->state, state);
}

void PhotonicCircuitSimulator::deallocateStateImpl() {
  qpp::ket tmp;
  state = tmp;
}

void PhotonicCircuitSimulator::deallocateState() {
  deallocateStateImpl();
  nQuditsAllocated = 0;
  stateDimension = 0;
}

bool PhotonicCircuitSimulator::measureQudit(const std::size_t index) {
  const auto quditIdx = convertQuditIndex(index);
  // If here, then we care about the result bit, so compute it.
  const auto measurement_tuple =
      qpp::measure(state, qpp::cmat::Identity(levels, levels), {quditIdx},
                   /*qudit dimension=*/levels, /*destructive measmt=*/false);
  const auto measurement_result = std::get<qpp::RES>(measurement_tuple);
  const auto &post_meas_states = std::get<qpp::ST>(measurement_tuple);
  const auto &collapsed_state = post_meas_states[measurement_result];

  state = Eigen::Map<const qpp::ket>(collapsed_state.data(),
                                     collapsed_state.size());

  cudaq::info("Measured qudit {} -> {}", quditIdx, measurement_result);
  return measurement_result;
}

std::unique_ptr<PhotonicState> PhotonicCircuitSimulator::getSimulationState() {
  flushGateQueue();
  return std::make_unique<PhotonicState>(std::move(state), levels);
}

bool PhotonicCircuitSimulator::handleBasicSampling(const std::size_t quditIdx,
                                                   const std::string &regName) {
  if (executionContext && executionContext->name == "sample" &&
      !executionContext->hasConditionalsOnMeasureResults) {
    // Add the qudit to the sampling list
    sampleQudits.push_back(quditIdx);

    auto processForRegName = [&](const std::string &regStr) {
      // Insert the sample qudit into the register name map
      auto iter = registerNameToMeasuredQudit.find(regStr);
      if (iter == registerNameToMeasuredQudit.end())
        registerNameToMeasuredQudit.emplace(regStr,
                                            std::vector<std::size_t>{quditIdx});
      else if (std::find(iter->second.begin(), iter->second.end(), quditIdx) ==
               iter->second.end())
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

void PhotonicCircuitSimulator::handleSamplingWithConditionals(
    const std::size_t quditIdx, const std::string bitResult,
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

std::string PhotonicCircuitSimulator::gateToString(
    const std::string_view gateName, const std::vector<std::size_t> &controls,
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
    ret << "(apply) ctrl-" << gateName << "(" << angleStr << bits.str() << ")";
  } else {
    bits << targets[0];
    for (size_t i = 1; i < targets.size(); i++) {
      bits << ", " << targets[i];
    }
    ret << "(apply) " << gateName << "(" << angleStr << bits.str() << ")";
  }
  return ret.str();
}

bool PhotonicCircuitSimulator::isInBatchMode() {
  if (!executionContext)
    return false;

  if (executionContext->totalIterations == 0)
    return false;

  return true;
}

bool PhotonicCircuitSimulator::isLastBatch() {
  return executionContext && executionContext->batchIteration > 0 &&
         executionContext->batchIteration ==
             executionContext->totalIterations - 1;
}

void PhotonicCircuitSimulator::flushAnySamplingTasks(bool force) {
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

  sampleQudits.clear();
  registerNameToMeasuredQudit.clear();
}

void PhotonicCircuitSimulator::enqueueGate(
    const std::string name, const std::vector<std::complex<double>> &matrix,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const std::vector<double> &params) {
  if (isInTracerMode()) {
    std::vector<cudaq::QuditInfo> controlsInfo, targetsInfo;
    for (auto &c : controls)
      controlsInfo.emplace_back(2, c);
    for (auto &t : targets)
      targetsInfo.emplace_back(2, t);

    std::vector<double> anglesProcessed;
    if constexpr (std::is_same_v<double, double>)
      anglesProcessed = params;
    else {
      for (auto &a : params)
        anglesProcessed.push_back(static_cast<double>(a));
    }

    executionContext->kernelTrace.appendInstruction(name, anglesProcessed,
                                                    controlsInfo, targetsInfo);
    return;
  }

  gateQueue.emplace(name, matrix, controls, targets, params);
}

void PhotonicCircuitSimulator::applyGate(const GateApplicationTask &task) {
  auto matrix = toQppMatrix(task.matrix, task.targets.size());
  // First, convert all of the qudit indices to big endian.
  std::vector<std::size_t> controls;
  for (auto index : task.controls) {
    controls.push_back(convertQuditIndex(index));
  }
  std::vector<std::size_t> targets;
  for (auto index : task.targets) {
    targets.push_back(convertQuditIndex(index));
  }

  if (controls.empty()) {
    state = qpp::apply(state, matrix, targets);
    return;
  }
  state = qpp::applyCTRL(state, matrix, controls, targets);
}

void PhotonicCircuitSimulator::flushGateQueueImpl() {
  while (!gateQueue.empty()) {
    auto &next = gateQueue.front();
    if (isStateVectorSimulator() && summaryData.enabled)
      summaryData.svGateUpdate(next.controls.size(), next.targets.size(),
                               stateDimension,
                               stateDimension * sizeof(std::complex<double>));
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
    if (executionContext && executionContext->noiseModel) {
      std::vector<std::size_t> noiseQudits{next.controls.begin(),
                                           next.controls.end()};
      noiseQudits.insert(noiseQudits.end(), next.targets.begin(),
                         next.targets.end());
      applyNoiseChannel(next.operationName, noiseQudits);
    }
    gateQueue.pop();
  }
  // // For CUDA-based simulators, this calls cudaDeviceSynchronize()
  // synchronize();
}

void PhotonicCircuitSimulator::setToZeroState() {
  state = qpp::ket::Zero(stateDimension);
  state(0) = 1.0;
}

bool PhotonicCircuitSimulator::shouldObserveFromSampling(bool defaultConfig) {
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

bool PhotonicCircuitSimulator::isSinglePrecision() const { return false; }

std::size_t
PhotonicCircuitSimulator::convertQuditIndex(std::size_t quditIndex) {
  assert(stateDimension > 0 && "The state is empty, and thus has no qudits");
  // return (std::log2(stateDimension) / std::log2(levels)) - quditIndex - 1;
  return quditIndex;
}

qpp::cmat PhotonicCircuitSimulator::toQppMatrix(
    const std::vector<std::complex<double>> &data, std::size_t nTargets) {
  auto nRows = std::pow(levels, nTargets);
  assert(data.size() == nRows * nRows &&
         "Invalid number of gate matrix elements passed to toQppMatrix");

  // we represent row major, they represent column major
  return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                  Eigen::Dynamic, Eigen::RowMajor>>(
      const_cast<std::complex<double> *>(data.data()), nRows, nRows);
}

qpp::cmat PhotonicCircuitSimulator::toQppMatrix(
    const std::vector<std::complex<double>> &data, std::size_t nControls,
    std::size_t nTargets) {
  auto nRows = std::pow(levels, (nTargets + nControls));
  assert(data.size() == nRows * nRows &&
         "Invalid number of gate matrix elements passed to toQppMatrix");

  // we represent row major, they represent column major
  return Eigen::Map<Eigen::Matrix<std::complex<double>, Eigen::Dynamic,
                                  Eigen::Dynamic, Eigen::RowMajor>>(
      const_cast<std::complex<double> *>(data.data()), nRows, nRows);
}

PhotonicCircuitSimulator::PhotonicCircuitSimulator() {
  // Populate the correct name so it is printed correctly during
  // deconstructor.
  summaryData.name = name();
}

/// @brief Allocate a single qudit, return the qudit as a logical index
std::size_t PhotonicCircuitSimulator::allocateQudit() {
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
std::vector<std::size_t> PhotonicCircuitSimulator::allocateQudits(
    std::size_t count, const void *state,
    cudaq::simulation_precision precision) {

  // Make sure if someone gives us state data, that the precision
  // is correct for this simulation.
  if (state != nullptr) {
    if (precision == cudaq::simulation_precision::fp32)
      throw std::runtime_error("Invalid user-provided state data. Simulator "
                               "is FP64 but state data is FP32.");
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

  // Tell the subtype to allocate more qudits
  addQuditsToState(count, state);

  // May be that the state grows enough that we
  // want to handle observation via sampling
  if (executionContext)
    executionContext->canHandleObserve = canHandleObserve();

  return qudits;
}

/// @brief Allocate `count` qudits in a specific state.
std::vector<std::size_t>
PhotonicCircuitSimulator::allocateQudits(std::size_t count,
                                         const PhotonicState *state) {
  if (!state)
    return allocateQudits(count);

  if (count != state->getNumQudits())
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

  // Tell the subtype to allocate more qudits
  addQuditsToState(*state);

  // May be that the state grows enough that we
  // want to handle observation via sampling
  if (executionContext)
    executionContext->canHandleObserve = canHandleObserve();

  return qudits;
}
/// @brief Allocate `count` qudits.
std::vector<std::size_t>
PhotonicCircuitSimulator::allocateQudits(std::size_t count) {
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

  // Tell the subtype to allocate more qudits
  addQuditsToState(count);

  // May be that the state grows enough that we
  // want to handle observation via sampling
  if (executionContext)
    executionContext->canHandleObserve = canHandleObserve();

  return qudits;
}

/// @brief Deallocate the qudit with give index
void PhotonicCircuitSimulator::deallocate(const std::size_t quditIdx) {
  if (executionContext) {
    cudaq::info("Deferring qudit {} deallocation", quditIdx);
    deferredDeallocation.push_back(quditIdx);
    return;
  }

  cudaq::info("Deallocating qudit {}", quditIdx);

  // Reset the qudit
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
void PhotonicCircuitSimulator::deallocateQudits(
    const std::vector<std::size_t> &qudits) {
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
void PhotonicCircuitSimulator::resetExecutionContext() {
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

      if (std::find(vectorRegisters.begin(), vectorRegisters.end(), regName) !=
          vectorRegisters.end()) {
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
    flushGateQueue();
    executionContext->simulationState = getSimulationState();
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
void PhotonicCircuitSimulator::setExecutionContext(
    cudaq::ExecutionContext *context) {
  executionContext = context;
  executionContext->canHandleObserve = canHandleObserve();
  currentCircuitName = context->kernelName;
  cudaq::info("Setting current circuit name to {}", currentCircuitName);
}

void PhotonicCircuitSimulator::applyCustomOperation(
    const std::vector<std::complex<double>> &matrix,
    const std::vector<std::size_t> &controls,
    const std::vector<std::size_t> &targets,
    const std::string_view customName) {

  flushAnySamplingTasks();
  auto numRows = std::sqrt(matrix.size());
  auto numQudits = std::log2(numRows) / std::log2(levels);
  std::vector<std::complex<double>> actual;
  if (numQudits > 1 && getQuditOrdering() != QuditOrdering::msb) {
    // Convert the matrix to LSB qudit ordering
    auto convertOrdering = [](std::size_t numQudits, std::size_t idx) {
      std::size_t newIdx = 0;
      // (std::log2(stateDimension) / std::log2(levels)) - quditIndex - 1;
      for (std::size_t i = 0; i < numQudits; ++i)
        if (idx & (1ULL << i))
          newIdx |= (1ULL << ((numQudits - 1) - i));
      return newIdx;
    };
    actual.resize(matrix.size());
    for (std::size_t i = 0; i < numRows; i++) {
      for (std::size_t j = 0; j < numRows; j++) {
        auto k = convertOrdering(numQudits, i);
        auto l = convertOrdering(numQudits, j);
        if (!std::is_same_v<double, double>) {
          actual[i * numRows + j] =
              static_cast<std::complex<double>>(matrix[k * numRows + l]);
        } else {
          auto element = matrix[k * numRows + l];
          actual[i * numRows + j] =
              std::complex<double>(element.real(), element.imag());
        }
      }
    }
  } else {
    std::transform(matrix.begin(), matrix.end(), std::back_inserter(actual),
                   [](auto &&element) -> std::complex<double> {
                     if (!std::is_same_v<double, double>) {
                       return static_cast<std::complex<double>>(element);
                     } else {
                       return std::complex<double>(element.real(),
                                                   element.imag());
                     }
                   });
  }
  if (cudaq::details::should_log(cudaq::details::LogLevel::info))
    cudaq::info(gateToString(customName.empty() ? "unknown op" : customName,
                             controls, {}, targets) +
                    " = {}",
                matrix);
  enqueueGate(customName.empty() ? "unknown op" : customName.data(), actual,
              controls, targets, {});
}

template <typename QuantumOperation>
void PhotonicCircuitSimulator::enqueueQuantumOperation(
    const std::vector<double> &angles, const std::vector<std::size_t> &controls,
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

void PhotonicCircuitSimulator::resetQudit(const std::size_t index) {
  flushGateQueue();
  const auto quditIdx = convertQuditIndex(index);
  state = qpp::reset(state, {quditIdx});
}

cudaq::ExecutionResult
PhotonicCircuitSimulator::sample(const std::vector<std::size_t> &qudits,
                                 const int shots) {

  // if (shots < 1) {
  //   double expectationValue = calculateExpectationValue(qudits);
  //   cudaq::info("Computed expectation value = {}", expectationValue);
  //   return cudaq::ExecutionResult{{}, expectationValue};
  // }

  std::vector<std::size_t> measuredDigits;
  for (auto index : qudits) {
    measuredDigits.push_back(convertQuditIndex(index));
  }

  auto sampleResult = qpp::sample(shots, state, measuredDigits, 2);
  // Convert to what we expect
  std::stringstream bitstream;
  cudaq::ExecutionResult counts;

  // Expectation value from the counts
  double expVal = 0.0;
  for (auto [result, count] : sampleResult) {
    // Push back each term in the vector of bits to the bitstring.
    for (const auto &bit : result) {
      bitstream << bit;
    }

    // Add to the sample result
    // in mid-circ sampling mode this will append 1 bitstring
    auto bitstring = bitstream.str();
    counts.appendResult(bitstring, count);
    auto par = cudaq::sample_result::has_even_parity(bitstring);
    auto p = count / (double)shots;
    if (!par) {
      p = -p;
    }
    expVal += p;
    // Reset the state.
    bitstream.str("");
    bitstream.clear();
  }

  counts.expectationValue = expVal;
  return counts;
}

} // namespace nvqir
