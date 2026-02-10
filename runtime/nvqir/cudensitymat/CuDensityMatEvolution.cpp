/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BatchingUtils.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatExpectation.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "common/FmtCore.h"
#include "cudaq/algorithms/evolve_internal.h"
#include "cudaq/algorithms/integrator.h"
#include <iterator>
#include <random>
#include <stdexcept>

namespace cudaq::__internal__ {
template <typename Key, typename Value>
std::map<Key, Value>
convertToOrderedMap(const std::unordered_map<Key, Value> &unorderedMap) {
  return std::map<Key, Value>(unorderedMap.begin(), unorderedMap.end());
}

state migrateState(const state &inputState) {
  const auto currentDeviceId =
      dynamics::Context::getCurrentContext()->getDeviceId();
  cudaPointerAttributes attributes;
  HANDLE_CUDA_ERROR(
      cudaPointerGetAttributes(&attributes, inputState.get_tensor().data));
  const auto stateDeviceId = attributes.device;
  if (currentDeviceId == stateDeviceId)
    return inputState;

  CUDAQ_INFO("Migrate state data from device {} to {}\n", stateDeviceId,
             currentDeviceId);
  const int64_t dim = inputState.get_tensor().get_num_elements();
  const int64_t arraySizeBytes = dim * sizeof(std::complex<double>);
  auto localizedState =
      cudaq::dynamics::DeviceAllocator::allocate(arraySizeBytes);
  HANDLE_CUDA_ERROR(cudaMemcpy(localizedState, inputState.get_tensor().data,
                               arraySizeBytes, cudaMemcpyDefault));
  return state(new CuDensityMatState(dim, localizedState));
}

bool checkBatchingCompatibility(
    const std::vector<cudaq::matrix_handler> &elemOps) {
  if (elemOps.size() == 1)
    return true;

  const auto &firstOp = elemOps[0];
  for (std::size_t i = 1; i < elemOps.size(); ++i) {
    if (elemOps[i].degrees() != firstOp.degrees()) {
      return false;
    }
  }
  return true;
}

bool checkBatchingCompatibility(
    const std::vector<sum_op<cudaq::matrix_handler>> &ops) {

  if (ops.size() == 1) {
    return true;
  }

  // Check if all sum_ops has the same number of terms
  const std::size_t num_terms = ops.front().num_terms();
  for (std::size_t i = 1; i < ops.size(); ++i) {
    if (ops[i].num_terms() != num_terms) {
      return false;
    }
  }

  // Split the sum_op to list of product_op
  std::vector<std::vector<product_op<cudaq::matrix_handler>>> productOpsList(
      ops.size());
  for (auto &productOps : productOpsList) {
    productOps.reserve(num_terms);
  }
  for (std::size_t i = 0; i < ops.size(); ++i) {
    for (std::size_t j = 0; j < num_terms; ++j) {
      productOpsList[i].emplace_back(ops[i][j]);
    }
  }

  // Sort by degrees
  for (auto &productOps : productOpsList) {
    std::ranges::stable_sort(productOps.begin(), productOps.end(),
                             [](const product_op<cudaq::matrix_handler> &a,
                                const product_op<cudaq::matrix_handler> &b) {
                               return a.degrees() < b.degrees();
                             });
  }

  // Use the first product_op as a reference
  auto &reference = productOpsList[0];
  for (std::size_t i = 1; i < ops.size(); ++i) {
    auto &current = productOpsList[i];
    assert(current.size() == reference.size());
    for (std::size_t j = 0; j < reference.size(); ++j) {
      // Check if the degrees of the product_op match
      if (current[j].degrees() != reference[j].degrees()) {
        return false;
      }
      // Check if the number of elementary operators matches
      if (current[j].num_ops() != reference[j].num_ops()) {
        return false;
      }
      for (std::size_t k = 0; k < current[j].num_ops(); ++k) {
        // Check if the elementary degrees match
        if (current[j][k].degrees() != reference[j][k].degrees()) {
          return false;
        }
      }
    }
  }
  return true;
}

bool checkBatchingCompatibility(
    const std::vector<sum_op<cudaq::matrix_handler>> &hamOps,
    const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
        &listCollapseOps) {
  if (!checkBatchingCompatibility(hamOps)) {
    return false;
  }
  if (!listCollapseOps.empty()) {
    // All collapse_ops must have the same length for batching
    const std::size_t collapseOpLength = listCollapseOps.front().size();
    for (const auto &collapseOps : listCollapseOps) {
      if (collapseOps.size() != collapseOpLength) {
        return false;
      }
    }

    for (std::size_t i = 0; i < collapseOpLength; ++i) {
      // Check all the collapse ops in the batch
      std::vector<sum_op<cudaq::matrix_handler>> collapseOps;
      collapseOps.reserve(listCollapseOps.size());
      for (const auto &collapseOp : listCollapseOps) {
        collapseOps.emplace_back(collapseOp[i]);
      }
      if (!checkBatchingCompatibility(collapseOps)) {
        return false;
      }
    }
  }
  return true;
}

bool checkBatchingCompatibility(const std::vector<super_op> &listSuperOp) {
  if (listSuperOp.empty()) {
    return false;
  }

  const auto &firstSuperOp = listSuperOp[0];
  const auto numberOfTerms = firstSuperOp.num_terms();

  for (std::size_t i = 1; i < listSuperOp.size(); ++i) {
    const auto &toCheck = listSuperOp[i];
    if (toCheck.num_terms() != numberOfTerms) {
      return false;
    }

    for (std::size_t j = 0; j < numberOfTerms; ++j) {
      const auto &termToCheck = toCheck[j];
      const auto &firstTerm = firstSuperOp[j];
      if (firstTerm.first.has_value()) {
        if (!termToCheck.first.has_value()) {
          return false;
        }
        if (!checkBatchingCompatibility(
                {firstTerm.first.value(), termToCheck.first.value()})) {
          return false;
        }
      }
      if (firstTerm.second.has_value()) {
        if (!termToCheck.second.has_value()) {
          return false;
        }
        if (!checkBatchingCompatibility(
                {firstTerm.second.value(), termToCheck.second.value()})) {
          return false;
        }
      }
    }
  }

  return true;
}

static CuDensityMatState *asCudmState(cudaq::state &cudaqState) {
  auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
  auto *cudmState = dynamic_cast<CuDensityMatState *>(simState);
  if (!cudmState)
    throw std::runtime_error("Invalid state.");
  return cudmState;
}

static evolve_result
evolveSingleImpl(const std::vector<int64_t> &dims, const schedule &schedule,
                 base_integrator &integrator,
                 const std::vector<sum_op<cudaq::matrix_handler>> &observables,
                 IntermediateResultSave storeIntermediateResults) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::vector<CuDensityMatExpectation> expectations;
  auto &opConverter =
      cudaq::dynamics::Context::getCurrentContext()->getOpConverter();
  for (auto &obs : observables)
    expectations.emplace_back(CuDensityMatExpectation(
        handle, opConverter.convertToCudensitymatOperator({}, obs, dims)));

  std::vector<std::vector<double>> expectationVals;
  std::vector<cudaq::state> intermediateStates;
  for (const auto &step : schedule) {
    integrator.integrate(step.real());
    auto [t, currentState] = integrator.getState();
    if (storeIntermediateResults != cudaq::IntermediateResultSave::None) {
      std::vector<double> expVals;

      for (auto &expectation : expectations) {
        auto *cudmState = asCudmState(currentState);
        expectation.prepare(cudmState->get_impl());
        const auto expVal = expectation.compute(cudmState->get_impl(),
                                                step.real(), /*batchSize=*/1);
        assert(expVal.size() == 1);
        expVals.emplace_back(expVal.front().real());
      }
      expectationVals.emplace_back(std::move(expVals));
      if (storeIntermediateResults == cudaq::IntermediateResultSave::All)
        intermediateStates.emplace_back(currentState);
    }
  }

  if (cudaq::details::should_log(cudaq::details::LogLevel::trace))
    cudaq::dynamics::dumpPerfTrace();

  if (storeIntermediateResults == cudaq::IntermediateResultSave::All) {
    return evolve_result(intermediateStates, expectationVals);
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.getState();

    if (storeIntermediateResults ==
        cudaq::IntermediateResultSave::ExpectationValue)
      return evolve_result({finalState}, expectationVals);

    std::vector<double> expVals;
    auto *cudmState = asCudmState(finalState);
    for (auto &expectation : expectations) {
      expectation.prepare(cudmState->get_impl());
      const auto expVal = expectation.compute(cudmState->get_impl(), finalTime,
                                              /*batchSize=*/1);
      assert(expVal.size() == 1);
      expVals.emplace_back(expVal.front().real());
    }
    return evolve_result(finalState, expVals);
  }
}

/// @brief Evolve the system for a single time step.
/// @param hamiltonian Hamiltonian operator.
/// @param dimensionsMap Dimension of the system.
/// @param schedule Time schedule.
/// @param initialState Initial state.
/// @param inIntegrator Integrator.
/// @param collapseOperators Collapse operators.
/// @param observables Observables.
/// @param storeIntermediateResults Store intermediate results.
/// @param shotsCount Number of shots.
/// @return evolve_result Result of the evolution.
evolve_result evolveSingle(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensionsMap, const schedule &schedule,
    const state &initialState, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapseOperators,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables,
    IntermediateResultSave storeIntermediateResults,
    std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::map<std::size_t, int64_t> dimensions =
      convertToOrderedMap(dimensionsMap);
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);

  auto *cudmState = asCudmState(const_cast<state &>(initialState));
  if (!cudmState->is_initialized())
    cudmState->initialize_cudm(handle, dims, /*batchSize=*/1);

  state initial_State = [&]() {
    if (!collapseOperators.empty() && !cudmState->is_density_matrix())
      return state(new CuDensityMatState(cudmState->to_density_matrix()));
    return initialState;
  }();

  SystemDynamics system(dims, hamiltonian, collapseOperators);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);
  integrator.setState(initial_State, 0.0);
  return evolveSingleImpl(dims, schedule, integrator, observables,
                          storeIntermediateResults);
}

/// @brief Evolve the system for a single time step.
/// @param hamiltonian Hamiltonian operator.
/// @param dimensions Dimension of the system.
/// @param schedule Time schedule.
/// @param initial_state Initial state enum.
/// @param integrator Integrator.
/// @param collapse_operators Collapse operators.
/// @param observables Observables.
/// @param store_intermediate_results Store intermediate results.
/// @param shots_count Number of shots.
/// @return evolve_result Result of the evolution.
evolve_result evolveSingle(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensions, const schedule &schedule,
    InitialState initial_state, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapse_operators,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables,
    IntermediateResultSave store_intermediate_results,
    std::optional<int> shots_count) {
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  auto cudmState = CuDensityMatState::createInitialState(
      handle, initial_state, dimensions, collapse_operators.size() > 0);
  return evolveSingle(
      hamiltonian, dimensions, schedule, state(cudmState.release()), integrator,
      collapse_operators, observables, store_intermediate_results, shots_count);
}

static std::vector<evolve_result>
evolveBatchedImpl(const std::vector<int64_t> dims, const schedule &schedule,
                  std::size_t batchSize, base_integrator &integrator,
                  const std::vector<sum_op<cudaq::matrix_handler>> &observables,
                  IntermediateResultSave storeIntermediateResults) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();

  // We requires an even partition for distributed batched states.
  if (batchSize > 1 &&
      batchSize % dynamics::Context::getCurrentContext()->getNumRanks() != 0) {
    throw std::runtime_error(fmt::format(
        "Distributed batched states require an even partition across ranks: "
        "batch size {} is not divisible by number of ranks {}. Please adjust "
        "the number of MPI ranks or the batch size.",
        batchSize, dynamics::Context::getCurrentContext()->getNumRanks()));
  }

  std::vector<CuDensityMatExpectation> expectations;
  auto &opConverter =
      cudaq::dynamics::Context::getCurrentContext()->getOpConverter();
  for (auto &obs : observables) {
    auto cudmObsOp = opConverter.convertToCudensitymatOperator({}, obs, dims);
    expectations.emplace_back(CuDensityMatExpectation(handle, cudmObsOp));
  }

  // Helper to compute the state idx within the batch for distributed mode
  const auto getDistributedGlobalIdx = [](int localIdx, int batchSize) {
    const auto mpiNumRanks =
        dynamics::Context::getCurrentContext()->getNumRanks();
    const auto mpiRank = dynamics::Context::getCurrentContext()->getRank();
    const auto statesPerRank = batchSize / mpiNumRanks;
    return mpiRank * statesPerRank + localIdx;
  };

  std::vector<std::vector<std::vector<double>>> expectationVals(batchSize);
  std::vector<std::vector<cudaq::state>> intermediateStates(batchSize);
  for (const auto &step : schedule) {
    integrator.integrate(step.real());
    auto [t, currentState] = integrator.getState();
    if (storeIntermediateResults != cudaq::IntermediateResultSave::None) {
      auto *cudmState = asCudmState(currentState);
      std::vector<std::vector<double>> expVals(batchSize);
      for (auto &expectation : expectations) {
        expectation.prepare(cudmState->get_impl());
        const auto expVal =
            expectation.compute(cudmState->get_impl(), step.real(), batchSize);
        assert(expVal.size() == batchSize);
        for (int i = 0; i < expVal.size(); ++i) {
          expVals[i].emplace_back(expVal[i].real());
        }
      }

      if (storeIntermediateResults == cudaq::IntermediateResultSave::All) {
        auto states = CuDensityMatState::splitBatchedState(*cudmState);
        // In distributed mode, the split operation only returns the local
        // states held by this rank. The number of split states may be less than
        // batch_size.
        assert(states.size() <= batchSize);

        const auto numLocalStates = states.size();
        if (numLocalStates == batchSize) {
          // Non-distributed mode: all states are local
          for (int i = 0; i < batchSize; ++i) {
            intermediateStates[i].emplace_back(states[i]);
          }
        } else {
          for (int i = 0; i < numLocalStates; ++i) {
            const auto globalIdx = getDistributedGlobalIdx(i, batchSize);
            if (globalIdx < batchSize) {
              intermediateStates[globalIdx].emplace_back(states[i]);
            }
          }
        }
      }
      for (int i = 0; i < batchSize; ++i) {
        expectationVals[i].emplace_back(expVals[i]);
      }
    }
  }

  if (storeIntermediateResults == cudaq::IntermediateResultSave::All) {
    std::vector<evolve_result> results;
    // Note: In distributed mode, each rank only has local states.
    // Hence, return results only for the states this rank holds.
    for (int i = 0; i < batchSize; ++i) {
      // Skip if we don't have the data
      if (intermediateStates[i].empty())
        continue;
      results.emplace_back(
          evolve_result(intermediateStates[i], expectationVals[i]));
    }
    return results;
  } else {
    // Only final state is needed
    auto [finalTime, finalState] = integrator.getState();
    auto *cudmState = asCudmState(finalState);
    auto states = CuDensityMatState::splitBatchedState(*cudmState);
    // In distributed mode, the split operation only returns the local
    // states held by this rank. The number of split states may be less than
    // batch_size.
    assert(states.size() <= batchSize);

    // Helper to construct results based on distribution mode (distributed or
    // non-distributed)
    const auto constructResults =
        [batchSize, getDistributedGlobalIdx](
            const auto &states,
            const auto &expVals) -> std::vector<evolve_result> {
      const auto numLocalStates = states.size();
      std::vector<evolve_result> results;
      if (numLocalStates == batchSize) {
        // Non-distributed mode: all states are local
        for (int i = 0; i < batchSize; ++i) {
          results.emplace_back(
              evolve_result({cudaq::state(states[i])}, expVals[i]));
        }
      } else {
        // Distributed mode: each rank contains only a subset of batch data.
        for (int i = 0; i < numLocalStates; ++i) {
          const auto globalIdx = getDistributedGlobalIdx(i, batchSize);
          if (globalIdx < batchSize) {
            results.emplace_back(
                evolve_result({cudaq::state(states[i])}, expVals[globalIdx]));
          }
        }
      }

      return results;
    };

    if (storeIntermediateResults ==
        cudaq::IntermediateResultSave::ExpectationValue) {
      // Construct results with only final states and all expectation values
      // (including intermediate expectation values)
      return constructResults(states, expectationVals);
    }

    assert(storeIntermediateResults == cudaq::IntermediateResultSave::None);
    // Save option is None: only the final state and final expectation value (no
    // intermediate expectation values).

    // Compute final expectation values
    std::vector<std::vector<double>> expVals(batchSize);
    for (auto &expectation : expectations) {
      expectation.prepare(cudmState->get_impl());
      const auto expVal =
          expectation.compute(cudmState->get_impl(), finalTime, batchSize);
      assert(expVal.size() == batchSize);
      for (int i = 0; i < expVal.size(); ++i) {
        expVals[i].emplace_back(expVal[i].real());
      }
    }
    // Construct results with only final states and final expectation values.
    return constructResults(states, expVals);
  }
}

std::vector<evolve_result> evolveBatched(
    const sum_op<cudaq::matrix_handler> &hamiltonian,
    const cudaq::dimension_map &dimensionsMap, const schedule &schedule,
    const std::vector<state> &initialStates, base_integrator &integrator,
    const std::vector<sum_op<cudaq::matrix_handler>> &collapseOperators,
    const std::vector<sum_op<cudaq::matrix_handler>> &observables,
    IntermediateResultSave storeIntermediateResults,
    std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::map<std::size_t, int64_t> dimensions =
      convertToOrderedMap(dimensionsMap);
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);
  std::vector<CuDensityMatState *> states;
  for (auto &initialState : initialStates) {
    states.emplace_back(asCudmState(const_cast<state &>(initialState)));
  }
  auto batchedState = CuDensityMatState::createBatchedState(
      handle, states, dims, !collapseOperators.empty());
  SystemDynamics system(dims, hamiltonian, collapseOperators);
  cudaq::integrator_helper::init_system_dynamics(integrator, system, schedule);
  integrator.setState(cudaq::state(batchedState.release()), 0.0);
  return evolveBatchedImpl(dims, schedule, initialStates.size(), integrator,
                           observables, storeIntermediateResults);
}

evolve_result
evolveSingle(const super_op &superOp, const cudaq::dimension_map &dimensionsMap,
             const schedule &schedule, const state &initialState,
             base_integrator &integrator,
             const std::vector<sum_op<cudaq::matrix_handler>> &observables,
             IntermediateResultSave storeIntermediateResults,
             std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::map<std::size_t, int64_t> dimensions =
      convertToOrderedMap(dimensionsMap);
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);

  auto *cudmState = asCudmState(const_cast<state &>(initialState));
  if (!cudmState->is_initialized())
    cudmState->initialize_cudm(handle, dims, /*batchSize=*/1);

  cudaq::integrator_helper::init_system_dynamics(integrator, {superOp}, dims,
                                                 schedule);
  integrator.setState(initialState, 0.0);

  return evolveSingleImpl(dims, schedule, integrator, observables,
                          storeIntermediateResults);
}

evolve_result
evolveSingle(const super_op &superOp, const cudaq::dimension_map &dimensionsMap,
             const schedule &schedule, InitialState initial_state,
             base_integrator &integrator,
             const std::vector<sum_op<cudaq::matrix_handler>> &observables,
             IntermediateResultSave storeIntermediateResults,
             std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  const bool has_right_apply = [&]() {
    for (const auto &[leftOp, rightOp] : superOp) {
      if (rightOp.has_value())
        return true;
    }
    return false;
  }();
  auto cudmState = CuDensityMatState::createInitialState(
      handle, initial_state, dimensionsMap, has_right_apply);
  return evolveSingle(superOp, dimensionsMap, schedule,
                      state(cudmState.release()), integrator, observables,
                      storeIntermediateResults, shotsCount);
}

std::vector<evolve_result>
evolveBatched(const super_op &superOp,
              const cudaq::dimension_map &dimensionsMap,
              const schedule &schedule, const std::vector<state> &initialStates,
              base_integrator &integrator,
              const std::vector<sum_op<cudaq::matrix_handler>> &observables,
              IntermediateResultSave storeIntermediateResults,
              std::optional<int> shotsCount) {
  LOG_API_TIME();
  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::map<std::size_t, int64_t> dimensions =
      convertToOrderedMap(dimensionsMap);
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : dimensions)
    dims.emplace_back(dim);
  std::vector<CuDensityMatState *> states;
  for (auto &initialState : initialStates) {
    states.emplace_back(asCudmState(const_cast<state &>(initialState)));
  }
  const bool has_right_apply = [&]() {
    for (const auto &[leftOp, rightOp] : superOp) {
      if (rightOp.has_value())
        return true;
    }
    return false;
  }();
  auto batchedState = CuDensityMatState::createBatchedState(
      handle, states, dims, has_right_apply);
  cudaq::integrator_helper::init_system_dynamics(integrator, {superOp}, dims,
                                                 schedule);
  integrator.setState(cudaq::state(batchedState.release()), 0.0);
  return evolveBatchedImpl(dims, schedule, initialStates.size(), integrator,
                           observables, storeIntermediateResults);
}

std::vector<evolve_result>
evolveBatched(const std::vector<sum_op<cudaq::matrix_handler>> &hamiltonians,
              const cudaq::dimension_map &dimensions, const schedule &schedule,
              const std::vector<state> &initial_states,
              base_integrator &integrator,
              const std::vector<std::vector<sum_op<cudaq::matrix_handler>>>
                  &collapse_operators,
              const std::vector<sum_op<cudaq::matrix_handler>> &observables,
              IntermediateResultSave store_intermediate_results,
              std::optional<int> batch_size) {
  LOG_API_TIME();

  if (!collapse_operators.empty() &&
      hamiltonians.size() != collapse_operators.size()) {
    throw std::runtime_error("Number of Hamiltonian operators must match "
                             "number of collapse operators.");
  }

  if (initial_states.size() != hamiltonians.size()) {
    throw std::runtime_error(
        "Number of initial states must match number of Hamiltonian operators.");
  }

  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : convertToOrderedMap(dimensions))
    dims.emplace_back(dim);

  const bool canBeBatched =
      checkBatchingCompatibility(hamiltonians, collapse_operators);
  if (!canBeBatched) {
    // If the batch size was specified:
    if (batch_size.has_value() && batch_size.value() > 1) {
      throw std::runtime_error(
          "Hamiltonian operators and collapse operators are not compatible for "
          "batching. Unable to run batched simulation with the requested batch "
          "size.");
    }

    if (!batch_size.has_value()) {
      // Otherwise, just log a warning:
      CUDAQ_WARN("Hamiltonian operators and collapse operators are not "
                 "compatible for batching. "
                 "Falling back to single evolution for each Hamiltonian.");
    }
  }

  const auto batchSizeToRun =
      canBeBatched ? std::min<int>(batch_size.value_or(hamiltonians.size()),
                                   hamiltonians.size())
                   : 1;
  assert(batchSizeToRun <= hamiltonians.size());
  std::unordered_map<std::string, std::complex<double>> params;
  for (const auto &param : schedule.get_parameters()) {
    params[param] = schedule.get_value_function()(param, 0.0);
  }
  const bool isMasterEquation =
      !collapse_operators.empty() && !collapse_operators[0].empty();

  // Run batched evolution up to the batch size and concatenate the results.
  std::vector<evolve_result> allResults;
  allResults.reserve(hamiltonians.size());
  // Split the input states into batches up to batchSizeToRun
  for (std::size_t i = 0; i < hamiltonians.size(); i += batchSizeToRun) {
    std::vector<CuDensityMatState *> states;
    states.reserve(batchSizeToRun);
    std::vector<sum_op<cudaq::matrix_handler>> batchHamOps;
    batchHamOps.reserve(batchSizeToRun);
    std::vector<std::vector<sum_op<cudaq::matrix_handler>>> batchCollapseOps;
    batchCollapseOps.reserve(batchSizeToRun);

    for (std::size_t j = i; j < i + batchSizeToRun && j < hamiltonians.size();
         ++j) {
      states.emplace_back(asCudmState(const_cast<state &>(initial_states[j])));
      batchHamOps.emplace_back(hamiltonians[j]);
      if (!collapse_operators.empty()) {
        batchCollapseOps.emplace_back(collapse_operators[j]);
      }
    }
    const bool isDensityMat = states[0]->is_density_matrix();
    const bool sameStateType = std::all_of(
        states.begin(), states.end(), [&](CuDensityMatState *state) {
          return state->is_density_matrix() == isDensityMat;
        });

    if (!sameStateType) {
      throw std::invalid_argument(
          "All initial states must be of the same type (density matrix or "
          "state vector).");
    }
    // Evolve the batch of states
    SystemDynamics system(dims, batchHamOps, batchCollapseOps);
    cudaq::integrator_helper::init_system_dynamics(integrator, system,
                                                   schedule);

    if (states.size() > 1) {
      auto batchedState = CuDensityMatState::createBatchedState(
          handle, states, dims, isMasterEquation);
      integrator.setState(cudaq::state(batchedState.release()), 0.0);
      auto results =
          evolveBatchedImpl(dims, schedule, states.size(), integrator,
                            observables, store_intermediate_results);
      assert(results.size() == states.size());
      allResults.insert(allResults.end(),
                        std::make_move_iterator(results.begin()),
                        std::make_move_iterator(results.end()));
    } else {
      if (!states[0]->is_initialized())
        states[0]->initialize_cudm(handle, dims, /*batchSize=*/1);
      state canonicalize_initial_state = [&]() {
        if (isMasterEquation && !states[0]->is_density_matrix())
          return state(new CuDensityMatState(states[0]->to_density_matrix()));
        return initial_states[i];
      }();
      integrator.setState(canonicalize_initial_state, 0.0);
      auto result = evolveSingleImpl(dims, schedule, integrator, observables,
                                     store_intermediate_results);
      allResults.emplace_back(std::move(result));
    }
  }

  return allResults;
}

std::vector<evolve_result>
evolveBatched(const std::vector<super_op> &superOps,
              const cudaq::dimension_map &dimensions, const schedule &schedule,
              const std::vector<state> &initial_states,
              base_integrator &integrator,
              const std::vector<sum_op<cudaq::matrix_handler>> &observables,
              IntermediateResultSave store_intermediate_results,
              std::optional<int> batch_size) {
  LOG_API_TIME();
  if (superOps.empty()) {
    throw std::runtime_error("No super operators provided for evolution.");
  }
  if (initial_states.size() != superOps.size()) {
    throw std::runtime_error(
        "Number of initial states must match number of super operators.");
  }

  cudensitymatHandle_t handle =
      dynamics::Context::getCurrentContext()->getHandle();
  std::vector<int64_t> dims;
  for (const auto &[id, dim] : convertToOrderedMap(dimensions))
    dims.emplace_back(dim);

  const bool canBeBatched = checkBatchingCompatibility(superOps);
  if (!canBeBatched) {
    // If the batch size was specified:
    if (batch_size.has_value() && batch_size.value() > 1) {
      throw std::runtime_error(
          "The input super-operators are not compatible for "
          "batching. Unable to run batched simulation with the requested batch "
          "size.");
    }

    if (!batch_size.has_value()) {
      // Otherwise, just log a warning:
      CUDAQ_WARN("The input super-operators are not "
                 "compatible for batching. "
                 "Falling back to single evolution for each Hamiltonian.");
    }
  }

  const auto batchSizeToRun =
      canBeBatched
          ? std::min<int>(batch_size.value_or(superOps.size()), superOps.size())
          : 1;
  assert(batchSizeToRun <= superOps.size());
  std::unordered_map<std::string, std::complex<double>> params;
  for (const auto &param : schedule.get_parameters()) {
    params[param] = schedule.get_value_function()(param, 0.0);
  }

  const bool has_right_apply = [&]() {
    for (const auto &superOp : superOps) {
      for (const auto &[leftOp, rightOp] : superOp) {
        if (rightOp.has_value())
          return true;
      }
    }
    return false;
  }();
  // Run batched evolution up to the batch size and concatenate the results.
  std::vector<evolve_result> allResults;
  allResults.reserve(superOps.size());
  // Split the input states into batches up to batchSizeToRun
  for (std::size_t i = 0; i < superOps.size(); i += batchSizeToRun) {
    std::vector<CuDensityMatState *> states;
    states.reserve(batchSizeToRun);
    std::vector<super_op> batchSuperOps;
    batchSuperOps.reserve(batchSizeToRun);

    for (std::size_t j = i; j < i + batchSizeToRun && j < superOps.size();
         ++j) {
      states.emplace_back(asCudmState(const_cast<state &>(initial_states[j])));
      batchSuperOps.emplace_back(superOps[j]);
    }
    const bool isDensityMat = states[0]->is_density_matrix();
    const bool sameStateType = std::all_of(
        states.begin(), states.end(), [&](CuDensityMatState *state) {
          return state->is_density_matrix() == isDensityMat;
        });

    if (!sameStateType) {
      throw std::invalid_argument(
          "All initial states must be of the same type (density matrix or "
          "state vector).");
    }
    // Evolve the batch of states
    integrator_helper::init_system_dynamics(integrator, batchSuperOps, dims,
                                            schedule);

    if (states.size() > 1) {
      auto batchedState = CuDensityMatState::createBatchedState(
          handle, states, dims, has_right_apply);
      integrator.setState(cudaq::state(batchedState.release()), 0.0);
      auto results =
          evolveBatchedImpl(dims, schedule, states.size(), integrator,
                            observables, store_intermediate_results);
      assert(results.size() == states.size());
      allResults.insert(allResults.end(),
                        std::make_move_iterator(results.begin()),
                        std::make_move_iterator(results.end()));
    } else {
      if (!states[0]->is_initialized())
        states[0]->initialize_cudm(handle, dims, /*batchSize=*/1);

      state canonicalize_initial_state = [&]() {
        if (has_right_apply && !states[0]->is_density_matrix())
          return state(new CuDensityMatState(states[0]->to_density_matrix()));
        return initial_states[i];
      }();

      integrator.setState(canonicalize_initial_state, 0.0);
      auto result = evolveSingleImpl(dims, schedule, integrator, observables,
                                     store_intermediate_results);
      allResults.emplace_back(std::move(result));
    }
  }

  return allResults;
}
} // namespace cudaq::__internal__
