/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "BatchingUtils.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatExpectation.h"
#include "CuDensityMatState.h"
#include "CuDensityMatTimeStepper.h"
#include "CuDensityMatUtils.h"
#include "cudaq/algorithms/base_integrator.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/schedule.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
namespace {
cudaq::CuDensityMatState *asCudmState(cudaq::state &cudaqState) {
  auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
  auto *cudmState = dynamic_cast<cudaq::CuDensityMatState *>(simState);
  if (!cudmState)
    throw std::runtime_error("Invalid state.");
  return cudmState;
}
} // namespace

// Internal dynamics bindings
PYBIND11_MODULE(nvqir_dynamics_bindings, m) {
  class PyCuDensityMatTimeStepper : public cudaq::CuDensityMatTimeStepper {
  public:
    PyCuDensityMatTimeStepper(cudensitymatHandle_t handle,
                              cudensitymatOperator_t liouvillian,
                              cudaq::schedule schedule)
        : cudaq::CuDensityMatTimeStepper(handle, liouvillian),
          m_schedule(schedule) {}
    cudaq::schedule m_schedule;
  };

  // Time stepper bindings
  py::class_<PyCuDensityMatTimeStepper>(m, "TimeStepper")
      .def(py::init(
          [](cudaq::schedule schedule, std::vector<int64_t> modeExtents,
             cudaq::sum_op<cudaq::matrix_handler> hamiltonian,
             std::vector<cudaq::sum_op<cudaq::matrix_handler>> collapse_ops,
             bool is_master_equation) {
            std::unordered_map<std::string, std::complex<double>> params;
            for (const auto &param : schedule.get_parameters()) {
              params[param] = schedule.get_value_function()(param, 0.0);
            }
            auto liouvillian = cudaq::dynamics::Context::getCurrentContext()
                                   ->getOpConverter()
                                   .constructLiouvillian(
                                       {hamiltonian}, {collapse_ops},
                                       modeExtents, params, is_master_equation);
            return PyCuDensityMatTimeStepper(
                cudaq::dynamics::Context::getCurrentContext()->getHandle(),
                liouvillian, schedule);
          }))
      .def(py::init([](cudaq::schedule schedule,
                       std::vector<int64_t> modeExtents,
                       cudaq::super_op superOp) {
        std::unordered_map<std::string, std::complex<double>> params;
        for (const auto &param : schedule.get_parameters()) {
          params[param] = schedule.get_value_function()(param, 0.0);
        }
        auto liouvillian =
            cudaq::dynamics::Context::getCurrentContext()
                ->getOpConverter()
                .constructLiouvillian({superOp}, modeExtents, params);
        return PyCuDensityMatTimeStepper(
            cudaq::dynamics::Context::getCurrentContext()->getHandle(),
            liouvillian, schedule);
      }))
      .def(py::init([](cudaq::schedule schedule,
                       std::vector<int64_t> modeExtents,
                       const std::vector<cudaq::sum_op<cudaq::matrix_handler>>
                           &hamiltonians,
                       const std::vector<
                           std::vector<cudaq::sum_op<cudaq::matrix_handler>>>
                           &list_collapse_ops,
                       bool is_master_equation) {
        std::unordered_map<std::string, std::complex<double>> params;
        for (const auto &param : schedule.get_parameters()) {
          params[param] = schedule.get_value_function()(param, 0.0);
        }
        auto liouvillian =
            cudaq::dynamics::Context::getCurrentContext()
                ->getOpConverter()
                .constructLiouvillian(hamiltonians, list_collapse_ops,
                                      modeExtents, params, is_master_equation);
        return PyCuDensityMatTimeStepper(
            cudaq::dynamics::Context::getCurrentContext()->getHandle(),
            liouvillian, schedule);
      }))
      .def(py::init([](cudaq::schedule schedule,
                       std::vector<int64_t> modeExtents,
                       const std::vector<cudaq::super_op> &superOps) {
        std::unordered_map<std::string, std::complex<double>> params;
        for (const auto &param : schedule.get_parameters()) {
          params[param] = schedule.get_value_function()(param, 0.0);
        }
        auto liouvillian =
            cudaq::dynamics::Context::getCurrentContext()
                ->getOpConverter()
                .constructLiouvillian(superOps, modeExtents, params);
        return PyCuDensityMatTimeStepper(
            cudaq::dynamics::Context::getCurrentContext()->getHandle(),
            liouvillian, schedule);
      }))
      .def("compute",
           [](PyCuDensityMatTimeStepper &self, cudaq::state &inputState,
              double t) {
             std::unordered_map<std::string, std::complex<double>> params;
             for (const auto &param : self.m_schedule.get_parameters()) {
               params[param] = self.m_schedule.get_value_function()(param, t);
             }
             return self.compute(inputState, t, params);
           })
      .def("compute",
           [](PyCuDensityMatTimeStepper &self, cudaq::state &inputState,
              double t, cudaq::state &outputState) {
             // Compute into the provided output state
             std::unordered_map<std::string, std::complex<double>> params;
             for (const auto &param : self.m_schedule.get_parameters()) {
               params[param] = self.m_schedule.get_value_function()(param, t);
             }

             auto *inputSimState =
                 cudaq::state_helper::getSimulationState(&inputState);
             auto *castInputSimState =
                 dynamic_cast<cudaq::CuDensityMatState *>(inputSimState);

             auto *outputSimState =
                 cudaq::state_helper::getSimulationState(&outputState);
             auto *castOutputSimState =
                 dynamic_cast<cudaq::CuDensityMatState *>(outputSimState);

             if (!castInputSimState || !castOutputSimState)
               throw std::runtime_error("Invalid input or output state.");

             assert(castInputSimState->getBatchSize() ==
                    castOutputSimState->getBatchSize());
             self.computeImpl(castInputSimState->get_impl(),
                              castOutputSimState->get_impl(), t, params,
                              castInputSimState->getBatchSize());
           });

  // System dynamics data class
  py::class_<cudaq::SystemDynamics>(m, "SystemDynamics")
      .def(py::init<>())
      .def_readwrite("modeExtents", &cudaq::SystemDynamics::modeExtents)
      .def_readwrite("hamiltonian", &cudaq::SystemDynamics::hamiltonian)
      .def_readwrite("collapseOps", &cudaq::SystemDynamics::collapseOps)
      .def_readwrite("parameters", &cudaq::SystemDynamics::parameters)
      .def_readwrite("superOp", &cudaq::SystemDynamics::superOp);

  // Expectation calculation
  py::class_<cudaq::CuDensityMatExpectation>(m, "CuDensityMatExpectation")
      .def(py::init([](cudaq::sum_op<cudaq::matrix_handler> &obs,
                       const std::vector<int64_t> &modeExtents) {
        return cudaq::CuDensityMatExpectation(
            cudaq::dynamics::Context::getCurrentContext()->getHandle(),
            cudaq::dynamics::Context::getCurrentContext()
                ->getOpConverter()
                .convertToCudensitymatOperator({}, obs, modeExtents));
      }))
      .def("prepare",
           [](cudaq::CuDensityMatExpectation &self, cudaq::state &state) {
             auto *cudmState = asCudmState(state);
             assert(cudmState->is_initialized());
             self.prepare(cudmState->get_impl());
           })
      .def("compute", [](cudaq::CuDensityMatExpectation &self,
                         cudaq::state &state, double t) {
        auto *cudmState = asCudmState(state);
        std::vector<double> expVals;
        const auto results =
            self.compute(cudmState->get_impl(), t, cudmState->getBatchSize());
        for (const auto &result : results)
          expVals.emplace_back(result.real());
        return expVals;
      });

  // Schedule class
  py::class_<cudaq::schedule>(m, "Schedule")
      .def(py::init<const std::vector<double> &,
                    const std::vector<std::string> &>());

  // Helper to initialize a data buffer state
  m.def("initializeState",
        [](cudaq::state &state, const std::vector<int64_t> &modeExtents,
           bool asDensityMat, int64_t batchSize) {
          auto &castSimState = *asCudmState(state);
          if (!castSimState.is_initialized())
            castSimState.initialize_cudm(
                cudaq::dynamics::Context::getCurrentContext()->getHandle(),
                modeExtents, batchSize);
          if (asDensityMat && !castSimState.is_density_matrix()) {
            return cudaq::state(
                new cudaq::CuDensityMatState(castSimState.to_density_matrix()));
          }
          return state;
        });
  // Helper to initialize a data buffer state from an unowned device pointer.
  // We wrap the device pointer as a `CuDensityMatState` without copying data.
  m.def("initializeState",
        [](int64_t deviceDataPtr, std::size_t size,
           const std::vector<int64_t> &modeExtents, int64_t batchSize) {
          auto *cudmState = new cudaq::CuDensityMatState(
              size, reinterpret_cast<void *>(deviceDataPtr), /*borrowed=*/true);
          cudmState->initialize_cudm(
              cudaq::dynamics::Context::getCurrentContext()->getHandle(),
              modeExtents, batchSize);
          return cudaq::state(cudmState);
        });

  // Helper to create an initial state
  m.def("createInitialState",
        [](cudaq::InitialState initialStateType,
           const std::unordered_map<std::size_t, std::int64_t> &dimensions,
           bool createDensityMatrix) {
          auto state = cudaq::CuDensityMatState::createInitialState(
              cudaq::dynamics::Context::getCurrentContext()->getHandle(),
              initialStateType, dimensions, createDensityMatrix);
          assert(state->is_initialized());
          return cudaq::state(state.release());
        });
  m.def("createBatchedState",
        [](std::vector<cudaq::state> &states,
           const std::vector<int64_t> &modeExtents, bool asDensityMat) {
          std::vector<cudaq::CuDensityMatState *> statePtrs;
          for (auto &state : states) {
            statePtrs.emplace_back(asCudmState(state));
          }
          auto batchedState = cudaq::CuDensityMatState::createBatchedState(
              cudaq::dynamics::Context::getCurrentContext()->getHandle(),
              statePtrs, modeExtents, asDensityMat);
          return cudaq::state(batchedState.release());
        });

  m.def("getBatchSize", [](cudaq::state &state) {
    auto &cudmSimState = *asCudmState(state);
    if (!cudmSimState.is_initialized())
      throw std::runtime_error(
          "Cannot query batch size of an uninitialized state");
    return cudmSimState.getBatchSize();
  });
  m.def("splitBatchedState",
        [](cudaq::state &state) -> std::vector<cudaq::state> {
          auto &castSimState = *asCudmState(state);
          if (!castSimState.is_initialized())
            throw std::runtime_error("Cannot split of an uninitialized state");
          if (castSimState.getBatchSize() == 1)
            return {state};
          std::vector<cudaq::state> splitStates;
          auto states =
              cudaq::CuDensityMatState::splitBatchedState(castSimState);

          for (auto *state : states) {
            splitStates.emplace_back(cudaq::state(state));
          }
          return splitStates;
        });

  // Helper to clear context (performance/callback) after an evolve simulation.
  // Note: the callback contexts contain Python functions, hence must be cleared
  // before the actual shutdown.
  m.def("clearContext", []() {
    cudaq::dynamics::Context::getCurrentContext()
        ->getOpConverter()
        .clearCallbackContext();
    if (cudaq::details::should_log(cudaq::details::LogLevel::trace))
      cudaq::dynamics::dumpPerfTrace();
  });

  // Helper to check if operators can be batched.
  m.def(
      "checkBatchingCompatibility",
      [](const std::vector<cudaq::sum_op<cudaq::matrix_handler>> &hamOps,
         const std::vector<std::vector<cudaq::sum_op<cudaq::matrix_handler>>>
             &listCollapseOps) {
        return cudaq::__internal__::checkBatchingCompatibility(hamOps,
                                                               listCollapseOps);
      },
      py::arg("hamiltonians"), py::arg("collapse_operators"));

  m.def(
      "checkSuperOpBatchingCompatibility",
      [](const std::vector<cudaq::super_op> &super_operators) {
        return cudaq::__internal__::checkBatchingCompatibility(super_operators);
      },
      py::arg("super_operators"));

  auto integratorsSubmodule = m.def_submodule("integrators");

  // Runge-Kutta integrator
  py::class_<cudaq::integrators::runge_kutta>(integratorsSubmodule,
                                              "runge_kutta")
      .def(py::init<int, std::optional<double>>(), py::kw_only(),
           py::arg("order") = cudaq::integrators::runge_kutta::default_order,
           py::arg("max_step_size") = py::none())
      .def("setState",
           [](cudaq::integrators::runge_kutta &self, cudaq::state &state,
              double t) { self.setState(state, t); })
      .def("setSystem",
           [](cudaq::integrators::runge_kutta &self,
              cudaq::SystemDynamics system, cudaq::schedule schedule) {
             cudaq::integrator_helper::init_system_dynamics(self, system,
                                                            schedule);
           })
      .def("integrate", &cudaq::integrators::runge_kutta::integrate)
      .def("getState", [](cudaq::integrators::runge_kutta &self) {
        return self.getState();
      });
}
