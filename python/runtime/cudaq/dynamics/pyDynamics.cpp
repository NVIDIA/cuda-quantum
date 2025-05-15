/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

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
  auto *castSimState = dynamic_cast<cudaq::CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");
  return castSimState;
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
                                       hamiltonian, collapse_ops, modeExtents,
                                       params, is_master_equation);
            return PyCuDensityMatTimeStepper(
                cudaq::dynamics::Context::getCurrentContext()->getHandle(),
                liouvillian, schedule);
          }))
      .def("compute", [](PyCuDensityMatTimeStepper &self, cudaq::state &inputState,
                         double t) {
        std::unordered_map<std::string, std::complex<double>> params;
        for (const auto &param : self.m_schedule.get_parameters()) {
          params[param] = self.m_schedule.get_value_function()(param, t);
        }
        return self.compute(inputState, t, params);
      });

  // System dynamics data class
  py::class_<cudaq::SystemDynamics>(m, "SystemDynamics")
      .def(py::init<>())
      .def_readwrite("modeExtents", &cudaq::SystemDynamics::modeExtents)
      .def_readwrite("hamiltonian", &cudaq::SystemDynamics::hamiltonian)
      .def_readwrite("collapseOps", &cudaq::SystemDynamics::collapseOps)
      .def_readwrite("parameters", &cudaq::SystemDynamics::parameters);

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
        assert(cudmState->is_initialized());
        return self.compute(cudmState->get_impl(), t).real();
      });

  // Schedule class
  py::class_<cudaq::schedule>(m, "Schedule")
      .def(py::init<const std::vector<double> &,
                    const std::vector<std::string> &>());

  // Helper to initialize a data buffer state
  m.def("initializeState",
        [](cudaq::state &state, const std::vector<int64_t> &modeExtents,
           bool asDensityMat) {
          auto &castSimState = *asCudmState(state);
          if (!castSimState.is_initialized())
            castSimState.initialize_cudm(
                cudaq::dynamics::Context::getCurrentContext()->getHandle(),
                modeExtents);
          if (asDensityMat && !castSimState.is_density_matrix()) {
            return cudaq::state(
                new cudaq::CuDensityMatState(castSimState.to_density_matrix()));
          }
          return state;
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
