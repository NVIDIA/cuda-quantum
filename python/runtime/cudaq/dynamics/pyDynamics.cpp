/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatTimeStepper.h"
#include "cudaq/algorithms/base_integrator.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/schedule.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "CuDensityMatState.h"
#include "CuDensityMatContext.h"
#include "CuDensityMatExpectation.h"

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
      .def("compute", [](PyCuDensityMatTimeStepper &self, int64_t inputStatePtr,
                         int64_t outputStatePtr, double t) {
        cudensitymatState_t inputState =
            reinterpret_cast<cudensitymatState_t>(inputStatePtr);
        cudensitymatState_t outputState =
            reinterpret_cast<cudensitymatState_t>(outputStatePtr);
        std::unordered_map<std::string, std::complex<double>> params;
        for (const auto &param : self.m_schedule.get_parameters()) {
          params[param] = self.m_schedule.get_value_function()(param, t);
        }
        self.computeImpl(inputState, outputState, t, params);
      });

  py::class_<cudaq::SystemDynamics>(m, "SystemDynamics")
      .def(py::init<>())
      .def_readwrite("modeExtents", &cudaq::SystemDynamics::modeExtents)
      .def_readwrite("hamiltonian", &cudaq::SystemDynamics::hamiltonian)
      .def_readwrite("collapseOps", &cudaq::SystemDynamics::collapseOps)
      .def_readwrite("parameters", &cudaq::SystemDynamics::parameters);

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
             self.prepare(cudmState->get_impl());
           })
      .def("compute",
           [](cudaq::CuDensityMatExpectation &self, cudaq::state &state,
              double t) {
             auto *cudmState = asCudmState(state);
             return self.compute(cudmState->get_impl(), t).real();
           })
      .def("prepare",
           [](cudaq::CuDensityMatExpectation &self, int64_t statePtr) {
             self.prepare(reinterpret_cast<cudensitymatState_t>(statePtr));
           })
      .def("compute", [](cudaq::CuDensityMatExpectation &self, int64_t statePtr,
                         double t) {
        return self.compute(reinterpret_cast<cudensitymatState_t>(statePtr), t)
            .real();
      });

  py::class_<cudaq::schedule>(m, "Schedule")
      .def(py::init<const std::vector<double> &,
                    const std::vector<std::string> &>());

  m.def("initializeState", [](cudaq::state &state,
                              const std::vector<int64_t> &modeExtents,
                              bool asDensityMat) {
    auto &castSimState = *asCudmState(state);
    if (!castSimState.is_initialized())
      castSimState.initialize_cudm(
          cudaq::dynamics::Context::getCurrentContext()->getHandle(),
          modeExtents);

    if (asDensityMat && !castSimState.is_density_matrix())
      return cudaq::state(castSimState.make_density_matrix());
    return state;
  });

  auto integratorsSubmodule = m.def_submodule("integrators");

  py::class_<cudaq::integrators::runge_kutta>(integratorsSubmodule,
                                              "runge_kutta")
      .def(py::init<>())
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
