/*******************************************************************************
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatState.h"
#include "cudaq/dynamics_integrators.h"
#include "cudm_error_handling.h"
#include "cudm_helpers.h"
#include "cudm_time_stepper.h"
namespace {
using namespace cudaq;
class cudmStepper : public TimeStepper {
public:
  explicit cudmStepper(cudensitymatHandle_t handle,
                       cudensitymatOperator_t liouvillian)
      : m_handle(handle), m_liouvillian(liouvillian){};

  state compute(const state &inputState, double t, double step_size,
                const std::unordered_map<std::string, std::complex<double>>
                    &parameters) override {
    if (step_size == 0.0) {
      throw std::runtime_error("Step size cannot be zero.");
    }

    auto *simState = cudaq::state_helper::getSimulationState(
        const_cast<state *>(&inputState));
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    CuDensityMatState &state = *castSimState;
    // Prepare workspace
    cudensitymatWorkspaceDescriptor_t workspace;
    HANDLE_CUDM_ERROR(cudensitymatCreateWorkspace(m_handle, &workspace));

    // Query free gpu memory and allocate workspace buffer
    std::size_t freeMem = 0, totalMem = 0;
    HANDLE_CUDA_ERROR(cudaMemGetInfo(&freeMem, &totalMem));
    // Take 80% of free memory
    freeMem = static_cast<std::size_t>(static_cast<double>(freeMem) * 0.80);

    // Create a new state for the next step
    auto next_state = CuDensityMatState::zero_like(state);

    if (!next_state.is_initialized()) {
      throw std::runtime_error("Next state failed to initialize.");
    }

    if (state.get_hilbert_space_dims() != next_state.get_hilbert_space_dims()) {
      throw std::runtime_error("As the dimensions of both the old and the new "
                               "state do no match, the "
                               "operator cannot act on the states.");
    }

    // Prepare the operator for action
    HANDLE_CUDM_ERROR(cudensitymatOperatorPrepareAction(
        m_handle, m_liouvillian, state.get_impl(), next_state.get_impl(),
        CUDENSITYMAT_COMPUTE_64F, freeMem, workspace, 0x0));

    // Query required workspace buffer size
    std::size_t requiredBufferSize = 0;
    HANDLE_CUDM_ERROR(cudensitymatWorkspaceGetMemorySize(
        m_handle, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
        CUDENSITYMAT_WORKSPACE_SCRATCH, &requiredBufferSize));

    void *workspaceBuffer = nullptr;
    if (requiredBufferSize > 0) {
      // Allocate GPU storage for workspace buffer
      const std::size_t bufferVolume =
          requiredBufferSize / sizeof(std::complex<double>);
      workspaceBuffer = cudm_helper::create_array_gpu(
          std::vector<std::complex<double>>(bufferVolume, {0.0, 0.0}));

      // Attach workspace buffer
      HANDLE_CUDM_ERROR(cudensitymatWorkspaceSetMemory(
          m_handle, workspace, CUDENSITYMAT_MEMSPACE_DEVICE,
          CUDENSITYMAT_WORKSPACE_SCRATCH, workspaceBuffer, requiredBufferSize));
    }

    // Apply the operator action
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    HANDLE_CUDM_ERROR(cudensitymatOperatorComputeAction(
        m_handle, m_liouvillian, t, 0, nullptr, state.get_impl(),
        next_state.get_impl(), workspace, 0x0));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());

    // Cleanup
    cudm_helper::destroy_array_gpu(workspaceBuffer);
    HANDLE_CUDM_ERROR(cudensitymatDestroyWorkspace(workspace));

    return cudaq::state(
        std::make_unique<CuDensityMatState>(std::move(next_state)).release());
  }

private:
  cudensitymatHandle_t m_handle;
  cudensitymatOperator_t m_liouvillian;
};
} // namespace

namespace cudaq {

void runge_kutta::set_system(const SystemDynamics &system) {
  m_system = system;
}

void runge_kutta::set_state(cudaq::state initial_state, double t0) {
  m_state = std::make_shared<cudaq::state>(initial_state);
  m_t = t0;
}

std::pair<double, cudaq::state> runge_kutta::get_state() {
  auto *simState = cudaq::state_helper::getSimulationState(m_state.get());
  auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
  if (!castSimState)
    throw std::runtime_error("Invalid state.");

  auto cudmState =
      new CuDensityMatState(castSimState->get_handle(), *castSimState,
                            castSimState->get_hilbert_space_dims());

  return std::make_pair(m_t, cudaq::state(cudmState));
}

void runge_kutta::integrate(double target_time) {
  const auto asCudmState = [](cudaq::state &cudaqState) -> CuDensityMatState * {
    auto *simState = cudaq::state_helper::getSimulationState(&cudaqState);
    auto *castSimState = dynamic_cast<CuDensityMatState *>(simState);
    if (!castSimState)
      throw std::runtime_error("Invalid state.");
    return castSimState;
  };
  auto &castSimState = *asCudmState(*m_state);
  if (!m_stepper) {
    static std::unordered_map<void *, std::unique_ptr<cudm_helper>> helpers;
    if (helpers.find(castSimState.get_handle()) == helpers.end())
      helpers[castSimState.get_handle()] =
          std::make_unique<cudm_helper>(castSimState.get_handle());
    auto &helper = *(helpers.find(castSimState.get_handle())->second);
    auto liouvillian = helper.construct_liouvillian(
        *m_system.hamiltonian, m_system.collapseOps, m_system.modeExtents, {},
        castSimState.is_density_matrix());
    m_stepper =
        std::make_unique<cudmStepper>(castSimState.get_handle(), liouvillian);
  }
  const auto substeps = order.value_or(4);
  while (m_t < target_time) {
    double step_size =
        std::min(dt.value_or(target_time - m_t), target_time - m_t);

    // std::cout << "Runge-Kutta step at time " << m_t
    //           << " with step size: " << step_size << std::endl;

    if (substeps == 1) {
      // Euler method (1st order)
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, {});
      auto &k1 = *asCudmState(k1State);
      // k1.dump(std::cout);
      k1 *= step_size;
      castSimState += k1;
    } else if (substeps == 2) {
      // Midpoint method (2nd order)
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, {});
      auto &k1 = *asCudmState(k1State);
      k1 *= (step_size / 2.0);

      castSimState += k1;

      auto k2State =
          m_stepper->compute(*m_state, m_t + step_size / 2.0, step_size, {});
      auto &k2 = *asCudmState(k2State);
      k2 *= (step_size / 2.0);

      castSimState += k2;
    } else if (substeps == 4) {
      // Runge-Kutta method (4th order)
      auto k1State = m_stepper->compute(*m_state, m_t, step_size, {});
      auto &k1 = *asCudmState(k1State);
      CuDensityMatState rho_temp = CuDensityMatState::clone(castSimState);
      rho_temp += (k1 * (step_size / 2));

      auto k2State = m_stepper->compute(
          cudaq::state(new CuDensityMatState(std::move(rho_temp))),
          m_t + step_size / 2.0, step_size, {});
      auto &k2 = *asCudmState(k2State);
      CuDensityMatState rho_temp_2 = CuDensityMatState::clone(castSimState);
      rho_temp_2 += (k2 * (step_size / 2));

      auto k3State = m_stepper->compute(
          cudaq::state(new CuDensityMatState(std::move(rho_temp_2))),
          m_t + step_size / 2.0, step_size, {});
      auto &k3 = *asCudmState(k3State);
      CuDensityMatState rho_temp_3 = CuDensityMatState::clone(castSimState);
      rho_temp_3 += (k3 * step_size);

      auto k4State = m_stepper->compute(
          cudaq::state(new CuDensityMatState(std::move(rho_temp_3))),
          m_t + step_size, step_size, {});
      auto &k4 = *asCudmState(k4State);
      castSimState += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (step_size / 6.0);
    } else {
      throw std::runtime_error("Invalid integrator order");
    }

    // Update time
    m_t += step_size;
  }
}

// void runge_kutta_integrator::set_state(cudaq::state initial_state, double t0) {
//   // TODO
// }
// std::pair<double, cudaq::state> runge_kutta_integrator::get_state() {
//   // TODO:
//   return std::make_pair(0.0, cudaq::state(nullptr));
// }

// // FIXME: remove this
// std::pair<double, cudm_state *> runge_kutta_integrator::get_cudm_state() {
//   return std::make_pair(m_t, m_state.get());
// }

// void runge_kutta_integrator::integrate(double target_time) {
//   if (!m_stepper) {
//     throw std::runtime_error("Time stepper is not initialized.");
//   }

//   if (dt.has_value() && dt.value() <= 0.0) {
//     throw std::invalid_argument("Invalid time step size for integration.");
//   }

//   if (!m_state) {
//     throw std::runtime_error("Initial state has not been set.");
//   }
//   const auto substeps = order.value_or(4);
//   while (m_t < target_time) {
//     double step_size =
//         std::min(dt.value_or(target_time - m_t), target_time - m_t);

//     // std::cout << "Runge-Kutta step at time " << m_t
//     //           << " with step size: " << step_size << std::endl;

//     if (substeps == 1) {
//       // Euler method (1st order)
//       cudm_state k1 = m_stepper->compute(*m_state, m_t, step_size);
//       k1 *= step_size;
//       *m_state += k1;
//     } else if (substeps == 2) {
//       // Midpoint method (2nd order)
//       cudm_state k1 = m_stepper->compute(*m_state, m_t, step_size);
//       k1 *= (step_size / 2.0);

//       *m_state += k1;

//       cudm_state k2 =
//           m_stepper->compute(*m_state, m_t + step_size / 2.0, step_size);
//       k2 *= (step_size / 2.0);

//       *m_state += k2;
//     } else if (substeps == 4) {
//       // Runge-Kutta method (4th order)
//       cudm_state k1 = m_stepper->compute(*m_state, m_t, step_size);

//       cudm_state rho_temp = cudm_state::clone(*m_state);
//       rho_temp += (k1 * (step_size / 2));

//       cudm_state k2 =
//           m_stepper->compute(rho_temp, m_t + step_size / 2.0, step_size);

//       cudm_state rho_temp_2 = cudm_state::clone(*m_state);
//       rho_temp_2 += (k2 * (step_size / 2));

//       cudm_state k3 =
//           m_stepper->compute(rho_temp_2, m_t + step_size / 2.0, step_size);

//       cudm_state rho_temp_3 = cudm_state::clone(*m_state);
//       rho_temp_3 += (k3 * step_size);

//       cudm_state k4 =
//           m_stepper->compute(rho_temp_3, m_t + step_size, step_size);

//       *m_state += (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (step_size / 6.0);
//     } else {
//       throw std::runtime_error("Invalid integrator order");
//     }

//     // Update time
//     m_t += step_size;
//   }

//   // std::cout << "Integration complete. Final time: " << m_t << std::endl;
// }

// // TODO: remove this
// runge_kutta_integrator::runge_kutta_integrator(
//     cudm_state &&initial_state, double t0,
//     std::shared_ptr<cudm_time_stepper> stepper, int substeps)
//     : m_t(t0), m_stepper(stepper), order(substeps) {

//   m_state = std::make_unique<cudm_state>(std::move(initial_state));
// }
// void runge_kutta_integrator::set_stepper(
//     std::shared_ptr<cudm_time_stepper> stepper) {
//   m_stepper = stepper;
// }

// void runge_kutta_integrator::set_state(cudm_state &&initial_state) {
//   m_state = std::make_unique<cudm_state>(std::move(initial_state));
//   m_t = 0.0;
// }
} // namespace cudaq
