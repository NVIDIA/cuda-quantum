/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once

#include "CuDensityMatState.h"
#include "cudaq/algorithms/base_time_stepper.h"
#include <cudensitymat.h>

namespace cudaq {
class CuDensityMatTimeStepper : public base_time_stepper {
public:
  explicit CuDensityMatTimeStepper(cudensitymatHandle_t handle,
                                   cudensitymatOperator_t liouvillian);
  CuDensityMatTimeStepper(const CuDensityMatTimeStepper &) = delete;
  CuDensityMatTimeStepper &operator=(const CuDensityMatTimeStepper &) = delete;

  state compute(const state &inputState, double t,
                const std::unordered_map<std::string, std::complex<double>>
                    &parameters) override;

  /// @brief Overwrite `outputState` with the action of the Liouvillian on
  /// `inputState`. Both states must have the same shape.
  void computeInto(
      const CuDensityMatState &inputState, CuDensityMatState &outputState,
      double t,
      const std::unordered_map<std::string, std::complex<double>> &parameters);

  /// @brief Return reusable state number `index`, shaped like `like`, for an
  /// integrator's intermediate results. Its contents are unspecified. The
  /// state is kept across calls and replaced only if its shape differs.
  CuDensityMatState &workspaceState(std::size_t index,
                                    const CuDensityMatState &like);

  void computeImpl(
      cudensitymatState_t inState, cudensitymatState_t outState, double t,
      const std::unordered_map<std::string, std::complex<double>> &parameters,
      int64_t batchSize);

private:
  cudensitymatHandle_t m_handle;
  cudensitymatOperator_t m_liouvillian;
  std::vector<std::unique_ptr<CuDensityMatState>> m_workspaceStates;
};
} // namespace cudaq
