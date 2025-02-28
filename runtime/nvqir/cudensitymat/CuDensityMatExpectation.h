/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include <complex>
#include <cudensitymat.h>
namespace cudaq {

class CuDensityMatExpectation {
  cudensitymatHandle_t m_handle{nullptr};
  cudensitymatOperator_t m_hamOp{nullptr};
  cudensitymatExpectation_t m_expectation{nullptr};
  cudensitymatWorkspaceDescriptor_t m_workspace{nullptr};

public:
  CuDensityMatExpectation(cudensitymatHandle_t handle,
                          cudensitymatOperator_t op);
  CuDensityMatExpectation(const CuDensityMatExpectation &) = delete;
  CuDensityMatExpectation &operator=(const CuDensityMatExpectation &) = delete;
  CuDensityMatExpectation(CuDensityMatExpectation &&src) {
    std::swap(m_handle, src.m_handle);
    std::swap(m_hamOp, src.m_hamOp);
    std::swap(m_expectation, src.m_expectation);
    std::swap(m_workspace, src.m_workspace);
  }
  ~CuDensityMatExpectation();
  void prepare(cudensitymatState_t state);
  std::complex<double> compute(cudensitymatState_t state, double time);
};

} // namespace cudaq
