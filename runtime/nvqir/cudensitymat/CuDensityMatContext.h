/*************************************************************** -*- C++ -*- ***
 * Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#pragma once
#include "CuDensityMatOpConverter.h"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cudensitymat.h>

namespace cudaq {
namespace dynamics {
class Context {
public:
  Context(Context const &) = delete;
  Context &operator=(Context const &) = delete;
  ~Context();

  cudensitymatHandle_t getHandle() const { return m_cudmHandle; }
  cublasHandle_t getCublasHandle() const { return m_cublasHandle; }
  OpConverter &getOpConverter() { return *m_opConverter; }
  static Context *getCurrentContext();
  void *getScratchSpace(std::size_t minSizeBytes);
  static std::size_t getRecommendedWorkSpaceLimit();

private:
  Context(int deviceId);
  cudensitymatHandle_t m_cudmHandle;
  cublasHandle_t m_cublasHandle;
  std::unique_ptr<OpConverter> m_opConverter;
  int m_deviceId;
  void *m_scratchSpace{nullptr};
  std::size_t m_scratchSpaceSizeBytes{0};
};
} // namespace dynamics
} // namespace cudaq
