/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "error_norm.h"
#include "CuDensityMatErrorHandling.h"
#include "CuDensityMatUtils.h"

#include <algorithm>
#include <cuComplex.h>
#include <cuda_runtime.h>

namespace cudaq::detail {

namespace {

/// Per-element scaled squared error with a block-level reduction. Each block
/// atomically accumulates its partial sum into a single device scalar, so the
/// whole error norm is computed without any full-state device-to-host copy.
__global__ void scaled_error_kernel(const cuDoubleComplex *__restrict__ y5,
                                    const cuDoubleComplex *__restrict__ y4,
                                    std::size_t n, double rtol, double atol,
                                    double *__restrict__ out) {
  extern __shared__ double sdata[];
  const unsigned tid = threadIdx.x;

  double local = 0.0;
  for (std::size_t i = blockIdx.x * blockDim.x + tid; i < n;
       i += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
    const double er = y5[i].x - y4[i].x;
    const double ei = y5[i].y - y4[i].y;
    const double errMag = sqrt(er * er + ei * ei);
    const double mag5 = sqrt(y5[i].x * y5[i].x + y5[i].y * y5[i].y);
    const double mag4 = sqrt(y4[i].x * y4[i].x + y4[i].y * y4[i].y);
    // scipy-style element-wise scale: atol + rtol * max(|y5|, |y4|).
    const double scale = atol + rtol * fmax(mag5, mag4);
    const double ratio = errMag / scale;
    local += ratio * ratio;
  }

  sdata[tid] = local;
  __syncthreads();
  for (unsigned stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride)
      sdata[tid] += sdata[tid + stride];
    __syncthreads();
  }
  if (tid == 0)
    atomicAdd(out, sdata[0]);
}

} // namespace

void scaled_error_sumsq(const void *y5, const void *y4, std::size_t n,
                        double rtol, double atol, double *sumsq_out) {
  const auto *p5 = static_cast<const cuDoubleComplex *>(y5);
  const auto *p4 = static_cast<const cuDoubleComplex *>(y4);

  auto *d_out = static_cast<double *>(
      cudaq::dynamics::DeviceAllocator::allocate(sizeof(double)));
  HANDLE_CUDA_ERROR(cudaMemset(d_out, 0, sizeof(double)));

  constexpr int block = 256;
  int grid = static_cast<int>((n + block - 1) / block);
  grid = std::max(1, std::min(grid, 1024));

  scaled_error_kernel<<<grid, block, block * sizeof(double)>>>(
      p5, p4, n, rtol, atol, d_out);
  HANDLE_CUDA_ERROR(cudaGetLastError());
  HANDLE_CUDA_ERROR(
      cudaMemcpy(sumsq_out, d_out, sizeof(double), cudaMemcpyDeviceToHost));

  cudaq::dynamics::DeviceAllocator::free(d_out);
}

} // namespace cudaq::detail
