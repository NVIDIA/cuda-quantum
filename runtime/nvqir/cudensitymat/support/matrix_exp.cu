/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#include "matrix_exp.h"
#include <algorithm>
#include <cmath>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <iostream>
#include <vector>

// Small-matrix Frobenius norm kernel
__global__ void
compute_frobenius_norm_kernel(const cuDoubleComplex *__restrict__ a, int n,
                              double *__restrict__ out) {
  extern __shared__ double sdata[];
  int tid = threadIdx.x;
  double sum = 0.0;
  for (int idx = tid; idx < n; idx += blockDim.x) {
    double re = a[idx].x;
    double im = a[idx].y;
    sum += re * re + im * im;
  }
  sdata[tid] = sum;
  __syncthreads();
  // Reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      sdata[tid] += sdata[tid + stride];
    }
    __syncthreads();
  }
  if (tid == 0) {
    out[0] = sdata[0];
  }
}

// Use standardized error checking (throws exceptions instead of return codes)
#include "cuda_memory.h"

#define MATRIX_EXP_CUDA_CHECK(call) CUDA_CHECK(call)
#define MATRIX_EXP_CUBLAS_CHECK(call) CUBLAS_CHECK(call)
#define MATRIX_EXP_CUSOLVER_CHECK(call) CUSOLVER_CHECK(call)

/// Kernel: Set matrix to identity
__global__ void set_identity_kernel(cuDoubleComplex *__restrict__ d_A, int N) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  const int col = blockIdx.y * blockDim.y + threadIdx.y;

  if (row < N && col < N) {
    d_A[row * N + col] = (row == col) ? make_cuDoubleComplex(1.0, 0.0)
                                      : make_cuDoubleComplex(0.0, 0.0);
  }
}

size_t get_matrix_exp_workspace_size(int N) {
  // Need space for:
  // - 8 matrices for Padé (A_scaled, A2, A4, A6, U, V, tmp, tmp2)
  // - Extra workspace for LU factorization (reserve 2 matrices worth)
  // - Pivot + info arrays
  const size_t matrix_bytes =
      static_cast<size_t>(N) * N * sizeof(cuDoubleComplex);
  const size_t matrix_storage = 8 * matrix_bytes;
  const size_t lu_workspace = 2 * matrix_bytes;
  const size_t pivots = static_cast<size_t>(N + 1) * sizeof(int);
  const size_t align_pad = alignof(cuDoubleComplex);
  return matrix_storage + lu_workspace + pivots + align_pad;
}

int compute_matrix_exp(cuDoubleComplex *d_A, cuDoubleComplex *d_expA, int N,
                       void *d_workspace, cublasHandle_t cublasH,
                       cusolverDnHandle_t cusolverH) {
  // Both handles must be provided
  if (!cublasH || !cusolverH) {
    throw std::runtime_error(
        "compute_matrix_exp requires valid cuBLAS and cuSOLVER handles");
  }

  cudaStream_t stream = nullptr;
  cublasGetStream(cublasH, &stream);

  // Workspace partitioning
  // Padé 13 uses 8 matrices + LU workspace/pivots
  cuDoubleComplex *d_A_scaled = (cuDoubleComplex *)d_workspace;
  cuDoubleComplex *d_A2 = d_A_scaled + N * N;
  cuDoubleComplex *d_A4 = d_A2 + N * N;
  cuDoubleComplex *d_A6 = d_A4 + N * N;
  cuDoubleComplex *d_U = d_A6 + N * N;
  cuDoubleComplex *d_V = d_U + N * N;
  cuDoubleComplex *d_tmp = d_V + N * N;
  cuDoubleComplex *d_tmp2 = d_tmp + N * N;
  char *extra = reinterpret_cast<char *>(d_tmp2 + N * N);
  int *d_pivots = reinterpret_cast<int *>(extra);
  extra += static_cast<size_t>(N) * sizeof(int);
  int *d_info = reinterpret_cast<int *>(extra);
  extra += sizeof(int);
  // Align LU workspace to cuDoubleComplex alignment
  constexpr std::size_t align = alignof(cuDoubleComplex);
  std::uintptr_t extra_addr = reinterpret_cast<std::uintptr_t>(extra);
  extra_addr = (extra_addr + align - 1) & ~(align - 1);
  cuDoubleComplex *d_lu_work = reinterpret_cast<cuDoubleComplex *>(extra_addr);

  // 1. Estimate norm of A (Frobenius norm)
  double norm_A = 0.0;

  if (N <= 16) {
    // Use custom CUDA norm for small matrices (avoids host-side dependency
    // issues or small-vec bugs) Use d_workspace as scratch for output (first
    // double)
    double *d_norm = reinterpret_cast<double *>(d_workspace);
    int nElems = N * N;
    int block = 256;
    int grid = 1;
    // Kernel defined above
    compute_frobenius_norm_kernel<<<grid, block, block * sizeof(double),
                                    stream>>>(
        reinterpret_cast<const cuDoubleComplex *>(d_A), nElems, d_norm);

    MATRIX_EXP_CUDA_CHECK(cudaGetLastError());
    MATRIX_EXP_CUDA_CHECK(
        cudaMemcpy(&norm_A, d_norm, sizeof(double), cudaMemcpyDeviceToHost));
    norm_A = std::sqrt(norm_A);
  } else {
    // Use cuBLAS for larger matrices
    MATRIX_EXP_CUBLAS_CHECK(cublasDznrm2(cublasH, N * N, d_A, 1, &norm_A));
  }

  // 2. Scaling: choose s such that ||A/2^s|| is small
  int s = 0;
  if (norm_A > 0.0) {
    s = std::max(0, static_cast<int>(std::ceil(std::log2(norm_A))));
  }
  const double scale = 1.0 / (1 << s); // 2^(-s)
  const cuDoubleComplex one = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex zero = make_cuDoubleComplex(0.0, 0.0);

  // A_scaled = A * scale
  cuDoubleComplex alpha = make_cuDoubleComplex(scale, 0.0);
  MATRIX_EXP_CUDA_CHECK(cudaMemcpy(d_A_scaled, d_A,
                                   N * N * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToDevice));
  MATRIX_EXP_CUBLAS_CHECK(cublasZscal(cublasH, N * N, &alpha, d_A_scaled, 1));

  if (norm_A == 0.0) {
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);
    set_identity_kernel<<<grid, block, 0, stream>>>(d_expA, N);
    MATRIX_EXP_CUDA_CHECK(cudaGetLastError());
    return 0;
  }

  // 3. Padé [13/13] approximation
  // Coefficients for Padé(13)
  constexpr double b0 = 64764752532480000.0;
  constexpr double b1 = 32382376266240000.0;
  constexpr double b2 = 7771770303897600.0;
  constexpr double b3 = 1187353796428800.0;
  constexpr double b4 = 129060195264000.0;
  constexpr double b5 = 10559470521600.0;
  constexpr double b6 = 670442572800.0;
  constexpr double b7 = 33522128640.0;
  constexpr double b8 = 1323241920.0;
  constexpr double b9 = 40840800.0;
  constexpr double b10 = 960960.0;
  constexpr double b11 = 16380.0;
  constexpr double b12 = 182.0;
  constexpr double b13 = 1.0;

  const int nn = N * N;
  cuDoubleComplex beta = zero;

  // A2 = A_scaled^2
  alpha = one;
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_A_scaled, N, d_A_scaled, N,
                                      &beta, d_A2, N));

  // A4 = A2^2
  alpha = one;
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_A2, N, d_A2, N, &beta, d_A4,
                                      N));

  // A6 = A2 * A4
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_A2, N, d_A4, N, &beta, d_A6,
                                      N));

  // --- Compute U ---
  // tmp = b13*A6 + b11*A4 + b9*A2
  MATRIX_EXP_CUDA_CHECK(cudaMemcpy(d_tmp, d_A6, N * N * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToDevice));
  alpha = make_cuDoubleComplex(b13, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZscal(cublasH, nn, &alpha, d_tmp, 1));
  alpha = make_cuDoubleComplex(b11, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A4, 1, d_tmp, 1));
  alpha = make_cuDoubleComplex(b9, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A2, 1, d_tmp, 1));

  // tmp2 = A6 * tmp
  alpha = one;
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_A6, N, d_tmp, N, &beta,
                                      d_tmp2, N));

  // tmp2 += b7*A6 + b5*A4 + b3*A2 + b1*I
  alpha = make_cuDoubleComplex(b7, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A6, 1, d_tmp2, 1));
  alpha = make_cuDoubleComplex(b5, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A4, 1, d_tmp2, 1));
  alpha = make_cuDoubleComplex(b3, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A2, 1, d_tmp2, 1));

  // Add b1 * I
  dim3 block(16, 16);
  dim3 grid((N + 15) / 16, (N + 15) / 16);
  set_identity_kernel<<<grid, block, 0, stream>>>(d_tmp, N);
  alpha = make_cuDoubleComplex(b1, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(
      cublasZaxpy(cublasH, nn, &alpha, d_tmp, 1, d_tmp2, 1));

  // U = A_scaled * tmp2
  alpha = one;
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_A_scaled, N, d_tmp2, N,
                                      &beta, d_U, N));

  // --- Compute V ---
  // tmp = b12*A6 + b10*A4 + b8*A2
  MATRIX_EXP_CUDA_CHECK(cudaMemcpy(d_tmp, d_A6, N * N * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToDevice));
  alpha = make_cuDoubleComplex(b12, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZscal(cublasH, nn, &alpha, d_tmp, 1));
  alpha = make_cuDoubleComplex(b10, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A4, 1, d_tmp, 1));
  alpha = make_cuDoubleComplex(b8, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A2, 1, d_tmp, 1));

  // tmp2 = A6 * tmp
  alpha = one;
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_A6, N, d_tmp, N, &beta,
                                      d_tmp2, N));

  // V = tmp2 + b6*A6 + b4*A4 + b2*A2 + b0*I
  MATRIX_EXP_CUDA_CHECK(cudaMemcpy(d_V, d_tmp2, N * N * sizeof(cuDoubleComplex),
                                   cudaMemcpyDeviceToDevice));
  alpha = make_cuDoubleComplex(b6, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A6, 1, d_V, 1));
  alpha = make_cuDoubleComplex(b4, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A4, 1, d_V, 1));
  alpha = make_cuDoubleComplex(b2, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_A2, 1, d_V, 1));

  set_identity_kernel<<<grid, block, 0, stream>>>(d_tmp, N);
  alpha = make_cuDoubleComplex(b0, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZaxpy(cublasH, nn, &alpha, d_tmp, 1, d_V, 1));

  // Solve (V - U) X = (V + U)
  // tmp2 = V + U
  alpha = one;
  beta = one;
  MATRIX_EXP_CUBLAS_CHECK(cublasZgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      &alpha, d_V, N, &beta, d_U, N, d_tmp2,
                                      N));

  // V = V - U
  beta = make_cuDoubleComplex(-1.0, 0.0);
  MATRIX_EXP_CUBLAS_CHECK(cublasZgeam(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      &alpha, d_V, N, &beta, d_U, N, d_V, N));

  // LU factorization of (V - U)
  int lwork = 0;
  MATRIX_EXP_CUSOLVER_CHECK(
      cusolverDnZgetrf_bufferSize(cusolverH, N, N, d_V, N, &lwork));
  const size_t lu_workspace_bytes =
      2 * static_cast<size_t>(N) * N * sizeof(cuDoubleComplex);
  const size_t required_bytes =
      static_cast<size_t>(lwork) * sizeof(cuDoubleComplex);
  cuDoubleComplex *lu_work = d_lu_work;
  bool lu_allocated = false;
  if (required_bytes > lu_workspace_bytes) {
    MATRIX_EXP_CUDA_CHECK(cudaMalloc(&lu_work, required_bytes));
    lu_allocated = true;
  }

  MATRIX_EXP_CUSOLVER_CHECK(
      cusolverDnZgetrf(cusolverH, N, N, d_V, N, lu_work, d_pivots, d_info));

  MATRIX_EXP_CUSOLVER_CHECK(cusolverDnZgetrs(cusolverH, CUBLAS_OP_N, N, N, d_V,
                                             N, d_pivots, d_tmp2, N, d_info));

  if (lu_allocated) {
    cudaFree(lu_work);
  }

  // 4. Squaring: exp(A) = (exp(A_scaled))^(2^s)
  // We perform 's' matrix multiplications using ping-pong buffering

  cuDoubleComplex *current = d_tmp2;
  cuDoubleComplex *next = d_tmp; // Use tmp as temp buffer

  alpha = make_cuDoubleComplex(1.0, 0.0);
  beta = make_cuDoubleComplex(0.0, 0.0);

  for (int i = 0; i < s; ++i) {
    // Square: next = current * current
    MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                        N, &alpha, current, N, current, N,
                                        &beta, next, N));

    // Swap pointers
    std::swap(current, next);
  }

  // Final result is in 'current'
  // Copy to d_expA (the output buffer)
  if (current != d_expA) {
    MATRIX_EXP_CUDA_CHECK(cudaMemcpy(d_expA, current,
                                     N * N * sizeof(cuDoubleComplex),
                                     cudaMemcpyDeviceToDevice));
  }

  return 0;
}

int apply_unitary_to_density_matrix(const cuDoubleComplex *d_U,
                                    cuDoubleComplex *d_rho, int N,
                                    void *d_workspace, cublasHandle_t cublasH) {
  if (!cublasH) {
    throw std::runtime_error(
        "apply_unitary_to_density_matrix requires valid cuBLAS handle");
  }

  // Workspace for temp = U * rho
  cuDoubleComplex *d_temp = static_cast<cuDoubleComplex *>(d_workspace);

  const cuDoubleComplex alpha = make_cuDoubleComplex(1.0, 0.0);
  const cuDoubleComplex beta = make_cuDoubleComplex(0.0, 0.0);

  // Step 1: temp = U * rho
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_N, N, N,
                                      N, &alpha, d_U, N, d_rho, N, &beta,
                                      d_temp, N));

  // Step 2: rho = temp * U† (U† = conjugate transpose of U)
  MATRIX_EXP_CUBLAS_CHECK(cublasZgemm(cublasH, CUBLAS_OP_N, CUBLAS_OP_C, N, N,
                                      N, &alpha, d_temp, N, d_U, N, &beta,
                                      d_rho, N));

  return 0;
}
