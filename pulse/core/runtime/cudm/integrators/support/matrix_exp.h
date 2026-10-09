/****************************************************************-*- C++ -*-****
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.
 * All rights reserved.
 *
 * This source code and the accompanying materials are made available under
 * the terms of the Apache License 2.0 which accompanies this distribution.
 ******************************************************************************/

#pragma once

#include <cstdint>
#include <cuComplex.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

/// \file matrix_exp.h
/// \brief Matrix exponential for exact PWC Hamiltonian evolution.
///
/// Implements exp(A) for complex matrices using scaling and squaring with
/// Padé [13/13] approximation (Higham 2005). Used by Magnus CF4 integrator.

/// \brief Compute matrix exponential: B = exp(A) using scaling and squaring.
///
/// Algorithm: Scaling and squaring with Padé [13/13] approximation.
/// Based on Higham 2005, using the standard Padé coefficients.
///
/// Steps:
///   1. Scale: A' = A / 2^s where s is chosen so ||A'|| is small.
///   2. Padé: Compute U and V polynomials for [13/13].
///   3. Solve: exp(A') = (V + U) × (V - U)⁻¹.
///   4. Square: exp(A) = (exp(A'))^(2^s).
///
/// \param d_A Input matrix A (N×N, will be modified as workspace).
/// \param d_expA Output matrix exp(A) (N×N).
/// \param N Matrix dimension.
/// \param d_workspace Workspace buffer (see get_matrix_exp_workspace_size).
/// \param cublasH cuBLAS handle (required, must be valid).
/// \param cusolverH cuSOLVER handle (required, must be valid).
/// \return 0 on success, throws exception on error.
int compute_matrix_exp(cuDoubleComplex *d_A, cuDoubleComplex *d_expA, int N,
                       void *d_workspace, cublasHandle_t cublasH,
                       cusolverDnHandle_t cusolverH);

/// \brief Query workspace size needed for matrix exponential.
/// \param N Matrix dimension.
/// \return Workspace size in bytes.
size_t get_matrix_exp_workspace_size(int N);

/// \brief Apply unitary propagator to density matrix: ρ_new = U ρ U†.
///
/// For Hamiltonian evolution with PWC Hamiltonians:
///   U = exp(-i H Δt), ρ(t+Δt) = U ρ(t) U†.
///
/// \param d_U Unitary operator U (N×N, read-only).
/// \param d_rho Density matrix ρ (N×N, modified in-place).
/// \param N Hilbert space dimension.
/// \param d_workspace Workspace buffer (at least N×N complex elements).
/// \param cublasH cuBLAS handle (required, must be valid).
/// \return 0 on success, throws exception on error.
int apply_unitary_to_density_matrix(const cuDoubleComplex *d_U,
                                    cuDoubleComplex *d_rho, int N,
                                    void *d_workspace, cublasHandle_t cublasH);
