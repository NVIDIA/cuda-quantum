// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#ifndef CUDAQ_PULSE_RUNTIME_CUDM_RUNTIME_H
#define CUDAQ_PULSE_RUNTIME_CUDM_RUNTIME_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *CudmHandle;
typedef void *CudmState;
typedef void *CudmWorkspace;
typedef void *CudmOperator;
typedef void *CudmElementaryOp;
typedef void *CudmOpTerm;

typedef enum {
  CUDM_SUCCESS = 0,
  CUDM_ERROR_INVALID_HANDLE = 1,
  CUDM_ERROR_INVALID_STATE = 2,
  CUDM_ERROR_NO_GPU = 3,
  CUDM_ERROR_CUDA = 4,
  CUDM_ERROR_INTERNAL = 99,
} CudmStatus;

// Context management
CudmStatus cudm_init(CudmHandle *handle);
CudmStatus cudm_destroy(CudmHandle handle);

// State management
CudmStatus cudm_state_alloc(CudmHandle handle, CudmState *state,
                            const int64_t *mode_extents, int32_t num_modes,
                            int32_t purity, int32_t data_type);
CudmStatus cudm_state_destroy(CudmState state);
CudmStatus cudm_state_init_zero(CudmHandle handle, CudmState state);

// Workspace management
CudmStatus cudm_workspace_create(CudmHandle handle, CudmWorkspace *ws);
CudmStatus cudm_workspace_destroy(CudmWorkspace ws);

// Operator management
CudmStatus cudm_operator_create(CudmHandle handle, CudmOperator *op,
                                const int64_t *mode_extents, int32_t num_modes);
CudmStatus cudm_operator_destroy(CudmOperator op);

// Elementary operator
CudmStatus cudm_elementary_op_create(CudmHandle handle,
                                     CudmElementaryOp *elem_op,
                                     const void *tensor_data);
CudmStatus cudm_elementary_op_destroy(CudmElementaryOp elem_op);

// Operator term
CudmStatus cudm_op_term_create(CudmHandle handle, CudmOpTerm *term);
CudmStatus cudm_op_term_append(CudmHandle handle, CudmOpTerm term,
                               int32_t num_elementary_ops, double coeff_real,
                               double coeff_imag);
CudmStatus cudm_op_term_destroy(CudmOpTerm term);

// Composite operator assembly
CudmStatus cudm_operator_append(CudmHandle handle, CudmOperator op,
                                CudmOpTerm term, int32_t duality,
                                double coeff_real, double coeff_imag);

// Time evolution
CudmStatus cudm_evolve_step(CudmHandle handle, CudmOperator op,
                            CudmState state_in, CudmState state_out,
                            CudmWorkspace ws, double t, double dt,
                            int32_t integrator);

// Full evolve (high-level convenience)
CudmStatus cudm_evolve(CudmHandle handle, CudmOperator op, CudmState state_in,
                       CudmState state_out, CudmWorkspace ws, double t_start,
                       double t_end, int64_t num_steps, int32_t integrator);

// Expectation value
CudmStatus cudm_observe(CudmHandle handle, CudmOperator op, CudmState state,
                        CudmWorkspace ws, double t, double *real, double *imag);

#ifdef __cplusplus
}
#endif

#endif // CUDAQ_PULSE_RUNTIME_CUDM_RUNTIME_H
