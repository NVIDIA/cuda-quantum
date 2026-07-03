// Copyright (c) 2026 NVIDIA Corporation & Affiliates.
// All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "cudm_runtime.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>

namespace {

void *getCudmLib() {
  static void *lib = [] {
    const char *path = std::getenv("CUDM_LIB_PATH");
    void *handle = nullptr;
    if (path)
      handle = dlopen(path, RTLD_LAZY | RTLD_GLOBAL);
    if (!handle)
      handle = dlopen("libcudensitymat.so", RTLD_LAZY | RTLD_GLOBAL);
    if (!handle)
      handle = dlopen("libcudensitymat.dylib", RTLD_LAZY | RTLD_GLOBAL);
    return handle;
  }();
  return lib;
}

template <typename FnT>
FnT *resolve(const char *name) {
  auto *lib = getCudmLib();
  if (!lib)
    return nullptr;
  return reinterpret_cast<FnT *>(dlsym(lib, name));
}

struct HandleData {
  void *cudm_handle;
};

} // namespace

extern "C" {

// ---- Context management ----

CudmStatus cudm_init(CudmHandle *handle) {
  if (!getCudmLib())
    return CUDM_ERROR_NO_GPU;

  auto *data = new HandleData{};
  using CreateFn = int(void **);
  auto *fn = resolve<CreateFn>("cudensitymatCreate");
  if (!fn)
    return CUDM_ERROR_INTERNAL;
  int status = fn(&data->cudm_handle);
  if (status != 0) {
    delete data;
    return CUDM_ERROR_CUDA;
  }
  *handle = static_cast<CudmHandle>(data);
  return CUDM_SUCCESS;
}

CudmStatus cudm_destroy(CudmHandle handle) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);
  using DestroyFn = int(void *);
  auto *fn = resolve<DestroyFn>("cudensitymatDestroy");
  if (fn)
    fn(data->cudm_handle);
  delete data;
  return CUDM_SUCCESS;
}

// ---- State management ----

CudmStatus cudm_state_alloc(CudmHandle handle, CudmState *state,
                            const int64_t *mode_extents, int32_t num_modes,
                            int32_t purity, int32_t data_type) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  // cudensitymatCreateState(handle, purity, numModes, modeExtents, batchSize,
  //                         statePtr)
  using Fn = int(void *, int32_t, int32_t, const int64_t *, int64_t, void **);
  auto *fn = resolve<Fn>("cudensitymatCreateState");
  if (!fn) {
    *state = nullptr;
    return CUDM_ERROR_INTERNAL;
  }
  void *st = nullptr;
  int status = fn(data->cudm_handle, purity, num_modes, mode_extents, 0, &st);
  if (status != 0)
    return CUDM_ERROR_CUDA;
  *state = st;
  return CUDM_SUCCESS;
}

CudmStatus cudm_state_destroy(CudmState state) {
  if (!state)
    return CUDM_ERROR_INVALID_STATE;
  using Fn = int(void *);
  auto *fn = resolve<Fn>("cudensitymatDestroyState");
  if (fn)
    fn(state);
  return CUDM_SUCCESS;
}

CudmStatus cudm_state_init_zero(CudmHandle handle, CudmState state) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  if (!state)
    return CUDM_ERROR_INVALID_STATE;
  // TODO: Forward to cudensitymatStateInitZero when available
  return CUDM_SUCCESS;
}

// ---- Workspace ----

CudmStatus cudm_workspace_create(CudmHandle handle, CudmWorkspace *ws) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  using Fn = int(void *, void **);
  auto *fn = resolve<Fn>("cudensitymatCreateWorkspace");
  if (!fn) {
    *ws = nullptr;
    return CUDM_ERROR_INTERNAL;
  }
  void *w = nullptr;
  int status = fn(data->cudm_handle, &w);
  if (status != 0)
    return CUDM_ERROR_CUDA;
  *ws = w;
  return CUDM_SUCCESS;
}

CudmStatus cudm_workspace_destroy(CudmWorkspace ws) {
  if (!ws)
    return CUDM_SUCCESS;
  using Fn = int(void *);
  auto *fn = resolve<Fn>("cudensitymatDestroyWorkspace");
  if (fn)
    fn(ws);
  return CUDM_SUCCESS;
}

// ---- Operator ----

CudmStatus cudm_operator_create(CudmHandle handle, CudmOperator *op,
                                const int64_t *mode_extents,
                                int32_t num_modes) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  using Fn = int(void *, int32_t, const int64_t *, void **);
  auto *fn = resolve<Fn>("cudensitymatCreateOperator");
  if (!fn) {
    *op = nullptr;
    return CUDM_ERROR_INTERNAL;
  }
  void *o = nullptr;
  int status = fn(data->cudm_handle, num_modes, mode_extents, &o);
  if (status != 0)
    return CUDM_ERROR_CUDA;
  *op = o;
  return CUDM_SUCCESS;
}

CudmStatus cudm_operator_destroy(CudmOperator op) {
  if (!op)
    return CUDM_SUCCESS;
  using Fn = int(void *);
  auto *fn = resolve<Fn>("cudensitymatDestroyOperator");
  if (fn)
    fn(op);
  return CUDM_SUCCESS;
}

// ---- Elementary operator ----

CudmStatus cudm_elementary_op_create(CudmHandle handle,
                                     CudmElementaryOp *elem_op,
                                     const void *tensor_data) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  using Fn = int(void *, const void *, void **);
  auto *fn = resolve<Fn>("cudensitymatCreateElementaryOp");
  if (!fn) {
    *elem_op = nullptr;
    return CUDM_ERROR_INTERNAL;
  }
  void *eo = nullptr;
  int status = fn(data->cudm_handle, tensor_data, &eo);
  if (status != 0)
    return CUDM_ERROR_CUDA;
  *elem_op = eo;
  return CUDM_SUCCESS;
}

CudmStatus cudm_elementary_op_destroy(CudmElementaryOp elem_op) {
  if (!elem_op)
    return CUDM_SUCCESS;
  using Fn = int(void *);
  auto *fn = resolve<Fn>("cudensitymatDestroyElementaryOp");
  if (fn)
    fn(elem_op);
  return CUDM_SUCCESS;
}

// ---- Operator term ----

CudmStatus cudm_op_term_create(CudmHandle handle, CudmOpTerm *term) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  using Fn = int(void *, void **);
  auto *fn = resolve<Fn>("cudensitymatCreateOpTerm");
  if (!fn) {
    *term = nullptr;
    return CUDM_ERROR_INTERNAL;
  }
  void *t = nullptr;
  int status = fn(data->cudm_handle, &t);
  if (status != 0)
    return CUDM_ERROR_CUDA;
  *term = t;
  return CUDM_SUCCESS;
}

CudmStatus cudm_op_term_append(CudmHandle handle, CudmOpTerm term,
                               int32_t num_elementary_ops, double coeff_real,
                               double coeff_imag) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  if (!term)
    return CUDM_ERROR_INTERNAL;

  using Fn = int(void *, void *, int32_t, double, double);
  auto *fn = resolve<Fn>("cudensitymatOpTermAppend");
  if (!fn)
    return CUDM_ERROR_INTERNAL;
  auto *data = static_cast<HandleData *>(handle);
  int status =
      fn(data->cudm_handle, term, num_elementary_ops, coeff_real, coeff_imag);
  return status == 0 ? CUDM_SUCCESS : CUDM_ERROR_CUDA;
}

CudmStatus cudm_op_term_destroy(CudmOpTerm term) {
  if (!term)
    return CUDM_SUCCESS;
  using Fn = int(void *);
  auto *fn = resolve<Fn>("cudensitymatDestroyOpTerm");
  if (fn)
    fn(term);
  return CUDM_SUCCESS;
}

// ---- Composite operator assembly ----

CudmStatus cudm_operator_append(CudmHandle handle, CudmOperator op,
                                CudmOpTerm term, int32_t duality,
                                double coeff_real, double coeff_imag) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  if (!op || !term)
    return CUDM_ERROR_INTERNAL;

  using Fn = int(void *, void *, void *, int32_t, double, double);
  auto *fn = resolve<Fn>("cudensitymatOperatorAppendTerm");
  if (!fn)
    return CUDM_ERROR_INTERNAL;
  auto *data = static_cast<HandleData *>(handle);
  int status = fn(data->cudm_handle, op, term, duality, coeff_real, coeff_imag);
  return status == 0 ? CUDM_SUCCESS : CUDM_ERROR_CUDA;
}

// ---- Time evolution ----

CudmStatus cudm_evolve_step(CudmHandle handle, CudmOperator op,
                            CudmState state_in, CudmState state_out,
                            CudmWorkspace ws, double t, double dt,
                            int32_t integrator) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  using Fn =
      int(void *, void *, void *, void *, void *, double, double, int32_t);
  auto *fn = resolve<Fn>("cudensitymatEvolveStep");
  if (!fn)
    return CUDM_ERROR_INTERNAL;
  int status =
      fn(data->cudm_handle, op, state_in, state_out, ws, t, dt, integrator);
  return status == 0 ? CUDM_SUCCESS : CUDM_ERROR_CUDA;
}

CudmStatus cudm_evolve(CudmHandle handle, CudmOperator op, CudmState state_in,
                       CudmState state_out, CudmWorkspace ws, double t_start,
                       double t_end, int64_t num_steps, int32_t integrator) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  double dt = (t_end - t_start) / static_cast<double>(num_steps);
  for (int64_t step = 0; step < num_steps; step++) {
    double t = t_start + step * dt;
    CudmStatus st = cudm_evolve_step(handle, op, state_in, state_out, ws, t, dt,
                                     integrator);
    if (st != CUDM_SUCCESS)
      return st;
  }
  return CUDM_SUCCESS;
}

// ---- Expectation ----

CudmStatus cudm_observe(CudmHandle handle, CudmOperator op, CudmState state,
                        CudmWorkspace ws, double t, double *real,
                        double *imag) {
  if (!handle)
    return CUDM_ERROR_INVALID_HANDLE;
  auto *data = static_cast<HandleData *>(handle);

  using Fn = int(void *, void *, void *, void *, double, double *, double *);
  auto *fn = resolve<Fn>("cudensitymatExpectationCompute");
  if (!fn) {
    *real = 0.0;
    *imag = 0.0;
    return CUDM_ERROR_INTERNAL;
  }
  int status = fn(data->cudm_handle, op, state, ws, t, real, imag);
  return status == 0 ? CUDM_SUCCESS : CUDM_ERROR_CUDA;
}

} // extern "C"
