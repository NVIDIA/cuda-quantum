/*******************************************************************************
 * Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                  *
 * All rights reserved.                                                        *
 *                                                                             *
 * This source code and the accompanying materials are made available under    *
 * the terms of the Apache License 2.0 which accompanies this distribution.    *
 ******************************************************************************/

#include "CuDensityMatContext.h"
#include "CuDensityMatIntegratorBase.h"
#include "CuDensityMatState.h"
#include "CuDensityMatUtils.h"
#include "support/cuda_memory.h"
#include "support/matrix_exp.h"
#include "support/propagator_cache.h"
#include "cudaq/algorithms/integrator.h"
#include "cudaq/runtime/logger/logger.h"

#include <cmath>
#include <complex>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace cudaq::integrators {

// High-order commutator-free Magnus integrator (CF4).
//
// For closed-system density-matrix evolution this materialises the dense
// Hamiltonian at the two Gauss-Legendre nodes of a step, forms the two
// commutator-free exponential factors, computes each propagator with a GPU
// matrix exponential (support/matrix_exp.cu), and applies them to the density
// matrix as rho <- U rho U^dagger. Propagators are cached (LRU) keyed by a
// quantized Hamiltonian signature so repeated piecewise-constant slices reuse a
// single matrix exponential.
//
// Reference (CF4, s=2): S. Blanes, P. C. Moan, "Fourth- and sixth-order
// commutator-free Magnus integrators for linear and non-linear dynamical
// systems", Appl. Numer. Math. 56 (2006).
//
// For open systems (collapse operators / super-operator) or state-vector
// evolution, this falls back to `magnus_expansion` for correctness parity.

using cudmIntHelp = CuDensityMatIntegratorHelper;

namespace {

// FNV-1a signature over a quantized dense matrix. Quantization (to ~1e-9)
// absorbs floating-point noise so that identical PWC slices hash identically.
std::size_t hashMatrixBuffer(const std::vector<std::complex<double>> &buf) {
  std::size_t h = 14695981039346656037ULL;
  constexpr std::size_t fnvPrime = 1099511628211ULL;
  constexpr double invQuantum = 1e9; // 1e-9 resolution.
  auto combine = [&](std::int64_t v) {
    h ^= static_cast<std::size_t>(v);
    h *= fnvPrime;
  };
  for (const auto &z : buf) {
    combine(static_cast<std::int64_t>(std::llround(z.real() * invQuantum)));
    combine(static_cast<std::int64_t>(std::llround(z.imag() * invQuantum)));
  }
  return h;
}

} // namespace

/// @brief Hidden CUDA/cache state for `magnus_cf4`.
struct magnus_cf4::Impl {
  cudaq::detail::PropagatorLRUCache cache;
  cudaq::detail::CudaComplexMemory dH;        // Scaled Hamiltonian / scratch.
  cudaq::detail::CudaComplexMemory applyWork; // Workspace for U rho U^dagger.
  void *expWorkspace = nullptr;
  std::size_t expWorkspaceBytes = 0;
  cusolverDnHandle_t cusolver = nullptr;
  int dim = 0;

  explicit Impl(std::size_t capacity) : cache(capacity) {}

  ~Impl() {
    if (expWorkspace)
      cudaFree(expWorkspace);
    if (cusolver)
      cusolverDnDestroy(cusolver);
  }

  // Lazily allocate device buffers / handles for a given Hilbert dimension.
  void ensureResources(int hilbertDim, cublasHandle_t cublas) {
    if (!cusolver) {
      CUSOLVER_CHECK(cusolverDnCreate(&cusolver));
      cudaStream_t stream = nullptr;
      cublasGetStream(cublas, &stream);
      CUSOLVER_CHECK(cusolverDnSetStream(cusolver, stream));
    }
    if (hilbertDim != dim) {
      dim = hilbertDim;
      const std::size_t nn = static_cast<std::size_t>(dim) * dim;
      dH.reallocate(nn);
      applyWork.reallocate(nn);
      const std::size_t needed = get_matrix_exp_workspace_size(dim);
      if (needed > expWorkspaceBytes) {
        if (expWorkspace)
          cudaFree(expWorkspace);
        CUDA_CHECK(cudaMalloc(&expWorkspace, needed));
        expWorkspaceBytes = needed;
      }
      // Dimension-specific cache entries are stale for a new dimension.
      cache.clear();
    }
  }
};

magnus_cf4::magnus_cf4(const std::optional<double> &max_step_size,
                       std::size_t cache_capacity)
    : m_t(0.0), m_dt(max_step_size), m_cache_capacity(cache_capacity),
      m_impl(std::make_shared<Impl>(cache_capacity)) {}

magnus_cf4::~magnus_cf4() = default;

std::shared_ptr<base_integrator> magnus_cf4::clone() {
  auto cloned =
      std::make_shared<cudaq::integrators::magnus_cf4>(m_dt, m_cache_capacity);
  cloned->m_t = this->m_t;
  cloned->m_state = this->m_state;
  cloned->m_system = this->m_system;
  cloned->m_schedule = this->m_schedule;
  // Note: the propagator cache is intentionally not shared; the clone starts
  // with an empty cache so async copies do not race on device buffers.
  return cloned;
}

void magnus_cf4::setState(const cudaq::state &initialState, double t0) {
  cudmIntHelp::setState(m_state, m_t, initialState, t0);
  resetStats();
}

std::pair<double, cudaq::state> magnus_cf4::getState() {
  return cudmIntHelp::getState(m_state, m_t);
}

namespace {

// Whether the closed-system, density-matrix unitary fast path applies.
bool canUseUnitaryFastPath(const SystemDynamics &system,
                           CuDensityMatState &state) {
  if (!state.is_density_matrix())
    return false;
  if (state.getBatchSize() != 1)
    return false;
  if (system.superOp.has_value())
    return false;
  for (const auto &ops : system.collapseOps)
    if (!ops.empty())
      return false;
  return true;
}

} // namespace

void magnus_cf4::integrate(double targetTime) {
  cudaq::dynamics::PerfMetricScopeTimer metricTimer("magnus_cf4::integrate");

  auto &state = *cudmIntHelp::asCudmState(*m_state);

  // Fall back to the general (open-system / state-vector) Magnus path when the
  // exact unitary propagator does not apply.
  if (!canUseUnitaryFastPath(m_system, state)) {
    cudaq::integrators::magnus_expansion fallback(
        magnus_expansion::default_num_taylor_terms, m_dt);
    cudaq::integrator_helper::init_system_dynamics(fallback, m_system,
                                                   m_schedule);
    auto [t0, currentState] = getState();
    fallback.setState(currentState, m_t);
    fallback.integrate(targetTime);
    auto [tFinal, finalState] = fallback.getState();
    auto *finalCudm = cudmIntHelp::asCudmState(finalState);
    m_state = std::make_shared<cudaq::state>(
        CuDensityMatState::clone(*finalCudm).release());
    m_t = tFinal;
    return;
  }

  auto *context = cudaq::dynamics::Context::getCurrentContext();
  cublasHandle_t cublas = context->getCublasHandle();

  const auto tensor = state.getTensor();
  const int dim = static_cast<int>(tensor.extents[0]);
  m_impl->ensureResources(dim, cublas);

  // Precompute the dimension map (degree index -> extent) for to_matrix.
  std::unordered_map<std::size_t, std::int64_t> dims;
  for (std::size_t i = 0; i < m_system.modeExtents.size(); ++i)
    dims[i] = m_system.modeExtents[i];

  // CF4 (s=2) Gauss-Legendre nodes and commutator-free weights.
  const double sqrt3 = std::sqrt(3.0);
  const double c1 = 0.5 - sqrt3 / 6.0;
  const double c2 = 0.5 + sqrt3 / 6.0;
  const double alpha1 = 0.25 + sqrt3 / 6.0;
  const double alpha2 = 0.25 - sqrt3 / 6.0;

  const std::size_t nn = static_cast<std::size_t>(dim) * dim;

  // Dense Hamiltonian at time t, flattened column-major to match the device
  // storage order used by CuDensityMatState.
  auto denseHamiltonianColumnMajor = [&](double t) {
    auto params = cudmIntHelp::scheduleParamsAt(m_schedule, t);
    const auto H = m_system.hamiltonian.front().to_matrix(dims, params);
    std::vector<std::complex<double>> flat;
    flat.reserve(nn);
    for (std::size_t col = 0; col < H.cols(); ++col)
      for (std::size_t row = 0; row < H.rows(); ++row)
        flat.push_back(H[{row, col}]);
    return flat;
  };

  // Weighted combination w1*H1 + w2*H2 (both already column-major).
  auto combine = [&](const std::vector<std::complex<double>> &H1,
                     const std::vector<std::complex<double>> &H2, double w1,
                     double w2) {
    std::vector<std::complex<double>> out(nn);
    for (std::size_t i = 0; i < nn; ++i)
      out[i] = w1 * H1[i] + w2 * H2[i];
    return out;
  };

  // Apply exp(-i * step * X) to the density matrix rho (in place), using the
  // propagator cache. X is the column-major exponent (a Hermitian matrix).
  auto applyExpFactor = [&](const std::vector<std::complex<double>> &X,
                            double step, cuDoubleComplex *dRho) {
    const std::size_t signature = hashMatrixBuffer(X);
    const std::size_t key = m_impl->cache.make_key(signature, step);
    auto *entry = m_impl->cache.get(key);
    if (!entry) {
      ++m_stats.cache_misses;
      entry = &m_impl->cache.insert(key, static_cast<std::size_t>(dim));
      m_impl->dH.copy_from_host(
          reinterpret_cast<const cuDoubleComplex *>(X.data()), nn);
      // Scale in place: dH <- (-i * step) * X.
      const cuDoubleComplex scale = make_cuDoubleComplex(0.0, -step);
      CUBLAS_CHECK(cublasZscal(cublas, static_cast<int>(nn), &scale,
                               m_impl->dH.get(), 1));
      compute_matrix_exp(m_impl->dH.get(), entry->U.get(), dim,
                         m_impl->expWorkspace, cublas, m_impl->cusolver);
    } else {
      ++m_stats.cache_hits;
    }
    apply_unitary_to_density_matrix(entry->U.get(), dRho, dim,
                                    m_impl->applyWork.get(), cublas);
  };

  auto *dRho = static_cast<cuDoubleComplex *>(state.get_device_pointer());

  while (m_t < targetTime) {
    const double step = cudmIntHelp::computeStepSize(m_t, targetTime, m_dt);

    const auto H1 = denseHamiltonianColumnMajor(m_t + c1 * step);
    const auto H2 = denseHamiltonianColumnMajor(m_t + c2 * step);
    const auto expA = combine(H1, H2, alpha1, alpha2);
    const auto expB = combine(H1, H2, alpha2, alpha1);

    // U = exp(-i*step*expA) * exp(-i*step*expB); applying expB first, then
    // expA, yields rho <- U rho U^dagger.
    applyExpFactor(expB, step, dRho);
    applyExpFactor(expA, step, dRho);

    ++m_stats.steps;
    m_t += step;
  }
}

} // namespace cudaq::integrators
