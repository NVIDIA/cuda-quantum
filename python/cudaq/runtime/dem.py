# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace
from .utils import _kernel_has_conditionals_on_measure

_VALID_DEM_OPTION_KEYS = frozenset({
    "decompose_errors",
    "fold_loops",
    "allow_gauge_detectors",
    "approximate_disjoint_errors_threshold",
    "ignore_decomposition_failures",
    "block_decomposition_from_introducing_remnant_edges",
    "return_measurement_matrices",
})


def _detail_check_conditionals_on_measure(kernel):
    if not _kernel_has_conditionals_on_measure(kernel):
        return
    kernel_name = kernel.name if hasattr(kernel, 'name') else '<unknown>'
    raise RuntimeError(
        f"`cudaq::dem_from_kernel`: kernel '{kernel_name}' branches on "
        "a measurement result. DEM analysis not supported.")


@trace.traced
def dem_from_kernel(kernel, *args, noise_model=None, **dem_kwargs):
    """Generate a detector error model (DEM) from a CUDA-Q kernel.

    Runs `kernel` under the internal `"dem"` execution context, captures
    the recorded circuit from the backend, and returns Stim's standard
    `.dem` text via `stim::DetectorErrorModel::str()`. The active CUDA-Q
    target is unaffected; the analysis simulator is an internal,
    thread-local override.

    Args:
      kernel (:class:`Kernel`): The :class:`Kernel` to analyze.
      *arguments: Concrete argument values forwarded to the kernel invocation.
      noise_model (:class:`NoiseModel`, optional): Noise model layered on
          top of any `apply_noise` ops already present in the kernel.
      decompose_errors (bool, optional): Decompose hyper-edge error
          mechanisms into pairs of two-detector edges. Default ``False``.
      fold_loops (bool, optional): Fold loop bodies in the circuit for a
          more compact DEM. Default ``False``.
      allow_gauge_detectors (bool, optional): Allow detectors whose parity
          is not determined by the circuit. Default ``False``.
      approximate_disjoint_errors_threshold (float, optional): Threshold
          for approximating disjoint-error products; set to ``0`` to
          disable. Default ``0.0``.
      ignore_decomposition_failures (bool, optional): Skip error mechanisms
          that cannot be decomposed instead of raising an exception.
          Default ``False``.
      block_decomposition_from_introducing_remnant_edges (bool, optional):
          Prevent the decomposer from introducing remnant edges.
          Default ``False``.
      return_measurement_matrices (bool, optional): When True, also return
          the sparse measurements-to-detectors (m2d) and
          measurements-to-observables (m2o) matrices alongside the DEM text.
          Default ``False``.

    Returns:
      If `return_measurement_matrices` is False (default): a UTF-8 string in
      Stim's standard `.dem` file format. Consumers that need a structured DEM
      can parse it with `stim.DetectorErrorModel(text)`.

      If `return_measurement_matrices` is True: a tuple
      ``(dem_text, m2d, m2o)`` where both matrices are
      ``scipy.sparse.csr_matrix`` with binary entries.
      ``m2d`` has shape ``(num_detectors, num_measurements)``: entry
      ``m2d[d, m] == 1`` means measurement ``m`` contributes to detector ``d``.
      ``m2o`` has shape ``(num_observables, num_measurements)``: entry
      ``m2o[k, m] == 1`` means measurement ``m`` contributes to observable ``k``.
      Measurement indices are chronological.
    """
    _detail_check_conditionals_on_measure(kernel)

    unknown = set(dem_kwargs) - _VALID_DEM_OPTION_KEYS
    if unknown:
        raise ValueError(
            f"dem_from_kernel: unknown keyword argument(s) {sorted(unknown)}. "
            f"Valid options: {sorted(_VALID_DEM_OPTION_KEYS)}")

    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    processedArgs, module = decorator.prepare_call(*args)
    result = cudaq_runtime.dem_from_kernel_impl(decorator.uniqName, module,
                                                noise_model, dem_kwargs,
                                                *processedArgs)

    if not dem_kwargs.get("return_measurement_matrices", False):
        return result

    import numpy as np
    import scipy.sparse as sp

    dem_text, num_measurements, det_rows, obs_rows = result

    def _make_csr(rows, num_cols):
        row_idx = [r for r, ms in enumerate(rows) for _ in ms]
        col_idx = [m for ms in rows for m in ms]
        return sp.csr_matrix(
            (np.ones(len(row_idx), dtype=np.uint8), (row_idx, col_idx)),
            shape=(len(rows), num_cols),
        )

    m2d = _make_csr(det_rows, num_measurements)
    m2o = _make_csr(obs_rows, num_measurements)
    return dem_text, m2d, m2o
