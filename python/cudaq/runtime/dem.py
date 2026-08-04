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

# Helpers called from the C++ binding (`py_dem.cpp`) for `scipy` interoperability.


def _make_csr(rows, num_cols):
    import numpy as np
    import scipy.sparse as sp
    row_idx = [r for r, ms in enumerate(rows) for _ in ms]
    col_idx = [m for ms in rows for m in ms]
    return sp.csr_matrix(
        (np.ones(len(row_idx), dtype=np.uint8), (row_idx, col_idx)),
        shape=(len(rows), num_cols),
    )


def _csr_to_rows(mat):
    mat = mat.tocsr()
    return [
        list(mat.indices[mat.indptr[i]:mat.indptr[i + 1]])
        for i in range(mat.shape[0])
    ]


DEMResult = cudaq_runtime.DEMResult

# ---------------------------------------------------------------------------


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

    Returns a :class:`DEMResult` carrying the DEM text, count fields, and
    (by default) the measurement matrices.

    ``str(result)`` returns the DEM text so existing print calls are unchanged.
    ``stim.DetectorErrorModel(result.dem)`` replaces the previous
    ``stim.DetectorErrorModel(result)``.

    Args:
      kernel: The kernel to analyze.
      *arguments: Forwarded to the kernel.
      noise_model: Optional noise model.
      decompose_errors (bool): Default ``False``.
      fold_loops (bool): Default ``False``.
      allow_gauge_detectors (bool): Default ``False``.
      approximate_disjoint_errors_threshold (float): Default ``0.0``.
      ignore_decomposition_failures (bool): Default ``False``.
      block_decomposition_from_introducing_remnant_edges (bool): Default ``False``.
      return_measurement_matrices (bool): When ``False``, ``m2d_matrix`` /
          ``m2o_matrix`` will be ``None``. Default ``True``.

    Returns:
      :class:`DEMResult`
    """
    _detail_check_conditionals_on_measure(kernel)

    unknown = set(dem_kwargs) - _VALID_DEM_OPTION_KEYS
    if unknown:
        raise ValueError(
            f"dem_from_kernel: unknown keyword argument(s) {sorted(unknown)}. "
            f"Valid options: {sorted(_VALID_DEM_OPTION_KEYS)}")

    dem_kwargs.setdefault("return_measurement_matrices", True)

    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    policy = cudaq_runtime.DemPolicy(decorator.uniqName, noise_model,
                                     dem_kwargs)
    return cudaq_runtime.launch_dem(policy, lambda: decorator(*args))
