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

# ---------------------------------------------------------------------------
# Attach Python-typed members to DEMResult.
#
# DEMResult is the bound C++ cudaq::dem_result. Members that require Python
# types (scipy matrices) or Python protocols (__str__, __repr__, classmethod)
# are attached here so that scipy never appears in the C++ binding layer, and
# DEMResult stays a single type identical in shape to SampleResult.
#
# The setup is deferred to first use via _get_dem_result_class() to avoid
# a circular-import issue: the extension module (which registers DEMResult in
# cudaq_runtime) may trigger cudaq/__init__.py mid-init, before
# `bindDemFromKernel` has run. Accessing `cudaq_runtime.DEMResult` at module
# level would fail then.
# ---------------------------------------------------------------------------

_DEMResult = None


def _make_csr(rows, num_cols):
    import numpy as np
    import scipy.sparse as sp
    row_idx = [r for r, ms in enumerate(rows) for _ in ms]
    col_idx = [m for ms in rows for m in ms]
    return sp.csr_matrix(
        (np.ones(len(row_idx), dtype=np.uint8), (row_idx, col_idx)),
        shape=(len(rows), num_cols),
    )


def _m2d_matrix(self):
    if not self.matrices_computed:
        return None
    return _make_csr(self.m2d, self.num_measurements)


def _m2o_matrix(self):
    if not self.matrices_computed:
        return None
    return _make_csr(self.m2o, self.num_measurements)


@classmethod
def _from_matrices(cls,
                   dem,
                   m2d_csr,
                   m2o_csr,
                   *,
                   num_detectors=0,
                   num_observables=0,
                   num_measurements=0,
                   annotations=None):
    """Build a DEMResult from scipy CSR matrices."""

    def _csr_to_rows(mat):
        mat = mat.tocsr()
        return [
            list(mat.indices[mat.indptr[i]:mat.indptr[i + 1]])
            for i in range(mat.shape[0])
        ]

    return cls(
        dem,
        m2d=_csr_to_rows(m2d_csr),
        m2o=_csr_to_rows(m2o_csr),
        num_detectors=num_detectors,
        num_observables=num_observables,
        num_measurements=num_measurements,
        annotations=annotations or {},
    )


def _dem_result_str(self):
    return self.dem


def _dem_result_repr(self):
    return (f"DEMResult(detectors={self.num_detectors}, "
            f"observables={self.num_observables}, "
            f"measurements={self.num_measurements})")


def _get_dem_result_class():
    """Return the DEMResult class, attaching Python-typed members on first call."""
    global _DEMResult
    if _DEMResult is not None:
        return _DEMResult
    cls = cudaq_runtime.DEMResult
    cls.m2d_matrix = property(_m2d_matrix,
                              doc="scipy CSR matrix (num_detectors × "
                              "num_measurements), or None when matrices "
                              "were not requested.")
    cls.m2o_matrix = property(_m2o_matrix,
                              doc="scipy CSR matrix (num_observables × "
                              "num_measurements), or None when matrices "
                              "were not requested.")
    cls.from_matrices = _from_matrices
    cls.__str__ = _dem_result_str
    cls.__repr__ = _dem_result_repr
    _DEMResult = cls
    return cls


def __getattr__(name):
    """Lazily resolve DEMResult so it is safe to import at extension-init time."""
    if name == "DEMResult":
        return _get_dem_result_class()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    # Ensure Python-typed members are attached before the first result arrives.
    _get_dem_result_class()

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
