# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import numpy as np
import scipy.sparse as sp

from cudaq.mlir._mlir_libs._quakeDialects import cudaq_runtime
from cudaq.kernel.kernel_decorator import (mk_decorator, isa_kernel_decorator)
from cudaq.util import trace
from .utils import _kernel_has_conditionals_on_measure


def _detail_check_conditionals_on_measure(kernel):
    if not _kernel_has_conditionals_on_measure(kernel):
        return
    kernel_name = kernel.name if hasattr(kernel, 'name') else '<unknown>'
    raise RuntimeError(
        f"`cudaq::dem_from_kernel`: kernel '{kernel_name}' branches on "
        "a measurement result. DEM analysis not supported.")


@trace.traced
def dem_from_kernel(kernel, *args, noise_model=None, return_m2d=False):
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
      return_m2d (bool, optional): When True, also return the sparse
          measurements-to-detectors (m2d) matrix alongside the DEM text.
          Defaults to False.

    Returns:
      If `return_m2d` is False (default): a UTF-8 string in Stim's standard
      `.dem` file format. Consumers that need a structured DEM can parse it
      with `stim.DetectorErrorModel(text)`.

      If `return_m2d` is True: a tuple ``(dem_text, m2d, m2o)`` where both
      matrices are ``scipy.sparse.csr_matrix`` with binary entries.
      ``m2d`` has shape ``(num_detectors, num_measurements)``: entry
      ``m2d[d, m] == 1`` means measurement ``m`` contributes to detector ``d``.
      ``m2o`` has shape ``(num_observables, num_measurements)``: entry
      ``m2o[k, m] == 1`` means measurement ``m`` contributes to observable ``k``.
      Measurement indices are chronological.
    """
    _detail_check_conditionals_on_measure(kernel)

    if isa_kernel_decorator(kernel):
        decorator = kernel
    else:
        decorator = mk_decorator(kernel)
    processedArgs, module = decorator.prepare_call(*args)
    result = cudaq_runtime.dem_from_kernel_impl(decorator.uniqName, module,
                                                noise_model, return_m2d,
                                                *processedArgs)

    if not return_m2d:
        return result

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
