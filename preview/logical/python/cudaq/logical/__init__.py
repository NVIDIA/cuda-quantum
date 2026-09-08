# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""CUDA-Q Logical fault-tolerant quantum-computing programming model.

The package root is a curated authoring facade. Common decorators, circuit
operations, and lifecycle verbs are available directly; complete semantic
domains remain discoverable through their owning namespaces. Historical flat
nouns remain available through exact lazy aliases but are not advertised by
:data:`__all__`.
"""

from __future__ import annotations

import warnings as _warnings
from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    distribution as _distribution,
)
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path


def _require_cudaq_runtime() -> None:
    """Fail fast with actionable guidance when the CUDA-Q runtime is absent.

    The base cudaq-logical wheel is runtime-agnostic: the CUDA-Q runtime
    arrives through the ``cu12``/``cu13`` extras or the ``cudaq``
    metapackage. It provides the ``cudaq`` import package
    (``cudaq/__init__.py``); without it, ``cudaq`` resolves as a bare
    namespace holding only this subpackage and the compiled modules below
    fail with inscrutable errors.
    """
    spec = _find_spec("cudaq")
    if spec is None or spec.origin is None:
        raise ImportError(
            "cudaq.logical requires the CUDA-Q runtime, which is not "
            "installed. Install it with "
            'pip install "cudaq-logical[cu13]" (CUDA 13) or '
            'pip install "cudaq-logical[cu12]" (CUDA 12), or with '
            "pip install cudaq."
        )


_require_cudaq_runtime()

_warnings.warn(
    "cudaq-logical is in preview. Its APIs, behavior, and documentation "
    "may change substantially in upcoming versions.",
    stacklevel=2,
)

from . import (  # noqa: E402
    analysis, algebra, architecture, codes, compiler, devices, errors, estimate,
    experiments, gadgets, ops, programs, protocols, qec, stages, std, targets,
    types,
)
from .compiler import compile, materialize  # noqa: E402
from .programs import (
    abort_on,
    condition_results,
    objective,
    program,
    require,
)  # noqa: E402
from .codes import code  # noqa: E402
from .gadgets import gadget, patch  # noqa: E402
from .protocols import protocol  # noqa: E402
from .architecture import machine  # noqa: E402
from .ops import (  # noqa: E402
    allocate, allocate_patch, barrier, ccz, cx, cz, discard, extract_syndrome,
    h, idle, measure, measure_x, measure_z, mpp, mz, prepare, prepare_plus,
    prepare_zero, pack_resource, postselect, request_many, reset, rx, ry, rz, s,
    sdg, t, tdg, unpack_resource, x, y, z,
)
from .targets import emit  # noqa: E402

try:
    _installed = _distribution("cudaq-logical")
    _package_root = _Path(__file__).resolve().parent.parent.parent
    _metadata_root = _Path(_installed.locate_file("")).resolve()
    if _package_root != _metadata_root:
        raise _PackageNotFoundError
    __version__ = _installed.version
except _PackageNotFoundError:
    # Source-tree and staged-build imports do not necessarily have wheel
    # metadata beside them.  Keep their development identity explicit while
    # installed packages always report the distribution version.
    __version__ = "0.0.0.dev0"
finally:
    globals().pop("_installed", None)
    globals().pop("_package_root", None)
    globals().pop("_metadata_root", None)

__all__ = [
    "program",
    "objective",
    "code",
    "gadget",
    "patch",
    "protocol",
    "machine",
    "require",
    "condition_results",
    "abort_on",
    "compile",
    "materialize",
    "estimate",
    "emit",
    "allocate",
    "allocate_patch",
    "prepare",
    "prepare_plus",
    "prepare_zero",
    "request_many",
    "unpack_resource",
    "pack_resource",
    "postselect",
    "reset",
    "discard",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "cx",
    "cz",
    "ccz",
    "measure",
    "measure_x",
    "measure_z",
    "mpp",
    "mz",
    "idle",
    "barrier",
    "extract_syndrome",
    "ops",
    "types",
    "stages",
    "programs",
    "algebra",
    "qec",
    "architecture",
    "compiler",
    "analysis",
    "targets",
    "errors",
    "codes",
    "gadgets",
    "protocols",
    "devices",
    "experiments",
    "std",
]


def __dir__():
    return sorted(__all__)
