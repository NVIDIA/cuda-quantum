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

from importlib.metadata import (
    PackageNotFoundError as _PackageNotFoundError,
    distribution as _distribution,
)
from importlib.util import find_spec as _find_spec
from pathlib import Path as _Path

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
    __version__ = "0.3.0.dev0"
finally:
    globals().pop("_installed", None)
    globals().pop("_package_root", None)
    globals().pop("_metadata_root", None)

# Fail loudly on a half-upgraded install: pip uninstall/upgrade of cuda-quantum
# can strand our files in cudaq/mlir/_mlir_libs next to a missing _mlir.so.
if _find_spec("cudaq.mlir._mlir_libs._qlx_ext") is None:
    raise ImportError(
        "cudaq.logical requires cudaq.mlir._mlir_libs._qlx_ext; the CUDA-Q "
        "runtime wheel may have been upgraded or uninstalled without "
        "reinstalling cudaq-logical"
    )

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
