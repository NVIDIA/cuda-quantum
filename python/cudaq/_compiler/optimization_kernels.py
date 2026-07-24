# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import cudaq

# A ``prepare`` pipeline that normalizes real frontend output into the
# straight-line, bounded-unitary domain: unroll statically-bounded ``cc.loop``s
# (from Python ``for range(...)``), fold the resulting constants, then move to
# reference semantics. Every kernel in this module is in-domain after it.
DOMAIN_PREPARE_PIPELINE = (
    "builtin.module(func.func(cc-loop-unroll,canonicalize,memtoreg))")


@cudaq.kernel
def bell():
    """Bell state"""
    q = cudaq.qvector(2)
    h(q[0])
    x.ctrl(q[0], q[1])


@cudaq.kernel
def ghz_linear():
    """4-qubit GHZ"""
    q = cudaq.qvector(4)
    h(q[0])
    for i in range(3):
        x.ctrl(q[i], q[i + 1])


@cudaq.kernel
def inverse_pairs():
    """Adjacent self-inverse pairs"""
    q = cudaq.qvector(3)
    for i in range(3):
        h(q[i])
        h(q[i])


@cudaq.kernel
def t_ladder():
    """Four T gates on one qubit"""
    q = cudaq.qvector(1)
    for _ in range(4):
        t(q[0])


@cudaq.kernel
def rotation_chain():
    """Mergeable z-rotations"""
    q = cudaq.qvector(2)
    for i in range(2):
        rz(0.5, q[i])
        rz(0.25, q[i])


@cudaq.kernel
def clifford_mix():
    """A small mixed Clifford circuit"""
    q = cudaq.qvector(2)
    h(q[0])
    s(q[1])
    x.ctrl(q[0], q[1])
    x(q[1])
    z(q[0])


_REAL_KERNELS = (
    ("bell", bell),
    ("ghz_linear", ghz_linear),
    ("inverse_pairs", inverse_pairs),
    ("t_ladder", t_ladder),
    ("rotation_chain", rotation_chain),
    ("clifford_mix", clifford_mix),
)
_REAL_BY_NAME = dict(_REAL_KERNELS)


def real_kernel_names() -> tuple:
    return tuple(name for name, _ in _REAL_KERNELS)


def real_kernel_module_text(name: str) -> str:
    """Return the frontend-lowered Quake IR text for real kernel ``name``.

    The text is produced by the CUDA-Q Python frontend, so it carries the full
    realistic module shape. It is not byte-reproducible. The mangled kernel
    name has a per-process unique suffix. Raises ``ValueError`` for an unknown
    name so a typo can never silently yield an empty corpus.
    """
    try:
        kernel = _REAL_BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown real kernel '{name}'; "
                         f"known: {list(real_kernel_names())}")
    return str(kernel)


def write_real_corpus(directory) -> list:
    """Write one ``<name>.qke`` per real kernel into ``directory``.

    Returns the list of written paths, ordered as :data:`real_kernel_names`.
    """
    from pathlib import Path

    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in real_kernel_names():
        path = out / f"{name}.qke"
        path.write_text(real_kernel_module_text(name))
        paths.append(path)
    return paths
