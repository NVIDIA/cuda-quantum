# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Device-free P0 hybrid Clifford-frame normalization."""

from __future__ import annotations

from .build import Build, EvidenceRecord
from .pipeline import Pipeline, pipelines


def _absorb_clifford_frame(source, *, pipeline: Pipeline) -> Build:
    if pipeline != pipelines.clifford_frame():
        raise ValueError("Clifford-frame normalization requires the canonical "
                         "cudaq.logical.compiler.pipelines.clifford_frame() "
                         "contract")
    if not isinstance(source, Build):
        raise TypeError(
            "cudaq.logical.compiler.absorb_clifford_frame expects a verified "
            "P0 Build")
    if source.profile != "p0":
        raise ValueError(
            "cudaq.logical.compiler.absorb_clifford_frame requires a P0 Build, "
            f"got {source.profile!r}")

    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = source._fresh_module()
    runtime.absorb_clifford_frame_module(module)
    if not runtime.verify_clifford_frame_module(module):
        raise RuntimeError(
            "native Clifford-frame normalization produced invalid normal form")

    return Build(
        context=module.context,
        module=module,
        root=source.root,
        profile="p0",
        facets=source.facets,
        pipeline=pipeline,
        evidence=(
            *source.evidence,
            EvidenceRecord(
                kind="clifford_frame_normalization",
                producer="cudaq-logical-native-clifford-frame@0.3",
                result="pass",
                obligations=(
                    "exact-clifford-frame-absorption",
                    "retained-pauli-conjugation",
                    "folded-repeat-frame-closure",
                    "p0-machine-independence",
                ),
                assumptions=(
                    "arbitrary-rotation-realization-remains-p2-owned",
                    "terminal-frame-discarded",
                    "source-build-content-sha256="
                    f"{source.content_sha256}",
                ),
            ),
        ),
        value_groups={
            name: len(group) for name, group in source.values._groups.items()
        },
        experiment=source.experiment,
        source_modules=source.source_modules,
    )


def absorb_clifford_frame(source: Build) -> Build:
    """Absorb exact Cliffords while retaining arbitrary P0 rotations.

    The result is a new immutable, device-free P0 build. Exact Clifford
    actions update a symbolic frame; each surviving Pauli-product rotation is
    conjugated by that frame without changing its angle or requested
    precision. Folded repeats stay folded and must close their frame at every
    iteration boundary.
    """

    return _absorb_clifford_frame(
        source,
        pipeline=pipelines.clifford_frame(),
    )


__all__ = ["absorb_clifford_frame"]
