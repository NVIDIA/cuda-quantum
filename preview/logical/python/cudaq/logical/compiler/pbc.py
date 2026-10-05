# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Device-free P0 Pauli-based-computation normalization."""

from __future__ import annotations

from .build import Build, EvidenceRecord
from .pipeline import Pipeline, pipelines


def _to_pbc(source, *, pipeline: Pipeline) -> Build:
    if pipeline != pipelines.pbc():
        raise ValueError("PBC normalization requires the canonical "
                         "cudaq.logical.compiler.pipelines.pbc() "
                         "contract")
    if not isinstance(source, Build):
        raise TypeError(
            "cudaq.logical.compiler.to_pbc expects a synthesized P0 Build")
    if source.profile != "p0":
        raise ValueError(
            "cudaq.logical.compiler.to_pbc requires a synthesized P0 Build, "
            f"got {source.profile!r}")
    synthesis = source.synthesis
    if synthesis is None or synthesis.gate_set != "clifford_t":
        raise ValueError(
            "cudaq.logical.compiler.to_pbc requires "
            "cudaq.logical.compiler.synthesize(..., "
            "gate_set=cudaq.logical.compiler.gate_sets.clifford_t) first")

    from cudaq.mlir._mlir_libs import _qlxRuntime as runtime

    module = runtime.clone_module(source.module)
    if not runtime.verify_clifford_t_module(module):
        raise ValueError(
            "cudaq.logical.compiler.to_pbc requires positive Clifford+T input; "
            "call cudaq.logical.compiler.synthesize(..., "
            "gate_set=cudaq.logical.compiler.gate_sets.clifford_t) first")
    runtime.lower_to_pbc_module(module)
    if not runtime.verify_pbc_module(module):
        raise RuntimeError("native PBC lowering produced invalid normal form")

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
                kind="pauli_based_computation_normalization",
                producer="cudaq-logical-native-pbc@0.3",
                result="pass",
                obligations=(
                    "positive-clifford-t-input",
                    "signed-quarter-turn-product-rotations",
                    "pairwise-commuting-terminal-measurements",
                ),
                assumptions=(
                    "clifford-frame-absorbed",
                    "device-free-p0-transform",
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


def to_pbc(source: Build) -> Build:
    """Normalize a synthesized Clifford+T P0 build into PBC form.

    The returned immutable P0 build contains one signed pi/4 Pauli-product
    rotation per synthesized T gate followed by pairwise-commuting terminal
    Pauli-product measurements. Clifford operations are absorbed into the
    tracked Pauli frame. This transform remains code- and device-independent.
    """

    return _to_pbc(source, pipeline=pipelines.pbc())


__all__ = ["to_pbc"]
