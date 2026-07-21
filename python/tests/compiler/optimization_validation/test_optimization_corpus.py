# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Tests for the reproducible bounded-unitary corpus generator."""

import pytest

from cudaq.mlir.ir import Context, Module
from cudaq.mlir._mlir_libs._quakeDialects import (cudaq_runtime,
                                                  register_all_dialects, quake,
                                                  cc)

from cudaq._compiler import optimization_corpus as corpus


def _context() -> Context:
    ctx = Context()
    register_all_dialects(ctx)
    quake.register_dialect(context=ctx)
    cc.register_dialect(context=ctx)
    return ctx


# Reproducibility
def test_generation_is_byte_reproducible():
    a = corpus.generate_module_text(42, num_qubits=3, length=8)
    b = corpus.generate_module_text(42, num_qubits=3, length=8)
    assert a == b


def test_different_seeds_differ():
    assert corpus.generate_module_text(1) != corpus.generate_module_text(2)


def test_write_corpus_is_reproducible(tmp_path):
    seeds = (1, 2, 3)
    first = {p.name: p.read_text() for p in corpus.write_corpus(tmp_path / "a", seeds)}
    second = {p.name: p.read_text() for p in corpus.write_corpus(tmp_path / "b", seeds)}
    assert first == second


# Provenance stamping
def test_provenance_header_records_version_and_params():
    text = corpus.generate_module_text(7, num_qubits=2, length=5)
    head = text.splitlines()[:2]
    assert f"v{corpus.CORPUS_GENERATOR_VERSION}" in head[0]
    assert "seed=7" in head[1]
    assert "num_qubits=2" in head[1]
    assert "length=5" in head[1]


# Pinned seed sets
def test_seed_sets_are_pinned_and_sized():
    sizes = {p: len(s) for p, s in corpus.CORPUS_SEED_SETS.items()}
    assert sizes == {
        "single-reproducer": 1,
        "smoke": 3,
        "quick": 8,
        "ci": 24,
        "full": 64,
    }
    # Depth ordering holds.
    assert (len(corpus.CORPUS_SEED_SETS["smoke"])
            < len(corpus.CORPUS_SEED_SETS["quick"])
            < len(corpus.CORPUS_SEED_SETS["ci"])
            < len(corpus.CORPUS_SEED_SETS["full"]))


def test_seeds_for_preset_matches_table():
    assert corpus.seeds_for_preset("smoke") == corpus.CORPUS_SEED_SETS["smoke"]


def test_seeds_for_preset_rejects_unknown():
    with pytest.raises(ValueError):
        corpus.seeds_for_preset("enormous")


def test_corpus_for_preset_writes_the_seed_set(tmp_path):
    paths = corpus.corpus_for_preset(tmp_path, "smoke")
    assert len(paths) == len(corpus.CORPUS_SEED_SETS["smoke"])
    for seed, path in zip(corpus.CORPUS_SEED_SETS["smoke"], paths):
        assert path.name == f"generated_{seed}.qke"
        assert path.read_text() == corpus.generate_module_text(seed)


# The corpus must actually be usable by the validator: every generated module is
# a valid, in-domain bounded-unitary circuit.
def test_generated_modules_are_valid_and_in_domain():
    ctx = _context()
    for seed in corpus.CORPUS_SEED_SETS["smoke"]:
        text = corpus.generate_module_text(seed)
        module = Module.parse(text, ctx)
        assert module.operation.verify()
        pf = cudaq_runtime.preflight_bounded_unitary(module, 14)
        assert pf["supported"], pf["rejections"]
