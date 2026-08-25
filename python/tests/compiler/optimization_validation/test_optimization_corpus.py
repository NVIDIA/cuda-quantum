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

# A few arbitrary, fixed seeds to exercise the generator over.
_SAMPLE_SEEDS = (184467, 184468, 184469)


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
    first = {
        p.name: p.read_text()
        for p in corpus.write_corpus(tmp_path / "a", seeds)
    }
    second = {
        p.name: p.read_text()
        for p in corpus.write_corpus(tmp_path / "b", seeds)
    }
    assert first == second


# The corpus must actually be usable by the validator: every generated module is
# a valid, in-domain bounded-unitary circuit.
def test_generated_modules_are_valid_and_in_domain():
    ctx = _context()
    for seed in _SAMPLE_SEEDS:
        text = corpus.generate_module_text(seed)
        module = Module.parse(text, ctx)
        assert module.operation.verify()
        pf = cudaq_runtime.preflight_bounded_unitary(module, 14)
        assert pf["supported"], pf["rejections"]


# Canonical corpus: a fixed, named, in-repo set of reference circuits.
def test_canonical_names_are_stable_and_unique():
    names = corpus.canonical_names()
    assert names == ("bell_pair", "ghz_3", "inverse_pair_h", "mergeable_rz",
                     "t_ladder", "clifford_mix")
    assert len(names) == len(set(names))


def test_canonical_modules_are_valid_and_in_domain():
    ctx = _context()
    for name in corpus.canonical_names():
        text = corpus.canonical_module_text(name)
        module = Module.parse(text, ctx)
        assert module.operation.verify(), name
        pf = cudaq_runtime.preflight_bounded_unitary(module, 14)
        assert pf["supported"], (name, pf["rejections"])


def test_canonical_module_text_is_reproducible():
    for name in corpus.canonical_names():
        assert corpus.canonical_module_text(
            name) == corpus.canonical_module_text(name)


def test_canonical_module_text_rejects_unknown():
    with pytest.raises(ValueError):
        corpus.canonical_module_text("no_such_circuit")


def test_write_canonical_corpus_writes_one_file_per_circuit(tmp_path):
    paths = corpus.write_canonical_corpus(tmp_path)
    assert [p.name for p in paths
           ] == [f"{n}.qke" for n in corpus.canonical_names()]
    for name, path in zip(corpus.canonical_names(), paths):
        assert path.read_text() == corpus.canonical_module_text(name)


# Clifford GHZ: the size-parameterized input for the scalable tableau oracle.
def test_clifford_ghz_is_reproducible_and_sized():
    text = corpus.clifford_ghz_module_text(20)
    assert text == corpus.clifford_ghz_module_text(20)
    assert "!quake.veq<20>" in text
    # h on qubit 0 plus a full 19-link CX chain.
    assert text.count("quake.h") == 1
    assert text.count("quake.x [") == 19


def test_clifford_ghz_chain_length_truncates_the_chain():
    assert corpus.clifford_ghz_module_text(
        20, chain_length=18).count("quake.x [") == 18


def test_clifford_ghz_is_in_the_clifford_domain_past_the_dense_bound():
    """The whole point: no qubit bound, so 20 qubits is in domain here and not
    in the dense one."""
    ctx = _context()
    module = Module.parse(corpus.clifford_ghz_module_text(20), ctx)
    assert module.operation.verify()
    clifford = cudaq_runtime.preflight_clifford(module)
    assert clifford["supported"], clifford["rejections"]
    assert clifford["max_qubits"] == 20
    dense = cudaq_runtime.preflight_bounded_unitary(module, 14)
    assert not dense["supported"]
