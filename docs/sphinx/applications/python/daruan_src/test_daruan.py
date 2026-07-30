# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

# Allow `pytest path/to/daruan_src` without installing a package.
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import daruan  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _cpu_target():
    daruan.select_target("qpp-cpu")


def test_param_layout_size():
    assert daruan.REPS == 3
    assert daruan.N_PARAMS == 14
    assert daruan.initial_params(0).shape == (14,)


def test_initial_params_geometric_scales():
    params = daruan.initial_params(1)
    for ell in range(daruan.REPS):
        assert params[4 * ell + 2] == pytest.approx(float(2**ell))
        assert params[4 * ell + 3] == pytest.approx(0.0)


def test_activation_bounded_and_finite():
    params = daruan.initial_params(0)
    for x in (-1.0, -0.3, 0.0, 0.5, 1.0):
        val = daruan.activation(x, params)
        assert np.isfinite(val)
        assert -1.0 - 1e-9 <= val <= 1.0 + 1e-9


def test_activation_deterministic():
    params = daruan.initial_params(2)
    a = daruan.activation(0.25, params)
    b = daruan.activation(0.25, params)
    assert a == pytest.approx(b, abs=1e-12)


def test_activations_vectorized_matches_scalar():
    params = daruan.initial_params(0)
    xs = np.array([-0.5, 0.0, 0.75])
    got = daruan.activations(xs, params)
    want = np.array([daruan.activation(float(x), params) for x in xs])
    assert np.allclose(got, want)


def test_target_j0_like_limit_at_zero():
    assert daruan.target_j0_like(np.array([0.0]))[0] == pytest.approx(1.0)


def test_make_dataset_shapes_and_labels():
    xs, ys = daruan.make_dataset("sin4", n_train=16, seed=7)
    assert xs.shape == (16,)
    assert ys.shape == (16,)
    assert np.allclose(ys, np.sin(4.0 * xs))


def test_select_target_rejects_unknown():
    with pytest.raises(ValueError, match="not available"):
        daruan.select_target("not-a-real-cudaq-target")


def test_short_training_reduces_mse():
    xs, ys = daruan.make_dataset("sin4", n_train=12, seed=0)
    params0 = daruan.initial_params(0)
    mse0 = daruan.mse_loss(params0, xs, ys)
    _, mse1, history = daruan.train(xs,
                                    ys,
                                    seed=0,
                                    maxiter=20,
                                    verbose=False)
    assert len(history) >= 2
    assert mse1 < mse0
    assert mse1 < 0.55
