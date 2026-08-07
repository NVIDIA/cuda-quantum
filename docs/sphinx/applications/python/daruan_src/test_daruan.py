"""Unit tests for DARUAN."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import daruan  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def _cpu_target():
    daruan.select_target("qpp-cpu")


def test_param_count():
    assert daruan.N_PARAMS == 14


def test_activation_bounded():
    val = daruan.activation(0.3, daruan.initial_params(0))
    assert np.isfinite(val)
    assert -1.0 <= val <= 1.0


def test_activation_deterministic():
    params = daruan.initial_params(1)
    assert daruan.activation(0.2, params) == pytest.approx(
        daruan.activation(0.2, params), abs=1e-12)


def test_training_reduces_mse():
    rng = np.random.default_rng(0)
    xs = rng.uniform(-1.0, 1.0, size=12)
    ys = np.sin(4.0 * xs)
    mse0 = daruan.mse_loss(daruan.initial_params(0), xs, ys)
    _, mse1, history = daruan.train(xs, ys, seed=0, maxiter=25)
    assert len(history) >= 2
    assert mse1 < mse0
