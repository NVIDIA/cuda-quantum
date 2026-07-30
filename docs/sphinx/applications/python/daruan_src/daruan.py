# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""
DARUAN: DatA Re-Uploading ActivatioN as a CUDA-Q kernel.

Implements the single-qubit QVAF instantiation from
Jiang et al., arXiv:2509.14026 (eqs. 4.6, A.36).
"""

from __future__ import annotations

import argparse
import math
from typing import Callable

import cudaq
import numpy as np
from cudaq import spin
from scipy.optimize import minimize

REPS = 3
N_PARAMS = 4 * REPS + 2  # 14
RY_MIXER = math.pi / 3.0


def select_target(name: str | None = None) -> str:
    """Prefer qpp-cpu; allow nvidia when present."""
    available = {t.name for t in cudaq.get_targets()}
    if name:
        if name not in available:
            raise ValueError(
                f"Target {name!r} not available. Have: {sorted(available)}")
        cudaq.set_target(name)
        return name
    for candidate in ("qpp-cpu", "density-matrix-cpu"):
        if candidate in available:
            cudaq.set_target(candidate)
            return candidate
    return cudaq.get_target().name


@cudaq.kernel
def daruan_kernel(x: float, params: list[float]):
    """Single-qubit DARUAN with REPS=3 re-uploads."""
    q = cudaq.qubit()
    rz(params[0], q)
    ry(math.pi / 3.0, q)
    rz(params[1], q)
    rz(params[2] * x + params[3], q)
    rz(params[4], q)
    ry(math.pi / 3.0, q)
    rz(params[5], q)
    rz(params[6] * x + params[7], q)
    rz(params[8], q)
    ry(math.pi / 3.0, q)
    rz(params[9], q)
    rz(params[10] * x + params[11], q)
    rz(params[12], q)
    ry(math.pi / 3.0, q)
    rz(params[13], q)


def activation(x: float, params: list[float] | np.ndarray) -> float:
    """Evaluate φ(x) = ⟨Z⟩ for the DARUAN circuit."""
    p = list(float(v) for v in params)
    return float(
        cudaq.observe(daruan_kernel, spin.z(0), float(x), p).expectation())


def activations(xs: np.ndarray, params: list[float] | np.ndarray) -> np.ndarray:
    p = list(float(v) for v in params)
    return np.array([activation(float(x), p) for x in xs], dtype=float)


def target_sin4(x: np.ndarray) -> np.ndarray:
    return np.sin(4.0 * x)


def target_j0_like(x: np.ndarray) -> np.ndarray:
    """Paper-style demo: sin(20x)/(20x), with limit 1 at x=0."""
    out = np.empty_like(x, dtype=float)
    near_zero = np.abs(x) < 1e-12
    out[near_zero] = 1.0
    xn = x[~near_zero]
    out[~near_zero] = np.sin(20.0 * xn) / (20.0 * xn)
    return out


TARGETS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sin4": target_sin4,
    "j0": target_j0_like,
}


def make_dataset(
    target_name: str = "sin4",
    n_train: int = 40,
    x_min: float = -1.0,
    x_max: float = 1.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs = rng.uniform(x_min, x_max, size=n_train)
    ys = TARGETS[target_name](xs)
    return xs, ys


def initial_params(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, 0.3, size=N_PARAMS)
    for ell in range(REPS):
        params[4 * ell + 2] = float(2**ell)
        params[4 * ell + 3] = 0.0
    return params


def mse_loss(params: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> float:
    preds = activations(xs, params)
    return float(np.mean((preds - ys)**2))


def train(
    xs: np.ndarray,
    ys: np.ndarray,
    seed: int = 0,
    maxiter: int = 80,
    verbose: bool = True,
) -> tuple[np.ndarray, float, list[float]]:
    params0 = initial_params(seed)
    history: list[float] = []

    def objective(theta: np.ndarray) -> float:
        loss = mse_loss(theta, xs, ys)
        history.append(loss)
        if verbose and (len(history) == 1 or len(history) % 10 == 0):
            print(f"  iter {len(history):4d}  mse={loss:.6e}")
        return loss

    if verbose:
        print(f"Initial MSE: {objective(params0):.6e}")

    result = minimize(
        objective,
        params0,
        method="COBYLA",
        options={
            "maxiter": max(maxiter // 2, 40),
            "rhobeg": 0.5,
            "tol": 1e-8
        },
    )
    result = minimize(
        objective,
        np.asarray(result.x, dtype=float),
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "ftol": 1e-12
        },
    )
    final_mse = float(result.fun)
    if verbose:
        print(f"Final MSE:   {final_mse:.6e}  (success={result.success})")
    return np.asarray(result.x, dtype=float), final_mse, history


def evaluate_grid(
    params: np.ndarray,
    target_name: str = "sin4",
    n_grid: int = 100,
    x_min: float = -1.0,
    x_max: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.linspace(x_min, x_max, n_grid)
    ys = TARGETS[target_name](xs)
    preds = activations(xs, params)
    return xs, ys, preds


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train a DARUAN CUDA-Q activation")
    parser.add_argument("--target", default=None)
    parser.add_argument("--function", choices=sorted(TARGETS), default="sin4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-train", type=int, default=40)
    parser.add_argument("--maxiter", type=int, default=80)
    args = parser.parse_args(argv)

    target = select_target(args.target)
    print(f"CUDA-Q target: {target}")
    xs, ys = make_dataset(args.function, n_train=args.n_train, seed=args.seed)
    _, final_mse, _ = train(xs, ys, seed=args.seed, maxiter=args.maxiter)
    return 0 if final_mse < 0.5 else 1


if __name__ == "__main__":
    raise SystemExit(main())
