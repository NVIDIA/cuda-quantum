"""
DARUAN single-qubit activation (Jiang et al., arXiv:2509.14026).

Circuit (r=3):
  U(x) = W4 S(w3 x + b3) W3 S(w2 x + b2) W2 S(w1 x + b1) W1
  W = Rz(a) Ry(pi/3) Rz(b),  S(phi) = Rz(phi)
  phi(x) = <Z>
"""

from __future__ import annotations

import math

import cudaq
import numpy as np
from cudaq import spin
from scipy.optimize import minimize

REPS = 3
N_PARAMS = 4 * REPS + 2  # 14


def select_target(name: str = "qpp-cpu") -> str:
    available = {t.name for t in cudaq.get_targets()}
    if name not in available:
        raise ValueError(f"Target {name!r} not available. Have: {sorted(available)}")
    cudaq.set_target(name)
    return name


@cudaq.kernel
def daruan_kernel(x: float, params: list[float]):
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


def activation(x: float, params) -> float:
    p = [float(v) for v in params]
    return float(cudaq.observe(daruan_kernel, spin.z(0), float(x), p).expectation())


def initial_params(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, 0.3, size=N_PARAMS)
    for ell in range(REPS):
        params[4 * ell + 2] = float(2**ell)
        params[4 * ell + 3] = 0.0
    return params


def mse_loss(params, xs, ys) -> float:
    preds = np.array([activation(float(x), params) for x in xs])
    return float(np.mean((preds - ys)**2))


def train(xs, ys, seed: int = 0, maxiter: int = 40):
    params0 = initial_params(seed)
    history = []

    def objective(theta):
        loss = mse_loss(theta, xs, ys)
        history.append(loss)
        return loss

    result = minimize(objective,
                      params0,
                      method="COBYLA",
                      options={
                          "maxiter": maxiter,
                          "rhobeg": 0.5
                      })
    return np.asarray(result.x, dtype=float), float(result.fun), history
