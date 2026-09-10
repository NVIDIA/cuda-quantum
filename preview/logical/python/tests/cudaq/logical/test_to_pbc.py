# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Native Pauli-based-computation lowering (the ``to_pbc`` P0 pass).

A synthesized Clifford+T program becomes pi/4 Pauli-product rotations (one per
T) followed by the terminal measurements conjugated into Pauli products. Each
run is checked against an independent stim tableau oracle: a T at forward
position p on qubit q rotates about ``V_before^dag Z_q V_before`` (V_before =
the Clifford gates before it), and a measurement of basis B on qubit i becomes
``C^dag B_i C`` (C = all Clifford gates).
"""
from __future__ import annotations

import random
import re
from typing import Tuple

import pytest

import cudaq.logical as qlx
from cudaq.mlir._mlir_libs import _qlxRuntime as rt

stim = pytest.importorskip("stim")


# -- build / parse ----------------------------------------------------------
def _synth(gates, meas):
    nq = 1 + max([g[1] for g in gates] + [g[2] for g in gates if g[0] == "cx"] +
                 [m[1] for m in meas])

    def prog():
        q = qlx.allocate(nq, state=qlx.types.zero)
        for g in gates:
            if g[0] == "h":
                q[g[1]] = qlx.h(q[g[1]])
            elif g[0] == "s":
                q[g[1]] = qlx.s(q[g[1]])
            elif g[0] == "sdg":
                q[g[1]] = qlx.sdg(q[g[1]])
            elif g[0] == "x":
                q[g[1]] = qlx.x(q[g[1]])
            elif g[0] == "z":
                q[g[1]] = qlx.z(q[g[1]])
            elif g[0] == "cx":
                q[g[1]], q[g[2]] = qlx.cx(q[g[1]], q[g[2]])
            elif g[0] == "t":
                (q[g[1]],) = qlx.ops.rotate(qlx.types.Z(q[g[1]]),
                                            angle=qlx.types.pi / 4)
            elif g[0] == "tdg":
                (q[g[1]],) = qlx.ops.rotate(qlx.types.Z(q[g[1]]),
                                            angle=-qlx.types.pi / 4)
        return tuple(
            qlx.measure_x(q[i]) if b == "X" else qlx.measure_z(q[i])
            for b, i in meas)

    prog.__annotations__["return"] = Tuple[tuple(bool for _ in meas)]
    prog = qlx.program(prog)
    mlir = qlx.compile(prog,
                       pipeline=qlx.compiler.pipelines.logical()).to_mlir()
    return rt.synthesize_qlx(mlir, 1e-10)


_PREP = re.compile(r"(%\w+) = qlx\.prepare .*value_index = (\d+)")
_ACT = re.compile(
    r"(%\w+(?::\d+)?) = qlx\.apply #qlx\.action<(\w+)>\(([^)]*)\)")
_MEAS = re.compile(r"%\w+ = qlx\.measure <(\w)> (%\w+(?:#\d+)?)")
_ROT = re.compile(
    r"(%\w+(?::\d+)?) = qlx\.apply #qlx\.action<pauli_rotation>\(([^)]*)\)")
_MPP = re.compile(
    r"(%\w+(?::\d+)?) = qlx\.instrument #qlx\.instrument<mpp>\(([^)]*)\)")


def _field(line, key):
    m = re.search(rf"\b{key} = (-?\d+)", line)
    return int(m.group(1)) if m else None


def _val2q(text):
    return {m.group(1): int(m.group(2)) for m in _PREP.finditer(text)}


def _thread(resbase, qubits, v):
    m = re.match(r"%(\w+)(?::(\d+))?", resbase)
    base = m.group(1)
    if m.group(2):
        for i, q in enumerate(qubits):
            v[f"%{base}#{i}"] = q
    elif qubits:
        v[f"%{base}"] = qubits[0]


def _pauli(qubits, xm, zm):
    d = {}
    for bit, q in enumerate(qubits):
        x, z = (xm >> bit) & 1, (zm >> bit) & 1
        if x and z:
            d[q] = "Y"
        elif x:
            d[q] = "X"
        elif z:
            d[q] = "Z"
    return d


def _parse_circuit(text):
    v = _val2q(text)
    nq = 1 + max(v.values())
    gates, meas = [], []
    for line in text.splitlines():
        line = line.strip()
        ma, mm = _ACT.match(line), _MEAS.match(line)
        if ma:
            res, act, ops = ma.group(1), ma.group(2), [
                o.strip() for o in ma.group(3).split(",")
            ]
            qs = [v[o] for o in ops if o in v]
            if act in ("h", "s", "sdg", "x", "y", "z", "cx", "cz", "t", "tdg"):
                gates.append((act, *qs))
            _thread(res, qs, v)
        elif mm:
            meas.append((mm.group(1), v[mm.group(2)]))
    return gates, meas, nq


def _parse_pbc(text):
    v = _val2q(text)
    rots, meass = [], []
    for line in text.splitlines():
        line = line.strip()
        mr, mm = _ROT.match(line), _MPP.match(line)
        if mr or mm:
            res, ops = (mr or mm).group(1), [
                o.strip() for o in (mr or mm).group(2).split(",")
            ]
            qs = [v[o] for o in ops if o in v]
            p = _pauli(qs, _field(line, "x_mask"), _field(line, "z_mask"))
            (rots if mr else meass).append((p, _field(line, "sign")))
            _thread(res, qs, v)
    return rots, meass


# -- stim oracle ------------------------------------------------------------
def _tableau(gates, nq, upto=None):
    c = stim.Circuit()
    for idx, g in enumerate(gates):
        if upto is not None and idx >= upto:
            break
        if g[0] in ("t", "tdg"):
            continue
        name = {
            "h": "H",
            "s": "S",
            "sdg": "S_DAG",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "cx": "CX",
            "cz": "CZ"
        }[g[0]]
        c.append(name, list(g[1:]))
    T = stim.Tableau.from_circuit(c)
    if len(T) < nq:
        full = stim.Tableau(nq)
        full.append(T, list(range(len(T))))
        T = full
    return T


def _key(ps):
    d = {i: "_XYZ"[ps[i]] for i in range(len(ps)) if "_XYZ"[ps[i]] != "_"}
    return frozenset(d.items()), int(round(ps.sign.real))


def _oracle(gates, meas, nq):
    rots = []
    for p, g in enumerate(gates):
        if g[0] in ("t", "tdg"):
            base = stim.PauliString("_" * g[1] + "Z" + "_" * (nq - g[1] - 1))
            rots.append(_key(_tableau(gates, nq, upto=p).inverse()(base)))
    tinv = _tableau(gates, nq).inverse()
    meass = [
        _key(tinv(stim.PauliString("_" * q + b + "_" * (nq - q - 1))))
        for b, q in meas
    ]
    return rots, meass


def _check(gates, meas):
    text = _synth(gates, meas)
    sg, sm, nq = _parse_circuit(text)
    pbc = rt.to_pbc(text)
    assert rt.verify_pbc(pbc) is True  # pass output is always PBC normal form
    prots, pmeas = _parse_pbc(pbc)
    orots, omeas = _oracle(sg, sm, nq)

    pmeas_k = sorted((frozenset(d.items()), s) for d, s in pmeas)
    prots_k = sorted((frozenset(d.items()), s) for d, s in prots)
    assert pmeas_k == sorted(
        omeas), f"measurements: {pmeas_k} != {sorted(omeas)}"
    assert prots_k == sorted(orots), f"rotations: {prots_k} != {sorted(orots)}"


HAND = {
    "T;Mz": ([("t", 0)], [("Z", 0)]),
    "H;T;Mz": ([("h", 0), ("t", 0)], [("Z", 0)]),
    "H;CX;Mzz": ([("h", 0), ("cx", 0, 1)], [("Z", 0), ("Z", 1)]),
    "T0;CX;T1;Mzz": ([("t", 0), ("cx", 0, 1), ("t", 1)], [("Z", 0), ("Z", 1)]),
    "S;T;H;T;Mx": ([("s", 0), ("t", 0), ("h", 0), ("t", 0)], [("X", 0)]),
    "prep-basis;Mx": ([("h", 0), ("s", 0), ("tdg", 0)], [("X", 0)]),
}


@pytest.mark.parametrize("name", list(HAND))
def test_pbc_hand_cases(name):
    _check(*HAND[name])


@pytest.mark.parametrize("trial", range(25))
def test_pbc_random_matches_stim(trial):
    rng = random.Random(1000 + trial)
    n = rng.randint(1, 4)
    gates = []
    for _ in range(rng.randint(3, 14)):
        k = rng.choice(["h", "s", "sdg", "x", "z", "cx", "t", "tdg"])
        if k == "cx" and n >= 2:
            a, b = rng.sample(range(n), 2)
            gates.append(("cx", a, b))
        elif k == "cx":
            gates.append(("h", 0))
        else:
            gates.append((k, rng.randrange(n)))
    meas = [(rng.choice(["Z", "X"]), i) for i in range(n)]
    _check(gates, meas)


def test_pbc_structure():
    # Rotations precede measurements; qubits are discarded; results are wired.
    text = rt.to_pbc(
        _synth([("h", 0), ("t", 0), ("cx", 0, 1)], [("Z", 0), ("Z", 1)]))
    body = [l.strip() for l in text.splitlines()]
    rot_idx = [i for i, l in enumerate(body) if "action<pauli_rotation>" in l]
    mpp_idx = [i for i, l in enumerate(body) if "instrument<mpp>" in l]
    assert rot_idx and mpp_idx and max(rot_idx) < min(mpp_idx)
    assert any("qlx.discard" in l for l in body)
    assert "action<h>" not in text and "action<cx>" not in text  # Cliffords absorbed


def test_pbc_rejects_unsynthesized():
    # A single-qubit pauli_rotation (pre-synthesis) is rejected with guidance.
    @qlx.program
    def raw() -> bool:
        q = qlx.allocate(1)
        (q[0],) = qlx.ops.rotate(qlx.types.Z(q[0]), angle=0.3, precision=1e-6)
        return qlx.measure_z(q[0])

    mlir = qlx.compile(raw, pipeline=qlx.compiler.pipelines.logical()).to_mlir()
    with pytest.raises(RuntimeError):
        rt.to_pbc(mlir)
