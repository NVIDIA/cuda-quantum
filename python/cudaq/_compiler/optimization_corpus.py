# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Seeded generators for straight-line, bounded-unitary Quake inputs."""

import random
from pathlib import Path

# Non-parametric single-qubit gates used as filler.
_SINGLE = ("h", "x", "y", "z", "s", "t")
# Involutions: g applied twice is the identity (up to nothing for these).
_INVOLUTIONS = ("h", "x", "y", "z")


def generate_module_text(seed: int,
                         num_qubits: int = 2,
                         length: int = 6,
                         kernel_name: str = "kern") -> str:
    """Return the Quake IR text for one seeded straight-line kernel."""
    rng = random.Random(seed)
    n = max(1, num_qubits)

    lines = [f"func.func @{kernel_name}() {{"]
    lines.append("  %cst = arith.constant 1.000000e+00 : f64")
    lines.append(f"  %q = quake.alloca !quake.veq<{n}>")
    refs = []
    for i in range(n):
        lines.append(f"  %r{i} = quake.extract_ref %q[{i}] : "
                     f"(!quake.veq<{n}>) -> !quake.ref")
        refs.append(f"%r{i}")

    for _ in range(length):
        choice = rng.random()
        target = rng.randrange(n)
        if choice < 0.6:
            gate = rng.choice(_SINGLE)
            lines.append(f"  quake.{gate} {refs[target]} : (!quake.ref) -> ()")
        elif choice < 0.8 or n < 2:
            lines.append(f"  quake.rz (%cst) {refs[target]} : "
                         f"(f64, !quake.ref) -> ()")
        else:
            control = rng.randrange(n)
            while control == target:
                control = rng.randrange(n)
            lines.append(f"  quake.x [{refs[control]}] {refs[target]} : "
                         f"(!quake.ref, !quake.ref) -> ()")

    # Injected motifs: a self-inverse pair and a pair of `mergeable` rotations.
    motif_target = rng.randrange(n)
    involution = rng.choice(_INVOLUTIONS)
    lines.append(f"  quake.{involution} {refs[motif_target]} : "
                 f"(!quake.ref) -> ()")
    lines.append(f"  quake.{involution} {refs[motif_target]} : "
                 f"(!quake.ref) -> ()")
    lines.append(f"  quake.rz (%cst) {refs[motif_target]} : "
                 f"(f64, !quake.ref) -> ()")
    lines.append(f"  quake.rz (%cst) {refs[motif_target]} : "
                 f"(f64, !quake.ref) -> ()")

    lines.append("  cc.return")
    lines.append("}")
    return "\n".join(lines) + "\n"


def write_corpus(directory,
                 seeds,
                 num_qubits: int = 2,
                 length: int = 6) -> list:
    """Write one ``generated_<seed>.qke`` per seed into ``directory``.

    Returns the list of written paths. Reproducible: same seeds and parameters
    always produce byte-identical files.
    """
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for seed in seeds:
        text = generate_module_text(seed, num_qubits, length)
        path = out / f"generated_{seed}.qke"
        path.write_text(text)
        paths.append(path)
    return paths
