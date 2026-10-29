# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Execute the memory tutorials cell by cell with test-specific parameters."""

from pathlib import Path
from types import SimpleNamespace
from cudaq.logical.architecture.geometry import rotated_surface_lattice
from cudaq.logical.architectures import surface

import pytest

import cudaq.logical as cql

EXAMPLES = Path(__file__).resolve().parents[4] / "examples/standalone"


@pytest.fixture(scope="module")
def run_memory_example(tmp_path_factory):

    def run(name, **parameters):
        parameters.setdefault("shots", 0)
        if name in ("07_memory_edem.py", "08_surface_memory_circuit.py"):
            parameters.setdefault("output",
                                  tmp_path_factory.mktemp("memory-edem"))
        path = EXAMPLES / name
        namespace = {"__name__": "__main__", "__file__": str(path)}
        lineno = 0
        for cell in path.read_text().split("# %%\n"):
            # Preserve source locations for authoring introspection/tracebacks.
            exec(compile("\n" * lineno + cell, str(path), "exec"), namespace)
            if cell.startswith("# Configure the memory experiment."):
                namespace.update(parameters)
            lineno += cell.count("\n") + 1
        return namespace

    return run


@pytest.fixture(scope="module")
def memory_result():
    # A test adapter for varying export noise without rebuilding the gadgets.
    def result(namespace):
        profiles = namespace["profiles"]

        def depolarizations(probability):
            return {
                index:
                    surface.data_depol(
                        contract, namespace["projections"][contract.gadget],
                        probability)
                for index, contract in enumerate(profiles[:-1])
            }

        def to_stim(noise=None, *, data_depolarization=0.0):
            return namespace["coordinates"] + "\n" + cql.compiler.emit_stim(
                profiles,
                noise=noise,
                depolarizations=depolarizations(data_depolarization))

        return SimpleNamespace(
            profiles=profiles,
            coordinates=namespace["implementation"].coordinates,
            extraction_entries=tuple((p[0].before, p[0].carriers)
                                     for p in namespace["placements"].values()),
            depolarizations=depolarizations,
            to_stim=to_stim)

    return result


@pytest.fixture(scope="module")
def make_memory(run_memory_example, memory_result):

    def make(distance=3, rounds=3):
        return memory_result(
            run_memory_example("06_stim_emission.py",
                               distance=distance,
                               rounds=rounds))

    return make


def _surface_memory_reference(stim, distance, rounds, probability):
    code = cql.codes.Surface[distance]
    lattice = rotated_surface_lattice(distance)
    checks = [("X", i, support) for i, support in enumerate(code.hx)
             ] + [("Z", i, support) for i, support in enumerate(code.hz)]
    lines = [
        f"QUBIT_COORDS({x}, {y}) {i}"
        for i, (x, y) in enumerate(lattice.coordinates)
    ]
    lines.append("R " + " ".join(map(str, range(code.n))))
    previous, detectors, measured = {}, [], 0
    for round_index in range(rounds):
        if probability:
            lines.append(f"DEPOLARIZE1({probability}) " +
                         " ".join(map(str, range(code.n))))
        current = {}
        for basis, i, support in checks:
            lines.append("MPP " + "*".join(f"{basis}{q}" for q in support))
            current[basis, i] = measured
            measured += 1
        if round_index == 0:
            detectors.extend((current["Z", i],) for i in range(len(code.hz)))
        else:
            detectors.extend((previous[basis, i], current[basis, i])
                             for basis, i, _ in checks)
        previous = current
    lines.append("MPP " + "*".join(f"Z{q}" for q in code.lz[0]))
    measured += 1
    for row in detectors:
        lines.append("DETECTOR " + " ".join(f"rec[{r-measured}]" for r in row))
    lines.append("OBSERVABLE_INCLUDE(0) rec[-1]")
    return stim.Circuit("\n".join(lines))


@pytest.fixture(scope="module")
def surface_memory_reference():
    return _surface_memory_reference
