# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Rotated surface-code Z memory from preparation, SEC, and readout profiles."""

import argparse
from dataclasses import dataclass
import sys

import cudaq.logical as cql
from cudaq.logical.architecture.geometry import rotated_surface_lattice


@dataclass(frozen=True)
class SurfaceMemory:
    profiles: tuple
    round_starts: dict
    coordinates: tuple

    def to_stim(self, noise=None) -> str:
        """Return circuit text, including coordinates, detectors and observable."""
        lines = []
        for q, coordinate in enumerate(self.coordinates):
            x, y = coordinate
            lines.append(f"QUBIT_COORDS({x}, {y}) {q}")
        return "\n".join(lines) + "\n" + cql.compiler.emit_stim(
            self.profiles, noise=noise, round_starts=self.round_starts)


def surface_memory(distance=3, rounds=3):
    """Build ``prep0 -> SEC ** (rounds - 2) -> readout_Z`` for rounds >= 2.

    Preparation and readout each include an extraction round. The four
    interleaved N/Z interaction layers preserve the circuit distance: X hooks
    run horizontally, across the vertical logical X string; Z hooks run
    vertically, across the horizontal logical Z string.
    """
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 2:
        raise ValueError("rounds must be an integer of at least two")
    code = cql.codes.Surface[distance]
    lattice = rotated_surface_lattice(distance)
    data = {point: i for i, point in enumerate(lattice.coordinates)}
    xs = tuple(face for face in lattice.faces if face.sublattice == 1)
    zs = tuple(face for face in lattice.faces if face.sublattice == 0)
    # Corner displacements from each check ancilla. The last two interactions
    # make horizontal X hooks and vertical Z hooks, transverse to the logical
    # strings of the same Pauli type. Each layer couples disjoint carriers.
    x_order = ((1, 1), (-1, 1), (1, -1), (-1, -1))
    z_order = ((1, 1), (1, -1), (-1, 1), (-1, -1))
    ancillas = sorted(
        [(face.coordinate, "sx", i) for i, face in enumerate(xs)] +
        [(face.coordinate, "sz", i) for i, face in enumerate(zs)],
        key=lambda item: (item[0][1], item[0][0]))
    z_boundary_order = sorted(range(len(zs)), key=lambda i: zs[i].coordinate)

    def define(function, name, objective, result):
        function.__name__ = function.__qualname__ = f"surface_d{distance}_{name}"
        function.__annotations__ = {"block": cql.patch[code], "return": result}
        return cql.gadget(function, implements=objective)

    def extract(block):
        block = cql.h(block.sx)
        for x_delta, z_delta in zip(x_order, z_order):
            for faces, delta, x_check in ((xs, x_delta, True), (zs, z_delta,
                                                                False)):
                for i, face in enumerate(faces):
                    point = (face.coordinate[0] + delta[0],
                             face.coordinate[1] + delta[1])
                    if point not in data:
                        continue
                    if x_check:
                        block = cql.cx(block.sx[(i,)],
                                       block.data[(data[point],)])
                    else:
                        block = cql.cx(block.data[(data[point],)],
                                       block.sz[(i,)])
        block = cql.h(block.sx)
        # Leave reset to the next extraction so its faults belong to the gadget
        # that uses the ancilla. Only data-code errors cross encoded boundaries.
        for _, partition, i in ancillas:
            block, _ = cql.ops.mz(getattr(block, partition)[(i,)],
                                  record=f"{partition}{i}")
        return block

    extraction = define(extract, "extract", cql.logical.idle, cql.patch[code])

    def syndrome_round(block):
        block = cql.reset(block.sx)
        block = cql.reset(block.sz)
        return extraction(block)

    sec = define(syndrome_round, "sec", cql.logical.idle, cql.patch[code])

    def prepare(block):
        block = cql.reset(block.data)
        block = cql.reset(block.sx)
        block = cql.reset(block.sz)
        return extraction(block)

    prep0 = define(prepare, "prep0", cql.logical.prepare_zero, cql.patch[code])

    def readout(block):
        block = sec(block)
        # Retain the reference circuit's terminal reset fault sites. These
        # ancillas have no subsequent interactions, so their responses are zero.
        block = cql.reset(block.sx)
        block = cql.reset(block.sz)
        logical = []
        for i in range(code.n):
            block, outcome = cql.ops.mz(block.data[(i,)], record=f"data{i}")
            if i in code.lz[0]:
                logical.append(outcome)
        cql.discard(block)
        return cql.ops.parity(*logical)

    readout_z = define(readout, "readout_z", cql.logical.measure_z, bool)

    def profile(gadget, kind):
        projection = cql.compiler.project_clifford(cql.compile(gadget))
        port, = projection.interface.inputs or projection.interface.outputs
        measured = {
            row.carriers[0]: gadget.record(row.record)
            for row in projection.interface.measurements
            if row.symbol == extraction.name
        }
        records = {
            (partition, i): measured[q] for partition in ("sx", "sz")
            for i, q in enumerate(port.carriers(partition))
        }
        graph = cql.gadgets.ProfileGraph(gadget.interface)
        if kind == "prep":
            for i in z_boundary_order:
                graph.detector(records["sz", i])
        else:
            incoming = gadget.interface.inputs.only().syndrome
            for _, partition, i in ancillas:
                index = i if partition == "sx" else len(xs) + i
                graph.detector(incoming[index] ^ records[partition, i])
        if kind == "readout":
            for i in z_boundary_order:
                parity = cql.gadgets.ProfileParity()
                for q in code.hz[i]:
                    parity ^= gadget.record(f"data{q}.data0")
                graph.detector(parity ^ records["sz", i])
        else:
            graph.bind(
                gadget.interface.outputs.only().syndrome,
                tuple(records[partition, i]
                      for partition, faces in (("sx", xs), ("sz", zs))
                      for i in range(len(faces))))
        return graph.freeze()

    profiles = (profile(prep0,
                        "prep"), *((profile(sec, "sec"),) * (rounds - 2)),
                profile(readout_z, "readout"))
    starts = {}
    for i, selected in enumerate(profiles):
        projection = cql.compiler.project_clifford(cql.compile(selected),
                                                   gadget=selected.gadget.name)
        # Each occurrence contains exactly one round. Its first H starts that
        # round after any preparation resets, before the check interactions.
        starts[i] = (next(
            op.site_id for op in projection.instructions if op.kind == "H"),)
    coordinates = (lattice.coordinates + tuple(face.coordinate for face in xs) +
                   tuple(face.coordinate for face in zs))
    return SurfaceMemory(profiles, starts, coordinates)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Sample this many shots with Stim (default: emit text only)")
    parser.add_argument("--seed", type=int, help="Optional Stim random seed")
    args = parser.parse_args(argv)
    if args.shots < 0:
        parser.error("--shots must be nonnegative")
    if args.seed is not None and not 0 <= args.seed < 2**64:
        parser.error("--seed must be in [0, 2**64)")

    # Simulation is an optional example dependency; the exporter itself emits
    # text without loading Stim.
    stim = None
    if args.shots:
        try:
            import stim
        except ImportError:
            parser.error("sampling requires Stim; install with "
                         "python -m pip install 'stim>=1.16,<2'")

    p = 0.001
    memory = surface_memory(distance=3, rounds=3)
    noise = cql.compiler.StandardDepolarizingNoise(
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p)
    text = memory.to_stim(noise)
    print(text, end="")
    if stim is not None:
        circuit = stim.Circuit(text)
        sampler = circuit.compile_detector_sampler(seed=args.seed)
        detectors, observables = sampler.sample(args.shots,
                                                separate_observables=True)
        # Keep stdout valid circuit text when redirected to a .stim file.
        print(
            f"Sampled {args.shots} shots "
            f"(detectors={detectors.shape[1]}, observables={observables.shape[1]})",
            file=sys.stderr)
        print(
            f"Shots with detection events: "
            f"{detectors.any(axis=1).sum()}/{args.shots}",
            file=sys.stderr)
        print(
            f"Shots with uncorrected observable flips: "
            f"{observables.any(axis=1).sum()}/{args.shots}",
            file=sys.stderr)


if __name__ == "__main__":
    main()
