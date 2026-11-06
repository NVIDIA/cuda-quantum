# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Rotated-surface gadgets with the interleaved N/Z gate schedule."""

from __future__ import annotations

from dataclasses import dataclass
from cudaq.logical import codes, gadgets, ops, std, types
from cudaq.logical.architecture.geometry import rotated_surface_lattice

# Displacements from each check ancilla, one interaction per carrier per layer.
# The final two interactions put X hooks horizontally and Z hooks vertically,
# transverse to the vertical logical X and horizontal logical Z strings.
_X_ORDER = ((1, 1), (-1, 1), (1, -1), (-1, -1))
_Z_ORDER = ((1, 1), (1, -1), (-1, 1), (-1, -1))


@dataclass(frozen=True, slots=True)
class SurfaceInterleavedNZDefinitions:
    """Reusable preparation, SEC and destructive Z readout for one distance.

    Each gadget includes one extraction round. Readout then measures every data
    qubit and closes the code interface. Coordinates cover data, X ancillas,
    then Z ancillas, in the compiler's physical carrier order.
    """

    distance: int
    code: codes.Code
    prepare_zero: gadgets.GadgetDefinition
    syndrome_extraction: gadgets.GadgetDefinition
    measure_z_with_syndrome: gadgets.GadgetDefinition
    coordinates: tuple[tuple[float, float], ...]


def build_definitions(distance):
    from cudaq.logical.compiler import compile, project_clifford

    code = codes.Surface[distance]
    encoding = code.default_encoding
    lattice = rotated_surface_lattice(distance)
    data = {point: i for i, point in enumerate(lattice.coordinates)}
    xs = tuple(face for face in lattice.faces if face.sublattice == 1)
    zs = tuple(face for face in lattice.faces if face.sublattice == 0)
    ancillas = sorted(
        [(face.coordinate, "sx", i) for i, face in enumerate(xs)] +
        [(face.coordinate, "sz", i) for i, face in enumerate(zs)],
        key=lambda item: (item[0][1], item[0][0]))
    z_boundary_order = sorted(range(len(zs)), key=lambda i: zs[i].coordinate)

    def define(function, name, action, result):
        function.__name__ = function.__qualname__ = (
            f"surface_d{distance}_circuit_{name}")
        function.__annotations__ = {
            "block": types.patch[encoding],
            "return": result
        }
        return gadgets.gadget(function,
                              implements=action,
                              metadata={"syndrome_rounds": 1})

    def extract(block):
        block = ops.h(block.sx)
        for x_delta, z_delta in zip(_X_ORDER, _Z_ORDER):
            for faces, delta, x_check in ((xs, x_delta, True), (zs, z_delta,
                                                                False)):
                for i, face in enumerate(faces):
                    point = (face.coordinate[0] + delta[0],
                             face.coordinate[1] + delta[1])
                    if point not in data:
                        continue
                    if x_check:
                        block = ops.cx(block.sx[(i,)],
                                       block.data[(data[point],)])
                    else:
                        block = ops.cx(block.data[(data[point],)],
                                       block.sz[(i,)])
        block = ops.h(block.sx)
        # Resets belong to the next gadget that uses each ancilla. Only data
        # errors cross the encoded interface between extraction rounds.
        for _, partition, i in ancillas:
            block, _ = ops.mz(getattr(block, partition)[(i,)],
                              record=f"{partition}{i}")
        return block

    extraction = define(extract, "extract", std.idle, types.patch[encoding])

    def syndrome_round(block):
        block = ops.reset(block.sx)
        block = ops.reset(block.sz)
        return extraction(block)

    sec = define(syndrome_round, "sec", std.idle, types.patch[encoding])

    def prepare(block):
        block = ops.reset(block.data)
        block = ops.reset(block.sx)
        block = ops.reset(block.sz)
        return extraction(block)

    prep = define(prepare, "prepare_zero", std.prepare_zero,
                  types.patch[encoding])

    def readout(block):
        block = sec(block)
        # Match the reference circuit's terminal reset fault sites. With no
        # subsequent ancilla interactions, those faults have zero response.
        block = ops.reset(block.sx)
        block = ops.reset(block.sz)
        logical = []
        for i in range(code.n):
            block, outcome = ops.mz(block.data[(i,)], record=f"data{i}")
            if i in code.lz[0]:
                logical.append(outcome)
        ops.discard(block)
        return ops.parity(*logical)

    mz = define(readout, "measure_z", std.measure_z, bool)

    sites = []

    def contract(gadget, kind):
        projection = project_clifford(compile(gadget), gadget=gadget.name)
        port, = projection.interface.inputs or projection.interface.outputs
        measured = {
            row.carriers[0]: gadget.record(row.record)
            for row in projection.interface.measurements
            if row.symbol == extraction.name
        }
        # The named site is the entry to this exact nested syndrome gadget,
        # before its first operation. Record provenance identifies the call;
        # no gate-kind heuristic is needed to find the boundary.
        call_path, = {
            row.call_path
            for row in projection.interface.measurements
            if row.symbol == extraction.name
        }
        prefix = ".".join((*call_path, ""))
        entry = next(op for op in projection.instructions
                     if op.site_id.startswith(prefix))
        sites.append((gadget, projection, entry.site_id, port.data_carriers))
        records = {
            (partition, i): measured[q] for partition in ("sx", "sz")
            for i, q in enumerate(port.carriers(partition))
        }
        graph = gadgets.GadgetContractBuilder(gadget.interface)
        if kind == "prep":
            for i in z_boundary_order:
                graph.detector(records["sz", i])
        else:
            incoming = gadget.inputs.only().syndrome
            for _, partition, i in ancillas:
                index = i if partition == "sx" else len(xs) + i
                graph.detector(incoming[index] ^ records[partition, i])
        if kind == "readout":
            for i in z_boundary_order:
                parity = gadgets.ProfileParity()
                for q in code.hz[i]:
                    parity ^= gadget.record(f"data{q}.data0")
                graph.detector(parity ^ records["sz", i])
        else:
            graph.bind(
                gadget.outputs.only().syndrome,
                tuple(records[partition, i]
                      for partition, faces in (("sx", xs), ("sz", zs))
                      for i in range(len(faces))))
        return graph.freeze()

    definitions = SurfaceInterleavedNZDefinitions(
        distance, code, prep, sec, mz,
        lattice.coordinates + tuple(face.coordinate for face in xs) +
        tuple(face.coordinate for face in zs))
    contracts = (contract(prep,
                          "prep"), contract(sec,
                                            "sec"), contract(mz, "readout"))
    return definitions, contracts, sites
