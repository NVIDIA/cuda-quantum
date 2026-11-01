# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.
#
# This source code and the accompanying materials are made available under
# the terms of the Apache License 2.0 which accompanies this distribution.
# ============================================================================ #
"""Virtual-syndrome transfer helpers for the Surface recipe."""

from cudaq.logical.algebra.clifford import _conjugate
from cudaq.logical.algebra.pauli import PauliGroupElement
from cudaq.logical.compiler import compile, project_clifford
from cudaq.logical.gadgets.contracts import GadgetContractBuilder
from cudaq.logical.gadgets.records import ProfileParity


def passthrough_profile(gadget):
    """Declare unchanged virtual syndromes for an encoded operation."""
    graph = GadgetContractBuilder(gadget.interface)
    for source, target in zip(gadget.inputs.blocks, gadget.outputs.blocks):
        graph.bind(target.syndrome, source.syndrome)
    return graph.freeze()


def clifford_profile(gadget):
    """Pull output stabilizers back through a measurement-free layer.

    Express each pulled-back operator in the input stabilizer basis. Its
    eigenvalue gives the output virtual syndrome, including affine signs.
    This handles both single-patch gates and gates coupling two patches.
    """
    projection = project_clifford(compile(gadget), gadget=gadget.name)
    width = projection.interface.carrier_count

    def stabilizers(endpoint, port):
        for row in endpoint.code_profile.effective_stabilizers.rows:
            n = len(row) // 2
            x = sum(row[i] << q for i, q in enumerate(port.data_carriers))
            z = sum(row[n + i] << q for i, q in enumerate(port.data_carriers))
            yield x, z, 0

    # Retain the actual signed Pauli product alongside its syndrome parity.
    basis = {}
    for endpoint, port in zip(gadget.inputs.blocks,
                              projection.interface.inputs):
        for index, (x, z, _) in enumerate(stabilizers(endpoint, port)):
            operator = PauliGroupElement(width, x, z, (x & z).bit_count())
            parity = ProfileParity.from_value(endpoint.syndrome[index])
            while (operator.x_mask | (operator.z_mask << width)):
                pivot = (operator.x_mask |
                         (operator.z_mask << width)).bit_length() - 1
                if pivot not in basis:
                    basis[pivot] = operator, parity
                    break
                row, value = basis[pivot]
                operator = operator.multiply(row)
                parity = parity ^ value

    images = [
        row for endpoint, port in zip(gadget.outputs.blocks,
                                      projection.interface.outputs)
        for row in stabilizers(endpoint, port)
    ]
    for op in reversed(projection.instructions):
        name = {"S": "sdg", "S_DAG": "s"}.get(op.kind, op.kind.lower())
        size = 2 if op.kind in ("CX", "CZ", "SWAP") else 1
        for start in reversed(range(0, len(op.carriers), size)):
            wires = op.carriers[start:start + size]
            if name == "swap":
                for pair in (wires, wires[::-1], wires):
                    _conjugate(images, "cx", pair)
            else:
                _conjugate(images, name, wires)

    values = []
    for x, z, sign in images:
        operator = PauliGroupElement(width, x, z,
                                     (x & z).bit_count() + 2 * sign)
        parity = ProfileParity()
        while (operator.x_mask | (operator.z_mask << width)):
            row, value = basis[(operator.x_mask |
                                (operator.z_mask << width)).bit_length() - 1]
            operator = operator.multiply(row)
            parity = parity ^ value
        values.append(parity ^ (operator.phase_exponent_mod_4 == 2))

    graph = GadgetContractBuilder(gadget.interface)
    start = 0
    for endpoint in gadget.outputs.blocks:
        end = start + endpoint.syndrome.width
        graph.bind(endpoint.syndrome, tuple(values[start:end]))
        start = end
    return graph.freeze()
