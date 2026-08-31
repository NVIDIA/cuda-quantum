# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Iterable


class NonCliffordAction(TypeError):
    """Raised when canonical Clifford analysis encounters non-Clifford intent."""


def _identity(arity):
    return [
        *((1 << index, 0, 0) for index in range(arity)),
        *((0, 1 << index, 0) for index in range(arity)),
    ]


def _conjugate(images, name, wires):
    if name == "idle":
        return
    if name in {"t", "tdg", "ccz"}:
        raise NonCliffordAction(
            f"logical objective contains non-Clifford action {name!r}")
    if name in {"x", "y", "z", "h", "s", "sdg"}:
        if len(wires) != 1:
            raise TypeError(f"logical action {name!r} requires one operand")
        qubit = wires[0]
        bit = 1 << qubit
        for index, (x_mask, z_mask, phase) in enumerate(images):
            x_bit = bool(x_mask & bit)
            z_bit = bool(z_mask & bit)
            if name == "x":
                phase ^= int(z_bit)
            elif name == "z":
                phase ^= int(x_bit)
            elif name == "y":
                phase ^= int(x_bit ^ z_bit)
            elif name == "h":
                phase ^= int(x_bit and z_bit)
                if x_bit != z_bit:
                    x_mask ^= bit
                    z_mask ^= bit
            elif name in {"s", "sdg"}:
                phase ^= int(x_bit and (z_bit if name == "s" else not z_bit))
                if x_bit:
                    z_mask ^= bit
            images[index] = (x_mask, z_mask, phase)
        return
    if name in {"cx", "cz"}:
        if len(wires) != 2 or wires[0] == wires[1]:
            raise TypeError(f"logical action {name!r} requires two operands")
        control, target = wires
        if name == "cz":
            _conjugate(images, "h", (target,))
            _conjugate(images, "cx", wires)
            _conjugate(images, "h", (target,))
            return
        control_bit = 1 << control
        target_bit = 1 << target
        for index, (x_mask, z_mask, phase) in enumerate(images):
            x_control = bool(x_mask & control_bit)
            x_target = bool(x_mask & target_bit)
            z_control = bool(z_mask & control_bit)
            z_target = bool(z_mask & target_bit)
            phase ^= int(x_control and z_target and
                         (x_target ^ z_control ^ True))
            if x_control:
                x_mask ^= target_bit
            if z_target:
                z_mask ^= control_bit
            images[index] = (x_mask, z_mask, phase)
        return
    raise NonCliffordAction(f"unsupported Clifford objective action {name!r}")


def _symbol_name(attribute):
    raw = getattr(attribute, "value", attribute)
    if isinstance(raw, (tuple, list)):
        raw = raw[-1]
    return str(raw).strip('"').removeprefix("@").split("::@")[-1]


def _derive_program(program, arity):
    block = program.regions[0].blocks[0]
    if len(block.arguments) != arity:
        raise ValueError("logical action semantics arity is inconsistent")
    wire_of = {
        argument: index for index, argument in enumerate(block.arguments)
    }
    images = _identity(arity)
    returned_wires = None
    for child in block.operations:
        operation = child.operation
        if operation.name == "qlx.apply":
            action_attr = str(operation.attributes["action"])
            if action_attr.startswith("#qlx.action<"):
                action_name = action_attr[len("#qlx.action<"):-1].strip('"')
            else:
                action_name = _symbol_name(operation.attributes["action"])
                action_name = action_name.removeprefix("qlx_standard_")
            try:
                wires = tuple(wire_of[value] for value in operation.operands)
            except KeyError as exc:
                raise NonCliffordAction(
                    "Clifford objective applies an action to a non-qubit value"
                ) from exc
            _conjugate(images, action_name, wires)
            if len(operation.results) != len(wires):
                raise NonCliffordAction(
                    "Clifford actions must preserve quantum arity")
            wire_of.update(zip(operation.results, wires))
        elif operation.name == "qlx.return":
            try:
                returned_wires = tuple(
                    wire_of[value] for value in operation.operands)
            except KeyError as exc:
                raise NonCliffordAction(
                    "Clifford objective returns a non-qubit value") from exc
        else:
            raise NonCliffordAction(
                "canonical Clifford analysis requires a straight-line logical "
                f"action; found {operation.name}")
    if returned_wires is None or sorted(returned_wires) != list(range(arity)):
        raise NonCliffordAction(
            "Clifford objective must return every logical operand once")
    output_of_wire = {
        wire: output for output, wire in enumerate(returned_wires)
    }

    def remap(mask):
        result = 0
        for wire in range(arity):
            if mask & (1 << wire):
                result |= 1 << output_of_wire[wire]
        return result

    return tuple((remap(x), remap(z), phase) for x, z, phase in images)


@dataclass(frozen=True, slots=True)
class CliffordAction:
    """Complete signed binary-symplectic logical action.

    Rows are ordered images of ``X_0..X_(k-1), Z_0..Z_(k-1)``. Columns are
    ordered ``x_0..x_(k-1), z_0..z_(k-1)``. ``phases[i]`` is 1 exactly when
    generator image ``i`` carries a minus sign.
    """

    matrix: tuple[tuple[int, ...], ...]
    phases: tuple[int, ...]
    ports: tuple[Hashable, ...]
    evidence: tuple[str, ...] = ()

    def __post_init__(self):
        matrix = tuple(tuple(int(bit) for bit in row) for row in self.matrix)
        phases = tuple(int(bit) for bit in self.phases)
        ports = tuple(self.ports)
        width = 2 * len(ports)
        if len(set(ports)) != len(ports):
            raise ValueError("CliffordAction ports must be unique")
        if len(matrix) != width or any(len(row) != width for row in matrix):
            raise ValueError(
                f"CliffordAction for {len(ports)} ports requires a {width}x{width} matrix"
            )
        if any(bit not in (0, 1) for row in matrix for bit in row):
            raise ValueError("CliffordAction matrix entries must be binary")
        if len(phases) != width or any(bit not in (0, 1) for bit in phases):
            raise ValueError(
                "CliffordAction phases require one binary sign per row")
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "phases", phases)
        object.__setattr__(self, "ports", ports)
        object.__setattr__(self, "evidence", tuple(self.evidence))
        self._verify_symplectic()

    @property
    def arity(self):
        return len(self.ports)

    @property
    def images(self):
        result = []
        for row, phase in zip(self.matrix, self.phases):
            x_mask = sum(
                bit << index for index, bit in enumerate(row[:self.arity]))
            z_mask = sum(
                bit << index for index, bit in enumerate(row[self.arity:]))
            result.append((x_mask, z_mask, phase))
        return tuple(result)

    def _verify_symplectic(self):
        k = self.arity
        for left, a in enumerate(self.matrix):
            for right, b in enumerate(self.matrix):
                actual = sum(a[index] * b[k + index] + a[k + index] * b[index]
                             for index in range(k)) % 2
                expected = int((left < k and right == k + left) or
                               (right < k and left == k + right))
                if actual != expected:
                    raise ValueError("CliffordAction matrix is not symplectic")

    @classmethod
    def from_symplectic(cls, *, matrix, phases, ports):
        return cls(tuple(map(tuple, matrix)), tuple(phases), tuple(ports))

    @classmethod
    def standard(cls, name: str, arity: int, *, ports=None):
        images = _identity(arity)
        _conjugate(images, name, tuple(range(arity)))
        return cls.from_images(
            images,
            ports=tuple(range(arity)) if ports is None else tuple(ports),
            evidence=("standard-logical-action",),
        )

    @classmethod
    def from_images(cls, images, *, ports, evidence=()):
        ports = tuple(ports)
        arity = len(ports)
        rows = []
        phases = []
        for x_mask, z_mask, phase in images:
            rows.append(
                tuple((x_mask >> index) & 1 for index in range(arity)) + tuple(
                    (z_mask >> index) & 1 for index in range(arity)))
            phases.append(phase)
        return cls(tuple(rows), tuple(phases), ports, tuple(evidence))

    @classmethod
    def from_mlir_program(cls, program, *, ports):
        ports = tuple(ports)
        return cls.from_images(
            _derive_program(program, len(ports)),
            ports=ports,
            evidence=("canonical-qlx-clifford-analysis",),
        )

    @classmethod
    def from_program(cls, definition):
        from ..programs.definition import ProgramDefinition
        from ..compiler import compile

        if not isinstance(definition,
                          ProgramDefinition) or definition.kind != "objective":
            raise TypeError(
                "CliffordAction.from_program expects @cudaq.logical.objective")
        build = compile(definition)
        # Clifford semantics are derived evidence, so inspect a private replay
        # of the immutable Build rather than its mutable cached module view.
        module = build._fresh_module()
        action = next(
            (view.operation
             for view in module.body.operations
             if view.operation.name == "qlx.action" and _symbol_name(
                 view.operation.attributes["sym_name"]) == build.root.symbol),
            None,
        )
        if action is None or "semantics" not in action.attributes:
            raise ValueError(
                "logical objective is not an action with a semantics program")
        semantics = _symbol_name(action.attributes["semantics"])
        program = next(
            (view.operation
             for view in module.body.operations
             if view.operation.name == "qlx.objective_body" and
             _symbol_name(view.operation.attributes["sym_name"]) == semantics),
            None,
        )
        if program is None:
            raise ValueError("logical objective semantics body is missing")
        return cls.from_mlir_program(program,
                                     ports=tuple(
                                         definition.signature.parameters))

    def to_mlir_attr(self, context):
        from cudaq.mlir import ir as mlir_ir

        matrix = ", ".join(str(bit) for row in self.matrix for bit in row)
        phases = ", ".join(str(bit) for bit in self.phases)
        ports = ", ".join(f'"{port}"' for port in self.ports)
        return mlir_ir.Attribute.parse(
            "#qlx.clifford_action<"
            f"matrix = [{matrix}], "
            f"phases = [{phases}], ports = [{ports}]>",
            context=context,
        )


__all__ = ["CliffordAction", "NonCliffordAction"]
