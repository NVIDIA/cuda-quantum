# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Small reusable realization factories, exported as ordinary Python values.

This module is a library, not a registry.  Calling a factory returns a normal
``GadgetDefinition`` that the importing module can expose to QLX's private
module linker alongside research-local realizations.
"""

from __future__ import annotations

import hashlib

from cudaq.logical.programs.decorators import objective
from cudaq.logical.ops._impl import (
    cx,
    cz,
    discard,
    extract_syndrome,
    h,
    reset,
    s,
    sdg,
    x,
    y,
    z,
)
from cudaq.logical.ops._impl import measure_x as logical_measure_x
from cudaq.logical.ops._impl import measure_z as logical_measure_z
from cudaq.logical.ops._impl import (
    mpp,
    prepare,
)
from cudaq.logical.types.values import logical_qubit
from cudaq.logical.codes import (
    Code,
    Encoding,
)
from cudaq.logical.algebra.pauli import (
    PauliGroupElement,
    PauliProduct,
    X,
    Z,
)
from cudaq.logical.gadgets.definition import gadget
from cudaq.logical.gadgets.interface import patch
from cudaq.logical.types.semantic import (
    plus,
    zero,
)
from ..std import LogicalInstrumentRef
from ..std import idle as idle_objective
from ..std import prepare_plus as prepare_plus_objective
from ..std import prepare_zero as prepare_zero_objective
from ..std import x as x_objective
from ..std import z as z_objective


@objective(name="qlx_standard_measure_z")
def _measure_z_intent(q: logical_qubit) -> bool:
    return logical_measure_z(q)


@objective(name="qlx_standard_measure_x")
def _measure_x_intent(q: logical_qubit) -> bool:
    return logical_measure_x(q)


def _encoding(value):
    if isinstance(value, Encoding):
        return value
    if isinstance(value, Code):
        return value.default_encoding
    raise TypeError(
        "logical measurement realization requires a Code or Encoding")


def _measurement(
    value,
    *,
    basis: str,
    logical: int,
    preserve_block: bool,
    name: str | None,
):
    encoding = _encoding(value)
    code = encoding.code
    if not isinstance(logical, int) or isinstance(logical, bool):
        raise TypeError("logical measurement port must be an int")
    if not 0 <= logical < code.k:
        raise ValueError(f"code {code.name} has no protected logical {logical}")
    objective = _measure_z_intent if basis == "z" else _measure_x_intent
    pauli = Z if basis == "z" else X

    def realization(block):
        block, outcome = mpp(pauli(block[logical]))
        if preserve_block:
            return block, outcome
        discard(block)
        return outcome

    suffix = "_preserve_block" if preserve_block else ""
    realization.__name__ = (name or
                            f"{code.name}_measure_{basis}{logical}{suffix}")
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": tuple[patch[encoding], bool] if preserve_block else bool,
    }
    return gadget(realization, implements=objective, name=realization.__name__)


def measure_z(
    value,
    *,
    logical: int = 0,
    preserve_block: bool = False,
    name: str | None = None,
):
    """Return a destructive logical-Z measurement realization."""
    return _measurement(
        value,
        basis="z",
        logical=logical,
        preserve_block=preserve_block,
        name=name,
    )


def measure_x(
    value,
    *,
    logical: int = 0,
    preserve_block: bool = False,
    name: str | None = None,
):
    """Return a destructive logical-X measurement realization."""
    return _measurement(
        value,
        basis="x",
        logical=logical,
        preserve_block=preserve_block,
        name=name,
    )


def logical_measure(
    value,
    *,
    basis="z",
    logical: int = 0,
    preserve_block: bool = False,
    name: str | None = None,
):
    """Return a destructive logical measurement realization for one basis.

    ``basis`` accepts ``"x"``/``"z"`` (or any object whose ``value``/``name``
    normalizes to one of those). This is the single-call spelling of the
    :func:`measure_x`/:func:`measure_z` pair.
    """
    text = getattr(basis, "value", None) or getattr(basis, "name",
                                                    None) or basis
    text = str(text).lower()
    if text not in ("x", "z"):
        raise ValueError("logical_measure basis must be 'x' or 'z'")
    return _measurement(
        value,
        basis=text,
        logical=logical,
        preserve_block=preserve_block,
        name=name,
    )


def logical_pauli(
    value,
    *,
    basis: str,
    logical: int = 0,
    name: str | None = None,
):
    """Return the canonical encoded logical-X or logical-Z realization."""

    encoding = _encoding(value)
    code = encoding.code
    basis = str(basis).lower()
    if basis not in {"x", "z"}:
        raise ValueError("logical_pauli basis must be 'x' or 'z'")
    if not isinstance(logical, int) or isinstance(logical, bool):
        raise TypeError("logical Pauli port must be an int")
    if not 0 <= logical < code.k:
        raise ValueError(f"code {code.name} has no protected logical {logical}")
    representative = (code.logical_x_basis.rows[logical]
                      if basis == "x" else code.logical_z_basis.rows[logical])
    support = (representative[:code.n]
               if basis == "x" else representative[code.n:])
    indices = tuple(index for index, bit in enumerate(support) if bit)

    def realization(block):
        action = x if basis == "x" else z
        return action(block.frame[indices])

    realization.__name__ = name or f"{code.name}_logical_{basis}{logical}"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(
        realization,
        implements=x_objective if basis == "x" else z_objective,
        name=realization.__name__,
        metadata={
            "logical": logical,
            "pauli": basis.upper(),
            "representative": indices,
        },
    )


def css_memory_round(value,
                     *,
                     record: str | None = None,
                     name: str | None = None):
    """Return a code-specific logical memory round.

    One syndrome-extraction pass over the encoded block: the gadget consumes
    and returns the same linear patch owner while its check records update the
    syndrome history. It implements the standard ``cudaq.logical.logical.idle``
    objective, so QEC selection binds it to explicit ``cudaq.logical.idle`` memory
    workloads placed on the code.
    """
    encoding = _encoding(value)
    code = encoding.code

    def realization(block):
        if record is None:
            block, _ = extract_syndrome(block)
        else:
            block, _ = extract_syndrome(block, record=record)
        return block

    realization.__name__ = name or f"{code.name}_memory_round"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(realization,
                  implements=idle_objective,
                  name=realization.__name__)


def _preparation(value, *, state, objective, name: str | None):
    encoding = _encoding(value)
    code = encoding.code

    def realization(block):
        return prepare(block, state=state)

    realization.__name__ = name or f"{code.name}_prepare_{state.name}"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(realization, implements=objective, name=realization.__name__)


def prepare_zero(value, *, name: str | None = None):
    """Return a code-specific encoded ``|0>`` preparation realization."""

    return _preparation(
        value,
        state=zero,
        objective=prepare_zero_objective,
        name=name,
    )


def prepare_plus(value, *, name: str | None = None):
    """Return a code-specific encoded ``|+>`` preparation realization."""

    return _preparation(
        value,
        state=plus,
        objective=prepare_plus_objective,
        name=name,
    )


def _stim_pauli(element: PauliGroupElement, *, width: int):
    """Convert one phase-complete Pauli-group element to a Stim Pauli product."""

    try:
        import stim
    except ImportError as exc:
        raise ImportError("stabilizer-state preparation requires stim") from exc
    if element.arity > width:
        raise ValueError("logical stabilizer exceeds the encoded logical width")
    x_mask = element.x_mask
    z_mask = element.z_mask
    paulis = []
    for index in range(width):
        x_bit = bool(x_mask & (1 << index))
        z_bit = bool(z_mask & (1 << index))
        paulis.append(
            "Y" if x_bit and z_bit else "X" if x_bit else "Z" if z_bit else "I")
    # Stim's Hermitian Y is iXZ. Remove that local convention from the
    # phase-complete i^p X^x Z^z representation to recover the overall sign.
    residual_phase = (element.phase_exponent_mod_4 -
                      (x_mask & z_mask).bit_count()) % 4
    if residual_phase not in {0, 2}:
        raise ValueError("stabilizer-state generator must be Hermitian")
    result = stim.PauliString("".join(paulis))
    return -result if residual_phase == 2 else result


def _positive_row_pauli(code: Code, row) -> PauliGroupElement:
    """Interpret one phase-free symplectic row as a positive Hermitian Pauli."""

    x_mask = sum(int(bit) << index for index, bit in enumerate(row[:code.n]))
    z_mask = sum(int(bit) << index for index, bit in enumerate(row[code.n:]))
    return PauliGroupElement(
        code.n,
        x_mask,
        z_mask,
        (x_mask & z_mask).bit_count() % 4,
    )


def _lift_logical_stabilizer(code: Code, value: PauliProduct):
    """Lift one formal logical Pauli through the code's chosen representatives."""

    if not isinstance(value, PauliProduct):
        raise TypeError(
            "logical_stabilizers entries must be formal qlx Pauli products")
    logical = PauliGroupElement.from_product(value)
    if logical.arity > code.k:
        raise ValueError(
            f"logical stabilizer addresses port {logical.arity - 1}, but "
            f"code {code.name!r} has only {code.k} protected logical port(s)")
    physical = PauliGroupElement(
        code.n,
        0,
        0,
        logical.phase_exponent_mod_4,
    )

    # PauliGroupElement uses the canonical i^p X^x Z^z order. Preserve that
    # order while replacing every formal logical X/Z by the code's physical
    # representative, including the phase induced where representatives cross.
    for logical_index, row in enumerate(code.logical_x_basis.rows):
        if logical.x_mask & (1 << logical_index):
            physical = physical.multiply(_positive_row_pauli(code, row))
    for logical_index, row in enumerate(code.logical_z_basis.rows):
        if logical.z_mask & (1 << logical_index):
            physical = physical.multiply(_positive_row_pauli(code, row))
    return physical


def _stabilizer_preparation_circuit(
    encoding: Encoding,
    logical_stabilizers: tuple[PauliProduct, ...],
):
    try:
        import stim
    except ImportError as exc:
        raise ImportError("stabilizer-state preparation requires stim") from exc
    code = encoding.code
    if code.r:
        raise NotImplementedError(
            "stabilizer-state preparation currently requires a stabilizer code; "
            "subsystem-code gauge-state policy must be explicit")
    if len(logical_stabilizers) != code.k:
        raise ValueError(
            "logical_stabilizers must provide exactly one independent "
            f"generator per protected logical port ({code.k} required)")
    physical_generators = [
        _stim_pauli(
            _positive_row_pauli(code, row),
            width=code.n,
        ) for row in code.stabilizer_basis.rows
    ]
    physical_generators.extend(
        _stim_pauli(
            _lift_logical_stabilizer(code, stabilizer),
            width=code.n,
        ) for stabilizer in logical_stabilizers)
    try:
        tableau = stim.Tableau.from_stabilizers(physical_generators)
    except ValueError as exc:
        raise ValueError(
            "logical_stabilizers must be independent, mutually commuting, and "
            "compatible with the code stabilizers") from exc
    return tableau.to_circuit(method="elimination")


def stabilizer_preparation(
    value,
    *,
    logical_stabilizers,
    name: str | None = None,
):
    """Return an explicit encoded stabilizer-state preparation gadget.

    ``logical_stabilizers`` contains exactly ``code.k`` independent, commuting
    formal Pauli products over the code's protected logical-port indices. For
    example, ``(cudaq.logical.Y(0),)`` selects encoded ``|+Y>`` for a one-logical-qubit
    code, while ``(cudaq.logical.X(0) @ cudaq.logical.X(1), cudaq.logical.Z(0) @ cudaq.logical.Z(1))`` selects an
    encoded Bell state for a two-logical-qubit code.

    The returned ordinary :class:`~cudaq.logical.GadgetDefinition` contains the concrete
    reset and Clifford operations produced by tableau elimination. It is not a
    device-side opaque preparation instruction and does not claim a
    fault-tolerant preparation protocol by itself.
    """

    encoding = _encoding(value)
    code = encoding.code
    logical_stabilizers = tuple(logical_stabilizers)
    circuit = _stabilizer_preparation_circuit(
        encoding,
        logical_stabilizers,
    )
    instructions = tuple((
        instruction.name,
        tuple(target.value for target in instruction.targets_copy()),
    ) for instruction in circuit)
    supported = {"H", "S", "S_DAG", "X", "Y", "Z", "CX", "CZ"}
    unsupported = tuple(instruction for instruction, _targets in instructions
                        if instruction not in supported)
    if unsupported:
        raise NotImplementedError(
            "stabilizer tableau synthesis emitted unsupported instruction "
            f"{unsupported[0]!r}")

    def realization(block):
        block = reset(block.data)
        unary = {
            "H": h,
            "S": s,
            "S_DAG": sdg,
            "X": x,
            "Y": y,
            "Z": z,
        }
        for instruction, targets in instructions:
            if instruction in unary:
                for target in targets:
                    block = unary[instruction](block.data[target])
                continue
            binary = cx if instruction == "CX" else cz
            for offset in range(0, len(targets), 2):
                block = binary(
                    block.data[targets[offset]],
                    block.data[targets[offset + 1]],
                )
        return block

    state_key = ";".join(f"{stabilizer.sign}:" +
                         ",".join(f"{factor.pauli}{factor.operand}"
                                  for factor in stabilizer.factors)
                         for stabilizer in logical_stabilizers)
    digest = hashlib.sha256(state_key.encode("utf-8")).hexdigest()[:10]
    realization.__name__ = name or f"{code.name}_prepare_stabilizer_{digest}"
    realization.__qualname__ = realization.__name__
    realization.__annotations__ = {
        "block": patch[encoding],
        "return": patch[encoding],
    }
    return gadget(
        realization,
        implements=LogicalInstrumentRef(
            f"prepare_stabilizer_{digest}",
            0,
            1,
        ),
        name=realization.__name__,
        metadata={
            "preparation_kind": "stabilizer_state",
            "logical_stabilizers": state_key,
        },
    )


__all__ = [
    "css_memory_round",
    "logical_measure",
    "logical_pauli",
    "measure_x",
    "measure_z",
    "prepare_plus",
    "prepare_zero",
    "stabilizer_preparation",
]
