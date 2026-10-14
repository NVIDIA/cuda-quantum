# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Pinnacle generalized-bicycle implementation library.

Importing this module links stable Mark III definitions for the five published
Pinnacle processing blocks.  Preparations, one extraction round, and
transversal CX are ordinary named gadget values.  The WSC construction remains
a factory because its logical Pauli product is part of the program; that input
is a typed :class:`cudaq.logical.PauliProduct`, never a string grammar.

The H/S values are evidence, not realizations: exhaustive auditing finds no
canonical global H or S in the declared circulant/fold candidate family.  The
WSC construction has an exact algebraic certificate, while its current serial
bare-ancilla extractor deliberately retains unknown circuit distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..codes.pinnacle import (
    PINNACLE_GB_INSTANCES,
    PUBLISHED_GB_SEEDS,
    PinnacleGBInstance,
    pinnacle_gb,
    pinnacle_gb_instance,
)
from cudaq.logical.codes import (
    Code,
    Encoding,
)
from cudaq.logical.algebra.pauli import PauliProduct
from ..qec.rounds import RoundPolicy
from ._measurement import JointMeasurement, _measurement_profile
from ._wsc import _HighRateCSSBuilder


def _code(value) -> Code:
    if isinstance(value, Encoding):
        return value.code
    if isinstance(value, Code):
        return value
    raise TypeError(
        "Pinnacle definitions require a cudaq.logical.Code or cudaq.logical.Encoding"
    )


def _cycle_rounds(code: Code) -> int:
    rounds = code.metadata.get("logical_cycle_rounds")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        distance = getattr(code.d, "value", None)
        if not isinstance(distance, int) or distance <= 0:
            raise ValueError(
                "the code needs logical_cycle_rounds metadata or distance evidence"
            )
        rounds = distance
    return rounds


def _resolve_rounds(code: Code, rounds: int | RoundPolicy | None) -> int:
    if rounds is None:
        return _cycle_rounds(code)
    if isinstance(rounds, RoundPolicy):
        return rounds.resolve(code)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise TypeError(
            "rounds must be a positive int or cudaq.logical.RoundPolicy")
    return rounds


def _formal_terms(code: Code,
                  product: PauliProduct) -> tuple[tuple[int, str], ...]:
    if not isinstance(product, PauliProduct):
        raise TypeError("WSC measurement expects a cudaq.logical.PauliProduct")
    if product.identities:
        raise ValueError(
            "WSC measurement does not accept identity-covered ports")
    terms = []
    for factor in product.factors:
        logical = factor.operand
        if not isinstance(logical, int) or isinstance(logical, bool):
            raise TypeError(
                "WSC Pauli factors must use integer logical-port operands")
        if not 0 <= logical < code.k:
            raise ValueError(
                f"code {code.name} has no protected logical port {logical}")
        terms.append((logical, factor.pauli))
    return tuple(terms)


def _builder(value) -> _HighRateCSSBuilder:
    code = _code(value)
    return _HighRateCSSBuilder(value, logical_cycle_rounds=_cycle_rounds(code))


def prepare_all_one(value):
    """Build a named preparation of ``|1>`` on every protected port."""

    return _builder(value).prepare_one()


def prepare_all_minus(value):
    """Build a named preparation of ``|->`` on every protected port."""

    return _builder(value).prepare_minus()


def transversal_cx(value):
    """Build carrierwise CX between two blocks of one high-rate CSS code."""

    return _builder(value).transversal_cx()


def global_h_evidence(value):
    """Return the exact canonical-action audit for the declared H candidates."""

    return _builder(value).h_evidence()


def global_s_evidence(value):
    """Return the exact canonical-action audit for the declared S candidates."""

    return _builder(value).s_evidence()


def wsc_measurement(
    value,
    product: PauliProduct,
    *,
    rounds: int | RoundPolicy | None = None,
) -> JointMeasurement:
    """Build an exact within-block WSC measurement for ``product``.

    Formal integer operands select protected ports of ``value``. For example,
    ``cudaq.logical.X(0) @ cudaq.logical.Y(2)`` requests ``X`` on port 0 times
    ``Y`` on port 2.
    """

    code = _code(value)
    resolved_rounds = _resolve_rounds(code, rounds)
    plan = _builder(value).build_wsc_measurement(
        _formal_terms(code, product),
        rounds=resolved_rounds,
        sign=product.sign,
    )
    return JointMeasurement(
        product=product,
        realization=plan.gadget,
        analysis=_measurement_profile(plan,
                                      name=f"{plan.gadget.name}_analysis"),
        evidence=plan.evidence,
        rounds=resolved_rounds,
        data_code=code,
        _diagnostics=plan,
    )


@dataclass(frozen=True, slots=True)
class _NamedDefinitions:
    code: Code
    prepare_zero: Any
    prepare_one: Any
    prepare_plus: Any
    prepare_minus: Any
    memory_round: Any
    transversal_cx: Any
    cycle_rounds: int


def _named(code: Code) -> _NamedDefinitions:
    builder = _builder(code)
    return _NamedDefinitions(
        code=code,
        prepare_zero=builder.prepare_zero(),
        prepare_one=builder.prepare_one(),
        prepare_plus=builder.prepare_plus(),
        prepare_minus=builder.prepare_minus(),
        memory_round=builder.memory(1),
        transversal_cx=builder.transversal_cx(),
        cycle_rounds=_cycle_rounds(code),
    )


_gb30 = _named(pinnacle_gb("gb30"))
_gb62 = _named(pinnacle_gb("gb62"))
_gb126 = _named(pinnacle_gb("gb126"))
_gb254 = _named(pinnacle_gb("gb254"))
_gb510 = _named(pinnacle_gb("gb510"))

gb30 = _gb30.code
gb30_prepare_zero = _gb30.prepare_zero
gb30_prepare_one = _gb30.prepare_one
gb30_prepare_plus = _gb30.prepare_plus
gb30_prepare_minus = _gb30.prepare_minus
gb30_memory_round = _gb30.memory_round
gb30_transversal_cx = _gb30.transversal_cx
gb30_cycle_rounds = _gb30.cycle_rounds

gb62 = _gb62.code
gb62_prepare_zero = _gb62.prepare_zero
gb62_prepare_one = _gb62.prepare_one
gb62_prepare_plus = _gb62.prepare_plus
gb62_prepare_minus = _gb62.prepare_minus
gb62_memory_round = _gb62.memory_round
gb62_transversal_cx = _gb62.transversal_cx
gb62_cycle_rounds = _gb62.cycle_rounds

gb126 = _gb126.code
gb126_prepare_zero = _gb126.prepare_zero
gb126_prepare_one = _gb126.prepare_one
gb126_prepare_plus = _gb126.prepare_plus
gb126_prepare_minus = _gb126.prepare_minus
gb126_memory_round = _gb126.memory_round
gb126_transversal_cx = _gb126.transversal_cx
gb126_cycle_rounds = _gb126.cycle_rounds

gb254 = _gb254.code
gb254_prepare_zero = _gb254.prepare_zero
gb254_prepare_one = _gb254.prepare_one
gb254_prepare_plus = _gb254.prepare_plus
gb254_prepare_minus = _gb254.prepare_minus
gb254_memory_round = _gb254.memory_round
gb254_transversal_cx = _gb254.transversal_cx
gb254_cycle_rounds = _gb254.cycle_rounds

gb510 = _gb510.code
gb510_prepare_zero = _gb510.prepare_zero
gb510_prepare_one = _gb510.prepare_one
gb510_prepare_plus = _gb510.prepare_plus
gb510_prepare_minus = _gb510.prepare_minus
gb510_memory_round = _gb510.memory_round
gb510_transversal_cx = _gb510.transversal_cx
gb510_cycle_rounds = _gb510.cycle_rounds

__all__ = [
    "PINNACLE_GB_INSTANCES",
    "PUBLISHED_GB_SEEDS",
    "PinnacleGBInstance",
    "JointMeasurement",
    "pinnacle_gb",
    "pinnacle_gb_instance",
    "prepare_all_one",
    "prepare_all_minus",
    "transversal_cx",
    "global_h_evidence",
    "global_s_evidence",
    "wsc_measurement",
]
for _preset in ("gb30", "gb62", "gb126", "gb254", "gb510"):
    __all__.extend((
        _preset,
        f"{_preset}_prepare_zero",
        f"{_preset}_prepare_one",
        f"{_preset}_prepare_plus",
        f"{_preset}_prepare_minus",
        f"{_preset}_memory_round",
        f"{_preset}_transversal_cx",
        f"{_preset}_cycle_rounds",
    ))
