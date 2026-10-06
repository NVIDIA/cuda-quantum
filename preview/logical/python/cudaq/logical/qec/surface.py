# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Distance-indexed rotated-surface-code implementations.

``surface[d]`` returns one cached, inspectable Mark III definition set.  The
distance is data, not part of the Python symbol names, so the same application
shape works at every supported odd distance.  Bind the definitions an
application offers at module scope when they should participate in automatic
QEC selection.

Joint products are ordinary :class:`cudaq.logical.PauliProduct` values. Their returned
gadgets take the temporary ancilla patch explicitly, keeping allocation and
ownership visible in the containing gadget or protocol.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from ..codes import Surface as SurfaceCode
from ..gadgets import (
    css_memory_round,
    logical_measure,
    prepare_plus,
    prepare_zero,
)
from ..qec.lattice_surgery import (
    SurgeryPrimitives,
    connected_ancilla,
    local_basis_change,
    mpp_compiler,
    surface_primitives,
)
from cudaq.logical.architecture.physical_definition import Basis
from cudaq.logical.codes import Code
from cudaq.logical.gadgets import GadgetDefinition
from cudaq.logical.algebra.pauli import PauliProduct
from cudaq.logical.qec.lowering import QECLowering
from ..protocols.actual_cultivation import (
    cultivate_t_d3_to_matchable_d6,
    prepare_cultivated_t,
)
from ..qec.rounds import RoundPolicy, from_code_distance
from ._measurement import JointMeasurement, _measurement_profile
from ._surface_deformations import _SurfaceBuilder


def _surface_rounds(code: Code, rounds: int | RoundPolicy | None) -> int:
    if rounds is None:
        return from_code_distance.resolve(code)
    if isinstance(rounds, RoundPolicy):
        return rounds.resolve(code)
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds <= 0:
        raise TypeError(
            "rounds must be a positive int or cudaq.logical.RoundPolicy")
    return rounds


def _surface_terms(product: PauliProduct) -> tuple[str, ...]:
    if not isinstance(product, PauliProduct):
        raise TypeError(
            "surface joint measurement expects a cudaq.logical.PauliProduct")
    if product.identities:
        raise ValueError(
            "surface joint measurement does not accept identity-covered patches"
        )
    operands = tuple(factor.operand for factor in product.factors)
    if any(not isinstance(operand, int) or isinstance(operand, bool)
           for operand in operands):
        raise TypeError(
            "surface Pauli factors must use consecutive integer patch operands")
    if operands != tuple(range(len(operands))):
        raise ValueError(
            "surface Pauli factors must cover consecutive patches 0..arity-1")
    return tuple(factor.pauli for factor in product.factors)


@lru_cache(maxsize=None)
def _joint_measurement(
    distance: int,
    product: PauliProduct,
    rounds: int,
) -> JointMeasurement:
    code = SurfaceCode[distance]
    plan = _SurfaceBuilder(distance).build_joint_measurement(
        _surface_terms(product),
        rounds=rounds,
        sign=product.sign,
    )
    return JointMeasurement(
        product=product,
        realization=plan.gadget,
        analysis=_measurement_profile(plan,
                                      name=f"{plan.gadget.name}_analysis"),
        evidence=plan.evidence,
        rounds=rounds,
        data_code=code,
        auxiliary_code=plan.auxiliary_code,
        _diagnostics=plan,
    )


@dataclass(frozen=True, slots=True)
class SurfaceDefinitions:
    """The complete reusable definition set for one surface-code distance."""

    distance: int
    code: Code
    prepare_zero: GadgetDefinition
    prepare_one: GadgetDefinition
    prepare_plus: GadgetDefinition
    prepare_minus: GadgetDefinition
    memory_round: GadgetDefinition
    measure_z: GadgetDefinition
    measure_x: GadgetDefinition
    h: GadgetDefinition
    fold_s: GadgetDefinition
    transversal_cx: GadgetDefinition
    surgery: SurgeryPrimitives
    mpp: QECLowering

    def joint_measurement(
        self,
        product: PauliProduct,
        *,
        rounds: int | RoundPolicy | None = None,
    ) -> JointMeasurement:
        """Build a live-owner WSC measurement for a typed patch product.

        Integer operands label separate input patches.  For example,
        ``cudaq.logical.Z(0) @ cudaq.logical.Z(1)`` requests a two-patch ZZ
        measurement. The
        realization also takes and consumes one patch of ``auxiliary_code``.
        """

        return _joint_measurement(
            self.distance,
            product,
            _surface_rounds(self.code, rounds),
        )


@lru_cache(maxsize=None)
def _definitions(distance: int) -> SurfaceDefinitions:
    builder = _SurfaceBuilder(distance)
    code = builder.code
    surgery = surface_primitives(code)
    return SurfaceDefinitions(
        distance=distance,
        code=code,
        prepare_zero=prepare_zero(code),
        prepare_one=builder.prepare_one(),
        prepare_plus=prepare_plus(code),
        prepare_minus=builder.prepare_minus(),
        memory_round=css_memory_round(code),
        measure_z=logical_measure(code, basis=Basis.Z),
        measure_x=logical_measure(code, basis=Basis.X),
        h=builder.logical_h(),
        fold_s=builder.fold_s(),
        transversal_cx=builder.transversal_cx(),
        surgery=surgery,
        mpp=mpp_compiler(
            code=code,
            max_weight=4,
            prepare=surgery.prepare,
            merge=surgery.merge,
            measure=surgery.measure,
            split=surgery.split,
            ancilla=connected_ancilla,
            y_basis=local_basis_change,
            rounds=from_code_distance,
            name=f"{code.name}_mpp",
        ),
    )


class SurfaceFamily:
    """Indexable family of cached surface implementation definitions."""

    __slots__ = ()

    def __getitem__(self, distance: int) -> SurfaceDefinitions:
        return _definitions(distance)

    def __repr__(self) -> str:
        return "surface"


surface = SurfaceFamily()

__all__ = [
    "SurfaceDefinitions",
    "SurfaceFamily",
    "surface",
    "JointMeasurement",
    "prepare_cultivated_t",
    "cultivate_t_d3_to_matchable_d6",
    "adjacent_rounds",
    "css_memory_round",
    "logical_measure",
    "mpp_compiler",
    "surface_primitives",
]
