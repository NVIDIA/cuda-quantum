# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Module-linked MPP compilation from typed lattice-surgery primitives."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, Signature

from cudaq.logical.programs.decorators import objective as _objective
from cudaq.logical.ops._impl import all_false as _all_false
from cudaq.logical.ops._impl import extract_syndrome as _extract_syndrome
from cudaq.logical.ops._impl import mpp as _mpp
from cudaq.logical.types.values import logical_qubit as _logical_qubit
from cudaq.logical.gadgets import (
    GadgetDefinition,
    gadget as _gadget,
    patch,
)
from cudaq.logical.qec.lowering import QECLowering
from cudaq.logical.architecture.logical import capability
from cudaq.logical.algebra.pauli import Z as _Z
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.std import mpp as _mpp_objective
from cudaq.logical.codes import (
    Code,
    Encoding,
)
from ..rounds import RoundPolicy, from_code_distance

_PUBLIC_MODULE = "cudaq.logical.qec.lattice_surgery"


@dataclass(frozen=True, slots=True)
class AncillaPolicy:
    name: str


@dataclass(frozen=True, slots=True)
class YBasisPolicy:
    name: str


AncillaPolicy.__module__ = _PUBLIC_MODULE
YBasisPolicy.__module__ = _PUBLIC_MODULE

connected_ancilla = AncillaPolicy("connected_ancilla")
local_basis_change = YBasisPolicy("local_basis_change")


def _stage_values(result, count, *, name, outcome=False):
    values = result if isinstance(result, tuple) else (result,)
    expected = count + (1 if outcome else 0)
    if len(values) != expected:
        raise TypeError(
            f"lattice-surgery {name} primitive returned {len(values)} values; "
            f"expected {expected}")
    return values


def _generated_protocol(
    *,
    site,
    context,
    prepare,
    merge,
    measure,
    split,
    max_weight,
    ancilla,
    y_basis,
    rounds,
):
    x_mask = int(site.parameters.get("x_mask", 0))
    z_mask = int(site.parameters.get("z_mask", 0))
    sign = int(site.parameters.get("sign", 1))
    weight = (x_mask | z_mask).bit_count()
    if weight <= 0:
        raise ValueError("lattice-surgery MPP requires a nonidentity product")
    if weight > max_weight:
        raise ValueError(
            f"MPP weight {weight} exceeds lattice-surgery max_weight={max_weight}"
        )
    if sign not in (-1, 1):
        raise ValueError("MPP sign must be +1 or -1")

    blocks = tuple(
        dict.fromkeys(binding.block or binding.placement
                      for binding in context.placements))
    boundary_count = len(blocks) or site.input_arity
    encoding = context.encoding
    annotation = patch[encoding]
    parameters = tuple(
        Parameter(
            f"block{index}",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        ) for index in range(boundary_count))
    result_annotation = tuple[tuple([annotation] * boundary_count + [bool])]

    def generated(*values):
        values = _stage_values(prepare(*values), boundary_count, name="prepare")
        values = _stage_values(merge(*values), boundary_count, name="merge")
        measured = _stage_values(
            measure(*values),
            boundary_count,
            name="measure",
            outcome=True,
        )
        values, outcome = measured[:-1], measured[-1]
        # Surgery primitives measure the unsigned seam product. A negative
        # requested Pauli product has the complementary eigenvalue bit, so the
        # folded protocol must carry that sign in its executable dataflow, not
        # only in its name and metadata.
        if sign < 0:
            outcome = _all_false(outcome)
        values = _stage_values(split(*values), boundary_count, name="split")
        return (*values, outcome)

    product = f"x{x_mask:x}_z{z_mask:x}_{'m' if sign < 0 else 'p'}"
    generated.__name__ = f"{context.lowering.name}_{product}"
    generated.__qualname__ = generated.__name__
    generated.__module__ = context.lowering.provider.__module__
    generated.__signature__ = Signature(
        parameters,
        return_annotation=result_annotation,
    )
    hints = {parameter.name: annotation for parameter in parameters}
    hints["return"] = result_annotation
    resolved_rounds = rounds.resolve(context.code, context.policy)
    return ProtocolDefinition(
        generated,
        implements=_mpp_objective,
        name=generated.__name__,
        type_hints=hints,
        metadata={
            "compiler": "lattice_surgery.mpp",
            "x_mask": x_mask,
            "z_mask": z_mask,
            "sign": sign,
            "weight": weight,
            "rounds": resolved_rounds,
            "ancilla": ancilla.name,
            "y_basis": y_basis.name,
            "prepare": prepare.name,
            "merge": merge.name,
            "measure": measure.name,
            "split": split.name,
        },
    )


def mpp_compiler(
    *,
    code,
    max_weight: int,
    prepare,
    merge,
    measure,
    split,
    ancilla=connected_ancilla,
    y_basis=local_basis_change,
    rounds=from_code_distance,
    name: str | None = None,
    version="0.3.9",
):
    """Create a module-linked MPP compiler from typed surgery primitives."""

    if not isinstance(code, (Code, Encoding)):
        raise TypeError("lattice-surgery code= must be a Code or Encoding")
    if (not isinstance(max_weight, int) or isinstance(max_weight, bool) or
            max_weight <= 0):
        raise TypeError("lattice-surgery max_weight must be a positive int")
    primitives = (prepare, merge, measure, split)
    if any(not isinstance(value, (GadgetDefinition, ProtocolDefinition))
           for value in primitives):
        raise TypeError(
            "lattice-surgery primitives must be gadgets or protocols")
    if not isinstance(ancilla, AncillaPolicy):
        raise TypeError("ancilla= must be a lattice-surgery AncillaPolicy")
    if not isinstance(y_basis, YBasisPolicy):
        raise TypeError("y_basis= must be a lattice-surgery YBasisPolicy")
    if not isinstance(rounds, RoundPolicy):
        raise TypeError(
            "rounds= must be a cudaq.logical.qec.rounds.RoundPolicy")

    compiler_name = name or f"{code.name}_lattice_surgery_mpp"

    def compile_site(site, context):
        return _generated_protocol(
            site=site,
            context=context,
            prepare=prepare,
            merge=merge,
            measure=measure,
            split=split,
            max_weight=max_weight,
            ancilla=ancilla,
            y_basis=y_basis,
            rounds=rounds,
        )

    compile_site.__name__ = f"compile_{compiler_name}"
    compile_site.__qualname__ = compile_site.__name__
    compile_site.__module__ = _PUBLIC_MODULE
    return QECLowering(
        compile_site,
        objective_family="pauli_product_measurement",
        codes=(code,),
        requires=(capability.lattice_surgery,),
        dependencies=primitives,
        plugin="cudaq.logical.providers.mpp_compiler",
        version=str(version),
        name=compiler_name,
        policy_schema={
            "rounds": "positive int",
            "max_weight": max_weight
        },
        metadata={
            "ancilla": ancilla.name,
            "y_basis": y_basis.name,
            "rounds": rounds.name,
        },
    )


def _surgery_stage_provider(
    a: _logical_qubit,
    b: _logical_qubit,
) -> tuple[_logical_qubit, _logical_qubit]:
    return a, b


_surgery_stage_provider.__module__ = _PUBLIC_MODULE
_surgery_stage = _objective(name="qlx_surgery_stage")(_surgery_stage_provider)


def _surgery_zz_provider(
    a: _logical_qubit,
    b: _logical_qubit,
) -> tuple[_logical_qubit, _logical_qubit, bool]:
    a, b, outcome = _mpp(_Z(a) @ _Z(b))
    return a, b, outcome


_surgery_zz_provider.__module__ = _PUBLIC_MODULE
_surgery_zz = _objective(name="qlx_surgery_zz")(_surgery_zz_provider)


@dataclass(frozen=True, slots=True)
class SurgeryPrimitives:
    """The prepare/merge/measure/split gadget set one MPP compiler consumes."""

    prepare: GadgetDefinition
    merge: GadgetDefinition
    measure: GadgetDefinition
    split: GadgetDefinition


SurgeryPrimitives.__module__ = _PUBLIC_MODULE


def surface_primitives(value, *, name_prefix: str | None = None):
    """Generate a CSS surgery primitive set for one code or encoding."""

    if isinstance(value, Encoding):
        encoding = value
    elif isinstance(value, Code):
        encoding = value.default_encoding
    else:
        raise TypeError("surface_primitives requires a Code or Encoding")
    code = encoding.code
    prefix = name_prefix or f"{code.name}_surgery"

    def stage(role, record_prefix):

        def realization(a, b):
            a, _ = _extract_syndrome(a, record=f"{record_prefix}_a")
            b, _ = _extract_syndrome(b, record=f"{record_prefix}_b")
            return a, b

        realization.__name__ = f"{prefix}_{role}"
        realization.__qualname__ = realization.__name__
        realization.__module__ = _PUBLIC_MODULE
        realization.__annotations__ = {
            "a": patch[encoding],
            "b": patch[encoding],
            "return": tuple[patch[encoding], patch[encoding]],
        }
        return _gadget(
            realization,
            implements=_surgery_stage,
            name=realization.__name__,
        )

    def seam_measure():

        def realization(a, b):
            return _mpp(_Z(a[0]) @ _Z(b[0]))

        realization.__name__ = f"{prefix}_measure"
        realization.__qualname__ = realization.__name__
        realization.__module__ = _PUBLIC_MODULE
        realization.__annotations__ = {
            "a": patch[encoding],
            "b": patch[encoding],
            "return": tuple[patch[encoding], patch[encoding], bool],
        }
        return _gadget(
            realization,
            implements=_surgery_zz,
            name=realization.__name__,
        )

    return SurgeryPrimitives(
        prepare=stage("prepare", "pre"),
        merge=stage("merge", "merge"),
        measure=seam_measure(),
        split=stage("split", "split"),
    )


mpp_compiler.__module__ = _PUBLIC_MODULE
surface_primitives.__module__ = _PUBLIC_MODULE

__all__ = [
    "AncillaPolicy",
    "SurgeryPrimitives",
    "YBasisPolicy",
    "connected_ancilla",
    "local_basis_change",
    "mpp_compiler",
    "surface_primitives",
]
