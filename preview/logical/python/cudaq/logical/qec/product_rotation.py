# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import math
from dataclasses import dataclass

from cudaq.logical.gadgets import GadgetDefinition
from cudaq.logical.architecture.capabilities import (
    NATIVE_PAULI_PRODUCT_ROTATION,
    PhysicalCapability,
)
from cudaq.logical.qec.lowering import (
    GeneratedQECArtifact,
    QECLowering,
)
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.std import pauli_rotation


def _fixed_dependencies(strategies):
    dependencies = []
    for strategy in strategies:
        if isinstance(strategy, (GadgetDefinition, ProtocolDefinition)):
            dependencies.append(strategy)
        else:
            dependencies.extend(getattr(strategy, "dependencies", ()))
    return tuple(dict.fromkeys(dependencies))


def _resolve(strategy, site, context):
    if isinstance(strategy, (GadgetDefinition, ProtocolDefinition)):
        return strategy, {}
    if callable(strategy):
        result = strategy(site, context)
        if isinstance(result, GeneratedQECArtifact):
            if not isinstance(result.definition,
                              (GadgetDefinition, ProtocolDefinition)):
                raise TypeError(
                    "generated RPP strategy artifact must contain a gadget or "
                    "protocol")
            return result.definition, dict(result.specialization)
        if isinstance(result, (GadgetDefinition, ProtocolDefinition)):
            return result, {}
        raise TypeError(
            "RPP strategy provider must return a gadget, protocol, or "
            "GeneratedQECArtifact")
    raise TypeError("RPP strategy must be a gadget, protocol, or provider")


def _rpp_objective_adapter(definition, site, selected_name):
    """Declare the exact RPP objective around a selected implementation step."""

    def generated(*blocks):
        return definition(*blocks)

    generated.__name__ = (
        f"{definition.name}_{selected_name}_{site.symbol}_rpp")
    generated.__qualname__ = generated.__name__
    generated.__module__ = definition.provider.__module__
    generated.__signature__ = definition.signature
    return ProtocolDefinition(
        generated,
        implements=pauli_rotation,
        name=generated.__name__,
        type_hints=definition.type_hints,
        metadata={
            "compiler": "cudaq.logical.product_rotation",
            "strategy": selected_name,
            "implementation": definition.name,
        },
    )


def _has_native_rpp(architecture) -> bool:
    if architecture is None:
        return False
    return any(
        any((value.key if isinstance(value, PhysicalCapability) else value
            ) == NATIVE_PAULI_PRODUCT_ROTATION.key
            for value in resource_class.capabilities) or any(
                getattr(action, "name", action) == "rpp"
                for action in resource_class.native_actions)
        for resource_class in architecture.resource_classes)


def _quarter_turn(angle: float, *, tolerance: float) -> int | None:
    value = angle / (math.pi / 4.0)
    nearest = round(value)
    if math.isclose(value, nearest, rel_tol=0.0, abs_tol=tolerance):
        return nearest
    return None


def _exact_quarter_turn(numer: int, denom: int) -> int | None:
    """Quarter-turn index of the exact angle ``(numer/denom) * pi``, or ``None``.

    ``theta = (numer/denom) * pi`` is ``k * pi/4`` iff ``denom`` divides
    ``4*numer``; the index is then ``k = 4*numer/denom``.  Authoritative -- no
    tolerance -- so a symbol-authored ``cudaq.logical.pi/4`` is the magic rotation by
    construction, never a near-lattice float that a tolerance might snap or miss.
    """
    if denom <= 0:
        return None
    if (4 * numer) % denom != 0:
        return None
    return (4 * numer) // denom


def signed_angle(parameters) -> float:
    """The effective signed rotation angle ``sign * |theta|``.

    A rotation stores its Pauli/angle sign canonically in the ``sign`` field
    with a nonnegative ``angle`` magnitude (uniform with ``mpp``). This is the
    sole sanctioned reader of the angle: every consumer asks for the *signed*
    angle so the sign can never be silently dropped (the original defect).
    """
    return float(parameters["angle"]) * int(parameters.get("sign", 1))


def signed_pi_fraction(parameters):
    """The exact signed angle as ``(numer, denom)`` units of pi, or ``None``.

    The ``sign`` field is folded into the numerator, so the returned pair is the
    self-contained signed rational a consumer can classify directly.
    """
    numer = parameters.get("angle_pi_numer")
    denom = parameters.get("angle_pi_denom")
    if numer is None or denom is None:
        return None
    return (int(numer) * int(parameters.get("sign", 1)), int(denom))


def _rz_distance(left: float, right: float) -> float:
    """Projective operator-norm distance between two same-axis rotations."""

    return 2.0 * abs(math.sin((left - right) / 4.0))


@dataclass(frozen=True, slots=True)
class ProductRotationCompiler:
    """Documentation-friendly wrapper around one linked RPP lowering."""

    lowering: QECLowering

    def materialize(self, module=None):
        return self.lowering.materialize(module=module)

    def __getattr__(self, name):
        return getattr(self.lowering, name)


def compiler(
    *,
    code,
    clifford=None,
    t_injection=None,
    native=None,
    rotation_state=None,
    synthesis=None,
    plugin="cudaq.logical.product_rotation",
    version="1.0.0",
    name=None,
    angle_tolerance=1e-12,
):
    """Create a versioned device-selected Pauli-product-rotation compiler.

    A strategy is a fixed gadget/protocol or a pure ``(site, context)``
    provider. Exact multiples of pi/2 select ``clifford``; odd multiples of
    pi/4 select ``t_injection``. Other angles prefer native RPP, then a
    rotation-state realization, then an explicit approximation provider.
    """

    if (not isinstance(angle_tolerance, (int, float)) or
            isinstance(angle_tolerance, bool) or angle_tolerance <= 0):
        raise TypeError("angle_tolerance must be positive")
    strategies = {
        "clifford": clifford,
        "t_injection": t_injection,
        "native": native,
        "rotation_state": rotation_state,
        "synthesis": synthesis,
    }
    if all(value is None for value in strategies.values()):
        raise ValueError(
            "product rotation compiler requires at least one strategy")

    def provider(site, context):
        if site.objective_family != "pauli_product_rotation":
            raise ValueError(
                "product rotation compiler received a non-RPP site")
        if site.parameters.get("dynamic_angle"):
            raise NotImplementedError(
                "compile-time RPP synthesis requires a static angle; use a "
                "dynamic native realization for runtime angles")
        if "angle" not in site.parameters:
            raise ValueError("RPP action site is missing its canonical angle")
        effective_angle = signed_angle(site.parameters)
        precision = site.parameters.get("precision",
                                        context.policy.get("rpp_precision"))
        if precision is not None:
            precision = float(precision)
            if not math.isfinite(precision) or precision <= 0:
                raise ValueError(
                    "RPP synthesis precision must be finite and positive")

        requested = context.policy.get("rpp_strategy", "auto")
        if requested != "auto":
            if requested not in strategies or strategies[requested] is None:
                raise ValueError(
                    f"requested RPP strategy {requested!r} is unavailable")
            if requested == "native" and not _has_native_rpp(context.physical):
                raise ValueError(
                    "requested native RPP strategy is not supported by the architecture"
                )
            if requested == "synthesis" and precision is None:
                raise ValueError(
                    "approximate RPP synthesis requires precision= or "
                    "policy['rpp_precision']")
            selected_name = requested
        else:
            exact = signed_pi_fraction(site.parameters)
            if exact is not None:
                # Symbol-authored angle: the quarter-turn is exact (distance 0),
                # so the precision-distance override never applies.
                quarter = _exact_quarter_turn(*exact)
            else:
                quarter = _quarter_turn(effective_angle,
                                        tolerance=float(angle_tolerance))
                if (quarter is not None and precision is not None and
                        _rz_distance(effective_angle, quarter *
                                     (math.pi / 4.0)) > precision):
                    quarter = None
            selected_name = None
            if quarter is not None:
                if quarter % 2 == 0 and clifford is not None:
                    selected_name = "clifford"
                elif quarter % 2 and t_injection is not None:
                    selected_name = "t_injection"
            if (selected_name is None and native is not None and
                    _has_native_rpp(context.physical)):
                selected_name = "native"
            if selected_name is None and rotation_state is not None:
                selected_name = "rotation_state"
            if selected_name is None and synthesis is not None:
                if precision is None:
                    raise ValueError(
                        "approximate RPP synthesis requires precision= or "
                        "policy['rpp_precision']")
                selected_name = "synthesis"
            if selected_name is None:
                raise NotImplementedError(
                    "no RPP strategy covers this angle and device")

        definition, strategy_evidence = _resolve(strategies[selected_name],
                                                 site, context)
        definition = _rpp_objective_adapter(definition, site, selected_name)
        evidence = {
            **strategy_evidence,
            "rpp_strategy": selected_name,
            "angle_convention": "exp(-i*theta*P/2)",
            "effective_angle": effective_angle,
        }
        if precision is not None:
            evidence["precision"] = precision
        return GeneratedQECArtifact(definition, evidence)

    provider.__name__ = name or f"{getattr(code, 'name', 'code')}_rpp_compiler"
    return QECLowering(
        provider,
        objective_family="pauli_product_rotation",
        codes=(code,),
        dependencies=_fixed_dependencies(strategies.values()),
        plugin=plugin,
        version=version,
        name=provider.__name__,
        policy_schema={
            "rpp_strategy":
                ("auto|clifford|t_injection|native|rotation_state|synthesis"),
            "rpp_precision": "positive float",
        },
        metadata={
            "angle_convention": "exp(-i*theta*P/2)",
            "strategy_order": "exact,native,rotation_state,synthesis",
        },
    )


__all__ = ["ProductRotationCompiler", "compiler"]
