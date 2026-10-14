# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Versioned arbitrary-rotation synthesis for generated QEC protocols.

The pure :func:`pygridsynth_rz` entry point turns one static Rz angle into a
normalized H/S/T word plus numerical and provenance evidence.  The
:func:`pygridsynth_rpp_compiler` factory wraps that word in a code-specific P2
protocol: basis changes map a Pauli product to Z, a CNOT ladder accumulates its
parity, the synthesized word acts on the accumulator, and the ladder/basis
changes are uncomputed.

``pygridsynth`` is optional and imported only when an off-lattice angle selects
the synthesis strategy.  Exact multiples of pi/4 use the built-in exact word
table and do not require the optional package.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version as package_version
from inspect import Parameter, Signature

from . import product_rotation
from cudaq.logical.codes import (
    Code,
    Encoding,
)
from cudaq.logical.gadgets import (
    GadgetDefinition,
    patch,
)
from cudaq.logical.qec.lowering import GeneratedQECArtifact
from cudaq.logical.protocols.definition import ProtocolDefinition
from cudaq.logical.std import pauli_rotation

_ANGLE_TOLERANCE = 1.0e-12
_ERROR_METRIC = "projective_operator_norm"


@dataclass(frozen=True, slots=True)
class CliffordTWord:
    """One normalized H/S/T word and its synthesis certificate."""

    gates: tuple[str, ...]
    requested_precision: float
    achieved_error: float
    algorithm: str
    implementation: str
    implementation_version: str
    error_metric: str = _ERROR_METRIC

    @property
    def text(self) -> str:
        return " ".join(self.gates)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    @property
    def t_count(self) -> int:
        return self.gates.count("t")

    @property
    def clifford_count(self) -> int:
        return len(self.gates) - self.t_count

    def evidence(self) -> dict[str, str | int | float]:
        return {
            "synthesis_algorithm": self.algorithm,
            "synthesis_implementation": self.implementation,
            "synthesis_version": self.implementation_version,
            "synthesis_error_metric": self.error_metric,
            "synthesis_requested_precision": self.requested_precision,
            "synthesis_achieved_error": self.achieved_error,
            "synthesis_word_sha256": self.sha256,
            "synthesis_t_count": self.t_count,
            "synthesis_clifford_count": self.clifford_count,
        }


def _validate_precision(precision) -> float:
    if (not isinstance(precision,
                       (int, float)) or isinstance(precision, bool) or
            not math.isfinite(float(precision)) or
            not 0.0 < float(precision) < 1.0):
        raise ValueError("PyGridSynth precision must be finite and in (0, 1)")
    return float(precision)


def _exact_word(angle: float) -> tuple[str, ...] | None:
    """Return an exact H/S/T word for a multiple of pi/4, up to phase."""

    quarter = angle / (math.pi / 4.0)
    nearest = round(quarter)
    if not math.isclose(quarter, nearest, rel_tol=0.0,
                        abs_tol=_ANGLE_TOLERANCE):
        return None
    value = nearest % 8
    # T-dagger is S-dagger T = S^3 T.  Keeping only H/S/T means every T in the
    # word has the same exact resource-injection contract.
    exact = {
        0: (),
        1: ("t",),
        2: ("s",),
        3: ("s", "t"),
        4: ("s", "s"),
        5: ("s", "s", "t"),
        6: ("s", "s", "s"),
        7: ("s", "s", "s", "t"),
    }
    return exact[value]


def _exact_word_error(angle: float) -> float:
    nearest_angle = round(angle / (math.pi / 4.0)) * (math.pi / 4.0)
    return 2.0 * abs(math.sin((angle - nearest_angle) / 4.0))


def _high_precision_word_error(mpmath, gates, angle) -> float:
    """Measure the emitted chronological word, independently of GridSynth."""

    root_half = mpmath.sqrt(mpmath.mpf("0.5"))
    matrices = {
        "h": ((root_half, root_half), (root_half, -root_half)),
        "s": ((1, 0), (0, 1j)),
        "t": ((1, 0), (0, mpmath.exp(0.25j * mpmath.pi))),
    }
    result = ((1, 0), (0, 1))
    for gate in gates:
        left = matrices[gate]
        result = tuple(
            tuple(
                sum(left[row][inner] * result[inner][column]
                    for inner in range(2))
                for column in range(2))
            for row in range(2))
    target = (
        (mpmath.exp(-0.5j * angle), 0),
        (0, mpmath.exp(0.5j * angle)),
    )

    def determinant(matrix):
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    phase = 0.5 * mpmath.arg(determinant(result) / determinant(target))

    def distance(candidate_phase):
        factor = mpmath.exp(1.0j * candidate_phase)
        delta = tuple(
            tuple(result[row][column] - factor * target[row][column]
                  for column in range(2))
            for row in range(2))
        h00 = abs(delta[0][0])**2 + abs(delta[1][0])**2
        h11 = abs(delta[0][1])**2 + abs(delta[1][1])**2
        h01 = (delta[0][0].conjugate() * delta[0][1] +
               delta[1][0].conjugate() * delta[1][1])
        trace = h00 + h11
        zero = mpmath.mpf("0")
        det = max(zero, h00 * h11 - abs(h01)**2)
        discriminant = max(zero, trace * trace - 4 * det)
        return mpmath.sqrt(max(zero, (trace + mpmath.sqrt(discriminant)) / 2))

    # Taking the square root of the determinant phase has two branches.
    return float(min(distance(phase), distance(phase + mpmath.pi)))


def _pygridsynth_version() -> str:
    try:
        return package_version("pygridsynth")
    except PackageNotFoundError:
        return "unknown"


def pygridsynth_rz(angle: float, precision: float) -> CliffordTWord:
    """Synthesize ``Rz(angle)`` into a normalized H/S/T word.

    The returned error is independently measured in projective operator norm,
    so global phase does not turn an otherwise equivalent word into a failure.
    """

    angle = float(angle)
    if not math.isfinite(angle):
        raise ValueError("PyGridSynth angle must be finite")
    precision = _validate_precision(precision)
    exact = _exact_word(angle)
    exact_error = _exact_word_error(angle) if exact is not None else None
    if exact is not None and exact_error <= precision:
        return CliffordTWord(
            exact,
            requested_precision=precision,
            achieved_error=exact_error,
            algorithm="exact-clifford-t",
            implementation="qlx",
            implementation_version="0.3",
        )

    try:
        import mpmath  # noqa: PLC0415
        from pygridsynth.gridsynth import gridsynth  # noqa: PLC0415
        from pygridsynth.synthesis_of_cliffordT import (  # noqa: PLC0415
            decompose_domega_unitary,)
    except ImportError as error:
        raise ImportError("arbitrary-angle RPP synthesis requires the optional "
                          "PyGridSynth dependency; install "
                          "`cudaq-logical[synthesis]`") from error

    work_digits = max(50, math.ceil(-math.log10(precision)) + 20)
    with mpmath.workdps(work_digits):
        unitary = gridsynth(mpmath.mpf(str(angle)), mpmath.mpf(str(precision)))
        circuit = tuple(decompose_domega_unitary(unitary, [0]))

    # PyGridSynth 2.x calls Pauli X ``SXGate``.  Normalize X and inverse phase
    # gates into H/S/T so the generated protocol needs only four code-specific
    # primitives: H, S, CX, and positive-T injection.  Its circuit sequence is
    # a left-to-right matrix product, so chronological execution is reversed.
    mapping = {
        "TGate": ("t",),
        "TInvGate": ("s", "s", "s", "t"),
        "TdgGate": ("s", "s", "s", "t"),
        "HGate": ("h",),
        "SGate": ("s",),
        "SInvGate": ("s", "s", "s"),
        "SdgGate": ("s", "s", "s"),
        "SXGate": ("h", "s", "s", "h"),
        "XGate": ("h", "s", "s", "h"),
        "WGate": (),
    }
    gates = []
    for gate in reversed(circuit):
        name = type(gate).__name__
        try:
            gates.extend(mapping[name])
        except KeyError as error:
            raise RuntimeError(
                f"PyGridSynth emitted unsupported gate type {name!r}"
            ) from error

    with mpmath.workdps(work_digits):
        achieved_error = _high_precision_word_error(mpmath, gates,
                                                    mpmath.mpf(str(angle)))
    if achieved_error > precision:
        raise RuntimeError(
            "PyGridSynth word failed the requested projective operator-norm "
            f"bound: achieved {achieved_error:.6e}, requested {precision:.6e}")
    return CliffordTWord(
        tuple(gates),
        requested_precision=precision,
        achieved_error=achieved_error,
        algorithm="ross-selinger-gridsynth",
        implementation="pygridsynth",
        implementation_version=_pygridsynth_version(),
    )


def _require_primitive(value, name):
    if not isinstance(value, (GadgetDefinition, ProtocolDefinition)):
        raise TypeError(f"PyGridSynth {name}= must be a gadget or protocol")
    return value


def _generated_rpp_protocol(*, site, context, word, h, s, cx, t, strategy):
    x_mask = int(site.parameters.get("x_mask", 0))
    z_mask = int(site.parameters.get("z_mask", 0))
    sign = int(site.parameters.get("sign", 1))
    support_mask = x_mask | z_mask
    factor_count = len(context.placements)
    if support_mask == 0:
        raise ValueError("RPP synthesis requires a nonidentity Pauli product")
    if support_mask.bit_length() > factor_count:
        raise ValueError("RPP masks exceed the placed logical operand count")

    block_keys = tuple(
        binding.block or binding.placement for binding in context.placements)
    if len(set(block_keys)) != len(block_keys):
        raise NotImplementedError(
            "PyGridSynth RPP synthesis currently requires at most one product "
            "factor per encoded patch")

    encoding = context.encoding
    annotation = patch[encoding]
    parameters = tuple(
        Parameter(
            f"block{index}",
            Parameter.POSITIONAL_OR_KEYWORD,
            annotation=annotation,
        ) for index in range(factor_count))
    result_annotation = tuple[tuple([annotation] * factor_count)]
    active = tuple(
        index for index in range(factor_count) if support_mask & (1 << index))

    def generated(*blocks):
        values = list(blocks)

        def apply_one(definition, index, count=1):
            for _ in range(count):
                values[index] = definition(values[index])

        def ladder(left, right):
            values[left], values[right] = cx(values[left], values[right])

        # C(P) maps every active X/Y factor to Z before parity accumulation.
        for index in active:
            x_bit = bool(x_mask & (1 << index))
            z_bit = bool(z_mask & (1 << index))
            if x_bit and not z_bit:
                apply_one(h, index)
            elif x_bit and z_bit:
                apply_one(s, index, 3)
                apply_one(h, index)

        for left, right in zip(active, active[1:]):
            ladder(left, right)

        accumulator = active[-1]
        for gate in word.gates:
            apply_one({"h": h, "s": s, "t": t}[gate], accumulator)

        for left, right in reversed(tuple(zip(active, active[1:]))):
            ladder(left, right)

        for index in active:
            x_bit = bool(x_mask & (1 << index))
            z_bit = bool(z_mask & (1 << index))
            if x_bit and not z_bit:
                apply_one(h, index)
            elif x_bit and z_bit:
                apply_one(h, index)
                apply_one(s, index)
        return tuple(values)

    generated.__name__ = (
        f"{context.lowering.name}_{strategy}_{word.sha256[:12]}")
    generated.__qualname__ = generated.__name__
    generated.__module__ = context.lowering.provider.__module__
    generated.__signature__ = Signature(parameters,
                                        return_annotation=result_annotation)
    hints = {parameter.name: annotation for parameter in parameters}
    hints["return"] = result_annotation
    metadata = {
        "compiler": "synthesis.pygridsynth_rpp",
        "strategy": strategy,
        "x_mask": x_mask,
        "z_mask": z_mask,
        "sign": sign,
        "word": word.text,
        **word.evidence(),
    }
    return ProtocolDefinition(
        generated,
        implements=pauli_rotation,
        name=generated.__name__,
        type_hints=hints,
        metadata=metadata,
    )


@dataclass(frozen=True, slots=True)
class _RPPStrategy:
    h: GadgetDefinition | ProtocolDefinition
    s: GadgetDefinition | ProtocolDefinition
    cx: GadgetDefinition | ProtocolDefinition
    t: GadgetDefinition | ProtocolDefinition
    approximate: bool

    @property
    def dependencies(self):
        return (self.h, self.s, self.cx, self.t)

    def __call__(self, site, context):
        angle = float(site.parameters["angle"])
        sign = int(site.parameters.get("sign", 1))
        effective_angle = angle * sign
        if self.approximate:
            precision = site.parameters.get("precision",
                                            context.policy.get("rpp_precision"))
            if precision is None:
                raise ValueError("PyGridSynth synthesis requires a precision")
            word = pygridsynth_rz(effective_angle, float(precision))
            strategy = "synthesis"
        else:
            gates = _exact_word(effective_angle)
            if gates is None:
                raise ValueError(
                    "exact RPP strategy requires a multiple of pi/4")
            requested_precision = float(
                site.parameters.get("precision", _ANGLE_TOLERANCE))
            achieved_error = _exact_word_error(effective_angle)
            if achieved_error > requested_precision:
                raise ValueError(
                    "exact RPP strategy cannot meet the requested precision; "
                    "reduce the compiler angle_tolerance or force synthesis")
            word = CliffordTWord(
                gates,
                requested_precision=requested_precision,
                achieved_error=achieved_error,
                algorithm="exact-clifford-t",
                implementation="qlx",
                implementation_version="0.3",
            )
            strategy = "exact"
        protocol = _generated_rpp_protocol(
            site=site,
            context=context,
            word=word,
            h=self.h,
            s=self.s,
            cx=self.cx,
            t=self.t,
            strategy=strategy,
        )
        return GeneratedQECArtifact(protocol, word.evidence())


def pygridsynth_strategy(*, h, s, cx, t):
    """Return an explicit ``synthesis=`` provider for an RPP compiler."""

    return _RPPStrategy(
        _require_primitive(h, "h"),
        _require_primitive(s, "s"),
        _require_primitive(cx, "cx"),
        _require_primitive(t, "t"),
        approximate=True,
    )


def pygridsynth_rpp_compiler(
    *,
    code,
    h,
    s,
    cx,
    t,
    native=None,
    rotation_state=None,
    plugin="cudaq.logical.synthesis.pygridsynth",
    version="1.0.0",
    name=None,
    angle_tolerance=_ANGLE_TOLERANCE,
):
    """Create a complete exact-plus-PyGridSynth RPP compiler for one code."""

    if not isinstance(code, (Code, Encoding)):
        raise TypeError("PyGridSynth RPP code= must be a Code or Encoding")
    primitives = {
        key: _require_primitive(value, key) for key, value in {
            "h": h,
            "s": s,
            "cx": cx,
            "t": t
        }.items()
    }
    exact = _RPPStrategy(**primitives, approximate=False)
    approximate = _RPPStrategy(**primitives, approximate=True)
    return product_rotation.compiler(
        code=code,
        clifford=exact,
        t_injection=exact,
        native=native,
        rotation_state=rotation_state,
        synthesis=approximate,
        plugin=plugin,
        version=version,
        name=name or f"{getattr(code, 'name', 'code')}_pygridsynth_rpp",
        angle_tolerance=angle_tolerance,
    )


__all__ = [
    "CliffordTWord",
    "pygridsynth_rz",
    "pygridsynth_strategy",
    "pygridsynth_rpp_compiler",
]
