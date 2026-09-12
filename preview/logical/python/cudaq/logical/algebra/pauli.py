# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable


def _operand_key(operand: Any) -> tuple[Any, ...]:
    key = getattr(operand, "semantic_ref", None)
    if key is not None:
        return tuple(key)
    if isinstance(operand, int):
        return ("formal", operand)
    return ("object", id(operand))


@dataclass(frozen=True, slots=True)
class PauliFactor:
    operand: Any
    pauli: str

    @property
    def key(self) -> tuple[Any, ...]:
        return _operand_key(self.operand)


@dataclass(frozen=True, slots=True)
class PauliProduct:
    """Canonical Hermitian Pauli product over bound values or formal ports.

    ``identities`` records operands covered by explicit :func:`I` factors.
    Identity coverage never changes the operator, so it is excluded from
    equality: ``X(a) @ I(b) == X(a)``.  Consumers that take ownership of every
    covered operand (for example destructive :func:`cudaq.logical.readout`) still see
    identity operands through :attr:`covered_operands`.
    """

    factors: tuple[PauliFactor, ...]
    sign: int = 1
    identities: tuple[PauliFactor, ...] = field(default=(), compare=False)

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError("PauliProduct sign must be +1 or -1")
        seen: set[tuple[Any, ...]] = set()
        normalized: list[PauliFactor] = []
        identities: list[PauliFactor] = list(self.identities)
        for factor in self.factors:
            if factor.pauli == "I":
                identities.append(factor)
                continue
            if factor.pauli not in {"X", "Y", "Z"}:
                raise ValueError("PauliProduct factors must be I, X, Y, or Z")
            normalized.append(factor)
        for factor in (*normalized, *identities):
            if factor.key in seen:
                raise ValueError(
                    "Pauli tensor composition requires disjoint operands; "
                    "use PauliGroupElement.multiply for overlapping algebra")
            seen.add(factor.key)
        if not normalized and not identities:
            raise ValueError("PauliProduct requires at least one factor")
        normalized.sort(key=lambda factor: factor.key)
        identities.sort(key=lambda factor: factor.key)
        object.__setattr__(self, "factors", tuple(normalized))
        object.__setattr__(self, "identities", tuple(identities))

    def __matmul__(self, other: "PauliProduct") -> "PauliProduct":
        if not isinstance(other, PauliProduct):
            return NotImplemented
        return PauliProduct(
            self.factors + other.factors,
            self.sign * other.sign,
            self.identities + other.identities,
        )

    def __neg__(self) -> "PauliProduct":
        return PauliProduct(self.factors, -self.sign, self.identities)

    @property
    def operands(self) -> tuple[Any, ...]:
        return tuple(factor.operand for factor in self.factors)

    @property
    def identity_operands(self) -> tuple[Any, ...]:
        return tuple(factor.operand for factor in self.identities)

    @property
    def covered_operands(self) -> tuple[Any, ...]:
        """All owned operands, including identity-covered ones, in key order."""
        merged = sorted((*self.factors, *self.identities), key=lambda f: f.key)
        return tuple(factor.operand for factor in merged)

    @property
    def paulis(self) -> tuple[str, ...]:
        return tuple(factor.pauli for factor in self.factors)

    @classmethod
    def from_symplectic(
        cls,
        *,
        operands: Iterable[Any],
        x_mask: int,
        z_mask: int,
        sign: int = 1,
    ) -> "PauliProduct":
        operands = tuple(operands)
        if x_mask < 0 or z_mask < 0:
            raise ValueError("symplectic masks must be nonnegative")
        if (x_mask | z_mask) >> len(operands):
            raise ValueError("symplectic mask exceeds operand arity")
        factors: list[PauliFactor] = []
        for i, operand in enumerate(operands):
            x = bool(x_mask & (1 << i))
            z = bool(z_mask & (1 << i))
            if x or z:
                factors.append(
                    PauliFactor(operand, "Y" if x and z else "X" if x else "Z"))
        if not factors:
            raise ValueError(
                "PauliProduct must contain at least one non-identity factor")
        return cls(tuple(factors), sign)

    @classmethod
    def formal(
        cls,
        *,
        arity: int,
        x_mask: int,
        z_mask: int,
        sign: int = 1,
    ) -> "PauliProduct":
        if arity <= 0:
            raise ValueError("formal PauliProduct arity must be positive")
        return cls.from_symplectic(operands=range(arity),
                                   x_mask=x_mask,
                                   z_mask=z_mask,
                                   sign=sign)

    def symplectic_for(self,
                       operands: Iterable[Any] |
                       None = None) -> tuple[int, int]:
        order = tuple(operands) if operands is not None else self.operands
        positions = {
            _operand_key(operand): i for i, operand in enumerate(order)
        }
        x_mask = z_mask = 0
        for factor in self.factors:
            try:
                i = positions[factor.key]
            except KeyError as exc:
                raise ValueError(
                    "operand order does not cover this PauliProduct") from exc
            if factor.pauli in {"X", "Y"}:
                x_mask |= 1 << i
            if factor.pauli in {"Z", "Y"}:
                z_mask |= 1 << i
        return x_mask, z_mask


@dataclass(frozen=True, slots=True)
class PauliGroupElement:
    """One element ``i^p * X^x Z^z`` of the formal n-qubit Pauli group.

    Unlike the Hermitian :class:`PauliProduct`, a group element tracks the
    complete phase exponent modulo four, so overlapping multiplication is
    well defined and never hides a ``+/-i`` phase.
    """

    arity: int
    x_mask: int
    z_mask: int
    phase_exponent_mod_4: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.arity, int) or isinstance(
                self.arity, bool) or self.arity <= 0:
            raise ValueError("PauliGroupElement arity must be a positive int")
        if self.x_mask < 0 or self.z_mask < 0:
            raise ValueError("symplectic masks must be nonnegative")
        if (self.x_mask | self.z_mask) >> self.arity:
            raise ValueError("symplectic mask exceeds the declared arity")
        object.__setattr__(self, "phase_exponent_mod_4",
                           int(self.phase_exponent_mod_4) % 4)

    @property
    def is_hermitian(self) -> bool:
        # i^p X^x Z^z collects one factor of -i per overlapping (Y) position,
        # so the element is Hermitian exactly when p and popcount(x & z) have
        # equal parity.
        return self.phase_exponent_mod_4 % 2 == (
            (self.x_mask & self.z_mask).bit_count() % 2)

    @classmethod
    def from_product(cls, product: PauliProduct) -> "PauliGroupElement":
        if not isinstance(product, PauliProduct):
            raise TypeError(
                "PauliGroupElement.from_product expects a PauliProduct")
        covered = product.covered_operands
        if any(not isinstance(operand, int) or isinstance(operand, bool)
               for operand in covered):
            raise TypeError(
                "PauliGroupElement.from_product requires a formal PauliProduct "
                "whose operands are integer ports")
        arity = max(covered) + 1
        x_mask = z_mask = overlap = 0
        for factor in product.factors:
            position = factor.operand
            if factor.pauli in {"X", "Y"}:
                x_mask |= 1 << position
            if factor.pauli in {"Z", "Y"}:
                z_mask |= 1 << position
            if factor.pauli == "Y":
                overlap += 1
        # Each Hermitian Y factor is i * X Z, and a -1 sign is i^2.
        phase = (overlap + (2 if product.sign == -1 else 0)) % 4
        return cls(arity, x_mask, z_mask, phase)

    def multiply(
            self,
            other: "PauliGroupElement | PauliProduct") -> "PauliGroupElement":
        """Group multiplication ``self * other`` with explicit phase tracking."""
        if isinstance(other, PauliProduct):
            other = PauliGroupElement.from_product(other)
        if not isinstance(other, PauliGroupElement):
            raise TypeError(
                "PauliGroupElement.multiply expects a group element or product")
        arity = max(self.arity, other.arity)
        # (X^x1 Z^z1)(X^x2 Z^z2): commuting Z^z1 across X^x2 contributes
        # (-1)^{popcount(z1 & x2)} = i^{2 popcount(z1 & x2)}.
        phase = (self.phase_exponent_mod_4 + other.phase_exponent_mod_4 + 2 *
                 (self.z_mask & other.x_mask).bit_count()) % 4
        return PauliGroupElement(
            arity,
            self.x_mask ^ other.x_mask,
            self.z_mask ^ other.z_mask,
            phase,
        )


def _factor(pauli: str, operand: Any) -> PauliProduct:
    return PauliProduct((PauliFactor(operand, pauli),))


def I(operand: Any) -> PauliProduct:  # noqa: E743 - spec-mandated factor name
    """Explicit identity coverage of one operand inside a tensor composition.

    ``X(a) @ I(b)`` equals ``X(a)`` as an operator while additionally covering
    ``b``; identity factors are legal only where the consumer accepts them.
    """
    return _factor("I", operand)


def X(operand: Any) -> PauliProduct:
    return _factor("X", operand)


def Y(operand: Any) -> PauliProduct:
    return _factor("Y", operand)


def Z(operand: Any) -> PauliProduct:
    return _factor("Z", operand)
