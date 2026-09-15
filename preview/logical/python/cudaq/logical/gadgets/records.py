# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from inspect import Signature, signature
from math import isfinite
from types import MappingProxyType, NoneType
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    Literal,
    Mapping,
    TypeVar,
    get_args,
    get_origin,
    get_type_hints,
)

from cudaq.logical.programs.binding import (
    LogicalPortRef,
    ObjectiveOperandRef,
)
from cudaq.logical._core.immutable import ImmutableValue

EncodingT = TypeVar("EncodingT")

from .interface import (
    BlockEndpoint,
    BlockFlow,
    EndpointCollection,
    GadgetInterface,
    _freeze_gadget_metadata,
    patch,
)


@dataclass(frozen=True, slots=True)
class RecordRef:
    """Stable reference to one record produced by a gadget realization."""

    gadget: "GadgetDefinition"
    name: str

    def __post_init__(self) -> None:
        if not self.name or self.name.startswith(".") or self.name.endswith(
                "."):
            raise ValueError("record names must be nonempty relative paths")

    def __xor__(self, other):
        return RecordParity((self,)) ^ other


@dataclass(frozen=True, slots=True)
class RecordParity:
    """An immutable XOR expression over stable gadget records."""

    records: tuple[RecordRef, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "records", tuple(self.records))
        if not self.records:
            raise ValueError("a record parity requires at least one record")

    def __xor__(self, other):
        if isinstance(other, RecordRef):
            other = RecordParity((other,))
        if not isinstance(other, RecordParity):
            return NotImplemented
        # Preserve first-seen order while applying GF(2) cancellation.
        ordered = []
        for record in (*self.records, *other.records):
            if record in ordered:
                ordered.remove(record)
            else:
                ordered.append(record)
        if not ordered:
            raise ValueError("record parity normalized to the zero expression")
        return RecordParity(tuple(ordered))


@dataclass(frozen=True, slots=True)
class InputSyndromeRef:
    """One effective syndrome bit entering a typed gadget block endpoint."""

    endpoint: BlockEndpoint
    index: int

    def __post_init__(self) -> None:
        if not isinstance(self.endpoint,
                          BlockEndpoint) or self.endpoint.side != "input":
            raise TypeError("input syndrome requires an input BlockEndpoint")
        if not isinstance(self.index, int) or isinstance(
                self.index, bool) or self.index < 0:
            raise TypeError("input syndrome index must be a nonnegative int")

    @property
    def port(self) -> BlockEndpoint:
        """The typed endpoint; retained as the algebra's historical field name."""

        return self.endpoint

    def __xor__(self, other):
        return ProfileParity(input_syndromes=(self,)) ^ other

    def __rxor__(self, other):
        return ProfileParity.from_value(other) ^ self


@dataclass(frozen=True, slots=True)
class ProfileParity:
    """Affine GF(2) parity over incoming syndromes and realization records."""

    records: tuple[RecordRef, ...] = ()
    input_syndromes: tuple[InputSyndromeRef, ...] = ()
    constant: bool = False

    def __post_init__(self) -> None:
        records = tuple(self.records)
        for record in records:
            if not isinstance(record, RecordRef):
                raise TypeError(
                    "profile parity records must be RecordRef values")
        if len(set(records)) != len(records):
            raise ValueError(
                "profile parity records must be duplicate-free; use XOR to "
                "cancel terms intentionally")
        syndromes = tuple(self.input_syndromes)
        for syndrome in syndromes:
            if not isinstance(syndrome, InputSyndromeRef):
                raise TypeError(
                    "profile parity input_syndromes must be InputSyndromeRef values"
                )
        if len(set(syndromes)) != len(syndromes):
            raise ValueError(
                "profile parity input_syndromes must be duplicate-free; use "
                "XOR to cancel terms intentionally")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "input_syndromes", syndromes)
        object.__setattr__(self, "constant", bool(self.constant))

    @classmethod
    def _from_xor_terms(
        cls,
        records,
        input_syndromes,
        constant,
    ) -> "ProfileParity":
        """Construct the canonical result of an intentional XOR operation."""

        def cancel(values):
            result = []
            for value in values:
                if value in result:
                    result.remove(value)
                else:
                    result.append(value)
            return tuple(result)

        return cls(
            records=cancel(records),
            input_syndromes=cancel(input_syndromes),
            constant=constant,
        )

    @classmethod
    def from_value(cls, value) -> "ProfileParity":
        if isinstance(value, cls):
            return value
        if isinstance(value, RecordRef):
            return cls(records=(value,))
        if isinstance(value, RecordParity):
            return cls(records=value.records)
        if isinstance(value, InputSyndromeRef):
            return cls(input_syndromes=(value,))
        if isinstance(value, bool):
            return cls(constant=value)
        raise TypeError(
            "expected a record, incoming syndrome, or profile parity")

    def __xor__(self, other):
        other = ProfileParity.from_value(other)
        return ProfileParity._from_xor_terms(
            (*self.records, *other.records),
            (*self.input_syndromes, *other.input_syndromes),
            self.constant ^ other.constant,
        )

    def __rxor__(self, other):
        return ProfileParity.from_value(other) ^ self


@dataclass(frozen=True, slots=True)
class RecordFamily:
    """Typed indexed family below one stable gadget record base."""

    gadget: "GadgetDefinition"
    base: str
    field: str
    indices: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = tuple(self.indices)
        if not indices:
            raise ValueError("a record family requires at least one index")
        if any(not isinstance(index, int) or isinstance(index, bool) or
               index < 0 for index in indices):
            raise TypeError(
                "record-family indices must be nonnegative integers")
        if len(set(indices)) != len(indices):
            raise ValueError("record-family indices must be unique")
        object.__setattr__(self, "indices", indices)

    def __getitem__(self, index: int) -> RecordRef:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise TypeError(
                "record-family indices must be nonnegative integers")
        try:
            selected = self.indices[index]
        except IndexError as exc:
            raise IndexError(
                "record-family index is outside its typed shape") from exc
        return RecordRef(self.gadget, f"{self.base}.{self.field}{selected}")

    def select(self, indices) -> "RecordFamily":
        return RecordFamily(
            self.gadget,
            self.base,
            self.field,
            tuple(self.indices[index] for index in indices),
        )

    def parity(self, indices=None) -> RecordParity:
        selected = range(len(self.indices)) if indices is None else indices
        refs = tuple(self[index] for index in selected)
        return RecordParity(refs)

    def __xor__(self, other):
        if not isinstance(other, RecordFamily):
            return NotImplemented
        return RecordVectorParity((self, other))


@dataclass(frozen=True, slots=True)
class RecordVectorParity:
    """Shape-preserving elementwise XOR over equal-width record families."""

    families: tuple[RecordFamily, ...]

    def __post_init__(self) -> None:
        ordered = []
        for family in self.families:
            if family in ordered:
                ordered.remove(family)
            else:
                ordered.append(family)
        families = tuple(ordered)
        if not families:
            raise ValueError(
                "a record-vector parity requires at least one family")
        widths = {len(family.indices) for family in families}
        if len(widths) != 1:
            raise ValueError("record-vector XOR requires equal-shaped families")
        object.__setattr__(self, "families", families)

    @property
    def width(self) -> int:
        return len(self.families[0].indices)

    def __xor__(self, other):
        if isinstance(other, RecordFamily):
            other = RecordVectorParity((other,))
        if not isinstance(other, RecordVectorParity):
            return NotImplemented
        return RecordVectorParity((*self.families, *other.families))

    def rows(self) -> tuple[RecordParity, ...]:
        return tuple(
            RecordParity(tuple(family[index]
                               for family in self.families))
            for index in range(self.width))


@dataclass(frozen=True, slots=True)
class ProfileVectorExpr:
    """Shape-preserving vector of affine profile rows."""

    rows: tuple[ProfileParity, ...]

    def __post_init__(self) -> None:
        rows = tuple(ProfileParity.from_value(row) for row in self.rows)
        if not rows:
            raise ValueError(
                "a profile vector expression requires at least one row")
        object.__setattr__(self, "rows", rows)

    @property
    def width(self) -> int:
        return len(self.rows)

    @classmethod
    def from_value(cls,
                   value,
                   *,
                   width: int | None = None) -> "ProfileVectorExpr":
        if isinstance(value, cls):
            result = value
        elif isinstance(value, SyndromeBundleRef):
            if value.endpoint.side != "input":
                raise TypeError(
                    "only input syndrome bundles are profile expressions")
            expression_width = value.width if width is None else width
            result = cls(
                tuple(
                    ProfileParity(input_syndromes=(
                        InputSyndromeRef(value.endpoint, index),))
                    for index in range(expression_width)))
        elif isinstance(value, RecordFamily):
            result = cls(
                tuple(
                    ProfileParity(records=(value[index],))
                    for index in range(len(value.indices))))
        elif isinstance(value, RecordVectorParity):
            result = cls(
                tuple(
                    ProfileParity(records=row.records) for row in value.rows()))
        elif isinstance(value, bool):
            if width is None:
                raise TypeError(
                    "a scalar profile constant requires a target width")
            result = cls(
                tuple(ProfileParity(constant=value) for _ in range(width)))
        elif isinstance(value, (tuple, list)):
            result = cls(tuple(ProfileParity.from_value(row) for row in value))
        else:
            if width == 1:
                result = cls((ProfileParity.from_value(value),))
            else:
                raise TypeError(
                    "expected a syndrome bundle, record family/vector, bool, or row sequence"
                )
        if width is not None and result.width != width:
            raise ValueError(
                f"profile vector width {result.width} does not match target width {width}"
            )
        return result

    def __xor__(self, other):
        other = ProfileVectorExpr.from_value(other, width=self.width)
        return ProfileVectorExpr(
            tuple(left ^ right for left, right in zip(self.rows, other.rows)))

    def __rxor__(self, other):
        return ProfileVectorExpr.from_value(other, width=self.width) ^ self


@dataclass(frozen=True, slots=True)
class SyndromeBundleRef:
    """The effective-syndrome bundle carried by one block endpoint."""

    endpoint: BlockEndpoint

    @property
    def width(self) -> int:
        return self.endpoint.code_profile.effective_stabilizers.nrows

    def __getitem__(self, index: int) -> InputSyndromeRef:
        if self.endpoint.side != "input":
            raise TypeError(
                "output syndrome bundles are assignment targets, not inputs")
        if not isinstance(index, int) or isinstance(
                index, bool) or not 0 <= index < self.width:
            raise IndexError(
                "syndrome index is outside the endpoint's typed shape")
        return InputSyndromeRef(self.endpoint, index)

    def __xor__(self, other):
        return ProfileVectorExpr.from_value(self).__xor__(other)

    def __rxor__(self, other):
        return ProfileVectorExpr.from_value(self).__rxor__(other)


@dataclass(frozen=True, slots=True)
class ProfileBinding:
    """One vector-valued output-boundary equation in a profile graph."""

    target: SyndromeBundleRef
    value: ProfileVectorExpr

    def __post_init__(self) -> None:
        if self.target.endpoint.side != "output":
            raise TypeError(
                "profile binding targets must be output syndrome bundles")


@dataclass(frozen=True, slots=True)
class StructuredRecord:
    """Typed view of one generated syndrome or data-readout record."""

    gadget: "GadgetDefinition"
    base: str
    check_count: int = 0
    data_count: int = 0
    bit_count: int = 0

    @property
    def checks(self) -> RecordFamily:
        return RecordFamily(self.gadget, self.base, "s",
                            tuple(range(self.check_count)))

    @property
    def data(self) -> RecordFamily:
        return RecordFamily(self.gadget, self.base, "data",
                            tuple(range(self.data_count)))

    @property
    def bits(self) -> RecordFamily:
        return RecordFamily(self.gadget, self.base, "bit",
                            tuple(range(self.bit_count)))

    def __xor__(self, other):
        if not isinstance(other, StructuredRecord):
            return NotImplemented
        if self.check_count or other.check_count:
            if self.check_count != other.check_count:
                raise ValueError(
                    "syndrome-record XOR requires equal check counts")
            return self.checks ^ other.checks
        if self.data_count or other.data_count:
            if self.data_count != other.data_count:
                raise ValueError("data-record XOR requires equal data widths")
            return self.data ^ other.data
        if self.bit_count or other.bit_count:
            if self.bit_count != other.bit_count:
                raise ValueError("record XOR requires equal bit widths")
            return self.bits ^ other.bits
        raise TypeError("structured record has no vector field to compare")


def _resolve_gadget_code_profile(
    gadget: "GadgetDefinition",
    explicit=None,
):
    """Resolve one typed boundary profile for record-family operations.

    A gadget may expose several endpoints, but a generated record family has
    one shape.  Preserve the ordinary single-profile shorthand while requiring
    an explicit compatible ``CodeProfile`` whenever the typed boundary is
    ambiguous.
    """

    from cudaq.logical.codes import CodeProfile

    endpoints = (*gadget.interface.inputs, *gadget.interface.outputs)
    profiles = [endpoint.code_profile for endpoint in endpoints]
    if not profiles and gadget.device is not None:
        profiles.append(gadget.device.default_qec_region.encoding.profile)

    if explicit is not None:
        if not isinstance(explicit, CodeProfile):
            raise TypeError("code_profile= must be a cudaq.logical.CodeProfile")
        if not profiles:
            raise ValueError(
                "an explicit record CodeProfile requires a typed gadget boundary"
            )
        if all(profile is not explicit for profile in profiles):
            raise ValueError(
                "record CodeProfile is not carried by any typed gadget "
                "boundary Encoding")
        return explicit

    unique = []
    for profile in profiles:
        if all(profile is not candidate for candidate in unique):
            unique.append(profile)
    if len(unique) != 1:
        raise ValueError(
            "generated record-bundle shape is ambiguous for a gadget without "
            "one typed boundary CodeProfile; pass code_profile= explicitly")
    return unique[0]


class GadgetRecords:
    """Typed selectors for deterministic auto-named gadget records."""

    __slots__ = ("gadget",)

    def __init__(self, gadget: "GadgetDefinition") -> None:
        self.gadget = gadget

    @staticmethod
    def _index(index: int) -> int:
        if not isinstance(index, int) or isinstance(index, bool) or index < 0:
            raise TypeError("record indices must be nonnegative integers")
        return index

    def _boundary_profile(self, code_profile=None):
        return _resolve_gadget_code_profile(self.gadget, code_profile)

    def _syndrome_count(self, code_profile=None) -> int:
        profile = self._boundary_profile(code_profile)
        checks = profile.effective_stabilizers.nrows
        if not checks:
            raise ValueError(
                "the gadget boundary CodeProfile has no syndrome record shape")
        return checks

    def syndrome(self, index: int, *, code_profile=None) -> StructuredRecord:
        return StructuredRecord(
            self.gadget,
            f"syndrome{self._index(index)}",
            check_count=self._syndrome_count(code_profile),
        )

    def measure_z(self, index: int) -> StructuredRecord:
        code = self._boundary_profile().code
        return StructuredRecord(
            self.gadget,
            f"mz{self._index(index)}",
            data_count=code.block.partitions.get("data", code.n),
        )

    def product(self, index: int) -> RecordRef:
        return RecordRef(self.gadget, f"mpp{self._index(index)}.outcome")

    def gauge(self, index: int) -> StructuredRecord:
        return StructuredRecord(self.gadget, f"gauge{self._index(index)}")

    def inferred_syndrome(
        self,
        index: int,
        *,
        code_profile=None,
    ) -> StructuredRecord:
        return StructuredRecord(
            self.gadget,
            f"inferred{self._index(index)}",
            check_count=self._syndrome_count(code_profile),
        )
