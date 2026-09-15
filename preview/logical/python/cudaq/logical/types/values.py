# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterator, Sequence

from ..errors import QLXError


class UseAfterConsume(QLXError, RuntimeError):
    pass


class _SSAProxy:
    __slots__ = ("mlir_value", "owner", "location")

    def __init__(self,
                 mlir_value: Any,
                 *,
                 owner: object,
                 location: Any = None) -> None:
        self.mlir_value = mlir_value
        self.owner = owner
        self.location = location

    @property
    def type(self):
        return self.mlir_value.type


class logical_qubit(_SSAProxy):
    """One live, linear P0 logical-qubit SSA owner."""

    __slots__ = ("_semantic_ref", "_live")

    def __init__(
        self,
        mlir_value: Any,
        *,
        owner: object,
        semantic_ref: tuple[Any, ...],
        location: Any = None,
    ) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self._semantic_ref = semantic_ref
        self._live = True

    @property
    def semantic_ref(self) -> tuple[Any, ...]:
        return self._semantic_ref

    @property
    def is_live(self) -> bool:
        return self._live

    def _consume(self, operation: str) -> None:
        if not self._live:
            raise UseAfterConsume(
                f"logical value {self._semantic_ref!r} was already consumed; "
                f"cannot use it in {operation}")
        self._live = False

    def __copy__(self):
        raise TypeError("live logical values cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("live logical values cannot be deep-copied")

    def __reduce__(self):
        raise TypeError("live logical values cannot be pickled")


class _ClassicalValue(_SSAProxy):
    __slots__ = ()


class LogicalBool(_ClassicalValue):
    __slots__ = ("producer",)

    def __init__(self, mlir_value, *, owner, location=None, producer=None):
        super().__init__(mlir_value, owner=owner, location=location)
        self.producer = producer

    def __bool__(self) -> bool:
        raise TypeError(
            "a traced logical Boolean cannot be inspected by Python; use "
            "cudaq.logical.cond(...) or cudaq.logical.if_(...)")

    def __xor__(self, other):
        return self.owner.xor(self, other)

    def __rxor__(self, other):
        return self.owner.xor(other, self)


class IndexValue(_ClassicalValue):
    pass


class Float64Value(_ClassicalValue):
    pass


class EventState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    FAILED = "failed"
    CANCELLED = "cancelled"
    EXHAUSTED = "exhausted"


class EventStatusValue(_ClassicalValue):
    """Copyable result of a nonconsuming event poll."""

    __slots__ = ()


class _LogicalLinearValue(_SSAProxy):
    __slots__ = ("_live",)

    def __init__(self, mlir_value, *, owner, location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self._live = True

    @property
    def is_live(self) -> bool:
        return self._live

    def _validate_consume(self, operation: str) -> None:
        if not self._live:
            raise UseAfterConsume(
                f"logical linear value was already consumed before {operation}")

    def _consume(self, operation: str) -> None:
        self._validate_consume(operation)
        self._live = False

    def __copy__(self):
        raise TypeError("logical linear values cannot be copied")

    def __deepcopy__(self, memo):
        raise TypeError("logical linear values cannot be deep-copied")

    def __reduce__(self):
        raise TypeError("logical linear values cannot be pickled")


class LogicalResourceValue(_LogicalLinearValue):
    __slots__ = ("kind",)

    def __init__(self, mlir_value, *, owner, kind, location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.kind = kind


class LogicalEventValue(_LogicalLinearValue):
    __slots__ = ("payload_type", "payload_kind")

    def __init__(
        self,
        mlir_value,
        *,
        owner,
        payload_type,
        payload_kind=None,
        location=None,
    ) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.payload_type = payload_type
        self.payload_kind = payload_kind


class LogicalFrameValue(_LogicalLinearValue):
    __slots__ = ("domain",)

    def __init__(self,
                 mlir_value,
                 *,
                 owner,
                 domain: str,
                 location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.domain = domain


class ResourceValue(_LogicalLinearValue):
    """One live P2 resource-state owner."""

    __slots__ = ("kind",)

    def __init__(self, mlir_value, *, owner, kind, location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.kind = kind


class FabricEventValue(_LogicalLinearValue):
    """One live P2 asynchronous resource-flow event."""

    __slots__ = ("payload_kind",)

    def __init__(self, mlir_value, *, owner, payload_kind, location=None):
        super().__init__(mlir_value, owner=owner, location=location)
        self.payload_kind = payload_kind


class MeasurementBits(_ClassicalValue):
    """Copyable tensor of measurement records with one stable record base."""

    __slots__ = ("record",)

    def __init__(self,
                 mlir_value,
                 *,
                 owner,
                 record: str,
                 location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.record = record


class SyndromeValue(_ClassicalValue):
    """Structured syndrome bundle produced by one extraction round."""

    __slots__ = ("encoding", "record")

    def __init__(self,
                 mlir_value,
                 *,
                 owner,
                 encoding,
                 record: str,
                 location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.encoding = encoding
        self.record = record


class GaugeRecordsValue(_ClassicalValue):
    """Raw outcomes from one ordered gauge-measurement basis."""

    __slots__ = ("encoding", "epoch", "record", "measurement_map")

    def __init__(
        self,
        mlir_value,
        *,
        owner,
        encoding,
        epoch,
        record: str,
        measurement_map,
        location=None,
    ) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.encoding = encoding
        self.epoch = epoch
        self.record = record
        self.measurement_map = measurement_map


class PhysicalState(_SSAProxy):
    __slots__ = ("resource", "resource_class", "_live")

    def __init__(
        self,
        mlir_value,
        *,
        owner,
        resource,
        resource_class=None,
        location=None,
    ) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.resource = resource
        self.resource_class = resource_class
        self._live = True

    @property
    def is_live(self) -> bool:
        return self._live

    def _consume(self, operation: str) -> None:
        if not self._live:
            raise UseAfterConsume(
                f"physical resource {self.resource!r} was already consumed before {operation}"
            )
        self._live = False


class PhysicalRecord(_ClassicalValue):
    __slots__ = ("record", "producer")

    def __init__(self,
                 mlir_value,
                 *,
                 owner,
                 record,
                 producer,
                 location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.record = record
        self.producer = producer


class LogicalRegister(Sequence[logical_qubit]):
    """A small mutable Python container of immutable SSA proxies."""

    __slots__ = ("_values", "name")

    def __init__(self,
                 values: list[logical_qubit],
                 name: str | None = None) -> None:
        self._values = values
        self.name = name

    def __getitem__(self, index):
        return self._values[index]

    def __setitem__(self, index, value: logical_qubit) -> None:
        if not isinstance(value, logical_qubit):
            raise TypeError("logical registers contain logical_qubit values")
        self._values[index] = value

    def __len__(self) -> int:
        return len(self._values)

    def __iter__(self) -> Iterator[logical_qubit]:
        return iter(self._values)

    def as_tuple(self) -> tuple[logical_qubit, ...]:
        return tuple(self._values)


def unwrap(value: Any):
    return value.mlir_value if isinstance(value, _SSAProxy) else value


@dataclass(frozen=True, slots=True)
class _BBSyndromeContinuation:
    """Internal proof that moment 8 initialized one code's Z ancillas."""

    code: Any


class PatchValue(_SSAProxy):
    __slots__ = (
        "encoding",
        "epoch",
        "_live",
        "_semantic_ref",
        "_encoded_state_live",
        "_terminal_only_reason",
        "_bb_syndrome_continuation",
        "carrier_frame",
    )

    def __init__(
        self,
        mlir_value,
        *,
        owner,
        encoding,
        epoch=None,
        semantic_ref,
        encoded_state_live=True,
        terminal_only_reason=None,
        bb_syndrome_continuation=None,
        carrier_frame=None,
        location=None,
    ) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.encoding = encoding
        self.epoch = encoding.initial_epoch if epoch is None else epoch
        self._semantic_ref = tuple(semantic_ref)
        self._live = True
        self._encoded_state_live = bool(encoded_state_live)
        self._terminal_only_reason = terminal_only_reason
        self._bb_syndrome_continuation = bb_syndrome_continuation
        self.carrier_frame = carrier_frame

    @property
    def semantic_ref(self) -> tuple[Any, ...]:
        return self._semantic_ref

    @property
    def is_live(self) -> bool:
        return self._live

    def _validate_consume(self, operation: str) -> None:
        guard = getattr(self.owner, "_before_operation", None)
        if guard is not None:
            guard(operation)
        self._validate_consume_state(operation)

    def _validate_consume_state(self, operation: str) -> None:
        """Check local linear state without invoking an owner-level guard."""

        if not self._live:
            raise UseAfterConsume(
                f"patch was already consumed before {operation}")
        if (not self._encoded_state_live and
                operation not in {"fabric.dealloc", "fabric.transform_end"}):
            raise UseAfterConsume(
                "the patch's encoded state was destroyed by terminal data "
                f"measurement; it may only be discarded, not used in {operation}"
            )
        if self._terminal_only_reason is not None and operation != "fabric.dealloc":
            raise UseAfterConsume(
                f"{self._terminal_only_reason}; the patch may only be "
                f"discarded, not used in {operation}")

    def _consume(self, operation: str) -> None:
        self._validate_consume(operation)
        self._live = False

    def __getattr__(self, name: str):
        block = self.carrier_frame.frame if self.carrier_frame else self.encoding.code.block
        if name in block.partitions:
            return PartitionView(self, name)
        raise AttributeError(name)

    @property
    def frame(self):
        """View every carrier reserved by a dynamic-code transform."""

        if self.carrier_frame is None:
            return PartitionView(self, "all")
        if len(self.carrier_frame.frame.partitions) == 1:
            return PartitionView(
                self, next(iter(self.carrier_frame.frame.partitions)))
        return PartitionView(self, "all")

    @property
    def carrier_block(self):
        return self.carrier_frame.frame if self.carrier_frame else self.encoding.code.block

    def __getitem__(self, index: int):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("encoded logical-port indices must be Python ints")
        if index < 0 or index >= self.encoding.code.k:
            raise IndexError(index)
        return PatchLogicalRef(self, index)

    @property
    def gauge(self):
        """Explicit non-protected subsystem degree view.

        Gauge degrees are intentionally absent from ordinary logical-port
        indexing. They may be addressed only by realization-side protocols.
        """

        return PatchGaugeView(self)


class PatchBundleValue(_SSAProxy):
    """One linear folded child-slot aggregate of a hierarchical encoding."""

    __slots__ = ("hierarchy", "slot_group", "_live")

    def __init__(self,
                 mlir_value,
                 *,
                 owner,
                 hierarchy,
                 slot_group: str,
                 location=None) -> None:
        super().__init__(mlir_value, owner=owner, location=location)
        self.hierarchy = hierarchy
        self.slot_group = slot_group
        self._live = True

    @property
    def is_live(self) -> bool:
        return self._live

    def _consume(self, operation: str) -> None:
        if not self._live:
            raise UseAfterConsume(
                f"patch bundle was already consumed before {operation}")
        self._live = False


@dataclass(frozen=True, slots=True)
class PatchLogicalRef:
    patch: PatchValue
    index: int

    @property
    def semantic_ref(self):
        return (*self.patch.semantic_ref, "logical", self.index)


@dataclass(frozen=True, slots=True)
class PatchGaugeRef:
    patch: PatchValue
    index: int

    @property
    def semantic_ref(self):
        return (*self.patch.semantic_ref, "gauge", self.index)


@dataclass(frozen=True, slots=True)
class PatchGaugeView:
    patch: PatchValue

    def __getitem__(self, index: int):
        if not isinstance(index, int) or isinstance(index, bool):
            raise TypeError("gauge indices must be Python ints")
        if index < 0 or index >= self.patch.encoding.code.r:
            raise IndexError(index)
        return PatchGaugeRef(self.patch, index)


@dataclass(frozen=True, slots=True)
class PartitionView:
    owner: PatchValue
    partition: str

    def __getitem__(self, indices):
        if isinstance(indices, int) and not isinstance(indices, bool):
            values = (indices,)
        else:
            try:
                values = tuple(indices)
            except TypeError as exc:
                raise TypeError(
                    "carrier selection must be an int or iterable") from exc
        if any(not isinstance(value, int) or isinstance(value, bool) or
               value < 0 for value in values):
            raise TypeError(
                "carrier selection indices must be nonnegative ints")
        width = (self.owner.carrier_block.size if self.partition == "all" else
                 self.owner.carrier_block.partitions[self.partition])
        if any(value >= width for value in values):
            raise IndexError("carrier selection exceeds its partition")
        return PartitionSelection(self.owner, self.partition, values)


@dataclass(frozen=True, slots=True)
class PartitionSelection:
    owner: PatchValue
    partition: str
    indices: tuple[int, ...]
