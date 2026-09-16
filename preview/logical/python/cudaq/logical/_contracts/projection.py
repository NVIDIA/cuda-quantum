# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import dataclass

_PORT_DIRECTIONS = frozenset({"input", "output", "inout"})


@dataclass(frozen=True, slots=True)
class ProjectedPort:
    """One typed gadget port bound to a projected circuit's carrier namespace.

    ``port_ordinal`` is the normalized ``fabric.gadget_spec`` ordinal.  The
    integer carrier identifiers are meaningful only within the circuit
    projection that owns the surrounding :class:`CompiledInterfaceManifest`.
    Reusable gadget specifications intentionally never contain these values.
    """

    port_ordinal: int
    name: str
    direction: str
    patch_id: int
    partitions: tuple[tuple[str, tuple[int, ...]], ...]

    def __post_init__(self) -> None:
        if (not isinstance(self.port_ordinal, int) or
                isinstance(self.port_ordinal, bool) or self.port_ordinal < 0):
            raise TypeError("projected port ordinal must be a nonnegative int")
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("projected port name must be nonempty")
        if self.direction not in _PORT_DIRECTIONS:
            raise ValueError(
                "projected port direction must be input, output, or inout")
        if (not isinstance(self.patch_id, int) or
                isinstance(self.patch_id, bool) or self.patch_id < 0):
            raise TypeError(
                "projected patch identity must be a nonnegative int")

        normalized = []
        names = set()
        for partition, carriers in self.partitions:
            if not isinstance(partition, str) or not partition:
                raise TypeError("projected partition names must be nonempty")
            if partition in names:
                raise ValueError(f"duplicate projected partition {partition!r}")
            names.add(partition)
            carriers = tuple(carriers)
            if any(not isinstance(carrier, int) or isinstance(carrier, bool) or
                   carrier < 0 for carrier in carriers):
                raise TypeError(
                    "projected carrier identifiers must be nonnegative ints")
            if len(set(carriers)) != len(carriers):
                raise ValueError(
                    "a projected partition cannot repeat a carrier")
            normalized.append((partition, carriers))
        if "data" not in names:
            raise ValueError(
                "every projected encoded port requires a data partition")
        object.__setattr__(self, "partitions", tuple(normalized))

    def carriers(self, partition: str) -> tuple[int, ...]:
        try:
            return dict(self.partitions)[partition]
        except KeyError as exc:
            raise KeyError(
                f"projected port {self.name!r} has no {partition!r} partition"
            ) from exc

    @property
    def data_carriers(self) -> tuple[int, ...]:
        return self.carriers("data")


@dataclass(frozen=True, slots=True)
class ProjectedMeasurement:
    """One stable realization record bound to a projected measurement index."""

    index: int
    record: str
    symbol: str
    call_path: tuple[str, ...]
    field: str
    lane: int
    carriers: tuple[int, ...]
    aliases: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (not isinstance(self.index, int) or isinstance(self.index, bool) or
                self.index < 0):
            raise TypeError(
                "projected measurement index must be a nonnegative int")
        for label, value in (("record", self.record), ("symbol", self.symbol)):
            if not isinstance(value, str) or not value:
                raise TypeError(
                    f"projected measurement {label} must be nonempty")
        call_path = tuple(self.call_path)
        if any(not isinstance(item, str) or not item for item in call_path):
            raise TypeError(
                "projected measurement call path must contain names")
        if not isinstance(self.field, str) or not self.field:
            raise TypeError("projected measurement field must be nonempty")
        if not isinstance(self.lane, int) or isinstance(self.lane,
                                                        bool) or self.lane < 0:
            raise TypeError(
                "projected measurement lane must be a nonnegative int")
        carriers = tuple(self.carriers)
        if any(not isinstance(carrier, int) or isinstance(carrier, bool) or
               carrier < 0 for carrier in carriers):
            raise TypeError(
                "projected measurement carriers must be nonnegative ints")
        aliases = tuple(self.aliases)
        if any(not isinstance(alias, str) or not alias for alias in aliases):
            raise TypeError(
                "projected measurement aliases must be nonempty strings")
        object.__setattr__(self, "call_path", call_path)
        object.__setattr__(self, "carriers", carriers)
        object.__setattr__(self, "aliases", aliases)


@dataclass(frozen=True, slots=True)
class CompiledInterfaceManifest:
    """Typed boundary and record bindings for one concrete circuit projection."""

    inputs: tuple[ProjectedPort, ...]
    outputs: tuple[ProjectedPort, ...]
    measurements: tuple[ProjectedMeasurement, ...]
    carrier_count: int

    def __post_init__(self) -> None:
        inputs = tuple(self.inputs)
        outputs = tuple(self.outputs)
        measurements = tuple(self.measurements)
        if any(port.direction not in {"input", "inout"} for port in inputs):
            raise ValueError(
                "compiled inputs must have input or inout direction")
        if any(port.direction not in {"output", "inout"} for port in outputs):
            raise ValueError(
                "compiled outputs must have output or inout direction")
        for side, ports in (("input", inputs), ("output", outputs)):
            ordinals = tuple(port.port_ordinal for port in ports)
            if len(set(ordinals)) != len(ordinals):
                raise ValueError(
                    f"compiled {side} port ordinals must be unique")
            if ordinals != tuple(sorted(ordinals)):
                raise ValueError(
                    f"compiled {side} ports must follow interface order")
        indices = tuple(measurement.index for measurement in measurements)
        if indices != tuple(range(len(measurements))):
            raise ValueError(
                "compiled measurement indices must be contiguous execution order"
            )
        if (not isinstance(self.carrier_count, int) or
                isinstance(self.carrier_count, bool) or self.carrier_count < 0):
            raise TypeError("compiled carrier count must be a nonnegative int")
        used = [
            carrier for port in (*inputs, *outputs)
            for _partition, carriers in port.partitions for carrier in carriers
        ]
        used.extend(carrier for measurement in measurements
                    for carrier in measurement.carriers)
        if used and max(used) >= self.carrier_count:
            raise ValueError(
                "compiled interface carrier exceeds its projection range")
        object.__setattr__(self, "inputs", inputs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "measurements", measurements)

    @property
    def input_data_carriers(self) -> tuple[int, ...]:
        return tuple(
            carrier for port in self.inputs for carrier in port.data_carriers)

    @property
    def output_data_carriers(self) -> tuple[int, ...]:
        return tuple(
            carrier for port in self.outputs for carrier in port.data_carriers)

    def record_order(self, gadget) -> tuple:
        """Return root-gadget ``RecordRef`` values in measurement order.

        Ambiguous repeated record names are rejected instead of being inferred
        from equal port widths.  Repeated call sites therefore require the
        compiler to have assigned distinct stable record paths.
        """

        names = tuple(measurement.record for measurement in self.measurements)
        if len(set(names)) != len(names):
            duplicate = next(name for name in names if names.count(name) > 1)
            raise ValueError(
                "projected measurement manifest has ambiguous repeated stable "
                f"record {duplicate!r}")
        return tuple(gadget.record(name) for name in names)


__all__ = [
    "CompiledInterfaceManifest",
    "ProjectedMeasurement",
    "ProjectedPort",
]

# Preserve public and pickle identities through
# cudaq.logical.compiler.projection.
for _compatibility_class in (
        ProjectedPort,
        ProjectedMeasurement,
        CompiledInterfaceManifest,
):
    _compatibility_class.__module__ = "cudaq.logical.compiler.projection"
del _compatibility_class
