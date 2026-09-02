# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from inspect import Parameter, Signature
from types import GenericAlias

from cudaq.mlir import ir as mlir_ir

from ..gadgets.builder import GadgetBuilder as _GadgetBackend
from ..programs.builder import UnplacedBuilder as _UnplacedBackend
from ..protocols.builder import ProtocolBuilder as _ProtocolBackend
from .build import Build, EvidenceRecord
from .context import CompilationContext
from .pipeline import pipelines
from ..types.values import (
    LogicalRegister,
    logical_qubit,
)
from ..codes import (
    Code,
    StabilizerCode,
    SubsystemCode,
)
from ..programs.definition import (
    DefinitionHandle,
    ProgramDefinition,
)
from ..gadgets import GadgetDefinition
from ..protocols.definition import ProtocolDefinition
from ..architecture.logical import (
    Space,
    SpaceSlot,
)
from ..architecture.constraints import local
from ..std import LogicalActionRef, LogicalInstrumentRef


def _return_annotation(results):
    results = tuple(results)
    if not results:
        return None
    if len(results) == 1:
        return results[0]
    return GenericAlias(tuple, results)


class _RawInsertionScope:
    """Context manager yielding one builder's live MLIR insertion point."""

    __slots__ = ("_backend",)

    def __init__(self, backend) -> None:
        self._backend = backend

    def __enter__(self):
        return self._backend.insertion_point

    def __exit__(self, exc_type, exc, traceback):
        return False


class UnplacedBuilder:
    """Advanced direct P0 authoring over the same MLIR backend as decorators."""

    profile = "p0"

    def __init__(self, name, *, arguments=(), results=(), _kind="program"):
        arguments = tuple(arguments)
        results = tuple(results)

        def provider(*_):  # Never executed by the direct surface.
            raise RuntimeError(
                "direct UnplacedBuilder provider must not execute")

        provider.__name__ = str(name)
        provider.__qualname__ = str(name)
        provider.__module__ = "__main__"
        provider.__signature__ = Signature(
            tuple(
                Parameter(
                    f"arg{index}",
                    Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=annotation,
                ) for index, annotation in enumerate(arguments)),
            return_annotation=_return_annotation(results),
        )
        provider.__annotations__ = {
            **{
                f"arg{index}": value for index, value in enumerate(arguments)
            },
            "return": _return_annotation(results),
        }
        definition_kind = "program" if _kind == "program" else "objective"
        objective_kind = "auto" if _kind == "program" else _kind
        self.definition = ProgramDefinition(
            provider,
            name=str(name),
            kind=definition_kind,
            objective_kind=objective_kind,
        )
        self.transaction = CompilationContext()
        self._backend = _UnplacedBackend(self.transaction, self.definition)
        self.context = self.transaction.context
        self.module = self.transaction.module
        self.insertion_point = self._backend.insertion_point
        self._finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raw_insertion_point(self):
        """Enter a controlled raw-MLIR insertion scope at the builder frontier.

        Yields the builder's current insertion point so generated operations
        (``cudaq.mlir.ir.Operation.create(..., ip=ip)``) land exactly where typed
        helpers would emit.  Raw results rejoin typed helpers through explicit
        adoption; operations that are not adopted remain visible only to MLIR.
        """
        return _RawInsertionScope(self._backend)

    def adopt(self, value, kind=None, *, consumes=()):
        """Adopt one raw MLIR result back into typed frontend liveness.

        Adoption is the explicit contract between generated raw operations
        and the typed builder: the result must be an MLIR value of this
        builder's ``!qlx.logical_qubit`` type produced inside this builder's
        module, and every input named in ``consumes`` leaves liveness exactly
        once so later typed use of a stale handle still fails closed.
        """
        from ..types.values import logical_qubit as _logical_qubit

        if kind is not None and kind is not _logical_qubit:
            raise TypeError(
                "raw adoption currently produces qlx.logical_qubit values")
        if not isinstance(value, mlir_ir.Value):
            raise TypeError(
                "adopt expects the MLIR result of a raw generated operation")
        backend = self._backend
        if value.type != backend.logical_type:
            raise TypeError(
                f"adopted value type {value.type} is not this builder's "
                "!qlx.logical_qubit (context and type must both match)")
        node = value.owner
        node = getattr(node, "operation", node)
        while node is not None and node.name != "builtin.module":
            node = node.parent
        if node is None or node != self.module.operation:
            raise ValueError(
                "adopted value was not produced inside this builder's module")
        backend._consume_qubits(tuple(consumes), "raw adoption")
        return backend._new_qubit(value)

    def arguments(self):
        return tuple(self._backend.arguments())

    def argument(self, index):
        return self.arguments()[index]

    def allocate(self, count, *, state="zero", name=None):
        return self._backend.allocate(count, state=state, name=name)

    def prepare(self, state="zero"):
        return self._backend.prepare(str(getattr(state, "name", state)))

    def apply(self, action, *values, **parameters):
        if isinstance(action, LogicalActionRef):
            results = self._backend.apply_standard(action.name, values,
                                                   **parameters)
        elif isinstance(action, ProgramDefinition):
            results = self._backend.apply_definition(action, values, parameters)
        else:
            raise TypeError(
                "UnplacedBuilder.apply expects a standard action or "
                "@cudaq.logical.objective definition")
        if isinstance(results, list) and len(results) == 1:
            return results[0]
        return tuple(results) if isinstance(results, list) else results

    def instrument(self, instrument, *values, **parameters):
        if isinstance(instrument, ProgramDefinition):
            return self._backend.apply_definition(instrument, values,
                                                  parameters)
        if not isinstance(instrument, LogicalInstrumentRef):
            raise TypeError(
                "UnplacedBuilder.instrument expects a standard instrument or "
                "@cudaq.logical.objective definition")
        if parameters:
            raise TypeError("standard direct instruments take no parameters")
        if instrument.name == "prepare_zero":
            if values:
                raise TypeError("prepare_zero takes no inputs")
            return self._backend.prepare("zero")
        if instrument.name == "prepare_plus":
            if values:
                raise TypeError("prepare_plus takes no inputs")
            return self._backend.prepare("plus")
        if instrument.name in {"measure_x", "measure_z"}:
            if len(values) != 1:
                raise TypeError(f"{instrument.name} takes one logical qubit")
            return self._backend.measure(instrument.name[-1], values[0])
        raise NotImplementedError(
            f"direct standard instrument {instrument.name!r} is not implemented"
        )

    def mpp(self, product):
        return self._backend.mpp(product)

    def rotate(self, product, *, angle, precision=None):
        return tuple(
            self._backend.rotate(product, angle=angle, precision=precision))

    def idle(self, *values, rounds):
        result = self._backend.idle(values, rounds=rounds)
        return result[0] if len(result) == 1 else tuple(result)

    def discard(self, *values, reason=None):
        self._backend.discard(values, reason=reason)

    def finish(self, *results):
        if self._finished:
            raise RuntimeError("UnplacedBuilder root is already finished")
        self._backend.finish(*results)
        self._finished = True
        if self.definition.kind == "program":
            handle = DefinitionHandle(self._backend.symbol, "program", "p0")
        else:
            family = self._backend.objective_family()
            objective = self.transaction.declare_objective(
                family=family,
                requested_symbol=self.definition.name,
                kind="composite",
                inputs=self._backend.input_types,
                results=self._backend.result_types,
                semantics=mlir_ir.FlatSymbolRefAttr.get(self._backend.symbol,
                                                        context=self.context),
            )
            handle = DefinitionHandle(objective, family, "p0")
        self.transaction.bind(self.definition, handle)
        build = Build(
            context=self.context,
            module=self.module,
            root=handle,
            profile="p0",
            pipeline=pipelines.logical(),
            evidence=(EvidenceRecord(
                kind="direct_profile_verification",
                producer="qlx-python@0.3",
                result="pass",
                obligations=("p0-signature", "linear-ownership"),
            ),),
            value_groups=self._backend.value_groups,
            source_modules=("__main__",),
        )
        self.definition._qlx_direct_snapshot = build
        return build


class ActionBuilder(UnplacedBuilder):

    def __init__(self, name, *, arity):
        super().__init__(
            name,
            arguments=(logical_qubit,) * int(arity),
            results=(logical_qubit,) * int(arity),
            _kind="action",
        )

    def ports(self):
        return self.arguments()


class PlacedBuilder(UnplacedBuilder):
    """Direct logical authoring followed by the canonical P0→P1 placer."""

    profile = "p1"

    def __init__(self, name, *, machine, arguments=(), results=()):
        super().__init__(name, arguments=arguments, results=results)
        self.machine = machine
        self._direct_bindings = []

    @staticmethod
    def _allocation_coordinate(value):
        if not isinstance(value, logical_qubit):
            raise TypeError("PlacedBuilder.bind expects logical qubit values")
        reference = tuple(value.semantic_ref)
        try:
            marker = reference.index("allocation")
            allocation = reference[marker + 1]
            index = reference[marker + 3]
        except (ValueError, IndexError) as error:
            raise ValueError(
                "direct placement requires a prepared logical allocation "
                "that survives P0 serialization") from error
        return int(allocation), int(index)

    def bind(self, values, *, at, slot=None, witness=None):
        """Attach exact local P1 placement to one or more authored values."""

        if isinstance(at, SpaceSlot):
            if slot is not None:
                raise TypeError("do not combine a SpaceSlot with slot=")
            at, slot = at.space, at.index
        if not isinstance(at, Space):
            raise TypeError("PlacedBuilder.bind at= requires a machine Space")
        scalar = isinstance(values, logical_qubit)
        authored = (values,) if scalar else tuple(values)
        if not authored:
            raise ValueError("PlacedBuilder.bind requires at least one value")
        for offset, value in enumerate(authored):
            allocation, index = self._allocation_coordinate(value)
            exact_slot = None if slot is None else slot + offset
            self._direct_bindings.append(
                (allocation, index, at, exact_slot, witness))
        return authored[0] if scalar else (
            values if isinstance(values, LogicalRegister) else authored)

    def finish(self, *results, placement=(), objective=None):
        portable = super().finish(*results)
        from ..compiler import place

        constraints = (tuple(placement(portable.values))
                       if callable(placement) else tuple(placement or ()))
        exact = tuple(
            local(
                portable.values[allocation][index],
                at=space,
                slot=slot,
                witness=witness,
            ) for allocation, index, space, slot, witness in
            self._direct_bindings)

        return place(
            portable,
            device=self.machine,
            placement=(*constraints, *exact),
            objective=objective,
        )


def _direct_provider(name, inputs, results):
    if isinstance(inputs, dict):
        items = tuple(inputs.items())
    else:
        items = tuple(
            (f"arg{index}", value) for index, value in enumerate(inputs))

    def provider(*_):
        raise RuntimeError("direct profile builder provider must not execute")

    provider.__name__ = str(name)
    provider.__qualname__ = str(name)
    provider.__module__ = "__main__"
    provider.__signature__ = Signature(
        tuple(
            Parameter(key, Parameter.POSITIONAL_OR_KEYWORD, annotation=value)
            for key, value in items),
        return_annotation=_return_annotation(results),
    )
    provider.__annotations__ = {
        **dict(items),
        "return": _return_annotation(results),
    }
    return provider, tuple(key for key, _ in items)


class GadgetBuilder:
    """Advanced direct P2A builder; decorator source is sugar over this form."""

    profile = "p2a"

    def __init__(
        self,
        name,
        *,
        implements,
        signature,
        results=None,
        logical_ports=None,
        reusable_body=False,
    ):
        if results is None:
            results = tuple(signature.values()) if isinstance(
                signature, dict) else tuple(signature)
        provider, self._names = _direct_provider(name, signature, results)
        self.definition = GadgetDefinition(
            provider,
            implements=implements,
            name=str(name),
            logical_ports=logical_ports,
            metadata={"realization_form": "circuit"} if reusable_body else None,
        )
        self.transaction = CompilationContext()
        self._backend = _GadgetBackend(self.transaction, self.definition)
        self.context = self.transaction.context
        self.module = self.transaction.module
        self.insertion_point = self._backend.insertion_point
        self._arguments = dict(zip(self._names, self._backend.arguments()))
        self._finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def input(self, name=0):
        if isinstance(name, int):
            name = self._names[name]
        return self._arguments[name]

    def inputs(self):
        return tuple(self._arguments[name] for name in self._names)

    def apply(self, action, *values, **options):
        if not isinstance(action, LogicalActionRef):
            raise TypeError(
                "GadgetBuilder.apply expects a standard logical action")
        result = self._backend.apply_standard(action.name, values, **options)
        return result[0] if len(result) == 1 else tuple(result)

    def h(self, value):
        return self.apply(LogicalActionRef("h", 1), value)

    def cx(self, control, target, **options):
        return self.apply(LogicalActionRef("cx", 2), control, target, **options)

    def mpp(self, product):
        return self._backend.mpp(product)

    def rotate(self, product, *, angle):
        return self._backend.rotate(product, angle=angle)

    def read_syndrome_ancillas(self, value, *, record=None):
        return self._backend.read_syndrome_ancillas(value, record=record)

    def extract_syndrome(
        self,
        value,
        *,
        record=None,
        schedule=None,
        cx_schedule=None,
        prime=None,
        final_cycle=False,
    ):
        return self._backend.extract_syndrome(
            value,
            record=record,
            schedule=schedule,
            cx_schedule=cx_schedule,
            prime=prime,
            final_cycle=final_cycle,
        )

    def measure_gauges(self, value, *, operators, record=None, phase=None):
        return self._backend.measure_gauges(value,
                                            operators=operators,
                                            record=record,
                                            phase=phase)

    def transition_epoch(self, value, *, to, evidence, logical_map=None):
        return self._backend.transition_epoch(value,
                                              to=to,
                                              evidence=evidence,
                                              logical_map=logical_map)

    def permute(self, value, permutation):
        return self._backend.permute(value, permutation)

    def unpack_resource(self,
                        resource,
                        *,
                        like,
                        encoding=None,
                        logical_ports=None):
        return self._backend.unpack_resource(
            resource,
            like=like,
            encoding=encoding,
            logical_ports=logical_ports,
        )

    def pack_resource(self, payload, *, kind):
        return self._backend.pack_resource(payload, kind=kind)

    def finish(self, *values, **named):
        if self._finished:
            raise RuntimeError("GadgetBuilder root is already finished")
        if values and named:
            raise TypeError(
                "finish accepts positional or named results, not both")
        if named:
            values = tuple(named.values())
        returned = values[0] if len(values) == 1 else tuple(values)
        self._backend.finish(returned)
        self._finished = True
        handle = DefinitionHandle(self._backend.symbol, "gadget", "p2a")
        self.transaction.bind(self.definition, handle)
        build = Build(
            context=self.context,
            module=self.module,
            root=handle,
            profile="p2a",
            pipeline=pipelines.gadgets(),
            evidence=(EvidenceRecord(
                "direct_gadget_verification",
                "qlx-python@0.3",
                "pass",
                ("typed-boundary", "logical-objective", "linear-ownership"),
            ),),
            source_modules=("__main__",),
        )
        self.definition._attach_direct_snapshot(build)
        return build


class ProtocolBuilder:
    """Advanced direct folded P2N call-graph builder."""

    profile = "p2n"

    def __init__(self, name, *, signature, results=None, implements=None):
        if results is None:
            results = tuple(signature.values()) if isinstance(
                signature, dict) else tuple(signature)
        provider, self._names = _direct_provider(name, signature, results)
        self.definition = ProtocolDefinition(provider,
                                             implements=implements,
                                             name=str(name))
        self.transaction = CompilationContext()
        self._backend = _ProtocolBackend(self.transaction, self.definition)
        self.context = self.transaction.context
        self.module = self.transaction.module
        self.insertion_point = self._backend.insertion_point
        self._arguments = dict(zip(self._names, self._backend.arguments()))
        self._finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def input(self, name=0):
        if isinstance(name, int):
            name = self._names[name]
        return self._arguments[name]

    def inputs(self):
        return tuple(self._arguments[name] for name in self._names)

    def call(self, definition, *args, analysis=None):
        kwargs = {} if analysis is None else {"analysis": analysis}
        return self._backend.call(definition, args, kwargs)

    def xor(self, lhs, rhs):
        """Combine two protocol Booleans while preserving their provenance."""
        return self._backend.xor(lhs, rhs)

    def all_false(self, *events):
        """Accept when every supplied attempt outcome is false."""
        return self._backend.all_false(*events)

    def retry(
        self,
        *values,
        until,
        max_attempts,
        exhaustion=None,
        commit_point=None,
    ):
        from ..gadgets import RetryExhaustion

        exhaustion = (RetryExhaustion.REPORT_FAILURE
                      if exhaustion is None else exhaustion)
        return self._backend.retry(
            values,
            until=until,
            max_attempts=max_attempts,
            exhaustion=exhaustion,
            commit_point=commit_point,
        )

    def transition_epoch(self, value, *, to, evidence, logical_map=None):
        return self._backend.transition_epoch(value,
                                              to=to,
                                              evidence=evidence,
                                              logical_map=logical_map)

    def unpack_resource(self,
                        resource,
                        *,
                        like,
                        encoding=None,
                        logical_ports=None):
        return self._backend.unpack_resource(
            resource,
            like=like,
            encoding=encoding,
            logical_ports=logical_ports,
        )

    def allocate_patch(self, encoding, *, region):
        return self._backend.allocate_patch(encoding, region=region)

    def postselect(self, predicate, *, expected=False):
        return self._backend.postselect(predicate, expected=expected)

    def pack_resource(self, payload, *, kind):
        return self._backend.pack_resource(payload, kind=kind)

    def finish(self, *values, **named):
        if self._finished:
            raise RuntimeError("ProtocolBuilder root is already finished")
        if values and named:
            raise TypeError(
                "finish accepts positional or named results, not both")
        if named:
            values = tuple(named.values())
        returned = values[0] if len(values) == 1 else tuple(values)
        self._backend.finish(returned)
        self._finished = True
        handle = DefinitionHandle(self._backend.symbol, "protocol", "p2n")
        self.transaction.bind(self.definition, handle)
        build = Build(
            context=self.context,
            module=self.module,
            root=handle,
            profile="p2n",
            pipeline=pipelines.protocols(),
            evidence=(EvidenceRecord(
                "direct_protocol_verification",
                "qlx-python@0.3",
                "pass",
                ("typed-calls", "linear-ownership", "folded-control"),
            ),),
            source_modules=("__main__",),
        )
        self.definition._attach_direct_snapshot(build)
        return build


class CodeBuilder:
    """Incremental construction utility that freezes to one normal Code."""

    def __init__(self, name, *, n, k, r=0, d=None, block=None, metadata=None):
        self.name, self.n, self.k, self.r = str(name), n, k, r
        self.d, self.block, self.metadata = d, block, metadata
        self._stabilizers = []
        self._gauges = []
        self._anti_stabilizers = []
        self._hx = self._hz = self._gx = self._gz = ()
        self._lx = self._lz = ()
        self._finished = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def add_stabilizers(self, rows, *, basis=None):
        rows = tuple(tuple(row) for row in rows)
        if basis == "x":
            self._hx = (*self._hx, *rows)
        elif basis == "z":
            self._hz = (*self._hz, *rows)
        elif basis is None:
            self._stabilizers.extend(rows)
        else:
            raise ValueError("stabilizer basis must be 'x', 'z', or omitted")
        return self

    def set_gauge_pairs(self, gx, gz):
        self._gx, self._gz = tuple(map(tuple, gx)), tuple(map(tuple, gz))
        return self

    def set_anti_stabilizers(self, rows):
        self._anti_stabilizers = list(rows)
        return self

    def set_logical_pairs(self, lx, lz):
        self._lx, self._lz = tuple(map(tuple, lx)), tuple(map(tuple, lz))
        return self

    def finish(self):
        if self._finished:
            raise RuntimeError("CodeBuilder is already finished")
        self._finished = True
        code_type = SubsystemCode if self.r else StabilizerCode
        return code_type(
            name=self.name,
            n=self.n,
            k=self.k,
            r=self.r,
            d=self.d,
            block=self.block,
            stabilizers=tuple(self._stabilizers),
            gauges=tuple(self._gauges),
            anti_stabilizers=tuple(self._anti_stabilizers),
            hx=self._hx,
            hz=self._hz,
            gx=self._gx,
            gz=self._gz,
            lx=self._lx,
            lz=self._lz,
            metadata=self.metadata,
        )


__all__ = [
    "UnplacedBuilder",
    "ActionBuilder",
    "PlacedBuilder",
    "GadgetBuilder",
    "ProtocolBuilder",
    "CodeBuilder",
]

from .._compat import preserve_legacy_module as _preserve_legacy_module

_preserve_legacy_module(globals(), "cudaq.logical.advanced")
