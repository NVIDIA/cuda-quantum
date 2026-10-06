# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

from dataclasses import InitVar, dataclass, field
import math
import os
import time
import weakref

import cudaq.mlir.ir as mlir_ir

from ._scheduling import SchedulingStrategy, scheduling
from ..errors import (
    RepeatCountOverflow,
    ScheduleConflict,
    UnsupportedSchedulingStrategy,
)
from .build import Build, EvidenceRecord
from .pipeline import PassSpec, Pipeline


@dataclass(frozen=True, slots=True)
class ScheduleEntry:
    event_id: str
    kind: str
    start_ns: float
    duration_ns: float
    resources: tuple[str, ...]
    dependencies: tuple[str, ...] = ()
    parent: str | None = None
    branch: str | None = None
    condition: str | None = None
    max_attempts: int | None = None
    exhaustion: str | None = None
    commit_point: str | None = None
    repeat_count: int | None = None
    repeat_period_ns: float | None = None
    repeat_epilogue_ns: float | None = None
    max_iterations: int | None = None
    callee: str | None = None
    instance: str | None = None
    profile: str | None = None
    template_event: str | None = None
    attempt: str | None = None
    attempt_event: str | None = None
    decision_event: str | None = None
    success_probability: float | None = None
    success_probability_source: str | None = None
    success_probability_evidence: str | None = None
    data_dependencies: tuple[str, ...] = ()
    resource_dependencies: tuple[str, ...] = ()
    domain_dependencies: tuple[str, ...] = ()

    @property
    def finish_ns(self) -> float:
        return self.start_ns + self.duration_ns


_SCHEDULE_AUTH = object()


@dataclass(frozen=True, slots=True)
class _FusedScheduleEstimateEvidence:
    """Private authenticated handoff for estimate-only native scheduling.

    The native boundary has independently verified the transient typed
    schedule and derived the payload before returning.  No portable
    ``phys.schedule`` is published on this path; callers that need one use the
    ordinary :func:`schedule` API.
    """

    payload: str
    _requested_root_symbol: str
    _schedule_symbol: str
    _machine_symbol: str
    _operating_point_symbol: str | None
    _lower_tier_symbol: str
    stage: object
    facets: tuple


class _WeakrefableSchedule:
    """Give slotted schedule artifacts weak-reference support on Python 3.10."""

    __slots__ = ("__weakref__",)


@dataclass(frozen=True, slots=True)
class PhysicalSchedule(_WeakrefableSchedule):
    build: Build
    entries: tuple[ScheduleEntry, ...]
    strategy: SchedulingStrategy
    strategy_domain: str
    provider: str
    provider_version: str
    constraint_profile: str
    constraints: tuple[str, ...]
    timing_profile: tuple[tuple[str, float], ...]
    tie_break: str
    optimization_status: str
    objective_value: float | None
    makespan_ns: float
    _requested_root_symbol: str = field(default="", repr=False)
    _graph_symbol: str = field(default="", repr=False)
    _schedule_symbol: str = field(default="", repr=False)
    _machine_symbol: str = field(default="", repr=False)
    _operating_point_symbol: str | None = field(default=None, repr=False)
    _auth: InitVar[object] = None
    _defer_entries: InitVar[bool] = False

    def __post_init__(self, _auth, _defer_entries=False) -> None:
        if self.strategy is not scheduling.greedy_asap:
            raise ScheduleConflict(
                "physical schedules require the canonical "
                "cudaq.logical.compiler.scheduling.greedy_asap strategy")
        if self.strategy_domain != _PHYSICAL_DOMAIN:
            raise ScheduleConflict(
                "physical schedule strategy_domain must be physical")
        if self.provider != _PROVIDER or self.provider_version != _PROVIDER_VERSION:
            raise ScheduleConflict(
                "physical schedule provider identity/version is invalid")
        if self.constraint_profile != _CONSTRAINT_PROFILE:
            raise ScheduleConflict(
                "physical schedule constraint profile is invalid")
        if tuple(self.constraints) != _ENFORCED_CONSTRAINTS:
            raise ScheduleConflict(
                "physical schedule constraints are not canonical")
        timing_profile = tuple(self.timing_profile)
        if timing_profile != tuple(sorted(timing_profile)):
            raise ScheduleConflict(
                "physical schedule timing profile must be sorted")
        if len({name for name, _ in timing_profile}) != len(timing_profile):
            raise ScheduleConflict(
                "physical schedule timing profile keys must be unique")
        if any(not isinstance(name, str) or not name or
               not isinstance(value, (int, float)) or
               not math.isfinite(float(value)) or float(value) < 0.0
               for name, value in timing_profile):
            raise ScheduleConflict(
                "physical schedule timing profile must contain finite "
                "nonnegative named values")
        object.__setattr__(self, "entries", tuple(self.entries))
        object.__setattr__(self, "constraints", tuple(self.constraints))
        object.__setattr__(
            self, "timing_profile",
            tuple((name, float(value)) for name, value in timing_profile))
        if self.tie_break != _TIE_BREAK:
            raise ScheduleConflict(
                "physical schedule tie-break policy is invalid")
        if self.optimization_status != _OPTIMIZATION_STATUS:
            raise ScheduleConflict(
                "physical schedule optimization status is invalid")
        if self.objective_value is not None:
            raise ScheduleConflict(
                "greedy ASAP schedules cannot record an optimization objective value"
            )
        if not math.isfinite(self.makespan_ns) or self.makespan_ns < 0.0:
            raise ScheduleConflict(
                "physical schedule makespan must be finite and nonnegative")
        if _defer_entries:
            if self.entries:
                raise ScheduleConflict(
                    "a deferred internal schedule cannot carry eager entries")
        else:
            actual_makespan = max(
                (entry.finish_ns
                 for entry in self.entries
                 if entry.parent is None),
                default=0.0,
            )
            if actual_makespan != self.makespan_ns:
                raise ScheduleConflict(
                    "physical schedule makespan must equal its latest "
                    "top-level finish")
        if _auth is not _SCHEDULE_AUTH:
            raise ScheduleConflict(
                "PhysicalSchedule is an opaque verified artifact; obtain it "
                "from cudaq.logical.compiler.schedule() or Build.schedule")
        if (not all(
                isinstance(value, str) and value for value in (
                    self._requested_root_symbol,
                    self._graph_symbol,
                    self._schedule_symbol,
                    self._machine_symbol,
                )) or (self._operating_point_symbol is not None and
                       (not isinstance(self._operating_point_symbol, str) or
                        not self._operating_point_symbol))):
            raise ScheduleConflict(
                "physical schedule retained provenance is incomplete")
        if (self.build.profile == "p3" and
                self.build.root.symbol != self._requested_root_symbol):
            raise ScheduleConflict(
                "physical schedule requested-root provenance differs from its Build"
            )

    @classmethod
    def _from_verified_ir(cls, **values):
        build = values.get("build")
        if build is None or not getattr(build, "_verified", False):
            raise ScheduleConflict(
                "cannot mint PhysicalSchedule without a native-verified "
                "phys.schedule")
        defer_entries = bool(values.pop("_defer_entries", False))
        return cls(
            **values,
            _auth=_SCHEDULE_AUTH,
            _defer_entries=defer_entries,
        )

    def _semantic_key(self):
        return (
            self.entries,
            self.strategy,
            self.strategy_domain,
            self.provider,
            self.provider_version,
            self.constraint_profile,
            self.constraints,
            self.timing_profile,
            self.tie_break,
            self.optimization_status,
            self.objective_value,
            self.makespan_ns,
            self._requested_root_symbol,
            self._graph_symbol,
            self._schedule_symbol,
            self._machine_symbol,
            self._operating_point_symbol,
        )

    def canonical(self):
        """Return the verified IR-derived schedule or reject detached drift."""

        canonical = getattr(self.build, "schedule", None)
        if canonical is None or self._semantic_key() != canonical._semantic_key(
        ):
            raise ScheduleConflict(
                "PhysicalSchedule no longer matches the selected verified "
                "phys.schedule in its Build")
        return canonical

    @property
    def profile(self):
        return self.build.profile

    @property
    def stage(self):
        return self.build.stage

    @property
    def facets(self):
        return self.build.facets

    @property
    def module(self):
        return self.build.module

    @property
    def root(self):
        return self.build.root

    def to_mlir(self):
        return self.build.to_mlir()


def _text(attribute):
    value = getattr(attribute, "value", None)
    return str(value if value is not None else attribute).strip('"').lstrip("@")


def _symbol(operation):
    try:
        return _text(operation.attributes["sym_name"])
    except KeyError:
        return None


_UNSET = object()
_PHYSICAL_DOMAIN = "physical"
_PROVIDER = "qlx.compiler.greedy_asap"
_PROVIDER_VERSION = "1"
_CONSTRAINT_PROFILE = "qlx.physical_schedule.constraints/v1"
_ENFORCED_CONSTRAINTS = (
    "graph_ssa_dependencies",
    "physical_resource_exclusion",
    "allocation_mapping_after",
    "structured_control_exclusivity",
    "folded_region_bounds",
    "resolved_event_durations",
)
_TIE_BREAK = "stable_graph_order"
_OPTIMIZATION_STATUS = "not_applicable"


def _unsupported_strategy(value, *,
                          channel: str) -> UnsupportedSchedulingStrategy:
    if isinstance(value, SchedulingStrategy):
        received = f"{value.name!r} in domains {value.domains!r}"
    else:
        received = f"{value!r} ({type(value).__name__})"
    if channel == "objective":
        reason = (
            "objective= is a rejection-only compatibility channel; no physical "
            "optimization provider implements this request")
    elif isinstance(value, str):
        reason = "raw/stale string scheduling aliases are not accepted"
    elif isinstance(
            value, SchedulingStrategy) and not value.supports(_PHYSICAL_DOMAIN):
        reason = "the typed strategy does not support the physical domain"
    elif isinstance(value,
                    SchedulingStrategy) and value == scheduling.greedy_asap:
        reason = "a constructed lookalike is not the canonical strategy singleton"
    else:
        reason = "the typed strategy has no registered physical provider"
    return UnsupportedSchedulingStrategy(
        f"unsupported physical scheduling request {received}: {reason}; use "
        "strategy=cudaq.logical.compiler.scheduling.greedy_asap")


def _resolve_strategy(*, strategy, objective) -> SchedulingStrategy:
    if objective is not _UNSET:
        if strategy is not _UNSET:
            raise UnsupportedSchedulingStrategy(
                "cudaq.logical.compiler.schedule accepts strategy= or the "
                "rejection-only objective= compatibility channel, not both; "
                "use strategy=cudaq.logical.compiler.scheduling.greedy_asap")
        raise _unsupported_strategy(objective, channel="objective")
    if strategy is _UNSET:
        return scheduling.greedy_asap
    if strategy is not scheduling.greedy_asap:
        raise _unsupported_strategy(strategy, channel="strategy")
    if not strategy.supports(_PHYSICAL_DOMAIN):
        # This is unreachable for the canonical singleton, but keeps the
        # provider boundary fail-closed if its declaration is edited.
        raise _unsupported_strategy(strategy, channel="strategy")
    return strategy


def schedule(
    root,
    *,
    device=None,
    strategy=_UNSET,
    objective=_UNSET,
    _estimate=False,
    _estimate_full_workload=False,
):
    """Produce one deterministic greedy-ASAP P3 physical schedule.

    ``strategy=`` accepts only the canonical typed
    :data:`cudaq.logical.compiler.scheduling.greedy_asap` provider.
    ``objective=`` is
    retained solely to diagnose the previously advertised but unimplemented
    optimization requests; it never falls back to greedy scheduling.
    """

    strategy = _resolve_strategy(strategy=strategy, objective=objective)
    if not isinstance(_estimate_full_workload, bool):
        raise TypeError("private full-workload estimate flag must be bool")
    if _estimate_full_workload and not _estimate:
        raise TypeError("full-workload costing requires fused estimation")
    from .compile import compile
    from .pipeline import pipelines

    if isinstance(root, PhysicalSchedule):
        raise ScheduleConflict(
            "cudaq.logical.compiler.schedule does not reschedule an existing "
            "PhysicalSchedule; use its retained verified schedule directly")
    elif isinstance(root, Build):
        # Only P3 can carry a phys.schedule. Inspecting an earlier published
        # stage would defensively clone its module just to prove an invariant
        # already guaranteed by the stage contract, before physical lowering
        # clones that same immutable input a second time.
        if root.profile == "p3" and root.schedule is not None:
            raise ScheduleConflict(
                "cudaq.logical.compiler.schedule does not reschedule a Build that "
                "already carries a selected phys.schedule")
        if root.profile == "p3" and device is not None:
            raise ScheduleConflict(
                "device= cannot override the machine or operating point of "
                "an existing P3 physical graph")
        build = root
    else:
        build = compile(
            root,
            pipeline=pipelines.physical() if device is not None else None,
            device=device,
            _transient=True,
        )
    if build.profile != "p3" and device is not None:
        build = compile(
            build,
            pipeline=pipelines.physical(),
            device=device,
            _transient=True,
        )
    if build.profile != "p3":
        raise ValueError(
            "cudaq.logical.schedule requires a P3 physical graph Build or a "
            "definition "
            "with device=; it never reconstructs a schedule directly from P2")
    # Lower against a private replayed copy; exposed module views are
    # disposable inspection values rather than compiler authorities.
    clone_started = None
    if os.getenv("QLX_PROFILE_SCHEDULE_ESTIMATE") is not None:
        clone_started = time.perf_counter()
    module = build._fresh_module()
    if clone_started is not None:
        print(
            "phys-estimate-schedule python-module-clone "
            f"{time.perf_counter() - clone_started:.6f}s",
            flush=True,
        )
    symbols = {
        _symbol(operation.operation): operation.operation
        for operation in module.body.operations
        if _symbol(operation.operation) is not None
    }
    graph_symbol = build.root.symbol
    root = symbols.get(graph_symbol)
    if root is not None and "graph" in root.attributes:
        graph_symbol = _text(root.attributes["graph"])
    graph = symbols.get(graph_symbol)
    if graph is None or graph.name != "phys.graph":
        raise ValueError(
            "cudaq.logical.schedule requires a phys.graph root, not a device "
            "definition")

    schedule_symbol = f"{build.root.symbol}_schedule"
    if schedule_symbol in symbols:
        raise ScheduleConflict(
            f"physical schedule symbol @{schedule_symbol} already exists")
    from cudaq.logical._native import native

    estimate_payload = None
    lower_tier_symbol = ""
    if _estimate:
        if not callable(_estimate):
            raise TypeError(
                "the private fused-estimation hook must be callable")
        lower_tier_symbol = _estimate(
            module,
            graph_symbol,
            schedule_symbol,
        )
        if not isinstance(lower_tier_symbol, str) or not lower_tier_symbol:
            raise ScheduleConflict(
                "fused Tier-3 estimation omitted its analytical lower tier")
    try:
        if _estimate:
            estimate_payload = native._schedule_verified_and_estimate_json(
                module,
                graph_symbol,
                schedule_symbol,
                lower_tier_symbol,
                _estimate_full_workload,
            )
        else:
            native.run_pass(
                module,
                f"phys-schedule{{graph={graph_symbol} result={schedule_symbol}}}",
                verify=False,
            )
    except (mlir_ir.MLIRError, RuntimeError) as error:
        message = str(error)
        if "multiplicity exceeds signed 64-bit" in message:
            raise RepeatCountOverflow(
                "physical repeat multiplicity exceeds signed 64-bit range"
            ) from error
        if "requires a bounded cflow.while" in message:
            raise NotImplementedError(
                "the generic physical scheduler requires a bounded "
                "cflow.while; use a symbolic controller scheduler for an "
                "unbounded runtime loop") from error
        if "physical scheduling does not support region control" in message:
            raise ValueError(message) from error
        raise ScheduleConflict(
            f"native greedy-ASAP scheduling failed for phys.graph "
            f"@{graph_symbol}: {message}") from error

    schedule_pass = PassSpec(
        "phys-schedule",
        provides_facets=("physical_schedule",),
    )
    schedule_pipeline = (Pipeline(
        (schedule_pass,), output_profile="p3") if build.pipeline is None else
                         build.pipeline.append(schedule_pass))
    if _estimate:
        return _FusedScheduleEstimateEvidence(
            payload=estimate_payload,
            _requested_root_symbol=build.root.symbol,
            _schedule_symbol=schedule_symbol,
            _machine_symbol=_text(graph.attributes["architecture"]),
            _operating_point_symbol=(_text(graph.attributes["operating_point"])
                                     if "operating_point" in graph.attributes
                                     else None),
            _lower_tier_symbol=lower_tier_symbol,
            stage=build.stage,
            facets=tuple(schedule_pipeline.apply_facets(build.facets)),
        )

    if root is not graph:
        # A derived P3 root owns the selected schedule explicitly.  This is a
        # top-level typed provenance link; all graph traversal and scheduling
        # remain in the native pass.
        root.attributes["schedule"] = mlir_ir.FlatSymbolRefAttr.get(
            schedule_symbol, context=module.context)
    # The native pass independently verifies its materialized schedule against
    # the retained graph before this verified increment is sealed.  The source
    # P3 Build remains authenticated and unchanged; graph-rooted results avoid
    # a second redundant Build-wide verification only after that native proof.
    build_constructor = (Build._from_verified_increment
                         if root is graph else Build)
    scheduled_build = build_constructor(
        context=module.context,
        module=module,
        root=build.root,
        profile="p3",
        facets=build.facets,
        pipeline=schedule_pipeline,
        evidence=(
            *build.evidence,
            EvidenceRecord(
                kind="physical_schedule_verification",
                producer="cudaq-logical-native@0.3",
                result="pass",
                obligations=(
                    "dependency-order",
                    "resource-exclusion",
                    "allocation-lifetime-exclusion",
                    "structured-control",
                    "resolved-durations",
                    "native-construction-invariants",
                ),
            ),
        ),
        placement=build.placement,
        qec_selection=build.qec_selection,
        experiment=build.experiment,
        source_modules=build.source_modules,
        device=getattr(build, "_device", None),
    )
    selected = scheduled_build._parse_schedule(scheduled_build._module)
    if selected is None:
        raise ScheduleConflict(
            f"native phys-schedule omitted @{schedule_symbol}")
    # PhysicalSchedule owns its authenticated Build.  Retaining the schedule
    # strongly from that Build would form a cycle which, at paper scale, keeps
    # the complete scheduled module and millions of parsed rows alive after a
    # caller releases the schedule.  A weak cache preserves identity while the
    # public schedule is live and lets reference counting reclaim it promptly.
    scheduled_build._cache["schedule"] = weakref.ref(selected)
    return selected
