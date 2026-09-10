# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
from __future__ import annotations

import json

import cudaq.logical as qlx
import pytest


@qlx.code
class ScratchCode:
    block = qlx.codes.CSSBlock(data=1)
    d = 1
    hx = ()
    hz = ()
    lx = ((0,),)
    lz = ((0,),)


@qlx.machine
class ScratchMachine:
    compute = qlx.architecture.Space(capacity=2)


def _scratch_device(name, carrier_count):
    carriers = qlx.architecture.ResourceClass(
        "qubit",
        carrier_count,
        native_actions=("x",),
        native_instruments=(qlx.architecture.physical_instruments.MPP,),
        name=f"{name}_carriers",
    )
    architecture = qlx.architecture.PhysicalMachine(
        f"{name}_architecture",
        resource_classes={carriers.name: carriers},
    )
    builder = qlx.devices.DeviceBuilder(
        name,
        logical=ScratchMachine,
        physical=architecture,
    )
    compute = builder.qec.bind(
        builder.logical.compute,
        encoding=ScratchCode,
    )
    builder.physical.bind(
        compute,
        to=getattr(builder.physical, carriers.name),
    )
    builder.physical.set_operating_point(timing={
        "cycle_ns": 10.0,
        "x_ns": 4.0
    },)
    return builder.build()


one_scratch_device = _scratch_device("one_scratch", 3)
two_scratch_device = _scratch_device("two_scratch", 4)


@qlx.protocol
def uses_internal_scratch(
    block: qlx.patch[ScratchCode],) -> qlx.patch[ScratchCode]:
    scratch = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    scratch = qlx.prepare_zero(scratch)
    qlx.discard(scratch)
    return block


@qlx.protocol
def two_disjoint_calls() -> None:
    left = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    right = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    left = uses_internal_scratch(left)
    right = uses_internal_scratch(right)
    qlx.discard((left, right))


@qlx.protocol
def two_chained_calls() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    block = uses_internal_scratch(block)
    block = uses_internal_scratch(block)
    qlx.discard(block)


@qlx.protocol
def repeated_scratch_call() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    block = qlx.ops.repeat(
        3,
        carries=(block,),
        body=lambda value: uses_internal_scratch(value),
    )
    qlx.discard(block)


@qlx.gadget(implements=qlx.logical.idle)
def ticked_call(block: qlx.patch[ScratchCode],) -> qlx.patch[ScratchCode]:
    qlx.ops.tick()
    return block


@qlx.protocol
def ticked_call_then_allocate() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    block = ticked_call(block)
    fresh = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    qlx.discard((block, fresh))


@qlx.protocol
def repeated_ticked_call_then_allocate() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    block = qlx.ops.repeat(
        2,
        carries=(block,),
        body=lambda value: ticked_call(value),
    )
    fresh = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    qlx.discard((block, fresh))


@qlx.protocol
def zero_repeat_ticked_call_then_allocate() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    block = qlx.ops.repeat(
        0,
        carries=(block,),
        body=lambda value: ticked_call(value),
    )
    fresh = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    qlx.discard((block, fresh))


@qlx.protocol
def alternative_branch_internal_scratch(
        block: qlx.patch[ScratchCode],
        selected: bool) -> qlx.patch[ScratchCode]:

    def scratch_once(live):
        scratch = qlx.ops.allocate_patch(ScratchCode,
                                         region=ScratchMachine.compute)
        scratch = qlx.prepare_zero(scratch)
        qlx.discard(scratch)
        return live

    def one_scratch(live):
        return (scratch_once(live),)

    def two_serial_scratch(live):
        return (scratch_once(scratch_once(live)),)

    block, = qlx.ops.cond(
        selected,
        carries=(block,),
        then=one_scratch,
        else_=two_serial_scratch,
    )
    return block


@qlx.protocol
def internally_conditioned_scratch() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)
    block = qlx.prepare_zero(block)
    block, selected = qlx.mpp(qlx.types.Z(block[0]))
    block = alternative_branch_internal_scratch(block, selected)
    qlx.discard(block)


_COUNT_MAX = (1 << 63) - 1


@qlx.protocol
def max_nested_scratch_repeat() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)

    def outer(value):
        return qlx.ops.repeat(
            _COUNT_MAX,
            carries=(value,),
            body=lambda nested: uses_internal_scratch(nested),
        )

    block = qlx.ops.repeat(1, carries=(block,), body=outer)
    qlx.discard(block)


@qlx.protocol
def overflowing_nested_scratch_repeat() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)

    def outer(value):
        return qlx.ops.repeat(
            _COUNT_MAX,
            carries=(value,),
            body=lambda nested: uses_internal_scratch(nested),
        )

    block = qlx.ops.repeat(2, carries=(block,), body=outer)
    qlx.discard(block)


@qlx.protocol
def zero_nested_scratch_repeat() -> None:
    block = qlx.ops.allocate_patch(ScratchCode, region=ScratchMachine.compute)

    def outer(value):
        return qlx.ops.repeat(
            _COUNT_MAX,
            carries=(value,),
            body=lambda nested: uses_internal_scratch(nested),
        )

    block = qlx.ops.repeat(0, carries=(block,), body=outer)
    qlx.discard(block)


def _schedule(device):
    build = qlx.compile(
        two_disjoint_calls,
        pipeline=qlx.compiler.pipelines.physical(),
        device=device,
    )
    return build, qlx.compiler.schedule(build)


def _raw_p3_build(text, *, root="g"):
    import cudaq.mlir.ir as mlir_ir
    from cudaq.logical.programs import DefinitionHandle

    context = mlir_ir.Context()
    module = mlir_ir.Module.parse(text, context)
    assert module.operation.verify()
    return qlx.compiler.Build(
        context=context,
        module=module,
        root=DefinitionHandle(root, "phys.graph", "p3"),
        profile="p3",
        pipeline=None,
    )


def _shared_call_graph(second_body: bool):
    second = """
    %2 = \"phys.call\"(%1) <{callee = @work, event_id = \"call1\",
      instance = \"root.work.call1\"}> ({
    ^bb0(%current: !phys.state<@q0>):
      %next = phys.delay %current {duration_ns = 3.000000e+00 : f64,
        event_id = \"delay1\"}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %next : !phys.state<@q0>
    }) : (!phys.state<@q0>) -> !phys.state<@q0>
    """ if second_body else """
    %2 = phys.call_template %1 {callee = @work, event_id = "call1",
      instance = "root.work.call1", template_event = "call0"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    """
    return _raw_p3_build(f"""
module attributes {{qlx.profiles = [\"p3\"]}} {{
  phys.machine @arch {{
    phys.resource_class @q {{count = 1 : i64, kind = \"qubit\",
      native_actions = []}}
  }}
  phys.resource @q0 {{index = 0 : i64, kind = \"qubit\",
    resource_class = @q}}
  phys.graph @g on @arch : () -> () {{
    %0 = phys.acquire [@q0] {{event_id = \"acquire0\"}}
      : !phys.state<@q0>
    %1 = \"phys.call\"(%0) <{{callee = @work, event_id = \"call0\",
      instance = \"root.work.call0\"}}> ({{
    ^bb0(%current: !phys.state<@q0>):
      %next = phys.delay %current {{duration_ns = 3.000000e+00 : f64,
        event_id = \"delay0\"}}
        : (!phys.state<@q0>) -> !phys.state<@q0>
      phys.yield %next : !phys.state<@q0>
    }}) : (!phys.state<@q0>) -> !phys.state<@q0>
    {second}
    phys.release %2 {{event_id = \"release0\"}} : !phys.state<@q0>
    phys.return
  }}
}}
""")


def test_call_template_schedule_matches_explicit_body_without_expanding_rows():
    shared = qlx.compiler.schedule(_shared_call_graph(False))
    explicit = qlx.compiler.schedule(_shared_call_graph(True))

    assert shared.makespan_ns == explicit.makespan_ns == 6.0
    invocation = next(
        entry for entry in shared.entries if entry.kind == "call_template")
    canonical = next(
        entry for entry in shared.entries if entry.event_id == "call0")
    assert invocation.duration_ns == canonical.duration_ns == 3.0
    assert invocation.resources == canonical.resources == ("q[0]",)
    assert not any(entry.event_id == "delay1" for entry in shared.entries)
    assert shared.build.module.operation.verify()


def test_schedule_checks_composed_i64_repeat_multiplicity_without_unrolling():
    near_max = qlx.compile(
        max_nested_scratch_repeat,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    near_schedule = qlx.compiler.schedule(near_max)
    repeats = [
        entry for entry in near_schedule.entries if entry.kind == "repeat"
    ]
    assert len(repeats) == 2
    assert {entry.repeat_count for entry in repeats} == {1, _COUNT_MAX}
    assert len(near_schedule.entries) < 20

    zero = qlx.compile(
        zero_nested_scratch_repeat,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    zero_schedule = qlx.compiler.schedule(zero)
    assert len([
        entry for entry in zero_schedule.entries if entry.kind == "repeat"
    ]) == 2
    outer = next(entry for entry in zero_schedule.entries
                 if entry.kind == "repeat" and entry.repeat_count == 0)
    assert outer.duration_ns == 0

    overflowing = qlx.compile(
        overflowing_nested_scratch_repeat,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    with pytest.raises(
            qlx.errors.RepeatCountOverflow,
            match="multiplicity exceeds signed 64-bit",
    ):
        qlx.compiler.schedule(overflowing)


def test_condition_envelope_covers_branch_local_allocations_and_estimate_is_exclusive(
):
    build = qlx.compile(
        internally_conditioned_scratch,
        pipeline=qlx.compiler.pipelines.physical(),
        device=two_scratch_device,
    )
    scheduled = qlx.compiler.schedule(build)
    by_id = {entry.event_id: entry for entry in scheduled.entries}
    condition = next(entry for entry in scheduled.entries if entry.kind == "if")
    descendants = []
    for entry in scheduled.entries:
        current = entry
        while current.parent is not None:
            if current.parent == condition.event_id:
                descendants.append(entry)
                break
            current = by_id[current.parent]
    assert descendants
    assert condition.duration_ns > 0.0
    assert all(condition.start_ns <= entry.start_ns and
               entry.finish_ns <= condition.finish_ns for entry in descendants)
    assert scheduled.build.module.operation.verify()

    envelopes = {
        "call", "repeat", "cond", "if", "while", "try_take", "spacetime_call"
    }
    leaves = tuple(
        entry for entry in scheduled.entries if entry.kind not in envelopes)

    def condition_branch(entry):
        current = entry
        while current.parent is not None:
            if current.parent == condition.event_id:
                return current.branch
            current = by_id[current.parent]
        return None

    outside = tuple(
        entry for entry in leaves if condition_branch(entry) is None)
    branches = {
        name:
            tuple(entry for entry in leaves if condition_branch(entry) == name)
        for name in ("then", "else")
    }

    def physical_resource_count(entry):
        return sum(
            not resource.startswith("control:") for resource in entry.resources)

    expected_active = sum(
        entry.duration_ns * physical_resource_count(entry)
        for entry in outside) + max(
            sum(entry.duration_ns * physical_resource_count(entry)
                for entry in branch)
            for branch in branches.values())

    def peak(entries):
        edges = []
        for entry in entries:
            if entry.duration_ns > 0.0:
                edges.extend(((entry.start_ns, 1), (entry.finish_ns, -1)))
        active = result = 0
        for _, delta in sorted(edges, key=lambda item: (item[0], item[1])):
            active += delta
            result = max(result, active)
        return result

    expected_peak = max(peak(outside + branch) for branch in branches.values())
    estimate = qlx.analysis.estimate(
        scheduled,
        tier=qlx.analysis.Tier.SCHEDULE,
        p_phys=0.0,
        failure_budget=1.0,
    )
    assert estimate.active_resource_time_ns == expected_active
    assert estimate.peak_concurrency == expected_peak
    assert any("worst-case executable branch" in assumption
               for assumption in estimate.assumptions)


def test_released_scratch_binding_is_reused_and_serializes_as_a_resource():
    build, schedule = _schedule(one_scratch_device)
    text = build.to_mlir()
    calls = [
        entry for entry in schedule.entries
        if entry.kind in {"call", "call_template"}
    ]
    canonical = next(entry for entry in calls if entry.kind == "call")
    invocation = next(entry for entry in calls if entry.kind == "call_template")
    scratch_events = [
        entry for entry in schedule.entries
        if entry.kind == "prepare" and entry.parent == canonical.event_id
    ]
    scratch_acquires = [
        entry for entry in schedule.entries
        if entry.kind == "acquire" and entry.parent == canonical.event_id
    ]
    scratch_releases = [
        entry for entry in schedule.entries
        if entry.kind == "release" and entry.parent == canonical.event_id
    ]

    assert build.module.operation.verify()
    assert "phys.allocation_mapping" in text
    assert len(calls) == 2
    assert len(scratch_events) == 1
    assert len(scratch_acquires) == len(scratch_releases) == 1
    assert invocation.template_event == canonical.event_id
    assert canonical.finish_ns <= invocation.start_ns
    assert invocation.data_dependencies == ()
    assert invocation.resource_dependencies == (canonical.event_id,)


def test_disjoint_calls_use_available_scratch_capacity_in_parallel():
    _, schedule = _schedule(two_scratch_device)
    calls = [entry for entry in schedule.entries if entry.kind == "call"]
    scratch_events = [
        entry for entry in schedule.entries if entry.kind == "prepare" and
        entry.parent in {call.event_id for call in calls}
    ]

    assert len(scratch_events) == 2
    assert scratch_events[0].resources != scratch_events[1].resources
    assert scratch_events[0].start_ns == scratch_events[1].start_ns
    assert not set(scratch_events[0].dependencies) & set(
        scratch_events[1].dependencies)


def test_schedule_dependency_causes_survive_ir_replay():
    _, schedule = _schedule(one_scratch_device)
    replayed = qlx.compiler.Build.replay(schedule.build.serialize()).schedule

    assert replayed is not None
    assert replayed.entries == schedule.entries
    assert any(entry.resource_dependencies for entry in replayed.entries)
    assert any(entry.data_dependencies for entry in replayed.entries)


def test_shared_call_resource_effect_serializes_without_boundary_ssa():
    build = qlx.compile(
        two_chained_calls,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    schedule = qlx.compiler.schedule(build)
    calls = [
        entry for entry in schedule.entries
        if entry.kind in {"call", "call_template"}
    ]

    assert len(calls) == 2
    assert [call.kind for call in calls] == ["call", "call_template"]
    assert any(
        record.producer == "fabric-to-phys@0.3" for record in build.evidence)
    assert "phys.allocation_mapping" in build.to_mlir()
    assert "phys.mapping" in build.to_mlir()
    assert not calls[1].data_dependencies
    assert calls[0].event_id in calls[1].resource_dependencies
    assert calls[0].finish_ns <= calls[1].start_ns


def test_folded_repeat_keeps_one_call_template_and_exact_tier3_occupancy():
    build = qlx.compile(
        repeated_scratch_call,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    schedule = qlx.compiler.schedule(build)
    estimate = qlx.estimate(
        schedule,
        tier=qlx.analysis.Tier.SCHEDULE,
        p_phys=0.0,
        failure_budget=1.0,
    )

    assert build.to_mlir().count("cflow.repeat") == 2
    assert len([entry for entry in schedule.entries if entry.kind == "call"
               ]) == 1
    repeat = next(entry for entry in schedule.entries if entry.kind == "repeat")
    outer_release = next(entry for entry in schedule.entries
                         if entry.kind == "release" and entry.parent is None)
    assert repeat.event_id in outer_release.data_dependencies
    assert repeat.event_id not in outer_release.resource_dependencies
    assert estimate.event_counts["prepare"] == 1
    assert estimate.active_resource_time_ns == 30.0
    assert estimate.makespan_ns == 30.0


@pytest.mark.parametrize(
    "protocol,envelope_kind",
    (
        (ticked_call_then_allocate, "call"),
        (repeated_ticked_call_then_allocate, "repeat"),
    ),
)
def test_ticked_call_threads_local_state_without_ordering_new_resources(
    protocol,
    envelope_kind,
):
    build = qlx.compile(
        protocol,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    scheduled = qlx.compiler.schedule(build)
    envelope = next(
        entry for entry in scheduled.entries if entry.kind == envelope_kind)
    barrier = next(entry for entry in scheduled.entries
                   if entry.kind == "barrier" and entry.parent is not None)
    top_level_acquires = [
        entry for entry in scheduled.entries
        if entry.kind == "acquire" and entry.parent is None
    ]

    assert barrier.parent is not None
    assert barrier.resources == envelope.resources
    assert barrier.data_dependencies == ("acquire0",)
    assert barrier.resource_dependencies == ("acquire0",)
    assert not barrier.domain_dependencies
    assert not top_level_acquires[-1].dependencies
    assert top_level_acquires[-1].start_ns == envelope.start_ns
    assert qlx.compiler.Build.replay(
        scheduled.build.serialize()).schedule.entries == scheduled.entries


def test_zero_count_repeat_does_not_export_its_inactive_local_boundary():
    build = qlx.compile(
        zero_repeat_ticked_call_then_allocate,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    scheduled = qlx.compiler.schedule(build)
    repeat = next(
        entry for entry in scheduled.entries if entry.kind == "repeat")
    nested_barrier = next(
        entry for entry in scheduled.entries if entry.kind == "barrier")
    by_id = {entry.event_id: entry for entry in scheduled.entries}
    ancestors = []
    parent = nested_barrier.parent
    while parent is not None:
        ancestors.append(parent)
        parent = by_id[parent].parent
    later_acquire = next(
        entry for entry in scheduled.entries
        if entry.kind == "acquire" and entry.parent is None and
        scheduled.entries.index(entry) > scheduled.entries.index(repeat))

    assert repeat.repeat_count == 0
    assert repeat.event_id in ancestors
    assert nested_barrier.resources == ("one_scratch_carriers[0]",)
    assert nested_barrier.data_dependencies == ("acquire0",)
    assert nested_barrier.resource_dependencies == ("acquire0",)
    assert not nested_barrier.domain_dependencies
    assert not later_acquire.dependencies
    assert qlx.compiler.Build.replay(scheduled.build.serialize()).verify()


def test_branch_clock_frontier_expands_its_structured_envelope():
    build = _raw_p3_build(r"""
module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %q = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %p = phys.prepare %q {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %r = phys.reset %p {event_id = "reset", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %c = "arith.constant"() {event_id = "condition", value = true}
      : () -> i1
    "cflow.if"(%c) <{event_id = "if0"}> ({
      phys.barrier {domains = ["clock"], event_id = "branch_tick"}
        : () -> ()
      cflow.yield
    }, {
      cflow.yield
    }) : (i1) -> ()
    phys.release %r {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}
""")

    scheduled = qlx.compiler.schedule(build)
    by_id = {entry.event_id: entry for entry in scheduled.entries}

    assert by_id["if0"].start_ns == 1.0
    assert by_id["branch_tick"].start_ns == 2.0
    assert by_id["if0"].finish_ns == by_id["branch_tick"].finish_ns == 2.0
    assert by_id["release"].start_ns == 2.0
    assert by_id["release"].domain_dependencies == ("if0",)
    assert scheduled.build.verify()


def test_event_dispatch_clock_frontier_covers_every_branch_and_continuation():
    build = _raw_p3_build(r"""
module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %q = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %p = phys.prepare %q {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %r = phys.reset %p {event_id = "reset", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %event = phys.resource_request "t_state" from @m::@magic {
      event_id = "request"
    } : !event.handle<!phys.resource_payload<@t_state>, "linear">
    "event.try_take"(%event) <{event_id = "take"}> ({
    ^bb0(%resource: !phys.resource_payload<@t_state>):
      phys.discard_resource_payload %resource {event_id = "discard.ready"}
        : !phys.resource_payload<@t_state>
      event.yield
    }, {
    ^bb0(%pending: !event.handle<!phys.resource_payload<@t_state>, "linear">):
      %resource = event.await %pending {event_id = "await.pending"}
        : !event.handle<!phys.resource_payload<@t_state>, "linear">
          -> !phys.resource_payload<@t_state>
      phys.discard_resource_payload %resource {event_id = "discard.pending"}
        : !phys.resource_payload<@t_state>
      event.yield
    }, {
    ^bb0(%status: i8):
      phys.barrier {domains = ["clock"], event_id = "branch_tick"}
        : () -> ()
      event.yield
    }) : (!event.handle<!phys.resource_payload<@t_state>, "linear">) -> ()
    phys.release %r {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}
""")

    scheduled = qlx.compiler.schedule(build)
    by_id = {entry.event_id: entry for entry in scheduled.entries}

    assert by_id["take"].start_ns == 1.0
    assert by_id["take"].finish_ns == 3.0
    assert by_id["branch_tick"].start_ns == 2.0
    assert by_id["discard.pending"].finish_ns == 3.0
    assert by_id["release"].start_ns == 3.0
    assert by_id["release"].domain_dependencies == ("take",)
    assert scheduled.build.verify()


def test_bounded_while_exports_a_branch_clock_frontier():
    build = _raw_p3_build(r"""
module attributes {qlx.profiles = ["p3"], qlx.stages = ["p3"]} {
  phys.machine @arch {
    phys.resource_class @qubits {
      count = 1 : i64, kind = "qubit", native_actions = []
    }
  }
  phys.resource @q0 {
    index = 0 : i64, kind = "qubit", resource_class = @qubits
  }
  phys.graph @g on @arch : () -> () {
    %q = phys.acquire [@q0] {event_id = "acquire"}
      : !phys.state<@q0>
    %p = phys.prepare %q {event_id = "prepare", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %r = phys.reset %p {event_id = "reset", state = "zero"}
      : (!phys.state<@q0>) -> !phys.state<@q0>
    %go = "arith.constant"() {event_id = "go", value = true} : () -> i1
    %out = "cflow.while"(%go) <{
      event_id = "loop", max_iterations = 1 : i64
    }> ({
    ^bb0(%predicate: i1):
      "cflow.while_condition"(%predicate, %predicate) : (i1, i1) -> ()
    }, {
    ^bb0(%predicate: i1):
      phys.barrier {domains = ["clock"], event_id = "loop_tick"}
        : () -> ()
      cflow.yield %predicate : i1
    }) : (i1) -> i1
    phys.release %r {event_id = "release"} : !phys.state<@q0>
    phys.return
  }
}
""")

    scheduled = qlx.compiler.schedule(build)
    by_id = {entry.event_id: entry for entry in scheduled.entries}

    assert by_id["loop"].start_ns == 1.0
    assert by_id["loop_tick"].start_ns == 2.0
    assert by_id["loop"].finish_ns == by_id["loop_tick"].finish_ns == 2.0
    assert by_id["release"].domain_dependencies == ("loop",)
    assert scheduled.build.verify()


def test_replay_rejects_a_redigested_schedule_with_a_missing_barrier_edge():
    import cudaq.mlir.ir as mlir_ir
    from cudaq.logical.compiler.build import _build_bundle_content_sha256

    build = qlx.compile(
        ticked_call_then_allocate,
        pipeline=qlx.compiler.pipelines.physical(),
        device=one_scratch_device,
    )
    scheduled = qlx.compiler.schedule(build)
    barrier = next(
        entry for entry in scheduled.entries if entry.kind == "barrier")
    assert barrier.data_dependencies == barrier.resource_dependencies
    assert len(barrier.data_dependencies) == 1
    predecessor = barrier.data_dependencies[0]
    bundle = json.loads(scheduled.build.serialize())
    line = next(line for line in bundle["module"].splitlines()
                if f'"{barrier.event_id}|' in line)
    forged = line.replace(
        f"deps={predecessor}|data_deps={predecessor}|"
        f"resource_deps={predecessor}|domain_deps=",
        "deps=|data_deps=|resource_deps=|domain_deps=",
    )
    assert forged != line
    bundle["module"] = bundle["module"].replace(line, forged, 1)
    bundle["content_sha256"] = _build_bundle_content_sha256(bundle)

    with pytest.raises(
            mlir_ir.MLIRError,
            match="data_deps must exactly match its graph SSA dependencies",
    ):
        qlx.compiler.Build.replay(
            json.dumps(bundle, sort_keys=True, separators=(",", ":")).encode())
