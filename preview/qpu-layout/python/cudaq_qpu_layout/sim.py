#!/usr/bin/env python3
# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Reference model for laying a virtual circuit out on a region-based QPU.

Walks value-semantics (wire-set) Quake IR and *makes its own* layout decisions:
where each virtual qubit lives, how qubits move to meet, and which timestep each
operation runs in. It does not simulate quantum state -- the output is a
`trace.py` document describing the placement and schedule it chose.

Because the model decides everything itself, it consumes plain virtual-qubit
Quake and deliberately ignores any `{region = @rN}` / `quake.move` the
region-lowering passes may have added: those passes are an alternative path to
compare against, not an input.

Movement follows the calling convention of `LoweringQubitsToRegions.md`: a
region is entered and left only through its ports, so carrying a qubit from one
region to another is three legs -- compute wire onto an out-port, out-port
across to the destination's in-port, in-port onto a compute wire. Displacing a
resident to make room is the same act: it leaves through an out-port and, being
still live, must be carried to a compute wire elsewhere.

Placement, routing and scheduling are separate policy objects so a better
heuristic can replace one without disturbing the others -- the contract
described in `QubitLayout.md`.

Not modeled yet:
  - Pinnacle Clifford-frame merging, which serializes non-Clifford operations
    across regions that have been entangled together.
  - Heterogeneous region sizes and magic-state factories.
  - Classically-conditioned operations; a measurement is recorded as an ordinary
    op and its outcome is never used.

Run standalone:
    python3 layout_sim.py payload.mlir --regions 2 --region-size 4
"""

from cudaq.mlir.ir import Module
from cudaq.kernel.utils import getMLIRContext

from .model import QpuModel
from .trace import (TraceBuilder, COMPUTE, IN, OUT, PORT_IN, PORT_OUT,
                    CROSS)

WIRE_TYPE = "!quake.wire"
WIRE_SOURCES = ("quake.borrow_wire", "quake.null_wire")
WIRE_SINKS = ("quake.return_wire", "quake.sink")
MEASUREMENTS = ("quake.mz", "quake.my", "quake.mx")
IGNORED = ("func.return", "cc.return", "quake.discriminate", "quake.wire_set")


class LayoutError(RuntimeError):
    """The circuit cannot be laid out on the given QPU model."""


# ===----------------------------------------------------------------------=== #
# Placement state
# ===----------------------------------------------------------------------=== #


class Placement:
    """Which virtual qubit occupies which site.

    A site is `(region, kind, slot)`. Compute wires are the scarce resource,
    bounded by `region_size`; ports are unbounded, so a port slot is handed out
    on demand.
    """

    def __init__(self, model):
        self.model = model
        self.pos = {}  # vq -> (region, kind, slot)
        self.sites = [{COMPUTE: {}, IN: {}, OUT: {}}
                      for _ in range(model.num_regions)]
        self.free_at = {}  # site -> timestep at which it becomes available

    def region_of(self, vq):
        return self.pos[vq][0]

    def kind_of(self, vq):
        return self.pos[vq][1]

    def slot_of(self, vq):
        return self.pos[vq][2]

    def free_compute_slots(self, region):
        taken = self.sites[region][COMPUTE]
        return [s for s in range(self.model.region_size) if s not in taken]

    def next_port(self, region, kind):
        """Ports are unbounded; hand out the lowest unused index."""
        taken = self.sites[region][kind]
        slot = 0
        while slot in taken:
            slot += 1
        return slot

    def residents(self, region):
        """The qubits on this region's compute wires."""
        return list(self.sites[region][COMPUTE].values())

    def occupancy(self, region):
        return len(self.sites[region][COMPUTE])

    def available_at(self, site):
        """When a site last emptied -- nothing may land there before then."""
        return self.free_at.get(site, 0)

    def assign(self, vq, region, kind, slot):
        self.pos[vq] = (region, kind, slot)
        self.sites[region][kind][slot] = vq

    def relocate(self, vq, region, kind, slot, vacated_at=0):
        old = self.pos[vq]
        del self.sites[old[0]][old[1]][old[2]]
        self.free_at[old] = vacated_at
        self.assign(vq, region, kind, slot)
        return old

    def release(self, vq, vacated_at=0):
        old = self.pos.pop(vq)
        del self.sites[old[0]][old[1]][old[2]]
        self.free_at[old] = vacated_at
        return old


# ===----------------------------------------------------------------------=== #
# Policies
# ===----------------------------------------------------------------------=== #


class FirstFitPlacer:
    """Interaction-aware first fit.

    A qubit whose first use entangles it with an already-placed partner is put
    in that partner's region, so an interaction never costs a move if capacity
    allows. Otherwise it takes the first region with room.
    """

    def __init__(self, model, placement):
        self.model = model
        self.placement = placement

    def place(self, vq, partners):
        preferred = [self.placement.region_of(p)
                     for p in partners
                     if p in self.placement.pos]
        order = preferred + [r for r in range(self.model.num_regions)
                             if r not in preferred]
        for region in order:
            free = self.placement.free_compute_slots(region)
            if free:
                self.placement.assign(vq, region, COMPUTE, free[0])
                return (region, COMPUTE, free[0])
        raise LayoutError(
            f"no free slot for virtual qubit {vq}: all {self.model.num_regions} "
            f"regions of size {self.model.region_size} are full "
            f"(capacity {self.model.capacity})")


class Router:
    """Emits the moves that bring an op's operands together.

    Every operand is drawn into one destination region. Regions are all-to-all,
    so once co-located the operands interact directly -- nothing moves within a
    region. Intra-region topology, when it is modeled, belongs here as a move,
    never as a swap.

    A region is only entered and left through its ports, so drawing a qubit into
    a region is always three legs -- compute wire onto an out-port, out-port
    across to the destination's in-port, in-port onto a compute wire. When the
    destination is full an idle resident vacates onto an out-port first; it is
    still live, so it must then be carried to a compute wire somewhere else.
    """

    def __init__(self, model, placement, scheduler, builder):
        self.model = model
        self.placement = placement
        self.sched = scheduler
        self.builder = builder

    def route(self, operands):
        if len(operands) < 2:
            return
        dest = self._choose_region(operands)
        for vq in operands:
            if self.placement.region_of(vq) != dest:
                self._move_into(vq, dest, operands)

    def _choose_region(self, operands):
        """The region already holding the most operands; ties go to the first."""
        counts = {}
        for vq in operands:
            r = self.placement.region_of(vq)
            counts[r] = counts.get(r, 0) + 1
        best = max(counts.values())
        for vq in operands:
            r = self.placement.region_of(vq)
            if counts[r] == best:
                return r

    def _leg(self, vq, kind, dest_region, dest_kind, dest_slot, cost):
        """One leg of a move. Waits for both the qubit and the target site."""
        site = (dest_region, dest_kind, dest_slot)
        t = max(self.sched.time_for([vq]), self.placement.available_at(site))
        src = self.placement.relocate(vq, *site, vacated_at=t + cost)
        self.builder.move(t, vq, kind, src, site, cost)
        self.sched.commit([vq], t, cost)
        self.builder.observe_occupancy(dest_region,
                                       self.placement.occupancy(dest_region))

    def _park_out(self, vq):
        """Compute wire -> out-port: the explicit act that keeps a qubit alive
        when it must give up its slot."""
        region = self.placement.region_of(vq)
        self._leg(vq, PORT_OUT, region, OUT,
                  self.placement.next_port(region, OUT), self.model.port_cost)

    def _land_from_out(self, vq, dest, slot):
        """Out-port -> destination in-port -> destination compute wire."""
        self._leg(vq, CROSS, dest, IN, self.placement.next_port(dest, IN),
                  self.model.move_cost)
        self._leg(vq, PORT_IN, dest, COMPUTE, slot, self.model.port_cost)

    def _move_into(self, vq, dest, operands):
        free = self.placement.free_compute_slots(dest)
        if free:
            self._park_out(vq)
            self._land_from_out(vq, dest, free[0])
            return

        # The region is full, so an idle resident must vacate onto an out-port
        # before the incoming qubit can land on its wire.
        idle = [r for r in self.placement.residents(dest) if r not in operands]
        if not idle:
            raise LayoutError(
                f"region {dest} holds only operands of the current operation; "
                f"cannot bring virtual qubit {vq} in "
                f"(region_size={self.model.region_size})")
        evicted = idle[0]
        slot = self.placement.slot_of(evicted)

        self._park_out(evicted)
        self._park_out(vq)
        self._land_from_out(vq, dest, slot)
        self._rehome(evicted)

    def _rehome(self, vq):
        """Carry a qubit waiting on an out-port to a compute wire with room.

        A qubit parked on a port is in transit, not stored: keeping it alive
        means landing it somewhere it can next be used.
        """
        for region in range(self.model.num_regions):
            free = self.placement.free_compute_slots(region)
            if free:
                self._land_from_out(vq, region, free[0])
                return
        raise LayoutError(
            f"virtual qubit {vq} was displaced but no region has a free compute "
            f"wire to receive it; the QPU is at capacity "
            f"({self.model.capacity} qubits)")


class AsapScheduler:
    """As-soon-as-possible list scheduling.

    Each virtual qubit carries the timestep at which it is next free. An
    operation runs as soon as all of its qubits are free, so operations on
    disjoint qubits -- in one region or across regions -- share a timestep.
    That parallelism is what the trace exists to show.
    """

    def __init__(self):
        self.ready = {}

    def ready_time(self, vq):
        return self.ready.get(vq, 0)

    def time_for(self, vqs):
        return max((self.ready_time(vq) for vq in vqs), default=0)

    def commit(self, vqs, t, cost):
        for vq in vqs:
            self.ready[vq] = t + cost


# ===----------------------------------------------------------------------=== #
# IR walking
# ===----------------------------------------------------------------------=== #


def _children(op):
    for region in op.regions:
        for block in region.blocks:
            for inner in block.operations:
                yield inner.operation


def _is_wire(value):
    return str(value.type) == WIRE_TYPE


def _find_entrypoint(module):
    for op in _children(module.operation):
        if op.name != "func.func":
            continue
        for attr in op.attributes:
            if attr == "cudaq-entrypoint":
                return op
    return None


def _check_straight_line(func):
    """Reject anything the model cannot cost: branches and unrolled-away loops."""
    body = func.regions[0]
    if len(body.blocks) > 1:
        raise LayoutError(
            f"kernel '{func.opview.sym_name.value}' has "
            f"{len(body.blocks)} basic blocks; the layout simulator handles "
            "straight-line kernels only. Lower with a pipeline that fully "
            "unrolls loops and flattens branches.")
    for op in _children(func):
        if op.regions:
            raise LayoutError(
                f"'{op.name}' carries a nested region; the layout simulator "
                "handles straight-line kernels only. Lower with a pipeline "
                "that fully unrolls loops and flattens branches.")


def _constant_of(value):
    """The literal value of a gate parameter, when it is a constant."""
    try:
        owner = value.owner.opview
    except Exception:
        return None
    for name in ("arith.constant", "complex.constant"):
        if value.owner.name == name:
            try:
                return owner.value.value
            except Exception:
                return None
    return None


class Simulator:

    def __init__(self, model):
        self.model = model
        self.builder = TraceBuilder(model)
        self.placement = Placement(model)
        self.sched = AsapScheduler()
        self.placer = FirstFitPlacer(model, self.placement)
        self.router = Router(model, self.placement, self.sched, self.builder)
        self.vq_of = {}  # MLIR wire Value -> virtual qubit id
        self._next_vq = 0

    # -- virtual qubit identity, recovered from the SSA data flow -------------

    def _new_vq(self, value):
        vq = self._next_vq
        self._next_vq += 1
        self.vq_of[value] = vq
        self.builder.num_vqubits += 1
        return vq

    def _vq(self, value):
        if value not in self.vq_of:
            return self._new_vq(value)
        return self.vq_of[value]

    def _thread(self, op):
        """Carry virtual qubit ids from `!quake.wire` operands to results."""
        ins = [o for o in op.operands if _is_wire(o)]
        outs = [r for r in op.results if _is_wire(r)]
        for src, dst in zip(ins, outs):
            self.vq_of[dst] = self._vq(src)

    # -- walking -------------------------------------------------------------

    def run(self, module):
        entry = _find_entrypoint(module)
        if entry is None:
            raise LayoutError("no `cudaq-entrypoint` function found in payload")
        _check_straight_line(entry)
        self.builder.entry = entry.opview.sym_name.value
        for op in _children(entry):
            self._visit(op)
        return self.builder

    def _visit(self, op):
        name = op.name

        if name in WIRE_SOURCES:
            self._new_vq(op.results[0])
            return

        if name in WIRE_SINKS:
            vq = self._vq(op.operands[0])
            if vq in self.placement.pos:
                t = self.sched.ready_time(vq)
                self.builder.delta(t, "release", vq,
                                   src=self.placement.release(vq, vacated_at=t))
            return

        if name in IGNORED:
            return

        view = op.opview
        if not hasattr(view, "targets"):
            self.builder.unmodeled.append(name)
            return

        self._gate(op, view)

    def _gate(self, op, view):
        # Measurements carry `targets` but no `controls` group.
        controls = [self._vq(c) for c in getattr(view, "controls", [])
                    if _is_wire(c)]
        targets = [self._vq(t) for t in view.targets]
        operands = controls + targets
        params = [_constant_of(p) for p in getattr(view, "parameters", [])]

        # Place on first use, hinting with the qubits this op entangles it with.
        for vq in operands:
            if vq not in self.placement.pos:
                others = [o for o in operands if o != vq]
                site = self.placer.place(vq, others)
                # A wire another qubit has just vacated is not free until then.
                self.sched.ready[vq] = max(self.sched.ready_time(vq),
                                           self.placement.available_at(site))
                self.builder.delta(self.sched.time_for(operands), "assign", vq,
                                   dst=site)
                self.builder.observe_occupancy(
                    site[0], self.placement.occupancy(site[0]))

        self.router.route(operands)

        t = self.sched.time_for(operands)
        gate = op.name.split(".", 1)[1]
        self.builder.op(t, gate, self.placement.region_of(operands[0]),
                        controls, targets, params, self.placement.pos)
        self.sched.commit(operands, t, self.model.gate_cost)
        if op.name in MEASUREMENTS:
            self.builder.num_measurements += 1
        self._thread(op)


def simulate_module(module, model=None):
    """Lay an already-parsed Quake module out on `model`."""
    if not module.operation.verify():
        raise LayoutError("Quake module failed verification before layout")
    return Simulator(model or QpuModel()).run(module)


def simulate(mlir_text, model=None, context=None):
    """Lay a value-semantics Quake payload out on `model`. Returns a TraceBuilder."""
    ctx = context or getMLIRContext()
    return simulate_module(Module.parse(mlir_text, context=ctx), model)


def main():
    import argparse, sys
    from .trace import summarize

    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("payload", nargs="?", help="Quake MLIR file (default: stdin)")
    p.add_argument("--regions", type=int, default=2)
    p.add_argument("--region-size", type=int, default=2)
    p.add_argument("--move-cost", type=int, default=4)
    p.add_argument("-o", "--output", help="write the trace JSON here")
    p.add_argument("--viewer", metavar="PATH",
                   help="write a standalone HTML viewer with the trace embedded")
    p.add_argument("-q", "--quiet", action="store_true",
                   help="print only the summary line")
    args = p.parse_args()

    model = QpuModel(num_regions=args.regions,
                     region_size=args.region_size,
                     move_cost=args.move_cost)
    src = open(args.payload).read() if args.payload else sys.stdin.read()
    builder = simulate(src, model)
    doc = builder.to_json()
    if args.output:
        with open(args.output, "w") as f:
            f.write(builder.dumps())
    if args.viewer:
        from .viewer import write_viewer
        print("viewer: " + write_viewer(doc, args.viewer))
    if args.quiet or args.output or args.viewer:
        print(summarize(doc))
    else:
        print(builder.dumps())

