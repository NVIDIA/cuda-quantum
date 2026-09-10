# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""The layout trace: what happened on the QPU at each timestep.

The schema is deliberately delta-based -- a step records the ops that fired, the
moves that were needed, and the occupancy *changes* -- rather than a full
occupancy snapshot. A consumer reconstructs the placement at any timestep by
replaying `deltas` from an empty QPU. This is the format the visualization tool
will read, so `SCHEMA` is versioned.

A qubit occupies a *site*: `{region, kind, slot}`, where `kind` is one of the
region's compute wires, its in-ports, or its out-ports. Ports are first-class
locations, so a qubit crossing between regions is visibly in flight -- it leaves
a compute wire onto an out-port, crosses to the destination's in-port, and only
then lands on a compute wire. Gates only ever run on `compute` sites.

    {
      "schema": "cudaq-qpu-layout-trace/2",
      "entry": "bell_cross",
      "model": {...},
      "steps": [ {"t": 0, "ops": [...], "moves": [...], "deltas": [...]} ],
      "summary": {...}
    }
"""

import json

SCHEMA = "cudaq-qpu-layout-trace/3"

# Site kinds.
COMPUTE = "compute"
IN = "in"
OUT = "out"

# Move kinds. A region crossing is always the three-leg sequence
# PORT_OUT -> CROSS -> PORT_IN. Regions are all-to-all, so nothing moves within
# one; when intra-region topology is modeled it will add a move kind here, not
# a swap.
PORT_OUT = "port-out"
CROSS = "cross"
PORT_IN = "port-in"
PORT_KINDS = (PORT_OUT, PORT_IN)


def site(region, kind, slot):
    return {"region": region, "kind": kind, "slot": slot}


class TraceBuilder:
    """Accumulates timestep events, then renders the trace document."""

    def __init__(self, model):
        self.model = model
        self.entry = None
        self.steps = {}  # t -> {"ops": [], "moves": [], "deltas": []}
        self.num_vqubits = 0
        self.num_measurements = 0
        self.peak = [0] * model.num_regions
        self.busy = [set() for _ in range(model.num_regions)]
        self.unmodeled = []  # op names walked but not interpreted

    def _step(self, t):
        return self.steps.setdefault(t, {"ops": [], "moves": [], "deltas": []})

    def op(self, t, gate, region, controls, targets, params, sites):
        """Record a gate firing at timestep `t`."""
        self._step(t)["ops"].append({
            "gate": gate,
            "region": region,
            "controls": [site_for(vq, sites) for vq in controls],
            "targets": [site_for(vq, sites) for vq in targets],
            "params": params,
        })
        self.busy[region].add(t)

    def move(self, t, vq, kind, src, dst, cost):
        """Record a qubit relocation at timestep `t`."""
        self._step(t)["moves"].append({
            "vq": vq,
            "kind": kind,
            "from": site(*src),
            "to": site(*dst),
            "cost": cost,
        })
        self.delta(t, "move", vq, src=src, dst=dst)

    def delta(self, t, event, vq, src=None, dst=None):
        entry = {"event": event, "vq": vq}
        if src is not None:
            entry["from"] = site(*src)
        if dst is not None:
            entry["to"] = site(*dst)
        self._step(t)["deltas"].append(entry)

    def observe_occupancy(self, region, count):
        self.peak[region] = max(self.peak[region], count)

    def to_json(self):
        steps = [{"t": t, **self.steps[t]} for t in sorted(self.steps)]
        moves = [m for s in steps for m in s["moves"]]
        # Depth is the span of the schedule, not of the trace: trailing steps
        # that only record releases do not lengthen the circuit.
        op_steps = [s["t"] for s in steps if s["ops"]]
        depth = max(op_steps) + 1 if op_steps else 0
        return {
            "schema": SCHEMA,
            "entry": self.entry,
            "model": self.model.to_json(),
            "steps": steps,
            "summary": {
                "depth": depth,
                "num_vqubits": self.num_vqubits,
                "num_measurements": self.num_measurements,
                "moves": {
                    # `cross` counts region crossings, so it is also the number
                    # of inter-region transfers; each carries two port hops.
                    "cross": sum(1 for m in moves if m["kind"] == CROSS),
                    "port": sum(1 for m in moves if m["kind"] in PORT_KINDS),
                },
                "total_move_cost": sum(m["cost"] for m in moves),
                "region_utilization": [{
                    "region": r,
                    "peak": self.peak[r],
                    "busy_steps": len(self.busy[r])
                } for r in range(self.model.num_regions)],
                "unmodeled_ops": sorted(set(self.unmodeled)),
            },
        }

    def dumps(self, indent=2):
        return json.dumps(self.to_json(), indent=indent)


def site_for(vq, sites):
    """An op operand's site. Gates only run on compute wires, so `kind` is
    implicit and left out."""
    region, kind, slot = sites[vq]
    return {"vq": vq, "region": region, "slot": slot}


def replay(trace):
    """Reconstruct occupancy step by step from the delta stream.

    Yields `(t, {vq: (region, kind, slot)})` after applying each step's deltas.
    This is how a consumer recovers the full placement the trace does not store.
    """
    pos = {}
    for step in trace["steps"]:
        for d in step["deltas"]:
            if d["event"] == "release":
                pos.pop(d["vq"], None)
            else:
                to = d["to"]
                pos[d["vq"]] = (to["region"], to["kind"], to["slot"])
        yield step["t"], dict(pos)


def summarize(trace):
    """One-line human-readable digest of a rendered trace."""
    s = trace["summary"]
    m = s["moves"]
    return (f"entry={trace['entry']} depth={s['depth']} "
            f"vqubits={s['num_vqubits']} "
            f"moves={m['cross']}cross/{m['port']}port "
            f"move_cost={s['total_move_cost']}")
