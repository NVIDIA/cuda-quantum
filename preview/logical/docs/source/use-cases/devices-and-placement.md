# Devices and placement

CUDA-Q Logical separates the machines that most stacks blur into one "backend"
object. The separation is a firewall: each stage sees only its own structure,
and a program can never branch on a later-stage cost fact it should not know.

## Machines, devices, architectures

- A **logical machine** (`@ql.machine`) is the P1 structure contract: regions,
  resource streams, capabilities, and plain integer capacities. No codes, no
  error rates:

  ```{eval-rst}
  .. literalinclude:: ../../../examples/02_p1_placement.py
     :language: python
     :lines: 13-21
     :caption: A logical machine with one capable region (examples/02_p1_placement.py).
  ```

  Capabilities are typed keys in an open `ql.machine` vocabulary —
  `logical_compute`, `logical_measurement`, `logical_factory`, and
  `resource_transfer` are the ones the compiler itself interprets.

- A **QEC machine** carries the P2 refinement: encoded block pools, one selected
  encoding per region, and block capacities.

- An immutable **device** binds adjacent machines explicitly. A device with only
  a logical machine is complete for placement work; binding an encoding per
  region grows it to a QEC device.

For homogeneous cases you never write the normalized containers explicitly. The
device builder separates logical region facts from the vertical encoding
binding:

```python
import cudaq.logical as ql

builder = ql.devices.DeviceBuilder("SteaneMemory")
memory = builder.logical.add_memory(capacity=8)
builder.qec.bind(memory, encoding=ql.codes.Steane)
SteaneMemory = builder.build()

assert SteaneMemory.layers == (ql.stages.P1, ql.stages.P2)
```

`builder.qec.bind(...)` fixes the QEC refinement of a logical region; the
singular name is deliberate — each region has one selected encoding.

### Stop at the layer your study needs

`ql.devices.DeviceBuilder` is progressively complete. `build()` does not demand
code facts that the requested compiler stage cannot use:

| Declared layers | Builder boundary                                                                                                | Suitable work                                  |
| --------------- | --------------------------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| P1              | `logical.add_compute(...)`, `logical.add_memory(...)`, `logical.add_factory(...)`, or `logical.add_region(...)` | logical placement, capacities, resource supply |
| P1 + P2         | `qec.bind(...)`                                                                                                 | code selection, gadgets, static estimation     |

A placement-only device is therefore complete as written:

```python
logical = ql.devices.DeviceBuilder("LogicalPlacement")
logical.logical.add_compute(capacity=64)
LogicalPlacement = logical.build()

assert LogicalPlacement.layers == (ql.stages.P1,)
```

The stack views contain only the layers actually declared, so a P1-only device
cannot accidentally display P2 block pools. Common member labels are inferred:
`add_compute`, `add_memory`, and `add_factory` attach typed capabilities and
generate region and resource-stream labels with suffixes as needed.
`DeviceBuilder` itself always has an explicit stable name.

Ready-made QEC device stacks also ship as compilation targets — the
`surface_target(distance=3, logical_capacity=1)` stack in
`examples/00_cudaq_logical_resource_estimate.py` binds a rotated-surface-code
QEC machine under a logical machine without any hand-written builder code.

## Automatic placement: constraints, preferences, witnesses

Placement solves hard constraints and soft preferences over the machine, and the
solution is inspectable evidence, not solver state:

```{eval-rst}
.. literalinclude:: ../../../examples/02_p1_placement.py
   :language: python
   :lines: 32-43
   :caption: Placing the Bell program and inspecting the witness (examples/02_p1_placement.py).
```

`place` materializes P0 before invoking the selector. Values are addressed
through its typed view — `p0.values.data` for a named allocation group,
`p0.values[0]` for structural ordinals — and resolved immediately to
`LogicalValueRef`s; Python variable names are diagnostics, never placement
identity. Pass an existing P0 `Build` only when you intentionally inspected or
reused it.

The witness records every binding, the machine, the objective, any _soft_
preference the solver had to give up, and the deterministic tie-break:

- `p1.placement.bindings` — one `PlacementBinding` per logical owner, naming its
  region and slot;
- `p1.placement.objective` and `p1.placement.relaxed_preferences` — what was
  optimized and what was surrendered;
- `ql.compiler.Build.replay(p1.serialize()).placement == p1.placement` — the
  witness is part of the immutable, replayable build.

Hard constraints are hard. Requiring a capability that no region provides fails
closed:

% invisible-code-block: python % % p0, TwoSlotMachine = load_ql_example( %
"preview/logical/examples/02_p1_placement.py", "p0", "TwoSlotMachine")

```python
try:
    # ValueError: no machine space satisfies the placement constraints
    ql.compiler.place(
        p0,
        device=TwoSlotMachine,
        placement=(ql.architecture.require_capability(
            ql.architecture.capability.logical_factory),),
    )
except ValueError as exc:
    assert "no machine space satisfies the placement constraints" in str(exc)
```

What is refused is silently relaxing a hard requirement. Soft preferences, by
contrast, may be surrendered — and the surrender is reported:

```python
p1 = ql.compiler.place(
    p0,
    device=TwoSlotMachine,
    placement=(
        ql.architecture.colocate(p0.values.data),
        ql.architecture.prefer(
            space=TwoSlotMachine.compute,
            for_=ql.architecture.lifecycle.ACTIVE,
        ),
    ),
    objective=ql.architecture.metric.expected_spacetime_volume,
)

assert p1.placement.relaxed_preferences == ()
```

The constraint vocabulary is `ql.architecture`: `colocate`, `allow_spaces`,
`require_capability`, `prefer`, and `local` for exact-slot pinning. Exact slots
are singleton constraints, not a verification bypass — hand-authored placements
produce the same verified witness as solved ones.

## Code-agnostic P1

P1 placement never selects an encoding.
`ql.architecture.colocate(p0.values.data)` keeps several logical owners in one
logical region, but each owner consumes a distinct P1 slot, and a
`PlacementBinding` carries no `encoding` field. Encodings enter at P2 — bound
per region through `DeviceBuilder.qec.bind(...)` or carried by a compilation
target — and the P2 witness is separate from `p1.placement`: changing the
encoding does not rewrite the verified P1 artifact.

## Where the firewall pays off

Placement constraints speak machine vocabulary — regions, slots, capabilities —
so a portable program re-places by swapping `device=`, and a machine can be
reused across many programs. Because capacities and capabilities live on the
machine and encodings live on the device binding, neither the program nor the
estimate code changes when the study moves from one machine to another.

## Continue from here

- [Magic states and protocols](magic-states-and-protocols.md) — typed resource
  supply and a concrete 15-to-1 factory.
- [Logical Clifford+T synthesis](logical-synthesis.md) — legalizing logical
  rotations before placement.
- [Examples](examples.md) — the placement example in context of the full shipped
  set.
