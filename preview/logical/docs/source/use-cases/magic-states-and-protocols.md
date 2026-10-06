# Magic states and protocols

Non-Clifford computation is where fault-tolerant resource estimation stops being
a memory benchmark: Pauli-product rotations and measurements need magic states,
and magic states come from distillation protocols. CUDA-Q Logical treats each
protocol as a verified, inspectable definition — never a string-named leaf the
compiler trusts.

In the programming model, rotations (`cudaq.logical.rz` and its siblings, and
`cudaq.logical.resource_rotate` where a magic state is consumed) and
Pauli-product measurement (`cudaq.logical.mpp`) are _logical primitives_:

```python
import cudaq.logical as cql


@cql.program
def parity_check() -> bool:
    q = cql.allocate(3, state=cql.types.zero)
    q[0], q[1], q[2], parity = cql.mpp(
        cql.types.X(q[0]) @ cql.types.Y(q[1]) @ cql.types.Z(q[2]))
    cql.discard(q)
    return parity
```

The program states the observable; which code-specific gadgets and protocols
realize it is a P2 selection decision, recorded as evidence — not a fact the
application author supplies.

## Phase convention

CUDA-Q Logical fixes one rotation convention everywhere — the frontend and the
synthesis pass both assume it:

```{math}
R_P(\theta) = \exp\!\left(-\tfrac{i}{2}\,\theta P\right),
\qquad P \in \{X, Y, Z, \dots\}.
```

This is the standard (Nielsen–Chuang) convention. Two consequences are worth
stating outright because they trip up cross-checks against other tools:

- **Litinski's angle is off by a factor of two.** Litinski writes rotations as
  $P_\phi = \exp(i\phi P)$, so his $\phi$ and our $\theta$ relate by
  $\theta = -2\phi$. A Litinski "$P_{\pi/8}$" is our $R_P(\pi/4)$ — the magic
  (T-class) rotation. Always convert before comparing angles.
- **The Clifford hierarchy lands on quarter-turns of $\theta$**, not eighths:
  $R_P(\pi)$ is a Pauli, $R_P(\pi/2)$ is a Clifford, and $R_P(\pi/4)$ is the
  non-Clifford magic rotation that consumes one T state.

Concretely, for $P = Z$ the diagonal rotation
{math}`R_Z(\theta) = \operatorname{diag}(e^{-i\theta/2},\, e^{+i\theta/2})`
equals the named gates **up to an unobservable global phase**:

| angle $\theta$ | $R_Z(\theta)$                                         | equals (up to global phase)                                   | class     |
| -------------- | ----------------------------------------------------- | ------------------------------------------------------------- | --------- |
| $\pi/4$        | {math}`\operatorname{diag}(e^{-i\pi/8}, e^{+i\pi/8})` | $e^{-i\pi/8}\,T$, with {math}`T=\operatorname{diag}(1, e^{i\pi/4})` | magic (T) |
| $\pi/2$        | {math}`\operatorname{diag}(e^{-i\pi/4}, e^{+i\pi/4})` | $e^{-i\pi/4}\,S$, with {math}`S=\operatorname{diag}(1, i)=T^2`      | Clifford  |
| $\pi$          | {math}`\operatorname{diag}(e^{-i\pi/2}, e^{+i\pi/2})` | $e^{-i\pi/2}\,Z$, with {math}`Z=\operatorname{diag}(1, -1)=S^2=T^4` | Pauli     |

So {math}`R_Z(k\,\pi/4) = T^k \pmod 8` up to global phase — the exact-word fast
path the native `qlx-synthesize-rotations` pass takes before ever calling
`gridsynth`: $k=1\to T$, $k=2\to S$, $k=4\to Z$, and $T^8 = I$. We drop
global phase throughout: it is unobservable and carries no logical content.

### Authoring exact angles with `cudaq.logical.algebra.pi`

Writing `angle=0.7853981633974483` leaves the compiler to _infer_ that you
meant $\pi/4$ from a float, within a tolerance. To state the intent exactly,
author with the symbol `cudaq.logical.algebra.pi` — an exact rational multiple
of $\pi$ that arithmetic keeps exact:

```python
@cql.program
def exact_angles() -> bool:
    q = cql.allocate(1, state=cql.types.zero)
    q[0] = cql.rz(q[0], cql.algebra.pi / 4)
    q[0] = cql.rz(q[0], 3 * cql.algebra.pi / 4)
    q[0] = cql.rz(q[0], cql.algebra.pi / 8)
    return cql.measure_z(q[0])

cql.compile(exact_angles)
```

An `Angle` authored this way stamps its reduced coefficient onto the rotation
op, so the synthesis dispatch classifies it **authoritatively** —
`cudaq.logical.algebra.pi / 2` is the Clifford point and
`cudaq.logical.algebra.pi / 4` the magic point by construction, never a
near-lattice float that a tolerance might snap or miss.
`float(cudaq.logical.algebra.pi / 4)` still yields the ordinary radian value,
so a raw float angle (e.g. `0.3`) keeps the numeric path unchanged — both
surfaces coexist.

Synthesis legalizes off-lattice rotations to Clifford+T; see
[Logical Clifford+T synthesis](logical-synthesis.md).

## Typed resource kinds

Magic states are typed resources, not ad-hoc qubits. The standard library
namespace `cudaq.logical.logical` declares the kinds — `T_STATE`,
`RAW_T_STATE`, `Y_STATE`, `CCZ_STATE`, `CS_STATE`, `ENCODED_BELL_PAIR` — each
with a typed consume action, and its `produce(...)` builds the objective a
production protocol claims to implement. A protocol body sees resources through
`cudaq.logical.types.resource[...]` handles: `cudaq.logical.request_many` draws
raw inputs, `cudaq.logical.unpack_resource` opens a resource into a patch, and
`cudaq.logical.pack_resource` certifies the output kind.

## 15-to-1: a concrete factory

`examples/standalone/03_magic_state_distillation.py` authors the real five-row
triorthogonal circuit with the root authoring facade — not an analytical
placeholder:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/03_magic_state_distillation.py
   :language: python
   :start-at: import cudaq.logical as cql
   :end-before: # %%
   :caption: The 15-to-1 T-state protocol (examples/standalone/03_magic_state_distillation.py).
```

The protocol unpacks fifteen linear raw-state inputs onto bare patches; eleven
resource-assisted product rotations from
`cudaq.logical.protocols.FIFTEEN_TO_ONE_ROTATION_STEPS` apply the triorthogonal
circuit; `cudaq.logical.protocols.bare_s` converts the resulting T† on the odd
row to the canonical T|+⟩; and exactly the four even rows must measure $+X$ —
recorded with `cudaq.logical.postselect`, so the acceptance condition is part
of the definition rather than a comment about it.

Because the protocol is an ordinary compiled definition, the static estimation
tier counts it directly:

```{eval-rst}
.. literalinclude:: ../../../examples/standalone/03_magic_state_distillation.py
   :language: python
   :start-at: # Estimate the protocol directly
   :end-at: peak live patches
```

```text
15-to-1 static resource estimate:
  raw T-state requests: 15
  peak live patches: 5
```

The `selection` operation count records the four postselection checks, so a
downstream study does not have to rediscover that this factory rejects.

The leading-order analytical curves attach to the _same_ protocol library entry
for supply/demand studies — output error $35p^3$ and acceptance $1 - 15p$:

```python
model = cql.protocols.DISTILL_15TO1_T
assert model.output_error(1e-3) == 3.510537795740123e-08
assert model.acceptance_probability(1e-3) == 0.9851045810483217
```

The `cudaq.logical.protocols` library also exposes the reusable pieces —
`FIFTEEN_TO_ONE_ROTATION_STEPS` and their supports, `bare_s`, `bare_measure_x`,
and the ready-made `distill_15to1` definition — so you can compose your own
production protocol from verified parts.

:::{admonition} Evidence boundary :class: note

A static estimate counts what the declared protocol costs; it does not sample
the factory, decode its checks, or model the noise that makes distillation
necessary. The estimate reports postselection as counts and success rows, not
as simulated accept/reject statistics. :::

## Compact factory models at P3

A protocol says what a factory does; at P3 a factory binding can also say how
fast it does it. `cudaq.logical.devices.FactoryModel` carries a startup
latency, a steady output interval, and the evidence behind both:

```{eval-rst}
.. literalinclude:: ../../../examples/gidney_ekera_factory.py
   :language: python
   :start-at: builder.physical.bind(factory_qec
   :end-before: builder.physical.set_operating_point
   :dedent: 4
   :caption: A declared factory model with explicit provenance (examples/gidney_ekera_factory.py).
```

Declaring the numbers is one option, and `evidence=` is where you say they are
an assumption rather than a measurement. The other option is to derive them:
`cudaq.logical.compiler.factory_model` reads a verified P3 schedule and
characterizes the factory it implements.

```{eval-rst}
.. literalinclude:: ../../../examples/gidney_ekera_factory.py
   :language: python
   :start-at: factory_model = cql.compiler.factory_model(
   :end-before: # Make sure the compiler
   :dedent: 4
```

The derived model reports the same two figures the declared one asserts —
`startup_cycles` and `output_interval_cycles` — plus a characterization
recording the physical units it took. A supply/demand study can then use a
compact factory model in place of the full protocol schedule, without losing
track of where its timing came from.

## Continue from here

- [Logical Clifford+T synthesis](logical-synthesis.md) — the other route for
  off-lattice rotations.
- [Devices and placement](devices-and-placement.md) — where factory regions and
  resource streams live on a machine.
- [Examples](examples/index.md) — the distillation example in the context of
  the full shipped set.
