# Defining codes

A code in CUDA-Q Logical is validated algebra: constructing one **proves** its
stabilizer and logical structure — mutual commutation, canonical pairing, and
rank against `n - k - r` — or fails with a diagnostic. You cannot hold an
algebraically invalid code and pass it to a gadget or a device.

## CSS codes: state the checks once

The shipped examples define the self-dual `[[7,1,3]]` Steane code by stating its
CSS check rows once:

```{eval-rst}
.. literalinclude:: ../../../examples/03_code_and_gadget.py
   :language: python
   :start-at: @ql.code
   :end-before: @ql.objective
   :caption: The Steane code, from examples/03_code_and_gadget.py.
```

Check rows are carrier-index supports. `n` comes from the block, `k` from the
logical pairs, and `r` from the (absent) gauge declarations — if you supply any
of them explicitly, it counts as a checked assertion, never a hint.
Construction derives the remaining structure, so downstream code can trust it.
The same shipped example asserts `Steane.n == 7`, `Steane.k == 1`, and
`Steane.d.value == 3` and finds six independent X/Z stabilizer checks. An
algebra that does not close fails at construction:

```python
import cudaq.logical as ql

try:
    @ql.code
    class Bad:
        block = ql.codes.CSSBlock(data=2, sx=1, sz=1)
        d = 1
        hx = ((0, 1),)
        hz = ((0, 1),)   # two checks for n-k-r = 1: overconstrained
        lx = ((0,),)
        lz = ((1,),)
except ValueError as exc:
    assert "stabilizer rank must equal n-k-r = 1, got 2" in str(exc)
```

For non-CSS stabilizer codes, the general Pauli spelling states generators as
Pauli products instead of support rows:

```python
import cudaq.logical as ql

@ql.code
class Repetition3:
    block = ql.codes.Block(data=3, syndrome=2)
    d = ql.codes.Distance.asymmetric(x=3, z=1)
    stabilizers = (ql.types.Z(0) @ ql.types.Z(1),
                   ql.types.Z(1) @ ql.types.Z(2))
    lx = (ql.types.X(0) @ ql.types.X(1) @ ql.types.X(2),)
    lz = (ql.types.Z(0),)
```

## Distance is evidence, not an integer

A distance in CUDA-Q Logical is a typed evidence value, not a bare integer:

```python
claimed  = ql.codes.Distance.claimed(12)
proved   = ql.codes.Distance.exact(12, method="exhaustive_search",
                                    provenance=ql.analysis.citation("search report"))
bounded  = ql.codes.Distance.lower_bound(5, method="topological_cycle_bound",
                                          provenance=ql.analysis.citation("geometry note"))
unknown  = ql.codes.Distance.unknown("derive by search")
```

A bare `d = 3` in a code body normalizes to `claimed` — a recorded assertion,
never a proof. The evidence-bearing constructors (`exact`, `lower_bound`,
`upper_bound`) require `method=` and `provenance=`; omit them and you get a
`TypeError`, not a silent upgrade of the claim:

```python
import cudaq.logical as ql

try:
    # TypeError: Distance.exact evidence requires method= and provenance=;
    # use Distance.claimed(...) to record an unproved assertion
    ql.codes.Distance.exact(12)
except TypeError as exc:
    assert "method= and provenance=" in str(exc)
```

`Distance.asymmetric` carries independent X- and Z-basis evidence — the catalog
repetition code is `d_x = 3, d_z = 1` — and `Distance.unknown(...)` is the
honest zero: estimators that need distance scaling report _missing evidence_
instead of inventing a number.

## The catalog and parameterized families

The shipped catalog covers the standard teaching codes:

```python
surface_3 = ql.codes.Surface[3]        # the rotated [[9, 1, 3]] code
assert surface_3 is ql.codes.rotated_surface(3)
assert (surface_3.n, surface_3.k, surface_3.d.value) == (9, 1, 3)
assert surface_3.block.size == 17       # 9 data + 8 syndrome carriers
```

Alongside the surface family, the catalog ships `ql.codes.Steane`,
`ql.codes.Repetition`, `ql.codes.RM15` (the `[[15,1,3]]` Reed–Muller code), and
`ql.codes.BareQubit` (the trivial distance-1 code used by the Stim-emission
fixture).

To define a parameterized family, decorate an ordinary function with
`@ql.code` and return a `ql.codes.CSSCode`; bracket syntax specializes it:

```python
import cudaq.logical as ql

@ql.code
def repetition(distance: int):
    """The ``[[distance, 1, distance]]`` bit-flip repetition family."""
    checks = tuple((i, i + 1) for i in range(distance - 1))
    return ql.codes.CSSCode(
        block=ql.codes.CSSBlock(data=distance, sx=0, sz=distance - 1),
        d=ql.codes.Distance.asymmetric(x=distance, z=1),
        hz=checks,
        lx=(tuple(range(distance)),),
        lz=((0,),),
    )

rep5 = repetition(5)      # or: repetition[5]
```

Specialization is interned — `repetition(5) is repetition(5)` and
`ql.codes.Surface[3] is ql.codes.rotated_surface(3)` — so code identity, and
hence selection and cache keys, never depends on spelling.

## What you get for free

Every validated code synthesizes a default encoding — all `k` logical qubits
in canonical order — and gadget signatures reference it by name; the compiled
artifacts show it as `@Steane_default_encoding`. `ql.materialize` lowers the
code to a named `fabric.code` artifact, and the gadget factories consume the
code directly:

```python
import cudaq.logical as ql

prep    = ql.gadgets.prepare_zero(ql.codes.Steane)     # |0>_L preparation
round_  = ql.gadgets.css_memory_round(ql.codes.Steane) # one syndrome round
readout = ql.gadgets.logical_measure(ql.codes.Steane, basis="z")
```

What those gadgets are, and how their logical claims are verified, is the
subject of [gadgets and verification](gadgets-and-verification.md).

## Where to go next

- [Gadgets and verification](gadgets-and-verification.md) — realize objectives
  on your code and check the claims.
- The [quick start](../getting-started/quickstart.md) runs the Steane code and
  gadget of this page end to end.
- [Examples](examples/index.md) links the shipped, test-executed sources this
  page draws from.
