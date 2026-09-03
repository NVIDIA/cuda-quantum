# Defining codes

A code in CUDA-Q Logical is validated algebra: constructing one **proves** its
stabilizer and logical structure — mutual commutation, canonical pairing, and
rank against `n - k - r` — or fails with a diagnostic. There is no way to hold
an algebraically invalid code and pass it to a gadget or a device.

## CSS codes: state the checks once

The shipped examples define the self-dual `[[7,1,3]]` Steane code by stating its
CSS check rows once:

```{literalinclude} ../../../examples/03_code_and_gadget.py
:language: python
:start-at: "@qlx.code"
:end-before: "@qlx.objective"
:caption: The Steane code, from examples/03_code_and_gadget.py.
```

Check rows are carrier-index supports. `n` comes from the block, `k` from the
logical pairs, and `r` from the (absent) gauge declarations — supplying any of
them explicitly is a checked assertion, never a hint. Construction derives the
remaining structure, so downstream code can trust it; the same shipped example
asserts `Steane.n == 7`, `Steane.k == 1`, and `Steane.d.value == 3` and finds
six independent X/Z stabilizer checks. An algebra that does not close fails at
construction:

```python
import cudaq.logical as qlx

try:
    @qlx.code
    class Bad:
        block = qlx.codes.CSSBlock(data=2, sx=1, sz=1)
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
import cudaq.logical as qlx

@qlx.code
class Repetition3:
    block = qlx.codes.Block(data=3, syndrome=2)
    d = qlx.codes.Distance.asymmetric(x=3, z=1)
    stabilizers = (qlx.types.Z(0) @ qlx.types.Z(1),
                   qlx.types.Z(1) @ qlx.types.Z(2))
    lx = (qlx.types.X(0) @ qlx.types.X(1) @ qlx.types.X(2),)
    lz = (qlx.types.Z(0),)
```

## Distance is evidence, not an integer

A distance in CUDA-Q Logical is a typed evidence value, not a bare integer:

```python
claimed  = qlx.codes.Distance.claimed(12)
proved   = qlx.codes.Distance.exact(12, method="exhaustive_search",
                                    provenance=qlx.analysis.citation("search report"))
bounded  = qlx.codes.Distance.lower_bound(5, method="topological_cycle_bound",
                                          provenance=qlx.analysis.citation("geometry note"))
unknown  = qlx.codes.Distance.unknown("derive by search")
```

A bare `d = 3` in a code body normalizes to `claimed` — a recorded assertion,
never a proof. The evidence-bearing constructors (`exact`, `lower_bound`,
`upper_bound`) require `method=` and `provenance=`; omitting them raises
`TypeError` rather than silently upgrading the claim:

```python
import cudaq.logical as qlx

try:
    # TypeError: Distance.exact evidence requires method= and provenance=;
    # use Distance.claimed(...) to record an unproved assertion
    qlx.codes.Distance.exact(12)
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
surface_3 = qlx.codes.Surface[3]        # the rotated [[9, 1, 3]] code
assert surface_3 is qlx.codes.rotated_surface(3)
assert (surface_3.n, surface_3.k, surface_3.d.value) == (9, 1, 3)
assert surface_3.block.size == 17       # 9 data + 8 syndrome carriers
```

alongside `qlx.codes.Steane`, `qlx.codes.Repetition`, `qlx.codes.RM15` (the
`[[15,1,3]]` Reed–Muller code), and `qlx.codes.BareQubit` (the trivial
distance-1 code used by the Stim-emission fixture).

A parameterized family is an ordinary function decorated with `@qlx.code` that
returns a `qlx.codes.CSSCode`; bracket syntax specializes it:

```python
import cudaq.logical as qlx

@qlx.code
def repetition(distance: int):
    """The ``[[distance, 1, distance]]`` bit-flip repetition family."""
    checks = tuple((i, i + 1) for i in range(distance - 1))
    return qlx.codes.CSSCode(
        block=qlx.codes.CSSBlock(data=distance, sx=0, sz=distance - 1),
        d=qlx.codes.Distance.asymmetric(x=distance, z=1),
        hz=checks,
        lx=(tuple(range(distance)),),
        lz=((0,),),
    )

rep5 = repetition(5)      # or: repetition[5]
```

Specialization is interned — `repetition(5) is repetition(5)` and
`qlx.codes.Surface[3] is qlx.codes.rotated_surface(3)` — so code identity, and
hence selection and cache keys, never depends on spelling.

## What you get for free

Every validated code synthesizes a default encoding (all `k` logicals in
canonical order) that gadget signatures reference by name — the compiled
artifacts show it as `@Steane_default_encoding`. `qlx.materialize` lowers the
code to a named `fabric.code` artifact, and the gadget factories consume the
code directly:

```python
import cudaq.logical as qlx

prep    = qlx.gadgets.prepare_zero(qlx.codes.Steane)     # |0>_L preparation
round_  = qlx.gadgets.css_memory_round(qlx.codes.Steane) # one syndrome round
readout = qlx.gadgets.logical_measure(qlx.codes.Steane, basis="z")
```

What those gadgets are, and how their logical claims are verified, is the
subject of [gadgets and verification](gadgets-and-verification.md).

## Where to go next

- [Gadgets and verification](gadgets-and-verification.md) — realize objectives
  on your code and check the claims.
- The [quickstart](../quickstart.md) runs the Steane code and gadget of this
  page end to end.
- The [example gallery](../example-gallery/index.md) embeds the shipped,
  test-executed sources this page draws from.
