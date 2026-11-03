# Bring your own rotation synthesizer

A Python object can supply Clifford+T words to the logical compiler. An external
synthesizer may be implemented in Python or wrap a Rust library with PyO3;
it needs no C++, MLIR, or C ABI integration.

```{code-block} python
import cudaq.logical as cql
import external  # Illustrative third-party package.

@cql.program
def circuit() -> bool:
    q = cql.prepare_zero()
    q = cql.u3(q, 0.3, 1.1, -0.4)
    return cql.measure_z(q)

build = cql.compiler.synthesize(
    circuit, gate_set=cql.compiler.clifford_t, precision=1e-10,
    synthesizer=external.Synthesizer(),
)
print(build.synthesis)
```

The same object works with an estimation target:

```{code-block} python
import cudaq

cudaq.set_target(cql.targets.clifford_t_target(
    precision=1e-10, synthesizer=external.Synthesizer()))
estimate = cudaq.estimate(my_kernel)
```

For an explicit pipeline, pass the object to
`cql.compiler.pipelines.clifford_t(precision=..., synthesizer=...)` or
`cql.compiler.clifford_t.pipeline(...)`, then compile as usual.

## External synthesizer contract

Implement `name`, `version`, and `rz(angle, precision)`. Optionally implement
`u3(theta, phi, lam, precision)` to synthesize the whole unitary. Both methods
return `cudaq.logical.compiler.CliffordTWord`:

```{code-block} python
from cudaq.logical.compiler import CliffordTWord

class Synthesizer:
    name = "external-synth"
    version = "0.1.0"

    def rz(self, angle: float, precision: float) -> CliffordTWord:
        gates, measured_error = self.synthesize_with_my_library(angle, precision)
        return CliffordTWord(
            gates=tuple(gates), requested_precision=precision,
            achieved_error=measured_error, algorithm="my-algorithm",
            implementation=self.name, implementation_version=self.version,
        )
```

Inputs are static floating-point angles in radians and a finite precision in
`(0, 1)`. The target is the exact binary value of each input float; decimal
round trips and reduction with a rounded value of 2π can change that target.
Output gates are **chronological**: the first tuple element is applied
first. The accepted alphabet is `h, s, sdg, t, tdg, x, y, z` (case insensitive).
CUDA-Q normalizes it to positive H/S/T. Unsupported gates, exceptions, and
words outside the requested precision fail compilation with a diagnostic.
Return an empty tuple for the identity. Raise an exception on synthesis failure.

The error metric is projective operator norm:
`min_phase ||word - exp(i phase) target||₂`. CUDA-Q independently multiplies
and checks every third-party word; it does not trust `achieved_error` as proof.
When available, `mpmath` verifies at a precision derived from the requested
bound and word length. Without it, double precision is used only when the
requested bound is at least `256 * machine_epsilon * (normalized_gate_count + 1)`.
Smaller requests fail with an instruction to install `mpmath`; the compiler
never relaxes a requested precision to its numerical floor. The verification
includes a numerical margin and is a numerical check, not an interval proof.

Methods must return the same word for the same inputs. The compiler caches
requests per pass run, including operation kind, all angles, and precision.
Compiler-owned Pauli basis changes, parity ladders, and exact π/4 shortcuts
remain outside the plugin. Exact rotations may never invoke a callback.
`Build.synthesis.synthesizer` and the legalization evidence identify the
selected plugin as `name/version`. Omitting `synthesizer` retains the native
GridSynth path and its existing evidence producer.

Reuse the same synthesizer instance across compilations: each distinct object
receives a process-local registry name and stays alive until interpreter exit.
Pipeline text can be replayed only in a process with that registration.
A directly synthesized Clifford+T Build can be reused with the same selection
and an equal or looser precision. Reuse preserves its provenance and original
precision. To change synthesizer or tighten precision, compile the original
program or an unsynthesized P0 Build. After transforming a synthesized Build
to PBC or absorbing its Clifford frame, further synthesis requests also require
the original source; the transformed Build retains earlier approximation error.

`PyGridSynth`, exported from `cql.compiler`, adapts the optional `pygridsynth`
package. For code-specific RPP lowering, use
`cql.qec.synthesis.synthesis_strategy(synthesizer, h=..., s=..., cx=..., t=...)`.
The original `qec.synthesis.CliffordTWord` import remains supported.

## The U3 action

`cql.u3(q, theta, phi, lam, precision=...)` means
`Rz(phi) Ry(theta) Rz(lam)`, up to global phase: `lam` acts first.
Numeric angles retain their original binary f64 values, including sign and
magnitude. `cql.pi` expressions are reduced exactly modulo 2π and retain rational
metadata. Traced `Float64Value` angles can be authored, but must be specialized
before synthesis, as with `rz`.

Precision bounds the **whole** U3. If the plugin has no `u3` method, CUDA-Q
splits it into Rz(lam), Ry(theta), Rz(phi), giving each rotation ε/3.
Exact π/4 Euler angles bypass whole-U3 dispatch. Mixed symbolic/numeric requests
also use the split path when an angle carries exact π/4-lattice metadata, so
the callback's floating-point inputs cannot discard that exact intent.
Exact Clifford U3 actions use a shortest positive H/S word, so
`u3(π/2, 0, π)` emits a single H and no T.

This prototype supports U3 only in P0 programs. Synthesize before placement;
the device path reports an error for an unlegalized U3. Quake U3 import,
single-qubit merging, probabilistic output, and repeat-until-success synthesis
are future work.

Run `examples/standalone/06_custom_synthesizer.py` for a self-contained example.
Its toy synthesizer delegates to CUDA-Q GridSynth and composes Euler rotations;
it exercises the interface and makes no quality or performance claim.
It measures each Rz word with `cudaq.synth.rz_error` and sums the three bounds
for U3, whose basis-change Cliffords are exact. It asks GridSynth for half the
callback precision to leave room for the independent verifier's numerical
margin when `mpmath` is absent; the requested contract remains unchanged.
