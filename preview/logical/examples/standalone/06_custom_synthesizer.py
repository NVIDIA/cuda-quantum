# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# Licensed under the Apache License, Version 2.0.
"""Plug a Python synthesizer into logical Clifford+T compilation.

ToySU2Synthesizer is no better than the default GridSynth. It exercises the
external interface using CUDA-Q itself, without optional synthesis packages.
"""

import cudaq
import cudaq.logical as cql
from cudaq.logical.compiler import CliffordTWord


class ToySU2Synthesizer:
    """Demonstrate whole-U3 dispatch by composing three native Rz words."""

    name = "toy-su2"
    version = "0.1.0"

    def rz(self, angle, precision):
        expansions = {
            "H": ("h",),
            "S": ("s",),
            "T": ("t",),
            "X": ("h", "s", "s", "h"),
            "W": (),
            "I": (),
        }
        # Leave room for the independent verifier's numerical margin even
        # when mpmath is absent and it must multiply the word in doubles.
        sequence = str(cudaq.synth.gridsynth(angle, precision / 2, seed=0))
        gates = tuple(gate for native in reversed(sequence)
                      for gate in expansions[native])
        error = cudaq.synth.rz_error(angle, sequence)
        return CliffordTWord(gates, precision, error, "toy-gridsynth",
                             self.name, self.version)

    def u3(self, theta, phi, lam, precision):
        first = self.rz(lam, precision / 3)
        middle = self.rz(theta, precision / 3)
        last = self.rz(phi, precision / 3)
        gates = (first.gates + ("s", "s", "s", "h") + middle.gates +
                 ("h", "s") + last.gates)
        # Unitary composition adds the three error bounds; the basis-change
        # Cliffords are exact. rz_error supplies a conservative norm bound.
        error = first.achieved_error + middle.achieved_error + last.achieved_error
        return CliffordTWord(gates, precision, error, "toy-euler", self.name,
                             self.version)


@cql.program
def contrived() -> tuple[bool, bool]:
    q = cql.allocate(2, state=cql.types.zero)
    q[0] = cql.u3(q[0], 0.3, 1.1, -0.4)
    q[1] = cql.rz(q[1], 0.1234)
    q[0], q[1] = cql.cx(q[0], q[1])
    q[1] = cql.u3(q[1], cql.pi / 2, 0.0, cql.pi)
    return cql.measure_z(q[0]), cql.measure_z(q[1])


@cql.program
def exact_h() -> bool:
    q = cql.u3(cql.prepare_zero(), cql.pi / 2, 0.0, cql.pi)
    return cql.measure_z(q)


def main():
    built_in = cql.compiler.synthesize(contrived,
                                       gate_set=cql.compiler.clifford_t,
                                       precision=1e-10)
    external = cql.compiler.synthesize(contrived,
                                       gate_set=cql.compiler.clifford_t,
                                       precision=1e-10,
                                       synthesizer=ToySU2Synthesizer())
    print("Built in:", built_in.synthesis)
    print("Python plugin:", external.synthesis)
    assert built_in.synthesis.synthesizer != external.synthesis.synthesizer
    for synthesizer in (None, ToySU2Synthesizer()):
        exact = cql.compiler.synthesize(exact_h,
                                        gate_set=cql.compiler.clifford_t,
                                        synthesizer=synthesizer)
        assert exact.synthesis.h_count == 1
        assert exact.synthesis.t_count == exact.synthesis.s_count == 0
    # synthesize() runs qlx-verify-clifford-t before returning either Build.


if __name__ == "__main__":
    main()
