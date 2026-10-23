# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# Licensed under the Apache License, Version 2.0.

from functools import lru_cache

import pytest


@lru_cache(maxsize=None)
def _native_rz(angle, precision):
    import cudaq
    from cudaq.logical.compiler import CliffordTWord

    expansions = {
        "H": ("h",),
        "S": ("s",),
        "T": ("t",),
        "X": ("h", "s", "s", "h"),
        "W": (),
        "I": ()
    }
    text = str(cudaq.synth.gridsynth(angle, precision, seed=0))
    gates = tuple(
        gate for native in reversed(text) for gate in expansions[native])
    error = cudaq.synth.rz_error(angle, text)
    return CliffordTWord(gates, precision, error, "test-gridsynth",
                         "test-synth", "1")


@pytest.fixture
def synthesizer_factory():

    class RecordingSynthesizer:
        name = "test-synth"
        version = "1"

        def __init__(self):
            self.calls = []

        def rz(self, angle, precision):
            self.calls.append(("rz", angle, precision))
            return _native_rz(angle, precision)

    class DirectSynthesizer(RecordingSynthesizer):

        def u3(self, theta, phi, lam, precision):
            from cudaq.logical.compiler import CliffordTWord
            self.calls.append(("u3", theta, phi, lam, precision))
            gates = (_native_rz(lam, precision / 3).gates +
                     ("s", "s", "s", "h") +
                     _native_rz(theta, precision / 3).gates + ("h", "s") +
                     _native_rz(phi, precision / 3).gates)
            return CliffordTWord(gates, precision, 0.0, "test-euler", self.name,
                                 self.version)

    def make(*, direct=False):
        return DirectSynthesizer() if direct else RecordingSynthesizer()

    return make
