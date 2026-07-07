# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

from ._cudaq_synth import gridsynth as _gridsynth

__all__ = ["CliffordTSequence", "gridsynth"]


class CliffordTSequence:
    """An ordered Clifford+T gate sequence approximating an R_z rotation.

    The sequence uses the gate alphabet {H, S, T, X, W}, where H is
    Hadamard, S is the phase gate (S = T^2), T is the pi/8 gate, X is
    Pauli-X, and W is the scalar global-phase gate W = omega * I with
    omega = e^{i*pi/4}.

    Gates are stored in matrix-multiplication order. The sequence
    "G0 G1 ... Gn-1" denotes the unitary U = G0 * G1 * ... * G(n-1). When
    read as a circuit (order of application to a state), gates apply
    right-to-left: G(n-1) first, G0 last.

    ``str(seq)`` yields the gate string, with the empty (identity)
    sequence rendered as the single character ``"I"``. Iteration,
    indexing, and ``len`` operate on the individual gate characters (the
    identity sequence has length 0).
    """

    __slots__ = ("_gates",)

    _ALPHABET = frozenset("HSTXW")

    def __init__(self, gates: str):
        """Create a sequence from a gate string over {H, S, T, X, W}.

        The identity sentinel ``"I"`` is accepted anywhere in the string
        and contributes no gate.
        """
        stripped = gates.replace("I", "")
        invalid = set(stripped) - self._ALPHABET
        if invalid:
            raise ValueError(
                f"invalid gate character(s) {sorted(invalid)}; "
                "expected only H, S, T, X, W, or I")
        self._gates = stripped

    def __str__(self):
        return self._gates or "I"

    def __repr__(self):
        return f"CliffordTSequence('{self}')"

    def __len__(self):
        return len(self._gates)

    def __iter__(self):
        return iter(self._gates)

    def __getitem__(self, index):
        return self._gates[index]

    def __eq__(self, other):
        if isinstance(other, CliffordTSequence):
            return self._gates == other._gates
        if isinstance(other, str):
            return str(self) == other
        return NotImplemented

    def __hash__(self):
        return hash(self._gates)

    @property
    def t_count(self) -> int:
        """Number of T gates in the sequence."""
        return self._gates.count("T")

    def to_kernel(self):
        """Build a CUDA-Q kernel applying this sequence to a qubit.

        Returns a kernel taking a single qubit argument, suitable for
        standalone sampling or composition into a larger kernel via
        ``kernel.apply_call``.
        """
        import cudaq

        kernel, qubit = cudaq.make_kernel(cudaq.qubit)
        for gate in reversed(self._gates):
            if gate == "H":
                kernel.h(qubit)
            elif gate == "S":
                kernel.s(qubit)
            elif gate == "T":
                kernel.t(qubit)
            elif gate == "X":
                kernel.x(qubit)
        return kernel


def gridsynth(theta,
              epsilon,
              diophantine_timeout_ms: int = 200,
              factoring_timeout_ms: int = 50) -> CliffordTSequence:
    """Synthesize a Clifford+T sequence approximating R_z(theta).

    Implements the grid-synthesis algorithm of Ross & Selinger
    (arXiv:1403.2975, Algorithm 7.6). The returned sequence is in
    Matsumoto-Amano normal form with minimum T-count up to search
    timeouts. The synthesized unitary U satisfies
    ``||R_z(theta) - U|| <= epsilon`` in the operator norm.

    Args:
        theta: Target rotation angle (float, or decimal str for
            arbitrary precision).
        epsilon: Approximation precision in operator norm, must be > 0
            (float, or str).
        diophantine_timeout_ms: Per-candidate timeout for the Diophantine
            solver. Higher values improve optimality at the cost of
            worst-case latency. Default 200.
        factoring_timeout_ms: Per-candidate timeout for integer factoring
            inside the Diophantine solver. Default 50.

    Returns:
        A :class:`CliffordTSequence`. ``str()`` of the result is the gate
        string over {H, S, T, X, W} in matrix-multiplication order (see
        the class docstring); ``to_kernel()`` builds a CUDA-Q kernel for
        the sequence.

    Raises:
        ValueError: if theta or epsilon is a string that does not parse
            as a number, if theta is not finite, if epsilon <= 0, or if
            synthesis fails (degenerate epsilon region or search space
            exhausted).
    """
    return CliffordTSequence(
        _gridsynth(theta, epsilon, diophantine_timeout_ms,
                   factoring_timeout_ms))
