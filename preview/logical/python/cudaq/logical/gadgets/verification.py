# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Verification of a gadget's detector contract and declared logical action."""

from __future__ import annotations

from dataclasses import dataclass
import json

from cudaq.logical.errors import GadgetVerificationError

from .definition import GadgetDefinition
from .profiles import GadgetProfile

_SCHEMA = "cudaq.logical.gadget-verification/v1"


@dataclass(frozen=True, slots=True)
class VerificationFailure:
    """One violated condition, named by ``kind``."""

    kind: str
    message: str


@dataclass(frozen=True, slots=True)
class ContractVerification:
    """Result of the detector-contract check.

    ``outcome_code_rank`` is the dimension of the deterministic parities of
    the probed realization. A contract is correct when every declared detector
    and output virtual-syndrome relation is one of them and, together with the
    deterministic combinations of its observables, they span all of them:
    ``outcome_code_rank == detector_rank +
    deterministic_logical_outcome_rank``.
    """

    ok: bool
    outcome_code_rank: int
    detector_rank: int
    deterministic_logical_outcome_rank: int
    outcome_count: int
    detector_count: int
    output_relation_count: int
    observable_count: int
    success_count: int
    failures: tuple[VerificationFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class LogicalActionVerification:
    """Result of comparing the realization with the declared logical action.

    ``status`` is ``"verified"``, ``"failed"``, or ``"unsupported"`` (the
    action is not a Clifford action over the gadget's logical qubits, so it
    was not checked; ``reason`` says why). ``logical_rank`` is the number of
    Choi stabilizers reproduced, with matching signs, by the realization.
    """

    status: str
    action: str | None = None
    logical_rank: int = 0
    reason: str | None = None
    failures: tuple[VerificationFailure, ...] = ()


@dataclass(frozen=True, slots=True)
class GadgetVerificationReceipt:
    """Evidence that a gadget and its contract passed :func:`verify_gadget`.

    Neither check proves fault tolerance: they establish ideal determinism and
    completeness of the contract, and the ideal logical action.
    """

    gadget: str
    profile: str
    detector_contract: ContractVerification
    logical_action: LogicalActionVerification

    @property
    def verified(self) -> bool:
        """Both checks ran and passed."""

        return (self.detector_contract.ok and
                self.logical_action.status == "verified")


def _action_label(implements) -> str | None:
    name = getattr(implements, "name", None)
    if name is None:
        return None
    from cudaq.logical.std import LogicalActionRef

    return f"ql.logical.{name}" if isinstance(implements,
                                              LogicalActionRef) else str(name)


def _declared_action(gadget: GadgetDefinition):
    """The declared Clifford action, or ``(None, reason)`` if there is none."""

    from cudaq.logical.algebra.clifford import NonCliffordAction

    from .analysis import clifford_action

    try:
        return clifford_action(gadget), None
    except NonCliffordAction as exc:
        return None, str(exc)
    except (TypeError, ValueError) as exc:
        return None, ("the declared logical action is not a Clifford action "
                      f"that can be compared with the realization: {exc}")


def _pauli_label(action, row: int) -> str:
    arity = action.arity
    generator = ("X" if row < arity else "Z")
    qubit = row % arity
    reference = f"{generator}_ref" if arity == 1 else f"{generator}_ref[{qubit}]"
    x_bits, z_bits = action.matrix[row][:arity], action.matrix[row][arity:]
    factors = []
    for index, (x, z) in enumerate(zip(x_bits, z_bits)):
        if x or z:
            letter = "Y" if x and z else "X" if x else "Z"
            factors.append(f"{letter}̄" if arity ==
                           1 else f"{letter}̄[{index}]")
    sign = "-" if action.phases[row] else ""
    return f"{reference} {sign}{' '.join(factors)}"


_ROW_FAILURES = {
    "not_stabilized": ("action_mismatch", "action mismatch",
                       "is not stabilized by the realization's Choi state"),
    "sign_mismatch":
        ("sign_mismatch", "sign mismatch",
         "stabilizes the realization's Choi state with the opposite sign"),
    "random_sign":
        ("random_sign", "nondeterministic sign",
         "stabilizes the realization's Choi state only up to a random "
         "outcome"),
}


def _action_verification(native, action, label) -> LogicalActionVerification:
    status = native["status"]
    if status == "unsupported":
        return LogicalActionVerification("unsupported", label, 0,
                                         native.get("reason"))
    failures = []
    for row, row_status in enumerate(native["rows"]):
        if row_status == "verified":
            continue
        kind, name, condition = _ROW_FAILURES[row_status]
        failures.append(
            VerificationFailure(
                kind, f"{name} on {_pauli_label(action, row)}: the expected "
                f"Choi stabilizer {condition}"))
    reproduced = sum(1 for value in native["rows"] if value == "verified")
    return LogicalActionVerification(status, label, reproduced, None,
                                     tuple(failures))


def _contract_verification(native) -> ContractVerification:
    contract = native["contract"]
    logical = contract["deterministic_logical_outcome_rank"]
    failures = []
    for failure in contract["failures"]:
        kind = failure["kind"]
        if kind == "rank_incomplete":
            message = (
                "rank condition fails: the realization has "
                f"{contract['outcome_code_rank']} independent deterministic "
                "parities, but the declared detectors and output virtual "
                f"syndromes span {contract['detector_rank']} + {logical}; "
                "declare the missing detector, output syndrome, or observable")
        elif kind == "detector_not_deterministic":
            message = (f"detector {failure['index']} is not a deterministic "
                       "parity of the realization")
        elif kind == "success_not_deterministic":
            message = (f"success row {failure['index']} is not deterministic "
                       "on the ideal realization")
        elif kind == "output_syndrome_mismatch":
            message = (f"output virtual syndrome {failure['index']} does not "
                       "follow from the realization's records and input "
                       "syndromes")
        else:
            message = failure["message"]
        failures.append(VerificationFailure(kind, message))
    return ContractVerification(
        ok=contract["ok"],
        outcome_code_rank=contract["outcome_code_rank"],
        detector_rank=contract["detector_rank"],
        deterministic_logical_outcome_rank=logical,
        outcome_count=contract["outcome_count"],
        detector_count=contract["detector_count"],
        output_relation_count=contract["output_relation_count"],
        observable_count=contract["observable_count"],
        success_count=contract["success_count"],
        failures=tuple(failures),
    )


def verify_gadget(contract: GadgetProfile,
                  *,
                  raise_on_failure: bool = True) -> GadgetVerificationReceipt:
    """Check a gadget and its detector contract from one Choi probe.

    Every logical input of the gadget's realization is Bell-paired with a
    reference qubit, the encoded input is placed in an arbitrary stabilizer
    coset, and the realization's Clifford projection is run through an
    outcome-complete symbolic stabilizer simulation. Two conditions are read
    off that one probe:

    * **Detector contract.** Every declared detector and output virtual
      syndrome relation must be a deterministic parity of the probed
      realization, and together with the deterministic combinations of its
      observables they must generate all of its deterministic parities (the
      rank condition).
    * **Logical action.** The post-realization Choi state must be stabilized,
      stabilizer by stabilizer and sign by sign, by the Choi stabilizers of
      the ``implements=`` action on the encoded logical qubits (input logical
      qubit ``i`` maps to output logical qubit ``i``).

    Failures name the condition. With ``raise_on_failure`` (the default) they
    raise :class:`GadgetVerificationError`, whose ``receipt`` holds the full
    result; otherwise the receipt is returned for inspection. A declared
    action that is not Clifford, or does not act on the gadget's logical
    qubits, is reported as ``"unsupported"`` and does not raise; check
    :attr:`GadgetVerificationReceipt.verified`.

    The checks establish ideal determinism and the ideal logical action. They
    do not prove fault tolerance.
    """

    if not isinstance(contract, GadgetProfile):
        raise TypeError("verify_gadget expects a GadgetProfile, the detector "
                        "contract declared for a gadget")
    from cudaq.mlir._mlir_libs import _qlxRuntime

    from ..compiler import compile

    gadget = contract.gadget
    build = compile(contract)
    action, reason = _declared_action(gadget)
    action_json = "" if action is None else json.dumps({
        "arity": action.arity,
        "matrix": [list(row) for row in action.matrix],
        "phases": list(action.phases),
    })
    native = json.loads(
        _qlxRuntime.verify_gadget_json(build._fresh_module(), build.root.symbol,
                                       action_json))
    if native.get("schema") != _SCHEMA:
        raise RuntimeError("unsupported native gadget verification schema")

    label = _action_label(gadget.implements)
    if action is None:
        logical = LogicalActionVerification("unsupported", label, 0, reason)
    else:
        logical = _action_verification(native["logical_action"], action, label)
    receipt = GadgetVerificationReceipt(gadget.name, native["profile"],
                                        _contract_verification(native), logical)
    failures = (*receipt.detector_contract.failures,
                *receipt.logical_action.failures)
    if failures and raise_on_failure:
        raise GadgetVerificationError(
            f"gadget {gadget.name!r} failed verification:\n" +
            "\n".join(f"  - {failure.message}" for failure in failures),
            receipt)
    return receipt


__all__ = [
    "ContractVerification",
    "GadgetVerificationReceipt",
    "LogicalActionVerification",
    "VerificationFailure",
    "verify_gadget",
]
