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
from typing import Literal

from cudaq.logical.errors import GadgetVerificationError

from .contracts import GadgetContract

_SCHEMA = "cudaq.logical.gadget-verification/v2"


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
    declared behavior is outside the supported stabilizer forms, so it was
    not checked; ``reason`` says why). ``verified_checks`` counts the signed
    logical relations proved by the realization. ``logical_rank`` retains
    the historical spelling of that count; it is not a general matrix rank.
    """

    status: Literal["verified", "failed", "unsupported"]
    action: str | None = None
    logical_rank: int = 0
    reason: str | None = None
    failures: tuple[VerificationFailure, ...] = ()

    @property
    def verified_checks(self) -> int:
        """Number of signed logical relations proved by the realization."""

        return self.logical_rank


@dataclass(frozen=True, slots=True)
class GadgetVerificationReceipt:
    """Results of checking a gadget's contract and declared logical behavior.

    Neither check proves fault tolerance: they establish ideal determinism and
    completeness of the contract, and the ideal logical action.
    """

    gadget: str
    contract: str
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
    from cudaq.logical.std import LogicalActionRef, LogicalInstrumentRef

    if isinstance(implements, (LogicalActionRef, LogicalInstrumentRef)):
        return f"ql.logical.{name}"
    return str(name)


_ROW_FAILURES = {
    "not_stabilized": ("action_mismatch", "action mismatch",
                       "does not hold in the realization's logical state"),
    "sign_mismatch": ("sign_mismatch", "sign mismatch",
                      "holds with the opposite deterministic sign"),
    "random_sign": ("random_sign", "nondeterministic sign",
                    "has a sign with unmatched ideal randomness"),
}


def _action_verification(native, label) -> LogicalActionVerification:
    status = native["status"]
    if status == "unsupported":
        return LogicalActionVerification("unsupported", label, 0,
                                         native.get("reason"))
    if status not in ("verified", "failed"):
        raise RuntimeError(
            f"invalid native logical verification status: {status!r}")
    rows = native.get("rows", ())
    labels = native.get("labels", ())
    if len(rows) != len(labels):
        raise RuntimeError(
            "native logical verification labels disagree with checks")
    failures = []
    for label_text, row_status in zip(labels, rows):
        if row_status == "verified":
            continue
        if row_status not in _ROW_FAILURES:
            raise RuntimeError(
                f"invalid native logical check status: {row_status!r}")
        kind, name, condition = _ROW_FAILURES[row_status]
        failures.append(
            VerificationFailure(
                kind, f"{name} on {label_text}: the expected logical relation "
                f"{condition}"))
    if native.get("reason") and status == "failed":
        failures.append(VerificationFailure("action_mismatch",
                                            native["reason"]))
    if (status == "verified" and
        (failures or not rows)) or (status == "failed" and not failures):
        raise RuntimeError("inconsistent native logical verification result")
    reproduced = sum(1 for value in rows if value == "verified")
    return LogicalActionVerification(status, label, reproduced,
                                     native.get("reason"), tuple(failures))


def _contract_verification(native) -> ContractVerification:
    contract = native["detector_contract"]
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


def verify_gadget(contract: GadgetContract,
                  *,
                  raise_on_failure: bool = True) -> GadgetVerificationReceipt:
    """Prove a gadget's detector contract and supported logical behavior.

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
    * **Logical behavior.** The realization must reproduce the signed Choi
      relations, preparation stabilizers, or measurement correlations of its
      declared objective. Logical operands follow the selected compiled
      specification's bindings. Measurement results use the complete affine
      outcome map, including its constant.

    Failures name the condition. With ``raise_on_failure`` (the default) they
    raise :class:`GadgetVerificationError`, whose ``receipt`` holds the full
    result. An unsupported claim also raises, with its status and reason in
    the receipt. With ``raise_on_failure=False``, failed and unsupported
    results are returned for inspection. Invalid declarations, compilation
    errors and unsupported realization projections still raise diagnostics;
    inspection receipts require a realization that can be analyzed.

    Supported logical forms are signed Clifford unitaries, single-logical
    signed X/Z eigenstate preparation, and destructive or preserving X/Z measurements.
    Other instruments, dynamic feed-forward, subsystem codes and non-Clifford
    realizations are outside this verifier's supported scope. A normal return
    in the default mode always has ``receipt.verified == True``.

    The checks establish ideal determinism and the ideal logical action. They
    do not prove fault tolerance.
    """

    if not isinstance(contract, GadgetContract):
        raise TypeError("verify_gadget expects a GadgetContract, the detector "
                        "contract declared for a gadget")
    if type(raise_on_failure) is not bool:
        raise TypeError("raise_on_failure must be a bool")
    from cudaq.mlir._mlir_libs import _qlxRuntime

    from ..compiler import compile

    gadget = contract.gadget
    build = compile(contract)
    # Resolve intent, bindings and affine results from this same authenticated
    # snapshot. Re-analyzing the Python action would bypass compiled bindings.
    action_json = json.dumps({"source": "gadget_spec"})
    native = json.loads(
        _qlxRuntime.verify_gadget_json(build._fresh_module(), build.root.symbol,
                                       action_json))
    if native.get("schema") != _SCHEMA:
        raise RuntimeError("unsupported native gadget verification schema")

    label = _action_label(gadget.implements)
    logical = _action_verification(native["logical_action"], label)
    receipt = GadgetVerificationReceipt(gadget.name, native["contract"],
                                        _contract_verification(native), logical)
    failures = (*receipt.detector_contract.failures,
                *receipt.logical_action.failures)
    if not receipt.verified and raise_on_failure:
        messages = [failure.message for failure in failures]
        if logical.status == "unsupported":
            messages.append("logical verification is unsupported: " + (
                logical.reason or "no supported declared objective"))
        raise GadgetVerificationError(
            f"gadget {gadget.name!r} did not pass verification:\n" +
            "\n".join(f"  - {message}" for message in messages), receipt)
    return receipt


__all__ = [
    "ContractVerification",
    "GadgetVerificationReceipt",
    "LogicalActionVerification",
    "VerificationFailure",
    "verify_gadget",
]
