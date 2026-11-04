# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Verify an encoded realization against its action and detector contract."""

# %%
# A gadget has three ingredients: an ideal logical action, an encoded circuit,
# and a detector contract describing the circuit's measurement parities.
import cudaq.logical as cql

RepetitionPatch = cql.patch[cql.codes.Repetition]


# %%
# The ideal action preserves a logical qubit. The physical realization extracts
# the repetition code's two Z checks and returns the encoded block.
@cql.gadget(implements=cql.std.idle)
def syndrome_round(block: RepetitionPatch) -> RepetitionPatch:
    block, _ = cql.extract_syndrome(block, record="round")
    return block


# %%
# Input and output bundles describe the encoded boundary. A syndrome component
# is a virtual check value there; "round.s0" and "round.s1" are actual records.
# Each detector XORs an incoming syndrome with its newly measured check.
# Binding the output syndrome makes those new records available to the next
# gadget. These declarations are separate from the physical circuit above.
def declare_contract(gadget):
    incoming = gadget.inputs.only()
    outgoing = gadget.outputs.only()
    checks = tuple(
        gadget.record(f"round.s{i}") for i in range(incoming.syndrome.width))
    detectors = tuple(
        cql.gadgets.Detector(incoming.syndrome[i] ^ checks[i])
        for i in range(incoming.syndrome.width))
    return cql.gadgets.GadgetContract(gadget,
                                      detectors=detectors,
                                      boundary={outgoing.syndrome: checks})


contract = declare_contract(syndrome_round)

# %%
# Verification checks both that the declared detector relations are complete
# and deterministic, and that the encoded circuit implements the ideal action.
receipt = cql.gadgets.verify_gadget(contract)
assert receipt.detector_contract.ok
assert receipt.logical_action.status == "verified"
assert receipt.verified
print("Syndrome round:")
print(f"  detector contract verified: {receipt.detector_contract.ok}")
print(f"  logical action: {receipt.logical_action.status}")
print(f"  gadget verified: {receipt.verified}")


# %%
# Deliberately claim logical X for the same physical realization. The detector
# relations still hold, but they cannot establish the claimed logical action.
@cql.gadget(implements=cql.std.x)
def wrong_action(block: RepetitionPatch) -> RepetitionPatch:
    block, _ = cql.extract_syndrome(block, record="round")
    return block


wrong_contract = declare_contract(wrong_action)
wrong_receipt = cql.gadgets.verify_gadget(wrong_contract,
                                          raise_on_failure=False)
assert wrong_receipt.detector_contract.ok
assert wrong_receipt.logical_action.status == "failed"
assert not wrong_receipt.verified
print("Deliberately wrong logical X declaration:")
print(f"  detector contract verified: {wrong_receipt.detector_contract.ok}")
print(f"  logical action: {wrong_receipt.logical_action.status}")
print(f"  gadget verified: {wrong_receipt.verified}")
