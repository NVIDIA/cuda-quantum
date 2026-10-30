# Verify an Authored Gadget

[`gadgets/00_verify_gadget.py`](../../../../examples/gadgets/00_verify_gadget.py)
authors a Repetition-code syndrome round, declares its detector contract, and
verifies its ideal logical action. It then changes the declared action to
demonstrate a failed verification receipt.

This is the first gadget example. The general standalone introduction,
[`standalone/00_logical_program.py`](../../../../examples/standalone/00_logical_program.py),
instead authors and estimates a logical Bell program. Here the question is
whether one encoded realization does what its declaration promises.

Run from the repository root with CUDA-Q Logical installed:

```bash
python3 preview/logical/examples/gadgets/00_verify_gadget.py
```

The example needs no simulator dependency. Run the complete script or step
through its `# %%` cells.

## Author the realization and its claim

A gadget combines an ideal logical action with a physical realization on an
encoded boundary. This realization extracts the Repetition code's two Z
checks and returns the block. Its `implements=` declaration claims that the
logical qubit is unchanged:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/00_verify_gadget.py
   :language: python
   :start-at: import cudaq.logical as cql
   :end-before: # %%
```

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/00_verify_gadget.py
   :language: python
   :start-at: @cql.gadget(implements=cql.std.idle)
   :end-before: # %%
   :caption: An idle logical claim with a syndrome-extraction realization.
```

The patch type identifies the code. `record="round"` gives the extracted
check results stable names that the contract can reference.

## Declare the detector contract

The contract relates the realization's measurement records to its encoded
input and output boundaries:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/00_verify_gadget.py
   :language: python
   :start-at: def declare_contract(gadget):
   :end-before: # %%
   :caption: Declare the virtual detectors and outgoing virtual syndrome.
```

Each detector computes the XOR of an incoming virtual syndrome and the newly
measured check. In an ideal round the two agree, so the detector parity is zero.
The output binding carries the measured checks forward as the next gadget's
expected stabilizer values. These declarations describe the realization's
measurement relations; they do not add physical operations.

## Inspect a successful receipt

Verification checks both the detector contract and the logical action:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/00_verify_gadget.py
   :language: python
   :start-at: receipt = cql.gadgets.verify_gadget(contract)
   :end-before: # %%
```

`detector_contract.ok` establishes that the declared detector and boundary
relations are deterministic and complete. `logical_action.status` reports
whether the realization implements the declared idle action.
`receipt.verified` requires both checks to pass. With the default arguments,
`verify_gadget` raises if either check fails.

## Inspect an incorrect logical claim

The final cell gives the same physical extraction a logical-X claim:

```{eval-rst}
.. literalinclude:: ../../../../examples/gadgets/00_verify_gadget.py
   :language: python
   :start-at: @cql.gadget(implements=cql.std.x)
   :caption: Return a failed receipt for inspection with raise_on_failure=False.
```

The detector contract still passes: those checks describe syndrome relations,
which cannot distinguish idle from a claimed logical X. The logical-action
check fails, and the combined receipt is unverified. Passing
`raise_on_failure=False` lets the example inspect this expected failure.

These checks establish ideal behavior for the supported stabilizer actions;
they do not prove fault tolerance or noisy circuit distance. See
[Detector contracts](../../reference/detector-contracts.md#verify-a-contract-and-the-logical-action)
for receipt details and supported claims.

Continue with [Physical surface-code memory](gadget-surface-memory.md) to
verify preparation, extraction, and readout, then export their noisy circuit
and composed detector error model.

[Back to examples](index.md)
