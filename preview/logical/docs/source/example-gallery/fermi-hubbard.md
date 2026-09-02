# Trotterized Fermi–Hubbard evolution

**Outcome.** A four-site Fermi–Hubbard Hamiltonian — Jordan–Wigner hopping
terms (`XZX`/`YZY`) plus on-site interaction — is evolved for eight Trotter
steps with `exp_pauli` inside one `cudaq.kernel`. The same kernel is then
estimated twice without modification: once in the logical action model
(`estimator` target), and once after Clifford+T synthesis (`clifford_t`
target), so the cost of rotation approximation shows up directly in the
action counts.

**Evidence boundary.** Both reports are logical action counts and depth
bounds — two estimation views of one workload. Neither attaches a code,
gadget, or machine, so no encoded or physical cost is claimed.

## Canonical source

```{literalinclude} ../../../examples/07_fermi_hubbard.py
:language: python
:linenos:
```

[Back to the gallery](../examples)
