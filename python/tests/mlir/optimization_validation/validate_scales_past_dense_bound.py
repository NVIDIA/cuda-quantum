# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Why the Clifford-tableau oracle exists, on one input. A 20-qubit Clifford
# kernel is past the dense oracle's 14-qubit bound, so the dense oracle fails
# closed with `too-many-qubits` and a nonzero exit. The tableau oracle has no
# qubit bound and certifies the same kernel exactly, at the exact-clifford-sim
# tier. Reach is bought with domain, not with a weaker verdict.

# RUN: PYTHONPATH=%cudaq_python_root not python3 -m cudaq._compiler \
# RUN:   --input %S/Inputs/clifford_20q.qke \
# RUN:   --prepare 'builtin.module(func.func(memtoreg))' \
# RUN:   --candidate 'builtin.module(func.func(phase-folding))' \
# RUN:   --oracle up-to-global-phase | FileCheck --check-prefix=DENSE %s

# DENSE-DAG: "status": "unsupported-domain"
# DENSE-DAG: too-many-qubits
# DENSE-DAG: "equal_up_to_global_phase": false

# RUN: PYTHONPATH=%cudaq_python_root python3 -m cudaq._compiler \
# RUN:   --input %S/Inputs/clifford_20q.qke \
# RUN:   --prepare 'builtin.module(func.func(memtoreg))' \
# RUN:   --candidate 'builtin.module(func.func(phase-folding))' \
# RUN:   --oracle clifford-tableau \
# RUN:   --metric operation-count:nonincreasing \
# RUN:   | FileCheck --check-prefix=CLIFFORD %s

# CLIFFORD-DAG: "assurance_tier": "exact-clifford-sim"
# CLIFFORD-DAG: "name": "equivalence"
# CLIFFORD-DAG: "satisfied": true
# CLIFFORD: "status": "passed"
