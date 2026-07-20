# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# A semantics-preserving candidate (phase-folding) over a bounded-unitary input
# passes at the exact assurance tier, up to a global phase, with a nonincreasing
# operation count. Exit status 0.

# RUN: PYTHONPATH=%cudaq_python_root python3 -m cudaq._compiler.optimization_cli \
# RUN:   --input %S/Inputs/good.qke \
# RUN:   --prepare 'builtin.module(func.func(memtoreg))' \
# RUN:   --candidate 'builtin.module(func.func(phase-folding))' \
# RUN:   --oracle up-to-global-phase \
# RUN:   --metric operation-count:nonincreasing | FileCheck %s

# CHECK-DAG: "assurance_tier": "exact"
# CHECK-DAG: "equal_up_to_global_phase": true
# CHECK-DAG: "name": "equivalence"
# CHECK-DAG: "name": "determinism"
# CHECK-DAG: "name": "fixed-point"
# CHECK-DAG: "satisfied": true
# CHECK: "status": "passed"
