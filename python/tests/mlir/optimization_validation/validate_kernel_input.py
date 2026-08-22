# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# A Python @cudaq.kernel file can be passed straight to --input: every kernel in
# the file is lowered through the frontend and validated. The `entangle` kernel's
# `for` loop is unrolled by the prepare pipeline into the bounded-unitary domain.
# Both kernels pass a semantics-preserving candidate. Exit status 0.

# RUN: PYTHONPATH=%cudaq_python_root python3 -m cudaq._compiler \
# RUN:   --input %S/Inputs/kernels.py \
# RUN:   --prepare 'builtin.module(func.func(cc-loop-unroll,canonicalize,memtoreg))' \
# RUN:   --candidate 'builtin.module(func.func(canonicalize))' \
# RUN:   --oracle up-to-global-phase \
# RUN:   --metric operation-count:nonincreasing | FileCheck %s

# Two cases, named for the kernels (not the .qke file), both passing.
# CHECK-DAG: "input": {{.*}}bell.qke
# CHECK-DAG: "input": {{.*}}entangle.qke
# CHECK-DAG: "assurance_tier": "exact-unitary"
# CHECK: "status": "passed"
