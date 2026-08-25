# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# RUN: PYTHONPATH=../../ CUDAQ_DUMP_JIT_IR=1 pytest -rP -k 'test_qir_profile' %s 2>&1 | FileCheck %s
# RUN: PYTHONPATH=../../ CUDAQ_DUMP_JIT_IR=1 pytest -rP -k 'test_ng_qir_profile' %s 2>&1 | FileCheck %s --check-prefix=CHECK-NG

import cudaq


def test_qir_profile():

    @cudaq.kernel
    def my_kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    # This device requires qir:0.1
    cudaq.set_target('quantinuum', emulate=True, machine='H2-2E')
    result = cudaq.sample(my_kernel)


def test_ng_qir_profile():

    @cudaq.kernel
    def my_kernel():
        q = cudaq.qubit()
        x(q)
        mz(q)

    # This device requires qir:1.0
    cudaq.set_target('quantinuum', emulate=True, machine='Helios-1E')
    result = cudaq.sample(my_kernel)


# CHECK: requiredQubits
# CHECK: requiredResults
# CHECK-NG: required_num_qubits
# CHECK-NG: required_num_results
