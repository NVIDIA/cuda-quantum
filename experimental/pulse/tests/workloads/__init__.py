# Copyright (c) 2026 NVIDIA Corporation & Affiliates.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from . import (
    w1_bell,
    w2_cnot_cr,
    w3_qaoa4,
    w4_syndrome,
    w5_dd_cpmg8,
    w6_vqe_hea,
)

ALL_WORKLOADS = {
    "bell": w1_bell,
    "cnot_cr": w2_cnot_cr,
    "qaoa4": w3_qaoa4,
    "syndrome": w4_syndrome,
    "dd_cpmg8": w5_dd_cpmg8,
    "vqe_hea": w6_vqe_hea,
}
