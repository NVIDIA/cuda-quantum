# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Algorithm libraries authored against the CUDA-Q Logical logical surface."""

from . import reversible
from .crypto import (
    GidneyEkeraProgram,
    GidneyEkeraResourceModel,
    GidneyEkeraEstimate,
    c_pad_for,
    exponent_length,
    gidney_ekera_factor,
    gidney_ekera_2019,
    estimate_gidney_ekera,
    lookup_additions_per_modmul,
    toffolis_per_lookup_addition,
)
from .fermi_hubbard import (
    FermiHubbardRotationClass,
    FermiHubbardStats,
    FermiHubbardWorkload,
    fermi_hubbard_jordan_wigner,
    fermi_hubbard_stats,
)

__all__ = [
    "reversible",
    "GidneyEkeraProgram",
    "GidneyEkeraResourceModel",
    "GidneyEkeraEstimate",
    "c_pad_for",
    "exponent_length",
    "gidney_ekera_factor",
    "gidney_ekera_2019",
    "estimate_gidney_ekera",
    "lookup_additions_per_modmul",
    "toffolis_per_lookup_addition",
    "FermiHubbardRotationClass",
    "FermiHubbardStats",
    "FermiHubbardWorkload",
    "fermi_hubbard_jordan_wigner",
    "fermi_hubbard_stats",
]
