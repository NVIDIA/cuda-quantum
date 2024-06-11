# ============================================================================ #
# Copyright (c) 2022 - 2024 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #


def install_qutip_request(*args, **kwargs):
    print(
        "The module QuTiP was not detected. Bloch Sphere visualization will not work.\nPlease install QuTip to use this feature. You can run \"pip install qutip\" or in a conda environment, run \"conda -c conda-forge install qutip\" to install qutip. "
    )
