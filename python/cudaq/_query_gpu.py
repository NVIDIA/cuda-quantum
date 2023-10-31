# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import os
from pathlib import Path
import subprocess

# Install directory is wherever this script is and up one directory
install_dir=Path(__file__).parent.parent.resolve()

def is_gpu_available():
    try:
        gpu_list = subprocess.check_output(["nvidia-smi", "-L"], encoding="utf-8")
        ngpus = len(gpu_list.splitlines())
        lib_path = Path(os.path.join(install_dir, "lib/libnvqir-custatevec-fp32.so"))
        if ngpus > 0 and lib_path.is_file():
            return True
    except Exception:
        pass
    return False
