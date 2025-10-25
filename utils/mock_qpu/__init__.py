# ============================================================================ #
# Copyright (c) 2022 - 2025 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import warnings
import traceback

try:
    from .anyon import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `anyon` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .braket import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `braket` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .infleqtion import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `infleqtion` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .ionq import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `ionq` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .iqm import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `iqm` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .oqc import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `oqc` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .qci import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `qci` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .quantinuum import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `quantinuum` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .quantum_machines import *
except ImportError as e:
    print(f"Import error details: {e}")
    print(f"Full traceback:\n{traceback.format_exc()}")
    warnings.warn(f"Failed to import `quantum_machines` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)
