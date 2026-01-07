# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates and Contributors.  #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import warnings

try:
    from .anyon import *
except ImportError as e:
    warnings.warn(f"Failed to import `anyon` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .braket import *
except ImportError as e:
    warnings.warn(f"Failed to import `braket` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .infleqtion import *
except ImportError as e:
    warnings.warn(f"Failed to import `infleqtion` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .ionq import *
except ImportError as e:
    warnings.warn(f"Failed to import `ionq` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .iqm import *
except ImportError as e:
    warnings.warn(f"Failed to import `iqm` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .oqc import *
except ImportError as e:
    warnings.warn(f"Failed to import `oqc` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .qci import *
except ImportError as e:
    warnings.warn(f"Failed to import `qci` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .quantinuum import *
except ImportError as e:
    warnings.warn(f"Failed to import `quantinuum` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)

try:
    from .quantum_machines import *
except ImportError as e:
    warnings.warn(f"Failed to import `quantum_machines` mock QPU: {e}",
                  ImportWarning,
                  stacklevel=2)
