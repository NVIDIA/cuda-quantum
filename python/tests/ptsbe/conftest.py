# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
import sys
from pathlib import Path

_ptsbe_dir = Path(__file__).resolve().parent
if str(_ptsbe_dir) not in sys.path:
    sys.path.insert(0, str(_ptsbe_dir))
