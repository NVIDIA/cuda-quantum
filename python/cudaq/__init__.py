# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import sys

from _pycudaq import *
from domains import chemistry

initKwargs = {'qpu': 'qpp', 'platform': 'default'}

if '-qpu' in sys.argv:
    initKwargs['qpu'] = sys.argv[sys.argv.index('-qpu') + 1]

if '--qpu' in sys.argv:
    initKwargs['qpu'] = sys.argv[sys.argv.index('--qpu') + 1]

if '-platform' in sys.argv:
    initKwargs['platform'] = sys.argv[sys.argv.index('-platform') + 1]

if '--platform' in sys.argv:
    initKwargs['platform'] = sys.argv[sys.argv.index('--platform') + 1]

initialize_cudaq(**initKwargs)

__all__ = [""]