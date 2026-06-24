# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

import logging
import numpy
from numbers import Number
from typing import Any, Mapping, List, Union
from ..operators import ElementaryOperator, OperatorArithmetics, ScalarOperator
from .schedule import Schedule

logger = logging.getLogger(__name__)
