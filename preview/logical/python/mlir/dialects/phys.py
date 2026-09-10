# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #
"""Import sentinel for the native-only ``phys`` MLIR dialect.

The dialect has no generated Python OpViews or type casters. Keeping this
module importable lets MLIR cache that fact instead of retrying a failed import
for every generic operation, type, and SSA value downcast.
"""

__all__ = ()
