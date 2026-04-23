#!/bin/bash
# ============================================================================ #
# Copyright (c) 2025 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -e
TARGET_LIB="$1"

# If nm is not available, don't fail. We have it available in at least one CI
# environment, and that is all that is necessary for this sanity check.
if [ ! -x "$(command -v nm)" ]; then
  echo "INFO: nm could not be found, skipping Stim symbol check."
  exit 0
fi

if [ ! -f "$TARGET_LIB" ]; then
    echo "ERROR: Library file not found: $TARGET_LIB" >&2
    exit 1
fi

# Search for 'stim' symbols, excluding the known entry point.
# The command fails if grep finds any matching lines.
if nm -D "$TARGET_LIB" | grep 'stim' | grep -q -v 'getCircuitSimulator_stim'; then
  echo "ERROR: Found unexpected exported symbols containing 'stim' in $TARGET_LIB" >&2
  echo '--- Offending Symbols ---' >&2
  nm -D "$TARGET_LIB" | grep 'stim' | grep -v 'getCircuitSimulator_stim' >&2
  echo '-------------------------' >&2
  exit 1
fi
