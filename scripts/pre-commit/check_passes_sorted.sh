#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Check that pass definitions in the given Passes.td files are sorted.

EXIT_CODE=0

for file in "$@"; do
    names=$(sed -nE 's/^def ([A-Za-z0-9_]+) *:.*/\1/p' "$file")
    if [ "$names" != "$(LC_ALL=C sort <<< "$names")" ]; then
        echo "ERROR: Passes are not in alphabetical order: $file" >&2
        LC_ALL=C sort -c <<< "$names" 2>&1 | sed 's/^sort: -:[0-9]*: disorder: /  First out of order: /' >&2
        EXIT_CODE=1
    fi
done

exit $EXIT_CODE
