#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Check that spelling allowlists are sorted alphabetically

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
EXIT_CODE=0

for file in "$REPO_ROOT"/.github/pre-commit/spelling_allowlist*.txt; do
    if [ ! -f "$file" ]; then
        continue
    fi

    sorted_content=$(LC_ALL=C sort "$file")
    current_content=$(cat "$file")

    if [ "$sorted_content" != "$current_content" ]; then
        echo "ERROR: Spelling allowlist is not sorted: $file" >&2
        echo "Run: LC_ALL=C sort -o $file $file" >&2
        EXIT_CODE=1
    else
        echo "OK: $(basename $file) is properly sorted"
    fi
done

exit $EXIT_CODE
