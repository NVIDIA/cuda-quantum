#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Pre-commit wrapper for markdown link checking
# Simplified version - checks markdown files only (no RST conversion)

set -e

REPO_ROOT=$(git rev-parse --show-toplevel)
CONFIG_FILE="$REPO_ROOT/.github/pre-commit/md_link_check_config.json"

# Check if markdown-link-check is installed
if ! command -v markdown-link-check &> /dev/null; then
    echo "ERROR: markdown-link-check not found" >&2
    echo "Install with: npm install -g markdown-link-check" >&2
    exit 1
fi

EXIT_CODE=0
FILES=("$@")

if [ ${#FILES[@]} -eq 0 ]; then
    echo "No markdown files to check"
    exit 0
fi

echo "Checking ${#FILES[@]} markdown file(s) for broken links..."

for file in "${FILES[@]}"; do
    if [ ! -f "$file" ]; then
        continue
    fi

    echo "Checking: $file"
    if ! markdown-link-check "$file" --config "$CONFIG_FILE" --quiet; then
        echo "FAILED: $file has broken links" >&2
        EXIT_CODE=1
    else
        echo "OK: $file"
    fi
done

if [ $EXIT_CODE -ne 0 ]; then
    echo "" >&2
    echo "Link check failed for one or more files." >&2
fi

exit $EXIT_CODE
