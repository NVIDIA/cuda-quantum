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

# markdown-link-check reaches marked >= 16, which ships only an ESM build, via a
# CommonJS markdown-link-extractor that requires() it. That combination needs a
# Node able to require() an ES module: 20.19 or newer. On older Node the tool
# exits with ERR_REQUIRE_ESM, which this script would otherwise report as broken
# links.
REQUIRED_NODE=20.19.0
NODE_VERSION=$(node --version 2>/dev/null | sed 's/^v//')

if [ -z "$NODE_VERSION" ]; then
    echo "ERROR: node not found; markdown-link-check needs Node >= $REQUIRED_NODE" >&2
    exit 1
fi

if [ "$(printf '%s\n%s\n' "$REQUIRED_NODE" "$NODE_VERSION" | sort -V | head -1)" != "$REQUIRED_NODE" ]; then
    echo "ERROR: node $NODE_VERSION is too old for markdown-link-check" >&2
    echo "       (need >= $REQUIRED_NODE for its ESM-only 'marked' dependency)" >&2
    echo "       See the Prerequisites section of Developing.md to install Node 20." >&2
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
