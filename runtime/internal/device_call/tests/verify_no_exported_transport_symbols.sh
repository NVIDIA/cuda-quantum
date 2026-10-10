#!/usr/bin/env bash
# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                         #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <shared-library>" >&2
  exit 2
fi

lib="$1"

# The --exclude-libs flag is ELF/GNU linker specific, so only run this check
# on ELF platforms.
case "$(uname -s)" in
  Linux) ;;
  *) echo "Skipping symbol-export check on non-ELF platform"
     exit 0
     ;;
esac

if [ ! -f "$lib" ]; then
  echo "Shared library not found: $lib" >&2
  exit 1
fi

if ! command -v nm >/dev/null 2>&1; then
  echo "nm not found; cannot verify symbol visibility" >&2
  exit 1
fi

# Defined dynamic symbols have an address; undefined references do not.
# --exclude-libs must hide the transport archive symbols, so no cpu_udp_*
# or cpu_roce_* definitions may appear in the dynamic symbol table.
exported=$(nm -D "$lib" | grep -E '^[0-9a-fA-F]+[[:space:]]+[^U]' | \
           grep -E 'cpu_udp_|cpu_roce_' || true)

if [ -n "$exported" ]; then
  echo "ERROR: transport symbols are exported from $lib:" >&2
  echo "$exported" >&2
  exit 1
fi
