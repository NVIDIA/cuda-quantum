#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This script migrates assets from an extracted installer directory to the expected locations.

# Default target location: /opt/nvidia/cudaq/realtime
target=/opt/nvidia/cudaq/realtime


# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":t:" opt; do
  case $opt in
    t) target="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
    ;;
  esac
done
OPTIND=$__optind__

mkdir -p "$target"

# Generate uninstall script from the full payload file list (not only what we
# move this run), so it is correct on first install and when re-running.
uninstall_script="$target/uninstall.sh"
target_quoted=$(printf '%q' "$target")

{
  printf '#!/bin/bash\nset -euo pipefail\n\ntarget=%s\n\n' "$target_quoted"
  printf 'echo "This will remove CUDA-Q Realtime files installed under $target."\n'
  printf 'echo "The following files will be removed:"\n'
  # List files for the user before asking for confirmation
  find . -type f -print0 | while IFS= read -r -d '' file; do
    [ "$file" = "./install.sh" ] && continue
    # Strip leading ./
    rel="${file#./}"
    printf 'echo "  $target/%s"\n' "$rel"
  done
  printf '\nread -r -p "Continue? [y/N] " answer\n'
  printf 'case "${answer,,}" in\n  y|yes) ;;\n  *) echo "Aborted."; exit 1 ;;\nesac\n\n'
  # Removal commands for all payload files (excluding install.sh)
  find . -type f -print0 | while IFS= read -r -d '' file; do
    [ "$file" = "./install.sh" ] && continue
    rel="${file#./}"
    printf 'rm -f "$target/%s"\n' "$rel"
  done
  printf 'find "$target" -type d -empty -delete\n'
  printf 'rm -f "$target/uninstall.sh"\n'
  printf 'rmdir "$target" 2>/dev/null || true\n'
} > "$uninstall_script"
chmod a+x "$uninstall_script"

echo "Migrating assets to $target..."
echo "Uninstall script: $uninstall_script"

find . -type f -print0 | while IFS= read -r -d '' file;
do
    [ "$file" = "./install.sh" ] && continue

    echo "Processing $file..."
    if [ ! -f "$target/$file" ]; then
        target_path="$target/$(dirname "$file")"
        mkdir -p "$target_path"
        mv "$file" "$target_path/"
        echo "Moved $file to $target_path/"
    else
        echo "File $target/$file already exists, skipping."
    fi
done
