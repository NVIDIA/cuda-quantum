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

echo "Migrating assets to $target..."

find . -type f -print0 | while IFS= read -r -d '' file;
do 
    echo "Processing $file..."  
    if [ ! -f "$target/$file" ]; then 
        # Move the file to the target location, preserving directory structure
        target_path="$target/$(dirname "$file")"
        mkdir -p "$target_path"
        mv "$file" "$target_path/"
        echo "Moved $file to $target_path/"
    else
        echo "File $target/$file already exists, skipping."
    fi    
done