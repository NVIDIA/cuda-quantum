#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# bash scripts/run_clang_format.sh
# -or-
# bash scripts/run_clang_format.sh -p /path/to/clang-format
#
# By default, this script will use the clang-format executable
# in your PATH, but you can modify that with the -p command line option. 

# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":p:" opt; do
  case $opt in
    p) clang_format_executable="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__
clang_format_executable=${clang_format_executable:-clang-format}

# Run the script from the top-level of the repo
cd $(git rev-parse --show-toplevel)

# Run Clang Format
git ls-files -- '*.cpp' '*.h' ':!:tpls/*' ':!:test' | xargs $clang_format_executable -i

# Take us back to where we were
cd -
