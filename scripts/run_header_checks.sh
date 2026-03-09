#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts is used to check that a copyright header is present 
# in all files where it is required. 
#
# Usage:
# bash scripts/run_header_checks.sh
# -or-
# bash scripts/run_header_checks.sh -c <command>
#
# where <command> is passed when invoking `license-eye header`.

# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":c:" opt; do
  case $opt in
    c) command="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__
command=${command:-check}

# Run the script from the top-level of the repo
cd $(git rev-parse --show-toplevel)

# The license-eye check ignores files ending in .txt by default;
# we create a temporary copy of CMakeLists.txt files to check them.
cmakelists=$(find . -name "CMakeLists.txt" -not -path "./tpls/*")
for file in $cmakelists; do
  cp "$file"{,.tmp}
done

go install github.com/apache/skywalking-eyes/cmd/license-eye@latest
# Use GOPATH if set, otherwise default to ~/go (Go's default)
"${GOPATH:-$HOME/go}/bin/license-eye" header $command
status=$?

for file in $cmakelists; do
  rm "$file".tmp
done

cd - && exit $status
