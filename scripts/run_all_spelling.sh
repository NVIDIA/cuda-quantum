#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Usage:
# bash scripts/run_all_spelling.sh
# -or-
# bash scripts/run_all_spelling.sh -d  (only checks files that have changed from main)

# Note: this script intentionally exits upon encountering the first spelling
# error so that you can go fix the error.

cd $(git rev-parse --show-toplevel)

RUN_ON_ALL_FILES=1
__optind__=$OPTIND
OPTIND=1
while getopts "::d" opt; do
  case $opt in
    d) RUN_ON_ALL_FILES=0
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__

CONFIG=.github/workflows/config/spellcheck_config.yml

if [ "$RUN_ON_ALL_FILES" = "1" ]; then
  ALL_FILES_TO_CONSIDER=`git ls-files --`
else
  ALL_FILES_TO_CONSIDER=`git diff --diff-filter=d --name-only main... --`
fi
NCHECK=0
for f in $ALL_FILES_TO_CONSIDER; do
  NCHECK=$((NCHECK+1))
done
echo "Going to check up to $NCHECK files"

# markdown
ff=`find $ALL_FILES_TO_CONSIDER -name "*.md" | grep -v -P "^tpls/"`
NCHECK=0; for f in $ff; do NCHECK=$((NCHECK+1)); done
echo "Beginning markdown check ($NCHECK files to check)"
for f in $ff; do
  echo "*** $f ***"
  pyspelling -n markdown -c ${CONFIG} -S $f
  if [ $? -ne 0 ]; then
    exit
  fi
done

# rst
ff=`find $ALL_FILES_TO_CONSIDER -name "*.rst" | grep -v -P "^tpls/|_templates"`
NCHECK=0; for f in $ff; do NCHECK=$((NCHECK+1)); done
echo "Beginning rst check ($NCHECK files to check)"
for f in $ff; do
  echo "*** $f ***"
  pyspelling -n rst -c ${CONFIG} -S $f
  if [ $? -ne 0 ]; then
    exit
  fi
done

# cxx_headers
ff=`find $ALL_FILES_TO_CONSIDER -name "*.h" -o -name "*.hpp" | grep -v -P "^test/|^tpls/|nlopt-src/"`
NCHECK=0; for f in $ff; do NCHECK=$((NCHECK+1)); done
echo "Beginning cxx_headers check ($NCHECK files to check)"
for f in $ff; do
  echo "*** $f ***"
  pyspelling -n cxx_headers -c ${CONFIG} -S $f
  if [ $? -ne 0 ]; then
    exit
  fi
done

# cxx_examples
ff=`find $ALL_FILES_TO_CONSIDER -name "*.cpp" | grep -E "^docs/sphinx/applications/|^docs/sphinx/targets/|^docs/sphinx/examples/"`
NCHECK=0; for f in $ff; do NCHECK=$((NCHECK+1)); done
echo "Beginning cxx_examples check ($NCHECK files to check)"
for f in $ff; do
  echo "*** $f ***"
  pyspelling -n cxx_examples -c ${CONFIG} -S $f
  if [ $? -ne 0 ]; then
    exit
  fi
done

# python
ff=`find $ALL_FILES_TO_CONSIDER -name "*.py" | grep -v -P "^python/tests/|^test/|^tpls/|^docs/sphinx/conf.py"`
NCHECK=0; for f in $ff; do NCHECK=$((NCHECK+1)); done
echo "Beginning python check ($NCHECK files to check)"
for f in $ff; do
  echo "*** $f ***"
  pyspelling -n python -c ${CONFIG} -S $f
  if [ $? -ne 0 ]; then
    exit
  fi
done
