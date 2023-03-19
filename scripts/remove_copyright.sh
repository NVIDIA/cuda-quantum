#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts is used to remove existing copyright headers recognized by
# the delimiters configured under settings in devcontainer.json.
# The script will modify all files that match the specified pattern in place,
# and exits with a non-zero exit code if the header for one or more file(s) 
# was not found.
#
# Usage:
# bash scripts/remove_copyright.sh
# -or-
# bash scripts/remove_copyright.sh -p *.cpp
# 
# Prerequisites:
# This script makes use of jq - a command line tool to work with JSON.
# This is not installed by default and can be installed with the command:
#   apt update && apt install -y jq
# Furthermore, jq does not deal with comments and the script does not
# strip them from the dev container definition - edit the devcontainer.json
# manually if needed, or update this script.

# Process command line arguments
__optind__=$OPTIND
OPTIND=1
while getopts ":p:" opt; do
  case $opt in
    p) pattern="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    exit 1
    ;;
  esac
done
OPTIND=$__optind__
pattern=${pattern:-"*"}

# Run the script from the top-level of the repo
cd $(git rev-parse --show-toplevel)

settings=`jq '.customizations.vscode.settings' .devcontainer/devcontainer.json`
languages=`echo "$settings" | jq '."psi-header.changes-tracking".include'`
header_configs=`echo "$settings" | jq '[."psi-header.lang-config"[]]'`
#language_delims=`echo "$header_configs" | jq 'map({(.language): {mapto: .mapTo, begin: .begin, end: .end}})'`
valid_delimiters=`echo "$header_configs" | jq 'map(select(has("begin"))) | map({begin: .begin, end: .end, prefix: .prefix, suffix: .suffix})'`

function string_value {
  entry=`echo "$1" | jq .$2`
  entry="${entry%\"}"
  entry="${entry#\"}"
  echo "$entry"
}

function first_match {
  line=`echo "$1" | grep -m 1 -n -F "$2" | sed  's/\([0-9]*\).*/\1/'` # -Fx to match the whole line
  echo $line
}

files=`find . -type f -name "$pattern"  \
        -a ! -wholename '*/build/*'     \
        -a ! -wholename '*/tpls/*'      \
        -a ! -wholename '*/.*/*'        \
        -a ! -name '.*'`

all_headers_found=true
for file in $files; do
  has_copyright_header=false
  echo "Processing file $file"
  echo "$valid_delimiters" | jq -c '.[]' | while read delims && ! $has_copyright_header; do
    begin_marker=`string_value "$delims" begin`
    end_marker=`string_value "$delims" end`

    content=`cat $file`
    start=`first_match "$content" "$begin_marker"`
    content=`echo "$content" | tail -n +$(($start + 1))`
    nr_lines=`first_match "$content" "$end_marker"`
    content=`echo "$content" | head -n $(($nr_lines + 1))`
    copyright=`first_match "$content" Copyright`

    if [ ! "$start" = "" ] && [ ! "$nr_lines" = "" ] && [ ! "$copyright" = "" ]; then 
      has_copyright_header=true
      cp "$file"{,.bak}
      for ((line_nr=0; line_nr <=$nr_lines; line_nr++)); do
        sed -i ${start}d "$file".bak
      done
      mv "$file".bak "$file"
    fi
  done

  if [ ! $has_copyright_header ]; then
    echo "Missing copyright header in $file"
    all_headers_found=false
  fi
done

cd - && exit $((! $all_headers_found))
