#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Installation location: the directory this script was installed into, which is
# /opt/nvidia/cudaq/realtime unless the installer was given --installpath.
install_dir="$(cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")" && pwd)"

# Help needs no initialized hardware.
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    exec bash "$install_dir/utils/validation_test.sh" --help
  fi
done

bin_dir="$install_dir/bin"
# Use the same provider-independent runner as a local realtime build. Preserve
# its exit status, including unsupported configurations and incomplete runs.
exec bash "$install_dir/utils/validation_test.sh" \
  "--validation-app=$bin_dir/validation_app" \
  "--playback-app=$bin_dir/playback_app" "$@"
