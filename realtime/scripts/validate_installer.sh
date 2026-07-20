#!/bin/bash

# ============================================================================ #
# Copyright (c) 2026 NVIDIA Corporation & Affiliates.                          #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Default installation location: /opt/nvidia/cudaq/realtime
install_dir=/opt/nvidia/cudaq/realtime

# Check LD_LIBRARY_PATH contains the install_dir/lib path
if [[ ":$LD_LIBRARY_PATH:" != *":$install_dir/lib:"* ]]; then
  echo "Warning: LD_LIBRARY_PATH does not contain $install_dir/lib. Please add it to your environment variables to ensure CUDA-Q Realtime works correctly." >&2
  echo "For example, you can run: export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$install_dir/lib"
fi

bin_dir="$install_dir/bin"
# Call the `hololink_test.sh` script to validate the installation, forward all the command line arguments to it.
bash "$install_dir/utils/hololink_test.sh" "$@" --bin-dir "$bin_dir"

# Check the status of the hololink test script to determine if the validation was successful.
if [[ $? -ne 0 ]]; then
  echo "Failed to validate hololink test application. Please refer to the documentation for troubleshooting." >&2
  exit 1
fi
