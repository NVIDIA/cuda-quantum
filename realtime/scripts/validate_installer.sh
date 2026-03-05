#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# Default installation location: /opt/nvidia/cudaq/realtime
install_dir=/opt/nvidia/cudaq/realtime

# First, make sure the "--hololink-dir <path>"" is provided and valid, since it is required for validation.
hololink_dir=""

# Process command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
      --hololink-dir)
        if [[ -n "$2" ]]; then
          hololink_dir="$2"
          shift 2
        else
          echo "Error: --hololink-dir requires a non-empty argument." >&2
          exit 1
        fi
        ;;
      *)
        # Ignore other command line arguments, they will be forwarded to the `hololink_test.sh` script.
        shift
        ;;
    esac
done

if [[ -z "$hololink_dir" ]]; then
  echo "Error: --hololink-dir <path> argument is required." >&2
  exit 1
fi

if [[ ! -d "$hololink_dir" ]]; then
  echo "Error: Provided --hololink-dir path '$hololink_dir' is not a valid directory." >&2
  exit 1
fi

# Now, build the hololink test application using the provided hololink directory.
# This will use the CUDA-Q Realtime installation (rather than building in the source tree).
utils_dir="$install_dir/utils"

# Check that the expected `hololink_test.sh` script exists in the installation.
if [[ ! -f "$utils_dir/hololink_test.sh" ]]; then
  echo "Error: Expected hololink_test.sh script not found in installation at '$utils_dir/hololink_test.sh'." >&2
  exit 1
fi
# Check that it contains the CMakeLists.txt file to build the hololink test application.
if [[ ! -f "$utils_dir/CMakeLists.txt" ]]; then
  echo "Error: Expected CMakeLists.txt for hololink test application not found in installation at '$utils_dir/CMakeLists.txt'." >&2
  exit 1
fi

# Now, build the hololink test application using the provided hololink directory and the installed CUDA-Q Realtime.
rm -rf "$utils_dir/build"
build_dir="$utils_dir/build"
mkdir -p "$build_dir"
cd "$build_dir"
echo "Building hololink test application for validation using hololink directory '$hololink_dir' and CUDA-Q Realtime installation at '$install_dir'..."
cmake .. -DCMAKE_BUILD_TYPE=Release -DHOLOSCAN_SENSOR_BRIDGE_BUILD_DIR=$hololink_dir/build -DHOLOSCAN_SENSOR_BRIDGE_SOURCE_DIR=$hololink_dir 

# Check the status of the cmake command before proceeding to make, since if it fails, the make will also fail and it will be less clear that the issue is with cmake configuration.
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to configure hololink test application for validation." >&2
  exit 1
fi

make

# Check the status of the make command to ensure the test application was built successfully before trying to run it.
if [[ $? -ne 0 ]]; then
  echo "Error: Failed to build hololink test application for validation." >&2
  exit 1
fi

echo "Successfully built hololink test application for validation."

# Add "--bin-dir <path>" argument to specify the location of the built test application, so that the `hololink_test.sh` script can find and run it.
bin_dir="$build_dir"
# Call the `hololink_test.sh` script to validate the installation, forward all the command line arguments to it.
bash "$install_dir/utils/hololink_test.sh" "$@" --bin-dir "$bin_dir"

# Check the status of the hololink test script to determine if the validation was successful.
if [[ $? -ne 0 ]]; then
  echo "Failed to validate hololink test application. Please refer to the documentation for troubleshooting." >&2
  exit 1
fi
