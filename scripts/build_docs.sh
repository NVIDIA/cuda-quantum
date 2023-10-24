#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2023 NVIDIA Corporation & Affiliates.                   #
# All rights reserved.                                                         #
#                                                                              #
# This source code and the accompanying materials are made available under     #
# the terms of the Apache License 2.0 which accompanies this distribution.     #
# ============================================================================ #

# This scripts is used to generate html documentation using sphinx. 
# By default, this script only builds the code if this is needed to generate documentation.
# For Python documentation, this means that any changes to the doc comments in the Python
# source code are not automatically reflected in the docs when running this script.
# The -u argument allows to force a rebuild of the Python package such that changes become 
# visible in the generated docs.
#
# Usage:
# bash scripts/build_docs.sh
# -or-
# bash scripts/build_docs.sh -u python
# -or-
# CUDAQ_INSTALL_PREFIX=/cudaq/installation/path/ bash scripts/build_docs.sh
# -or-
# DOCS_INSTALL_PREFIX=/path/to/put/docs/ CUDAQ_INSTALL_PREFIX=/cudaq/installation/path bash scripts/build_docs.sh
#
# Prerequisites:
# All tools required to run this script are installed when using the dev container definition
# in this repository. If you are not using this dev container, you may need to install
# wget, unzip, make, flex, and bison (available via apt install) in addition to the requirements listed in the 
# $CUDAQ_REPO_ROOT/docs/requirements.txt file.

# To view docs in a browser from within a dev container you can use the desktop-lite feature:
# https://github.com/devcontainers/features/tree/main/src/desktop-lite.
# To do so, uncomment this feature in devcontainer.json, and rebuild the container.
# Open a browser and enter the address `http://localhost:6080/`. In VS Code, this can be done by 
# opening the Ports View, and clicking on the `Open in Browser` symbol for the Local Address 
# `localhost:6080`.
# Click connect and enter the default password "cuda-quantum" (without the quotes).
# All GUI commands entered in the terminal of the dev container will now be shown in this minimal
# desktop. A <url> can be opened in the Chrome browser by entering the command 
#   google-chrome --no-sandbox <url>
# 
# The script prints the url to the index of the generated docs at the end. Open that url in
# the browser to preview the docs.

CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$HOME/.cudaq"}
DOCS_INSTALL_PREFIX=${DOCS_INSTALL_PREFIX:-"$CUDAQ_INSTALL_PREFIX/docs"}
export PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

# Process command line arguments
(return 0 2>/dev/null) && is_sourced=true || is_sourced=false
force_update=""

__optind__=$OPTIND
OPTIND=1
while getopts ":u:" opt; do
  case $opt in
    u) force_update="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    if $is_sourced; then return 1; else exit 1; fi
    ;;
  esac
done
OPTIND=$__optind__

# Need to know the top-level of the repo
working_dir=`pwd`
repo_root=$(git rev-parse --show-toplevel)
docs_exit_code=0 # updated in each step

# Make sure these are full path so that it doesn't matter where we use them
docs_build_output="$repo_root/build/docs"
sphinx_output_dir="$docs_build_output/sphinx"
doxygen_output_dir="$docs_build_output/doxygen"
dialect_output_dir="$docs_build_output/Dialects"

# Check if the cudaq Python package is installed and if not, build and install it
build_include_dir="$repo_root/build/include"
python3 -c "import cudaq" 2>/dev/null
if [ ! "$?" -eq "0" ] || [ ! -d "$build_include_dir" ] || [ "${force_update,,}" = "python" ] || [ "${force_update,,}" = "py" ]; then
    echo "Building cudaq package."
    CUDAQ_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" bash "$repo_root/scripts/build_cudaq.sh"
    cudaq_build_exit_code=$?

    python3 -c "import cudaq" 2>/dev/null
    if [ ! "$?" -eq "0" ] || [ ! "$cudaq_build_exit_code" -eq "0" ]; then
        echo "Failed to build and install the CUDA Quantum Python package needed for docs generation."
        cd "$working_dir" && if $is_sourced; then return 2; else exit 2; fi
    else 
        echo "Python package has been installed in $CUDAQ_INSTALL_PREFIX."
        echo "You may need to add it to your PYTHONPATH to use it outside this script."
    fi
fi

# Extract documentation from tablegen files
mkdir -p "$repo_root/build" && cd "$repo_root/build" && mkdir -p logs
logs_dir=`pwd`/logs
cmake .. 1>/dev/null && cmake --build . --target cudaq-doc 1>/dev/null
cmake_exit_code=$?
if [ ! "$cmake_exit_code" -eq "0" ]; then
    echo "Failed to generate documentation from the cudaq-doc build target."
    echo "CMake exit code: $cmake_exit_code"
    docs_exit_code=10
fi

# Check if a new enough version of doxygen is installed, and otherwise build it from source
doxygen_version=`doxygen --version 2>/dev/null | grep -o '^1\.9\.[0-9]*' | cut -d ' ' -f 3`
doxygen_revision=`echo $doxygen_version | cut -d '.' -f 3`
if [ "$doxygen_version" = "" ] || [ "$doxygen_revision" -lt "7" ]; then
    echo "A suitable doxygen installation was not found."
    echo "Attempting to build one from source."
    mkdir -p "$repo_root/build/doxygen" && cd "$repo_root/build/doxygen"

    wget https://github.com/doxygen/doxygen/archive/9a5686aeebff882ebda518151bc5df9d757ea5f7.zip -q -O repo.zip
    (unzip repo.zip && mv doxygen* repo && rm repo.zip) 1> /dev/null
    echo "The progress of the build is being logged to $logs_dir/docstools_output.txt."

    (cmake -G "Unix Makefiles" repo -DCMAKE_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
        && cmake --build . --target install -j $(nproc) --config Release \
        && rm -rf repo) \
        2> "$logs_dir/docstools_error.txt" 1> "$logs_dir/docstools_output.txt"
    if [ ! "$?" -eq "0" ]; then
        echo "Build failed. The build logs can be found in the $logs_dir directory."
        echo "You may need to install the prerequisites listed in the comment at the top of this file."
        cd "$working_dir" && if $is_sourced; then return 3; else exit 3; fi
    else
        doxygen_exe="$CUDAQ_INSTALL_PREFIX/bin/doxygen"
    fi
else
    doxygen_exe=doxygen
fi

# Generate API documentation using Doxygen
echo "Generating XML documentation using Doxygen..."
mkdir -p "${doxygen_output_dir}"
sed 's@${DOXYGEN_OUTPUT_PREFIX}@'"${doxygen_output_dir}"'@' "$repo_root/docs/Doxyfile.in" | \
sed 's@${CUDAQ_REPO_ROOT}@'"${repo_root}"'@' > "${doxygen_output_dir}/Doxyfile"
"$doxygen_exe" "${doxygen_output_dir}/Doxyfile" 2> "$logs_dir/doxygen_error.txt" 1> "$logs_dir/doxygen_output.txt"
doxygen_exit_code=$?
if [ ! "$doxygen_exit_code" -eq "0" ]; then
    echo "Failed to generate documentation using doxygen."
    echo "Doxygen exit code: $doxygen_exit_code"
    docs_exit_code=11
fi

echo "Building CUDA Quantum documentation using Sphinx..."
cd "$repo_root/docs"
# The docs build so far is fast such that we do not care about the cached outputs.
# Revisit this when caching becomes necessary.

rm -rf sphinx/_doxygen/
rm -rf sphinx/_mdgen/
cp -r "$doxygen_output_dir" sphinx/_doxygen/
# cp -r "$dialect_output_dir" sphinx/_mdgen/ # uncomment once we use the content from those files

rm -rf "$sphinx_output_dir"
sphinx-build -v -n -W --keep-going -b html sphinx "$sphinx_output_dir" -j auto 2> "$logs_dir/sphinx_error.txt" 1> "$logs_dir/sphinx_output.txt"
sphinx_exit_code=$?
if [ ! "$sphinx_exit_code" -eq "0" ]; then
    echo "Failed to generate documentation using sphinx-build."
    echo "Sphinx exit code: $sphinx_exit_code"
    echo "======== logs ========"
    cat "$logs_dir/sphinx_output.txt" "$logs_dir/sphinx_error.txt"
    echo "======================"
    docs_exit_code=12
fi

rm -rf sphinx/_doxygen/
rm -rf sphinx/_mdgen/

mkdir -p "$DOCS_INSTALL_PREFIX"
if [ "$docs_exit_code" -eq "0" ]; then
    cp -r "$sphinx_output_dir"/* "$DOCS_INSTALL_PREFIX"
    touch "$DOCS_INSTALL_PREFIX/.nojekyll"
    echo "Documentation was generated in $DOCS_INSTALL_PREFIX."
    echo "To browse it, open this url in a browser: file://$DOCS_INSTALL_PREFIX/index.html"
else
    echo "Documentation generation failed with exit code $docs_exit_code."
    echo "Check the logs in $logs_dir, and the documentation build output in $docs_build_output."
fi

cd "$working_dir" && if $is_sourced; then return $docs_exit_code; else exit $docs_exit_code; fi
