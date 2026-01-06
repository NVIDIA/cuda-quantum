#!/bin/bash

# ============================================================================ #
# Copyright (c) 2022 - 2026 NVIDIA Corporation & Affiliates.                   #
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

# The script prints the url to the index of the generated docs at the end. Open that url in
# the browser to preview the docs. If you are working within a dev container, you can use
# the Live Server extension in VS Code to view the docs in a Browser:
#   https://marketplace.visualstudio.com/items?itemName=ritwickdey.LiveServer
# After building the docs, go to the build/docs/sphinx directory, right click on index.html, and 
# select "Open With Live Server".

CUDAQ_INSTALL_PREFIX=${CUDAQ_INSTALL_PREFIX:-"$HOME/.cudaq"}
DOCS_INSTALL_PREFIX=${DOCS_INSTALL_PREFIX:-"$CUDAQ_INSTALL_PREFIX/docs"}
export PYTHONPATH="$CUDAQ_INSTALL_PREFIX:${PYTHONPATH}"

# Process command line arguments
force_update=""

__optind__=$OPTIND
OPTIND=1
while getopts ":u:" opt; do
  case $opt in
    u) force_update="$OPTARG"
    ;;
    \?) echo "Invalid command line option -$OPTARG" >&2
    (return 0 2>/dev/null) && return 1 || exit 1
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
rm -rf "$docs_build_output"

# Check if the cudaq Python package is installed and if not, build and install it
build_include_dir="$repo_root/build/include"
python3 -c "import cudaq" 2>/dev/null
if [ ! "$?" -eq "0" ] || [ ! -d "$build_include_dir" ] || [ "${force_update,,}" = "python" ] || [ "${force_update,,}" = "py" ]; then
    echo "Building cudaq package."
    CUDAQ_INSTALL_PREFIX="$CUDAQ_INSTALL_PREFIX" \
    CUDAQ_BUILD_TESTS=OFF \
    bash "$repo_root/scripts/build_cudaq.sh"
    cudaq_build_exit_code=$?

    python3 -c "import cudaq" 2>/dev/null
    if [ ! "$?" -eq "0" ] || [ ! "$cudaq_build_exit_code" -eq "0" ]; then
        echo "Failed to build and install the CUDA-Q Python package needed for docs generation."
        cd "$working_dir" && (return 0 2>/dev/null) && return 2 || exit 2
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
        cd "$working_dir" && (return 0 2>/dev/null) && return 3 || exit 3
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
    cat "$logs_dir/doxygen_output.txt" "$logs_dir/doxygen_error.txt"
    echo "Failed to generate documentation using doxygen."
    echo "Doxygen exit code: $doxygen_exit_code"
    docs_exit_code=11
fi

# Create Python readme from template
echo "Creating README.md for cudaq package"
package_name=cudaq
cuda_version_requirement="12.x or 13.x"
cuda_version_conda=12.4.0 # only used as example in the install script
deprecation_notice=""
cat "$repo_root/python/README.md.in" > "$repo_root/python/README.md"
for variable in package_name cuda_version_requirement cuda_version_conda deprecation_notice; do
    sed -i "s/.{{[ ]*$variable[ ]*}}/${!variable}/g" "$repo_root/python/README.md"
done
if [ -n "$(cat "$repo_root/python/README.md" | grep -e '.{{.*}}')" ]; then 
    echo "Incomplete template substitutions in README."
    docs_exit_code=1
fi

echo "Building CUDA-Q documentation using Sphinx..."
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

    # Generate markdown files from HTML using pandoc
    echo "Converting HTML documentation to Markdown using pandoc..."

    # Verify that pandoc is installed
    if ! command -v pandoc &> /dev/null; then
        echo "pandoc is required but not installed. Please install pandoc to convert HTML to Markdown."
        exit 4
    fi

    # Convert each html file to markdown
    find "$DOCS_INSTALL_PREFIX" -type f -name "*.html" | while read -r html_file; do
        md_file="${html_file%.html}.md"
        pandoc "$html_file" -f html -t markdown -o "$md_file"
        if [ "$?" -ne "0" ]; then
            echo "Failed to convert $html_file"
            docs_exit_code=13
        fi
    done

    if [ "$docs_exit_code" -eq "0" ]; then
        echo "Markdown documentation generated successfully."
        find "$sphinx_output_dir" -type f -name "*.md" \
            -exec cp --parents '{}' "$DOCS_INSTALL_PREFIX" \;
        echo "Markdown files copied successfully to $DOCS_INSTALL_PREFIX."

        # Copy llms.txt from the repository root to the docs install prefix
        if [ -f "$CUDAQ_REPO_ROOT/llms.txt" ]; then
            cp "$CUDAQ_REPO_ROOT/llms.txt" "$DOCS_INSTALL_PREFIX/"
            echo "Copied llms.txt to $DOCS_INSTALL_PREFIX."
        else
            echo "Warning: llms.txt not found in $CUDAQ_REPO_ROOT, skipping copy."
        fi
    else
        echo "Markdown documentation encountered issues."
    fi
else
    echo "Documentation generation failed with exit code $docs_exit_code."
    echo "Check the logs in $logs_dir, and the documentation build output in $docs_build_output."
fi

cd "$working_dir" && (return 0 2>/dev/null) && return $docs_exit_code || exit $docs_exit_code
